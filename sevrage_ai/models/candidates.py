"""Candidate estimators: enumeration from the training config and construction. Owner: WS4.

Untyped scikit-learn objects stay behind the :class:`Classifier` / :class:`Regressor`
protocols.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sevrage_ai.config.schema import FeatureSpec
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.features.preprocessing import (
    build_preprocessor,
    categorical_feature_names,
    monotonic_constraints,
)
from sevrage_ai.logging_config import get_logger
from sevrage_ai.types import ModelFamily, TaskKind

__all__ = [
    "ISOTONIC_MIN_ROWS",
    "Calibration",
    "CandidateSpec",
    "Classifier",
    "ConstantClassifier",
    "EstimatorFamily",
    "ParamValue",
    "Regressor",
    "build",
    "enumerate_candidates",
]

EstimatorFamily = Literal["logistic", "forest", "hgb", "ridge", "constant"]
Calibration = Literal["none", "sigmoid", "isotonic"]
ParamValue = float | int | str | None

ISOTONIC_MIN_ROWS: Final = 1000
"""Isotonic calibration is refused below this number of training rows (design 3.2)."""

_LOGGER = get_logger(__name__)
_FOREST_BASE: Final = 50
_HGB_BASE: Final = 100
_CALIBRATION_PENALTY: Final = 5


class Classifier(Protocol):
    """Fitted or unfitted probabilistic classifier (scikit-learn API)."""

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.intp]) -> object:
        """Fit on encoded labels ``0..C-1``."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Probabilities, shape ``(n, C')``, columns in ``classes_`` order."""
        ...


class Regressor(Protocol):
    """Fitted or unfitted regressor (scikit-learn API)."""

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.float64]) -> object:
        """Fit on the target in original units (the transform is inside the estimator)."""
        ...

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predictions, shape ``(n,)``."""
        ...


class ConstantClassifier:
    """Predicts one class with probability 1 (task with a single remaining class)."""

    def __init__(self) -> None:
        self.classes_ = np.array([0], dtype=np.intp)

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.intp]) -> ConstantClassifier:  # noqa: ARG002
        """Nothing to learn."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Column of ones."""
        return np.ones((len(X), 1), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One point of a candidate grid.

    ``params`` holds sorted ``(name, value)`` pairs, so the spec is hashable. ``complexity``
    orders candidates for the one-standard-error rule: lower is simpler and is preferred
    among candidates within one SE of the best (logistic/ridge < forest < HGB by depth;
    stronger regularisation and no calibration are simpler).
    """

    family: EstimatorFamily
    task_kind: TaskKind
    params: tuple[tuple[str, ParamValue], ...]
    calibration: Calibration = "none"
    complexity: int = 0

    @property
    def name(self) -> str:
        """Stable readable identifier, e.g. ``"logistic[C=0.1,class_weight=balanced]+sigmoid"``."""
        inner = ",".join(f"{key}={value}" for key, value in self.params)
        suffix = "" if self.calibration == "none" else f"+{self.calibration}"
        return f"{self.family}[{inner}]{suffix}"

    def params_dict(self) -> dict[str, ParamValue]:
        """``params`` as a dictionary."""
        return dict(self.params)


def _grid(fields: dict[str, Sequence[ParamValue]]) -> list[dict[str, ParamValue]]:
    keys = sorted(fields)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(fields[k] for k in keys))
    ]


def _number(value: ParamValue) -> float:
    if isinstance(value, int | float):
        return float(value)
    msg = f"numeric grid value expected, got {value!r}"
    raise TypeError(msg)


def _rank(value: ParamValue, values: Sequence[float], *, descending: bool = False) -> int:
    ordered = sorted(set(values), reverse=descending)
    return ordered.index(_number(value))


def enumerate_candidates(kind: TaskKind, cfg: TrainingConfig) -> list[CandidateSpec]:
    """Cartesian product of the configured grids (x calibration methods for classifiers).

    Named ``enumerate_candidates`` rather than ``enumerate`` (design) to avoid shadowing the
    builtin.
    """
    specs: list[CandidateSpec] = []
    if kind.is_classification:
        grids = cfg.candidates.classification
        bases: list[tuple[EstimatorFamily, dict[str, ParamValue], int]] = []
        if grids.logistic is not None:
            for params in _grid(
                {
                    "C": grids.logistic.C,
                    "class_weight": grids.logistic.class_weight,
                    "max_iter": (grids.logistic.max_iter,),
                }
            ):
                bases.append(("logistic", params, _rank(params["C"], grids.logistic.C)))
        if grids.forest is not None:
            forest = grids.forest
            for params in _grid(
                {
                    "n_estimators": forest.n_estimators,
                    "min_samples_leaf": forest.min_samples_leaf,
                    "max_features": forest.max_features,
                    "class_weight": forest.class_weight,
                }
            ):
                complexity = _FOREST_BASE + _rank(
                    params["min_samples_leaf"], forest.min_samples_leaf, descending=True
                )
                bases.append(("forest", params, complexity))
        if grids.hgb is not None:
            hgb = grids.hgb
            for params in _grid(
                {
                    "max_depth": hgb.max_depth,
                    "learning_rate": hgb.learning_rate,
                    "max_iter": hgb.max_iter,
                    "min_samples_leaf": hgb.min_samples_leaf,
                    "l2_regularization": hgb.l2_regularization,
                    "class_weight": hgb.class_weight,
                }
            ):
                complexity = (
                    _HGB_BASE
                    + 10 * int(_number(params["max_depth"]))
                    + _rank(params["min_samples_leaf"], hgb.min_samples_leaf, descending=True)
                )
                bases.append(("hgb", params, complexity))
        for (family, params, complexity), method in itertools.product(
            bases, cfg.calibration.methods
        ):
            specs.append(
                CandidateSpec(
                    family=family,
                    task_kind=kind,
                    params=tuple(sorted(params.items())),
                    calibration=method,
                    complexity=complexity + (0 if method == "none" else _CALIBRATION_PENALTY),
                )
            )
        return specs
    reg = cfg.candidates.regression
    if reg.ridge is not None:
        for alpha in reg.ridge.alpha:
            specs.append(
                CandidateSpec(
                    family="ridge",
                    task_kind=kind,
                    params=(("alpha", alpha),),
                    complexity=_rank(alpha, reg.ridge.alpha, descending=True),
                )
            )
    if reg.hgb is not None:
        hgb_r = reg.hgb
        for params in _grid(
            {
                "max_depth": hgb_r.max_depth,
                "loss": hgb_r.loss,
                "min_samples_leaf": hgb_r.min_samples_leaf,
                "learning_rate": hgb_r.learning_rate,
                "max_iter": hgb_r.max_iter,
                "l2_regularization": hgb_r.l2_regularization,
            }
        ):
            complexity = (
                _HGB_BASE
                + 10 * int(_number(params["max_depth"]))
                + _rank(params["min_samples_leaf"], hgb_r.min_samples_leaf, descending=True)
            )
            specs.append(
                CandidateSpec(
                    family="hgb",
                    task_kind=kind,
                    params=tuple(sorted(params.items())),
                    complexity=complexity,
                )
            )
    return specs


def _hgb_extras(features: Sequence[FeatureSpec], *, monotonic: bool) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    categorical = categorical_feature_names(features)
    if categorical:
        extras["categorical_features"] = categorical
    constraints = monotonic_constraints(features)
    if monotonic and constraints:
        extras["monotonic_cst"] = constraints
    return extras


def build(
    candidate: CandidateSpec,
    features: Sequence[FeatureSpec],
    *,
    seed: int,
    calibration_cv: int,
    smallest_class_count: int | None = None,
    transform: Literal["none", "log1p"] = "none",
) -> Classifier | Regressor:
    """Build an unfitted pipeline (preprocessor + estimator).

    Classifiers are optionally wrapped in ``CalibratedClassifierCV(cv=calibration_cv)``.
    **Guard:** if ``smallest_class_count < calibration_cv``, fall back to no calibration and
    log it. Regressors are wrapped in ``TransformedTargetRegressor`` when ``transform`` is
    ``"log1p"`` (``log1p``/``expm1``). ``seed`` sets every ``random_state``. Monotonic
    constraints are only used where HGB supports them (binary classification, regression).
    """
    params = candidate.params_dict()
    if candidate.task_kind.is_classification:
        return _build_classifier(
            candidate, params, features, seed, calibration_cv, smallest_class_count
        )
    estimator: Any
    if candidate.family == "ridge":
        estimator = Ridge(alpha=params["alpha"])
        family = ModelFamily.LINEAR
    else:
        estimator = HistGradientBoostingRegressor(
            **params,
            early_stopping=False,
            random_state=seed,
            **_hgb_extras(features, monotonic=True),
        )
        family = ModelFamily.TREE
    pipeline = Pipeline([("prep", build_preprocessor(features, family)), ("model", estimator)])
    if transform == "log1p":
        regressor: Regressor = TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
        return regressor
    result: Regressor = pipeline
    return result


def _build_classifier(
    candidate: CandidateSpec,
    params: dict[str, ParamValue],
    features: Sequence[FeatureSpec],
    seed: int,
    calibration_cv: int,
    smallest_class_count: int | None,
) -> Classifier:
    estimator: Any
    if candidate.family == "logistic":
        estimator = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=params["max_iter"],
            random_state=seed,
        )
        family = ModelFamily.LINEAR
    elif candidate.family == "forest":
        estimator = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
        family = ModelFamily.TREE
    else:
        monotonic = candidate.task_kind is TaskKind.BINARY
        estimator = HistGradientBoostingClassifier(
            **params,
            early_stopping=False,
            random_state=seed,
            **_hgb_extras(features, monotonic=monotonic),
        )
        family = ModelFamily.TREE
    pipeline = Pipeline([("prep", build_preprocessor(features, family)), ("model", estimator)])
    method = candidate.calibration
    if method == "none":
        result: Classifier = pipeline
        return result
    if smallest_class_count is not None and smallest_class_count < calibration_cv:
        _LOGGER.info(
            "Calibration %s abandonnée : classe minoritaire trop petite (%d < %d).",
            method,
            smallest_class_count,
            calibration_cv,
        )
        fallback: Classifier = pipeline
        return fallback
    calibrated: Classifier = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=StratifiedKFold(n_splits=calibration_cv, shuffle=True, random_state=seed),
    )
    return calibrated
