"""Model selection, cross-fitting and per-target fitting (D4). Owner: WS4.

Each target is served by an ensemble of K cross-fit fold models; probabilities or
predictions are averaged. The same folds produce the out-of-fold (OOF) predictions used
for conformal calibration and for the treatment decision threshold.

Model space vs original units (regression)
    ``CrossFit.oof`` and :meth:`FittedTarget.fold_predictions` are in the **transformed**
    space (``log1p`` when ``task.transform == "log1p"``), where CV+ intervals are
    computed. :meth:`FittedTarget.predict` returns original units (back-transform of the
    mean in model space).

Selection scores (higher is better) are computed with numpy, on probabilities clipped
away from 0, so that a class absent from an inner training fold cannot produce an
infinite log-loss.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from sevrage_ai.confidence.conformal import CrossConformalClassifier, CVPlusRegressor
from sevrage_ai.config.schema import FeatureSpec
from sevrage_ai.config.training import ThresholdConfig, TrainingConfig
from sevrage_ai.logging_config import get_logger
from sevrage_ai.models.candidates import (
    ISOTONIC_MIN_ROWS,
    CandidateSpec,
    Classifier,
    ConstantClassifier,
    Regressor,
    build,
    enumerate_candidates,
)
from sevrage_ai.models.splits import make_splitter, quantile_strata
from sevrage_ai.models.staged import (
    GATE_CLOSED,
    GATE_OPEN,
    GATE_UNKNOWN,
    TwoStageClassifier,
    gate_of,
    joint_codes,
    label_of,
)
from sevrage_ai.models.tasks import TargetTask, encode_labels, regression_values
from sevrage_ai.types import CanonicalData, TargetName, TaskKind

__all__ = [
    "CrossFit",
    "FittedTarget",
    "choose_threshold",
    "cross_fit",
    "fit_target",
    "score_candidate",
    "select",
]

_LOGGER = get_logger(__name__)
_PROBA_FLOOR: Final = 1e-15
_MIN_STAGE_ROWS: Final = 10


@dataclass(frozen=True, slots=True, eq=False)
class CrossFit:
    """K fold models fitted on the task's rows.

    ``fold_ids[i]`` is the fold in which task row ``i`` was held out. ``oof`` holds the
    out-of-fold outputs of the task rows: probabilities ``(n, C)`` for classifiers, model
    space predictions ``(n,)`` for regressors.
    """

    fold_models: tuple[Classifier | Regressor, ...]
    fold_ids: npt.NDArray[np.intp]
    oof: npt.NDArray[np.float64]


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------


def _fit(model: Classifier | Regressor, X: pd.DataFrame, y: npt.NDArray[Any]) -> None:
    with warnings.catch_warnings():
        # lbfgs may stop a few iterations short on tiny folds; the model is still usable
        # and is judged on held-out data like every other candidate.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)


def _fit_model(
    task: TargetTask, model: Classifier | Regressor, X: pd.DataFrame, y_fit: npt.NDArray[Any]
) -> None:
    """Fit on ``y_fit``; a single model given joint codes gets the class indices."""
    if task.classes is not None and not isinstance(model, TwoStageClassifier):
        y_fit = label_of(np.asarray(y_fit, dtype=np.intp), len(task.classes))
    _fit(model, X, y_fit)


def aligned_proba(model: Classifier, X: pd.DataFrame, n_classes: int) -> npt.NDArray[np.float64]:
    """``predict_proba`` with one column per task class (classes unseen in the fit set get
    probability 0), rows renormalised to sum to 1."""
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])), dtype=np.intp)
    out = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
    out[:, classes] = raw
    totals = out.sum(axis=1, keepdims=True)
    totals[totals <= 0.0] = 1.0
    normalised: npt.NDArray[np.float64] = out / totals
    return normalised


def model_space(task: TargetTask, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Original units -> model space (``log1p`` when the task is transformed)."""
    if task.transform == "log1p":
        transformed: npt.NDArray[np.float64] = np.log1p(np.maximum(values, 0.0))
        return transformed
    return np.asarray(values, dtype=np.float64)


def original_units(task: TargetTask, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Model space -> original units."""
    if task.transform == "log1p":
        back: npt.NDArray[np.float64] = np.expm1(values)
        return back
    return np.asarray(values, dtype=np.float64)


def _regression_model_space(
    task: TargetTask, model: Regressor, X: pd.DataFrame
) -> npt.NDArray[np.float64]:
    inner = getattr(model, "regressor_", None)
    if task.transform == "log1p" and inner is not None:
        return np.asarray(inner.predict(X), dtype=np.float64)
    return model_space(task, np.asarray(model.predict(X), dtype=np.float64))


def _log_loss(y: npt.NDArray[np.intp], proba: npt.NDArray[np.float64]) -> float:
    picked = np.clip(proba[np.arange(y.size), y], _PROBA_FLOOR, 1.0)
    return float(-np.log(picked).mean())


def _brier(y: npt.NDArray[np.intp], proba: npt.NDArray[np.float64]) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(y.size), y] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def score_candidate(
    task: TargetTask,
    cfg: TrainingConfig,
    y_true: npt.NDArray[Any],
    output: npt.NDArray[np.float64],
) -> float:
    """Selection score (higher is better): negative log-loss / Brier for classifiers,
    negative MAE / RMSE in original units for regressors (``output`` in original units)."""
    if task.classes is not None:
        y = np.asarray(y_true, dtype=np.intp)
        if cfg.selection.classification == "neg_log_loss":
            return -_log_loss(y, output)
        return -_brier(y, output)
    errors = np.asarray(y_true, dtype=np.float64) - output
    if cfg.selection.regression == "neg_mean_absolute_error":
        return -float(np.abs(errors).mean())
    return -float(np.sqrt((errors**2).mean()))


def _task_arrays(
    task: TargetTask, data: CanonicalData, features: Sequence[FeatureSpec]
) -> tuple[pd.DataFrame, npt.NDArray[Any], npt.NDArray[np.str_] | None]:
    if data.y is None:
        msg = "training needs labelled data"
        raise ValueError(msg)
    names = [f.name for f in features]
    X = data.X.loc[task.row_mask, names].reset_index(drop=True)
    labels = data.y[task.name.value]
    y: npt.NDArray[Any] = (
        encode_labels(task, labels) if task.classes is not None else regression_values(task, labels)
    )
    groups = None if data.groups is None else data.groups[task.row_mask].to_numpy(dtype=str)
    return X, y, groups


def _strata(task: TargetTask, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return y if task.classes is not None else quantile_strata(np.asarray(y, dtype=np.float64))


def _smallest(y: npt.NDArray[Any]) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if counts.size else 0


def fit_labels(
    task: TargetTask, data: CanonicalData, cfg: TrainingConfig, y: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Labels given to ``fit``: ``y``, or joint codes for a two-stage target.

    A target listed in ``cfg.two_stage`` gets the joint codes of ``models.staged`` (class
    and gate). When the gate target is absent from the data, ``y`` is returned unchanged
    and a single model is fitted.
    """
    stage = cfg.two_stage.get(task.name)
    if stage is None or task.classes is None or data.y is None:
        return y
    if stage.gate.value not in data.y or stage.otherwise not in task.classes:
        return y
    column = data.y[stage.gate.value][task.row_mask]
    known = column.notna().to_numpy(dtype=bool)
    opened = (column == stage.equals).fillna(value=False).to_numpy(dtype=bool)
    gate = np.where(known, np.where(opened, GATE_OPEN, GATE_CLOSED), GATE_UNKNOWN)
    return joint_codes(np.asarray(y, dtype=np.intp), gate.astype(np.intp), len(task.classes))


def _build_two_stage(
    task: TargetTask,
    candidate: CandidateSpec,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
    codes: npt.NDArray[np.intp],
) -> Classifier | None:
    """Two-stage model for the training ``codes``, or ``None`` if a stage cannot be learned
    (too few gated rows, or a single class among them)."""
    classes = task.classes or ()
    labels, gate = label_of(codes, len(classes)), gate_of(codes, len(classes))
    staged = labels[gate == GATE_OPEN]
    if staged.size < _MIN_STAGE_ROWS or np.unique(staged).size < 2:
        return None
    gate_spec = CandidateSpec(
        family=candidate.family,
        task_kind=TaskKind.BINARY,
        params=candidate.params,
        calibration=candidate.calibration,
        complexity=candidate.complexity,
    )
    gate_model = build(
        gate_spec,
        features,
        seed=cfg.seed,
        calibration_cv=cfg.calibration.cv,
        smallest_class_count=_smallest(gate[gate != GATE_UNKNOWN]),
    )
    stage_model = build(
        candidate,
        features,
        seed=cfg.seed,
        calibration_cv=cfg.calibration.cv,
        smallest_class_count=_smallest(staged),
    )
    model: Classifier = TwoStageClassifier(
        gate_model,  # type: ignore[arg-type]
        stage_model,  # type: ignore[arg-type]
        n_classes=len(classes),
        otherwise=classes.index(cfg.two_stage[task.name].otherwise),
    )
    return model


def _build(
    task: TargetTask,
    candidate: CandidateSpec,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
    y_train: npt.NDArray[Any],
) -> Classifier | Regressor:
    """Unfitted model for the training labels ``y_train`` (see :func:`fit_labels`)."""
    if task.classes is not None and int(np.max(y_train, initial=0)) >= len(task.classes):
        staged = _build_two_stage(task, candidate, features, cfg, y_train)
        if staged is not None:
            return staged
    return build(
        candidate,
        features,
        seed=cfg.seed,
        calibration_cv=cfg.calibration.cv,
        smallest_class_count=_smallest(y_train) if task.classes is not None else None,
        transform=task.transform,
    )


def _predict_original(
    task: TargetTask, model: Classifier | Regressor, X: pd.DataFrame
) -> npt.NDArray[np.float64]:
    if task.classes is not None:
        return aligned_proba(model, X, len(task.classes))  # type: ignore[arg-type]
    return np.asarray(model.predict(X), dtype=np.float64)  # type: ignore[union-attr]


def _evaluate_fold(
    task: TargetTask,
    candidate: CandidateSpec,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
    X: pd.DataFrame,
    y: npt.NDArray[Any],
    y_fit: npt.NDArray[Any],
    train: npt.NDArray[np.intp],
    test: npt.NDArray[np.intp],
) -> float:
    model = _build(task, candidate, features, cfg, y_fit[train])
    _fit_model(task, model, X.iloc[train], y_fit[train])
    return score_candidate(task, cfg, y[test], _predict_original(task, model, X.iloc[test]))


# --------------------------------------------------------------------------------------
# Selection and cross-fitting
# --------------------------------------------------------------------------------------


def select(
    task: TargetTask,
    data: CanonicalData,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
) -> tuple[CandidateSpec, dict[str, float]]:
    """Inner-CV model selection on the task's rows.

    Returns the chosen candidate and the mean score of every candidate. With
    ``cfg.selection.one_se_rule``, the simplest candidate within one standard error of the
    best is chosen. Isotonic calibration is skipped below ``ISOTONIC_MIN_ROWS`` rows.
    """
    X, y, groups = _task_arrays(task, data, features)
    y_fit = fit_labels(task, data, cfg, y)
    candidates = [
        c
        for c in enumerate_candidates(task.kind, cfg)
        if c.calibration != "isotonic" or len(y) >= ISOTONIC_MIN_ROWS
    ]
    splits = list(make_splitter(_strata(task, y), groups, cfg.cv.inner_splits, cfg.seed))
    jobs = [(c, train, test) for c in candidates for train, test in splits]
    scores = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_evaluate_fold)(task, c, features, cfg, X, y, y_fit, train, test)
        for c, train, test in jobs
    )
    per_candidate: dict[str, list[float]] = {c.name: [] for c in candidates}
    for (candidate, _, _), score in zip(jobs, scores, strict=True):
        per_candidate[candidate.name].append(float(score))
    means = {name: float(np.mean(values)) for name, values in per_candidate.items()}
    by_name = {c.name: c for c in candidates}
    best = max(candidates, key=lambda c: (means[c.name], -c.complexity))
    chosen = best
    if cfg.selection.one_se_rule and len(splits) > 1:
        values = np.asarray(per_candidate[best.name])
        se = float(values.std(ddof=1) / math.sqrt(values.size))
        eligible = [c for c in candidates if means[c.name] >= means[best.name] - se]
        chosen = min(eligible, key=lambda c: (c.complexity, -means[c.name]))
    _LOGGER.info(
        "Cible %s : %d candidats x %d plis, retenu %s.",
        task.name.value,
        len(candidates),
        len(splits),
        by_name[chosen.name].name,
    )
    return chosen, means


def _fit_fold(
    task: TargetTask,
    candidate: CandidateSpec,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
    X: pd.DataFrame,
    y_fit: npt.NDArray[Any],
    train: npt.NDArray[np.intp],
    test: npt.NDArray[np.intp],
) -> tuple[Classifier | Regressor, npt.NDArray[np.float64]]:
    model = _build(task, candidate, features, cfg, y_fit[train])
    _fit_model(task, model, X.iloc[train], y_fit[train])
    X_test = X.iloc[test]
    if task.classes is not None:
        return model, aligned_proba(model, X_test, len(task.classes))  # type: ignore[arg-type]
    return model, _regression_model_space(task, model, X_test)  # type: ignore[arg-type]


def cross_fit(
    task: TargetTask,
    candidate: CandidateSpec,
    data: CanonicalData,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
) -> CrossFit:
    """Fit ``cfg.cv.crossfit_splits`` fold models and collect OOF outputs."""
    X, y, groups = _task_arrays(task, data, features)
    y_fit = fit_labels(task, data, cfg, y)
    splits = list(make_splitter(_strata(task, y), groups, cfg.cv.crossfit_splits, cfg.seed + 1))
    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_fit_fold)(task, candidate, features, cfg, X, y_fit, train, test)
        for train, test in splits
    )
    fold_ids = np.zeros(len(y), dtype=np.intp)
    oof = np.zeros((len(y), len(task.classes)) if task.classes is not None else len(y))
    models: list[Classifier | Regressor] = []
    for fold, ((train, test), (model, output)) in enumerate(zip(splits, results, strict=True)):
        del train
        fold_ids[test] = fold
        oof[test] = output
        models.append(model)
    return CrossFit(fold_models=tuple(models), fold_ids=fold_ids, oof=oof.astype(np.float64))


def choose_threshold(
    y_positive: npt.NDArray[np.bool_], p_positive: npt.NDArray[np.float64], cfg: ThresholdConfig
) -> float:
    """Decision threshold on P(positive) from out-of-fold probabilities (``ThresholdConfig``).

    ``target_sensitivity``: the highest threshold whose sensitivity reaches ``value``;
    ``youden``: the threshold maximising sensitivity + specificity - 1; ``fixed``: ``value``.
    Without positives or negatives, falls back to 0.5.
    """
    if cfg.strategy == "fixed":
        return float(cfg.value)
    positive = np.asarray(y_positive, dtype=bool)
    p = np.asarray(p_positive, dtype=np.float64)
    if positive.all() or not positive.any():
        return 0.5
    thresholds = np.unique(p)[::-1]
    if cfg.strategy == "target_sensitivity":
        for threshold in thresholds:
            if float((p[positive] >= threshold).mean()) >= cfg.value:
                return float(threshold)
        return float(thresholds[-1])
    best, best_j = 0.5, -math.inf
    for threshold in thresholds:
        predicted = p >= threshold
        j = float(predicted[positive].mean() - predicted[~positive].mean())
        if j > best_j:
            best, best_j = float(threshold), j
    return best


# --------------------------------------------------------------------------------------
# Fitted target
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class FittedTarget:
    """Everything needed to predict one target.

    ``conformal`` is ``None`` for a constant task. ``threshold`` is only set for
    ``treatment`` (decision threshold on P(yes)). ``selection_scores`` maps candidate names
    to their mean inner-CV score.
    """

    task: TargetTask
    candidate: CandidateSpec
    crossfit: CrossFit
    conformal: CrossConformalClassifier | CVPlusRegressor | None
    threshold: float | None = None
    selection_scores: Mapping[str, float] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Mean fold probabilities, shape ``(n, C)`` in ``task.classes`` order."""
        mean: npt.NDArray[np.float64] = self.fold_predictions(X).mean(axis=0)
        return mean

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Regression point predictions in original units, shape ``(n,)``."""
        return original_units(self.task, self.fold_predictions(X).mean(axis=0))

    def fold_predictions(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Per-fold outputs: ``(K, n, C)`` probabilities or ``(K, n)`` model-space values."""
        if self.task.classes is not None:
            n_classes = len(self.task.classes)
            return np.stack(
                [aligned_proba(m, X, n_classes) for m in self.crossfit.fold_models]  # type: ignore[arg-type]
            )
        return np.stack(
            [_regression_model_space(self.task, m, X) for m in self.crossfit.fold_models]  # type: ignore[arg-type]
        )


def fit_target(
    task: TargetTask,
    data: CanonicalData,
    features: Sequence[FeatureSpec],
    cfg: TrainingConfig,
) -> FittedTarget:
    """Selection, then cross-fit, then conformal fit, then (treatment only) the threshold."""
    if task.is_constant:
        n = task.n_rows
        constant = CrossFit(
            fold_models=(ConstantClassifier(),),
            fold_ids=np.zeros(n, dtype=np.intp),
            oof=np.ones((n, 1), dtype=np.float64),
        )
        spec = CandidateSpec(family="constant", task_kind=task.kind, params=())
        return FittedTarget(task=task, candidate=spec, crossfit=constant, conformal=None)
    candidate, scores = select(task, data, features, cfg)
    crossfit = cross_fit(task, candidate, data, features, cfg)
    _, y, _ = _task_arrays(task, data, features)
    conformal: CrossConformalClassifier | CVPlusRegressor
    threshold: float | None = None
    if task.classes is not None:
        y_idx = np.asarray(y, dtype=np.intp)
        conformal = CrossConformalClassifier(
            cfg.conformal.alpha, class_conditional=cfg.conformal.class_conditional
        ).fit(y_idx, crossfit.oof, crossfit.fold_ids)
        if task.name is TargetName.TREATMENT and task.positive_class is not None:
            positive = task.classes.index(task.positive_class)
            threshold = choose_threshold(
                y_idx == positive, crossfit.oof[:, positive], cfg.treatment_threshold
            )
    else:
        conformal = CVPlusRegressor(cfg.conformal.alpha).fit(
            model_space(task, np.asarray(y, dtype=np.float64)), crossfit.oof, crossfit.fold_ids
        )
    return FittedTarget(
        task=task,
        candidate=candidate,
        crossfit=crossfit,
        conformal=conformal,
        threshold=threshold,
        selection_scores=scores,
    )
