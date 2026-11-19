"""Out-of-distribution detection against the training data. Owner: WS3.

Flag codes (``Flag.feature`` = canonical feature name, no values in messages):

* ``ood.below_range`` / ``ood.above_range`` (warning): outside the training
  ``range_quantiles`` interval but inside the training min/max;
* ``ood.beyond_training`` (critical): outside the training min/max;
* ``ood.unseen_category`` (warning): category never seen in training;
* ``ood.missing_required`` (critical): a ``required`` feature is missing;
* ``ood.novelty`` (warning): LOF score below the training ``novelty_quantile``.

Range checks apply to numeric features only (a binary value is always 0 or 1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from sevrage_ai.config.schema import FeatureSpec
from sevrage_ai.config.training import OODConfig
from sevrage_ai.types import FeatureKind, Flag

__all__ = [
    "OOD_ABOVE_RANGE",
    "OOD_BELOW_RANGE",
    "OOD_BEYOND_TRAINING",
    "OOD_MISSING_REQUIRED",
    "OOD_NOVELTY",
    "OOD_UNSEEN_CATEGORY",
    "FeatureRange",
    "NoveltyScorer",
    "OODReference",
    "detect",
    "fit_reference",
    "missing_fraction",
]

OOD_BELOW_RANGE: Final = "ood.below_range"
OOD_ABOVE_RANGE: Final = "ood.above_range"
OOD_BEYOND_TRAINING: Final = "ood.beyond_training"
OOD_UNSEEN_CATEGORY: Final = "ood.unseen_category"
OOD_MISSING_REQUIRED: Final = "ood.missing_required"
OOD_NOVELTY: Final = "ood.novelty"

_MIN_NOVELTY_ROWS: Final = 10
_SCALE_EPS: Final = 1e-9


class NoveltyScorer(Protocol):
    """Fitted novelty model (``LocalOutlierFactor(novelty=True)``): higher = more normal."""

    def score_samples(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Novelty score of each row of the prepared matrix."""
        ...


@dataclass(frozen=True, slots=True)
class FeatureRange:
    """Training distribution summary of one numeric feature."""

    q_low: float
    q_high: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True, eq=False)
class OODReference:
    """Everything needed to flag a new patient.

    Novelty scoring runs on the numeric and binary ``novelty_columns``, imputed with
    ``novelty_center`` and divided by ``novelty_scale``. ``novelty_model`` and
    ``novelty_threshold`` are ``None`` when novelty detection is disabled (or the training
    set is too small). ``labels`` gives the French label of each feature (messages).
    """

    ranges: Mapping[str, FeatureRange]
    categories: Mapping[str, frozenset[str]]
    required: tuple[str, ...]
    novelty_columns: tuple[str, ...]
    novelty_center: npt.NDArray[np.float64]
    novelty_scale: npt.NDArray[np.float64]
    novelty_model: NoveltyScorer | None
    novelty_threshold: float | None
    labels: Mapping[str, str] = field(default_factory=dict)


def _novelty_matrix(
    X: pd.DataFrame,
    columns: Sequence[str],
    center: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # ``reindex``: a column absent from ``X`` counts as missing (imputed with the centre),
    # like the range and category checks, which skip absent features.
    values = X.reindex(columns=list(columns)).to_numpy(dtype=np.float64, na_value=np.nan)
    values = np.where(np.isnan(values), center[np.newaxis, :], values)
    scaled: npt.NDArray[np.float64] = (values - center) / scale
    return scaled


def fit_reference(
    X: pd.DataFrame, features: Sequence[FeatureSpec], cfg: OODConfig, *, seed: int
) -> OODReference:
    """Summarise the training predictors ``X[features]``. (``seed`` is kept for API
    stability; LOF is deterministic.)"""
    del seed
    q_low, q_high = cfg.range_quantiles
    ranges: dict[str, FeatureRange] = {}
    categories: dict[str, frozenset[str]] = {}
    novelty_columns: list[str] = []
    for feature in features:
        column = X[feature.name]
        if feature.kind is FeatureKind.CATEGORICAL:
            categories[feature.name] = frozenset(str(v) for v in column.dropna().unique())
            continue
        values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        novelty_columns.append(feature.name)
        if feature.kind is FeatureKind.NUMERIC:
            ranges[feature.name] = FeatureRange(
                q_low=float(np.quantile(values, q_low)),
                q_high=float(np.quantile(values, q_high)),
                minimum=float(values.min()),
                maximum=float(values.max()),
            )
    center = np.array(
        [
            float(np.nanmedian(X[c].to_numpy(dtype=np.float64, na_value=np.nan)))
            for c in novelty_columns
        ],
        dtype=np.float64,
    )
    scale = np.array(
        [_robust_scale(X[c].to_numpy(dtype=np.float64, na_value=np.nan)) for c in novelty_columns],
        dtype=np.float64,
    )
    model: LocalOutlierFactor | None = None
    threshold: float | None = None
    if cfg.novelty == "lof" and len(X) >= _MIN_NOVELTY_ROWS and novelty_columns:
        matrix = _novelty_matrix(X, novelty_columns, center, scale)
        model = LocalOutlierFactor(n_neighbors=min(cfg.lof_n_neighbors, len(X) - 1), novelty=True)
        model.fit(matrix)
        threshold = float(np.quantile(model.negative_outlier_factor_, cfg.novelty_quantile))
    return OODReference(
        ranges=ranges,
        categories=categories,
        required=tuple(f.name for f in features if f.required),
        novelty_columns=tuple(novelty_columns),
        novelty_center=center,
        novelty_scale=scale,
        novelty_model=model,
        novelty_threshold=threshold,
        labels={f.name: f.label for f in features},
    )


def _robust_scale(values: npt.NDArray[np.float64]) -> float:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return 1.0
    q75, q25 = np.quantile(finite, [0.75, 0.25])
    spread = float(q75 - q25)
    if spread < _SCALE_EPS:
        spread = float(finite.std())
    return spread if spread > _SCALE_EPS else 1.0


def _range_flag(name: str, label: str, value: float, rng: FeatureRange) -> Flag | None:
    if value < rng.minimum or value > rng.maximum:
        side = "inférieure" if value < rng.minimum else "supérieure"
        return Flag(
            OOD_BEYOND_TRAINING,
            "critical",
            f"« {label} » : valeur {side} à toutes celles de l'échantillon d'apprentissage.",
            name,
        )
    if value < rng.q_low:
        return Flag(
            OOD_BELOW_RANGE, "warning", f"« {label} » : valeur inhabituellement basse.", name
        )
    if value > rng.q_high:
        return Flag(
            OOD_ABOVE_RANGE, "warning", f"« {label} » : valeur inhabituellement haute.", name
        )
    return None


def detect(X: pd.DataFrame, ref: OODReference) -> list[list[Flag]]:
    """Flags for each row of ``X`` (same order), possibly empty lists."""
    flags: list[list[Flag]] = [[] for _ in range(len(X))]
    for name, rng in ref.ranges.items():
        if name not in X:
            continue
        label = ref.labels.get(name, name)
        values = X[name].to_numpy(dtype=np.float64, na_value=np.nan)
        for i, value in enumerate(values):
            if not math.isnan(value) and (flag := _range_flag(name, label, value, rng)):
                flags[i].append(flag)
    for name, seen in ref.categories.items():
        if name not in X:
            continue
        label = ref.labels.get(name, name)
        for i, value in enumerate(X[name].tolist()):
            if isinstance(value, str) and value not in seen:
                flags[i].append(
                    Flag(
                        OOD_UNSEEN_CATEGORY, "warning", f"« {label} » : modalité jamais vue.", name
                    )
                )
    for name in ref.required:
        missing = X[name].isna().to_numpy(dtype=bool) if name in X else np.ones(len(X), bool)
        label = ref.labels.get(name, name)
        for row in np.flatnonzero(missing):
            flags[int(row)].append(
                Flag(
                    OOD_MISSING_REQUIRED,
                    "critical",
                    f"« {label} » : variable requise manquante.",
                    name,
                )
            )
    if ref.novelty_model is not None and ref.novelty_threshold is not None and len(X):
        matrix = _novelty_matrix(X, ref.novelty_columns, ref.novelty_center, ref.novelty_scale)
        scores = ref.novelty_model.score_samples(matrix)
        for row in np.flatnonzero(scores < ref.novelty_threshold):
            flags[int(row)].append(
                Flag(
                    OOD_NOVELTY,
                    "warning",
                    "Profil global inhabituel par rapport aux patients d'apprentissage.",
                )
            )
    return flags


def missing_fraction(X: pd.DataFrame, features: Sequence[FeatureSpec]) -> npt.NDArray[np.float64]:
    """Per-row share of missing values among ``features``, shape ``(n,)``."""
    if not features:
        return np.zeros(len(X), dtype=np.float64)
    names = [f.name for f in features]
    present = X.reindex(columns=names)
    fraction: npt.NDArray[np.float64] = present.isna().to_numpy(dtype=bool).mean(axis=1)
    return fraction.astype(np.float64)
