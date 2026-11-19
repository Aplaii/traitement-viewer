"""The whole trained system: one fitted target per enabled target plus references. Owner: WS4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from sevrage_ai.confidence.ood import OODReference, fit_reference
from sevrage_ai.config.schema import DatasetSchema, FeatureSpec, schema_hash
from sevrage_ai.config.training import TrainingConfig
from sevrage_ai.errors import DataValidationError
from sevrage_ai.logging_config import get_logger
from sevrage_ai.models.tasks import build_tasks, encode_labels
from sevrage_ai.models.training import FittedTarget, fit_target
from sevrage_ai.types import FLOAT_DTYPE, STRING_DTYPE, CanonicalData, FeatureKind, TargetName

__all__ = ["MIN_TASK_ROWS", "TrainedSystem", "compute_reference_patient", "train_all"]

MIN_TASK_ROWS: Final = 10
"""A target needs at least this many labelled rows to be trained."""

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class TrainedSystem:
    """What ``model.joblib`` contains (with the schema and the training config).

    ``features`` are the predictors in model column order. ``reference_patient`` is a
    one-row frame of canonical dtypes (medians for numeric features, modes otherwise) used
    for occlusion explanations. ``classes`` holds the frozen class order of every class
    target. ``class_counts`` holds the training counts per target and class.
    """

    targets: Mapping[TargetName, FittedTarget]
    features: tuple[FeatureSpec, ...]
    feature_set: str
    ood_reference: OODReference
    reference_patient: pd.DataFrame
    classes: Mapping[TargetName, tuple[str, ...]]
    class_counts: Mapping[TargetName, Mapping[str, int]]
    n_train: int
    is_synthetic: bool
    seed: int
    schema_hash: str


def compute_reference_patient(X: pd.DataFrame, features: Sequence[FeatureSpec]) -> pd.DataFrame:
    """One-row frame: median (numeric), mode (binary, categorical), canonical dtypes.

    A binary feature that is missing for most rows (an OUI / NON / NON CONNU question that
    is usually not known at admission) gets a missing reference: the explanations then
    compare the patient with "not known", as the model sees it at admission.
    """
    columns: dict[str, pd.Series[float] | pd.Series[str]] = {}
    for feature in features:
        column = X[feature.name]
        if feature.kind is FeatureKind.CATEGORICAL:
            modes = column.dropna().mode()
            value = str(sorted(modes.tolist())[0]) if len(modes) else None
            columns[feature.name] = pd.Series([value], dtype=STRING_DTYPE)
            continue
        all_values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        values = all_values[~np.isnan(all_values)]
        mostly_unknown = feature.kind is FeatureKind.BINARY and 2 * values.size < all_values.size
        if values.size == 0 or mostly_unknown:
            reference = np.nan
        elif feature.kind is FeatureKind.BINARY:
            reference = float(values.mean() > 0.5)
        else:
            reference = float(np.median(values))
        columns[feature.name] = pd.Series([reference], dtype=FLOAT_DTYPE)
    return pd.DataFrame(columns, index=pd.RangeIndex(1))


def train_all(data: CanonicalData, schema: DatasetSchema, cfg: TrainingConfig) -> TrainedSystem:
    """Fit every enabled target on ``data`` with ``cfg.feature_set`` (final model, D4).

    Raises:
        DataValidationError: no labels, or a target with fewer than :data:`MIN_TASK_ROWS`
            labelled rows.
    """
    if data.y is None:
        msg = "Données sans cibles : entraînement impossible."
        raise DataValidationError(msg)
    features = tuple(schema.predictors(cfg.feature_set))
    tasks = build_tasks(schema, data.y)
    too_small = [t.name.value for t in tasks if t.n_rows < MIN_TASK_ROWS]
    if too_small:
        msg = "Trop peu de lignes étiquetées pour entraîner certaines cibles."
        raise DataValidationError(msg, details=[f"{name} < {MIN_TASK_ROWS}" for name in too_small])
    fitted: dict[TargetName, FittedTarget] = {}
    for task in tasks:
        _LOGGER.info("Entraînement de la cible %s (%d lignes).", task.name.value, task.n_rows)
        fitted[task.name] = fit_target(task, data, features, cfg)
    X = data.X.loc[:, [f.name for f in features]]
    classes = {t.name: t.classes for t in tasks if t.classes is not None}
    counts: dict[TargetName, dict[str, int]] = {}
    for task in tasks:
        if task.classes is None:
            continue
        encoded = encode_labels(task, data.y[task.name.value])
        counts[task.name] = {c: int((encoded == i).sum()) for i, c in enumerate(task.classes)}
    return TrainedSystem(
        targets=fitted,
        features=features,
        feature_set=cfg.feature_set,
        ood_reference=fit_reference(X, features, cfg.ood, seed=cfg.seed),
        reference_patient=compute_reference_patient(X, features),
        classes=classes,
        class_counts=counts,
        n_train=data.n_rows,
        is_synthetic=data.is_synthetic,
        seed=cfg.seed,
        schema_hash=schema_hash(schema),
    )
