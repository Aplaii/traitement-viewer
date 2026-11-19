"""Target tasks: which rows, which classes, which encoding. Owner: WS4.

Gating (D6): the ``treatment`` and ``severity`` tasks use every row with a label. The
``molecule``, dose and duration tasks only use rows whose ``conditional_on`` holds
(treated patients) and whose label is present. For a target whose classes are inferred
from the data (``molecule``), classes rarer than ``min_class_count`` are merged into
``"other"``; if ``"other"`` itself stays below ``min_class_count``, its rows are left out
of the task (``model.rare_class_dropped`` flag). If only one class remains, the task is
*constant* (``constant_class`` is set) and carries a ``model.constant_task`` warning flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from sevrage_ai.config.schema import DatasetSchema, TargetSpec
from sevrage_ai.types import OTHER_CLASS, Flag, TargetName, TaskKind

__all__ = [
    "TargetTask",
    "build_tasks",
    "encode_labels",
    "predicted_indices",
    "regression_values",
]


@dataclass(frozen=True, slots=True, eq=False)
class TargetTask:
    """One supervised task derived from the schema and the training labels.

    ``classes`` is the frozen label order after merging (``None`` for regression).
    ``row_mask`` has one entry per ``CanonicalData`` row: ``True`` if the row is used by this
    task. ``merged_classes`` maps original labels to their merged label (e.g.
    ``{"chlordiazepoxide": "other"}``). ``class_priors`` holds the Laplace-smoothed class
    prevalence of the task rows, in ``classes`` order (``None`` for regression).
    """

    name: TargetName
    kind: TaskKind
    classes: tuple[str, ...] | None
    row_mask: npt.NDArray[np.bool_]
    positive_class: str | None = None
    transform: Literal["none", "log1p"] = "none"
    unit: str | None = None
    merged_classes: Mapping[str, str] = field(default_factory=dict)
    constant_class: str | None = None
    flags: tuple[Flag, ...] = ()
    class_priors: tuple[float, ...] | None = None

    @property
    def is_constant(self) -> bool:
        """``True`` when only one class remains (a constant model is used)."""
        return self.constant_class is not None

    @property
    def n_rows(self) -> int:
        """Number of rows used by this task."""
        return int(self.row_mask.sum())


def _condition_mask(y: pd.DataFrame, spec: TargetSpec) -> npt.NDArray[np.bool_]:
    condition = spec.conditional_on
    if condition is None:
        return np.ones(len(y), dtype=bool)
    if condition.target.value not in y:
        return np.zeros(len(y), dtype=bool)
    holds = (y[condition.target.value] == condition.equals).fillna(value=False)
    return holds.to_numpy(dtype=bool)


def _class_task(
    name: TargetName, spec: TargetSpec, labels: pd.Series[str], mask: npt.NDArray[np.bool_]
) -> TargetTask:
    observed = labels[mask]
    flags: list[Flag] = []
    merged: dict[str, str] = {}
    if spec.classes is not None:
        classes = tuple(spec.classes)
    else:
        counts = observed.value_counts()
        # A label that already reads "other" belongs to the "other" pool: otherwise the
        # class order would hold "other" twice once rare classes are merged into it.
        rare = sorted(
            str(c)
            for c, k in counts.items()
            if int(k) < spec.min_class_count or str(c) == OTHER_CLASS
        )
        merged = {label: OTHER_CLASS for label in rare if label != OTHER_CLASS}
        kept = sorted(str(c) for c in counts.index if str(c) not in rare)
        other_count = int(sum(int(counts[c]) for c in rare))
        if other_count >= spec.min_class_count:
            kept.append(OTHER_CLASS)
        elif other_count:
            mask = mask & ~labels.isin(rare).fillna(value=False).to_numpy(dtype=bool)
            flags.append(
                Flag(
                    "model.rare_class_dropped",
                    "info",
                    f"« {spec.label} » : {other_count} cas de classes trop rares "
                    "écartés de l'apprentissage.",
                )
            )
        classes = tuple(kept)
    constant = None
    if len(classes) == 1:
        constant = classes[0]
        flags.append(
            Flag(
                "model.constant_task",
                "warning",
                f"« {spec.label} » : une seule classe observée, prédiction constante.",
            )
        )
    final = labels[mask].astype(object).map(lambda v: merged.get(str(v), str(v)))
    class_counts = np.array([float((final == c).sum()) for c in classes])
    priors = tuple(float(v) for v in (class_counts + 1.0) / (class_counts.sum() + len(classes)))
    return TargetTask(
        name=name,
        kind=spec.kind,
        classes=classes,
        row_mask=mask,
        positive_class=spec.positive_class,
        merged_classes=merged,
        constant_class=constant,
        flags=tuple(flags),
        class_priors=priors,
    )


def build_tasks(schema: DatasetSchema, y: pd.DataFrame) -> list[TargetTask]:
    """One task per enabled target, in :class:`~sevrage_ai.types.TargetName` order."""
    tasks: list[TargetTask] = []
    for name, spec in schema.enabled_targets().items():
        labels = y[name.value]
        mask = labels.notna().to_numpy(dtype=bool) & _condition_mask(y, spec)
        if spec.kind.is_classification:
            tasks.append(_class_task(name, spec, labels, mask))
        else:
            tasks.append(
                TargetTask(
                    name=name,
                    kind=spec.kind,
                    classes=None,
                    row_mask=mask,
                    transform=spec.transform,
                    unit=spec.unit,
                )
            )
    return tasks


def predicted_indices(
    proba: npt.NDArray[np.float64],
    task: TargetTask,
    *,
    threshold: float | None = None,
    prior_corrected: bool = False,
) -> npt.NDArray[np.intp]:
    """Predicted class index per row of ``proba`` (``(n, C)``, ``task.classes`` order).

    The one decision rule shared by inference and evaluation:

    * binary task with a positive class and a ``threshold``: the positive class iff its
      probability is at least ``threshold`` (treatment);
    * ``prior_corrected`` (see ``config.training.DecisionConfig``): argmax of
      ``p_k / task.class_priors[k]``;
    * otherwise the argmax.
    """
    classes = task.classes or ()
    if threshold is not None and task.positive_class is not None and len(classes) == 2:
        positive = classes.index(task.positive_class)
        decided: npt.NDArray[np.intp] = np.where(
            proba[:, positive] >= threshold, positive, 1 - positive
        ).astype(np.intp)
        return decided
    scores = proba
    if prior_corrected and task.class_priors is not None:
        scores = proba / np.asarray(task.class_priors, dtype=np.float64)
    return np.argmax(scores, axis=1).astype(np.intp)


def encode_labels(task: TargetTask, y: pd.Series[str]) -> npt.NDArray[np.intp]:
    """Encode the labels of the task's rows (``y[task.row_mask]``) as indices into ``classes``."""
    if task.classes is None:
        msg = f"task {task.name.value} is not a classification task"
        raise ValueError(msg)
    index = {label: i for i, label in enumerate(task.classes)}
    labels = [task.merged_classes.get(str(v), str(v)) for v in y[task.row_mask].tolist()]
    return np.array([index[label] for label in labels], dtype=np.intp)


def regression_values(task: TargetTask, y: pd.Series[float]) -> npt.NDArray[np.float64]:
    """Values of the task's rows (``y[task.row_mask]``) in original units."""
    values: npt.NDArray[np.float64] = np.asarray(y[task.row_mask], dtype=np.float64)
    return values
