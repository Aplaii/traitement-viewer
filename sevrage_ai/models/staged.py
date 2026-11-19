"""Two-stage classifier: a gate, then the classes among the gated rows.

Used for the severity of the withdrawal (``TrainingConfig.two_stage``). In the data, a
withdrawal that is not treated is almost always mild, while a treated one may be mild,
moderate or severe: "mild" mixes two populations. The model is therefore split in two:

* the *gate* model gives ``P(gate)`` (treated or not), learned on the rows whose gate is
  known;
* the *stage* model gives ``P(class | gate)``, learned on the gated rows only.

``P(class) = P(gate) * P(class | gate)``, and the rest ``1 - P(gate)`` goes to the
``otherwise`` class (mild). Rows whose gate is unknown are used by neither model.

Labels travel through ``fit(X, y)`` as *joint codes* (:func:`joint_codes`), so that the
classifier keeps the scikit-learn API of the other candidates.
"""

from __future__ import annotations

from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "GATE_CLOSED",
    "GATE_OPEN",
    "GATE_UNKNOWN",
    "TwoStageClassifier",
    "gate_of",
    "joint_codes",
    "label_of",
]

GATE_CLOSED: Final = 0
GATE_OPEN: Final = 1
GATE_UNKNOWN: Final = 2


def joint_codes(
    labels: npt.NDArray[np.intp], gate: npt.NDArray[np.intp], n_classes: int
) -> npt.NDArray[np.intp]:
    """One code per row holding the class index and the gate (``GATE_*``)."""
    codes: npt.NDArray[np.intp] = (labels + n_classes * gate).astype(np.intp)
    return codes


def label_of(codes: npt.NDArray[np.intp], n_classes: int) -> npt.NDArray[np.intp]:
    """Class index of each joint code."""
    labels: npt.NDArray[np.intp] = (codes % n_classes).astype(np.intp)
    return labels


def gate_of(codes: npt.NDArray[np.intp], n_classes: int) -> npt.NDArray[np.intp]:
    """Gate (``GATE_*``) of each joint code."""
    gate: npt.NDArray[np.intp] = (codes // n_classes).astype(np.intp)
    return gate


class _Probabilistic(Protocol):
    """The two inner models (same API as ``models.candidates.Classifier``)."""

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.intp]) -> object: ...

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]: ...


def _aligned(model: _Probabilistic, X: pd.DataFrame, n_columns: int) -> npt.NDArray[np.float64]:
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])), dtype=np.intp)
    out = np.zeros((len(X), n_columns), dtype=np.float64)
    out[:, classes] = raw
    return out


class TwoStageClassifier:
    """``P(class) = P(gate) * P(class | gate)``, the rest going to class ``otherwise``.

    ``gate_model`` and ``stage_model`` are unfitted probabilistic classifiers. ``fit``
    takes joint codes; ``predict_proba`` has one column per class (``classes_`` is
    ``0..n_classes-1``).
    """

    def __init__(
        self,
        gate_model: _Probabilistic,
        stage_model: _Probabilistic,
        *,
        n_classes: int,
        otherwise: int,
    ) -> None:
        self.gate_model = gate_model
        self.stage_model = stage_model
        self.n_classes = n_classes
        self.otherwise = otherwise
        self.classes_ = np.arange(n_classes, dtype=np.intp)
        self.gate_rate_: float | None = None

    def fit(self, X: pd.DataFrame, y: npt.NDArray[np.intp]) -> TwoStageClassifier:
        """Fit the gate on the rows of known gate and the stage on the gated rows."""
        codes = np.asarray(y, dtype=np.intp)
        labels, gate = label_of(codes, self.n_classes), gate_of(codes, self.n_classes)
        known = gate != GATE_UNKNOWN
        opened = gate == GATE_OPEN
        if np.unique(gate[known]).size < 2:
            # Every known row has the same gate: its rate replaces the gate model.
            self.gate_rate_ = float(opened[known].mean()) if known.any() else 1.0
        else:
            self.gate_rate_ = None
            self.gate_model.fit(X.loc[known], gate[known])
        self.stage_model.fit(X.loc[opened], labels[opened])
        return self

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Probabilities, shape ``(n, n_classes)``."""
        if self.gate_rate_ is not None:
            gate = np.full(len(X), self.gate_rate_, dtype=np.float64)
        else:
            gate = _aligned(self.gate_model, X, 2)[:, GATE_OPEN]
        proba: npt.NDArray[np.float64] = _aligned(self.stage_model, X, self.n_classes)
        proba *= gate[:, None]
        proba[:, self.otherwise] += 1.0 - gate
        return proba
