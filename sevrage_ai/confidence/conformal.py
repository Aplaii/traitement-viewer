"""In-house conformal prediction (D5): cross-conformal sets and CV+ intervals. Owner: WS3.

Classification (LAC score ``s = 1 - p_y``)
    Calibration point ``i`` (held out in fold ``k(i)``) has score ``s_i`` computed from its
    OOF probability. For a test point with per-fold probabilities ``p^k``, label ``y`` is in
    the set iff ``#{i : s_i < 1 - p^{k(i)}_y} < (1 - alpha)(n + 1)``. With
    ``class_conditional=True`` (Mondrian), the calibration points are restricted to those
    of class ``y`` (and ``n`` becomes the class count).

Regression (CV+, in model space)
    ``lower`` = the ``floor(alpha (n+1))``-th smallest of ``mu_{k(i)}(x) - R_i``;
    ``upper`` = the ``ceil((1-alpha)(n+1))``-th smallest of ``mu_{k(i)}(x) + R_i``, where
    ``R_i = |y_i - oof_i|``. An order statistic outside ``1..n`` gives ``-inf`` / ``+inf``.

References: Barber et al. (2021), "Predictive inference with the jackknife+"; Vovk (2015),
"Cross-conformal predictors".
"""

from __future__ import annotations

import math
from typing import Literal, Self

import numpy as np
import numpy.typing as npt

__all__ = [
    "CVPlusRegressor",
    "CrossConformalClassifier",
    "interval_coverage",
    "mean_interval_width",
    "mean_set_size",
    "set_coverage",
    "singleton_rate",
]


def _check_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie in (0, 1)"
        raise ValueError(msg)
    return alpha


def _check_fold_ids(folds: npt.NDArray[np.intp]) -> None:
    if folds.size and int(folds.min()) < 0:
        msg = "fold_ids must be non-negative fold indices"
        raise ValueError(msg)


def _check_models_cover_folds(n_models: int, folds: npt.NDArray[np.intp]) -> None:
    # Every calibration point needs the model of the fold it was held out in; a silent
    # mismatch would drop calibration points and inflate the sets / intervals.
    if n_models <= int(folds.max()):
        msg = "per-fold outputs must have one entry per calibration fold (K > max fold id)"
        raise ValueError(msg)


class CrossConformalClassifier:
    """Cross-conformal prediction sets for a K-fold ensemble."""

    def __init__(
        self,
        alpha: float,
        score: Literal["lac"] = "lac",
        *,
        class_conditional: bool = False,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        self.score = score
        self.class_conditional = class_conditional
        self._scores: npt.NDArray[np.float64] | None = None
        self._folds: npt.NDArray[np.intp] | None = None
        self._labels: npt.NDArray[np.intp] | None = None
        self._n_classes = 0

    def fit(
        self,
        y_idx: npt.NDArray[np.intp],
        oof_proba: npt.NDArray[np.float64],
        fold_ids: npt.NDArray[np.intp],
    ) -> Self:
        """Store the calibration scores. ``oof_proba`` has shape ``(n, C)``."""
        y = np.asarray(y_idx, dtype=np.intp)
        proba = np.asarray(oof_proba, dtype=np.float64)
        folds = np.asarray(fold_ids, dtype=np.intp)
        if (
            y.ndim != 1
            or folds.ndim != 1
            or proba.ndim != 2
            or proba.shape[0] != y.size
            or folds.size != y.size
            or y.size == 0
        ):
            msg = "fit expects y (n,), oof_proba (n, C) and fold_ids (n,) with n > 0"
            raise ValueError(msg)
        if int(y.min()) < 0 or int(y.max()) >= proba.shape[1]:
            msg = "y must hold class indices in 0..C-1"
            raise ValueError(msg)
        _check_fold_ids(folds)
        self._scores = 1.0 - proba[np.arange(y.size), y]
        self._folds = folds
        self._labels = y
        self._n_classes = proba.shape[1]
        return self

    def predict_set(self, fold_proba: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Boolean membership, shape ``(m, C)``, from per-fold probabilities ``(K, m, C)``."""
        if self._scores is None or self._folds is None or self._labels is None:
            msg = "CrossConformalClassifier is not fitted"
            raise RuntimeError(msg)
        proba = np.asarray(fold_proba, dtype=np.float64)
        if proba.ndim != 3 or proba.shape[2] != self._n_classes:
            msg = "predict_set expects fold probabilities of shape (K, m, C)"
            raise ValueError(msg)
        n_models, m, n_classes = proba.shape
        _check_models_cover_folds(n_models, self._folds)
        test_scores = 1.0 - proba
        counts = np.zeros((m, n_classes), dtype=np.float64)
        limits = np.full(n_classes, (1.0 - self.alpha) * (self._scores.size + 1))
        for label in range(n_classes):
            members = (
                self._labels == label if self.class_conditional else np.ones_like(self._folds, bool)
            )
            if self.class_conditional:
                limits[label] = (1.0 - self.alpha) * (int(members.sum()) + 1)
            for fold in range(n_models):
                calibration = np.sort(self._scores[members & (self._folds == fold)])
                counts[:, label] += np.searchsorted(
                    calibration, test_scores[fold, :, label], "left"
                )
        included: npt.NDArray[np.bool_] = counts < limits[np.newaxis, :]
        return included


class CVPlusRegressor:
    """CV+ prediction intervals for a K-fold ensemble (theoretical coverage 1 - 2 alpha)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = _check_alpha(alpha)
        self._residuals: npt.NDArray[np.float64] | None = None
        self._folds: npt.NDArray[np.intp] | None = None

    def fit(
        self,
        y: npt.NDArray[np.float64],
        oof_pred: npt.NDArray[np.float64],
        fold_ids: npt.NDArray[np.intp],
    ) -> Self:
        """Store the absolute OOF residuals (model space)."""
        target = np.asarray(y, dtype=np.float64)
        prediction = np.asarray(oof_pred, dtype=np.float64)
        folds = np.asarray(fold_ids, dtype=np.intp)
        if (
            target.ndim != 1
            or target.shape != prediction.shape
            or target.shape != folds.shape
            or target.size == 0
        ):
            msg = "fit expects y, oof_pred and fold_ids of the same shape (n,) with n > 0"
            raise ValueError(msg)
        _check_fold_ids(folds)
        self._residuals = np.abs(target - prediction)
        self._folds = folds
        return self

    def predict_interval(
        self, fold_pred: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(lower, upper)``, each of shape ``(m,)``, from per-fold predictions ``(K, m)``."""
        if self._residuals is None or self._folds is None:
            msg = "CVPlusRegressor is not fitted"
            raise RuntimeError(msg)
        predictions = np.asarray(fold_pred, dtype=np.float64)
        if predictions.ndim != 2:
            msg = "predict_interval expects per-fold predictions of shape (K, m)"
            raise ValueError(msg)
        _check_models_cover_folds(predictions.shape[0], self._folds)
        n = self._residuals.size
        centres = predictions[self._folds, :]  # (n, m): mu_{k(i)}(x_j)
        low_rank = math.floor(self.alpha * (n + 1))
        high_rank = math.ceil((1.0 - self.alpha) * (n + 1))
        m = predictions.shape[1]
        lower = np.full(m, -np.inf)
        upper = np.full(m, np.inf)
        if low_rank >= 1:
            lows = centres - self._residuals[:, np.newaxis]
            lower = np.partition(lows, low_rank - 1, axis=0)[low_rank - 1]
        if high_rank <= n:
            highs = centres + self._residuals[:, np.newaxis]
            upper = np.partition(highs, high_rank - 1, axis=0)[high_rank - 1]
        return lower.astype(np.float64), upper.astype(np.float64)


def set_coverage(y_idx: npt.NDArray[np.intp], sets: npt.NDArray[np.bool_]) -> float:
    """Share of rows whose true label is in its prediction set."""
    y = np.asarray(y_idx, dtype=np.intp)
    if y.size == 0:
        return math.nan
    return float(np.asarray(sets, dtype=bool)[np.arange(y.size), y].mean())


def mean_set_size(sets: npt.NDArray[np.bool_]) -> float:
    """Mean number of labels per prediction set."""
    array = np.asarray(sets, dtype=bool)
    return math.nan if array.shape[0] == 0 else float(array.sum(axis=1).mean())


def singleton_rate(sets: npt.NDArray[np.bool_]) -> float:
    """Share of prediction sets with exactly one label."""
    array = np.asarray(sets, dtype=bool)
    return math.nan if array.shape[0] == 0 else float((array.sum(axis=1) == 1).mean())


def interval_coverage(
    y: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> float:
    """Share of rows with ``lower <= y <= upper``."""
    target = np.asarray(y, dtype=np.float64)
    if target.size == 0:
        return math.nan
    return float(((np.asarray(lower) <= target) & (target <= np.asarray(upper))).mean())


def mean_interval_width(lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64]) -> float:
    """Mean ``upper - lower`` (``inf`` if any interval is unbounded)."""
    widths = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    return math.nan if widths.size == 0 else float(widths.mean())
