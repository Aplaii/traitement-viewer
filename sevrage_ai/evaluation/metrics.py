"""Performance metrics on arrays (pure functions). Owner: WS5a.

Metric keys (contract for the report, the model card and ``ModelInfo.metrics_summary``):

* classification: ``accuracy``, ``balanced_accuracy``, ``macro_f1``, ``log_loss``,
  ``brier`` (multiclass Brier score), ``ece``; ``recall_<class>`` for every class;
  binary: ``roc_auc``, ``pr_auc``, ``calibration_slope``, ``calibration_intercept``, and
  at ``threshold``: ``sensitivity``, ``specificity``, ``ppv``, ``npv``; more than 2
  classes: ``roc_auc_ovr``, ``top2_accuracy``; ``ordinal=True``: ``qwk``;
* regression: ``mae``, ``median_ae``, ``rmse``, ``r2`` (original units);
* conformal classification: ``coverage``, ``mean_set_size``, ``singleton_rate``;
* conformal regression: ``coverage``, ``mean_width``, ``median_width``.

An undefined metric (for example AUC with one class) is ``nan``. It must never raise or
warn.

Definitions (numpy only, identical to scikit-learn where both exist):

* ``balanced_accuracy`` is the mean recall over the classes present in ``y``;
* ``macro_f1`` averages F1 over the classes present in ``y`` or in the predictions (an
  F1 with no true positive is 0);
* ``log_loss`` clips the probability of the true class to ``[1e-15, 1]``;
* ``brier`` is the sum over classes of the squared error, averaged over rows;
* ``ece`` is the ECE of ``P(classes[1])`` for binary tasks and the top-label (confidence)
  ECE otherwise, with 10 uniform bins;
* ``roc_auc`` is the Mann-Whitney AUC (ties count 1/2), ``pr_auc`` the average precision;
* ``roc_auc_ovr`` is the macro one-vs-rest AUC over the classes present in ``y``;
* ``top2_accuracy`` counts a row as a hit when fewer than two classes have a strictly
  higher probability than the true class (ties are hits).

Inconsistent shapes or out-of-range labels are programming errors and raise
``ValueError``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from sevrage_ai.confidence.calibration import calibration_slope_intercept, ece
from sevrage_ai.confidence.conformal import (
    interval_coverage,
    mean_interval_width,
    mean_set_size,
    set_coverage,
    singleton_rate,
)

__all__ = [
    "classification",
    "conformal_classification",
    "conformal_regression",
    "decision_metrics",
    "quadratic_weighted_kappa",
    "regression",
]

_LOG_LOSS_EPS: Final = 1e-15
_NAN: Final = math.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else _NAN


def _mean_defined(values: Sequence[float]) -> float:
    defined = [value for value in values if not math.isnan(value)]
    return math.fsum(defined) / len(defined) if defined else _NAN


# --------------------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------------------


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """1-based ranks, ties receiving the mean of their ranks."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(np.float64)
    means = ends - (counts.astype(np.float64) - 1.0) / 2.0
    ranks: npt.NDArray[np.float64] = means[inverse.ravel()]
    return ranks


def _roc_auc(positive: npt.NDArray[np.bool_], score: npt.NDArray[np.float64]) -> float:
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return _NAN
    rank_sum = float(_average_ranks(score)[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(positive: npt.NDArray[np.bool_], score: npt.NDArray[np.float64]) -> float:
    n_pos = int(positive.sum())
    if n_pos == 0:
        return _NAN
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    hits = positive[order].astype(np.float64)
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_score)), sorted_score.size - 1]
    true_pos = np.cumsum(hits)[last_of_tie]
    precision = true_pos / (last_of_tie + 1.0)
    recall = true_pos / n_pos
    recall_step = np.diff(np.r_[0.0, recall])
    return float((recall_step * precision).sum())


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def _check_classification(
    y: npt.NDArray[np.intp], proba: npt.NDArray[np.float64], n_classes: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    labels = np.asarray(y, dtype=np.intp).ravel()
    probabilities = np.asarray(proba, dtype=np.float64)
    if n_classes < 1:
        msg = "classification: at least one class is required"
        raise ValueError(msg)
    if probabilities.ndim != 2 or probabilities.shape != (labels.size, n_classes):
        msg = "classification: proba must have shape (len(y), len(classes))"
        raise ValueError(msg)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= n_classes):
        msg = "classification: labels must lie in 0..len(classes)-1"
        raise ValueError(msg)
    return labels, probabilities


def _confusion(
    labels: npt.NDArray[np.intp], predicted: npt.NDArray[np.intp], n_classes: int
) -> npt.NDArray[np.int64]:
    flat = np.bincount(labels * n_classes + predicted, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes).astype(np.int64)


def _predictions(
    probabilities: npt.NDArray[np.float64], threshold: float | None
) -> npt.NDArray[np.intp]:
    if probabilities.shape[1] == 2 and threshold is not None:
        return (probabilities[:, 1] >= threshold).astype(np.intp)
    return np.argmax(probabilities, axis=1).astype(np.intp)


def _top2_accuracy(labels: npt.NDArray[np.intp], probabilities: npt.NDArray[np.float64]) -> float:
    if labels.size == 0:
        return _NAN
    true_proba = probabilities[np.arange(labels.size), labels]
    higher = (probabilities > true_proba[:, np.newaxis]).sum(axis=1)
    return float((higher < 2).mean())


def classification(
    y: npt.NDArray[np.intp],
    proba: npt.NDArray[np.float64],
    classes: Sequence[str],
    *,
    threshold: float | None = None,
    ordinal: bool = False,
    predicted: npt.NDArray[np.intp] | None = None,
) -> dict[str, float]:
    """Classification metrics. ``y`` holds label indices, ``proba`` has shape ``(n, C)``.

    For binary tasks the positive class is ``classes[1]`` and ``threshold`` applies to
    ``proba[:, 1]``. Otherwise predictions are the argmax. ``predicted`` (class indices)
    overrides both, so that the metrics follow the production decision rule
    (``models.tasks.predicted_indices``).
    """
    n_classes = len(classes)
    labels, probabilities = _check_classification(y, proba, n_classes)
    n = labels.size
    if predicted is None:
        predicted = _predictions(probabilities, threshold)
    else:
        predicted = np.asarray(predicted, dtype=np.intp)
        if predicted.shape != labels.shape:
            msg = "predicted must have one class index per row"
            raise ValueError(msg)
    confusion = _confusion(labels, predicted, n_classes)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_pos = np.diag(confusion)
    recalls = [_ratio(float(true_pos[k]), float(support[k])) for k in range(n_classes)]
    f1s = [
        2.0 * float(true_pos[k]) / float(support[k] + predicted_count[k])
        for k in range(n_classes)
        if support[k] + predicted_count[k] > 0
    ]
    true_proba = probabilities[np.arange(n), labels]
    onehot = np.eye(n_classes, dtype=np.float64)[labels]
    if n_classes == 2:
        calibration_error = ece(labels, probabilities[:, 1])
    else:
        correct = (np.argmax(probabilities, axis=1) == labels).astype(np.intp)
        calibration_error = ece(correct, probabilities.max(axis=1, initial=0.0))

    result: dict[str, float] = {
        "accuracy": _ratio(float(true_pos.sum()), float(n)),
        "balanced_accuracy": _mean_defined(recalls),
        "macro_f1": _mean_defined(f1s),
        "log_loss": (float(-np.log(np.clip(true_proba, _LOG_LOSS_EPS, 1.0)).mean()) if n else _NAN),
        "brier": float(((probabilities - onehot) ** 2).sum(axis=1).mean()) if n else _NAN,
        "ece": calibration_error,
    }
    for name, recall in zip(classes, recalls, strict=True):
        result[f"recall_{name}"] = recall
    if n_classes == 2:
        positive = labels == 1
        score = probabilities[:, 1]
        slope, intercept = calibration_slope_intercept(labels, score)
        decision = decision_metrics(positive, predicted == 1)
        result |= {
            "roc_auc": _roc_auc(positive, score),
            "pr_auc": _average_precision(positive, score),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "sensitivity": decision["sensitivity"],
            "specificity": decision["specificity"],
            "ppv": decision["ppv"],
            "npv": decision["npv"],
        }
    elif n_classes > 2:
        aucs = [
            _roc_auc(labels == k, probabilities[:, k]) for k in range(n_classes) if support[k] > 0
        ]
        result["roc_auc_ovr"] = _mean_defined(aucs)
        result["top2_accuracy"] = _top2_accuracy(labels, probabilities)
    if ordinal:
        result["qwk"] = quadratic_weighted_kappa(labels, predicted, n_classes)
    return result


def decision_metrics(
    y_positive: npt.NDArray[np.bool_], predicted_positive: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Sensitivity, specificity, PPV, NPV and balanced accuracy of a binary decision.

    ``balanced_accuracy`` is the mean of the defined values among sensitivity and
    specificity (the same definition as :func:`classification`).
    """
    truth = np.asarray(y_positive, dtype=bool).ravel()
    decision = np.asarray(predicted_positive, dtype=bool).ravel()
    if truth.shape != decision.shape:
        msg = "decision_metrics: y_positive and predicted_positive must have the same length"
        raise ValueError(msg)
    true_pos = float((truth & decision).sum())
    false_neg = float((truth & ~decision).sum())
    true_neg = float((~truth & ~decision).sum())
    false_pos = float((~truth & decision).sum())
    sensitivity = _ratio(true_pos, true_pos + false_neg)
    specificity = _ratio(true_neg, true_neg + false_pos)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": _ratio(true_pos, true_pos + false_pos),
        "npv": _ratio(true_neg, true_neg + false_neg),
        "balanced_accuracy": _mean_defined([sensitivity, specificity]),
    }


def quadratic_weighted_kappa(
    y: npt.NDArray[np.intp], y_pred: npt.NDArray[np.intp], n_classes: int
) -> float:
    """Cohen's kappa with quadratic weights (ordinal agreement)."""
    truth = np.asarray(y, dtype=np.intp).ravel()
    predicted = np.asarray(y_pred, dtype=np.intp).ravel()
    if truth.shape != predicted.shape:
        msg = "quadratic_weighted_kappa: y and y_pred must have the same length"
        raise ValueError(msg)
    if truth.size == 0 or n_classes < 2:
        return _NAN
    both = np.r_[truth, predicted]
    if int(both.min()) < 0 or int(both.max()) >= n_classes:
        msg = "quadratic_weighted_kappa: labels must lie in 0..n_classes-1"
        raise ValueError(msg)
    observed = _confusion(truth, predicted, n_classes).astype(np.float64)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / truth.size
    index = np.arange(n_classes, dtype=np.float64)
    weights = (index[:, np.newaxis] - index[np.newaxis, :]) ** 2
    denominator = float((weights * expected).sum())
    if denominator <= 0.0:
        return _NAN
    return 1.0 - float((weights * observed).sum()) / denominator


# --------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------


def regression(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> dict[str, float]:
    """Regression metrics in original units."""
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    prediction = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.shape != prediction.shape:
        msg = "regression: y_true and y_pred must have the same length"
        raise ValueError(msg)
    if truth.size == 0:
        return {"mae": _NAN, "median_ae": _NAN, "rmse": _NAN, "r2": _NAN}
    error = prediction - truth
    absolute = np.abs(error)
    total = float(((truth - truth.mean()) ** 2).sum())
    r2 = _NAN if truth.size < 2 or total <= 0.0 else 1.0 - float((error**2).sum()) / total
    return {
        "mae": float(absolute.mean()),
        "median_ae": float(np.median(absolute)),
        "rmse": math.sqrt(float((error**2).mean())),
        "r2": r2,
    }


# --------------------------------------------------------------------------------------
# Conformal
# --------------------------------------------------------------------------------------


def conformal_classification(
    y: npt.NDArray[np.intp], sets: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Empirical coverage, mean set size and singleton rate."""
    labels = np.asarray(y, dtype=np.intp).ravel()
    membership = np.asarray(sets, dtype=bool)
    if membership.ndim != 2 or membership.shape[0] != labels.size:
        msg = "conformal_classification: sets must have shape (len(y), n_classes)"
        raise ValueError(msg)
    return {
        "coverage": set_coverage(labels, membership),
        "mean_set_size": mean_set_size(membership),
        "singleton_rate": singleton_rate(membership),
    }


def conformal_regression(
    y_true: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Empirical coverage, mean and median interval width."""
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    low = np.asarray(lower, dtype=np.float64).ravel()
    high = np.asarray(upper, dtype=np.float64).ravel()
    if not truth.shape == low.shape == high.shape:
        msg = "conformal_regression: y_true, lower and upper must have the same length"
        raise ValueError(msg)
    return {
        "coverage": interval_coverage(truth, low, high),
        "mean_width": mean_interval_width(low, high),
        "median_width": float(np.median(high - low)) if truth.size else _NAN,
    }
