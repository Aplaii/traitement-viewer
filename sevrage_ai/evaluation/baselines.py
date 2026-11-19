"""Baselines evaluated on the same folds as the models. Owner: WS5a.

* dummy: majority-class probabilities (classification) or training median (regression);
* clinical rules from ``schema.baselines`` (e.g. PAWSS >= 4 for high severity, CIWA-Ar >= 8
  for treatment). A missing feature value predicts the negative class.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from sevrage_ai.config.schema import BaselineSpec
from sevrage_ai.evaluation.metrics import decision_metrics

__all__ = ["dummy_class_proba", "dummy_regression", "evaluate_rule", "rule_predictions"]

_Comparison = Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.bool_]]
_OPERATORS: Final[dict[str, _Comparison]] = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def dummy_class_proba(
    y_train: npt.NDArray[np.intp], n_classes: int, n_test: int
) -> npt.NDArray[np.float64]:
    """Training class frequencies repeated ``n_test`` times, shape ``(n_test, n_classes)``.

    An empty ``y_train`` gives uniform probabilities. Raises ``ValueError`` if
    ``n_classes < 1`` or a label lies outside ``0..n_classes-1``.
    """
    labels = np.asarray(y_train, dtype=np.intp).ravel()
    if n_classes < 1:
        msg = "dummy_class_proba: n_classes must be >= 1"
        raise ValueError(msg)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= n_classes):
        msg = "dummy_class_proba: labels must lie in 0..n_classes-1"
        raise ValueError(msg)
    if labels.size:
        frequencies = np.bincount(labels, minlength=n_classes) / labels.size
    else:
        frequencies = np.full(n_classes, 1.0 / n_classes)
    return np.tile(frequencies.astype(np.float64), (n_test, 1))


def dummy_regression(y_train: npt.NDArray[np.float64], n_test: int) -> npt.NDArray[np.float64]:
    """Training median repeated ``n_test`` times.

    Missing (``nan``) training values are ignored; with no value left the prediction is
    ``nan``.
    """
    values = np.asarray(y_train, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    median = float(np.median(values)) if values.size else np.nan
    return np.full(n_test, median, dtype=np.float64)


def rule_predictions(X: pd.DataFrame, spec: BaselineSpec) -> npt.NDArray[np.bool_]:
    """``X[spec.feature] <op> spec.threshold``. A missing value is ``False``.

    Non-numeric values count as missing. Raises ``ValueError`` if the feature is not a
    column of ``X``.
    """
    if spec.feature not in X.columns:
        msg = f"rule_predictions: feature {spec.feature!r} is not a column of X"
        raise ValueError(msg)
    numeric = pd.to_numeric(X[spec.feature], errors="coerce")
    values = np.asarray(numeric.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
    compare = _OPERATORS[spec.op]
    return np.asarray(compare(values, spec.threshold), dtype=np.bool_)  # nan compares False


def evaluate_rule(
    spec: BaselineSpec, X: pd.DataFrame, y_positive: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Rule metrics against ``y_positive``: sensitivity, specificity, ppv, npv,
    balanced_accuracy.

    Undefined metrics are ``nan`` (see :func:`sevrage_ai.evaluation.metrics.decision_metrics`).
    """
    truth = np.asarray(y_positive, dtype=bool).ravel()
    if truth.size != len(X):
        msg = "evaluate_rule: y_positive must have one value per row of X"
        raise ValueError(msg)
    return decision_metrics(truth, rule_predictions(X, spec))
