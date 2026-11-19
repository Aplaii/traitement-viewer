"""Calibration diagnostics (numpy). Owner: WS3.

``calibration_slope_intercept`` fits an unpenalised logistic regression of ``y`` on
``logit(p)`` (Newton-Raphson): slope 1 and intercept 0 mean perfect calibration; a slope
below 1 means over-confident probabilities. Undefined cases (one class, constant
probabilities, divergence) return ``nan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

__all__ = ["ReliabilityBins", "calibration_slope_intercept", "ece", "reliability_bins"]

_EPS: Final = 1e-6
_MAX_ITER: Final = 50


@dataclass(frozen=True, slots=True, eq=False)
class ReliabilityBins:
    """Reliability-diagram data. Empty bins are dropped, so all arrays have the same length."""

    mean_predicted: npt.NDArray[np.float64]
    observed_frequency: npt.NDArray[np.float64]
    counts: npt.NDArray[np.intp]


def _bin_ids(
    proba: npt.NDArray[np.float64], n_bins: int, strategy: Literal["uniform", "quantile"]
) -> npt.NDArray[np.intp]:
    if strategy == "quantile" and proba.size:
        edges = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    ids: npt.NDArray[np.intp] = np.searchsorted(edges, proba, side="right").astype(np.intp)
    return ids


def reliability_bins(
    y_true: npt.NDArray[np.intp],
    proba: npt.NDArray[np.float64],
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> ReliabilityBins:
    """Bin ``proba`` (probability of the positive class, 0/1 ``y_true``)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    ids = _bin_ids(p, n_bins, strategy)
    means, freqs, counts = [], [], []
    for bin_id in np.unique(ids):
        members = ids == bin_id
        means.append(float(p[members].mean()))
        freqs.append(float(y[members].mean()))
        counts.append(int(members.sum()))
    return ReliabilityBins(
        mean_predicted=np.asarray(means, dtype=np.float64),
        observed_frequency=np.asarray(freqs, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.intp),
    )


def ece(y_true: npt.NDArray[np.intp], proba: npt.NDArray[np.float64], *, n_bins: int = 10) -> float:
    """Expected calibration error (uniform bins, weighted by bin counts)."""
    y = np.asarray(y_true)
    if y.size == 0:
        return math.nan
    bins = reliability_bins(y_true, proba, n_bins=n_bins, strategy="uniform")
    gaps = np.abs(bins.mean_predicted - bins.observed_frequency)
    return float((gaps * bins.counts).sum() / bins.counts.sum())


def _separated(positive: npt.NDArray[np.float64], negative: npt.NDArray[np.float64]) -> bool:
    """``True`` when a threshold on the logit (quasi-)separates the classes.

    With one covariate and an intercept, the maximum likelihood estimate exists iff the
    classes overlap on both sides: ``min(pos) < max(neg)`` and ``max(pos) > min(neg)``.
    Newton's method on (quasi-)separated data never converges to a finite estimate, and
    the clipped linear predictor can make it stop at an arbitrary finite value instead.
    """
    return not (positive.min() < negative.max() and positive.max() > negative.min())


def calibration_slope_intercept(
    y_true: npt.NDArray[np.intp], proba: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """``(slope, intercept)`` of the logistic recalibration of ``proba`` (clipped away from 0/1)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(proba, dtype=np.float64), _EPS, 1.0 - _EPS)
    if y.size < 2 or np.unique(y).size < 2 or np.ptp(p) < _EPS:
        return math.nan, math.nan
    logit = np.log(p / (1.0 - p))
    positive = y > 0.5
    if _separated(logit[positive], logit[~positive]):
        return math.nan, math.nan  # (quasi-)separation: the maximum likelihood diverges
    design = np.column_stack([np.ones_like(logit), logit])
    beta = np.array([0.0, 1.0])
    for _ in range(_MAX_ITER):
        eta = np.clip(design @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = mu * (1.0 - mu)
        hessian = design.T @ (design * weights[:, np.newaxis])
        gradient = design.T @ (y - mu)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        beta = beta + step
        if float(np.abs(step).max()) < 1e-8:
            break
    else:
        return math.nan, math.nan  # no convergence
    if not np.all(np.isfinite(beta)) or float(np.abs(beta).max()) > 1e3:
        return math.nan, math.nan  # separation: the estimate diverged
    return float(beta[1]), float(beta[0])
