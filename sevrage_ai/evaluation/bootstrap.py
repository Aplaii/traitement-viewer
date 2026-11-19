"""Percentile bootstrap confidence intervals. Owner: WS5a."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import numpy.typing as npt

from sevrage_ai.types import MetricEstimate

__all__ = ["ci"]


def _strata_members(
    stratify: npt.NDArray[np.generic] | None, n_rows: int
) -> list[npt.NDArray[np.intp]] | None:
    if stratify is None:
        return None
    labels = np.asarray(stratify)
    if labels.shape[:1] != (n_rows,):
        msg = "ci: stratify must have one label per row"
        raise ValueError(msg)
    _, inverse = np.unique(labels, return_inverse=True)
    codes = inverse.ravel()
    return [np.flatnonzero(codes == code) for code in range(int(codes.max(initial=-1)) + 1)]


def _resamples(
    rng: np.random.Generator,
    n_rows: int,
    n_resamples: int,
    strata: Sequence[npt.NDArray[np.intp]] | None,
) -> Iterator[npt.NDArray[np.intp]]:
    for _ in range(n_resamples):
        if strata is None:
            yield rng.integers(0, n_rows, size=n_rows, dtype=np.intp)
        else:
            yield np.concatenate(
                [members[rng.integers(0, members.size, size=members.size)] for members in strata]
            )


def _quantile(ordered: npt.NDArray[np.float64], q: float) -> float:
    """Linear-interpolation quantile of sorted values (numpy's default), safe with ``inf``."""
    position = q * (ordered.size - 1)
    below = math.floor(position)
    above = min(below + 1, ordered.size - 1)
    fraction = position - below
    low, high = float(ordered[below]), float(ordered[above])
    if fraction == 0.0 or low == high:
        return low
    if math.isinf(low):
        return low
    if math.isinf(high):
        return high
    return low + (high - low) * fraction


def ci(
    fn: Callable[..., float],
    *arrays: npt.NDArray[np.generic],
    n: int,
    level: float,
    seed: int,
    stratify: npt.NDArray[np.generic] | None = None,
    sd_repeats: float | None = None,
) -> MetricEstimate:
    """Point estimate ``fn(*arrays)`` and percentile bootstrap CI over rows.

    Rows are resampled jointly across ``arrays`` (within strata if ``stratify``). Resamples
    where ``fn`` is undefined (``nan``) are dropped; if none remain, ``lo`` and ``hi`` are
    ``nan``. Deterministic for a given ``seed``.

    ``n`` is the number of bootstrap resamples and ``level`` the CI level; the returned
    :class:`MetricEstimate` has ``n`` = number of rows and ``sd_repeats`` passed through.
    ``fn`` must accept arrays indexed along their first axis. Raises ``ValueError`` on
    invalid parameters or arrays of different lengths.
    """
    if not arrays:
        msg = "ci: at least one array is required"
        raise ValueError(msg)
    if n < 1:
        msg = "ci: n must be >= 1 (number of bootstrap resamples)"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = "ci: level must lie in (0, 1)"
        raise ValueError(msg)
    data = [np.asarray(array) for array in arrays]
    n_rows = data[0].shape[0] if data[0].ndim else 0
    if any(array.ndim == 0 or array.shape[0] != n_rows for array in data):
        msg = "ci: every array must have the same number of rows"
        raise ValueError(msg)
    strata = _strata_members(stratify, n_rows)

    value = float(fn(*data))
    statistics: list[float] = []
    if n_rows:
        rng = np.random.default_rng(seed)
        for rows in _resamples(rng, n_rows, n, strata):
            statistic = float(fn(*(array[rows] for array in data)))
            if not math.isnan(statistic):
                statistics.append(statistic)
    lo = hi = math.nan
    if statistics:
        ordered = np.sort(np.asarray(statistics, dtype=np.float64))
        tail = (1.0 - level) / 2.0
        lo, hi = _quantile(ordered, tail), _quantile(ordered, 1.0 - tail)
    return MetricEstimate(value=value, lo=lo, hi=hi, sd_repeats=sd_repeats, level=level, n=n_rows)
