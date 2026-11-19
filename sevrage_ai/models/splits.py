"""Cross-validation splitting. Owner: WS4.

``StratifiedGroupKFold`` when groups exist, else ``StratifiedKFold``, always shuffled with
the given seed. The number of splits is clamped to
``max(2, min(n_splits, smallest stratum count))``, so scikit-learn's "least populated
class" warning can never fire (``filterwarnings = error``). When the smallest stratum has
a single member, stratification is impossible and a shuffled (group) K-fold is used.
Regression targets are stratified on quantile bins.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from sevrage_ai.config.schema import DatasetSchema
from sevrage_ai.types import TargetName

__all__ = [
    "Split",
    "effective_n_splits",
    "make_splitter",
    "outer_stratification_key",
    "quantile_strata",
    "repeated_splits",
]

Split = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
"""``(train_positions, test_positions)``."""

_MISSING_STRATUM: Final = "__missing__"
_RARE_STRATUM: Final = "__rare__"
_REPEAT_SEED_STEP: Final = 1_000


def _smallest_count(y_strat: npt.ArrayLike) -> int:
    _, counts = np.unique(np.asarray(y_strat).astype(str), return_counts=True)
    return int(counts.min()) if counts.size else 0


def effective_n_splits(y_strat: npt.ArrayLike, n_splits: int) -> int:
    """``max(2, min(n_splits, smallest stratum count))``."""
    return max(2, min(n_splits, _smallest_count(y_strat)))


def make_splitter(
    y_strat: npt.ArrayLike,
    groups: npt.ArrayLike | None,
    n_splits: int,
    seed: int,
) -> Iterator[Split]:
    """Yield stratified (grouped when ``groups`` is given) folds, clamped (module docstring).

    Raises:
        ValueError: fewer than 2 rows (or 2 groups).
    """
    labels = np.asarray(y_strat).astype(str)
    n = labels.size
    units = n if groups is None else int(np.unique(np.asarray(groups).astype(str)).size)
    if units < 2:
        msg = "cross-validation needs at least 2 rows (or groups)"
        raise ValueError(msg)
    k = min(effective_n_splits(labels, n_splits), units)
    stratify = _smallest_count(labels) >= 2
    X_dummy = np.zeros((n, 1))
    splitter: StratifiedGroupKFold | StratifiedKFold | GroupKFold | KFold
    if groups is not None:
        group_array = np.asarray(groups).astype(str)
        if stratify:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = splitter.split(X_dummy, labels, group_array)
    else:
        if stratify:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = splitter.split(X_dummy, labels)
    for train, test in folds:
        yield train.astype(np.intp), test.astype(np.intp)


def repeated_splits(
    y_strat: npt.ArrayLike,
    groups: npt.ArrayLike | None,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[Split]:
    """``n_repeats`` x ``make_splitter`` with seeds derived from ``seed``.

    Used for the outer CV, with one split set shared by all targets.
    """
    splits: list[Split] = []
    for repeat in range(n_repeats):
        splits += list(make_splitter(y_strat, groups, n_splits, seed + _REPEAT_SEED_STEP * repeat))
    return splits


def quantile_strata(values: npt.NDArray[np.float64], n_bins: int = 5) -> npt.NDArray[np.intp]:
    """Stratification labels for a regression target (quantile bins, ties merged)."""
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0:
        return np.zeros(0, dtype=np.intp)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    strata: npt.NDArray[np.intp] = np.searchsorted(edges, finite, side="right").astype(np.intp)
    return strata


def outer_stratification_key(
    y: pd.DataFrame, schema: DatasetSchema, *, min_cell: int
) -> npt.NDArray[np.str_]:
    """Outer-CV stratification key: severity x treatment.

    Missing labels get their own stratum. Strata with fewer than ``min_cell`` rows are
    pooled into one ``__rare__`` stratum, itself merged into the largest stratum when it is
    still too small: a handful of rows (for example 2 patients without a severity label)
    can then never clamp the number of outer folds below ``min_cell``.
    """
    del schema  # the key only uses the canonical target columns

    def column(name: TargetName) -> npt.NDArray[np.str_]:
        if name.value not in y:
            return np.full(len(y), _MISSING_STRATUM)
        values = y[name.value].astype(object).where(y[name.value].notna(), _MISSING_STRATUM)
        return np.asarray(values, dtype=str)

    severity = column(TargetName.SEVERITY)
    combined = np.char.add(np.char.add(severity, "|"), column(TargetName.TREATMENT))
    return _pool_small_strata(combined, min_cell)


def _pool_small_strata(labels: npt.NDArray[np.str_], min_cell: int) -> npt.NDArray[np.str_]:
    values, counts = np.unique(labels, return_counts=True)
    small = values[counts < min_cell]
    if small.size == 0:
        return labels
    pooled = np.where(np.isin(labels, small), _RARE_STRATUM, labels).astype(str)
    n_rare = int(np.isin(labels, small).sum())
    if n_rare >= min_cell or small.size == values.size:
        return pooled
    largest = str(values[counts >= min_cell][np.argmax(counts[counts >= min_cell])])
    return np.where(pooled == _RARE_STRATUM, largest, pooled).astype(str)
