"""Global explanations by permutation importance on held-out folds (D8). Owner: WS6."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from sevrage_ai.config.schema import FeatureSpec

__all__ = ["permutation", "permutation_batched"]

ScoreFn = Callable[[pd.DataFrame], float]
"""Maps a (possibly permuted) canonical test frame to a score, where higher is better."""


def permutation(
    score_fn: ScoreFn,
    X: pd.DataFrame,
    features: Sequence[FeatureSpec],
    *,
    n_repeats: int,
    seed: int,
) -> dict[str, float]:
    """Mean score drop when each canonical feature column is shuffled (``n_repeats`` times).

    Keys are canonical feature names, in ``features`` order. The caller wraps the model and
    the true labels of an outer test fold in ``score_fn``. Deterministic for a given ``seed``.

    ``score_fn`` is called once on the untouched ``X`` (baseline), then ``n_repeats`` times
    per feature on a copy of ``X`` where only that column is shuffled (same index, same
    dtype). One ``numpy`` generator seeded with ``seed`` drives every shuffle, feature by
    feature. ``X`` itself is never modified. A positive value means the feature helps.

    Raises:
        ValueError: ``n_repeats < 1`` or a feature column is absent from ``X``.
    """
    if n_repeats < 1:
        msg = f"permutation: n_repeats must be >= 1, got {n_repeats}"
        raise ValueError(msg)
    missing = [feature.name for feature in features if feature.name not in X.columns]
    if missing:
        msg = f"permutation: feature columns missing from X: {missing}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    baseline = float(score_fn(X))
    importances: dict[str, float] = {}
    for feature in features:
        drops = [
            baseline - float(score_fn(_shuffle_column(X, feature.name, rng)))
            for _ in range(n_repeats)
        ]
        importances[feature.name] = math.fsum(drops) / n_repeats
    return importances


PredictFn = Callable[[pd.DataFrame], "npt.NDArray[Any]"]
"""Maps a canonical frame to one prediction row per input row (probabilities or values)."""

PredictionScoreFn = Callable[["npt.NDArray[Any]"], float]
"""Maps the predictions of the ``len(X)`` test rows to a score, where higher is better."""


def permutation_batched(
    predict_fn: PredictFn,
    score_fn: PredictionScoreFn,
    X: pd.DataFrame,
    features: Sequence[FeatureSpec],
    *,
    n_repeats: int,
    seed: int,
) -> dict[str, float]:
    """Same result as :func:`permutation`, with a single call to ``predict_fn``.

    The untouched ``X`` and every shuffled copy (same shuffles, same order, same ``seed``)
    are stacked into one frame and predicted at once; ``score_fn`` then scores each block
    of ``len(X)`` predictions. Much faster when a prediction call has a large fixed cost
    (pipelines, calibrated cross-fit ensembles), and identical to :func:`permutation` for
    models whose predictions are row-independent.

    Raises:
        ValueError: ``n_repeats < 1``, a feature column is absent from ``X``, or
            ``predict_fn`` does not return one row per input row.
    """
    if n_repeats < 1:
        msg = f"permutation: n_repeats must be >= 1, got {n_repeats}"
        raise ValueError(msg)
    missing = [feature.name for feature in features if feature.name not in X.columns]
    if missing:
        msg = f"permutation: feature columns missing from X: {missing}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    blocks = [X] + [
        _shuffle_column(X, feature.name, rng) for feature in features for _ in range(n_repeats)
    ]
    stacked = pd.concat(blocks, ignore_index=True)
    predictions = np.asarray(predict_fn(stacked))
    n = len(X)
    if predictions.shape[0] != n * len(blocks):
        msg = "permutation: predict_fn must return one prediction per input row"
        raise ValueError(msg)
    baseline = float(score_fn(predictions[:n]))
    importances: dict[str, float] = {}
    for position, feature in enumerate(features):
        drops = [
            baseline - float(score_fn(predictions[start : start + n]))
            for start in (n * (1 + position * n_repeats + repeat) for repeat in range(n_repeats))
        ]
        importances[feature.name] = math.fsum(drops) / n_repeats
    return importances


def _shuffle_column(X: pd.DataFrame, name: str, rng: np.random.Generator) -> pd.DataFrame:
    """Copy of ``X`` whose column ``name`` is shuffled, keeping its index and dtype."""
    column = X[name]
    shuffled = pd.Series(rng.permutation(column.to_numpy()), index=X.index, dtype=column.dtype)
    return X.assign(**{name: shuffled})
