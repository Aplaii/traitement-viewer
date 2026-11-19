"""Local explanations by occlusion against a reference patient (D8). Owner: WS6.

For each feature ``f``, replace ``x[f]`` by ``reference[f]`` and measure the change of
``predict_fn``: ``delta_f = predict_fn(x) - predict_fn(x with f from the reference)``.
Everything happens in the raw canonical feature space, through the whole pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from sevrage_ai.config.schema import FeatureSpec
from sevrage_ai.types import FLOAT_DTYPE, STRING_DTYPE, Contribution, FeatureKind

__all__ = ["MISSING_DISPLAY", "format_value", "occlusion"]

PredictFn = Callable[[pd.DataFrame], npt.NDArray[np.float64]]
"""Maps an ``(m, p)`` canonical frame to the ``(m,)`` quantity to explain."""

MISSING_DISPLAY: Final = "Inconnu"
"""French display of a missing value."""

_BINARY_DISPLAY: Final[Mapping[float, str]] = {1.0: "oui", 0.0: "non"}
_DECIMALS: Final = 2


def occlusion(
    predict_fn: PredictFn,
    x: pd.DataFrame,
    reference: pd.DataFrame,
    features: Sequence[FeatureSpec],
    top_k: int = 8,
) -> tuple[Contribution, ...]:
    """Top-``top_k`` contributions by ``|delta|`` (descending, ties by feature order).

    ``x`` and ``reference`` are one-row canonical frames (``reference`` is
    ``TrainedSystem.reference_patient``). All occluded variants are scored in one call to
    ``predict_fn``. Pass a ``predict_fn`` already scaled to the display unit, e.g. 100 x
    P(predicted class) for percentage points. Features whose value equals the reference
    value, or is missing in both, get ``delta = 0`` and are dropped.

    Details:

    * ``predict_fn`` receives a frame indexed ``0..m-1`` with every column of ``x`` (extra
      columns are passed through unchanged). Row 0 is ``x``; row ``i`` is ``x`` with the
      ``i``-th differing feature (in ``features`` order) set to its reference value.
      Feature columns have canonical dtypes (``float64`` for numeric and binary features,
      :data:`~sevrage_ai.types.STRING_DTYPE` for categorical ones). It is not called when
      no feature differs from the reference.
    * A feature that differs from the reference but leaves the score unchanged is kept
      with ``delta = 0.0``. A non-finite ``delta`` is skipped.

    Raises:
        ValueError: ``x`` or ``reference`` does not have exactly one row, a feature column
            is absent or cannot take its canonical dtype, ``top_k < 0``, or ``predict_fn``
            does not return an ``(m,)`` array.
    """
    if top_k < 0:
        msg = f"occlusion: top_k must be >= 0, got {top_k}"
        raise ValueError(msg)
    patient = _canonical_row(x, features, "x")
    baseline = _canonical_row(reference, features, "reference")
    changed = [
        feature
        for feature in features
        if _differs(patient.loc[0, feature.name], baseline.loc[0, feature.name])
    ]
    if not changed:
        return ()

    scores = _score(predict_fn, _stack_variants(patient, baseline, changed))
    contributions: list[Contribution] = []
    for row, feature in enumerate(changed, start=1):
        # Python floats: inf - inf gives NaN without a numpy RuntimeWarning.
        delta = float(scores[0]) - float(scores[row])
        if math.isfinite(delta):
            display = format_value(patient.loc[0, feature.name], feature)
            contributions.append(
                Contribution(feature=feature.name, value_display=display, delta=delta)
            )
    contributions.sort(key=lambda contribution: -abs(contribution.delta))  # stable sort
    return tuple(contributions[:top_k])


def format_value(value: object, feature: FeatureSpec) -> str:
    """French display of a canonical value with its unit.

    Examples: ``"4000 UI/L"``, ``"oui"``/``"non"`` for binaries, the category, ``"Inconnu"``
    for missing values. Uses a decimal comma.

    Numbers are rounded to at most two decimals, trailing zeros removed (``3.57`` ->
    ``"3,57"``, ``0.5`` -> ``"0,5"``, ``4000.0`` -> ``"4000"``), with no thousands
    separator. A non-zero number that would round to zero keeps two significant digits
    (``0.0042`` -> ``"0,0042"``). A binary value other than 0/1 is shown as a number, and a
    non-numeric value of a numeric feature is shown verbatim.
    """
    if _is_missing(value):
        return MISSING_DISPLAY
    if feature.kind is FeatureKind.CATEGORICAL:
        return str(value)
    if not isinstance(value, int | float | np.integer | np.floating):
        return str(value)
    number = float(value)
    if feature.kind is FeatureKind.BINARY and number in _BINARY_DISPLAY:
        return _BINARY_DISPLAY[number]
    text = _format_number(number)
    return f"{text} {feature.unit}" if feature.unit else text


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _canonical_row(
    frame: pd.DataFrame, features: Sequence[FeatureSpec], owner: str
) -> pd.DataFrame:
    """``frame`` re-indexed ``0``, with feature columns cast to their canonical dtype."""
    if len(frame) != 1:
        msg = f"occlusion: {owner} must have exactly one row, got {len(frame)}"
        raise ValueError(msg)
    missing = [feature.name for feature in features if feature.name not in frame.columns]
    if missing:
        msg = f"occlusion: feature columns missing from {owner}: {missing}"
        raise ValueError(msg)
    dtypes = {
        feature.name: STRING_DTYPE if feature.kind is FeatureKind.CATEGORICAL else FLOAT_DTYPE
        for feature in features
    }
    try:
        return frame.reset_index(drop=True).astype(dtypes)
    except (TypeError, ValueError):
        msg = f"occlusion: feature columns of {owner} cannot take their canonical dtype"
        raise ValueError(msg) from None


def _stack_variants(
    patient: pd.DataFrame, baseline: pd.DataFrame, changed: Sequence[FeatureSpec]
) -> pd.DataFrame:
    """``patient`` followed by one variant per feature of ``changed`` (dtypes preserved)."""
    frames = [patient]
    for feature in changed:
        variant = patient.copy()
        variant[feature.name] = baseline[feature.name]
        frames.append(variant)
    return pd.concat(frames, ignore_index=True)


def _score(predict_fn: PredictFn, stacked: pd.DataFrame) -> npt.NDArray[np.float64]:
    scores = np.asarray(predict_fn(stacked), dtype=np.float64)
    expected = (len(stacked),)
    if scores.shape != expected:
        msg = f"occlusion: predict_fn must return an array of shape {expected}, got {scores.shape}"
        raise ValueError(msg)
    return scores


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float | np.floating) and math.isnan(value)


def _differs(value: object, reference: object) -> bool:
    """``False`` when both values are missing or both are equal."""
    value_missing, reference_missing = _is_missing(value), _is_missing(reference)
    if value_missing or reference_missing:
        return value_missing != reference_missing
    return bool(value != reference)


def _format_number(number: float) -> str:
    if number == 0.0:
        return "0"  # also -0.0
    text = np.format_float_positional(number, precision=_DECIMALS, trim="-")
    if float(text) == 0.0:
        text = np.format_float_positional(number, precision=_DECIMALS, fractional=False, trim="-")
    return text.replace(".", ",")
