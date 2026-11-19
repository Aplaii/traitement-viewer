"""Declarative condition language: a closed union of pydantic models, never evaluated as code.

Conditions are used by target derivation (``targets.severity.source.derived``), safety
rules and nowhere else. A condition is one of:

* a comparison ``{feature: <name>, op: eq|ne|gt|gte|lt|lte, value: <number|string>}``;
* a membership test ``{feature: <name>, op: in, value: [<v1>, <v2>, ...]}``;
* ``{any: [<condition>, ...]}`` (logical OR) or ``{all: [<condition>, ...]}`` (logical AND).

``feature`` is a **canonical** feature name, evaluated on a frame of canonical columns.
Ordering operators (``gt``, ``gte``, ``lt``, ``lte``) need a numeric value. ``eq``, ``ne``
and ``in`` accept numbers (for numeric or binary features) or strings (for categorical
features). Booleans are rejected: write ``1``/``0``.

Missing values
    :func:`evaluate` is two-valued and NaN-safe: a comparison on a missing value is
    ``False``, **including** ``ne``. :func:`evaluate_3v` uses Kleene three-valued logic
    (missing leaf -> ``<NA>``; ``any``/``all`` propagate ``<NA>`` only when the result is
    really unknown). :func:`derive_labels` builds on it, so a missing outcome never
    silently falls through to the ``else`` class (see ``on_missing``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from sevrage_ai.errors import ConfigError
from sevrage_ai.types import STRING_DTYPE

__all__ = [
    "AllOf",
    "AnyOf",
    "Comparison",
    "Condition",
    "DerivationRule",
    "ElseRule",
    "Membership",
    "OnMissing",
    "WhenRule",
    "derive_labels",
    "evaluate",
    "evaluate_3v",
    "iter_leaves",
    "referenced_features",
]

ComparisonOp = Literal["eq", "ne", "gt", "gte", "lt", "lte"]
OnMissing = Literal["undetermined", "else"]
Scalar = float | str

_ORDERING_OPS: Final = frozenset({"gt", "gte", "lt", "lte"})
_FEATURE_NAME = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        msg = "les booléens ne sont pas acceptés dans une condition : utiliser 1 ou 0"
        raise ValueError(msg)
    return value


class Comparison(_Frozen):
    """``{feature, op, value}`` with ``op`` in ``eq, ne, gt, gte, lt, lte``."""

    feature: Annotated[str, _FEATURE_NAME]
    op: ComparisonOp
    value: Scalar

    @field_validator("value", mode="before")
    @classmethod
    def _no_bool(cls, value: object) -> object:
        return _reject_bool(value)

    @model_validator(mode="after")
    def _ordering_needs_number(self) -> Comparison:
        if self.op in _ORDERING_OPS and isinstance(self.value, str):
            msg = f"l'opérateur « {self.op} » exige une valeur numérique"
            raise ValueError(msg)
        return self


class Membership(_Frozen):
    """``{feature, op: in, value: [...]}``: the feature value belongs to a list."""

    feature: Annotated[str, _FEATURE_NAME]
    op: Literal["in"]
    value: tuple[Scalar, ...] = Field(min_length=1)

    @field_validator("value", mode="before")
    @classmethod
    def _no_bool(cls, value: object) -> object:
        if isinstance(value, list | tuple):
            for item in value:
                _reject_bool(item)
        return value

    @model_validator(mode="after")
    def _homogeneous(self) -> Membership:
        kinds = {isinstance(item, str) for item in self.value}
        if len(kinds) > 1:
            msg = "« in » exige des valeurs toutes numériques ou toutes textuelles"
            raise ValueError(msg)
        return self


class AnyOf(_Frozen):
    """Logical OR of at least one condition."""

    any: tuple[Condition, ...] = Field(min_length=1)


class AllOf(_Frozen):
    """Logical AND of at least one condition."""

    all: tuple[Condition, ...] = Field(min_length=1)


def _condition_tag(value: object) -> str | None:
    if isinstance(value, dict):
        if "any" in value:
            return "any"
        if "all" in value:
            return "all"
        op = value.get("op")
        if op == "in":
            return "membership"
        if op is not None:
            return "comparison"
        return None
    for tag, cls in _TAGGED:
        if isinstance(value, cls):
            return tag
    return None


Condition = Annotated[
    Annotated[Comparison, Tag("comparison")]
    | Annotated[Membership, Tag("membership")]
    | Annotated[AnyOf, Tag("any")]
    | Annotated[AllOf, Tag("all")],
    Discriminator(
        _condition_tag,
        custom_error_type="invalid_condition",
        custom_error_message=(
            "condition invalide : attendu {feature, op, value}, {any: [...]} ou {all: [...]}"
        ),
    ),
]
"""Closed discriminated union of the condition models."""

_TAGGED: Final[tuple[tuple[str, type[BaseModel]], ...]] = (
    ("comparison", Comparison),
    ("membership", Membership),
    ("any", AnyOf),
    ("all", AllOf),
)

AnyOf.model_rebuild()
AllOf.model_rebuild()


class WhenRule(_Frozen):
    """Derivation rule ``{when: <condition>, then: <class>}``."""

    when: Condition
    then: str = Field(min_length=1)


class ElseRule(_Frozen):
    """Final derivation rule ``{else: <class>}`` (Python attribute ``otherwise``)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True)

    otherwise: str = Field(min_length=1, alias="else")


DerivationRule = WhenRule | ElseRule
"""One step of a first-match-wins derivation."""


# --------------------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------------------


def iter_leaves(cond: Condition) -> Iterator[Comparison | Membership]:
    """Yield every comparison and membership leaf of ``cond``, depth first."""
    if isinstance(cond, AnyOf):
        for child in cond.any:
            yield from iter_leaves(child)
    elif isinstance(cond, AllOf):
        for child in cond.all:
            yield from iter_leaves(child)
    else:
        yield cond


def referenced_features(cond: Condition) -> frozenset[str]:
    """Canonical feature names referenced anywhere in ``cond``."""
    return frozenset(leaf.feature for leaf in iter_leaves(cond))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _leaf_3v(leaf: Comparison | Membership, frame: pd.DataFrame) -> pd.Series[bool]:
    if leaf.feature not in frame.columns:
        msg = f"Condition : la variable « {leaf.feature} » est absente des données."
        raise ConfigError(msg)
    column = frame[leaf.feature]
    values: tuple[Scalar, ...] = leaf.value if isinstance(leaf, Membership) else (leaf.value,)
    numeric_values = not isinstance(values[0], str)
    numeric_column = pd.api.types.is_numeric_dtype(column.dtype)
    if numeric_values != numeric_column:
        expected = "numérique" if numeric_column else "textuelle"
        msg = f"Condition sur « {leaf.feature} » : la valeur de comparaison doit être {expected}."
        raise ConfigError(msg)
    missing = column.isna().to_numpy(dtype=bool)
    result: npt.NDArray[np.bool_]
    if numeric_column:
        data = column.to_numpy(dtype=np.float64, na_value=np.nan)
        numbers = np.asarray(values, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            result = _compare(data, leaf.op, numbers)
    else:
        # Missing cells are masked to <NA> below, so the fill value never matters.
        text = column.astype(STRING_DTYPE).fillna("").to_numpy(dtype=object)
        strings = np.asarray(values, dtype=object)
        result = _compare(text, leaf.op, strings)
    masked = pd.array(result, dtype="boolean")
    masked[missing] = pd.NA
    return pd.Series(masked, index=frame.index, name=leaf.feature)


def _compare(
    data: npt.NDArray[np.float64] | npt.NDArray[np.object_],
    op: str,
    values: npt.NDArray[np.float64] | npt.NDArray[np.object_],
) -> npt.NDArray[np.bool_]:
    if op == "in":
        return np.isin(data, values)
    target = values[0]
    outcome: npt.NDArray[np.bool_]
    match op:
        case "eq":
            outcome = np.asarray(data == target, dtype=bool)
        case "ne":
            outcome = np.asarray(data != target, dtype=bool)
        case "gt":
            outcome = np.asarray(data > target, dtype=bool)
        case "gte":
            outcome = np.asarray(data >= target, dtype=bool)
        case "lt":
            outcome = np.asarray(data < target, dtype=bool)
        case "lte":
            outcome = np.asarray(data <= target, dtype=bool)
        case _:  # pragma: no cover - excluded by the pydantic Literal
            msg = f"opérateur inconnu : {op}"
            raise ConfigError(msg)
    return outcome


def evaluate_3v(cond: Condition, frame: pd.DataFrame) -> pd.Series[bool]:
    """Evaluate ``cond`` row-wise with Kleene logic. Returns a ``boolean`` (nullable) Series.

    A leaf on a missing value is ``<NA>``. ``any`` is ``True`` if one child is ``True``,
    ``<NA>`` if none is ``True`` and one is ``<NA>``, else ``False``. ``all`` is the dual.
    The result shares ``frame.index``.

    Raises:
        ConfigError: a referenced feature is absent from ``frame``, or a value's type does
            not match the column (numeric vs text).
    """
    if isinstance(cond, AnyOf):
        children = [evaluate_3v(child, frame) for child in cond.any]
        result = children[0]
        for child in children[1:]:
            result = result | child
        return result
    if isinstance(cond, AllOf):
        children = [evaluate_3v(child, frame) for child in cond.all]
        result = children[0]
        for child in children[1:]:
            result = result & child
        return result
    return _leaf_3v(cond, frame)


def evaluate(cond: Condition, frame: pd.DataFrame) -> pd.Series[bool]:
    """Evaluate ``cond`` row-wise, vectorised and NaN-safe.

    Returns a plain ``bool`` Series aligned on ``frame.index``. A comparison on a missing
    value is ``False`` (including ``ne``). Equivalent to ``evaluate_3v(...).fillna(False)``,
    because the language has no negation.

    Raises:
        ConfigError: see :func:`evaluate_3v`.
    """
    return evaluate_3v(cond, frame).fillna(value=False).astype(bool)


def derive_labels(
    rules: Sequence[DerivationRule],
    frame: pd.DataFrame,
    *,
    on_missing: OnMissing = "undetermined",
) -> pd.Series[str]:
    """Derive one label per row with first-match-wins rules (used for ``severity``).

    For each row, the rules are tried in order. A ``when`` rule that is ``True`` assigns its
    ``then`` label. An :class:`ElseRule` assigns its label to every row still undecided.
    When a ``when`` rule is *unknown* for a row (``<NA>``, i.e. it depends on missing values):

    * ``on_missing="undetermined"`` (default): the row gets ``<NA>``, because an earlier,
      more severe rule might have matched;
    * ``on_missing="else"``: unknown is treated as ``False`` and the next rules are tried.

    Returns:
        A :data:`~sevrage_ai.types.STRING_DTYPE` Series aligned on ``frame.index``.
    """
    labels: npt.NDArray[np.object_] = np.full(len(frame), None, dtype=object)
    pending = np.ones(len(frame), dtype=bool)
    for rule in rules:
        if isinstance(rule, ElseRule):
            labels[pending] = rule.otherwise
            break
        verdict = evaluate_3v(rule.when, frame)
        matched = verdict.fillna(value=False).to_numpy(dtype=bool) & pending
        labels[matched] = rule.then
        pending &= ~matched
        if on_missing == "undetermined":
            pending &= ~verdict.isna().to_numpy(dtype=bool)
    return pd.Series(labels, index=frame.index, dtype=STRING_DTYPE)
