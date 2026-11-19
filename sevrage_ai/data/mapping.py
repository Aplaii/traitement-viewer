"""Raw table -> canonical data (:class:`~sevrage_ai.types.CanonicalData`). Owner: WS1.

``to_canonical`` does, in order:

1. drop ``dataset.metadata_columns``; match raw headers to ``source_column`` (see
   :func:`sevrage_ai.config.schema.normalize_column_name`); undeclared columns are
   dropped and listed in the report;
2. treat ``dataset.na_values`` as missing; apply ``value_mapping`` (same normalisation);
3. coerce types with the lenient rules of :mod:`sevrage_ai.data.text_parsing`, then apply
   ``scale``. A value that cannot be read, that falls outside ``valid_range`` or that is
   not a known category **becomes missing** and produces a ``RowIssue``
   (``mapping.unparseable``, ``mapping.out_of_range``, ``mapping.unknown_value``). Raw
   exports are cleaned, never rejected, for these cell-level problems;
4. build ``X`` (every admission feature, schema order, canonical dtypes);
5. with ``with_targets=True``: derive ``severity`` with
   :func:`sevrage_ai.config.conditions.derive_labels` on the canonical post-admission
   features; read the other targets (``value_mapping`` or text ``parser``); convert doses
   to mg diazepam-equivalent; force conditional targets to missing where their condition
   does not hold;
6. build ``ids`` (``dataset.id_column`` or 1-based row numbers) and ``groups``.

Reports carry column names and counts only, never cell values.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from sevrage_ai.config.conditions import derive_labels
from sevrage_ai.config.schema import (
    ColumnSource,
    DatasetSchema,
    DerivedSource,
    FeatureSpec,
    TargetSpec,
    normalize_column_name,
)
from sevrage_ai.data.text_parsing import (
    cell_text,
    compile_patterns,
    is_missing_cell,
    matches_any,
    parse_leading_number,
    parse_number,
    parse_prescription,
)
from sevrage_ai.types import (
    FLOAT_DTYPE,
    STRING_DTYPE,
    STRING_TARGETS,
    SYNTHETIC_MARKER_COLUMN,
    CanonicalData,
    FeatureKind,
    RowIssue,
    TargetName,
)

__all__ = [
    "ISSUE_OUT_OF_RANGE",
    "ISSUE_UNKNOWN_VALUE",
    "ISSUE_UNPARSEABLE",
    "MappingReport",
    "MappingResult",
    "from_canonical_features",
    "scaffold_mapping",
    "to_canonical",
]

ISSUE_UNPARSEABLE: Final = "mapping.unparseable"
ISSUE_OUT_OF_RANGE: Final = "mapping.out_of_range"
ISSUE_UNKNOWN_VALUE: Final = "mapping.unknown_value"
_IDENTIFIER_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_SCAFFOLD_CUTOFF: Final = 0.6
_SCAFFOLD_KEYS: Final = frozenset(
    {"source_column", "negative_column", "column", "id_column", "group_column"}
)


@dataclass(frozen=True, slots=True)
class MappingReport:
    """What the mapping did. French ``warnings``, no cell values.

    ``renamed``: raw header -> canonical name (features and target sources).
    ``missing_columns``: canonical names whose source column is absent.
    ``dropped_columns``: raw headers not declared in the schema.
    ``unparseable_counts`` / ``out_of_range_counts`` / ``unknown_value_counts``: canonical
    column -> number of cells set to missing for that reason. ``pattern_missing_counts``:
    target -> number of texts set to missing by ``missing_patterns``.
    """

    renamed: Mapping[str, str]
    missing_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    unparseable_counts: Mapping[str, int]
    row_issues: tuple[RowIssue, ...]
    warnings: tuple[str, ...]
    out_of_range_counts: Mapping[str, int] = field(default_factory=dict)
    unknown_value_counts: Mapping[str, int] = field(default_factory=dict)
    pattern_missing_counts: Mapping[str, int] = field(default_factory=dict)

    def issues_for_row(self, row_id: str) -> tuple[RowIssue, ...]:
        """Cell-level issues of one row."""
        return tuple(issue for issue in self.row_issues if issue.row_id == row_id)


@dataclass(frozen=True, slots=True, eq=False)
class MappingResult:
    """Canonical data plus the mapping report."""

    data: CanonicalData
    report: MappingReport


# --------------------------------------------------------------------------------------
# Issue collection
# --------------------------------------------------------------------------------------

_MESSAGES: Final[Mapping[str, str]] = {
    ISSUE_UNPARSEABLE: "valeur illisible, considérée comme manquante",
    ISSUE_OUT_OF_RANGE: "valeur hors des bornes plausibles, considérée comme manquante",
    ISSUE_UNKNOWN_VALUE: "modalité non reconnue, considérée comme manquante",
}


class _Issues:
    """Collects cell issues, grouped by code and column."""

    def __init__(self, ids: Sequence[str]) -> None:
        self._ids = ids
        self.rows: list[RowIssue] = []
        self.counts: dict[str, Counter[str]] = {code: Counter() for code in _MESSAGES}

    def add(self, code: str, column: str, label: str, positions: npt.NDArray[np.intp]) -> None:
        if positions.size == 0:
            return
        self.counts[code][column] += int(positions.size)
        message = f"« {label} » : {_MESSAGES[code]}."
        self.rows += [RowIssue(self._ids[int(i)], column, code, message) for i in positions]


# --------------------------------------------------------------------------------------
# Column conversion
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Normaliser:
    na: frozenset[str]
    censored: str

    def text(self, value: object) -> str | None:
        """Cell text, ``None`` when missing or an ``na_value``."""
        text = cell_text(value)
        if text is None or normalize_column_name(text) in self.na:
            return None
        return text


def _positions(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    return np.flatnonzero(mask).astype(np.intp)


def _numeric_values(
    raw: pd.Series[Any], norm: _Normaliser
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Numbers and an ``unparseable`` mask (missing and ``na_values`` are just NaN)."""
    values = np.full(len(raw), np.nan)
    bad = np.zeros(len(raw), dtype=bool)
    censored = "missing" if norm.censored == "missing" else "bound"
    for i, cell in enumerate(raw.tolist()):
        if is_missing_cell(cell):
            continue
        if isinstance(cell, str) and norm.text(cell) is None:
            continue
        number = parse_number(cell, censored=censored)  # type: ignore[arg-type]
        if number is None:
            bad[i] = True
        else:
            values[i] = number
    return values, bad


def _apply_range(
    values: npt.NDArray[np.float64], valid_range: tuple[float, float] | None
) -> npt.NDArray[np.bool_]:
    if valid_range is None:
        return np.zeros(values.size, dtype=bool)
    low, high = valid_range
    with np.errstate(invalid="ignore"):
        outside: npt.NDArray[np.bool_] = (values < low) | (values > high)
    values[outside] = np.nan
    return outside


def _mapped_values(
    raw: pd.Series[Any], mapping: Mapping[str, object], norm: _Normaliser
) -> tuple[list[object], npt.NDArray[np.bool_]]:
    """Apply ``value_mapping``: mapped values (``None`` = missing) and an ``unmapped`` mask."""
    lookup = {normalize_column_name(key): value for key, value in mapping.items()}
    out: list[object] = []
    unmapped = np.zeros(len(raw), dtype=bool)
    for i, cell in enumerate(raw.tolist()):
        text = norm.text(cell)
        if text is None:
            out.append(None)
            continue
        key = normalize_column_name(text)
        if key in lookup:
            out.append(lookup[key])
        else:
            out.append(text)
            unmapped[i] = True
    return out, unmapped


def _convert_feature(
    raw: pd.Series[Any], feature: FeatureSpec, norm: _Normaliser, issues: _Issues
) -> pd.Series[float] | pd.Series[str]:
    if feature.kind is FeatureKind.CATEGORICAL:
        return _convert_categorical(raw, feature, norm, issues)
    if feature.value_mapping is not None:
        mapped, unmapped = _mapped_values(raw, feature.value_mapping, norm)
        values = np.full(len(raw), np.nan)
        for i, value in enumerate(mapped):
            if value is not None and not unmapped[i]:
                values[i] = float(value)  # type: ignore[arg-type]
        leftovers = pd.Series(
            [m if u else None for m, u in zip(mapped, unmapped, strict=True)], dtype=object
        )
        numbers, bad = _numeric_values(leftovers, norm)
        usable = unmapped & ~bad
        values[usable] = numbers[usable]
        unparseable = unmapped & bad
    else:
        values, unparseable = _numeric_values(raw, norm)
    values = values * feature.scale
    code_unknown = np.zeros(len(raw), dtype=bool)
    if feature.kind is FeatureKind.BINARY:
        with np.errstate(invalid="ignore"):
            code_unknown = ~np.isnan(values) & ~np.isin(values, (0.0, 1.0))
        values[code_unknown] = np.nan
        issues.add(ISSUE_UNKNOWN_VALUE, feature.name, feature.label, _positions(code_unknown))
    outside = _apply_range(values, feature.valid_range)
    issues.add(ISSUE_UNPARSEABLE, feature.name, feature.label, _positions(unparseable))
    issues.add(ISSUE_OUT_OF_RANGE, feature.name, feature.label, _positions(outside))
    return pd.Series(values, dtype=FLOAT_DTYPE)


def _convert_categorical(
    raw: pd.Series[Any], feature: FeatureSpec, norm: _Normaliser, issues: _Issues
) -> pd.Series[str]:
    categories = {normalize_column_name(c): c for c in feature.categories or ()}
    mapped, _ = _mapped_values(raw, feature.value_mapping or {}, norm)
    out: list[str | None] = []
    unknown = np.zeros(len(raw), dtype=bool)
    for i, value in enumerate(mapped):
        if value is None:
            out.append(None)
            continue
        category = categories.get(normalize_column_name(str(value)))
        if category is None:
            unknown[i] = True
        out.append(category)
    issues.add(ISSUE_UNKNOWN_VALUE, feature.name, feature.label, _positions(unknown))
    return pd.Series(out, dtype=STRING_DTYPE)


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------


@dataclass
class _TargetContext:
    schema: DatasetSchema
    norm: _Normaliser
    issues: _Issues
    pattern_counts: Counter[str]
    warnings: list[str]


def _texts(
    raw: pd.Series[Any], source: ColumnSource, name: str, ctx: _TargetContext
) -> list[str | None]:
    patterns = compile_patterns(source.missing_patterns)
    texts: list[str | None] = []
    for cell in raw.tolist():
        text = ctx.norm.text(cell)
        if text is not None and patterns and matches_any(text, patterns):
            ctx.pattern_counts[name] += 1
            text = None
        texts.append(text)
    return texts


def _class_target(
    raw: pd.Series[Any],
    name: TargetName,
    spec: TargetSpec,
    source: ColumnSource,
    ctx: _TargetContext,
) -> pd.Series[str]:
    texts = _texts(raw, source, name.value, ctx)
    classes = {normalize_column_name(c): c for c in spec.classes or ()}
    lookup = {normalize_column_name(k): v for k, v in (source.value_mapping or {}).items()}
    out: list[str | None] = []
    unknown = np.zeros(len(raw), dtype=bool)
    unparseable = np.zeros(len(raw), dtype=bool)
    for i, text in enumerate(texts):
        if text is None:
            out.append(None)
            continue
        if source.parser == "prescription_molecule":
            molecule = parse_prescription(text, ctx.schema.molecule_aliases).first_molecule
            unparseable[i] = molecule is None
            if molecule is not None:
                molecule = lookup.get(normalize_column_name(molecule), molecule)
            out.append(molecule)
            continue
        key = normalize_column_name(text)
        if key in lookup:
            out.append(lookup[key])
        elif key in classes:
            out.append(classes[key])
        elif spec.classes is None and _IDENTIFIER_RE.match(key):
            out.append(key)
        else:
            unknown[i] = True
            out.append(None)
    ctx.issues.add(ISSUE_UNKNOWN_VALUE, name.value, spec.label, _positions(unknown))
    ctx.issues.add(ISSUE_UNPARSEABLE, name.value, spec.label, _positions(unparseable))
    return pd.Series(out, dtype=STRING_DTYPE)


def _regression_target(
    raw: pd.Series[Any],
    name: TargetName,
    spec: TargetSpec,
    source: ColumnSource,
    molecule: pd.Series[str] | None,
    ctx: _TargetContext,
) -> pd.Series[float]:
    equivalence = ctx.schema.dose_equivalence_to_diazepam
    values = np.full(len(raw), np.nan)
    unparseable = np.zeros(len(raw), dtype=bool)
    if source.parser is None:
        numbers, unparseable = _numeric_values(raw, ctx.norm)
        values = numbers * source.scale
    else:
        for i, text in enumerate(_texts(raw, source, name.value, ctx)):
            if text is None:
                continue
            number: float | None
            if source.parser == "leading_number":
                number = parse_leading_number(text)
            else:
                prescription = parse_prescription(text, ctx.schema.molecule_aliases)
                number = prescription.total_diazepam_equivalent(equivalence)
            if number is None:
                unparseable[i] = True
            else:
                values[i] = number * source.scale
    if source.convert_with == "molecule":
        values = _convert_dose(values, molecule, equivalence, ctx, spec.label)
    outside = _apply_range(values, spec.valid_range)
    ctx.issues.add(ISSUE_UNPARSEABLE, name.value, spec.label, _positions(unparseable))
    ctx.issues.add(ISSUE_OUT_OF_RANGE, name.value, spec.label, _positions(outside))
    return pd.Series(values, dtype=FLOAT_DTYPE)


def _convert_dose(
    values: npt.NDArray[np.float64],
    molecule: pd.Series[str] | None,
    equivalence: Mapping[str, float],
    ctx: _TargetContext,
    label: str,
) -> npt.NDArray[np.float64]:
    if molecule is None:
        ctx.warnings.append(f"« {label} » : molécule inconnue, dose non convertie (manquante).")
        return np.full(values.size, np.nan)
    labels: list[object] = molecule.astype(object).tolist()
    factors = np.array(
        [equivalence.get(m, np.nan) if isinstance(m, str) else np.nan for m in labels],
        dtype=np.float64,
    )
    lost = int((~np.isnan(values) & np.isnan(factors)).sum())
    if lost:
        ctx.warnings.append(
            f"« {label} » : {lost} dose(s) sans molécule ou sans équivalence connue (manquantes)."
        )
    converted: npt.NDArray[np.float64] = values * factors
    return converted


def _build_targets(
    lookup: Callable[[str], pd.Series[Any] | None],
    schema: DatasetSchema,
    ctx: _TargetContext,
    n: int,
    renamed: dict[str, str],
    missing: list[str],
) -> pd.DataFrame:
    outcomes = pd.DataFrame(
        {
            feature.name: _feature_or_missing(lookup, feature, ctx, n, renamed, missing)
            for feature in schema.post_admission_features()
        },
        index=pd.RangeIndex(n),
    )
    columns: dict[str, pd.Series[str] | pd.Series[float]] = {}
    for name, spec in schema.enabled_targets().items():
        source = spec.source
        if isinstance(source, DerivedSource):
            labels = derive_labels(source.derived, outcomes, on_missing=source.on_missing)
            columns[name.value] = labels.reset_index(drop=True)
            continue
        raw = lookup(source.column)
        if raw is None:
            missing.append(name.value)
            columns[name.value] = _empty_target(name, n)
            continue
        renamed.setdefault(source.column, name.value)
        if name in STRING_TARGETS:
            columns[name.value] = _class_target(raw, name, spec, source, ctx)
        else:
            molecule = _labels_or_none(columns.get(TargetName.MOLECULE.value))
            columns[name.value] = _regression_target(raw, name, spec, source, molecule, ctx)
    y = pd.DataFrame(columns, index=pd.RangeIndex(n))
    for name, spec in schema.enabled_targets().items():
        condition = spec.conditional_on
        if condition is None or condition.target.value not in y:
            continue
        holds = (y[condition.target.value] == condition.equals).fillna(value=False)
        column = y[name.value].copy()
        column[~holds.to_numpy(dtype=bool)] = pd.NA if name in STRING_TARGETS else np.nan
        y[name.value] = column
    return y


def _labels_or_none(column: pd.Series[Any] | None) -> pd.Series[str] | None:
    """The molecule column (text) if it has been built, else ``None``."""
    return None if column is None or column.dtype == FLOAT_DTYPE else column


def _empty_target(name: TargetName, n: int) -> pd.Series[str] | pd.Series[float]:
    if name in STRING_TARGETS:
        return pd.Series([None] * n, dtype=STRING_DTYPE)
    return pd.Series(np.full(n, np.nan), dtype=FLOAT_DTYPE)


def _feature_or_missing(
    lookup: Callable[[str], pd.Series[Any] | None],
    feature: FeatureSpec,
    ctx: _TargetContext,
    n: int,
    renamed: dict[str, str],
    missing: list[str],
) -> pd.Series[float] | pd.Series[str]:
    raw = lookup(feature.source_column)
    if raw is None:
        missing.append(feature.name)
        if feature.kind is FeatureKind.CATEGORICAL:
            return pd.Series([None] * n, dtype=STRING_DTYPE)
        return pd.Series(np.full(n, np.nan), dtype=FLOAT_DTYPE)
    renamed[str(raw.name)] = feature.name
    values = _convert_feature(raw, feature, ctx.norm, ctx.issues)
    if feature.negative_column is None:
        return values
    return _three_state(values, feature, feature.negative_column, lookup, ctx, renamed, missing)


def _three_state(
    positive: pd.Series[Any],
    feature: FeatureSpec,
    negative_column: str,
    lookup: Callable[[str], pd.Series[Any] | None],
    ctx: _TargetContext,
    renamed: dict[str, str],
    missing: list[str],
) -> pd.Series[float]:
    """OUI / NON / NON CONNU: 1 if OUI is ticked, 0 if NON is ticked, missing otherwise."""
    yes = positive.to_numpy(dtype=np.float64, na_value=np.nan) == 1.0
    raw_no = lookup(negative_column)
    if raw_no is None:
        missing.append(f"{feature.name} (NON)")
        no = np.zeros(len(positive), dtype=bool)
    else:
        renamed[str(raw_no.name)] = feature.name
        converted = _convert_feature(raw_no, feature, ctx.norm, ctx.issues)
        no = converted.to_numpy(dtype=np.float64, na_value=np.nan) == 1.0
    values = np.where(yes, 1.0, np.where(no, 0.0, np.nan))
    return pd.Series(values, dtype=FLOAT_DTYPE)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def _row_ids(raw: pd.DataFrame, column: str | None, index: Mapping[str, str]) -> list[str]:
    n = len(raw)
    fallback = [str(i + 1) for i in range(n)]
    if column is None or normalize_column_name(column) not in index:
        return fallback
    values = raw[index[normalize_column_name(column)]].tolist()
    return [cell_text(v) or fallback[i] for i, v in enumerate(values)]


def to_canonical(
    raw: pd.DataFrame,
    schema: DatasetSchema,
    *,
    with_targets: bool,
    is_synthetic: bool = False,
) -> MappingResult:
    """Map a raw table to canonical form (steps in the module docstring).

    ``is_synthetic`` is OR-ed with the presence of the ``__synthetic__`` column in ``raw``.
    With ``with_targets=False``, ``data.y`` is ``None`` and target columns are ignored.
    """
    n = len(raw)
    frame = raw.reset_index(drop=True)
    index: dict[str, str] = {}
    for column in frame.columns:
        index.setdefault(normalize_column_name(str(column)), str(column))
    is_synthetic = is_synthetic or SYNTHETIC_MARKER_COLUMN in index
    metadata = {normalize_column_name(c) for c in schema.dataset.metadata_columns}

    def lookup(column: str) -> pd.Series[Any] | None:
        key = normalize_column_name(column)
        return frame[index[key]] if key in index else None

    ids = _row_ids(frame, schema.dataset.id_column, index)
    norm = _Normaliser(
        na=frozenset(normalize_column_name(v) for v in schema.dataset.na_values),
        censored=schema.dataset.censored_numbers,
    )
    issues = _Issues(ids)
    renamed: dict[str, str] = {}
    missing: list[str] = []
    warnings: list[str] = []
    ctx = _TargetContext(schema, norm, issues, Counter(), warnings)
    X = pd.DataFrame(
        {
            f.name: _feature_or_missing(lookup, f, ctx, n, renamed, missing)
            for f in schema.admission_features()
        },
        index=pd.RangeIndex(n),
    )
    y = _build_targets(lookup, schema, ctx, n, renamed, missing) if with_targets else None
    groups = None
    if schema.dataset.group_column is not None:
        group_raw = lookup(schema.dataset.group_column)
        if group_raw is None:
            warnings.append("Colonne de regroupement absente : chaque ligne est un patient.")
        else:
            groups = pd.Series(
                [cell_text(v) or ids[i] for i, v in enumerate(group_raw.tolist())],
                dtype=STRING_DTYPE,
            )
    declared = set(schema.source_columns()) | metadata
    dropped = tuple(c for key, c in index.items() if key not in declared)
    if missing:
        warnings.append(f"{len(missing)} colonne(s) déclarée(s) absente(s) du fichier.")
    data = CanonicalData(
        X=X,
        y=y,
        groups=groups,
        ids=pd.Series(ids, dtype=STRING_DTYPE),
        is_synthetic=is_synthetic,
    )
    report = MappingReport(
        renamed=renamed,
        missing_columns=tuple(missing),
        dropped_columns=dropped,
        unparseable_counts=dict(issues.counts[ISSUE_UNPARSEABLE]),
        row_issues=tuple(issues.rows),
        warnings=tuple(warnings),
        out_of_range_counts=dict(issues.counts[ISSUE_OUT_OF_RANGE]),
        unknown_value_counts=dict(issues.counts[ISSUE_UNKNOWN_VALUE]),
        pattern_missing_counts=dict(ctx.pattern_counts),
    )
    return MappingResult(data=data, report=report)


def from_canonical_features(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    *,
    ids: Sequence[str] | None = None,
    is_synthetic: bool = False,
) -> MappingResult:
    """Coerce a frame that already uses **canonical names and values** into ``CanonicalData``.

    Used for the Streamlit form (``predict_record``) and ``predict_frame(raw_columns=False)``.
    Missing admission columns are added as missing; unknown columns are dropped with a
    warning; dtypes are coerced to the canonical conventions (values that cannot be
    converted become missing, with a ``mapping.unparseable`` issue); ``y`` is ``None``.
    Values are **not** range-checked here: validation rejects them. Idempotent: applying it
    to ``result.data.X`` returns an equal ``X``.
    """
    n = len(frame)
    source = frame.reset_index(drop=True)
    row_ids = [str(i + 1) for i in range(n)] if ids is None else [str(i) for i in ids]
    if len(row_ids) != n:
        msg = "from_canonical_features: ids must have one entry per row"
        raise ValueError(msg)
    issues = _Issues(row_ids)
    columns: dict[str, pd.Series[float] | pd.Series[str]] = {}
    missing: list[str] = []
    for feature in schema.admission_features():
        if feature.name not in source.columns:
            missing.append(feature.name)
            columns[feature.name] = (
                pd.Series([None] * n, dtype=STRING_DTYPE)
                if feature.kind is FeatureKind.CATEGORICAL
                else pd.Series(np.full(n, np.nan), dtype=FLOAT_DTYPE)
            )
            continue
        column = source[feature.name]
        if feature.kind is FeatureKind.CATEGORICAL:
            text = [None if is_missing_cell(v) else str(v) for v in column.tolist()]
            columns[feature.name] = pd.Series(text, dtype=STRING_DTYPE)
            continue
        numbers = pd.to_numeric(column.astype(object), errors="coerce").astype(np.float64)
        bad = numbers.isna().to_numpy() & ~column.isna().to_numpy()
        issues.add(ISSUE_UNPARSEABLE, feature.name, feature.label, _positions(bad))
        columns[feature.name] = pd.Series(numbers.to_numpy(), dtype=FLOAT_DTYPE)
    admission = {f.name for f in schema.admission_features()}
    unknown = tuple(c for c in source.columns if c not in admission)
    warnings = [f"{len(unknown)} colonne(s) inconnue(s) ignorée(s)."] if unknown else []
    data = CanonicalData(
        X=pd.DataFrame(columns, index=pd.RangeIndex(n)),
        y=None,
        groups=None,
        ids=pd.Series(row_ids, dtype=STRING_DTYPE),
        is_synthetic=is_synthetic,
    )
    report = MappingReport(
        renamed={},
        missing_columns=tuple(missing),
        dropped_columns=tuple(str(c) for c in unknown),
        unparseable_counts=dict(issues.counts[ISSUE_UNPARSEABLE]),
        row_issues=tuple(issues.rows),
        warnings=tuple(warnings),
    )
    return MappingResult(data=data, report=report)


# --------------------------------------------------------------------------------------
# Scaffold
# --------------------------------------------------------------------------------------


def _best_matches(
    wanted: Sequence[tuple[str, str]], columns: Sequence[str]
) -> dict[str, tuple[str, float]]:
    """Greedy one-to-one assignment of raw columns to schema entries by difflib ratio."""
    keys = {normalize_column_name(c): c for c in columns}
    pairs: list[tuple[float, str, str]] = []
    for entry, declared in wanted:
        target = normalize_column_name(declared)
        for key, column in keys.items():
            ratio = difflib.SequenceMatcher(None, target, key).ratio()
            if ratio >= _SCAFFOLD_CUTOFF / 2:
                pairs.append((ratio, entry, column))
    pairs.sort(key=lambda item: -item[0])
    chosen: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for ratio, entry, column in pairs:
        if entry in chosen or column in used:
            continue
        chosen[entry] = (column, ratio)
        used.add(column)
    return chosen


def scaffold_mapping(columns: Sequence[str], schema: DatasetSchema) -> str:
    """Draft a schema YAML mapping ``columns`` (raw headers) onto the schema.

    Each feature, each target source column and the dataset's ``id_column`` /
    ``group_column`` get the closest raw header (``difflib``, one-to-one) as a suggestion,
    with ``# TODO`` markers on uncertain or missing matches. Metadata columns
    (``__synthetic__``) are never suggested. Targets that share a raw column keep sharing
    it. The result is meant to be saved as ``configs/schema.local.yaml`` and edited by hand,
    and loads with :func:`sevrage_ai.config.load_schema`.
    """
    data = schema.model_dump(mode="json", by_alias=True, exclude_defaults=True)
    metadata = {normalize_column_name(c) for c in schema.dataset.metadata_columns}
    candidates = [" ".join(c.split()) for c in columns if normalize_column_name(c) not in metadata]
    wanted: list[tuple[str, str]] = [
        (f"feature:{f.name}", f.source_column) for f in schema.features
    ]
    wanted += [
        (f"feature_no:{f.name}", f.negative_column) for f in schema.features if f.negative_column
    ]
    dataset_keys = {
        key: column
        for key, column in (
            ("id_column", schema.dataset.id_column),
            ("group_column", schema.dataset.group_column),
        )
        if column is not None
    }
    wanted += [(f"dataset:{key}", column) for key, column in dataset_keys.items()]
    target_columns: dict[str, str] = {}
    for name, spec in schema.targets.items():
        if isinstance(spec.source, ColumnSource):
            target_columns.setdefault(normalize_column_name(spec.source.column), name.value)
            wanted.append((f"target:{name.value}", spec.source.column))
    wanted = [
        (entry, column)
        for entry, column in wanted
        if not entry.startswith("target:")
        or target_columns[normalize_column_name(column)] == entry.split(":", 1)[1]
    ]
    matches = _best_matches(wanted, candidates)
    todo: dict[str, str] = {}
    for key in dataset_keys:
        _apply_match(data["dataset"], key, f"dataset:{key}", matches, todo)
    for feature in data["features"]:
        _apply_match(feature, "source_column", f"feature:{feature['name']}", matches, todo)
        if "negative_column" in feature:
            _apply_match(feature, "negative_column", f"feature_no:{feature['name']}", matches, todo)
    shared: dict[str, str] = {}
    for name, target in data["targets"].items():
        source = target.get("source", {})
        if "column" not in source:
            continue
        key = normalize_column_name(source["column"])
        owner = target_columns[key]
        if owner == name:
            _apply_match(source, "column", f"target:{name}", matches, todo)
            shared[key] = source["column"]
        else:
            source["column"] = shared.get(key, source["column"])
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=10_000)
    lines = []
    for line in text.splitlines():
        key, _, value = line.strip().removeprefix("- ").partition(":")
        note = todo.get(value.strip()) if key in _SCAFFOLD_KEYS else None
        lines.append(f"{line}  # TODO {note}" if note else line)
    header = [
        "# Brouillon généré par `sevrage scaffold-mapping` : vérifier chaque ligne « TODO »,",
        "# puis valider avec `sevrage validate --schema <ce fichier>`.",
    ]
    return "\n".join(header + lines) + "\n"


def _apply_match(
    node: dict[str, object],
    key: str,
    entry: str,
    matches: Mapping[str, tuple[str, float]],
    todo: dict[str, str],
) -> None:
    declared = str(node[key])
    match = matches.get(entry)
    if match is None:
        todo[_yaml_scalar(declared)] = "aucune colonne correspondante trouvée"
        return
    column, ratio = match
    node[key] = column
    if normalize_column_name(column) != normalize_column_name(declared) and (
        ratio < 1.0 or not math.isfinite(ratio)
    ):
        todo[_yaml_scalar(column)] = f"à vérifier (similarité {ratio:.2f})"


def _yaml_scalar(value: str) -> str:
    return yaml.safe_dump(value, allow_unicode=True, width=10_000).splitlines()[0]
