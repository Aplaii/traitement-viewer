"""Validation of canonical data, per row and per table (D3). Owner: WS1.

Per-row checks: numeric values inside ``valid_range``, binary in {0, 1}, categorical in
``categories``; in ``train`` mode, also target classes and regression ``valid_range``.
:func:`build_record_model` expresses the same rules as a pydantic model (used by the
Streamlit form); :func:`validate` applies them column-wise (fast, same results).

Table-level checks (:func:`validate`): at least one row, ids unique, required features
not entirely missing, labels present and at least two observed classes for targets with
declared classes in ``train`` mode (errors); share of missing values per column above
50 %, classes rarer than ``min_class_count``, rows without label per target, a single
observed class for a target whose classes are inferred from the data, such as
``molecule``, which then gets a constant model (warnings).

Raw exports rarely produce row issues here: the mapping already turned unreadable or
implausible cells into missing values. Canonical input (the form, ``raw_columns=False``)
is checked strictly.

Messages are French and never contain cell values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model

from sevrage_ai.config.schema import DatasetSchema, FeatureSpec, TargetSpec
from sevrage_ai.types import (
    STRING_TARGETS,
    CanonicalData,
    FeatureKind,
    RowIssue,
    TargetName,
    TaskKind,
)

__all__ = [
    "TableIssue",
    "ValidationMode",
    "ValidationReport",
    "build_record_model",
    "validate",
]

ValidationMode = Literal["train", "predict"]

_MISSING_WARNING_SHARE: Final = 0.5
ISSUE_INVALID: Final = "validation.invalid_value"


@dataclass(frozen=True, slots=True)
class TableIssue:
    """A table-level problem. ``code`` is namespaced (``validation.*``)."""

    code: str
    message_fr: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of :func:`validate`.

    ``ok`` is ``False`` when there is at least one blocking ``errors`` entry. In ``train``
    mode, invalid rows are blocking. In ``predict`` mode they are not: they are rejected
    individually through ``row_issues``.
    """

    mode: ValidationMode
    ok: bool
    n_rows: int
    errors: tuple[TableIssue, ...]
    warnings: tuple[TableIssue, ...]
    row_issues: tuple[RowIssue, ...]

    def rejected_row_ids(self) -> frozenset[str]:
        """Ids of the rows with at least one row issue."""
        return frozenset(issue.row_id for issue in self.row_issues)

    def to_markdown(self) -> str:
        """French Markdown report (``sevrage validate --report``), without cell values."""
        verdict = "valide" if self.ok else "NON valide"
        lines = [
            "# Rapport de validation",
            "",
            f"- Mode : `{self.mode}`",
            f"- Lignes : {self.n_rows}",
            f"- Résultat : **{verdict}**",
            f"- Lignes avec au moins une valeur invalide : {len(self.rejected_row_ids())}",
            "",
        ]
        for title, issues in (
            ("Erreurs bloquantes", self.errors),
            ("Avertissements", self.warnings),
        ):
            lines += [f"## {title}", ""]
            lines += [f"- {_format_issue(i)}" for i in issues] or ["- Aucun."]
            lines.append("")
        if self.row_issues:
            counts: dict[tuple[str, str], int] = {}
            for issue in self.row_issues:
                key = (issue.column or "?", issue.code)
                counts[key] = counts.get(key, 0) + 1
            lines += ["## Valeurs invalides par colonne", "", "| Colonne | Code | Lignes |"]
            lines.append("|---|---|---|")
            lines += [f"| {c} | `{code}` | {n} |" for (c, code), n in sorted(counts.items())]
            lines.append("")
        return "\n".join(lines)


def _format_issue(issue: TableIssue) -> str:
    column = f" (`{issue.column}`)" if issue.column else ""
    return f"{issue.message_fr}{column}"


# --------------------------------------------------------------------------------------
# Record model
# --------------------------------------------------------------------------------------


def _is_binary(value: float) -> float:
    if value not in (0.0, 1.0):
        msg = "une variable binaire vaut 0 ou 1"
        raise ValueError(msg)
    return value


def _feature_field(feature: FeatureSpec) -> tuple[Any, Any]:
    if feature.kind is FeatureKind.CATEGORICAL:
        return (Literal[tuple(feature.categories or ())] | None, None)
    if feature.kind is FeatureKind.BINARY:
        return (Annotated[float, AfterValidator(_is_binary)] | None, None)
    low, high = feature.valid_range or (None, None)
    return (float | None, Field(default=None, ge=low, le=high, allow_inf_nan=False))


def _target_field(spec: TargetSpec) -> tuple[Any, Any]:
    if spec.kind is TaskKind.REGRESSION:
        low, high = spec.valid_range or (0.0, None)
        return (float | None, Field(default=None, ge=low, le=high, allow_inf_nan=False))
    if spec.classes is not None:
        return (Literal[tuple(spec.classes)] | None, None)
    return (str | None, None)


def build_record_model(
    schema: DatasetSchema,
    mode: ValidationMode,
    *,
    feature_set: str | None = None,
) -> type[BaseModel]:
    """Create the pydantic model of one canonical record (``pydantic.create_model``).

    Fields are canonical feature names (all admission features, or only ``feature_set``),
    all optional (``None`` = unknown), constrained by kind and ``valid_range``. In ``train``
    mode, the enabled targets are added. The model forbids extra keys. The Streamlit form
    uses the same model.
    """
    features = (
        schema.admission_features() if feature_set is None else schema.predictors(feature_set)
    )
    fields: dict[str, Any] = {f.name: _feature_field(f) for f in features}
    if mode == "train":
        for name, spec in schema.enabled_targets().items():
            fields[name.value] = _target_field(spec)
    config = ConfigDict(extra="forbid", frozen=True)
    model: type[BaseModel] = create_model(f"Record_{mode}", __config__=config, **fields)
    return model


# --------------------------------------------------------------------------------------
# Column-wise validation (same rules as the record model)
# --------------------------------------------------------------------------------------


def _invalid_feature_rows(column: pd.Series[Any], feature: FeatureSpec) -> npt.NDArray[np.bool_]:
    present = column.notna().to_numpy(dtype=bool)
    if feature.kind is FeatureKind.CATEGORICAL:
        allowed = set(feature.categories or ())
        inside = np.array([v in allowed for v in column.tolist()], dtype=bool)
        invalid: npt.NDArray[np.bool_] = present & ~inside
        return invalid
    values = column.to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(values)
        if feature.kind is FeatureKind.BINARY:
            bad |= ~np.isin(values, (0.0, 1.0))
        elif feature.valid_range is not None:
            low, high = feature.valid_range
            bad |= (values < low) | (values > high)
    return present & bad


def _invalid_target_rows(column: pd.Series[Any], spec: TargetSpec) -> npt.NDArray[np.bool_]:
    present = column.notna().to_numpy(dtype=bool)
    if spec.kind is TaskKind.REGRESSION:
        values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        low, high = spec.valid_range or (0.0, np.inf)
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(values) | (values < low) | (values > high)
        return present & bad
    if spec.classes is None:
        return np.zeros(len(column), dtype=bool)
    allowed = set(spec.classes)
    inside = np.array([v in allowed for v in column.tolist()], dtype=bool)
    invalid: npt.NDArray[np.bool_] = present & ~inside
    return invalid


def _row_issues(data: CanonicalData, schema: DatasetSchema, mode: ValidationMode) -> list[RowIssue]:
    ids = data.ids.tolist()
    issues: list[RowIssue] = []
    for feature in schema.admission_features():
        if feature.name not in data.X:
            continue
        rows = np.flatnonzero(_invalid_feature_rows(data.X[feature.name], feature))
        message = f"« {feature.label} » : valeur invalide ({_expected(feature)})."
        issues += [RowIssue(ids[i], feature.name, ISSUE_INVALID, message) for i in rows]
    if mode == "train" and data.y is not None:
        for name, spec in schema.enabled_targets().items():
            if name.value not in data.y:
                continue
            rows = np.flatnonzero(_invalid_target_rows(data.y[name.value], spec))
            message = f"Cible « {spec.label} » : valeur invalide."
            issues += [RowIssue(ids[i], name.value, ISSUE_INVALID, message) for i in rows]
    return issues


def _expected(feature: FeatureSpec) -> str:
    if feature.kind is FeatureKind.CATEGORICAL:
        return "modalités : " + ", ".join(feature.categories or ())
    if feature.kind is FeatureKind.BINARY:
        return "0 ou 1"
    if feature.valid_range is not None:
        low, high = feature.valid_range
        return f"entre {low:g} et {high:g}"
    return "nombre fini"


# --------------------------------------------------------------------------------------
# Table-level validation
# --------------------------------------------------------------------------------------


def _table_checks(
    data: CanonicalData, schema: DatasetSchema, mode: ValidationMode
) -> tuple[list[TableIssue], list[TableIssue]]:
    errors: list[TableIssue] = []
    warnings: list[TableIssue] = []
    if data.n_rows == 0:
        errors.append(TableIssue("validation.empty", "Aucune ligne de données."))
        return errors, warnings
    duplicated = int(data.ids.duplicated().sum())
    if duplicated:
        issue = TableIssue(
            "validation.duplicate_ids", f"{duplicated} identifiant(s) en double.", None
        )
        (errors if mode == "train" else warnings).append(issue)
    for feature in schema.admission_features():
        column = data.X.get(feature.name)
        share = 1.0 if column is None else float(column.isna().mean())
        if feature.required and share == 1.0:
            errors.append(
                TableIssue(
                    "validation.required_missing",
                    f"Variable requise entièrement manquante : « {feature.label} ».",
                    feature.name,
                )
            )
        elif share > _MISSING_WARNING_SHARE:
            warnings.append(
                TableIssue(
                    "validation.many_missing",
                    f"« {feature.label} » : {share:.0%} de valeurs manquantes.",
                    feature.name,
                )
            )
    if mode == "train":
        _label_checks(data, schema, errors, warnings)
    return errors, warnings


def _label_checks(
    data: CanonicalData,
    schema: DatasetSchema,
    errors: list[TableIssue],
    warnings: list[TableIssue],
) -> None:
    if data.y is None:
        errors.append(TableIssue("validation.no_labels", "Aucune cible : entraînement impossible."))
        return
    for name, spec in schema.enabled_targets().items():
        column = data.y.get(name.value)
        if column is None or not bool(column.notna().any()):
            errors.append(
                TableIssue(
                    "validation.target_empty",
                    f"Cible « {spec.label} » : aucune valeur renseignée.",
                    name.value,
                )
            )
            continue
        missing = int(column.isna().sum()) - _not_applicable(data, schema, name)
        if missing > 0:
            warnings.append(
                TableIssue(
                    "validation.missing_labels",
                    f"Cible « {spec.label} » : {missing} ligne(s) sans valeur, ignorée(s) "
                    "pour cette cible.",
                    name.value,
                )
            )
        if name in STRING_TARGETS:
            counts = column.dropna().value_counts()
            rare = sorted(str(c) for c, k in counts.items() if int(k) < spec.min_class_count)
            if rare:
                warnings.append(
                    TableIssue(
                        "validation.rare_classes",
                        f"Cible « {spec.label} » : classe(s) rare(s) {rare} "
                        f"(< {spec.min_class_count} cas), regroupées ou écartées.",
                        name.value,
                    )
                )
            if counts.size < 2:
                # Classes inferred from the data (molecule): a constant model is used
                # (``models.tasks``), so a single class is not blocking.
                inferred = spec.classes is None
                suffix = " (prédiction constante)" if inferred else ""
                (warnings if inferred else errors).append(
                    TableIssue(
                        "validation.single_class",
                        f"Cible « {spec.label} » : une seule classe observée{suffix}.",
                        name.value,
                    )
                )


def _not_applicable(data: CanonicalData, schema: DatasetSchema, name: TargetName) -> int:
    """Rows where a conditional target does not apply (they are missing by design)."""
    condition = schema.enabled_targets()[name].conditional_on
    if condition is None or data.y is None or condition.target.value not in data.y:
        return 0
    holds = (data.y[condition.target.value] == condition.equals).fillna(value=False)
    return int((~holds.to_numpy(dtype=bool)).sum())


def validate(data: CanonicalData, schema: DatasetSchema, mode: ValidationMode) -> ValidationReport:
    """Validate ``data`` row by row and at table level (see module docstring)."""
    errors, warnings = _table_checks(data, schema, mode)
    row_issues = _row_issues(data, schema, mode)
    if mode == "train" and row_issues:
        errors.append(
            TableIssue(
                "validation.invalid_rows",
                f"{len({i.row_id for i in row_issues})} ligne(s) avec des valeurs invalides.",
            )
        )
    return ValidationReport(
        mode=mode,
        ok=not errors,
        n_rows=data.n_rows,
        errors=tuple(errors),
        warnings=tuple(warnings),
        row_issues=tuple(row_issues),
    )
