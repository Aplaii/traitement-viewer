"""Export of predictions (CSV / JSON), with formula-injection protection. Owner: WS8a.

Every text cell starting with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return is
prefixed with ``'`` before being written, so spreadsheet software never runs it as a
formula. Rejected rows appear with their issue codes and French messages.

``to_frame`` produces one row per input row (predictions and rejected rows merged back in
their original order, best-effort on ``row_id`` when it is not a plain row number) with a
fixed set of columns, plus one probability column per class actually seen in the batch for
``severity`` and ``molecule_if_treated`` (multiclass targets); ``treatment`` stays a single
``treatment_proba`` column (probability of :data:`~sevrage_ai.types.TREATED`) since it is
binary. Only :class:`~sevrage_ai.types.FinalRecommendation` values, model outputs and
metadata are exported: no patient feature value ever appears, beyond the row id.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from sevrage_ai.types import (
    TREATED,
    BatchResult,
    ClassOutput,
    ConfidenceCheck,
    FinalRecommendation,
    Flag,
    PatientPrediction,
    RegressionOutput,
    RowIssue,
)

__all__ = [
    "FORMULA_PREFIXES",
    "escape_formula",
    "prediction_to_dict",
    "to_csv_bytes",
    "to_frame",
    "to_json",
    "write_csv",
]

FORMULA_PREFIXES: Final = ("=", "+", "-", "@", "\t", "\r")

_FLAG_SEP: Final = "|"
_MESSAGE_SEP: Final = "; "
_REASON_SEP: Final = " | "


def escape_formula(value: str) -> str:
    """Prefix ``value`` with ``'`` if it starts with one of :data:`FORMULA_PREFIXES`."""
    return f"'{value}" if value.startswith(FORMULA_PREFIXES) else value


def _finite_or_none(value: float) -> float | None:
    """``value`` unless it is infinite (unbounded conformal interval): then ``None``."""
    return value if math.isfinite(value) else None


def _check_to_dict(check: ConfidenceCheck) -> dict[str, object]:
    return {
        "code": check.code,
        "met": check.met,
        "caps_at": None if check.caps_at is None else check.caps_at.value,
        "message_fr": check.message_fr,
    }


def _class_output_to_dict(output: ClassOutput) -> dict[str, object]:
    return {
        "label": output.label,
        "probabilities": {label: float(p) for label, p in output.probabilities.items()},
        "prediction_set": sorted(output.prediction_set),
        "confidence": output.confidence.value,
        "reasons": list(output.reasons),
        "checks": [_check_to_dict(check) for check in output.checks],
    }


def _regression_output_to_dict(output: RegressionOutput | None) -> dict[str, object] | None:
    if output is None:
        return None
    return {
        "point": float(output.point),
        "lower": _finite_or_none(float(output.lower)),
        "upper": _finite_or_none(float(output.upper)),
        "unit": output.unit,
        "confidence": output.confidence.value,
        "reasons": list(output.reasons),
        "checks": [_check_to_dict(check) for check in output.checks],
    }


def _final_to_dict(final: FinalRecommendation) -> dict[str, object]:
    return {
        "treat": final.treat,
        "molecule": final.molecule,
        "dose_day1_dze": final.dose_day1_dze,
        "dose_day1_molecule_mg": final.dose_day1_molecule_mg,
        "dose_cumulative_dze": final.dose_cumulative_dze,
        "dose_cumulative_molecule_mg": final.dose_cumulative_molecule_mg,
        "duration_days": final.duration_days,
    }


def _flag_to_dict(flag: Flag) -> dict[str, object]:
    return {
        "code": flag.code,
        "level": flag.level,
        "message_fr": flag.message_fr,
        "feature": flag.feature,
    }


def _issue_to_dict(issue: RowIssue) -> dict[str, object]:
    return {
        "row_id": issue.row_id,
        "column": issue.column,
        "code": issue.code,
        "message_fr": issue.message_fr,
    }


def prediction_to_dict(prediction: PatientPrediction) -> dict[str, object]:
    """JSON-serialisable nested dictionary of one prediction (for the JSON download).

    Excludes ``explanations``: they display the patient's own feature values, which never
    leave the interactive UI.
    """
    return {
        "row_id": prediction.row_id,
        "model_id": prediction.model_id,
        "trained_on_synthetic": prediction.trained_on_synthetic,
        "overall_confidence": prediction.overall_confidence.value,
        "severity": _class_output_to_dict(prediction.severity),
        "treatment": _class_output_to_dict(prediction.treatment),
        "molecule_if_treated": _class_output_to_dict(prediction.molecule_if_treated),
        "dose_day1_if_treated": _regression_output_to_dict(prediction.dose_day1_if_treated),
        "dose_cumulative_if_treated": _regression_output_to_dict(
            prediction.dose_cumulative_if_treated
        ),
        "duration_if_treated": _regression_output_to_dict(prediction.duration_if_treated),
        "final": _final_to_dict(prediction.final),
        "flags": [_flag_to_dict(flag) for flag in prediction.flags],
        "overall_checks": [_check_to_dict(check) for check in prediction.overall_checks],
    }


def _classes(
    predictions: Sequence[PatientPrediction], getter: Callable[[PatientPrediction], ClassOutput]
) -> list[str]:
    """Sorted union of the classes seen across ``predictions`` for one classification target."""
    seen: set[str] = set()
    for prediction in predictions:
        seen.update(getter(prediction).probabilities)
    return sorted(seen)


def _column_order(severity_classes: Sequence[str], molecule_classes: Sequence[str]) -> list[str]:
    return [
        "row_id",
        "status",
        "severity_pred",
        *[f"severity_proba_{c}" for c in severity_classes],
        "severity_set",
        "severity_confidence",
        "treat",
        "treatment_proba",
        "treatment_confidence",
        "molecule",
        "molecule_if_treated_pred",
        *[f"molecule_proba_{c}" for c in molecule_classes],
        "molecule_set",
        "molecule_confidence",
        "dose_day1_mg_eq_diazepam",
        "dose_day1_lo",
        "dose_day1_hi",
        "dose_day1_molecule_mg",
        "dose_cumulative_mg_eq_diazepam",
        "dose_cumulative_lo",
        "dose_cumulative_hi",
        "dose_cumulative_molecule_mg",
        "duration_days",
        "duration_lo",
        "duration_hi",
        "confidence_global",
        "confidence_global_reasons",
        "flags",
        "model_id",
        "trained_on_synthetic",
        "rejection_codes",
        "rejection_message",
    ]


def _prediction_row(
    prediction: PatientPrediction,
    severity_classes: Sequence[str],
    molecule_classes: Sequence[str],
) -> dict[str, object]:
    severity = prediction.severity
    treatment = prediction.treatment
    molecule = prediction.molecule_if_treated
    final = prediction.final
    day1 = prediction.dose_day1_if_treated
    cumulative = prediction.dose_cumulative_if_treated
    duration = prediction.duration_if_treated
    row: dict[str, object] = {
        "row_id": prediction.row_id,
        "status": "ok",
        "severity_pred": severity.label,
        "severity_set": _FLAG_SEP.join(sorted(severity.prediction_set)),
        "severity_confidence": severity.confidence.value,
        "treat": final.treat,
        "treatment_proba": float(treatment.probabilities.get(TREATED, 0.0)),
        "treatment_confidence": treatment.confidence.value,
        "molecule": final.molecule,
        "molecule_if_treated_pred": molecule.label,
        "molecule_set": _FLAG_SEP.join(sorted(molecule.prediction_set)),
        "molecule_confidence": molecule.confidence.value,
        "dose_day1_mg_eq_diazepam": final.dose_day1_dze,
        "dose_day1_lo": None if day1 is None else _finite_or_none(day1.lower),
        "dose_day1_hi": None if day1 is None else _finite_or_none(day1.upper),
        "dose_day1_molecule_mg": final.dose_day1_molecule_mg,
        "dose_cumulative_mg_eq_diazepam": final.dose_cumulative_dze,
        "dose_cumulative_lo": None if cumulative is None else _finite_or_none(cumulative.lower),
        "dose_cumulative_hi": None if cumulative is None else _finite_or_none(cumulative.upper),
        "dose_cumulative_molecule_mg": final.dose_cumulative_molecule_mg,
        "duration_days": final.duration_days,
        "duration_lo": None if duration is None else _finite_or_none(duration.lower),
        "duration_hi": None if duration is None else _finite_or_none(duration.upper),
        "confidence_global": prediction.overall_confidence.value,
        "confidence_global_reasons": _REASON_SEP.join(
            check.message_fr for check in prediction.overall_checks if not check.met
        ),
        "flags": _FLAG_SEP.join(flag.code for flag in prediction.flags),
        "model_id": prediction.model_id,
        "trained_on_synthetic": prediction.trained_on_synthetic,
    }
    for label in severity_classes:
        row[f"severity_proba_{label}"] = float(severity.probabilities.get(label, 0.0))
    for label in molecule_classes:
        row[f"molecule_proba_{label}"] = float(molecule.probabilities.get(label, 0.0))
    return row


def _rejected_rows(issues: Sequence[RowIssue]) -> list[dict[str, object]]:
    grouped: dict[str, list[RowIssue]] = {}
    for issue in issues:
        grouped.setdefault(issue.row_id, []).append(issue)
    return [
        {
            "row_id": row_id,
            "status": "rejete",
            "rejection_codes": _FLAG_SEP.join(issue.code for issue in group),
            "rejection_message": _MESSAGE_SEP.join(issue.message_fr for issue in group),
        }
        for row_id, group in grouped.items()
    ]


def _row_sort_key(row_id: str) -> tuple[int, int | str]:
    """Best-effort original order: numeric row ids first (numerically), then alphabetical."""
    return (0, int(row_id)) if row_id.isdigit() else (1, row_id)


def to_frame(result: BatchResult) -> pd.DataFrame:
    """One row per input row: predictions and final recommendation, or the rejection reason."""
    severity_classes = _classes(result.predictions, lambda p: p.severity)
    molecule_classes = _classes(result.predictions, lambda p: p.molecule_if_treated)
    rows = [
        _prediction_row(prediction, severity_classes, molecule_classes)
        for prediction in result.predictions
    ]
    rows.extend(_rejected_rows(result.rejected))
    rows.sort(key=lambda row: _row_sort_key(str(row["row_id"])))
    columns = _column_order(severity_classes, molecule_classes)
    return pd.DataFrame(rows, columns=columns)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Escaped CSV (UTF-8 with BOM, ``;`` separator), in memory (Streamlit download)."""
    escaped = df.copy()
    for column in escaped.columns:
        # ``object`` columns may mix text with numbers or None (``is_string_dtype`` is then
        # False), and pandas >= 3 has its own ``str`` dtype: both are escaped cell by cell.
        series = escaped[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            escaped[column] = escaped[column].map(
                lambda value: escape_formula(value) if isinstance(value, str) else value
            )
    text = escaped.to_csv(sep=";", decimal=",", index=False, lineterminator="\n")
    return text.encode("utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write :func:`to_csv_bytes` to ``path`` and return it."""
    path.write_bytes(to_csv_bytes(df))
    return path


def to_json(result: BatchResult) -> str:
    """Whole batch as JSON (``predictions`` and ``rejected``)."""
    payload = {
        "predictions": [prediction_to_dict(prediction) for prediction in result.predictions],
        "rejected": [_issue_to_dict(issue) for issue in result.rejected],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
