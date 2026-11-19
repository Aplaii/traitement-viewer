"""Heuristic identification checks, run before training and batch prediction. Owner: WS1.

Checks column names against ``anonymization.forbidden_column_patterns`` (blocking). When
``scan_values`` is set, also scans text cells (and numeric cells written as text, so that
an XLSX number is checked like its CSV export) for NIR numbers (including the Corsica
codes 2A/2B), e-mail addresses and French phone numbers (blocking), and for full dates and
high-cardinality free-text columns (non-blocking warnings: they are quasi-identifiers
that the data protection officer must assess, e.g. a date of care in a comment).

**Reports contain column names and counts only, never the offending values.** Passing the
check does not prove the data are anonymous: the data controller or data protection
officer must confirm it.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from sevrage_ai.config.schema import AnonymizationSpec
from sevrage_ai.data.text_parsing import cell_text
from sevrage_ai.errors import AnonymizationError

__all__ = [
    "AnonymizationFinding",
    "AnonymizationReport",
    "FindingKind",
    "check",
    "ensure_anonymized",
]

FindingKind = Literal["column_name", "nir", "email", "phone", "full_date", "free_text"]

_NIR_RE: Final = re.compile(
    r"(?<![\d.,])[12]\s?\d{2}\s?(?:0[1-9]|1[0-2]|[2-9]\d)\s?(?:\d{2}|2[AaBb])\s?\d{3}\s?\d{3}"
    r"(?:\s?\d{2})?(?!\d)"
)
_EMAIL_RE: Final = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE: Final = re.compile(r"(?<![\d.,])(?:(?:\+|00)33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)")
_DATE_RE: Final = re.compile(
    r"(?<!\d)(?:\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2})(?!\d)"
)
_DECIMAL_RE: Final = re.compile(r"[-+]?\d+[.,]\d+")
"""A plain decimal number (e.g. a ratio exported as text) is never an identifier."""
_VALUE_PATTERNS: Final[tuple[tuple[FindingKind, re.Pattern[str], bool], ...]] = (
    ("nir", _NIR_RE, True),
    ("email", _EMAIL_RE, True),
    ("phone", _PHONE_RE, True),
    ("full_date", _DATE_RE, False),
)
_LABELS: Final[dict[FindingKind, str]] = {
    "column_name": "nom de colonne évoquant une donnée identifiante",
    "nir": "cellule(s) ressemblant à un numéro de sécurité sociale (NIR)",
    "email": "cellule(s) ressemblant à une adresse e-mail",
    "phone": "cellule(s) ressemblant à un numéro de téléphone",
    "full_date": "cellule(s) contenant une date complète (quasi-identifiant)",
    "free_text": "texte libre à forte variété (peut contenir des informations identifiantes)",
}


@dataclass(frozen=True, slots=True)
class AnonymizationFinding:
    """One suspicious column. ``count`` is the number of matching cells (1 for names)."""

    column: str
    kind: FindingKind
    count: int
    blocking: bool
    message_fr: str


@dataclass(frozen=True, slots=True)
class AnonymizationReport:
    """Result of :func:`check`. ``ok`` is ``False`` if any finding is blocking."""

    ok: bool
    findings: tuple[AnonymizationFinding, ...]
    n_columns: int
    n_rows: int

    def blocking(self) -> tuple[AnonymizationFinding, ...]:
        """The blocking findings."""
        return tuple(f for f in self.findings if f.blocking)


def _finding(column: str, kind: FindingKind, count: int, blocking: bool) -> AnonymizationFinding:
    suffix = "" if kind == "column_name" else f" : {count}"
    message = f"Colonne « {column} »{suffix} {_LABELS[kind]}."
    return AnonymizationFinding(column, kind, count, blocking, message)


def _number_texts(values: list[object]) -> list[str]:
    """Numeric cells (XLSX) as text, so that a NIR typed as a number is found like in CSV."""
    texts: list[str] = []
    for value in values:
        if isinstance(value, int | float | np.integer | np.floating) and not isinstance(
            value, bool | np.bool_
        ):
            text = cell_text(value)
            if text is not None:
                texts.append(text)
    return texts


def _scan_column(
    column: str, series: pd.Series[Any], spec: AnonymizationSpec, exempt: bool
) -> list[AnonymizationFinding]:
    values = series.tolist()
    texts = [v for v in values if isinstance(v, str) and v.strip()]
    candidates = texts + _number_texts(values)
    words = [t for t in candidates if not _DECIMAL_RE.fullmatch(t.strip())]
    findings: list[AnonymizationFinding] = []
    for kind, pattern, blocking in _VALUE_PATTERNS:
        count = sum(1 for text in words if pattern.search(text))
        if kind == "full_date":
            count += sum(1 for v in values if isinstance(v, dt.date | pd.Timestamp))
        if count:
            findings.append(_finding(column, kind, count, blocking))
    non_missing = len(texts)
    if not exempt and non_missing >= spec.min_rows_for_cardinality:
        ratio = len(set(texts)) / non_missing
        numeric_like = sum(1 for t in texts if re.fullmatch(r"[<>]?\s*[\d.,\s]+", t)) / non_missing
        if ratio > spec.high_cardinality_ratio and numeric_like < 0.5:
            findings.append(_finding(column, "free_text", non_missing, False))
    return findings


def check(
    raw: pd.DataFrame,
    spec: AnonymizationSpec,
    *,
    exempt_columns: Collection[str] = (),
) -> AnonymizationReport:
    """Scan ``raw`` (raw headers and raw values).

    ``exempt_columns`` (typically ``dataset.id_column`` and ``group_column``) skip the
    free-text cardinality check only. Name and value patterns still apply to them.
    """
    exempt = {" ".join(c.split()).casefold() for c in exempt_columns}
    patterns = spec.compiled_patterns()
    findings: list[AnonymizationFinding] = []
    for column in raw.columns:
        name = str(column)
        if any(pattern.search(name) for pattern in patterns):
            findings.append(_finding(name, "column_name", 1, True))
        if spec.scan_values:
            is_exempt = " ".join(name.split()).casefold() in exempt
            findings += _scan_column(name, raw[column], spec, is_exempt)
    return AnonymizationReport(
        ok=not any(f.blocking for f in findings),
        findings=tuple(findings),
        n_columns=raw.shape[1],
        n_rows=raw.shape[0],
    )


def ensure_anonymized(report: AnonymizationReport) -> None:
    """Raise if ``report`` has blocking findings.

    Raises:
        AnonymizationError: with the column names and counts in ``details``.
    """
    blocking = report.blocking()
    if not blocking:
        return
    msg = (
        f"Le fichier semble contenir des données identifiantes ({len(blocking)} colonne(s)). "
        "Retirez-les avant de continuer."
    )
    raise AnonymizationError(msg, details=[f.message_fr for f in blocking])
