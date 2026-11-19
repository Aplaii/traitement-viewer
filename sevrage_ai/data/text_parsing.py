"""Parsing of messy raw cells: lenient numbers, leading numbers and free-text prescriptions.

Hospital exports mix numbers and comments in the same cell. The rules below are
deliberately conservative: anything ambiguous is *unparseable* (it becomes missing and is
counted in the mapping report), never guessed.

Numbers (:func:`parse_number`)
    ``"12"``, ``"12,5"``, ``"12.5"``, ``"3.57 g/L de sang"`` (a number followed by text
    **without digits**), ``"<5"`` / ``"> 120"`` (censored: read as the bound, or missing).
    Ranges (``"20 - 50"``), several numbers (``"31  25 dans APAddicto"``) and words
    (``"Adolescence"``) are unparseable.

Prescriptions (:func:`parse_prescription`)
    The text is split into segments, one per run of mentions of the same molecule; the
    text before the first mention belongs to the first segment (``"180 mg Diazepam"``).
    A segment's amount is the last number after ``=`` when it details a sum, else its
    first ``<number> mg``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from sevrage_ai.config.schema import fold_text

__all__ = [
    "Prescription",
    "cell_text",
    "compile_patterns",
    "is_missing_cell",
    "matches_any",
    "parse_leading_number",
    "parse_number",
    "parse_prescription",
]

_NUMBER_RE: Final = re.compile(
    r"^(?P<cmp>[<>≤≥]\s*=?)?\s*=?\s*(?P<num>[-+]?\d+(?:[.,]\d+)?)(?P<rest>\D*)$"
)
_LEADING_RE: Final = re.compile(r"^\s*(?P<num>\d+(?:[.,]\d+)?)")
_SUM_RE: Final = re.compile(r"=\s*(\d+(?:[.,]\d+)?)")
_MG_RE: Final = re.compile(r"(\d+(?:[.,]\d+)?)\s*mg\b")
_QUOTES: Final = "\"'«»“”"


def is_missing_cell(value: object) -> bool:
    """``True`` for ``None``, ``NaN``, ``pd.NA`` and ``NaT``."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and math.isnan(float(value))


def cell_text(value: object) -> str | None:
    """Text of a raw cell with whitespace collapsed, or ``None`` when missing or blank.

    Integral floats lose their ``.0`` (``1.0`` -> ``"1"``), so that XLSX numbers match the
    text keys of ``value_mapping``; dates use the ISO format.
    """
    if is_missing_cell(value):
        return None
    if isinstance(value, bool | np.bool_):
        return "1" if value else "0"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    text = " ".join(str(value).split())
    return text or None


def parse_number(value: object, *, censored: Literal["bound", "missing"]) -> float | None:
    """Parse a raw cell as a number (rules in the module docstring).

    Returns ``nan`` for a censored value when ``censored="missing"`` and ``None`` when the
    cell is not a number (the caller counts it as unparseable). Missing cells must be
    filtered out by the caller.
    """
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None  # dates, times and other objects
    text = " ".join(value.split()).strip(_QUOTES).strip()
    match = _NUMBER_RE.match(text)
    if match is None:
        return None
    if match.group("cmp") and censored == "missing":
        return math.nan
    return float(match.group("num").replace(",", "."))


def parse_leading_number(text: str) -> float | None:
    """The number at the very start of ``text`` (``"7 jours"`` -> 7), else ``None``."""
    match = _LEADING_RE.match(text)
    return None if match is None else float(match.group("num").replace(",", "."))


def compile_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Compile schema regexes (they are matched against :func:`fold_text` output)."""
    return tuple(re.compile(pattern) for pattern in patterns)


def matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    """``True`` if one of ``patterns`` is found in the folded ``text``."""
    folded = fold_text(text)
    return any(pattern.search(folded) for pattern in patterns)


@dataclass(frozen=True, slots=True)
class Prescription:
    """Molecules named in a free-text prescription, in order, with their amounts (mg of
    that molecule, ``None`` when no amount could be read)."""

    segments: tuple[tuple[str, float | None], ...]

    @property
    def first_molecule(self) -> str | None:
        """The molecule named first (the initial prescribing choice), if any."""
        return self.segments[0][0] if self.segments else None

    def total_diazepam_equivalent(self, equivalence: Mapping[str, float]) -> float | None:
        """Sum of the amounts converted to mg diazepam-equivalent.

        ``None`` when no molecule is named, or when one segment has no amount or its
        molecule has no equivalence (a partial total would be misleading).
        """
        if not self.segments:
            return None
        total = 0.0
        for molecule, amount in self.segments:
            factor = equivalence.get(molecule)
            if amount is None or factor is None:
                return None
            total += amount * factor
        return total


def _alias_regex(aliases: Mapping[str, Sequence[str]]) -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for molecule, patterns in aliases.items():
        for pattern in (re.escape(molecule), *patterns):
            compiled.append((re.compile(pattern), molecule))
    return compiled


def _segment_amount(segment: str) -> float | None:
    sums = _SUM_RE.findall(segment)
    if sums:
        return float(sums[-1].replace(",", "."))
    match = _MG_RE.search(segment)
    return None if match is None else float(match.group(1).replace(",", "."))


def parse_prescription(text: str, aliases: Mapping[str, Sequence[str]]) -> Prescription:
    """Read the molecules and amounts of a free-text prescription (module docstring)."""
    folded = fold_text(text)
    mentions: list[tuple[int, int, str]] = []
    for pattern, molecule in _alias_regex(aliases):
        mentions += [(m.start(), m.end(), molecule) for m in pattern.finditer(folded)]
    mentions.sort(key=lambda item: (item[0], -item[1]))
    kept: list[tuple[int, int, str]] = []
    for start, end, molecule in mentions:
        if kept and start < kept[-1][1]:
            continue  # overlapping alias (e.g. "oxazepam" inside a longer match)
        kept.append((start, end, molecule))
    runs: list[tuple[int, str]] = []
    for start, _end, molecule in kept:
        if not runs or runs[-1][1] != molecule:
            runs.append((start, molecule))
    segments: list[tuple[str, float | None]] = []
    for position, (start, molecule) in enumerate(runs):
        begin = 0 if position == 0 else start
        stop = runs[position + 1][0] if position + 1 < len(runs) else len(folded)
        segments.append((molecule, _segment_amount(folded[begin:stop])))
    return Prescription(segments=tuple(segments))
