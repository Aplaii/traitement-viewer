"""Reading raw tables (CSV / XLSX) safely. Owner: WS1.

Raw table contract (:class:`RawTable.df`)
    * CSV: every column is text (:data:`~sevrage_ai.types.STRING_DTYPE`); empty cells are
      ``<NA>`` (``na_values`` are handled later, by the mapping). The separator and the
      encoding are sniffed when set to ``"auto"`` (French Excel exports: ``;``, cp1252 or
      utf-8-sig). Decimal commas are handled by the mapping, whatever ``decimal`` says.
    * XLSX: first sheet (or ``sheet``), native cell values kept in ``object`` columns
      (``None``/``NaN`` = missing).
    * Headers are stripped. Headers that collide once normalised (case, spaces) get a
      ``.1``, ``.2``... suffix and a warning (REDCap exports repeat some labels); empty
      headers are named ``colonne_<position>``. CSV rows with trailing separators are read
      without shifting their cells.
    * The ``__synthetic__`` marker column is removed; its **presence** sets
      ``is_synthetic=True``.
    * Guards: file size, number of rows and number of columns (``CsvOptions``).
    * ``sha256`` is computed on the raw bytes; ``source_name`` is the file name only (never
      a full path, which may contain user names).

Errors raise :class:`~sevrage_ai.errors.DataValidationError` with a French message that
contains no cell values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final, Literal, Self

import pandas as pd

from sevrage_ai.config.schema import DatasetSpec, normalize_column_name
from sevrage_ai.data.text_parsing import cell_text
from sevrage_ai.errors import DataValidationError
from sevrage_ai.types import STRING_DTYPE, SYNTHETIC_MARKER_COLUMN

__all__ = [
    "MAX_COLUMNS",
    "MAX_FILE_BYTES",
    "MAX_ROWS",
    "SUPPORTED_SUFFIXES",
    "CsvOptions",
    "RawTable",
    "read_bytes",
    "read_table",
]

MAX_FILE_BYTES: Final = 20 * 1024 * 1024
"""Default maximum input size (20 MB)."""
MAX_ROWS: Final = 100_000
MAX_COLUMNS: Final = 2_000
SUPPORTED_SUFFIXES: Final = frozenset({".csv", ".txt", ".tsv", ".xlsx"})
_ENCODINGS: Final = ("utf-8-sig", "cp1252")
_SEPARATORS: Final = (";", ",", "\t", "|")


@dataclass(frozen=True, slots=True)
class CsvOptions:
    """Reading options. Build them from the schema with :meth:`from_dataset`."""

    sep: Literal["auto", ",", ";", "\t", "|"] = "auto"
    decimal: Literal["auto", ".", ","] = "auto"
    encoding: Literal["auto", "utf-8", "utf-8-sig", "cp1252", "latin-1"] = "auto"
    sheet: str | int | None = None
    na_values: tuple[str, ...] = ()
    max_bytes: int = MAX_FILE_BYTES
    max_rows: int = MAX_ROWS
    max_columns: int = MAX_COLUMNS

    @classmethod
    def from_dataset(cls, dataset: DatasetSpec) -> Self:
        """Options from ``schema.dataset`` (``csv`` block and ``na_values``), default guards."""
        return cls(
            sep=dataset.csv.sep,
            decimal=dataset.csv.decimal,
            encoding=dataset.csv.encoding,
            sheet=dataset.csv.sheet,
            na_values=dataset.na_values,
        )


@dataclass(frozen=True, slots=True, eq=False)
class RawTable:
    """A raw table as read from disk or from an upload (see module docstring)."""

    df: pd.DataFrame
    is_synthetic: bool
    sha256: str
    source_name: str
    warnings: tuple[str, ...] = ()


def read_table(path: Path, opts: CsvOptions) -> RawTable:
    """Read a CSV/TXT/TSV/XLSX file from disk.

    Raises:
        DataValidationError: unsupported extension, too large, unreadable, empty, too many
            rows or columns.
    """
    _check_suffix(path.name)
    try:
        size = path.stat().st_size
    except OSError as exc:
        msg = f"Fichier introuvable : « {path.name} »."
        raise DataValidationError(msg) from exc
    _check_size(size, path.name, opts)
    try:
        data = path.read_bytes()
    except OSError as exc:
        msg = f"Lecture impossible : « {path.name} »."
        raise DataValidationError(msg) from exc
    return read_bytes(data, path.name, opts)


def read_bytes(data: bytes, filename: str, opts: CsvOptions) -> RawTable:
    """Read an in-memory upload (Streamlit). Never writes to disk.

    ``filename`` is only used for its extension and for ``RawTable.source_name``.

    Raises:
        DataValidationError: see :func:`read_table`.
    """
    # A browser upload may carry a full client-side path, with / or \ separators whatever
    # the server's platform: PureWindowsPath splits on both.
    name = PureWindowsPath(filename).name
    suffix = _check_suffix(name)
    _check_size(len(data), name, opts)
    frame = (
        _read_excel(data, name, opts) if suffix == ".xlsx" else _read_delimited(data, name, opts)
    )
    if frame.shape[1] == 0 or frame.shape[0] == 0:
        msg = f"Le fichier « {name} » est vide (aucune ligne de données)."
        raise DataValidationError(msg)
    if frame.shape[0] > opts.max_rows or frame.shape[1] > opts.max_columns:
        msg = f"Le fichier « {name} » est trop grand."
        details = [
            f"{frame.shape[0]} lignes (max {opts.max_rows})",
            f"{frame.shape[1]} colonnes (max {opts.max_columns})",
        ]
        raise DataValidationError(msg, details=details)
    frame, notes = _clean_headers(frame)
    marker = [c for c in frame.columns if normalize_column_name(c) == SYNTHETIC_MARKER_COLUMN]
    frame = frame.drop(columns=marker).reset_index(drop=True)
    return RawTable(
        df=frame,
        is_synthetic=bool(marker),
        sha256=hashlib.sha256(data).hexdigest(),
        source_name=name,
        warnings=tuple(notes),
    )


def _check_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        msg = f"Extension non prise en charge pour « {name} » (attendu : CSV ou XLSX)."
        raise DataValidationError(msg, details=[f"extensions : {sorted(SUPPORTED_SUFFIXES)}"])
    return suffix


def _check_size(size: int, name: str, opts: CsvOptions) -> None:
    if size > opts.max_bytes:
        msg = f"Le fichier « {name} » dépasse la taille maximale autorisée."
        raise DataValidationError(msg, details=[f"{size} octets > {opts.max_bytes}"])
    if size == 0:
        msg = f"Le fichier « {name} » est vide."
        raise DataValidationError(msg)


def _read_excel(data: bytes, name: str, opts: CsvOptions) -> pd.DataFrame:
    sheet: str | int = 0 if opts.sheet is None else opts.sheet
    try:
        with warnings.catch_warnings():
            # openpyxl warns about cosmetic workbook features (styles, data validation).
            warnings.simplefilter("ignore", UserWarning)
            frame = pd.read_excel(
                io.BytesIO(data), sheet_name=sheet, header=None, dtype=object, engine="openpyxl"
            )
    except Exception as exc:  # openpyxl raises many unrelated types on corrupt files
        msg = f"Lecture du classeur impossible : « {name} »."
        raise DataValidationError(msg, details=[type(exc).__name__]) from exc
    return _promote_header(frame.dropna(how="all"))


def _promote_header(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the first row as the (raw) header: pandas would silently rename duplicates."""
    if frame.shape[0] == 0:
        return frame.reset_index(drop=True)
    body = frame.iloc[1:].reset_index(drop=True)
    body.columns = pd.Index(frame.iloc[0].tolist(), dtype=object)
    return body


def _decode(data: bytes, name: str, encoding: str) -> str:
    candidates = _ENCODINGS if encoding == "auto" else (encoding,)
    for candidate in candidates:
        try:
            # An explicit "utf-8" keeps the BOM of Excel's "CSV UTF-8" exports.
            return data.decode(candidate).removeprefix("﻿")
        except UnicodeDecodeError:
            continue
    msg = f"Encodage du fichier « {name} » non reconnu (UTF-8 ou Windows-1252 attendu)."
    raise DataValidationError(msg)


def _sniff_separator(text: str) -> str:
    """The separator giving the same number (> 1) of fields on the first rows, preferring
    the most fields; ``;`` first on ties (French exports). Quotes are honoured."""
    sample = text.splitlines()[:30]
    best, best_score = ",", (False, 0)
    for sep in _SEPARATORS:
        try:
            counts = {len(row) for row in csv.reader(sample, delimiter=sep) if row}
        except csv.Error:
            continue
        width = max(counts, default=0)
        score = (len(counts) == 1 and width > 1, width)
        if score > best_score:
            best, best_score = sep, score
    return best


def _read_delimited(data: bytes, name: str, opts: CsvOptions) -> pd.DataFrame:
    text = _decode(data, name, opts.encoding)
    sep = _sniff_separator(text) if opts.sep == "auto" else opts.sep
    try:
        # index_col=False: rows ending with a separator must not turn the first column
        # into the index (which would shift every cell of the row by one column).
        frame = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            skip_blank_lines=True,
            index_col=False,
        )
        # The raw header row: pandas renames duplicate headers ("a", "a.1") silently.
        header = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            header=None,
            nrows=1,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            skip_blank_lines=True,
        )
        frame.columns = pd.Index(header.iloc[0].tolist(), dtype=object)
    except (pd.errors.ParserError, ValueError) as exc:
        msg = f"Lecture du fichier « {name} » impossible (séparateur ou guillemets incohérents)."
        raise DataValidationError(msg, details=[type(exc).__name__]) from exc
    return frame.astype(STRING_DTYPE)


def _clean_headers(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    used: set[str] = set()
    headers: list[str] = []
    renamed = 0
    for position, column in enumerate(frame.columns):
        base = cell_text(column) or f"colonne_{position + 1}"
        header, suffix = base, 0
        while normalize_column_name(header) in used:
            suffix += 1
            header = f"{base}.{suffix}"
        if suffix:
            renamed += 1
        used.add(normalize_column_name(header))
        headers.append(header)
    notes = []
    if renamed:
        notes.append(f"{renamed} en-tête(s) en double renommé(s) avec un suffixe « .1 », « .2 »...")
    cleaned = frame.copy()
    cleaned.columns = pd.Index(headers)
    return cleaned, notes
