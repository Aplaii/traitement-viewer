"""Data fingerprint stored in bundle metadata. Owner: WS1."""

from __future__ import annotations

import hashlib

import pandas as pd

__all__ = ["data_fingerprint"]


def data_fingerprint(df: pd.DataFrame) -> str:
    """Hex sha256 of the canonical CSV bytes of ``df``.

    Columns are sorted by name; index excluded; ``,`` separator; ``.`` decimal; missing
    written as empty; ``\\n`` line endings. Independent of column order and of the platform.
    """
    # Positional selection: works for non-string and duplicate column labels, which a
    # reindex on the str-converted names would turn into all-missing columns (or reject).
    order = sorted(range(df.shape[1]), key=lambda position: str(df.columns[position]))
    ordered = df.iloc[:, order]
    text = ordered.to_csv(index=False, sep=",", decimal=".", na_rep="", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
