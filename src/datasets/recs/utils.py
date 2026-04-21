"""Shared helpers for RECS analysis scripts (02–06).

Centralises the curated-file loader and TYPEHUQ filter that were previously
duplicated across every analysis script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_curated(path: Path) -> pd.DataFrame:
    """Read a curated RECS parquet (or CSV fallback)."""
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def filter_unit_type(df: pd.DataFrame, unit_type: str) -> pd.DataFrame:
    """Filter by TYPEHUQ if the column is present.

    Parameters
    ----------
    unit_type:
        ``"mf"`` → TYPEHUQ 3–4, ``"sf"`` → TYPEHUQ 1–2, ``"all"`` → no filter.
    """
    if "TYPEHUQ" not in df.columns or unit_type == "all":
        return df
    if unit_type == "mf":
        return df[df["TYPEHUQ"].isin([3, 4])].copy()
    if unit_type == "sf":
        return df[df["TYPEHUQ"].isin([1, 2])].copy()
    return df
