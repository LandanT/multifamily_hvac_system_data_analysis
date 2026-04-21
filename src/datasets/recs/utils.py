"""Shared helpers for RECS analysis scripts (02–07).

Centralises the curated-file loader, TYPEHUQ filter, segment filter, and
classification-mode filter that were previously duplicated across scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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


# ---------------------------------------------------------------------------
# Segment and classification-mode helpers (added for heating refocus)
# ---------------------------------------------------------------------------

MF_SEGMENTS = {
    "all_mf": None,
    "2_to_4_units": "2_to_4_units",
    "5plus_units": "5plus_units",
}

FUEL_STRATA = {
    "all_fuels": None,
    "electric": "ELWARM",
    "gas": "UGWARM",
}

CLASSIFICATION_VIEWS = ["pooled_binary", "explicit_only", "inferred_only"]


def filter_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Filter to a multifamily segment.

    Parameters
    ----------
    segment:
        ``"all_mf"`` → no additional filter,
        ``"2_to_4_units"`` → TYPEHUQ 3 only,
        ``"5plus_units"`` → TYPEHUQ 4 only.
    """
    target = MF_SEGMENTS.get(segment)
    if target is None or "mf_segment" not in df.columns:
        return df
    return df[df["mf_segment"] == target].copy()


def filter_fuel(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    """Filter to a heating-fuel stratum.

    Parameters
    ----------
    fuel:
        ``"all_fuels"`` → no filter,
        ``"electric"`` → ELWARM == 1,
        ``"gas"`` → UGWARM == 1.
    """
    col = FUEL_STRATA.get(fuel)
    if col is None or col not in df.columns:
        return df
    return df[df[col] == 1].copy()


def filter_classification_view(
    df: pd.DataFrame,
    view: str,
    binary_col: str = "heating_system_type_binary",
    mode_col: str = "heating_classification_mode",
) -> pd.DataFrame:
    """Filter to a classification view and return rows with valid binary labels.

    Parameters
    ----------
    view:
        ``"pooled_binary"`` → all Central + Distributed (explicit + inferred),
        ``"explicit_only"`` → only explicitly classified rows,
        ``"inferred_only"`` → only inferred rows.
    binary_col:
        Column with ``"Central"`` / ``"Distributed"`` labels.
    mode_col:
        Column with ``"explicit"`` / ``"inferred"`` / ``"unknown"`` labels.
    """
    sub = df[df[binary_col].isin(["Central", "Distributed"])].copy()
    if view == "explicit_only" and mode_col in sub.columns:
        sub = sub[sub[mode_col] == "explicit"]
    elif view == "inferred_only" and mode_col in sub.columns:
        sub = sub[sub[mode_col] == "inferred"]
    # pooled_binary → keep all classifiable rows
    return sub


def iter_segment_fuel_view():
    """Yield (segment_label, fuel_label, view_label) for the standard analysis grid."""
    for seg in MF_SEGMENTS:
        for fuel in FUEL_STRATA:
            for view in CLASSIFICATION_VIEWS:
                yield seg, fuel, view


def friendly_label(segment: str, fuel: str, view: str) -> str:
    """Human-readable label for a segment × fuel × view combination."""
    parts = []
    if segment != "all_mf":
        parts.append(segment.replace("_", " "))
    else:
        parts.append("All MF")
    parts.append(fuel.replace("_", " ").title())
    parts.append(view.replace("_", " "))
    return " | ".join(parts)
