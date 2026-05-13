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
    """Filter to a heating-fuel stratum (mutually exclusive).

    Parameters
    ----------
    fuel:
        ``"all_fuels"`` → no filter,
        ``"electric"`` → ELWARM == 1 AND UGWARM != 1,
        ``"gas"`` → UGWARM == 1 AND ELWARM != 1.

    Mutual exclusivity ensures Electric Only + Gas Only subsets do not overlap
    and sample sizes sum sensibly relative to All Fuels.
    """
    if fuel == "all_fuels" or fuel not in FUEL_STRATA:
        return df
    col = FUEL_STRATA[fuel]
    if col is None or col not in df.columns:
        return df
    # Determine the "other" fuel column to exclude
    other_col = "UGWARM" if col == "ELWARM" else "ELWARM"
    mask = df[col] == 1
    if other_col in df.columns:
        mask = mask & (df[other_col] != 1)
    return df[mask].copy()


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


# ---------------------------------------------------------------------------
# YEARMADERANGE labels and energy-code-aligned vintage bins
# ---------------------------------------------------------------------------

# RECS 2020 YEARMADERANGE ordinal code → human-readable decade label
YEARMADERANGE_LABELS: dict[int, str] = {
    1: "Before 1950",
    2: "1950–1959",
    3: "1960–1969",
    4: "1970–1979",
    5: "1980–1989",
    6: "1990–1999",
    7: "2000–2009",
    8: "2010–2015",
    9: "2016–2020",
}

# Energy-code-aligned bins (aligned with ASHRAE 90.1 editions):
#   Pre-1980 (codes 1–4): Before federal/state energy codes
#   1980–1999 (codes 5–6): Early code era (90A-1980, 90.1-1989)
#   2000+ (codes 7–9): Modern code era (90.1-2001 through 90.1-2019)
VINTAGE_BIN_MAP: dict[int, str] = {
    1: "Pre-1980",
    2: "Pre-1980",
    3: "Pre-1980",
    4: "Pre-1980",
    5: "1980–1999",
    6: "1980–1999",
    7: "2000+",
    8: "2000+",
    9: "2000+",
}

# Ordered categories for plotting
VINTAGE_BIN_ORDER = ["Pre-1980", "1980–1999", "2000+"]


def add_vintage_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``vintage_bin`` column with energy-code-aligned categories.

    Groups YEARMADERANGE into three bins aligned with major energy code eras:
    Pre-1980, 1980–1999, 2000+.  Returns the DataFrame with the new column.
    """
    ymr = pd.to_numeric(df.get("YEARMADERANGE"), errors="coerce")
    df["vintage_bin"] = ymr.map(VINTAGE_BIN_MAP)
    return df

