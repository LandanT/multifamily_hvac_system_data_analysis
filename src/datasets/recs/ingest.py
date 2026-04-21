"""RECS 2020 microdata ingestion.

Loads the public-use microdata file (recs2020_public_v7.csv) and computes
derived energy-intensity metrics.  All energy values in the RECS file are
already expressed in thousand Btu (kBtu), so no unit conversion is required —
unlike the FM/RBSA pipelines that convert kWh / therms to kBtu.

Usage::

    from src.datasets.recs.ingest import load_recs

    df = load_recs("path/to/recs2020_public_v7.csv")            # MF only (default)
    df_all = load_recs("path/to/recs2020_public_v7.csv", unit_type="all")
    df_sf  = load_recs("path/to/recs2020_public_v7.csv", unit_type="sf")

TYPEHUQ values
--------------
    1  Mobile home
    2  Single-family detached
    3  Apartment in building with 2–4 units
    4  Apartment in building with 5+ units

unit_type="mf" retains TYPEHUQ 3 and 4 (matches FM/RBSA multifamily focus).

Missing-value conventions
--------------------------
RECS uses -2 ("not applicable") and -1 ("missing") as numeric sentinels.
Both are converted to NaN in this function so that downstream code can use
pd.isna() uniformly.  The distinction between "not applicable" and "missing"
is noted here for reference but treated identically in classification:
both trigger the inference path rather than an explicit classification.

Survey weights
--------------
NWEIGHT (sample weight) and replicate weights BRRWT1–BRRWT96 are retained in
the returned DataFrame.  The initial pipeline produces unweighted estimates;
future analyses should apply NWEIGHT for nationally representative results.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Columns where -2 / -1 sentinel codes should be replaced with NaN
# ---------------------------------------------------------------------------

# Classification indicator columns
_SENTINEL_COLS = [
    "HEATHOME",
    "HEATAPT",
    "EQUIPM",
    "AIRCOND",
    "COOLAPT",
    "ACEQUIPM_PUB",
    "H2OAPT",
    "MORETHAN1H2O",
    "TYPEHUQ",
    # Fuel flags — heating
    "UGWARM",
    "LPWARM",
    "FOWARM",
    "ELWARM",
    # Fuel flags — water heating
    "UGWATER",
    "LPWATER",
    "FOWATER",
    "ELWATER",
]

# Energy and size columns that may contain sentinels
_ENERGY_COLS = [
    "TOTALBTU",
    "TOTALBTUSPH",
    "TOTALBTUWTH",
    "TOTALBTUOTH",
    "BTUEL",
    "BTUNG",
    "BTULP",
    "BTUFO",
    "TOTSQFT_EN",
    # Heated-area column referenced in spec (may not be present in all vintages)
    "TOTHSQFT",
    # Climate zone identifiers
    "YEARMADERANGE",
]


def load_recs(
    csv_path: str | Path,
    unit_type: str = "mf",
) -> pd.DataFrame:
    """Load the RECS 2020 public-use microdata file.

    Parameters
    ----------
    csv_path:
        Path to ``recs2020_public_v7.csv`` (or similar public-use CSV).
    unit_type:
        Housing unit filter.  One of:

        ``"mf"`` (default)
            Retain only multifamily units (TYPEHUQ 3 or 4).  Consistent with
            the FM/RBSA analyses; HEATAPT and COOLAPT are only meaningful for
            multi-unit buildings.
        ``"sf"``
            Retain only single-family / mobile homes (TYPEHUQ 1 or 2).
        ``"all"``
            No filter — retain all housing unit types.

    Returns
    -------
    pd.DataFrame
        One row per housing unit with sentinel codes replaced by NaN and
        derived EUI columns appended.  Ready for
        :func:`src.datasets.recs.classify.add_system_classifications`.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Replace -2 (not applicable) and -1 (missing) with NaN across known cols.
    for col in _SENTINEL_COLS + _ENERGY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(~df[col].isin([-2, -1]), other=np.nan)

    # Apply housing-unit-type filter
    if unit_type == "mf":
        if "TYPEHUQ" in df.columns:
            df = df[df["TYPEHUQ"].isin([3, 4])].copy()
    elif unit_type == "sf":
        if "TYPEHUQ" in df.columns:
            df = df[df["TYPEHUQ"].isin([1, 2])].copy()
    elif unit_type == "all":
        pass
    else:
        raise ValueError(f"unit_type must be 'mf', 'sf', or 'all'; got {unit_type!r}")

    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Derived energy-intensity metrics (kBtu/sqft/yr)
    # No unit conversion needed: RECS already provides thousand-Btu totals.
    # Rows with zero or missing floor area are set to NaN rather than dropped
    # here; script 01 filters them out after classification.
    # ------------------------------------------------------------------
    sqft = pd.to_numeric(df.get("TOTSQFT_EN"), errors="coerce")
    valid_sqft = sqft.where(sqft > 0)  # zero area → NaN denominator

    def _eui(numerator_col: str) -> pd.Series:
        num = pd.to_numeric(df.get(numerator_col, pd.Series(np.nan, index=df.index)), errors="coerce")
        return num / valid_sqft

    df["Site_EUI_kBtu_sqft"] = _eui("TOTALBTU")

    # Heating EUI: the spec references TOTHSQFT (heated area) but that column
    # may not exist in all public-use file versions.  We fall back to TOTSQFT_EN
    # and document the decision here.
    if "TOTHSQFT" in df.columns:
        heated_sqft = pd.to_numeric(df["TOTHSQFT"], errors="coerce")
        heated_sqft = heated_sqft.where(heated_sqft > 0)
        heat_num = pd.to_numeric(df.get("TOTALBTUSPH", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["Heating_EUI_kBtu_sqft"] = heat_num / heated_sqft
    else:
        # TOTHSQFT absent — use total energy-consuming floor area
        df["Heating_EUI_kBtu_sqft"] = _eui("TOTALBTUSPH")

    df["DHW_EUI_kBtu_sqft"] = _eui("TOTALBTUWTH")
    df["Other_EUI_kBtu_sqft"] = _eui("TOTALBTUOTH")
    df["Electric_EUI_kBtu_sqft"] = _eui("BTUEL")
    df["Gas_EUI_kBtu_sqft"] = _eui("BTUNG")

    # Fall back for TOTALBTU: sum fuel components if primary total is missing.
    if "TOTALBTU" in df.columns:
        missing_total = df["TOTALBTU"].isna()
        if missing_total.any():
            fuel_cols = ["BTUEL", "BTUNG", "BTULP", "BTUFO"]
            present_fuels = [c for c in fuel_cols if c in df.columns]
            if present_fuels:
                fallback = df.loc[missing_total, present_fuels].apply(
                    pd.to_numeric, errors="coerce"
                ).sum(axis=1)
                df.loc[missing_total, "TOTALBTU"] = fallback
                # Recompute Site_EUI_kBtu_sqft with filled TOTALBTU
                df["Site_EUI_kBtu_sqft"] = (
                    pd.to_numeric(df["TOTALBTU"], errors="coerce") / valid_sqft
                )

    return df
