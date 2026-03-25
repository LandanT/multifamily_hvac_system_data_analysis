"""Sanity checks for the curated RBSA site master and MF buildings tables.

Usage::

    # Check site master
    python analysis/rbsa/02_sanity_checks.py --curated outputs/rbsa/rbsa_site_master_*.parquet

    # Check MF buildings table
    python analysis/rbsa/02_sanity_checks.py --curated outputs/rbsa/rbsa_mf_buildings_*.parquet \\
        --table-type mf_buildings
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common.log import get_logger

logger = get_logger("rbsa.sanity")

_SITE_MASTER_CLASSIFICATION_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]
_MF_BUILDING_CLASSIFICATION_COLS = ["hvac_system_type", "dhw_system_type"]

_SITE_ENERGY_COLS = [
    "Annual Electric Usage (kWh)",
    "Annual Electric Usage (kBtu)",
    "Annual Gas Usage (therms)",
    "Annual Gas Usage (kBtu)",
    "Annual Usage Total (kBtu)",
    "Site_EUI_kBtu_sqft",
    "Electric_EUI_kBtu_sqft",
]


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _check_site_master(df: pd.DataFrame) -> None:
    _section("Classification label distributions")
    for col in _SITE_MASTER_CLASSIFICATION_COLS:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            total = len(df)
            classifiable = (df[col] != "Unknown").sum()
            pct = 100.0 * classifiable / total if total else 0.0
            print(f"\n  {col}:")
            print(counts.to_string())
            print(f"    Classifiable: {classifiable}/{total} ({pct:.1f}%)")
        else:
            logger.warning("Missing classification column: %s", col)

    _section("Energy column missingness")
    present_energy = [c for c in _SITE_ENERGY_COLS if c in df.columns]
    if present_energy:
        miss = df[present_energy].isna().mean().sort_values(ascending=False)
        print(miss.to_string())
    else:
        logger.warning("No energy columns found.")

    _section("Building type split")
    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in df.columns), None)
    if bt_col:
        print(df[bt_col].value_counts(dropna=False).to_string())

    _section("EUI range check (Site_EUI_kBtu_sqft)")
    if "Site_EUI_kBtu_sqft" in df.columns:
        eui = df["Site_EUI_kBtu_sqft"].dropna()
        print(f"  n={len(eui)}  min={eui.min():.1f}  median={eui.median():.1f}  max={eui.max():.1f}")
        outliers = (eui > 500) | (eui < 0)
        print(f"  Suspected outliers (EUI < 0 or > 500): {outliers.sum()}")

    _section("Duplicate SiteID check")
    dups = df.duplicated("SiteID").sum() if "SiteID" in df.columns else "N/A (SiteID not found)"
    print(f"  Duplicate SiteIDs: {dups}")


def _check_mf_buildings(df: pd.DataFrame) -> None:
    _section("MF building classification distributions")
    for col in _MF_BUILDING_CLASSIFICATION_COLS:
        if col in df.columns:
            print(f"\n  {col}:")
            print(df[col].value_counts(dropna=False).to_string())
        else:
            logger.warning("Missing classification column: %s", col)

    if all(c in df.columns for c in _MF_BUILDING_CLASSIFICATION_COLS):
        _section("Cross-tab: hvac_system_type × dhw_system_type")
        xtab = pd.crosstab(
            df["hvac_system_type"],
            df["dhw_system_type"],
            margins=True,
            margins_name="Total",
        )
        print(xtab.to_string())

    _section("Duplicate Building_ID check")
    dups = df.duplicated("Building_ID").sum() if "Building_ID" in df.columns else "N/A"
    print(f"  Duplicate Building_IDs: {dups}")

    _section("reason column samples (first 10 rows)")
    for col in ["hvac_reason", "dhw_reason"]:
        if col in df.columns:
            print(f"\n  {col} (top values):")
            print(df[col].value_counts(dropna=False).head(10).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity checks for curated RBSA outputs.")
    ap.add_argument("--curated", type=Path, required=True, help="Path to curated parquet or csv file.")
    ap.add_argument(
        "--table-type",
        choices=["site_master", "mf_buildings"],
        default="site_master",
        help="Which table type to check (default: site_master).",
    )
    args = ap.parse_args()

    df = (
        pd.read_parquet(args.curated)
        if args.curated.suffix.lower() == ".parquet"
        else pd.read_csv(args.curated, low_memory=False)
    )
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), args.curated)

    if args.table_type == "mf_buildings":
        _check_mf_buildings(df)
    else:
        _check_site_master(df)


if __name__ == "__main__":
    main()
