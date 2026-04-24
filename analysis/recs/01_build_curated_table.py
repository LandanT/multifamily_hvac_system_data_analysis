"""RECS 2020 — Step 1: Build curated analysis table.

Loads the RECS 2020 microdata, classifies HVAC/DHW systems for each
household, filters to rows with usable energy and floor-area data, and
writes a timestamped parquet file for downstream analysis scripts.

Outputs (written to --outdir):
  recs2020_curated_YYYYMMDD.parquet  — classified households with EUI columns

Usage::

    python analysis/recs/01_build_curated_table.py \\
        --data path/to/recs2020_public_v7.csv \\
        --outdir outputs/recs

    # single-family or all units:
    python analysis/recs/01_build_curated_table.py \\
        --data path/to/recs2020_public_v7.csv \\
        --outdir outputs/recs --unit-type sf
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.log import get_logger
from src.datasets.recs.ingest import load_recs
from src.datasets.recs.classify import add_system_classifications, classifiable_mask

logger = get_logger("recs.01_build")

SYSTEM_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def _coverage_summary(df: pd.DataFrame) -> None:
    _section("Classification coverage summary")
    print(f"  Total rows after filtering : {len(df)}")

    for col in SYSTEM_COLS:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        n_classifiable = classifiable_mask(df, col).sum()
        pct = 100 * n_classifiable / len(df) if len(df) else 0
        print(f"\n  {col}:")
        print(counts.to_string())
        print(f"    → classifiable (non-Unknown): {n_classifiable} / {len(df)} ({pct:.1f}%)")

    # Binary breakdown
    print()
    for col in SYSTEM_COLS:
        binary_col = f"{col}_binary"
        if binary_col not in df.columns:
            continue
        central = (df[binary_col] == "Central").sum()
        distributed = (df[binary_col] == "Distributed").sum()
        total_binary = central + distributed
        print(
            f"  {binary_col}: Central={central}, Distributed={distributed}"
            f", classifiable={total_binary}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build RECS 2020 curated analysis table."
    )
    ap.add_argument(
        "--data", type=Path, required=True,
        help="Path to recs2020_public_v7.csv (or similar public-use file).",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/recs"),
        help="Directory for output files (default: outputs/recs).",
    )
    ap.add_argument(
        "--unit-type",
        choices=["mf", "sf", "all"],
        default="mf",
        help=(
            "Housing unit filter: 'mf' = multifamily only / TYPEHUQ 3–4 (default), "
            "'sf' = single-family / TYPEHUQ 1–2, 'all' = no filter."
        ),
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and classify
    # ------------------------------------------------------------------
    logger.info("Loading RECS data from %s (unit_type=%s)", args.data, args.unit_type)
    df = load_recs(args.data, unit_type=args.unit_type)
    logger.info("Loaded %d rows", len(df))

    logger.info("Applying system classifications …")
    df = add_system_classifications(df)

    # ------------------------------------------------------------------
    # Filter: require non-missing TOTALBTU and non-zero floor area
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df[df["TOTALBTU"].notna() & df["TOTSQFT_EN"].notna() & (df["TOTSQFT_EN"] > 0)].copy()
    n_after = len(df)
    logger.info(
        "Filtered to rows with valid TOTALBTU and TOTSQFT_EN: %d → %d (dropped %d)",
        n_before, n_after, n_before - n_after,
    )

    # ------------------------------------------------------------------
    # EUI outlier flagging
    # ------------------------------------------------------------------
    EUI_LOW, EUI_HIGH = 1.0, 500.0
    eui = df["Site_EUI_kBtu_sqft"]
    outlier_mask = eui.notna() & ((eui < EUI_LOW) | (eui > EUI_HIGH))
    df["eui_outlier"] = outlier_mask
    n_outliers = int(outlier_mask.sum())
    if n_outliers:
        logger.warning(
            "Flagged %d rows with Site_EUI outside [%.0f, %.0f] kBtu/sqft/yr "
            "(column 'eui_outlier'=True). Review before analysis.",
            n_outliers, EUI_LOW, EUI_HIGH,
        )
        print(f"\n  ⚠ EUI outliers: {n_outliers} rows outside [{EUI_LOW}, {EUI_HIGH}] kBtu/sqft/yr")
    else:
        logger.info("No EUI outliers detected (range [%.0f, %.0f]).", EUI_LOW, EUI_HIGH)

    # ------------------------------------------------------------------
    # Derived columns for downstream segment / sensitivity analysis
    # ------------------------------------------------------------------
    logger.info("Adding derived analysis columns …")

    # mf_segment: multifamily building-size segment
    if "TYPEHUQ" in df.columns:
        df["mf_segment"] = df["TYPEHUQ"].map({3: "2_to_4_units", 4: "5plus_units"})
    else:
        df["mf_segment"] = pd.NA

    # unit_size_bin: floor area bins
    sqft = pd.to_numeric(df.get("TOTSQFT_EN"), errors="coerce")
    df["unit_size_bin"] = pd.cut(
        sqft,
        bins=[0, 750, 1000, 1500, 2000, float("inf")],
        labels=["<750", "750-999", "1000-1499", "1500-1999", "2000+"],
        right=False,
    )

    # heating_classification_mode: explicit / inferred / unknown
    _MODE_MAP = {
        "Central": "explicit",
        "Distributed": "explicit",
        "Central (inferred)": "inferred",
        "Distributed (inferred)": "inferred",
        "Unknown": "unknown",
    }
    if "heating_system_type" in df.columns:
        df["heating_classification_mode"] = df["heating_system_type"].map(_MODE_MAP).fillna("unknown")
    else:
        df["heating_classification_mode"] = "unknown"

    # amenity_flag: pool or hot tub present
    pool = pd.to_numeric(df.get("SWIMPOOL"), errors="coerce") == 1
    tub = pd.to_numeric(df.get("RECBATH"), errors="coerce") == 1
    df["amenity_flag"] = pool | tub

    # ev_flag: household owns an electric vehicle
    ev = pd.to_numeric(df.get("ELECVEH"), errors="coerce") == 1
    df["ev_flag"] = ev

    # heating_fuel_group: simplified primary heating fuel
    elwarm = pd.to_numeric(df.get("ELWARM"), errors="coerce") == 1
    ugwarm = pd.to_numeric(df.get("UGWARM"), errors="coerce") == 1
    df["heating_fuel_group"] = "other"
    df.loc[elwarm & ~ugwarm, "heating_fuel_group"] = "electric"
    df.loc[ugwarm & ~elwarm, "heating_fuel_group"] = "gas"
    df.loc[elwarm & ugwarm, "heating_fuel_group"] = "both"

    logger.info(
        "Derived columns added: mf_segment, unit_size_bin, "
        "heating_classification_mode, amenity_flag, ev_flag, heating_fuel_group"
    )

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    _coverage_summary(df)

    # ------------------------------------------------------------------
    # Metadata summary
    # ------------------------------------------------------------------
    _section("Derived column summary")
    if "mf_segment" in df.columns:
        print("\n  mf_segment:")
        print(df["mf_segment"].value_counts(dropna=False).to_string())
    if "unit_size_bin" in df.columns:
        print("\n  unit_size_bin:")
        print(df["unit_size_bin"].value_counts(dropna=False).sort_index().to_string())
    if "heating_classification_mode" in df.columns:
        print("\n  heating_classification_mode:")
        print(df["heating_classification_mode"].value_counts(dropna=False).to_string())
    if "heating_system_type_binary" in df.columns and "mf_segment" in df.columns:
        print("\n  heating_system_type_binary × mf_segment:")
        xtab = pd.crosstab(
            df["mf_segment"], df["heating_system_type_binary"],
            margins=True, margins_name="Total", dropna=False,
        )
        print(xtab.to_string())
    print(f"\n  amenity_flag=True : {int(df['amenity_flag'].sum())}")
    print(f"  ev_flag=True      : {int(df['ev_flag'].sum())}")
    print(f"\n  heating_fuel_group:")
    print(df["heating_fuel_group"].value_counts(dropna=False).to_string())

    # Key variable missingness
    _section("Key variable missingness")
    _miss_cols = [
        "TOTALBTU", "TOTALBTUSPH", "TOTSQFT_EN", "TYPEHUQ",
        "HEATAPT", "EQUIPM", "ELWARM", "UGWARM",
        "SWIMPOOL", "RECBATH", "ELECVEH",
        "Heating_EUI_kBtu_sqft", "Site_EUI_kBtu_sqft",
    ]
    for col in _miss_cols:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            pct = 100.0 * n_miss / len(df) if len(df) else 0
            print(f"  {col:30s}: {n_miss:5d} missing ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    date_str = dt.date.today().strftime("%Y%m%d")
    out_path = args.outdir / f"recs2020_curated_{date_str}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote curated table → %s", out_path)
    print(f"\n  Output: {out_path}")


if __name__ == "__main__":
    main()
