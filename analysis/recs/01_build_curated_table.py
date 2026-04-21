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
    # Coverage report
    # ------------------------------------------------------------------
    _coverage_summary(df)

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
