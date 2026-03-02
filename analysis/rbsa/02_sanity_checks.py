"""Sanity checks for curated RBSA MF table."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from src.common.log import get_logger

logger = get_logger("rbsa.sanity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", type=Path, required=True, help="Path to curated parquet/csv.")
    ap.add_argument("--weight-col", type=str, default="Site_Case_Weight", help="Weight column name (if present).")
    args = ap.parse_args()

    df = pd.read_parquet(args.curated) if args.curated.suffix.lower() == ".parquet" else pd.read_csv(args.curated, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.curated)

    for c in ["hvac_system_type", "dhw_system_type"]:
        if c in df.columns:
            logger.info("%s counts:\\n%s", c, df[c].value_counts(dropna=False))
        else:
            logger.warning("Missing %s", c)

    # missingness for annual energy-like columns
    annual = [c for c in df.columns if c.lower().startswith("annual") and ("kwh" in c.lower() or "therm" in c.lower())]
    if annual:
        miss = df[annual].isna().mean().sort_values(ascending=False)
        logger.info("Annual energy missingness (top):\\n%s", miss.head(10))
    else:
        logger.warning("No obvious annual energy columns found.")

if __name__ == "__main__":
    main()
