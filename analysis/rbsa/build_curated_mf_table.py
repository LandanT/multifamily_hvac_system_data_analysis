"""Example RBSA workflow entrypoint (stub).

Usage:
  python analysis/rbsa/build_curated_mf_table.py --data-dir data/rbsa_2022 --outdir outputs/rbsa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.log import get_logger
from src.common.schema import require_columns

logger = get_logger("rbsa")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Path to RBSA dataset folder.")
    ap.add_argument("--outdir", type=Path, required=True, help="Output folder.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    usage_path = args.data_dir / "Usage_One_Line.csv"
    if not usage_path.exists():
        logger.warning("Expected %s does not exist. Stage RBSA raw files into data-dir first.", usage_path)
        return

    usage = pd.read_csv(usage_path)
    require_columns(usage, ["SiteID"], "Usage_One_Line.csv")

    logger.info("Loaded %d rows from %s", len(usage), usage_path)

    out_path = args.outdir / "rbsa_usage_preview.csv"
    usage.head(1000).to_csv(out_path, index=False)
    logger.info("Wrote preview: %s", out_path)

if __name__ == "__main__":
    main()
