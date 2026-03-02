"""RBSA MF curated table builder (thin structure)."""

from __future__ import annotations

import argparse
from pathlib import Path
import datetime as dt
import pandas as pd

from src.common.log import get_logger
from src.datasets.rbsa.ingest import load_rbsa_from_zip, load_rbsa_from_dir
from src.datasets.rbsa.classify import classify_hvac, classify_dhw

logger = get_logger("rbsa.curate")


def detect_multifamily(site_detail: pd.DataFrame) -> pd.Series:
    if "Building Type" in site_detail.columns:
        s = site_detail["Building Type"].astype(str).str.lower()
        return s.str.contains("multi") | s.str.contains("mf")
    # fallback: keep all
    logger.warning("No obvious multifamily column found; keeping all sites. Use --include-all-sites to silence.")
    return pd.Series([True] * len(site_detail), index=site_detail.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None, help="Path to RBSA zip file.")
    ap.add_argument("--data-dir", type=Path, default=None, help="Directory containing extracted RBSA CSVs.")
    ap.add_argument("--outdir", type=Path, required=True, help="Output folder (e.g., outputs/rbsa).")
    ap.add_argument("--include-all-sites", action="store_true", help="Do not filter to multifamily; keep all.")
    args = ap.parse_args()

    if (args.zip is None) == (args.data_dir is None):
        raise SystemExit("Provide exactly one of --zip or --data-dir")

    args.outdir.mkdir(parents=True, exist_ok=True)

    inputs = load_rbsa_from_zip(args.zip) if args.zip else load_rbsa_from_dir(args.data_dir)

    site = inputs.site_detail.copy()
    usage = inputs.usage_one_line.copy()
    hvac = inputs.mf_hvac.copy()
    dhw = inputs.mf_dhw.copy()

    mf_mask = pd.Series([True] * len(site), index=site.index) if args.include_all_sites else detect_multifamily(site)
    site_mf = site.loc[mf_mask].copy()

    logger.info("SiteDetail: %d rows total; keeping %d rows", len(site), len(site_mf))

    # Classify systems (Building_ID level)
    hvac_cls = hvac[["Building_ID"]].copy()
    hvac_cls["hvac_system_type"] = hvac.apply(lambda r: classify_hvac(r).label, axis=1)
    hvac_cls["hvac_reason"] = hvac.apply(lambda r: classify_hvac(r).reason, axis=1)
    hvac_cls = hvac_cls.drop_duplicates("Building_ID")

    dhw_cls = dhw[["Building_ID"]].copy()
    dhw_cls["dhw_system_type"] = dhw.apply(lambda r: classify_dhw(r).label, axis=1)
    dhw_cls["dhw_reason"] = dhw.apply(lambda r: classify_dhw(r).reason, axis=1)
    dhw_cls = dhw_cls.drop_duplicates("Building_ID")

    curated = site_mf[["SiteID", "Building_ID"]].merge(hvac_cls, on="Building_ID", how="left")
    curated = curated.merge(dhw_cls, on="Building_ID", how="left")
    curated = curated.merge(usage, on="SiteID", how="left")

    logger.info("Curated rows: %d", len(curated))
    if "hvac_system_type" in curated.columns:
        logger.info("HVAC types:\\n%s", curated["hvac_system_type"].value_counts(dropna=False))
    if "dhw_system_type" in curated.columns:
        logger.info("DHW types:\\n%s", curated["dhw_system_type"].value_counts(dropna=False))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = args.outdir / f"rbsa_mf_curated_{ts}"
    curated.to_parquet(out_base.with_suffix(".parquet"), index=False)
    curated.to_csv(out_base.with_suffix(".csv"), index=False)
    logger.info("Wrote %s.(parquet,csv)", out_base)

if __name__ == "__main__":
    main()
