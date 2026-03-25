"""RBSA curated table builder — Prompts 1–4.

Implements:
  Prompt 1: Master analysis table (SiteDetail + Usage_One_Line + Mechanical_One_Line + EUI)
  Prompt 2: Site-level HVAC classification (Central / Distributed / Unknown)
  Prompt 3: Site-level DHW classification (from Mechanical_WaterHeater.csv)
  Prompt 4: MF building-level HVAC/DHW classification (from MF_Building_* tables)

Outputs (written to --outdir):
  rbsa_site_master.{parquet,csv}    — all sites with system classifications and EUI
  rbsa_mf_buildings.{parquet,csv}  — MF building-level HVAC/DHW classifications and EUI
                                      (EUI joined via Building_ID → SiteID bridge in SiteDetail)

Usage::

    python analysis/rbsa/01_build_curated_mf_table.py \\
        --data-dir path/to/2022_RBSA_Datasets/ \\
        --outdir outputs/rbsa

    # or from a zip:
    python analysis/rbsa/01_build_curated_mf_table.py \\
        --zip path/to/rbsa_2022.zip \\
        --outdir outputs/rbsa
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import datetime as dt

import pandas as pd

from src.common.log import get_logger
from src.datasets.rbsa.ingest import load_rbsa_from_zip, load_rbsa_from_dir
from src.datasets.rbsa.classify import (
    add_site_classifications,
    add_mf_building_classifications,
    classify_site_heating,
    classify_site_cooling,
)

logger = get_logger("rbsa.build")


# ---------------------------------------------------------------------------
# Step 1: Master site analysis table
# ---------------------------------------------------------------------------

_MECH_OL_KEEP = [
    "SiteID",
    "Primary_Heating_System_Type",
    "Primary_Cooling_System_Type",
    "Primary_Heating_Fuel_Type",
]

_USAGE_KEEP = [
    "SiteID",
    "Annual Electric Usage (kWh)",
    "Annual Electric Usage (kBtu)",
    "Annual Gas Usage (therms)",
    "Annual Gas Usage (kBtu)",
    "Annual Usage Total (kBtu)",
    "Annual Usage Total for Heating (kBtu)",
    "Annual Electric Usage for Heating (kBtu)",
    "Annual Gas Usage for Heating (kBtu)",
    "Total Delivered Fuel (kBtu)",
    "Ownership",
    "Qty_Occupants",
]


def _build_site_master(inputs) -> pd.DataFrame:
    """Prompt 1: Join SiteDetail + Usage + Mechanical_One_Line; compute EUI."""
    site = inputs.site_detail.copy()

    # Conditioned_Area lives in SiteDetail
    usage_cols = [c for c in _USAGE_KEEP if c in inputs.usage_one_line.columns]
    usage = inputs.usage_one_line[usage_cols].copy()

    mol_cols = [c for c in _MECH_OL_KEEP if c in inputs.mech_one_line.columns]
    mol = inputs.mech_one_line[mol_cols].copy()

    # Coerce energy columns to numeric
    for col in usage.columns:
        if col != "SiteID":
            usage[col] = pd.to_numeric(usage[col], errors="coerce")

    master = site.merge(usage, on="SiteID", how="left")
    master = master.merge(mol, on="SiteID", how="left")

    logger.info("Master join: %d rows (expected ~2,279)", len(master))
    assert master["SiteID"].nunique() == len(master), "Duplicate SiteIDs after join"

    # Conditioned_Area may be in site_detail or usage; resolve
    if "Conditioned_Area" not in master.columns:
        logger.warning("Conditioned_Area not found in SiteDetail — EUI cannot be computed.")
        master["Site_EUI_kBtu_sqft"] = None
        master["Electric_EUI_kBtu_sqft"] = None
        return master

    master["Conditioned_Area"] = pd.to_numeric(master["Conditioned_Area"], errors="coerce")

    if "Annual Usage Total (kBtu)" in master.columns:
        master["Site_EUI_kBtu_sqft"] = (
            pd.to_numeric(master["Annual Usage Total (kBtu)"], errors="coerce")
            / master["Conditioned_Area"]
        )
    else:
        master["Site_EUI_kBtu_sqft"] = None

    if "Annual Electric Usage (kBtu)" in master.columns:
        master["Electric_EUI_kBtu_sqft"] = (
            pd.to_numeric(master["Annual Electric Usage (kBtu)"], errors="coerce")
            / master["Conditioned_Area"]
        )
    else:
        master["Electric_EUI_kBtu_sqft"] = None

    return master


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def _report_master(master: pd.DataFrame) -> None:
    _section("Prompt 1 — Master table summary")
    print(f"  Total rows  : {len(master)}")
    print(f"  Columns     : {len(master.columns)}")

    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in master.columns), None)
    if bt_col:
        print(f"\n  Building_Type breakdown:")
        print(master[bt_col].value_counts(dropna=False).to_string())

    key_cols = [
        "Conditioned_Area",
        "Annual Usage Total (kBtu)",
        "Annual Electric Usage (kBtu)",
        "Annual Gas Usage (kBtu)",
        "Site_EUI_kBtu_sqft",
        "Electric_EUI_kBtu_sqft",
        "Primary_Heating_System_Type",
        "Primary_Cooling_System_Type",
        "Primary_Heating_Fuel_Type",
    ]
    print("\n  Non-null counts for key columns:")
    for c in key_cols:
        if c in master.columns:
            nn = master[c].notna().sum()
            print(f"    {c:<45s}: {nn:>5d} / {len(master)}")


def _report_hvac_classification(master: pd.DataFrame) -> None:
    _section("Prompt 2 — Site-level HVAC classification")
    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in master.columns), None)

    for cls_col in ["heating_system_type", "cooling_system_type"]:
        if cls_col not in master.columns:
            continue
        print(f"\n  {cls_col}:")
        if bt_col:
            print(master.groupby(bt_col)[cls_col].value_counts(dropna=False).to_string())
        else:
            print(master[cls_col].value_counts(dropna=False).to_string())


def _report_dhw_classification(master: pd.DataFrame) -> None:
    _section("Prompt 3 — Site-level DHW classification")
    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in master.columns), None)

    if "dhw_system_type" not in master.columns:
        print("  dhw_system_type column not found.")
        return

    print("\n  dhw_system_type:")
    if bt_col:
        print(master.groupby(bt_col)["dhw_system_type"].value_counts(dropna=False).to_string())
    else:
        print(master["dhw_system_type"].value_counts(dropna=False).to_string())


def _report_mf_buildings(mf_bldg: pd.DataFrame) -> None:
    _section("Prompt 4 — MF building-level HVAC/DHW classification")
    print(f"  Total MF buildings: {len(mf_bldg)}")
    if "n_sites" in mf_bldg.columns:
        print(f"  Surveyed units linked: {mf_bldg['n_sites'].sum():.0f}")
    if "hvac_source" in mf_bldg.columns:
        print(f"\n  Classification source:")
        print(mf_bldg["hvac_source"].value_counts(dropna=False).to_string())

    for col in ["hvac_system_type", "dhw_system_type"]:
        if col in mf_bldg.columns:
            print(f"\n  {col}:")
            print(mf_bldg[col].value_counts(dropna=False).to_string())

    if "hvac_system_type" in mf_bldg.columns and "dhw_system_type" in mf_bldg.columns:
        print("\n  Cross-tab: hvac_system_type × dhw_system_type")
        xtab = pd.crosstab(
            mf_bldg["hvac_system_type"],
            mf_bldg["dhw_system_type"],
            margins=True,
            margins_name="Total",
        )
        print(xtab.to_string())

    eui_cols = ["Site_EUI_kBtu_sqft", "Electric_EUI_kBtu_sqft"]
    present_eui = [c for c in eui_cols if c in mf_bldg.columns]
    if present_eui:
        print("\n  Median EUI by hvac_system_type (building-level medians):")
        if "hvac_system_type" in mf_bldg.columns:
            print(mf_bldg.groupby("hvac_system_type")[present_eui].median().round(1).to_string())
        else:
            print(mf_bldg[present_eui].describe().round(1).to_string())


# ---------------------------------------------------------------------------
# Step 4 extension: Enrich MF buildings with energy data
# ---------------------------------------------------------------------------

_MF_ENERGY_COLS = [
    "Site_EUI_kBtu_sqft",
    "Electric_EUI_kBtu_sqft",
    "Annual Usage Total (kBtu)",
    "Annual Electric Usage (kBtu)",
    "Annual Gas Usage (kBtu)",
]


def _enrich_mf_with_energy(
    mf_bldg: pd.DataFrame,
    site_detail: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Join MF building records to energy data via Building_ID → SiteID bridge.

    SiteDetail contains both Building_ID and SiteID. MF buildings (keyed on
    Building_ID) can be linked to Usage_One_Line energy data (keyed on SiteID)
    through this bridge. One Building_ID may map to multiple surveyed SiteIDs;
    energy metrics are aggregated to building level using the median.
    """
    if "Building_ID" not in site_detail.columns or "SiteID" not in site_detail.columns:
        logger.warning("SiteDetail missing Building_ID or SiteID — skipping energy enrichment.")
        return mf_bldg

    # Bridge: Building_ID → SiteID (MF rows only — exclude null/string "NA")
    bridge = site_detail[["Building_ID", "SiteID"]].copy()
    bridge = bridge[bridge["Building_ID"].notna() & (bridge["Building_ID"] != "NA")]

    # Pull available energy columns from master
    energy_cols = ["SiteID"] + [c for c in _MF_ENERGY_COLS if c in master.columns]
    energy = master[energy_cols].copy()
    for col in energy_cols[1:]:
        energy[col] = pd.to_numeric(energy[col], errors="coerce")

    bridge_energy = bridge.merge(energy, on="SiteID", how="left")

    # Aggregate to building level: median for energy, count for n_sites
    numeric_cols = [c for c in _MF_ENERGY_COLS if c in bridge_energy.columns]
    agg_dict: dict = {c: "median" for c in numeric_cols}
    agg_dict["SiteID"] = "count"
    bldg_energy = bridge_energy.groupby("Building_ID").agg(agg_dict).reset_index()
    bldg_energy = bldg_energy.rename(columns={"SiteID": "n_sites"})

    result = mf_bldg.merge(bldg_energy, on="Building_ID", how="left")
    n_with_eui = result["Site_EUI_kBtu_sqft"].notna().sum() if "Site_EUI_kBtu_sqft" in result.columns else 0
    logger.info("MF energy enrichment: %d buildings, EUI available for %d", len(result), n_with_eui)
    return result


# ---------------------------------------------------------------------------
# Site-level fallback for MF buildings where MF questionnaire gave "unknown"
# ---------------------------------------------------------------------------

def _apply_site_fallback(
    mf_bldg: pd.DataFrame,
    master: pd.DataFrame,
    site_detail: pd.DataFrame,
) -> pd.DataFrame:
    """For MF buildings where hvac/dhw classification is 'unknown', fall back to
    site-level Mechanical_One_Line classification via Building_ID -> SiteID bridge.

    Adds an 'hvac_source' column: 'mf_questionnaire' or 'site_mechanical_fallback'.
    When a building has multiple SiteIDs, takes the modal (most common) site
    classification across its units.
    """
    df = mf_bldg.copy()
    df["hvac_source"] = "mf_questionnaire"

    if "Building_ID" not in site_detail.columns or "SiteID" not in site_detail.columns:
        logger.warning("SiteDetail missing Building_ID or SiteID — skipping fallback.")
        return df

    # Build Building_ID -> site classifications (mode per building)
    site_cls_cols = ["SiteID", "heating_system_type", "cooling_system_type", "dhw_system_type"]
    site_cls_cols = [c for c in site_cls_cols if c in master.columns]
    if len(site_cls_cols) < 2:
        return df

    bridge = site_detail[["Building_ID", "SiteID"]].copy()
    bridge = bridge[bridge["Building_ID"].notna() & (bridge["Building_ID"] != "NA")]
    site_cls = master[site_cls_cols].copy()
    bridge_cls = bridge.merge(site_cls, on="SiteID", how="left")

    # Take modal classification per building
    agg = {}
    for col in ["heating_system_type", "cooling_system_type", "dhw_system_type"]:
        if col in bridge_cls.columns:
            agg[col] = lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"
    bldg_site_cls = bridge_cls.groupby("Building_ID").agg(agg).reset_index()
    bldg_site_cls = bldg_site_cls.rename(columns={
        "heating_system_type": "_site_heating",
        "cooling_system_type": "_site_cooling",
        "dhw_system_type": "_site_dhw",
    })

    df = df.merge(bldg_site_cls, on="Building_ID", how="left")

    # Apply fallback for each unknown column
    for mf_col, site_col in [
        ("hvac_system_type", "_site_heating"),
        ("dhw_system_type", "_site_dhw"),
    ]:
        if mf_col not in df.columns or site_col not in df.columns:
            continue
        unknown_mask = df[mf_col] == "unknown"
        if unknown_mask.any():
            df.loc[unknown_mask, mf_col] = df.loc[unknown_mask, site_col].fillna("unknown")
            df.loc[unknown_mask, "hvac_source"] = "site_mechanical_fallback"

    drop_cols = [c for c in ["_site_heating", "_site_cooling", "_site_dhw"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    n_fallback = (df["hvac_source"] == "site_mechanical_fallback").sum()
    logger.info("Site-level fallback applied to %d / %d MF buildings", n_fallback, len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build RBSA curated tables (Prompts 1–4).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path, default=None, help="Path to RBSA zip file.")
    src.add_argument("--data-dir", type=Path, default=None, help="Directory containing extracted RBSA CSVs.")
    ap.add_argument("--outdir", type=Path, required=True, help="Output folder (e.g., outputs/rbsa).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all RBSA files
    # ------------------------------------------------------------------
    logger.info("Loading RBSA data from %s", args.zip or args.data_dir)
    inputs = load_rbsa_from_zip(args.zip) if args.zip else load_rbsa_from_dir(args.data_dir)
    logger.info(
        "Loaded — SiteDetail: %d | Usage: %d | MechOL: %d | MechHC: %d | MechWH: %d"
        " | MF_Site: %d | MF_HVAC: %d | MF_DHW: %d",
        len(inputs.site_detail), len(inputs.usage_one_line), len(inputs.mech_one_line),
        len(inputs.mech_hc), len(inputs.mech_wh),
        len(inputs.mf_site_detail), len(inputs.mf_hvac), len(inputs.mf_dhw),
    )

    # ------------------------------------------------------------------
    # Prompts 1–3: site-level master table
    # ------------------------------------------------------------------
    master = _build_site_master(inputs)
    master = add_site_classifications(master, inputs.mech_wh)

    _report_master(master)
    _report_hvac_classification(master)
    _report_dhw_classification(master)

    # ------------------------------------------------------------------
    # Prompt 4: MF building-level classification
    # ------------------------------------------------------------------
    mf_bldg = add_mf_building_classifications(inputs.mf_hvac, inputs.mf_dhw)

    # Enrich with MF_Building_Site_Detail (floors, units, area, vintage)
    mf_bldg = inputs.mf_site_detail.merge(mf_bldg, on="Building_ID", how="left")

    # Enrich with energy data via Building_ID → SiteID bridge in SiteDetail
    mf_bldg = _enrich_mf_with_energy(mf_bldg, inputs.site_detail, master)

    # Fall back to site-level Mechanical_One_Line classification for unknowns
    mf_bldg = _apply_site_fallback(mf_bldg, master, inputs.site_detail)

    _report_mf_buildings(mf_bldg)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    site_base = args.outdir / f"rbsa_site_master_{ts}"
    master.to_parquet(site_base.with_suffix(".parquet"), index=False)
    master.to_csv(site_base.with_suffix(".csv"), index=False)
    logger.info("Wrote site master → %s.(parquet,csv)", site_base)

    mf_base = args.outdir / f"rbsa_mf_buildings_{ts}"
    mf_bldg.to_parquet(mf_base.with_suffix(".parquet"), index=False)
    mf_bldg.to_csv(mf_base.with_suffix(".csv"), index=False)
    logger.info("Wrote MF buildings → %s.(parquet,csv)", mf_base)

    print(f"\n  Site master   : {site_base}.parquet / .csv")
    print(f"  MF buildings  : {mf_base}.parquet / .csv")


if __name__ == "__main__":
    main()
