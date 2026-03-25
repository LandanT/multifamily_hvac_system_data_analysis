"""Prompt 6 — Integration with the RBSA Dataset (side-by-side comparison).

Loads BOTH the 2023 Multifamily Survey dataset AND the RBSA dataset
independently.  Does NOT merge them.

For each dataset:
  1. Applies its own central/distributed classification logic.
  2. Computes median Site EUI and Source EUI by system type
     (HVAC and DHW separately).
  3. Produces a side-by-side comparison table.
  4. Notes structural differences (classification scheme, survey year,
     sample sizes, energy metric definitions, climate variables).

Usage::

    python analysis/fm_mf_survey/06_compare_with_rbsa.py \\
        --mf-data  path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        --rbsa-zip path/to/rbsa_data.zip       # OR --rbsa-dir path/to/rbsa/
        [--outdir outputs/fm_mf_survey]

Notes:
  * The RBSA dataset uses column-prefix-based classification (COMMON* vs TENANT*).
  * The 2023 MF Survey uses plant-type + equipment-type based classification.
  * Source EUI is not directly available in RBSA's Usage_One_Line.csv — the
    script looks for a 'Source EUI' column and skips gracefully if absent.
  * Do NOT merge the datasets — each row means something different.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Optional

import pandas as pd

from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.fm_mf_survey.classify import add_system_classifications, to_binary
from src.datasets.rbsa.ingest import load_rbsa_from_zip, load_rbsa_from_dir
from src.datasets.rbsa.classify import add_site_classifications
from src.common.log import get_logger

logger = get_logger("fm_mf_survey.06_compare")

_CENTRAL = {"Central", "Central (inferred)", "central", "mixed"}
_DISTRIBUTED = {"Distributed", "Distributed (inferred)", "distributed"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_binary_rbsa(label: str) -> Optional[str]:
    if label in _CENTRAL:
        return "Central"
    if label in _DISTRIBUTED:
        return "Distributed"
    return None


def _eui_summary(
    df: pd.DataFrame,
    binary_col: str,
    eui_cols: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Return a summary DataFrame: rows = Central/Distributed, cols = EUI metrics."""
    rows = []
    for grp in ["Central", "Distributed"]:
        gdf = df[df[binary_col] == grp]
        row = {"Dataset": dataset_name, "System": grp, "n": len(gdf)}
        for col in eui_cols:
            if col in gdf.columns:
                vals = pd.to_numeric(gdf[col], errors="coerce").dropna()
                row[f"median_{col}"] = round(vals.median(), 1) if len(vals) else None
                row[f"n_{col}"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2023 MF Survey side
# ---------------------------------------------------------------------------


def mf_survey_summaries(mf_path: Path) -> dict[str, pd.DataFrame]:
    """Load, classify, and summarise MF Survey data.

    Returns dict keyed by system type ('dhw', 'heating', 'cooling').
    """
    logger.info("Loading 2023 MF Survey from %s", mf_path)
    df = load_energy_subset(load_mf_survey(mf_path))
    df = add_system_classifications(df)

    eui_cols = ["Site EUI (kBTU/sqft/yr)", "Source EUI (kBTU/sqft/yr)"]
    summaries = {}

    for sys_col, key in [
        ("dhw_system_type", "dhw"),
        ("heating_system_type", "heating"),
        ("cooling_system_type", "cooling"),
    ]:
        binary = to_binary(df[sys_col])
        df2 = df.copy()
        df2["_binary"] = binary
        sub = df2[df2["_binary"].notna()]
        summaries[key] = _eui_summary(sub, "_binary", eui_cols, "2023 MF Survey")

    return summaries


# ---------------------------------------------------------------------------
# RBSA side
# ---------------------------------------------------------------------------


def rbsa_summaries(rbsa_zip: Optional[Path], rbsa_dir: Optional[Path]) -> dict[str, pd.DataFrame]:
    """Load, classify, and summarise RBSA data using site-level classifications."""
    if rbsa_zip:
        inputs = load_rbsa_from_zip(rbsa_zip)
    else:
        inputs = load_rbsa_from_dir(rbsa_dir)

    # Build master site table: SiteDetail + Usage + Mechanical_One_Line columns
    mol_keep = [c for c in [
        "SiteID", "Primary_Heating_System_Type", "Primary_Cooling_System_Type", "Primary_Heating_Fuel_Type",
    ] if c in inputs.mech_one_line.columns]
    master = inputs.site_detail.merge(inputs.usage_one_line, on="SiteID", how="left")
    master = master.merge(inputs.mech_one_line[mol_keep], on="SiteID", how="left")

    # Compute Site EUI
    if "Conditioned_Area" in master.columns and "Annual Usage Total (kBtu)" in master.columns:
        master["Conditioned_Area"] = pd.to_numeric(master["Conditioned_Area"], errors="coerce")
        master["Annual Usage Total (kBtu)"] = pd.to_numeric(master["Annual Usage Total (kBtu)"], errors="coerce")
        master["Site_EUI_kBtu_sqft"] = master["Annual Usage Total (kBtu)"] / master["Conditioned_Area"]
    else:
        master["Site_EUI_kBtu_sqft"] = None

    # Apply site-level HVAC + DHW classifications
    master = add_site_classifications(master, inputs.mech_wh)

    eui_col = "Site_EUI_kBtu_sqft"
    summaries = {}

    for sys_col, key in [
        ("heating_system_type", "heating"),
        ("cooling_system_type", "cooling"),
        ("dhw_system_type", "dhw"),
    ]:
        if sys_col not in master.columns:
            continue
        # Map RBSA labels to binary for _eui_summary
        master["_rbsa_binary"] = master[sys_col].apply(
            lambda v: _to_binary_rbsa(v) if isinstance(v, str) else None
        )
        sub = master[master[eui_col].notna() & master["_rbsa_binary"].notna()]
        if sub.empty:
            logger.warning("No data for RBSA %s — skipping", key)
            continue
        summaries[key] = _eui_summary(sub, "_rbsa_binary", [eui_col], "RBSA")

    return summaries


# ---------------------------------------------------------------------------
# Comparison table + structural notes
# ---------------------------------------------------------------------------


STRUCTURAL_DIFFERENCES = """
=============================================================================
STRUCTURAL DIFFERENCES: 2023 MF Survey vs. RBSA
=============================================================================

Dimension               2023 MF Survey                  RBSA
----------------------- ------------------------------- ----------------------
HVAC classification     Plant type (central chiller /   Column-prefix-based:
                        heating boiler) + in-unit        COMMON* vs TENANT*
                        equipment type                   equipment columns

DHW classification      'Hot Water Heating Location'     Column-prefix-based:
                        (central plant / in-unit) +      COMMON* vs TENANT*
                        'Residential Hot Water System'   water-heating columns

Central vs. Distributed Must be derived — not a         Must be derived —
                        direct survey field             not a direct field

Energy metric           Site EUI (kBTU/sqft/yr)         Varies by version;
                        Source EUI (kBTU/sqft/yr)       may be kWh or kBTU
                        Per-unit energy                 annual totals

Climate control         HDD65, CDD65 per property       Building location →
                        (2022 actuals)                  HDD/CDD lookup needed

Sample size (HVAC       ~69–107 with system detail      Varies; typically
classifiable)           out of 980 energy-QC props      100–300 MF buildings

Survey year             2022 data, collected 2023       2011–2012 vintage
                                                        (Pacific NW only)

Geography               National (US)                   Pacific Northwest
                                                        (OR, WA, ID, MT)

Multifamily scope       All MF property types           Multifamily buildings
                        (high-rise to townhouse)        only

IMPORTANT: Do NOT merge these datasets. Classification schemes differ,
geography differs, and vintage differs. Compare findings side-by-side only.
=============================================================================
"""


def print_comparison(mf: dict, rbsa: dict) -> None:
    print(STRUCTURAL_DIFFERENCES)

    for key, label in [("dhw", "DHW"), ("heating", "Heating"), ("cooling", "Cooling")]:
        print(f"\n{'='*60}")
        print(f"  {label} — Median EUI by Central vs. Distributed")
        print(f"{'='*60}")

        mf_tbl = mf.get(key)
        rbsa_tbl = rbsa.get(key)

        if mf_tbl is not None:
            print("\n  [2023 MF Survey]")
            print(mf_tbl.to_string(index=False))
        else:
            print("  [2023 MF Survey] — Not available")

        if rbsa_tbl is not None:
            print("\n  [RBSA]")
            print(rbsa_tbl.to_string(index=False))
        else:
            print("  [RBSA] — Not available or no matching EUI columns")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side MF Survey vs RBSA comparison.")
    ap.add_argument("--mf-data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--rbsa-zip", type=Path, default=None,
                    help="Path to RBSA zip file.")
    ap.add_argument("--rbsa-dir", type=Path, default=None,
                    help="Path to directory containing extracted RBSA CSVs.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/fm_mf_survey"),
                    help="Directory for output files.")
    args = ap.parse_args()

    if (args.rbsa_zip is None) == (args.rbsa_dir is None):
        raise SystemExit("Provide exactly one of --rbsa-zip or --rbsa-dir")

    args.outdir.mkdir(parents=True, exist_ok=True)

    mf = mf_survey_summaries(args.mf_data)

    try:
        rbsa = rbsa_summaries(args.rbsa_zip, args.rbsa_dir)
    except Exception as exc:
        logger.error("RBSA load failed: %s", exc)
        rbsa = {}

    print_comparison(mf, rbsa)

    # Save CSVs
    for key, label in [("dhw", "DHW"), ("heating", "Heating"), ("cooling", "Cooling")]:
        frames = []
        if key in mf and mf[key] is not None:
            frames.append(mf[key])
        if key in rbsa and rbsa[key] is not None:
            frames.append(rbsa[key])
        if frames:
            out = pd.concat(frames, ignore_index=True)
            fname = args.outdir / f"06_comparison_{key}.csv"
            out.to_csv(fname, index=False)
            logger.info("Saved %s", fname)

    logger.info("Done.")


if __name__ == "__main__":
    main()
