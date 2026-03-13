"""Prompt 7 — Side-by-Side Comparison: RBSA vs. 2023 FM MF Survey.

Loads BOTH datasets independently. Does NOT merge them.

For each dataset:
  1. Applies its own classification logic (already done for RBSA in step 01).
  2. Computes % Central / % Distributed and median Site EUI by system type.
  3. Runs Mann-Whitney U (Central vs. Distributed) per dataset.
  4. Presents a side-by-side table with a note on comparability limits.

Usage::

    python analysis/rbsa/06_compare_with_fm.py \\
        --site-master outputs/rbsa/rbsa_site_master_*.parquet \\
        --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        --outdir outputs/rbsa

Notes:
  * RBSA: Site_EUI_kBtu_sqft (kBtu/sqft/yr), Pacific NW residential sites.
  * FM Survey: Site EUI (kBTU/sqft/yr), national multifamily properties.
  * Do NOT merge — different units of analysis, geography, vintage, and
    classification schemes. Compare findings side-by-side only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.fm_mf_survey.classify import add_system_classifications, to_binary
from src.common.log import get_logger

logger = get_logger("rbsa.06_compare")

ALPHA = 0.05

STRUCTURAL_DIFFERENCES = """
=============================================================================
STRUCTURAL DIFFERENCES: RBSA 2022 vs. 2023 FM MF Survey
=============================================================================

Dimension               RBSA 2022                       2023 FM MF Survey
----------------------- ------------------------------- ----------------------
HVAC classification     System_Type → frozenset lookup  Plant type (explicit)
                        (Central/Distributed/Unknown)   + equipment type
                        from Mechanical_One_Line.csv    (Central/Distributed/
                                                        Central inferred/
                                                        Unknown)

DHW classification      Technology_Description +        Hot Water Heating
                        Provided_by_Heating_System      Location (explicit) +
                        (Mechanical_WaterHeater.csv)    system-type inference

Energy metric           Site_EUI_kBtu_sqft              Site EUI (kBTU/sqft/yr)
                        = Annual Usage Total /           Pre-calculated from
                        Conditioned_Area (computed)      benchmarked billing data

Coverage                836 of 2,279 sites have         ~980 of 2,273 props
                        Annual Usage Total (both         pass energy QC filter
                        electric + gas confirmed)        (Include in energy
                                                        analysis? == 1)

Climate                 Heating_Zone integer (1–3)       2022 Annual HDD65,
                        (field-assessed climate zone)    CDD65 per property

Sample size             Pacific NW: ID, MT, OR, WA      National (US)
(geography)             Residential (SF + MF units)     Multifamily only

Assessment              Physical field inspection        Self-reported survey
Survey year             2022 (data collected ~2022)     2022 data, survey 2023
Building type           Mostly single-family;           All MF: high-rise to
                        369 MF units; 219 MF buildings   townhome

IMPORTANT: Do NOT merge these datasets — different keys, geography, vintage,
           and assessment methodologies. Compare findings side-by-side only.
=============================================================================
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_site_master(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0 * u / (n1 * n2))


def _eui_summary_row(
    df: pd.DataFrame,
    binary_col: str,
    eui_col: str,
    dataset_name: str,
    system_key: str,
) -> dict:
    """Compute comparison metrics for one dataset × system type."""
    n_total = len(df)
    c = df.loc[df[binary_col] == "Central", eui_col].dropna().values
    d = df.loc[df[binary_col] == "Distributed", eui_col].dropna().values
    n_classifiable = (df[binary_col].isin(["Central", "Distributed"])).sum()

    row: dict = {
        "dataset": dataset_name,
        "system_key": system_key,
        "n_total": n_total,
        "n_classifiable": n_classifiable,
        "pct_central": round(100 * len(c) / n_classifiable, 1) if n_classifiable else None,
        "pct_distributed": round(100 * len(d) / n_classifiable, 1) if n_classifiable else None,
        "n_central": len(c),
        "n_distributed": len(d),
        "median_eui_central": round(float(np.median(c)), 1) if len(c) else None,
        "median_eui_distributed": round(float(np.median(d)), 1) if len(d) else None,
        "diff_central_minus_distributed": None,
        "p_value": None,
        "significant": None,
        "note": "",
    }

    if row["median_eui_central"] is not None and row["median_eui_distributed"] is not None:
        row["diff_central_minus_distributed"] = round(
            row["median_eui_central"] - row["median_eui_distributed"], 1
        )

    if len(c) >= 2 and len(d) >= 2:
        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        row["p_value"] = round(float(p), 4)
        row["significant"] = bool(p < ALPHA)
    else:
        row["note"] = f"Insufficient data (C={len(c)}, D={len(d)})"

    return row


# ---------------------------------------------------------------------------
# Per-dataset summaries
# ---------------------------------------------------------------------------


def rbsa_summaries(site_master_path: Path) -> dict[str, pd.DataFrame]:
    """Load RBSA site master and compute per-system summaries."""
    df = _load_site_master(site_master_path)
    logger.info("RBSA site master: %d rows", len(df))

    eui_col = "Site_EUI_kBtu_sqft"
    if eui_col not in df.columns or not df[eui_col].notna().any():
        logger.warning("Site_EUI_kBtu_sqft not found or all-null in RBSA site master.")
        return {}

    summaries = {}
    for sys_col, key in [
        ("heating_system_type", "heating"),
        ("cooling_system_type", "cooling"),
        ("dhw_system_type", "dhw"),
    ]:
        if sys_col not in df.columns:
            continue
        sub = df[df[eui_col].notna()].copy()
        row = _eui_summary_row(sub, sys_col, eui_col, "RBSA 2022", key)
        summaries[key] = row

    return summaries


def fm_summaries(fm_path: Path) -> dict[str, dict]:
    """Load FM MF Survey and compute per-system summaries."""
    logger.info("Loading FM MF Survey from %s", fm_path)
    df = load_energy_subset(load_mf_survey(fm_path))
    df = add_system_classifications(df)

    eui_col = "Site EUI (kBTU/sqft/yr)"
    summaries = {}

    for sys_col, key in [
        ("heating_system_type", "heating"),
        ("cooling_system_type", "cooling"),
        ("dhw_system_type", "dhw"),
    ]:
        binary = to_binary(df[sys_col])
        sub = df.copy()
        sub["_binary"] = binary
        sub = sub[sub[eui_col].notna()]
        row = _eui_summary_row(sub, "_binary", eui_col, "2023 FM MF Survey", key)
        summaries[key] = row

    return summaries


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def print_comparison(rbsa: dict, fm: dict, outdir: Path | None) -> None:
    print(STRUCTURAL_DIFFERENCES)

    all_rows = []
    for key, label in [("heating", "Heating"), ("cooling", "Cooling"), ("dhw", "DHW")]:
        print(f"\n{'='*70}")
        print(f"  {label} — Central vs. Distributed EUI (median kBtu/sqft/yr)")
        print(f"{'='*70}")

        headers = [
            "Dataset", "n_classifiable", "% Central", "% Distributed",
            "n Central", "n Distributed",
            "Median EUI Central", "Median EUI Distributed",
            "Diff (C-D)", "p-value", "Significant",
        ]
        rows_for_table = []

        for dataset_name, src in [("RBSA 2022", rbsa), ("2023 FM MF Survey", fm)]:
            r = src.get(key)
            if r is None:
                rows_for_table.append([dataset_name] + ["—"] * (len(headers) - 1))
                continue
            rows_for_table.append([
                r["dataset"],
                r["n_classifiable"],
                f"{r['pct_central']}%" if r["pct_central"] is not None else "—",
                f"{r['pct_distributed']}%" if r["pct_distributed"] is not None else "—",
                r["n_central"],
                r["n_distributed"],
                r["median_eui_central"],
                r["median_eui_distributed"],
                r["diff_central_minus_distributed"],
                r["p_value"],
                ("✓" if r["significant"] else "✗") if r["significant"] is not None else "—",
            ])
            all_rows.append(r)

        tbl = pd.DataFrame(rows_for_table, columns=headers)
        print(tbl.to_string(index=False))

        if rbsa.get(key) and fm.get(key):
            r_r = rbsa[key]
            r_f = fm[key]
            if r_r.get("diff_central_minus_distributed") is not None and r_f.get("diff_central_minus_distributed") is not None:
                direction_rbsa = "Central > Distributed" if r_r["diff_central_minus_distributed"] > 0 else "Distributed > Central"
                direction_fm = "Central > Distributed" if r_f["diff_central_minus_distributed"] > 0 else "Distributed > Central"
                print(f"\n  Direction: RBSA → {direction_rbsa}   |   FM Survey → {direction_fm}")

    if outdir and all_rows:
        out = pd.DataFrame(all_rows)
        fname = outdir / "06_comparison_rbsa_vs_fm.csv"
        out.to_csv(fname, index=False)
        logger.info("Saved comparison table to %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side RBSA vs. FM MF Survey comparison.")
    ap.add_argument("--site-master", type=Path, required=True,
                    help="Path to rbsa_site_master_*.parquet (output of 01_build_curated_mf_table.py)")
    ap.add_argument("--fm-data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/rbsa"),
                    help="Directory for output files.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    rbsa = rbsa_summaries(args.site_master)

    try:
        fm = fm_summaries(args.fm_data)
    except Exception as exc:
        logger.error("FM MF Survey load failed: %s", exc)
        fm = {}

    print_comparison(rbsa, fm, args.outdir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
