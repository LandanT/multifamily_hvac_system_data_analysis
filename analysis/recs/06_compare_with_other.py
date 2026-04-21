"""RECS 2020 — Step 6: Side-by-side comparison with FM and/or RBSA.

Loads RECS alongside the FM MF Survey and/or RBSA datasets independently.
Does NOT merge them — each dataset has different classification schemes,
geographic scope, sample populations, and survey vintage.

For each dataset provided:
  1. Computes % Central / % Distributed and median Site EUI by system type.
  2. Runs Mann-Whitney U (Central vs. Distributed).
  3. Presents a side-by-side table with a structural differences note.

At least one comparison dataset (--fm-data or --rbsa-zip / --rbsa-dir)
must be provided.

Usage::

    # RECS vs FM only:
    python analysis/recs/06_compare_with_other.py \\
        --recs-curated outputs/recs/recs2020_curated_*.parquet \\
        --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        --outdir outputs/recs

    # RECS vs RBSA only:
    python analysis/recs/06_compare_with_other.py \\
        --recs-curated outputs/recs/recs2020_curated_*.parquet \\
        --rbsa-dir path/to/2022_RBSA_Datasets/ \\
        --outdir outputs/recs

    # RECS vs both:
    python analysis/recs/06_compare_with_other.py \\
        --recs-curated outputs/recs/recs2020_curated_*.parquet \\
        --fm-data path/to/survey.xlsx \\
        --rbsa-dir path/to/rbsa/ \\
        --outdir outputs/recs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats

from src.common.log import get_logger

logger = get_logger("recs.06_compare")

ALPHA = 0.05

STRUCTURAL_DIFFERENCES = """
=============================================================================
STRUCTURAL DIFFERENCES: RECS 2020 vs. RBSA 2022 vs. 2023 FM MF Survey
=============================================================================

Dimension           RECS 2020               RBSA 2022               FM MF Survey 2023
------------------- ----------------------- ----------------------- -----------------------
Classification      HEATAPT / COOLAPT       System_Type frozenset   Plant type (explicit) +
                    explicit indicators +   lookup from             equipment inference
                    EQUIPM / ACEQUIPM_PUB   Mechanical_One_Line     (Central/Distributed/
                    inference (integer       (Central/Distributed/   Central inferred/Unknown)
                    codes)                  Unknown)

DHW classification  H2OAPT explicit +       Technology_Description  Hot Water Heating
                    MORETHAN1H2O mixed      + Provided_by_Heating_   Location (explicit) +
                    indicator               System flag              system-type inference

Energy metric       Site_EUI_kBtu_sqft      Site_EUI_kBtu_sqft      Site EUI (kBTU/sqft/yr)
                    = TOTALBTU /            = Annual Usage Total /   Pre-calculated from
                    TOTSQFT_EN              Conditioned_Area         benchmarked billing data
                    (no conversion needed)  (computed)

Sample scope        National US             Pacific NW (ID, MT,     National US
                    All housing types       OR, WA)                 Multifamily only
                    (MF filter: TYPEHUQ 3-4) Residential (SF + MF)

Survey year         2020 (EIA RECS)         2022 (field inspection) 2022–2023 (self-reported)

Climate controls    IECC_climate_code       Heating_Zone (1–3,      HDD65, CDD65 (per prop)
                    (alphanumeric)          field-assessed)

IMPORTANT: Do NOT merge these datasets — different keys, geography, vintage,
           classification schemes, and units of analysis.
           Compare findings side-by-side only.
=============================================================================
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parquet_or_csv(path: Path) -> pd.DataFrame:
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
    c = df.loc[df[binary_col] == "Central", eui_col].dropna().values
    d = df.loc[df[binary_col] == "Distributed", eui_col].dropna().values
    n_classifiable = int(df[binary_col].isin(["Central", "Distributed"]).sum())

    row: dict = {
        "dataset": dataset_name,
        "system_key": system_key,
        "n_total": len(df),
        "n_classifiable": n_classifiable,
        "pct_central": round(100.0 * len(c) / n_classifiable, 1) if n_classifiable else None,
        "pct_distributed": round(100.0 * len(d) / n_classifiable, 1) if n_classifiable else None,
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


def recs_summaries(recs_path: Path) -> dict[str, dict]:
    """Load RECS curated parquet and compute per-system summaries."""
    df = _load_parquet_or_csv(recs_path)
    logger.info("RECS curated: %d rows", len(df))

    eui_col = "Site_EUI_kBtu_sqft"
    summaries = {}
    for binary_col, key in [
        ("heating_system_type_binary", "heating"),
        ("cooling_system_type_binary", "cooling"),
        ("dhw_system_type_binary", "dhw"),
    ]:
        if binary_col not in df.columns or eui_col not in df.columns:
            continue
        sub = df[df[eui_col].notna()].copy()
        summaries[key] = _eui_summary_row(sub, binary_col, eui_col, "RECS 2020", key)

    return summaries


def fm_summaries(fm_path: Path) -> dict[str, dict]:
    """Load FM MF Survey and compute per-system summaries."""
    from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset
    from src.datasets.fm_mf_survey.classify import add_system_classifications, to_binary

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
        if sys_col not in df.columns or eui_col not in df.columns:
            continue
        sub = df.copy()
        sub["_binary"] = to_binary(sub[sys_col])
        sub = sub[sub[eui_col].notna()]
        summaries[key] = _eui_summary_row(sub, "_binary", eui_col, "FM MF Survey 2023", key)

    return summaries


def rbsa_summaries(rbsa_path: Path, is_zip: bool) -> dict[str, dict]:
    """Load RBSA and compute per-system summaries from site master."""
    from src.datasets.rbsa.ingest import load_rbsa_from_zip, load_rbsa_from_dir
    from src.datasets.rbsa.classify import add_site_classifications

    logger.info("Loading RBSA data from %s", rbsa_path)
    inputs = load_rbsa_from_zip(rbsa_path) if is_zip else load_rbsa_from_dir(rbsa_path)

    # Build a minimal site master for EUI
    usage = inputs.usage_one_line[
        [c for c in ["SiteID", "Annual Usage Total (kBtu)", "Annual Electric Usage (kBtu)"]
         if c in inputs.usage_one_line.columns]
    ].copy()
    site = inputs.site_detail.copy()
    if "Conditioned_Area" not in site.columns:
        logger.warning("Conditioned_Area missing from RBSA — cannot compute EUI")
        return {}

    master = site.merge(usage, on="SiteID", how="left")
    for col in ["Annual Usage Total (kBtu)", "Annual Electric Usage (kBtu)", "Conditioned_Area"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    sqft = master["Conditioned_Area"].where(master["Conditioned_Area"] > 0)
    if "Annual Usage Total (kBtu)" in master.columns:
        master["Site_EUI_kBtu_sqft"] = master["Annual Usage Total (kBtu)"] / sqft

    master = add_site_classifications(master, inputs.mech_wh)

    # Filter to MF buildings
    bt_col = next((c for c in ["Building_Type"] if c in master.columns), None)
    if bt_col:
        master = master[master[bt_col].str.lower().str.contains("multi", na=False)].copy()
        logger.info("RBSA MF filter: %d rows", len(master))

    eui_col = "Site_EUI_kBtu_sqft"
    summaries = {}
    for sys_col, key in [
        ("heating_system_type", "heating"),
        ("cooling_system_type", "cooling"),
        ("dhw_system_type", "dhw"),
    ]:
        if sys_col not in master.columns or eui_col not in master.columns:
            continue
        sub = master[master[eui_col].notna()].copy()
        summaries[key] = _eui_summary_row(sub, sys_col, eui_col, "RBSA 2022", key)

    return summaries


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def print_comparison(
    datasets: dict[str, dict[str, dict]],
    outdir: Path | None,
) -> None:
    print(STRUCTURAL_DIFFERENCES)

    all_rows = []
    for key, label in [("heating", "Heating"), ("cooling", "Cooling"), ("dhw", "DHW")]:
        print(f"\n{'=' * 70}")
        print(f"  {label} — Central vs. Distributed EUI (median kBtu/sqft/yr)")
        print(f"{'=' * 70}")

        headers = [
            "Dataset", "n_classifiable", "% Central", "% Distributed",
            "n Central", "n Distributed",
            "Median EUI Central", "Median EUI Distributed",
            "Diff (C-D)", "p-value", "Significant",
        ]
        rows_for_table = []

        for ds_name, ds_summaries in datasets.items():
            r = ds_summaries.get(key)
            if r is None:
                rows_for_table.append([ds_name] + ["—"] * (len(headers) - 1))
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

    if outdir and all_rows:
        out = pd.DataFrame(all_rows)
        fname = outdir / "06_comparison_recs_vs_other.csv"
        out.to_csv(fname, index=False)
        logger.info("Saved comparison table to %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side RECS 2020 vs. FM MF Survey / RBSA comparison."
    )
    ap.add_argument(
        "--recs-curated", type=Path, required=True,
        help="Path to recs2020_curated_*.parquet (output of 01_build_curated_table.py).",
    )
    ap.add_argument(
        "--fm-data", type=Path, default=None,
        help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx (optional).",
    )
    rbsa_src = ap.add_mutually_exclusive_group()
    rbsa_src.add_argument("--rbsa-zip", type=Path, default=None, help="Path to RBSA zip file.")
    rbsa_src.add_argument("--rbsa-dir", type=Path, default=None, help="Directory with extracted RBSA CSVs.")
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/recs"),
        help="Directory for output files (default: outputs/recs).",
    )
    args = ap.parse_args()

    if args.fm_data is None and args.rbsa_zip is None and args.rbsa_dir is None:
        ap.error("At least one comparison dataset is required: --fm-data or --rbsa-zip / --rbsa-dir.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict[str, dict]] = {}

    # Load RECS
    datasets["RECS 2020"] = recs_summaries(args.recs_curated)

    # Load FM (optional)
    if args.fm_data:
        try:
            datasets["FM MF Survey 2023"] = fm_summaries(args.fm_data)
        except Exception as exc:
            logger.error("FM MF Survey load failed: %s", exc)

    # Load RBSA (optional)
    rbsa_path = args.rbsa_zip or args.rbsa_dir
    if rbsa_path:
        try:
            datasets["RBSA 2022"] = rbsa_summaries(rbsa_path, is_zip=bool(args.rbsa_zip))
        except Exception as exc:
            logger.error("RBSA load failed: %s", exc)

    print_comparison(datasets, args.outdir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
