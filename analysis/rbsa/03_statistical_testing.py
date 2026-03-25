"""Prompt 5 (part 2) — Statistical Testing (Mann-Whitney U).

For each pair (Central vs. Distributed) × (Heating / Cooling / DHW):
  - Mann-Whitney U test on Site_EUI_kBtu_sqft, Annual Electric Usage (kBtu),
    Annual Gas Usage (kBtu)
  - Reports: U statistic, p-value, rank-biserial correlation (effect size),
    sample sizes, significance at α=0.05
  - Stratified by Primary_Heating_Fuel_Type (Electric vs. Natural gas) to
    avoid fuel-type confounding per Prompt 5 guidance

RBSA classifications are already binary (Central / Distributed / Unknown) —
no "inferred" label collapsing is needed; only filter != "Unknown".

Usage::

    python analysis/rbsa/03_statistical_testing.py \\
        --site-master outputs/rbsa/rbsa_site_master_*.parquet \\
        --outdir outputs/rbsa
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

logger = get_logger("rbsa.03_stats")

ALPHA = 0.05

OUTCOME_COLS = [
    "Site_EUI_kBtu_sqft",
    "Annual Electric Usage (kBtu)",
    "Annual Gas Usage (kBtu)",
]

SYSTEM_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]

FUEL_STRATA = [None, "Electric", "Natural gas"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_site_master(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation: r_rb = 1 - 2U / (n1 * n2). Range: [-1, 1]."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0 * u_stat / (n1 * n2))


def compare_groups(
    central: pd.Series,
    distributed: pd.Series,
    metric: str,
    system_col: str,
    stratum: str | None = None,
) -> dict:
    """Run Mann-Whitney U and return a result dict."""
    c = central.dropna().values
    d = distributed.dropna().values

    result = {
        "system_col": system_col,
        "metric": metric,
        "fuel_stratum": stratum or "All",
        "n_central": len(c),
        "n_distributed": len(d),
        "median_central": round(float(np.median(c)), 2) if len(c) else None,
        "median_distributed": round(float(np.median(d)), 2) if len(d) else None,
        "U_statistic": None,
        "p_value": None,
        "rank_biserial_r": None,
        "significant": None,
        "note": "",
    }

    if len(c) < 2 or len(d) < 2:
        result["note"] = f"Insufficient data (n_central={len(c)}, n_distributed={len(d)})"
        return result

    u, p = stats.mannwhitneyu(c, d, alternative="two-sided")
    r = rank_biserial(c, d)
    result["U_statistic"] = round(float(u), 1)
    result["p_value"] = round(float(p), 4)
    result["rank_biserial_r"] = round(r, 3)
    result["significant"] = bool(p < ALPHA)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Mann-Whitney U tests for RBSA site master.")
    ap.add_argument("--site-master", type=Path, required=True,
                    help="Path to rbsa_site_master_*.parquet (or .csv)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="If provided, save results CSV here.")
    ap.add_argument(
        "--building-type",
        choices=["mf", "sf", "all"],
        default="mf",
        help="Building type filter: 'mf' = multifamily only (default), 'sf' = single-family only, 'all' = no filter.",
    )
    args = ap.parse_args()

    logger.info("Loading site master from %s", args.site_master)
    df = _load_site_master(args.site_master)

    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in df.columns), None)
    if args.building_type == "sf" and bt_col:
        df = df[df[bt_col].str.lower().str.contains("single", na=False)].copy()
        logger.info("SF-only filter: %d rows", len(df))
    elif args.building_type == "mf" and bt_col:
        df = df[df[bt_col].str.lower().str.contains("multi", na=False)].copy()
        logger.info("MF-only filter: %d rows", len(df))

    results = []

    for sys_col in SYSTEM_COLS:
        if sys_col not in df.columns:
            logger.warning("%s not found — skipping", sys_col)
            continue

        # Filter to classifiable rows
        classifiable = df[df[sys_col] != "Unknown"].copy()

        for stratum in FUEL_STRATA:
            if stratum is not None:
                fuel_col = "Primary_Heating_Fuel_Type"
                if fuel_col not in df.columns:
                    continue
                sub = classifiable[classifiable[fuel_col] == stratum].copy()
            else:
                sub = classifiable

            central = sub[sub[sys_col] == "Central"]
            distributed = sub[sub[sys_col] == "Distributed"]

            stratum_label = stratum or "All"
            print(f"\n{'='*70}")
            print(
                f"  {sys_col}  [{stratum_label}]  —  "
                f"Central (n={len(central)}) vs Distributed (n={len(distributed)})"
            )
            print(f"{'='*70}")

            for metric in OUTCOME_COLS:
                if metric not in df.columns:
                    continue

                res = compare_groups(
                    central[metric], distributed[metric],
                    metric, sys_col, stratum,
                )
                results.append(res)

                sig_str = "✓ SIGNIFICANT" if res["significant"] else "✗ not significant"
                note_str = f"  [{res['note']}]" if res["note"] else ""
                print(
                    f"\n  {metric}\n"
                    f"    n_central={res['n_central']}, n_distributed={res['n_distributed']}\n"
                    f"    median_central={res['median_central']},  "
                    f"median_distributed={res['median_distributed']}\n"
                    f"    U={res['U_statistic']},  p={res['p_value']},  "
                    f"r_rb={res['rank_biserial_r']}\n"
                    f"    → {sig_str} (α={ALPHA}){note_str}"
                )

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        print(result_df[[
            "system_col", "fuel_stratum", "metric", "n_central", "n_distributed",
            "U_statistic", "p_value", "rank_biserial_r", "significant",
        ]].to_string(index=False))

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        out = args.outdir / "03_mann_whitney_results.csv"
        result_df.to_csv(out, index=False)
        logger.info("Saved results to %s", out)


if __name__ == "__main__":
    main()
