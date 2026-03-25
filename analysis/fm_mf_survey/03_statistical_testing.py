"""Prompt 3 — Statistical Testing (Mann-Whitney U).

For each pair (Central vs. Distributed) × (DHW / Heating / Cooling):
  - Mann-Whitney U test on Site EUI, Source EUI, Site Energy/Unit
  - Reports: U statistic, p-value, rank-biserial correlation (effect size),
    sample sizes
  - Uses alpha = 0.05
  - Does NOT use t-tests (small n, normality not guaranteed)

"Inferred" labels are pooled with their base category
(e.g., "Central (inferred)" → "Central").

Usage::

    python analysis/fm_mf_survey/03_statistical_testing.py \\
        --data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        [--outdir outputs/fm_mf_survey]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats

from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.fm_mf_survey.classify import add_system_classifications, to_binary
from src.common.log import get_logger

logger = get_logger("fm_mf_survey.03_stats")

ALPHA = 0.05

OUTCOME_COLS = [
    "Site EUI (kBTU/sqft/yr)",
    "Source EUI (kBTU/sqft/yr)",
    "Site Energy/Unit (kBtu/Unit/yr)",
]

SYSTEM_COLS = ["dhw_system_type", "heating_system_type", "cooling_system_type"]


# ---------------------------------------------------------------------------
# Effect size: rank-biserial correlation
# ---------------------------------------------------------------------------


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation from two arrays.

    r_rb = 1 - 2U / (n1 * n2)
    where U is the Mann-Whitney statistic for group x.
    Range: [-1, 1].  Positive → x tends to have higher values.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0 * u_stat / (n1 * n2))


# ---------------------------------------------------------------------------
# Single comparison
# ---------------------------------------------------------------------------


def compare_groups(
    central: pd.Series,
    distributed: pd.Series,
    metric: str,
    system_col: str,
) -> dict:
    """Run Mann-Whitney U and return a result dict."""
    c = central.dropna().values
    d = distributed.dropna().values

    result = {
        "system_col": system_col,
        "metric": metric,
        "n_central": len(c),
        "n_distributed": len(d),
        "median_central": round(np.median(c), 2) if len(c) else None,
        "median_distributed": round(np.median(d), 2) if len(d) else None,
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
    ap = argparse.ArgumentParser(description="Mann-Whitney U tests for 2023 MF Survey.")
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="If provided, save results CSV here.")
    args = ap.parse_args()

    logger.info("Loading data …")
    df = load_energy_subset(load_mf_survey(args.data))
    df = add_system_classifications(df)

    results = []

    for sys_col in SYSTEM_COLS:
        # Collapse inferred labels to binary Central / Distributed
        binary = to_binary(df[sys_col])
        df_bin = df.copy()
        df_bin["_binary"] = binary

        central = df_bin[df_bin["_binary"] == "Central"]
        distributed = df_bin[df_bin["_binary"] == "Distributed"]

        print(f"\n{'='*70}")
        print(f"  {sys_col}  —  Central (n={len(central)}) vs Distributed (n={len(distributed)})")
        print(f"{'='*70}")

        for metric in OUTCOME_COLS:
            if metric not in df.columns:
                continue

            res = compare_groups(
                central[metric], distributed[metric], metric, sys_col
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

    # ------------------------------------------------------------------
    # Print compact table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    result_df = pd.DataFrame(results)
    print(result_df[[
        "system_col", "metric", "n_central", "n_distributed",
        "U_statistic", "p_value", "rank_biserial_r", "significant",
    ]].to_string(index=False))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        out = args.outdir / "03_mann_whitney_results.csv"
        result_df.to_csv(out, index=False)
        logger.info("Saved results to %s", out)


if __name__ == "__main__":
    main()
