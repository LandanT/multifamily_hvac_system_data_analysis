"""RECS 2020 — Step 3: Statistical testing (Mann-Whitney U).

For each pair (Central vs. Distributed) × (Heating / Cooling / DHW):
  - Mann-Whitney U test on Site_EUI_kBtu_sqft, Heating_EUI_kBtu_sqft,
    DHW_EUI_kBtu_sqft
  - Reports: U statistic, p-value, rank-biserial correlation (effect size),
    sample sizes, significance at α=0.05
  - Stratified by primary heating fuel (ELWARM vs. UGWARM) to avoid
    fuel-type confounding, following the RBSA approach

"Inferred" labels are collapsed to binary before testing (e.g.,
"Central (inferred)" → "Central") via the binary columns added in step 01.

Usage::

    python analysis/recs/03_statistical_testing.py \\
        --curated outputs/recs/recs2020_curated_*.parquet \\
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
from src.datasets.recs.utils import load_curated, filter_unit_type

logger = get_logger("recs.03_stats")

ALPHA = 0.05

OUTCOME_COLS = [
    "Site_EUI_kBtu_sqft",
    "Heating_EUI_kBtu_sqft",
    "DHW_EUI_kBtu_sqft",
]

# Binary classification columns (set by add_system_classifications in step 01)
BINARY_COLS = [
    "heating_system_type_binary",
    "cooling_system_type_binary",
    "dhw_system_type_binary",
]

# Fuel strata: None = all households; "electric" / "gas" = subset by fuel flag
FUEL_STRATA = [None, "electric", "gas"]


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
    ap = argparse.ArgumentParser(description="Mann-Whitney U tests for RECS 2020.")
    ap.add_argument(
        "--curated", type=Path, required=True,
        help="Path to recs2020_curated_*.parquet (output of 01_build_curated_table.py).",
    )
    ap.add_argument(
        "--outdir", type=Path, default=None,
        help="If provided, save results CSV here.",
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

    logger.info("Loading curated data from %s", args.curated)
    df = load_curated(args.curated)
    df = filter_unit_type(df, args.unit_type)
    logger.info("Rows after unit-type filter (%s): %d", args.unit_type, len(df))

    results = []

    for binary_col in BINARY_COLS:
        if binary_col not in df.columns:
            logger.warning("%s not found — skipping", binary_col)
            continue

        sys_label = binary_col.replace("_binary", "")
        classifiable = df[df[binary_col].isin(["Central", "Distributed"])].copy()

        for stratum in FUEL_STRATA:
            if stratum == "electric":
                if "ELWARM" not in df.columns:
                    continue
                sub = classifiable[classifiable["ELWARM"] == 1].copy()
                stratum_label = "Electric (ELWARM=1)"
            elif stratum == "gas":
                if "UGWARM" not in df.columns:
                    continue
                sub = classifiable[classifiable["UGWARM"] == 1].copy()
                stratum_label = "Natural Gas (UGWARM=1)"
            else:
                sub = classifiable
                stratum_label = "All"

            central = sub[sub[binary_col] == "Central"]
            distributed = sub[sub[binary_col] == "Distributed"]

            print(f"\n{'=' * 70}")
            print(
                f"  {sys_label}  [{stratum_label}]  —  "
                f"Central (n={len(central)}) vs Distributed (n={len(distributed)})"
            )
            print(f"{'=' * 70}")

            for metric in OUTCOME_COLS:
                if metric not in df.columns:
                    continue

                res = compare_groups(
                    central[metric], distributed[metric],
                    metric, sys_label, stratum_label,
                )
                results.append(res)

                sig_str = "✓ SIGNIFICANT" if res["significant"] else "✗ not significant"
                if res["significant"] is None:
                    sig_str = f"— [{res['note']}]"
                note_str = f"  [{res['note']}]" if res["note"] and res["significant"] is not None else ""
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
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        print(result_df[[
            "system_col", "fuel_stratum", "metric", "n_central", "n_distributed",
            "U_statistic", "p_value", "rank_biserial_r", "significant",
        ]].to_string(index=False))

        # Multiple-testing note
        n_tests = len(result_df[result_df["p_value"].notna()])
        if n_tests > 1:
            bonferroni_alpha = round(ALPHA / n_tests, 4)
            n_bonf_sig = int((result_df["p_value"].dropna() < bonferroni_alpha).sum())
            print(
                f"\n  ⚠ Multiple testing: {n_tests} tests performed at α={ALPHA}."
                f"\n    Bonferroni-adjusted α = {bonferroni_alpha}"
                f" → {n_bonf_sig} of {n_tests} remain significant after correction."
            )
            result_df["bonferroni_significant"] = result_df["p_value"] < bonferroni_alpha

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        out = args.outdir / "03_mann_whitney_results.csv"
        result_df.to_csv(out, index=False)
        logger.info("Saved results to %s", out)


if __name__ == "__main__":
    main()
