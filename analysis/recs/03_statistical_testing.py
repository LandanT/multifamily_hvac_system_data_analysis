"""RECS 2020 — Step 3: Statistical testing (Mann-Whitney U).

Refocused around heating:

  - **Primary outcome**: Heating_EUI_kBtu_sqft
  - **Secondary outcome**: Site_EUI_kBtu_sqft
  - Tests are run across a grid of:
      * MF segment  (all MF / 2–4 units / 5+ units)
      * Fuel stratum (all fuels / electric-heated / gas-heated)
      * Classification view (pooled binary / explicit only / inferred only)
  - Reports: U statistic, p-value, rank-biserial correlation (effect size),
    sample sizes, significance at α = 0.05

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
from src.datasets.recs.utils import (
    load_curated,
    filter_unit_type,
    filter_segment,
    filter_fuel,
    filter_classification_view,
    MF_SEGMENTS,
    FUEL_STRATA,
    CLASSIFICATION_VIEWS,
    friendly_label,
)

logger = get_logger("recs.03_stats")

ALPHA = 0.05

# Primary outcome first, secondary second
OUTCOME_COLS = [
    "Heating_EUI_kBtu_sqft",
    "Site_EUI_kBtu_sqft",
    "DHW_EUI_kBtu_sqft",
]

# We focus on heating system type for the refocused analysis, but keep
# cooling and DHW as secondary passes (pooled binary, all MF, all fuels only)
BINARY_COL_PRIMARY = "heating_system_type_binary"
BINARY_COLS_SECONDARY = [
    "cooling_system_type_binary",
    "dhw_system_type_binary",
]


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
) -> dict:
    """Run Mann-Whitney U and return a result dict."""
    c = central.dropna().values
    d = distributed.dropna().values

    result = {
        "system_col": system_col,
        "metric": metric,
        "n_central": len(c),
        "n_distributed": len(d),
        "median_central": round(float(np.median(c)), 2) if len(c) else None,
        "median_distributed": round(float(np.median(d)), 2) if len(d) else None,
        "delta_median": None,
        "U_statistic": None,
        "p_value": None,
        "rank_biserial_r": None,
        "significant": None,
        "note": "",
    }

    if result["median_central"] is not None and result["median_distributed"] is not None:
        result["delta_median"] = round(result["median_central"] - result["median_distributed"], 2)

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

    # ==================================================================
    # PRIMARY: heating system type across segment × fuel × view grid
    # ==================================================================
    binary_col = BINARY_COL_PRIMARY
    sys_label = binary_col.replace("_binary", "")

    if binary_col not in df.columns:
        logger.error("%s not found — cannot proceed", binary_col)
        return

    for seg_key in MF_SEGMENTS:
        seg_df = filter_segment(df, seg_key)
        for fuel_key in FUEL_STRATA:
            fuel_df = filter_fuel(seg_df, fuel_key)
            for view in CLASSIFICATION_VIEWS:
                view_df = filter_classification_view(fuel_df, view, binary_col)

                central = view_df[view_df[binary_col] == "Central"]
                distributed = view_df[view_df[binary_col] == "Distributed"]

                label = friendly_label(seg_key, fuel_key, view)
                print(f"\n{'=' * 70}")
                print(
                    f"  {sys_label}  [{label}]\n"
                    f"  Central (n={len(central)}) vs Distributed (n={len(distributed)})"
                )
                print(f"{'=' * 70}")

                for metric in OUTCOME_COLS:
                    if metric not in df.columns:
                        continue

                    res = compare_groups(
                        central[metric], distributed[metric],
                        metric, sys_label,
                    )
                    res["mf_segment"] = seg_key
                    res["fuel_stratum"] = fuel_key
                    res["classification_view"] = view
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

    # ==================================================================
    # SECONDARY: cooling / DHW — pooled binary, all MF, all fuels only
    # ==================================================================
    for binary_col in BINARY_COLS_SECONDARY:
        if binary_col not in df.columns:
            logger.warning("%s not found — skipping", binary_col)
            continue

        sys_label = binary_col.replace("_binary", "")
        classifiable = df[df[binary_col].isin(["Central", "Distributed"])].copy()
        central = classifiable[classifiable[binary_col] == "Central"]
        distributed = classifiable[classifiable[binary_col] == "Distributed"]

        print(f"\n{'=' * 70}")
        print(
            f"  {sys_label}  [All MF | All Fuels | pooled binary]\n"
            f"  Central (n={len(central)}) vs Distributed (n={len(distributed)})"
        )
        print(f"{'=' * 70}")

        for metric in OUTCOME_COLS:
            if metric not in df.columns:
                continue
            res = compare_groups(
                central[metric], distributed[metric],
                metric, sys_label,
            )
            res["mf_segment"] = "all_mf"
            res["fuel_stratum"] = "all_fuels"
            res["classification_view"] = "pooled_binary"
            results.append(res)

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}")
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        display_cols = [
            "system_col", "mf_segment", "fuel_stratum", "classification_view",
            "metric", "n_central", "n_distributed",
            "median_central", "median_distributed",
            "U_statistic", "p_value", "rank_biserial_r", "significant",
        ]
        display_cols = [c for c in display_cols if c in result_df.columns]
        print(result_df[display_cols].to_string(index=False))

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
