"""Prompt 4 — Climate-Controlled Comparison.

Two strategies to control for climate when comparing central vs. distributed EUI:

  Strategy A — OLS regression
      Site EUI ~ system_type_dummy + HDD65 + CDD65

  Strategy B — Within-Region stratification
      Run Mann-Whitney U for Central vs. Distributed within each of the
      4 Fannie Mae regions (West, Midwest, South, Northeast).

Both strategies are run for DHW, Heating, and Cooling system types.

Usage::

    python analysis/mf_survey/04_climate_controlled.py \\
        --data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        [--outdir outputs/mf_survey]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.datasets.mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.mf_survey.classify import add_system_classifications, to_binary
from src.common.log import get_logger

logger = get_logger("mf_survey.04_climate")

ALPHA = 0.05
OUTCOME_COLS = ["Site EUI (kBTU/sqft/yr)", "Source EUI (kBTU/sqft/yr)"]
SYSTEM_COLS = ["dhw_system_type", "heating_system_type", "cooling_system_type"]
REGIONS = ["West", "Midwest", "South", "Northeast"]


# ---------------------------------------------------------------------------
# Strategy A: OLS regression
# ---------------------------------------------------------------------------


def run_ols(df: pd.DataFrame, outcome: str, sys_col: str) -> dict:
    """OLS: outcome ~ system_dummy + HDD65 + CDD65.

    Returns a dict with coefficient estimates and p-values.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels not installed; skipping OLS"}

    binary = to_binary(df[sys_col])
    sub = df.copy()
    sub["_binary"] = binary
    sub = sub[sub["_binary"].notna() & sub[outcome].notna()]
    sub = sub[sub["2022 Annual HDD65"].notna() & sub["2022 Annual CDD65"].notna()]

    if len(sub) < 5:
        return {"error": f"Too few obs for OLS (n={len(sub)})"}

    # Dummy: 1 = Central, 0 = Distributed
    sub["_central_dummy"] = (sub["_binary"] == "Central").astype(int)

    X = sub[["_central_dummy", "2022 Annual HDD65", "2022 Annual CDD65"]]
    X = sm.add_constant(X)
    y = sub[outcome]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit()

    coef_row = model.params.get("_central_dummy", None)
    pval_row = model.pvalues.get("_central_dummy", None)
    r2 = model.rsquared
    n = int(model.nobs)

    return {
        "n": n,
        "coef_central_dummy": round(float(coef_row), 2) if coef_row is not None else None,
        "p_central_dummy": round(float(pval_row), 4) if pval_row is not None else None,
        "R2": round(r2, 3),
        "significant": bool(float(pval_row) < ALPHA) if pval_row is not None else None,
        "summary": model.summary().as_text(),
    }


# ---------------------------------------------------------------------------
# Strategy B: Within-region Mann-Whitney U
# ---------------------------------------------------------------------------


def run_within_region(
    df: pd.DataFrame, outcome: str, sys_col: str, region_col: str = "Region"
) -> list[dict]:
    """Mann-Whitney within each region."""
    binary = to_binary(df[sys_col])
    sub = df.copy()
    sub["_binary"] = binary
    sub = sub[sub["_binary"].notna() & sub[outcome].notna()]

    results = []
    for region in REGIONS:
        rdf = sub[sub[region_col] == region] if region_col in sub.columns else pd.DataFrame()

        c = rdf.loc[rdf["_binary"] == "Central", outcome].dropna().values
        d = rdf.loc[rdf["_binary"] == "Distributed", outcome].dropna().values

        row = {
            "region": region,
            "n_central": len(c),
            "n_distributed": len(d),
            "median_central": round(float(np.median(c)), 2) if len(c) else None,
            "median_distributed": round(float(np.median(d)), 2) if len(d) else None,
            "U": None,
            "p_value": None,
            "significant": None,
            "note": "",
        }

        if len(c) >= 2 and len(d) >= 2:
            u, p = stats.mannwhitneyu(c, d, alternative="two-sided")
            row["U"] = round(float(u), 1)
            row["p_value"] = round(float(p), 4)
            row["significant"] = bool(p < ALPHA)
        else:
            row["note"] = f"Insufficient data (C={len(c)}, D={len(d)})"

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Scatter: EUI vs HDD65 / CDD65 coloured by system type
# ---------------------------------------------------------------------------


def make_scatter(df: pd.DataFrame, sys_col: str, outcome: str, outdir: Path) -> None:
    binary = to_binary(df[sys_col])
    sub = df.copy()
    sub["_binary"] = binary
    sub = sub[sub["_binary"].notna() & sub[outcome].notna()]

    if sub.empty:
        return

    for climate_col, xlabel in [("2022 Annual HDD65", "HDD65"), ("2022 Annual CDD65", "CDD65")]:
        if climate_col not in sub.columns:
            continue
        sub2 = sub[sub[climate_col].notna()]
        if sub2.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color in [("Central", "#2196F3"), ("Distributed", "#FF7043")]:
            grp = sub2[sub2["_binary"] == label]
            ax.scatter(grp[climate_col], grp[outcome],
                       label=f"{label} (n={len(grp)})", color=color, alpha=0.6, s=30)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(outcome)
        ax.set_title(f"{outcome} vs {xlabel}\n({sys_col})", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()

        fname = outdir / f"04_scatter_{sys_col}_{xlabel}_{outcome[:8].replace('/', '')}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Climate-controlled comparison for 2023 MF Survey.")
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/mf_survey"),
                    help="Directory for output files.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data …")
    df = load_energy_subset(load_mf_survey(args.data))
    df = add_system_classifications(df)

    all_ols = []
    all_region = []

    for sys_col in SYSTEM_COLS:
        print(f"\n{'='*70}")
        print(f"  {sys_col}")
        print(f"{'='*70}")

        for outcome in OUTCOME_COLS:
            if outcome not in df.columns:
                continue

            # --- Strategy A: OLS ---
            print(f"\n  [OLS] {outcome}")
            ols = run_ols(df, outcome, sys_col)
            if "error" in ols:
                print(f"    {ols['error']}")
            else:
                sig_str = "✓ SIGNIFICANT" if ols["significant"] else "✗ not significant"
                print(
                    f"    n={ols['n']},  coef_central={ols['coef_central_dummy']},  "
                    f"p={ols['p_central_dummy']},  R²={ols['R2']}\n"
                    f"    → {sig_str} (α={ALPHA})"
                )

            all_ols.append({
                "system_col": sys_col,
                "outcome": outcome,
                **{k: v for k, v in ols.items() if k != "summary"},
            })

            # --- Strategy B: Within-region ---
            print(f"\n  [Within-Region] {outcome}")
            region_rows = run_within_region(df, outcome, sys_col)
            for rr in region_rows:
                sig_str = ("✓ sig" if rr["significant"] else "✗ n.s.") if rr["significant"] is not None else "—"
                note = f" [{rr['note']}]" if rr["note"] else ""
                print(
                    f"    {rr['region']:12s}  C(n={rr['n_central']}) D(n={rr['n_distributed']})"
                    f"  p={rr['p_value']}  {sig_str}{note}"
                )
                all_region.append({"system_col": sys_col, "outcome": outcome, **rr})

            # --- Scatterplot ---
            make_scatter(df, sys_col, outcome, args.outdir)

    # Save results
    pd.DataFrame(all_ols).to_csv(args.outdir / "04_ols_results.csv", index=False)
    pd.DataFrame(all_region).to_csv(args.outdir / "04_within_region_results.csv", index=False)
    logger.info("Saved OLS and within-region CSV results.")


if __name__ == "__main__":
    main()
