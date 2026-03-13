"""Prompt 6 — Climate and Vintage Controls.

Two strategies to control for climate and vintage when comparing
Central vs. Distributed EUI:

  Strategy A — OLS regression
      Site_EUI_kBtu_sqft ~ C(heating_system_type) + Heating_Zone
                           + Home_Vintage + Conditioned_Area

  Strategy B — Within-zone Mann-Whitney U
      Run Central vs. Distributed comparison within each Heating_Zone
      (1 = mild, 2 = moderate, 3 = cold).

Both strategies are run for Heating, Cooling, and DHW system types.

Also produces a scatter of Site_EUI vs. Home_Vintage coloured by system type.

Usage::

    python analysis/rbsa/04_climate_vintage_controlled.py \\
        --site-master outputs/rbsa/rbsa_site_master_*.parquet \\
        --outdir outputs/rbsa
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

from src.common.log import get_logger

logger = get_logger("rbsa.04_climate")

ALPHA = 0.05
OUTCOME_COL = "Site_EUI_kBtu_sqft"
SYSTEM_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]
HEATING_ZONES = [1, 2, 3]
ZONE_LABELS = {1: "Zone 1 (Mild)", 2: "Zone 2 (Moderate)", 3: "Zone 3 (Cold)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_site_master(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


# ---------------------------------------------------------------------------
# Strategy A: OLS regression
# ---------------------------------------------------------------------------


def run_ols(df: pd.DataFrame, sys_col: str) -> dict:
    """OLS: Site_EUI ~ system_type + Heating_Zone + Home_Vintage + Conditioned_Area."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not installed; skipping OLS"}

    sub = df[df[sys_col].isin(["Central", "Distributed"])].copy()
    needed = [OUTCOME_COL, sys_col, "Heating_Zone", "Home_Vintage", "Conditioned_Area"]
    for c in needed:
        if c not in sub.columns:
            return {"error": f"Required column missing: {c}"}
    sub = sub[needed].dropna()

    if len(sub) < 10:
        return {"error": f"Too few obs for OLS (n={len(sub)})"}

    # Encode system type: 1 = Distributed (Central is the reference)
    sub["_distributed"] = (sub[sys_col] == "Distributed").astype(int)
    sub["Heating_Zone"] = pd.to_numeric(sub["Heating_Zone"], errors="coerce")
    sub["Home_Vintage"] = pd.to_numeric(sub["Home_Vintage"], errors="coerce")
    sub["Conditioned_Area"] = pd.to_numeric(sub["Conditioned_Area"], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 10:
        return {"error": f"Too few complete obs after coercion (n={len(sub)})"}

    X_cols = ["_distributed", "Heating_Zone", "Home_Vintage", "Conditioned_Area"]
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels not installed"}

    X = sm.add_constant(sub[X_cols])
    y = sub[OUTCOME_COL]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit()

    coef = model.params.get("_distributed", None)
    pval = model.pvalues.get("_distributed", None)
    ci = model.conf_int().loc["_distributed"] if "_distributed" in model.conf_int().index else (None, None)

    return {
        "n": int(model.nobs),
        "coef_distributed": round(float(coef), 2) if coef is not None else None,
        "p_distributed": round(float(pval), 4) if pval is not None else None,
        "ci_low": round(float(ci[0]), 2) if ci[0] is not None else None,
        "ci_high": round(float(ci[1]), 2) if ci[1] is not None else None,
        "R2": round(model.rsquared, 3),
        "significant": bool(float(pval) < ALPHA) if pval is not None else None,
        "summary": model.summary().as_text(),
    }


# ---------------------------------------------------------------------------
# Strategy B: Within-zone Mann-Whitney U
# ---------------------------------------------------------------------------


def run_within_zone(df: pd.DataFrame, sys_col: str) -> list[dict]:
    """Mann-Whitney within each Heating_Zone."""
    sub = df[df[sys_col].isin(["Central", "Distributed"])].copy()
    sub = sub[sub[OUTCOME_COL].notna()]

    if "Heating_Zone" not in sub.columns:
        return [{"error": "Heating_Zone column not found"}]

    results = []
    for zone in HEATING_ZONES:
        zdf = sub[pd.to_numeric(sub["Heating_Zone"], errors="coerce") == zone]
        c = zdf.loc[zdf[sys_col] == "Central", OUTCOME_COL].dropna().values
        d = zdf.loc[zdf[sys_col] == "Distributed", OUTCOME_COL].dropna().values

        row: dict = {
            "zone": zone,
            "zone_label": ZONE_LABELS[zone],
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
# Scatter: EUI vs Home_Vintage coloured by system type
# ---------------------------------------------------------------------------


def make_scatter(df: pd.DataFrame, sys_col: str, outdir: Path) -> None:
    sub = df[df[sys_col].isin(["Central", "Distributed"])].copy()
    sub = sub[sub[OUTCOME_COL].notna()]

    if "Home_Vintage" not in sub.columns or sub.empty:
        return

    sub["_vintage"] = pd.to_numeric(sub["Home_Vintage"], errors="coerce")
    sub = sub.dropna(subset=["_vintage"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [("Central", "#2196F3"), ("Distributed", "#FF7043")]:
        grp = sub[sub[sys_col] == label]
        ax.scatter(grp["_vintage"], grp[OUTCOME_COL],
                   label=f"{label} (n={len(grp)})", color=color, alpha=0.5, s=25)

    ax.set_xlabel("Home_Vintage (year built)")
    ax.set_ylabel("Site EUI (kBtu/sqft/yr)")
    ax.set_title(f"Site EUI vs. Home Vintage\n({sys_col})", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fname = outdir / f"04_scatter_vintage_{sys_col}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Climate/vintage-controlled comparison for RBSA.")
    ap.add_argument("--site-master", type=Path, required=True,
                    help="Path to rbsa_site_master_*.parquet (or .csv)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/rbsa"),
                    help="Directory for output files.")
    ap.add_argument("--sf-only", action="store_true",
                    help="Restrict to Single Family sites.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading site master from %s", args.site_master)
    df = _load_site_master(args.site_master)

    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in df.columns), None)
    if args.sf_only and bt_col:
        df = df[df[bt_col].str.lower().str.contains("single", na=False)].copy()
        logger.info("SF-only filter: %d rows", len(df))

    all_ols = []
    all_zone = []

    for sys_col in SYSTEM_COLS:
        if sys_col not in df.columns:
            logger.warning("%s not found — skipping", sys_col)
            continue

        print(f"\n{'='*70}")
        print(f"  {sys_col}")
        print(f"{'='*70}")

        # --- Strategy A: OLS ---
        print(f"\n  [OLS] {OUTCOME_COL}")
        ols = run_ols(df, sys_col)
        if "error" in ols:
            print(f"    {ols['error']}")
        else:
            sig_str = "✓ SIGNIFICANT" if ols["significant"] else "✗ not significant"
            print(
                f"    n={ols['n']},  coef_distributed={ols['coef_distributed']},  "
                f"p={ols['p_distributed']},  "
                f"CI=[{ols['ci_low']}, {ols['ci_high']}],  R²={ols['R2']}\n"
                f"    → {sig_str} (α={ALPHA})\n"
                f"    Interpretation: Distributed EUI is {ols['coef_distributed']:+.1f} kBtu/sqft/yr "
                f"relative to Central, after controlling for Heating_Zone, "
                f"Home_Vintage, and Conditioned_Area."
            )
        all_ols.append({"system_col": sys_col, "outcome": OUTCOME_COL,
                         **{k: v for k, v in ols.items() if k != "summary"}})

        # --- Strategy B: Within-zone ---
        print(f"\n  [Within-Zone] {OUTCOME_COL}")
        zone_rows = run_within_zone(df, sys_col)
        for zr in zone_rows:
            if "error" in zr:
                print(f"    {zr['error']}")
                continue
            sig_str = ("✓ sig" if zr["significant"] else "✗ n.s.") if zr["significant"] is not None else "—"
            note = f" [{zr['note']}]" if zr["note"] else ""
            print(
                f"    {zr['zone_label']:22s}  C(n={zr['n_central']}) D(n={zr['n_distributed']})"
                f"  median C={zr['median_central']}  D={zr['median_distributed']}"
                f"  p={zr['p_value']}  {sig_str}{note}"
            )
            all_zone.append({"system_col": sys_col, "outcome": OUTCOME_COL, **zr})

        # --- Scatter ---
        make_scatter(df, sys_col, args.outdir)

    # Save results
    pd.DataFrame(all_ols).to_csv(args.outdir / "04_ols_results.csv", index=False)
    pd.DataFrame(all_zone).to_csv(args.outdir / "04_within_zone_results.csv", index=False)
    logger.info("Saved OLS and within-zone results CSVs.")


if __name__ == "__main__":
    main()
