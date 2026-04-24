"""RECS 2020 — Step 4: Climate and vintage controlled comparison.

Two strategies to control for climate zone and vintage when comparing
Central vs. Distributed EUI:

  Strategy A — OLS regression
      Site_EUI_kBtu_sqft ~ central_dummy + C(IECC_climate_code)
                           + YEARMADERANGE + log(TOTSQFT_EN)
      Optional: include NWEIGHT for survey-weighted estimates.

  Strategy B — Within-climate-zone Mann-Whitney U
      Run Central vs. Distributed comparison within each unique
      IECC_climate_code value.

Both strategies are run for heating, cooling, and DHW system types.

Also produces scatter plots of Site_EUI vs. YEARMADERANGE coloured by
system type.

Usage::

    python analysis/recs/04_climate_vintage_controlled.py \\
        --curated outputs/recs/recs2020_curated_*.parquet \\
        --outdir outputs/recs
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    friendly_label,
)

logger = get_logger("recs.04_climate")

ALPHA = 0.05
OUTCOME_COLS = ["Heating_EUI_kBtu_sqft", "Site_EUI_kBtu_sqft"]
BINARY_COL = "heating_system_type_binary"

# For backward compat, also run cooling/DHW in all-MF pooled only
BINARY_COLS_SECONDARY = [
    "cooling_system_type_binary",
    "dhw_system_type_binary",
]


# ---------------------------------------------------------------------------
# Strategy A: OLS regression
# ---------------------------------------------------------------------------


def run_ols(df: pd.DataFrame, binary_col: str, outcome_col: str = "Heating_EUI_kBtu_sqft", use_weights: bool = False) -> dict:
    """OLS: outcome ~ distributed_dummy + C(IECC_climate_code) + YEARMADERANGE + log(TOTSQFT_EN)."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels not installed; skipping OLS"}

    sub = df[df[binary_col].isin(["Central", "Distributed"])].copy()
    needed = [outcome_col, binary_col, "IECC_climate_code", "YEARMADERANGE", "TOTSQFT_EN"]
    for c in needed:
        if c not in sub.columns:
            return {"error": f"Required column missing: {c}"}
    sub = sub[needed + (["NWEIGHT"] if use_weights and "NWEIGHT" in sub.columns else [])].dropna()

    if len(sub) < 10:
        return {"error": f"Too few observations for OLS (n={len(sub)})"}

    sub["_distributed"] = (sub[binary_col] == "Distributed").astype(int)
    sub["YEARMADERANGE"] = pd.to_numeric(sub["YEARMADERANGE"], errors="coerce")
    sub["TOTSQFT_EN"] = pd.to_numeric(sub["TOTSQFT_EN"], errors="coerce")
    sub["_log_sqft"] = np.log(sub["TOTSQFT_EN"].clip(lower=1))
    sub = sub.dropna(subset=["_distributed", "YEARMADERANGE", "_log_sqft", outcome_col])

    if len(sub) < 10:
        return {"error": f"Too few complete observations after coercion (n={len(sub)})"}

    # One-hot encode IECC_climate_code; drop first for reference category
    climate_dummies = pd.get_dummies(
        sub["IECC_climate_code"].astype(str), prefix="climate", drop_first=True
    )

    X_cols_df = pd.concat(
        [sub[["_distributed", "YEARMADERANGE", "_log_sqft"]], climate_dummies],
        axis=1,
    ).astype(float)
    X = sm.add_constant(X_cols_df)
    y = sub[outcome_col].astype(float)

    kwargs: dict = {}
    if use_weights and "NWEIGHT" in sub.columns:
        kwargs["weights"] = sub["NWEIGHT"].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X, **kwargs).fit()

    coef = model.params.get("_distributed")
    pval = model.pvalues.get("_distributed")
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
# Strategy A′: OLS with vintage × system-type interaction
# ---------------------------------------------------------------------------


def run_ols_interaction(df: pd.DataFrame, binary_col: str, outcome_col: str = "Heating_EUI_kBtu_sqft", use_weights: bool = False) -> dict:
    """OLS with interaction term: _distributed * YEARMADERANGE.

    Tests whether the system-type effect on EUI varies with building age.
    A significant interaction means the Central-vs-Distributed gap changes
    across vintage categories.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels not installed; skipping OLS"}

    sub = df[df[binary_col].isin(["Central", "Distributed"])].copy()
    needed = [outcome_col, binary_col, "IECC_climate_code", "YEARMADERANGE", "TOTSQFT_EN"]
    for c in needed:
        if c not in sub.columns:
            return {"error": f"Required column missing: {c}"}
    sub = sub[needed + (["NWEIGHT"] if use_weights and "NWEIGHT" in sub.columns else [])].dropna()

    if len(sub) < 10:
        return {"error": f"Too few observations for OLS (n={len(sub)})"}

    sub["_distributed"] = (sub[binary_col] == "Distributed").astype(int)
    sub["YEARMADERANGE"] = pd.to_numeric(sub["YEARMADERANGE"], errors="coerce")
    sub["TOTSQFT_EN"] = pd.to_numeric(sub["TOTSQFT_EN"], errors="coerce")
    sub["_log_sqft"] = np.log(sub["TOTSQFT_EN"].clip(lower=1))
    sub["_dist_x_vintage"] = sub["_distributed"] * sub["YEARMADERANGE"]
    sub = sub.dropna(subset=["_distributed", "YEARMADERANGE", "_log_sqft", "_dist_x_vintage", outcome_col])

    if len(sub) < 10:
        return {"error": f"Too few complete observations after coercion (n={len(sub)})"}

    climate_dummies = pd.get_dummies(
        sub["IECC_climate_code"].astype(str), prefix="climate", drop_first=True
    )

    X_cols_df = pd.concat(
        [sub[["_distributed", "YEARMADERANGE", "_log_sqft", "_dist_x_vintage"]], climate_dummies],
        axis=1,
    ).astype(float)
    X = sm.add_constant(X_cols_df)
    y = sub[outcome_col].astype(float)

    kwargs: dict = {}
    if use_weights and "NWEIGHT" in sub.columns:
        kwargs["weights"] = sub["NWEIGHT"].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X, **kwargs).fit()

    result: dict = {"n": int(model.nobs), "R2": round(model.rsquared, 3)}

    for term in ["_distributed", "_dist_x_vintage", "YEARMADERANGE"]:
        coef = model.params.get(term)
        pval = model.pvalues.get(term)
        ci = model.conf_int().loc[term] if term in model.conf_int().index else (None, None)
        result[f"coef_{term}"] = round(float(coef), 2) if coef is not None else None
        result[f"p_{term}"] = round(float(pval), 4) if pval is not None else None
        result[f"ci_low_{term}"] = round(float(ci[0]), 2) if ci[0] is not None else None
        result[f"ci_high_{term}"] = round(float(ci[1]), 2) if ci[1] is not None else None

    result["interaction_significant"] = bool(
        float(result.get("p__dist_x_vintage", 1.0) or 1.0) < ALPHA
    )
    result["summary"] = model.summary().as_text()
    return result


# ---------------------------------------------------------------------------
# Strategy B: Within-climate-zone Mann-Whitney U
# ---------------------------------------------------------------------------


def run_within_zone(df: pd.DataFrame, binary_col: str, outcome_col: str = "Heating_EUI_kBtu_sqft") -> list[dict]:
    """Mann-Whitney Central vs. Distributed within each IECC_climate_code."""
    sub = df[df[binary_col].isin(["Central", "Distributed"])].copy()
    sub = sub[sub[outcome_col].notna()]

    if "IECC_climate_code" not in sub.columns:
        return [{"error": "IECC_climate_code column not found"}]

    results = []
    zones = sorted(sub["IECC_climate_code"].dropna().unique())

    for zone in zones:
        zdf = sub[sub["IECC_climate_code"] == zone]
        c = zdf.loc[zdf[binary_col] == "Central", outcome_col].dropna().values
        d = zdf.loc[zdf[binary_col] == "Distributed", outcome_col].dropna().values

        row: dict = {
            "zone": zone,
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
# Scatter: EUI vs YEARMADERANGE coloured by system type
# ---------------------------------------------------------------------------


def make_scatter(df: pd.DataFrame, binary_col: str, outdir: Path, outcome_col: str = "Heating_EUI_kBtu_sqft") -> None:
    sub = df[df[binary_col].isin(["Central", "Distributed"])].copy()
    sub = sub[sub[outcome_col].notna()]

    if "YEARMADERANGE" not in sub.columns or sub.empty:
        return

    sub["_vintage"] = pd.to_numeric(sub["YEARMADERANGE"], errors="coerce")
    sub = sub.dropna(subset=["_vintage"])
    if sub.empty:
        return

    sys_label = binary_col.replace("_binary", "")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [("Central", "#2196F3"), ("Distributed", "#FF7043")]:
        grp = sub[sub[binary_col] == label]
        ax.scatter(grp["_vintage"], grp[outcome_col],
                   label=f"{label} (n={len(grp)})", color=color, alpha=0.5, s=25)

    short_outcome = outcome_col.replace("_kBtu_sqft", "")
    ax.set_xlabel("YEARMADERANGE (home vintage category)")
    ax.set_ylabel(f"{short_outcome} (kBtu/sqft/yr)")
    ax.set_title(f"{short_outcome} vs. Home Vintage\n({sys_label})", fontsize=9)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    fname = outdir / f"04_scatter_vintage_{sys_label}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Climate/vintage-controlled comparison for RECS 2020.")
    ap.add_argument(
        "--curated", type=Path, required=True,
        help="Path to recs2020_curated_*.parquet (output of 01_build_curated_table.py).",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/recs"),
        help="Directory for output files (default: outputs/recs).",
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
    ap.add_argument(
        "--use-weights",
        action="store_true",
        default=False,
        help="Include NWEIGHT survey weights in OLS models.",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading curated data from %s", args.curated)
    df = load_curated(args.curated)
    df = filter_unit_type(df, args.unit_type)
    logger.info("Rows after unit-type filter (%s): %d", args.unit_type, len(df))

    all_ols = []
    all_ols_interact = []
    all_zone = []

    # ==================================================================
    # PRIMARY: heating system type across segment × fuel × outcome grid
    # ==================================================================
    binary_col = BINARY_COL
    sys_label = binary_col.replace("_binary", "")

    for outcome_col in OUTCOME_COLS:
        short = outcome_col.replace("_kBtu_sqft", "")
        for seg_key in MF_SEGMENTS:
            seg_df = filter_segment(df, seg_key)
            for fuel_key in FUEL_STRATA:
                fuel_df = filter_fuel(seg_df, fuel_key)

                # Run pooled binary (primary) and explicit only (secondary)
                for view in ["pooled_binary", "explicit_only"]:
                    view_df = filter_classification_view(fuel_df, view, binary_col)
                    label = friendly_label(seg_key, fuel_key, view)

                    print(f"\n{'=' * 70}")
                    print(f"  {sys_label} | {short} | {label}")
                    print(f"{'=' * 70}")

                    # --- OLS ---
                    ols = run_ols(view_df, binary_col, outcome_col=outcome_col, use_weights=args.use_weights)
                    if "error" in ols:
                        print(f"    [OLS] {ols['error']}")
                    else:
                        sig_str = "✓ SIG" if ols["significant"] else "✗ n.s."
                        print(
                            f"    [OLS] n={ols['n']}  coef={ols['coef_distributed']:+.1f}  "
                            f"p={ols['p_distributed']}  CI=[{ols['ci_low']},{ols['ci_high']}]  "
                            f"R²={ols['R2']}  → {sig_str}"
                        )
                    all_ols.append({
                        "system_col": sys_label,
                        "outcome": outcome_col,
                        "mf_segment": seg_key,
                        "fuel_stratum": fuel_key,
                        "classification_view": view,
                        **{k: v for k, v in ols.items() if k != "summary"},
                    })

                    # --- OLS + vintage interaction (only for pooled binary, all fuels) ---
                    if view == "pooled_binary" and fuel_key == "all_fuels":
                        ols_int = run_ols_interaction(view_df, binary_col, outcome_col=outcome_col, use_weights=args.use_weights)
                        if "error" in ols_int:
                            print(f"    [OLS×vintage] {ols_int['error']}")
                        else:
                            int_sig = "✓ SIG" if ols_int["interaction_significant"] else "✗ n.s."
                            print(
                                f"    [OLS×vintage] n={ols_int['n']}  R²={ols_int['R2']}  "
                                f"interaction coef={ols_int.get('coef__dist_x_vintage', '?')}  "
                                f"p={ols_int.get('p__dist_x_vintage', '?')}  → {int_sig}"
                            )
                        all_ols_interact.append({
                            "system_col": sys_label,
                            "outcome": outcome_col,
                            "mf_segment": seg_key,
                            "fuel_stratum": fuel_key,
                            "classification_view": view,
                            **{k: v for k, v in ols_int.items() if k != "summary"},
                        })

                    # --- Within-zone (only pooled binary, all fuels, Heating EUI) ---
                    if view == "pooled_binary" and fuel_key == "all_fuels" and outcome_col == "Heating_EUI_kBtu_sqft":
                        zone_rows = run_within_zone(view_df, binary_col, outcome_col=outcome_col)
                        for zr in zone_rows:
                            if "error" in zr:
                                continue
                            all_zone.append({
                                "system_col": sys_label,
                                "outcome": outcome_col,
                                "mf_segment": seg_key,
                                **zr,
                            })

        # Scatter for heating EUI (all MF, pooled binary)
        make_scatter(df, binary_col, args.outdir, outcome_col=outcome_col)

    # ==================================================================
    # SECONDARY: cooling / DHW — pooled binary, all MF, all fuels, Site EUI
    # ==================================================================
    for bcol in BINARY_COLS_SECONDARY:
        if bcol not in df.columns:
            continue
        slabel = bcol.replace("_binary", "")
        ols = run_ols(df, bcol, outcome_col="Site_EUI_kBtu_sqft", use_weights=args.use_weights)
        all_ols.append({
            "system_col": slabel,
            "outcome": "Site_EUI_kBtu_sqft",
            "mf_segment": "all_mf",
            "fuel_stratum": "all_fuels",
            "classification_view": "pooled_binary",
            **{k: v for k, v in ols.items() if k != "summary"},
        })

    # ==================================================================
    # Save results
    # ==================================================================
    ols_df = pd.DataFrame(all_ols)
    zone_df = pd.DataFrame(all_zone)

    if not zone_df.empty and "p_value" in zone_df.columns:
        tested = zone_df[zone_df["p_value"].notna()]
        n_tests = len(tested)
        if n_tests > 1:
            bonferroni_alpha = round(ALPHA / n_tests, 4)
            n_bonf_sig = int((tested["p_value"] < bonferroni_alpha).sum())
            print(
                f"\n  ⚠ Within-zone multiple testing: {n_tests} tests at α={ALPHA}."
                f"\n    Bonferroni-adjusted α = {bonferroni_alpha}"
                f" → {n_bonf_sig} of {n_tests} remain significant after correction."
            )
            zone_df["bonferroni_significant"] = zone_df["p_value"] < bonferroni_alpha

    # Print OLS summary table
    print(f"\n{'=' * 70}")
    print("  OLS RESULTS SUMMARY")
    print(f"{'=' * 70}")
    if not ols_df.empty:
        display = ["system_col", "outcome", "mf_segment", "fuel_stratum",
                    "classification_view", "n", "coef_distributed", "p_distributed",
                    "ci_low", "ci_high", "R2", "significant"]
        display = [c for c in display if c in ols_df.columns]
        print(ols_df[display].to_string(index=False))

    ols_df.to_csv(args.outdir / "04_ols_results.csv", index=False)
    zone_df.to_csv(args.outdir / "04_within_zone_results.csv", index=False)
    pd.DataFrame(all_ols_interact).to_csv(
        args.outdir / "04_ols_interaction_results.csv", index=False
    )
    logger.info("Saved OLS, interaction-OLS, and within-zone results CSVs.")


if __name__ == "__main__":
    main()
