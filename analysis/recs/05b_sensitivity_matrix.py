"""RECS 2020 — Step 5b: Sensitivity matrix.

Builds a summary table showing the central-vs-distributed heating signal
across the most important slice combinations:
  - mf_segment × fuel × classification_view × outcome
  - amenity / EV exclusion sensitivities on key slices

Each row contains: class counts, medians, delta median, controlled OLS
coefficient + p-value, and notes on sample filters.

Outputs:
  outputs/recs/heating_sensitivity_matrix.csv
  outputs/recs/heating_sensitivity_matrix.md

Usage::

    python analysis/recs/05b_sensitivity_matrix.py \\
        --curated outputs/recs/recs2020_curated_*.parquet \\
        --outdir outputs/recs
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.log import get_logger
from src.datasets.recs.utils import (
    load_curated,
    filter_unit_type,
    filter_segment,
    filter_fuel,
    filter_classification_view,
)

logger = get_logger("recs.05b_sensitivity")


# ---------------------------------------------------------------------------
# Mini OLS helper (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def _quick_ols(df: pd.DataFrame, outcome_col: str, binary_col: str) -> dict:
    """Run a lightweight OLS and return coef, p, CI for the distributed indicator."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {}

    sub = df[[outcome_col, binary_col, "IECC_climate_code", "YEARMADERANGE", "TOTSQFT_EN"]].dropna()
    sub = sub[sub[binary_col].isin(["Central", "Distributed"])].copy()
    if len(sub) < 20:
        return {"ols_note": f"n={len(sub)}<20, skipped"}

    sub["_dist"] = (sub[binary_col] == "Distributed").astype(int)
    sub["YEARMADERANGE"] = pd.to_numeric(sub["YEARMADERANGE"], errors="coerce")
    sub["TOTSQFT_EN"] = pd.to_numeric(sub["TOTSQFT_EN"], errors="coerce")
    sub["_log_sqft"] = np.log(sub["TOTSQFT_EN"].clip(lower=1))
    sub = sub.dropna()
    if len(sub) < 20:
        return {"ols_note": f"n={len(sub)}<20 after drop, skipped"}

    climate_dummies = pd.get_dummies(
        sub["IECC_climate_code"].astype(str), prefix="cz", drop_first=True,
    )
    X = pd.concat(
        [sub[["_dist", "YEARMADERANGE", "_log_sqft"]], climate_dummies], axis=1,
    ).astype(float)
    X = sm.add_constant(X)
    y = sub[outcome_col].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit()

    coef = model.params.get("_dist")
    pval = model.pvalues.get("_dist")
    ci = model.conf_int().loc["_dist"] if "_dist" in model.conf_int().index else (None, None)
    return {
        "ols_n": int(model.nobs),
        "ols_coef": round(float(coef), 2) if coef is not None else None,
        "ols_p": round(float(pval), 4) if pval is not None else None,
        "ols_ci_low": round(float(ci[0]), 2) if ci[0] is not None else None,
        "ols_ci_high": round(float(ci[1]), 2) if ci[1] is not None else None,
        "ols_R2": round(model.rsquared, 3),
    }


# ---------------------------------------------------------------------------
# One sensitivity row
# ---------------------------------------------------------------------------

def _sensitivity_row(
    df: pd.DataFrame,
    outcome_col: str,
    binary_col: str,
    label: str,
    segment: str,
    fuel: str,
    view: str,
    extra_filter: str = "none",
) -> dict:
    """Compute one row of the sensitivity matrix."""
    classifiable = df[df[binary_col].isin(["Central", "Distributed"])]
    c = classifiable.loc[classifiable[binary_col] == "Central", outcome_col].dropna()
    d = classifiable.loc[classifiable[binary_col] == "Distributed", outcome_col].dropna()

    row: dict = {
        "label": label,
        "outcome": outcome_col,
        "mf_segment": segment,
        "fuel": fuel,
        "classification_view": view,
        "extra_filter": extra_filter,
        "n_central": len(c),
        "n_distributed": len(d),
        "median_central": round(float(c.median()), 2) if len(c) else None,
        "median_distributed": round(float(d.median()), 2) if len(d) else None,
        "delta_median": None,
    }

    if row["median_central"] is not None and row["median_distributed"] is not None:
        row["delta_median"] = round(row["median_central"] - row["median_distributed"], 2)

    # OLS
    ols = _quick_ols(classifiable, outcome_col, binary_col)
    row.update(ols)

    return row


# ---------------------------------------------------------------------------
# Slicing definitions
# ---------------------------------------------------------------------------

# The ~20 most important slices
SLICES = [
    # (segment, fuel, view, extra_filter_label)
    ("all_mf", "all_fuels", "pooled_binary", "none"),
    ("all_mf", "all_fuels", "explicit_only", "none"),
    ("all_mf", "all_fuels", "inferred_only", "none"),
    ("2_to_4_units", "all_fuels", "pooled_binary", "none"),
    ("5plus_units", "all_fuels", "pooled_binary", "none"),
    ("5plus_units", "all_fuels", "explicit_only", "none"),
    ("5plus_units", "all_fuels", "inferred_only", "none"),
    ("5plus_units", "electric", "pooled_binary", "none"),
    ("5plus_units", "gas", "pooled_binary", "none"),
    ("2_to_4_units", "electric", "pooled_binary", "none"),
    ("2_to_4_units", "gas", "pooled_binary", "none"),
    # Amenity / EV sensitivities on the lead segment
    ("5plus_units", "all_fuels", "pooled_binary", "excl_amenity"),
    ("5plus_units", "all_fuels", "pooled_binary", "excl_ev"),
    ("5plus_units", "all_fuels", "pooled_binary", "excl_both"),
    ("all_mf", "all_fuels", "pooled_binary", "excl_amenity"),
    ("all_mf", "all_fuels", "pooled_binary", "excl_ev"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build heating sensitivity matrix for RECS 2020.")
    ap.add_argument("--curated", type=Path, required=True,
                    help="Path to recs2020_curated_*.parquet.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/recs"),
                    help="Directory for output files.")
    ap.add_argument("--unit-type", choices=["mf", "sf", "all"], default="mf",
                    help="Housing unit filter (default: mf).")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading curated data from %s", args.curated)
    df = load_curated(args.curated)
    df = filter_unit_type(df, args.unit_type)
    logger.info("Rows after unit-type filter (%s): %d", args.unit_type, len(df))

    binary_col = "heating_system_type_binary"
    outcomes = ["Heating_EUI_kBtu_sqft", "Site_EUI_kBtu_sqft"]
    rows = []

    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for seg, fuel, view, extra in SLICES:
            sub = filter_segment(df, seg)
            sub = filter_fuel(sub, fuel)
            sub = filter_classification_view(sub, view, binary_col)

            # Apply extra filters
            if extra == "excl_amenity" and "amenity_flag" in sub.columns:
                sub = sub[~sub["amenity_flag"]].copy()
            elif extra == "excl_ev" and "ev_flag" in sub.columns:
                sub = sub[~sub["ev_flag"]].copy()
            elif extra == "excl_both":
                if "amenity_flag" in sub.columns:
                    sub = sub[~sub["amenity_flag"]]
                if "ev_flag" in sub.columns:
                    sub = sub[~sub["ev_flag"]].copy()

            label = f"{seg} | {fuel} | {view}" + (f" | {extra}" if extra != "none" else "")
            row = _sensitivity_row(sub, outcome, binary_col, label, seg, fuel, view, extra)
            rows.append(row)

    result_df = pd.DataFrame(rows)

    # Print
    print(f"\n{'=' * 100}")
    print("  HEATING SENSITIVITY MATRIX")
    print(f"{'=' * 100}")
    display_cols = [
        "outcome", "mf_segment", "fuel", "classification_view", "extra_filter",
        "n_central", "n_distributed", "median_central", "median_distributed",
        "delta_median", "ols_coef", "ols_p",
    ]
    display_cols = [c for c in display_cols if c in result_df.columns]
    print(result_df[display_cols].to_string(index=False))

    # Save CSV
    csv_path = args.outdir / "heating_sensitivity_matrix.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    # Save markdown table
    md_path = args.outdir / "heating_sensitivity_matrix.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Heating Sensitivity Matrix — RECS 2020 Multifamily\n\n")
        f.write("Each row shows a different analytical slice. "
                "`delta_median` = median Central − median Distributed. "
                "`ols_coef` = OLS coefficient on the Distributed indicator after "
                "controlling for climate, vintage, and log(unit area).\n\n")
        f.write(result_df[display_cols].to_markdown(index=False))
        f.write("\n")
    logger.info("Saved %s", md_path)


if __name__ == "__main__":
    main()
