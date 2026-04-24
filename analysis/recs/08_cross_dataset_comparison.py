"""RECS vs RBSA — Side-by-side comparison (presentation-ready).

Reads the pre-computed CSV outputs from each pipeline and produces:
  1. Forest plot — OLS coefficients from both datasets on the same axes.
  2. Grouped bars — median EUI (Central vs Distributed) by dataset × system type.
  3. Summary table CSV with all comparable metrics.

Does NOT merge raw microdata — only compares aggregated results.

Usage::

    python analysis/recs/08_cross_dataset_comparison.py \\
        --recs-ols outputs/recs/04_ols_results.csv \\
        --recs-mw  outputs/recs/03_mann_whitney_results.csv \\
        --rbsa-ols outputs/rbsa/04_ols_results.csv \\
        --rbsa-mw  outputs/rbsa/03_mann_whitney_results.csv \\
        --outdir   outputs/recs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.log import get_logger

logger = get_logger("recs.08_compare")

C_RECS = "#2196F3"
C_RBSA = "#FF7043"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalise_system_col(s: str) -> str:
    """Map varied system_col names to a short canonical label."""
    s = s.lower().replace("_binary", "")
    if "heat" in s and "dhw" not in s and "water" not in s:
        return "Heating"
    if "cool" in s:
        return "Cooling"
    if "dhw" in s or "water" in s:
        return "DHW"
    return s


# ---------------------------------------------------------------------------
# 1. Forest plot — OLS coefficients, both datasets
# ---------------------------------------------------------------------------

def plot_forest(recs_ols: pd.DataFrame, rbsa_ols: pd.DataFrame, outdir: Path) -> None:
    """Forest plot: OLS distributed coefficient ± 95% CI, RECS vs RBSA."""
    rows = []

    for src_label, src_df, color in [("RECS 2020", recs_ols, C_RECS), ("RBSA 2022", rbsa_ols, C_RBSA)]:
        for _, r in src_df.iterrows():
            sys_label = _normalise_system_col(str(r.get("system_col", "")))
            coef = r.get("coef_distributed")
            ci_lo = r.get("ci_low")
            ci_hi = r.get("ci_high")
            pval = r.get("p_distributed")
            n = r.get("n")
            if pd.isna(coef):
                continue
            rows.append({
                "label": f"{sys_label} — {src_label}",
                "dataset": src_label,
                "system": sys_label,
                "coef": float(coef),
                "ci_lo": float(ci_lo) if not pd.isna(ci_lo) else float(coef),
                "ci_hi": float(ci_hi) if not pd.isna(ci_hi) else float(coef),
                "pval": float(pval) if not pd.isna(pval) else 1.0,
                "n": int(n) if not pd.isna(n) else 0,
                "color": color,
            })

    if not rows:
        logger.warning("No OLS rows to plot — skipping forest plot")
        return

    # Drop extreme outlier rows (e.g. RBSA DHW with CI spanning hundreds)
    rows = [r for r in rows if abs(r["coef"]) < 100]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(rows))))
    y_pos = list(range(len(rows)))

    for i, r in enumerate(rows):
        marker_color = r["color"] if r["pval"] < 0.05 else "#BDBDBD"
        ax.errorbar(
            r["coef"], i,
            xerr=[[r["coef"] - r["ci_lo"]], [r["ci_hi"] - r["coef"]]],
            fmt="o", color=marker_color, ecolor=marker_color,
            elinewidth=2, capsize=4, markersize=8,
        )
        sig = "**" if r["pval"] < 0.01 else ("*" if r["pval"] < 0.05 else "")
        ax.text(
            r["ci_hi"] + 1.0, i,
            f'  {r["coef"]:+.1f}  (p={r["pval"]:.3f}){sig}   n={r["n"]}',
            va="center", fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_xlabel("Distributed − Central  (kBtu/sqft/yr)", fontsize=10)
    ax.set_title(
        "OLS coefficient: Distributed system type effect\n"
        "RECS 2020 (national) vs RBSA 2022 (Pacific NW)",
        fontsize=12, fontweight="bold",
    )
    ax.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_RECS, markersize=8, label="RECS 2020"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_RBSA, markersize=8, label="RBSA 2022"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#BDBDBD", markersize=8, label="Not significant"),
    ]
    ax.legend(handles=handles, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    fig.tight_layout()
    fname = outdir / "08_forest_recs_vs_rbsa.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 2. Grouped bars — median EUI by dataset × system type
# ---------------------------------------------------------------------------

def plot_median_bars(recs_mw: pd.DataFrame, rbsa_mw: pd.DataFrame, outdir: Path) -> None:
    """Side-by-side bars: median Central vs Distributed EUI per dataset."""
    # Filter to Site_EUI, "All" stratum, heating only (most comparable)
    datasets = []
    for src_label, mw_df, color in [("RECS", recs_mw, C_RECS), ("RBSA", rbsa_mw, C_RBSA)]:
        for _, r in mw_df.iterrows():
            metric = str(r.get("metric", ""))
            stratum = str(r.get("fuel_stratum", "All"))
            sys_label = _normalise_system_col(str(r.get("system_col", "")))
            if "Site_EUI" not in metric or stratum != "All":
                continue
            mc = r.get("median_central")
            md = r.get("median_distributed")
            if pd.isna(mc) or pd.isna(md):
                continue
            datasets.append({
                "dataset": src_label,
                "system": sys_label,
                "median_central": float(mc),
                "median_distributed": float(md),
                "n_central": int(r.get("n_central", 0)),
                "n_distributed": int(r.get("n_distributed", 0)),
                "p_value": float(r.get("p_value")) if not pd.isna(r.get("p_value")) else None,
            })

    if not datasets:
        return

    df = pd.DataFrame(datasets)
    systems = [s for s in ["Heating", "Cooling", "DHW"] if s in df["system"].values]

    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 5), sharey=False)
    if len(systems) == 1:
        axes = [axes]

    for ax, sys_type in zip(axes, systems):
        sub = df[df["system"] == sys_type].sort_values("dataset")
        if sub.empty:
            continue

        x = np.arange(len(sub))
        width = 0.32

        bars_c = ax.bar(x - width / 2, sub["median_central"].values, width,
                        label="Central", color="#2196F3", edgecolor="white")
        bars_d = ax.bar(x + width / 2, sub["median_distributed"].values, width,
                        label="Distributed", color="#FF7043", edgecolor="white")

        # Annotate n= and p
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(x[i] - width / 2, row["median_central"] - 1.5,
                    f'n={row["n_central"]}', ha="center", va="top", fontsize=7, color="white", fontweight="bold")
            ax.text(x[i] + width / 2, row["median_distributed"] - 1.5,
                    f'n={row["n_distributed"]}', ha="center", va="top", fontsize=7, color="white", fontweight="bold")

            if row["p_value"] is not None:
                y_max = max(row["median_central"], row["median_distributed"])
                if row["p_value"] < 0.001:
                    sig = "***"
                elif row["p_value"] < 0.01:
                    sig = "**"
                elif row["p_value"] < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."
                ax.text(x[i], y_max + 3, sig, ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"].values, fontsize=11)
        ax.set_ylabel("Median Site EUI (kBtu/sqft/yr)", fontsize=10)
        ax.set_title(f"{sys_type} System Type", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    fig.suptitle(
        "Median Site EUI: Central vs Distributed\nRECS 2020 (national) vs RBSA 2022 (Pacific NW)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fname = outdir / "08_median_bars_recs_vs_rbsa.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 3. Summary table
# ---------------------------------------------------------------------------

def build_summary_table(
    recs_ols: pd.DataFrame, rbsa_ols: pd.DataFrame,
    recs_mw: pd.DataFrame, rbsa_mw: pd.DataFrame,
    outdir: Path,
) -> None:
    """Combined comparison CSV with key metrics from both datasets."""
    rows = []
    for src_label, ols_df, mw_df in [("RECS 2020", recs_ols, recs_mw), ("RBSA 2022", rbsa_ols, rbsa_mw)]:
        for _, r in ols_df.iterrows():
            sys_label = _normalise_system_col(str(r.get("system_col", "")))
            row = {
                "dataset": src_label,
                "system": sys_label,
                "ols_n": r.get("n"),
                "ols_coef": r.get("coef_distributed"),
                "ols_p": r.get("p_distributed"),
                "ols_ci_low": r.get("ci_low"),
                "ols_ci_high": r.get("ci_high"),
                "ols_R2": r.get("R2"),
                "ols_significant": r.get("significant"),
            }

            # Find matching MannWhitney row (Site_EUI, All stratum)
            mw_match = mw_df[
                (mw_df["system_col"].apply(_normalise_system_col) == sys_label)
                & mw_df["metric"].str.contains("Site_EUI", na=False)
                & (mw_df["fuel_stratum"] == "All")
            ]
            if len(mw_match):
                m = mw_match.iloc[0]
                row["mw_n_central"] = m.get("n_central")
                row["mw_n_distributed"] = m.get("n_distributed")
                row["mw_median_central"] = m.get("median_central")
                row["mw_median_distributed"] = m.get("median_distributed")
                row["mw_p"] = m.get("p_value")
                row["mw_r_rb"] = m.get("rank_biserial_r")
                row["mw_significant"] = m.get("significant")

            rows.append(row)

    out = pd.DataFrame(rows)
    fname = outdir / "08_recs_vs_rbsa_summary.csv"
    out.to_csv(fname, index=False)
    logger.info("Saved %s", fname)

    # Print to console
    print(f"\n{'=' * 80}")
    print("  RECS 2020 vs RBSA 2022 — Summary")
    print(f"{'=' * 80}")
    print(out.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RECS vs RBSA side-by-side comparison.")
    ap.add_argument("--recs-ols", type=Path, required=True, help="RECS 04_ols_results.csv")
    ap.add_argument("--recs-mw", type=Path, required=True, help="RECS 03_mann_whitney_results.csv")
    ap.add_argument("--rbsa-ols", type=Path, required=True, help="RBSA 04_ols_results.csv")
    ap.add_argument("--rbsa-mw", type=Path, required=True, help="RBSA 03_mann_whitney_results.csv")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/recs"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    recs_ols = _load(args.recs_ols)
    recs_mw = _load(args.recs_mw)
    rbsa_ols = _load(args.rbsa_ols)
    rbsa_mw = _load(args.rbsa_mw)

    plot_forest(recs_ols, rbsa_ols, args.outdir)
    plot_median_bars(recs_mw, rbsa_mw, args.outdir)
    build_summary_table(recs_ols, rbsa_ols, recs_mw, rbsa_mw, args.outdir)

    logger.info("Done — all cross-dataset outputs saved to %s", args.outdir)


if __name__ == "__main__":
    main()
