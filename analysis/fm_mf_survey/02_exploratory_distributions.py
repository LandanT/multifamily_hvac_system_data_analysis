"""Prompt 2 — Exploratory Distributions.

Produces:
  1. Box plots of Site EUI and Source EUI by dhw_system_type,
     heating_system_type, and cooling_system_type.
  2. Distribution of building types and regions within each system-type
     category (confounding check).
  3. Cross-tab of heating_system_type × cooling_system_type.
  4. Summary stats (n, median, IQR, mean) for each group.

All charts include sample-size annotations because n is small (~40–100
per category).

Usage::

    python analysis/fm_mf_survey/02_exploratory_distributions.py \\
        --data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        --outdir outputs/fm_mf_survey
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.fm_mf_survey.classify import add_system_classifications
from src.common.log import get_logger

logger = get_logger("fm_mf_survey.02_explore")

# EUI columns used throughout
EUI_COLS = ["Site EUI (kBTU/sqft/yr)", "Source EUI (kBTU/sqft/yr)"]
SYSTEM_COLS = ["dhw_system_type", "heating_system_type", "cooling_system_type"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_n_labels(ax, groups: list[str], data_by_group: dict[str, pd.Series]) -> None:
    """Annotate each box with sample size."""
    for i, grp in enumerate(groups):
        n = len(data_by_group.get(grp, []))
        ax.text(
            i + 1, ax.get_ylim()[0],
            f"n={n}",
            ha="center", va="bottom", fontsize=8, color="dimgray",
        )


def make_boxplots(df: pd.DataFrame, system_col: str, eui_cols: list[str], outdir: Path) -> None:
    """Save one figure with subplots for each EUI metric."""
    fig, axes = plt.subplots(1, len(eui_cols), figsize=(6 * len(eui_cols), 5))
    if len(eui_cols) == 1:
        axes = [axes]

    # Order groups by median for readability (Unknown last)
    groups_raw = [g for g in df[system_col].unique() if g != "Unknown"] + (
        ["Unknown"] if "Unknown" in df[system_col].unique() else []
    )

    for ax, eui_col in zip(axes, eui_cols):
        if eui_col not in df.columns:
            ax.set_visible(False)
            continue

        data_by_group = {
            g: df.loc[df[system_col] == g, eui_col].dropna()
            for g in groups_raw
        }
        plot_data = [data_by_group[g].values for g in groups_raw]

        bp = ax.boxplot(plot_data, labels=groups_raw, patch_artist=True, notch=False)

        colors = plt.cm.Set2(np.linspace(0, 1, len(groups_raw)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{eui_col}\nby {system_col}", fontsize=10)
        ax.set_ylabel("kBTU/sqft/yr")
        ax.set_xticklabels(groups_raw, rotation=20, ha="right", fontsize=8)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        # Sample-size annotations below each box
        ylim = ax.get_ylim()
        for i, grp in enumerate(groups_raw):
            n = len(data_by_group[grp])
            ax.text(
                i + 1, ylim[0] - 0.03 * (ylim[1] - ylim[0]),
                f"n={n}",
                ha="center", va="top", fontsize=8, color="dimgray",
            )

        ax.text(
            0.5, 1.01,
            "⚠ Small samples — interpret with caution",
            transform=ax.transAxes,
            ha="center", fontsize=7, color="red",
        )

    fig.suptitle(f"EUI Distribution by {system_col}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    fname = outdir / f"02_boxplot_{system_col}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def make_confounding_charts(df: pd.DataFrame, system_col: str, outdir: Path) -> None:
    """Stacked bar of Region and Building Type by system type."""
    for cat_col, label in [("Region", "region"), ("Building Type #1", "building_type")]:
        if cat_col not in df.columns:
            continue

        sub = df[[system_col, cat_col]].dropna()
        if sub.empty:
            continue

        xtab = pd.crosstab(sub[system_col], sub[cat_col], normalize="index")
        if xtab.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        xtab.plot(kind="bar", stacked=True, ax=ax, colormap="Set3", edgecolor="white")

        # Sample-size labels on bars
        counts = sub[system_col].value_counts()
        for i, grp in enumerate(xtab.index):
            n = counts.get(grp, 0)
            ax.text(i, 1.02, f"n={n}", ha="center", fontsize=8, color="dimgray")

        ax.set_title(f"{cat_col} distribution within {system_col}", fontsize=10)
        ax.set_ylabel("Proportion")
        ax.set_xticklabels(xtab.index, rotation=20, ha="right", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, title=cat_col)
        fig.tight_layout()

        fname = outdir / f"02_confounding_{system_col}_{label}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)


def print_summary_stats(df: pd.DataFrame, system_col: str, eui_cols: list[str]) -> None:
    """Print n, median, IQR, mean for each group."""
    print(f"\n{'='*60}")
    print(f"Summary stats — {system_col}")
    print(f"{'='*60}")

    rows = []
    for grp, gdf in df.groupby(system_col):
        row = {"system_type": grp}
        for col in eui_cols:
            if col in gdf.columns:
                vals = gdf[col].dropna()
                row[f"n_{col[:8]}"] = len(vals)
                row[f"median_{col[:8]}"] = round(vals.median(), 1) if len(vals) else None
                row[f"IQR_{col[:8]}"] = (
                    round(vals.quantile(0.75) - vals.quantile(0.25), 1) if len(vals) else None
                )
                row[f"mean_{col[:8]}"] = round(vals.mean(), 1) if len(vals) else None
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("system_type")
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory distributions for 2023 MF Survey.")
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/fm_mf_survey"),
                    help="Directory for output figures and CSVs.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", args.data)
    df = load_energy_subset(load_mf_survey(args.data))
    df = add_system_classifications(df)
    logger.info("Energy subset: %d rows", len(df))

    # ---- 1. Box plots ------------------------------------------------
    logger.info("Generating box plots …")
    for sys_col in SYSTEM_COLS:
        make_boxplots(df, sys_col, EUI_COLS, args.outdir)

    # ---- 2. Confounding checks ----------------------------------------
    logger.info("Generating confounding charts …")
    for sys_col in SYSTEM_COLS:
        make_confounding_charts(df, sys_col, args.outdir)

    # ---- 3. Cross-tab heating × cooling ------------------------------
    print("\n--- Cross-tab: heating_system_type × cooling_system_type ---")
    xtab = pd.crosstab(
        df["heating_system_type"],
        df["cooling_system_type"],
        margins=True,
        margins_name="Total",
    )
    print(xtab.to_string())
    xtab.to_csv(args.outdir / "02_xtab_heating_x_cooling.csv")

    # ---- 4. Summary stats -------------------------------------------
    for sys_col in SYSTEM_COLS:
        print_summary_stats(df, sys_col, EUI_COLS)

    # Also save full summary to CSV
    rows = []
    for sys_col in SYSTEM_COLS:
        for grp, gdf in df.groupby(sys_col):
            for col in EUI_COLS:
                if col not in gdf.columns:
                    continue
                vals = gdf[col].dropna()
                rows.append({
                    "system_col": sys_col,
                    "group": grp,
                    "eui_metric": col,
                    "n": len(vals),
                    "median": round(vals.median(), 2) if len(vals) else None,
                    "q25": round(vals.quantile(0.25), 2) if len(vals) else None,
                    "q75": round(vals.quantile(0.75), 2) if len(vals) else None,
                    "iqr": round(vals.quantile(0.75) - vals.quantile(0.25), 2) if len(vals) else None,
                    "mean": round(vals.mean(), 2) if len(vals) else None,
                    "std": round(vals.std(), 2) if len(vals) else None,
                })

    pd.DataFrame(rows).to_csv(args.outdir / "02_summary_stats.csv", index=False)
    logger.info("Saved summary stats CSV.")


if __name__ == "__main__":
    main()
