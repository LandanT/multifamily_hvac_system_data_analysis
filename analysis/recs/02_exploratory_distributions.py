"""RECS 2020 — Step 2: Exploratory distributions.

Produces:
  1. Box plots of Site_EUI_kBtu_sqft and Electric_EUI_kBtu_sqft by
     heating_system_type, cooling_system_type, and dhw_system_type.
  2. Confounding check: IECC_climate_code and YEARMADERANGE stacked bars
     within each system-type category.
  3. Cross-tab of heating_system_type × cooling_system_type.
  4. Summary stats (n, median, IQR, mean, std) for each group — saved as CSV.

Input is the pre-built curated parquet (output of 01_build_curated_table.py).

Usage::

    python analysis/recs/02_exploratory_distributions.py \\
        --curated outputs/recs/recs2020_curated_*.parquet \\
        --outdir outputs/recs
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

from src.common.log import get_logger
from src.datasets.recs.utils import load_curated, filter_unit_type

logger = get_logger("recs.02_explore")

EUI_COLS = ["Site_EUI_kBtu_sqft", "Electric_EUI_kBtu_sqft", "Gas_EUI_kBtu_sqft"]
SYSTEM_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]


def make_boxplots(df: pd.DataFrame, system_col: str, eui_cols: list[str], outdir: Path) -> None:
    """Save one figure with one subplot per EUI metric."""
    present = [c for c in eui_cols if c in df.columns and df[c].notna().any()]
    if not present:
        logger.warning("No EUI columns found for boxplot — skipping %s", system_col)
        return

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5))
    if len(present) == 1:
        axes = [axes]

    groups_raw = [g for g in df[system_col].dropna().unique() if g != "Unknown"]
    if "Unknown" in df[system_col].unique():
        groups_raw.append("Unknown")

    for ax, eui_col in zip(axes, present):
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
        ax.set_ylabel("kBtu/sqft/yr")
        ax.set_xticklabels(groups_raw, rotation=20, ha="right", fontsize=8)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        ylim = ax.get_ylim()
        for i, grp in enumerate(groups_raw):
            n = len(data_by_group[grp])
            ax.text(
                i + 1, ylim[0] + 0.01 * (ylim[1] - ylim[0]),
                f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray",
            )

    fig.suptitle(f"EUI Distribution by {system_col}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.text(
        0.5, 0.01,
        "⚠ Interpret with caution — check sample sizes",
        ha="center", fontsize=7, color="red",
    )

    fname = outdir / f"02_boxplot_{system_col}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def make_confounding_charts(df: pd.DataFrame, system_col: str, outdir: Path) -> None:
    """Stacked bars showing climate zone and vintage distribution within each system type."""
    confound_cols = [
        ("IECC_climate_code", "climate_zone"),
        ("YEARMADERANGE", "vintage"),
    ]
    for cat_col, label in confound_cols:
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
    present = [c for c in eui_cols if c in df.columns]
    print(f"\n{'=' * 60}")
    print(f"Summary stats — {system_col}")
    print(f"{'=' * 60}")

    _SHORT = {
        "Site_EUI_kBtu_sqft": "SiteEUI",
        "Electric_EUI_kBtu_sqft": "ElecEUI",
        "Gas_EUI_kBtu_sqft": "GasEUI",
    }

    rows = []
    for grp, gdf in df.groupby(system_col):
        row = {"system_type": grp}
        for col in present:
            vals = gdf[col].dropna()
            short = _SHORT.get(col, col)
            row[f"n_{short}"] = len(vals)
            row[f"median_{short}"] = round(vals.median(), 1) if len(vals) else None
            row[f"IQR_{short}"] = (
                round(vals.quantile(0.75) - vals.quantile(0.25), 1) if len(vals) else None
            )
            row[f"mean_{short}"] = round(vals.mean(), 1) if len(vals) else None
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("system_type")
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory distributions for RECS 2020.")
    ap.add_argument(
        "--curated", type=Path, required=True,
        help="Path to recs2020_curated_*.parquet (output of 01_build_curated_table.py).",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/recs"),
        help="Directory for output figures and CSVs (default: outputs/recs).",
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

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading curated data from %s", args.curated)
    df = load_curated(args.curated)
    logger.info("Loaded %d rows", len(df))

    df = filter_unit_type(df, args.unit_type)
    logger.info("After unit-type filter (%s): %d rows", args.unit_type, len(df))

    # Filter to rows with at least one EUI value
    has_eui = df[EUI_COLS].notna().any(axis=1)
    df = df[has_eui].copy()
    logger.info("Rows with at least one EUI value: %d", len(df))

    # ---- 1. Box plots ------------------------------------------------
    logger.info("Generating box plots …")
    for sys_col in SYSTEM_COLS:
        if sys_col in df.columns:
            make_boxplots(df, sys_col, EUI_COLS, args.outdir)

    # ---- 2. Confounding checks ----------------------------------------
    logger.info("Generating confounding charts …")
    for sys_col in SYSTEM_COLS:
        if sys_col in df.columns:
            make_confounding_charts(df, sys_col, args.outdir)

    # ---- 3. Cross-tab heating × cooling ------------------------------
    if "heating_system_type" in df.columns and "cooling_system_type" in df.columns:
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
        if sys_col in df.columns:
            print_summary_stats(df, sys_col, EUI_COLS)

    # Save summary CSV
    rows = []
    for sys_col in SYSTEM_COLS:
        if sys_col not in df.columns:
            continue
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
