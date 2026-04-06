"""Prompt 5 (part 3) — Fuel-Type Breakdown.

Breaks down energy consumption by fuel type for each HVAC/DHW system type:

  - Annual Electric Usage (kBtu)   — available for ~1,033 sites
  - Annual Gas Usage (kBtu)        — available for ~446 sites
  - Combined (Annual Usage Total)  — available for ~836 sites (both fuels confirmed)

Two views per system type:
  1. Absolute median kBtu stacked bar [Electric | Gas]
  2. Normalised (% share) stacked bar
  3. Summary CSV

Also produces box plots of Electric kBtu (electric-primary subset) and
Gas kBtu (gas-primary subset) separately to avoid fuel confounding.

Usage::

    python analysis/rbsa/05_fuel_breakdown.py \\
        --site-master outputs/rbsa/rbsa_site_master_*.parquet \\
        --outdir outputs/rbsa
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.log import get_logger

logger = get_logger("rbsa.05_fuel")

FUEL_COLS = [
    "Annual Electric Usage (kBtu)",
    "Annual Gas Usage (kBtu)",
    "Total Delivered Fuel (kBtu)",
]

FUEL_LABELS = {
    "Annual Electric Usage (kBtu)": "Electricity",
    "Annual Gas Usage (kBtu)": "Natural Gas",
    "Total Delivered Fuel (kBtu)": "Delivered Fuel",
    "Electric_EUI_kBtu_sqft": "Electric EUI",
    "Gas_EUI_kBtu_sqft": "Gas EUI",
}

FUEL_COLORS = [
    "#4CAF50",  # Electricity — green
    "#FF9800",  # Natural Gas — orange
    "#795548",  # Delivered Fuel — brown
]

SYSTEM_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_site_master(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def _present_fuels(df: pd.DataFrame) -> list[str]:
    return [c for c in FUEL_COLS if c in df.columns and df[c].notna().any()]


def make_stacked_bar(
    group_medians: pd.DataFrame,
    group_counts: dict[str, int],
    fuel_cols: list[str],
    title: str,
    ylabel: str,
    fname: Path,
    normalize: bool = False,
) -> None:
    """Save a stacked bar chart."""
    groups = group_medians.index.tolist()
    labels = [FUEL_LABELS.get(c, c) for c in fuel_cols]
    colors = [FUEL_COLORS[FUEL_COLS.index(c)] for c in fuel_cols if c in FUEL_COLS]

    data = group_medians[fuel_cols].values.astype(float)
    data = np.nan_to_num(data, nan=0.0)

    if normalize:
        row_totals = data.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1
        data = 100.0 * data / row_totals

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(groups)), 5))
    bottoms = np.zeros(len(groups))

    for j, (col, label, color) in enumerate(zip(fuel_cols, labels, colors)):
        vals = data[:, j]
        ax.bar(groups, vals, bottom=bottoms, label=label, color=color, edgecolor="white", width=0.5)
        bottoms += vals

    for i, grp in enumerate(groups):
        n = group_counts.get(grp, 0)
        ax.text(i, bottoms[i] + 0.5, f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, title="Fuel Type", bbox_to_anchor=(1.3, 1.0))

    fig.tight_layout()
    fig.text(
        0.5, 0.01,
        "⚠ Median values — check sample sizes before interpreting",
        ha="center", fontsize=7, color="red",
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def fuel_summary_table(df: pd.DataFrame, system_col: str, fuel_cols: list[str]) -> pd.DataFrame:
    """Return per-group median fuel kBtu and share (%)."""
    sub = df[df[system_col] != "Unknown"].copy()

    rows = []
    for grp, gdf in sub.groupby(system_col):
        row: dict = {"system_type": grp, "n": len(gdf)}
        total = 0.0
        for fc in fuel_cols:
            med = gdf[fc].median() if fc in gdf.columns else 0.0
            label = FUEL_LABELS.get(fc, fc)
            row[f"median_{label}"] = round(med, 1) if pd.notna(med) else 0.0
            total += row[f"median_{label}"]
        row["total_median_kBtu"] = round(total, 1)
        for fc in fuel_cols:
            label = FUEL_LABELS.get(fc, fc)
            med = row[f"median_{label}"]
            row[f"share_{label}"] = round(100 * med / total, 1) if total > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).set_index("system_type")


def make_fuel_boxplots(
    df: pd.DataFrame,
    system_col: str,
    fuel_col: str,
    fuel_stratum: str,
    outdir: Path,
) -> None:
    """Box plot of a single fuel metric by system type within a fuel stratum."""
    sub = df[df[system_col] != "Unknown"].copy()
    if fuel_col not in sub.columns or not sub[fuel_col].notna().any():
        return

    groups = [g for g in sub[system_col].unique() if g != "Unknown"]
    data_by_group = {g: sub.loc[sub[system_col] == g, fuel_col].dropna() for g in groups}

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        [data_by_group[g].values for g in groups],
        labels=groups,
        patch_artist=True,
    )
    colors = cm.Set2(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ylim = ax.get_ylim()
    for i, grp in enumerate(groups):
        n = len(data_by_group[grp])
        ax.text(i + 1, ylim[0] + 0.01 * (ylim[1] - ylim[0]),
                f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    label = FUEL_LABELS.get(fuel_col, fuel_col)
    ax.set_title(f"{label} by {system_col}\n[{fuel_stratum} homes]", fontsize=9)
    ax.set_ylabel("kBtu/sqft/yr")
    ax.set_xticklabels(groups, rotation=15, ha="right")
    fig.tight_layout()

    slug = fuel_stratum.replace(" ", "_").lower()
    fname = outdir / f"05_boxplot_{system_col}_{slug}_{label[:8]}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuel-type breakdown for RBSA site master.")
    ap.add_argument("--site-master", type=Path, required=True,
                    help="Path to rbsa_site_master_*.parquet (or .csv)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/rbsa"),
                    help="Directory for output files.")
    ap.add_argument(
        "--building-type",
        choices=["mf", "sf", "all"],
        default="mf",
        help="Building type filter: 'mf' = multifamily only (default), 'sf' = single-family only, 'all' = no filter.",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading site master from %s", args.site_master)
    df = _load_site_master(args.site_master)

    bt_col = next((c for c in ["Building_Type", "Building Type"] if c in df.columns), None)
    if args.building_type == "sf" and bt_col:
        df = df[df[bt_col].str.lower().str.contains("single", na=False)].copy()
        logger.info("SF-only filter: %d rows", len(df))
    elif args.building_type == "mf" and bt_col:
        df = df[df[bt_col].str.lower().str.contains("multi", na=False)].copy()
        logger.info("MF-only filter: %d rows", len(df))

    fuel_cols = _present_fuels(df)
    logger.info("Fuel columns present: %s", fuel_cols)

    fuel_col = "Primary_Heating_Fuel_Type"
    electric_sub = df[df[fuel_col] == "Electric"].copy() if fuel_col in df.columns else df.copy()
    gas_sub = df[df[fuel_col] == "Natural gas"].copy() if fuel_col in df.columns else df.copy()

    for sys_col in SYSTEM_COLS:
        if sys_col not in df.columns:
            logger.warning("%s not found — skipping", sys_col)
            continue

        sub = df[df[sys_col] != "Unknown"].copy()
        if sub.empty:
            logger.warning("No classifiable rows for %s — skipping", sys_col)
            continue

        group_counts = sub[sys_col].value_counts().to_dict()
        group_medians = sub.groupby(sys_col)[fuel_cols].median()

        print(f"\n{'='*70}")
        print(f"  Fuel breakdown — {sys_col}")
        print(f"{'='*70}")

        tbl = fuel_summary_table(df, sys_col, fuel_cols)
        print(tbl.to_string())
        tbl.to_csv(args.outdir / f"05_fuel_summary_{sys_col}.csv")

        # Absolute stacked bar
        make_stacked_bar(
            group_medians, group_counts, fuel_cols,
            title=f"Median Energy by Fuel — {sys_col}",
            ylabel="Median kBtu/yr",
            fname=args.outdir / f"05_fuel_absolute_{sys_col}.png",
            normalize=False,
        )

        # Normalised (% share) stacked bar
        make_stacked_bar(
            group_medians, group_counts, fuel_cols,
            title=f"Fuel Mix (% of total) — {sys_col}",
            ylabel="Share (%)",
            fname=args.outdir / f"05_fuel_share_{sys_col}.png",
            normalize=True,
        )

        # Per-fuel box plots within electric and gas subsets (EUI normalised by sqft)
        for stratum, sub_df, fuel_metric in [
            ("Electric", electric_sub, "Electric_EUI_kBtu_sqft"),
            ("Gas", gas_sub, "Gas_EUI_kBtu_sqft"),
        ]:
            if sys_col in sub_df.columns and fuel_metric in sub_df.columns:
                make_fuel_boxplots(sub_df, sys_col, fuel_metric, stratum, args.outdir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
