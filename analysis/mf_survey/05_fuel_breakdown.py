"""Prompt 5 — Fuel-Type Breakdown.

For properties with classifiable DHW system types, breaks down energy
consumption by fuel:

  * Grid Electricity Site kBTU
  * Natural Gas Site kBTU
  * District Steam Site kBTU
  (plus Propane, Fuel Oil #1, Fuel Oil #2 when present)

Outputs:
  - Stacked bar charts of fuel mix (% of total site energy) by DHW system type
  - Absolute median kBTU stacked bars
  - Summary CSV

Usage::

    python analysis/mf_survey/05_fuel_breakdown.py \\
        --data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        [--outdir outputs/mf_survey]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.mf_survey.classify import add_system_classifications, to_binary
from src.common.log import get_logger

logger = get_logger("mf_survey.05_fuel")

FUEL_COLS = [
    "Grid Electricity Site kBTU",
    "Natural Gas Site kBTU",
    "District Steam Site kBTU",
    "District Hot Water Site kBTU",
    "Propane Site kBTU",
    "Fuel Oil #1 Site kBTU",
    "Fuel Oil #2 Site kBTU",
]

# Short display names for plot labels
FUEL_LABELS = {
    "Grid Electricity Site kBTU": "Electricity",
    "Natural Gas Site kBTU": "Natural Gas",
    "District Steam Site kBTU": "District Steam",
    "District Hot Water Site kBTU": "District HW",
    "Propane Site kBTU": "Propane",
    "Fuel Oil #1 Site kBTU": "Fuel Oil #1",
    "Fuel Oil #2 Site kBTU": "Fuel Oil #2",
}

FUEL_COLORS = [
    "#4CAF50",  # Electricity — green
    "#FF9800",  # Natural Gas — orange
    "#9C27B0",  # District Steam — purple
    "#2196F3",  # District HW — blue
    "#795548",  # Propane — brown
    "#607D8B",  # Fuel Oil #1 — blue-grey
    "#455A64",  # Fuel Oil #2 — dark blue-grey
]


def _present_fuels(df: pd.DataFrame) -> list[str]:
    """Return fuel columns that exist in df and have at least one non-null value."""
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
    # Replace NaN with 0 for stacking
    data = np.nan_to_num(data, nan=0.0)

    if normalize:
        row_totals = data.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1  # avoid /0
        data = 100.0 * data / row_totals

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(groups)), 5))
    bottoms = np.zeros(len(groups))

    for j, (col, label, color) in enumerate(zip(fuel_cols, labels, colors)):
        vals = data[:, j]
        ax.bar(groups, vals, bottom=bottoms, label=label, color=color, edgecolor="white", width=0.5)
        bottoms += vals

    # Sample-size annotations
    for i, grp in enumerate(groups):
        n = group_counts.get(grp, 0)
        ax.text(i, bottoms[i] + 0.5, f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, title="Fuel Type", bbox_to_anchor=(1.3, 1.0))
    ax.text(
        0.5, -0.12,
        "⚠ Median values; small samples — interpret with caution",
        transform=ax.transAxes,
        ha="center", fontsize=7, color="red",
    )

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def fuel_summary_table(df: pd.DataFrame, system_col: str, fuel_cols: list[str]) -> pd.DataFrame:
    """Return per-group median fuel kBTU and share (%)."""
    binary = to_binary(df[system_col])
    df2 = df.copy()
    df2["_binary"] = binary
    sub = df2[df2["_binary"].notna()]

    rows = []
    for grp, gdf in sub.groupby("_binary"):
        row = {"system_type": grp, "n": len(gdf)}
        total = 0.0
        for fc in fuel_cols:
            med = gdf[fc].median() if fc in gdf.columns else 0.0
            row[f"median_{FUEL_LABELS.get(fc, fc)}"] = round(med, 1) if pd.notna(med) else 0.0
            total += row[f"median_{FUEL_LABELS.get(fc, fc)}"]
        row["total_median_kBTU"] = round(total, 1)
        for fc in fuel_cols:
            key = f"share_{FUEL_LABELS.get(fc, fc)}"
            med = row[f"median_{FUEL_LABELS.get(fc, fc)}"]
            row[key] = round(100 * med / total, 1) if total > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).set_index("system_type")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuel-type breakdown for 2023 MF Survey.")
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/mf_survey"),
                    help="Directory for output files.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data …")
    df = load_energy_subset(load_mf_survey(args.data))
    df = add_system_classifications(df)

    fuel_cols = _present_fuels(df)
    logger.info("Fuel columns present: %s", fuel_cols)

    # Focus on DHW system type (as specified in Prompt 5)
    # Also run for heating and cooling for completeness
    for sys_col in ["dhw_system_type", "heating_system_type", "cooling_system_type"]:
        binary = to_binary(df[sys_col])
        df2 = df.copy()
        df2["_binary"] = binary
        sub = df2[df2["_binary"].notna()].copy()

        if sub.empty:
            logger.warning("No classifiable rows for %s — skipping", sys_col)
            continue

        group_counts = sub["_binary"].value_counts().to_dict()
        group_medians = sub.groupby("_binary")[fuel_cols].median()

        print(f"\n{'='*70}")
        print(f"  Fuel breakdown — {sys_col}")
        print(f"{'='*70}")

        tbl = fuel_summary_table(df, sys_col, fuel_cols)
        print(tbl.to_string())

        # Save table
        tbl.to_csv(args.outdir / f"05_fuel_summary_{sys_col}.csv")

        # Absolute stacked bar
        make_stacked_bar(
            group_medians, group_counts, fuel_cols,
            title=f"Median Site Energy by Fuel — {sys_col}",
            ylabel="Median Site kBTU",
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

    logger.info("Done.")


if __name__ == "__main__":
    main()
