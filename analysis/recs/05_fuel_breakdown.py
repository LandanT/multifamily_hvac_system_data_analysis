"""RECS 2020 — Step 5: Fuel-type breakdown.

Shows the share of households using each fuel type (electricity, natural gas,
propane, fuel oil) for heating and water heating within each system class.

Unlike FM/RBSA (which use fuel-specific kBtu columns), RECS supplies binary
fuel-use flags.  This script counts households and computes percentage shares
per fuel per system group.

Heating fuel flags  : ELWARM, UGWARM, LPWARM, FOWARM
Water-heating flags : ELWATER, UGWATER, LPWATER, FOWATER

Outputs per system type:
  1. Stacked bar chart — % of households by fuel type (heating)
  2. Stacked bar chart — % of households by fuel type (water heating)
  3. Summary CSV with counts and proportions

Usage::

    python analysis/recs/05_fuel_breakdown.py \\
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
import numpy as np
import pandas as pd

from src.common.log import get_logger

logger = get_logger("recs.05_fuel")

# Heating fuel flags and display labels
HEATING_FUEL_FLAGS = ["ELWARM", "UGWARM", "LPWARM", "FOWARM"]
HEATING_FUEL_LABELS = {
    "ELWARM": "Electricity",
    "UGWARM": "Natural Gas",
    "LPWARM": "Propane",
    "FOWARM": "Fuel Oil",
}

# Water heating fuel flags and display labels
DHW_FUEL_FLAGS = ["ELWATER", "UGWATER", "LPWATER", "FOWATER"]
DHW_FUEL_LABELS = {
    "ELWATER": "Electricity",
    "UGWATER": "Natural Gas",
    "LPWATER": "Propane",
    "FOWATER": "Fuel Oil",
}

# Consistent colors (matching FM/RBSA palette)
FUEL_COLORS = [
    "#4CAF50",  # Electricity — green
    "#FF9800",  # Natural Gas — orange
    "#795548",  # Propane — brown
    "#607D8B",  # Fuel Oil — blue-grey
]

SYSTEM_COLS = ["heating_system_type", "cooling_system_type", "dhw_system_type"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_curated(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )


def _filter_unit_type(df: pd.DataFrame, unit_type: str) -> pd.DataFrame:
    if "TYPEHUQ" not in df.columns or unit_type == "all":
        return df
    if unit_type == "mf":
        return df[df["TYPEHUQ"].isin([3, 4])].copy()
    if unit_type == "sf":
        return df[df["TYPEHUQ"].isin([1, 2])].copy()
    return df


def fuel_share_table(
    df: pd.DataFrame,
    system_col: str,
    fuel_flags: list[str],
    fuel_labels: dict[str, str],
) -> pd.DataFrame:
    """Compute household counts and % share per fuel flag per system group.

    Parameters
    ----------
    df:
        Curated DataFrame (non-Unknown rows only, passed in from caller).
    system_col:
        Classification column (e.g., ``'heating_system_type'``).
    fuel_flags:
        List of binary flag columns (e.g., ``['ELWARM', 'UGWARM', ...]``).
    fuel_labels:
        Mapping from flag name to display label.

    Returns
    -------
    pd.DataFrame
        Rows: system groups; columns: n, count_<fuel>, pct_<fuel>.
    """
    present_flags = [f for f in fuel_flags if f in df.columns]
    rows = []
    for grp, gdf in df.groupby(system_col):
        row: dict = {"system_type": grp, "n": len(gdf)}
        for flag in present_flags:
            label = fuel_labels[flag]
            count = int((gdf[flag] == 1).sum())
            pct = round(100.0 * count / len(gdf), 1) if len(gdf) else 0.0
            row[f"count_{label}"] = count
            row[f"pct_{label}"] = pct
        rows.append(row)

    return pd.DataFrame(rows).set_index("system_type")


def make_fuel_share_bar(
    tbl: pd.DataFrame,
    fuel_flags: list[str],
    fuel_labels: dict[str, str],
    title: str,
    fname: Path,
) -> None:
    """Stacked bar chart: % share of households per fuel type per system group."""
    present_flags = [f for f in fuel_flags if f"pct_{fuel_labels[f]}" in tbl.columns]
    if not present_flags:
        return

    groups = tbl.index.tolist()
    labels = [fuel_labels[f] for f in present_flags]
    colors = FUEL_COLORS[: len(present_flags)]

    data = np.array([
        tbl[f"pct_{fuel_labels[f]}"].fillna(0).values for f in present_flags
    ]).T  # shape: (n_groups, n_fuels)

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(groups)), 5))
    bottoms = np.zeros(len(groups))

    for j, (flag, label, color) in enumerate(zip(present_flags, labels, colors)):
        vals = data[:, j]
        ax.bar(
            groups, vals, bottom=bottoms, label=label,
            color=color, edgecolor="white", width=0.5,
        )
        bottoms += vals

    # Annotate n= above each bar
    for i, grp in enumerate(groups):
        n = tbl.loc[grp, "n"] if "n" in tbl.columns else 0
        ax.text(i, bottoms[i] + 0.5, f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("% of households")
    ax.set_ylim(0, max(110, bottoms.max() + 10))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, title="Fuel Type", bbox_to_anchor=(1.3, 1.0))
    ax.text(
        0.5, -0.12,
        "⚠ Households may use more than one fuel type (flags are not mutually exclusive)",
        transform=ax.transAxes,
        ha="center", fontsize=7, color="gray",
    )

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuel-type breakdown for RECS 2020.")
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
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading curated data from %s", args.curated)
    df = _load_curated(args.curated)
    df = _filter_unit_type(df, args.unit_type)
    logger.info("Rows after unit-type filter (%s): %d", args.unit_type, len(df))

    for sys_col in SYSTEM_COLS:
        if sys_col not in df.columns:
            logger.warning("%s not found — skipping", sys_col)
            continue

        sub = df[df[sys_col] != "Unknown"].copy()
        if sub.empty:
            logger.warning("No classifiable rows for %s — skipping", sys_col)
            continue

        print(f"\n{'=' * 70}")
        print(f"  Fuel breakdown — {sys_col}")
        print(f"{'=' * 70}")

        # Heating fuel breakdown (for all system types — shows what fuel heats the home)
        heating_tbl = fuel_share_table(sub, sys_col, HEATING_FUEL_FLAGS, HEATING_FUEL_LABELS)
        print("\n  Heating fuel share (% of households):")
        print(heating_tbl.to_string())
        heating_tbl.to_csv(args.outdir / f"05_heating_fuel_summary_{sys_col}.csv")

        make_fuel_share_bar(
            heating_tbl,
            HEATING_FUEL_FLAGS,
            HEATING_FUEL_LABELS,
            title=f"Heating Fuel Mix — {sys_col}\n(% of households by system class)",
            fname=args.outdir / f"05_fuel_heating_{sys_col}.png",
        )

        # DHW fuel breakdown (where water-heating flags are present)
        dhw_present = [f for f in DHW_FUEL_FLAGS if f in sub.columns]
        if dhw_present:
            dhw_tbl = fuel_share_table(sub, sys_col, DHW_FUEL_FLAGS, DHW_FUEL_LABELS)
            print("\n  Water-heating fuel share (% of households):")
            print(dhw_tbl.to_string())
            dhw_tbl.to_csv(args.outdir / f"05_dhw_fuel_summary_{sys_col}.csv")

            make_fuel_share_bar(
                dhw_tbl,
                DHW_FUEL_FLAGS,
                DHW_FUEL_LABELS,
                title=f"Water-Heating Fuel Mix — {sys_col}\n(% of households by system class)",
                fname=args.outdir / f"05_fuel_dhw_{sys_col}.png",
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
