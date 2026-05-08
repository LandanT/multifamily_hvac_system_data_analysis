"""RECS 2020 — Step 2b: Sample balance and heating system prevalence.

Answers the critical question: In which multifamily segments and unit-size
ranges do central and distributed heating systems coexist in meaningful
numbers for comparison?

Produces:
  1. Prevalence tables: heating system type share by mf_segment, unit_size_bin,
     and the cross of both — for each classification view (explicit, inferred,
     pooled binary).
  2. Stacked bar charts of heating system type share.
  3. Heatmap of sample counts by mf_segment × unit_size_bin × system type.
  4. CSV summaries saved to --outdir.

Usage::

    python analysis/recs/02b_sample_balance_and_prevalence.py \\
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
from src.datasets.recs.utils import (
    load_curated,
    filter_unit_type,
    filter_classification_view,
    CLASSIFICATION_VIEWS,
)

logger = get_logger("recs.02b_prevalence")

C_CENTRAL = "#2196F3"
C_DISTRIBUTED = "#FF7043"
C_UNKNOWN = "#9E9E9E"

SYSTEM_COLORS = {
    "Central": C_CENTRAL,
    "Distributed": C_DISTRIBUTED,
    "Central (inferred)": "#90CAF9",
    "Distributed (inferred)": "#FFAB91",
    "Unknown": C_UNKNOWN,
}


# ---------------------------------------------------------------------------
# Prevalence tables
# ---------------------------------------------------------------------------

def prevalence_table(
    df: pd.DataFrame,
    group_col: str,
    system_col: str = "heating_system_type",
    weight_col: str | None = "NWEIGHT",
) -> pd.DataFrame:
    """Cross-tab of system type within each group, with counts and shares."""
    rows = []
    for grp, gdf in df.groupby(group_col, dropna=False):
        n_total = len(gdf)
        for sys_type in gdf[system_col].dropna().unique():
            n = int((gdf[system_col] == sys_type).sum())
            pct = round(100.0 * n / n_total, 1) if n_total else 0.0
            row = {
                group_col: grp,
                "system_type": sys_type,
                "n": n,
                "pct": pct,
            }
            # Weighted share if available
            if weight_col and weight_col in gdf.columns:
                w_total = gdf[weight_col].sum()
                w_n = gdf.loc[gdf[system_col] == sys_type, weight_col].sum()
                row["weighted_pct"] = round(100.0 * w_n / w_total, 1) if w_total > 0 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def prevalence_table_binary(
    df: pd.DataFrame,
    group_col: str,
    binary_col: str = "heating_system_type_binary",
    weight_col: str | None = "NWEIGHT",
) -> pd.DataFrame:
    """Prevalence table using the binary (Central/Distributed) classification."""
    classifiable = df[df[binary_col].isin(["Central", "Distributed"])].copy()
    rows = []
    for grp, gdf in classifiable.groupby(group_col, dropna=False):
        n_total = len(gdf)
        for label in ["Central", "Distributed"]:
            n = int((gdf[binary_col] == label).sum())
            pct = round(100.0 * n / n_total, 1) if n_total else 0.0
            row = {group_col: grp, "system_type": label, "n": n, "pct": pct}
            if weight_col and weight_col in gdf.columns:
                w_total = gdf[weight_col].sum()
                w_n = gdf.loc[gdf[binary_col] == label, weight_col].sum()
                row["weighted_pct"] = round(100.0 * w_n / w_total, 1) if w_total > 0 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stacked bar chart
# ---------------------------------------------------------------------------

def plot_stacked_bar(
    tbl: pd.DataFrame,
    group_col: str,
    title: str,
    fname: Path,
    use_weighted: bool = False,
) -> None:
    """Stacked horizontal bar of system type share per group."""
    pct_col = "weighted_pct" if (use_weighted and "weighted_pct" in tbl.columns) else "pct"

    pivot = tbl.pivot_table(
        index=group_col, columns="system_type", values=pct_col, fill_value=0,
    )
    # Ensure consistent ordering
    ordered_cols = [c for c in ["Central", "Distributed", "Central (inferred)",
                                "Distributed (inferred)", "Unknown"] if c in pivot.columns]
    pivot = pivot[ordered_cols]

    colors = [SYSTEM_COLORS.get(c, "#BDBDBD") for c in ordered_cols]

    fig, ax = plt.subplots(figsize=(9.2, 5))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor="white")

    # Annotate counts
    n_lookup = tbl.groupby(group_col)["n"].sum()
    for i, grp in enumerate(pivot.index):
        total_n = n_lookup.get(grp, 0)
        ax.text(102, i, f"n={total_n}", va="center", fontsize=8, color="dimgray")

    ax.set_xlabel("% of households")
    ax.set_xlim(0, 115)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, title="System Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------

def plot_count_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Heatmap of sample counts by mf_segment × unit_size_bin for Central vs Distributed."""
    bcol = "heating_system_type_binary"
    if bcol not in df.columns or "mf_segment" not in df.columns or "unit_size_bin" not in df.columns:
        return

    classifiable = df[df[bcol].isin(["Central", "Distributed"])].copy()
    if classifiable.empty:
        return

    for label in ["Central", "Distributed"]:
        sub = classifiable[classifiable[bcol] == label]
        pivot = pd.crosstab(sub["mf_segment"], sub["unit_size_bin"], dropna=False)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Unit Size Bin")
        ax.set_ylabel("MF Segment")
        ax.set_title(f"Sample Count — {label} Heating\n(mf_segment × unit_size_bin)", fontsize=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val > pivot.values.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=10, color=color)

        fig.colorbar(im, ax=ax, label="Count")
        fig.tight_layout()
        fname = outdir / f"02b_heatmap_{label.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Classification-view prevalence table
# ---------------------------------------------------------------------------

def classification_view_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts by classification view × system type within each mf_segment."""
    if "heating_classification_mode" not in df.columns:
        return pd.DataFrame()

    rows = []
    segments = ["all_mf"] + list(df["mf_segment"].dropna().unique())
    for seg in segments:
        sub = df if seg == "all_mf" else df[df["mf_segment"] == seg]
        for view in CLASSIFICATION_VIEWS:
            vdf = filter_classification_view(sub, view)
            n_central = int((vdf["heating_system_type_binary"] == "Central").sum())
            n_distributed = int((vdf["heating_system_type_binary"] == "Distributed").sum())
            n_total = n_central + n_distributed
            rows.append({
                "mf_segment": seg,
                "classification_view": view,
                "n_central": n_central,
                "n_distributed": n_distributed,
                "n_total": n_total,
                "pct_central": round(100.0 * n_central / n_total, 1) if n_total else 0,
                "pct_distributed": round(100.0 * n_distributed / n_total, 1) if n_total else 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample balance and heating system prevalence for RECS 2020.",
    )
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

    # ---- 1. Prevalence by mf_segment (full classification) ----
    if "mf_segment" in df.columns:
        print("\n" + "=" * 70)
        print("  Heating system prevalence by mf_segment (all labels)")
        print("=" * 70)
        tbl_seg = prevalence_table(df, "mf_segment")
        print(tbl_seg.to_string(index=False))
        tbl_seg.to_csv(args.outdir / "02b_prevalence_by_segment.csv", index=False)

        plot_stacked_bar(
            tbl_seg, "mf_segment",
            "Heating System Type Prevalence by MF Segment\n(RECS 2020 Multifamily)",
            args.outdir / "02b_prevalence_by_segment.png",
        )

    # ---- 2. Prevalence by unit_size_bin (binary) ----
    if "unit_size_bin" in df.columns:
        print("\n" + "=" * 70)
        print("  Heating system prevalence by unit_size_bin (binary)")
        print("=" * 70)
        tbl_size = prevalence_table_binary(df, "unit_size_bin")
        print(tbl_size.to_string(index=False))
        tbl_size.to_csv(args.outdir / "02b_prevalence_by_unit_size.csv", index=False)

        plot_stacked_bar(
            tbl_size, "unit_size_bin",
            "Heating System Type Prevalence by Unit Size\n(RECS 2020 Multifamily, binary classification)",
            args.outdir / "02b_prevalence_by_unit_size.png",
        )

    # ---- 3. Prevalence by mf_segment × unit_size_bin ----
    if "mf_segment" in df.columns and "unit_size_bin" in df.columns:
        print("\n" + "=" * 70)
        print("  Heating system prevalence by mf_segment × unit_size_bin (binary)")
        print("=" * 70)
        df["_seg_x_size"] = df["mf_segment"].astype(str) + " | " + df["unit_size_bin"].astype(str)
        tbl_cross = prevalence_table_binary(df, "_seg_x_size")
        print(tbl_cross.to_string(index=False))
        tbl_cross.to_csv(args.outdir / "02b_prevalence_segment_x_size.csv", index=False)

        plot_stacked_bar(
            tbl_cross, "_seg_x_size",
            "Heating System Type by MF Segment × Unit Size\n(RECS 2020 Multifamily)",
            args.outdir / "02b_prevalence_segment_x_size.png",
        )
        df.drop(columns=["_seg_x_size"], inplace=True)

    # ---- 4. Heatmap of counts ----
    plot_count_heatmap(df, args.outdir)

    # ---- 5. Classification-view summary ----
    print("\n" + "=" * 70)
    print("  Classification view summary (explicit / inferred / pooled)")
    print("=" * 70)
    cv_tbl = classification_view_summary(df)
    if not cv_tbl.empty:
        print(cv_tbl.to_string(index=False))
        cv_tbl.to_csv(args.outdir / "02b_classification_view_summary.csv", index=False)

    logger.info("Done — all prevalence outputs saved to %s", args.outdir)


if __name__ == "__main__":
    main()
