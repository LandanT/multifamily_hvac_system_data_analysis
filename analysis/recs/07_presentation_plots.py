"""RECS 2020 — Step 7: Presentation-quality visualisations.

Targeted plots designed to communicate the Central vs. Distributed story
to colleagues.  Uses binary labels (collapsing inferred) throughout.

Produces:
  1. Forest plot — OLS coefficients with 95% CIs for all three system types.
  2. Grouped bar — median Site EUI, Central vs. Distributed, by system type
     with significance annotations.
  3. Top climate zones — median EUI side-by-side for the largest IECC zones,
     heating system type only.
  4. Fuel-stratified comparison — median EUI by fuel stratum for heating.
  5. Climate zone lollipop — median EUI difference (Central − Distributed) per
     climate zone with significance markers.

Usage::

    python analysis/recs/07_presentation_plots.py \\
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
from scipy import stats

from src.common.log import get_logger
from src.datasets.recs.utils import (
    load_curated,
    filter_unit_type,
    filter_segment,
    filter_fuel,
    filter_classification_view,
)

logger = get_logger("recs.07_plots")

ALPHA = 0.05
BINARY_COLS = {
    "Heating": "heating_system_type_binary",
    "Cooling": "cooling_system_type_binary",
    "DHW": "dhw_system_type_binary",
}

# End-use EUI column paired with each system type for the median comparison plot
ENDUSE_EUI_COLS = {
    "Heating": "Heating_EUI_kBtu_sqft",
    "Cooling": "Cooling_EUI_kBtu_sqft",
    "DHW": "DHW_EUI_kBtu_sqft",
}

# Consistent palette
C_CENTRAL = "#2196F3"
C_DISTRIBUTED = "#FF7043"
C_NEUTRAL = "#9E9E9E"


# ---------------------------------------------------------------------------
# 1. Forest plot — OLS coefficients
# ---------------------------------------------------------------------------

def plot_forest(df: pd.DataFrame, outdir: Path) -> None:
    """Forest plot of OLS distributed coefficient ± 95% CI per system type."""
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not installed — skipping forest plot")
        return

    rows = []
    for label, bcol in BINARY_COLS.items():
        if bcol not in df.columns:
            continue
        sub = df[df[bcol].isin(["Central", "Distributed"])].copy()
        needed = ["Site_EUI_kBtu_sqft", bcol, "IECC_climate_code", "YEARMADERANGE", "TOTSQFT_EN"]
        sub = sub[[c for c in needed if c in sub.columns]].dropna()
        if len(sub) < 20:
            continue

        sub["_dist"] = (sub[bcol] == "Distributed").astype(int)
        sub["YEARMADERANGE"] = pd.to_numeric(sub["YEARMADERANGE"], errors="coerce")
        sub["TOTSQFT_EN"] = pd.to_numeric(sub["TOTSQFT_EN"], errors="coerce")
        sub["_log_sqft"] = np.log(sub["TOTSQFT_EN"].clip(lower=1))
        sub = sub.dropna()

        climate_dummies = pd.get_dummies(
            sub["IECC_climate_code"].astype(str), prefix="cz", drop_first=True,
        )
        X = sm.add_constant(
            pd.concat([sub[["_dist", "YEARMADERANGE", "_log_sqft"]], climate_dummies], axis=1).astype(float)
        )
        model = sm.OLS(sub["Site_EUI_kBtu_sqft"].astype(float), X).fit()

        coef = model.params.get("_dist", np.nan)
        ci = model.conf_int().loc["_dist"] if "_dist" in model.conf_int().index else (np.nan, np.nan)
        pval = model.pvalues.get("_dist", np.nan)
        rows.append({
            "label": label, "coef": coef,
            "ci_lo": ci[0], "ci_hi": ci[1],
            "pval": pval, "n": int(model.nobs),
        })

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = list(range(len(rows)))

    for i, r in enumerate(rows):
        color = C_CENTRAL if r["pval"] < ALPHA else C_NEUTRAL
        ax.errorbar(
            r["coef"], i,
            xerr=[[r["coef"] - r["ci_lo"]], [r["ci_hi"] - r["coef"]]],
            fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, markersize=8,
        )
        sig = "**" if r["pval"] < 0.01 else ("*" if r["pval"] < 0.05 else "")
        ax.text(
            r["ci_hi"] + 0.5, i,
            f'  {r["coef"]:+.1f}  (p={r["pval"]:.3f})   n={r["n"]}',
            va="center", fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_xlabel("Distributed − Central  (kBtu/sqft/yr)", fontsize=10)
    ax.set_title(
        "OLS coefficient for Distributed system type\n"
        "(controlling for IECC climate zone, vintage, floor area)",
        fontsize=11, fontweight="bold",
    )
    ax.invert_yaxis()
    fig.tight_layout()

    fname = outdir / "07_forest_ols_coefficients.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 2. Grouped bar — median end-use EUI by system type (binary)
# ---------------------------------------------------------------------------

def plot_median_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """Side-by-side median end-use EUI bars for each system type with Mann-Whitney p."""
    labels = []
    medians_c, medians_d = [], []
    ns_c, ns_d = [], []
    pvals = []

    for label, bcol in BINARY_COLS.items():
        if bcol not in df.columns:
            continue
        eui_col = ENDUSE_EUI_COLS.get(label)
        if eui_col is None or eui_col not in df.columns:
            continue
        c = df.loc[df[bcol] == "Central", eui_col].dropna()
        d = df.loc[df[bcol] == "Distributed", eui_col].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        labels.append(label)
        medians_c.append(c.median())
        medians_d.append(d.median())
        ns_c.append(len(c))
        ns_d.append(len(d))
        pvals.append(p)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 6.3))

    bars_c = ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    bars_d = ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    # Annotate n= inside bars
    for bar, n in zip(bars_c, ns_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                f"n={n}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    for bar, n in zip(bars_d, ns_d):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                f"n={n}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    # Significance annotations
    for i, p in enumerate(pvals):
        y_max = max(medians_c[i], medians_d[i])
        sig_text = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(x[i], y_max + 2, sig_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(bottom=0, top=max(max(medians_c), max(medians_d)) * 1.25)
    ax.set_ylabel("Median End-Use EUI (kBtu/sqft/yr)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title("Median End-Use EUI: Central vs. Distributed\n(RECS 2020, Multifamily)", fontsize=12, fontweight="bold")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()

    fname = outdir / "07_median_eui_by_system_type.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 3. Top climate zones — faceted grouped bar (heating only)
# ---------------------------------------------------------------------------

def plot_top_climate_zones(df: pd.DataFrame, outdir: Path, top_n: int = 8) -> None:
    """Median EUI by Central vs. Distributed within the largest climate zones."""
    bcol = "heating_system_type_binary"
    if bcol not in df.columns or "IECC_climate_code" not in df.columns:
        return

    sub = df[df[bcol].isin(["Central", "Distributed"]) & df["Site_EUI_kBtu_sqft"].notna()].copy()

    # Select top N zones by total count
    zone_counts = sub["IECC_climate_code"].value_counts()
    top_zones = zone_counts.head(top_n).index.tolist()
    sub = sub[sub["IECC_climate_code"].isin(top_zones)]

    zones = sorted(top_zones)
    medians_c, medians_d, ns_c, ns_d, pvals = [], [], [], [], []

    for zone in zones:
        zdf = sub[sub["IECC_climate_code"] == zone]
        c = zdf.loc[zdf[bcol] == "Central", "Site_EUI_kBtu_sqft"].dropna()
        d = zdf.loc[zdf[bcol] == "Distributed", "Site_EUI_kBtu_sqft"].dropna()
        medians_c.append(c.median() if len(c) else 0)
        medians_d.append(d.median() if len(d) else 0)
        ns_c.append(len(c))
        ns_d.append(len(d))
        if len(c) >= 2 and len(d) >= 2:
            _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(np.nan)

    x = np.arange(len(zones))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars_c = ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    bars_d = ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    for i, zone in enumerate(zones):
        ax.text(x[i] - width / 2, medians_c[i] + 0.5, f"n={ns_c[i]}",
                ha="center", va="bottom", fontsize=7, color="dimgray")
        ax.text(x[i] + width / 2, medians_d[i] + 0.5, f"n={ns_d[i]}",
                ha="center", va="bottom", fontsize=7, color="dimgray")

        if not np.isnan(pvals[i]) and pvals[i] < ALPHA:
            y_max = max(medians_c[i], medians_d[i])
            ax.text(x[i], y_max + 4, f"p={pvals[i]:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#D32F2F")

    ax.set_ylabel("Median Site EUI (kBtu/sqft/yr)", fontsize=10)
    ax.set_xlabel("IECC Climate Zone", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=10)
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(
        "Median Site EUI by Climate Zone \u2014 Heating System Type\n"
        f"(top {top_n} zones by sample size)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    fname = outdir / "07_climate_zone_heating_comparison.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 4. Fuel-stratified median comparison (heating)
# ---------------------------------------------------------------------------

def plot_fuel_stratified(df: pd.DataFrame, outdir: Path) -> None:
    """Grouped bars: Central vs Distributed median Heating EUI, split by fuel stratum.

    Fuel strata use mutually exclusive logic to avoid overlap:
    - Electric Only: ELWARM==1 AND UGWARM!=1
    - Gas Only: UGWARM==1 AND ELWARM!=1
    """
    bcol = "heating_system_type_binary"
    eui = "Heating_EUI_kBtu_sqft"
    if bcol not in df.columns or eui not in df.columns:
        return

    # Build mutually exclusive fuel subsets
    has_el = "ELWARM" in df.columns
    has_ng = "UGWARM" in df.columns
    elec_only = df[(df["ELWARM"] == 1) & (df["UGWARM"] != 1)] if (has_el and has_ng) else pd.DataFrame()
    gas_only = df[(df["UGWARM"] == 1) & (df["ELWARM"] != 1)] if (has_el and has_ng) else pd.DataFrame()

    strata = [
        ("All Fuels", df),
        ("Electric Only", elec_only),
        ("Gas Only", gas_only),
    ]

    labels, medians_c, medians_d, ns_c, ns_d, pvals = [], [], [], [], [], []

    for slabel, sdf in strata:
        if sdf.empty or bcol not in sdf.columns:
            continue
        c = sdf.loc[sdf[bcol] == "Central", eui].dropna()
        d = sdf.loc[sdf[bcol] == "Distributed", eui].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        labels.append(slabel)
        medians_c.append(c.median())
        medians_d.append(d.median())
        ns_c.append(len(c))
        ns_d.append(len(d))
        pvals.append(p)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_c = ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    bars_d = ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    for bar, n in zip(bars_c, ns_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                f"n={n}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    for bar, n in zip(bars_d, ns_d):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                f"n={n}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    for i, p in enumerate(pvals):
        y_max = max(medians_c[i], medians_d[i])
        sig_text = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(x[i], y_max + 1.5, sig_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Headroom so annotations don't collide with the title
    ax.set_ylim(bottom=0, top=max(max(medians_c), max(medians_d)) * 1.25)
    ax.set_ylabel("Median Heating EUI (kBtu/sqft/yr)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(
        "Heating EUI by Fuel — 5+ Unit MF Buildings\n"
        "Central vs. Distributed (RECS 2020)\n"
        "(fuel strata are mutually exclusive)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    fname = outdir / "07_fuel_stratified_heating.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 5. Climate zone lollipop — EUI difference per zone
# ---------------------------------------------------------------------------

def plot_cz_lollipop(df: pd.DataFrame, outdir: Path, min_n: int = 10) -> None:
    """Lollipop: median EUI difference (Central − Distributed) per IECC zone."""
    bcol = "heating_system_type_binary"
    if bcol not in df.columns or "IECC_climate_code" not in df.columns:
        return

    sub = df[df[bcol].isin(["Central", "Distributed"]) & df["Site_EUI_kBtu_sqft"].notna()].copy()
    zones = sorted(sub["IECC_climate_code"].dropna().unique())

    rows = []
    for zone in zones:
        zdf = sub[sub["IECC_climate_code"] == zone]
        c = zdf.loc[zdf[bcol] == "Central", "Site_EUI_kBtu_sqft"].dropna()
        d = zdf.loc[zdf[bcol] == "Distributed", "Site_EUI_kBtu_sqft"].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        total_n = len(c) + len(d)
        if total_n < min_n:
            continue
        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        rows.append({
            "zone": zone,
            "diff": c.median() - d.median(),
            "n": total_n,
            "p": p,
            "sig": p < ALPHA,
        })

    if not rows:
        return

    rdf = pd.DataFrame(rows).sort_values("diff")
    y = np.arange(len(rdf))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(rdf))))

    colors = [C_CENTRAL if r["sig"] else C_NEUTRAL for _, r in rdf.iterrows()]
    for i, (_, r) in enumerate(rdf.iterrows()):
        ax.hlines(i, 0, r["diff"], color=colors[i], linewidth=2)
        marker = "o" if r["sig"] else "s"
        ax.plot(r["diff"], i, marker, color=colors[i], markersize=7)
        side = "left" if r["diff"] >= 0 else "right"
        ha = "left" if r["diff"] >= 0 else "right"
        offset = 0.8 if r["diff"] >= 0 else -0.8
        sig_label = f'n={r["n"]}  p={r["p"]:.3f}'
        ax.text(r["diff"] + offset, i, sig_label, va="center", ha=ha, fontsize=7.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(rdf["zone"].values, fontsize=9)
    ax.set_xlabel("Median Site EUI difference: Central − Distributed (kBtu/sqft/yr)", fontsize=10)
    ax.set_title(
        "Per-Climate-Zone EUI Difference — Heating System Type\n"
        "(positive = Central higher EUI; blue ● = significant at α=0.05)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    fname = outdir / "07_climate_zone_lollipop.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 6. Heating EUI by central vs distributed, split by MF segment
# ---------------------------------------------------------------------------

def plot_heating_by_segment(df: pd.DataFrame, outdir: Path) -> None:
    """Grouped bar: Heating EUI by Central vs Distributed, faceted by mf_segment."""
    bcol = "heating_system_type_binary"
    eui = "Heating_EUI_kBtu_sqft"
    if bcol not in df.columns or eui not in df.columns or "mf_segment" not in df.columns:
        return

    segments = ["2_to_4_units", "5plus_units"]
    labels, medians_c, medians_d, ns_c, ns_d, pvals = [], [], [], [], [], []

    for seg in segments:
        sub = filter_segment(df, seg)
        c = sub.loc[sub[bcol] == "Central", eui].dropna()
        d = sub.loc[sub[bcol] == "Distributed", eui].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        labels.append(seg.replace("_", " "))
        medians_c.append(c.median())
        medians_d.append(d.median())
        ns_c.append(len(c))
        ns_d.append(len(d))
        pvals.append(p)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_c = ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    bars_d = ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    for bar, n in zip(bars_c, ns_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")
    for bar, n in zip(bars_d, ns_d):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    for i, p in enumerate(pvals):
        y_max = max(medians_c[i], medians_d[i])
        sig_text = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if sig_text:
            ax.text(x[i], y_max + 1.5, sig_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Headroom so annotations don't collide with the title
    ax.set_ylim(bottom=0, top=max(max(medians_c), max(medians_d)) * 1.25)
    ax.set_ylabel("Median Heating EUI (kBtu/sqft/yr)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(
        "Heating EUI: Central vs. Distributed by MF Segment\n(RECS 2020 Multifamily)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    fname = outdir / "07_heating_eui_by_segment.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 7. Heating EUI by fuel within 5+ unit buildings
# ---------------------------------------------------------------------------

def plot_heating_by_fuel_5plus(df: pd.DataFrame, outdir: Path) -> None:
    """Grouped bars: Heating EUI Central vs Distributed for 5+ unit, by fuel."""
    bcol = "heating_system_type_binary"
    eui = "Heating_EUI_kBtu_sqft"
    if bcol not in df.columns or eui not in df.columns:
        return

    seg_df = filter_segment(df, "5plus_units")

    strata = [
        ("All Fuels", seg_df),
        ("Electric Only", filter_fuel(seg_df, "electric")),
        ("Gas Only", filter_fuel(seg_df, "gas")),
    ]

    labels, medians_c, medians_d, ns_c, ns_d, pvals = [], [], [], [], [], []

    for slabel, sdf in strata:
        if sdf.empty:
            continue
        c = sdf.loc[sdf[bcol] == "Central", eui].dropna()
        d = sdf.loc[sdf[bcol] == "Distributed", eui].dropna()
        if len(c) < 2 or len(d) < 2:
            continue
        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
        labels.append(slabel)
        medians_c.append(c.median())
        medians_d.append(d.median())
        ns_c.append(len(c))
        ns_d.append(len(d))
        pvals.append(p)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_c = ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    bars_d = ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    for bar, n in zip(bars_c, ns_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")
    for bar, n in zip(bars_d, ns_d):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    for i, p in enumerate(pvals):
        y_max = max(medians_c[i], medians_d[i])
        sig_text = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(x[i], y_max + 1.5, sig_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Headroom so annotations don't collide with the title
    ax.set_ylim(bottom=0, top=max(max(medians_c), max(medians_d)) * 1.25)
    ax.set_ylabel("Median Heating EUI (kBtu/sqft/yr)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(
        "Heating EUI by Fuel — 5+ Unit MF Buildings\n"
        "Central vs. Distributed (RECS 2020)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    fname = outdir / "07_heating_eui_fuel_5plus.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 8. Explicit vs Inferred sensitivity comparison
# ---------------------------------------------------------------------------

def plot_explicit_vs_inferred(df: pd.DataFrame, outdir: Path) -> None:
    """Show how Heating EUI results differ across classification views."""
    bcol = "heating_system_type_binary"
    eui = "Heating_EUI_kBtu_sqft"
    if bcol not in df.columns or eui not in df.columns:
        return

    views = [
        ("Pooled Binary", "pooled_binary"),
        ("Explicit Only", "explicit_only"),
        ("Inferred Only", "inferred_only"),
    ]

    labels, medians_c, medians_d, ns_c, ns_d = [], [], [], [], []

    for vlabel, vkey in views:
        vdf = filter_classification_view(df, vkey, bcol)
        c = vdf.loc[vdf[bcol] == "Central", eui].dropna()
        d = vdf.loc[vdf[bcol] == "Distributed", eui].dropna()
        if len(c) < 1 and len(d) < 1:
            continue
        labels.append(vlabel)
        medians_c.append(c.median() if len(c) else 0)
        medians_d.append(d.median() if len(d) else 0)
        ns_c.append(len(c))
        ns_d.append(len(d))

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    for i in range(len(labels)):
        ax.text(x[i] - width / 2, medians_c[i] + 0.3, f"n={ns_c[i]}",
                ha="center", va="bottom", fontsize=8, color="dimgray")
        ax.text(x[i] + width / 2, medians_d[i] + 0.3, f"n={ns_d[i]}",
                ha="center", va="bottom", fontsize=8, color="dimgray")

    ax.set_ylabel("Median Heating EUI (kBtu/sqft/yr)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(
        "Heating EUI by Classification View\n"
        "How much depends on explicit vs. inferred labels?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    fname = outdir / "07_explicit_vs_inferred.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 9. Explicit-only deep dive — Electric EUI and Gas EUI
# ---------------------------------------------------------------------------

def plot_explicit_electric_gas_eui(
    df: pd.DataFrame, outdir: Path, segment: str = "all_mf",
) -> None:
    """Grouped bars: Electric EUI and Gas EUI for explicitly classified systems."""
    bcol = "heating_system_type_binary"
    if bcol not in df.columns:
        return

    src = filter_segment(df, segment) if segment != "all_mf" else df

    eui_metrics = [
        ("Electric_EUI_kBtu_sqft", "Electric EUI"),
        ("Gas_EUI_kBtu_sqft", "Natural Gas EUI"),
    ]
    eui_metrics = [(col, lbl) for col, lbl in eui_metrics if col in src.columns]
    if not eui_metrics:
        return

    explicit = filter_classification_view(src, "explicit_only", bcol)
    if len(explicit) < 10:
        logger.warning("Too few explicit rows (%d) — skipping explicit EUI deep dive", len(explicit))
        return

    seg_label = "5+ Unit MF" if segment == "5plus_units" else "All MF"
    fig, axes = plt.subplots(1, len(eui_metrics), figsize=(6.5 * len(eui_metrics), 5))
    if len(eui_metrics) == 1:
        axes = [axes]

    for ax, (eui_col, eui_label) in zip(axes, eui_metrics):
        c = explicit.loc[explicit[bcol] == "Central", eui_col].dropna()
        d = explicit.loc[explicit[bcol] == "Distributed", eui_col].dropna()
        if len(c) < 2 or len(d) < 2:
            ax.set_visible(False)
            continue

        _, p = stats.mannwhitneyu(c, d, alternative="two-sided")

        x = np.arange(2)
        width = 0.5
        bars = ax.bar(x, [c.median(), d.median()], width,
                       color=[C_CENTRAL, C_DISTRIBUTED], edgecolor="white")

        ax.text(0, c.median() + 0.3, f"n={len(c)}\nmed={c.median():.1f}",
                ha="center", va="bottom", fontsize=9, color="dimgray")
        ax.text(1, d.median() + 0.3, f"n={len(d)}\nmed={d.median():.1f}",
                ha="center", va="bottom", fontsize=9, color="dimgray")

        sig_text = f"p={p:.4f}"
        y_max = max(c.median(), d.median())
        ax.text(0.5, y_max * 1.15, sig_text, ha="center", va="bottom",
                fontsize=9, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(["Central", "Distributed"], fontsize=11)
        ax.set_ylabel(f"{eui_label} (kBtu/sqft/yr)", fontsize=10)
        ax.set_title(f"{eui_label}", fontsize=12, fontweight="bold")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.suptitle(
        f"Explicit Classifications Only — Electric & Gas EUI\n"
        f"Central vs. Distributed (RECS 2020, {seg_label})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    suffix = f"_{segment}" if segment != "all_mf" else ""
    fname = outdir / f"07_explicit_electric_gas_eui{suffix}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 10. Explicit-only — Heating EUI by IECC climate zone, electric vs gas panels
# ---------------------------------------------------------------------------

def plot_explicit_eui_by_climate_zone(
    df: pd.DataFrame, outdir: Path, top_n: int = 8, segment: str = "all_mf",
) -> None:
    """Two-panel: Heating EUI by climate zone, electric-heated vs gas-heated (explicit only).

    Both panels share a Y-axis so the magnitude difference between fuels is
    immediately visible.
    """
    bcol = "heating_system_type_binary"
    cz_col = "IECC_climate_code"
    eui_col = "Heating_EUI_kBtu_sqft"
    if bcol not in df.columns or cz_col not in df.columns or eui_col not in df.columns:
        return

    src = filter_segment(df, segment) if segment != "all_mf" else df

    explicit = filter_classification_view(src, "explicit_only", bcol)
    if len(explicit) < 10:
        logger.warning("Too few explicit rows (%d) — skipping climate zone EUI", len(explicit))
        return

    seg_label = "5+ Unit MF" if segment == "5plus_units" else "All MF"

    # Build mutually exclusive fuel subsets
    has_el = "ELWARM" in explicit.columns
    has_ng = "UGWARM" in explicit.columns
    if not (has_el and has_ng):
        return

    fuel_panels = [
        ("All Fuels", explicit),
        ("Electric-Heated Only", explicit[(explicit["ELWARM"] == 1) & (explicit["UGWARM"] != 1)]),
        ("Gas-Heated Only", explicit[(explicit["UGWARM"] == 1) & (explicit["ELWARM"] != 1)]),
    ]

    # Top N zones by total explicit count (union of both fuels for consistency)
    zone_counts = explicit[cz_col].value_counts()
    top_zones = zone_counts.head(top_n).index.tolist()
    zones = sorted(top_zones)

    n_panels = len(fuel_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.5 * n_panels), sharex=True, sharey=True)

    for ax, (fuel_label, fuel_df) in zip(axes, fuel_panels):
        fuel_df = fuel_df[fuel_df[cz_col].isin(top_zones)]
        medians_c, medians_d, ns_c, ns_d, pvals = [], [], [], [], []

        for zone in zones:
            zdf = fuel_df[fuel_df[cz_col] == zone]
            c = zdf.loc[zdf[bcol] == "Central", eui_col].dropna()
            d = zdf.loc[zdf[bcol] == "Distributed", eui_col].dropna()
            medians_c.append(c.median() if len(c) else 0)
            medians_d.append(d.median() if len(d) else 0)
            ns_c.append(len(c))
            ns_d.append(len(d))
            if len(c) >= 2 and len(d) >= 2:
                _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
                pvals.append(p)
            else:
                pvals.append(np.nan)

        x = np.arange(len(zones))
        width = 0.35

        ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
        ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

        for i in range(len(zones)):
            ax.text(x[i] - width / 2, medians_c[i] + 0.3, f"n={ns_c[i]}",
                    ha="center", va="bottom", fontsize=7, color="dimgray")
            ax.text(x[i] + width / 2, medians_d[i] + 0.3, f"n={ns_d[i]}",
                    ha="center", va="bottom", fontsize=7, color="dimgray")

            if not np.isnan(pvals[i]) and pvals[i] < ALPHA:
                y_max = max(medians_c[i], medians_d[i])
                sig_text = "***" if pvals[i] < 0.001 else ("**" if pvals[i] < 0.01 else "*")
                ax.text(x[i], y_max + 1.5, sig_text, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#D32F2F")

        ax.set_ylabel("Median Heating EUI\n(kBtu/sqft/yr)", fontsize=10)
        ax.set_title(f"{fuel_label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(zones, fontsize=10)
    axes[-1].set_xlabel("IECC Climate Zone", fontsize=10)

    fig.suptitle(
        f"Explicit Classifications Only — Heating EUI by Climate Zone\n"
        f"Central vs. Distributed, {seg_label} (top {top_n} zones, * p<.05  ** p<.01  *** p<.001)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    suffix = f"_{segment}" if segment != "all_mf" else ""
    fname = outdir / f"07_explicit_eui_by_climate_zone{suffix}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def plot_explicit_eui_by_climate_zone_allfuels(
    df: pd.DataFrame, outdir: Path, top_n: int = 8, segment: str = "all_mf",
) -> None:
    """Single-panel: Heating EUI by climate zone, all fuels combined (explicit only)."""
    bcol = "heating_system_type_binary"
    cz_col = "IECC_climate_code"
    eui_col = "Heating_EUI_kBtu_sqft"
    if bcol not in df.columns or cz_col not in df.columns or eui_col not in df.columns:
        return

    src = filter_segment(df, segment) if segment != "all_mf" else df

    explicit = filter_classification_view(src, "explicit_only", bcol)
    if len(explicit) < 10:
        logger.warning("Too few explicit rows (%d) — skipping climate zone EUI (all fuels)", len(explicit))
        return

    seg_label = "5+ Unit MF" if segment == "5plus_units" else "All MF"

    # Top N zones by total explicit count
    zone_counts = explicit[cz_col].value_counts()
    top_zones = zone_counts.head(top_n).index.tolist()
    zones = sorted(top_zones)

    fig, ax = plt.subplots(figsize=(12, 5))

    fuel_df = explicit[explicit[cz_col].isin(top_zones)]
    medians_c, medians_d, ns_c, ns_d, pvals = [], [], [], [], []

    for zone in zones:
        zdf = fuel_df[fuel_df[cz_col] == zone]
        c = zdf.loc[zdf[bcol] == "Central", eui_col].dropna()
        d = zdf.loc[zdf[bcol] == "Distributed", eui_col].dropna()
        medians_c.append(c.median() if len(c) else 0)
        medians_d.append(d.median() if len(d) else 0)
        ns_c.append(len(c))
        ns_d.append(len(d))
        if len(c) >= 2 and len(d) >= 2:
            _, p = stats.mannwhitneyu(c, d, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(np.nan)

    x = np.arange(len(zones))
    width = 0.35

    ax.bar(x - width / 2, medians_c, width, label="Central", color=C_CENTRAL, edgecolor="white")
    ax.bar(x + width / 2, medians_d, width, label="Distributed", color=C_DISTRIBUTED, edgecolor="white")

    for i in range(len(zones)):
        ax.text(x[i] - width / 2, medians_c[i] + 0.3, f"n={ns_c[i]}",
                ha="center", va="bottom", fontsize=8, color="dimgray")
        ax.text(x[i] + width / 2, medians_d[i] + 0.3, f"n={ns_d[i]}",
                ha="center", va="bottom", fontsize=8, color="dimgray")

        if not np.isnan(pvals[i]) and pvals[i] < ALPHA:
            y_max = max(medians_c[i], medians_d[i])
            sig_text = "***" if pvals[i] < 0.001 else ("**" if pvals[i] < 0.01 else "*")
            ax.text(x[i], y_max + 1.5, sig_text, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#D32F2F")

    ax.set_ylabel("Median Heating EUI\n(kBtu/sqft/yr)", fontsize=11)
    ax.set_xlabel("IECC Climate Zone", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.set_title(
        f"Explicit Classifications Only — Heating EUI by Climate Zone\n"
        f"Central vs. Distributed, {seg_label} (top {top_n} zones, * p<.05  ** p<.01  *** p<.001)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    suffix = f"_{segment}" if segment != "all_mf" else ""
    fname = outdir / f"07_explicit_eui_by_climate_zone_allfuels{suffix}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# 11. Utility payment structure — EUI by who pays the bill
# ---------------------------------------------------------------------------

# ELPAY/NGPAY codes: 1=household pays, 2=included in rent, 3=some/some, 99=other, -2=N/A
_PAY_LABELS = {1: "Household Pays", 2: "Included in Rent", 3: "Split"}
_PAY_FUELS = [
    ("ELPAY", "Electricity", "Electric_EUI_kBtu_sqft"),
    ("NGPAY", "Natural Gas", "Gas_EUI_kBtu_sqft"),
]


def plot_utility_payment_eui(df: pd.DataFrame, outdir: Path) -> None:
    """Grouped bar: EUI by who pays the utility bill (household vs included in rent)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (pay_col, fuel_label, eui_col) in zip(axes, _PAY_FUELS):
        if pay_col not in df.columns or eui_col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[df[pay_col].isin([1, 2, 3]) & df[eui_col].notna()].copy()
        sub["_pay_label"] = sub[pay_col].map(_PAY_LABELS)

        groups = ["Household Pays", "Included in Rent", "Split"]
        groups = [g for g in groups if g in sub["_pay_label"].values]

        medians, counts = [], []
        for g in groups:
            vals = sub.loc[sub["_pay_label"] == g, eui_col].dropna()
            medians.append(vals.median())
            counts.append(len(vals))

        x = np.arange(len(groups))
        colors = ["#4CAF50", "#FF9800", "#9E9E9E"][:len(groups)]
        bars = ax.bar(x, medians, 0.5, color=colors, edgecolor="white")

        for i, (med, n) in enumerate(zip(medians, counts)):
            ax.text(i, med + 0.3, f"n={n}\nmed={med:.1f}",
                    ha="center", va="bottom", fontsize=8, color="dimgray")

        # Mann-Whitney: household pays vs included in rent
        hp = sub.loc[sub["_pay_label"] == "Household Pays", eui_col].dropna()
        ir = sub.loc[sub["_pay_label"] == "Included in Rent", eui_col].dropna()
        if len(hp) >= 2 and len(ir) >= 2:
            _, p = stats.mannwhitneyu(hp, ir, alternative="two-sided")
            y_max = max(hp.median(), ir.median())
            ax.text(0.5, y_max * 1.15, f"p={p:.4f}",
                    ha="center", va="bottom", fontsize=9, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=10)
        ax.set_ylabel(f"{fuel_label} EUI (kBtu/sqft/yr)", fontsize=10)
        ax.set_title(f"Who Pays for {fuel_label}?", fontsize=12, fontweight="bold")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.suptitle(
        "EUI by Utility Payment Structure\n(RECS 2020 Multifamily)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    fname = outdir / "07_utility_payment_eui.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


def plot_utility_payment_by_system_type(df: pd.DataFrame, outdir: Path) -> None:
    """Faceted: EUI by system type × payment structure (Site, Heating, Cooling)."""
    bcol = "heating_system_type_binary"
    if bcol not in df.columns:
        return

    eui_metrics = [
        ("Site_EUI_kBtu_sqft", "Site EUI"),
        ("Heating_EUI_kBtu_sqft", "Heating EUI"),
        ("Cooling_EUI_kBtu_sqft", "Cooling EUI"),
    ]
    eui_metrics = [(c, l) for c, l in eui_metrics if c in df.columns]
    if not eui_metrics:
        return

    # Use ELPAY as primary payment indicator (most complete)
    pay_col = "ELPAY"
    if pay_col not in df.columns:
        return

    sub = df[df[pay_col].isin([1, 2]) & df[bcol].isin(["Central", "Distributed"])].copy()
    sub["_pay"] = sub[pay_col].map({1: "Pays Own", 2: "In Rent"})

    combos = [
        ("Central", "Pays Own"),
        ("Central", "In Rent"),
        ("Distributed", "Pays Own"),
        ("Distributed", "In Rent"),
    ]
    combo_labels = ["Central\nPays Own", "Central\nIn Rent",
                    "Distributed\nPays Own", "Distributed\nIn Rent"]
    combo_colors = [C_CENTRAL, "#90CAF9", C_DISTRIBUTED, "#FFAB91"]

    n_metrics = len(eui_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    global_max = 0
    for ax, (eui_col, eui_label) in zip(axes, eui_metrics):
        medians, counts = [], []
        for sys_type, pay in combos:
            vals = sub.loc[(sub[bcol] == sys_type) & (sub["_pay"] == pay), eui_col].dropna()
            medians.append(vals.median() if len(vals) else 0)
            counts.append(len(vals))

        x = np.arange(len(combos))
        bars = ax.bar(x, medians, 0.6, color=combo_colors, edgecolor="white")

        for i, (med, n) in enumerate(zip(medians, counts)):
            ax.text(i, med + 0.3, f"n={n}", ha="center", va="bottom", fontsize=7, color="dimgray")

        if medians:
            global_max = max(global_max, max(medians))

        ax.set_xticks(x)
        ax.set_xticklabels(combo_labels, fontsize=8)
        ax.set_ylabel(f"{eui_label}\n(kBtu/sqft/yr)", fontsize=9)
        ax.set_title(eui_label, fontsize=10, fontweight="bold")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Shared y-axis with headroom for n= labels
    if global_max:
        axes[0].set_ylim(bottom=0, top=global_max * 1.25)

    fig.suptitle(
        "EUI by System Type × Electricity Payment\n"
        "(RECS 2020 MF — Household Pays vs. Included in Rent)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    fname = outdir / "07_utility_payment_by_system_type.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Presentation-quality RECS plots.")
    ap.add_argument("--curated", type=Path, required=True,
                    help="Path to recs2020_curated_*.parquet.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/recs"),
                    help="Directory for output files (default: outputs/recs).")
    ap.add_argument("--unit-type", choices=["mf", "sf", "all"], default="mf",
                    help="Housing unit filter (default: mf).")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_curated(args.curated)
    df = filter_unit_type(df, args.unit_type)
    logger.info("Loaded %d rows (unit_type=%s)", len(df), args.unit_type)

    # Existing plots (backward compatible)
    plot_forest(df, args.outdir)
    plot_median_comparison(df, args.outdir)
    plot_top_climate_zones(df, args.outdir)
    plot_fuel_stratified(df, args.outdir)
    plot_cz_lollipop(df, args.outdir)

    # Heating-focused plots
    plot_heating_by_segment(df, args.outdir)
    plot_heating_by_fuel_5plus(df, args.outdir)
    plot_explicit_vs_inferred(df, args.outdir)

    # Explicit-only deep dive (all MF + 5+ units)
    for seg in ("all_mf", "5plus_units"):
        plot_explicit_electric_gas_eui(df, args.outdir, segment=seg)
        plot_explicit_eui_by_climate_zone(df, args.outdir, segment=seg)
        plot_explicit_eui_by_climate_zone_allfuels(df, args.outdir, segment=seg)

    # Utility payment analysis
    plot_utility_payment_eui(df, args.outdir)
    plot_utility_payment_by_system_type(df, args.outdir)

    logger.info("All presentation plots saved to %s", args.outdir)


if __name__ == "__main__":
    main()
