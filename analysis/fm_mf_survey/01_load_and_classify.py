"""Prompt 1 — Data Loading and Classification.

Loads the 2023 Multifamily Survey dataset, applies the three HVAC/DHW
classification functions, and prints a summary of:

  * Classification label counts for each system type
  * % of the energy-analysis subset that has classifiable systems
  * Basic data-quality notes (e.g. "Not Provided" prevalence)

Usage::

    python analysis/fm_mf_survey/01_load_and_classify.py \\
        --data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \\
        [--outdir outputs/fm_mf_survey]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset
from src.datasets.fm_mf_survey.classify import add_system_classifications, classifiable_mask
from src.common.log import get_logger

logger = get_logger("fm_mf_survey.01_load_classify")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_separator(title: str = "") -> None:
    line = "=" * 70
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


def _classification_summary(df: pd.DataFrame, col: str) -> None:
    counts = df[col].value_counts(dropna=False)
    total = len(df)
    classifiable = (df[col] != "Unknown").sum()
    pct = 100.0 * classifiable / total if total else 0.0

    print(f"\n--- {col} ---")
    print(counts.to_string())
    print(f"  Classifiable (non-Unknown): {classifiable} / {total}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Load and classify 2023 MF Survey data.")
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to 2023_Multifamily_Survey_dataset_FINAL.xlsx")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="If provided, save classified CSV here.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load full dataset
    # ------------------------------------------------------------------
    _print_separator("Step 1: Load full dataset")
    logger.info("Loading %s", args.data)
    df_full = load_mf_survey(args.data)
    print(f"Full dataset shape: {df_full.shape}  (rows × columns)")

    # ------------------------------------------------------------------
    # 2. Filter to energy-analysis subset
    # ------------------------------------------------------------------
    _print_separator("Step 2: Filter to energy-analysis subset")
    df = load_energy_subset(df_full)
    print(f"Energy-analysis subset: {len(df)} properties")

    # ------------------------------------------------------------------
    # 3. Apply classifications
    # ------------------------------------------------------------------
    _print_separator("Step 3: Apply HVAC/DHW classifications")
    df = add_system_classifications(df)

    cls_cols = ["dhw_system_type", "heating_system_type", "cooling_system_type"]
    for col in cls_cols:
        _classification_summary(df, col)

    # ------------------------------------------------------------------
    # 4. Key source-field missingness (non-numeric columns)
    # ------------------------------------------------------------------
    _print_separator("Step 4: Source-field coverage")
    source_fields = {
        "Hot Water Heating Location": "DHW location",
        "Residential Hot Water System": "DHW equipment",
        "Type of Heating Plant": "Heating plant",
        "Residential Heating Equipment": "Heating equipment",
        "Type of Cooling Plant": "Cooling plant",
        "Residential Cooling Equipment": "Cooling equipment",
    }
    n = len(df)
    for col, label in source_fields.items():
        if col in df.columns:
            not_missing = df[col].notna() & ~df[col].isin(
                {"Not Provided", "Not Applicable", "N/A", "Not Available"}
            )
            print(f"  {label:35s}: {not_missing.sum():4d} / {n}  ({100*not_missing.sum()/n:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Energy payer fields (proxy for central vs. distributed)
    # ------------------------------------------------------------------
    _print_separator("Step 5: Energy payer distributions (proxy for system type)")
    payer_cols = {
        "Hot Water Heating Energy Payer": "DHW payer",
        "Space Heating Energy Payer": "Heating payer",
        "Cooling Energy Payer": "Cooling payer",
    }
    for col, label in payer_cols.items():
        if col in df.columns:
            valid = df[col].dropna()
            valid = valid[~valid.isin({"Not Provided", "Not Applicable"})]
            print(f"\n  {label}:")
            if len(valid):
                print(valid.value_counts().to_string())
            else:
                print("    (no responses)")

    # ------------------------------------------------------------------
    # 6. Cross-tab heating × cooling classification
    # ------------------------------------------------------------------
    _print_separator("Step 6: Cross-tab heating_system_type × cooling_system_type")
    xtab = pd.crosstab(
        df["heating_system_type"],
        df["cooling_system_type"],
        margins=True,
        margins_name="Total",
    )
    print(xtab.to_string())

    # ------------------------------------------------------------------
    # 7. Save (optional)
    # ------------------------------------------------------------------
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        out_path = args.outdir / "fm_mf_survey_classified.csv"
        save_cols = cls_cols + [
            "Hot Water Heating Location",
            "Residential Hot Water System",
            "Type of Heating Plant",
            "Residential Heating Equipment",
            "Type of Cooling Plant",
            "Residential Cooling Equipment",
            "Site EUI (kBTU/sqft/yr)",
            "Source EUI (kBTU/sqft/yr)",
            "Site Energy/Unit (kBtu/Unit/yr)",
            "Source Energy/Unit (kBtu/Unit/yr)",
            "Region",
            "Building Type #1",
            "Year Built",
            "Grand Total Number of Units",
            "2022 Annual HDD65",
            "2022 Annual CDD65",
        ]
        save_cols = [c for c in save_cols if c in df.columns]
        df[save_cols].to_csv(out_path, index=False)
        logger.info("Saved classified data to %s", out_path)
        print(f"\nClassified data saved to: {out_path}")

    _print_separator("Done")


if __name__ == "__main__":
    main()
