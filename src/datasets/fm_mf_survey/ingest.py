"""2023 Multifamily Survey dataset ingestion.

The Excel file has two header rows:
  - Row 0 (index 0): Excel column letters (A, B, C, ...)
  - Row 1 (index 1): Actual field names

So we load with ``header=1`` to use the field-name row as the column index.

Usage::

    from src.datasets.fm_mf_survey.ingest import load_mf_survey, load_energy_subset

    df = load_mf_survey("path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx")
    df_energy = load_energy_subset(df)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# Columns whose string values "Not Provided" or "Not Applicable" should be
# treated as missing (NaN) after load.
_NOT_PROVIDED_VALUES = {"Not Provided", "Not Applicable", "N/A", "Not Available"}

# Energy columns that may contain non-numeric strings.
ENERGY_NUMERIC_COLS = [
    "Total Site Energy kBTU",
    "Total Source Energy kBTU",
    "Site EUI (kBTU/sqft/yr)",
    "Source EUI (kBTU/sqft/yr)",
    "Site Energy/Unit (kBtu/Unit/yr)",
    "Source Energy/Unit (kBtu/Unit/yr)",
    "Total Energy Cost",
    "Energy Cost ($/sqft/yr)",
    "Energy Cost ($/Unit/yr)",
    # Fuel-specific site kBTU
    "Grid Electricity Site kBTU",
    "Natural Gas Site kBTU",
    "District Steam Site kBTU",
    "District Hot Water Site kBTU",
    "Propane Site kBTU",
    "Fuel Oil #1 Site kBTU",
    "Fuel Oil #2 Site kBTU",
    # Fuel-specific source kBTU
    "Grid Electricity Source kBTU",
    "Natural Gas Source kBTU",
    "District Steam Source kBTU",
    "District Hot Water Source kBTU",
    "Propane Source kBTU",
    "Fuel Oil #1 Source kBTU",
    "Fuel Oil #2 Source kBTU",
    # Climate
    "2022 Annual HDD65",
    "2022 Annual CDD65",
    # Size
    "Grand Total Number of Units",
    "Total Gross Floor Area (sqft)",
    "Percent of Gross Floor Area that is Heated",
    "Percent of Gross Floor Area that is Cooled",
    "Year Built",
]


def load_mf_survey(path: str | Path) -> pd.DataFrame:
    """Load the 2023 Multifamily Survey Excel file.

    Parameters
    ----------
    path:
        Path to ``2023_Multifamily_Survey_dataset_FINAL.xlsx``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with field-name columns (2,273 rows).
        Numeric energy/climate columns are coerced to float.
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Database", header=1, engine="openpyxl")

    # Coerce known numeric columns; non-parseable strings become NaN.
    for col in ENERGY_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_energy_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to properties that pass all energy QC filters (n≈980).

    Parameters
    ----------
    df:
        Full dataframe returned by :func:`load_mf_survey`.

    Returns
    -------
    pd.DataFrame
        Copy filtered to ``Include in energy analysis? == 1``.
    """
    col = "Include in energy analysis?"
    if col not in df.columns:
        raise KeyError(
            f"Column '{col}' not found. Ensure the file was loaded with header=1."
        )
    return df[df[col] == 1].copy()


def is_not_provided(series: pd.Series) -> pd.Series:
    """Return boolean mask: True where value is NaN or a 'Not Provided' string."""
    return series.isna() | series.isin(_NOT_PROVIDED_VALUES)


def clean_categorical(series: pd.Series) -> pd.Series:
    """Replace 'Not Provided' / 'Not Applicable' strings with NaN."""
    return series.replace(list(_NOT_PROVIDED_VALUES), pd.NA)
