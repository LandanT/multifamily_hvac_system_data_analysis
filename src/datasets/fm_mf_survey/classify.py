"""HVAC/DHW system classification for the 2023 Multifamily Survey dataset.

Three derived classification columns are produced:

- ``dhw_system_type``      : Central | Distributed | Central (inferred) |
                             Distributed (inferred) | Unknown
- ``heating_system_type``  : Central | Distributed | Central (inferred) | Unknown
- ``cooling_system_type``  : Central | Distributed | Central (inferred) | Unknown

Classification logic follows the rules documented in the project brief.
Critically:
  * "Not Provided" is NOT the same as NaN — both must be treated as missing.
  * We first trust explicit survey responses, then fall back to inferred rules.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------

_MISSING = {"Not Provided", "Not Applicable", "N/A", "Not Available", "nan", ""}


def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() in _MISSING


def _val(row: pd.Series, col: str):
    """Return row[col] or None if the column doesn't exist."""
    if col not in row.index:
        return None
    return row[col]


# ---------------------------------------------------------------------------
# DHW classification
# ---------------------------------------------------------------------------

_DHW_CENTRAL_INFERRED = {
    "Combined with heating boiler",
    "Central steam heat exchanger",
    "Tankless hot water heater, central",
}

_DHW_DISTRIBUTED_INFERRED = {
    "Tank hot water heater, standard",
    "Tank hot water heater, condensing",
    "Tank hot water heater, heat pump",
    "Tankless hot water heater, point-of-use",
}


def classify_dhw(row: pd.Series) -> str:
    """Classify DHW system as Central / Distributed / inferred / Unknown."""
    loc = _val(row, "Hot Water Heating Location")
    sys = _val(row, "Residential Hot Water System")

    # Explicit survey response takes priority
    if not _is_missing(loc):
        if str(loc).strip() == "Central plant supplying multiple units":
            return "Central"
        if str(loc).strip() == "In-unit system":
            return "Distributed"

    # Fall back to system-type inference
    if not _is_missing(sys):
        sys_str = str(sys).strip()
        if sys_str in _DHW_CENTRAL_INFERRED:
            return "Central (inferred)"
        if sys_str in _DHW_DISTRIBUTED_INFERRED:
            return "Distributed (inferred)"

    return "Unknown"


# ---------------------------------------------------------------------------
# Heating classification
# ---------------------------------------------------------------------------

_HEATING_DISTRIBUTED_EQUIP = {
    "Ductless split system/mini-split",
    "Through-wall heat pump or PTHP",
    "Portable electric heaters",
}

_HEATING_CENTRAL_INFERRED_EQUIP = {
    "Ducted central heating",
    "Fan coil (not ducted) or PTAC",
    "Radiator or baseboard",
}


def classify_heating(row: pd.Series) -> str:
    """Classify space heating as Central / Distributed / Central (inferred) / Unknown."""
    plant = _val(row, "Type of Heating Plant")
    equip = _val(row, "Residential Heating Equipment")

    # A non-missing heating plant means central generation
    if not _is_missing(plant):
        return "Central"

    if not _is_missing(equip):
        equip_str = str(equip).strip()
        if equip_str in _HEATING_DISTRIBUTED_EQUIP:
            return "Distributed"
        if equip_str in _HEATING_CENTRAL_INFERRED_EQUIP:
            return "Central (inferred)"

    return "Unknown"


# ---------------------------------------------------------------------------
# Cooling classification
# ---------------------------------------------------------------------------

_COOLING_DISTRIBUTED_EQUIP = {
    "Ductless split system/mini-split",
    "Window A/C",
    "Through-the-wall A/C",
    "Electric PTAC or PTHP",
    "Ducted central AC or heat pump, split system, air-cooled",
    "Ducted central AC or heat pump, split system, water-cooled",
}

_COOLING_CENTRAL_INFERRED_EQUIP = {
    "Chilled water fan coil (not ducted) or PTAC",
    "Ducted central AC, chilled water",
}


def classify_cooling(row: pd.Series) -> str:
    """Classify cooling as Central / Distributed / Central (inferred) / Unknown."""
    plant = _val(row, "Type of Cooling Plant")
    equip = _val(row, "Residential Cooling Equipment")

    # A non-missing cooling plant means central chiller
    if not _is_missing(plant):
        return "Central"

    if not _is_missing(equip):
        equip_str = str(equip).strip()
        if equip_str in _COOLING_DISTRIBUTED_EQUIP:
            return "Distributed"
        if equip_str in _COOLING_CENTRAL_INFERRED_EQUIP:
            return "Central (inferred)"

    return "Unknown"


# ---------------------------------------------------------------------------
# Convenience: add all three columns at once
# ---------------------------------------------------------------------------


def add_system_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``dhw_system_type``, ``heating_system_type``, ``cooling_system_type`` columns.

    Parameters
    ----------
    df:
        DataFrame (typically the energy-analysis subset) from
        :func:`src.datasets.fm_mf_survey.ingest.load_mf_survey`.

    Returns
    -------
    pd.DataFrame
        Same dataframe with three new classification columns appended.
    """
    df = df.copy()
    df["dhw_system_type"] = df.apply(classify_dhw, axis=1)
    df["heating_system_type"] = df.apply(classify_heating, axis=1)
    df["cooling_system_type"] = df.apply(classify_cooling, axis=1)
    return df


# ---------------------------------------------------------------------------
# Helpers for collapsing inferred labels into binary Central / Distributed
# ---------------------------------------------------------------------------

_CENTRAL_LABELS = {"Central", "Central (inferred)"}
_DISTRIBUTED_LABELS = {"Distributed", "Distributed (inferred)"}


def to_binary(series: pd.Series) -> pd.Series:
    """Map classification labels to 'Central', 'Distributed', or NaN."""
    def _map(v):
        if v in _CENTRAL_LABELS:
            return "Central"
        if v in _DISTRIBUTED_LABELS:
            return "Distributed"
        return pd.NA

    return series.map(_map)


def classifiable_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask: True where ``col`` is not 'Unknown'."""
    return df[col] != "Unknown"
