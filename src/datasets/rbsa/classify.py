"""RBSA HVAC/DHW system classification.

Two separate classification layers are provided:

1. **Site-level** (from Mechanical_One_Line.csv + Mechanical_WaterHeater.csv):
   - ``classify_site_heating(system_type)``  → Central | Distributed | Unknown
   - ``classify_site_cooling(system_type)``  → Central | Distributed | Unknown
   - ``classify_site_dhw_row(row)``          → per water-heater record label
   - ``aggregate_site_dhw(wh_df)``           → one label per SiteID
   - ``add_site_classifications(master_df, wh_df)`` → add columns to master table

2. **MF building-level** (from MF_Building_HVAC.csv / MF_Building_WaterHeating.csv):
   - ``classify_hvac(row)``  → SystemClassification (central|distributed|mixed|unknown)
   - ``classify_dhw(row)``   → SystemClassification (central|distributed|mixed|unknown)

MF classification logic
-----------------------
The MF questionnaire captures Yes/No flags for equipment observed in common areas
and in tenant units, plus surveyor hints ("Likely the Same"). Classification uses an
equipment-aware approach rather than treating all COMMON/TENANT columns equally:

- **"Likely the Same" = Yes**: surveyor confirmed a single system serves both common
  and tenant spaces → Central (strongest signal, checked first).
- **Definitive central equipment** (COMMON columns implying building-wide service):
  boilers, central AC/HP, WSHP loop, cooling tower, AHU/RTU.
- **Common-area-only equipment** (COMMON corridors/lobby heaters that do NOT imply
  tenant central service): baseboard, wall heater, unit heater, fireplace, PTAC/PTHP.
- **Definitive distributed equipment** (TENANT columns implying per-unit systems):
  baseboard, PTAC/PTHP, furnace, mini-split, through-wall, window/portable units.
- **DHW top-level flags**: ``Common Areas Have Hot Water`` and
  ``Tenant Units Have Water Heating Equipment`` resolve buildings where no specific
  equipment type was recorded.  Common=Yes + Tenant=No → Central;
  Common=Yes + Tenant=Yes → Mixed.
- If the MF questionnaire yields "unknown", callers may fall back to the site-level
  classification from Mechanical_One_Line.csv via the Building_ID → SiteID bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Site-level system type sets (from Mechanical_One_Line.Primary_*_System_Type)
# ---------------------------------------------------------------------------

CENTRAL_HEATING: frozenset[str] = frozenset({
    "Furnace",
    "Central Air Source Heat Pump",
    "Boiler",
    "Geothermal Heat Pump",
    "Water Source Heat Pump",
})

DISTRIBUTED_HEATING: frozenset[str] = frozenset({
    "Baseboard",
    "Ductless Mini-split Heat Pump (indoor)",
    "Wall Heater",
    "Unit Heater",
    "PTAC Packaged Terminal Air Conditioner",
    "PTHP Packaged Terminal Heat Pump",
    "Through-wall Heat Pump",
    "Radiant Heating",
    "Portable Heater",
    "Fireplace or Stove",
})

CENTRAL_COOLING: frozenset[str] = frozenset({
    "Central AC",
    "Central Air Source Heat Pump",
    "Geothermal Heat Pump",
    "Water Source Heat Pump",
})

DISTRIBUTED_COOLING: frozenset[str] = frozenset({
    "Ductless Mini-split Heat Pump (indoor)",
    "Ductless Mini-split AC (indoor)",
    "Window AC",
    "Through-wall AC",
    "PTAC Packaged Terminal Air Conditioner",
    "PTHP Packaged Terminal Heat Pump",
    "Portable AC",
    "Evaporative Cooler",
    "Portable Heat Pump",
})

# DHW technology types that are considered in-unit distributed
_DISTRIBUTED_DHW_TECHS: frozenset[str] = frozenset({
    "Electric Resistance",
    "Fossil Fuel Non-Condensing",
    "Fossil Fuel Condensing",
    "Electric Heat Pump (Packaged)",
    "Electric Heat Pump (Split System)",
})


# ---------------------------------------------------------------------------
# Site-level HVAC classification
# ---------------------------------------------------------------------------

def classify_site_heating(system_type) -> str:
    """Map Primary_Heating_System_Type to 'Central', 'Distributed', or 'Unknown'."""
    if pd.isna(system_type):
        return "Unknown"
    s = str(system_type).strip()
    if s in CENTRAL_HEATING:
        return "Central"
    if s in DISTRIBUTED_HEATING:
        return "Distributed"
    return "Unknown"


def classify_site_cooling(system_type) -> str:
    """Map Primary_Cooling_System_Type to 'Central', 'Distributed', or 'Unknown'."""
    if pd.isna(system_type):
        return "Unknown"
    s = str(system_type).strip()
    if s in CENTRAL_COOLING:
        return "Central"
    if s in DISTRIBUTED_COOLING:
        return "Distributed"
    return "Unknown"


# ---------------------------------------------------------------------------
# Site-level DHW classification
# ---------------------------------------------------------------------------

def classify_site_dhw_row(row: pd.Series) -> str:
    """Classify a single Mechanical_WaterHeater.csv record."""
    tech = row.get("Technology_Description")
    provided_by_heating = row.get("Provided_by_Heating_System")

    provided_str = "" if pd.isna(provided_by_heating) else str(provided_by_heating).strip()
    tech_str = "" if pd.isna(tech) else str(tech).strip()

    if provided_str == "Yes":
        return "Combined with Heating System"
    if tech_str == "Indirect Water Heater":
        return "Central (Indirect/Boiler-fed)"
    if tech_str in _DISTRIBUTED_DHW_TECHS:
        return "Distributed (In-unit)"
    return "Unknown"


def aggregate_site_dhw(wh_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Mechanical_WaterHeater.csv to one DHW classification per SiteID.

    Priority order per site:
      1. Whole-house serving unit (Serves_Whole_House == 'Yes')
      2. First record if no whole-house flag

    Returns a DataFrame with columns [SiteID, dhw_system_type].
    """
    wh = wh_df.copy()
    wh["_dhw_label"] = wh.apply(classify_site_dhw_row, axis=1)

    # Prefer the record that serves the whole house
    whole_house = wh[wh.get("Serves_Whole_House", pd.Series(dtype=str)) == "Yes"] if "Serves_Whole_House" in wh.columns else pd.DataFrame()

    if len(whole_house):
        primary = whole_house.groupby("SiteID")["_dhw_label"].first().reset_index()
    else:
        primary = wh.groupby("SiteID")["_dhw_label"].first().reset_index()

    return primary.rename(columns={"_dhw_label": "dhw_system_type"})


def add_site_classifications(master_df: pd.DataFrame, wh_df: pd.DataFrame) -> pd.DataFrame:
    """Add heating_system_type, cooling_system_type, and dhw_system_type to master_df.

    Parameters
    ----------
    master_df:
        Master site table that already has Primary_Heating_System_Type and
        Primary_Cooling_System_Type (joined from Mechanical_One_Line.csv).
    wh_df:
        Mechanical_WaterHeater.csv DataFrame (one row per water heater).

    Returns
    -------
    pd.DataFrame
        Copy of master_df with three new classification columns.
    """
    df = master_df.copy()

    if "Primary_Heating_System_Type" in df.columns:
        df["heating_system_type"] = df["Primary_Heating_System_Type"].map(classify_site_heating)
    else:
        df["heating_system_type"] = "Unknown"

    if "Primary_Cooling_System_Type" in df.columns:
        df["cooling_system_type"] = df["Primary_Cooling_System_Type"].map(classify_site_cooling)
    else:
        df["cooling_system_type"] = "Unknown"

    dhw_per_site = aggregate_site_dhw(wh_df)
    df = df.merge(dhw_per_site, on="SiteID", how="left")
    df["dhw_system_type"] = df["dhw_system_type"].fillna("Unknown")

    # Normalize verbose DHW labels to binary Central/Distributed so that
    # downstream analysis scripts (steps 02–05) can filter on the same
    # labels used for heating/cooling classification.
    _DHW_LABEL_MAP = {
        "Distributed (In-unit)": "Distributed",
        "Combined with Heating System": "Central",
        "Central (Indirect/Boiler-fed)": "Central",
    }
    df["dhw_system_type"] = df["dhw_system_type"].replace(_DHW_LABEL_MAP)

    return df


# ---------------------------------------------------------------------------
# MF building-level classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemClassification:
    label: str   # central | distributed | mixed | unknown
    reason: str


def _is_yes(v) -> bool:
    """Return True only for explicit affirmative values (Yes/1/True). Unknown → False."""
    if v is None:
        return False
    if isinstance(v, (int, float)) and pd.isna(v):
        return False
    return str(v).strip().lower() in ("1", "y", "yes", "true", "t")


# HVAC equipment columns that indicate a building-wide central plant
# (common-area equipment that distributes conditioning to tenant units)
_HVAC_CENTRAL_COLS: tuple[str, ...] = (
    "COMMON Boiler",
    "COMMON Central AC",
    "COMMON Central Air Source Heat Pump",
    "COMMON Water Source Heat Pump",
    "COMMON Tower",
    "COMMON AHU/RTU Heating",
    "COMMON AHU/RTU Cooling",
    "COMMON MAU",
)

# HVAC equipment columns that indicate per-unit distributed systems in tenant spaces
_HVAC_DISTRIBUTED_COLS: tuple[str, ...] = (
    "TENANT Baseboard",
    "TENANT Wall Heater",
    "TENANT Unit Heater",
    "TENANT PTAC Packaged Terminal Air Conditioner",
    "TENANT PTHP Packaged Terminal Heat Pump",
    "TENANT Furnace",
    "TENANT Ductless Mini-split Heat Pump",
    "TENANT Through-Wall AC",
    "TENANT Through-Wall HP",
    "TENANT Window AC",
    "TENANT Portable AC",
    "TENANT Portable Electric Heater",
    "TENANT Portable Fossil Fuel Heater",
    "TENANT Portable Heat Pump",
    "TENANT Central AC",
    "TENANT Central Air Source Heat Pump",
    "TENANT AHU/RTU Cooling",
    "TENANT Boiler",
    "TENANT Water Source Heat Pump",
)

# DHW equipment columns that indicate a building-wide central DHW plant
_DHW_CENTRAL_COLS: tuple[str, ...] = (
    "COMMON Fossil Fuel Condensing Boiler",
    "COMMON Fossil Fuel Non-Condensing Boiler",
    "COMMON Electric Resistance Boiler",
    "COMMON Fossil Fuel Condensing Storage",
    "COMMON Fossil Fuel Non-Condensing Storage",
    "COMMON Electric Resistance Storage",
    "COMMON Fossil Fuel Condensing Instantaneous",
    "COMMON Electric Resistance Instantaneous",
    "COMMON Converter/Indirect Water Heater",
)

# DHW equipment columns that indicate per-unit distributed water heating
_DHW_DISTRIBUTED_COLS: tuple[str, ...] = (
    "TENANT Fossil Fuel Condensing Storage",
    "TENANT Fossil Fuel Non-Condensing Storage",
    "TENANT Electric Resistance Storage",
    "TENANT Fossil Fuel Condensing Instantaneous",
    "TENANT Electric Resistance Instantaneous",
    "TENANT Converter/Indirect Water Heater",
    "TENANT Fossil Fuel Condensing Boiler",
    "TENANT Fossil Fuel Non-Condensing Boiler",
    "TENANT Electric Resistance Boiler",
)


def _check_cols(row: pd.Series, cols: tuple[str, ...]) -> list[str]:
    """Return list of cols present in row that have explicit Yes values."""
    return [c for c in cols if c in row.index and _is_yes(row[c])]


def classify_hvac(row: pd.Series) -> SystemClassification:
    """Classify MF building-level HVAC from a MF_Building_HVAC.csv row.

    Decision order:
    1. "Likely the Same" hint = Yes → Central (surveyor confirmed single system)
    2. Definitive central equipment (boiler, central AC/HP, WSHP, tower, AHU/RTU)
    3. Definitive distributed equipment (per-unit PTAC/baseboard/mini-split/etc.)
    4. central-only → Central; distributed-only → Distributed; both → Mixed; neither → Unknown
    """
    # Step 1: Surveyor hint — "Likely the Same" = Yes is the strongest signal
    heat_hint = str(row.get("Common and Tenant Heating Likely the Same", "")).strip().lower()
    cool_hint = str(row.get("Common and Tenant Cooling Likely the Same", "")).strip().lower()
    if heat_hint == "yes" or cool_hint == "yes":
        hints = []
        if heat_hint == "yes":
            hints.append("heating_likely_same")
        if cool_hint == "yes":
            hints.append("cooling_likely_same")
        return SystemClassification("central", f"hint:{','.join(hints)}")

    # Step 2 & 3: Equipment evidence
    central_evidence = _check_cols(row, _HVAC_CENTRAL_COLS)
    distributed_evidence = _check_cols(row, _HVAC_DISTRIBUTED_COLS)

    if central_evidence and not distributed_evidence:
        return SystemClassification("central", f"equipment:{central_evidence[0]}")
    if distributed_evidence and not central_evidence:
        return SystemClassification("distributed", f"equipment:{distributed_evidence[0]}")
    if central_evidence and distributed_evidence:
        return SystemClassification("mixed", f"central:{central_evidence[0]}+distributed:{distributed_evidence[0]}")
    return SystemClassification("unknown", "no definitive evidence")


def classify_dhw(row: pd.Series) -> SystemClassification:
    """Classify MF building-level DHW from a MF_Building_WaterHeating.csv row.

    Decision order:
    1. "Likely the Same" hint = Yes → Central
    2. "Tenant In-Unit Water Heater Included" = Yes → Distributed or Mixed
    3. "Common Areas Have Hot Water" = Yes + "Tenant Units Have Water Heating Equipment" = No → Central
    4. "Common Areas Have Hot Water" = Yes + "Tenant Units Have Water Heating Equipment" = Yes → Mixed
       (check equipment first; fall back to mixed if no specific equipment found)
    5. Definitive central DHW equipment (common boilers, storage, instantaneous)
    6. Definitive distributed DHW equipment (tenant-unit water heaters)
    7. central-only → Central; distributed-only → Distributed; both → Mixed; neither → Unknown
    """
    # Step 1: Surveyor hints
    hints = (
        "Common and Tenant Water Heating Likely the Same",
        "Common and Tenant Water Heating the Same",
    )
    for hint_col in hints:
        if str(row.get(hint_col, "")).strip().lower() == "yes":
            return SystemClassification("central", f"hint:{hint_col}")

    # Step 2: Explicit "in-unit water heater included" flag
    if _is_yes(row.get("Tenant In-Unit Water Heater Included")):
        distributed_evidence = _check_cols(row, _DHW_DISTRIBUTED_COLS)
        central_evidence = _check_cols(row, _DHW_CENTRAL_COLS)
        if central_evidence:
            return SystemClassification("mixed", f"in-unit+central:{central_evidence[0]}")
        return SystemClassification("distributed", "Tenant In-Unit Water Heater Included=Yes")

    # Step 3 & 4: Top-level area flags
    common_has_hw = _is_yes(row.get("Common Areas Have Hot Water"))
    tenant_has_wh = _is_yes(row.get("Tenant Units Have Water Heating Equipment"))

    if common_has_hw and not tenant_has_wh:
        return SystemClassification("central", "Common Areas Have Hot Water=Yes, no tenant equipment")

    if common_has_hw and tenant_has_wh:
        central_evidence = _check_cols(row, _DHW_CENTRAL_COLS)
        distributed_evidence = _check_cols(row, _DHW_DISTRIBUTED_COLS)
        if central_evidence and distributed_evidence:
            return SystemClassification("mixed", f"common_hw+tenant_wh:{central_evidence[0]}+{distributed_evidence[0]}")
        if central_evidence:
            return SystemClassification("mixed", f"common_hw+tenant_wh:central={central_evidence[0]}")
        if distributed_evidence:
            return SystemClassification("mixed", f"common_hw+tenant_wh:distributed={distributed_evidence[0]}")
        return SystemClassification("mixed", "Common Areas Have Hot Water=Yes + Tenant Units Have Water Heating Equipment=Yes")

    # Step 5 & 6: Equipment evidence
    central_evidence = _check_cols(row, _DHW_CENTRAL_COLS)
    distributed_evidence = _check_cols(row, _DHW_DISTRIBUTED_COLS)

    if central_evidence and not distributed_evidence:
        return SystemClassification("central", f"equipment:{central_evidence[0]}")
    if distributed_evidence and not central_evidence:
        return SystemClassification("distributed", f"equipment:{distributed_evidence[0]}")
    if central_evidence and distributed_evidence:
        return SystemClassification("mixed", f"central:{central_evidence[0]}+distributed:{distributed_evidence[0]}")
    return SystemClassification("unknown", "no definitive evidence")


def add_mf_building_classifications(
    mf_hvac: pd.DataFrame,
    mf_dhw: pd.DataFrame,
) -> pd.DataFrame:
    """Classify MF buildings and return a table keyed on Building_ID.

    Returns a DataFrame with columns:
      Building_ID, hvac_system_type, hvac_reason, dhw_system_type, dhw_reason
    """
    hvac_cls = mf_hvac[["Building_ID"]].copy()
    hvac_cls["hvac_system_type"] = mf_hvac.apply(lambda r: classify_hvac(r).label, axis=1)
    hvac_cls["hvac_reason"] = mf_hvac.apply(lambda r: classify_hvac(r).reason, axis=1)

    dhw_cls = mf_dhw[["Building_ID"]].copy()
    dhw_cls["dhw_system_type"] = mf_dhw.apply(lambda r: classify_dhw(r).label, axis=1)
    dhw_cls["dhw_reason"] = mf_dhw.apply(lambda r: classify_dhw(r).reason, axis=1)

    return hvac_cls.merge(dhw_cls, on="Building_ID", how="outer")
