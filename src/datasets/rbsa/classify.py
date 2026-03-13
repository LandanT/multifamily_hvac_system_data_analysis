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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

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

    return df


# ---------------------------------------------------------------------------
# MF building-level classification helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemClassification:
    label: str   # central | distributed | mixed | unknown
    reason: str


def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)) and pd.isna(v):
        return False
    s = str(v).strip().lower()
    if s in ("1", "y", "yes", "true", "t"):
        return True
    if s in ("0", "n", "no", "false", "f", "", "nan", "none"):
        return False
    try:
        return float(s) != 0.0
    except Exception:
        return True


def evidence_from_prefix(row: pd.Series, prefixes: Iterable[str]) -> Tuple[bool, list[str]]:
    cols = []
    for c in row.index:
        cu = c.upper()
        if any(cu.startswith(p) for p in prefixes):
            if _truthy(row[c]):
                cols.append(c)
    return (len(cols) > 0, cols)


def classify_system_from_common_tenant(
    row: pd.Series,
    common_prefixes: Iterable[str],
    tenant_prefixes: Iterable[str],
    extra_hint_cols: Optional[Iterable[str]] = None,
) -> SystemClassification:
    common_has, common_cols = evidence_from_prefix(row, common_prefixes)
    tenant_has, tenant_cols = evidence_from_prefix(row, tenant_prefixes)

    hint_cols = []
    if extra_hint_cols:
        for hc in extra_hint_cols:
            if hc in row.index and _truthy(row[hc]):
                hint_cols.append(hc)

    if common_has and tenant_has:
        return SystemClassification("mixed", f"common({len(common_cols)})+tenant({len(tenant_cols)})")
    if common_has and not tenant_has:
        suffix = f"+hints({','.join(hint_cols)})" if hint_cols else ""
        return SystemClassification("central", f"common({len(common_cols)}){suffix}")
    if tenant_has and not common_has:
        if hint_cols:
            return SystemClassification("mixed", f"tenant({len(tenant_cols)})+hints({','.join(hint_cols)})")
        return SystemClassification("distributed", f"tenant({len(tenant_cols)})")
    if hint_cols:
        return SystemClassification("central", f"hints({','.join(hint_cols)})")
    return SystemClassification("unknown", "no evidence")


def classify_hvac(row: pd.Series) -> SystemClassification:
    """Classify MF building-level HVAC from MF_Building_HVAC.csv row."""
    common_prefixes = ("COMMON", "COM ", "COMMON_")
    tenant_prefixes = ("TENANT", "TEN ", "TENANT_")
    hint_cols = (
        "Common and Tenant Heating Likely the Same",
        "Common and Tenant Cooling Likely the Same",
    )
    return classify_system_from_common_tenant(row, common_prefixes, tenant_prefixes, hint_cols)


def classify_dhw(row: pd.Series) -> SystemClassification:
    """Classify MF building-level DHW from MF_Building_WaterHeating.csv row."""
    common_prefixes = ("COMMON", "COM ", "COMMON_")
    tenant_prefixes = ("TENANT", "TEN ", "TENANT_")
    hint_cols = (
        "Common and Tenant Water Heating Likely the Same",
        "Common and Tenant Water Heating the Same",
    )
    return classify_system_from_common_tenant(row, common_prefixes, tenant_prefixes, hint_cols)


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
