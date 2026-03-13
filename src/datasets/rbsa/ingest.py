"""RBSA ingestion utilities (zip or directory).

Loads all relevant RBSA CSV files into a typed dataclass:

Site-level (one row per site):
  - SiteDetail.csv        → site_detail
  - Usage_One_Line.csv    → usage_one_line
  - Mechanical_One_Line.csv → mech_one_line

Equipment-level (one row per piece of equipment — must aggregate before joining):
  - Mechanical_HeatingAndCooling.csv → mech_hc
  - Mechanical_WaterHeater.csv       → mech_wh

Multifamily building-level (separate analytical population, keyed on Building_ID):
  - MF_Building_Site_Detail.csv   → mf_site_detail
  - MF_Building_HVAC.csv          → mf_hvac
  - MF_Building_WaterHeating.csv  → mf_dhw
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd

from src.common.schema import require_columns


@dataclass(frozen=True)
class RbsaInputs:
    # Site-level (one row per SiteID)
    site_detail: pd.DataFrame
    usage_one_line: pd.DataFrame
    mech_one_line: pd.DataFrame

    # Equipment-level (one row per equipment unit; SiteID is FK)
    mech_hc: pd.DataFrame   # Mechanical_HeatingAndCooling.csv
    mech_wh: pd.DataFrame   # Mechanical_WaterHeater.csv

    # MF building-level (one row per Building_ID — separate population)
    mf_site_detail: pd.DataFrame
    mf_hvac: pd.DataFrame
    mf_dhw: pd.DataFrame


def _read_csv_from_zip(z: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    with z.open(member_name) as f:
        return pd.read_csv(f, low_memory=False)


def _find_member(z: zipfile.ZipFile, filename: str) -> str:
    candidates = [
        n for n in z.namelist()
        if n.lower().endswith("/" + filename.lower()) or n.lower() == filename.lower()
    ]
    if not candidates:
        candidates = [n for n in z.namelist() if n.lower().endswith(filename.lower())]
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} in zip")
    candidates.sort(key=lambda s: (len(s), s))
    return candidates[0]


def _validate_inputs(inputs: RbsaInputs) -> None:
    require_columns(inputs.site_detail, ["SiteID"], "SiteDetail.csv")
    require_columns(inputs.usage_one_line, ["SiteID"], "Usage_One_Line.csv")
    require_columns(inputs.mech_one_line, ["SiteID"], "Mechanical_One_Line.csv")
    require_columns(inputs.mech_hc, ["SiteID"], "Mechanical_HeatingAndCooling.csv")
    require_columns(inputs.mech_wh, ["SiteID"], "Mechanical_WaterHeater.csv")
    require_columns(inputs.mf_site_detail, ["Building_ID"], "MF_Building_Site_Detail.csv")
    require_columns(inputs.mf_hvac, ["Building_ID"], "MF_Building_HVAC.csv")
    require_columns(inputs.mf_dhw, ["Building_ID"], "MF_Building_WaterHeating.csv")


def load_rbsa_from_zip(zip_path: Path) -> RbsaInputs:
    with zipfile.ZipFile(zip_path, "r") as z:
        site = _read_csv_from_zip(z, _find_member(z, "SiteDetail.csv"))
        usage = _read_csv_from_zip(z, _find_member(z, "Usage_One_Line.csv"))
        mol = _read_csv_from_zip(z, _find_member(z, "Mechanical_One_Line.csv"))
        hc = _read_csv_from_zip(z, _find_member(z, "Mechanical_HeatingAndCooling.csv"))
        wh = _read_csv_from_zip(z, _find_member(z, "Mechanical_WaterHeater.csv"))
        mf_site = _read_csv_from_zip(z, _find_member(z, "MF_Building_Site_Detail.csv"))
        mf_hvac = _read_csv_from_zip(z, _find_member(z, "MF_Building_HVAC.csv"))
        mf_dhw = _read_csv_from_zip(z, _find_member(z, "MF_Building_WaterHeating.csv"))

    inputs = RbsaInputs(site, usage, mol, hc, wh, mf_site, mf_hvac, mf_dhw)
    _validate_inputs(inputs)
    return inputs


def load_rbsa_from_dir(data_dir: Path) -> RbsaInputs:
    site = pd.read_csv(data_dir / "SiteDetail.csv", low_memory=False)
    usage = pd.read_csv(data_dir / "Usage_One_Line.csv", low_memory=False)
    mol = pd.read_csv(data_dir / "Mechanical_One_Line.csv", low_memory=False)
    hc = pd.read_csv(data_dir / "Mechanical_HeatingAndCooling.csv", low_memory=False)
    wh = pd.read_csv(data_dir / "Mechanical_WaterHeater.csv", low_memory=False)
    mf_site = pd.read_csv(data_dir / "MF_Building_Site_Detail.csv", low_memory=False)
    mf_hvac = pd.read_csv(data_dir / "MF_Building_HVAC.csv", low_memory=False)
    mf_dhw = pd.read_csv(data_dir / "MF_Building_WaterHeating.csv", low_memory=False)

    inputs = RbsaInputs(site, usage, mol, hc, wh, mf_site, mf_hvac, mf_dhw)
    _validate_inputs(inputs)
    return inputs
