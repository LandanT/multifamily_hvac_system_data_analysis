"""RBSA ingestion utilities (zip or directory)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd

from src.common.schema import require_columns


@dataclass(frozen=True)
class RbsaInputs:
    site_detail: pd.DataFrame
    usage_one_line: pd.DataFrame
    mf_hvac: pd.DataFrame
    mf_dhw: pd.DataFrame


def _read_csv_from_zip(z: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    with z.open(member_name) as f:
        return pd.read_csv(f, low_memory=False)


def _find_member(z: zipfile.ZipFile, filename: str) -> str:
    candidates = [n for n in z.namelist() if n.lower().endswith("/" + filename.lower()) or n.lower() == filename.lower()]
    if not candidates:
        candidates = [n for n in z.namelist() if n.lower().endswith(filename.lower())]
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} in zip")
    candidates.sort(key=lambda s: (len(s), s))
    return candidates[0]


def load_rbsa_from_zip(zip_path: Path) -> RbsaInputs:
    with zipfile.ZipFile(zip_path, "r") as z:
        site = _read_csv_from_zip(z, _find_member(z, "SiteDetail.csv"))
        usage = _read_csv_from_zip(z, _find_member(z, "Usage_One_Line.csv"))
        hvac = _read_csv_from_zip(z, _find_member(z, "MF_Building_HVAC.csv"))
        dhw = _read_csv_from_zip(z, _find_member(z, "MF_Building_WaterHeating.csv"))

    require_columns(site, ["SiteID", "Building_ID"], "SiteDetail.csv")
    require_columns(usage, ["SiteID"], "Usage_One_Line.csv")
    require_columns(hvac, ["Building_ID"], "MF_Building_HVAC.csv")
    require_columns(dhw, ["Building_ID"], "MF_Building_WaterHeating.csv")
    return RbsaInputs(site, usage, hvac, dhw)


def load_rbsa_from_dir(data_dir: Path) -> RbsaInputs:
    site = pd.read_csv(data_dir / "SiteDetail.csv", low_memory=False)
    usage = pd.read_csv(data_dir / "Usage_One_Line.csv", low_memory=False)
    hvac = pd.read_csv(data_dir / "MF_Building_HVAC.csv", low_memory=False)
    dhw = pd.read_csv(data_dir / "MF_Building_WaterHeating.csv", low_memory=False)

    require_columns(site, ["SiteID", "Building_ID"], "SiteDetail.csv")
    require_columns(usage, ["SiteID"], "Usage_One_Line.csv")
    require_columns(hvac, ["Building_ID"], "MF_Building_HVAC.csv")
    require_columns(dhw, ["Building_ID"], "MF_Building_WaterHeating.csv")
    return RbsaInputs(site, usage, hvac, dhw)
