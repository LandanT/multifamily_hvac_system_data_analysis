"""RBSA multifamily HVAC/DHW system classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import pandas as pd


@dataclass(frozen=True)
class SystemClassification:
    label: str
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
        return SystemClassification("central", f"common({len(common_cols)})" + (f"+hints({','.join(hint_cols)})" if hint_cols else ""))
    if tenant_has and not common_has:
        if hint_cols:
            return SystemClassification("mixed", f"tenant({len(tenant_cols)})+hints({','.join(hint_cols)})")
        return SystemClassification("distributed", f"tenant({len(tenant_cols)})")
    if hint_cols:
        return SystemClassification("central", f"hints({','.join(hint_cols)})")
    return SystemClassification("unknown", "no evidence")


def classify_hvac(row: pd.Series) -> SystemClassification:
    common_prefixes = ("COMMON", "COM ", "COMMON_")
    tenant_prefixes = ("TENANT", "TEN ", "TENANT_")
    hint_cols = (
        "Common and Tenant Heating Likely the Same",
        "Common and Tenant Cooling Likely the Same",
    )
    return classify_system_from_common_tenant(row, common_prefixes, tenant_prefixes, hint_cols)


def classify_dhw(row: pd.Series) -> SystemClassification:
    common_prefixes = ("COMMON", "COM ", "COMMON_")
    tenant_prefixes = ("TENANT", "TEN ", "TENANT_")
    hint_cols = (
        "Common and Tenant Water Heating Likely the Same",
        "Common and Tenant Water Heating the Same",
    )
    return classify_system_from_common_tenant(row, common_prefixes, tenant_prefixes, hint_cols)
