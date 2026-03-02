"""Unit conversions and derived energy metrics."""

from __future__ import annotations

KWH_TO_KBTU = 3.412141633
THERM_TO_KBTU = 100.0  # 1 therm = 100 kBtu


def kwh_to_kbtu(kwh: float) -> float:
    return kwh * KWH_TO_KBTU


def therm_to_kbtu(therms: float) -> float:
    return therms * THERM_TO_KBTU


def kbtu_total(electric_kwh: float | None = None, gas_therms: float | None = None) -> float | None:
    if electric_kwh is None and gas_therms is None:
        return None
    total = 0.0
    if electric_kwh is not None:
        total += kwh_to_kbtu(float(electric_kwh))
    if gas_therms is not None:
        total += therm_to_kbtu(float(gas_therms))
    return total


def eui_kbtu_per_ft2(total_kbtu: float | None, floor_area_ft2: float | None) -> float | None:
    if total_kbtu is None or floor_area_ft2 in (None, 0):
        return None
    return float(total_kbtu) / float(floor_area_ft2)
