"""Unit tests for RECS 2020 classification logic.

Tests cover all classification branches plus the to_binary() helper.
Each test constructs a minimal pd.Series representing one RECS household row
with only the columns relevant to the function under test.

Key invariant: explicit survey indicators (HEATAPT, COOLAPT, H2OAPT)
always take priority over equipment-type inference (EQUIPM, ACEQUIPM_PUB).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.datasets.recs.classify import (
    classify_dhw,
    classify_heating,
    classify_cooling,
    to_binary,
    add_system_classifications,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs) -> pd.Series:
    """Build a minimal row with NaN defaults for all expected columns."""
    defaults = {
        "HEATHOME": float("nan"),
        "HEATAPT": float("nan"),
        "EQUIPM": float("nan"),
        "AIRCOND": float("nan"),
        "COOLAPT": float("nan"),
        "ACEQUIPM_PUB": float("nan"),
        "H2OAPT": float("nan"),
        "MORETHAN1H2O": float("nan"),
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# Heating tests
# ---------------------------------------------------------------------------


class TestClassifyHeating:
    def test_no_heating_returns_unknown(self):
        row = _row(HEATHOME=0)
        assert classify_heating(row) == "Unknown"

    def test_no_heating_nan_returns_unknown(self):
        row = _row(HEATHOME=float("nan"))
        assert classify_heating(row) == "Unknown"

    def test_explicit_central_heatapt_1(self):
        row = _row(HEATHOME=1, HEATAPT=1)
        assert classify_heating(row) == "Central"

    def test_explicit_distributed_heatapt_0(self):
        row = _row(HEATHOME=1, HEATAPT=0)
        assert classify_heating(row) == "Distributed"

    def test_inferred_distributed_ductless(self):
        """EQUIPM=13 (ductless mini-split) → Distributed (inferred)."""
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=13)
        assert classify_heating(row) == "Distributed (inferred)"

    def test_inferred_distributed_built_in_electric(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=5)
        assert classify_heating(row) == "Distributed (inferred)"

    def test_inferred_distributed_built_in_gas(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=7)
        assert classify_heating(row) == "Distributed (inferred)"

    def test_inferred_distributed_wood_stove(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=8)
        assert classify_heating(row) == "Distributed (inferred)"

    def test_inferred_distributed_portable_electric(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=10)
        assert classify_heating(row) == "Distributed (inferred)"

    def test_inferred_central_furnace(self):
        """EQUIPM=3 (central furnace) → Central (inferred)."""
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=3)
        assert classify_heating(row) == "Central (inferred)"

    def test_inferred_central_steam_radiator(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=2)
        assert classify_heating(row) == "Central (inferred)"

    def test_inferred_central_heat_pump(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=4)
        assert classify_heating(row) == "Central (inferred)"

    def test_unknown_when_equipm_unknown_code(self):
        row = _row(HEATHOME=1, HEATAPT=float("nan"), EQUIPM=99)
        assert classify_heating(row) == "Unknown"

    def test_explicit_priority_over_inference(self):
        """HEATAPT=1 should win even if EQUIPM suggests distributed."""
        row = _row(HEATHOME=1, HEATAPT=1, EQUIPM=13)
        assert classify_heating(row) == "Central"

    def test_explicit_distributed_priority_over_central_inference(self):
        row = _row(HEATHOME=1, HEATAPT=0, EQUIPM=3)
        assert classify_heating(row) == "Distributed"


# ---------------------------------------------------------------------------
# Cooling tests
# ---------------------------------------------------------------------------


class TestClassifyCooling:
    def test_no_ac_returns_unknown(self):
        row = _row(AIRCOND=0)
        assert classify_cooling(row) == "Unknown"

    def test_no_ac_nan_returns_unknown(self):
        row = _row(AIRCOND=float("nan"))
        assert classify_cooling(row) == "Unknown"

    def test_explicit_central_coolapt_1(self):
        row = _row(AIRCOND=1, COOLAPT=1)
        assert classify_cooling(row) == "Central"

    def test_explicit_distributed_coolapt_0(self):
        row = _row(AIRCOND=1, COOLAPT=0)
        assert classify_cooling(row) == "Distributed"

    def test_inferred_central_acequipm_1(self):
        """ACEQUIPM_PUB=1 (central AC / central HP) → Central (inferred)."""
        row = _row(AIRCOND=1, COOLAPT=float("nan"), ACEQUIPM_PUB=1)
        assert classify_cooling(row) == "Central (inferred)"

    def test_inferred_distributed_ductless(self):
        """ACEQUIPM_PUB=3 (ductless mini-split) → Distributed (inferred)."""
        row = _row(AIRCOND=1, COOLAPT=float("nan"), ACEQUIPM_PUB=3)
        assert classify_cooling(row) == "Distributed (inferred)"

    def test_inferred_distributed_window_wall(self):
        row = _row(AIRCOND=1, COOLAPT=float("nan"), ACEQUIPM_PUB=4)
        assert classify_cooling(row) == "Distributed (inferred)"

    def test_inferred_distributed_portable(self):
        row = _row(AIRCOND=1, COOLAPT=float("nan"), ACEQUIPM_PUB=5)
        assert classify_cooling(row) == "Distributed (inferred)"

    def test_inferred_distributed_evaporative(self):
        row = _row(AIRCOND=1, COOLAPT=float("nan"), ACEQUIPM_PUB=6)
        assert classify_cooling(row) == "Distributed (inferred)"

    def test_unknown_when_acequipm_unknown_code(self):
        row = _row(AIRCOND=1, COOLAPT=float("nan"), ACEQUIPM_PUB=99)
        assert classify_cooling(row) == "Unknown"

    def test_explicit_priority_over_inference(self):
        """COOLAPT=1 wins even if ACEQUIPM_PUB suggests distributed."""
        row = _row(AIRCOND=1, COOLAPT=1, ACEQUIPM_PUB=4)
        assert classify_cooling(row) == "Central"


# ---------------------------------------------------------------------------
# DHW tests
# ---------------------------------------------------------------------------


class TestClassifyDhw:
    def test_central_h2oapt_1(self):
        row = _row(H2OAPT=1)
        assert classify_dhw(row) == "Central"

    def test_distributed_h2oapt_0(self):
        row = _row(H2OAPT=0)
        assert classify_dhw(row) == "Distributed"

    def test_mixed_morethan1_heater(self):
        """MORETHAN1H2O=1 with H2OAPT not central → Mixed."""
        row = _row(H2OAPT=float("nan"), MORETHAN1H2O=1)
        assert classify_dhw(row) == "Mixed"

    def test_central_takes_priority_over_morethan1(self):
        """H2OAPT=1 should win over MORETHAN1H2O=1."""
        row = _row(H2OAPT=1, MORETHAN1H2O=1)
        assert classify_dhw(row) == "Central"

    def test_unknown_when_h2oapt_missing(self):
        row = _row(H2OAPT=float("nan"), MORETHAN1H2O=0)
        assert classify_dhw(row) == "Unknown"

    def test_unknown_when_all_missing(self):
        row = _row()
        assert classify_dhw(row) == "Unknown"


# ---------------------------------------------------------------------------
# to_binary() tests
# ---------------------------------------------------------------------------


class TestToBinary:
    def test_central_maps_to_central(self):
        s = pd.Series(["Central"])
        assert to_binary(s).iloc[0] == "Central"

    def test_central_inferred_maps_to_central(self):
        s = pd.Series(["Central (inferred)"])
        assert to_binary(s).iloc[0] == "Central"

    def test_distributed_maps_to_distributed(self):
        s = pd.Series(["Distributed"])
        assert to_binary(s).iloc[0] == "Distributed"

    def test_distributed_inferred_maps_to_distributed(self):
        s = pd.Series(["Distributed (inferred)"])
        assert to_binary(s).iloc[0] == "Distributed"

    def test_unknown_maps_to_na(self):
        result = to_binary(pd.Series(["Unknown"]))
        assert pd.isna(result.iloc[0])

    def test_mixed_maps_to_na(self):
        result = to_binary(pd.Series(["Mixed"]))
        assert pd.isna(result.iloc[0])

    def test_mixed_series(self):
        s = pd.Series(["Central", "Distributed (inferred)", "Unknown", "Mixed", "Central (inferred)"])
        result = to_binary(s)
        assert result.iloc[0] == "Central"
        assert result.iloc[1] == "Distributed"
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == "Central"


# ---------------------------------------------------------------------------
# add_system_classifications() smoke test
# ---------------------------------------------------------------------------


class TestAddSystemClassifications:
    def test_adds_all_columns(self):
        df = pd.DataFrame([
            {
                "HEATHOME": 1, "HEATAPT": 1, "EQUIPM": float("nan"),
                "AIRCOND": 1, "COOLAPT": 0, "ACEQUIPM_PUB": float("nan"),
                "H2OAPT": 0, "MORETHAN1H2O": 0,
            }
        ])
        result = add_system_classifications(df)
        for col in [
            "heating_system_type", "cooling_system_type", "dhw_system_type",
            "heating_system_type_binary", "cooling_system_type_binary", "dhw_system_type_binary",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_values_correct(self):
        df = pd.DataFrame([
            {
                "HEATHOME": 1, "HEATAPT": 1, "EQUIPM": float("nan"),
                "AIRCOND": 1, "COOLAPT": 0, "ACEQUIPM_PUB": float("nan"),
                "H2OAPT": 0, "MORETHAN1H2O": 0,
            }
        ])
        result = add_system_classifications(df)
        assert result["heating_system_type"].iloc[0] == "Central"
        assert result["cooling_system_type"].iloc[0] == "Distributed"
        assert result["dhw_system_type"].iloc[0] == "Distributed"
        assert result["heating_system_type_binary"].iloc[0] == "Central"
        assert result["cooling_system_type_binary"].iloc[0] == "Distributed"
        assert result["dhw_system_type_binary"].iloc[0] == "Distributed"
