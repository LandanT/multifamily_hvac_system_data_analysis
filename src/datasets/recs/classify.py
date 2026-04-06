"""HVAC/DHW system classification for the RECS 2020 microdata.

Three derived classification columns are produced:

- ``heating_system_type``  : Central | Distributed | Central (inferred) |
                             Distributed (inferred) | Unknown
- ``cooling_system_type``  : Central | Distributed | Central (inferred) |
                             Distributed (inferred) | Unknown
- ``dhw_system_type``      : Central | Distributed | Mixed | Unknown

Binary variants collapse inferred labels:
- ``heating_system_type_binary``  : Central | Distributed | NaN
- ``cooling_system_type_binary``  : Central | Distributed | NaN
- ``dhw_system_type_binary``      : Central | Distributed | NaN

Classification logic
--------------------
RECS supplies explicit "serves multiple units" indicators (HEATAPT, COOLAPT,
H2OAPT) that take priority over equipment-type inference.  Values of -2 and -1
are replaced with NaN in :func:`src.datasets.recs.ingest.load_recs`, so
downstream code uses ``pd.isna()`` only — no string sentinel handling is needed
(unlike the FM pipeline which checks for "Not Provided" strings).

Equipment type codes (EQUIPM, ACEQUIPM_PUB) are integer values after ingest;
the classification sets below use integer literals accordingly.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Equipment type classification sets  (integer codes from RECS codebook)
# ---------------------------------------------------------------------------

# EQUIPM — main space-heating equipment type
_HEATING_DISTRIBUTED_EQUIP: frozenset[int] = frozenset({
    5,   # Built-in electric units (baseboard, etc.)
    7,   # Built-in gas / oil room heater
    8,   # Wood or pellet stove
    10,  # Portable electric heaters
    13,  # Ductless heat pump (mini-split)
})

_HEATING_CENTRAL_EQUIP: frozenset[int] = frozenset({
    2,  # Steam or hot-water system with radiators or pipes
    3,  # Central furnace
    4,  # Central heat pump
})

# ACEQUIPM_PUB — main air-conditioning equipment type
_COOLING_CENTRAL_EQUIP: frozenset[int] = frozenset({
    1,  # Central air conditioner (includes central heat pump)
})

_COOLING_DISTRIBUTED_EQUIP: frozenset[int] = frozenset({
    3,  # Ductless heat pump (mini-split)
    4,  # Window or wall air conditioner
    5,  # Portable air conditioner
    6,  # Evaporative cooler
})


# ---------------------------------------------------------------------------
# Heating classification
# ---------------------------------------------------------------------------


def classify_heating(row: pd.Series) -> str:
    """Classify space-heating system as Central / Distributed / inferred / Unknown.

    Decision order (highest priority first):
    1. No heating (HEATHOME != 1) → ``"Unknown"``
    2. HEATAPT == 1 → ``"Central"``   (explicit: equipment serves multiple units)
    3. HEATAPT == 0 → ``"Distributed"``   (explicit: each unit has own equipment)
    4. EQUIPM in distributed set → ``"Distributed (inferred)"``
    5. EQUIPM in central set     → ``"Central (inferred)"``
    6. Fallthrough → ``"Unknown"``
    """
    heathome = row.get("HEATHOME") if hasattr(row, "get") else row["HEATHOME"]
    heatapt = row.get("HEATAPT") if hasattr(row, "get") else row["HEATAPT"]
    equipm = row.get("EQUIPM") if hasattr(row, "get") else row["EQUIPM"]

    if pd.isna(heathome) or heathome != 1:
        return "Unknown"

    if not pd.isna(heatapt):
        if heatapt == 1:
            return "Central"
        if heatapt == 0:
            return "Distributed"

    # Inference via equipment type
    if not pd.isna(equipm):
        code = int(equipm)
        if code in _HEATING_DISTRIBUTED_EQUIP:
            return "Distributed (inferred)"
        if code in _HEATING_CENTRAL_EQUIP:
            return "Central (inferred)"

    return "Unknown"


# ---------------------------------------------------------------------------
# Cooling classification
# ---------------------------------------------------------------------------


def classify_cooling(row: pd.Series) -> str:
    """Classify cooling system as Central / Distributed / inferred / Unknown.

    Decision order:
    1. No AC (AIRCOND != 1) → ``"Unknown"``
    2. COOLAPT == 1 → ``"Central"``
    3. COOLAPT == 0 → ``"Distributed"``
    4. ACEQUIPM_PUB == 1 → ``"Central (inferred)"``
    5. ACEQUIPM_PUB in {3,4,5,6} → ``"Distributed (inferred)"``
    6. Fallthrough → ``"Unknown"``
    """
    aircond = row.get("AIRCOND") if hasattr(row, "get") else row["AIRCOND"]
    coolapt = row.get("COOLAPT") if hasattr(row, "get") else row["COOLAPT"]
    acequipm = row.get("ACEQUIPM_PUB") if hasattr(row, "get") else row["ACEQUIPM_PUB"]

    if pd.isna(aircond) or aircond != 1:
        return "Unknown"

    if not pd.isna(coolapt):
        if coolapt == 1:
            return "Central"
        if coolapt == 0:
            return "Distributed"

    # Inference via equipment type
    if not pd.isna(acequipm):
        code = int(acequipm)
        if code in _COOLING_CENTRAL_EQUIP:
            return "Central (inferred)"
        if code in _COOLING_DISTRIBUTED_EQUIP:
            return "Distributed (inferred)"

    return "Unknown"


# ---------------------------------------------------------------------------
# DHW classification
# ---------------------------------------------------------------------------


def classify_dhw(row: pd.Series) -> str:
    """Classify domestic hot-water system as Central / Distributed / Mixed / Unknown.

    Decision order:
    1. H2OAPT == 1 → ``"Central"``  (explicit: serves multiple units)
    2. MORETHAN1H2O == 1 (and H2OAPT not explicitly central) → ``"Mixed"``
    3. H2OAPT == 0 → ``"Distributed"``
    4. H2OAPT missing / NaN → ``"Unknown"``

    Note: no equipment-type inference is defined for DHW in the RECS microdata
    beyond the H2OAPT indicator and the MORETHAN1H2O flag.
    """
    h2oapt = row.get("H2OAPT") if hasattr(row, "get") else row["H2OAPT"]
    morethan1 = row.get("MORETHAN1H2O") if hasattr(row, "get") else row["MORETHAN1H2O"]

    if not pd.isna(h2oapt) and h2oapt == 1:
        return "Central"

    if not pd.isna(morethan1) and morethan1 == 1:
        # Multiple heaters present and not explicitly central → Mixed
        return "Mixed"

    if not pd.isna(h2oapt) and h2oapt == 0:
        return "Distributed"

    return "Unknown"


# ---------------------------------------------------------------------------
# Convenience: add all three columns at once
# ---------------------------------------------------------------------------


def add_system_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Add classification columns for heating, cooling and DHW.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`src.datasets.recs.ingest.load_recs`.

    Returns
    -------
    pd.DataFrame
        Copy with six new columns appended:
        ``heating_system_type``, ``cooling_system_type``, ``dhw_system_type``
        and their ``_binary`` variants.
    """
    df = df.copy()
    df["heating_system_type"] = df.apply(classify_heating, axis=1)
    df["cooling_system_type"] = df.apply(classify_cooling, axis=1)
    df["dhw_system_type"] = df.apply(classify_dhw, axis=1)

    df["heating_system_type_binary"] = to_binary(df["heating_system_type"])
    df["cooling_system_type_binary"] = to_binary(df["cooling_system_type"])
    df["dhw_system_type_binary"] = to_binary(df["dhw_system_type"])

    return df


# ---------------------------------------------------------------------------
# Helpers for collapsing inferred labels into binary Central / Distributed
# ---------------------------------------------------------------------------

_CENTRAL_LABELS: frozenset[str] = frozenset({"Central", "Central (inferred)"})
_DISTRIBUTED_LABELS: frozenset[str] = frozenset({"Distributed", "Distributed (inferred)"})


def to_binary(series: pd.Series) -> pd.Series:
    """Map classification labels to ``'Central'``, ``'Distributed'``, or ``pd.NA``.

    ``"Mixed"`` and ``"Unknown"`` become ``pd.NA`` (not classifiable as binary).
    """
    def _map(v):
        if v in _CENTRAL_LABELS:
            return "Central"
        if v in _DISTRIBUTED_LABELS:
            return "Distributed"
        return pd.NA

    return series.map(_map)


def classifiable_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask: True where ``col`` is not ``'Unknown'``."""
    return df[col] != "Unknown"
