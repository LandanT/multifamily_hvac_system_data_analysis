# Building Systems & Energy Use Analysis

Repeatable workflows to classify **central vs. distributed HVAC/DHW** systems and compare **energy consumption** across residential and multifamily building datasets.

## Overview

Each dataset goes through the same pipeline:

1. **Ingest** — load raw data, coerce types, filter missing values
2. **Classify** — label heating, cooling, and DHW systems as `Central`, `Distributed`, `Mixed`, or `Unknown`
3. **Explore** — box plots, cross-tabs, confounding checks (climate zone, vintage, fuel type)
4. **Test** — Mann-Whitney U tests (Central vs. Distributed EUI); rank-biserial effect sizes
5. **Control** — OLS regression (EUI ~ system type + climate + vintage + size)
6. **Fuel breakdown** — stacked bar charts by fuel type and system type
7. **Cross-dataset compare** — side-by-side summaries (datasets are never merged)

## Repository layout

```
src/
  common/           # shared utilities (logging, unit conversion, schema validation)
  datasets/
    fm_mf_survey/   # 2023 Fannie Mae Multifamily Survey  [implemented]
    rbsa/           # 2022 RBSA residential/multifamily   [implemented]
    cbecs/          # CBECS commercial                    [placeholder]
analysis/
  fm_mf_survey/     # analysis scripts 01–06
  rbsa/             # analysis scripts 00–06
tests/              # pytest tests (in progress)
```

---

## Datasets

### FM Multifamily Survey 2023

> **Status: Implemented**

**Source:** 2023 Fannie Mae Multifamily Survey (`2023_Multifamily_Survey_dataset_FINAL.xlsx`, 2,273 properties)
**Geography:** National multifamily portfolio
**Key fields:** HVAC/DHW equipment type, site/source EUI, fuel breakdown, HDD/CDD, building size

> **Note for future users:** The FM energy-analysis subset is thin (~40–107 classifiable properties per system type). Treat FM outputs as directional reference only; do not draw statistical conclusions from FM comparisons alone until a larger energy-data subset is available.

**Source modules:**
```
src/datasets/fm_mf_survey/
    ingest.py    — load_mf_survey(), load_energy_subset()
    classify.py  — classify_dhw(), classify_heating(), classify_cooling()
```

**Quick start:**
```bash
pip install -e ".[dev]"

python analysis/fm_mf_survey/01_load_and_classify.py \
    --data /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey
```

See [`analysis/fm_mf_survey/README.md`](analysis/fm_mf_survey/README.md) for the full script sequence and classification logic.

---

### RBSA 2022

> **Status: Implemented**

**Source:** 2022 Residential Building Stock Assessment — Pacific Northwest
**Geography:** Pacific Northwest (WA, OR, ID, MT)
**Key fields:** Primary heating/cooling system type, water heater records, site EUI, Heating Zone, vintage

**Source modules:**
```
src/datasets/rbsa/
    ingest.py    — load_rbsa_site_master(), load_rbsa_mf_buildings()
    classify.py  — site-level and MF-building-level classification rules
```

**Quick start:**
```bash
pip install -e ".[dev]"

# Build curated tables first
python analysis/rbsa/01_build_curated_mf_table.py \
    --data-dir /path/to/2022_RBSA_Datasets/ \
    --outdir outputs/rbsa

# Then run analysis scripts 02–06 (all read the parquet from step 1)
python analysis/rbsa/02_exploratory_distributions.py \
    --site-master outputs/rbsa/rbsa_site_master_*.parquet \
    --outdir outputs/rbsa
```

See [`analysis/rbsa/README.md`](analysis/rbsa/README.md) for the full script sequence and classification rules.

---

### RECS (Residential Energy Consumption Survey)

> **Status: Planned**

EIA's national household-level survey. Intended to extend the central vs. distributed HVAC/DHW analysis to a nationally representative residential sample.

When implemented, it will follow the same pipeline pattern:
```
src/datasets/recs/
    ingest.py    — (to be added)
    classify.py  — (to be added)
analysis/recs/
    01_build_curated_table.py
    02_exploratory_distributions.py
    ...
```

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Classification rules live here — edit to adjust system labels:
src/datasets/fm_mf_survey/classify.py
src/datasets/rbsa/classify.py
```

Raw data files are `.gitignore`d. Only code and analysis outputs are versioned.

---

## Important caveats

- **"Not Provided" ≠ NaN** — the FM survey uses both; both must be filtered.
- **Do NOT merge datasets** — FM MF Survey, RBSA, and future RECS have different classification schemes, geographies, and survey vintages. Script 06 in each analysis folder does side-by-side comparison only.
- **Non-parametric tests** — distributions are non-normal; use Mann-Whitney U, not t-tests.
- **Small samples** — the FM energy-analysis subset yields ~40–107 classifiable properties per system type; all outputs label sample sizes.
