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
    recs/           # 2020 RECS national household survey [implemented]
    cbecs/          # CBECS commercial                    [placeholder]
analysis/
  fm_mf_survey/     # analysis scripts 01–06
  rbsa/             # analysis scripts 00–06
  recs/             # analysis scripts 01–08 + run_workflow.py
tests/              # pytest suite (44 tests passing)
FINDINGS.md         # living summary of cross-dataset results
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

### RECS 2020 (Residential Energy Consumption Survey)

> **Status: Implemented**

**Source:** EIA 2020 RECS microdata — national household-level survey
**Geography:** National (all 50 states + DC)
**Sample:** 2,764 multifamily households (TYPEHUQ 3–4)
**Key fields:** HEATAPT (central flag), EQUIPM (equipment type), IECC climate zone, YEARMADERANGE, TOTSQFT_EN, NWEIGHT

**Source modules:**
```
src/datasets/recs/
    ingest.py    — load RECS CSV, sentinel replacement, EUI computation
    classify.py  — vectorized heating/cooling/DHW classifiers (Central/Distributed/Mixed/Unknown)
    utils.py     — shared helpers (load_curated, filter_unit_type)
```

**Quick start:**
```bash
pip install -e ".[dev]"

# Build curated table
python analysis/recs/01_build_curated_table.py \
    --data data/recs/recs2020_public_v7.csv \
    --outdir outputs/recs

# Run full pipeline (steps 02–06)
python analysis/recs/run_workflow.py \
    --curated outputs/recs/recs2020_curated_*.parquet \
    --outdir outputs/recs

# Presentation plots
python analysis/recs/07_presentation_plots.py

# Cross-dataset comparison (RECS vs RBSA)
python analysis/recs/08_cross_dataset_comparison.py \
    --recs-ols outputs/recs/04_ols_results.csv \
    --recs-mw  outputs/recs/03_mann_whitney_results.csv \
    --rbsa-ols outputs/rbsa/04_ols_results.csv \
    --rbsa-mw  outputs/rbsa/03_mann_whitney_results.csv \
    --outdir   outputs/recs
```

See [`analysis/recs/README.md`](analysis/recs/README.md) for the full script sequence and classification rules.

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Classification rules live here — edit to adjust system labels:
src/datasets/fm_mf_survey/classify.py
src/datasets/rbsa/classify.py
src/datasets/recs/classify.py
```

Raw data files are `.gitignore`d. Only code and analysis outputs are versioned.

---

## Important caveats

- **"Not Provided" ≠ NaN** — the FM survey uses both; both must be filtered.
- **Do NOT merge datasets** — FM MF Survey, RBSA, and RECS have different classification schemes, geographies, and survey vintages. Script 06/08 does side-by-side comparison only.
- **Non-parametric tests** — distributions are non-normal; use Mann-Whitney U, not t-tests.
- **Small samples** — FM energy subset ~40–107 properties; RBSA has only 11–35 Central heating MF units. All outputs label sample sizes.
- **Fuel confounding** — gas-heated distributed buildings can have *higher* EUI; always check fuel-stratified results alongside overall comparisons.
- **Observational data** — no causal claims. System type is confounded with building age, size, fuel type, and income.

See [`FINDINGS.md`](FINDINGS.md) for a living summary of cross-dataset results and interpretation.
