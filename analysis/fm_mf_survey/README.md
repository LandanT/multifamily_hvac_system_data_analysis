# 2023 Multifamily Survey — HVAC/DHW Energy Analysis

End-to-end analysis of HVAC and DHW system type (central vs. distributed)
vs. energy consumption, using the 2023 Multifamily Survey dataset.

## Data

| File | Description |
|------|-------------|
| `2023_Multifamily_Survey_dataset_FINAL.xlsx` | Main dataset (2,273 properties, sheet `Database`, `header=1`) |
| `2023_Multifamily_Survey_Data_Dictionary.xlsx` | Field definitions |

> **Critical loading note**: Row 0 of the Excel sheet contains Excel column
> letters (A, B, C…). Use `pd.read_excel(..., header=1)` to skip it.

## Source modules

```
src/datasets/fm_mf_survey/
    ingest.py    — load_mf_survey(), load_energy_subset(), helpers
    classify.py  — classify_dhw(), classify_heating(), classify_cooling(),
                   add_system_classifications(), to_binary()
```

## Analysis scripts

| Script | Prompt | What it does |
|--------|--------|--------------|
| `01_load_and_classify.py` | 1 | Load data, apply classifications, print coverage summary |
| `02_exploratory_distributions.py` | 2 | Box plots, confounding charts, cross-tabs, summary stats |
| `03_statistical_testing.py` | 3 | Mann-Whitney U tests (Central vs. Distributed), effect sizes |
| `04_climate_controlled.py` | 4 | OLS regression + within-region stratification |
| `05_fuel_breakdown.py` | 5 | Fuel-mix stacked bar charts by system type |
| `06_compare_with_rbsa.py` | 6 | Side-by-side comparison with RBSA dataset |

## Running

```bash
# Install dependencies first
pip install -e ".[dev]"   # or: pip install openpyxl scipy matplotlib statsmodels

# Prompt 1 — classification summary
python analysis/fm_mf_survey/01_load_and_classify.py \
    --data /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey

# Prompt 2 — exploratory distributions
python analysis/fm_mf_survey/02_exploratory_distributions.py \
    --data /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey

# Prompt 3 — statistical tests
python analysis/fm_mf_survey/03_statistical_testing.py \
    --data /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey

# Prompt 4 — climate-controlled comparison
python analysis/fm_mf_survey/04_climate_controlled.py \
    --data /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey

# Prompt 5 — fuel breakdown
python analysis/fm_mf_survey/05_fuel_breakdown.py \
    --data /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey

# Prompt 6 — side-by-side comparison with RBSA
python analysis/fm_mf_survey/06_compare_with_rbsa.py \
    --mf-data  /path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --rbsa-zip /path/to/rbsa_data.zip \
    --outdir   outputs/fm_mf_survey
```

## Classification logic

### DHW (`dhw_system_type`)

| Label | Condition |
|-------|-----------|
| `Central` | `Hot Water Heating Location == 'Central plant supplying multiple units'` |
| `Distributed` | `Hot Water Heating Location == 'In-unit system'` |
| `Central (inferred)` | System is "Combined with heating boiler" / "Central steam HX" / "Tankless central" |
| `Distributed (inferred)` | System is tank standard/condensing/HP or point-of-use tankless |
| `Unknown` | All else |

### Heating (`heating_system_type`)

| Label | Condition |
|-------|-----------|
| `Central` | `Type of Heating Plant` is non-null and not "Not Provided" |
| `Distributed` | No plant + equipment is mini-split / PTHP / portable electric |
| `Central (inferred)` | No plant + equipment is radiator / baseboard / fan coil / ducted |
| `Unknown` | All else |

### Cooling (`cooling_system_type`)

| Label | Condition |
|-------|-----------|
| `Central` | `Type of Cooling Plant` is non-null and not "Not Provided" |
| `Distributed` | No plant + equipment is mini-split / window AC / PTAC / split system |
| `Central (inferred)` | No plant + equipment is chilled water fan coil or ducted CHW AC |
| `Unknown` | All else |

## Important caveats

1. **"Not Provided" ≠ NaN** — always filter both.
2. **Small samples** — only ~40–107 properties have classifiable system types
   within the energy-analysis subset. All outputs label sample sizes.
3. **Non-parametric tests only** — Mann-Whitney U (not t-tests).
4. **Do NOT merge with RBSA** — different classification schemes, geography,
   and survey vintage.
