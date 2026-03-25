# RBSA analysis scripts

Run scripts in order. Scripts 02–06 consume the parquet output of `01`.

| Script | Prompt | Description |
|--------|--------|-------------|
| `00_sanity_checks.py` | — | QC checks on curated outputs (classification counts, EUI ranges, duplicates) |
| `01_build_curated_mf_table.py` | 1–4 | Build master site table (SiteDetail + Usage + MechOL + EUI); classify HVAC/DHW at site and MF-building levels; output `rbsa_site_master_*.parquet` and `rbsa_mf_buildings_*.parquet` |
| `02_exploratory_distributions.py` | 5a | Box plots of Site EUI by system type; Heating_Zone confounding checks; summary stats CSV |
| `03_statistical_testing.py` | 5b | Mann-Whitney U (Central vs Distributed EUI); rank-biserial effect size; stratified by Primary_Heating_Fuel_Type |
| `04_climate_vintage_controlled.py` | 6 | OLS regression (EUI ~ system_type + Heating_Zone + Home_Vintage + Conditioned_Area); within-zone Mann-Whitney; vintage scatter plots |
| `05_fuel_breakdown.py` | 5c | Stacked bar charts [Electric &#124; Gas] by system type; fuel summary CSVs; per-fuel box plots split by electric vs gas homes |
| `06_compare_with_fm.py` | 7 | Side-by-side comparison vs 2023 FM MF Survey — NOT merged; see structural differences note in output |

## Typical workflow

```bash
# Step 1: Build curated tables
python analysis/rbsa/01_build_curated_mf_table.py \
    --data-dir path/to/2022_RBSA_Datasets/ \
    --outdir outputs/rbsa

# Step 0: QC the output
python analysis/rbsa/00_sanity_checks.py \
    --curated outputs/rbsa/rbsa_site_master_*.parquet \
    --table-type site_master

python analysis/rbsa/00_sanity_checks.py \
    --curated outputs/rbsa/rbsa_mf_buildings_*.parquet \
    --table-type mf_buildings

# Steps 2–5: Analysis (all take --site-master from step 1)
# --building-type: 'mf' = multifamily only (default), 'sf' = single-family only, 'all' = no filter
python analysis/rbsa/02_exploratory_distributions.py \
    --site-master outputs/rbsa/rbsa_site_master_*.parquet \
    --outdir outputs/rbsa --building-type mf

python analysis/rbsa/03_statistical_testing.py \
    --site-master outputs/rbsa/rbsa_site_master_*.parquet \
    --outdir outputs/rbsa --building-type mf

python analysis/rbsa/04_climate_vintage_controlled.py \
    --site-master outputs/rbsa/rbsa_site_master_*.parquet \
    --outdir outputs/rbsa --building-type mf

python analysis/rbsa/05_fuel_breakdown.py \
    --site-master outputs/rbsa/rbsa_site_master_*.parquet \
    --outdir outputs/rbsa --building-type mf

# Step 6: Cross-dataset comparison
python analysis/rbsa/06_compare_with_fm.py \
    --site-master outputs/rbsa/rbsa_site_master_*.parquet \
    --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/rbsa --building-type mf
```

## Edit system classification rules in

- `src/datasets/rbsa/classify.py` — site-level: `CENTRAL_HEATING`, `DISTRIBUTED_HEATING`,
  `CENTRAL_COOLING`, `DISTRIBUTED_COOLING`, `_DISTRIBUTED_DHW_TECHS`
- `src/datasets/rbsa/classify.py` — MF building-level: `classify_hvac()`, `classify_dhw()`
