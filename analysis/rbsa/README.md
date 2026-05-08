# RBSA 2022 Analysis Scripts

The RBSA (Residential Building Stock Assessment) is a Pacific Northwest field-inspection
dataset maintained by NEEA. Unlike RECS (self-reported survey), RBSA equipment data comes
from on-site inspections, giving higher classification confidence but smaller sample sizes.

Run scripts in order. Scripts 02–06 consume the Parquet output of `01`.

## Data Model

Step 01 produces **two** curated tables (both as Parquet + CSV):

| Output | Unit of analysis | Key columns | Typical MF rows |
|--------|-----------------|-------------|-----------------|
| `rbsa_site_master_*.parquet` | One row per **SiteID** | Site EUI, electric/gas kBtu, heating/cooling/DHW classification, Heating Zone, vintage | ~370 |
| `rbsa_mf_buildings_*.parquet` | One row per **Building_ID** (within a site) | MF-building-level HVAC and DHW classifications | Varies |

**Important:** Multifamily system fields are at the **Building_ID** level, while energy usage
is at the **SiteID** level. The site_master table is the primary analysis table for EUI
comparisons. Both identifiers are retained in outputs.

## Script Inventory

| Script | Description |
|--------|-------------|
| `00_sanity_checks.py` | QC checks on curated outputs: classification counts, EUI ranges, duplicate detection. Supports `--table-type site_master` or `mf_buildings` |
| `01_build_curated_mf_table.py` | Join SiteDetail + Usage_One_Line + Mechanical_One_Line → master table; classify HVAC/DHW at site and MF-building levels; output two Parquets |
| `02_exploratory_distributions.py` | Box plots of Site EUI by system type; Heating Zone confounding checks; cross-tabs; summary stats CSV |
| `03_statistical_testing.py` | Mann-Whitney U (Central vs Distributed EUI); rank-biserial effect size; stratified by `Primary_Heating_Fuel_Type` |
| `04_climate_vintage_controlled.py` | OLS regression (`EUI ~ system_type + Heating_Zone + Home_Vintage + Conditioned_Area`); within-zone Mann-Whitney; vintage scatter plots |
| `05_fuel_breakdown.py` | Stacked bar charts [Electric \| Gas] by system type; fuel summary CSVs; per-fuel box plots split by electric- vs gas-primary homes |
| `06_compare_with_fm.py` | Side-by-side comparison vs 2023 FM MF Survey — NOT merged; documents structural differences |
| `run_workflow.py` | Sequential runner for steps 00→01→02→03→04→05→06; supports `--skip-steps`, `--dry-run` |

## Typical Workflow

### Option A: Use the workflow runner (recommended)

```bash
# Full pipeline from raw CSVs
python analysis/rbsa/run_workflow.py \
    --data-dir data/rbsa/ \
    --outdir outputs/rbsa

# Skip sanity checks
python analysis/rbsa/run_workflow.py \
    --data-dir data/rbsa/ \
    --outdir outputs/rbsa \
    --skip-steps 00

# From a zip file, with FM cross-comparison
python analysis/rbsa/run_workflow.py \
    --zip path/to/rbsa_2022.zip \
    --outdir outputs/rbsa \
    --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx
```

### Option B: Run scripts individually

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

`src/datasets/rbsa/classify.py` contains two classification layers:

- **Site-level classifiers** — map equipment names via lookup sets:
  - `CENTRAL_HEATING`, `DISTRIBUTED_HEATING` (e.g., "Furnace" → Central, "Baseboard" → Distributed)
  - `CENTRAL_COOLING`, `DISTRIBUTED_COOLING`
  - `_DISTRIBUTED_DHW_TECHS` (e.g., "Standard Tank" → Distributed)
- **MF building-level classifiers** — use binary indicator columns from `MF_Building_HVAC`/`MF_Building_WaterHeating`:
  - `classify_hvac()` — checks `_HVAC_CENTRAL_COLS` vs `_HVAC_DISTRIBUTED_COLS`
  - `classify_dhw()` — checks `_DHW_CENTRAL_COLS` vs `_DHW_DISTRIBUTED_COLS`

## RBSA Data Files Required

The RBSA loader expects these 8 CSV files in the data directory:

| File | Purpose |
|------|---------|
| `SiteDetail.csv` | Site metadata (building type, vintage, heating zone, area) |
| `Usage_One_Line.csv` | Annual energy consumption by site |
| `Mechanical_One_Line.csv` | Primary equipment type per site |
| `Mechanical_HeatingAndCooling.csv` | Detailed heating/cooling equipment records |
| `Mechanical_WaterHeater.csv` | Water heater equipment records |
| `MF_Building_Site_Detail.csv` | MF building-level site info |
| `MF_Building_HVAC.csv` | MF building-level HVAC indicators |
| `MF_Building_WaterHeating.csv` | MF building-level DHW indicators |

## Key Limitations

- **Small Central heating sample:** Only ~11–35 Central heating units in MF. CIs are wide; treat results as directional.
- **Pacific NW only:** Mild climate, less heating load variance than national datasets. Heating Zones 1–3 only.
- **Field-inspected ≠ self-reported:** Classification confidence is higher than RECS, but the datasets should not be directly merged.
