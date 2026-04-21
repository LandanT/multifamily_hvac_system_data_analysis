# RECS 2020 analysis scripts

Run scripts in order. Scripts 02–06 consume the parquet output of `01`.

| Script | Description |
|--------|-------------|
| `01_build_curated_table.py` | Load RECS microdata → classify HVAC/DHW → filter valid energy rows → output `recs2020_curated_YYYYMMDD.parquet` |
| `02_exploratory_distributions.py` | Box plots of Site EUI by system type; IECC climate zone and vintage confounding checks; summary stats CSV |
| `03_statistical_testing.py` | Mann-Whitney U (Central vs Distributed EUI); rank-biserial effect size; stratified by ELWARM / UGWARM fuel flag |
| `04_climate_vintage_controlled.py` | OLS regression (EUI ~ system_type + IECC_climate_code + YEARMADERANGE + log(TOTSQFT_EN)); within-climate-zone Mann-Whitney; vintage scatter plots |
| `05_fuel_breakdown.py` | Stacked bar charts of household fuel-use share (heating + water heating) by system class |
| `06_compare_with_other.py` | Side-by-side comparison vs FM MF Survey and/or RBSA — NOT merged; see structural differences note in output |

## Typical workflow

```bash
# Step 1: Place the RECS 2020 public-use file in data/recs/
# Download from: https://www.eia.gov/consumption/residential/data/2020/index.php?view=microdata

# Step 2: Build curated table (multifamily only by default)
python analysis/recs/01_build_curated_table.py \
    --data data/recs/recs2020_public_v7.csv \
    --outdir outputs/recs

# Step 3: Exploratory distributions
# --unit-type: 'mf' = multifamily only (default), 'sf' = single-family, 'all' = no filter
python analysis/recs/02_exploratory_distributions.py \
    --curated outputs/recs/recs2020_curated_*.parquet \
    --outdir outputs/recs --unit-type mf

# Step 4: Statistical testing (Mann-Whitney U)
python analysis/recs/03_statistical_testing.py \
    --curated outputs/recs/recs2020_curated_*.parquet \
    --outdir outputs/recs --unit-type mf

# Step 5: Climate/vintage-controlled comparison
python analysis/recs/04_climate_vintage_controlled.py \
    --curated outputs/recs/recs2020_curated_*.parquet \
    --outdir outputs/recs --unit-type mf

# Step 6: Fuel breakdown
python analysis/recs/05_fuel_breakdown.py \
    --curated outputs/recs/recs2020_curated_*.parquet \
    --outdir outputs/recs --unit-type mf

# Step 7: Cross-dataset comparison (provide at least one comparison dataset)
python analysis/recs/06_compare_with_other.py \
    --recs-curated outputs/recs/recs2020_curated_*.parquet \
    --fm-data path/to/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --rbsa-dir path/to/2022_RBSA_Datasets/ \
    --outdir outputs/recs
```

## Edit system classification rules in

- `src/datasets/recs/classify.py` — equipment code sets:
  - `_HEATING_DISTRIBUTED_EQUIP` / `_HEATING_CENTRAL_EQUIP` (EQUIPM codes)
  - `_COOLING_CENTRAL_EQUIP` / `_COOLING_DISTRIBUTED_EQUIP` (ACEQUIPM_PUB codes)

## Key assumptions and caveats

### Housing unit type filter
- **Default: `--unit-type mf`** retains TYPEHUQ 3 and 4 (apartments in buildings with 2+ units).
- This is consistent with the FM/RBSA multifamily focus and ensures that `HEATAPT` / `COOLAPT`
  (the primary central vs. distributed signals) are meaningful.
- Single-family homes (TYPEHUQ 1–2) have `HEATAPT = -2` (not applicable), which converts to NaN
  and triggers equipment-type inference. Use `--unit-type sf` or `--unit-type all` explicitly.

### Missing value codes
- RECS uses `-2` ("not applicable") and `-1` ("missing") as numeric sentinels.
  Both are replaced with `NaN` in `ingest.py` and treated identically in classification
  (both trigger the inference path from EQUIPM / ACEQUIPM_PUB).

### No unit conversion
- RECS `TOTALBTU` is already in thousand Btu.  No conversion is applied.
  FM/RBSA pipelines convert kWh / therms to kBtu; RECS does not need this.

### Heating EUI denominator
- The spec references `TOTHSQFT` (heated area) for `Heating_EUI_kBtu_sqft`.
  If `TOTHSQFT` is absent from the CSV, `TOTSQFT_EN` (total energy-consuming floor area) is used
  as a fallback and documented in the column comment in `ingest.py`.

### Survey weights
- `NWEIGHT` (sample weight) is retained in all outputs.
  Initial analyses are unweighted for simplicity.
  For nationally representative estimates, apply `NWEIGHT` — see script 04 `--use-weights` flag.
  Official EIA RECS tables are always weighted; unweighted MF subsets may differ.

### Cross-dataset comparisons
- RECS, RBSA, and FM use different classification schemes, geographic scopes, and survey vintages.
  **Do not merge these datasets.**  Script 06 presents results side-by-side only.
- RECS 2020 covers the full US; RBSA covers the Pacific Northwest; FM covers national MF properties.
  Differences in results reflect both real physical differences and methodological differences.
