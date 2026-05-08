# Building Systems & Energy Use Analysis

Repeatable workflows to classify **central vs. distributed HVAC/DHW** systems and compare **energy consumption** across multifamily building datasets.

## What This Project Does

This repo answers one core question: **Do multifamily buildings with central
HVAC/DHW systems use more or less energy than those with distributed (in-unit)
systems?** We analyze three independent datasets (RECS, RBSA, FM Multifamily
Survey), classify each building's heating, cooling, and domestic hot water
system as `Central`, `Distributed`, `Mixed`, or `Unknown`, and then compare
energy use intensities (EUI, in kBtu/sqft/yr) using statistical tests and
regression models.

Key outputs include:
- Mann-Whitney U tests comparing median EUI between system types
- OLS regressions controlling for climate zone, building vintage, and size
- Fuel-stratified analysis (electric-only vs. gas-only buildings)
- Climate-zone-level breakdowns
- Presentation-quality charts

See [`FINDINGS.md`](FINDINGS.md) for the summary of results across all datasets.

## Analysis Pipeline

Each dataset goes through the same 7-step pipeline (numbered scripts in `analysis/`):

| Step | Script prefix | Purpose |
|------|--------------|---------|
| 1 | `01_` | **Ingest & classify** — load raw data, apply system-type labels, output a curated Parquet file |
| 2 | `02_` | **Explore** — box plots, cross-tabs, confounding checks (climate zone, vintage, fuel type) |
| 3 | `03_` | **Test** — Mann-Whitney U tests (Central vs. Distributed EUI); rank-biserial effect sizes |
| 4 | `04_` | **Control** — OLS regression (EUI ~ system type + climate + vintage + size); within-zone tests |
| 5 | `05_` | **Fuel breakdown** — stacked bar charts of fuel mix by system type |
| 6 | `06_` | **Cross-dataset compare** — side-by-side summaries (datasets are never merged) |
| 7 | `07_`/`08_` | **Presentation plots & cross-dataset forest plots** (RECS only) |

Scripts are designed to be run in order. Steps 02+ consume the Parquet output from step 01.

## Repository Layout

```
├── README.md                 # ← you are here
├── FINDINGS.md               # living summary of cross-dataset results
├── WHAT_CHANGED_RECS.md      # changelog for the RECS heating-refocus iteration
├── pyproject.toml            # package metadata & dependencies
│
├── src/                      # reusable library code (imported by analysis scripts)
│   ├── common/               #   logging, unit conversion, schema validation
│   │   ├── log.py            #     get_logger(name) → configured Python logger
│   │   ├── schema.py         #     require_columns(df, cols) — fail-fast validation
│   │   └── units.py          #     kWh↔kBtu, therm↔kBtu, EUI calculators
│   └── datasets/
│       ├── fm_mf_survey/     #   2023 Fannie Mae MF Survey loader + classifiers
│       │   ├── ingest.py     #     load_mf_survey(), load_energy_subset()
│       │   └── classify.py   #     classify_dhw/heating/cooling(), to_binary()
│       ├── rbsa/             #   2022 RBSA loader + classifiers
│       │   ├── ingest.py     #     load_rbsa_from_dir(), load_rbsa_from_zip()
│       │   └── classify.py   #     site-level + MF-building-level classifiers
│       ├── recs/             #   2020 RECS loader + classifiers
│       │   ├── ingest.py     #     load_recs(), sentinel replacement, EUI derivation
│       │   ├── classify.py   #     vectorized heating/cooling/DHW classifiers
│       │   └── utils.py      #     load_curated(), filter helpers, segment constants
│       ├── cbecs/            #   (placeholder — not yet implemented)
│       └── building_data/    #   (placeholder — not yet implemented)
│
├── analysis/                 # numbered runnable scripts (one folder per dataset)
│   ├── fm_mf_survey/         #   scripts 01–06
│   ├── rbsa/                 #   scripts 00–06 + run_workflow.py
│   └── recs/                 #   scripts 01–08 + run_workflow.py
│
├── data/                     # raw source data (gitignored — you must supply these)
│   ├── fm_mf_survey/         #   2023_Multifamily_Survey_dataset_FINAL.xlsx
│   ├── rbsa/                 #   2022 RBSA CSVs (SiteDetail, Mechanical_*, etc.)
│   └── recs/                 #   recs2020_public_v7.csv (EIA download)
│
├── outputs/                  # generated artifacts: parquets, PNGs, CSVs (gitignored)
│   ├── fm_mf_survey/
│   ├── rbsa/
│   └── recs/
│
└── tests/                    # pytest suite — 44 tests for RECS classifiers
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.9
- Raw data files (not included — see each dataset section below for download links)

### Installation

```bash
git clone https://github.com/LandanT/multifamily_hvac_system_data_analysis.git
cd multifamily_hvac_system_data_analysis
pip install -e .
```

This installs `pandas`, `numpy`, `pyarrow`, `openpyxl`, `scipy`, `matplotlib`, and `statsmodels`.

For running tests you also need `pytest`:

```bash
pip install pytest
pytest tests/ -v          # 44 tests, all should pass
```

### Data Setup

Raw data files go in `data/` (gitignored). You need to obtain them separately:

| Dataset | Where to get it | Place it in |
|---------|----------------|-------------|
| RECS 2020 | [EIA RECS microdata](https://www.eia.gov/consumption/residential/data/2020/index.php?view=microdata) | `data/recs/recs2020_public_v7.csv` |
| RBSA 2022 | NEEA (request access) | `data/rbsa/` (extract CSVs into this folder) |
| FM MF Survey 2023 | Fannie Mae (internal) | `data/fm_mf_survey/2023_Multifamily_Survey_dataset_FINAL.xlsx` |

---

## Datasets

### RECS 2020 (Residential Energy Consumption Survey)

> **Status: Implemented** — most complete pipeline

**Source:** EIA 2020 RECS microdata — national household-level survey
**Geography:** National (all 50 states + DC)
**Sample:** 2,764 multifamily households (TYPEHUQ 3–4)
**Key fields:** `HEATAPT` (central flag), `EQUIPM` (equipment type), IECC climate zone, `YEARMADERANGE`, `TOTSQFT_EN`, `NWEIGHT`

**Source modules:**
```
src/datasets/recs/
    ingest.py    — load RECS CSV, sentinel replacement (-2/-1 → NaN), EUI computation
    classify.py  — vectorized heating/cooling/DHW classifiers (Central/Distributed/Mixed/Unknown)
    utils.py     — load_curated(), filter_unit_type(), filter_segment(), filter_fuel(),
                   filter_classification_view() — shared helpers for all analysis scripts
```

**Quick start:**
```bash
# Build curated table
python analysis/recs/01_build_curated_table.py \
    --data data/recs/recs2020_public_v7.csv \
    --outdir outputs/recs

# Run full pipeline (steps 02 through 07)
python analysis/recs/run_workflow.py \
    --curated outputs/recs/recs2020_curated_*.parquet \
    --outdir outputs/recs

# Or run individual steps — see analysis/recs/README.md for details
```

The RECS pipeline includes extra analysis steps not shared by other datasets:
- `02b` — sample balance & prevalence analysis
- `05b` — sensitivity matrix across all analytical slices
- `07` — presentation-quality plots (forest plots, grouped bars, climate zone charts)
- `08` — cross-dataset comparison (RECS vs RBSA from pre-computed CSVs)

See [`analysis/recs/README.md`](analysis/recs/README.md) for the full script sequence, CLI options, and classification rules.

---

### RBSA 2022

> **Status: Implemented**

**Source:** 2022 Residential Building Stock Assessment — Pacific Northwest
**Geography:** Pacific Northwest (WA, OR, ID, MT)
**Key fields:** Primary heating/cooling system type, water heater records, site EUI, Heating Zone, vintage

**Source modules:**
```
src/datasets/rbsa/
    ingest.py    — load_rbsa_from_dir(), load_rbsa_from_zip() — loads 8 CSV files
    classify.py  — site-level + MF-building-level classification rules
                   Uses equipment name lookup tables (CENTRAL_HEATING, DISTRIBUTED_HEATING, etc.)
```

**Quick start:**
```bash
# Build curated tables (produces two parquets: site_master + mf_buildings)
python analysis/rbsa/01_build_curated_mf_table.py \
    --data-dir data/rbsa/ \
    --outdir outputs/rbsa

# Run full pipeline
python analysis/rbsa/run_workflow.py \
    --data-dir data/rbsa/ \
    --outdir outputs/rbsa
```

See [`analysis/rbsa/README.md`](analysis/rbsa/README.md) for the full script sequence and classification rules.

---

### FM Multifamily Survey 2023

> **Status: Implemented** — smallest energy subset, treat as directional only

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
python analysis/fm_mf_survey/01_load_and_classify.py \
    --data data/fm_mf_survey/2023_Multifamily_Survey_dataset_FINAL.xlsx \
    --outdir outputs/fm_mf_survey
```

See [`analysis/fm_mf_survey/README.md`](analysis/fm_mf_survey/README.md) for the full script sequence and classification logic.

---

## How the Classifiers Work

Each dataset has its own `classify.py` in `src/datasets/<dataset>/`. Classifiers examine
equipment fields and return a 4-level label: `Central`, `Distributed`, `Mixed`, or `Unknown`.
Some classifiers also distinguish **explicit** (directly reported) from **inferred**
(derived from equipment codes) classifications.

To adjust classification rules, edit the relevant `classify.py` — each uses lookup sets
or conditional logic that can be modified without touching the pipeline scripts:

```
src/datasets/recs/classify.py          # RECS: equipment code sets + HEATAPT flag
src/datasets/rbsa/classify.py          # RBSA: equipment name lookup tables
src/datasets/fm_mf_survey/classify.py  # FM: plant type + equipment type conditionals
```

The `to_binary()` function in each classifier collapses inferred labels into their base
category (e.g., `"Central (inferred)"` → `"Central"`) and maps `Mixed`/`Unknown` to `pd.NA`
for binary analysis.

---

## Development

```bash
pip install -e .

# Run tests
pip install pytest
pytest tests/ -v     # 44 tests — all RECS classifier unit tests

# Linting (optional)
pip install ruff
ruff check src/ analysis/
```

### Code style
- Line length: 100 (configured in `pyproject.toml` for both Ruff and Black)
- Logging via `src/common/log.py` — use `get_logger(__name__)`, not `print()`
- CLI args via `argparse` — no hardcoded paths in scripts
- Parquet for intermediate outputs; CSV exports for sharing

### Adding a new dataset

1. Create `src/datasets/<name>/ingest.py` and `classify.py`
2. Classifiers should return a `label` in `{Central, Distributed, Mixed, Unknown}` and a `reason` string
3. Create `analysis/<name>/` with numbered scripts following the 01–06 pattern
4. Add a README in `analysis/<name>/` documenting the script sequence

---

## Important Caveats

- **"Not Provided" ≠ NaN** — the FM survey uses both; both must be filtered.
- **Do NOT merge datasets** — FM MF Survey, RBSA, and RECS have different classification schemes, geographies, and survey vintages. Script 06/08 does side-by-side comparison only.
- **Non-parametric tests** — distributions are non-normal; use Mann-Whitney U, not t-tests.
- **Small samples** — FM energy subset ~40–107 properties; RBSA has only 11–35 Central heating MF units. All outputs label sample sizes.
- **Fuel confounding** — gas-heated distributed buildings can have *higher* EUI; always check fuel-stratified results alongside overall comparisons.
- **Observational data** — no causal claims. System type is confounded with building age, size, fuel type, and income.
- **HEATAPT availability** — in RECS, the explicit central/distributed flag (`HEATAPT`) is only populated for 5+ unit buildings (TYPEHUQ=4). 2-to-4 unit buildings rely entirely on inferred classifications. See [`WHAT_CHANGED_RECS.md`](WHAT_CHANGED_RECS.md) for the full analysis of this discovery.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| [`FINDINGS.md`](FINDINGS.md) | Living summary of cross-dataset results and interpretation |
| [`WHAT_CHANGED_RECS.md`](WHAT_CHANGED_RECS.md) | Detailed changelog for the RECS pipeline "heating refocus" iteration |
| [`analysis/recs/README.md`](analysis/recs/README.md) | RECS script sequence, CLI options, assumptions, caveats |
| [`analysis/rbsa/README.md`](analysis/rbsa/README.md) | RBSA script sequence, CLI options, data model notes |
| [`analysis/fm_mf_survey/README.md`](analysis/fm_mf_survey/README.md) | FM script sequence, classification logic tables |
| [`tests/README.md`](tests/README.md) | Test suite overview and how to run |

## License

MIT — see [LICENSE](LICENSE).
