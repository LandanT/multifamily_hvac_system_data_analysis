# Copilot Instructions — Building Systems & Energy Use Analysis Repo

You are helping develop a Python repo that analyzes relationships between building system configurations
(e.g., central vs distributed HVAC and DHW) and energy consumption. Three datasets are implemented:
RECS 2020 (national), RBSA 2022 (Pacific NW), and FM Multifamily Survey 2023 (national MF).
CBECS is a placeholder for future work.

**Current phase:** The core pipelines are established for all three datasets. Current work focuses on
refining the RECS heating-focused analysis (fuel stratification, explicit-vs-inferred classification,
presentation plots) and cross-dataset comparisons.

## Goals
1. Produce repeatable workflows that transform source datasets into analysis-ready tables.
2. Provide system-type classifications (HVAC and DHW): `Central`, `Distributed`, `Mixed`, `Unknown`.
3. Compare energy use (annual; electric + gas) across system types (weighted if available).
4. Make it easy to refine classification rules without rewriting the pipeline.

## Repo Structure
- `data/` raw/staged datasets by source (gitignored).
- `src/` reusable code:
  - `src/datasets/recs/` RECS-specific ingest + classifiers + shared utils
  - `src/datasets/rbsa/` RBSA-specific ingest + classifiers
  - `src/datasets/fm_mf_survey/` FM MF Survey ingest + classifiers
  - `src/common/` shared utilities (logging, schema checks, unit conversions)
- `analysis/` numbered runnable scripts (fast iteration):
  - `analysis/recs/` scripts 01–08 + run_workflow.py (most complete pipeline)
  - `analysis/rbsa/` scripts 00–06 + run_workflow.py
  - `analysis/fm_mf_survey/` scripts 01–06
- `outputs/` generated artifacts (gitignored).
- `tests/` pytest suite (44 tests for RECS classifiers).

## Dataset-Specific Guidance

### RECS (primary dataset)
- `HEATAPT` (explicit central flag) is only populated for 5+ unit buildings (TYPEHUQ=4).
- 2-to-4 unit buildings (TYPEHUQ=3) rely on inferred classifications from EQUIPM codes.
- Use `filter_classification_view()` in `src/datasets/recs/utils.py` to select explicit-only vs all.
- Sentinel values -2 (not applicable) and -1 (missing) are replaced with NaN in `ingest.py`.

### RBSA
- Multifamily system fields are at **Building_ID** level, while usage is often at **SiteID** level.
- Keep both identifiers in curated outputs, and document the unit-of-analysis in scripts.
- Small Central heating sample (~11–35 units) — treat results as directional.

### FM MF Survey
- `"Not Provided"` ≠ NaN — always filter both.
- Energy-analysis subset is thin (~40–107 properties). Do not draw statistical conclusions alone.

## System Type Classifiers
- Each dataset has its own `classify.py` in `src/datasets/<dataset>/`
- Classifiers return:
  - `label` in {Central, Distributed, Mixed, Unknown}
  - `reason` (short human-readable string)
- Prefer conservative labeling; when ambiguous, return `Unknown` and log.
- `to_binary()` collapses inferred labels and maps Mixed/Unknown → pd.NA.

## Implementation Preferences
- pandas for transforms; Parquet outputs when possible; CSV exports for sharing
- logging via `src/common/log.py` (no print for pipeline steps)
- CLI args via argparse (no hardcoded paths)
- Do NOT merge datasets across sources — use side-by-side comparison only
