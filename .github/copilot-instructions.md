# Copilot Instructions — Building Systems & Energy Use Analysis Repo

You are helping develop a Python repo that analyzes relationships between building system configurations
(e.g., central vs distributed HVAC and DHW) and energy consumption. The project starts with NEAA RBSA data,
but will later integrate other sources such as CBECS and real building datasets.

**Current phase:** iterate quickly on RBSA multifamily definitions and signal tests, with light structure.
Keep pipelines transparent: join keys, row counts, and missingness checks.

## Goals
1. Produce repeatable workflows that transform source datasets into analysis-ready tables.
2. Provide system-type classifications (HVAC and DHW): `central`, `distributed`, `mixed`, `unknown`.
3. Compare energy use (annual + monthly; electric + gas) across system types (weighted if available).
4. Make it easy to refine classification rules without rewriting the pipeline.

## Repo Structure
- `data/` raw/staged datasets by source (gitignored).
- `src/` reusable code:
  - `src/datasets/rbsa/` RBSA-specific ingest + classifiers
  - `src/common/` shared utilities (logging, schema checks, unit conversions)
- `analysis/` numbered runnable scripts (fast iteration):
  - `analysis/rbsa/01_build_curated_mf_table.py`
  - `analysis/rbsa/02_sanity_checks.py`
- `outputs/` generated artifacts (gitignored).

## RBSA Guidance (important)
- Multifamily system fields are at **Building_ID** level, while usage is often at **SiteID** level.
- Keep both identifiers in curated outputs, and document the unit-of-analysis in scripts.

## System Type Classifiers
- Implement rules in `src/datasets/rbsa/classify.py`
- Classifiers return:
  - `label` in {central, distributed, mixed, unknown}
  - `reason` (short human-readable string)
- Prefer conservative labeling; when ambiguous, return `unknown` and log.

## Implementation Preferences
- pandas for transforms; Parquet outputs when possible; CSV exports for sharing
- logging via `src/common/log.py` (no print for pipeline steps)
- CLI args via argparse (no hardcoded paths)
