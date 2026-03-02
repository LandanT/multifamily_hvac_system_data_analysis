# Copilot Instructions — Building Systems & Energy Use Analysis Repo

You are helping develop a Python repo that analyzes relationships between building system configurations
(e.g., central vs distributed HVAC and DHW) and energy consumption. The project starts with NEAA RBSA data,
but will later integrate other sources such as CBECS and real building datasets.

## Goals
1. Produce repeatable, auditable workflows that transform source datasets into analysis-ready tables.
2. Provide clear system-type classifications (HVAC and DHW) such as: central, distributed, mixed, unknown.
3. Support comparisons of energy use (annual and monthly; electric and gas) across system types.
4. Enable layering multiple datasets with consistent schemas and documented assumptions.

## Guiding Principles
- Prefer reproducible pipelines over one-off exploration.
- Write modular code with clear I/O boundaries: raw → staged → curated → analysis.
- Keep transformations explicit and logged (row counts, join keys, missingness checks).
- Every derived variable (especially system classifications) must be documented with rules and rationale.
- Avoid hardcoding file paths; use config and CLI arguments.
- Be conservative with assumptions; if something is ambiguous, output "unknown" and log why.

## Repository Structure
- `data/` contains raw and staged datasets by source (rbsa_2022/, cbecs_2018/, building_data/). Large data is not committed.
- `src/` contains reusable code:
  - `src/datasets/<source>/` for source-specific ingest/cleaning/mapping
  - `src/common/` for shared utilities (logging, config, schema validation, unit conversions)
- `analysis/` contains runnable workflows per dataset and cross-dataset comparisons.
- `outputs/` contains generated artifacts (usually gitignored).

## Workflow Pattern (standard)
Each dataset workflow should:
1. Load source data using robust dtype handling.
2. Validate schema (required columns, expected types, sanity checks).
3. Produce "staged" tables (cleaned + minimal transforms).
4. Produce "curated" tables (joined + derived fields, e.g., system types).
5. Save outputs to `outputs/<dataset>/` with versioned filenames.
6. Emit a summary log: input rows, output rows, join keys, dropped records, missingness summary.

## System Type Classification Requirements
- Provide explicit rule-based classifiers for HVAC and DHW:
  - Allowed labels: `central`, `distributed`, `mixed`, `unknown`.
- Classifier functions must:
  - Return both the label and an explanation string (reason).
  - Be easy to modify without rewriting the pipeline.
- In multifamily data, treat building-level vs unit-level carefully:
  - Keep separate identifiers for site/unit vs building.
  - If energy use is unit-level but system type is building-level, document it and keep both IDs.

## Common Analysis Outputs
For each curated dataset:
- `*_combined_systems_energy.(csv|parquet)` containing:
  - identifiers (site_id, building_id if applicable)
  - hvac_system_type, dhw_system_type (+ optional heat_type/cool_type)
  - annual electric/gas usage, plus monthly usage columns if available
  - floor area and normalization fields (EUI) when possible
  - survey weights when available
- Summary tables:
  - weighted mean/median energy metrics by hvac_system_type and dhw_system_type
  - counts by category and missingness rates

## Implementation Preferences
- Prefer pandas and/or polars for tabular transforms.
- Use `argparse` for CLI entrypoints.
- Use `logging` (not print) with INFO summaries and DEBUG detail.
- Save curated tables as Parquet when possible; also export CSV for sharing.
- Put unit conversion utilities in `src/common/units.py`.

## Testing & Quality
- Include lightweight tests for classifiers and joins (pytest).
- Never commit raw proprietary datasets; document expected structure and schemas.

## When Uncertain
If you are unsure about columns, mappings, or normalization:
1) preserve metadata, 2) classify as unknown, 3) log a warning, and 4) make it easy to refine later.
