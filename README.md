# Building Systems & Energy Use Analysis

Repeatable workflows to classify **central vs distributed HVAC/DHW** and compare **energy consumption**
across datasets (NEAA RBSA first; later CBECS and real building data).

## Folder layout
- `data/` — raw and staged datasets (gitignored)
- `src/` — reusable dataset ingest + shared utilities
- `analysis/` — runnable workflows per dataset and cross-dataset comparisons
- `outputs/` — generated artifacts (gitignored)
- `.github/copilot-instructions.md` — guidance for Copilot

## Quick start
1. Put source data under `data/<dataset>/` (e.g., `data/rbsa_2022/`).
2. Run a workflow script from `analysis/<dataset>/`.
3. Write curated outputs to `outputs/<dataset>/`.
