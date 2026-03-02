# RBSA analysis scripts

- `01_build_curated_mf_table.py` — build curated table (SiteID ↔ Building_ID, classify HVAC/DHW, join usage)
- `02_sanity_checks.py` — quick QC summaries

Edit system classification rules in:
- `src/datasets/rbsa/classify.py`
