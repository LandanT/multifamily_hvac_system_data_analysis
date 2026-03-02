# Building Systems & Energy Use Analysis

This repo supports repeatable workflows to classify **central vs distributed HVAC/DHW** and compare **energy consumption**.

## Quick start (RBSA)
1. Put RBSA raw files under `data/rbsa_2022/` **or** point scripts at the RBSA ZIP.
2. Run:

```bash
python analysis/rbsa/01_build_curated_mf_table.py --zip /path/to/RBSA.zip --outdir outputs/rbsa
python analysis/rbsa/02_sanity_checks.py --curated outputs/rbsa/rbsa_mf_curated_*.parquet
```

## Notes
- Raw data is gitignored by default.
- Classification rules live in `src/datasets/rbsa/classify.py`.
