# Tests

## Running

```bash
pip install pytest
pytest tests/ -v
```

All 44 tests should pass. No raw data files are needed — tests use synthetic fixtures.

## Test Coverage

All tests are in `test_recs_classify.py` and cover the RECS classification logic in
`src/datasets/recs/classify.py`:

| Test class | Tests | What it covers |
|-----------|-------|----------------|
| `TestClassifyHeating` | 16 | Explicit Central/Distributed via `HEATAPT`, all 5 inferred distributed EQUIPM codes (5/7/8/10/13), all 3 inferred central EQUIPM codes (2/3/4), unknown codes, priority logic |
| `TestClassifyCooling` | 11 | Explicit via `COOLAPT`, 4 distributed ACEQUIPM codes (3/4/5/6), 1 central code (1), priority logic |
| `TestClassifyDhw` | 6 | Central/Distributed/Mixed/Unknown via `H2OAPT` and `MORETHAN1H2O`, priority logic |
| `TestToBinary` | 7 | Collapsing inferred labels → base labels, Mixed/Unknown → `pd.NA`, mixed series handling |
| `TestAddSystemClassifications` | 2 | Column presence and value correctness after full classification pipeline |
| `TestVectorizedMatchesRowWise` | 3 | Consistency between vectorized (fast) and row-wise (reference) classifiers for heating, cooling, DHW |

## What's Not Tested Yet

- RBSA classifiers (`src/datasets/rbsa/classify.py`)
- FM MF Survey classifiers (`src/datasets/fm_mf_survey/classify.py`)
- Ingest functions (all datasets)
- Analysis script outputs (integration tests)
