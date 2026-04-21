# Findings — Multifamily HVAC System Type & Energy Use

> **Last updated:** 2026-04-20  
> **Status:** Active — update as new analyses land  
> **Datasets:** RECS 2020 (national), RBSA 2022 (Pacific NW)

---

## 1. Dataset Summaries

| Dataset | Scope | MF sample | Central (htg) | Distributed (htg) | Key controls |
|---------|-------|-----------|---------------|-------------------|--------------|
| RECS 2020 | National, self-reported survey | 2,764 households (TYPEHUQ 3–4) | 1,893 | 720 | IECC climate zone, YEARMADERANGE (ordinal 1–8), log(TOTSQFT_EN), NWEIGHT |
| RBSA 2022 | Pacific NW, field-inspected | 369 MF units | 35 (heating) | 299 (heating) | Heating Zone (1–3), Home Vintage (year), Conditioned Area |

**Classification approaches differ:**
- RECS: Binary survey question (`HEATAPT`=1 → Central) + equipment-type inference. Mixed/Unknown → `pd.NA` in binary column.
- RBSA: Field-inspected equipment name mapped via lookup table. Site-level aggregation can yield Mixed.

---

## 2. Headline Results — Heating System Type

### 2a. Unadjusted (Mann-Whitney U)

| Dataset | Median Central | Median Distributed | p-value | r_rb | Significant? |
|---------|---------------|-------------------|---------|------|-------------|
| RECS 2020 | 43.67 | 43.45 | 0.518 | −0.02 | No |
| RBSA 2022 | 56.93 | 34.14 | 0.006 | −0.51 | **Yes** |

RECS shows no raw difference; RBSA shows a large effect favouring distributed — but only 11 Central units.

### 2b. OLS (controlled)

| Dataset | n | coef_distributed | p | 95% CI | R² |
|---------|---|-----------------|---|--------|-----|
| RECS 2020 | 2,613 | **−10.27** | **<0.001** | [−12.8, −7.8] | 0.277 |
| RBSA 2022 | 84 | −20.58 | 0.067 | [−42.6, 1.5] | 0.094 |

Both datasets point negative (distributed = lower EUI after controls). RECS is highly significant with large n; RBSA is borderline, underpowered with n=84.

### 2c. Fuel confounding (RECS)

| Fuel stratum | Median Central | Median Distributed | p | Direction |
|-------------|---------------|-------------------|---|-----------|
| All fuels | 43.67 | 43.45 | 0.518 | ≈ equal |
| Electric only | 37.48 | 36.93 | 0.650 | ≈ equal |
| **Natural gas** | **50.15** | **60.60** | **<0.001** | **Distributed higher** |

Gas-heated distributed MF homes have *higher* Site EUI — the OLS result (distributed = lower) is driven by composition: central systems are more likely to be gas-fired, and gas buildings have higher total EUI. After controlling for climate/vintage/size the net effect reverses.

---

## 3. Headline Results — Cooling System Type

### 3a. Unadjusted

| Dataset | Median Central | Median Distributed | p-value | r_rb |
|---------|---------------|-------------------|---------|------|
| RECS 2020 | 39.08 | 50.20 | <0.001 | +0.24 |
| RBSA 2022 | 52.89 | 35.90 | 0.249 | −0.42 |

RECS raw: distributed *higher*. RBSA raw: distributed lower but n.s. (only 3 Central).

### 3b. OLS (controlled)

| Dataset | n | coef_distributed | p | 95% CI | R² |
|---------|---|-----------------|---|--------|-----|
| RECS 2020 | 2,313 | **−3.81** | **0.004** | [−6.4, −1.2] | 0.268 |
| RBSA 2022 | 50 | −21.57 | 0.222 | [−56.7, 13.5] | 0.087 |

RECS: small but significant effect after controls. RBSA: large point estimate but very wide CI / n.s.

---

## 4. Headline Results — DHW System Type

### 4a. OLS (controlled)

| Dataset | n | coef_distributed | p | 95% CI | R² |
|---------|---|-----------------|---|--------|-----|
| RECS 2020 | 1,003 | +0.77 | 0.706 | [−3.3, 4.8] | 0.199 |
| RBSA 2022 | 50 | +378.0 | 0.114 | [−94.6, 850.6] | 0.073 |

Neither dataset shows a significant DHW system-type effect on Site EUI. RBSA's huge coefficient is a modeling artifact from extreme sample imbalance (0 Central DHW sites).

---

## 5. Building Age Interaction (RECS only)

Added `_distributed × YEARMADERANGE` interaction term to OLS.

| System | Interaction coef | p | Significant? | Interpretation |
|--------|-----------------|---|-------------|----------------|
| Heating | −0.48 | 0.364 | No | Gap is consistent across vintages |
| **Cooling** | **−1.85** | **<0.001** | **Yes** | Distributed advantage grows in newer buildings |
| DHW | −0.36 | 0.687 | No | No interaction |

For cooling: in older buildings (low YEARMADERANGE) there's essentially no Central-vs-Distributed difference; in newer buildings, distributed systems show increasingly lower EUI. This likely reflects newer mini-split / ductless equipment in newer construction.

---

## 6. Within-Climate-Zone Highlights

### RECS — Heating (17 IECC zones tested, Bonferroni α = 0.0011)

Only **2A** (p=0.005) and **3A** (p=0.001) survive Bonferroni — both show distributed *higher* raw EUI. This reversal vs. OLS again reflects fuel-mix confounding within those zones.

### RECS — Cooling (16 zones tested)

Zones **2A**, **3A**, **4A**, **5A**, **6A**, **6B** show significant raw differences (distributed higher), with 4A and 5A surviving Bonferroni.

### RBSA — Heating (3 Heating Zones)

Zone 1 (Mild): Central 57.2 vs Distributed 36.1 (p=0.011). Zones 2–3 have insufficient Central units.

---

## 7. Cross-Dataset Comparison

Both datasets consistently estimate a **negative** OLS coefficient for distributed heating (RECS: −10.3, RBSA: −20.6), suggesting that after controlling for confounders, distributed heating systems are associated with lower Site EUI. The RBSA effect is roughly 2× larger but with 30× less statistical power.

Key differences:
- **Geographic scope**: RECS is national; RBSA is Pacific NW only (mild climate, less heating load variance)
- **Sample balance**: RECS has ~72% Central / 28% Distributed; RBSA has ~13% Central / 87% Distributed
- **Classification method**: RECS uses survey self-report; RBSA uses field inspection
- **Controls**: RECS uses IECC climate dummies (17 zones); RBSA uses ordinal Heating Zone (1–3) + continuous vintage year

Despite these differences, the direction of the heating effect is concordant — a reassuring signal.

---

## 8. Key Caveats & Limitations

1. **Observational data** — no causal claims. System type is confounded with building age, size, fuel type, and income.
2. **Fuel confounding is substantial** — gas-heated distributed buildings have *higher* EUI; the OLS "distributed = lower" result reflects composition differences.
3. **RBSA sample size** — only 11–35 Central heating units. CIs are wide; results are directional, not definitive.
4. **RECS classification** — self-reported `HEATAPT` may mis-classify; equipment inference adds complexity.
5. **DHW analysis** — RBSA has 0 Central DHW observations in MF; RECS has limited sample (n=1,003). Neither dataset supports strong DHW conclusions.
6. **FM Multifamily Survey** — excluded from cross-dataset comparison due to data quality concerns.

---

## 9. Output Artifacts

| File | Description |
|------|-------------|
| `outputs/recs/04_ols_results.csv` | RECS base OLS coefficients |
| `outputs/recs/04_ols_interaction_results.csv` | RECS OLS with vintage interaction |
| `outputs/recs/04_within_zone_results.csv` | RECS within-zone Mann-Whitney |
| `outputs/recs/03_mann_whitney_results.csv` | RECS unadjusted Mann-Whitney (all strata) |
| `outputs/recs/08_forest_recs_vs_rbsa.png` | Forest plot — RECS vs RBSA OLS coefficients |
| `outputs/recs/08_median_bars_recs_vs_rbsa.png` | Grouped bars — median EUI comparison |
| `outputs/recs/08_recs_vs_rbsa_summary.csv` | Combined summary table |
| `outputs/recs/07_*.png` | Presentation plots (5 figures) |
| `outputs/rbsa/04_ols_results.csv` | RBSA OLS coefficients |
| `outputs/rbsa/03_mann_whitney_results.csv` | RBSA Mann-Whitney results |
