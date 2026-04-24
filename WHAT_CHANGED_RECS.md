# What Changed — RECS Multifamily Heating Analysis Refocus

## TL;DR

The original analysis pooled all multifamily units together and asked
"does Site EUI differ between central and distributed heating?"
The refocused analysis splits multifamily into **2-to-4 unit** vs **5+ unit**
buildings, prioritizes **Heating EUI** over Site EUI, stratifies by
**fuel type**, and separates **explicit** from **inferred** system
classifications so we can see where the signal is real vs where it is noisy.

---

## 1. Changes to the Pipeline

### 1a. Curated Table (`01_build_curated_table.py`)

| Aspect | Before | After |
|---|---|---|
| **Derived columns** | None beyond EUI and outlier flags | Added 6 new columns: `mf_segment`, `unit_size_bin`, `heating_classification_mode`, `amenity_flag`, `ev_flag`, `heating_fuel_group` |
| **Segment split** | All MF pooled | `mf_segment` = "2_to_4_units" (TYPEHUQ 3, n=1751) or "5plus_units" (TYPEHUQ 4, n=1013) |
| **Classification provenance** | Central/Distributed treated equally regardless of source | `heating_classification_mode` = "explicit" (from HEATAPT) vs "inferred" (from equipment scan) vs "unknown" |
| **Fuel grouping** | Not tracked | `heating_fuel_group` = electric / gas / both / other |
| **Amenity/EV flags** | Not tracked | Flags for pools, hot tubs (127 units), EVs (33 units) for sensitivity exclusions |
| **Metadata logging** | Minimal | Script now prints segment counts, fuel distribution, missingness report |

### 1b. New Script: Sample Balance & Prevalence (`02b_sample_balance_and_prevalence.py`)

Entirely new. Answers "where do Central and Distributed coexist in
meaningful numbers?" Outputs prevalence tables, stacked bar charts, count
heatmaps, and a classification-view summary CSV.

### 1c. Statistical Testing (`03_statistical_testing.py`)

| Aspect | Before | After |
|---|---|---|
| **Primary outcome** | Site_EUI_kBtu_sqft | **Heating_EUI_kBtu_sqft** (Site EUI still included) |
| **Loop structure** | Single pass over all MF | Triple loop: MF_SEGMENTS × FUEL_STRATA × CLASSIFICATION_VIEWS |
| **Classification views** | Pooled binary only | pooled_binary + explicit_only + inferred_only |
| **Secondary systems** | Cooling + DHW tested with same granularity | Cooling & DHW tested only at All-MF / pooled-binary level |
| **Output columns** | Basic | Added `mf_segment`, `fuel_stratum`, `classification_view`, `delta_median` |

### 1d. OLS Models (`04_climate_vintage_controlled.py`)

| Aspect | Before | After |
|---|---|---|
| **Outcome variables** | Site_EUI only | **Heating_EUI + Site_EUI** |
| **Loop structure** | Single OLS + interaction | outcomes × segments × fuels × views (pooled_binary + explicit_only) |
| **Interaction models** | Ran for all slices | Only pooled binary + all fuels (where it makes sense) |
| **Within-zone** | Ran for all slices | Only pooled binary + all fuels + Heating EUI |

### 1e. New Script: Sensitivity Matrix (`05b_sensitivity_matrix.py`)

Entirely new. One row per analytical slice (~32 slices), showing:
n_central, n_distributed, median difference, OLS coefficient, OLS p-value.
Includes amenity-exclusion and EV-exclusion sensitivity rows. Outputs CSV + markdown.

### 1f. Presentation Plots (`07_presentation_plots.py`)

Three new plots added:
- **Heating EUI by segment** (2-to-4 vs 5+ faceted box plots)
- **Heating EUI by fuel for 5+ unit buildings** (electric vs gas strata)
- **Explicit vs inferred** classification view comparison

### 1g. Shared Utilities (`src/datasets/recs/utils.py`)

New helper functions: `filter_segment()`, `filter_fuel()`,
`filter_classification_view()`, `iter_segment_fuel_view()`, `friendly_label()`.
Constants: `MF_SEGMENTS`, `FUEL_STRATA`, `CLASSIFICATION_VIEWS`.

### 1h. Workflow Runner (`run_workflow.py`)

Steps expanded from `[01, 02, 03, 04, 05, 06]` to
`[01, 02, 02b, 03, 04, 05, 05b, 06, 07]`.

---

## 2. Critical Structural Discovery

**HEATAPT (apartment heating type) is ONLY populated for 5+ unit buildings
(TYPEHUQ = 4).** For 2-to-4 unit buildings (TYPEHUQ = 3), HEATAPT is always
NaN — all classifications are *inferred* from the equipment scan.

This means:
- `explicit_only` view = 5+ unit buildings only (n=950)
- `inferred_only` view = 2-to-4 unit buildings only (n=1663)
- The pooled binary view mixes these two structurally different populations

---

## 3. Key Findings — Before vs After

### What the old analysis showed

- Site EUI: Central vs Distributed medians nearly identical (43.67 vs 43.45, p=0.52 on simple Mann-Whitney).
- After OLS controls: significant coefficient (~-10 kBtu/ft²), but the raw signal was basically flat.
- Conclusion was unclear: "maybe there's something, maybe not."

### What the refocused analysis reveals

**Heating EUI is the clearer signal** (which makes sense — it isolates the system being tested):

| Slice | Central median | Distributed median | Δ median | OLS coef | OLS p |
|---|---|---|---|---|---|
| All MF, all fuels, pooled | 19.05 | 13.24 | +5.81 | -7.11 | <0.001 |
| All MF, all fuels, explicit only | 25.25 | 12.29 | +12.96 | -3.78 | 0.006 |
| 5+ units, gas, pooled | 39.55 | 26.50 | +13.05 | -5.23 | 0.012 |
| 5+ units, electric, pooled | 11.88 | 8.08 | +3.80 | -2.13 | 0.16 (n.s.) |
| 2-4 units, gas, pooled | 22.90 | 20.13 | +2.77 | -1.84 | 0.47 (n.s.) |
| 2-4 units, electric, pooled | 11.94 | 14.74 | **-2.80** | -4.34 | 0.004 |

**Takeaways:**

1. **Central heating uses more energy than distributed** — the signal is consistent
   across most slices, especially for gas-heated 5+ unit buildings (Δ ≈ 13 kBtu/ft²).

2. **Gas is where the big gap lives.** Electric heating shows smaller or
   non-significant differences. The 5+ unit / electric slice (n=566) has the
   right direction but doesn't reach significance in OLS (p=0.16).

3. **2-to-4 unit gas buildings show no significant difference** (p=0.47),
   likely because the sample is extremely unbalanced (1047 Central vs 54 Distributed)
   and all classifications are inferred.

4. **2-to-4 unit electric buildings flip sign on median** (distributed is
   *higher*), though the OLS coefficient is still negative — suggesting
   confounders (climate, vintage) explain the reversal once controlled.

5. **Amenity/EV exclusion doesn't change the story.** Removing pool/hot-tub
   or EV-charging units barely moves the coefficients (5+ units, all fuels:
   -3.78 → -3.59 after both exclusions).

6. **Site EUI still shows no raw median gap when pooled** (0.22 kBtu/ft²),
   but the OLS-controlled gap is ~10 kBtu/ft². The gap becomes visible in raw
   medians when you look at 5+ units explicitly (59.80 vs 45.48).

---

## 4. Why This Matters

The original analysis was asking the right question but mixing dissimilar
populations and looking at the wrong outcome variable:

- **Site EUI pools heating + cooling + DHW + plug loads** — heating system
  type signal gets diluted.
- **Pooling 2-to-4 and 5+ unit buildings** mixes explicit classifications
  (reliable) with inferred ones (noisier) and conflates different building
  typologies.
- **Ignoring fuel type** masks the fact that the central-vs-distributed gap
  is primarily a gas-heating story.

The refocused analysis isolates the signal by:
1. Looking at Heating EUI first (direct outcome of the system being tested)
2. Separating building segments (different data quality, different physics)
3. Stratifying by fuel (different energy intensities)
4. Tracking classification provenance (explicit vs inferred)

This gives us much more defensible claims about where and why central
heating systems use more energy per square foot than distributed ones.
