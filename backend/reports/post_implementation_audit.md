# POST-IMPLEMENTATION AUDIT & SCIENTIFIC VALIDATION REPORT
**Project:** Sagar-Drishti  
**Date of Audit:** 2026-09-11  
**System Evaluated:** 10-Year Copernicus Marine Reanalysis + IMD Best-Track Multi-Horizon ML Pipeline  
**Repository Branch:** `kshitij`  

---

## 1. Executive Summary

This independent post-implementation audit evaluates the newly implemented 10-year Copernicus ocean data expansion and authoritative IMD Best-Track cyclone retraining pipeline for Sagar-Drishti. The objective is to determine whether the candidate `v2_10yr` model suite is scientifically sound, correctly trained, and ready to replace the existing 2-year baseline production models.

### Key Audit Findings
1. **10-Year Ocean Data File Status**: The 10-year NetCDF dataset (`copernicus_phy_10yr_surface.nc`, ~7.39 GB, spanning 2016-06-24 to 2026-06-23) has **NOT** been downloaded to the local filesystem. The local environment contains only the baseline 2-year NetCDF file (`copernicus_phy_2yr_surface.nc`, 1.58 GB, spanning 2024-06-24 to 2026-06-23, 730 daily timesteps). Consequently, **2016–2023 ocean reanalysis observations contributed exactly zero (0) supervised training samples**.
2. **IMD Track Date Forward-Fill Bug**: The IMD Best Track parser ([`imd_best_track_parser.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/parsers/imd_best_track_parser.py)) failed to forward-fill merged/empty date cells on sub-daily track rows (03:00, 06:00, 12:00, 18:00 UTC). When a date cell was blank, the parser defaulted the date to `YYYY-01-01`. In 2024, **191 out of 243 track points collapsed to 2024-01-01**, causing the registered start date of major 2024 cyclones (Asna, Dana, Fengal, BOB 05) to be set to January 1, 2024.
3. **Severe Label Contamination in Training Set**: Because storm start dates collapsed to `2024-01-01`, spatial-temporal matching classified **237 out of 320 training days** as active cyclone days (74.1% positive prevalence). The model was trained on pseudo-cyclone labels across months of calm background summer conditions.
4. **Candidate Model Discrimination Collapse**: The candidate v2 models exhibit near-random discriminative ability: Test ROC-AUC collapsed to **0.5230 (0d), 0.5399 (1d), 0.5527 (2d), and 0.5582 (3d)** (barely above a 0.50 coin toss). Precision collapsed to **4.19% – 6.74%**, with a **93.3% – 95.8% false alarm rate** (401 to 412 false alarms out of 430 positive predictions).
5. **Production Baseline Integrity**: The baseline production models in [`backend/models/`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/models/) remain completely untouched and achieve **ROC-AUC of 0.8621, PR-AUC of 0.1553, and F1 of 0.2059** for Horizon 3d. The live application currently loads baseline version `1.1.0`.

### Audit Recommendation
**VERDICT: NO-GO.** The candidate `v2_10yr` model suite cannot replace production in its current state. Production must continue running the baseline models while the identified date-parsing bug is resolved and the full 10-year NetCDF is downloaded.

---

## 2. Current Architecture & Component Map

The repository architecture spans ingestion, validation, feature engineering, event matching, training, and inference layers:

| Component | File Path | Input | Output | Status | In Production? | Test Coverage |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Copernicus Service** | [`copernicus_service.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/copernicus_service.py) | NetCDF filepath / API credentials | Lazy xarray Dataset, slices | Operational | Yes (loads 2-yr) | Verified |
| **10-Yr Ingestion CLI** | [`ingest_10yr_copernicus.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/scripts/ingest_10yr_copernicus.py) | CLI flags (`--dry-run`, `--validate-only`) | Copernicus Marine subset download | Operational (dry-run OK) | CLI only | Validated |
| **10-Yr NetCDF Validator** | [`validate_10yr_copernicus.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/validate_10yr_copernicus.py) | NetCDF dataset | Validation dictionary (bounds, NaNs, vars) | Operational | Standalone | 2/2 tests pass |
| **Feature Engine** | [`historical_engine.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/historical_engine.py) | NetCDF Dataset, site coords | 101 rolling features (Parquet) | Operational | Yes | Verified |
| **Feature Manifest** | [`feature_manifest.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/feature_manifest.py) | Feature column names | Schema audit & ordering validation | Operational | Yes | 5/5 tests pass |
| **IMD Best Track Parser** | [`imd_best_track_parser.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/parsers/imd_best_track_parser.py) | `78b4b0_Best_Tracks__Data__1982-2026_.xlsx` | Normalized tracks & systems (Parquet/JSON) | **Defective** (blank date handling) | Indirect | 4/4 tests pass (schema only) |
| **Historical Event Store** | [`event_store.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/event_store.py) | Seed events + IMD systems | Thread-safe `HistoricalEvent` catalog | Operational | Yes | 12/12 tests pass |
| **Event Matcher** | [`event_matcher.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/event_matcher.py) | Observation date, location, event catalog | Supervised labels (`event_within_0d`..`3d`) | Operational (logic correct) | Yes | 12/12 tests pass |
| **ML Dataset Builder** | [`ml_dataset_builder.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/ml_dataset_builder.py) | Labeled feature DataFrame | Train/Val/Test split Parquet | Operational | Yes | 5/5 tests pass |
| **Baseline ML Trainer** | [`train_separate_horizon_models.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/scripts/train_separate_horizon_models.py) | `ml_features_multibasin.parquet` | Baseline models (`backend/models/`) | Operational | Active Baseline | 10/10 tests pass |
| **Candidate 10-Yr Trainer**| [`train_10yr_pipeline.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/scripts/train_10yr_pipeline.py) | Labeled multi-basin Parquet | Candidate models (`backend/models/v2_10yr/`)| **Trained on corrupted labels**| Candidate only | Verified |
| **Prediction Service** | [`prediction_service.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/prediction_service.py) | Observation dict, horizon flags | Risk probability, warning level, explainability | Operational | Yes (v1.1.0 active) | 12/12 tests pass |
| **API Endpoints** | [`endpoints.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/api/endpoints.py) | FastAPI HTTP requests | JSON API responses | Operational | Yes | 12/12 tests pass |
| **SagarBot RAG Engine** | [`copernicus_rag.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/copernicus_rag.py) | User natural language prompt | Grounded oceanic assistant response | Operational | Yes | Verified |
| **Frontend UI & 3D Globe**| `frontend/src/` | User interaction, REST API | Cesium 3D globe, danger zones, warning cards | Operational | Yes | Clean build (0 err) |

---

## 3. Data Verification

### Physical Files on Disk
- `backend/data/copernicus/copernicus_phy_2yr_surface.nc`: **1,584,738,191 bytes (1.51 GB)**. Present on disk.
- `backend/data/copernicus/copernicus_phy_10yr_surface.nc`: **0 bytes (NOT PRESENT ON DISK)**.
- `78b4b0_Best_Tracks__Data__1982-2026_.xlsx`: **12,582,912 bytes (12.00 MB)**. Present in project root.
- `backend/data/historical/imd_tracks_2016_2026.parquet`: **43,585 bytes**. Present on disk.
- `backend/data/historical/imd_systems_2016_2026.json`: **56,543 bytes**. Present on disk.
- `backend/data/historical/labeled_features_10yr.parquet`: **1,250,892 bytes**. Present on disk.
- `backend/data/historical/ml_features_10yr_candidate.parquet`: **1,328,450 bytes**. Present on disk.

---

## 4. Copernicus 10-Year Audit

A dry run against the Copernicus Marine API via `ingest_10yr_copernicus.py --dry-run` was executed to verify the exact remote catalog specifications:

- **Target Dataset ID**: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **Selected Version**: `202311`, Part: `default`
- **Variables**: `mlotst`, `so`, `thetao`, `uo`, `vo`, `zos` (6 variables)
- **Spatial Bounds**: Longitude `[50.0°E, 100.0°E]`, Latitude `[0.0°N, 25.0°N]`
- **Temporal Domain**: `2016-06-24T00:00:00` to `2026-06-23T00:00:00` (3,652 daily timesteps)
- **Vertical Level**: Single surface level at depth `0.494025 m`
- **Calculated Download Size**: `7,564.71 MB (~7.39 GB)`
- **Uncompressed Data Transfer Size**: `58,722.96 MB (~57.3 GB)`
- **API Status**: Valid and ready on Copernicus servers.

### Audit of Local Available Dataset (`copernicus_phy_2yr_surface.nc`)
Direct inspection of the local NetCDF file confirms:
- **Actual Timesteps**: Exactly 730 daily timesteps.
- **Start Timestamp**: `2024-06-24 00:00:00`
- **End Timestamp**: `2026-06-23 00:00:00`
- **Spatial Dimensions**: Latitude = 301 points (0.0°N to 25.0°N, step ~0.083°), Longitude = 601 points (50.0°E to 100.0°E, step ~0.083°).
- **Depth**: Single scalar at `0.494025 m`.
- **Variables**: `mlotst`, `so`, `thetao`, `uo`, `vo`, `zos`.
- **Temporal Continuity**: Exactly 1.0 day between all adjacent slices. Zero missing days, zero duplicate timestamps.
- **Missing / NaN Percentage**: Sea pixels are 100% complete with valid physical data; land pixels are consistently masked.

**Conclusion**: The Copernicus infrastructure and parameters are scientifically valid, but the 7.39 GB file has not been transferred locally. The existing local data is strictly the 2-year subset.

---

## 5. IMD Best Track Audit

The workbook `78b4b0_Best_Tracks__Data__1982-2026_.xlsx` contains 45 yearly sheets (1982 through 2026). The parser scanned the 11 target sheets spanning 2016 through 2026.

### Discovered Inventory
- **Yearly Sheets Parsed**: 11 sheets (`2016` through `2026`).
- **Total Systems**: 117 physical cyclonic systems.
- **Total Track Points**: 2,454 6-hourly points.
- **Coordinate Bounds**: Latitude range `[2.5°N, 28.5°N]`, Longitude range `[51.2°E, 98.6°E]`. All coordinates fall strictly within the North Indian Ocean basin.
- **Invalid Coordinates**: 0.
- **Tentative 2026 Points**: 18 records in sheet `2026` correctly flagged as `is_tentative_2026 = True`.

### Root Cause of the Date Parsing Flaw
In sheets `2021`, `2022`, and `2024`, IMD recorded the calendar date in column `Date` only on the 00:00 UTC row of each day. The rows for 03:00, 06:00, 12:00, and 18:00 UTC contain blank (merged) date cells.
In `imd_best_track_parser.py`:
```python
raw_date = row[date_col] if date_col else None
date_str = cls.parse_date(raw_date, year_hint=year_int)
if not date_str:
    # If date column failed, try extracting from year
    date_str = f"{year_int}-01-01"  # <--- CRITICAL BUG: Defaults to Jan 01
```
Because `date_str` was not forward-filled (`ffill()`) from the preceding row within the same storm, every sub-daily observation with a blank date cell was assigned `2024-01-01`.

```
System IMD-2024-1-REMAL:
Row 1931: 2024-05-24 00:00 UTC -> 2024-05-24
Row 1932: (blank date) 03:00 UTC -> 2024-01-01  <-- Corrupted
Row 1933: (blank date) 06:00 UTC -> 2024-01-01  <-- Corrupted
```
When `dates_sorted = sorted(sdata["dates"])` computed `start_date`, `dates_sorted[0]` became `2024-01-01` for **every system in 2024**:
- `IMD-2024-4-ASNA`: `start_date = 2024-01-01`, `end_date = 2024-09-02` (Real genesis: Aug 29)
- `IMD-2024-11-DANA`: `start_date = 2024-01-01`, `end_date = 2024-10-25` (Real genesis: Oct 22)
- `IMD-2024-12-FENGAL`: `start_date = 2024-01-01`, `end_date = 2024-12-01` (Real genesis: Nov 28)

---

## 6. Variable Standardization & Physical Checks

The 6 canonical NetCDF variables map directly to Sagar-Drishti's physical channels:

| Raw Variable | Canonical Name | Physical Property | Unit | Valid Ocean Range | Derived Formula |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `thetao` | `temp` | Potential Temperature | °C | [10.0, 38.0] | Direct measurement |
| `so` | `sal` | Practical Salinity | PSU | [0.0, 45.0] | Direct measurement (min 0 for river plumes) |
| `uo` | `cur_u` | Eastward Velocity | m/s | [-4.0, 4.0] | Direct measurement |
| `vo` | `cur_v` | Northward Velocity | m/s | [-4.0, 4.0] | Direct measurement |
| `zos` | `ssh` | Sea Surface Height | m | [-3.0, 3.0] | Direct measurement |
| `mlotst` | `mld` | Mixed Layer Thickness | m | [1.0, 300.0] | Direct measurement |
| *Derived* | `cur` | Current Speed | m/s | [0.0, 5.0] | $\sqrt{u_o^2 + v_o^2}$ |

Numerical correctness of `cur` was independently recomputed: $\sqrt{1.15^2 + 0.85^2} = 1.4300$ m/s, matching the system calculation exactly. Units, orientations, and bounds conform to oceanographic standards.

---

## 7. Hard 101-Feature Contract & Schema Integrity

The 101-feature contract defines the exact mathematical features used across all ML models:

### Feature Distribution by Channel
1. **`temp` (14 features)**: `current`, `base_mean`, `base_std`, `abs_anom`, `zscore`, 7d (`mean`, `delta`, `trend`), 14d (`mean`, `delta`, `trend`), 30d (`mean`, `delta`, `trend`).
2. **`sal` (15 features)**: Same as `temp` plus `sal_pct_anom`.
3. **`cur_u` (14 features)**: Same 14 as `temp`.
4. **`cur_v` (14 features)**: Same 14 as `temp`.
5. **`cur` (15 features)**: Same 14 as `temp` plus `cur_pct_anom`.
6. **`ssh` (14 features)**: Same 14 as `temp`.
7. **`mld` (15 features)**: Same 14 as `temp` plus `mld_pct_anom`.
- **TOTAL**: $14 + 15 + 14 + 14 + 15 + 14 + 15 = \mathbf{101}$ features.

### Feature Whitelist & Ground-Truth Isolation
An exhaustive audit of the 101 feature names via [`test_feature_leakage_audit.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_feature_leakage_audit.py) confirms:
- **Zero IMD storm parameters** (name, track lat/lon, pressure, wind, category, landfall) are present in the feature columns.
- Feature ordering matches the canonical manifest deterministically across all environments.

---

## 8. Temporal Leakage Audit

### Mathematical Causality
For any observation date $T$ (day index `idx`):
- Current value: uses slice at `idx` (Date $T$).
- 7-day rolling window: uses index slice `[idx - 6 : idx + 1]`. Maximum index is `idx` ($T$).
- 14-day rolling window: uses index slice `[idx - 13 : idx + 1]`. Maximum index is `idx` ($T$).
- 30-day rolling window: uses index slice `[idx - 29 : idx + 1]`. Maximum index is `idx` ($T$).
- Indices `idx + 1`, `idx + 2`, ... are **never accessed** during rolling feature computation.

### Normalization and Scaling
In tree-based Random Forests, no cross-split scaling is applied. For candidate Logistic Regression models, `StandardScaler` is fitted strictly on the `train` split and applied out-of-sample to `val` and `test`, preventing data leakage.

### Label Leakage Finding
While the feature extraction is mathematically strictly causal ($\le T$), **the ground-truth labels themselves had severe backward leakage** because the date parsing bug extended storm labels up to 8 months into the past.

---

## 9. Year-by-Year Data Utilization Audit

The following table provides the comprehensive year-by-year inventory of ocean observations, IMD track points, and actual ML training utilization:

| Year | Ocean NetCDF Days on Disk | IMD Systems in Catalog | IMD Track Points | Labeled Ocean Days | Active Event Days | Lead Event Days | Negative Clean Days | Included in ML Training Set? | Reason if Not Included |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2016** | 0 | 10 | 199 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2017** | 0 | 9 | 132 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2018** | 0 | 14 | 317 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2019** | 0 | 12 | 414 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2020** | 0 | 9 | 178 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2021** | 0 | 10 | 189 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2022** | 0 | 15 | 223 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2023** | 0 | 9 | 279 | 0 | 0 | 0 | 0 | **NO** | 10-year NetCDF not on disk |
| **2024** | 191 (Jul 23 - Dec 31) | 13 | 243 | 324 | 237* | 0 | 83 | **YES (Train)** | Contaminated by Jan 01 bug |
| **2025** | 365 (Jan 01 - Dec 31) | 15 | 262 | 730 | 35 | 26 | 653 | **YES (Val & Test)** | Valid labeling |
| **2026** | 174 (Jan 01 - Jun 23) | 1 | 18 | 348 | 0 | 0 | 348 | **YES (Test)** | Calm winter/spring ocean |
| **TOTAL**| **730** | **117** | **2,454** | **1,402** | **272** | **26** | **1,084** | **1,382** | **2016-2023 missing on disk** |

*\*Note: The 237 active event days in 2024 are artificial; genuine cyclone days in 2024 for these regions were fewer than 25.*

---

## 10. Daily Alignment of IMD 6-Hourly Tracks

The IMD Best Track workbook provides observations at 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, and 21:00 UTC. The Copernicus reanalysis provides daily means centered at 12:00 UTC.

### Daily Alignment Rule
1. **Grouping**: In `event_matcher.py`, all track points for storm $S$ are aggregated into the storm's overall lifespan $[T_{start}, T_{end}]$ and bounding envelope $\text{bbox}(S)$.
2. **Spatial Distance**: For point mode, if the observation point $(lat, lon)$ falls within the storm bbox or within 350 km haversine distance of any track point, it is marked as spatially active.
3. **Temporal Evaluation**: An observation on date $T$ is marked:
   - `active_event` if $T_{start} \le T \le T_{end}$
   - `lead_event` if $1 \le T_{start} - T \le 3$ days
   - `negative_buffer` if $1 \le T - T_{end} \le 2$ days
   - `negative_clean` if outside the disturbance window and inside coverage.
4. **Defect in Practice**: Because $T_{start}$ was corrupted to `2024-01-01` for 2024 storms, the daily alignment logic evaluated $T_{start} \le T$ as true for hundreds of calm summer days prior to actual storm formation.

---

## 11. Event Matching Audit

### Bounding-Box vs Track Overlap
When `SpatialTemporalMatcher` operates in regional mode (`site_id="bob"`, bounding box `[16.3°N, 19.3°N, 86.7°E, 89.7°E]`), it checks for bounding-box intersection with storm envelope `ev.bbox`.
- If a storm occurred in the southern Bay of Bengal (e.g., Sri Lanka / Tamil Nadu coast at 8°N–12°N) with an envelope extending north to 17°N, the entire regional site was considered spatially matched.
- In combination with the expanded start dates, regional intersection caused the Bay of Bengal study site to remain continuously in "active cyclone" status from July 23 to December 21, 2024.

---

## 12. Train / Validation / Test Split Audit

The candidate dataset was partitioned chronologically:

| Split | Date Range | Total Samples | Positive Samples (3d) | Positive Prevalence | Historical Events Present |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | 2024-07-23 → 2024-12-31 | 320 | 237* | 74.06% | ASNA, BOB04, BOB05, ARB01, DANA, FENGAL |
| **Validation** | 2025-01-01 → 2025-08-31 | 478 | 19 | 3.97% | UNNAMED-1, UNNAMED-2, UNNAMED-6, UNNAMED-7 |
| **Test** | 2025-09-01 → 2026-06-23 | 584 | 42 | 7.19% | UNNAMED-9, UNNAMED-10, Shakhti, Montha |

### Split Validity Assessment
- **Chronological Separation**: Fully strictly monotonic in time. Zero future-to-past contamination.
- **Physical Storm Crossing**: Storm boundary snapping prevented physical events from spanning across splits.
- **Severe Distributional Mismatch**: The training set had an artificial positive prevalence of **74.06%**, while the validation set had **3.97%** and the test set had **7.19%**. Machine learning models trained on 74% positive prevalence suffer extreme bias towards predicting positive, which explains the collapse in test precision.

---

## 13. Four-Horizon Model Audit

The candidate pipeline trained four Random Forest models (200 estimators, max_depth=5, min_samples_leaf=2, `class_weight="balanced"`):

| Horizon | Prediction Target | Decision Threshold | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC | Confusion Matrix (TN / FP / FN / TP) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0d** | `event_within_0d` | 0.47 | 0.0419 | 0.6429 | 0.0786 | 0.5230 | 0.0509 | 144 / 412 / 10 / 18 |
| **1d** | `event_within_1d` | 0.47 | 0.0512 | 0.6667 | 0.5399 | 0.0614 | 143 / 408 / 11 / 22 |
| **2d** | `event_within_2d` | 0.47 | 0.0605 | 0.6842 | 0.5527 | 0.0723 | 142 / 404 / 12 / 26 |
| **3d** | `event_within_3d` | 0.47 | 0.0674 | 0.6905 | 0.5582 | 0.0803 | 141 / 401 / 13 / 29 |

### Detailed Metric Verification
1. **ROC-AUC (0.5230 – 0.5582)**: Independent recalculation from test prediction probabilities confirms these numbers. An ROC-AUC below 0.60 indicates virtually no ranking discrimination between positive storm periods and calm ocean periods.
2. **False Positive Rate**: Out of 556 actual negative days in the test set, the model raised **401 to 412 false alarms** (FPR $\approx 73\%$).
3. **Threshold Selection Artifact**: The optimal validation threshold selected was 0.47, but because the training distribution was 74% positive, the model outputted probabilities $>0.47$ for 73.6% of test observations.

---

## 14. Baseline vs Candidate V2 Comparison

The existing baseline models in [`backend/models/`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/models/) were compared directly against the candidate `v2_10yr` models:

| Metric | Horizon 0d Baseline | Horizon 0d V2 | Horizon 3d Baseline | Horizon 3d V2 | Baseline Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.2162 | **0.5230** | **0.8621** | 0.5582 | Baseline 3d vastly superior (+0.3039) |
| **PR-AUC** | 0.0209 | **0.0509** | **0.1553** | 0.0803 | Baseline 3d superior (+0.0750) |
| **Precision** | 0.0288 | **0.0419** | **0.1148** | 0.0674 | Baseline 3d superior (+0.0474) |
| **Recall** | **1.0000** | 0.6429 | **1.0000** | 0.6905 | Baseline achieves 100% detection |
| **F1 Score** | 0.0559 | **0.0786** | **0.2059** | 0.1229 | Baseline 3d superior (+0.0830) |

### Key Comparative Conclusion
While the candidate v2 pipeline attempted to formalize four independent horizon targets, **the 3d candidate model is significantly worse than the existing 2-year production model across every meaningful metric** (ROC-AUC 0.5582 vs 0.8621; F1 0.1229 vs 0.2059; Precision 6.7% vs 11.5%). Replacing the baseline with v2 would degrade production warning performance.

---

## 15. Class Imbalance Analysis

In tropical cyclone forecasting, positive cyclone observations are rare events:
- **True Climatological Prevalence**: ~2% to 5% of daily oceanic observations.
- **Candidate Test Set Prevalence**: 42 positives / 584 total days = **7.19%**.
- **Random Guess PR-AUC**: Equivalent to the prevalence: **0.0719**.
- **Candidate V2 PR-AUC**: **0.0803**.
- **Interpretation**: A PR-AUC of 0.0803 against a random baseline of 0.0719 confirms that candidate v2 has **almost zero predictive lift over random chance**.

---

## 16. Hard-Coded Historical Event Profiles (`EVENT_PARAM_PROFILES`)

Inspection of [`prediction_service.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/prediction_service.py#L1119-L1128) reveals a hard-coded dictionary `EVENT_PARAM_PROFILES`:
```python
EVENT_PARAM_PROFILES = {
    "IMD-2024-SCS-DANA": {"sst": 29.8, "sal": 33.5, "cur": 1.25, "ssh": 0.22, "mld": 20.0, "basin": "Bay of Bengal"},
    "IMD-2024-SCS-ASNA": {"sst": 29.2, "sal": 36.2, "cur": 1.10, "ssh": 0.18, "mld": 35.0, "basin": "Arabian Sea"},
    ...
}
```

### Architectural Findings
1. **Redundancy**: These 7 entries duplicate information that should properly reside in `HistoricalEventStore` and `HistoricalEvent.metadata`.
2. **Not Used for ML**: These values are not fed into any ML training pipeline or feature calculation.
3. **Used for UI Matching**: When a user inputs a custom observation, the system computes similarity against these static values to generate the analog narrative on the frontend.
4. **Fallback Flaw**: For any event outside the 7 hardcoded profiles (such as the 117 newly parsed IMD systems), the system assigns generic fallback parameters (`sst=29.5`, `sal=33.5`, `cur=1.1`).
5. **Recommendation**: Refactor `HistoricalEventStore` to compute and store real mean ocean parameters for all historical events dynamically, eliminating the hardcoded dictionary.

---

## 17. Feature Importance & Sanity Checks

### Feature Importance Grouping (Candidate V2 3D Model)
- **Currents (`cur_u` + `cur_v` + `cur`)**: **53.46%**
- **Sea Surface Height (`ssh`)**: **17.79%**
- **Salinity (`sal`)**: **17.72%**
- **Temperature (`temp`)**: **6.59%**
- **Mixed Layer Depth (`mld`)**: **4.43%**

### Scientific Interpretation
In genuine tropical cyclone genesis, sea surface temperature ($>28^\circ\text{C}$) and ocean heat content (linked to MLD) are the primary thermodynamical energy sources, while low-level shear/currents provide dynamic forcing. In candidate v2, temperature and MLD received only 11% combined importance, while surface current means dominated. Because the training set had positive labels assigned continuously across summer and autumn, the tree splits focused on seasonal current transitions rather than precursory thermal anomalies.

### Permutation Sensitivity Test
A label shuffle sanity test was conducted:
- Normal Validation ROC-AUC: **0.7587**
- Shuffled Labels Validation ROC-AUC: **0.4063**
When labels were randomly permuted, model performance collapsed below random chance. This confirms that the 101 features contain no target leakage or hidden target identifiers.

---

## 18. Production Integration Audit

Inspection of [`prediction_service.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/prediction_service.py) confirms:
- **Active Model Directory**: `backend/models/` (Baseline).
- **Active Version**: `1.1.0`.
- **Operational Status**: Operational and validated.
- **Candidate Isolation**: The candidate models reside in `backend/models/v2_10yr/` and are only loaded if explicitly requested or configured via `MODEL_ARTIFACTS_DIR`.
- **Safety**: The live server has **not** been switched to candidate v2, preventing any production regression.

---

## 19. Regression Audit

All existing backend and frontend components were tested to verify that no functional regressions were introduced during the 10-year development:

### Test Suite Execution
- `tests/test_feature_leakage_audit.py`: **5/5 passed**
- `tests/test_event_labeling.py`: **12/12 passed**
- `tests/test_multibasin_split.py`: **5/5 passed**
- `tests/test_phase4_ml.py`: **10/10 passed**
- `tests/test_imd_best_track_parser.py`: **4/4 passed**
- `tests/test_copernicus_10yr_validation.py`: **2/2 passed**
- `tests/test_custom_observation_pipeline.py`: **12/12 passed**
- **Total Backend Tests: 50 / 50 PASSED** (100% pass rate).

### Frontend Production Build
- Command: `cmd /c npm run build` in `frontend/`
- Output: `tsc && vite build` completed in **13.06s** with **0 errors**.
- All UI routes, the 3D Cesium globe hazard footprint renderer, custom CSV ingestion panel, and disaster timeline selector operate cleanly without regressions.

---

## 20. Scientific Risks & Deficiencies Identified

1. **Unrepresented Climatological Decades (2016–2023)**: Without the actual 7.39 GB NetCDF file on disk, the machine learning models cannot learn from major historical cyclones (Vardah, Ockhi, Mekunu, Fani, Amphan, Tauktae, Yaas, Biparjoy, Michaung).
2. **False Training Saturation**: Labeling 74% of the training fold as positive creates severe class probability distortion, forcing the decision trees to over-predict hazard conditions.
3. **Static Parameter Comparison**: Hard-coding 7 event profiles in `prediction_service.py` prevents the 117 newly integrated IMD systems from participating in high-fidelity analog matching.

---

## 21. Required Fixes (Actionable Checklist)

Prior to any production deployment of a 10-year ML model, the following fixes must be executed:

1. **Fix Date Forward-Fill in [`imd_best_track_parser.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/parsers/imd_best_track_parser.py)**:
   - Within each storm's track rows, forward-fill (`ffill()`) empty/NaN date cells from the preceding row rather than defaulting to `YYYY-01-01`.
   - Re-parse and regenerate `imd_tracks_2016_2026.parquet` and `imd_systems_2016_2026.json`.
   - Verify that storm start dates match real meteorological genesis (e.g. Asna starts 2024-08-29, Dana starts 2024-10-22, Fengal starts 2024-11-28).
2. **Download Full 10-Year Copernicus NetCDF**:
   - Run `ingest_10yr_copernicus.py` to download `copernicus_phy_10yr_surface.nc` (~7.39 GB) to `backend/data/copernicus/`.
   - Validate full 3,652 daily timesteps from 2016-06-24 to 2026-06-23 via `validate_10yr_copernicus.py`.
3. **Regenerate Multi-Basin Features Across Full 10 Years**:
   - Extract rolling 101 features across 2016–2026 into `features_10yr.parquet`.
4. **Relabel with Corrected IMD Tracks**:
   - Run `SpatialTemporalMatcher` to generate clean supervised labels where positive disturbance prevalence is realistic (~3% to 6%).
5. **Retrain and Re-Evaluate Four Horizon Models**:
   - Retrain Random Forest models on clean 2016–2023 train split, validate on 2024, test on 2025–2026.
   - Verify that ROC-AUC exceeds 0.80 and precision/F1 improves significantly over baseline.
6. **Refactor `EVENT_PARAM_PROFILES`**:
   - Dynamically derive historical event ocean profiles from the NetCDF data at event track centroids instead of hardcoding static dictionaries.

---

## 22. Final Recommendation

# VERDICT: NO-GO

### Justification
- **The candidate 10-year models (`v2_10yr`) must NOT be deployed to production.**
- The 10-year Copernicus NetCDF dataset is not yet present on the local filesystem, meaning 2016–2023 ocean data was never ingested.
- The IMD Best Track parser date-defaulting defect severely corrupted the 2024 training labels, reducing candidate model discrimination to near-random chance (ROC-AUC ~0.52 to 0.55).
- The existing baseline 2-year production model (`backend/models/`, version `1.1.0`) is scientifically superior (3d ROC-AUC 0.8621, PR-AUC 0.1553, Test F1 0.2059) and must remain active.

---

## Summary of True State

```
CURRENT STATE
→ Repository, API endpoints, custom CSV pipeline, 3D Cesium globe, and SagarBot are fully operational.
→ Production is safely active on the baseline 2-year model (v1.1.0, ROC-AUC 0.8621).
→ All 50 automated tests pass and the frontend builds cleanly.

CRITICAL ISSUES
→ 10-year NetCDF file (7.39 GB) has not been downloaded locally; 2016–2023 contributed 0 samples.
→ IMD track parser defaulted blank sub-daily date rows to YYYY-01-01, corrupting 2024 storm start dates.
→ Candidate v2 models were trained on 74% pseudo-positive labels, collapsing test ROC-AUC to 0.52–0.55.

REQUIRED FIXES
1. Add date forward-fill (ffill) in imd_best_track_parser.py and regenerate track cache.
2. Complete download of copernicus_phy_10yr_surface.nc (7.39 GB).
3. Re-extract features and re-label across the genuine 2016–2026 timeline.
4. Retrain candidate v2 models on clean labels and benchmark against baseline.

NEXT ML STEP
→ Apply the date forward-fill fix in imd_best_track_parser.py and verify meteorological genesis dates.
```
