# Sagar-Drishti — Cyclone Intensity 6-Hourly Feature Extraction Dataset
## RESEARCH-ONLY CANDIDATE DATASET — ZERO ML TRAINING / NO OPERATIONAL AUTHORITY

**Pipeline Version:** `SD-INTENSITY-FEATURE-PIPELINE-V1.0`  
**Dataset Path:** `research/cyclone_intensity/features_6hourly_candidate.parquet`  
**Dataset SHA-256:** `8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0`  
**Audit Status:** `POST_IMPLEMENTATION_RECONCILED` (2026-09-13)  
**Target Variable:** Primary continuous $V_{\max}(T+24\text{h})$ (knots) | Secondary continuous $P_c(T+24\text{h})$ (hPa)  
**RI Module Status:** **`DATA_NOT_READY`** (Quarantined holdout 2024–2026 has zero positive cases)  

---

## 1. Governance & Legal Firewalls

> [!IMPORTANT]
> **RESEARCH-ONLY CLASSIFICATION**  
> This dataset is constructed exclusively for retrospective offline scientific research and experimental design.
> It is **NOT**:
> - An operational forecast model.
> - Approved for production decision-making.
> - An operational warning issuance authority.
> - Ground truth for atmospheric or ocean variables.
> Production v1.1.0 (`backend/models/risk_model_3d.joblib`) remains the sole operational decision authority.

---

## 2. Scientific Motivation & Spatial-Temporal Architecture

### A. Why Storm-Centered Spatial Domains Are Required
Fixed-site marine buoys (e.g. AS1-6, BOB1-6) and basin-wide geographic averages fail to represent the physical micro-environment of moving tropical cyclones.
Two concentric storm-centered spatial zones are extracted centered on $(\text{lat}_T, \text{lon}_T)$:
1. **Inner Core ($0 - 100\text{ km}$ radius):** Captures eyewall latent heat supply, inner-core SST enthalpy, mixed layer depth buffering against cyclonic cold wakes, and core relative vorticity convergence.
2. **Environmental Annulus ($200 - 800\text{ km}$ radius):** Captures synoptic vertical wind shear ($850 - 200\text{ hPa}$) and mid-tropospheric dry air inflow ($700\text{ hPa}$ and $500\text{ hPa}$ relative humidity) that ventilate and disrupt the warm core.

### B. Why Daily 18Z-Derived Features Were Insufficient
Daily atmospheric features aggregated across 00Z–18Z contain afternoon/evening atmospheric conditions. When evaluated for morning synoptic fixes ($00\text{Z}$ or $06\text{Z}$), daily composites cause catastrophic acausal look-ahead leakage. This pipeline enforces **strictly synoptic 6-hourly atmospheric extraction at $00\text{Z}, 06\text{Z}, 12\text{Z}, 18\text{Z}$**, guaranteeing that no post-$T$ atmospheric analysis is ingested.

### C. The Critical Ocean Temporal-Resolution Rule
Copernicus Marine physical reanalysis is natively **daily** (`00:00:00 UTC`).
- Daily ocean data is **never linearly interpolated or relabeled as 6-hourly**.
- For each forecast origin $T$, the latest daily ocean slice satisfying $t_{\text{ocean}} \le T$ is used.
- Every record explicitly tracks: `ocean_source_timestamp`, `ocean_age_hours = T - ocean_source_timestamp`, and `ocean_temporal_resolution = 'daily'`.
- Configured maximum age: **48.0 hours**. If no observation exists within 48h (e.g. *ROANU* in May 2016 before Copernicus begins), ocean features are marked `NaN` and recorded in `exclusion_manifest.csv`.

---

## 3. Post-Implementation Forensic Reconciliations

### A. NADA Raw-Row 168 Anomaly
- **Raw Parquet Value:** `IMD-2016-8-NADA`, Date: `2016-12-02 00:00:00`, $P_c = 25.0\text{ hPa}$, $V_{\max} = 2.0\text{ kt}$, Grade: `UNKNOWN`.
- **Finding:** The datetime `2016-12-02T00:00:00` is **completely valid**. The actual anomaly is physical transcription corruption ($25.0\text{ hPa}$ is stratospheric; preceding fix was $1006\text{ hPa}$).
- **Action:** Row 168 is excluded from candidate data (`RAW_PHYSICAL_CORRUPTION_NADA_ROW_168`). Raw file remains unmodified.

### B. Target-Pair Reconciliation (Raw 1,907 vs. Candidate 1,899)
The difference of **-8 pairs** is mathematically proved:
- **-6 pairs:** 3 conflicting duplicate pairs in raw data (PHETHAI rows 542 & 543, UNNAMED14 rows 1576 & 1577, MICHAUNG rows 1969 & 1970) formed pairs in raw data, but duplicate origin rows were correctly excluded from candidate data.
- **-2 pairs:** Candidate fixes 163 and 164 in NADA had their only potential future target pointing to corrupt Row 168. With Row 168 removed, they correctly have no valid target.
- **Formula:** $1,907 - 6 - 2 = 1,899$ valid candidate pairs.

### C. Ocean Provenance Decomposition
Decomposed into mutually exclusive categories:
- **A. `PRE_COPERNICUS_NO_SOURCE`:** 32 fixes (1.26%) — Cyclone *ROANU* (May 2016) predates dataset start.
- **B. `ACTUAL_SOURCE_AGE_GT_48H`:** 0 fixes (0.00%) — Zero stale observations.
- **C. `OVERLAND_NO_MARINE_PIXELS`:** 401 fixes (15.82%) — Cyclone core inland with 0 ocean water pixels.
- **D. `MISSING_SOURCE_DATA`:** 0 fixes (0.00%).
- **E. `INVALID_SOURCE_DATA`:** 0 fixes (0.00%).
- **F. `VALID_MARINE_OBSERVATIONS`:** 2,102 fixes (82.92%) — Strictly causal marine observations.

---

## 4. Dataset Summary & Balance Sheet

- **Total Raw Best Track Fixes:** 2,544
- **Actual Excluded Raw Rows:** 9 (8 duplicate rows + 1 corrupt row 168)
- **Candidate Valid Fixes:** 2,535
- **Valid 24h Target Pairs:** 1,899
- **Rows with Missing 24h Target (Terminal Dissipation):** 636
- **Unique Storm Systems:** 117
- **Chronological Partitions:**
  - **TRAIN (2016–2021):** 64 storms | 1,488 fixes | 1,121 valid pairs
  - **VALIDATION (2022–2023):** 24 storms | 506 fixes | 382 valid pairs
  - **TEST (2024–2026):** 29 storms | 541 fixes | 396 valid pairs
- **Zero Storm Overlap:** Confirmed $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$.
- **Exclusion Manifest Breakdown:**
  - `ROW_EXCLUDED`: 9 rows
  - `TARGET_UNAVAILABLE`: 636 rows
  - `FEATURE_MISSING`: 965 rows (913 intermediate 3h atmos + 20 pre-ERA5 atmos + 32 pre-Copernicus ocean)
  - `DIAGNOSTIC_ONLY`: 401 rows (overland core landfall)
  - `FEATURE_INVALID`: 0 rows

---

## 5. Benchmark Provenance Status

The aspirational target benchmark `MAE < 11.5 kt at 24h` is officially designated:
**`PROVISIONAL / UNPROVEN`**  
It reflects operational IMD/NHC 24-hour verification statistics in annual cyclone bulletins, but currently has no formal empirical ablation baseline within Sagar-Drishti. Model evaluation will benchmark against persistence baselines and historical mean models rather than uncalibrated thresholds.

---

## 6. Anti-Leakage & Physical Integrity Assertions

1. All features satisfy $\text{feature\_timestamp} \le T$.
2. Target $V_{\max}(T+24\text{h})$ and $P_c(T+24\text{h})$ are isolated strictly into target columns and never included in feature vectors.
3. Zero models trained, fitted, or saved.
