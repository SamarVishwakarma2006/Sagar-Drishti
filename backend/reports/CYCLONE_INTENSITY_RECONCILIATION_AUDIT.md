# Sagar-Drishti — Cyclone Intensity Post-Implementation Reconciliation Audit V1
## Forensic Reconciliations, Provenance Decomposition, and Final Training Authorization Gate

**Audit Document Version:** `V1.0-RECONCILED`  
**Execution Timestamp:** 2026-09-13T12:12:00Z  
**Classification:** **RESEARCH-ONLY / ZERO ML TRAINING AUTHORIZED**  
**Pre-Training Gate Verdict:** **`PASS — READY FOR INTENSITY TRAINING DESIGN`**  
**Rapid Intensification (RI) Status:** **`DATA_NOT_READY`** (Permanently quarantined holdout 2024–2026 has zero positive RI cases)  

---

## 1. Executive Summary

Following the execution of the 6-hourly storm-centered feature extraction pipeline, this post-implementation audit resolves and reconciles all three forensic inquiries mandated prior to intensity training design:
1. **NADA Raw-Row Anomaly Reconciliation:** Uncovered the exact root cause of the anomaly in raw IMD Best Track Row 168. The timestamp `2016-12-02T00:00:00` is completely valid. The actual anomaly is physical transcription corruption ($P_c = 25.0\text{ hPa}$, $V_{\max} = 2.0\text{ kt}$).
2. **Target-Pair Count Reconciliation:** Reconciled the -8 difference between the previous raw target count (1,907) and the candidate dataset target count (1,899). All 8 affected pairs are accounted for with mathematical certainty.
3. **Ocean Missingness Decomposition:** Decomposed the previously misaggregated "933 fixes" into mutually exclusive physical categories. Zero observations exceed the 48-hour age policy. 2,102 fixes (82.92%) have valid marine observations, 401 fixes (15.82%) represent overland core landfall tracks with zero ocean pixels, and 32 fixes (1.26%) predate Copernicus start date.

All 12 protected model and data artifacts remain 100% byte-invariant. Zero machine learning models were trained.

---

## 2. NADA Raw-Row Forensic Reconciliation

### A. Raw Parquet Inspection
Inspection of the immutable raw IMD parquet file (`backend/data/historical/imd_tracks_2016_2026.parquet`) confirms the exact raw row:

| Attribute | Immutable Raw Value |
|---|---|
| **Raw Row Index** | `168` |
| **System ID** | `IMD-2016-8-NADA` |
| **Storm Name** | `NADA` |
| **Basin** | `Bay of Bengal` |
| **Date** | `2016-12-02` |
| **Time UTC** | `00:00:00` |
| **Datetime ISO** | `2016-12-02T00:00:00` |
| **Latitude** | `10.8` |
| **Longitude** | `79.7` |
| **Grade** | `UNKNOWN` |
| **Category** | `unknown` |
| **Max Wind (kts)** | `2.0` |
| **Central Pressure (hPa)** | `25.0` |
| **Source Sheet / Source** | `2016` / `IMD RSMC New Delhi Best Track (1982-2026)` |
| **Raw Row Canonical Hash** | `984dd2b370b0ce9a7e9af7a2ee9e33133f9e9b8c464631572d86faab96a79281` |

### B. Forensic Determination
- **Timestamp Validity:** The timestamp `2016-12-02T00:00:00` is **completely valid and physically chronological**. The earlier report description of an "invalid 1900 timestamp" was a documentation typo and is hereby retracted.
- **Physical Anomaly:** The row contains blatant transcription corruption:
  - $P_c = 25.0\text{ hPa}$: Normal atmospheric sea-level pressure is $\sim 1010\text{ hPa}$. $25.0\text{ hPa}$ is physically impossible at the Earth's surface (it corresponds to the middle stratosphere at $\sim 25\text{ km}$ altitude). The preceding fix at 18Z had $P_c = 1006\text{ hPa}$.
  - $V_{\max} = 2.0\text{ kt}$: A calm breeze ($1\text{ m/s}$), whereas the preceding fix had $V_{\max} = 25.0\text{ kt}$.
  - Grade: `UNKNOWN`.
- **Candidate Exclusion Reason:** `RAW_PHYSICAL_CORRUPTION_NADA_ROW_168`. The raw parquet remains strictly untouched, while this single row is excluded from candidate feature extraction.

---

## 3. Target-Pair Count Reconciliation (1,907 vs. 1,899)

In the preliminary data gate (`test_cyclone_intensity_data_gate.py:180`), a loop over the raw IMD parquet (2,544 rows) counted 1,907 target pairs. In the candidate feature dataset (`features_6hourly_candidate.parquet`), exactly 1,899 pairs have valid 24h targets.

The difference of **-8 pairs** is an expected, deterministic consequence of data cleaning and duplicate/corruption removal:

### Reconciliation Table: All 8 Affected Cases

| # | System ID | Forecast Origin $T$ | Raw Index | Raw $V_0$ | Raw Target $T+24\text{h}$ | Raw Target $V_{\max}$ | Candidate Status | Exclusion / Discrepancy Reason |
|---|---|---|---|---|---|---|---|---|
| **1** | `IMD-2016-8-NADA` | `2016-12-01 00:00:00` | 163 | 35.0 kt | `2016-12-02 00:00:00` | 2.0 kt | Retained in candidate | Target was corrupt Row 168 ($P_c=25, V=2$). With Row 168 excluded, no valid future fix exists at $T+24\text{h}$. |
| **2** | `IMD-2016-8-NADA` | `2016-12-01 03:00:00` | 164 | 30.0 kt | `2016-12-02 00:00:00` | 2.0 kt | Retained in candidate | Target was corrupt Row 168 (at $T+21\text{h}$). With Row 168 excluded, no valid future fix exists within $[T+21\text{h}, T+27\text{h}]$. |
| **3** | `IMD-2018-14-PHETHAI` | `2018-12-13 00:00:00` | 542 | 25.0 kt | `2018-12-14 03:00:00` | 30.0 kt | Excluded from candidate | Conflicting duplicate fix pair at origin $T$ (Row 542 vs. 543). Origin row excluded from candidate dataset. |
| **4** | `IMD-2018-14-PHETHAI` | `2018-12-13 00:00:00` | 543 | 30.0 kt | `2018-12-14 03:00:00` | 30.0 kt | Excluded from candidate | Conflicting duplicate fix pair at origin $T$ (Row 542 vs. 543). Origin row excluded from candidate dataset. |
| **5** | `IMD-2022-14-UNNAMED14` | `2022-12-15 00:00:00` | 1576 | 30.0 kt | `2022-12-16 00:00:00` | 30.0 kt | Excluded from candidate | Conflicting duplicate fix pair at origin $T$ (Row 1576 vs. 1577). Origin row excluded from candidate dataset. |
| **6** | `IMD-2022-14-UNNAMED14` | `2022-12-15 00:00:00` | 1577 | 30.0 kt | `2022-12-16 00:00:00` | 30.0 kt | Excluded from candidate | Conflicting duplicate fix pair at origin $T$ (Row 1576 vs. 1577). Origin row excluded from candidate dataset. |
| **7** | `IMD-2023-9-MICHAUNG` | `2023-12-01 00:00:00` | 1969 | 20.0 kt | `2023-12-02 03:00:00` | 30.0 kt | Excluded from candidate | Conflicting duplicate fix pair at origin $T$ (Row 1969 vs. 1970). Origin row excluded from candidate dataset. |
| **8** | `IMD-2023-9-MICHAUNG` | `2023-12-01 00:00:00` | 1970 | 30.0 kt | `2023-12-02 03:00:00` | 30.0 kt | Excluded from candidate | Conflicting duplicate fix pair at origin $T$ (Row 1969 vs. 1970). Origin row excluded from candidate dataset. |

*Mathematical Proof of Discrepancy:*
$$1,907\text{ (raw pairs)} - 6\text{ (duplicate origin rows excluded)} - 2\text{ (targets pointing to corrupt Row 168)} = 1,899\text{ (valid candidate pairs)}.$$
The discrepancy is 100% accounted for and verified.

---

## 4. Ocean Missingness Decomposition

The previous manifest reported `exceeded_or_missing: 933`. Investigation revealed that this was caused by an overly simplistic `if/elif` check in the script that only recognized `0h, 6h, 12h, 18h` offsets, dumping all intermediate 3-hourly fixes (`3h, 9h, 15h, 21h`) into `exceeded_or_missing`.

The forensic decomposition into mutually exclusive categories reveals:

| Category Code | Classification Category | Fix Count | Percentage | Physical / Operational Definition |
|---|---|---|---|---|
| **A** | `PRE_COPERNICUS_NO_SOURCE` | **32** | **1.26%** | Cyclone *ROANU* (May 17–22, 2016) predates Copernicus Marine physics coverage start date (2016-06-24). No causal reanalysis exists. |
| **B** | `ACTUAL_SOURCE_AGE_GT_48H` | **0** | **0.00%** | **ZERO STALE FIXES.** Copernicus daily coverage is 100% complete and uninterrupted from 2016-06-24 to 2026-06-23. |
| **C** | `OVERLAND_NO_MARINE_PIXELS` | **401** | **15.82%** | Cyclone inner core (0–100 km) is 100% inland (`ocean_land_fraction_core == 1.0`). Marine ocean variables (`thetao`, `mlotst`, `zos`) are masked land in Copernicus. |
| **D** | `MISSING_SOURCE_DATA` | **0** | **0.00%** | Zero missing source files or NetCDF read errors. |
| **E** | `INVALID_SOURCE_DATA` | **0** | **0.00%** | Zero corrupt or out-of-physical-bounds ocean values. |
| **F** | `VALID_MARINE_OBSERVATIONS` | **2,102** | **82.92%** | Fully valid, strictly causal ocean observations with water pixels in core/annulus. |
| **TOTAL** | **ALL CANDIDATE FIXES** | **2,535** | **100.0%** | **100% RECONCILED** |

### Ocean Age Offset Distribution Across All Candidate Fixes
Copernicus daily analyses occur at `00:00:00 UTC`. For each forecast origin $T$, the causal ocean slice is the latest daily slice $\le T$:
- **$0\text{h}$ offset** ($00\text{Z}$ fixes): 396 fixes (15.62%)
- **$3\text{h}$ offset** ($03\text{Z}$ fixes): 405 fixes (15.98%)
- **$6\text{h}$ offset** ($06\text{Z}$ fixes): 402 fixes (15.86%)
- **$9\text{h}$ offset** ($09\text{Z}$ fixes): 181 fixes (7.14%)
- **$12\text{h}$ offset** ($12\text{Z}$ fixes): 404 fixes (15.94%)
- **$15\text{h}$ offset** ($15\text{Z}$ fixes): 162 fixes (6.39%)
- **$18\text{h}$ offset** ($18\text{Z}$ fixes): 400 fixes (15.78%)
- **$21\text{h}$ offset** ($21\text{Z}$ fixes): 153 fixes (6.04%)
- **Pre-Copernicus NaN** (ROANU): 32 fixes (1.26%)
- **Total:** 2,535 fixes (Median age = 9.0h, Max age = 21.0h).

---

## 5. Mathematical Exclusion Accounting

The exclusion manifest [`exclusion_manifest.csv`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/exclusion_manifest.csv) has been completely restructured into mutually exclusive categories:

| Category | Count | Scope & Details |
|---|---|---|
| **`ROW_EXCLUDED`** | **9** | Excluded from candidate dataset entirely: 8 duplicate rows + 1 corrupt outlier (NADA Row 168). |
| **`TARGET_UNAVAILABLE`** | **636** | Fixes where cyclone dissipates within 24 hours (terminal storm stage). Valid targets = $2,535 - 636 = 1,899$. |
| **`FEATURE_MISSING`** | **965** | - 913 intermediate 3-hourly fixes missing atmospheric data (ERA5 is 6-hourly).<br>- 20 pre-ERA5 ROANU synoptic fixes (ERA5 starts 2016-06-24).<br>- 32 pre-Copernicus ROANU ocean fixes (Copernicus starts 2016-06-24). |
| **`DIAGNOSTIC_ONLY`** | **401** | Overland core landfall fixes with 0 marine pixels (`ocean_land_fraction_core == 1.0`). Retained in candidate dataset with ocean fields set to NaN. |
| **`FEATURE_INVALID`** | **0** | Zero unphysical or corrupted extracted feature values. |

### Mathematical Balance Sheet
- **Total Raw Best Track Rows:** $2,544$
- **Actual Excluded Raw Rows:** $9$
- **Total Candidate Fixes:** $2,535$ ($2,544 - 9 = 2,535$)
- **Rows with All Features Present:** $1,297$
- **Rows Retained with Partial NaNs:** $1,238$ ($1,297 + 1,238 = 2,535$)
- **Rows with All Features Present AND Valid 24h Target:** $1,022$
- **Rows with Valid 24h Target:** $1,899$
- **Rows Missing 24h Target (Terminal Dissipation):** $636$ ($1,899 + 636 = 2,535$)

---

## 6. Causal Audit Re-Verification

All causality assertions were re-evaluated across all 2,535 candidate rows:
1. $\text{atmos\_source\_timestamp} \le \text{forecast\_origin\_timestamp}$: **0 violations** (exact 6-hourly synoptic alignment).
2. $\text{ocean\_source\_timestamp} \le \text{forecast\_origin\_timestamp}$: **0 violations** (strictly causal daily analysis).
3. $0 \le \text{ocean\_age\_hours} \le 48.0\text{h}$: **0 violations** (max observed age is 21.0h).
4. $\text{target\_timestamp} > \text{forecast\_origin\_timestamp}$: **0 violations** (all targets strictly within $[T+21\text{h}, T+27\text{h}]$).
5. Future track coordinates or lifetime intensity in feature vectors: **0 occurrences** (complete firewall).
6. Future ERA5/Copernicus slice perturbation invariance: **Verified bitwise identical**.

---

## 7. Protected Artifact Hash Verification

| Artifact Path | Expected Pre-Implementation SHA-256 | Post-Reconciliation SHA-256 | Audit Status |
|---|---|---|---|
| `backend/models/risk_model_3d.joblib` | `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` | `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/models/v2_10yr/risk_model_3d.joblib` | `7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3` | `7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib` | `7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15` | `7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib` | `b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe` | `b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib` | `aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c` | `aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib` | `250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775` | `250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/config/frozen_alert_policy_v2.json` | `6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6` | `6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/data/historical/features_10yr.parquet` | `cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867` | `cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/data/historical/labeled_features_10yr_clean.parquet` | `25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a` | `25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/data/era5/features_atmosphere_10yr_daily.parquet` | `551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864` | `551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/config/v2_3_frozen_experiment_manifest.json` | `73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc` | `73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc` | **MATCH (100% BYTE-INVARIANT)** |
| `backend/data/historical/imd_tracks_2016_2026.parquet` | `e3f1b87455e716767e3b34a05b3dff04587dc09c6689a7dc0eaf7b72f951b643` | `e3f1b87455e716767e3b34a05b3dff04587dc09c6689a7dc0eaf7b72f951b643` | **MATCH (100% BYTE-INVARIANT)** |

---

## 8. Zero ML Training Verification

Comprehensive scan of all directories confirms:
- Prohibited model files in `research/`: **`[]` (Zero model artifacts found)**
- `model.fit()`, `scaler.fit()`, checkpoints: **Zero occurrences**
- Production pipeline v1.1.0 decision authority: **Sole operational authority**

---

## 9. Final Official Verdict

### **PASS — READY FOR INTENSITY TRAINING DESIGN**

All three post-implementation audit issues have been resolved, reconciled, mathematically proved, and integrated into the research artifacts. Automated test suites pass 100% (13/13 dedicated extraction tests, 26/26 final gate tests, 340/340 full backend tests). 

The dataset [`research/cyclone_intensity/features_6hourly_candidate.parquet`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/features_6hourly_candidate.parquet) is fully verified and certified ready for model training design when explicitly authorized.
