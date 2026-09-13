# Sagar-Drishti — Cyclone Intensity Final Data Sufficiency & Pre-Training Gate V2
## FORENSIC DATA / SCIENCE / LEAKAGE / EXPERIMENTAL DESIGN AUDIT — ZERO ML TRAINING

**Document ID:** `SD-GATE-2026-CYCLONE-INTENSITY-FINAL-V2`  
**Classification:** `RESEARCH_ONLY` | `NOT_FOR_PRODUCTION` | `NO_OPERATIONAL_AUTHORITY`  
**Date:** September 13, 2026  
**Evaluated Modules:** 
1. **Cyclone Intensity Module:** Continuous 24h Maximum Sustained Wind ($V_{\max}$) Regression
2. **Rapid Intensification (RI) Module:** Discrete 24h Intensification Classifier ($\Delta V_{\max} \ge 30\text{ kt}$)

---

## 1. Absolute Guardrails & Governance Compliance (Section 1)

In strict accordance with the mandatory non-negotiable project guardrails:
1. **ZERO ML TRAINING PERFORMED:**
   - No models were trained, retrained, fine-tuned, or calibrated.
   - No regressors, classifiers, scalers, or encoders were fitted.
   - No hyperparameters were searched or optimized.
   - No checkpoints, `.joblib`, `.pkl`, `.onnx`, or model weight artifacts were generated.
2. **Operational Production Immutability:**
   - Production model v1.1.0 (`backend/models/risk_model_3d.joblib`) remains 100% byte-for-byte untouched.
   - Candidate V2.3 Model D (`backend/models/candidates/v2_3/ocean_atmos_all/`) remains 100% untouched.
   - Frozen operational alert policy (`backend/config/frozen_alert_policy_v2.json`) remains 100% untouched.
   - Raw IMD Best Track dataset (`backend/data/historical/imd_tracks_2016_2026.parquet`) remains 100% byte-for-byte immutable.
3. **Legal & Operational Firewall:**
   - All work within this gate is designated: `RESEARCH_ONLY` | `NOT_FOR_PRODUCTION` | `NO_OPERATIONAL_AUTHORITY`.

---

## 2. Cryptographic Protected Artifact Hash Audit (Section 2)

Before and after the forensic implementation, SHA-256 cryptographic hashes were verified for all 11 protected artifacts and the raw IMD dataset:

```
CRYPTOGRAPHIC PROTECTED ARTIFACT HASH AUDIT
┌───────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┬────────┐
│ Protected Artifact Path                                           │ SHA-256 Checksum                                                 │ Status │
├───────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┼────────┤
│ backend/models/risk_model_3d.joblib                               │ 3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740│ PASS   │
│ backend/models/v2_10yr/risk_model_3d.joblib                       │ 7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3│ PASS   │
│ backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib│ 7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15│ PASS   │
│ backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib│ b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe│ PASS   │
│ backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib│ aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c│ PASS   │
│ backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib│ 250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775│ PASS   │
│ backend/config/frozen_alert_policy_v2.json                        │ 6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6│ PASS   │
│ backend/data/historical/features_10yr.parquet                     │ cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867│ PASS   │
│ backend/data/historical/labeled_features_10yr_clean.parquet       │ 25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a│ PASS   │
│ backend/data/era5/features_atmosphere_10yr_daily.parquet         │ 551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864│ PASS   │
│ backend/config/v2_3_frozen_experiment_manifest.json               │ 73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc│ PASS   │
│ [RAW DATA] backend/data/historical/imd_tracks_2016_2026.parquet   │ e3f1b87455e716767e3b34a05b3dff04587dc09c6689a7dc0eaf7b72f951b643│ PASS   │
└───────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────┴────────┘
```
**Conclusion:** 100% match across all 11 protected artifacts and raw IMD tracks. Zero unintended mutations occurred.

---

## 3. IMD Raw Data Forensic Re-Audit (Section 3)

The raw IMD Best Track Parquet archive was re-audited across all rows:
- **Total Unique Storm Systems:** 117 storms (2016-05-17 to 2026-01-10).
- **Total Synoptic Fixes:** 2,544 rows.
- **Fix Count per Storm Distribution:**
  - Minimum: 3 fixes
  - 25th Percentile: 11 fixes
  - Median: 17 fixes
  - Mean: 21.7 fixes
  - 75th Percentile: 29 fixes
  - Maximum: 97 fixes (Extremely Severe Cyclonic Storm *BIPARJOY* 2023)
- **Storm Duration Distribution:**
  - Minimum duration: 12.0 hours
  - Median duration: 69.0 hours (~2.9 days)
  - Mean duration: 115.2 hours (~4.8 days)
  - Maximum duration: 408.0 hours (~17.0 days, *KYAAR* 2019)
- **Timestamp Completeness:**
  - 2,544 valid ISO timestamps; 0 null timestamps.
  - Fix interval: Standard 6-hourly synoptic observations (00Z, 06Z, 12Z, 18Z), with 3-hourly special observations during active landfalls.
- **Wind Speed Distribution:**
  - Minimum: 2.0 kt (NADA row 168)
  - 25th percentile: 25.0 kt
  - Median: 30.0 kt
  - Mean: 37.9 kt
  - 75th percentile: 45.0 kt
  - Maximum: 130.0 kt (*KYAAR* 2019, *AMPHAN* 2020)
- **Central Pressure Distribution:**
  - Minimum: 25.0 hPa (NADA row 168 outlier)
  - Normal Physical Range: 906.0 hPa (*AMPHAN*) to 1008.0 hPa (weak depressions).
- **Missing Values:** Zero missing values in `system_id`, `datetime_iso`, `latitude`, `longitude`, `max_wind_kts`, `central_pressure_hpa`.

---

## 4. IMD-2016-8-NADA Anomaly Forensic Provenance (Section 4)

### Forensic Profile
- **System:** `IMD-2016-8-NADA`
- **Raw Row Index:** Row 168
- **Timestamp:** 2016-12-02T00:00:00
- **Recorded Values:** `central_pressure_hpa = 25.0`, `max_wind_kts = 2.0`, `grade = 'UNKNOWN'`

### Surrounding Synoptic Sequence
- **2016-12-01T12:00:00 (Row 166):** $V_{\max} = 30.0\text{ kt}$, $P_c = 1004.0\text{ hPa}$, Grade: `DD`
- **2016-12-01T18:00:00 (Row 167):** $V_{\max} = 25.0\text{ kt}$, $P_c = 1006.0\text{ hPa}$, Grade: `D`
- **2016-12-02T00:00:00 (Row 168):** $V_{\max} = 2.0\text{ kt}$, $P_c = 25.0\text{ hPa}$, Grade: `UNKNOWN`
- **2016-12-02T03:00:00 (Row 169):** $V_{\max} = 25.0\text{ kt}$, $P_c = 1006.0\text{ hPa}$, Grade: `D`

### Forensic Classification & Candidate Handling
- **Classification:** `FORENSIC_ANOMALY_REQUIRING_PROVENANCE_CONFIRMATION`
- **Scientific Rationale:** A surface pressure of $25\text{ hPa}$ is physically impossible at sea level (the standard atmosphere is $\approx 1013.25\text{ hPa}$; $25\text{ hPa}$ corresponds to the middle stratosphere at $\approx 25\text{ km}$ altitude). While the sequence is consistent with a clerical transcription truncation (e.g. pressure deficit $25\text{ hPa}$ or intended $1008\text{ hPa}$), scientific integrity dictates that without official published corrigenda, we must **not** modify the raw dataset.
- **Candidate Dataset Handling:** `EXCLUDED_FROM_CANDIDATE_DATASET`.
- **Raw Data Status:** Row 168 remains unmodified in `imd_tracks_2016_2026.parquet`.

---

## 5. Duplicate Fix Conflicts Audit (Section 5)

Forensic scanning of all $(system\_id, datetime\_iso)$ tuples identifies exactly 4 duplicate pairs (8 rows total):

```
CONFLICTING DUPLICATE ROWS AUDIT
┌──────┬──────────────────────┬─────────────────────┬──────────┬──────────┬──────────┬──────────┬───────┬─────────────────────────────┐
│ Row  │ System ID            │ Timestamp (ISO)     │ Lat (°N) │ Lon (°E) │ Vmax (kt)│ Pc (hPa) │ Grade │ Physical Conflict Detected  │
├──────┼──────────────────────┼─────────────────────┼──────────┼──────────┼──────────┼──────────┼───────┼─────────────────────────────┤
│ 542  │ IMD-2018-14-PHETHAI  │ 2018-12-13T00:00:00 │ 6.5      │ 88.7     │ 25.0     │ 1004.0   │ D     │ Position Δ1.7°N, Δ1.1°E     │
│ 543  │ IMD-2018-14-PHETHAI  │ 2018-12-13T00:00:00 │ 8.2      │ 87.6     │ 30.0     │ 1002.0   │ DD    │ Wind Δ5 kt, Press Δ2 hPa    │
├──────┼──────────────────────┼─────────────────────┼──────────┼──────────┼──────────┼──────────┼───────┼─────────────────────────────┤
│ 1488 │ IMD-2021-9-UNNAMED9  │ 2021-11-18T03:00:00 │ 11.0     │ 82.3     │ 25.0     │ 1000.0   │ D     │ Position Δ1.7°N, Δ2.6°E     │
│ 1489 │ IMD-2021-9-UNNAMED9  │ 2021-11-18T03:00:00 │ 12.7     │ 79.7     │ 20.0     │ 1002.0   │ D     │ Wind Δ5 kt, Press Δ2 hPa    │
├──────┼──────────────────────┼─────────────────────┼──────────┼──────────┼──────────┼──────────┼───────┼─────────────────────────────┤
│ 1576 │ IMD-2022-14-UNNAMED14│ 2022-12-15T00:00:00 │ 13.9     │ 68.2     │ 30.0     │ 1000.0   │ DD    │ Longitude Δ0.7°E            │
│ 1577 │ IMD-2022-14-UNNAMED14│ 2022-12-15T00:00:00 │ 13.9     │ 67.5     │ 30.0     │ 1000.0   │ DD    │ Positional uncertainty      │
├──────┼──────────────────────┼─────────────────────┼──────────┼──────────┼──────────┼──────────┼───────┼─────────────────────────────┤
│ 1969 │ IMD-2023-9-MICHAUNG  │ 2023-12-01T00:00:00 │ 9.1      │ 86.4     │ 20.0     │ 1002.0   │ D     │ Position Δ1.4°N, Δ2.3°E     │
│ 1970 │ IMD-2023-9-MICHAUNG  │ 2023-12-01T00:00:00 │ 10.5     │ 84.1     │ 30.0     │ 998.0    │ DD    │ Wind Δ10 kt, Press Δ4 hPa   │
└──────┴──────────────────────┴─────────────────────┴──────────┴──────────┴──────────┴──────────┴───────┴─────────────────────────────┘
```

### Deterministic Resolution Policy
- Because no authoritative IMD corrigenda exist specifying which conflicting fix is canonical, **arbitrarily choosing one row is prohibited**.
- **Candidate Dataset Handling:** All 8 rows receive status `EXCLUDED_FROM_CANDIDATE_DATASET`.
- **Raw Parquet:** All 8 rows remain byte-for-byte immutable in raw storage.
- **Candidate Valid Fix Count:** $2,544 - 1\text{ (NADA)} - 8\text{ (Duplicates)} = 2,535\text{ valid fixes}$.

---

## 6. Storm-Level Effective Sample Size & Temporal Dependence (Section 6)

> [!IMPORTANT]
> **STATISTICAL INDEPENDENCE WARNING**  
> "Fix-level observations are temporally dependent and must not be interpreted as independent training examples."

### Quantitative Verification
- **Total Raw Fixes:** 2,544 fixes across 117 storms.
- **Temporal Autocorrelation:**
  - Across all 89 storms with variance: mean lag-1 $r = 0.7712$, median lag-1 $r = 0.8485$.
  - Across the 49 cyclonic storms ($\ge 34\text{ kt}$): mean lag-1 $r = 0.9142$, median lag-1 $r = 0.9330$.
  - Pooled lag-1 within-storm correlation: $r = 0.9843$ for $V_{\max}$, $r = 0.9874$ for $P_c$.
- **Intensity Distribution by Storm Peak:**
  - 68 storms (58.1%) never exceed depression / deep depression strength ($\le 33\text{ kt}$).
  - 49 storms (41.9%) attain Cyclonic Storm intensity ($\ge 34\text{ kt}$).
  - 21 storms (17.9%) attain Very Severe Cyclonic Storm intensity ($\ge 64\text{ kt}$).
  - Only 2 storms (1.7%) attain Super Cyclonic Storm intensity ($\ge 120\text{ kt}$): *KYAAR* (2019, 130 kt) and *AMPHAN* (2020, 130 kt).
- **Implication:** The effective degrees of freedom are governed by storm count ($N=117$) rather than row count ($N=2,544$).

---

## 7. Intensity Target Specification (Section 7)

- **Primary Machine Learning Target:**
  $$V_{\max}(T + 24\text{h}) \quad [\text{knots, continuous regression}]$$
- **Target Timing & Window:** Evaluated at synoptic interval $T + 24\text{h}$ with tolerance $\pm 3.0\text{h}$ ($[21.0\text{h}, 27.0\text{h}]$).
- **Prohibition on Categorical Target:** Categorical IMD cyclone grade (D, DD, CS, SCS, VSCS, ESCS, SuCS) must **not** serve as the primary loss objective. Categories may be derived from predicted continuous $V_{\max}$.
- **Target Distribution:** 1,907 valid 24h pairs across the dataset, with $V_{\max}(T+24\text{h})$ spanning 20 kt to 130 kt.

---

## 8. Central Pressure Multi-Target Specification (Section 8)

- Central pressure $P_c(T + 24\text{h})$ [hPa] is retained as a **secondary research target**.
- **Dynamical Consistency Clarification:**
  > "Joint $V_{\max}/P_c$ prediction may provide a physically informative multi-target formulation, but does not itself enforce dynamical consistency."
- **Surge Modeling Clarification:**
  Central pressure alone does not govern storm surge. Storm surge is controlled by complex interactions among the wind field, storm size, translation speed, bathymetry, coastline geometry, and astronomical tides. No surge model is constructed in this phase.

---

## 9. Rapid Intensification (RI) Independent Gate (Sections 9 & 10)

- **Primary RI Threshold:** $\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$ (IMD NIO 93rd percentile; NOAA/NHC operational).
- **Secondary Sensitivity Benchmarks:** $25\text{ kt} / 24\text{h}$ and $20\text{ kt} / 24\text{h}$ (exploratory only).
- **Effective Sample Size:** 103 positive pairs occur across only 18 unique storms in 10 years.
- **The "Zero-RI" Holdout Blocker:**
  In the 2024–2026 independent holdout partition, **zero positive standard RI events exist ($N_{\text{pos}} = 0$)**.
  - Positive-class metrics (PR-AUC, Recall, CSI) are mathematically undefined.
  - Fabricating metrics is strictly prohibited.
- **RI Readiness Verdict:** **`DATA_NOT_READY`**.

---

## 10. Temporal Causality & Anti-Leakage Firewalls (Sections 11 & 12)

- **Causal Cutoff Rule:** For any forecast at time $T$, all predictive inputs must satisfy $\text{timestamp} \le T$.
- **Atmospheric 6-Hourly Matching (Section 13):**
  - 00Z prediction: Same-day 18Z is **FORBIDDEN**.
  - 06Z prediction: Same-day 18Z is **FORBIDDEN**.
  - 12Z prediction: Same-day 18Z is **FORBIDDEN**.
  - 18Z prediction: Same-day 18Z is **ALLOWED**.
- **Storm Identity Audit:** `system_id`, storm name, lifetime peak intensity, lifetime minimum pressure, and final duration are strictly excluded from the feature contract.
- **Spatial Geometry (Section 14):**
  Storm-centered extraction uses 0–100 km inner core and 200–800 km environmental annulus centered strictly on coordinates available at or before $T$.

---

## 11. Feature Contract Summary (Sections 15, 16, 20)

Formally documented in `backend/reports/CYCLONE_INTENSITY_FEATURE_CONTRACT.md`:
- **8 Directly Reusable Features:** Best Track causal kinematics ($V_{\max}$, $\Delta_{6\text{h}}$, $\Delta_{12\text{h}}$, $\Delta_{24\text{h}}$, $P_c$, translation speed, latitude, longitude).
- **8 Proxy-Only Features:** Daily basin-averaged ERA5 and fixed-site ocean variables (retained solely for legacy comparison).
- **8 Requires New Extraction Features:** 6-hourly storm-centered annular SST, MLD, TCHP, SLA, VWS (850–200 hPa), vorticity (850 hPa), RH (700 hPa), RH (500 hPa).
- **Unsuitable & Prohibited Features:** Fixed-point salinity trends, buoy current shear, and all future-derived lifetime properties.

---

## 12. Split Manifest Summary & Forensic Reconciliation (Sections 18 & 21)

Formally documented in `backend/reports/CYCLONE_INTENSITY_SPLIT_MANIFEST.md`:
- **TRAIN (2016–2021):** 64 storms, 1,493 raw fixes, 1,488 candidate fixes.
- **VALIDATION (2022–2023):** 24 storms, 510 raw fixes, 506 candidate fixes.
- **TEST (2024–2026):** 29 storms, 541 raw fixes, 541 candidate fixes.
- **Zero Overlap:** Confirmed empty intersection across all partition sets ($\mathcal{S}_A \cap \mathcal{S}_B = \emptyset$).
- **Test Quarantine:** 2024–2026 partition permanently locked from model tuning.

### Forensic Reconciliation Note: 74/22/21 Proposal vs Authoritative 64/24/29 Split
1. **Date Equivalence:** Every storm system in the 10-year dataset satisfies $\text{year}(\text{first\_fix}) == \text{year}(\text{last\_fix}) == \text{year}(\text{system\_id})$. No storm crosses calendar year boundaries. The first-fix date, storm start date, and system ID year rules are 100% equivalent.
2. **Deterministic Uniqueness:** Every `system_id` belongs to exactly one partition; zero storms or fixes are split across partitions.
3. **Discrepancy Cause:** The earlier 74/22/21 numbers originated from an integer array index partition (index 0..73, index 74..95, index 96..116) that erroneously cut through mid-2022 and mid-2024. Slicing 74 storms absorbed the first 10 storms of 2022 into Train (cutting at 2022-09-11), while slicing 22 storms absorbed the first 8 storms of 2024 into Validation (cutting at 2024-09-13), leaving only 21 late-2024/2025/2026 storms in the test holdout.
4. **Differing Storms Identified:**
   - 10 storms from 2022 (*UNNAMED-1* to *UNNAMED-10*, including *ASANI*) properly belong to Validation (2022–2023), not Train.
   - 8 storms from 2024 (*REMAL*, *UNNAMED-2*, *UNNAMED-3*, *ASNA*, *UNNAMED-5*, *UNNAMED-6*, *UNNAMED-7*, *UNNAMED-8*) properly belong to Test Holdout (2024–2026), not Validation.
5. **Scientific Standard Retained:** The 64/24/29 partition enforces true, intact calendar-season boundaries (2016–2021 Train, 2022–2023 Validation, 2024–2026 Quarantined Test) without intra-season contamination. The deterministic physical rule prevails.

---

## 13. Future Model Evaluation Protocol (Section 19)

- **Primary Benchmark:** Mean Absolute Error on $V_{\max}$ ($\text{MAE} < 11.5\text{ kt}$ at 24h).
- **Secondary Metrics:** RMSE, mean systematic bias, median absolute error, error distribution, per-storm error, grade-stratified error, and basin-stratified error.
- **RI Evaluation:** PR-AUC, CSI, Precision, Recall, F1, FAR, Brier Skill Score where mathematically defined; marked `UNDEFINED / INSUFFICIENT_EVIDENCE` on zero-positive test sets.

---

## 14. Final Data-Sufficiency Verdicts (Section 23)

### A. Cyclone Intensity: **`TRAINING_READY`**
**Scientific Rationale:**
1. **Target Clarity:** Continuous $V_{\max}(T+24\text{h})$ is physically rigorous, well-sampled across all years (1,907 candidate 24h pairs), and present in all partitions.
2. **Deterministic Quality Controls:** NADA anomaly and conflicting duplicates are resolved deterministically through candidate exclusions without mutating raw data.
3. **Causal Integrity:** Strict 6-hourly synoptic bounds, annular spatial extraction, and anti-leakage rules prevent look-ahead bias.
4. **Split Rigor:** Event-grouped chronological partitions guarantee zero storm overlap and quarantine the 2024–2026 holdout.
5. **Clear Scope:** `TRAINING_READY` authorizes a separate, controlled future training experiment; it does **not** imply operational deployment or production readiness.

### B. Rapid Intensification (RI): **`DATA_NOT_READY`**
**Scientific Rationale:**
1. **Zero Test Positives:** The 2024–2026 test partition contains zero standard RI events ($N_{\text{pos}} = 0$). Positive evaluation metrics (PR-AUC, CSI, Recall) are mathematically undefined.
2. **Small Effective Positive Count:** Only 18 unique storms over 10 years exhibited standard RI. Fix-level observations are strongly autocorrelated.
3. **Threshold Integrity:** Lowering the threshold to 20 or 25 kt to artificially manufacture positives is scientifically rejected.

---

## 15. Final Forensic Output (Section 28)

==================================================
FINAL FORENSIC GATE
==================================================

CYCLONE INTENSITY:
TRAINING_READY

RAPID INTENSIFICATION:
DATA_NOT_READY

ML TRAINING PERFORMED:
NO

PRODUCTION MODIFIED:
NO

V2.3 MODIFIED:
NO

FROZEN POLICY MODIFIED:
NO

RAW IMD DATA MODIFIED:
NO

PROTECTED HASHES:
PASS

AUTOMATED TESTS:
26/26 PASSED

CRITICAL BLOCKERS:
- None for Cyclone Intensity continuous regression research training.
- Rapid Intensification blocked by zero positive events in 2024–2026 holdout test partition and low effective sample size (18 storms).

REMAINING SCIENTIFIC LIMITATIONS:
- Cyclone intensity effective degrees of freedom are bounded by 117 unique storms due to high temporal autocorrelation.
- Storm-centered annular atmospheric features require new 6-hourly NetCDF extraction pipeline prior to training.
- 2024–2026 test holdout has limited storm-level statistical power for extreme category cyclones.

NEXT ALLOWED ACTION:
Implement the 6-hourly storm-centered feature extraction pipeline for candidate Cyclone Intensity features (without ML model training).

==================================================
