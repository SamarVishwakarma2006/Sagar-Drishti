# Sagar-Drishti — Cyclone Intensity V1
# Post-Training Documentation Reconciliation Report
**Document ID:** `SD-REPORT-2026-INTENSITY-RECONCILIATION-GATE-V1`  
**Classification:** **RESEARCH-ONLY — PERMANENTLY FROZEN**  
**Final Verdict:** **`PASS — RESEARCH SUCCESS, PROMISING BUT LIMITED EVIDENCE`**  
**Audit Scope:** Strict Post-Implementation Documentation & Forensic Inconsistency Reconciliation  
**Enforcement Directives:** NO RETRAINING / NO RETUNING / NO NEW TEST ACCESS / NO PRODUCTION CHANGES  

---

## 1. Executive Summary

This report delivers the authoritative documentation reconciliation for the already completed **Cyclone Intensity Machine Learning Training & Evaluation Gate V1**. 

All 16 non-negotiable rules were strictly observed during this reconciliation audit:
- Zero models were retrained.
- Zero hyperparameters were retuned.
- Zero model selection runs were re-executed.
- Zero requests for test partition data were made.
- Test access counter remains frozen at `1`.
- All 12 protected production and baseline artifacts were verified 100% byte-invariant.
- The previously completed scientific results remain frozen.

---

## 2. Authoritative Dataset Identity & Partition Counts

The canonical 6-hourly storm-centered dataset is locked and verified:
- **Canonical Parquet:** [`research/cyclone_intensity/features_6hourly_candidate.parquet`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/features_6hourly_candidate.parquet)
- **SHA-256:** `8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0`
- **Total Historical Storms:** 117
- **Total Synoptic Fixes:** 2,535
- **Continuous 24h Target Pairs:** 1,899 ($V_{\max}(T+24\text{h})$ and $P_c(T+24\text{h})$)

### Partition Distribution
| Partition | Period | Total Storms | Storms with Valid Targets | Synoptic Fixes | Valid 24h Targets |
|---|---|:---:|:---:|:---:|:---:|
| **TRAIN** | 2016–2021 | 64 | 58 | 1,488 | 1,121 |
| **VALIDATION** | 2022–2023 | 24 | 22 | 506 | 382 |
| **TEST** | 2024–2026 | 29 | 24 | 541 | 396 |
| **TOTAL** | **2016–2026** | **117** | **104** | **2,535** | **1,899** |

---

## 3. Authoritative Scientific Results & Model Configuration

- **Selected Model:** `XGBoost` (`EXP-E-CORE-ENVIRONMENT-RADIAL-CONTRASTS`)
- **Feature Configuration:** Kinematic (9) + Atmospheric (11) + Oceanic (4) + Radial Contrasts (5) = **29 Total Features**
- **Base Features:** 24
- **Final Feature Count:** 29
- **Primary Target:** $V_{\max}(T+24\text{h})$
- **Model Storage:** Sealed in [`research/cyclone_intensity/models/final_intensity_model.joblib`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/models/final_intensity_model.joblib) (SHA-256: `3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f`)

### Validation Performance (Selection Partition — 24 Storms, 382 Targets)
- **Storm MAE:** `7.352 kt`
- **Row MAE:** `8.880 kt`
- **RMSE:** `12.096 kt`
- **Row Bias:** `-0.417 kt`
- **Skill over Persistence:** `7.17%`
- **Skill over Damped Trend ($\alpha=0.5$):** `4.15%`

### Independent Holdout Performance (2024–2026 — 29 Storms, 396 Targets)
- **Storm MAE:** `5.629 kt`
- **Row MAE:** `6.147 kt`
- **RMSE:** `7.879 kt`
- **Row Bias:** `+3.022 kt`
- **Point Improvement vs Persistence:** `9.20%` (Persistence Storm MAE: `6.199 kt`)
- **Point Improvement vs Damped Trend:** `10.06%` (Damped Trend Storm MAE: `6.259 kt`)

### Clustered Storm-Level Bootstrap ($B=1,000$)
- **Holdout Storm MAE 95% CI:** `[4.403, 6.980] kt`
- **Persistence Skill 95% CI:** `[-20.26%, +28.75%]`
- **Damped-Trend Skill 95% CI:** `[-16.64%, +28.42%]`

---

## 4. Task 1 — Test-Suite Count Reconciliation

The four different test counts appearing across project documentation represent historical milestones in chronological order:

| Test Suite | Stage | Result | Detailed Composition & Architectural Meaning |
|---|---|:---:|---|
| **6-Hourly Extraction Suite** | Post-Implementation Audit | **13/13** | Dedicated extraction tests in [`test_cyclone_intensity_6hourly_extraction.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_6hourly_extraction.py). Verifies grid alignment, temporal resolution, causality, and manifest accounting. |
| **Cyclone Intensity Final Gate** | Data Reconciliation Gate | **26/26** | Forensic gate tests in [`test_cyclone_intensity_final_gate.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_final_gate.py). Verifies NADA row resolution, target accounting (-8), and ocean decomposition. |
| **Cyclone Intensity Data Gate** | Data Ingestion Gate | **22/22** | Preliminary sufficiency tests in [`test_cyclone_intensity_data_gate.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_data_gate.py). |
| **GPU Acceleration Suite** | Hardware Optimization Stage | **5/5** | Dedicated tests in [`test_gpu_acceleration.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_gpu_acceleration.py). Verifies DirectML inference on AMD GPU. |
| **Training Design Suite** | Training Design & Quarantine Gate | **10/10** | Quarantine, keycard verification, and baseline freeze in [`test_cyclone_intensity_training_design.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_training_design.py) (originally 9 tests at design; 1 test added for single-access tamper verification). |
| **Full Backend Suite (Stage 1)** | Pre-Training Historical Baseline | **340/340** | Total backend test count during the Post-Implementation Reconciliation Audit prior to GPU enablement. |
| **Full Backend Suite (Stage 2)** | Post-GPU Historical Milestone | **345/345** | Total backend test count after adding the 5 GPU acceleration tests ($340 + 5 = 345$). |
| **Full Backend Suite (Stage 3)** | Post-Design Historical Milestone | **354/354** | Total backend test count after adding the 9 training design tests ($345 + 9 = 354$). |
| **Full Backend Suite (Stage 4 - Current)** | **Authoritative Post-Execution Final** | **355/355** | Current full backend test suite including the single-access keycard tamper test ($354 + 1 = 355$). All pass with zero failures. |

---

## 5. Task 2 — Bias Sign Convention Reconciliation

### Code-Level Formula
Traced directly from codebase implementation in [`train_cyclone_intensity_v1.py:77`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/scripts/train_cyclone_intensity_v1.py#L77) and [`intensity_data_manager.py:260`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/intensity_data_manager.py#L260):

$$\text{Bias} = \text{mean}(y_{\text{pred}} - y_{\text{true}}) = \text{Prediction} - \text{Observation}$$

### Interpretation of Results
- **Validation Bias = $-0.417\text{ kt}$:** The model slightly **underpredicted** cyclone intensity by an average of $0.417\text{ kt}$ across the 24 validation storms.
- **Test Bias = $+3.022\text{ kt}$:** The model **overpredicted** cyclone intensity by an average of $3.022\text{ kt}$ across the 29 test holdout storms.
- **Meteorological Context:** The 2024–2026 holdout was composed predominantly of weak storms ($V_{\max} \le 60\text{ kt}$, with 76% having $V_{\max} \le 35\text{ kt}$). Because the model was trained on the broader 2016–2021 distribution containing major cyclones, predictions on weak systems exhibited a mild positive baseline shift.

---

## 6. Task 3 — Read-Only Test Quarantine Evidence

A read-only forensic inspection confirms:
1. **Access Count Frozen:** `test_access_count = 1`.
2. **Keycard Verification:** Test partition access was unlocked only after verifying all 8 cryptographic conditions in [`test_keycard.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/test_keycard.json):
   - `MODEL_CONFIGURATION_FROZEN == true`
   - `VALIDATION_SELECTION_COMPLETE == true`
   - `DATASET_HASH == "8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0"`
   - `FEATURE_CONTRACT_HASH == "258690562dc9a79589ddfabd6956feb4b36726f2a9bcb093177ea7e0dd26f896"`
   - `MODEL_SELECTION_HASH == "b53b196e6c29e837bf2fb0ebc8780a3bfe927b535ff580a7667b74fa37db7bad"`
   - `FINAL_MODEL_HASH == "3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f"`
   - `PREPROCESSING_HASH == "591143af5765a2680612b13572b3ec0a058d0e478547bbc0482f027f8f4cbe62"`
   - `TEST_ACCESS_COUNT == 0` (at pre-unlock check)
3. **Audit Log:** Logged to [`test_access_audit.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/test_access_audit.json) at `2026-09-13T07:56:42Z`.
4. **Single-Access Enforcement:** A second access attempt is blocked at code level, raising `TestSetQuarantineViolationError`.
5. **Post-Test Invariance:** Zero retraining, zero hyperparameter adjustment, and zero code changes occurred post-test.

---

## 7. Task 4 — Model Selection Evidence (Validation Isolation)

The model selection record [`research/cyclone_intensity/model_selection.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/model_selection.json) confirms:
- **Selection Decision:** Model `EXP-E-CORE-ENVIRONMENT-RADIAL-CONTRASTS` was selected using **VALIDATION METRICS ONLY** (lowest validation storm MAE of `7.352 kt`, positive persistence skill `+7.17%`, positive trend skill `+4.15%`).
- **Zero Test Contamination:** The selection record was hashed (`b53b196e...`) and locked into the keycard at `07:50:10 UTC`, **6 minutes before** the test set was unlocked (`07:56:42 UTC`).
- Test MAE, test RMSE, test bootstrap, and test error analysis played zero role in selecting the winning model.

---

## 8. Task 5 — Research-Only Status & RI Preservation

Cyclone Intensity V1 is strictly a **RESEARCH-ONLY** model:
- **NOT** production integrated.
- **NOT** operationally authorized.
- **NOT** a replacement for production systems.
- **NOT** validated for live warning issuance.
- **Rapid Intensification (RI):** Remains permanently classified as **`DATA_NOT_READY`** ($\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$, zero positive cases in the holdout). No RI classifier was trained or evaluated.

---

## 9. Task 6 — Provisional 11.5 kt Benchmark Wording

The final model's observed holdout MAE is below the currently provisional 11.5 kt reference threshold; the threshold itself remains unvalidated.

The wording "passed the 11.5 kt benchmark" is strictly retracted and avoided because the 11.5 kt benchmark lacks independent peer-reviewed provenance in the North Indian Ocean basin.

---

## 10. Known Scientific Limitations

1. Only 117 cyclone systems exist in the 2016–2026 record.
2. The storm-level effective sample size is limited (64 Train, 24 Val, 29 Test).
3. Strong temporal dependence exists across fixes within each storm system.
4. Only 29 independent test storms were available in the 2024–2026 holdout.
5. Clustered bootstrap confidence intervals for skill encompass zero on both validation and test sets.
6. The 2024–2026 holdout contains zero RI-positive events and maximum intensity was limited to $60\text{ kt}$.
7. The 11.5 kt reference threshold remains provisional.
8. Holdout performance must not be interpreted as guaranteed future performance.
9. No operational deployment is authorized.

---

## 11. Integrity Verification Summary

All 12 protected model, configuration, and historical data artifacts are verified 100% byte-invariant:

| Protected Artifact Path | SHA-256 Checksum | Status |
|---|---|:---:|
| `backend/models/risk_model_3d.joblib` | `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` | **100% BYTE-INVARIANT** |
| `backend/models/v2_10yr/risk_model_3d.joblib` | `7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3` | **100% BYTE-INVARIANT** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib` | `7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15` | **100% BYTE-INVARIANT** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib` | `b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe` | **100% BYTE-INVARIANT** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib` | `aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c` | **100% BYTE-INVARIANT** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib` | `250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775` | **100% BYTE-INVARIANT** |
| `backend/config/frozen_alert_policy_v2.json` | `6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6` | **100% BYTE-INVARIANT** |
| `backend/data/historical/features_10yr.parquet` | `cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867` | **100% BYTE-INVARIANT** |
| `backend/data/historical/labeled_features_10yr_clean.parquet` | `25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a` | **100% BYTE-INVARIANT** |
| `backend/data/era5/features_atmosphere_10yr_daily.parquet` | `551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864` | **100% BYTE-INVARIANT** |
| `backend/config/v2_3_frozen_experiment_manifest.json` | `73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc` | **100% BYTE-INVARIANT** |
| `backend/data/historical/imd_tracks_2016_2026.parquet` | `e3f1b87455e716767e3b34a05b3dff04587dc09c6689a7dc0eaf7b72f951b643` | **100% BYTE-INVARIANT** |

---

## 12. Final Verdict

# **`FINAL VERDICT: PASS — RESEARCH SUCCESS, PROMISING BUT LIMITED EVIDENCE`**

```
ENFORCEMENT AUDIT CONFIRMATION:
- NO RETRAINING
- NO RETUNING
- NO NEW TEST ACCESS
- NO PRODUCTION CHANGES
```
