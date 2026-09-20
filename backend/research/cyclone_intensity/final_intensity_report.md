# Sagar-Drishti — Cyclone Intensity ML Training & Generalization Final Report
## Document ID: SD-REPORT-2026-INTENSITY-FINAL-V1
**Execution Phase:** Machine Learning Training & Generalization Evaluation Gate V1  
**Audit Phase:** Post-Training Documentation Reconciliation Gate  
**Status:** **RESEARCH COMPLETED — PERMANENTLY FROZEN**  
**Final Verdict:** **`PASS — RESEARCH SUCCESS, PROMISING BUT LIMITED EVIDENCE`**  
**Governance Authority:** Sagar-Drishti Scientific Review Board  

---

## 1. Executive Summary

This report documents the forensic reconciliation and final scientific evaluation of the research-only machine learning model for **Cyclone Intensity 24-Hour Maximum Sustained Wind Speed Regression ($V_{\max}(T+24\text{h})$)** in the North Indian Ocean (Bay of Bengal and Arabian Sea).

The experimental pipeline was executed across Steps 1–17 under strict quarantine controls. Following validation-based selection among candidate architectures and feature ablations, the winning model—**`EXP-E-CORE-ENVIRONMENT-RADIAL-CONTRASTS`** (XGBoost with 29 features)—was frozen. After multi-factor cryptographic keycard verification, the quarantined 2024–2026 test partition was unlocked for a single evaluation pass.

All documentation, metric definitions, test-suite counts, and scientific limitations have been reconciled in this authoritative report. **Zero model retraining, zero hyperparameter retuning, zero post-test modification, and zero operational changes have occurred.**

---

## 2. Authoritative Training / Validation / Test Dataset Counts

- **Canonical Dataset Artifact:** [`research/cyclone_intensity/features_6hourly_candidate.parquet`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/features_6hourly_candidate.parquet)
- **Dataset SHA-256:** `8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0`
- **Total Historical Systems:** 117 tropical cyclone systems (2016–2026)
- **Total Synoptic Fixes:** 2,535 candidate 6-hourly fixes ($00\text{Z}, 06\text{Z}, 12\text{Z}, 18\text{Z}$)
- **Valid 24-Hour Target Pairs:** 1,899 continuous target pairs ($V_{\max}(T+24\text{h})$ and $P_c(T+24\text{h})$)

### Partition Distribution Breakdown
| Partition | Calendar Years | Total Storms | Storms with Valid Targets | Candidate Fixes | Valid 24h Targets | Target Availability Rate |
|---|---|---|---|---|---|---|
| **TRAIN** | 2016–2021 | 64 | 58 | 1,488 | 1,121 | 75.34% |
| **VALIDATION** | 2022–2023 | 24 | 22 | 506 | 382 | 75.49% |
| **TEST (Holdout)** | 2024–2026 | 29 | 24 | 541 | 396 | 73.20% |
| **TOTAL** | **2016–2026** | **117** | **104** | **2,535** | **1,899** | **74.91%** |

*Partition Isolation Note:* Storm systems are assigned chronologically and strictly by system ID. Zero storm crossover exists between partitions:
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset.$$

---

## 3. Final Model Configuration

- **Selected Model ID:** `EXP-E-CORE-ENVIRONMENT-RADIAL-CONTRASTS`
- **Algorithm Family:** `XGBoost` (`xgb.XGBRegressor`)
- **Objective Function:** `reg:absoluteerror` (MAE loss optimization)
- **Tree Construction:** `tree_method="hist"`, `device="cpu"` (deterministic OpenMP execution, 12 threads)
- **Frozen Hyperparameters:**
  ```json
  {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42
  }
  ```
- **Feature Set Breakdown (29 Total Features):**
  - **Base Features (24):**
    - *Kinematics (9):* `vmax_current`, `dvmax_6h`, `dvmax_12h`, `dvmax_24h`, `pc_current`, `dpc_6h`, `translation_speed_kts`, `latitude_current`, `longitude_current`
    - *Atmospheric (11):* `vws_env_mean_200_800km`, `vws_env_min_200_800km`, `vws_core_mean_0_100km`, `vort_core_mean_0_100km`, `vort_core_max_0_100km`, `vort_env_mean_200_800km`, `rh_700_env_mean_200_800km`, `rh_700_env_min_200_800km`, `rh_700_core_mean_0_100km`, `rh_500_env_mean_200_800km`, `rh_500_core_mean_0_100km`
    - *Oceanic (4):* `sst_core_mean_0_100km`, `sst_env_mean_200_800km`, `mld_core_mean_0_100km`, `sla_core_mean_0_100km`
  - **Radial Contrast Differentials (5):**
    - `delta_vws_core_minus_env` ($=\text{VWS}_{\text{core}} - \text{VWS}_{\text{env}}$)
    - `delta_vort_core_minus_env` ($=\text{Vort}_{\text{core}} - \text{Vort}_{\text{env}}$)
    - `delta_rh700_core_minus_env` ($=\text{RH700}_{\text{core}} - \text{RH700}_{\text{env}}$)
    - `delta_rh500_core_minus_env` ($=\text{RH500}_{\text{core}} - \text{RH500}_{\text{env}}$)
    - `delta_sst_core_minus_env` ($=\text{SST}_{\text{core}} - \text{SST}_{\text{env}}$)
- **Learned Transformations:** Train-only median imputation fitted strictly on the 64 training storms; validation and test sets were transformed using frozen training statistics.
- **Model Checkpoint Artifact:** [`research/cyclone_intensity/models/final_intensity_model.joblib`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/models/final_intensity_model.joblib) (SHA-256: `3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f`).

---

## 4. Validation Results & Ablation Progression

All models were fitted exclusively on the 64 Train storms (1,121 valid targets) and evaluated on the 24 Validation storms (382 valid targets).

### Deterministic Reference Baselines (Validation)
- **Persistence Baseline:** Row MAE = `12.12 kt`, Storm MAE = `7.92 kt`, RMSE = `18.20 kt`, Bias = `-0.34 kt`
- **Damped Trend Baseline ($\alpha=0.5$, clipping $[15.0, 165.0]\text{ kt}$):** Row MAE = `11.32 kt`, Storm MAE = `7.67 kt`, RMSE = `17.53 kt`, Bias = `+0.80 kt`

### Ablation Suite Results (Validation Partition)
| Ablation ID | Model Tier | Features | Storm MAE (Mean) | Storm MAE (Median) | Row MAE | Row RMSE | Row Bias | Skill vs Pers (%) | Skill vs Trend (%) |
|---|---|---|---|---|---|---|---|---|---|
| **EXP-A** | Kinematics Only | 9 | **7.684 kt** | 7.021 kt | 9.463 kt | 12.507 kt | -0.621 kt | +2.98% | -0.18% |
| **EXP-B** | Kinematics + Atmosphere | 20 | **7.389 kt** | 6.845 kt | 9.314 kt | 12.387 kt | -0.512 kt | +6.70% | +3.66% |
| **EXP-C** | Kinematics + Ocean | 13 | **7.786 kt** | 7.150 kt | 8.995 kt | 12.305 kt | -0.385 kt | +1.69% | -1.51% |
| **EXP-D** | Flat Spatial (Core + Env) | 24 | **7.559 kt** | 7.187 kt | 9.016 kt | 12.191 kt | -0.450 kt | +4.56% | +1.45% |
| **EXP-E** | **Radial Contrasts (Winning)** | **29** | **7.352 kt** | **7.220 kt** | **8.880 kt** | **12.096 kt** | **-0.417 kt** | **+7.17%** | **+4.15%** |

---

## 5. Independent Test Results (2024–2026 Holdout)

The final frozen model was evaluated on the quarantined holdout of 29 storms (396 valid target pairs):

| Metric | Model EXP-E | Persistence Baseline | Damped Trend Baseline ($\alpha=0.5$) | Point Improvement vs Baseline |
|---|---|---|---|---|
| **Storm-Aggregated MAE** | **5.629 kt** | 6.199 kt | 6.259 kt | **+9.20%** vs Pers / **+10.06%** vs Trend |
| **Median Storm MAE** | **4.469 kt** | 4.820 kt | 4.950 kt | — |
| **Storm MAE Std Dev** | **3.312 kt** | 3.510 kt | 3.480 kt | — |
| **Storm MAE Range** | **[0.235, 13.366] kt** | [0.500, 14.200] kt | [0.450, 14.100] kt | — |
| **Row-Level MAE** | **6.147 kt** | 7.045 kt | 6.932 kt | **+12.75%** vs Pers / **+11.32%** vs Trend |
| **Root Mean Squared Error (RMSE)** | **7.879 kt** | 10.158 kt | 10.042 kt | **+22.44%** vs Pers / **+21.54%** vs Trend |
| **Row Bias** | **+3.022 kt** | +0.831 kt | +1.450 kt | Mean overprediction across test fixes |
| **Median Absolute Error** | **4.987 kt** | 5.210 kt | 5.140 kt | — |

---

## 6. Clustered Storm-Level Bootstrap Results ($B=1,000$)

To account for intra-storm temporal autocorrelation, clustered bootstrap resampling was executed by sampling entire storm systems with replacement ($B=1,000$ iterations, seed 42):

### Validation Clustered Bootstrap (24 Storms)
- **Model Storm MAE 95% CI:** `[5.567, 9.213] kt` (Mean: `7.352 kt`)
- **Skill over Persistence 95% CI:** `[-40.50%, +25.67%]`
- **Skill over Damped Trend 95% CI:** `[-41.01%, +21.50%]`

### Test Clustered Bootstrap (29 Storms)
- **Model Storm MAE 95% CI:** `[4.403, 6.980] kt` (Mean: `5.629 kt`)
- **Skill over Persistence 95% CI:** `[-20.26%, +28.75%]`
- **Skill over Damped Trend 95% CI:** `[-16.64%, +28.42%]`

*Statistical Implication:* Because the 95% confidence intervals for skill cross zero on both validation and holdout partitions, the point-estimate improvement (+9.20% and +10.06%), while promising, does not attain asymptotic statistical significance.

---

## 7. Test Quarantine Evidence (Read-Only Audit)

A strict read-only audit confirms that the test partition was structurally isolated and opened only after every prerequisite was satisfied:

1. **Test Access Count:** `1` (single evaluation pass completed; subsequent requests are blocked).
2. **Access Log Inspection:** [`research/cyclone_intensity/test_access_audit.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/test_access_audit.json):
   ```json
   {
     "unlock_timestamp_utc": "2026-09-13T07:56:42.824541+00:00",
     "keycard_hash": "1d29b1fa6dab59bd9b2f95192c6de3a0460a9527db846f5f2e1d8282d5cc168f",
     "model_hash": "3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f",
     "dataset_hash": "8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0",
     "test_partition": "2024-2026",
     "evaluator_version": "SD-INTENSITY-EVALUATOR-V1.0",
     "access_count": 1,
     "read_only_enforced": true
   }
   ```
3. **Keycard Verification:** All 8 cryptographic conditions in [`test_keycard.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/test_keycard.json) were verified prior to unlocking:
   - `MODEL_CONFIGURATION_FROZEN == true`
   - `VALIDATION_SELECTION_COMPLETE == true`
   - `DATASET_HASH == "8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0"`
   - `FEATURE_CONTRACT_HASH == "258690562dc9a79589ddfabd6956feb4b36726f2a9bcb093177ea7e0dd26f896"`
   - `MODEL_SELECTION_HASH == "b53b196e6c29e837bf2fb0ebc8780a3bfe927b535ff580a7667b74fa37db7bad"`
   - `FINAL_MODEL_HASH == "3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f"`
   - `PREPROCESSING_HASH == "591143af5765a2680612b13572b3ec0a058d0e478547bbc0482f027f8f4cbe62"`
   - `TEST_ACCESS_COUNT == 0` (prior to opening)
4. **Post-Test Invariance:** Zero retraining, zero hyperparameter adjustment, and zero code changes occurred post-test. The final model is permanently sealed.

---

## 8. Model Selection Evidence (Validation Isolation)

The model selection record [`research/cyclone_intensity/model_selection.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/research/cyclone_intensity/model_selection.json) confirms:
- **Selection Timestamp:** `2026-09-13T07:50:10.138080+00:00`
- **Selection SHA-256:** `b53b196e6c29e837bf2fb0ebc8780a3bfe927b535ff580a7667b74fa37db7bad`
- **Selection Rationale:** Selected exclusively because Model EXP-E achieved the lowest validation storm-level MAE (`7.352 kt`) with positive persistence skill (`+7.17%`) and trend skill (`+4.15%`) on 24 validation storms.
- **Chronological Isolation:** The selection artifact was sealed and hashed **6 minutes BEFORE** the test set was unlocked (`07:56:42 UTC`). Test metrics, test MAE, test RMSE, and test bootstrap were physically inaccessible and played zero role in model selection.

---

## 9. Authoritative Test-Count Reconciliation

The repository test execution totals correspond to distinct chronological development stages:

| Test Suite | Stage | Result | Meaning & Detailed Composition |
|---|---|:---:|---|
| **6-Hourly Extraction Suite** | Post-Implementation Audit | **13/13** | Dedicated tests in [`test_cyclone_intensity_6hourly_extraction.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_6hourly_extraction.py) verifying 6h synoptic grid, temporal tracking, causality, and exclusion manifests. |
| **Cyclone Intensity Final Gate** | Data Reconciliation Gate | **26/26** | Gate tests in [`test_cyclone_intensity_final_gate.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_final_gate.py) verifying NADA row resolution, target accounting (-8), and ocean decomposition. |
| **Cyclone Intensity Data Gate** | Data Ingestion Gate | **22/22** | Pre-extraction sufficiency tests in [`test_cyclone_intensity_data_gate.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_data_gate.py). |
| **GPU Acceleration Suite** | Hardware Optimization Stage | **5/5** | Dedicated DirectML/GPU verification tests in [`test_gpu_acceleration.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_gpu_acceleration.py). |
| **Cyclone Intensity Training Design** | Training Design Gate | **10/10** | Code-level quarantine, 8-condition keycard check, single-access enforcement, and baseline freeze in [`test_cyclone_intensity_training_design.py`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/tests/test_cyclone_intensity_training_design.py) (originally 9 tests at design; 1 test added for single-access tamper verification). |
| **Full Backend Suite (Pre-Training)** | Historical (Post-Reconciliation) | **340/340** | Total backend test count prior to GPU acceleration implementation. Historical baseline count. |
| **Full Backend Suite (Post-GPU)** | Historical (Post-GPU Enablement) | **345/345** | Total backend test count after integrating 5 GPU acceleration tests ($340 + 5 = 345$). |
| **Full Backend Suite (Post-Design)** | Historical (ML Training Design) | **354/354** | Total backend test count after adding 9 initial training design tests ($345 + 9 = 354$). |
| **Full Backend Suite (Current Final)** | **Authoritative Post-Execution** | **355/355** | Current full backend suite including the keycard cryptographic single-access test ($354 + 1 = 355$). All pass with zero errors. |

---

## 10. Bias Sign Convention & Physical Interpretation

### Mathematical Formula
Traced directly from implementation in [`train_cyclone_intensity_v1.py:77`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/scripts/train_cyclone_intensity_v1.py#L77) and [`intensity_data_manager.py:260`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/intensity_data_manager.py#L260):

$$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (y_{\text{pred}, i} - y_{\text{true}, i}) = \text{mean}(y_{\text{pred}} - y_{\text{true}}) = \text{Prediction} - \text{Observation}$$

### Physical Interpretation of Values
- **Validation Bias ($-0.417\text{ kt}$):**
  A negative value indicates that the model **underpredicted** cyclone intensity by an average of $0.417\text{ kt}$ across the 24 validation storms (382 fixes).
- **Test Bias ($+3.022\text{ kt}$):**
  A positive value indicates that the model **overpredicted** cyclone intensity by an average of $3.022\text{ kt}$ across the 29 test holdout storms (396 fixes).
- **Physical Reason for Test Overprediction:** The 2024–2026 test partition was composed predominantly of weaker tropical systems ($V_{\max} \le 60\text{ kt}$, with 301 of 396 fixes having $V_{\max} \le 35\text{ kt}$). Because the model was trained on the broader 2016–2021 distribution (which included intense tropical cyclones up to $140\text{ kt}$), its predictions on weak disturbances exhibited a mild positive baseline shift.

---

## 11. Provisional 11.5 kt Benchmark Wording

The final model's observed holdout MAE is below the currently provisional 11.5 kt reference threshold; the threshold itself remains unvalidated.

*Policy Constraint:* Sagar-Drishti documentation does not claim that the model has "passed the 11.5 kt benchmark" in a validated operational sense, because the 11.5 kt figure represents a preliminary heuristic reference rather than an independently peer-reviewed North Indian Ocean operational standard.

---

## 12. Research-Only Restrictions & Operational Boundaries

Cyclone Intensity V1 is strictly a **RESEARCH-ONLY** model.

### Explicit Prohibitions
1. **NOT Production Integrated:** The model artifact resides strictly in `research/cyclone_intensity/models/` and is physically absent from `backend/models/`.
2. **NOT Operationally Authorized:** Zero authority is granted to use this model for operational cyclone forecasting, marine bulletins, or public disaster alerts.
3. **NOT a Replacement for Production Systems:** Production v1.1.0 (`risk_model_3d.joblib`), V2.0.0 10-year operational baseline (`v2_10yr`), and V2.3 Shadow systems remain authoritative.
4. **NOT Validated for Live Warnings:** Real-time ingestion pipelines MUST NOT feed live IMD or JTWC bulletins into this model.
5. **Rapid Intensification (RI) Status:** RI ($\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$) remains permanently classified as **`DATA_NOT_READY`**. Zero positive RI events exist in the 2024–2026 holdout. No RI classifier is authorized or trained.

---

## 13. Known Scientific Limitations

The experimental results must be interpreted in light of the following physical and statistical constraints:

1. **Limited Sample Size:** Only 117 cyclone systems over 10 years are available across the entire North Indian Ocean basin, representing a modest sample of independent meteorological life cycles.
2. **Effective Sample Size & Temporal Autocorrelation:** Synoptic fixes within the same storm exhibit strong serial correlation. The effective sample size is closer to the number of storms (64 Train, 24 Val, 29 Test) than the number of 6-hourly fixes.
3. **Limited Holdout Dynamic Range:** The 2024–2026 test partition contained only 29 storms, with a maximum observed intensity of $60\text{ kt}$ (Severe Cyclonic Storm). Zero Very Severe, Extremely Severe, or Super Cyclonic Storms occurred during this holdout window.
4. **Bootstrap Confidence Intervals Cross Zero Skill:** Clustered bootstrap 95% confidence intervals for skill over persistence (`[-20.26%, +28.75%]`) and damped trend (`[-16.64%, +28.42%]`) encompass zero, indicating that observed superiority on this holdout cannot guarantee superior performance on future seasons.
5. **Zero RI-Positive Events:** Because no rapid intensification events occurred in the 2024–2026 holdout, the model's behavior during sudden explosive deepening remains unverified.
6. **Generalization Extrapolation:** Holdout performance must not be interpreted as guaranteed future operational performance.

---

## 14. Official Final Verdict

# **`PASS — RESEARCH SUCCESS, PROMISING BUT LIMITED EVIDENCE`**

The Cyclone Intensity V1 experiment successfully met all experimental design criteria, established strict train-only and validation-only separation, preserved permanent quarantine of the holdout data, demonstrated positive point-estimate skill over persistence (+9.20%) and damped trend (+10.06%), and preserved 100% byte invariance of all production assets.

Due to the limited dynamic range of the 2024–2026 holdout and wide bootstrap confidence intervals, the evidence is officially classified as promising research evidence. Operational deployment is strictly prohibited pending future multi-season validation.

```
SUMMARY OF ENFORCEMENT:
- NO RETRAINING
- NO RETUNING
- NO NEW TEST ACCESS
- NO PRODUCTION CHANGES
```
