# Sagar-Drishti — Cyclone Intensity ML Training & Experimental Design Specification
## Document ID: SD-SPEC-2026-INTENSITY-TRAINING-V1
**Phase:** Machine Learning Training & Experimental Design Gate V1  
**Status:** **DESIGN ONLY — ZERO MODEL TRAINING AUTHORIZED**  
**Governance Authority:** Sagar-Drishti Scientific Review Board  

---

## 1. Executive Summary & Protection Rules

This document establishes the authoritative scientific, statistical, and operational specification for the future machine learning training of Sagar-Drishti's **Cyclone Intensity Regression Model**.

### Absolute Non-Negotiable Protection Rules
Under this design gate:
1. **Zero Model Fitting:** No estimator fitting, no `model.fit()`, no `scaler.fit()`, no hyperparameter search, no checkpoint creation, and no model serialization are permitted.
2. **Zero Production Mutation:** Production v1.1.0 (`backend/models/risk_model_3d.joblib`), V2.3 Model D, `v2_10yr`, the frozen V2.0.0 operational alert policy, the existing 101-feature ocean contract, and the existing daily ERA5 dataset remain strictly immutable and byte-invariant.
3. **Research-Only Scope:** All workflows remain within research sandboxes. No predictions or outputs may interface with operational hazard alerts or SagarBot.

---

## 2. Dataset Forensic Inventory

The training pipeline will strictly consume the validated research dataset:
`research/cyclone_intensity/features_6hourly_candidate.parquet`

### Canonical Dataset Characteristics
- **Total Cyclone Systems:** `117`
- **Total Candidate Fixes:** `2,535` synoptic fixes ($00\text{Z}, 06\text{Z}, 12\text{Z}, 18\text{Z}$)
- **Valid 24-Hour Target Pairs:** `1,899`
- **Total Extracted Features:** `24` (9 Kinematic, 11 Atmospheric, 4 Oceanic)

### Chronological Event-Grouped Partitioning
The dataset is partitioned chronologically with zero storm crossover:

| Partition | Calendar Years | Storm Systems | Candidate Fixes | Valid $V_{\max}(T+24\text{h})$ Targets | Governance Status |
|---|---|---|---|---|---|
| **TRAIN** | 2016–2021 | 64 | 1,488 | 1,121 | Accessible for training & imputation parameter estimation |
| **VALIDATION** | 2022–2023 | 24 | 506 | 382 | Accessible for hyperparameter & ablation selection |
| **TEST** | 2024–2026 | 29 | 541 | 396 | **PERMANENTLY QUARANTINED** (Single-pass final evaluation only) |
| **TOTAL** | **2016–2026** | **117** | **2,535** | **1,899** | **100% RECONCILED** |

---

## 3. Primary Model Task: 24-Hour Intensity Continuous Regression

- **Target Variable:** $V_{\max}(T+24\text{h})$ (Maximum 3-minute sustained surface wind speed in knots at $T+24\text{ hours}$).
- **Nature of Target:** Strictly continuous regression problem.
  - *Prohibition:* The primary target must **NOT** be discretized into categorical intensity bins or hazard tiers.
- **Input Feature Vector:** Environmental and causal kinematic state at origin $T$ ($feature\_timestamp \le T$).
- **Primary Metric:** Mean Absolute Error (MAE) in knots.
- **Secondary Metrics:**
  - Root Mean Squared Error (RMSE) in knots
  - Median Absolute Error in knots
  - Mean Error (Bias: $\hat{y} - y$) in knots
  - Skill score relative to persistence baseline ($\text{Skill}_{\text{pers}} = (1 - \text{MAE}_{\text{model}} / \text{MAE}_{\text{pers}}) \times 100\%$)
  - $R^2$ only as supplementary context.

---

## 4. Secondary Target: Central Pressure $P_c(T+24\text{h})$

- **Secondary Variable:** $P_c(T+24\text{h})$ (Central pressure in hPa at $T+24\text{ hours}$).
- **Scientific Boundaries:**
  1. *Dynamical Independence:* Joint prediction of $V_{\max}$ and $P_c$ does **NOT** automatically guarantee physical dynamical consistency (e.g. empirical Holland wind-pressure relationships).
  2. *Secondary Status:* $P_c$ remains an independent auxiliary diagnostic and will never supersede $V_{\max}$ as the primary target.
  3. *No Storm Surge Claims:* $P_c$ alone must **NOT** be claimed or represented as a direct surrogate for storm surge prediction (surge requires bathymetry, astronomical tide, coastal geometry, and wind fetch).

---

## 5. Rapid Intensification (RI) Governance

- **Status:** **`DATA_NOT_READY`**
- **Definition:** Standard meteorological definition: $\Delta V_{\max} \ge 30\text{ kt} / 24\text{ hours}$.
- **Sensitivity Thresholds:** $25\text{ kt}$ and $20\text{ kt}$ are sensitivity definitions only.
- **Prohibitions:**
  - **DO NOT** train an RI classifier.
  - **DO NOT** circumvent data deficiency through SMOTE, synthetic oversampling, class weighting, or moving positive cases across chronological partitions.
  - *Reasoning:* The quarantined holdout (2024–2026) contains zero positive standard RI cases ($0 / 396$). An unverified classifier cannot be scientifically validated on the holdout.

---

## 6. Permanent Test Quarantine & Code-Level Isolation

The 2024–2026 test partition is under permanent forensic quarantine:
1. **Forbidden Operations:** Test data must never be inspected, tuned against, or used for feature selection or model selection.
2. **Code-Level Enforcement:** The [`CycloneIntensityDataManager`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/intensity_data_manager.py) raises a hard [`TestSetQuarantineViolationError`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/app/services/intensity_data_manager.py) if any training or validation harness attempts to access the test set.
3. **Single-Pass Protocol:** Evaluated exactly once after all training, hyperparameters, and baselines are permanently frozen.

---

## 7. Effective Sample Size & Temporal Autocorrelation

Treating 6-hourly cyclone fixes as independent statistical entities is scientifically invalid. 

Fixes within the same storm exhibit severe temporal autocorrelation along their track:
- **Train Partition (2016–2021):** Lag-1 Autocorrelation $r = 0.876$
- **Validation Partition (2022–2023):** Lag-1 Autocorrelation $r = 0.852$
- **Test Partition (2024–2026):** Lag-1 Autocorrelation $r = 0.833$

### Effective Sample Size Limits
- **Upper Bound of Independent Units:** $N = 117\text{ storms}$ (64 Train, 24 Validation, 29 Test).
- **Fixes per Storm:** Mean $21.7\text{ fixes}$, Median $17.0\text{ fixes}$ (range 3 to 97 fixes).
- **Statistical Implication:** Row-level statistics substantially underestimate sampling variance. Model selection must rely on **Storm-Aggregated Metrics**, and confidence intervals must use **Storm-Level Clustered Bootstrapping**.

---

## 8. Deterministic Scientific Baselines (No Learned Models)

Per Final Design-Gate Corrections, CLIPER-lite is **removed from V1**. The baseline suite consists strictly of deterministic, non-learned formulations calculated without accessing the quarantined test set:

### Baseline A — Persistence
- **Formula:** $\hat{V}_{\max}(T+24\text{h}) = V_{\max}(T)$
- **Physical Meaning:** Zero net change in intensity over 24 hours. Represents the primary operational hurdle.

### Baseline B — Damped Kinematic Trend
- **Formula:** $\hat{V}_{\max}(T+24\text{h}) = \text{clip}\left(V_{\max}(T) + \alpha \cdot \text{trend}_{12\text{h}}, 15.0, 165.0\right)$
- **Damping Parameter:** $\alpha = 0.5$, classified explicitly as **`PRE-DECLARED HEURISTIC`** (frozen prior to validation without tuning).
- **Trend Rule:** $\Delta V_{\max}(12\text{h})$ if available; if NaN, $2.0 \times \Delta V_{\max}(6\text{h})$; if both NaN, $0.0$.
- **Clipping Bounds Provenance:**
  - $V_{\min} = 15.0\text{ kt}$: IMD Best Track depression boundary floor.
  - $V_{\max} = 165.0\text{ kt}$: Empirical North Indian Ocean physical ceiling for Super Cyclonic Storms (e.g. 1999 Odisha, Amphan 2020, Mocha 2023).

### Empirical Baseline Reference Metrics (Train & Validation Only)

| Metric | Train Persistence | Train Damped Trend ($\alpha=0.5$) | Validation Persistence | Validation Damped Trend ($\alpha=0.5$) | Quarantined Test |
|---|---|---|---|---|---|
| **Row MAE** | 15.16 kt | 14.12 kt | 12.12 kt | 11.32 kt | **QUARANTINED** |
| **Row RMSE** | 20.76 kt | 19.69 kt | 18.20 kt | 17.53 kt | **QUARANTINED** |
| **Row Bias** | +0.62 kt | +1.69 kt | -0.34 kt | +0.80 kt | **QUARANTINED** |
| **Storm MAE** | 10.26 kt | 9.95 kt | 7.92 kt | 7.67 kt | **QUARANTINED** |

*Crucial Scientific Observation:* Damped Trend on Validation achieves **11.32 kt Row MAE** and **7.67 kt Storm MAE** without any machine learning. Any proposed ML model must outperform these baseline figures to demonstrate legitimate predictive skill.

---

## 9. Candidate ML Architectures & Hardware Execution

### Model Families
1. **LightGBM Regressor (`lightgbm.LGBMRegressor`):** Primary gradient boosted decision tree using histogram binning with native missing-value routing.
2. **Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`):** Conservative bagged ensemble baseline.
3. **XGBoost Regressor (`xgboost.XGBRegressor`):** Exact/histogram tree boosting.

### Hyperparameter Search Space (Restricted Pre-Declared Grid)
Unrestricted hyperparameter search is forbidden. Search is limited to:
- `n_estimators`: $\{100, 200, 300\}$
- `max_depth`: $\{3, 5, 7\}$
- `learning_rate`: $\{0.03, 0.05, 0.1\}$
- `subsample`: $\{0.7, 0.85, 1.0\}$
- `colsample_bytree`: $\{0.7, 0.85, 1.0\}$
- `min_child_samples`: $\{10, 20\}$

### Hardware Execution Policy
- **Deterministic CPU Reference:** All training, cross-validation, and baseline verification must execute deterministically on CPU (`device="cpu"`, `n_jobs=12`, `random_state=42`).
- **DirectML Acceleration:** DirectML (`DmlExecutionProvider` on AMD Radeon Graphics) is supported for tensor evaluation and inference but scientific reproducibility must not depend on accelerator-specific floating-point variations.

---

## 10. Feature Taxonomy & Group Definitions

### Group A — Causal Kinematics & Position (9 Features)
- `vmax_current`, `dvmax_6h`, `dvmax_12h`, `dvmax_24h`
- `pc_current`, `dpc_6h`, `translation_speed_kts`
- `latitude_current`, `longitude_current`

### Group B — Atmospheric Predictors (ERA5, 11 Features)
- `vws_env_mean_200_800km`, `vws_env_min_200_800km`, `vws_core_mean_0_100km`
- `vort_core_mean_0_100km`, `vort_core_max_0_100km`, `vort_env_mean_200_800km`
- `rh_700_env_mean_200_800km`, `rh_700_env_min_200_800km`, `rh_700_core_mean_0_100km`
- `rh_500_env_mean_200_800km`, `rh_500_core_mean_0_100km`

### Group C — Ocean Thermodynamic Predictors (Copernicus, 4 Features)
- `sst_core_mean_0_100km`, `sst_env_mean_200_800km`
- `mld_core_mean_0_100km`, `sla_core_mean_0_100km`

### Group D — Storm-Centered Spatial-Structure Features (15 Features)
Explicit machine-readable separation of spatial domains:
- **Core Domain ($0\text{–}100\text{ km}$, 8 features):** `vws_core_mean_0_100km`, `vort_core_mean_0_100km`, `vort_core_max_0_100km`, `rh_700_core_mean_0_100km`, `rh_500_core_mean_0_100km`, `sst_core_mean_0_100km`, `mld_core_mean_0_100km`, `sla_core_mean_0_100km`.
- **Environmental Annulus ($200\text{–}800\text{ km}$, 7 features):** `vws_env_mean_200_800km`, `vws_env_min_200_800km`, `vort_env_mean_200_800km`, `rh_700_env_mean_200_800km`, `rh_700_env_min_200_800km`, `rh_500_env_mean_200_800km`, `sst_env_mean_200_800km`.

---

## 11. Ablation Experiment Suite

The feature ablation plan evaluates whether environmental complexity provides measurable incremental value:

| Ablation ID | Model Name | Feature Ingestion | Total Features | Scientific Objective |
|---|---|---|---|---|
| **EXP-A** | Kinematics Only | Group A | 9 | Quantify predictive power of past intensity & position alone |
| **EXP-B** | Kinematics + Atmosphere | Group A + Group B | 20 | Measure incremental value of vertical shear, vorticity, & humidity |
| **EXP-C** | Kinematics + Ocean | Group A + Group C | 13 | Measure incremental value of ocean thermodynamic fuel (SST, MLD, SLA) |
| **EXP-D** | Full Flat Model | Group A + Group B + Group C | 24 | Unconstrained pooling of all atmospheric and oceanic features |
| **EXP-E** | Core-Environment Contrasts | 24 Base Features + 5 Radial Contrasts | 29 | Explicitly test whether core-minus-annulus radial differentials outperform flat feature pooling |

### Explicit Differentiation of Ablation E
Ablation E introduces 5 pre-declared physical core-minus-environment gradients:
1. $\Delta \text{VWS} = \text{vws\_core\_mean\_0\_100km} - \text{vws\_env\_mean\_200\_800km}$ (ventilation shear differential)
2. $\Delta \text{Vorticity} = \text{vort\_core\_mean\_0\_100km} - \text{vort\_env\_mean\_200\_800km}$ (vortex spin-up concentration gradient)
3. $\Delta \text{RH}_{700} = \text{rh\_700\_core\_mean\_0\_100km} - \text{rh\_700\_env\_mean\_200\_800km}$ (inner-core moist envelope vs dry air intrusion)
4. $\Delta \text{RH}_{500} = \text{rh\_500\_core\_mean\_0\_100km} - \text{rh\_500\_env\_mean\_200\_800km}$ (mid-tropospheric humidity contrast)
5. $\Delta \text{SST} = \text{sst\_core\_mean\_0\_100km} - \text{sst\_env\_mean\_200\_800km}$ (core local SST anomaly vs ambient basin temperature)

---

## 12. Missing Data, Imputation & Scaling Policy

1. **Native NaN Routing:** Tree models (LightGBM, XGBoost) natively route missing values during split finding based on optimal objective gain. Native NaN routing is preferred.
2. **Train-Only Imputation:** If an architecture requires non-null inputs (e.g. Random Forest), median imputation parameters must be fitted **strictly on the TRAIN partition** and applied without modification to Validation and Test.
3. **Feature Scaling:** Monotonic decision tree algorithms are strictly invariant to monotonic feature transformations. Scaling is omitted by design to prevent unnecessary transformations.

---

## 13. Temporal Causal Safety Verification

All 24 feature variables strictly satisfy the temporal causality boundary:
$$T_{\text{source}} \le T_{\text{origin}}$$
No target variable (`target_vmax_24h`, `target_pc_24h`, `target_delta_vmax_24h`), no future storm position, and no future atmospheric/oceanic observation enters the feature matrix.

---

## 14. Multi-Level Validation & Model Selection Hierarchy

Validation will report both **Row-Level** and **Storm-Level** metrics across the 24 validation storms.

### Decision Hierarchy for Model Selection
1. **Primary Criterion:** Lowest Validation **Storm-Aggregated MAE**.
2. **Secondary Criterion:** Lowest Validation **RMSE**.
3. **Skill Hurdle:** Statistically significant positive skill score over the Damped Trend Baseline ($\text{Skill}_{\text{trend}} > 0\%$) and Persistence Baseline ($\text{Skill}_{\text{pers}} > 0\%$).
4. **Inter-Storm Stability:** Minimum variance / inter-quartile range in storm-specific MAEs across the 24 validation storms.
5. **Parsimony:** If two configurations differ by $\le 0.2\text{ kt}$ Storm MAE, choose the model with fewer features and lower tree depth.

---

## 15. Clustered Uncertainty Protocol

Confidence intervals for intensity predictions must account for temporal autocorrelation:
- **Resampling Method:** Storm-Level Clustered Bootstrap ($B = 1,000$ iterations).
- **Unit of Resampling:** `system_id` (entire storm sequences resampled with replacement).
- **Reported Bound:** 95% empirical percentile confidence intervals $[2.5\text{th}, 97.5\text{th}\text{ percentiles}]$.

---

## 16. Benchmark Provenance

The historical benchmark figure of `MAE < 11.5 kt at 24h` is officially classified as:
### **`PROVISIONAL / UNPROVEN`**
It is a research aspiration, not an established operational requirement. In fact, the deterministic Damped Trend baseline already reaches $11.32\text{ kt}$ on the validation partition. Models will be judged by their empirical skill over persistence and damped trend, not solely by crossing 11.5 kt.

---

## 17. Authoritative Protected Artifact Inventory

The 12 protected model, dataset, and configuration artifacts are permanently registered in [`backend/config/protected_artifact_manifest.json`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/config/protected_artifact_manifest.json):

```
AUTHORITATIVE 12 PROTECTED ARTIFACTS
1. backend/models/risk_model_3d.joblib                                  (3f52f16be035c663...)
2. backend/models/v2_10yr/risk_model_3d.joblib                          (7c4f17861c0d48e9...)
3. backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib  (7f78946c01442793...)
4. backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib  (b0245d154f537861...)
5. backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib  (aca2e9423af471e5...)
6. backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib  (250948b2aa72babe...)
7. backend/config/frozen_alert_policy_v2.json                           (6ff3fe00ff9354c4...)
8. backend/data/historical/features_10yr.parquet                        (cdf3837f9ebe68ea...)
9. backend/data/historical/labeled_features_10yr_clean.parquet          (25070aad573c23bb...)
10. backend/data/era5/features_atmosphere_10yr_daily.parquet           (551aa9cb4ca460ec...)
11. backend/config/v2_3_frozen_experiment_manifest.json                 (73d407d798f68de2...)
12. backend/data/historical/imd_tracks_2016_2026.parquet               (e3f1b87455e71676...)
```

---

## 18. Training Design Verdict

### **PASS — TRAINING DESIGN READY**
The training and experimental design specification is complete, mathematically bounded, causally audited, and locked. **Zero machine learning models were trained.** Training remains strictly forbidden until explicit subsequent authorization.
