# Sagar-Drishti — Multi-Hazard Implementation Roadmap (Phases 1 to 6)

**Document ID:** `SD-ROADMAP-2026-MULTI-HAZARD-PHASE-1-6`  
**Phase:** Pre-Implementation Research & Architecture Phase  
**Date:** September 13, 2026  
**Status:** **[ROADMAP APPROVED — ZERO TRAINING CONDUCTED]**  
**Classification:** Strategic Engineering Roadmap & Phased Implementation Plan  

---

## Executive Summary

This roadmap defines the multi-year engineering and scientific strategy for expanding Sagar-Drishti from its current operational cyclone genesis baseline into an integrated, modular **Multi-Hazard Ocean & Coastal Intelligence Platform**.

Each phase is governed by strict **Gate Criteria**:
- No phase may proceed to model training without an audited dataset and validated ground-truth labels.
- No model may enter operational production without passing a full season of shadow observation.
- Production `v1.1.0` remains the sole operational decision authority until formal decommissioning.

```
MULTI-HAZARD IMPLEMENTATION PHASING OVERVIEW
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: CYCLONE GENESIS & BASELINE MATURATION [CURRENT - OPERATIONAL]      │
│ Status: Production v1.1.0 active; V2.3 Model D in Shadow Observation        │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: CYCLONE INTENSITY & TRACK DISPLACEMENT [NEXT TRANCHE]              │
│ Scope: Vmax / Central Pressure Regressor, Rapid Intensification Classifier, │
│        and 24h/48h Track Vector Forecaster                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: EXTREME SYNOPTIC WEATHER (WIND & PRECIPITATION)                    │
│ Scope: Gale-Force Wind Exceedance Engine & Gridded Rainfall Classifier       │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: MARINE & OCEAN HAZARDS (WAVES & MARINE HEATWAVES)                  │
│ Scope: Significant Wave Height (Hs) Regressor & Marine Heatwave Detector    │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 5: COASTAL LANDFALL IMPACTS (STORM SURGE & FLOODING)                  │
│ Scope: Coastal Surge Residual Regressor & Compound Inundation Modeling      │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 6: MULTI-HAZARD COMPOUND RISK FUSION                                  │
│ Scope: Non-Linear Copula Modeling & Multi-Hazard Alert Synthesis             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Tropical Cyclone Genesis Risk (Current Baseline)

### Objective
Maintain uninterrupted operational cyclone genesis alerts while accumulating longitudinal evidence for candidate V2.3 Model D.

| Dimension | Specification |
| :--- | :--- |
| **Datasets** | `features_10yr.parquet` (101 ocean features) + `features_atmosphere_10yr_daily.parquet` (123 ERA5 features) |
| **Labels** | `labeled_features_10yr_clean.parquet` (IMD RSMC Best Track with 48h clean negative buffer) |
| **Features** | 224 combined ocean-atmosphere features (thermodynamics, dynamics, kinematics, moisture) |
| **Models** | Production: `risk_model_3d.joblib` (v1.1.0); Shadow: `risk_model_{0,1,2,3}d.joblib` (V2.3 Model D) |
| **Validation Protocol** | Chronological test split (2025–2026), Event-level PR-AUC, Brier score, Reliability curve |
| **Automated Tests** | 256 unit and integration tests (`test_v2_3_shadow_evaluation.py`, `test_v2_3_shadow_monitoring.py`) |
| **Shadow Requirements** | Minimum 3 full cyclone seasons, $\ge 10$ tracked storms, coverage $\ge 85\%$, drift monitoring |
| **Operationalization Exit Gate** | Unanimous scientific review, zero production regressions, formal promotion protocol |

---

## Phase 2: Cyclone Intensity & Track Displacement Modules

### Objective
Develop specialized models to forecast cyclone intensity ($V_{\max}$, $\Delta P$, Rapid Intensification) and storm center trajectory displacement $(\Delta \phi, \Delta \lambda)$ for active cyclonic systems.

```
PHASE 2 WORKFLOW
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│  DATA & LABELS INGESTION             │     │  SPECIALIZED MODEL ARCHITECTURES     │
│  - IMD Best Track 2,544 synoptic     │ ──> │  - Intensity: Multi-Task Gradient    │
│    fixes (Vmax, Pc, coordinates)     │     │    Boosted Trees (LightGBM / XGBoost)│
│  - ERA5 850 hPa vorticity, shear, RH │     │  - Rapid Intensification: Calibrated │
│  - Copernicus SST & Ocean Heat Cont. │     │    Ensemble Classifier               │
└──────────────────────────────────────┘     │  - Track: Deep Layer Steering Flow   │
                                             │    Autoregressive Regressor          │
                                             └──────────────────────────────────────┘
```

| Dimension | Specification |
| :--- | :--- |
| **Target Definitions** | 1. Intensity: Continuous $V_{\max}(T+H)$ in knots and central pressure $P_c(T+H)$ in hPa.<br>2. Rapid Intensification: Binary indicator $\mathbf{1}(V_{\max}(T+24\text{h}) - V_{\max}(T) \ge 30\text{ kt})$.<br>3. Track: 2D vector displacement $(\Delta \phi, \Delta \lambda)$ at $H \in \{24\text{h}, 48\text{h}, 72\text{h}\}$. |
| **Authoritative Labels** | `OBSERVATIONAL GROUND TRUTH`: IMD RSMC New Delhi Official Cyclone Best Track Archive. |
| **Feature Layer** | Common ocean-atmosphere features + inner-core ocean heat content, radius of maximum wind ($R_{\max}$), 200 hPa divergence, translation speed, 850–200 hPa environmental steering wind vectors. |
| **Model Architectures** | Multi-Task Gradient Boosted Regressor (LightGBM) for intensity; Cost-sensitive Random Forest for RI; Bidirectional LSTM / Spatial Attention Transformer for trajectory. |
| **Validation Protocol** | Track-level leave-one-season-out cross-validation; Primary Metrics: Mean Absolute Error ($MAE$) on $V_{\max}$ (target: $<12\text{ kt}$ at 24h), Mean Track Error ($MTE$) in km, RI Critical Success Index ($CSI$). |
| **Shadow Requirements** | Execute in shadow observation alongside active IMD cyclonic disturbances for $\ge 2$ seasons before operational alerting consideration. |
| **Operationalization Gate** | Outperform climatology-persistence (CLIPER) benchmark by $\ge 20\%$ at 24h lead time. |

---

## Phase 3: Extreme Synoptic Weather (Wind & Precipitation Modules)

### Objective
Predict basin-wide and coastal gale-force wind exceedance and high-impact daily rainfall accumulation.

| Dimension | Specification |
| :--- | :--- |
| **Target Definitions** | 1. Extreme Wind: Probability $P(U_{10} \ge 34.0\text{ kt} \text{ within } H)$ (provisional threshold).<br>2. Extreme Rain: Probability $P(R_{24} \ge 64.5\text{ mm/day} \text{ within } H)$ (IMD Heavy Rain). |
| **Required Datasets** | Ingest IMD High-Resolution 0.25° Gridded Daily Rainfall (Pai et al.) and NASA GPM IMERG; acquire IMD coastal AWS surface anemometer records. |
| **Feature Layer** | Horizontal pressure gradients ($\nabla MSLP$), Total Column Water Vapor ($TCWV$), boundary layer stability, low-level moisture convergence. |
| **Model Architectures** | Extreme Value Quantile Regression Forests & Calibrated Spatio-Temporal Gradient Boosted Classifiers. |
| **Validation Protocol** | Spatially blocked cross-validation; Primary Metrics: Brier Score, Expected Calibration Error ($ECE < 0.05$), Critical Success Index ($CSI$), Frequency Bias ($0.9 \le \text{Bias} \le 1.1$). |
| **Shadow Requirements** | 1 full southwest and northeast monsoon season under shadow monitoring. |
| **Operationalization Gate** | Demonstration of reliable calibration across coastal warning districts. |

---

## Phase 4: Marine & Ocean State Hazards (Waves & Marine Heatwaves)

### Objective
Predict hazardous sea state ($H_s \ge 4.0\text{ m}$) and prolonged thermal anomalies devastating marine ecosystems.

| Dimension | Specification |
| :--- | :--- |
| **Target Definitions** | 1. Extreme Waves: Significant wave height $H_s \ge 4.0\text{ m}$ (provisional threshold).<br>2. Marine Heatwaves: $SST \ge 90\text{th}$ percentile for $\ge 5$ consecutive days (Hobday et al., 2016). |
| **Required Datasets** | Ingest Copernicus Marine Global Ocean Waves Reanalysis (WAVERYS) and INCOIS Wave Rider Buoy archives; compute 30-year SST climatological baseline. |
| **Feature Layer** | Wave fetch, surface wind stress, friction velocity $u_*$, swell flux from Southern Ocean, mixed layer depth, solar insolation. |
| **Model Architectures** | Physics-Informed Neural Network (PINN) or Gradient Boosted Wave Regressor; Thermal Accumulation Degree Heating Days Regressor. |
| **Validation Protocol** | Moored buoy point-to-point comparison; Primary Metrics: Wave Height RMSE ($<0.5\text{ m}$), Scatter Index ($SI < 0.20$), MHW Day F1 Score ($>0.75$). |
| **Shadow Requirements** | Minimum 12 months continuous shadow comparison against INCOIS operational wave forecasts. |

---

## Phase 5: Coastal Landfall Impacts (Storm Surge & Coastal Flooding)

### Objective
Model coastal water level surge residuals above astronomical tide and compound coastal flood inundation.

| Dimension | Specification |
| :--- | :--- |
| **Target Definitions** | 1. Storm Surge: Peak water level residual $\eta_{\text{peak}} \ge 1.0\text{ m}$ above astronomical tide.<br>2. Coastal Flooding: Spatial flood inundation extent and depth $d_{\text{flood}} \ge 0.5\text{ m}$. |
| **Required Datasets** | Acquire GEBCO 15 arc-second coastal bathymetry, TPXO9 tidal constituent grids, INCOIS coastal tide gauge time series, and 30m Coastal DEM (Copernicus GLO-30). |
| **Feature Layer** | Bathymetric shelf slope, storm approach angle, astronomical tidal phase, coastal levee height, river discharge, rainfall accumulation. |
| **Model Architectures** | Hydrodynamic surrogate metamodel (training ML on ADCIRC / SLOSH numerical simulation outputs). |
| **Validation Protocol** | Coastal tide gauge station verification; Primary Metrics: Peak Surge Height MAE ($<25\text{ cm}$), Arrival Timing Error ($<2\text{ hours}$), Inundation Spatial IoU ($>0.65$). |
| **Shadow Requirements** | Formal hindcast validation on all historical landfalling cyclones 2016–2026 before any shadow activation. |

---

## Phase 6: Multi-Hazard Compound Risk Fusion (Synthesis Layer)

### Objective
Synthesize individual, calibrated hazard predictions into a non-linear, multi-hazard coastal compound risk indicator.

```
PHASE 6 MULTI-HAZARD RISK SYNTHESIS
┌───────────────────────────────┐     ┌───────────────────────────────┐
│ INDIVIDUAL HAZARD ENGINES     │     │ COMPOUND FUSION ARCHITECTURE  │
│ - Cyclone Genesis Probability │     │ - Joint Copula Tail Modeling  │
│ - Cyclone Intensity (Vmax)    │ ──> │   (Clayton / Vine Copula)     │ ──> MULTI-HAZARD
│ - Gale-Force Wind Exceedance  │     │ - Physical Cascade Matrix     │     INTELLIGENCE
│ - Extreme Rainfall Exceedance │     │ - Calibrated Risk Tiers       │     BULLETIN
│ - Storm Surge Residual        │     │ - Uncertainty Propagation     │
└───────────────────────────────┘     └───────────────────────────────┘
```

| Dimension | Specification |
| :--- | :--- |
| **Methodology** | Non-additive joint probability estimation using Archimedean Copulas; Bayesian compound vulnerability mapping. |
| **Safety Firewall** | Multi-hazard synthesis must never alter or suppress individual hazard alerts. Individual hazard exceedances must always trigger primary warnings. |
| **Operational Output** | Sagar-Drishti Coastal Multi-Hazard Threat Bulletin (Tiered Warning: Normal, Advisory, Watch, Warning, Emergency). |

---

## 7. Phased Implementation Timeline & Resource Allocation

```
ROADMAP IMPLEMENTATION SCHEDULE
─────────────────────────────────────────────────────────────────────────────
Phase 1: Cyclone Genesis Baseline (Current)      [ACTIVE - SHADOW & PROD]
Phase 2: Cyclone Intensity & Track Modules       [MONTHS 1 – 6]
Phase 3: Extreme Wind & Rainfall Modules         [MONTHS 6 – 12]
Phase 4: Marine Waves & Heatwaves Modules        [MONTHS 12 – 18]
Phase 5: Storm Surge & Coastal Flooding          [MONTHS 18 – 24]
Phase 6: Multi-Hazard Compound Risk Fusion       [MONTHS 24 – 30]
─────────────────────────────────────────────────────────────────────────────
```

### Exit Criteria for Each Phase
1. **Zero Production Mutation:** Existing production pipelines remain untouched.
2. **Cryptographic Checksum Verification:** Checksums of protected baseline artifacts remain identical.
3. **Comprehensive Automated Test Coverage:** $\ge 90\%$ test coverage on all newly added services and registry endpoints.
4. **Peer-Reviewed Scientific Report:** Generation of an audited validation report prior to initiating any shadow observation.
