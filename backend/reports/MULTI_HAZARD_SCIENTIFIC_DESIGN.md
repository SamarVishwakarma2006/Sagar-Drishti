# Sagar-Drishti — Multi-Hazard Scientific & Architecture Design

**Document ID:** `SD-DESIGN-2026-MULTI-HAZARD-SCIENTIFIC-ARCHITECTURE`  
**Phase:** Pre-Implementation Research & Architecture Phase  
**Date:** September 13, 2026  
**Status:** **[RESEARCH SPECIFICATION COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Scientific Specification & Engineering Blueprint  

---

## Executive Summary

This document defines the foundational scientific, physical, mathematical, and architectural design required to expand Sagar-Drishti from a single cyclone genesis risk predictor into a modular, multi-hazard **Ocean & Coastal Intelligence Platform**.

### Immutable Operational Baseline
- **Production `v1.1.0`:** Sole operational decision authority. 101 ocean features, 3-day risk horizon. Completely untouched.
- **Candidate `V2.3 Model D`:** Strictly isolated in shadow observation mode. 224 combined ocean-atmosphere features across 0d, 1d, 2d, and 3d horizons.
- **Protected Checksum Invariance:** All 11 protected artifacts (production model, candidate models, alert policy, historical Parquets, ERA5 archives, and experiment manifests) remain 100% byte-for-byte immutable.
- **Zero-Training Mandate:** Zero machine learning models are trained, fine-tuned, or calibrated in this phase.

---

## 1. Mathematical Target Formulations & Threshold Standards

A foundational principle of Sagar-Drishti is **Target Independence**: *No hazard is permitted to recycle or proxy the target of another hazard.* Each physical phenomenon is governed by distinct equations and requires an independent, mathematically rigorous prediction target.

```
TARGET FORMULATION MATRIX
┌──────────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
│ Hazard Category              │ Explicit Mathematical Target Equation                                       │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ 1. Tropical Cyclone Genesis  │ Y_genesis(T+H) = 1 if Active Cyclone Fix within 500km at T+H, else 0        │
│ 2. Cyclone Intensity         │ V_max(T+H) in knots (Regression) & 1(V_max(T+24) - V_max(T) >= 30 kt) (RI)  │
│ 3. Cyclone Track             │ Delta_pos(T+H) = [lat(T+H) - lat(T), lon(T+H) - lon(T)] in degrees         │
│ 4. Extreme Wind              │ 1(max_{t in [T, T+H]} U_10(t) >= 34.0 kt) [PROVISIONAL THRESHOLD]           │
│ 5. Extreme Rainfall          │ 1(R_24(T+H) >= 64.5 mm/day) [PROVISIONAL THRESHOLD]                         │
│ 6. Extreme Waves             │ 1(max_{t in [T, T+H]} H_s(t) >= 4.0 m) [PROVISIONAL THRESHOLD]             │
│ 7. Storm Surge               │ eta_peak(T+H) = max_{t in [T, T+H]} [h(t) - zeta_tide(t)] >= 1.0 m [PROV.]  │
│ 8. Coastal Flooding          │ A_flood(T+H) in km^2 & 1(d_flood(x, y, T+H) >= 0.5 m) [PROVISIONAL]         │
│ 9. Extreme Sea-Level Events  │ 1(h(T+H) >= h_crit,99.5) [PROVISIONAL THRESHOLD]                            │
│ 10. Marine Heatwaves (MHW)   │ 1(SST(t) >= SST_90th(day_of_year) for >= 5 consecutive days within [T, T+H])│
└──────────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Mathematical Definitions

#### Target 1: Tropical Cyclone Genesis / Event Risk
- **Formulation:** Binary classification of whether an organized cyclonic system (Depression or higher, $\text{grade} \ge \text{D}$) exists or forms within radius $R_{\text{crit}} = 500\text{ km}$ of the forecast coordinate at lead time $H \in \{0, 1, 2, 3\}\text{ days}$:
  $$Y_{\text{genesis}}(T+H) = \begin{cases} 1 & \text{if } \exists \, \text{IMD track fix with } \text{dist}(\mathbf{x}_{\text{site}}, \mathbf{x}_{\text{storm}}) \le 500\text{ km at } T+H \\ 0 & \text{if no storm fix and } \Delta t_{\text{event}} \ge 48\text{ h (clean negative buffer)} \end{cases}$$
- **Threshold Justification:** IMD RSMC standard criterion for cyclonic disturbance classification. Fully operationalized in Sagar-Drishti v1.1.0 and V2.3.

#### Target 2: Cyclone Intensity & Rapid Intensification (RI)
- **Continuous Formulation:** Maximum 3-minute sustained surface wind speed $V_{\max}(T+H)$ in knots and central pressure drop $\Delta P(T+H) = P_{\text{env}} - P_c$ in hPa:
  $$\hat{V}_{\max}(T+H) = f_{\text{intensity}}(\mathbf{x}(T), H)$$
- **Rapid Intensification (RI) Formulation:** Binary indicator of intensity increase $\ge 30\text{ knots}$ ($\approx 15.4\text{ m/s}$) within a 24-hour window:
  $$Y_{\text{RI}}(T+24\text{h}) = \mathbf{1}\left(V_{\max}(T+24\text{h}) - V_{\max}(T) \ge 30\text{ kt}\right)$$
- **Threshold Justification:** Official WMO and IMD standard operational definition for tropical cyclone Rapid Intensification.

#### Target 3: Cyclone Track Trajectory Displacement
- **Formulation:** Vector displacement $(\Delta \phi, \Delta \lambda)$ of the storm center from initialization time $T$ to lead time $T+H$:
  $$\Delta \mathbf{p}(T+H) = \begin{bmatrix} \phi(T+H) - \phi(T) \\ \lambda(T+H) - \lambda(T) \end{bmatrix}$$
- **Verification Metric:** Great-circle Along-Track Error ($ATE$) and Cross-Track Error ($CTE$) in kilometers relative to the empirical track direction.

#### Target 4: Extreme Wind Exceedance
- **Formulation:** Probability that 10-meter wind speed $U_{10}$ exceeds gale-force threshold within forecast window $[T, T+H]$:
  $$Y_{\text{wind}}(T, H) = \mathbf{1}\left(\max_{t \in [T, T+H]} U_{10}(t) \ge U_{\text{threshold}}\right)$$
- **Threshold Standard:** $U_{\text{threshold}} = 34.0\text{ knots}$ ($17.2\text{ m/s}$, Beaufort Force 8).  
  *Status:* `is_provisional = True` pending coastal anemometer station calibration.

#### Target 5: Extreme Rainfall Accumulation
- **Formulation:** Probability of daily accumulated precipitation $R_{24}$ exceeding IMD "Heavy Rainfall" threshold:
  $$Y_{\text{rain}}(T+H) = \mathbf{1}\left(R_{24}(T+H) \ge 64.5\text{ mm/day}\right)$$
  - Extreme Rainfall Category 2 (Very Heavy): $R_{24} \ge 115.6\text{ mm/day}$ (`is_provisional = True`)
  - Extreme Rainfall Category 3 (Extremely Heavy): $R_{24} \ge 204.5\text{ mm/day}$ (`is_provisional = True`)
- **Threshold Standard:** IMD National Meteorological Classification.

#### Target 6: Extreme Waves / Dangerous Sea State
- **Formulation:** Significant wave height ($H_s$, mean of highest one-third of waves) exceeding dangerous sea state criteria:
  $$Y_{\text{wave}}(T, H) = \mathbf{1}\left(\max_{t \in [T, T+H]} H_s(t) \ge 4.0\text{ meters}\right)$$
- **Threshold Standard:** WMO Sea State Code 6 ("Very Rough Sea", $H_s \in [4.0, 6.0]\text{ m}$).  
  *Status:* `is_provisional = True` until wave rider buoy calibration is completed.

#### Target 7: Storm Surge Water Level Residual
- **Formulation:** Peak non-tidal sea level residual $\eta_{\text{peak}}$ above predicted astronomical tide $\zeta_{\text{tide}}$ along the coastline:
  $$\eta_{\text{peak}}(T, H) = \max_{t \in [T, T+H]} \left[ h(t) - \zeta_{\text{tide}}(t) \right]$$
  $$Y_{\text{surge}}(T, H) = \mathbf{1}\left(\eta_{\text{peak}}(T, H) \ge 1.0\text{ meter}\right)$$
- **Threshold Standard:** $1.0\text{ meter}$ coastal surge represents standard IMD warning threshold for coastal inundation risk.  
  *Status:* `is_provisional = True`.

#### Target 8: Coastal Flooding
- **Formulation:** Spatial inundation extent area $A_{\text{flood}}$ in $\text{km}^2$ and binary inundation at coastal coordinate:
  $$Y_{\text{flood}}(\mathbf{x}, T+H) = \mathbf{1}\left(d_{\text{flood}}(\mathbf{x}, T+H) \ge 0.5\text{ meters}\right)$$
- **Threshold Standard:** Water depth $\ge 0.5\text{ m}$ disrupts vehicular evacuation and structural integrity.  
  *Status:* `is_provisional = True`.

#### Target 9: Extreme Sea-Level Events
- **Formulation:** Exceedance of local 99.5th percentile total coastal water level:
  $$Y_{\text{esl}}(T+H) = \mathbf{1}\left(h(T+H) \ge h_{\text{crit}, 99.5}\right)$$
- **Threshold Standard:** Station-specific extreme value return level.  
  *Status:* `is_provisional = True`.

#### Target 10: Marine / Ocean Anomalies (Marine Heatwaves)
- **Formulation:** Internationally standardized definition of Marine Heatwaves (Hobday et al., 2016):
  $$Y_{\text{MHW}}(t) = \mathbf{1}\left(SST(t) \ge SST_{90\text{th}}(\text{doy}) \quad \forall t \in [\tau, \tau + 4]\right)$$
  where $SST_{90\text{th}}(\text{doy})$ is the day-of-year 90th percentile calculated over an 11-day centered window across a 30-year climatological reference period.
- **Threshold Standard:** Authoritative international oceanographic standard.

---

## 2. Authoritative Label Strategy

Each hazard must be paired with an authoritative label source. The distinction between observational ground truth and modeled reference data must be rigorously enforced:

```
LABEL SOURCE ARCHITECTURE
┌──────────────────────────────┬───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Hazard                       │ Primary Label Source          │ Data Classification           │ Known Biases / Limitations    │
├──────────────────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Tropical Cyclone Genesis     │ IMD RSMC Best Track Archive   │ DERIVED LABELS                │ Spatial buffer boundary edge  │
│ Cyclone Intensity            │ IMD RSMC Official Fixes       │ OBSERVATIONAL GROUND TRUTH    │ Open-ocean Dvorak uncertainty │
│ Cyclone Track                │ IMD RSMC Synoptic Fixes       │ OBSERVATIONAL GROUND TRUTH    │ 6-hour temporal granularity   │
│ Extreme Wind                 │ ERA5 / IMD Coastal AWS Gauges │ REANALYSIS / OBSERVATIONAL    │ ERA5 underestimates gusts     │
│ Extreme Rainfall             │ IMD 0.25° Gridded Rainfall    │ OBSERVATIONAL GROUND TRUTH    │ Sparse oceanic rain gauges    │
│ Extreme Waves                │ INCOIS Wave Rider Buoys/WAVE  │ OBSERVATIONAL / REANALYSIS    │ Buoys confined to coast/shelf │
│ Storm Surge                  │ INCOIS Coastal Tide Gauges    │ OBSERVATIONAL GROUND TRUTH    │ Estuarine gauge silting       │
│ Coastal Flooding             │ Sentinel-1 SAR / Field Surveys│ OBSERVATIONAL / DERIVED       │ Satellite overpass latency    │
│ Extreme Sea-Level Events     │ Survey of India Tide Gauges   │ OBSERVATIONAL GROUND TRUTH    │ Benchmark datum shifts        │
│ Marine Heatwaves             │ Copernicus GLORYS / OISST     │ DERIVED LABELS                │ Nearshore satellite bias      │
└──────────────────────────────┴───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## 3. Causal Cutoff & Anti-Leakage Firewall

Machine learning models deployed for operational maritime forecasting are highly vulnerable to subtle forms of future information leakage. Sagar-Drishti enforces an unbreachable **Causal Firewall**:

```
CAUSAL FIREWALL AT FORECAST INITIALIZATION TIME T
─────────────────────────────────────────────────────────────────────────────
Past & Available: tau <= T                               Future: tau > T
[Ocean State: GLORYS / NRT]                               [TARGET WINDOW]
[Atmospheric State: ERA5 18Z]                             [T+0d, T+1d, T+2d, T+3d]
─────────────────────────────────────────────┬───────────────────────────────
                     FORECAST TIME T          │
             (Strict Cutoff: 18:00 UTC)      │  ABSOLUTE LEAKAGE PROHIBITION
                                             │  NO future observations tau > T
                                             │  NO post-season track revisions
                                             │  NO future rolling window values
```

### Critical Leakage Risks & Prevention Protocols
1. **Reanalysis Ingestion Latency Leakage:** ERA5 final quality-controlled archives have a 2-to-3 month latency. Real-time inference cannot utilize final ERA5. Models trained on final reanalysis must be evaluated against preliminary real-time analyses (ERA5T or GFS) to detect degradation.
2. **Target Contamination in Rolling Windows:** Rolling statistics (e.g., 7-day, 14-day, 30-day means) must strictly use backward-looking intervals $[T - \tau, T]$. Forward-looking centered windows $[T - \tau/2, T + \tau/2]$ are mathematically prohibited.
3. **Post-Event Best Track Revisions:** Post-season best tracks adjust storm genesis times by 6 to 12 hours based on retrospective satellite re-analysis. Operational inference only has real-time synoptic bulletins. Testing must include operational bulletins to measure real-world performance.
4. **Spatial Auto-Correlation Leakage:** Splitting train and test sets by random row shuffling leaks identical storm structures across time. All splits must be **chronologically separated by full seasons or independent multi-year blocks**.

---

## 4. Specialized Modular Architecture

Sagar-Drishti transitions from a monolithic pipeline into a decoupled, modular multi-hazard architecture. Each hazard module operates with independent target schemas, independent feature contracts, independent training lifecycles, and independent uncertainty quantification.

```
SAGAR-DRISHTI MULTI-HAZARD MODULAR PLATFORM
│
├── [1] DATA INGESTION & QUALITY FIREWALL
│   ├── Copernicus Marine Reanalysis / Analysis Ingestion Engine
│   ├── ECMWF ERA5 / Operational Atmospheric Ingestion Engine
│   └── In-Situ Sensor Ingestion (IMD Synoptic, INCOIS Buoys & Gauges)
│
├── [2] COMMON ENVIRONMENTAL FEATURE LAYER
│   ├── Unified Ocean State Engine (SST, MLD, Salinity, Currents, SLA)
│   └── Unified Atmospheric State Engine (Vorticity, VWS, RH, Divergence, MSLP)
│
├── [3] SPECIALIZED HAZARD PREDICTION MODULES
│   ├── Module A: Cyclone Intelligence
│   │   ├── Cyclone Genesis / Event Risk (v1.1.0 Prod / V2.3 Model D Shadow)
│   │   ├── Cyclone Intensity Regressor & Rapid Intensification Classifier
│   │   └── Cyclone Track & Trajectory Displacement Forecaster
│   ├── Module B: Extreme Synoptic Weather
│   │   ├── Extreme Wind Exceedance Engine
│   │   └── Extreme Precipitation Exceedance Engine
│   ├── Module C: Marine & Ocean State Hazards
│   │   ├── Extreme Waves & Sea State Forecaster
│   │   └── Marine Heatwave (MHW) Thermal Anomaly Forecaster
│   └── Module D: Coastal & Landfall Impacts
│       ├── Coastal Storm Surge Surge-Tide Forecaster
│       └── Coastal Compound Inundation Flooding Forecaster
│
├── [4] VERIFICATION & SHADOW OBSERVATION SUITE
│   ├── Independent Event-Level Contingency Evaluators
│   ├── Calibration Evaluator (Brier Score, Expected Calibration Error)
│   └── Feature Drift & Data Quality Telemetry (KS-test, PSI)
│
└── [5] MULTI-HAZARD RISK FUSION ENGINE (FUTURE RESEARCH ONLY)
    ├── Joint Dependence Copula Engine (Non-Additive)
    └── Compound Vulnerability Mapping
```

---

## 5. Common Environmental Feature Layer vs Hazard-Specific Features

To maintain computational efficiency and physical coherence, Sagar-Drishti establishes a **Common Environmental Feature Layer** that is shared across modules, while preserving specialized feature inputs for specific physical hazards.

### A. Common Environmental Features (Shared Across All Modules)
- **Thermodynamic Ocean Drivers:** Sea Surface Temperature ($SST$), SST 30-day baseline anomaly, SST z-score, Mixed Layer Depth ($MLD$), Ocean Heat Content proxy ($OHC \approx \rho c_p \int (T - 26)\,dz$).
- **Dynamic Ocean Drivers:** Sea Level Anomaly ($SLA$), surface geostrophic current vectors ($u_{\text{curr}}, v_{\text{curr}}$), current velocity magnitude $\sqrt{u^2 + v^2}$.
- **Atmospheric Kinematics:** 850 hPa relative vorticity ($\zeta_{850}$), area-weighted vorticity exceedance fraction, Vertical Wind Shear ($VWS = \|\mathbf{v}_{200} - \mathbf{v}_{850}\|$), 850 hPa horizontal convergence.
- **Atmospheric Moisture & Mass:** Relative humidity at 850, 700, 500 hPa ($RH_{850}, RH_{700}, RH_{500}$), 500 hPa geopotential height ($z_{500}$), Mean Sea Level Pressure ($MSLP$).

### B. Hazard-Specific Feature Requirements
- **Cyclone Intensity Specific:** Inner-core ocean heat content, radius of maximum winds ($R_{\max}$), translation speed, upper-tropospheric divergence ($d_{200}$), diurnal convective cooling pulse.
- **Extreme Wind Specific:** Surface pressure horizontal gradient ($\nabla MSLP$), boundary layer stability index, low-level jet momentum transport, surface roughness length ($z_0$).
- **Extreme Rainfall Specific:** Total Column Water Vapor ($TCWV$), convective available potential energy ($CAPE$), moisture flux convergence ($\nabla \cdot (q\mathbf{v})$), lifting condensation level.
- **Extreme Waves Specific:** Wind-sea fetch length, wind duration, surface friction velocity ($u_*$), directional wave spectra, remote swell flux from Southern Indian Ocean.
- **Storm Surge Specific:** Coastal bathymetric depth gradient ($dD/dx$), distance from continental shelf break, astronomical tidal phase, coastline curvature angle, storm approach angle.
- **Coastal Flooding Specific:** Topographic elevation ($z_{\text{DEM}}$), distance to river outlet, antecedent soil moisture saturation (API), urban imperviousness fraction, estuarine tidal prism.

---

## 6. Validation Design: Metrics & Split Protocols

Evaluation methodology must align with the mathematical nature of the hazard. **Event-based hazards must never be validated purely on row-level accuracy.**

```
VALIDATION METHODOLOGY TAXONOMY
┌──────────────────────────────┬───────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Hazard                       │ Evaluation Level  │ Primary Metric                │ Secondary Metrics             │
├──────────────────────────────┼───────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Cyclone Genesis              │ EVENT-LEVEL       │ PR-AUC / Lead Brier Score     │ Precision, Recall, F1, CSI    │
│ Cyclone Intensity            │ TRACK / FIX-LEVEL │ MAE & RMSE on V_max (knots)   │ Bias, RI Brier Score, RI CSI  │
│ Cyclone Track                │ TRACK-LEVEL       │ Mean Track Error (MTE, km)    │ Along-Track & Cross-Track Err │
│ Extreme Wind                 │ ROW & EVENT-LEVEL │ Brier Score & PR-AUC          │ ECE, CSI, False Alarm Ratio   │
│ Extreme Rainfall             │ EVENT & SPATIAL   │ Critical Success Index (CSI)  │ Frequency Bias, ETS, F1       │
│ Extreme Waves                │ CONTINUOUS & PROB │ RMSE on H_s (meters) & PR-AUC │ Scatter Index, Peak Bias, CSI │
│ Storm Surge                  │ STATION EVENT     │ Peak Surge Error (MAE, cm)    │ Arrival Time Error (hours)    │
│ Coastal Flooding             │ SPATIAL PIXEL     │ Intersection over Union (IoU) │ Critical Success Index, MAE   │
│ Extreme Sea-Level Events     │ STATION EXTREME   │ Quantile Score (99.5th %ile)  │ Brier Score, Return Period Err│
│ Marine Heatwaves             │ EVENT & DURATION  │ MHW Day F1 & Duration MAE     │ Peak Intensity Error (°C), ECE│
└──────────────────────────────┴───────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### Chronological Partitioning Rules
1. **Three-Tier Chronological Partition:**
   - **Training Set (Years 1–7):** Model optimization and feature selection.
   - **Validation Set (Years 8–9):** Calibration, probability threshold tuning, hyperparameter selection.
   - **Final Test Set (Year 10):** Single-pass blind evaluation. Repeated tuning on the final test set is strictly prohibited.
2. **Event-Level Aggregation:** For cyclone-related hazards, metrics must be reported per unique storm system ($N_{\text{storms}}$), positive fix observations ($N_{\text{pos}}$), and independent seasons ($N_{\text{seasons}}$).

---

## 7. Multi-Hazard Compound Risk Fusion (Future Research Architecture)

> [!CAUTION]
> **Prohibition on Naive Additive Probability:** Multi-hazard risk must **never** be calculated by simply adding individual probabilities ($P_{\text{total}} \neq P_{\text{cyclone}} + P_{\text{wind}} + P_{\text{rain}} + P_{\text{surge}}$). Hazards are physically coupled through joint atmospheric and oceanic dynamics. Naive addition violates the axioms of probability, causes double-counting, and generates severe operational false alarm rates.

```
FUTURE COMPOUND RISK FUSION CONCEPT (RESEARCH ONLY)
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NON-LINEAR COPULA FUSION                             │
│                                                                             │
│   P(Compound Hazard) = C_theta( F_wind(u), F_rain(r), F_surge(eta) )        │
│                                                                             │
│   Where:                                                                    │
│   - F_i are marginal calibrated cumulative distribution functions           │
│   - C_theta is an Archimedean / Vine Copula modeling physical tail dependence│
│   - Propagates joint uncertainty without naive probability summation        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Future Fusion Research Agenda
1. **Tail Dependence Modeling:** Heavy rainfall and storm surge exhibit strong upper-tail dependence during tropical cyclone landfalls. A Clayton or Gumbel copula must be investigated to model concurrent extremes.
2. **Uncertainty Propagation:** Probabilistic ensemble spreads from individual hazard engines must propagate into the compound risk metric rather than point-estimate collapse.
3. **Double-Counting Remediation:** Separating wind damage risk from surge damage risk when both are forced by the identical cyclonic pressure field.

---

## 8. Operational Architecture & Three-Tier Model Isolation

To safeguard operational reliability, Sagar-Drishti enforces strict architectural isolation between production and research models:

```
THREE-TIER ISOLATION MODEL
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: PRODUCTION                                                          │
│ - Model: v1.1.0 (Sole operational decision authority)                       │
│ - Alert Policy: Frozen v2 (Strictly immutable)                              │
│ - Zero dependencies on research code or experimental hazard models          │
├─────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: SHADOW CANDIDATES                                                   │
│ - Model: V2.3 Model D (224 features, 4 horizons)                            │
│ - Execution: Read-only background shadow inference                          │
│ - Telemetry: SQLite evidence store; zero impact on live API alerts          │
├─────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: RESEARCH PROPOSED HAZARDS                                           │
│ - Modules: Intensity, Track, Wind, Rain, Waves, Surge, Flooding, MHW        │
│ - Status: Strictly RESEARCH_PROPOSED                                        │
│ - Runtime: Decoupled metadata registry; cannot be invoked by alert engine   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Provisional Prioritization of Next Hazards

Based on the completed data inventory, label audit, and physical feasibility matrix, the candidate hazards are prioritized into three development tranches:

```
DEVELOPMENT TRANCHE RANKING
┌─────────┬──────────────────────────────────┬──────────────┬────────────────────────────────────────┐
│ Tranche │ Hazard Module                    │ Feasibility  │ Primary Recommendation Rationale       │
├─────────┼──────────────────────────────────┼──────────────┼────────────────────────────────────────┤
│ TRANCHE │ 1. Cyclone Intensity (Vmax, RI)  │ HIGH         │ IMD ground-truth fixes resident;       │
│ 1 (TOP) │ 2. Extreme Wind (Open Ocean)     │ HIGH         │ ERA5 10m wind resident; high value;    │
│         │ 3. Marine Heatwaves (MHW)        │ HIGH         │ 10-yr SST resident; established metric.│
├─────────┼──────────────────────────────────┼──────────────┼────────────────────────────────────────┤
│ TRANCHE │ 4. Cyclone Track Displacement    │ MEDIUM       │ Fixes resident; needs NWP steering;    │
│ 2       │ 5. Extreme Rainfall Accumulation │ MEDIUM       │ Critical value; needs IMD 0.25° rain;  │
│         │ 6. Extreme Waves (Hs, Period)    │ LOW–MED      │ High marine value; needs WAVERYS.      │
├─────────┼──────────────────────────────────┼──────────────┼────────────────────────────────────────┤
│ TRANCHE │ 7. Storm Surge Coastal Residual  │ LOW          │ Needs coastal tide gauges & GEBCO;     │
│ 3       │ 8. Extreme Sea-Level Events      │ LOW          │ Needs coastal tide gauge network;      │
│         │ 9. Coastal Inundation Flooding   │ NOT FEASIBLE │ Needs 30m Coastal DEM & river runoff.  │
└─────────┴──────────────────────────────────┴──────────────┴────────────────────────────────────────┘
```

---

## 10. Conservative Novelty Analysis

In compliance with academic rigor, Sagar-Drishti distinguishes between established meteorological science and its potentially distinctive implementation:

```
NOVELTY DIFFERENTIATION MATRIX
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ ESTABLISHED IN SCIENTIFIC LITERATURE         │ POTENTIALLY DISTINCTIVE IN SAGAR-DRISHTI     │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Using SST and VWS to predict TC intensity    │ Unified 224-feature ocean-atmosphere causal  │
│ (e.g., DeMaria SHIPS statistical model).     │ vector spanning both Arabian Sea & BoB.      │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Marine Heatwave thresholding using 90th %ile │ Longitudinal shadow-observation architecture │
│ climatology (Hobday et al., 2016).           │ logging live calibrated evidence pre-deploy. │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Numerical wave modeling via WaveWatch III    │ Decoupled multi-hazard registry enforcing    │
│ driven by atmospheric winds.                 │ explicit causal cutoffs and test isolation.  │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Deep learning for cyclone track prediction   │ Strict separation of operational authority   │
│ using convolutional LSTM / Transformers.     │ from shadow and research prediction engines. │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 11. Explicit Limitations & Scientific Constraints

1. **Zero High-Resolution Eyewall Dynamics:** ERA5 reanalysis (0.25° grid, ~28 km) cannot resolve tropical cyclone eye thermodynamics ($<15\text{ km}$).
2. **Absence of In-Situ Oceanic Rain Gauges:** Open-ocean precipitation ground truth is non-existent; satellite precipitation (GPM IMERG) must be utilized as reference data rather than direct station observation.
3. **Delayed Reanalysis Availability:** ERA5 final archives have a 2-to-3 month latency, precluding operational use without real-time analysis feeds.
4. **Shallow-Water Bathymetric Omission:** Nearshore wave breaking, surf zone setup, and tidal shoaling cannot be captured without coastal bathymetry grids.
5. **Sample Size Constraints in North Indian Ocean:** The North Indian Ocean experiences only 4 to 6 cyclonic storms annually, limiting the sample of Category 4 and 5 super cyclonic storms for deep neural network training.
6. **Compound Flood Interdependencies:** Flooding caused by the coincidence of extreme rainfall and storm surge cannot be modeled without hydrodynamic elevation grids.
7. **Dvorak Intensity Uncertainty:** IMD Best Track intensity estimates in the open ocean have an inherent $\pm 5$ to $10\text{ knot}$ satellite estimation uncertainty.
8. **Lack of Coastal Tide Gauges:** Storm surge models cannot be validated along coastal reaches lacking continuous INCOIS acoustic tide gauges.
9. **Wave Swell Separation:** Wave hazard modeling from local wind fields alone misses remotely generated Southern Ocean swells.
10. **Provisional Nature of Thresholds:** All operational hazard thresholds remain provisional until officially calibrated with national disaster management authorities.
11. **No Operational Authority for Research Hazards:** All newly designed modules are strictly advisory and possess zero decision-making authority in production.
