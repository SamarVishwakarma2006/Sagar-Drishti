# Sagar-Drishti — Multi-Hazard Feasibility Matrix & Capability Assessment

**Document ID:** `SD-REPORT-2026-MULTI-HAZARD-FEASIBILITY-MATRIX`  
**Phase:** Pre-Implementation Research & Architecture Phase  
**Date:** September 13, 2026  
**Status:** **[FEASIBILITY ANALYSIS COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Research Architecture & Feasibility Audit  

---

## Executive Summary

To expand Sagar-Drishti from a cyclone genesis prediction tool into a comprehensive **Multi-Hazard Ocean & Coastal Intelligence Platform**, this report systematically evaluates ten candidate ocean and coastal hazards.

In strict compliance with meteorological physics and project safety mandates:
1. **Target Independence:** No hazard is permitted to reuse the binary cyclone genesis target.
2. **Authoritative Label Rigor:** Labels are explicitly segregated into `OBSERVATIONAL GROUND TRUTH`, `REANALYSIS / REFERENCE DATA`, `OPERATIONAL FORECAST DATA`, and `DERIVED LABELS`.
3. **No Feasibility Inflation:** Feasibility ratings are grounded in actual repository contents and physical prerequisites.
4. **Provisional Prioritization:** Any preliminary ranking of next hazards is explicitly designated as `PROVISIONAL PRIORITIZATION`, subject to empirical data readiness.

---

## 1. Multi-Hazard Feasibility Matrix

| Hazard | Target Formulation | Label Source & Category | Data Available in Repo | Additional Data Required | ML Formulation | Difficulty | Feasibility Rating |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Tropical Cyclone Genesis** | $P(\text{event within } H) \in [0, 1]$ for $H \in \{0, 1, 2, 3\}\text{d}$ | `DERIVED LABELS`<br>(IMD RSMC Best Track with 48h negative buffer) | 101 ocean + 123 ERA5 features (224 features total) | None for current operational scope | Calibrated Binary Classification (Random Forest / GBDT) | **MEDIUM** | **HIGH**<br>*(Active in Production v1.1.0 & Shadow Model D)* |
| **2. Cyclone Intensity** | $V_{\max}(T+H)$ in knots (continuous) & $\Delta V_{24} \ge 30\text{ kt}$ (Rapid Intensification) | `OBSERVATIONAL GROUND TRUTH`<br>(IMD RSMC Official 3-min sustained $V_{\max}, \Delta P$) | IMD Best Track 2016–2026 fixes, ERA5 VWS, vorticity, RH, Copernicus SST/MLD | Operational real-time NWP analysis feeds | Multi-Task Regression ($V_{\max}, P_c$) + Binary Classification (RI) | **MEDIUM** | **HIGH**<br>*(Research Readiness High)* |
| **3. Cyclone Track** | Trajectory displacement $(\Delta \phi, \Delta \lambda)_{T+H}$ & Along/Cross-Track Error (km) | `OBSERVATIONAL GROUND TRUTH`<br>(IMD RSMC Official Best Track center coordinates) | IMD Best Track fixes 2016–2026, ERA5 500/700 hPa steering flow | Dynamic operational NWP steering forecasts, high-frequency fixes | Sequence-to-Sequence / Autoregressive Trajectory Regression | **HIGH** | **MEDIUM**<br>*(Statistical track degrades >24h without NWP)* |
| **4. Extreme Wind** | $P(U_{10} > 34\text{ kt} \text{ within } H)$<br>*(Provisional Threshold)* | `REANALYSIS / REFERENCE DATA`<br>(ERA5 10m wind speed maxima) | ERA5 hourly/daily 10m wind fields ($u_{10}, v_{10}$), pressure gradient | Coastal AWS/anemometer ground truth records | Calibrated Probabilistic Classification / Extreme Value Regression | **MEDIUM** | **HIGH** *(for open ocean)*<br>**MEDIUM** *(for coastal stations)* |
| **5. Extreme Rainfall** | $P(R_{24} \ge 64.5\text{ mm/d} \text{ within } H)$<br>*(Provisional Threshold)* | `OBSERVATIONAL GROUND TRUTH`<br>(IMD 0.25° Gridded Daily Rainfall / AWS gauges) | ERA5 moisture columns, divergence (proxies only) | IMD gridded daily rainfall (Pai et al.), NASA GPM IMERG | Spatio-temporal Quantile Regression / Calibrated Binary Classifier | **HIGH** | **MEDIUM**<br>*(Blocked by missing rainfall ground truth)* |
| **6. Extreme Waves / Dangerous Sea State** | $P(H_s \ge 4.0\text{ m} \text{ within } H)$<br>*(Provisional Threshold)* | `REANALYSIS / REFERENCE DATA`<br>(Copernicus WAVERYS / ERA5 WAM) | ERA5 surface wind vectors ($u_{10}, v_{10}$) | Copernicus Marine Wave Reanalysis (WAVERYS), INCOIS wave buoys | Physics-Informed Neural Network (PINN) / GBDT for $H_s, T_p$ | **HIGH** | **LOW**<br>*(Blocked by absence of wave spectra on disk)* |
| **7. Storm Surge** | Peak coastal water level residual $\eta_{\text{peak}} \ge 1.0\text{ m}$<br>*(Provisional Threshold)* | `OBSERVATIONAL GROUND TRUTH`<br>(INCOIS / Survey of India Coastal Tide Gauges) | IMD cyclone attributes, ERA5 MSLP & wind stress | GEBCO 15 arc-sec bathymetry, TPXO tidal constituents, coastal tide gauges | Hydrodynamic surrogate model / Gradient Boosted Surge Regressor | **VERY HIGH** | **LOW**<br>*(Blocked by absence of bathymetry & tide gauges)* |
| **8. Coastal Flooding** | Inundation extent ($\text{km}^2$) & flood depth $d \ge 0.5\text{ m}$<br>*(Provisional Threshold)* | `OBSERVATIONAL GROUND TRUTH` / `DERIVED`<br>(Sentinel-1 SAR flood extent & field reports) | None | 30m Coastal DEM, river discharge (CWC), estuarine hydrodynamic grids | Spatial Segmentation (U-Net) / Coupled Inundation Metamodel | **VERY HIGH** | **NOT CURRENTLY FEASIBLE**<br>*(Prerequisite geospatial assets absent)* |
| **9. Extreme Sea-Level Events** | Total water level exceedance $P(h \ge h_{\text{crit}})$ from tide + surge + steric SLA | `OBSERVATIONAL GROUND TRUTH`<br>(Continuous Coastal Tide Gauge time series) | Copernicus Sea Level Anomaly ($SLA$) offshore | Nearshore tide gauge network, astronomical harmonic tidal models | Extreme Value Distribution (GEV / POT) with ML covariate conditioning | **HIGH** | **LOW**<br>*(Altimetry is offshore only; nearshore gauges missing)* |
| **10. Marine / Ocean Anomalies (MHW)** | Binary Marine Heatwave $P(\text{MHW within } H)$: $SST \ge 90\text{th}\%$ile for $\ge 5$ consecutive days | `DERIVED LABELS`<br>(Calculated from daily Copernicus SST vs 30-year climatology) | Copernicus 10-year daily SST (`temp_current`, `temp_abs_anom`, `temp_zscore`) | 30-year daily climatological baseline (1982–2011) to freeze threshold | Calibrated Binary Classification / Thermal Stress Index Regressor | **LOW–MEDIUM** | **HIGH**<br>*(Data resident in repo; clear physical standard)* |

---

## 2. In-Depth Analysis of Individual Hazards

### Hazard 1: Tropical Cyclone Genesis (Current Operational Baseline)
- **Scientific Rationale:** Genesis risk represents the probability that an organized tropical depression will form or enter a designated ocean domain within lead time $H$.
- **Current Status:** Fully implemented. Production `v1.1.0` serves as the sole operational decision authority. Candidate `V2.3 Model D` (224 ocean-atmosphere features) is active in strict shadow observation.
- **Feasibility:** `HIGH` (Operational/Shadow).

### Hazard 2: Cyclone Intensity ($V_{\max}$, $\Delta P$, Rapid Intensification)
- **Scientific Rationale:** Cyclone intensity governs destructive potential at landfall. The governing physical drivers—Vertical Wind Shear (VWS), low-level cyclonic vorticity, mid-tropospheric relative humidity, Sea Surface Temperature ($SST$), and Ocean Heat Content ($OHC$)—are directly captured in Sagar-Drishti's combined 224-feature space.
- **Label Availability:** `imd_tracks_2016_2026.parquet` contains 2,544 official synoptic fixes with 3-minute sustained $V_{\max}$ and $\Delta P$ from IMD RSMC.
- **Feasibility:** `HIGH` (Scientific formulation is mature; authoritative ground-truth labels exist directly on disk).

### Hazard 3: Cyclone Track & Displacement
- **Scientific Rationale:** Predicting the future center location $(\phi, \lambda)$ at 24h, 48h, and 72h lead times.
- **Physical Bottleneck:** While storm fixes exist in the repository, cyclone steering past 24 hours is dominated by synoptic-scale upper-tropospheric steering flow (500–200 hPa). Pure statistical or ML track models uncoupled from dynamic NWP ensemble forecasts experience severe track error growth beyond 24–36 hours.
- **Feasibility:** `MEDIUM` (Requires coupled NWP ensemble steering flow inputs at inference time).

### Hazard 4: Extreme Wind (Synoptic & Monsoonal Wind Hazard)
- **Scientific Rationale:** Extreme surface winds cause major offshore marine disruptions, structural damage, and storm surges. Predicting exceedance of Gale-force wind ($>34\text{ kt}$ / 17.2 m/s, provisional threshold) provides critical multi-sector warnings.
- **Data Reality:** ERA5 provides continuous 10m wind fields ($u_{10}, v_{10}$). However, ERA5 is a reanalysis reference dataset, not direct ground truth, and systematically under-resolves peak gusts in convective squalls.
- **Feasibility:** `HIGH` for open-ocean synoptic gale risk; `MEDIUM` for nearshore gust ground truth.

### Hazard 5: Extreme Rainfall (Pluvial Weather Hazard)
- **Scientific Rationale:** Heavy monsoonal and cyclonic precipitation is a leading cause of loss of life and economic disruption in coastal India.
- **Data Reality:** The repository currently lacks gridded precipitation records. ERA5 relative humidity and divergence provide moisture convergence proxies, but cannot serve as training labels for rainfall accumulation. IMD 0.25° gridded daily rainfall (Pai et al.) or NASA GPM IMERG must be acquired.
- **Feasibility:** `MEDIUM` (Operationally critical, but currently data-blocked).

### Hazard 6: Extreme Waves / Dangerous Sea State
- **Scientific Rationale:** Significant wave height ($H_s$) exceeding 4.0 meters (WMO "Rough to Very Rough" sea state) represents extreme danger to shipping, fishing, and offshore infrastructure.
- **Physical Bottleneck:** Ocean waves consist of locally generated wind-sea and remotely generated swell. Swell propagating from the Southern Ocean across the equator into the Arabian Sea and Bay of Bengal cannot be predicted from local wind alone. Wave model output (Copernicus WAVERYS or ERA5 WAM) is strictly required.
- **Feasibility:** `LOW` (Blocked until wave reanalysis is ingested).

### Hazard 7: Storm Surge
- **Scientific Rationale:** Storm surge is the abnormal rise of water generated by a storm, over and above astronomical tide.
- **Physical Bottleneck:** Storm surge amplitude is non-linearly governed by coastal bathymetry (depth $D$), coastline curvature (bays, estuaries), astronomical tidal phase, and wind stress. Without high-resolution bathymetric grids (GEBCO) and coastal tide gauge validation data (INCOIS), ML surge modeling is ungrounded.
- **Feasibility:** `LOW` (Blocked until bathymetry and tide gauge archives are integrated).

### Hazard 8: Coastal Flooding (Compound Inundation)
- **Scientific Rationale:** Spatial inundation depth and extent across coastal zones.
- **Physical Bottleneck:** Coastal flooding is a compound hazard driven by astronomical tide, storm surge, wave setup, intense pluvial precipitation, and riverine discharge. Modeling inundation requires a high-resolution Digital Elevation Model (DEM $\le 30\text{m}$), hydraulic roughness, and river discharge data.
- **Feasibility:** `NOT CURRENTLY FEASIBLE` (Zero topographic or hydrological data resident in repository).

### Hazard 9: Extreme Sea-Level Events
- **Scientific Rationale:** Total coastal water level $h(t) \ge h_{\text{crit}}$ resulting from astronomical spring tides combined with seasonal thermal expansion and non-cyclonic wind setup.
- **Data Reality:** Altimetry SLA in `features_10yr.parquet` is offshore only ($\ge 20\text{ km}$ from coast) and cannot resolve coastal wave setup or astronomical tide.
- **Feasibility:** `LOW` (Requires coastal tide gauges).

### Hazard 10: Marine / Ocean Anomalies (Marine Heatwaves)
- **Scientific Rationale:** Prolonged abnormally warm sea surface temperatures devastate marine ecosystems, collapse fisheries, and provide high-enthalpy fuel for explosive cyclogenesis.
- **Data Reality:** 10-year daily Copernicus SST, rolling means, and standard deviations are already computed in `features_10yr.parquet`. The mathematical standard (Hobday et al., 2016: $SST \ge 90\text{th}$ percentile for $\ge 5$ consecutive days) is mathematically clear and directly computable.
- **Feasibility:** `HIGH` (Highest data and label readiness among non-cyclone marine hazards).

---

## 3. Provisional Prioritization of Next Hazards

In strict accordance with the mandatory pre-implementation instructions:

> [!IMPORTANT]
> **PROVISIONAL PRIORITIZATION NOTICE:** The following ranking represents a **provisional research prioritization** based on the data audit and physical feasibility. It is **NOT** a decision to initiate model training. Final development authorization requires formal scientific review and dedicated data acquisition gates.

```
PROVISIONAL RESEARCH PRIORITIZATION
┌───────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Rank  │ Hazard Candidate                 │ Primary Justification                                       │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1     │ Cyclone Intensity                │ Authoritative ground truth (IMD RSMC Best Track) in repo;   │
│       │ (Vmax, ΔP, Rapid Intensification)│ 224 thermodynamic/dynamic features already aligned.         │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2     │ Extreme Wind                     │ ERA5 10m wind fields resident; well-understood synoptic     │
│       │ (Open-Ocean / Basin Gale Risk)   │ physics; direct multi-sector utility for marine safety.     │
├───────┼──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3     │ Marine Heatwaves / Anomalies     │ 10-year daily SST and anomaly baselines resident on disk;   │
│       │ (SST Percentile Exceedance)      │ established international standard (Hobday et al., 2016).   │
└───────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### Why These 3 Rank Highest:
1. **Direct Data Availability:** All three hazards utilize datasets that are already curated, verified, and resident in the repository (`imd_tracks_2016_2026.parquet`, `features_atmosphere_10yr_daily.parquet`, and `features_10yr.parquet`). Zero speculative external downloads are required to conduct preliminary target verification.
2. **Authoritative Label Quality:** Cyclone intensity relies on IMD RSMC official post-season best tracks; Marine Heatwaves rely on established international definitions applied to Copernicus satellite-assimilated SST.
3. **Physical Coupling with Sagar-Drishti's Core:** All three hazards directly leverage Sagar-Drishti's existing ocean-atmosphere state engine (SST, MLD, VWS, vorticity, and pressure fields).

### Why Other Hazards Are Deferred:
- **Extreme Rainfall:** Blocked until IMD 0.25° gridded daily rainfall is acquired.
- **Extreme Waves:** Blocked until Copernicus Marine WAVERYS or ERA5 wave spectra are acquired.
- **Storm Surge & Extreme Sea Level:** Blocked until coastal tide gauges (INCOIS) and coastal bathymetry (GEBCO) are integrated.
- **Coastal Flooding:** Completely blocked due to absence of coastal Digital Elevation Models (DEM) and river discharge networks.
