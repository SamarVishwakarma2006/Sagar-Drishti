# Sagar-Drishti — Multi-Hazard Data Inventory & Source Audit

**Document ID:** `SD-REPORT-2026-MULTI-HAZARD-DATA-INVENTORY`  
**Phase:** Pre-Implementation Research & Architecture Phase  
**Date:** September 13, 2026  
**Status:** **[AUDIT COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Research Specification & Data Provenance Archive  

---

## Executive Summary

Sagar-Drishti currently operates a validated machine learning pipeline for tropical cyclone genesis and event risk prediction (Production `v1.1.0` as the sole operational decision authority; candidate `V2.3 Model D` in shadow observation mode across 0d, 1d, 2d, and 3d lead times).

Expanding Sagar-Drishti into a comprehensive **Multi-Hazard Ocean & Coastal Intelligence Platform** requires a rigorous, physical-domain data audit. A primary failure mode in operational meteorological machine learning is the conflation of different physical processes (e.g., treating cyclone probability as a proxy for storm surge, extreme waves, or heavy precipitation). Each hazard exhibits unique spatial scales, governing hydrodynamic and thermodynamic equations, and distinct ground-truth observation networks.

This audit establishes a rigorous baseline inventory of all datasets currently resident in the repository versus datasets required for future expansion.

---

## 1. Taxonomic Classification of Data Sources

To prevent scientific mischaracterization and leakage, all data sources evaluated in this inventory are categorized into four explicit tiers:

```
DATA SOURCE TAXONOMY
│
├── 1. OBSERVATIONAL GROUND TRUTH
│   └── Direct in-situ sensor or surface synoptic observations (IMD AWS/manual stations,
│       INCOIS moored ocean buoys, coastal tide gauge networks, calibrated satellite scatterometry).
│
├── 2. REANALYSIS / REFERENCE DATA
│   └── Physically constrained numerical assimilation blending historical observations with
│       dynamical models (ECMWF ERA5, Copernicus GLORYS12V1). Reference standard; NOT pure observation.
│
├── 3. OPERATIONAL FORECAST DATA
│   └── Forward numerical weather and ocean prediction model runs (NCEP GFS, ECMWF IFS, NCMRWF).
│       Prone to spatial displacement, intensity bias, and model drift.
│
└── 4. DERIVED LABELS
    └── Algorithmic or heuristic products extracted from primary sources (e.g., cyclone genesis buffer
        windows, marine heatwave duration thresholds). Dependent on extraction logic.
```

> [!WARNING]
> **Scientific Integrity Requirement:** Reanalysis data (such as ERA5 or GLORYS12V1) must **never** be uncritically termed "ground truth". While reanalysis provides spatially continuous, dynamically balanced reference fields, it contains known systematic biases (e.g., underestimation of peak cyclonic wind gusts, smoothed coastal boundary layers, and convective precipitation parameterization errors). Observational ground truth must be sought wherever validation is conducted.

---

## 2. Currently Available Datasets (Resident in Repository)

The following datasets have been audited directly on disk within the `backend/data/` directory:

```
CURRENT REPOSITORY DATA ASSETS
backend/data/
├── historical/
│   ├── features_10yr.parquet                        (7,246 rows, 106 columns, Copernicus Ocean Reanalysis)
│   ├── labeled_features_10yr_clean.parquet          (7,246 rows, 110 columns, Clean Cyclone Genesis Labels)
│   ├── imd_tracks_2016_2026.parquet                 (2,544 synoptic fixes, IMD RSMC Best Track)
│   ├── imd_systems_2016_2026.json                   (117 tropical systems metadata, IMD RSMC)
│   └── aligned_ocean_era5_10yr_ablation.parquet     (2,572 rows, 224 aligned features for V2.3 ablation)
└── era5/
    ├── features_atmosphere_10yr_daily.parquet       (21,912 rows, 48 base spatial summary features)
    └── chunks/                                      (10 annual NetCDF archives: era5_pressure_2016.nc to 2026.nc)
```

### Dataset 1: Copernicus Global Ocean Physics Reanalysis (`features_10yr.parquet`)

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | Copernicus Global Ocean Physics Reanalysis (GLORYS12V1 Derivative) |
| **Provider** | Copernicus Marine Environment Monitoring Service (CMEMS) / Mercator Ocean International |
| **Data Tier** | `REANALYSIS / REFERENCE DATA` |
| **Primary Variables** | `sea_surface_temperature` ($SST$, °C), `sea_water_salinity` ($SSS$, PSU), `mixed_layer_depth` ($MLD$, m), `sea_surface_height_above_geoid` ($SLA$, m), `eastward_sea_water_velocity` ($u_{\text{curr}}$, m/s), `northward_sea_water_velocity` ($v_{\text{curr}}$, m/s) |
| **Derived Features** | 101 features per site: current value, 30-day baseline mean/std, absolute anomaly, z-score, 7-day rolling mean/std/min/max/trend, 14-day rolling mean/std/trend, 30-day rolling mean/std/trend |
| **Spatial Resolution** | Native GLORYS: 0.083° (~9 km) horizontal grid; Extracted at 12 representative offshore monitoring coordinates |
| **Temporal Resolution** | Daily averages |
| **Date Coverage** | July 23, 2016 to June 23, 2026 (3,623 calendar days, 10 complete years) |
| **Geographic Coverage** | North Indian Ocean: Arabian Sea (6 sites: AS1–AS6) and Bay of Bengal (6 sites: BOB1–BOB6) |
| **Operational Latency** | Reanalysis delayed mode: ~1 to 3 months; NRT analysis products: ~24 hours |
| **Licensing / Access** | Copernicus Open Access Policy (free for commercial and research use) |
| **Authoritative Status** | World reference standard for global ocean reanalysis |
| **Training Suitability** | Suitable for large-scale ocean thermodynamic and dynamic feature representation |
| **Validation Suitability** | Suitable for reference validation of large-scale SST anomalies; unsuitable for nearshore surf zone or estuarine dynamics |
| **Known Missingness** | Complete time series on disk; zero missing calendar days across the 10-year record |
| **Known Limitations** | Coarse coastal representation (~9 km resolution cannot resolve coastal estuaries, shallow lagoons, or bathymetric wave shoaling); smoothed boundary layer currents |

---

### Dataset 2: ECMWF ERA5 Atmospheric Reanalysis (`features_atmosphere_10yr_daily.parquet`)

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | ECMWF ERA5 Atmospheric Reanalysis (V2.1 Spatial Feature Engineering Derivative) |
| **Provider** | European Centre for Medium-Range Weather Forecasts (ECMWF) via Copernicus C3S |
| **Data Tier** | `REANALYSIS / REFERENCE DATA` |
| **Primary Variables** | Relative humidity ($r$ at 850, 700, 500 hPa), Divergence ($d$ at 850, 200 hPa), Vorticity ($\zeta$ at 850 hPa), Geopotential height ($z$ at 500 hPa), Temperature ($t$ at 850, 200 hPa), Horizontal winds ($u, v$ at 850, 200 hPa), Vertical Wind Shear ($VWS = \sqrt{(u_{200}-u_{850})^2 + (v_{200}-v_{850})^2}$), Mean Sea Level Pressure ($MSLP$), 10m horizontal winds ($u_{10}, v_{10}$) |
| **Spatial Resolution** | Native ERA5: 0.25° $\times$ 0.25° (~28 km) latitude/longitude grid across two synoptic ocean basins |
| **Basin Domains** | Arabian Sea: 0.0°N–26.0°N, 45.0°E–80.0°E; Bay of Bengal: 0.0°N–24.0°N, 78.0°E–100.0°E |
| **Temporal Resolution** | 6-hourly native analysis slices (00:00, 06:00, 12:00, 18:00 UTC) aggregated to daily spatial summaries with strict $18:00\text{ UTC}$ causal cutoff |
| **Date Coverage** | June 24, 2016 to June 23, 2026 (10 complete years) |
| **Geographic Coverage** | North Indian Ocean basin-wide atmospheric columns |
| **Operational Latency** | ERA5 final release: ~2 to 3 months; ERA5T (preliminary): ~5 days |
| **Licensing / Access** | Copernicus Open Access / ECMWF License terms |
| **Authoritative Status** | Global benchmark for atmospheric climate reanalysis |
| **Training Suitability** | Suitable for cyclone environment, synoptic wind, and thermodynamic moisture modeling |
| **Validation Suitability** | Reference data benchmark; must not replace station anemometer ground truth for coastal extreme wind |
| **Known Missingness** | Complete spatial coverage; 10 annual NetCDF files resident in repository; zero missing days |
| **Known Limitations** | Underestimates peak tropical cyclone core winds ($V_{\max}$) by 15–35% due to 28 km grid smoothing; convective precipitation is parameterized, leading to spatial dispersion of heavy rainfall cores |

---

### Dataset 3: IMD RSMC New Delhi Cyclone Best Track Dataset (`imd_tracks_2016_2026.parquet`)

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | IMD RSMC New Delhi Official Tropical Cyclone Best Track Dataset |
| **Provider** | Regional Specialised Meteorological Centre (RSMC) for Tropical Cyclones Over North Indian Ocean / India Meteorological Department (IMD) |
| **Data Tier** | `OBSERVATIONAL GROUND TRUTH` (Post-season reanalyzed official meteorological record) |
| **Primary Variables** | `system_id`, `year`, `storm_name`, `basin`, `date`, `time_utc`, `datetime_iso`, `latitude`, `longitude`, `grade` (D, DD, CS, SCS, VSCS, ESCS, SuCS), `category`, `max_wind_kts` ($V_{\max}$ 3-minute sustained), `central_pressure_hpa` ($P_c$), `pressure_drop_hpa` ($\Delta P$) |
| **Spatial Resolution** | Point fixes with 0.1° latitude and longitude precision |
| **Temporal Resolution** | 6-hourly synoptic fixes (00, 06, 12, 18 UTC); increased to 3-hourly fixes during active cyclonic stages near landfall |
| **Date Coverage** | Full archive 1982–2026; Resident active research slice: 2016–2026 (117 systems, 2,544 synoptic fixes) |
| **Geographic Coverage** | North Indian Ocean (Arabian Sea and Bay of Bengal) |
| **Operational Latency** | Post-season final publication: 3 to 6 months following cyclone season review |
| **Licensing / Access** | Official Government of India Meteorological Archive (Publicly accessible for scientific research) |
| **Authoritative Status** | **Sole official authority** for tropical cyclone classification, naming, track, and intensity in the North Indian Ocean under WMO mandate |
| **Training Suitability** | Highly suitable for Cyclone Genesis, Intensity ($V_{\max}$, $\Delta P$), and Track trajectory modeling |
| **Validation Suitability** | The definitive ground truth benchmark for cyclone track and intensity validation |
| **Known Missingness** | 100% of recognized cyclonic disturbances (Depression and above) are cataloged. Minor gaps in pressure drop ($\Delta P$) for weak depressions in open seas |
| **Known Limitations** | Intensity estimates over open ocean rely heavily on Dvorak satellite techniques prior to coastal radar tracking; wind speeds represent 3-minute sustained averages (requires conversion factor ~0.88 to compare with 1-minute US JTWC wind speeds) |

---

### Dataset 4: Sagar-Drishti Clean Binary Labeled Dataset (`labeled_features_10yr_clean.parquet`)

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | Sagar-Drishti 10-Year Clean Historical Cyclone Genesis Labels |
| **Provider** | Sagar-Drishti Internal Feature/Label Alignment Engine |
| **Data Tier** | `DERIVED LABELS` |
| **Primary Variables** | 101 ocean features + multi-horizon binary event targets: `label_0d`, `label_1d`, `label_2d`, `label_3d` |
| **Derivation Logic** | Active cyclone day mapped from IMD Best Track within 500 km radius of monitoring site; clean negative buffer enforces a mandatory 48-hour gap prior to genesis to prevent lookahead contamination |
| **Spatial Resolution** | 12 monitoring sites (6 Arabian Sea, 6 Bay of Bengal) |
| **Temporal Resolution** | Daily rows (7,246 total observations) |
| **Date Coverage** | July 23, 2016 to June 23, 2026 |
| **Authoritative Status** | Derived internal research dataset |
| **Training Suitability** | Frozen baseline for Cyclone Genesis classification only |
| **Validation Suitability** | Frozen baseline for Model D shadow evaluation |
| **Known Limitations** | **Strictly binary.** Contains zero intensity, track displacement, rainfall, or wave height information. Must NOT be used for any hazard other than cyclone genesis |

---

## 3. Required But Not Available Datasets (Data Gaps Audit)

To support the 10 candidate hazards, the following datasets are strictly required but currently absent from the repository. **None of these datasets exist in the local workspace.**

```
DATA GAPS REQUIRING ACQUISITION
┌──────────────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Hazard Module                        │ Missing Dataset Asset         │ Authoritative Target Provider │
├──────────────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Extreme Waves / Dangerous Sea State  │ Gridded Wave Spectra & Hs     │ Copernicus Marine WAVE / ERA5 │
│ Extreme Rainfall                     │ Gridded Daily Precipitation   │ IMD 0.25° Gridded / GPM IMERG │
│ Storm Surge                          │ Coastal Tide Gauge Water Level│ INCOIS / Survey of India      │
│ Storm Surge / Flooding               │ Coastal Bathymetry & DEM      │ GEBCO 15 arc-sec / SRTM 30m   │
│ Extreme Sea Level                    │ Harmonic Tidal Constituents   │ TPXO9 / FES2014 Global Tides  │
│ Coastal Flooding                     │ River Discharge / Drainage    │ Central Water Commission (CWC)│
└──────────────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### Missing Dataset 1: High-Resolution Gridded Wave Reanalysis & Forecast

- **Target Hazard:** Extreme Waves / Dangerous Sea State
- **Required Variables:** Significant wave height ($H_s$, meters), peak wave period ($T_p$, seconds), mean wave direction ($\theta_m$, degrees), swell wave height ($H_{s,\text{swell}}$), wind-sea significant wave height ($H_{s,\text{wind}}$).
- **Candidate Providers:**
  1. *Copernicus Marine Global Ocean Waves Reanalysis (WAVERYS)*: 0.2° grid, 3-hourly, MFWAM model driven by ERA5 winds, assimilated with altimeter and SAR wave spectra.
  2. *ECMWF ERA5 Ocean Waves (WAM)*: 0.5° grid (native ERA5 wave model), coupled directly to atmospheric forcing.
  3. *INCOIS Wave Watch III (WW3) Operational Archives*: Indian Ocean coastal domain, 0.1° resolution.
- **Observational Ground Truth Benchmark:** INCOIS Coastal and Deep-Sea Moored Wave Rider Buoy network (WRB).
- **Current Status:** `NOT AVAILABLE` in repository.
- **Feasibility Impact:** ML wave models cannot be trained or validated without acquiring either WAVERYS or ERA5 wave parameters.

### Missing Dataset 2: High-Resolution Gridded & Synoptic Precipitation

- **Target Hazard:** Extreme Rainfall
- **Required Variables:** Daily accumulated rainfall ($R_{24}$, mm/day), hourly rainfall rate ($R_{\text{hr}}$, mm/hr), heavy precipitation flag.
- **Candidate Providers:**
  1. *IMD High-Resolution Daily Gridded Rainfall*: 0.25° $\times$ 0.25° spatial grid based on ~3,500 daily reporting rain-gauge stations across mainland India (Pai et al., 2014). Authoritative national ground truth.
  2. *NASA/JAXA GPM IMERG (Global Precipitation Measurement)*: 0.1° $\times$ 0.1° half-hourly multi-satellite precipitation with gauge calibration. Essential for oceanic rainfall where surface gauges do not exist.
  3. *ERA5 Total Precipitation (`tp`)*: Continuous atmospheric reanalysis precipitation (accumulated liquid and frozen water).
- **Observational Ground Truth Benchmark:** IMD AWS/ARG surface rain gauges and coastal Doppler Weather Radars (DWR).
- **Current Status:** `NOT AVAILABLE` in repository (ERA5 chunk currently stores pressure-level thermodynamic variables, not surface precipitation).
- **Feasibility Impact:** Extreme rainfall classification cannot proceed until IMD gridded rainfall or GPM IMERG is ingested.

### Missing Dataset 3: Coastal Water Level & Sea Surface Height Observations

- **Target Hazard:** Storm Surge & Extreme Sea-Level Events
- **Required Variables:** Total water level ($h$, meters above chart datum or mean sea level), non-tidal storm surge residual ($\eta = h - \zeta_{\text{tide}}$), astronomical tide height ($\zeta_{\text{tide}}$).
- **Candidate Providers:**
  1. *INCOIS Real-Time Tide Gauge Network*: Continuous coastal acoustic/radar tide gauge records along the Indian coastline (Paradip, Visakhapatnam, Chennai, Tuticorin, Kochi, Mumbai, Porbandar).
  2. *Survey of India Geodetic Tide Gauge Archives*: Long-term authoritative baseline sea level records.
  3. *PSMSL (Permanent Service for Mean Sea Level)*: Global coastal gauge repository.
- **Current Status:** `NOT AVAILABLE` in repository.
- **Feasibility Impact:** Storm surge is defined strictly as the meteorological residual water level above astronomical tide. Without coastal tide gauge time series, ground-truth surge residuals cannot be labeled or evaluated.

### Missing Dataset 4: High-Resolution Coastal Bathymetry & Coastline Geometry

- **Target Hazard:** Storm Surge & Coastal Flooding
- **Required Variables:** Ocean bottom depth ($D$, meters), continental shelf slope, coastal shoreline geometry, barrier island configurations.
- **Candidate Providers:**
  1. *GEBCO 2024 Grid (General Bathymetric Chart of the Oceans)*: Global 15 arc-second (~450m) continuous terrain and bathymetry model.
  2. *ETOPO 2022 Global Relief Model*: NOAA 15 arc-second relief.
- **Current Status:** `NOT AVAILABLE` in repository.
- **Feasibility Impact:** Surge shoaling physics ($\eta \propto \frac{\tau_w L}{\rho g D}$) fundamentally depends on water depth $D$. Machine learning models cannot generalize surge heights along coastlines with wide shallow shelves (e.g., northern Bay of Bengal, Gulf of Khambhat) without bathymetric features.

### Missing Dataset 5: Coastal Digital Elevation Models (DEM) & Topography

- **Target Hazard:** Coastal Flooding (Inundation Extent and Depth)
- **Required Variables:** Bare-earth land elevation above mean sea level ($z$, meters), coastal levees, embankments, drainage channels.
- **Candidate Providers:**
  1. *Copernicus DEM (GLO-30)*: 30m resolution radar elevation data with high vertical accuracy.
  2. *NASA SRTM (Shuttle Radar Topography Mission)*: 30m global elevation model.
  3. *CartoDEM (ISRO Bhuvan)*: Indian territory high-resolution elevation model.
- **Current Status:** `NOT AVAILABLE` in repository.
- **Feasibility Impact:** Coastal flood modeling without high-resolution topography produces physically absurd, uncalibrated flood polygons. Coastal flooding remains `NOT CURRENTLY FEASIBLE` until a calibrated coastal DEM is integrated.

### Missing Dataset 6: Astronomical Tidal Constituents

- **Target Hazard:** Storm Surge & Extreme Sea-Level Events
- **Required Variables:** Harmonic tidal amplitude and phase constituents ($M_2, S_2, N_2, K_1, O_1, P_1$, etc.) for predicting astronomical tide $\zeta_{\text{tide}}(t)$.
- **Candidate Providers:**
  1. *TPXO9-atlas*: Global inverse tidal model (0.16° grid).
  2. *FES2014 (Finite Element Solution)*: Global tide database assimilating altimetry.
- **Current Status:** `NOT AVAILABLE` in repository.
- **Feasibility Impact:** Total water level $h(t) = \zeta_{\text{tide}}(t) + \eta_{\text{surge}}(t) + \eta_{\text{wave setup}}(t)$. Deconvolving surge from tide requires an astronomical tidal solver.

---

## 4. Summary Matrix: Available vs Missing Data Assets

| Hazard Module | Currently Available Data Assets | Missing Data Prerequisites | Readiness Status |
| :--- | :--- | :--- | :--- |
| **Cyclone Genesis** | `features_10yr.parquet`, `features_atmosphere_10yr_daily.parquet`, `labeled_features_10yr_clean.parquet` | None for current operational scope | **READY (In Shadow/Production)** |
| **Cyclone Intensity** | `imd_tracks_2016_2026.parquet` ($V_{\max}, \Delta P$), ERA5 atmospheric fields, Copernicus SST/MLD | Operational real-time NWP analysis feeds | **HIGH DATA READINESS (Research)** |
| **Cyclone Track** | `imd_tracks_2016_2026.parquet` (fixes), ERA5 steering flow (500/700 hPa winds) | High-resolution satellite cloud center fixes | **HIGH DATA READINESS (Research)** |
| **Extreme Wind** | ERA5 10m wind speeds ($u_{10}, v_{10}$), pressure gradients | IMD synoptic coastal anemometer records | **MEDIUM DATA READINESS (Reanalysis only)** |
| **Extreme Rainfall** | ERA5 atmospheric moisture/divergence (proxy only) | IMD 0.25° gridded rainfall, GPM IMERG | **DATA BLOCKED (Missing rainfall truth)** |
| **Extreme Waves** | ERA5 surface wind forcing ($u_{10}, v_{10}$) | Copernicus WAVERYS / ERA5 WAM ($H_s, T_p$) | **DATA BLOCKED (Missing wave spectra)** |
| **Storm Surge** | IMD cyclone parameters, ERA5 MSLP and wind | INCOIS coastal tide gauges, GEBCO bathymetry, TPXO tides | **DATA BLOCKED (Missing gauges & bathymetry)** |
| **Coastal Flooding** | None | 30m Coastal DEM, river discharge, flood extents | **DATA BLOCKED (Missing elevation & drainage)** |
| **Extreme Sea Level** | Copernicus SLA (satellite altimetry offshore) | Coastal tide gauges, astronomical tidal model | **DATA BLOCKED (Missing nearshore water levels)** |
| **Marine Heatwaves** | Copernicus 10-year SST, climatological baselines | High-resolution coastal satellite SST (OISST v2.1) | **HIGH DATA READINESS (Research)** |

---

## 5. Architectural Recommendations

1. **Do Not Download Huge External Datasets in this Phase:** In compliance with the pre-implementation mandate, zero bulk downloads of multi-gigabyte wave, rainfall, or DEM grids should occur during this architecture phase.
2. **Standardize Ingestion Interfaces:** Define declarative schemas and interface specifications in `backend/app/services/hazard_registry.py` that future pipeline phases can populate as datasets are incrementally licensed and verified.
3. **Preserve Baseline Dataset Cryptographic Hashes:** All existing Parquet and JSON assets must remain unmodified, serving as immutable baselines for Model D shadow observation.
