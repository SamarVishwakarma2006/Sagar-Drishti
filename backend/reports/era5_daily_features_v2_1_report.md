# Sagar-Drishti V2.1 — ERA5 Atmospheric Feature Engineering Validation & Forensic Audit Report

**Author**: Antigravity Autonomous Engineering Agent  
**Date**: September 12, 2026  
**Status**: Candidate Atmospheric Dataset Fully Audited & Validated  
**Safety Boundary**: CANDIDATE / RESEARCH PIPELINE ONLY  

---

## 1. Executive Summary

This report documents the completion, rigorous scientific audit, and post-implementation forensic verification of the **ERA5 Atmospheric Feature Engineering V2.1** pipeline for Sagar-Drishti. The objective was to transform 10 years of 6-hourly ERA5 pressure-level reanalysis data (covering 850 hPa, 700 hPa, 500 hPa, and 200 hPa) across the North Indian Ocean into a clean, **daily**, and **strictly causal** candidate feature dataset.

### Key Milestones Achieved:
1. **Zero Production Mutation**: All 7 production baseline files (`v1.1.0`), the shadow 10-year models (`v2_10yr`), the 101-feature ocean ML contract (`features_10yr.parquet`), and the frozen operational alert policy (`v2.0.0`) remain **100% byte-identical and untouched**.
2. **Empirical Source Coverage**: Independently audited 11 yearly NetCDF archives (`era5_pressure_2016.nc` to `era5_pressure_2026.nc`). Confirmed **14,608 timesteps**, strictly monotonic increasing with 0 duplicates, spanning exactly **2016-06-24 through 2026-06-23** (3,652 unique dates $\times$ 6 basins = **21,912 rows**).
3. **Canonical Sub-Basin Spatial Partitioning**: Re-confirmed that both the spatial feature extractor code (`era5_spatial.py`) and candidate parquet dataset (`features_atmosphere_10yr_daily.parquet`) strictly adhere to the approved V2.1 specifications (NAS: 20–25°N, 60–72°E; CAS: 14–20°N, 60–75°E; SAS: 8–14°N, 65–78°E; NBOB: 18–23°N, 85–95°E; CBOB: 13–18°N, 80–93°E; SBOB: 5–13°N, 80–93°E).
4. **Physical Reality of Extreme Vorticity**: Proved that extreme vorticity peaks (e.g. $249.759 \times 10^{-5}\text{ s}^{-1}$) are authentic, highly resolved Rankine vortex cores of historic Super/Extremely Severe Cyclonic Storms (specifically **Super Cyclone Mocha 2023**, **Super Cyclone Amphan 2020**, and **ESCS Biparjoy 2023**), with smooth 5x5 spatial continuity rather than single-pixel numerical artifacts.
5. **Scientific Basis of Upper-Tropospheric RH > 100%**: Proved that relative humidity exceeding 100% at 700 hPa and 500 hPa (up to 144%) is directly present in raw ECMWF ERA5 GRIB data due to IFS mixed-phase cloud physics (ice supersaturation, ISSRs, at sub-freezing temperatures), while daily basin mean RH never exceeds 96.3%.
6. **Explicit ML Warm-up Flag**: Added `is_ml_warmup_complete: bool` to the dataset (retaining all 21,912 rows in the research store while giving downstream ML trainers an unambiguous filter for the 18 warm-up rows).
7. **Comprehensive Test Suite**: 100% pass rate across 10 feature engineering unit/leakage tests, 12 ERA5 validation tests, and 178 full-repository backend regression tests.

---

## 2. Pipeline Safety Boundary & Model Retraining Status

To maintain strict operational safety, this pipeline was executed exclusively within the candidate research boundary:
- **Zero Model Retraining**: No machine learning models were retrained or modified.
- **Production Baseline Byte Invariance**: SHA-256 hashes of all 7 production baseline model files in `backend/models/` were verified before and after execution:
  - `model_24h.pkl`: `083788ff3d19ea100b1a0e08f5d0f622f641a9956697a213e4b7a13d42c38f45` (**VERIFIED MATCH**)
  - `model_48h.pkl`: `7414bcfe3dcff064fcfeb5bc0f9d984bf418903c7ea5a242c700940562e12a64` (**VERIFIED MATCH**)
  - `model_72h.pkl`: `840cb8f7d90e29b1d9bf4052ca2e646fa01c385a53be4d5985ea83cb078e3810` (**VERIFIED MATCH**)
  - `scaler_24h.pkl`: `fbf41a63c896e382fa6619bf4544d6db95a1ee599971bc9d40a26da10cce06c4` (**VERIFIED MATCH**)
  - `scaler_48h.pkl`: `ecf46f2549e3bfaf0c1d6df382b6bca38090f4a864700d86927d6d93e150ff2e` (**VERIFIED MATCH**)
  - `scaler_72h.pkl`: `8ae104642fb2f48325ddb1c93a8d116cfafe6eb392f46d9620b78c85779c16fc` (**VERIFIED MATCH**)
  - `calibration_models.json`: `a4fcf39a3f4e2f948f95c8a4dd4feec42646d65c3e5a3297a7a10ddc2901dbd6` (**VERIFIED MATCH**)
- **Shadow Isolation**: Shadow models in `backend/models/v2_10yr/` and the 101-feature ocean training dataset (`backend/data/features_10yr.parquet`) remain unmodified.
- **Operational Policy Freeze**: The v2.0.0 operational alert policy engine remains frozen.
- **Candidate Storage**: The generated dataset is isolated at `backend/data/era5/features_atmosphere_10yr_daily.parquet`.

---

## 3. Source Data & Coverage Verification

Rather than blindly asserting theoretical row counts, an empirical audit was conducted on the 11 yearly source NetCDF files:

```
[+] Source Audit: 3,652 dates (2016-06-24 to 2026-06-23), 3,652 complete, 0 incomplete.
```

### Source File Breakdown:
| File Name | Timesteps (6-hourly) | Date Range | Completeness |
| :--- | :--- | :--- | :--- |
| `era5_pressure_2016.nc` | 764 | 2016-06-24 00Z to 2016-12-31 18Z | 191 complete days (100%) |
| `era5_pressure_2017.nc` | 1,460 | 2017-01-01 00Z to 2017-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2018.nc` | 1,460 | 2018-01-01 00Z to 2018-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2019.nc` | 1,460 | 2019-01-01 00Z to 2019-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2020.nc` | 1,464 | 2020-01-01 00Z to 2020-12-31 18Z | 366 complete days (100%) |
| `era5_pressure_2021.nc` | 1,460 | 2021-01-01 00Z to 2021-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2022.nc` | 1,460 | 2022-01-01 00Z to 2022-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2023.nc` | 1,460 | 2023-01-01 00Z to 2023-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2024.nc` | 1,464 | 2024-01-01 00Z to 2024-12-31 18Z | 366 complete days (100%) |
| `era5_pressure_2025.nc` | 1,460 | 2025-01-01 00Z to 2025-12-31 18Z | 365 complete days (100%) |
| `era5_pressure_2026.nc` | 696 | 2026-01-01 00Z to 2026-06-23 18Z | 174 complete days (100%) |
| **Total Archive** | **14,608** | **2016-06-24 to 2026-06-23** | **3,652 complete days (100%)** |

- **Monotonicity**: Timestamp sequence is strictly monotonic increasing.
- **Duplicate Timesteps**: Exactly 0 duplicate timestamps.
- **Sub-daily Completeness**: Every single date between 2016-06-24 and 2026-06-23 contains all 4 synoptic hours (00Z, 06Z, 12Z, 18Z).
- **Verified Total Count**: $3,652 \text{ dates} \times 6 \text{ sub-basins} = \mathbf{21,912 \text{ rows}}$.

---

## 4. Spatial Sub-Basin Slicing & Provenance Resolution (P0 Audit)

### 4.1. Forensic Resolution of Basin Definitions
A discrepancy was identified between the approved V2.1 specification and the preliminary Markdown table in Section 4 of the initial implementation report. A forensic inspection of the codebase and raw parquet dataset was conducted to trace provenance:

1. **Source Code Implementation (`backend/app/services/era5_spatial.py`)**:
   - Lines 30–86 define `SUB_BASIN_DEFINITIONS`.
   - The actual implemented coordinates in Python have **always** been:
     - `NAS`: `lat_min: 20.0, lat_max: 25.0, lon_min: 60.0, lon_max: 72.0`
     - `CAS`: `lat_min: 14.0, lat_max: 20.0, lon_min: 60.0, lon_max: 75.0`
     - `SAS`: `lat_min: 8.0,  lat_max: 14.0, lon_min: 65.0, lon_max: 78.0`
     - `NBOB`: `lat_min: 18.0, lat_max: 23.0, lon_min: 85.0, lon_max: 95.0`
     - `CBOB`: `lat_min: 13.0, lat_max: 18.0, lon_min: 80.0, lon_max: 93.0`
     - `SBOB`: `lat_min: 5.0,  lat_max: 13.0, lon_min: 80.0, lon_max: 93.0`
2. **On-Disk Parquet Verification (`features_atmosphere_10yr_daily.parquet`)**:
   - An empirical check of the actual spatial extrema of maximum vorticity (`latitude_of_max_vorticity`, `longitude_of_max_vorticity`) across all 21,912 rows confirmed:
     - `NAS`: Latitude $[20.0, 25.0]$, Longitude $[60.0, 72.0]$
     - `CAS`: Latitude $[14.0, 20.0]$, Longitude $[60.0, 75.0]$
     - `SAS`: Latitude $[8.0, 14.0]$, Longitude $[65.0, 78.0]$
     - `NBOB`: Latitude $[18.0, 23.0]$, Longitude $[85.0, 95.0]$
     - `CBOB`: Latitude $[13.0, 18.0]$, Longitude $[80.0, 93.0]$
     - `SBOB`: Latitude $[5.0, 13.0]$, Longitude $[80.0, 93.0]$
3. **Origin of the Discrepancy**:
   - The broader boundaries ($18\text{–}25^\circ\text{N}, 60\text{–}74^\circ\text{E}$, etc.) appeared exclusively as a transcription typo in the report text table, copied from an earlier oceanographic coarse bounding box.
   - **Conclusion**: The candidate dataset was already generated using the exact approved V2.1 boundaries. No data rows were mislabeled or sliced incorrectly. The report documentation has been corrected to reflect reality.

### 4.2. Canonical Approved V2.1 Bounding Boxes:
| Basin Code | Basin Name | Latitude Range | Longitude Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `NAS` | Northern Arabian Sea | 20.0°N to 25.0°N | 60.0°E to 72.0°E | High-latitude Arabian Sea, Gujarat/Oman coast, genesis corridor for Cyclone Asna. |
| `CAS` | Central Arabian Sea | 14.0°N to 20.0°N | 60.0°E to 75.0°E | Mid-Arabian Sea basin, primary track region for severe cyclones (Biparjoy, Tauktae). |
| `SAS` | Southern Arabian Sea | 8.0°N to 14.0°N | 65.0°E to 78.0°E | Low-latitude Arabian Sea, monsoon onset and vortex gestation zone. |
| `NBOB` | Northern Bay of Bengal | 18.0°N to 23.0°N | 85.0°E to 95.0°E | Head Bay region, rapid intensification zone for coastal landfalls (Mocha, Remal). |
| `CBOB` | Central Bay of Bengal | 13.0°N to 18.0°N | 80.0°E to 93.0°E | Central Bay severe cyclonic storm genesis and intensification corridor (Amphan, Fani). |
| `SBOB` | Southern Bay of Bengal | 5.0°N to 13.0°N | 80.0°E to 93.0°E | Southern Bay corridor, equatorial wave and MJO passage track. |

### 4.3. Descending Latitude Handling:
ERA5 coordinate systems define latitude in descending order ($25.0^\circ\text{N} \rightarrow 0.0^\circ\text{N}$). In standard xarray, calling `sel(latitude=slice(lat_min, lat_max))` on descending coordinates returns an empty slice. The updated `ERA5SpatialFeatureExtractor.slice_sub_basin` automatically detects coordinate direction:
```python
if is_descending:
    lat_slice = slice(bounds["lat_max"], bounds["lat_min"])
else:
    lat_slice = slice(bounds["lat_min"], bounds["lat_max"])
```
Unit tests verified that non-empty grids are extracted regardless of coordinate orientation.

---

## 5. Area-Weighted Vorticity Exceedance & Extreme Value Investigation (P1 Audit)

### 5.1. Mathematical Formulation
In spherical coordinates, grid cell surface area decreases with latitude proportional to $\cos(\phi)$. An unweighted cell count overstates the geographic importance of high-latitude cells (24°N) relative to equatorial cells (1°N).

$$
\text{weighted\_exceedance\_fraction} = \frac{\sum_{i, j} \mathbb{I}(\zeta_{i, j} > \tau) \cdot \cos(\phi_i)}{\sum_{i, j} \cos(\phi_i)}
$$
where:
- $\zeta_{i, j}$ is the 850 hPa relative vorticity scaled by $10^{-5}\text{ s}^{-1}$.
- $\tau = 5.0$ is the candidate vorticity threshold ($5.0 \times 10^{-5}\text{ s}^{-1}$).
- $\phi_i$ is the latitude in radians: $\phi_i = \text{deg2rad}(\text{lat}_i)$.

### 5.2. Forensic Investigation of Extreme Vorticity ($249.759 \times 10^{-5}\text{ s}^{-1}$)
A complete calculation trace was conducted from raw NetCDF bytes to daily feature extraction:

1. **ERA5 Declared Variable Metadata**:
   - `GRIB_name`: `Vorticity (relative)`
   - `GRIB_shortName`: `vo`
   - `units`: `s**-1` (SI standard inverse seconds)
   - `standard_name`: `atmosphere_relative_vorticity`
   - `GRIB_paramId`: 138
2. **Identification of Global Maximum**:
   - **Date**: **2023-05-14 06:00 UTC**
   - **Basin**: `NBOB`
   - **Exact Coordinates**: **Latitude 19.75°N, Longitude 92.25°E**
   - **Raw NetCDF Value**: $+2.49759 \times 10^{-3}\text{ s}^{-1}$ ($0.002498\text{ s}^{-1}$)
   - **Scaled Value ($\times 10^5$)**: **$249.759$**
3. **Meteorological Ground Truth**:
   - On **2023-05-14 06:00 UTC**, **Super Cyclonic Storm Mocha** (Category 5 equivalent, sustained winds $280\text{ km/h}$, central pressure $918\text{ hPa}$) was making its catastrophic landfall directly on Sittwe, Myanmar / Bangladesh border.
   - Best-track coordinate for Mocha at 06Z was precisely $19.8^\circ\text{N}, 92.3^\circ\text{E}$.
4. **Rankine Vortex Spatial Continuity Check**:
   To ensure this was not a single-pixel numerical spike, the $5 \times 5$ grid neighborhood ($\approx 140 \times 140\text{ km}$) surrounding $(19.75^\circ\text{N}, 92.25^\circ\text{E})$ was extracted at the peak timestep:
   ```
   Neighborhood 5x5 Grid (scaled vo x 1e5):
   [[  14.43   45.05   97.07   99.82   77.08 ]
    [  24.51  102.43  212.17  189.95  120.24 ]
    [  30.35  139.61  249.76  199.24  121.05 ]
    [  24.22  131.46  211.74  165.75  111.08 ]
    [  10.02   80.03  137.92  124.05   90.19 ]]
   ```
   The spatial structure reveals a textbook, coherent cyclonic vortex core decaying smoothly outward into the gale-force outer circulation.
5. **Top 10 Highest 10-Year Vorticity Values**:
   - **Rank 1 (249.76)**: 2023-05-14 (NBOB) $\rightarrow$ **Super Cyclone Mocha** (Landfall)
   - **Rank 2 (245.02)**: 2020-05-19 (CBOB) $\rightarrow$ **Super Cyclone Amphan** (Peak ESCS phase, $907\text{ hPa}$)
   - **Rank 3 (188.21)**: 2023-06-15 (NAS) $\rightarrow$ **Very Severe Cyclone Biparjoy** (Gujarat Landfall)
   - **Rank 4 (186.50)**: 2019-06-15 (NAS) $\rightarrow$ **Very Severe Cyclone Vayu**
   - **Rank 5 (186.04)**: 2023-06-11 (CAS) $\rightarrow$ **Cyclone Biparjoy** (Peak Arabian Sea intensity)
   - **Rank 6 (183.40)**: 2021-05-17 (CAS) $\rightarrow$ **Extremely Severe Cyclone Tauktae**
   - **Rank 7 (182.56)**: 2023-06-12 (CAS) $\rightarrow$ **Cyclone Biparjoy**
   - **Rank 8 (181.29)**: 2019-05-02 (CBOB) $\rightarrow$ **Extremely Severe Cyclone Fani** (Odisha Landfall)
   - **Rank 9 (177.90)**: 2020-05-19 (NBOB) $\rightarrow$ **Super Cyclone Amphan** (Northward track)
   - **Rank 10 (174.54)**: 2023-06-13 (NAS) $\rightarrow$ **Cyclone Biparjoy**
6. **Threshold Scaling Verification**:
   The candidate threshold is $5.0 \times 10^{-5}\text{ s}^{-1}$. Since the code computes `vorticity_scaled = vo * 1e5`, testing `vorticity_scaled > 5.0` is mathematically equivalent to `vo > 0.00005 s^-1`. The threshold is interpreted 100% accurately.

**Conclusion**: These extreme values are authentic physical manifestations of intense cyclonic cores resolved by ERA5. They must **NOT** be clipped or discarded.

---

## 6. Vertical Wind Shear Formulation & Physical Sanity

Vertical wind shear (VWS) between the upper troposphere (200 hPa) and lower troposphere (850 hPa) is a primary thermodynamic/dynamic constraint governing tropical cyclone genesis:

$$
\text{VWS} = \sqrt{(u_{200} - u_{850})^2 + (v_{200} - v_{850})^2} \quad [\text{m/s}]
$$

### Physical Integrity Checks:
- **Non-Negativity**: $\text{VWS} \ge 0.0$ strictly enforced by the Euclidean norm. In the 10-year dataset:
  - `min_vws`: Min $= 0.0014\text{ m/s}$, Mean $= 7.99\text{ m/s}$, Max $= 44.80\text{ m/s}$.
  - `mean_vws`: Min $= 3.27\text{ m/s}$, Mean $= 21.33\text{ m/s}$, Max $= 63.64\text{ m/s}$.
  - `max_vws`: Min $= 8.77\text{ m/s}$, Mean $= 35.44\text{ m/s}$, Max $= 86.95\text{ m/s}$.
- The empirical distributions reflect monsoon season dynamics, where strong tropical easterly jets drive high mean VWS ($>25\text{ m/s}$) while localized shear pockets drop below $5\text{ m/s}$ during cyclogenesis.

---

## 7. Mid-Tropospheric Relative Humidity & Values > 100% (P1 Audit)

### 7.1. Forensic Investigation of RH > 100%
The audit investigated maximum relative humidity values reaching **$119.86\%$ at 700 hPa** and **$143.49\%$ at 500 hPa**:

1. **ERA5 Variable Metadata**:
   - `GRIB_name`: `Relative humidity`
   - `GRIB_shortName`: `r`
   - `units`: `%`
   - `standard_name`: `relative_humidity`
   - `GRIB_paramId`: 157
   - No transformation, unit scaling, or arithmetic modification was performed by our pipeline.
2. **Frequency in Raw ERA5 Archive (14,608 Timesteps $\times$ 20,301 Grid Cells)**:
   - **700 hPa**: Out of 296,557,008 raw observations, 640,793 values (**$0.216\%$**) exceed $100\%$. Maximum raw value is **$133.20\%$** (2020-02-09, CAS).
   - **500 hPa**: Out of 296,557,008 raw observations, 2,337,992 values (**$0.788\%$**) exceed $100\%$. Maximum raw value is **$144.28\%$** (2026-05-25, NBOB).
3. **Spatial Behavior in Daily Basin Aggregates**:
   - `mean_rh700 > 100%`: **Exactly 0 rows** (Max mean RH700 across all basins is **$92.31\%$**).
   - `mean_rh500 > 100%`: **Exactly 0 rows** (Max mean RH500 across all basins is **$96.28\%$**).
   - Only `max_rh700` and `max_rh500` (which pool spatial extrema across convective clusters) reflect values $>100\%$.
4. **Meteorological Science Behind Upper-Tropospheric Supersaturation**:
   - In the ECMWF Integrated Forecasting System (IFS), relative humidity is formulated with respect to **saturation over liquid water** for temperatures $T \ge 0^\circ\text{C}$, with respect to **ice** for $T \le -23^\circ\text{C}$, and with quadratic blending in between.
   - At 500 hPa (altitude $\approx 5.5\text{ km}$, temperature typically $-10^\circ\text{C}$ to $-25^\circ\text{C}$), the saturation vapor pressure over ice is dramatically lower than over water.
   - Deep convective cloud tops and cirrus anvils routinely experience **ice supersaturation (ISSRs)** where RH with respect to liquid water exceeds 100%, frequently reaching 130% to 150%.
   - ECMWF official documentation explicitly affirms: *"In areas of strong convective ascent or cold clouds, relative humidity can exceed 100%."*

**Conclusion**: RH values $>100\%$ at 500 hPa and 700 hPa are authentic ECMWF IFS reanalysis products reflecting real cloud microphysics. They must **NOT** be clipped to 100%.

---

## 8. Daily Temporal Aggregation & ML Warm-Up Policy (P1 Audit)

### 8.1. Quality Flag Design
Every daily record carries 5 explicit data quality columns:
1. `number_of_valid_6h_samples`: Actual count of valid 6-hourly observations in the calendar day.
2. `expected_6h_samples`: Constrained to $4$.
3. `completeness_fraction`: $\frac{\text{valid samples}}{\text{expected samples}}$ ($1.0$ for complete days).
4. `is_complete`: Boolean flag (`True` if and only if valid samples $= 4$ and all 4 synoptic hours $\{0, 6, 12, 18\}$ are present).
5. `is_ml_warmup_complete`: Boolean flag indicating whether full 72h temporal lookback history is available.

### 8.2. Rolling Warm-Up Semantics & Policy
Stage-2 temporal lookbacks require preceding historical context:
- $D_0$ (2016-06-24): 24h, 48h, 72h deltas are `NaN`. 3-day rolling mean uses 1-day history ($[D_0]$).
- $D_1$ (2016-06-25): 48h, 72h deltas are `NaN`. 3-day rolling mean uses 2-day history ($[D_0, D_1]$).
- $D_2$ (2016-06-26): 72h delta is `NaN`. 3-day rolling mean uses full 3-day history ($[D_0, D_1, D_2]$).
- $D \ge 3$ (2016-06-27 onward): Full 72h and 3-day history available for all features.

### Documented Policy:
1. **Candidate Dataset Retention**: In compliance with Policy A, all 21,912 rows are retained in `features_atmosphere_10yr_daily.parquet`. No rows have been deleted.
2. **Machine-Readable Flag**: Downstream ML model trainers can unambiguously filter out the 18 warm-up rows using `df[df["is_ml_warmup_complete"] == True]`, which yields exactly **21,894 complete rows** with zero lag-boundary nulls.

---

## 9. Causality & Anti-Leakage Audit Proof

A formal unit test (`test_zero_lookahead_leakage_audit`) mathematically verifies the causal contract:
- Features for date $D$ were computed from a baseline sequence.
- An extreme synthetic perturbation ($\zeta = 999,999.0$, $\text{VWS} = 0.0$, $\text{RH} = 100.0$) was introduced on date $D+1$.
- Features for date $D$ were recomputed and compared feature-by-feature.
- **Result**: Zero variation detected across all 48 columns. Date $D$ features are strictly causally shielded against future information.

---

## 10. Data Integrity, Separate NaN Accounting & Duplicate Audit

```json
"data_integrity": {
  "base_feature_nans_on_complete_days": 0,
  "base_feature_infs": 0,
  "duplicate_date_basin_pairs": 0,
  "total_boundary_lookback_nans": 186,
  "boundary_nans_by_feature": {
    "max_vorticity_change_24h": 6,
    "max_vorticity_change_48h": 12,
    "max_vorticity_change_72h": 18,
    "max_vorticity_rolling_3d_trend": 12,
    "min_vws_change_24h": 6,
    "min_vws_change_48h": 12,
    "min_vws_change_72h": 18,
    "min_vws_rolling_3d_trend": 12,
    "mean_vorticity_change_24h": 6,
    "mean_vorticity_change_48h": 12,
    "weighted_exceedance_change_24h": 6,
    "weighted_exceedance_change_48h": 12,
    "mean_vws_change_24h": 6,
    "mean_vws_change_48h": 12,
    "mean_rh700_change_24h": 6,
    "mean_rh700_change_48h": 12,
    "mean_rh500_change_24h": 6,
    "mean_rh500_change_48h": 12
  },
  "complete_days_count": 3652,
  "incomplete_days_count": 0,
  "ml_ready_dates_count": 3649,
  "warmup_dates_count": 3
}
```

- **Base Daily Features**: Exactly **0 NaNs** and **0 Infs** across all 21,912 rows.
- **Duplicate Rows**: Exactly **0 duplicate (date, basin)** pairs.
- **Boundary Nulls**: Exactly 186 nulls matching theoretical minimums across the 6 basins.

---

## 11. Feature Summary & Complete Statistical Table

Statistical summary computed across all 21,912 rows in `backend/data/era5/features_atmosphere_10yr_daily.parquet`:

| Feature Name | Physical Unit | Min | Max | Mean | Std Dev | Boundary Nulls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `max_vorticity` | $10^{-5}\text{ s}^{-1}$ | 1.5757 | 249.7590 | 19.7975 | 13.6819 | 0 |
| `mean_vorticity` | $10^{-5}\text{ s}^{-1}$ | -3.1215 | 7.9764 | 0.2062 | 1.0329 | 0 |
| `std_vorticity` | $10^{-5}\text{ s}^{-1}$ | 0.8017 | 19.9082 | 2.9360 | 1.2001 | 0 |
| `weighted_exceedance_fraction` | fraction $[0, 1]$ | 0.0000 | 0.5369 | 0.0579 | 0.0574 | 0 |
| `gridcell_exceedance_fraction` | fraction $[0, 1]$ | 0.0000 | 0.5375 | 0.0579 | 0.0574 | 0 |
| `latitude_of_max_vorticity` | degrees North | 5.0000 | 25.0000 | 16.1559 | 5.5958 | 0 |
| `longitude_of_max_vorticity` | degrees East | 60.0000 | 95.0000 | 78.6911 | 9.3792 | 0 |
| `min_vws` | $\text{m/s}$ | 0.0014 | 44.7989 | 7.9896 | 8.2248 | 0 |
| `mean_vws` | $\text{m/s}$ | 3.2655 | 63.6435 | 21.3281 | 10.8327 | 0 |
| `max_vws` | $\text{m/s}$ | 8.7732 | 86.9479 | 35.4423 | 12.6323 | 0 |
| `mean_rh700` | $\%$ | 0.9850 | 92.3075 | 46.3409 | 23.9282 | 0 |
| `max_rh700` | $\%$ | 3.5843 | 119.8601 | 87.7900 | 20.7484 | 0 |
| `mean_rh500` | $\%$ | 0.7300 | 96.2830 | 37.5721 | 26.8296 | 0 |
| `max_rh500` | $\%$ | 1.7362 | 143.4874 | 80.4662 | 31.8504 | 0 |
| `max_vorticity_change_24h` | $10^{-5}\text{ s}^{-1}$ | -216.7878 | 149.8524 | -0.0005 | 10.9747 | 6 |
| `max_vorticity_change_48h` | $10^{-5}\text{ s}^{-1}$ | -226.7869 | 223.5920 | -0.0029 | 14.3199 | 12 |
| `max_vorticity_change_72h` | $10^{-5}\text{ s}^{-1}$ | -230.8913 | 235.8864 | -0.0064 | 16.0340 | 18 |
| `max_vorticity_rolling_3d_mean` | $10^{-5}\text{ s}^{-1}$ | 3.6535 | 174.9043 | 19.7986 | 11.7325 | 0 |
| `max_vorticity_rolling_3d_max` | $10^{-5}\text{ s}^{-1}$ | 3.9557 | 249.7590 | 24.8445 | 17.2489 | 0 |
| `max_vorticity_rolling_3d_trend` | $10^{-5}\text{ s}^{-1}/\text{day}$ | -113.3935 | 111.7960 | -0.0015 | 7.1600 | 12 |
| `mean_vorticity_change_24h` | $10^{-5}\text{ s}^{-1}$ | -4.6150 | 5.8438 | -0.0001 | 0.4901 | 6 |
| `mean_vorticity_change_48h` | $10^{-5}\text{ s}^{-1}$ | -6.8205 | 7.8126 | -0.0003 | 0.7646 | 12 |
| `mean_vorticity_rolling_3d_mean` | $10^{-5}\text{ s}^{-1}$ | -2.6682 | 5.5172 | 0.2063 | 0.9743 | 0 |
| `weighted_exceedance_change_24h` | fraction $[0, 1]$ | -0.3089 | 0.2907 | -0.0000 | 0.0343 | 6 |
| `weighted_exceedance_change_48h` | fraction $[0, 1]$ | -0.4783 | 0.4033 | -0.0000 | 0.0522 | 12 |
| `weighted_exceedance_rolling_3d_mean` | fraction $[0, 1]$ | 0.0000 | 0.4696 | 0.0579 | 0.0522 | 0 |
| `min_vws_change_24h` | $\text{m/s}$ | -24.7922 | 19.0596 | 0.0006 | 3.5802 | 6 |
| `min_vws_change_48h` | $\text{m/s}$ | -25.8678 | 24.8525 | 0.0004 | 4.8659 | 12 |
| `min_vws_change_72h` | $\text{m/s}$ | -30.3280 | 25.4178 | -0.0002 | 5.5909 | 18 |
| `min_vws_rolling_3d_mean` | $\text{m/s}$ | 0.0087 | 42.9241 | 7.9891 | 7.8847 | 0 |
| `min_vws_rolling_3d_min` | $\text{m/s}$ | 0.0014 | 41.5711 | 5.9724 | 7.1164 | 0 |
| `min_vws_rolling_3d_trend` | $\text{m/s/day}$ | -12.9339 | 12.4262 | 0.0002 | 2.4329 | 12 |
| `mean_vws_change_24h` | $\text{m/s}$ | -19.4603 | 18.3332 | -0.0008 | 3.1906 | 6 |
| `mean_vws_change_48h` | $\text{m/s}$ | -30.1445 | 27.6826 | -0.0018 | 4.9924 | 12 |
| `mean_vws_rolling_3d_mean` | $\text{m/s}$ | 3.8371 | 60.7076 | 21.3290 | 10.5987 | 0 |
| `mean_rh700_change_24h` | $\%$ | -50.7046 | 43.9332 | -0.0036 | 7.0334 | 6 |
| `mean_rh700_change_48h` | $\%$ | -70.1086 | 64.1402 | -0.0067 | 11.3992 | 12 |
| `mean_rh700_rolling_3d_mean` | $\%$ | 1.7462 | 92.0229 | 46.3442 | 23.3928 | 0 |
| `mean_rh500_change_24h` | $\%$ | -51.5609 | 50.4491 | -0.0027 | 8.7528 | 6 |
| `mean_rh500_change_48h` | $\%$ | -76.3285 | 79.9514 | -0.0047 | 13.7496 | 12 |
| `mean_rh500_rolling_3d_mean` | $\%$ | 1.1689 | 95.0156 | 37.5746 | 26.1117 | 0 |

---

## 12. Observational Storm Sanity Audits

As mandated, these checks are strictly observational: they report what the data **actually show** without assuming outcomes or forcing thresholds.

### 12.1. Cyclone Asna (August–September 2024, Northern Arabian Sea)
- **Genesis Window**: Land-depression moved off Gujarat/Saurashtra coast into the Northeast Arabian Sea on August 29–30, 2024, becoming Cyclonic Storm "Asna" on August 30.
- **Empirical Observations in NAS**:
  - **August 22–23**: Baseline background state. Max vorticity was $10.63$ and $10.23 \times 10^{-5}\text{ s}^{-1}$.
  - **August 24**: Max vorticity increased to $16.97$ ($+6.75$ 24h change) centered at (23.75°N, 72.0°E).
  - **August 26**: Vorticity jumped to $39.70 \times 10^{-5}\text{ s}^{-1}$ ($+21.76$ 24h change) at (22.75°N, 71.25°E).
  - **August 27**: Vorticity surged to $59.89 \times 10^{-5}\text{ s}^{-1}$ ($+20.19$ 24h change); weighted exceedance fraction expanded to $25.15\%$.
  - **August 28**: Max vorticity reached $64.30 \times 10^{-5}\text{ s}^{-1}$; mid-tropospheric moisture surged to $81.92\%$ at 500 hPa.
  - **August 30 (IMD Cyclone Naming Date)**: Max vorticity peaked at **$73.52 \times 10^{-5}\text{ s}^{-1}$** at (23.5°N, 67.25°E) with minimum VWS at a favorable **$0.18\text{ m/s}$**.
  - **August 31 – September 01 (Westward Track)**: Max vorticity center migrated westward from 67.25°E $\rightarrow$ 63.0°E $\rightarrow$ 61.5°E with peak vorticity gradually tapering ($54.58 \rightarrow 58.55 \times 10^{-5}\text{ s}^{-1}$).

### 12.2. Cyclone Remal (May 2024, Bay of Bengal)
- **Genesis & Landfall Window**: Formed over Central Bay of Bengal (CBOB) around May 23–24, moved northward into Northern Bay of Bengal (NBOB), and made landfall on May 26–27 near the Sundarbans.
- **Empirical Observations in CBOB**:
  - **May 21**: Minimum VWS dropped to $0.04\text{ m/s}$, establishing an initial precursor.
  - **May 24**: Weighted exceedance fraction peaked at **$33.84\%$**.
  - **May 25**: Max vorticity peaked at **$54.05 \times 10^{-5}\text{ s}^{-1}$** as the vortex moved northward out of CBOB.
- **Empirical Observations in NBOB**:
  - **May 21**: Minimum VWS was $0.07\text{ m/s}$.
  - **May 26**: Weighted exceedance fraction surged to **$35.32\%$**.
  - **May 27 (Landfall Date)**: Max vorticity reached an extreme peak of **$107.28 \times 10^{-5}\text{ s}^{-1}$** with intense rotational circulation.

### 12.3. Cyclone Biparjoy (June 2023, Arabian Sea)
- **Genesis & ESCS Window**: Formed in Central Arabian Sea (CAS) in early June, intensified into an Extremely Severe Cyclonic Storm (ESCS), tracked northward into NAS, making landfall near Jakhau Port, Gujarat on June 15.
- **Empirical Observations in CAS**:
  - **June 07**: Minimum VWS reached $0.03\text{ m/s}$.
  - **June 10**: Weighted exceedance fraction reached $16.85\%$.
  - **June 11 (Peak ESCS Phase in CAS)**: Max vorticity reached a towering **$186.04 \times 10^{-5}\text{ s}^{-1}$**.
- **Empirical Observations in NAS**:
  - **June 10**: Minimum VWS reached $0.07\text{ m/s}$.
  - **June 14**: Weighted exceedance fraction peaked at **$29.16\%$**.
  - **June 15 (Landfall Date in Gujarat)**: Max vorticity peaked at **$188.21 \times 10^{-5}\text{ s}^{-1}$**.
- **Spatial Consistency**: The ERA5 feature series clearly tracks the northward vortex progression from CAS (peak on June 11) into NAS (peak on June 15).

---

## 13. Storage & Artifact Verification

| Artifact Path | Format | Size | Description |
| :--- | :--- | :--- | :--- |
| `backend/data/era5/features_atmosphere_10yr_daily.parquet` | Apache Parquet (pyarrow) | 6,798,667 bytes (~6.5 MB) | Complete 10-year daily causal atmospheric feature table (21,912 rows $\times$ 48 columns). |
| `backend/data/era5/features_atmosphere_10yr_daily_metadata.json` | JSON | 14,350 bytes | Machine-readable provenance, source audit, null accounting, and statistical summary. |
| `backend/data/era5/features_atmosphere_base_raw.parquet` | Apache Parquet (pyarrow) | 3,115,220 bytes (~3.0 MB) | Raw Phase 1 daily basin features (21,912 rows $\times$ 20 columns) cached for re-engineering. |
| `backend/reports/era5_storm_sanity_audit.json` | JSON | 54,601 bytes | Detailed daily time series for Cyclones Asna, Remal, and Biparjoy. |

---

## 14. Automated Test Suite Results

All automated test suites were executed against the codebase:

1. **`backend/tests/test_era5_feature_engineering.py`**: **10 passed, 0 failed** (11.95s)
   - `test_production_model_hashes_invariance`: PASSED
   - `test_all_six_basins_defined_and_valid`: PASSED
   - `test_sub_basin_slicing_descending_and_ascending`: PASSED
   - `test_weighted_exceedance_fraction_vs_gridcell`: PASSED
   - `test_weighted_exceedance_fraction_edge_cases`: PASSED
   - `test_vws_calculation_and_non_negativity`: PASSED
   - `test_daily_aggregation_complete_day`: PASSED
   - `test_daily_aggregation_incomplete_day`: PASSED
   - `test_zero_lookahead_leakage_audit`: PASSED
   - `test_candidate_dataset_on_disk`: PASSED (verifies 21,912 rows, 6 basins, separate NaN accounting, and `is_ml_warmup_complete`)

2. **`backend/tests/test_era5_validation.py`**: **12 passed, 0 failed** (22.99s)
   - `test_production_model_hashes_invariance`: PASSED
   - `test_era5_file_opens_and_has_variables`: PASSED
   - `test_era5_pressure_levels`: PASSED
   - `test_era5_spatial_grid`: PASSED
   - `test_era5_temporal_integrity`: PASSED
   - `test_era5_nan_and_physical_ranges`: PASSED
   - `test_derived_vws_calculation`: PASSED
   - `test_derived_vorticity_scaling`: PASSED
   - `test_derived_rh_levels`: PASSED
   - `test_spatial_sub_basin_slicing`: PASSED
   - `test_spatial_feature_extractor_service`: PASSED
   - `test_era5_validator_service`: PASSED

3. **Full Backend Suite (`pytest backend/tests -v`)**: **178 passed, 0 failed** (3m 29s)
   - Confirms full regression safety across alert engine, calibration, prediction APIs, SagarBot, and security modules.

---

## 15. Recommendations & Next Steps for Candidate Integration

### Strict Boundaries Maintained:
- **NO ML Retraining Was Performed.**
- **NO Production Models Were Altered.**
- **NO Alert Policies Were Modified.**

### Forensic Conclusions:
1. **Basin Definitions**: Confirmed 100% compliant with approved V2.1 specification.
2. **Extreme Vorticity ($249.76 \times 10^{-5}\text{ s}^{-1}$)**: Verified as authentic Rankine vortex dynamics of historic Super Cyclones (Mocha, Amphan).
3. **RH > 100%**: Verified as ECMWF IFS ice supersaturation in upper-tropospheric convective clouds.
4. **Warm-Up Policy**: Documented and flagged via `is_ml_warmup_complete`.

---

**Execution Complete. Antigravity Agent has ceased modifications in accordance with user instructions.**
