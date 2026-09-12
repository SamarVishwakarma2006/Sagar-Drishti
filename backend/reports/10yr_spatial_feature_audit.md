# FORENSIC AUDIT: 10-YEAR SPATIAL GRID & FEATURE ENGINE CONVERSION
**Project:** Sagar-Drishti  
**Branch:** `kshitij`  
**Date:** September 11, 2026  
**Auditor:** Antigravity AI Data & Systems Engineering  
**Scope:** Forensic trace from `backend/data/copernicus/copernicus_phy_10yr_surface.nc` to `backend/data/historical/features_10yr.parquet`

---

## 1. Executive Summary & Direct Answers

| Question | Forensic Answer | Evidence / Details |
| :--- | :--- | :--- |
| **1. Total Rows in `features_10yr.parquet`** | **7,246 rows** | Directly verified via `pd.read_parquet()` |
| **2. Unique Dates** | **3,623 dates** | Spanning `2016-07-23` to `2026-06-23` (no gaps) |
| **3. Unique Spatial Identifiers** | **2 sites / basins** | `['bob', 'aras']` / `['Bay of Bengal', 'Arabian Sea']` |
| **4. Spatial Identifiers List** | `['bob', 'aras']` | Both defined in `SiteRegistry` |
| **5. Spatial Site Details** | `bob`: Lat 17.8°N, Lon 88.2°E<br>`aras`: Lat 16.5°N, Lon 67.5°E | `bob` bbox: [14.0–22.0°N, 83.0–93.0°E] (3,623 rows)<br>`aras` bbox: [12.0–21.0°N, 62.0–73.0°E] (3,623 rows) |
| **6. What Each ML Row Represents** | **C. Regional bounding-box average** | Area mean across lat/lon of Copernicus grid cells inside site bounding box |
| **7. Converting Code / Function** | `HistoricalFeatureEngine.export_training_dataset` + `_extract_1d_series` | `backend/app/services/historical_engine.py` (lines 111–117, 735–749) |
| **8. Full Copernicus Grid Used?** | **NO. Only site bounding-box subsets** | Only the bounding boxes of `bob` and `aras` are sliced and averaged |
| **9. Theoretical Grid vs. Actual Sites** | Theoretical: **180,901 cells/day**<br>Actual: **2 site area averages/day** | $301 \times 601 = 180,901$ spatial cells per day. Over 10 years: 660,650,452 grid points. Actual: 2 spatial sites |
| **10. 7,246 Mathematical Relationship** | **GENUINELY TRUE: 100% Verified** | $3,652 \text{ total days} - 29 \text{ warmup days} = 3,623 \text{ days}$<br>$3,623 \times 2 \text{ sites} = \mathbf{7,246 \text{ rows}}$ |

---

## 2. Raw NetCDF vs. Feature Dataset Verification

### Raw NetCDF: `backend/data/copernicus/copernicus_phy_10yr_surface.nc`
- **File Size:** 7,927,862,543 bytes (7.383 GB)
- **Time Coordinate:** Exactly 3,652 daily timesteps from `2016-06-24` to `2026-06-23`.
- **Latitude Points:** 301 points from $0.0^\circ\text{N}$ to $25.0^\circ\text{N}$ ($\Delta = 0.0833^\circ \approx 9.25\text{ km}$).
- **Longitude Points:** 601 points from $50.0^\circ\text{E}$ to $100.0^\circ\text{E}$ ($\Delta = 0.0833^\circ \approx 9.25\text{ km}$).
- **Vertical Level:** Single surface depth coordinate: `depth = 0.494025 m` (1 level).
- **Physical Variables:** 6 ocean variables:
  - `thetao` (Sea Water Potential Temperature, °C)
  - `so` (Sea Water Salinity, PSU)
  - `uo` (Eastward Sea Water Velocity, m/s)
  - `vo` (Northward Sea Water Velocity, m/s)
  - `zos` (Sea Surface Height above Geoid, m)
  - `mlotst` (Ocean Mixed Layer Thickness, m)
- **Theoretical Domain Size:**
  $$\text{Grid cells per daily snapshot} = 301 \times 601 = 180,901$$
  $$\text{Total 10-year grid observations} = 180,901 \times 3,652 = 660,650,452 \text{ data points}$$

### ML Feature Dataset: `backend/data/historical/features_10yr.parquet`
- **File Size:** 3,076,259 bytes (~3.0 MB)
- **Total Rows:** Exactly **7,246 rows**
- **Columns:** 106 (5 metadata columns: `date`, `day_index`, `mode`, `site_id`, `basin` + 101 continuous physical features).
- **Unique Dates:** Exactly **3,623 dates** from `2016-07-23` to `2026-06-23`.
- **Unique Spatial Sites:** Exactly **2**: `bob` (3,623 rows) and `aras` (3,623 rows).

---

## 3. Mathematical Derivation of 7,246 Rows

The exact count of 7,246 rows is derived as follows:

1. **Total Days in Copernicus 10-Year Record:**
   $$\text{Days from 2016-06-24 to 2026-06-23 (inclusive)} = 3,652 \text{ calendar days}$$
2. **Causal Warmup Window Requirement:**
   The project contract specifies causal rolling windows of 7, 14, and 30 days. To compute the 30-day rolling mean, delta, and linear trend at day $t$, the engine requires observations from $t - 29$ through $t$.
   In `HistoricalFeatureEngine.export_training_dataset` (`backend/app/services/historical_engine.py`, lines 766–767):
   ```python
   # Iterate over all eligible days: index 29 through total_obs - 1
   for idx in range(29, total_obs):
   ```
   Day indices $0$ to $28$ (29 days, spanning `2016-06-24` to `2016-07-22`) are strictly discarded to eliminate lookback truncation and prevent forward-leakage.
3. **Usable Daily Timesteps:**
   $$3,652 \text{ total days} - 29 \text{ warmup days} = 3,623 \text{ usable days}$$
4. **Spatial Expansion:**
   Two representative basin study regions are extracted:
   - Bay of Bengal (`bob`): 3,623 daily observations
   - Arabian Sea (`aras`): 3,623 daily observations
5. **Total Feature Matrix Rows:**
   $$3,623 \times 2 = \mathbf{7,246 \text{ rows}}$$

This relationship is **100% verified and mathematically exact**.

---

## 4. Spatial Sites Inventory

| Spatial ID | Basin / Site Name | Region | Center Lat | Center Lon | Feature Extraction Bounding Box | Total Rows | First Date | Last Date |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bob`** | Bay of Bengal — Fresh Plume | N. Indian Ocean | 17.8°N | 88.2°E | Lat: [14.0°N, 22.0°N]<br>Lon: [83.0°E, 93.0°E] | 3,623 | 2016-07-23 | 2026-06-23 |
| **`aras`** | Arabian Sea — OMZ Core | N. Indian Ocean | 16.5°N | 67.5°E | Lat: [12.0°N, 21.0°N]<br>Lon: [62.0°E, 73.0°E] | 3,623 | 2016-07-23 | 2026-06-23 |

---

## 5. Exact Feature Extraction Path & Implementation Logic

The conversion from raw NetCDF to Parquet is executed by `HistoricalFeatureEngine.export_training_dataset` in `backend/app/services/historical_engine.py`.

### Slicing & Spatial Averaging Logic (`_extract_1d_series`, lines 101–117):
```python
else:
    # Regional aggregation
    if bbox is None:
        raise ValueError("Region mode requires a valid bounding box.")
    lats = ds[coord_lat].values
    lat_slice = slice(min(bbox.min_lat, bbox.max_lat), max(bbox.min_lat, bbox.max_lat))
    if len(lats) > 1 and lats[0] > lats[-1]:
        lat_slice = slice(max(bbox.min_lat, bbox.max_lat), min(bbox.min_lat, bbox.max_lat))

    reg_da = var_da.sel({
        coord_lat: lat_slice,
        coord_lon: slice(bbox.min_lon, bbox.max_lon)
    })
    # Area mean across lat/lon
    mean_da = reg_da.mean(dim=[coord_lat, coord_lon], skipna=True)
    return mean_da.values.astype(float)
```

### Script Invocations (`backend/scripts/build_clean_10yr_dataset.py`, lines 61–82):
```python
res_bob = HistoricalFeatureEngine.export_training_dataset(
    mode="region",
    site_id="bob",
    nc_path=NC_10YR_PATH,
    output_dir=raw_bob_dir,
)

res_aras = HistoricalFeatureEngine.export_training_dataset(
    mode="region",
    site_id="aras",
    nc_path=NC_10YR_PATH,
    output_dir=raw_aras_dir,
)
```

---

## 6. Spatial Aggregation Methodology

Each row in `features_10yr.parquet` corresponds to **Option C: A regional bounding-box average**:
1. For every daily timestep $T$ and each of the 6 ocean variables, the engine extracts a 2D spatial slice bounded by `bbox.min_lat` to `bbox.max_lat` and `bbox.min_lon` to `bbox.max_lon`.
2. Land pixels (which have `NaN` values in the Copernicus physical variables) are skipped via `skipna=True`.
3. The remaining ocean pixels are averaged together into a single scalar value representing the basin-wide ocean state on day $T$.
4. The derived channel `cur` (total current magnitude) is calculated as $\sqrt{u^2 + v^2}$.
5. From these 7 daily time series (`temp`, `sal`, `cur_u`, `cur_v`, `cur`, `ssh`, `mld`), the engine computes full-period baseline statistics and causal rolling features (7d, 14d, 30d), producing the frozen 101 features.

---

## 7. Year-by-Year Coverage Table

The table below is generated directly from `backend/data/historical/features_10yr.parquet`:

| Year | Unique Days | Spatial Locations | Total Rows | Expected Rows | Missing Days |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2016** | 162 | 2 | 324 | 324 | **0** |
| **2017** | 365 | 2 | 730 | 730 | **0** |
| **2018** | 365 | 2 | 730 | 730 | **0** |
| **2019** | 365 | 2 | 730 | 730 | **0** |
| **2020** | 366 | 2 | 732 | 732 | **0** |
| **2021** | 365 | 2 | 730 | 730 | **0** |
| **2022** | 365 | 2 | 730 | 730 | **0** |
| **2023** | 365 | 2 | 730 | 730 | **0** |
| **2024** | 366 | 2 | 732 | 732 | **0** |
| **2025** | 365 | 2 | 730 | 730 | **0** |
| **2026** | 174 | 2 | 348 | 348 | **0** |
| **TOTAL** | **3,623** | **2** | **7,246** | **7,246** | **0** |

*Notes:*
- 2016 begins on `2016-07-23` (day 29 of the record), yielding 162 days.
- 2020 and 2024 have 366 days (leap years).
- 2026 ends on `2026-06-23`, yielding 174 days.
- **Zero missing days exist across the entire 10-year span.**

---

## 8. Raw NetCDF → Parquet Feature Trace

To certify that features genuinely originate from the 10-year NetCDF, 4 dates were sampled across the 10-year span. For each date, the raw NetCDF data array was sliced by the site's bounding box, area-averaged, and compared to the values stored in `features_10yr.parquet`:

### 1. Date: 2017-09-15
- **`bob` (Bay of Bengal):**
  - Raw NetCDF `thetao` bbox mean: **29.6633°C** | Parquet `temp_current`: **29.6633°C** (diff: $0.000032^\circ$)
  - Raw NetCDF `so` bbox mean: **29.1056 PSU** | Parquet `sal_current`: **29.1056 PSU** (diff: $0.000042\text{ PSU}$)
- **`aras` (Arabian Sea):**
  - Raw NetCDF `thetao` bbox mean: **28.4815°C** | Parquet `temp_current`: **28.4815°C** (diff: $0.000006^\circ$)
  - Raw NetCDF `so` bbox mean: **36.2570 PSU** | Parquet `sal_current`: **36.2570 PSU** (diff: $0.000040\text{ PSU}$)

### 2. Date: 2020-05-20 (Super Cyclone Amphan)
- **`bob` (Bay of Bengal):**
  - Raw NetCDF `thetao` bbox mean: **29.8729°C** | Parquet `temp_current`: **29.8729°C** (diff: $0.000023^\circ$)
  - Raw NetCDF `so` bbox mean: **32.7190 PSU** | Parquet `sal_current`: **32.7190 PSU** (diff: $0.000016\text{ PSU}$)
- **`aras` (Arabian Sea):**
  - Raw NetCDF `thetao` bbox mean: **30.6325°C** | Parquet `temp_current`: **30.6325°C** (diff: $0.000034^\circ$)
  - Raw NetCDF `so` bbox mean: **36.1149 PSU** | Parquet `sal_current`: **36.1149 PSU** (diff: $0.000048\text{ PSU}$)

### 3. Date: 2024-10-24 (Severe Cyclonic Storm Dana)
- **`bob` (Bay of Bengal):**
  - Raw NetCDF `thetao` bbox mean: **29.6185°C** | Parquet `temp_current`: **29.6185°C** (diff: $0.000017^\circ$)
  - Raw NetCDF `so` bbox mean: **29.5260 PSU** | Parquet `sal_current`: **29.5260 PSU** (diff: $0.000048\text{ PSU}$)
- **`aras` (Arabian Sea):**
  - Raw NetCDF `thetao` bbox mean: **29.3579°C** | Parquet `temp_current`: **29.3579°C** (diff: $0.000046^\circ$)
  - Raw NetCDF `so` bbox mean: **35.7900 PSU** | Parquet `sal_current`: **35.7900 PSU** (diff: $0.000020\text{ PSU}$)

### 4. Date: 2026-03-10
- **`bob` (Bay of Bengal):**
  - Raw NetCDF `thetao` bbox mean: **27.1219°C** | Parquet `temp_current`: **27.1219°C** (diff: $0.000012^\circ$)
  - Raw NetCDF `so` bbox mean: **30.9673 PSU** | Parquet `sal_current`: **30.9673 PSU** (diff: $0.000020\text{ PSU}$)
- **`aras` (Arabian Sea):**
  - Raw NetCDF `thetao` bbox mean: **27.6234°C** | Parquet `temp_current`: **27.6234°C** (diff: $0.000044^\circ$)
  - Raw NetCDF `so` bbox mean: **35.6665 PSU** | Parquet `sal_current`: **35.6665 PSU** (diff: $0.000025\text{ PSU}$)

**Conclusion:** The values in `features_10yr.parquet` match the raw NetCDF values down to the 4th decimal place (exact rounding precision). They genuinely originate from the 10-year NetCDF.

---

## 9. Duplication & Physical Divergence Analysis

- **Duplicate rows:** Exactly **0 duplicate `(date, site_id)` rows**.
- **Identical feature vectors between sites:** Exactly **0 days** out of 3,623 where `bob` and `aras` share identical temperatures.
- **Oceanographic Divergence:**
  - **Sea Surface Salinity (SSS):**
    - Bay of Bengal Mean: **30.740 PSU** (freshwater cap driven by Ganga–Brahmaputra runoff)
    - Arabian Sea Mean: **36.023 PSU** (high evaporation, hyper-saline)
    - Physical Difference: **$-5.284\text{ PSU}$**
    - Salinity Correlation between basins: **$-0.0353$** (completely uncoupled dynamics)
  - **Sea Surface Temperature (SST):**
    - Bay of Bengal Mean: **28.649°C**
    - Arabian Sea Mean: **28.299°C**
    - SST Correlation: **$0.8466$** (coupled Northern Indian Ocean seasonal monsoon cycle)
  - **Mixed Layer Depth (MLD):**
    - Bay of Bengal Mean: **16.332 m** (shallow barrier layer)
    - Arabian Sea Mean: **26.062 m** (deep monsoon convective mixing)

The two spatial sites represent genuinely distinct, physically realistic ocean regimes.

---

## 10. Copernicus vs. IMD Spatial Consistency Audit

### CRITICAL FINDING: Spatial Resolution Mismatch Between Feature Extraction and Event Labeling

The forensic audit revealed a subtle spatial discrepancy between how ocean features were extracted and how cyclone labels were matched:

1. **Ocean Feature Extraction Spatial Box:**
   When `HistoricalFeatureEngine.export_training_dataset(mode="region", site_id="bob")` ran, it fetched `SiteRegistry.get_site("bob").bbox`:
   $$\text{Feature Box (bob)} = [14.0^\circ\text{N}, 22.0^\circ\text{N}, 83.0^\circ\text{E}, 93.0^\circ\text{E}]$$
   - Extent: $8^\circ \text{ latitude} \times 10^\circ \text{ longitude}$ (97 lat $\times$ 121 lon grid points = **11,737 Copernicus cells**, $\approx 800,000\text{ km}^2$).
   - This represents the **entire northern/central Bay of Bengal basin**.

2. **Cyclone Label Matching Spatial Box:**
   When `build_clean_10yr_dataset.py` labeled observations with `SpatialTemporalMatcher.match_observation`, it explicitly passed:
   $$\text{Label Matching Box (bob)} = [16.3^\circ\text{N}, 19.3^\circ\text{N}, 86.7^\circ\text{E}, 89.7^\circ\text{E}]$$
   - Extent: $3^\circ \text{ latitude} \times 3^\circ \text{ longitude}$ (37 lat $\times$ 37 lon grid points = **1,369 Copernicus cells**, $\approx 100,000\text{ km}^2$).
   - This represents a **tight $\pm 1.5^\circ$ box around the primary mooring site (17.8°N, 88.2°E)**.

3. **Inference Consistency (`PredictionService.predict`):**
   When an API user calls `/predict` with `site_id="bob"` (the default in production), `PredictionService` calls `HistoricalFeatureEngine.compute_comparison(mode="region", site_id="bob")`, which evaluates the **full $8^\circ \times 10^\circ$ basin box**.

### Impact of this Mismatch:
- **Feature Matrix:** Represents large-scale regional ocean thermodynamics across the entire North Bay of Bengal (monsoon heat content, plume salinity, basin-wide SSH).
- **Target Labels:** Represents whether a cyclone center tracks into the central 3° × 3° corridor around the mooring buoy.
- **Why this was done:**
  - If the broad $8^\circ \times 10^\circ$ box were used for IMD event matching, 61 of 117 cyclones would intersect the box, falsely flagging the central site as having active storms when cyclones were actually 600 km away in the south/east bay.
  - Using the 3° × 3° box matched 29 genuine direct-impact cyclones, keeping prevalence at a realistic 5.42%.
- **Scientific Assessment:**
  Using basin-wide thermodynamic features to forecast risk of storm passage into a key maritime transit zone is meteorologically sound (ocean heat content over a broad upstream area fuels storm development). However, the spatial discrepancy must be made explicit and documented in the model contract.

---

## 11. Architectural Intentionality & Why 7,246 Rows is Expected

### Is 7,246 rows expected?
**YES.**
It is the exact mathematical product of $3,623 \text{ usable days} \times 2 \text{ representative basins}$.

### Was the architecture intended to be pixel-level or site/basin-level?
**SITE / BASIN LEVEL.**
- The production baseline model (`v1.1.0`) was trained on `labeled_features.parquet` (136 rows for a single site) and `ml_features_multibasin.parquet` (374 rows: 185 for `bob` + 189 for `aras` across 6 months).
- The 10-year expansion in Sagar-Drishti was designed to expand the **temporal baseline from 6 months/2 years to 10 continuous years (2016–2026)** for the multi-basin architecture, **not** to convert the application into a 660-million-row pixel-level gridded forecasting model.
- A 660-million-row dataset ($180,901 \text{ cells} \times 3,652 \text{ days}$) would require >500 GB of RAM/disk and would violate the project's lightweight Random Forest architecture.

---

## 12. Recommendations Prior to Retraining

1. **Retraining Permission:**
   Retraining is scientifically sound and safe to proceed under candidate isolation (`backend/models/v2_10yr/`).
2. **Spatial Contract Documentation:**
   The training documentation must explicitly state that features represent **basin-level ocean thermodynamic state** ($8^\circ \times 10^\circ$ for BoB, $9^\circ \times 11^\circ$ for Arabian Sea), while risk targets represent **regional storm intrusion** ($3^\circ \times 3^\circ$ central zone).
3. **Keep Baseline v1.1.0 Active:**
   Maintain production baseline `v1.1.0` in `backend/models/` untouched while candidate models are evaluated.

---

## Required Final Summary

RAW DATA:  
→ **GENUINELY 10-YEAR** (3,652 daily timesteps from 2016-06-24 to 2026-06-23, 301×601 grid, 7.383 GB NetCDF verified on disk).

FEATURE DATA:  
→ **GENUINELY USING 10-YEAR DATA** (3,623 daily observations from 2016-07-23 to 2026-06-23 after discarding 29-day causal warmup; zero missing days; exact physical values match raw NetCDF to 4 decimal places).

SPATIAL COVERAGE:  
→ **2 REPRESENTATIVE BASIN SITES** (`bob` = Bay of Bengal, `aras` = Arabian Sea), area-averaged across their respective ocean bounding boxes.

7,246 ROW EXPLANATION:  
→ Exactly **3,623 usable daily timesteps $\times$ 2 spatial basin sites = 7,246 rows**.

RETRAINING:  
→ **SAFE TO PROCEED** (under strict candidate isolation in `backend/models/v2_10yr/` using the chronological split; production baseline v1.1.0 remains active).
