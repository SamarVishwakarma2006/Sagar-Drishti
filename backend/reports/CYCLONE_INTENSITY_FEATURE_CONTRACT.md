# Sagar-Drishti — Cyclone Intensity Feature Contract
## RESEARCH SPECIFICATION — ZERO OPERATIONAL AUTHORITY

**Document ID:** `SD-CONTRACT-2026-CYCLONE-INTENSITY-FEATURES-V2`  
**Classification:** `RESEARCH_ONLY` | `NOT_FOR_PRODUCTION` | `NO_OPERATIONAL_AUTHORITY`  
**Date:** September 13, 2026  
**Module Target:** Cyclone Intensity (Primary Continuous $V_{\max}$) & Rapid Intensification (Research Sensitivity)  
**Parent Pipeline Status:** Production v1.1.0 and Shadow V2.3 Model D remain 100% frozen and untouched.

---

## 1. Governance & Scope

This feature contract defines the candidate predictors, spatial geometries, temporal causal bounds, and missing-data policies for future research modeling of tropical cyclone intensity over the North Indian Ocean (Arabian Sea and Bay of Bengal).

> [!IMPORTANT]
> **LEGAL & OPERATIONAL NOTICE**  
> This feature contract is strictly for retrospective offline research and evaluation. It has **no operational forecasting authority** and does not alter the production operational pipeline (`backend/models/risk_model_3d.joblib`) or the shadow V2.3 pipeline (`backend/models/candidates/v2_3/ocean_atmos_all/`). No model training is conducted within this specification.

---

## 2. Hard Temporal Causality Firewall (Section 11)

For any forecast or evaluation issued at synoptic prediction timestamp $T$:

$$\forall f \in \mathcal{F}_{\text{candidate}}, \quad \text{timestamp}(f) \le T$$

### Absolute Rejection List (Anti-Leakage Firewall)
The following candidate data sources and transformations are **strictly prohibited**:
1. **Future Track Coordinates:** Lat/Lon at $t > T$.
2. **Future Storm Fixes:** Any observation, intensity, or pressure recorded at $t > T$.
3. **Future Environmental Fields:** Atmospheric or ocean analysis slices with timestamp $> T$.
4. **Centered Rolling Windows:** Windows spanning $[T - \Delta t, T + \Delta t]$ (e.g., centered 24h moving averages). Only strictly backward-looking causal windows $[T - \Delta t, T]$ are permissible.
5. **Future Lifetime Statistics:** Storm lifetime maximum intensity, lifetime minimum pressure, final storm duration, or final post-season classification.
6. **Full-Track Spline Interpolation:** Interpolations derived from full life-cycle coordinates that smooth past positions using future points.
7. **Post-Event Global Normalization:** Scalers or normalizers fitted on datasets containing observations from time periods subsequent to the training partition.

---

## 3. Storm Identity & Future-Lifetime Leakage Audit (Section 12)

| Identifier / Metadata Field | Predictive Status | Feature Vector Inclusion | Leakage Risk & Forensic Rationale |
| :--- | :--- | :--- | :--- |
| `system_id` | Metadata Only | **EXCLUDED** | May encode season length, historical sequence, or facilitate memorization of specific storm trajectories. |
| `storm_name` | Metadata Only | **EXCLUDED** | Names are assigned only upon reaching Cyclonic Storm intensity ($\ge 34\text{ kt}$); inclusion leaks that depression attained named status. |
| `final_category` | Target Derivation | **PROHIBITED** | Directly encodes peak lifetime intensity. Fatal leakage if present at time $T$. |
| `lifetime_max_vmax` | Target Derivation | **PROHIBITED** | Encodes future peak wind speed. Fatal target leakage. |
| `lifetime_min_pc` | Target Derivation | **PROHIBITED** | Encodes future minimum central pressure. Fatal target leakage. |
| `storm_duration_hours` | Metadata Only | **PROHIBITED** | Known only after storm dissipation. Fatal temporal leakage. |
| `basin_id` | Static Geographic | **PERMITTED (Metadata/Stratification)** | Encodes static geographic basin (Arabian Sea vs Bay of Bengal); non-predictive of future trajectory. |

---

## 4. 6-Hourly Atmospheric Synoptic Requirement (Section 13)

Existing daily ERA5 features in `features_atmosphere_10yr_daily.parquet` are evaluated as daily averages (00Z to 18Z). For synoptic cyclone intensity prediction at time $T$, **daily composites contain severe acausal leakage** if a morning forecast (e.g., 00Z or 06Z) ingests daily averages that include subsequent afternoon/evening conditions (12Z or 18Z).

### Required Synoptic Times
Predictions occur at standard synoptic intervals: **00Z, 06Z, 12Z, 18Z**.

### Causal Availability Rule
For a prediction issued at synoptic time $T$:

| Prediction Synoptic Hour ($T$) | Allowed Atmospheric Timestamps | Explicitly Prohibited Timestamps |
| :--- | :--- | :--- |
| **00:00 UTC (00Z)** | Day $D$ 00Z, Day $D-1$ 18Z, 12Z, ... | Day $D$ 06Z, Day $D$ 12Z, **Day $D$ 18Z (FORBIDDEN)** |
| **06:00 UTC (06Z)** | Day $D$ 06Z, Day $D$ 00Z, Day $D-1$ 18Z, ... | Day $D$ 12Z, **Day $D$ 18Z (FORBIDDEN)** |
| **12:00 UTC (12Z)** | Day $D$ 12Z, Day $D$ 06Z, Day $D$ 00Z, ... | **Day $D$ 18Z (FORBIDDEN)** |
| **18:00 UTC (18Z)** | Day $D$ 18Z, Day $D$ 12Z, Day $D$ 06Z, ... | Day $D+1$ 00Z and all future timestamps |

---

## 5. Storm-Centered Spatial Extraction Specification (Section 14)

Rather than fixed geographic buoys or whole-basin averages, candidate features are evaluated across two concentric storm-relative spatial zones centered on the storm position $(\text{lat}_T, \text{lon}_T)$ observed at or before $T$:

```
STORM-CENTERED SPATIAL SPECIFICATION
                  [ 200 - 800 km: Environmental Annulus ]
                ┌─────────────────────────────────────────┐
                │                                         │
                │        [ 0 - 100 km: Inner Core ]       │
                │              ┌───────────┐              │
                │              │  (•) (T)  │              │
                │              └───────────┘              │
                │                                         │
                └─────────────────────────────────────────┘
```

1. **Inner-Core Disk ($0 - 100\text{ km}$ radius):**
   - Captures inner-core thermodynamics, sea surface temperature beneath eyewall, ocean mixed-layer depth, and local eye vorticity.
   - Land masking: Grid points over land are masked out for ocean fields. If $>50\%$ of the inner-core disk intersects land, an `inner_core_land_fraction` flag is set.
2. **Environmental Annulus ($200 - 800\text{ km}$ radius):**
   - Captures synoptic environmental vertical wind shear (850–200 hPa), mid-tropospheric dry air inflow (700 hPa and 500 hPa relative humidity), and upper-tropospheric divergence (200 hPa).
   - Basin boundary handling: Great-circle distances are computed using Haversine formulation on standard $0.25^\circ \times 0.25^\circ$ ERA5 / Copernicus grids.
3. **Status Note:** These radii represent the proposed research baseline (consistent with NHC/SHIPS operational conventions) and are not claimed as scientifically optimal for all cyclone sizes.

---

## 6. Ocean Feature Reusability & Feasibility Audit (Section 15)

Audit of candidate ocean features from Copernicus Marine Service and `features_10yr.parquet`:

| Feature Name | Proposed Unit | Source & Grid | Spatial Zone | Causal Cutoff | Missing Policy | Reusability Status | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sst_inner_core` | $^\circ\text{C}$ | Copernicus L4 ($0.083^\circ$) | $0-100\text{ km}$ disk | $\le T$ (prior day analysis) | Median if offshore, mask if land | **REQUIRES_NEW_EXTRACTION** | Eyewall thermodynamic enthalpy supply; SST $< 26.5^\circ\text{C}$ typically caps intensity. |
| `temp_current` | $^\circ\text{C}$ | `features_10yr.parquet` | Fixed buoy sites (AS1-6, BOB1-6) | $\le T$ | Imputed | **PROXY_ONLY** | Fixed-site SST indicates general basin warmth but lacks proximity to moving eyewall. |
| `temp_abs_anom` | $^\circ\text{C}$ | `features_10yr.parquet` | Fixed buoy sites | $\le T$ | Imputed | **PROXY_ONLY** | Climatic thermal anomaly proxy; does not capture local storm-wake cooling. |
| `temp_zscore` | Dimensionless | `features_10yr.parquet` | Fixed buoy sites | $\le T$ | Imputed | **PROXY_ONLY** | Normalized thermal excess at fixed sites; proxy-level preconditioning only. |
| `mld_inner_core` | $\text{m}$ | Copernicus Marine Physics ($0.083^\circ$) | $0-100\text{ km}$ disk | $\le T$ | Nearest ocean cell | **REQUIRES_NEW_EXTRACTION** | Shallow MLD ($<30\text{ m}$) permits rapid storm-induced upwelling and SST drop; deep MLD ($>60\text{ m}$) buffers enthalpy. |
| `mld_current` | $\text{m}$ | `features_10yr.parquet` | Fixed buoy sites | $\le T$ | Imputed | **PROXY_ONLY** | Fixed-point MLD does not track vortex passage. |
| `ssh_inner_core` | $\text{m}$ | Copernicus Altimetry ($0.25^\circ$) | $0-100\text{ km}$ disk | $\le T$ (7-day latency) | Spatial median | **REQUIRES_NEW_EXTRACTION** | Sea level anomaly identifies warm anticyclonic ocean eddies that trigger sudden intensification. |
| `ssh_current` | $\text{m}$ | `features_10yr.parquet` | Fixed buoy sites | $\le T$ | Imputed | **PROXY_ONLY** | Regional SLA indicator; spatially uncoupled from cyclone track. |
| `tchp_estimated` | $\text{kJ/cm}^2$ | Derived ($\int_{z_{26}}^0 \rho c_p (T-26) dz$) | $0-100\text{ km}$ disk | $\le T$ | Zero if $T < 26^\circ\text{C}$ | **REQUIRES_NEW_EXTRACTION** | Tropical Cyclone Heat Potential integrates heat capacity above $26^\circ\text{C}$ isotherm. |
| `sal_30d_std` | $\text{PSU}$ | `features_10yr.parquet` | Fixed buoy sites | N/A | Excluded | **UNSUITABLE** | Fixed-site salinity standard deviation has no direct physical link to storm wind speed. |
| `cur_shear_fixed` | $\text{m/s}$ | `features_10yr.parquet` | Fixed buoy sites | N/A | Excluded | **UNSUITABLE** | Local ocean current shear at remote buoys is irrelevant to cyclone vortex dynamics. |

---

## 7. Atmospheric Feature Reusability & Feasibility Audit (Section 16)

Audit of candidate atmospheric features from ERA5 Reanalysis:

| Feature Name | Proposed Unit | Source & Level | Spatial Zone | Causal Cutoff | Missing Policy | Reusability Status | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vws_env_annulus` | $\text{kt}$ | ERA5 ($850 - 200\text{ hPa}$) | $200-800\text{ km}$ annulus | $6\text{h synoptic} \le T$ | Reanalysis interpolation | **REQUIRES_NEW_EXTRACTION** | Magnitude of vector wind difference; $>20\text{ kt}$ ventilates warm core, $<10\text{ kt}$ favors rapid intensification. |
| `mean_vws` | $\text{m/s}$ | `features_atmosphere_10yr_daily` | Basin aggregate | Daily composite | Imputed | **PROXY_ONLY** | Daily basin-wide mean masks localized shear corridors encountered along track. |
| `min_vws` | $\text{m/s}$ | `features_atmosphere_10yr_daily` | Basin aggregate | Daily composite | Imputed | **PROXY_ONLY** | Basin minimum indicates favorable regional window but not local storm encounter. |
| `vort_850_core` | $10^{-5}\text{ s}^{-1}$ | ERA5 ($850\text{ hPa}$) | $0-100\text{ km}$ disk | $6\text{h synoptic} \le T$ | Bilinear interpolation | **REQUIRES_NEW_EXTRACTION** | Low-level relative cyclonic spin-up and convergence feeding the primary circulation. |
| `max_vorticity` | $10^{-5}\text{ s}^{-1}$ | `features_atmosphere_10yr_daily` | Basin aggregate | Daily composite | Imputed | **PROXY_ONLY** | Peak basin vorticity identifies monsoon trough activity, not moving eyewall intensity. |
| `rh_700_env` | $\%$ | ERA5 ($700\text{ hPa}$) | $200-800\text{ km}$ annulus | $6\text{h synoptic} \le T$ | Bilinear interpolation | **REQUIRES_NEW_EXTRACTION** | Environmental mid-level moisture; dry air entrainment causes convective asymmetry and weakens core. |
| `mean_rh_700` | $\%$ | `features_atmosphere_10yr_daily` | Basin aggregate | Daily composite | Imputed | **PROXY_ONLY** | Basin-scale humidity average; lacks directional sectoring of dry intrusions. |
| `rh_500_env` | $\%$ | ERA5 ($500\text{ hPa}$) | $200-800\text{ km}$ annulus | $6\text{h synoptic} \le T$ | Bilinear interpolation | **REQUIRES_NEW_EXTRACTION** | Deep tropospheric moisture buffer protecting convective towers from evaporative downdrafts. |
| `mean_rh_500` | $\%$ | `features_atmosphere_10yr_daily` | Basin aggregate | Daily composite | Imputed | **PROXY_ONLY** | Regional background moisture only. |
| `div_200_core` | $10^{-5}\text{ s}^{-1}$ | ERA5 ($200\text{ hPa}$) | $0-200\text{ km}$ disk | $6\text{h synoptic} \le T$ | Bilinear interpolation | **REQUIRES_NEW_EXTRACTION** | Upper-level outflow channel efficiency pumping mass out of the cyclonic column. |

---

## 8. Kinematic & Persistence Feature Specification (Directly Reusable)

Kinematic features derived strictly from historical Best Track fixes available at or before $T$:

| Feature Name | Unit | Source | Formulation | Causal Cutoff | Missing Policy | Reusability Status | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vmax_current` | $\text{kt}$ | IMD Best Track | $V_{\max}(T)$ | $\le T$ | Strict requirement | **DIRECTLY_REUSABLE** | Primary persistence predictor; cyclone intensity exhibits strong auto-correlation. |
| `dvmax_6h` | $\text{kt}$ | IMD Best Track | $V_{\max}(T) - V_{\max}(T-6\text{h})$ | $\le T$ | $0.0$ if first fix | **DIRECTLY_REUSABLE** | Short-term intensification trend / momentum. |
| `dvmax_12h` | $\text{kt}$ | IMD Best Track | $V_{\max}(T) - V_{\max}(T-12\text{h})$ | $\le T$ | Backfill with $2 \times \Delta_{6\text{h}}$ | **DIRECTLY_REUSABLE** | Intermediate intensification trend. |
| `dvmax_24h` | $\text{kt}$ | IMD Best Track | $V_{\max}(T) - V_{\max}(T-24\text{h})$ | $\le T$ | Backfill with $4 \times \Delta_{6\text{h}}$ | **DIRECTLY_REUSABLE** | Full-day intensification history. |
| `pc_current` | $\text{hPa}$ | IMD Best Track | $P_c(T)$ (excluding NADA row 168) | $\le T$ | Standard pressure profile | **DIRECTLY_REUSABLE** | Core pressure depth; physical correlate of wind field. |
| `translation_speed` | $\text{kt}$ | IMD Best Track | $\text{dist}(P_T, P_{T-6\text{h}}) / 6\text{h}$ | $\le T$ | $0.0$ if stationary | **DIRECTLY_REUSABLE** | Slow-moving storms ($<5\text{ kt}$) self-induce cold wakes, capping intensity. |
| `latitude_current` | $^\circ\text{N}$ | IMD Best Track | $\text{lat}(T)$ | $\le T$ | Strict requirement | **DIRECTLY_REUSABLE** | Proxy for Coriolis parameter $f = 2\Omega \sin(\phi)$, steering regimes, and proximity to land. |
| `longitude_current`| $^\circ\text{E}$ | IMD Best Track | $\text{lon}(T)$ | $\le T$ | Strict requirement | **DIRECTLY_REUSABLE** | Geographic basin localization and longitudinal thermal gradients. |

---

## 9. Label Revision Control & Provenance (Section 17)

1. **Retrospective Best Track vs Real-Time Advisory:**
   - The training and validation labels use IMD post-season Best Track archives (`backend/data/historical/imd_tracks_2016_2026.parquet`).
   - Post-season Best Track represents retrospective expert consensus incorporating satellite re-analyses, radar records, and land-station barograms.
2. **Operational Reality:**
   - Real-time operational warnings operate on preliminary Dvorak estimates ($T$-numbers) that may carry $\pm 5 - 10\text{ kt}$ uncertainty compared to post-season Best Track.
3. **Firewall Rule:**
   - Best Track retrospective revisions apply exclusively to the **target labels** ($V_{\max}$ at $T+24\text{h}$).
   - Under no circumstances may retrospective post-season track or intensity data from $t > T$ be utilized as predictive features at time $T$.

---

## 10. Summary Classification & Feature Contract Freeze

- **Directly Reusable Features:** 8 kinematic/persistence features derived from causal Best Track observations.
- **Proxy-Only Features:** 8 basin/fixed-site ocean and atmospheric features (retained for baseline comparison only).
- **Requires New Extraction:** 8 storm-centered 6-hourly ocean-atmosphere features (annular rings, 6-hourly synoptic matching).
- **Unsuitable Features:** All fixed-site salinity trends, localized buoy current shear, and static post-event metadata.
- **Prohibited / Leakage Features:** 10 future-derived track, lifetime, and centered-window variables permanently banned.

**Contract Approval Status:** FROZEN FOR RESEARCH PURPOSES ONLY. ZERO ML TRAINING PERFORMED.
