# Sagar-Drishti — Cyclone Intensity & Rapid Intensification: Causal Leakage Audit

**Document ID:** `SD-AUDIT-2026-INTENSITY-LEAKAGE`  
**Phase:** Pre-Training Forensic Data & Scientific Gate  
**Date:** September 13, 2026  
**Status:** **[LEAKAGE AUDIT COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Scientific Integrity, Causal Cutoffs & Anti-Leakage Protocols  

---

## Executive Summary

Machine learning models designed for tropical cyclone intensity prediction and Rapid Intensification (RI) are exceptionally vulnerable to **temporal and target leakage**. Because tropical cyclones are dynamic, rapidly propagating vortices, contaminating feature sets with environmental fields or storm attributes from even a few hours into the future produces artificially inflated validation metrics that fail catastrophically in operational forecasting.

This audit establishes a comprehensive **Feature-by-Feature Causal Firewall** for all candidate predictors, analyzes the scientific implications of **daily vs. 6-hourly temporal resolution**, and formalizes the data alignment protocol connecting IMD Best Track observations with ocean and atmospheric reanalyses.

---

## 1. The Fundamental Causal Cutoff Principle

$$\mathbf{X}(T) \subset \mathcal{F}_{\le T} \quad \text{and} \quad \mathbf{Y}(T+H) \subset \mathcal{F}_{T+H}$$

At forecast initialization time $T$, the feature vector $\mathbf{X}(T)$ must strictly be measurable from observations available at or before $T$. No data, observation, derivative, or rolling window extending into time $\tau > T$ may enter the predictor space.

```
CAUSAL INFORMATION BOUNDARY AT INITIALIZATION TIME T
─────────────────────────────────────────────────────────────────────────────
PAST & CURRENT: tau <= T                                  FUTURE: tau > T
[Storm Fix at T: lat, lon, Vmax, Pc]                       [TARGET HORIZONS]
[Ocean State: SST, MLD, Salinity at tau <= T]              [Intensity at T+24h]
[Atmospheric State: Shear, Vorticity at tau <= T]          [Intensity at T+48h]
─────────────────────────────────────────────┬─────────────[RI: V(T+24)-V(T)]
                     FORECAST TIME T          │
                                             │  ABSOLUTE LEAKAGE PROHIBITION
                                             │  NO future storm fixes tau > T
                                             │  NO future environmental fields
                                             │  NO centered rolling windows
```

---

## 2. Feature-by-Feature Leakage Audit Table (Task 9)

Every candidate predictor proposed for Cyclone Intensity and Rapid Intensification is audited against the strict causal cutoff at forecast time $T$:

| Predictor Name | Description | Proposed Time Window | Available at $T$? | Leakage Assessment | Action / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `current_vmax` | Current storm intensity $V_{\max}(T)$ | Fix at $T$ | **YES** | Legitimate causal predictor | **APPROVED** |
| `current_pc` | Current central pressure $P_c(T)$ | Fix at $T$ | **YES** | Legitimate causal predictor | **APPROVED** |
| `vmax_change_6h` | Historical 6h intensity trend | $[T-6\text{h}, T]$ | **YES** | Backward-looking historical trend | **APPROVED** |
| `vmax_change_12h`| Historical 12h intensity trend | $[T-12\text{h}, T]$| **YES** | Backward-looking historical trend | **APPROVED** |
| `vmax_change_24h`| Historical 24h intensity trend | $[T-24\text{h}, T]$| **YES** | Backward-looking historical trend | **APPROVED** |
| `current_lat_lon`| Storm center location at $T$ | Fix at $T$ | **YES** | Legitimate causal coordinate | **APPROVED** |
| `storm_speed_6h` | Translation speed over prior 6h | $[T-6\text{h}, T]$ | **YES** | Backward-looking translation | **APPROVED** |
| `storm_dir_6h`   | Heading direction over prior 6h| $[T-6\text{h}, T]$ | **YES** | Backward-looking heading | **APPROVED** |
| `sst_at_t`       | Sea Surface Temp at center ($T$) | Day of $T$ ($\tau \le T$)| **YES** | Ocean preconditioning | **APPROVED** |
| `mld_at_t`       | Mixed Layer Depth at center ($T$)| Day of $T$ ($\tau \le T$)| **YES** | Ocean thermal buffer | **APPROVED** |
| `vws_850_200_at_t`| Vertical Wind Shear at $T$ | Slice at $T$ | **YES** | Atmospheric shear forcing | **APPROVED** |
| `vo_850_at_t`    | 850 hPa relative vorticity at $T$| Slice at $T$ | **YES** | Low-level vortex spin-up | **APPROVED** |
| `rh_700_500_at_t`| Mid-tropospheric humidity at $T$| Slice at $T$ | **YES** | Dry-air intrusion inhibitor | **APPROVED** |
| `div_200_at_t`   | Upper-tropospheric divergence | Slice at $T$ | **YES** | Upper-level ventilation | **APPROVED** |
| *future_vmax*    | $V_{\max}$ at $T+6\text{h}$ or $T+12\text{h}$ | $\tau > T$ | **NO** | **SEVERE TARGET LEAKAGE** | **REJECTED** |
| *future_lat_lon* | Future storm coordinates ($T+24$)| $\tau > T$ | **NO** | **SEVERE TRACK LEAKAGE** | **REJECTED** |
| *future_env_sst* | SST along future storm path | $\tau > T$ | **NO** | Future path is unknown at $T$ | **REJECTED** |
| *future_vws*     | Future shear along storm path | $\tau > T$ | **NO** | Unknown without forecast NWP | **REJECTED** |
| *post_season_rev*| Reanalyzed genesis / peak wind | Post-event review | **NO** | Unavailable in real time | **REJECTED** |
| *centered_roll*  | 7-day centered rolling mean | $[T-3\text{d}, T+3\text{d}]$| **NO** | Forward rolling lookahead | **REJECTED** |
| *lifetime_max*   | Peak storm intensity reached | Full lifetime | **NO** | Only known after dissipation | **REJECTED** |
| *total_duration* | Storm total active duration | Full lifetime | **NO** | Only known after dissipation | **REJECTED** |

---

## 3. Temporal Resolution Audit: Daily vs 6-Hourly Features (Task 8)

A critical architectural finding of this forensic gate is the **temporal misalignment between daily atmospheric aggregates and synoptic cyclone fixes**:

```
TEMPORAL GRANULARITY COMPARISON
┌───────────────────────┬───────────────────────────────┬─────────────────────────────────────────────┐
│ Feature Asset         │ Native Temporal Resolution    │ Operational Causal Implication              │
├───────────────────────┼───────────────────────────────┼─────────────────────────────────────────────┤
│ IMD Synoptic Fixes    │ 3-hourly and 6-hourly fixes   │ Fixes occur at 00:00, 06:00, 12:00, 18:00 Z│
│ Existing V2.1 ERA5    │ Daily spatial summaries       │ Aggregates up to strict 18:00 UTC cutoff    │
│ Raw ERA5 NetCDF Chunks│ 6-hourly pressure slices      │ Native slices at 00:00, 06:00, 12:00, 18:00Z│
└───────────────────────┴───────────────────────────────┴─────────────────────────────────────────────┘
```

### The 18Z Daily Aggregation Risk
- In the existing candidate dataset (`features_atmosphere_10yr_daily.parquet`), features represent daily statistics calculated over all four 6-hourly time steps of that calendar day (ending at $18:00\text{ UTC}$).
- **Leakage Vulnerability:** If a forecast is initialized at $T = 06:00\text{ UTC}$ on day $D$, and the model inputs the daily feature for day $D$, the feature incorporates atmospheric slices from $12:00\text{ UTC}$ and $18:00\text{ UTC}$—which occur **in the future relative to $T = 06:00\text{ UTC}$**!

### Causal Synchronization Mandate
To maintain zero leakage, future Cyclone Intensity data alignment must strictly enforce one of two rules:
1. **Rule A (Daily Restriction):** If using daily atmospheric summaries, **forecasts must strictly be initialized at $T = 18:00\text{ UTC}$**, or use the prior day's ($D-1$) daily summary for morning forecasts ($00\text{Z}$ and $06\text{Z}$).
2. **Rule B (6-Hourly Extraction):** For optimal meteorological fidelity, feature extraction for intensity should extract **instantaneous 6-hourly atmospheric slices matching exact synoptic fix times $T \in \{00, 06, 12, 18\text{ UTC}\}$** from the raw NetCDF archives resident in `backend/data/era5/chunks/`.

---

## 4. Ocean & Atmospheric Data Alignment Protocols (Task 13)

```
TEMPORAL ALIGNMENT TIMELINE
T - 24h               T - 6h                 T (Forecast Initialization)          T + 24h (Target)
───┼─────────────────────┼───────────────────┼──────────────────────────────────────────┼───
   │                     │                   │                                          │
   Fix: Vmax(T-24)       Fix: Vmax(T-6)      Fix: Vmax(T), Pc(T), (lat, lon)            Target: Vmax(T+24)
   Ocean: SST(D-1)       Ocean: SST(D-1)     Ocean: SST(D-1) or SST(D) if available     Target: Pc(T+24)
   Atmos: Slice(T-24)    Atmos: Slice(T-6)   Atmos: Slice(T) [Strict Cutoff]            Delta V24 = V(T+24)-V(T)
```

### Spatial Alignment Rules
1. **Storm-Centered Environmental Extraction (Recommended):** Ocean and atmospheric features must be sampled within an annular radius centered on the storm's current position $(\phi(T), \lambda(T))$:
   - Inner core: $r \in [0, 100\text{ km}]$ (SST cooling, mixed-layer heat content, core vorticity).
   - Environmental annulus: $r \in [200, 800\text{ km}]$ (Deep-layer vertical wind shear, mid-tropospheric moisture).
2. **Nearest-Site Approximation (Unsuitable for Intensity):** Sagar-Drishti's existing 12 fixed monitoring sites (AS1–AS6, BOB1–BOB6) are spaced hundreds of kilometers apart. Mapping a cyclone located at $14.2°\text{N}, 87.5°\text{E}$ to the nearest site (BOB3 at $12.0°\text{N}, 85.0°\text{E}$) introduces spatial displacement errors of $>300\text{ km}$, misrepresenting the true local wind shear and ocean enthalpy feeding the eyewall.

### Boundary Cases & Land Interaction
1. **Landfalling Systems:** When a storm moves inland, intense surface friction and the loss of oceanic latent heat flux cause rapid decay.
   - *Rule:* If the target fix at $T+24\text{h}$ is over land, the sample remains valid for continuous $V_{\max}$ regression (learning decay physics). However, for Rapid Intensification, landfalling systems within 24 hours must be flagged (`is_landfall_24h = True`) because RI over land is physically impossible.
2. **Dissipating Systems:** If a storm dissipates before $T+21\text{h}$, it has no valid fix at $T+24\text{h}$. The sample cannot have a 24h change target and must be excluded from RI evaluation.

---

## 5. Summary of Anti-Leakage Rules for Future Implementation

1. **Zero Future Lookahead:** No variable with timestamp $\tau > T$ may enter $\mathbf{X}(T)$.
2. **Zero Interpolation Leakage:** Target $V_{\max}(T+24\text{h})$ must be an observed synoptic fix within $[T+21\text{h}, T+27\text{h}]$, not a forward spline.
3. **Strict Post-Season Revision Quarantine:** All experiment training manifests must record the cryptographic hash of the IMD Best Track Parquet to prevent silent historical revisions from altering benchmark reproducibility.
4. **Chronological Partitioning:** Cross-validation folds must group all fixes of each storm together, preventing auto-correlated track slices from leaking across training and test partitions.
