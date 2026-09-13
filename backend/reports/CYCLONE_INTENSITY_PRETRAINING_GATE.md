# Sagar-Drishti — Cyclone Intensity & Rapid Intensification: Pre-Training Forensic Gate Report

**Document ID:** `SD-GATE-2026-INTENSITY-PRETRAINING-FINAL`  
**Phase:** Pre-Training Forensic Data & Scientific Gate  
**Date:** September 13, 2026  
**Status:** **[PRE-TRAINING FORENSIC GATE COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Scientific Gate Evaluation & Official Pre-Training Readiness Verdicts  

---

## Executive Summary & Immutability Verification

In strict compliance with the project charter, **zero machine learning models were trained, fine-tuned, or calibrated**. Production `v1.1.0` remains the sole operational decision authority. Candidate `V2.3 Model D` remains strictly isolated in shadow observation mode.

All 11 protected artifacts were independently audited using SHA-256 cryptographic checksums and verified to be 100% byte-for-byte identical.

---

## 1. Official Pre-Training Readiness Verdicts (Task 22)

In accordance with strict meteorological scientific standards, Cyclone Intensity and Rapid Intensification are evaluated as **separate hazard modules with independent readiness verdicts**:

```
=============================================================================
                       OFFICIAL PRE-TRAINING GATE VERDICTS
=============================================================================

MODULE A: CYCLONE INTENSITY (Vmax & Central Pressure)
VERDICT: CONDITIONAL_DATA_READY

EVIDENCE:
- 2,535 valid candidate fixes across 117 unique storms over 10 complete years (2016-2026).
- Continuous, well-sampled target distribution spanning 20.0 to 130.0 knots (mean 40.9 kt).
- Authoritative ground-truth labels from official IMD RSMC Best Track archives.
- High physical synergy with existing ocean-atmosphere thermodynamic drivers.

UNRESOLVED BLOCKERS BEFORE TRAINING:
1. Candidate alignment pipeline must execute frozen deduplication (excluding 4 duplicate pairs).
2. Clerical outlier (IMD-2016-8-NADA row 168) must be excluded from training targets.
3. Storm-centered environmental annular feature extraction must be compiled.

-----------------------------------------------------------------------------

MODULE B: RAPID INTENSIFICATION (RI: ΔVmax >= 30 kt / 24h)
VERDICT: DATA_NOT_READY

EVIDENCE:
- Extreme event-level sample size scarcity: Across 10 years, only 18 unique storms
  exhibited standard Rapid Intensification (103 fix pairs, 5.40% prevalence).
- Complete absence of positive cases in recent years: Between January 1, 2024 and
  June 23, 2026, ZERO standard RI storms occurred in the North Indian Ocean (N_pos = 0).
- Severe chronological holdout invalidation: Standard test splits on 2024-2026 yield
  zero positive examples, causing PR-AUC, CSI, and F1 metrics to collapse mathematically.

RECOMMENDED NEXT STEPS:
- Do NOT initiate standalone operational machine learning training for RI.
- Formulate RI as a secondary threshold output from the continuous intensity regressor
  rather than a standalone binary classification network.
- Acquire multi-decadal historical archives (1982-2015) to build an adequate sample of RI storms.
=============================================================================
```

---

## 2. Answers to Mandated Gate Inquiries

### 1. Is the IMD dataset sufficient for intensity?
**YES, conditionally.** The 10-year IMD Best Track dataset contains 2,544 synoptic fixes across 117 storms. After excluding 8 conflicting duplicate rows and 1 clerical pressure outlier, 2,535 valid candidate fixes remain. The continuous intensity span (20 to 130 kt) is adequate for training gradient boosted regression trees.

### 2. Is it sufficient for RI?
**NO.** Standard Rapid Intensification ($\ge 30\text{ kt} / 24\text{h}$) occurred in **only 18 unique storms** across the entire 10-year record. 18 positive events is statistically insufficient to train an independent operational classifier without severe overfitting.

### 3. Is Vmax the preferred intensity target?
**YES.** Maximum 3-minute sustained wind speed ($V_{\max}$) in knots directly measures kinetic destruction potential and represents the primary operational forecast metric utilized by IMD and WMO.

### 4. Is central pressure useful?
**YES.** Central pressure ($P_c$) and pressure drop ($\Delta P = 1010 - P_c$) govern the barometric storm surge. Training $P_c$ as a joint multi-task regression output alongside $V_{\max}$ enforces cyclostrophic balance consistency.

### 5. Is intensity classification useful?
**NO as a primary target; YES as a derived product.** Categorical grade boundaries (e.g. CS at 34 kt vs DD at 33 kt) create artificial step functions. It is scientifically superior to predict continuous $\hat{V}_{\max}$ and map it to IMD warning tiers.

### 6. Is 30 kt/24h RI defensible?
**YES.** $30\text{ kt} / 24\text{h}$ is the authoritative international standard established by WMO, NOAA, and IMD. It represents the 95th percentile of 24h intensity changes in the North Indian Ocean.

### 7. Are 25/20 kt useful sensitivity definitions?
**YES, as secondary research benchmarks.** A 25 kt threshold yields 25 storms (150 pairs), while 20 kt yields 37 storms (253 pairs). They are valuable for evaluating model sensitivity to threshold choice, but must **never** replace 30 kt as the primary operational definition.

### 8. How many independent RI storms exist?
**Exactly 18 unique storms** exhibited $\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$ between May 2016 and January 2026.

### 9. Is 2024–2026 zero-RI confirmed?
**YES, 100% confirmed.** In the 2.5-year observation period from 2024-01-01 to 2026-06-23, exactly zero cyclonic systems in the North Indian Ocean achieved $\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$.

### 10. What temporal resolution is required?
**Synoptic 6-hourly resolution.** Tropical cyclone intensification occurs on synoptic timescales. Atmospheric shear and moisture must be synchronized with synoptic fix hours ($00, 06, 12, 18\text{ UTC}$).

### 11. Are daily ERA5 features sufficient?
**Only with strict 18Z initialization constraints.** Daily features aggregated up to 18Z contain afternoon atmospheric data that would leak future information if evaluated for morning fixes ($00\text{Z}$ or $06\text{Z}$).

### 12. What 6-hourly features may be needed?
Instantaneous 6-hourly slices of 850–200 hPa Vertical Wind Shear ($VWS$), 850 hPa relative vorticity ($\zeta_{850}$), and 700/500 hPa relative humidity ($RH$) extracted at exact synoptic fix timestamps.

### 13. Which ocean features are reusable?
`temp_current` (SST), `temp_abs_anom`, `temp_zscore`, `mld_current` (Mixed Layer Depth), and `ssh_current` (Sea Level Anomaly) from the existing 101-feature dataset are directly and potentially reusable. Fixed-buoy salinity rolling trends are not appropriate.

### 14. Which atmospheric features are reusable?
Basin-level `min_vws`, `mean_vws`, `max_vorticity`, `mean_rh_700`, and `mean_rh_500` are directly reusable as synoptic background proxies. Storm-centered annular shear requires new spatial extraction.

### 15. What spatial representation is best?
**Storm-centered annular radius extraction (0–100 km inner core, 200–800 km environmental annulus).** Fixed monitoring sites introduce unacceptable spatial errors of $>300\text{ km}$.

### 16. What leakage risks exist?
Future storm fixes ($\tau > T$), future reanalyzed environmental shear along the unobserved future path, post-season track revisions, and centered rolling windows.

### 17. What event grouping is required?
**Strict event-level grouping.** All fixes belonging to a unique `system_id` must reside in the identical cross-validation partition. Zero storm overlap is permitted between train and test.

### 18. What chronological split is defensible?
- **For Intensity:** Train on 2016–2021 (74 storms), Validate on 2022–2023 (22 storms), Test on 2024–2026 (21 storms).
- **For RI:** Leave-One-Season-Out (LOSO) Cross-Validation across 2016–2023. The 2024–2026 period must only be evaluated for False Alarm Ratio due to zero positive events.

### 19. What metrics should be primary?
- **Intensity:** Mean Absolute Error ($MAE$) on $V_{\max}$ (knots).
- **Rapid Intensification:** Precision-Recall AUC ($PR\text{-}AUC$) and Critical Success Index ($CSI$).

### 20. What is the final readiness status for INTENSITY?
$$\mathbf{CONDITIONAL\_DATA\_READY}$$

### 21. What is the final readiness status for RI?
$$\mathbf{DATA\_NOT\_READY}$$

---

## 3. Cryptographic Integrity Audit

| Protected Artifact File Path | SHA-256 Checksum | Audit Match |
| :--- | :--- | :--- |
| `backend/models/risk_model_3d.joblib` | `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` | **MATCH (100%)** |
| `backend/models/v2_10yr/risk_model_3d.joblib` | `7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3` | **MATCH (100%)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib` | `7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15` | **MATCH (100%)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib` | `b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe` | **MATCH (100%)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib` | `aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c` | **MATCH (100%)** |
| `backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib` | `250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775` | **MATCH (100%)** |
| `backend/config/frozen_alert_policy_v2.json` | `6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6` | **MATCH (100%)** |
| `backend/data/historical/features_10yr.parquet` | `cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867` | **MATCH (100%)** |
| `backend/data/historical/labeled_features_10yr_clean.parquet` | `25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a` | **MATCH (100%)** |
| `backend/data/era5/features_atmosphere_10yr_daily.parquet` | `551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864` | **MATCH (100%)** |
| `backend/config/v2_3_frozen_experiment_manifest.json` | `73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc` | **MATCH (100%)** |
