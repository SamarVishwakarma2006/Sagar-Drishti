# Sagar-Drishti — Cyclone Intensity & Rapid Intensification: Training & Experiment Design

**Document ID:** `SD-DESIGN-2026-INTENSITY-TRAINING-DESIGN`  
**Phase:** Pre-Training Forensic Data & Scientific Gate  
**Date:** September 13, 2026  
**Status:** **[EXPERIMENT DESIGN COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Scientific Methodology, Feature Categorization & Validation Protocols  

---

## Executive Summary

This document establishes the scientific and methodological blueprint for the future training of Sagar-Drishti's **Cyclone Intensity** and **Rapid Intensification (RI)** modules.

In accordance with strict project guardrails:
1. **Zero Training Mandate:** No models are trained, fine-tuned, or generated in this phase.
2. **Contract Preservation:** The existing 101-feature ocean contract and 224-feature V2.3 ocean-atmosphere contract are preserved without modification.
3. **Event-Grouped Splitting:** No storm system may appear partially in training and partially in validation or testing.
4. **Separation of Readiness Status:** Cyclone Intensity and Rapid Intensification are evaluated independently.

---

## 1. Feature Reusability Audit (Tasks 10 & 11)

### A. Ocean Features Audit (Existing 101-Feature Contract)
The existing 101 ocean features in `features_10yr.parquet` were designed for basin/site-level cyclone genesis preconditioning. For cyclone intensity and RI, their physical relevance is classified into four categories:

```
OCEAN FEATURE CLASSIFICATION (101-FEATURE CONTRACT)
┌───────────────────────┬─────────────────────────────────────────────────────────────┐
│ Classification Tier   │ Variables & Physical Rationale                              │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ DIRECTLY REUSABLE     │ temp_current (SST), temp_abs_anom, temp_zscore.             │
│                       │ Physical thermodynamic fuel supplying latent heat to core.  │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ POTENTIALLY REUSABLE  │ mld_current, mld_abs_anom, ssh_current (SLA).               │
│                       │ Deep mixed layer and high sea surface height indicate thick │
│                       │ warm water lens that resists cyclonic cold-wake upwelling.  │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ HAZARD-SPECIFIC       │ Tropical Cyclone Heat Potential (TCHP = rho*cp*int(T-26)dz),│
│ (REQUIRES DERIVATION) │ Inner-core SST cooling anomaly (cold-wake feedback).        │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ NOT APPROPRIATE       │ Site-fixed salinity rolling trends (sal_30d_std), local     │
│                       │ fixed-point current shear at fixed buoys hundreds of km away│
└───────────────────────┴─────────────────────────────────────────────────────────────┘
```

### B. Atmospheric Features Audit (Existing ERA5 Contract)
The existing atmospheric fields in `features_atmosphere_10yr_daily.parquet` and raw NetCDF archives are classified for intensity modeling:

```
ATMOSPHERIC FEATURE CLASSIFICATION (ERA5 CONTRACT)
┌───────────────────────┬─────────────────────────────────────────────────────────────┐
│ Classification Tier   │ Variables & Physical Rationale                              │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ DIRECTLY REUSABLE     │ min_vws, mean_vws (Vertical Wind Shear 850-200 hPa).        │
│                       │ Strong shear (>20 kt) tears vortex apart; low shear (<10 kt)│
│                       │ is a prerequisite for Rapid Intensification.                │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ DIRECTLY REUSABLE     │ max_vorticity, mean_vorticity (850 hPa relative vorticity). │
│                       │ Measures low-level spin-up and angular momentum convergence.│
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ DIRECTLY REUSABLE     │ mean_rh_700, mean_rh_500 (Mid-tropospheric humidity).       │
│                       │ Dry-air intrusion into inner core induces downdrafts and    │
│                       │ arrests rapid intensification.                              │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ POTENTIALLY REUSABLE  │ div_200 (Upper-level divergence / outflow efficiency).      │
│                       │ Dual outflow channels accelerate intensification.           │
├───────────────────────┼─────────────────────────────────────────────────────────────┤
│ HAZARD-SPECIFIC       │ Storm-centered azimuthally averaged radial shear;           │
│ (REQUIRES DERIVATION) │ Environmental steering flow vectors (850-200 hPa mean wind).│
└───────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 2. Spatial Representation Architecture (Task 12)

Six candidate spatial representations were evaluated for associating ocean-atmosphere features with moving tropical cyclones:

```
SPATIAL REPRESENTATION EVALUATION
┌───────────────────────────┬──────────────┬──────────────┬───────────────────────────────────────────┐
│ Representation Method     │ Resolution   │ Complexity   │ Scientific & Operational Assessment       │
├───────────────────────────┼──────────────┼──────────────┼───────────────────────────────────────────┤
│ 1. Nearest Fixed Site     │ ~300 - 500 km│ Very Low     │ REJECTED: Unacceptable spatial errors;    │
│    (AS1-6 / BOB1-6)       │              │              │ misses local eyewall ocean enthalpy.      │
├───────────────────────────┼──────────────┼──────────────┼───────────────────────────────────────────┤
│ 2. Basin-Wide Aggregate   │ ~1,000 km    │ Very Low     │ REJECTED: Averages out local shear and    │
│    (Whole BoB / Arabian S)│              │              │ localized warm ocean eddies.              │
├───────────────────────────┼──────────────┼──────────────┼───────────────────────────────────────────┤
│ 3. Storm Center Point     │ 0.25 deg     │ Low          │ ACCEPTABLE BASELINE: Extracts grid cell   │
│    Extraction             │ (~28 km)     │              │ containing storm center (lat, lon).       │
├───────────────────────────┼──────────────┼──────────────┼───────────────────────────────────────────┤
│ 4. Storm-Centered Annular │ Multi-ring   │ Medium       │ RECOMMENDED SCIENTIFIC STANDARD:          │
│    Radius Extraction      │ (0-100, 200- │              │ Inner ring (0-100 km) for SST/core;       │
│    (SHIPS-style rings)    │  800 km)     │              │ Outer ring (200-800 km) for shear/moisture│
└───────────────────────────┴──────────────┴──────────────┴───────────────────────────────────────────┘
```

---

## 3. Event-Grouped Chronological Splitting Strategy (Tasks 12 & 14)

### The Anti-Leakage Event-Grouping Rule
Sequential synoptic fixes from the same cyclone exhibit strong temporal autocorrelation. Random row-level splitting leaks storm-specific structural information across folds, generating artificially optimistic validation metrics.

**Mandatory Splitting Rule:** *Every storm system ($N=117$) must belong entirely to a single partition.*

```
EVENT-GROUPED CHRONOLOGICAL PARTITIONS
┌────────────────────────┬─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Partition Set          │ Calendar Span       │ Unique Storms│ Total Fixes  │ RI Storms    │
├────────────────────────┼─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Training Set           │ 2016-05 to 2021-12  │ 74 storms    │ 1,595 fixes  │ 14 storms    │
│ Validation Set         │ 2022-01 to 2023-12  │ 22 storms    │   528 fixes  │  4 storms    │
│ Final Holdout Test Set │ 2024-01 to 2026-06  │ 21 storms    │   421 fixes  │  0 storms*   │
└────────────────────────┴─────────────────────┴──────────────┴──────────────┴──────────────┘
*Note on Zero RI: The recent 2024-2026 holdout contains zero standard RI storms. See Section 4.
```

### Leave-One-Season-Out Cross-Validation (Alternative for RI)
Because the 2024–2026 final holdout has zero positive standard RI events, testing Rapid Intensification requires **Leave-One-Season-Out (LOSO) Cross-Validation** across the 8 active seasons (2016–2023). Each season is systematically held out as a blind test set while the remaining 9 seasons serve as training data.

---

## 4. Remediation Strategy for the "Zero-RI" Holdout Problem (Task 15)

In our forensic audit, the 2024–2026 period contains 21 unique storms and 421 synoptic fixes, but **zero storms achieved standard Rapid Intensification ($\ge 30\text{ kt} / 24\text{h}$)**:

$$\text{RI Positives in } [2024, 2026] = 0 \quad (N_{\text{pos}} = 0)$$

### Operational Governance Rules for the Zero-RI Test Set:
1. **No Metric Fabrication:** Under zero positive test examples, Precision-Recall AUC ($PR\text{-}AUC$) and Critical Success Index ($CSI$) cannot be evaluated. The evaluation pipeline must report:
   $$\mathbf{INSUFFICIENT\_EVIDENCE\_FOR\_RI\_TESTING}$$
2. **Evaluation of False Alarm Rate Only:** The 2024–2026 holdout set remains valid for evaluating **False Alarm Ratio (FAR)** and **Specificity** (verifying that the model does not trigger false RI alerts during quiet, non-intensifying seasons).
3. **Primary RI Benchmark on Active Years:** Formal RI evaluation must be performed on holdout seasons from 2018–2023 that contain verified RI events (e.g., holding out the 2019 season with 6 RI storms, or 2023 with 4 RI storms).

---

## 5. Class Imbalance Remediation Architecture (Task 17)

Across 1,907 candidate 24h pairs, standard RI ($\ge 30\text{ kt} / 24\text{h}$) occurs in only 103 pairs (**$5.40\%$ prevalence**, imbalance ratio ~17.5 to 1).

### Permissible Imbalance Strategies for Future Training:
1. **Class-Weighted Cross-Entropy Loss:** Weight positive RI errors proportionally to inverse class prevalence:
   $$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 17.5$$
2. **Focal Loss Formulation:** Suppresses gradients from well-classified negative examples ($\gamma = 2.0$, $\alpha = 0.75$):
   $$\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
3. **Threshold-Independent Optimization:** Optimize hyperparameters on **Precision-Recall AUC ($PR\text{-}AUC$)** and **Brier Skill Score ($BSS$)** rather than raw classification accuracy.

---

## 6. Validation Metrics Hierarchy (Task 18)

Validation metrics must be frozen prior to model execution:

```
METRIC HIERARCHY
┌────────────────────────┬───────────────────────────────────┬────────────────────────────────────────┐
│ Hazard Module          │ Primary Benchmark Metric          │ Secondary Operational Metrics          │
├────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ Cyclone Intensity      │ Mean Absolute Error (MAE on Vmax) │ - Root Mean Squared Error (RMSE)       │
│ (Continuous Regressor) │ Target: < 11.5 kt at 24h          │ - Systematic Intensity Bias (knots)    │
│                        │                                   │ - MAE by Grade (D, CS, VSCS, SuCS)     │
├────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ Rapid Intensification  │ 1. Precision-Recall AUC (PR-AUC)  │ - Brier Score & Expected Calib. Error  │
│ (Binary Classifier)    │ 2. Critical Success Index (CSI)   │ - Precision, Recall, F1-Score          │
│                        │ Target: CSI > 0.35 at 24h         │ - False Alarm Ratio (FAR < 0.45)       │
│                        │                                   │ - Storm-Level Detection Rate           │
└────────────────────────┴───────────────────────────────────┴────────────────────────────────────────┘
```

---

## 7. Pre-Training Dataset Manifest Specification (Task 21)

Prior to initiating any future model training, an immutable JSON manifest (`cyclone_intensity_dataset_manifest.json`) must be permanently committed:

```json
{
  "manifest_id": "SD-MANIFEST-2026-CYCLONE-INTENSITY-CANDIDATE-V1",
  "created_at": "2026-09-13T00:00:00Z",
  "provenance_hashes": {
    "imd_tracks_sha256": "4f1cf8bf463b2241cfd14e9fdfcf0773d22b640ce970034a17fe68eb7ca0fefc",
    "features_ocean_sha256": "cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867",
    "features_era5_sha256": "551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864"
  },
  "dataset_specifications": {
    "total_raw_fixes": 2544,
    "excluded_duplicate_rows": 8,
    "excluded_outlier_rows": 1,
    "valid_candidate_fixes": 2535,
    "total_unique_storms": 117
  },
  "target_definitions": {
    "intensity_primary": "Vmax(T+24h) in knots, tolerance +/- 3.0h",
    "ri_primary_provisional": "Delta_Vmax >= 30.0 kt / 24h, tolerance +/- 3.0h",
    "ri_secondary_sensitivity_25": "Delta_Vmax >= 25.0 kt / 24h",
    "ri_secondary_sensitivity_20": "Delta_Vmax >= 20.0 kt / 24h"
  },
  "temporal_alignment_rule": "Strict causal cutoff at T; 6-hourly atmospheric slices matching synoptic fix hour",
  "spatial_alignment_rule": "Storm-centered annular extraction (0-100km core, 200-800km environment)",
  "partitions": {
    "train_storm_ids": ["IMD-2016-1-ROANU", "...", "IMD-2021-10-JAWAD"],
    "val_storm_ids": ["IMD-2022-1-ASANI", "...", "IMD-2023-9-MICHAUNG"],
    "test_storm_ids": ["IMD-2024-1-REMAL", "...", "IMD-2026-1-TENTATIVE"]
  },
  "zero_positive_test_handling": "INSUFFICIENT_EVIDENCE_FOR_RI_TESTING flag active for 2024-2026 holdout"
}
```
