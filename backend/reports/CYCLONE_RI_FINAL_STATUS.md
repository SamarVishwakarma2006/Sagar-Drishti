# Sagar-Drishti — Rapid Intensification (RI) Final Status Report
## FORENSIC DATA SUFFICIENCY EVALUATION — ZERO ML TRAINING

**Document ID:** `SD-REPORT-2026-CYCLONE-RI-FINAL-STATUS-V2`  
**Classification:** `RESEARCH_ONLY` | `NOT_FOR_PRODUCTION` | `NO_OPERATIONAL_AUTHORITY`  
**Date:** September 13, 2026  
**Hazard Module:** Tropical Cyclone Rapid Intensification (RI)  
**Evaluated Primary Definition:** $\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$  
**Final Scientific Verdict:** **`DATA_NOT_READY`**

---

## 1. Scientific Definition & Governance (Section 9)

### Primary Research Definition
The primary definition of Rapid Intensification for Sagar-Drishti is established as:

$$\Delta V_{\max} = V_{\max}(T + 24\text{h}) - V_{\max}(T) \ge 30\text{ kt} \quad (\approx 15.4\text{ m/s})$$

evaluated across synoptic fix intervals with temporal tolerance $24\text{h} \pm 3\text{h}$ ($[21.0\text{h}, 27.0\text{h}]$).

> [!NOTE]
> **SCIENTIFIC NOMENCLATURE & JURISDICTIONAL CONTEXT**  
> $30\text{ kt}/24\text{h}$ is a defensible primary RI definition for Sagar-Drishti because it is explicitly used by IMD for the North Indian Ocean and is also used operationally by NOAA/NHC. It must **not** be described as a "universal WMO-mandated threshold", as WMO regional associations accommodate differing regional convective baselines. Furthermore, for the North Indian Ocean basin, $30\text{ kt}/24\text{h}$ corresponds to approximately the **93rd percentile** of 24-hour intensity changes in IMD Best Track climatology, rather than the 95th percentile typical of the Atlantic basin.

### Secondary Sensitivity Benchmarks
To evaluate model sensitivity across lower intensification thresholds, two secondary benchmarks are defined:
1. **$25\text{ kt} / 24\text{h}$ ($\approx 12.9\text{ m/s}$):** Captures borderline rapid intensification and moderate strengthening in short-lived monsoon systems.
2. **$20\text{ kt} / 24\text{h}$ ($\approx 10.3\text{ m/s}$):** Captures standard tropical storm intensification.

> [!CAUTION]
> **PROHIBITION AGAINST THRESHOLD LOWERING**  
> These secondary thresholds are exploratory sensitivity benchmarks only. Lowering the operational definition of Rapid Intensification below $30\text{ kt}/24\text{h}$ merely to inflate positive sample counts is scientifically unacceptable and strictly prohibited.

---

## 2. Quantitative Sample Size & Temporal Dependence Audit (Section 10)

Across the full 10-year historical IMD Best Track dataset (2016–2026, 2,544 raw fixes, 117 unique storms):

```
RAPID INTENSIFICATION HISTORICAL PREVALENCE (2016–2026)
┌────────────────────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Metric                         │ Primary (30 kt) │ Secondary (25 kt│ Secondary (20 kt│
├────────────────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Total Candidate 24h Fix Pairs  │ 1,907 pairs     │ 1,907 pairs     │ 1,907 pairs     │
│ Positive Fix Pairs (Baseline)  │ 103 pairs*      │ 154 pairs       │ 249 pairs       │
│ Fix-Pair Level Prevalence      │ 5.40%           │ 8.08%           │ 13.06%          │
│ Unique Positive Storms         │ 18 storms       │ 24 storms       │ 37 storms       │
│ Storm-Level Prevalence         │ 15.38% (18/117) │ 20.51% (24/117) │ 31.62% (37/117) │
│ Positive Active Seasons (Years)│ 5 seasons       │ 6 seasons       │ 8 seasons       │
└────────────────────────────────┴─────────────────┴─────────────────┴─────────────────┘
*103 positive pairs when evaluating all overlapping 21-27h fix intervals across multi-day RI episodes.
```

### The Effective Sample Size Reality
- The 103 positive fix pairs occur across only **18 unique storms**.
- On average, each positive storm contributes $\approx 5.7$ overlapping positive fix pairs throughout its intensification phase (e.g., Super Cyclonic Storm AMPHAN and Extremely Severe Cyclonic Storm FANI each generated 10+ consecutive overlapping positive 24h pairs as they intensified across multiple synoptic updates).
- **Temporal Dependence Rule:** Fix-level observations within the same cyclone episode are strongly autocorrelated. **103 positive pairs do NOT represent 103 independent statistical events.** Statistically, the effective sample size is bounded by the 18 independent storm events.

### Annual Distribution of Positive RI Storms ($\ge 30\text{ kt}/24\text{h}$)
- **2016:** 0 RI storms
- **2017:** 0 RI storms
- **2018:** 2 RI storms (*TITLI*, *MEKUNU*)
- **2019:** 6 RI storms (*FANI*, *VAYU*, *HIKKA*, *KYAAR*, *MAHA*, *BULBUL*)
- **2020:** 2 RI storms (*AMPHAN*, *GATI*)
- **2021:** 1 RI storm (*TAUKTAE*)
- **2022:** 0 RI storms
- **2023:** 4 RI storms (*MOCHA*, *BIPARJOY*, *TEJ*, *HAMOON*)
- **2024:** 0 RI storms
- **2025:** 0 RI storms
- **2026 (Jan):** 0 RI storms

---

## 3. The "Zero-RI" Holdout Blocker (2024–2026)

In the independent chronological test partition (2024-01-01 through 2026-01-10):
- **Total Storms in Test Set:** 29 storms
- **Total Fixes in Test Set:** 541 fixes
- **Total 24h Fix Pairs:** 396 pairs
- **Positive RI Events ($\ge 30\text{ kt}/24\text{h}$):** **EXACTLY 0 ($N_{\text{pos}} = 0$)**

### Methodological & Scientific Implications
1. **Mathematical Impossibility of Positive Evaluation:**
   When $N_{\text{pos}} = 0$, True Positives ($TP$) must equal 0 regardless of model predictions:
   $$\text{Recall} = \frac{TP}{TP + FN} = \frac{0}{0} \implies \text{UNDEFINED}$$
   $$\text{Precision-Recall AUC (PR-AUC)} \implies \text{MATHEMATICALLY UNDEFINED}$$
   $$\text{Critical Success Index (CSI)} = \frac{TP}{TP + FP + FN} = \frac{0}{0 + FP + 0} = 0.0$$
2. **Prohibition of Metric Fabrication:**
   Reporting a fabricated or interpolated PR-AUC on a test partition with zero positive cases is scientifically invalid. Any evaluation pipeline must emit:
   $$\mathbf{UNDEFINED \ /\ INSUFFICIENT\_EVIDENCE}$$
3. **Negative-Side Analysis Only:**
   The 2024–2026 partition can legitimately be used solely for evaluating **False Alarm Ratio (FAR)**, **Specificity**, and negative class calibration (verifying that the system does not issue spurious RI alarms during non-RI seasons).
4. **Prohibition of Split Manipulation:**
   Moving test boundaries (e.g., cutting the test set to 2023 to capture Mocha) simply to artificially generate positive test cases violates chronological testing integrity and compromises future generalization testing.

---

## 4. Requirements to Clear the RI Readiness Gate

Rapid Intensification cannot legitimately advance to `TRAINING_READY` until the following empirical conditions are satisfied:
1. **Accumulation of Independent Test Positives:** The holdout test window must encompass at least 2–3 independent positive RI storms to allow well-defined, non-degenerate calculation of PR-AUC, CSI, and Recall.
2. **Leave-One-Season-Out (LOSO) Cross-Validation Framework:** Prior to holdout evaluation, models must be cross-validated across historical seasons containing verified RI activity (2018, 2019, 2020, 2021, 2023) using rigorous storm-grouped folding.
3. **Inner-Core Dynamic Feature Extraction:** High-frequency (6-hourly) storm-centered 0–100 km SST cooling and 200–800 km environmental wind shear must be extracted to provide physical predictors of rapid vortex spin-up.

---

## 5. Final Verdict & Governance Sign-Off

```
============================================================
RAPID INTENSIFICATION (RI) READINESS GATE VERDICT
============================================================
Module Status:                 DATA_NOT_READY
Primary Reason:                Zero Positive Events in 2024–2026 Test Partition
Effective Positive Count:      Only 18 independent storms across 10 years
Mathematical Feasibility:      PR-AUC and CSI are undefined on holdout test set
Operational Authority:         NONE (Research Only)
ML Training Conducted:         ZERO
============================================================
```

**Sign-off:** The evidence confirms that Rapid Intensification module must remain **`DATA_NOT_READY`**.
