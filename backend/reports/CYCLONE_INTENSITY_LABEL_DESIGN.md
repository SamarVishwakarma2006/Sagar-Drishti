# Sagar-Drishti — Cyclone Intensity & Rapid Intensification: Label Design Specification

**Document ID:** `SD-SPEC-2026-INTENSITY-LABEL-DESIGN`  
**Phase:** Pre-Training Forensic Data & Scientific Gate  
**Date:** September 13, 2026  
**Status:** **[LABEL DESIGN COMPLETE — ZERO TRAINING CONDUCTED]**  
**Classification:** Target Formulation & Meteorological Label Specification  

---

## Executive Summary

This specification establishes the mathematical, temporal, and physical definitions of target variables for the future **Cyclone Intensity** and **Rapid Intensification (RI)** modules of Sagar-Drishti.

In accordance with strict meteorological physics:
1. **Target Independence:** Cyclone intensity ($V_{\max}$) and pressure drop ($\Delta P$) measure physical vortex strength and are mathematically decoupled from Cyclone Genesis ($P_{\text{genesis}}$).
2. **Dual-Level Metric Reporting:** All RI evidence and performance numbers must be reported simultaneously at the **Fix-Pair / Row Level** and the **Unique-Storm Level**.
3. **No Interpolation Artifacts:** Targets must be derived from discrete synoptic fixes within a formalized temporal tolerance window, strictly prohibiting forward interpolation.

---

## 1. Cyclone Intensity Target Family (Task 4)

Cyclone intensity represents the thermodynamic and dynamic vigor of the cyclonic vortex. In the North Indian Ocean, IMD RSMC New Delhi measures surface intensity using a **3-minute sustained wind speed average** ($V_{\max}$, knots) and **central sea-level pressure** ($P_c$, hPa).

```
INTENSITY TARGET FAMILY
┌──────────────────────┬───────────────────────────────┬──────────────┬──────────────────────────────┐
│ Target Identifier    │ Mathematical Definition       │ Unit         │ Recommended ML Formulation   │
├──────────────────────┼───────────────────────────────┼──────────────┼──────────────────────────────┤
│ Vmax_24h (PRIMARY)   │ V_max(T + 24h)                │ knots (kt)   │ Continuous Gradient Boosting │
│ Vmax_48h             │ V_max(T + 48h)                │ knots (kt)   │ Continuous Gradient Boosting │
│ Vmax_72h             │ V_max(T + 72h)                │ knots (kt)   │ Continuous Gradient Boosting │
│ Pc_24h (SECONDARY)   │ P_c(T + 24h)                  │ hPa          │ Multi-Task Joint Regressor   │
│ DeltaP_24h           │ 1010.0 - P_c(T + 24h)         │ hPa          │ Continuous Regression        │
│ Grade_24h            │ Categorical Grade (D to SuCS) │ Ordinal (1-7)│ Ordinal Classification       │
└──────────────────────┴───────────────────────────────┴──────────────┴──────────────────────────────┘
```

### Detailed Scientific Target Specifications

#### Target 1.1: Maximum Sustained Wind Speed $V_{\max}(T+H)$ (Primary Target)
- **Mathematical Definition:**
  $$Y_{V_{\max}}(T+H) = V_{\max,\text{IMD}}(t^*)$$
  where $t^*$ is the verified IMD synoptic fix matching initialization time $T$ shifted by lead time $H \in \{24\text{h}, 48\text{h}, 72\text{h}\}$ within a defined tolerance.
- **Physical Meaning:** Direct indicator of surface kinetic energy and destructive wind hazard ($E_k \propto V_{\max}^2$).
- **Label Source:** `max_wind_kts` in `imd_tracks_2016_2026.parquet` (IMD RSMC official best track).
- **Target Distribution:** Continuous range from $20.0\text{ kt}$ to $130.0\text{ kt}$ (median $30.0\text{ kt}$, mean $40.9\text{ kt}$).
- **Validation Metrics:** Mean Absolute Error ($MAE$, target $<12\text{ kt}$ at 24h), Root Mean Squared Error ($RMSE$), Systematic Bias ($\bar{e} = \frac{1}{N}\sum (\hat{y} - y)$).

#### Target 1.2: Central Sea-Level Pressure $P_c(T+H)$ and Pressure Drop $\Delta P(T+H)$
- **Mathematical Definition:**
  $$Y_{P_c}(T+H) = P_{c,\text{IMD}}(t^*)$$
  $$\Delta P(T+H) = P_{\text{ambient}} - P_{c,\text{IMD}}(t^*)$$
  where $P_{\text{ambient}} \approx 1010.0\text{ hPa}$ (standard tropical North Indian Ocean ambient pressure).
- **Physical Meaning:** Governs the barometric storm surge potential via the inverse barometer effect (~1 cm rise per 1 hPa drop).
- **Recommendation:** Train as a **secondary multi-task output** alongside $V_{\max}$. Because $V_{\max}$ and $\Delta P$ are physically bound through cyclostrophic balance ($V_{\max} \approx C \sqrt{\Delta P}$), joint prediction regularizes neural and tree representations.

#### Target 1.3: IMD Cyclone Category / Grade (Ordinal Classification)
- **Categories:**
  1. Depression (D: 17–27 kt)
  2. Deep Depression (DD: 28–33 kt)
  3. Cyclonic Storm (CS: 34–47 kt)
  4. Severe Cyclonic Storm (SCS: 48–63 kt)
  5. Very Severe Cyclonic Storm (VSCS: 64–89 kt)
  6. Extremely Severe Cyclonic Storm (ESCS: 90–119 kt)
  7. Super Cyclonic Storm (SuCS: $\ge 120\text{ kt}$)
- **Recommendation:** Unsuitable as a primary target due to arbitrary category boundaries ($33\text{ kt}$ vs $34\text{ kt}$ splits an operational threshold). Best derived post-hoc from continuous $\hat{V}_{\max}$.

---

## 2. Rapid Intensification (RI) Target Design (Task 5)

Rapid Intensification represents the extreme upper-tail of tropical cyclone development, defined as an intensification rate exceeding the 95th percentile of all synoptic changes.

### Primary Provisional Target Definition
$$\Delta V_{24} = V_{\max}(t^*) - V_{\max}(T)$$

$$Y_{\text{RI}, 30}(T, 24\text{h}) = \begin{cases} 1 & \text{if } \Delta V_{24} \ge 30.0\text{ knots} \\ 0 & \text{if } \Delta V_{24} < 30.0\text{ knots} \end{cases}$$

- **Status:** `is_provisional = True` (Standard WMO / IMD operational criterion).
- **Row-Level Prevalence:** 103 positive pairs / 1,907 total pairs = **$5.40\%$**.
- **Event-Level Prevalence:** 18 positive storms / 117 total storms = **$15.38\%$**.

### Secondary Research Sensitivity Definitions
To assess statistical sensitivity to threshold choice, two secondary definitions are designated:
1. **Moderate Rapid Intensification ($\ge 25\text{ kt} / 24\text{h}$):**
   - Row-Level: 150 positive pairs / 1,907 pairs = **$7.87\%$**
   - Event-Level: 25 unique storms / 117 storms = **$21.37\%$**
2. **Marginal Intensification ($\ge 20\text{ kt} / 24\text{h}$):**
   - Row-Level: 253 positive pairs / 1,907 pairs = **$13.27\%$**
   - Event-Level: 37 unique storms / 117 storms = **$31.62\%$**

> [!WARNING]
> **Strict Scientific Mandate:** The primary operational RI threshold must remain strictly at $\mathbf{30\text{ kt} / 24\text{h}}$. Secondary thresholds ($25\text{ kt}$ and $20\text{ kt}$) must **never** be substituted as the primary operational definition simply to inflate the training sample size.

---

## 3. Temporal Tolerance & Selection Rules (Task 6)

In the real-world operational IMD Best Track dataset, fixes are recorded at 3-hourly intervals during active stages and 6-hourly intervals during depressions. Because consecutive fixes are not spaced at continuous infinitesimal increments, calculating a 24-hour change requires an explicit **Temporal Tolerance Window**:

```
TEMPORAL TOLERANCE WINDOW FOR 24-HOUR TARGET MATCHING
Initialization Time: T (e.g., 00:00 UTC)
Ideal Target Time:   T + 24.0 hours (e.g., next day 00:00 UTC)
Admissible Window:   [T + 21.0 hours, T + 27.0 hours]  (Tolerance: Delta_t = +/- 3.0h)
```

### Deterministic Selection Algorithm
1. Query all synoptic fixes belonging to the identical `system_id` within the time interval:
   $$t \in [T + 21\text{h}, T + 27\text{h}]$$
2. If no fix exists within $[T + 21\text{h}, T + 27\text{h}]$ (e.g., the storm made landfall or dissipated prior to 21h):
   - The sample is marked as `TARGET_TRUNCATED_LIFECYCLE` and excluded from the 24h change target.
3. If one or more candidate fixes exist:
   - Calculate the absolute time difference to exact 24 hours: $|\delta t| = |t - (T + 24\text{h})|$.
   - Select the fix $t^*$ that minimizes $|\delta t|$.
   - If two candidate fixes have identical absolute distance (e.g., $T+21\text{h}$ and $T+27\text{h}$), **deterministically select the earlier fix ($T+21\text{h}$)**.
4. **Zero Future Interpolation Rule:** Linear or spline interpolation between $T+18\text{h}$ and $T+30\text{h}$ to synthesize an artificial $T+24\text{h}$ value is **strictly prohibited**. Interpolation introduces synthetic smoothing, alters peak winds, and violates empirical best-track observation fidelity.

---

## 4. Multi-Horizon Feasibility Analysis: +24h, +48h, +72h (Tasks 3 & 7)

```
TARGET HORIZON DATA AVAILABILITY
Horizon     Candidate Pairs    RI Positive Pairs (>=30 kt)    Target Feasibility Status
─────────────────────────────────────────────────────────────────────────────
+24 hours        1,907                    103                  FEASIBLE (Primary Scope)
+48 hours        1,307               [NOT FINALIZED]           FEASIBLE FOR VMAX; RI AMBIGUOUS
+72 hours          856               [NOT FINALIZED]           FEASIBLE FOR VMAX; RI AMBIGUOUS
─────────────────────────────────────────────────────────────────────────────
```

### The 48h and 72h Rapid Intensification Dilemma (Task 7)
In atmospheric literature, "Rapid Intensification" is almost universally defined over a 24-hour window. Extending RI to 48 hours or 72 hours introduces two conflicting mathematical formulations:
1. **Cumulative Intensification Formulation:** $\Delta V_{48} \ge 45\text{ kt} / 48\text{h}$ (Kaplan & DeMaria criterion).
2. **Windowed Sub-Interval Formulation:** $\exists \, \tau \in [T, T+48\text{h}]$ such that $\Delta V_{24}(\tau) \ge 30\text{ kt}$.

Because no official WMO or IMD consensus exists for the North Indian Ocean regarding 48h/72h RI thresholds, **the 48h and 72h RI targets are officially designated as:**

$$\mathbf{TARGET\_NOT\_FINALIZED}$$

Continuous intensity prediction ($V_{\max}(T+48\text{h})$ and $V_{\max}(T+72\text{h})$) remains scientifically defensible and feasible, but multi-day RI classification must not be trained without prior international consensus.

---

## 5. Summary Table: Label Design Specifications

| Dimension | Intensity Regression Target | Rapid Intensification (RI) Target |
| :--- | :--- | :--- |
| **Primary Variable** | Maximum 3-min sustained wind $V_{\max}$ | Binary indicator of 24h change $\ge 30\text{ kt}$ |
| **Formula** | $\hat{V}_{\max}(T+24\text{h})$ | $\mathbf{1}(V_{\max}(t^*) - V_{\max}(T) \ge 30\text{ kt})$ |
| **Units** | Knots (kt) | Probability $P(\text{RI}) \in [0, 1]$ |
| **Temporal Tolerance** | $\pm 3.0\text{ hours}$ centered on $T+24\text{h}$ | $\pm 3.0\text{ hours}$ centered on $T+24\text{h}$ |
| **Secondary Horizons** | $+48\text{h}$ (1,307 pairs), $+72\text{h}$ (856 pairs) | Marked `TARGET_NOT_FINALIZED` for $\ge 48\text{h}$ |
| **Authoritative Source** | IMD RSMC Official Best Track | IMD RSMC Official Best Track |
| **Provisional Status** | Final operational metric | `is_provisional = True` |
| **Handling of Missing Target**| Excluded if storm dissipates $<21\text{h}$ | Excluded if storm dissipates $<21\text{h}$ |
