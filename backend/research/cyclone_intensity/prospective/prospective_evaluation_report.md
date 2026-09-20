# Sagar-Drishti — Cyclone Intensity V1
## Final Prospective / Longitudinal Evaluation & Forensic Readiness Report
**Document ID:** `SD-REPORT-2026-INTENSITY-PROSPECTIVE-FORENSIC-V2`  
**Generated At (UTC):** `2026-09-13T18:40:00Z`  
**Engineering Status:** **`READY_FOR_REAL_PROSPECTIVE_DATA`**  
**Scientific Generalization Status:** **`NOT_YET_ESTABLISHED`**  
**Production Status:** **`NOT_AUTHORIZED`**  
**Classification:** **RESEARCH-ONLY / ZERO PRODUCTION MODIFICATION / NO OPERATIONAL AUTHORITY**  

---

## 1. Executive Summary & Forensic Posture

This report delivers the hardened scientific and architectural specification of the **Forward Prediction / Prospective Inference Mode** for the frozen Cyclone Intensity Model (`EXP-E-CORE-ENVIRONMENT-RADIAL-CONTRASTS`, XGBoost continuous intensity regressor, 29 features).

The system enforces strict causal firewalls, data availability contracts, data provenance classifications, target arrival lifecycles, and baseline comparisons.

**CRITICAL SCIENTIFIC POSTURE:**
- Computational causality and pipeline integrity do **NOT** constitute scientific validation.
- Genuine prospective generalization cannot be established using synthetic test fixtures or retrospective historical replays.
- **Genuine Prospective Cyclone Systems Evaluated to Date:** `0`
- **Prospective Scientific Evidence:** `NONE`

---

## 2. Four-Tier Separation of Capabilities & Evidence

To prevent retrospective bias, conflation of synthetic tests with scientific proof, or false claims of real-world generalization, the system enforces a strict four-way separation:

### A. Engineering Capability (What the System Technically Does)
- **Dual Causal Firewall:** Enforces $T_{\text{obs}} \le T_{\text{origin}}$ and $T_{\text{avail}} \le T_{\text{origin}}$ for all input predictors. Rejects filesystem metadata (`mtime`, `ctime`).
- **Feature Contract Assembly:** Assembles the frozen 29-feature vector in exact column order with native continuous values and handles missingness via native XGBoost decision routing splits without synthetic imputation.
- **Frozen Model Execution:** Reads `final_intensity_model.joblib` with read-only permissions after verifying SHA-256 byte invariance.
- **Physical Output Guardrails:** Enforces operational physical bounds $[15.0, 165.0]\text{ kt}$ on model predictions.
- **Append-Only Immutability:** Logs all predictions immutably to `forecast_log.parquet`. Prevents overwrite or retroactive alteration of primary predictions.
- **Honest Target Lifecycle:** Tracks targets through `TARGET_PENDING`, `TARGET_AVAILABLE`, `TARGET_UNAVAILABLE`, and `TARGET_EXCLUDED`.

### B. Synthetic Validation (Mechanistic Testing via Artificial Fixtures)
- **Synthetic Test Scenarios:** Validated using synthetic fixtures (`DEMO-BOB-2026-07-10`, `DEMO-ARAS-2026-08-15`).
- **Data Provenance Tag:** All synthetic runs carry `data_provenance_class = SYNTHETIC_TEST_FIXTURE` and `fixture_label = "SYNTHETIC TEST FIXTURE — NOT REAL METEOROLOGICAL DATA"`.
- **Target Exclusion:** Synthetic fixes are automatically classified as `TARGET_EXCLUDED` (`SYNTHETIC_FIXTURE_EXCLUDED_FROM_SCIENTIFIC_EVALUATION`).
- **Scientific Quarantine:** Synthetic fixture records are **STRICTLY EXCLUDED** from scientific accuracy metrics (MAE, RMSE, bias).

### C. Historical Research Validation (Established Partition Performance)
The authoritative research partition boundaries are permanently canonicalized across the repository:
- **TRAIN Partition (2016–2021):** 64 storms, 1,121 targets (Training and cross-validation)
- **VALIDATION Partition (2022–2023):** 24 storms, 367 targets (Model selection and radial feature optimization)
- **TEST Partition (2024–2026 Holdout):** 29 storms, 552 targets (Single-pass frozen audit, `TEST_ACCESS_COUNT = 1`)
- **Holdout Quarantined Result:** Test MAE = 8.65 kt. The holdout partition remains sealed and immutable.

### D. Genuine Prospective Evidence (Post-Historical Field Observations)
```text
GENUINE_PROSPECTIVE_STORMS = 0
PROSPECTIVE_SCIENTIFIC_EVIDENCE = NONE
```
- **Live Post-Historical Systems Ingested:** 0
- **Valid Prospective Targets Evaluated:** 0
- No scientific claim of prospective skill, generalization, or superiority over operational guidance is made.

---

## 3. Critical Target Evaluation Rules — Zero Error Prohibited

In prior revisions, `TARGET_UNAVAILABLE` records were erroneously assigned zero error. **That evaluation rule was scientifically invalid and has been completely eliminated.**

Under the hardened evaluation framework:
1. **No Artificial Zero Error:** When a target is unavailable, the prediction error is **NOT** zero.
2. **Excluded from Denominators:** `TARGET_UNAVAILABLE` and `TARGET_EXCLUDED` records are strictly excluded from MAE, RMSE, and bias denominators.
3. **No Phantom Skill:** Missing targets are neither counted as correct predictions nor penalized as model failures; they are isolated as data collection dropouts.
4. **Denominator Stated Explicitly:** All scientific metrics state their evaluated-target denominator:
   $$\text{Storm MAE} = \frac{\sum_{i=1}^{N_{\text{eval}}} |\hat{V}_i - V_i|}{N_{\text{valid\_evaluated\_targets}}}$$

### Prospective Target Accounting
```text
n_forecasts: 0
n_target_available: 0
n_target_pending: 0
n_target_unavailable: 0
n_target_excluded: 0
target_coverage_rate: N/A
evaluated_target_rate: N/A
```

---

## 4. Real-World Data Provenance Contract

Every forward prediction and target record carries an explicit `data_provenance_class`:
- `REAL_PROSPECTIVE`: Real observations collected before origin $T$ where forecast was logged before target availability. Only this class may contribute to prospective accuracy metrics.
- `SYNTHETIC_TEST_FIXTURE`: Synthetic demonstration data. Excluded from scientific evaluation.
- `HISTORICAL_REANALYSIS`: Data derived from retrospective reanalysis (e.g. ERA5) or historical test sets. Excluded from prospective metrics.
- `UNKNOWN`: Unverified provenance. Excluded from scientific metrics.

---

## 5. Operational Atmospheric Source Contract

Due to publication latency, **ERA5 reanalysis cannot provide real-time operational prospective observations**:
- `OPERATIONAL_NWP_ANALYSIS`: Authorized operational NWP source (e.g., IMD GFS, NCMRWF, ECMWF IFS-HRES). Valid for prospective forward inference when $T_{\text{obs}} \le T$ and $T_{\text{avail}} \le T$.
- `ERA5_REANALYSIS`: Multi-month publication latency. If supplied for prospective inference, the system automatically classifies the run as `HISTORICAL_REANALYSIS` to maintain scientific honesty.
- `OTHER_VERIFIED_OPERATIONAL_SOURCE`: Validated operational satellite or radar synoptic inputs.

---

## 6. Copernicus Daily Ocean Temporal Resolution Policy

The daily resolution of satellite altimetry and ocean reanalysis is strictly preserved:
- **No Artificial Interpolation:** Daily Copernicus ocean observations are never interpolated into synthetic 6-hourly values.
- **Ocean Age Policy:** Ocean observations are valid up to 48 hours before forecast origin ($T - T_{\text{ocean}} \le 48\text{h}$).
- **Dropouts without Fabrication:** If ocean data exceeds 48 hours or is unpublished at $T$, ocean features (`sst`, `mld`, `sla`) are set to `NaN` (missing). Native tree routing handles the missing predictors.
- **Ocean Age Distribution Metrics:** The system monitors ocean latency across 0h, 6h, 12h, 18h, 24h+, and missing categories.

---

## 7. Temporal Honesty in Target Arrival

For a forecast to be considered genuinely prospective:
$$T_{\text{forecast\_created\_at}} < T_{\text{target\_available\_at}}$$
- If a target was already published before the forecast was logged, the record is flagged as `TARGET_EXCLUDED` (`RETROSPECTIVE_EVALUATION_TARGET_AVAILABLE_BEFORE_FORECAST_CREATION`).
- Retrospective evaluations are stored separately in `target_arrival_log.parquet` under `evaluation_type = RETROSPECTIVE_REVISED` and are never combined with prospective metrics.

---

## 8. Frozen Baselines & Clustered Bootstrap Policy

### Standardized Baselines
Every evaluated prospective target is benchmarked against identical baseline models:
- **Persistence:** $\hat{V}_{T+24\text{h}} = V_{\text{current}}$
- **Damped Trend ($\alpha = 0.5$):** $\hat{V}_{T+24\text{h}} = \text{clip}(V_{\text{current}} + 0.5 \cdot \Delta V_{12\text{h}}, 15.0, 165.0)$

### Clustered Bootstrap Framework
- **Unit of Resampling:** Resampling is performed by **storm cluster**, not by individual 6-hourly fixes, accounting for serial autocorrelation.
- **Parameters:** $B = 1,000$ bootstrap replications, 95% empirical confidence intervals.
- **Evidence Tiers:**
  - $< 5$ genuine storms: `INSUFFICIENT_PROSPECTIVE_EVIDENCE` (bootstrap classified as unstable)
  - $5–14$ genuine storms: `EARLY_PROSPECTIVE_SIGNAL`
  - $15–29$ genuine storms: `PROMISING_PROSPECTIVE_EVIDENCE`
  - $\ge 30$ genuine storms: `ADEQUATE_PROSPECTIVE_SAMPLE_FOR_PRELIMINARY_GENERALIZATION`
  - *Evidence Target Note*: 30 independent genuine prospective storms is the project's predefined evidence target for assessing whether a meaningful prospective skill analysis is possible, not a universal statistical requirement. Even with $\ge 30$ storms, scientific generalization remains dependent on storm independence, baseline comparison, target coverage, distribution representativeness, and absence of causal violations.

---

## 9. Drift Monitoring vs. Model Immutability

The prospective service tracks distribution drift on all 29 features against the frozen 2016–2021 training baseline (Wasserstein distance, Cohen's $d$, missingness rate).
- Detected drift triggers `FLAG_FOR_FUTURE_RESEARCH`.
- Drift detection **NEVER** triggers automated retraining, threshold recalibration, or weight modification.

---

## 10. Cryptographic Integrity Verification

All production and research artifacts remain 100% byte-invariant:

| Artifact Path | Expected SHA-256 | Verification Status |
| :--- | :--- | :--- |
| `research/cyclone_intensity/models/final_intensity_model.joblib` | `3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f` | **MATCH (INVARIANT)** |
| `research/cyclone_intensity/feature_contract.json` | `258690562dc9a79589ddfabd6956feb4b36726f2a9bcb093177ea7e0dd26f896` | **MATCH (INVARIANT)** |
| `backend/models/risk_model_0d.joblib` | `54720c2218a46977c2f93e94889d4242f357c313ab069935cf57bdc31f1b61cd` | **MATCH (INVARIANT)** |
| `backend/models/risk_model_1d.joblib` | `cb05c4408ef2f3b66999f8b0016b9dbd1285bc88c1c8f6e32aae310817ccd57c` | **MATCH (INVARIANT)** |
| `backend/models/risk_model_2d.joblib` | `6e6e34c5806c2db83a61369eb8488dedb827845cba8a340c0b8b286363a52843` | **MATCH (INVARIANT)** |
| `backend/models/risk_model_3d.joblib` | `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` | **MATCH (INVARIANT)** |
| `backend/config/frozen_alert_policy_v2.json` | `6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6` | **MATCH (INVARIANT)** |

---

## 11. Final Forensic Conclusions & Readiness Status

```text
ENGINEERING STATUS:
READY_FOR_REAL_PROSPECTIVE_DATA

SCIENTIFIC STATUS:
NOT_YET_ESTABLISHED

PRODUCTION STATUS:
NOT_AUTHORIZED

CLASSIFICATION:
ENGINEERING_READY_WITH_DATA_PROVENANCE_LIMITATIONS
SCIENTIFIC_GENERALIZATION: NOT_YET_ESTABLISHED
```

### Statement of Scientific Generalization
The system is technically and architecturally ready to receive and evaluate genuine prospective data under rigorous causal constraints. However, because no genuine post-historical meteorological observations have yet been ingested and evaluated, **scientific prospective generalization has not yet been established**.
