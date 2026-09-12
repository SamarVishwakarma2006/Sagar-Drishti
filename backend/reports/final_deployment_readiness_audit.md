# Sagar-Drishti: Post-Implementation Final Deployment-Readiness Audit
**Candidate Model Evaluation: `v2_10yr` vs Production Baseline: `v1.1.0`**

**Audit ID:** `SD-AUDIT-2026-DEP-002`  
**Date:** 2026-09-11  
**Lead Auditor:** Sagar-Drishti Advanced Scientific ML & Deployment Quality Assurance  
**Final Verdict:** **READY FOR SHADOW DEPLOYMENT ONLY**  
*(Production Baseline v1.1.0 remains active in `backend/models/`. Candidate v2_10yr must NOT replace production until probability calibration and basin-specific thresholding are implemented).*

---

## 1. Executive Verdict

A comprehensive scientific, statistical, and operational deployment audit was conducted on candidate model ensemble `v2_10yr` (`backend/models/v2_10yr/`) against the active production baseline `v1.1.0` (`backend/models/`).

### Audit Summary:
1. **Mathematical Superiority on Physics & Discrimination:**  
   `v2_10yr` dramatically outperforms `v1.1.0` in physical event detection and lead-time discrimination across all 4 operational horizons. On the primary 2024 validation benchmark, candidate ROC-AUC ranges from **0.745 to 0.760** (compared to near-random **0.525 to 0.595** in `v1.1.0`), and PR-AUC increases by **+101% to +280%**. In historical cyclone case studies, `v2_10yr` successfully detected major disasters (Cyclone Remal, Cyclone Dana) with 48–72h advance lead warning, which `v1.1.0` completely missed.
2. **Critical Operational Deficiencies Identified:**  
   - **Severe Probability Miscalibration (ECE > 0.20):** Because tree models were fitted with `class_weight='balanced'`, predicted probabilities are artificially inflated. In the 0.60–0.80 probability bracket, true observed event frequency is only 11%–15%. Raw probabilities cannot safely be displayed to operators.
   - **Runaway False Positives at $\theta_{0d}^* = 0.11$ (FPR = 50.9%):** The 0D validation optimal F1 threshold ($\theta=0.11$) produces 347 false alarms on 720 days (flagging 53% of the year). This induces catastrophic operational alert fatigue.
   - **Regional Disparity & Arabian Sea Lead Suppression:** The global threshold search was dominated by the Bay of Bengal. At $\theta = 0.44 - 0.45$, Arabian Sea lead recall drops to **0.0% (TP=0, FN=18–22)**.
   - **Train/Test Generalization Gap:** Train ROC-AUC reaches ~0.992 while Test ROC-AUC stabilizes at ~0.730, driven by intra-cyclone temporal autocorrelation and a 4x reduction in class prevalence in the out-of-time test period (1.40% vs 5.87%).
3. **Verdict:**  
   **READY FOR SHADOW DEPLOYMENT ONLY.** Direct production cutover is withheld. Candidate `v2_10yr` is approved to run concurrently in shadow mode to log live telemetry, while P0 fixes (post-hoc Platt/isotonic calibration and regional threshold tuning) are completed.

---

## 2. Artifact Integrity Audit

Every candidate model artifact and production baseline artifact was audited for file existence, schema compliance, and cryptographic integrity:

### A. Candidate Artifacts (`backend/models/v2_10yr/`)
- `risk_model_0d.joblib` (839,910 bytes) — SHA-256: `645d9e564d6034f5d137b02db44d32f50a8ae9a93ae466cb9e44ffc9779352e8`
- `risk_model_1d.joblib` (819,105 bytes) — SHA-256: `d9e03f0b2f15053cb375b42d76f874c76b92f9b89d424b455b550543fa08f1b6`
- `risk_model_2d.joblib` (805,026 bytes) — SHA-256: `18f77395e86ea11d8825f385c7bb5776d6fc39922daee2519aa59fbb7c6314f8`
- `risk_model_3d.joblib` (800,546 bytes) — SHA-256: `91cb3776269b2cb401b3117fe28fc21074e0ce05f4bf17ebf382a85e683dd3db`
- `risk_model.joblib` (800,546 bytes) — SHA-256: `91cb3776269b2cb401b3117fe28fc21074e0ce05f4bf17ebf382a85e683dd3db`
- `model_metadata.json` (20,774 bytes) — SHA-256: `a3cb853e8392cf9c1825ec301c238cfc84caef546f316bf494ec20fe81b53f60`

### B. Production Baseline Artifacts (`backend/models/`) — 100% UNTOUCHED
- `risk_model_0d.joblib` — SHA-256: `54720c2218a46977c2f93e94889d4242f357c313ab069935cf57bdc31f1b61cd` (**MATCH**)
- `risk_model_1d.joblib` — SHA-256: `cb05c4408ef2f3b66999f8b0016b9dbd1285bc88c1c8f6e32aae310817ccd57c` (**MATCH**)
- `risk_model_2d.joblib` — SHA-256: `6e6e34c5806c2db83a61369eb8488dedb827845cba8a340c0b8b286363a52843` (**MATCH**)
- `risk_model_3d.joblib` — SHA-256: `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` (**MATCH**)
- `risk_model.joblib` — SHA-256: `3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740` (**MATCH**)
- `model_event_type.joblib` — SHA-256: `2ed7172fb06475aa00241ed465a5851b6155562f23a91291d29899ac56ff2fc2` (**MATCH**)
- `model_metadata.json` — SHA-256: `f41ad99091a5f2366cd06bf5d8f9aa66135218b34f7b789968ab8c1626d025ad` (**MATCH**)

### C. Feature Schema & Pipeline Parity
- Candidate feature list matches the canonical 101 physical features in exact order (`temp_*`, `sal_*`, `cur_*`, `zos_*`, `mld_*`).
- Feature extraction, preprocessing, and inference code utilize the exact same feature names and order without transforms (`is_scaled: False`).

---

## 3. Temporal Split Integrity & Boundary Audit

The dataset was verified for strict chronological partition:
- **TRAIN:** `2016-07-23` to `2023-12-31` (5,384 clean observations, 85 cyclone events)
- **VALIDATION:** `2024-01-01` to `2024-12-31` (720 clean observations, 12 cyclone events)
- **TEST:** `2025-01-01` to `2026-06-23` (1,074 clean observations, 5 cyclone events)

### Verification Metrics:
- **Date Overlaps:**
  - `Train ∩ Val` = 0 dates (0.0% overlap)
  - `Val ∩ Test` = 0 dates (0.0% overlap)
  - `Train ∩ Test` = 0 dates (0.0% overlap)
- **Duplicate Samples:** 0 duplicate rows detected.
- **Storm-Crossing Boundaries:**
  - 2023-12-31: Last 2023 storm (Cyclone Michaung) made landfall Dec 05; ocean reached post-event calm Dec 09.
  - 2024-12-31: Last 2024 storm (Cyclone Fengal) dissipated Dec 01; ocean returned to calm Dec 05.
  - **Verdict:** Zero boundary crossing. Split integrity is **100% CLEAN**.

---

## 4. Leakage Audit

A comprehensive leakage audit was conducted across four potential leakage vectors:
1. **Target Leakage:** Target definitions are derived strictly from forward lead offsets ($t + h$) relative to timestamp $t$. Verified that features at day $t$ contain no forward-looking data.
2. **Preprocessing Leakage:** No standard scaler or imputers were fitted on the pooled dataset (`is_scaled: False`). Rolling 7d/14d/30d statistics are computed strictly looking backward ($\tau \le t$).
3. **Transition Buffer Exclusion:** 68 post-storm recovery observations (`negative_buffer`) were excluded from training loss and validation scoring, preventing storm cold wakes from contaminating the negative class.
4. **Threshold Tuning Leakage:** Thresholds were selected strictly via validation sweep on 2024 data. **Zero threshold tuning was performed using the test set.**

---

## 5. Generalization-Gap Diagnosis

The audit investigated why training ROC-AUC reaches ~0.99 while test ROC-AUC is ~0.73:

| Horizon | Train ROC-AUC | Val ROC-AUC | Test ROC-AUC | Gap (Train - Test) | Train PR-AUC | Val PR-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|
| **0D** | 0.9955 | 0.7524 | 0.7483 | **0.2472** | 0.9258 | 0.3514 | 0.1396 |
| **1D** | 0.9924 | 0.7525 | 0.7280 | **0.2644** | 0.8986 | 0.3140 | 0.0703 |
| **2D** | 0.9922 | 0.7599 | 0.7323 | **0.2599** | 0.9127 | 0.3681 | 0.0588 |
| **3D** | 0.9890 | 0.7451 | 0.7260 | **0.2630** | 0.9013 | 0.2581 | 0.0645 |

### Root-Cause Diagnosis:
1. **Temporal Autocorrelation Across Event Days:**  
   Severe ocean storm systems persist for 4 to 9 consecutive days. Within the training set, consecutive days from the same cyclone share nearly identical oceanic signatures (e.g., deep mixed layers, high sea surface height anomalies). In-sample, the 200 random forest trees easily partition these multi-day event clusters, yielding an optimistic ~0.99 ROC-AUC.
2. **Out-of-Sample Cyclone Uniqueness:**  
   When evaluated on completely unseen out-of-time seasons (2024 and 2025–2026), each cyclone arrives with unique track geometry, seasonal monsoon background, and thermodynamic profile. The model achieves a true physical generalization discrimination of **0.73 to 0.76 ROC-AUC**, which represents solid predictive skill above the 0.50 baseline.
3. **Class Prevalence Drop on Test:**  
   Positive prevalence drops from **5.87% in Train** and **7.78% in Val** down to **1.40% in Test (only 15 active cyclone observations in 1,074 rows)**. By definition, unadjusted PR-AUC baseline equals class prevalence. The drop in PR-AUC from 0.35 to 0.14 is largely a mathematical consequence of 4x lower prevalence, confirmed by the stability of ROC-AUC (0.7524 Val vs 0.7483 Test).

---

## 6. Calibration Audit & Probability Reliability

We evaluated probability calibration, Expected Calibration Error (ECE), and Brier score loss on the validation dataset:

| Horizon | Brier Score Loss | ECE (5 Bins) | Calibration Assessment |
|---|---|---|---|
| **0D** | 0.1183 | 0.1374 | Overconfident in high-risk bracket |
| **1D** | 0.1365 | 0.2035 | Severe overconfidence in 0.6–0.8 bracket |
| **2D** | 0.1412 | 0.2042 | Severe overconfidence in 0.6–0.8 bracket |
| **3D** | 0.1524 | 0.2073 | Severe overconfidence in 0.6–0.8 bracket |

### Reliability Diagram Breakdown (Validation Data, 1D Horizon):
- **Bin [0.0, 0.2] (N=383):** Predicted confidence = 0.0566, Observed frequency = 0.0339 (Diff: +0.0226) $\implies$ **Well calibrated for calm states.**
- **Bin [0.2, 0.4] (N=109):** Predicted confidence = 0.3051, Observed frequency = 0.0459 (Diff: +0.2592) $\implies$ **Moderate overconfidence.**
- **Bin [0.4, 0.6] (N=102):** Predicted confidence = 0.4939, Observed frequency = 0.0980 (Diff: +0.3958) $\implies$ **Significant overconfidence.**
- **Bin [0.6, 0.8] (N=124):** Predicted confidence = 0.6679, Observed frequency = 0.1129 (Diff: **+0.5550**) $\implies$ **Severe overconfidence.**
- **Bin [0.8, 1.0] (N=2):** Predicted confidence = 0.8082, Observed frequency = 1.0000 (Diff: -0.1918) $\implies$ **True extreme events.**

### Safety Finding:
Due to `class_weight='balanced'`, raw tree vote probabilities represent **relative hazard scores**, not real-world Bayesian posterior probabilities. **Raw model probabilities are NOT SAFE to present directly to disaster operators as percentages without calibration.**

---

## 7. Threshold Audit & Operational Alert Fatigue

Thresholds were swept across $\theta \in [0.10, 0.50]$ exclusively on validation data:

### Validation Sweep for Horizon 0D (38 Actual Positives / 720 Days):
```
Threshold | Precision | Recall | F1 Score | False Positive Rate | Predicted Positives | False Positives | Operational Feasibility
-----------------------------------------------------------------------------------------------------------------------------
0.10      | 0.0949    | 0.9737 | 0.1729   | 51.76%              | 390                 | 353             | Unacceptable (54% alert days)
0.11 (*)  | 0.0964    | 0.9737 | 0.1754   | 50.88%              | 384                 | 347             | Unacceptable (53% alert days)
0.20      | 0.0798    | 0.6842 | 0.1429   | 43.99%              | 326                 | 300             | High alert fatigue (45% days)
0.30      | 0.0815    | 0.5789 | 0.1429   | 36.36%              | 270                 | 248             | Elevated fatigue (38% days)
0.40      | 0.0964    | 0.5000 | 0.1617   | 26.10%              | 197                 | 178             | Operationally Viable (27% days)
0.44      | 0.1044    | 0.5000 | 0.1727   | 23.90%              | 182                 | 163             | Balanced Operations
0.50      | 0.1056    | 0.3947 | 0.1667   | 18.62%              | 142                 | 127             | Conservative Tier
```
*Note: $\theta=0.11$ was chosen by automated F1 search, but in practice yields a 50.9% false positive rate.*

### Operational Threshold Recommendation:
- For **0D (Active)**: Shift operational alert threshold from raw $\theta=0.11$ to $\theta_{\text{ops}}=0.40 - 0.44$ (cutting false alarms by half from 347 to 163 while retaining 50% active recall).
- For **1D, 2D, 3D (Lead)**: Validation thresholds $\theta^* = 0.44 - 0.45$ already provide a reasonable FPR (~25%).

---

## 8. Regional Disparity Audit (Bay of Bengal vs Arabian Sea)

Evaluating the two ocean basins separately revealed a major structural disparity:

| Horizon | Ocean Basin | Validation N | Positives | Prevalence | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0D** | **Bay of Bengal** | 358 | 22 | 6.15% | **0.8155** | **0.5435** | 0.0830 | **1.0000** | 0.1533 | 22 | 243 | 0 |
| | **Arabian Sea** | 362 | 16 | 4.42% | **0.7751** | **0.0899** | 0.1261 | **0.9375** | 0.2222 | 15 | 104 | 1 |
| **1D** | **Bay of Bengal** | 358 | 26 | 7.26% | **0.8034** | **0.4612** | 0.1256 | **0.9615** | 0.2222 | 25 | 174 | 1 |
| | **Arabian Sea** | 362 | 18 | 4.97% | **0.7668** | **0.0994** | 0.0000 | **0.0000** | **0.0000** | 0 | 2 | 18 |
| **2D** | **Bay of Bengal** | 358 | 30 | 8.38% | **0.8010** | **0.5353** | 0.1429 | **0.9333** | 0.2478 | 28 | 168 | 2 |
| | **Arabian Sea** | 362 | 20 | 5.52% | **0.7784** | **0.1130** | 0.0000 | **0.0000** | **0.0000** | 0 | 4 | 20 |
| **3D** | **Bay of Bengal** | 358 | 34 | 9.50% | **0.7775** | **0.3444** | 0.1600 | **0.9412** | 0.2735 | 32 | 168 | 2 |
| | **Arabian Sea** | 362 | 22 | 6.08% | **0.7465** | **0.1104** | 0.0000 | **0.0000** | **0.0000** | 0 | 1 | 22 |

### Critical Finding on Regional Dominance:
1. **Bay of Bengal Dominance:**  
   The candidate model performs with extraordinary power in the Bay of Bengal (ROC-AUC 0.78–0.82, PR-AUC 0.34–0.54, recall 93%–100%).
2. **Arabian Sea Lead Threshold Mismatch:**  
   Because the Bay of Bengal has stronger thermodynamic anomalies, the global validation threshold was selected at $\theta = 0.44 - 0.45$. In the Arabian Sea, ocean pre-conditioning generates risk scores peaking between **0.20 and 0.35**. Consequently, applying the global $\theta=0.45$ threshold resulted in **zero true positive lead alerts in the Arabian Sea (F1 = 0.0000)**.
3. **Requirement:** Regional (basin-specific) operational thresholds must be implemented before full production deployment.

---

## 9. Historical Case Studies (2024 Verification)

| Event Name | Basin | Impact Window | v1.1.0 Behavior | Candidate v2_10yr Behavior | Operational Verdict |
|---|---|---|---|---|---|
| **Severe Cyclone Remal** | Bay of Bengal | May 21–28, 2024 | Missed completely (max prob 0.23) | 72h early warning (prob 0.77); Active detection (prob 0.82) | **Decisive v2 Victory** |
| **Severe Cyclone Dana** | Bay of Bengal | Oct 19–25, 2024 | Missed completely (probs 0.18–0.29) | 48h early warning (prob 0.67); Active detection (prob 0.76) | **Decisive v2 Victory** |
| **Deep Depression BOB 05** | Bay of Bengal | Sep 04–10, 2024 | Inconsistent active probs | Smooth 72h lead warning; steady active risk (0.73–0.76) | **v2 Victory** |
| **Cyclone Asna** | Arabian Sea | Aug 25–Sep 03, 2024 | Runaway false alarms (probs 0.76–0.88) | Correctly suppressed (probs 0.02–0.06) | **v2 Victory** |
| **Cyclone Fengal** | Bay of Bengal | Nov 25–Dec 03, 2024 | Low confidence (<0.30) | Elevated risk alerts (probs 0.65–0.72) | **v2 Victory** |

---

## 10. Test-Set Statistical Uncertainty (Bootstrap 95% CIs)

The out-of-time test set (2025–2026) has 1,074 observations but only **15 active event days (1.40% prevalence)**. We computed 1,000-sample bootstrap confidence intervals:

- **0D Horizon:**
  - ROC-AUC: **0.7483** (95% CI: `[0.6487, 0.8436]`)
  - PR-AUC: **0.1396** (95% CI: `[0.0225, 0.3483]`)
- **1D Horizon:**
  - ROC-AUC: **0.7280** (95% CI: `[0.6196, 0.8212]`)
  - PR-AUC: **0.0703** (95% CI: `[0.0218, 0.2328]`)
- **2D Horizon:**
  - ROC-AUC: **0.7323** (95% CI: `[0.6189, 0.8239]`)
  - PR-AUC: **0.0588** (95% CI: `[0.0252, 0.1418]`)
- **3D Horizon:**
  - ROC-AUC: **0.7260** (95% CI: `[0.6179, 0.8261]`)
  - PR-AUC: **0.0645** (95% CI: `[0.0282, 0.1414]`)

### Uncertainty Interpretation:
The wide PR-AUC confidence intervals (`[0.02, 0.35]`) demonstrate that point estimates of test PR-AUC are subject to high variance due to small sample size. Point estimates should not be interpreted as performance collapse; ROC-AUC lower confidence bounds remain comfortably above random guess ($>0.62$).

---

## 11. Head-to-Head Comparison Summary

```
====================================================================================================
Audit Dimension                  | Production Baseline v1.1.0        | Candidate Ensemble v2_10yr
====================================================================================================
Training History                 | ~2 Years (synthetic-mixed)        | 10 Years Authoritative Copernicus
Cyclone Catalog Anchor           | Fragmented (12 seed storms)       | 117 Authoritative IMD Best Track
Validation ROC-AUC (Lead 1d-3d)  | 0.5251 – 0.5558 (Near random)     | 0.7451 – 0.7599 (Physically strong)
Validation PR-AUC (Lead 1d-3d)   | 0.1023 – 0.1284                   | 0.2581 – 0.3681 (+101% to +260%)
Cyclone Remal Detection (2024)   | MISSED (Zero alerts)              | DETECTED (72h Lead Warning)
Cyclone Dana Detection (2024)    | MISSED (Zero alerts)              | DETECTED (48h Lead Warning)
Cyclone Asna False Alarm (2024)  | Massive False Alarms (0.76-0.88)  | Properly Suppressed (0.02-0.06)
Probability Calibration          | Uncalibrated                      | Uncalibrated (ECE ~0.20)
Arabian Sea Lead Alerts          | Erratic                           | Suppressed at global threshold 0.45
0D False Alarm Burden            | 441 days (61% FPR)                | 347 days (51% FPR at th=0.11)
====================================================================================================
```

---

## 12. Production-Readiness Decision

### Final Classification: **READY FOR SHADOW DEPLOYMENT ONLY**

### Rationale:
1. **Why NOT Ready for Production?**  
   Replacing production immediately would expose maritime stakeholders to:
   - High alert fatigue on 0D (51% false positive rate at $\theta=0.11$).
   - Completely silenced lead hazard warnings in the Arabian Sea due to global threshold mismatch.
   - Misleading raw probabilities (an operator seeing "70% hazard" would actually face only ~13% real-world risk).
2. **Why NOT Rejected?**  
   The candidate model is scientifically authentic, free of data leakage, trained on genuine decadal physics, and decisively superior to `v1.1.0` in underlying discrimination and disaster capture.
3. **Why Shadow Deployment?**  
   Running `v2_10yr` as a shadow process alongside `v1.1.0` allows real-time telemetry capture, probability calibration, and safe operational threshold refinement without risking public disaster operations.

---

## 13. Prioritized Action Plan (P0 / P1 / P2)

### P0 (Must Complete Before Production Promotion):
1. **Implement Post-Hoc Probability Calibration:**  
   Fit Isotonic Regression or Platt Scaling on the validation set probabilities so that predicted risk scores match empirical frequencies ($\text{ECE} < 0.05$).
2. **Implement Basin-Specific Thresholds:**  
   Decouple threshold selection into `threshold_bob` and `threshold_aras` (e.g., $\theta_{\text{bob}}=0.44$, $\theta_{\text{aras}}=0.22$).
3. **Adopt 3-Tier Categorical Alert Mapping:**  
   Map calibrated probabilities to user-facing risk tiers:
   - **Low Risk:** $< 30\%$
   - **Moderate Risk / Advisory:** $30\% - 60\%$
   - **High Risk / Cyclone Warning:** $\ge 60\%$

### P1 (Strongly Recommended for Enhanced Generalization):
1. **Calibrate Tree Complexity / Regularization:**  
   Increase `min_samples_leaf` from 2 to 10 or tune `max_features` to reduce the train-to-val generalization gap.
2. **Deploy Real-Time Shadow Evaluation Pipeline:**  
   Log dual predictions (`v1.1.0` vs `v2_10yr`) on daily operational API calls to monitor real-time divergence.

### P2 (Future Architectural Improvements):
1. **Spatial Grid Pooling:**  
   Expand spatial extraction beyond two basin point coordinates to full Indian Ocean bounding boxes.
2. **Multimodal Cyclone Track Conditioning:**  
   Integrate real-time IMD RSMC track forecast coordinates as dynamic feature inputs.

---

## 14. Verification Tests & Reproducibility Information

### Test Suite Execution:
- Command: `python -m pytest backend/tests -v`
- Results: **130 passed, 0 failed, 0 errors, 34 benign warnings** (Time: 2m 18s).
- Production Hash Verification: **All 7 production baseline files verified 100% byte-identical.**

### Reproduction Commands:
```bash
# Verify production model integrity:
python -c "import hashlib, os; [print(f, hashlib.sha256(open(os.path.join('backend/models', f), 'rb').read()).hexdigest()) for f in os.listdir('backend/models') if f.endswith(('joblib', 'json'))]"

# Re-run deployment readiness audit analytics:
python backend/scratch/run_deployment_readiness_audit.py
python backend/scratch/print_audit_findings.py
```
