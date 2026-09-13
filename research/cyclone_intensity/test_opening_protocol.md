# Sagar-Drishti — Cyclone Intensity Test-Set Opening Protocol

**Protocol ID:** `SD-PROTO-TEST-QUARANTINE-V1`  
**Status:** **LOCKED & QUARANTINED**  
**Quarantined Partition:** `TEST` (Years 2024–2026, 29 Storm Systems, 541 Candidate Fixes, 396 Valid Target Pairs)  
**Governance Authority:** Cyclone Intensity Pre-Training & Scientific Gate Committee  

---

## 1. Principle of Permanent Quarantine

The chronological test holdout (2024–2026) represents the definitive out-of-sample operational generalization test for Sagar-Drishti's Cyclone Intensity model.

To guarantee zero information leakage, zero p-hacking, and zero test-driven hyperparameter overfitting:
1. **Zero Intermediate Evaluation:** The test set must NEVER be accessed during exploratory analysis, baseline verification, feature selection, or hyperparameter optimization.
2. **Code-Level Enforcement:** Data-loading pipelines and training harnesses must reject any request to read or filter the `TEST` partition prior to the unlock authorization, raising an explicit `TestSetQuarantineViolationError`.
3. **Single-Pass Evaluation:** Once unlocked, the test set is evaluated exactly **ONCE**. The outputs of this single run represent the official reported figures of record. No revisions, post-hoc parameter tweaks, or secondary runs are permitted.

---

## 2. Mandatory Eight-Keycard Unlock Conditions

The quarantined test partition may only be unlocked when all eight keycards are satisfied and verified:

```
┌───────┬──────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Key # │ Prerequisite Gate                │ Verification Criteria                                  │
├───────┼──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1     │ Feature Contract Frozen          │ SHA-256 of feature_contract.json verified and logged.  │
│ 2     │ Feature Groups & Ablations Frozen│ feature_groups.json verified; no new features added.   │
│ 3     │ Baselines Frozen & Documented    │ Persistence & Damped Trend metrics logged on Val.      │
│ 4     │ Imputation Parameters Frozen     │ Train-only median statistics fitted and serialized.    │
│ 5     │ Model Family & Grid Frozen       │ Hyperparameter search space locked before execution.   │
│ 6     │ Validation Selection Complete    │ Winning architecture selected strictly on Val Storm-MAE│
│ 7     │ Reproducibility Seeds Frozen     │ Seed 42 locked across all sampling and random states.  │
│ 8     │ Explicit Stakeholder Authorization│ User explicitly issues "AUTHORIZE_TEST_EVALUATION"    │
└───────┴──────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 3. Post-Unlock Execution Sequence

When authorization is granted:
1. Load the frozen model checkpoint trained on Train (or Train+Validation if explicitly pre-declared).
2. Execute a single inference pass over the 396 valid target rows of the 2024–2026 test partition.
3. Compute the primary metric (Storm-aggregated MAE) and secondary metrics (Row MAE, RMSE, Bias, Skill over Persistence).
4. Run the pre-declared 1,000-iteration storm-level clustered bootstrap on the test partition to establish 95% confidence intervals.
5. Generate the final unedited forensic test report [`CYCLONE_INTENSITY_TEST_EVALUATION_REPORT.md`](file:///c:/Users/krishna/OneDrive/Desktop/Sagar-Drishti/backend/reports/CYCLONE_INTENSITY_TEST_EVALUATION_REPORT.md).
6. Permanently seal the test evaluation lock.
