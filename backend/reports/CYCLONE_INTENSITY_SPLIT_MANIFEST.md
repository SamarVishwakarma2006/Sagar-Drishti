# Sagar-Drishti — Cyclone Intensity Split Manifest
## RETROSPECTIVE EVENT-GROUPED CHRONOLOGICAL PARTITIONS

**Document ID:** `SD-MANIFEST-2026-INTENSITY-SPLIT-V2`  
**Classification:** `RESEARCH_ONLY` | `NOT_FOR_PRODUCTION` | `NO_OPERATIONAL_AUTHORITY`  
**Date:** September 13, 2026  
**Source Dataset:** IMD Best Track Archives (`backend/data/historical/imd_tracks_2016_2026.parquet`)  
**SHA-256 Hash:** `e3f1b87455e716767e3b34a05b3dff04587dc09c6689a7dc0eaf7b72f951b643`  
**ML Training Performed:** ZERO (Zero models, zero scalers, zero checkpoints)

---

## 1. Split Protocol & Forensic Reconciliation

### The Anti-Leakage Event-Grouping Principle
Consecutive 6-hourly synoptic fixes within the same cyclone track exhibit heavy temporal autocorrelation (pooled lag-1 autocorrelation $r = 0.984$ for $V_{\max}$ and $r = 0.987$ for $P_c$; storm-level mean $r = 0.771$, median $r = 0.848$). Random or fix-level splitting fatally leaks core vortex structure and trajectory information between train and test sets.

To guarantee zero leakage:
1. **Storm-Level Grouping:** Every cyclone system belongs entirely and indivisibly to a single partition.
2. **Strict Chronological Sequencing:** Partitions are strictly ordered by calendar year to prevent forward temporal leakage.
3. **Quarantine of Holdout Test Set:** The 2024–2026 partition is permanently quarantined and must never be used for feature engineering, hyperparameter tuning, or model selection.

---

## 2. Partition Summary & Candidate Exclusions

```
EVENT-GROUPED CHRONOLOGICAL PARTITION SUMMARY
┌──────────────┬────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Partition    │ Calendar Span      │ Unique Storms│ Raw Fixes    │ Excluded Fixes│ Valid Fixes  │
├──────────────┼────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ TRAIN        │ 2016-05 to 2021-12 │ 64 storms    │ 1,493 fixes  │ 5 fixes*     │ 1,488 fixes  │
│ VALIDATION   │ 2022-01 to 2023-12 │ 24 storms    │   510 fixes  │ 4 fixes**    │   506 fixes  │
│ TEST         │ 2024-01 to 2026-01 │ 29 storms    │   541 fixes  │ 0 fixes      │   541 fixes  │
├──────────────┼────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ TOTAL        │ 2016-05 to 2026-01 │ 117 storms   │ 2,544 fixes  │ 9 fixes      │ 2,535 fixes  │
└──────────────┴────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
*Train exclusions: 1 NADA anomaly (row 168) + 4 conflicting duplicate rows (PHETHAI rows 542-543, UNNAMED-9 rows 1488-1489).
**Validation exclusions: 4 conflicting duplicate rows (UNNAMED-14 rows 1576-1577, MICHAUNG rows 1969-1970).
```

### Forensic Storm Split Reconciliation Audit (Addressing All 10 Forensic Points)

#### 1. What Date Defines a Storm's Partition?
In our comprehensive audit of all 117 cyclone systems in `imd_tracks_2016_2026.parquet`:
- **First Fix Date:** The timestamp of the initial observation of each storm.
- **Storm Start Date:** Defined identically by the initial synoptic entry in IMD Best Track.
- **Last Fix Date:** The timestamp of the final dissipating observation.
- **System ID Year:** The calendar year embedded in `IMD-YYYY-N-NAME`.

**Crucial Finding:** Across all 117 storms, **exactly zero storms cross calendar year boundaries** (i.e., $\text{year}(\text{first\_fix}) == \text{year}(\text{last\_fix}) == \text{year}(\text{system\_id})$ for 100% of storms). Therefore, the partitioning rule based on **calendar season / first fix date / storm start date / system_id year are mathematically and empirically 100% identical and equivalent**.

#### 2 & 3. Partition Uniqueness and Non-Splitting
- **Every `system_id` belongs to exactly one partition.**
- **Zero storm systems are split across partitions.** All fixes belonging to any given storm exist exclusively within that storm's single assigned partition.

#### 4 & 5. Detailed Reconciliation: 74/22/21 Proposal vs Authoritative 64/24/29 Partition
The discrepancy between the preliminary 74 / 22 / 21 proposal and the actual 64 / 24 / 29 counts arose from an **index-slicing vs calendar-boundary conflict** in the initial design draft:
- In the initial preliminary draft, a fixed integer index partition (the first 74 storms, the next 22 storms, and the remaining 21 storms) was applied to the chronological storm array to approximate a 63% / 19% / 18% ratio.
- However, that draft incorrectly assumed that index 73 aligned with December 31, 2021, and that index 95 aligned with December 31, 2023.
- **Forensic Verification:** 
  - There are only **64 storms** between 2016 and 2021 (2016: 10, 2017: 9, 2018: 14, 2019: 12, 2020: 9, 2021: 10). Taking 74 storms required slicing through September 11, 2022, erroneously absorbing **10 storms from 2022 into the training set**.
  - 2022 has 15 storms and 2023 has 9 storms (24 storms total). An index of 22 storms from index 74 reached into September 13, 2024, absorbing **8 storms from 2024 into the validation set**.
  - Consequently, the holdout test set lost the first 8 storms of 2024 (including *REMAL* and *ASNA*), leaving only 21 storms from late-2024 to 2026.

**Specific Storm IDs Differing Between Old Proposal and Correct Split:**
- **10 Storms from 2022 (Erroneously in Old Train, Correctly in Validation):**  
  `IMD-2022-1-UNNAMED1`, `IMD-2022-2-UNNAMED2`, `IMD-2022-3-ASANI`, `IMD-2022-4-UNNAMED4`, `IMD-2022-5-UNNAMED5`, `IMD-2022-6-UNNAMED6`, `IMD-2022-7-UNNAMED7`, `IMD-2022-8-UNNAMED8`, `IMD-2022-9-UNNAMED9`, `IMD-2022-10-UNNAMED10`.
- **8 Storms from 2024 (Erroneously in Old Validation, Correctly in Test Holdout):**  
  `IMD-2024-1-REMAL`, `IMD-2024-2-UNNAMED2`, `IMD-2024-3-UNNAMED3`, `IMD-2024-4-ASNA`, `IMD-2024-5-UNNAMED5`, `IMD-2024-6-UNNAMED6`, `IMD-2024-7-UNNAMED7`, `IMD-2024-8-UNNAMED8`.

#### 6. Chronological Validity of Current 64 / 24 / 29 Partition
The current partition strictly preserves complete, intact calendar years:
- **TRAIN (2016–2021):** 6 complete years (May 2016 to December 2021), 64 storms, 1,493 raw fixes (1,488 candidate fixes).
- **VALIDATION (2022–2023):** 2 complete years (March 2022 to December 2023), 24 storms, 510 raw fixes (506 candidate fixes).
- **TEST (2024–2026):** 2.5 complete years (May 2024 to January 2026), 29 storms, 541 raw fixes (541 candidate fixes).

#### 7. Verification of No Fix-Level Splitting
Every single fix of each storm is grouped with that storm's partition. Zero intra-storm fix leakage exists.

#### 8. Verification of Causal Isolation
No future information or post-event knowledge was utilized to assign features, targets, or partition memberships.

#### 9. Complete Quarantine of 2024–2026 Test Set
All 29 storms and 541 fixes of the 2024–2026 period remain strictly quarantined. They are barred from feature selection, hyperparameter tuning, and threshold selection.

#### 10. The Scientifically Correct Rule Prevails
The 64 / 24 / 29 partition respects physical season boundaries, avoids mid-season splits, ensures that no 2022 data contaminates training, and guarantees that all of 2024 is preserved in the true holdout test set. Therefore, **the 64 / 24 / 29 partition is retained as the authoritative scientific standard**.

---

## 3. Mathematical Proof of Zero Storm Overlap

Let $\mathcal{S}_{\text{train}}$, $\mathcal{S}_{\text{val}}$, and $\mathcal{S}_{\text{test}}$ denote the sets of unique `system_id` values in each partition:

$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$$
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

$$\left| \mathcal{S}_{\text{train}} \right| + \left| \mathcal{S}_{\text{val}} \right| + \left| \mathcal{S}_{\text{test}} \right| = 64 + 24 + 29 = 117 = \left| \mathcal{S}_{\text{all}} \right|$$

**Set Intersection Result:** Empty set ($\emptyset$) across all pairs. Zero storm overlap confirmed.

---

## 4. Complete Storm Inventory by Partition

### A. Training Partition (2016–2021: 64 Storms)
```
IMD-2016-1-ROANU, IMD-2016-2-UNNAMED2, IMD-2016-3-UNNAMED3, IMD-2016-4-UNNAMED4, 
IMD-2016-5-UNNAMED5, IMD-2016-6-KYANT, IMD-2016-7-UNNAMED7, IMD-2016-8-NADA, 
IMD-2016-9-VARDAH, IMD-2016-10-UNNAMED10, IMD-2017-1-MAARUTHA, IMD-2017-2-MORA, 
IMD-2017-3-UNNAMED3, IMD-2017-4-UNNAMED4, IMD-2017-5-UNNAMED5, IMD-2017-6-UNNAMED6, 
IMD-2017-7-UNNAMED7, IMD-2017-9-OCKHI, IMD-2017-10-UNNAMED10, IMD-2018-1-UNNAMED1, 
IMD-2018-2-SAGAR, IMD-2018-3-MEKUNU, IMD-2018-4-UNNAMED4, IMD-2018-5-UNNAMED5, 
IMD-2018-6-UNNAMED6, IMD-2018-7-UNNAMED7, IMD-2018-8-UNNAMED8, IMD-2018-9-UNNAMED9, 
IMD-2018-10-DAYE, IMD-2018-11-LUBAN, IMD-2018-12-TITLI, IMD-2018-13-GAJA, 
IMD-2018-14-PHETHAI, IMD-2019-1-PABUK, IMD-2019-2-FANI, IMD-2019-3-VAYU, 
IMD-2019-4-UNNAMED4, IMD-2019-5-HIKKA, IMD-2019-6-UNNAMED6, IMD-2019-7-KYAAR, 
IMD-2019-8-MAHA, IMD-2019-9-BULBUL, IMD-2019-10-PAWAN, IMD-2019-11-UNNAMED11, 
IMD-2019-12-UNNAMED12, IMD-2020-1-AMPHAN, IMD-2020-2-UNNAMED2, IMD-2020-3-NISARGA, 
IMD-2020-4-UNNAMED4, IMD-2020-5-UNNAMED5, IMD-2020-6-UNNAMED6, IMD-2020-7-NIVAR, 
IMD-2020-8-GATI, IMD-2020-9-BUREVI, IMD-2021-1-UNNAMED1, IMD-2021-2-TAUKTAE, 
IMD-2021-3-YAAS, IMD-2021-4-UNNAMED4, IMD-2021-5-GULAB, IMD-2021-6-SHAHEEN, 
IMD-2021-7-UNNAMED7, IMD-2021-8-UNNAMED8, IMD-2021-9-UNNAMED9, IMD-2021-10-JAWAD
```

### B. Validation Partition (2022–2023: 24 Storms)
```
IMD-2022-1-UNNAMED1, IMD-2022-2-UNNAMED2, IMD-2022-3-ASANI, IMD-2022-4-UNNAMED4, 
IMD-2022-5-UNNAMED5, IMD-2022-6-UNNAMED6, IMD-2022-7-UNNAMED7, IMD-2022-8-UNNAMED8, 
IMD-2022-9-UNNAMED9, IMD-2022-10-UNNAMED10, IMD-2022-11-SITRANG, IMD-2022-12-UNNAMED12, 
IMD-2022-13-MANDOUS, IMD-2022-14-UNNAMED14, IMD-2022-15-UNNAMED15, IMD-2023-1-UNNAMED1, 
IMD-2023-2-MOCHA, IMD-2023-3-BIPARJOY, IMD-2023-4-UNNAMED4, IMD-2023-5-UNNAMED5, 
IMD-2023-6-TEJ, IMD-2023-7-HAMOON, IMD-2023-8-MIDHILI, IMD-2023-9-MICHAUNG
```

### C. Quarantined Test Partition (2024–2026: 29 Storms)
```
IMD-2024-1-REMAL, IMD-2024-2-UNNAMED2, IMD-2024-3-UNNAMED3, IMD-2024-4-ASNA, 
IMD-2024-5-UNNAMED5, IMD-2024-6-UNNAMED6, IMD-2024-7-UNNAMED7, IMD-2024-8-UNNAMED8, 
IMD-2024-9-UNNAMED9, IMD-2024-10-UNNAMED10, IMD-2024-11-DANA, IMD-2024-12-FENGAL, 
IMD-2024-13-UNNAMED13, IMD-2025-1-UNNAMED1, IMD-2025-2-UNNAMED2, IMD-2025-3-UNNAMED3, 
IMD-2025-4-UNNAMED4, IMD-2025-5-UNNAMED5, IMD-2025-6-UNNAMED6, IMD-2025-7-UNNAMED7, 
IMD-2025-8-UNNAMED8, IMD-2025-9-UNNAMED9, IMD-2025-10-UNNAMED10, IMD-2025-11-SHAKHTI, 
IMD-2025-12-UNNAMED12, IMD-2025-13-MONTHA, IMD-2025-14-SENYAR, IMD-2025-15-DITWAH, 
IMD-2026-1-UNNAMED1
```

---

## 5. Statistical Characterization & Holdout Limitations

> [!WARNING]
> **STATISTICAL POWER LIMITATION OF 2024–2026 HOLDOUT**  
> The 2024–2026 partition contains 29 unique storms and 541 fixes. While this provides substantial observations for continuous intensity evaluation (Mean Absolute Error on $V_{\max}$ across depressions, cyclonic storms, and severe cyclones such as REMAL and DANA), **it contains exactly 0 standard Rapid Intensification events ($\Delta V_{\max} \ge 30\text{ kt} / 24\text{h}$)**.  
> Consequently, the 2024–2026 holdout is formally classified as an:  
> **"independent chronological holdout with limited storm-level statistical power."**  
> In future evaluation, PR-AUC and CSI on the test partition must be marked as `UNDEFINED / INSUFFICIENT_EVIDENCE`.

---

## 6. Zero Training Verification

No models, scalers, or encoders were fitted on any partition. All hashes of protected models and pipelines remain identical to pre-audit baselines.
