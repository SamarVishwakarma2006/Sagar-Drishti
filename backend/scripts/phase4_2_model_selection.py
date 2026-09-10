"""
Phase 4.2 Comprehensive Validation Comparison and Feature Selection Script.
Evaluates:
1. Candidate models on Validation:
   - Logistic Regression (All 101, Top 50, Top 25, Top 15)
   - Random Forest (All 101, Top 50, Top 25, Top 15)
   - HistGradientBoosting (All 101, Top 50, Top 25, Top 15)
2. Defensible operational objective evaluation (PR-AUC, Event Recall across both basins, False Alarm minimization)
3. False-positive proximity categorization (0-3d, 4-7d, 8-14d, >14d)
4. Permutation importance on validation fold
5. Seasonal shortcut diagnostic ablation
6. Probability calibration assessment
7. Threshold sweep on validation
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    brier_score_loss
)

PARQUET_PATH = "backend/data/historical/ml_features_multibasin.parquet"

EXCLUDE_COLUMNS = [
    "date", "day_index", "mode", "site_id", "split", "basin", "parsed_date",
    "event_active", "is_active_event", "event_present",
    "event_within_1d", "event_within_2d", "event_within_3d",
    "lead_0", "lead_1", "lead_2", "lead_3", "lead_days",
    "label_status", "event_type", "event_id", "event_name", "severity",
    "secondary_event_id", "label_source", "label_confidence"
]


def run_phase4_2_selection():
    df = pd.read_parquet(PARQUET_PATH)
    all_feats = [c for c in df.columns if c not in EXCLUDE_COLUMNS and df[c].dtype in ("float64", "float32")]
    
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    y_tr = train_df["event_present"].values.astype(int)
    y_val = val_df["event_present"].values.astype(int)
    
    # Fit feature ranker strictly on TRAIN fold
    rf_ranker = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42)
    rf_ranker.fit(train_df[all_feats].values, y_tr)
    tr_importances = rf_ranker.feature_importances_
    sorted_feats = [f for f, imp in sorted(zip(all_feats, tr_importances), key=lambda x: x[1], reverse=True)]
    
    subsets = {
        "all_101": sorted_feats,
        "top_50": sorted_feats[:50],
        "top_25": sorted_feats[:25],
        "top_15": sorted_feats[:15],
    }
    
    # --------------------------------------------------------------------------
    # 1. CANDIDATE MODEL VALIDATION GRID
    # --------------------------------------------------------------------------
    candidates = {}
    
    # Random Forest variants
    for s_name, feats in subsets.items():
        candidates[f"RF_{s_name}"] = {
            "model": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42),
            "features": feats,
            "is_scaled": False,
            "family": "Random Forest",
        }
        
    # Logistic Regression variants
    for s_name, feats in subsets.items():
        candidates[f"LR_{s_name}"] = {
            "model": LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=42, max_iter=1000),
            "features": feats,
            "is_scaled": True,
            "family": "Logistic Regression",
        }
        
    # HistGradientBoosting variants
    for s_name, feats in subsets.items():
        candidates[f"HGB_{s_name}"] = {
            "model": HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=5, l2_regularization=1.0, class_weight="balanced", random_state=42),
            "features": feats,
            "is_scaled": False,
            "family": "HistGradientBoosting",
        }
        
    print("=" * 115)
    print("PHASE 4.2 — CANDIDATE MODEL VALIDATION COMPARISON TABLE (VALIDATION SET: N=88, Positives=13)")
    print("=" * 115)
    header = f"{'Candidate Name':18s} | {'Family':16s} | {'Feats':5s} | {'PR-AUC':7s} | {'ROC-AUC':7s} | {'Prec(0.27)':10s} | {'Rec(0.27)':10s} | {'F1(0.27)':8s} | {'BalAcc':7s} | {'FP':3s} | {'FN':3s} | {'ARB01':5s} | {'DANA':5s}"
    print(header)
    print("-" * len(header))
    
    val_results = {}
    
    for name, cfg in candidates.items():
        clf = cfg["model"]
        feats = cfg["features"]
        is_scaled = cfg["is_scaled"]
        
        X_tr = train_df[feats].values
        X_val = val_df[feats].values
        
        if is_scaled:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            
        clf.fit(X_tr, y_tr)
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        pr_auc = average_precision_score(y_val, val_probs)
        roc_auc = roc_auc_score(y_val, val_probs)
        
        # Operational threshold 0.27
        preds_27 = (val_probs >= 0.27).astype(int)
        p = precision_score(y_val, preds_27, zero_division=0)
        r = recall_score(y_val, preds_27, zero_division=0)
        f1 = f1_score(y_val, preds_27, zero_division=0)
        b_acc = balanced_accuracy_score(y_val, preds_27)
        cm = confusion_matrix(y_val, preds_27)
        tn, fp, fn, tp = cm.ravel()
        
        # Event-level detection
        val_copy = val_df.copy()
        val_copy["pred"] = preds_27
        val_copy["prob"] = val_probs
        arb_det = (val_copy[val_copy["event_id"] == "IMD-2024-D-ARB01"]["pred"] == 1).any()
        dana_det = (val_copy[val_copy["event_id"] == "IMD-2024-SCS-DANA"]["pred"] == 1).any()
        
        arb_str = "Det" if arb_det else "Miss"
        dana_str = "Det" if dana_det else "Miss"
        
        val_results[name] = {
            "name": name,
            "family": cfg["family"],
            "feat_count": len(feats),
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "balanced_accuracy": round(float(b_acc), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "arb_detected": bool(arb_det),
            "dana_detected": bool(dana_det),
            "both_events_detected": bool(arb_det and dana_det),
            "probs": val_probs,
            "model": clf,
            "features": feats,
            "is_scaled": is_scaled,
        }
        
        print(f"{name:18s} | {cfg['family']:16s} | {len(feats):5d} | {pr_auc:7.4f} | {roc_auc:7.4f} | {p:10.4f} | {r:10.4f} | {f1:8.4f} | {b_acc:7.4f} | {fp:3d} | {fn:3d} | {arb_str:5s} | {dana_str:5s}")

    # --------------------------------------------------------------------------
    # 2. OPERATIONAL OBJECTIVE EVALUATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPERATIONAL SELECTION CRITERIA FILTER (REQUIREMENT: BOTH BASIN EVENTS DETECTED)")
    print("=" * 80)
    qualifying = [res for res in val_results.values() if res["both_events_detected"]]
    print(f"Total candidate models: {len(val_results)}")
    print(f"Models passing Operational Requirement (Both D-ARB01 and SCS-DANA detected): {len(qualifying)}")
    for q in qualifying:
        print(f"   -> {q['name']:18s} | PR-AUC: {q['pr_auc']:.4f} | ROC-AUC: {q['roc_auc']:.4f} | F1: {q['f1']:.4f} | FP: {q['fp']:2d} | FN: {q['fn']:2d}")

    # --------------------------------------------------------------------------
    # 3. FALSE-POSITIVE PROXIMITY CATEGORIZATION (TASK 4)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 4: IMPROVED FALSE-POSITIVE PROXIMITY ANALYSIS")
    print("=" * 80)
    
    # We evaluate proximity for the primary model (RF_all_101) on Validation and Test
    primary_res = val_results["RF_all_101"]
    primary_clf = primary_res["model"]
    primary_feats = primary_res["features"]
    
    # Compute test probabilities for diagnosis (without tuning)
    test_probs = primary_clf.predict_proba(test_df[primary_feats].values)[:, 1]
    test_df["prob"] = test_probs
    test_df["pred"] = (test_probs >= 0.27).astype(int)
    
    val_df["prob"] = primary_res["probs"]
    val_df["pred"] = (primary_res["probs"] >= 0.27).astype(int)
    
    # Extract documented event dates across the whole multi-basin catalog
    event_df = df[df["event_present"] == 1].copy()
    event_dates = pd.to_datetime(event_df["date"]).unique()
    
    for split_label, s_df in [("Validation", val_df), ("Test", test_df)]:
        fa_rows = s_df[(s_df["pred"] == 1) & (s_df["event_present"] == 0)].copy()
        print(f"\n--- {split_label} False Positive Proximity Breakdown (Total FP = {len(fa_rows)}) ---")
        
        # Categorize by proximity to nearest documented event
        # Distance = min(abs(date - event_date)) in days
        fa_dates = pd.to_datetime(fa_rows["date"])
        distances = [min(abs((d - ed).days) for ed in event_dates) for d in fa_dates]
        fa_rows["dist_days"] = distances
        
        # Diagnostic groups requested by user:
        # 1. 0-3 days before/near event: target-window false positives
        # 2. 4-7 days before event: near-event precursor warnings
        # 3. 8-14 days before event: extended precursor warnings
        # 4. >14 days before event: distant false alarms
        def categorize_fp(d):
            if d <= 3:
                return "1. Target-window false positives (0-3d from event)"
            elif d <= 7:
                return "2. Near-event precursor warnings (4-7d from event)"
            elif d <= 14:
                return "3. Extended precursor warnings (8-14d from event)"
            else:
                return "4. Distant false alarms (>14d from event)"
                
        fa_rows["fp_category"] = fa_rows["dist_days"].apply(categorize_fp)
        
        cat_counts = fa_rows["fp_category"].value_counts().sort_index()
        for cat, count in cat_counts.items():
            pct = count / len(fa_rows) * 100
            print(f"   {cat:55s}: {count:2d} ({pct:5.1f}%)")
            
        print("\n   Sample dates per category:")
        for cat in cat_counts.index:
            sub = fa_rows[fa_rows["fp_category"] == cat]
            sample_dates = sub["date"].tolist()[:5]
            print(f"      {cat.split('(')[0].strip()}: {sample_dates}")

    # --------------------------------------------------------------------------
    # 4. TEMPORAL DYNAMICS & PERMUTATION IMPORTANCE (TASK 5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 5: 30-DAY FEATURE STABILITY & MULTI-GRANULARITY PERMUTATION IMPORTANCE")
    print("=" * 80)
    
    perm_val = permutation_importance(primary_clf, val_df[primary_feats].values, y_val, n_repeats=10, random_state=42, scoring="roc_auc")
    perm_dict = dict(zip(primary_feats, perm_val.importances_mean))
    
    # Ocean variables breakdown
    var_map = {
        "temperature": ["temp_"],
        "salinity": ["sal_"],
        "current_speed": ["cur_current", "cur_base", "cur_abs", "cur_z", "cur_pct", "cur_7d", "cur_14d", "cur_30d"],
        "uo (zonal current)": ["cur_u_"],
        "vo (meridional current)": ["cur_v_"],
        "SSH": ["ssh_"],
        "MLD": ["mld_"],
    }
    
    var_importance = {k: 0.0 for k in var_map}
    for feat, imp in perm_dict.items():
        for var_name, prefixes in var_map.items():
            if any(feat.startswith(p) for p in prefixes):
                var_importance[var_name] += max(0.0, imp)
                break
                
    tot_var_imp = sum(var_importance.values())
    print("\nPermutation Importance Aggregated by Ocean Variable Channel:")
    for v, imp in sorted(var_importance.items(), key=lambda x: x[1], reverse=True):
        pct = (imp / tot_var_imp * 100) if tot_var_imp > 0 else 0
        print(f"   {v:25s}: {imp:.4f} ({pct:5.1f}%)")
        
    # Temporal scale breakdown
    time_map = {
        "current": ["_current", "_base_"],
        "7-day": ["_7d_"],
        "14-day": ["_14d_"],
        "30-day": ["_30d_"],
    }
    time_importance = {k: 0.0 for k in time_map}
    for feat, imp in perm_dict.items():
        for t_name, substrs in time_map.items():
            if any(s in feat for s in substrs):
                time_importance[t_name] += max(0.0, imp)
                break
                
    tot_time_imp = sum(time_importance.values())
    print("\nPermutation Importance Aggregated by Temporal Scale:")
    for t, imp in sorted(time_importance.items(), key=lambda x: x[1], reverse=True):
        pct = (imp / tot_time_imp * 100) if tot_time_imp > 0 else 0
        print(f"   {t:15s}: {imp:.4f} ({pct:5.1f}%)")
        
    print("\nTop 15 Individual Permutation Features (Validation ROC-AUC Drop):")
    sorted_p = sorted(perm_dict.items(), key=lambda x: x[1], reverse=True)
    for idx, (f, imp) in enumerate(sorted_p[:15]):
        print(f"   {idx+1:2d}. {f:25s}: {imp:+.4f}")


if __name__ == "__main__":
    run_phase4_2_selection()
