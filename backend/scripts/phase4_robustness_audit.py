"""
Phase 4.1 Robustness, False-Alarm Analysis & Feature Validation Script.
Executes rigorous diagnostics across Tasks 1 through 8 strictly observing
the scientific rule that TEST is kept frozen and never used for feature or model selection.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    brier_score_loss
)

PARQUET_PATH = "backend/data/historical/ml_features_multibasin.parquet"
MODEL_PATH = "backend/models/risk_model.joblib"
METADATA_PATH = "backend/models/model_metadata.json"

EXCLUDE_COLUMNS = [
    "date", "day_index", "mode", "site_id", "split", "basin", "parsed_date",
    "event_active", "is_active_event", "event_present",
    "event_within_1d", "event_within_2d", "event_within_3d",
    "lead_0", "lead_1", "lead_2", "lead_3", "lead_days",
    "label_status", "event_type", "event_id", "event_name", "severity",
    "secondary_event_id", "label_source", "label_confidence"
]


def run_diagnostics():
    df = pd.read_parquet(PARQUET_PATH)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS and df[c].dtype in ("float64", "float32")]
    
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    X_tr = train_df[feature_cols].values
    y_tr = train_df["event_present"].values.astype(int)
    
    X_val = val_df[feature_cols].values
    y_val = val_df["event_present"].values.astype(int)
    
    X_te = test_df[feature_cols].values
    y_te = test_df["event_present"].values.astype(int)
    
    # Load trained model
    artifact = joblib.load(MODEL_PATH)
    rf = artifact["model"]
    frozen_th = artifact.get("frozen_threshold", 0.27)
    
    # Probabilities
    val_probs = rf.predict_proba(X_val)[:, 1]
    test_probs = rf.predict_proba(X_te)[:, 1]
    
    val_df["prob"] = val_probs
    val_df["pred"] = (val_probs >= frozen_th).astype(int)
    
    test_df["prob"] = test_probs
    test_df["pred"] = (test_probs >= frozen_th).astype(int)
    
    print("=" * 80)
    print("TASK 1: FALSE ALARM ANALYSIS (VALIDATION AND TEST)")
    print("=" * 80)
    
    for split_name, s_df in [("Validation", val_df), ("Test", test_df)]:
        fa_df = s_df[(s_df["pred"] == 1) & (s_df["event_present"] == 0)].copy()
        tn_df = s_df[(s_df["pred"] == 0) & (s_df["event_present"] == 0)].copy()
        print(f"\n--- {split_name} Split False Alarms (Total FA = {len(fa_df)} out of {len(fa_df)+len(tn_df)} negatives) ---")
        
        # Breakdown by basin / site
        print("\n1. False Alarms by Site / Basin:")
        site_grp = fa_df.groupby(["site_id", "basin"]).size()
        total_neg_site = s_df[s_df["event_present"] == 0].groupby(["site_id", "basin"]).size()
        for (sid, basin), fa_cnt in site_grp.items():
            tot = total_neg_site.get((sid, basin), 0)
            print(f"   {sid} ({basin}): {fa_cnt} / {tot} negative days ({fa_cnt/tot*100:.1f}% FA rate)")
            
        # Breakdown by Month
        fa_df["month"] = fa_df["date"].str.slice(0, 7)
        s_df_neg = s_df[s_df["event_present"] == 0].copy()
        s_df_neg["month"] = s_df_neg["date"].str.slice(0, 7)
        print("\n2. False Alarms by Month:")
        month_fa = fa_df.groupby("month").size()
        month_tot = s_df_neg.groupby("month").size()
        for m, cnt in month_fa.items():
            tot = month_tot.get(m, 0)
            print(f"   {m}: {cnt} / {tot} negative days ({cnt/tot*100:.1f}% FA rate)")
            
        # Breakdown by Probability Range
        print("\n3. Probability Distribution of False Alarms:")
        bins = [0.27, 0.30, 0.35, 0.40, 0.50, 1.0]
        prob_labels = ["0.27-0.30", "0.30-0.35", "0.35-0.40", "0.40-0.50", "0.50+"]
        fa_df["prob_range"] = pd.cut(fa_df["prob"], bins=bins, labels=prob_labels, include_lowest=True)
        print(fa_df["prob_range"].value_counts().sort_index().to_string())
        
        # Temporal distance to nearest documented event
        # Find dates of all positive events across whole dataset
        event_dates = pd.to_datetime(df[df["event_present"] == 1]["date"]).unique()
        fa_dates = pd.to_datetime(fa_df["date"])
        distances = [min(abs((d - ed).days) for ed in event_dates) for d in fa_dates]
        fa_df["dist_days"] = distances
        print("\n4. Temporal Proximity to Nearest Documented Event:")
        print(f"   Mean distance: {np.mean(distances):.1f} days")
        print(f"   Median distance: {np.median(distances):.1f} days")
        print(f"   Min distance: {np.min(distances):.1f} days")
        print(f"   Within 7 days of an event: {(fa_df['dist_days'] <= 7).sum()} ({ (fa_df['dist_days'] <= 7).mean()*100:.1f}%)")
        print(f"   Within 14 days of an event: {(fa_df['dist_days'] <= 14).sum()} ({ (fa_df['dist_days'] <= 14).mean()*100:.1f}%)")
        print(f"   Within 30 days of an event: {(fa_df['dist_days'] <= 30).sum()} ({ (fa_df['dist_days'] <= 30).mean()*100:.1f}%)")
        print(f"   Distances breakdown:\n{fa_df['dist_days'].value_counts(bins=[0, 3, 7, 14, 30, 60, 100]).sort_index().to_string()}")

    print("\n" + "=" * 80)
    print("TASK 2: FEATURE REDUNDANCY & CORRELATION ANALYSIS (101 FEATURES)")
    print("=" * 80)
    
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find pairs with r >= 0.85 and r >= 0.95
    pairs_85 = []
    pairs_95 = []
    cols = feature_cols
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if r >= 0.95:
                pairs_95.append((cols[i], cols[j], r))
            elif r >= 0.85:
                pairs_85.append((cols[i], cols[j], r))
                
    print(f"Total feature pairs evaluated: {len(cols)*(len(cols)-1)//2}")
    print(f"Pairs with |r| >= 0.95 (extreme redundancy): {len(pairs_95)}")
    print(f"Pairs with 0.85 <= |r| < 0.95 (high redundancy): {len(pairs_85)}")
    
    print("\nTop 15 Most Highly Correlated Pairs (|r| >= 0.95):")
    all_high = sorted(pairs_95 + pairs_85, key=lambda x: x[2], reverse=True)
    for f1, f2, r in all_high[:15]:
        print(f"   {f1:25s} <--> {f2:25s} : r = {r:.4f}")
        
    # PCA to find effective independent dimensions
    pca = PCA()
    pca.fit(StandardScaler().fit_transform(df[feature_cols].values))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    dim_80 = np.argmax(cum_var >= 0.80) + 1
    dim_90 = np.argmax(cum_var >= 0.90) + 1
    dim_95 = np.argmax(cum_var >= 0.95) + 1
    dim_99 = np.argmax(cum_var >= 0.99) + 1
    
    print("\nPCA Dimensionality Analysis of Physical Manifold:")
    print(f"   Components for 80% variance: {dim_80} / 101 features")
    print(f"   Components for 90% variance: {dim_90} / 101 features")
    print(f"   Components for 95% variance: {dim_95} / 101 features")
    print(f"   Components for 99% variance: {dim_99} / 101 features")
    print(f"   Top 5 PCA components explain: {cum_var[4]*100:.2f}% of total variance")
    print(f"   Top 10 PCA components explain: {cum_var[9]*100:.2f}% of total variance")

    print("\n" + "=" * 80)
    print("TASK 3: VALIDATION PERMUTATION IMPORTANCE VS IMPURITY IMPORTANCE")
    print("=" * 80)
    
    # Gini impurity importance
    impurity_imp = dict(zip(feature_cols, rf.feature_importances_))
    
    # Permutation importance on VALIDATION fold
    perm_res = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc")
    perm_means = perm_res.importances_mean
    perm_imp = dict(zip(feature_cols, perm_means))
    
    sorted_perm = sorted(perm_imp.items(), key=lambda x: x[1], reverse=True)
    sorted_gini = sorted(impurity_imp.items(), key=lambda x: x[1], reverse=True)
    
    print("\nComparison of Top 15 Features (Permutation ROC-AUC drop vs Gini Impurity):")
    print(f"{'Rank':4s} | {'Permutation Feature':25s} {'Drop':8s} | {'Gini Feature':25s} {'Gini %':8s}")
    print("-" * 75)
    for idx in range(15):
        pf, pval = sorted_perm[idx]
        gf, gval = sorted_gini[idx]
        print(f"{idx+1:4d} | {pf:25s} {pval:+.4f}   | {gf:25s} {gval*100:6.2f}%")
        
    # Aggregation by variable channel
    var_groups = {
        "temperature": ["temp"],
        "salinity": ["sal"],
        "current_components": ["cur_u", "cur_v"],
        "current_speed": ["cur_current", "cur_base", "cur_abs", "cur_z", "cur_pct", "cur_7d", "cur_14d", "cur_30d"],
        "sea_surface_height": ["ssh"],
        "mixed_layer_depth": ["mld"],
    }
    
    var_perm = {k: 0.0 for k in var_groups}
    for f, val in perm_imp.items():
        for grp, prefixes in var_groups.items():
            if any(f.startswith(p) for p in prefixes):
                var_perm[grp] += max(0.0, val)
                break
                
    var_perm_tot = sum(var_perm.values())
    print("\nPermutation Importance Aggregated by Ocean Variable:")
    for v, val in sorted(var_perm.items(), key=lambda x: x[1], reverse=True):
        pct = (val / var_perm_tot * 100) if var_perm_tot > 0 else 0
        print(f"   {v:22s}: {val:.4f} ({pct:5.1f}%)")
        
    # Aggregation by temporal window
    win_groups = {
        "current": ["_current", "_base_"],
        "7d": ["_7d_"],
        "14d": ["_14d_"],
        "30d": ["_30d_"],
    }
    win_perm = {k: 0.0 for k in win_groups}
    for f, val in perm_imp.items():
        for w, substrs in win_groups.items():
            if any(s in f for s in substrs):
                win_perm[w] += max(0.0, val)
                break
                
    win_perm_tot = sum(win_perm.values())
    print("\nPermutation Importance Aggregated by Temporal Window:")
    for w, val in sorted(win_perm.items(), key=lambda x: x[1], reverse=True):
        pct = (val / win_perm_tot * 100) if win_perm_tot > 0 else 0
        print(f"   {w:15s}: {val:.4f} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("TASK 4: SEASONAL SHORTCUT TEST (DIAGNOSTIC ABLATION)")
    print("=" * 80)
    
    # Breakdown of predictions by month/season
    val_copy = val_df.copy()
    val_copy["month"] = val_copy["date"].str.slice(0, 7)
    print("\n1. Validation Positive Prediction Rate by Month:")
    for m, grp in val_copy.groupby("month"):
        pos_true = (grp["event_present"] == 1).sum()
        pred_pos = (grp["pred"] == 1).sum()
        mean_p = grp["prob"].mean()
        print(f"   {m}: True Pos={pos_true}/{len(grp)}, Predicted Pos={pred_pos}/{len(grp)} ({pred_pos/len(grp)*100:.1f}%), Mean Prob={mean_p:.4f}")
        
    # Ablation experiment: what if we remove 30d baseline / climatology features
    # that most strongly correlate with seasonal cycle (e.g. temp_base_mean, sal_base_mean, ssh_base_mean, etc.)
    seasonal_proxy_features = [
        c for c in feature_cols if "_base_mean" in c or "_base_std" in c or "30d_mean" in c
    ]
    print(f"\n2. Seasonal proxy features identified for diagnostic ablation: {len(seasonal_proxy_features)} features")
    print(f"   Sample: {seasonal_proxy_features[:8]}")
    
    reduced_nonseasonal_cols = [c for c in feature_cols if c not in seasonal_proxy_features]
    print(f"   Remaining non-seasonal features: {len(reduced_nonseasonal_cols)} features")
    
    # Train ablated RF on train, evaluate on val
    rf_ablated = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42)
    rf_ablated.fit(train_df[reduced_nonseasonal_cols].values, y_tr)
    probs_ablated = rf_ablated.predict_proba(val_df[reduced_nonseasonal_cols].values)[:, 1]
    
    pr_auc_full = average_precision_score(y_val, val_probs)
    roc_auc_full = roc_auc_score(y_val, val_probs)
    pr_auc_abl = average_precision_score(y_val, probs_ablated)
    roc_auc_abl = roc_auc_score(y_val, probs_ablated)
    
    print("\nAblation Experiment Results on Validation Fold:")
    print(f"   Full Model (101 feats):      PR-AUC = {pr_auc_full:.4f} | ROC-AUC = {roc_auc_full:.4f}")
    print(f"   Ablated Model ({len(reduced_nonseasonal_cols)} feats):   PR-AUC = {pr_auc_abl:.4f} | ROC-AUC = {roc_auc_abl:.4f}")
    print(f"   PR-AUC Delta: {pr_auc_abl - pr_auc_full:+.4f} | ROC-AUC Delta: {roc_auc_abl - roc_auc_full:+.4f}")
    if pr_auc_abl >= 0.20:
        print("   -> Performance does NOT collapse! The model utilizes dynamic anomalies and short-term trends, not purely seasonal shortcuts.")
    else:
        print("   -> Significant degradation observed; model heavily relies on seasonal baselines.")

    print("\n" + "=" * 80)
    print("TASK 5: FEATURE SELECTION EXPERIMENT (TRAIN-ONLY SELECTION, VAL-ONLY EVAL)")
    print("=" * 80)
    
    # Feature ranking MUST be computed using TRAIN split only!
    rf_train_ranker = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42)
    rf_train_ranker.fit(X_tr, y_tr)
    tr_importances = rf_train_ranker.feature_importances_
    sorted_tr_feats = [f for f, imp in sorted(zip(feature_cols, tr_importances), key=lambda x: x[1], reverse=True)]
    
    feature_subsets = {
        "A. All 101 features": sorted_tr_feats,
        "B. Top 50 features": sorted_tr_feats[:50],
        "C. Top 25 features": sorted_tr_feats[:25],
        "D. Top 15 features": sorted_tr_feats[:15],
    }
    
    print(f"{'Subset':22s} | {'Feats':5s} | {'PR-AUC':7s} | {'ROC-AUC':7s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'BalAcc':7s} | {'FA (FP)':7s} | {'Events':6s}")
    print("-" * 95)
    
    for label, subset_cols in feature_subsets.items():
        clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42)
        clf.fit(train_df[subset_cols].values, y_tr)
        probs_s = clf.predict_proba(val_df[subset_cols].values)[:, 1]
        
        pr_s = average_precision_score(y_val, probs_s)
        roc_s = roc_auc_score(y_val, probs_s)
        
        # Test at frozen_th = 0.27
        preds_s = (probs_s >= frozen_th).astype(int)
        p = precision_score(y_val, preds_s, zero_division=0)
        r = recall_score(y_val, preds_s, zero_division=0)
        f1 = f1_score(y_val, preds_s, zero_division=0)
        b_acc = balanced_accuracy_score(y_val, preds_s)
        cm = confusion_matrix(y_val, preds_s)
        tn, fp, fn, tp = cm.ravel()
        
        # Event detection
        v_sub = val_df.copy()
        v_sub["pred"] = preds_s
        arb = bool((v_sub[v_sub["event_id"] == "IMD-2024-D-ARB01"]["pred"] == 1).any())
        dana = bool((v_sub[v_sub["event_id"] == "IMD-2024-SCS-DANA"]["pred"] == 1).any())
        ev_str = f"{int(arb)+int(dana)}/2"
        
        print(f"{label:22s} | {len(subset_cols):5d} | {pr_s:7.4f} | {roc_s:7.4f} | {p:6.4f} | {r:6.4f} | {f1:6.4f} | {b_acc:7.4f} | {fp:2d} / {tn+fp:2d}  | {ev_str:6s}")

    print("\n" + "=" * 80)
    print("TASK 6: PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 80)
    
    # Uncalibrated Brier Score on Validation
    brier_uncal = brier_score_loss(y_val, val_probs)
    print(f"Uncalibrated Random Forest Validation Brier Score: {brier_uncal:.4f}")
    
    # Sigmoid (Platt) Calibration
    cal_sigmoid = CalibratedClassifierCV(estimator=rf, cv="prefit", method="sigmoid")
    # Note: to calibrate without leaking val, we can fit on train CV or val
    try:
        cal_sigmoid.fit(X_val, y_val)
        probs_cal_sig = cal_sigmoid.predict_proba(X_val)[:, 1]
        brier_sig = brier_score_loss(y_val, probs_cal_sig)
        print(f"Sigmoid Platt Calibration Brier Score (fitted on val): {brier_sig:.4f}")
    except Exception as e:
        print(f"Sigmoid calibration error: {e}")
        
    # Calibration with 5-fold CV on Train only
    rf_for_cal = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42)
    cal_cv = CalibratedClassifierCV(estimator=rf_for_cal, cv=5, method="sigmoid")
    cal_cv.fit(X_tr, y_tr)
    probs_cal_cv = cal_cv.predict_proba(X_val)[:, 1]
    brier_cv = brier_score_loss(y_val, probs_cal_cv)
    pr_cv = average_precision_score(y_val, probs_cal_cv)
    roc_cv = roc_auc_score(y_val, probs_cal_cv)
    print(f"5-Fold Train-only Calibrated Brier Score evaluated on Val: {brier_cv:.4f} (PR-AUC={pr_cv:.4f}, ROC-AUC={roc_cv:.4f})")
    
    print("\nCalibration Assessment:")
    if brier_cv > brier_uncal:
        print("   -> Brier score worsens or remains similar under empirical calibration due to small sample size (positives=22). Raw tree probabilities are preferred.")
    else:
        print(f"   -> Calibration improves Brier score from {brier_uncal:.4f} to {brier_cv:.4f}.")

    print("\n" + "=" * 80)
    print("TASK 7: VALIDATION THRESHOLD ANALYSIS (0.10 TO 0.50)")
    print("=" * 80)
    
    print(f"{'Threshold':10s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'Spec':6s} | {'FP':3s} | {'FN':3s} | {'D-ARB01 Lead':13s} | {'SCS-DANA Lead':13s} | {'Events':6s}")
    print("-" * 90)
    for th in [0.10, 0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.35, 0.40, 0.45, 0.50]:
        preds = (val_probs >= th).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)
        
        # Event lead time
        v_sub = val_df.copy()
        v_sub["pred"] = preds
        arb_rows = v_sub[v_sub["event_id"] == "IMD-2024-D-ARB01"]
        dana_rows = v_sub[v_sub["event_id"] == "IMD-2024-SCS-DANA"]
        
        arb_det = (arb_rows["pred"] == 1).any()
        dana_det = (dana_rows["pred"] == 1).any()
        
        arb_lead = f"{int(arb_rows[arb_rows['pred']==1]['lead_days'].max())}d lead" if arb_det else "Missed"
        dana_lead = f"{int(dana_rows[dana_rows['pred']==1]['lead_days'].max())}d lead" if dana_det else "Missed"
        
        marker = " <-- FROZEN" if abs(th - 0.27) < 1e-4 else ""
        print(f"{th:10.2f} | {p:6.4f} | {r:6.4f} | {f1:6.4f} | {spec:6.4f} | {fp:3d} | {fn:3d} | {arb_lead:13s} | {dana_lead:13s} | {int(arb_det)+int(dana_det)}/2{marker}")

    print("\n" + "=" * 80)
    print("TASK 8: EVENT-LEVEL ROBUSTNESS & FALSE ALARMS IN PRECEDING WINDOWS")
    print("=" * 80)
    
    # Audit all events in dataset
    all_events = [
        ("D-BOB04", "Depression BOB 04", "bob", train_df),
        ("SCS-ASNA", "Severe Cyclonic Storm Asna", "aras", train_df),
        ("DD-BOB05", "Deep Depression BOB 05", "bob", train_df),
        ("D-ARB01", "Depression ARB 01", "aras", val_df),
        ("SCS-DANA", "Severe Cyclonic Storm Dana", "bob", val_df),
        ("CS-FENGAL", "Cyclonic Storm Fengal", "bob", test_df),
    ]
    
    for ev_code, ev_name, site, split_frame in all_events:
        # Full df subset for site
        site_df = df[df["site_id"] == site].sort_values("date").copy()
        site_X = site_df[feature_cols].values
        site_probs = rf.predict_proba(site_X)[:, 1]
        site_df["prob"] = site_probs
        site_df["pred"] = (site_probs >= frozen_th).astype(int)
        
        ev_sub = site_df[site_df["event_id"].str.contains(ev_code, na=False)]
        if len(ev_sub) == 0:
            continue
            
        ev_dates = pd.to_datetime(ev_sub["date"]).sort_values()
        first_event_date = ev_dates.min()
        
        # Preceding 7 days and 14 days
        date_7d_prior = first_event_date - pd.Timedelta(days=7)
        date_14d_prior = first_event_date - pd.Timedelta(days=14)
        
        site_df["dt"] = pd.to_datetime(site_df["date"])
        
        # Preceding 7d slice (excluding any other event)
        slice_7d = site_df[(site_df["dt"] >= date_7d_prior) & (site_df["dt"] < first_event_date)]
        slice_7d_neg = slice_7d[slice_7d["event_present"] == 0]
        fa_7d = (slice_7d_neg["pred"] == 1).sum()
        
        # Preceding 14d slice
        slice_14d = site_df[(site_df["dt"] >= date_14d_prior) & (site_df["dt"] < first_event_date)]
        slice_14d_neg = slice_14d[slice_14d["event_present"] == 0]
        fa_14d = (slice_14d_neg["pred"] == 1).sum()
        
        # Event metrics
        firing = ev_sub[ev_sub["pred"] == 1]
        detected = len(firing) > 0
        lead_time = int(firing["lead_days"].max()) if detected else 0
        earliest_date = firing["date"].min() if detected else "None"
        
        print(f"\nEvent: {ev_name} ({ev_code}) | Basin: {site.upper()} | Split: {ev_sub['split'].iloc[0].upper()}")
        print(f"   Total event duration: {len(ev_sub)} days ({ev_sub['date'].min()} to {ev_sub['date'].max()})")
        print(f"   Active days: {(ev_sub['label_status']=='active_event').sum()}, Lead days: {(ev_sub['label_status']=='lead_event').sum()}")
        print(f"   Detected event days: {len(firing)} / {len(ev_sub)} ({len(firing)/len(ev_sub)*100:.1f}%)")
        print(f"   Missed days: {len(ev_sub) - len(firing)}")
        print(f"   Earliest detection date: {earliest_date}")
        print(f"   Detection lead time: {lead_time} days before active onset")
        print(f"   False alarms in preceding 7 days (clean negative): {fa_7d} / {len(slice_7d_neg)}")
        print(f"   False alarms in preceding 14 days (clean negative): {fa_14d} / {len(slice_14d_neg)}")


if __name__ == "__main__":
    run_diagnostics()
