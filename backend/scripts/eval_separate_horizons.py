"""
Evaluation script for separate horizon models:
1. event_within_0d (active event)
2. event_within_1d (active OR within 1d)
3. event_within_2d (active OR within 2d)
4. event_within_3d (active OR within 3d)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix
)

PARQUET_PATH = 'backend/data/historical/ml_features_multibasin.parquet'
df = pd.read_parquet(PARQUET_PATH)

EXCLUDE_COLUMNS = [
    'date', 'day_index', 'mode', 'site_id', 'split', 'basin', 'parsed_date',
    'event_active', 'is_active_event', 'event_present',
    'event_within_1d', 'event_within_2d', 'event_within_3d',
    'lead_0', 'lead_1', 'lead_2', 'lead_3', 'lead_days',
    'label_status', 'event_type', 'event_id', 'event_name', 'severity',
    'secondary_event_id', 'label_source', 'label_confidence'
]
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS and df[c].dtype in ('float64', 'float32')]
print(f"Canonical feature count: {len(feature_cols)}")

train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()
test_df = df[df['split'] == 'test'].copy()

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values

targets = {
    '0d': {
        'name': 'event_within_0d',
        'y_tr': (train_df['event_active'] == 1).astype(int).values,
        'y_val': (val_df['event_active'] == 1).astype(int).values,
        'y_test': (test_df['event_active'] == 1).astype(int).values,
    },
    '1d': {
        'name': 'event_within_1d',
        'y_tr': ((train_df['event_active'] == 1) | (train_df['event_within_1d'] == 1)).astype(int).values,
        'y_val': ((val_df['event_active'] == 1) | (val_df['event_within_1d'] == 1)).astype(int).values,
        'y_test': ((test_df['event_active'] == 1) | (test_df['event_within_1d'] == 1)).astype(int).values,
    },
    '2d': {
        'name': 'event_within_2d',
        'y_tr': ((train_df['event_active'] == 1) | (train_df['event_within_2d'] == 1)).astype(int).values,
        'y_val': ((val_df['event_active'] == 1) | (val_df['event_within_2d'] == 1)).astype(int).values,
        'y_test': ((test_df['event_active'] == 1) | (test_df['event_within_2d'] == 1)).astype(int).values,
    },
    '3d': {
        'name': 'event_within_3d',
        'y_tr': ((train_df['event_active'] == 1) | (train_df['event_within_3d'] == 1)).astype(int).values,
        'y_val': ((val_df['event_active'] == 1) | (val_df['event_within_3d'] == 1)).astype(int).values,
        'y_test': ((test_df['event_active'] == 1) | (test_df['event_within_3d'] == 1)).astype(int).values,
    },
}

summary_rows = []

for h, t_info in targets.items():
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42)
    rf.fit(X_train, t_info['y_tr'])
    val_probs = rf.predict_proba(X_val)[:, 1]
    test_probs = rf.predict_proba(X_test)[:, 1]
    
    val_pr_auc = average_precision_score(t_info['y_val'], val_probs)
    val_roc_auc = roc_auc_score(t_info['y_val'], val_probs)
    
    # Threshold sweep on validation
    best_th = 0.27
    best_criterion = -1
    best_val_row = {}
    
    # We test thresholds from 0.15 to 0.50
    for th in np.arange(0.15, 0.51, 0.01):
        th_round = round(float(th), 2)
        v_pred = (val_probs >= th_round).astype(int)
        rec = recall_score(t_info['y_val'], v_pred, zero_division=0)
        prec = precision_score(t_info['y_val'], v_pred, zero_division=0)
        f1 = f1_score(t_info['y_val'], v_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(t_info['y_val'], v_pred)
        
        # Validation event detection check
        val_df['p'] = v_pred
        ev_arb = val_df[val_df['event_name'] == 'Depression ARB 01']['p'].sum() > 0
        ev_dana = val_df[val_df['event_name'] == 'Severe Cyclonic Storm Dana']['p'].sum() > 0
        both_events = ev_arb and ev_dana
        
        # Primary objective: detect both validation events if possible, then maximize F1
        criterion = (10.0 if both_events else (5.0 if (ev_arb or ev_dana) else 0.0)) + f1
        if criterion > best_criterion:
            best_criterion = criterion
            best_th = th_round
            best_val_row = {
                'threshold': best_th,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'balanced_accuracy': bal_acc,
                'arb01_detected': ev_arb,
                'dana_detected': ev_dana,
            }
            
    # Evaluate frozen threshold on Test
    t_pred = (test_probs >= best_val_row['threshold']).astype(int)
    t_prec = precision_score(t_info['y_test'], t_pred, zero_division=0)
    t_rec = recall_score(t_info['y_test'], t_pred, zero_division=0)
    t_f1 = f1_score(t_info['y_test'], t_pred, zero_division=0)
    t_bal_acc = balanced_accuracy_score(t_info['y_test'], t_pred)
    t_pr_auc = average_precision_score(t_info['y_test'], test_probs)
    t_roc_auc = roc_auc_score(t_info['y_test'], test_probs)
    
    tn, fp, fn, tp = confusion_matrix(t_info['y_test'], t_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    test_df['p'] = t_pred
    fengal_det = test_df[test_df['event_name'] == 'Cyclonic Storm Fengal']['p'].sum() > 0
    fengal_rows = test_df[test_df['event_name'] == 'Cyclonic Storm Fengal']
    fengal_dates_detected = fengal_rows[fengal_rows['p'] == 1]['date'].tolist()
    
    summary_rows.append({
        'horizon': h,
        'target': t_info['name'],
        'train_pos': int(t_info['y_tr'].sum()),
        'val_pos': int(t_info['y_val'].sum()),
        'test_pos': int(t_info['y_test'].sum()),
        'val_pr_auc': val_pr_auc,
        'val_roc_auc': val_roc_auc,
        'frozen_threshold': best_val_row['threshold'],
        'val_f1': best_val_row['f1'],
        'val_recall': best_val_row['recall'],
        'val_precision': best_val_row['precision'],
        'test_precision': t_prec,
        'test_recall': t_rec,
        'test_f1': t_f1,
        'test_bal_acc': t_bal_acc,
        'test_pr_auc': t_pr_auc,
        'test_roc_auc': t_roc_auc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'fpr': fpr,
        'fengal_detected': bool(fengal_det),
        'fengal_dates': fengal_dates_detected,
    })

res_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 115)
print("HORIZON-SPECIFIC MODEL EVALUATION COMPARISON TABLE")
print("=" * 115)
header = f"{'Horizon':7s} | {'Target':16s} | {'TrainPos':8s} | {'ValPos':6s} | {'TestPos':7s} | {'Thresh':6s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'PR-AUC':7s} | {'ROC-AUC':7s} | {'Fengal Det':10s}"
print(header)
print("-" * 115)
for r in summary_rows:
    row = f"{r['horizon']:7s} | {r['target']:16s} | {r['train_pos']:8d} | {r['val_pos']:6d} | {r['test_pos']:7d} | {r['frozen_threshold']:6.2f} | {r['test_precision']:6.4f} | {r['test_recall']:6.4f} | {r['test_f1']:6.4f} | {r['test_pr_auc']:7.4f} | {r['test_roc_auc']:7.4f} | {str(r['fengal_detected']):10s}"
    print(row)
print("=" * 115)

print("\n" + "=" * 110)
print("DETAILED TEST PERFORMANCE BY HORIZON:")
print("=" * 110)
for r in summary_rows:
    print(f"\nHorizon: {r['horizon']} ({r['target']})")
    print(f"  Class Balance: Train Pos={r['train_pos']}/134, Val Pos={r['val_pos']}/88, Test Pos={r['test_pos']}/152")
    print(f"  Validation Selected Threshold: {r['frozen_threshold']} (Val F1: {r['val_f1']:.4f}, Val Recall: {r['val_recall']:.4f}, Val Prec: {r['val_precision']:.4f})")
    print(f"  Test Evaluation: Recall={r['test_recall']:.4f}, Precision={r['test_precision']:.4f}, F1={r['test_f1']:.4f}, PR-AUC={r['test_pr_auc']:.4f}, ROC-AUC={r['test_roc_auc']:.4f}")
    print(f"  Confusion Matrix: TP={r['tp']}, FP={r['fp']}, TN={r['tn']}, FN={r['fn']} (FPR: {r['fpr']*100:.1f}%)")
    print(f"  Unseen Fengal Detected: {r['fengal_detected']} (Dates: {r['fengal_dates']})")
