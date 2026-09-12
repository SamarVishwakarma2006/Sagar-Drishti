import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix
)

df = pd.read_parquet('backend/data/historical/ml_features_multibasin.parquet')
exclude_cols = [
    'date', 'day_index', 'mode', 'site_id', 'split', 'basin', 'parsed_date',
    'event_active', 'is_active_event', 'event_present',
    'event_within_1d', 'event_within_2d', 'event_within_3d',
    'lead_0', 'lead_1', 'lead_2', 'lead_3', 'lead_days',
    'label_status', 'event_type', 'event_id', 'event_name', 'severity',
    'secondary_event_id', 'label_source', 'label_confidence'
]
all_feats = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ('float64', 'float32')]

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

y_tr = train_df['event_present'].values.astype(int)
y_val = val_df['event_present'].values.astype(int)

# Rank features using Train RF
rf_ranker = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42)
rf_ranker.fit(train_df[all_feats].values, y_tr)
ranked_feats = [f for f, imp in sorted(zip(all_feats, rf_ranker.feature_importances_), key=lambda x: x[1], reverse=True)]

top_50 = ranked_feats[:50]
top_25 = ranked_feats[:25]
non_base_feats = [c for c in all_feats if '_base_mean' not in c and '_base_std' not in c]

models = {
    'RF (All 101 feats)': (RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42), all_feats, False),
    'RF (Top 50 feats)': (RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42), top_50, False),
    'RF (Top 25 feats)': (RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42), top_25, False),
    'RF (Non-Base 87 feats)': (RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=42), non_base_feats, False),
    'LR (All 101 feats)': (LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000), all_feats, True),
    'LR (Top 25 feats)': (LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000), top_25, True),
    'HGB (All 101 feats)': (HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=5, l2_regularization=1.0, class_weight='balanced', random_state=42), all_feats, False),
    'HGB (Top 25 feats)': (HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=5, l2_regularization=1.0, class_weight='balanced', random_state=42), top_25, False),
}

print('=== MODEL ROBUSTNESS COMPARISON ON VALIDATION SET (N=88, Pos=13) ===\n')
header = f"{'Model Configuration':25s} | {'Feats':5s} | {'PR-AUC':7s} | {'ROC-AUC':7s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'FA':3s} | {'ARB01':5s} | {'DANA':5s}"
print(header)
print('-' * len(header))

for name, (clf, feat_subset, is_scaled) in models.items():
    X_tr_s = train_df[feat_subset].values
    X_val_s = val_df[feat_subset].values
    
    if is_scaled:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_s)
        X_val_s = scaler.transform(X_val_s)
        
    clf.fit(X_tr_s, y_tr)
    probs = clf.predict_proba(X_val_s)[:, 1]
    
    pr_auc = average_precision_score(y_val, probs)
    roc_auc = roc_auc_score(y_val, probs)
    
    preds_27 = (probs >= 0.27).astype(int)
    p = precision_score(y_val, preds_27, zero_division=0)
    r = recall_score(y_val, preds_27, zero_division=0)
    f1 = f1_score(y_val, preds_27, zero_division=0)
    cm = confusion_matrix(y_val, preds_27)
    tn, fp, fn, tp = cm.ravel()
    
    val_c = val_df.copy()
    val_c['pred'] = preds_27
    arb_det = (val_c[val_c['event_id'] == 'IMD-2024-D-ARB01']['pred'] == 1).any()
    dana_det = (val_c[val_c['event_id'] == 'IMD-2024-SCS-DANA']['pred'] == 1).any()
    
    arb_str = 'Det' if arb_det else 'Miss'
    dana_str = 'Det' if dana_det else 'Miss'
    
    print(f"{name:25s} | {len(feat_subset):5d} | {pr_auc:7.4f} | {roc_auc:7.4f} | {p:6.4f} | {r:6.4f} | {f1:6.4f} | {fp:3d} | {arb_str:5s} | {dana_str:5s}")
