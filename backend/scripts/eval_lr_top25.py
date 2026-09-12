import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

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
sorted_feats = [f for f, imp in sorted(zip(all_feats, rf_ranker.feature_importances_), key=lambda x: x[1], reverse=True)]
top_25 = sorted_feats[:25]

print('=== TOP 25 FEATURES SELECTED STRICTLY ON TRAIN ===')
for i, f in enumerate(top_25, 1):
    print(f'{i:2d}. {f}')

# Train Logistic Regression on Top 25
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(train_df[top_25].values)
X_val_sc = scaler.transform(val_df[top_25].values)

lr = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_tr_sc, y_tr)
val_probs = lr.predict_proba(X_val_sc)[:, 1]

print('\n=== VALIDATION PR-AUC & ROC-AUC ===')
print(f'PR-AUC:  {average_precision_score(y_val, val_probs):.4f}')
print(f'ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}')

print('\n=== VALIDATION THRESHOLD SWEEP (0.10 to 0.50) ===')
header = f"{'Thresh':8s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'Spec':6s} | {'FP':3s} | {'FN':3s} | {'ARB01':6s} | {'DANA':6s}"
print(header)
print('-' * len(header))
for th in [0.10, 0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.35, 0.40, 0.45, 0.50]:
    preds = (val_probs >= th).astype(int)
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    
    val_c = val_df.copy()
    val_c['pred'] = preds
    arb_det = (val_c[val_c['event_id'] == 'IMD-2024-D-ARB01']['pred'] == 1).any()
    dana_det = (val_c[val_c['event_id'] == 'IMD-2024-SCS-DANA']['pred'] == 1).any()
    arb_s = 'Det' if arb_det else 'Miss'
    dana_s = 'Det' if dana_det else 'Miss'
    print(f"{th:8.2f} | {p:6.4f} | {r:6.4f} | {f1:6.4f} | {spec:6.4f} | {fp:3d} | {fn:3d} | {arb_s:6s} | {dana_s:6s}")
