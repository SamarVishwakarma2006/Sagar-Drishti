import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import (
    brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression

def compute_ece(y_true, y_prob, n_bins=5):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    bin_stats = []
    n = len(y_true)
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            obs_freq = float(np.mean(y_true[mask]))
            pred_conf = float(np.mean(y_prob[mask]))
            abs_err = abs(obs_freq - pred_conf)
            ece += (count / n) * abs_err
            bin_stats.append({
                "bin": i,
                "range": [round(float(bin_edges[i]), 2), round(float(bin_edges[i+1]), 2)],
                "count": count,
                "obs_freq": round(obs_freq, 4),
                "pred_conf": round(pred_conf, 4),
                "diff": round(abs_err, 4)
            })
    return round(float(ece), 4), bin_stats

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models", "v2_10yr")
    calib_dir = os.path.join(models_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    
    data_path = os.path.join(base_dir, "data", "historical", "ml_features_10yr_candidate.parquet")
    df = pd.read_parquet(data_path)
    val_df = df[df["split"] == "val"].copy()
    
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "r") as f:
        meta_v2 = json.load(f)
    feature_cols = meta_v2.get("feature_list", meta_v2.get("feature_columns"))
    
    horizons = ["0d", "1d", "2d", "3d"]
    models = {h: joblib.load(os.path.join(models_dir, f"risk_model_{h}.joblib"))["model"] for h in horizons}
    
    calib_metadata = {
        "metadata_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "2.0.0-10yr-candidate",
        "calibration_method": "IsotonicRegression",
        "training_split": "val",
        "training_date_range": ["2024-01-01", "2024-12-31"],
        "horizons": {},
        "basin_thresholds": {},
        "risk_tiers": {
            "LOW": {"min": 0.0, "max": 0.30, "label": "Low Risk / Normal Ocean State"},
            "MODERATE": {"min": 0.30, "max": 0.60, "label": "Moderate Risk / Advisory"},
            "HIGH": {"min": 0.60, "max": 1.00, "label": "High Risk / Warning Alert"}
        }
    }
    
    print("================================================================================")
    print("FITTING ISOTONIC CALIBRATORS & DETERMINING BASIN THRESHOLDS")
    print("================================================================================")
    
    # 1. Fit Calibrators per horizon
    for h in horizons:
        tgt = f"event_within_{h}"
        y_val = val_df[tgt].values.astype(int)
        X_val = val_df[feature_cols].values
        raw_p = models[h].predict_proba(X_val)[:, 1]
        
        # Fit Isotonic Regression strictly on validation data
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_p, y_val)
        cal_p = np.clip(iso.predict(raw_p), 0.0, 1.0)
        
        # Save calibrator artifact
        art_path = os.path.join(calib_dir, f"calibrator_{h}.joblib")
        joblib.dump({"calibrator": iso, "horizon": h, "method": "IsotonicRegression"}, art_path)
        
        # Before / After metrics
        raw_brier = round(float(brier_score_loss(y_val, raw_p)), 4)
        raw_ece, raw_bins = compute_ece(y_val, raw_p, n_bins=5)
        
        cal_brier = round(float(brier_score_loss(y_val, cal_p)), 4)
        cal_ece, cal_bins = compute_ece(y_val, cal_p, n_bins=5)
        
        calib_metadata["horizons"][h] = {
            "artifact_file": f"calibrator_{h}.joblib",
            "samples": len(y_val),
            "actual_positives": int(y_val.sum()),
            "before_calibration": {
                "brier_score": raw_brier,
                "ece": raw_ece,
                "reliability_bins": raw_bins
            },
            "after_calibration": {
                "brier_score": cal_brier,
                "ece": cal_ece,
                "reliability_bins": cal_bins
            }
        }
        print(f"Horizon {h.upper()}:")
        print(f"  Brier: {raw_brier:.4f} -> {cal_brier:.4f} (Reduction: {((raw_brier-cal_brier)/raw_brier):.1%})")
        print(f"  ECE:   {raw_ece:.4f} -> {cal_ece:.4f}")
        
    # 2. Determine Basin-Specific Thresholds
    # In Bay of Bengal: scores are higher. Operational threshold theta_bob = 0.44 gives >93% recall.
    # In Arabian Sea: scores are lower. Operational threshold theta_aras = 0.15 restores recall to 62-75%.
    basin_configs = {
        "0d": {"bob": 0.44, "aras": 0.15},
        "1d": {"bob": 0.44, "aras": 0.15},
        "2d": {"bob": 0.44, "aras": 0.15},
        "3d": {"bob": 0.44, "aras": 0.15},
    }
    
    calib_metadata["basin_thresholds"] = basin_configs
    
    # Calculate detailed stats for each basin/horizon
    print("\n=== BASIN-SPECIFIC OPERATIONAL PERFORMANCE ===")
    basin_metrics = {}
    for h in horizons:
        tgt = f"event_within_{h}"
        basin_metrics[h] = {}
        for b_name, b_key in [("Bay of Bengal", "bob"), ("Arabian Sea", "aras")]:
            b_df = val_df[val_df["basin"] == b_name].copy()
            y_b = b_df[tgt].values.astype(int)
            X_b = b_df[feature_cols].values
            raw_p = models[h].predict_proba(X_b)[:, 1]
            
            th = basin_configs[h][b_key]
            preds = (raw_p >= th).astype(int)
            cm = confusion_matrix(y_b, preds)
            tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)
            
            prec = round(float(precision_score(y_b, preds, zero_division=0)), 4)
            rec = round(float(recall_score(y_b, preds, zero_division=0)), 4)
            f1 = round(float(f1_score(y_b, preds, zero_division=0)), 4)
            fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
            
            basin_metrics[h][b_key] = {
                "basin_name": b_name,
                "threshold": th,
                "samples": len(y_b),
                "actual_positives": int(y_b.sum()),
                "predicted_positives": int(preds.sum()),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
            print(f"Horizon {h.upper()} | {b_name:14} (th={th:.2f}) | Rec={rec:.1%}, Prec={prec:.1%}, F1={f1:.4f}, FPR={fpr:.1%}, TP={tp}/{y_b.sum()}, FP={fp}")

    calib_metadata["basin_evaluation"] = basin_metrics
    
    with open(os.path.join(calib_dir, "calibration_metadata.json"), "w") as f:
        json.dump(calib_metadata, f, indent=2)
        
    print(f"\n[+] Calibration artifacts and metadata saved to {calib_dir}")

if __name__ == "__main__":
    main()
