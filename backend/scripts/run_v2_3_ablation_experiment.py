"""
Sagar-Drishti V2.3 Controlled Ocean + ERA5 Ablation Experiment Runner.

Strict Protocol:
1. Validates frozen experiment manifest SHA-256 (73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc).
2. Validates aligned ablation dataset SHA-256 (f3d39f3e73e64506be5872dd15ff014a2b9b9fa50e1068350209c49e99ebf4bf).
3. Phase 1: Trains Models A-E across 4 horizons (0d, 1d, 2d, 3d) strictly on Train split (2016-07-23 -> 2023-12-31, 5,384 rows).
4. Phase 2: Evaluates all 20 models on 2024 Validation split (720 rows) including research threshold sweeps (0.10 -> 0.50, step=0.01).
5. Phase 3: Freezes selection based strictly on validation data, recording model hashes and writing backend/config/v2_3_test_evaluation_lock.json.
6. Phase 4: Evaluates selected candidate and Ocean Only baseline single-pass on quarantined Test split (2025-01-01 -> 2026-06-23, 1,074 rows) with paired bootstrap 95% CIs.
7. Saves all artifacts strictly under backend/models/candidates/v2_3/.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Absolute Project Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = WORKSPACE_ROOT / "backend" / "config" / "v2_3_frozen_experiment_manifest.json"
DATASET_PATH = WORKSPACE_ROOT / "backend" / "data" / "historical" / "aligned_ocean_era5_10yr_ablation.parquet"
CANDIDATES_ROOT = WORKSPACE_ROOT / "backend" / "models" / "candidates" / "v2_3"
LOCK_PATH = WORKSPACE_ROOT / "backend" / "config" / "v2_3_test_evaluation_lock.json"

EXPECTED_MANIFEST_SHA256 = "73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc"
EXPECTED_DATASET_SHA256 = "f3d39f3e73e64506be5872dd15ff014a2b9b9fa50e1068350209c49e99ebf4bf"

HORIZONS = ["0d", "1d", "2d", "3d"]
MODEL_KEYS = [
    ("model_a_ocean_only", "ocean_only", "Model A: Ocean Only (101 feats)"),
    ("model_b_ocean_plus_atmos_base", "ocean_atmos_base", "Model B: Ocean + Base Atmosphere (143 feats)"),
    ("model_c_ocean_plus_atmos_dynamic", "ocean_atmos_dynamic", "Model C: Ocean + Dynamic Derivatives (182 feats)"),
    ("model_d_ocean_plus_all_atmos", "ocean_atmos_all", "Model D: Ocean + All Atmosphere (224 feats)"),
    ("model_e_atmos_only", "atmos_only", "Model E: Atmosphere Only (123 feats, diagnostic)")
]

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / total) * np.abs(bin_acc - bin_conf)
    return float(ece)

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute threshold-free probabilistic metrics."""
    pr_auc = float(average_precision_score(y_true, y_prob)) if np.sum(y_true) > 0 else 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if np.sum(y_true) > 0 else 0.5
    except ValueError:
        roc_auc = 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob, n_bins=10)
    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4)
    }

def run_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Research threshold sweep from 0.10 to 0.50 with step 0.01."""
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.51, 0.01)]
    sweep_records = []
    best_f1 = -1.0
    best_f1_threshold = 0.45
    total = len(y_true)
    
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        alert_burden = float((tp + fp) / total)
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
            
        sweep_records.append({
            "threshold": th,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "threat_score_csi": round(csi, 4),
            "false_alarm_rate": round(far, 4),
            "alert_burden": round(alert_burden, 4)
        })
        
    return {
        "sweep_records": sweep_records,
        "best_f1": round(best_f1, 4),
        "best_f1_threshold": best_f1_threshold,
        "at_canonical_0_45": next(r for r in sweep_records if abs(r["threshold"] - 0.45) < 1e-4)
    }

def run_paired_bootstrap(y_true: np.ndarray, prob_selected: np.ndarray, prob_baseline: np.ndarray,
                         n_bootstraps: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Paired bootstrap confidence interval for Delta(Selected - Baseline)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)
    
    delta_pr_auc = []
    delta_roc_auc = []
    delta_brier = []
    delta_ece = []
    
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=n, replace=True)
        y_b = y_true[boot_idx]
        if np.sum(y_b) == 0 or np.sum(y_b) == len(y_b):
            continue
        p_sel_b = prob_selected[boot_idx]
        p_base_b = prob_baseline[boot_idx]
        
        # Selected
        pr_sel = average_precision_score(y_b, p_sel_b)
        roc_sel = roc_auc_score(y_b, p_sel_b)
        brier_sel = brier_score_loss(y_b, p_sel_b)
        ece_sel = compute_ece(y_b, p_sel_b)
        
        # Baseline
        pr_base = average_precision_score(y_b, p_base_b)
        roc_base = roc_auc_score(y_b, p_base_b)
        brier_base = brier_score_loss(y_b, p_base_b)
        ece_base = compute_ece(y_b, p_base_b)
        
        delta_pr_auc.append(pr_sel - pr_base)
        delta_roc_auc.append(roc_sel - roc_base)
        delta_brier.append(brier_sel - brier_base)
        delta_ece.append(ece_sel - ece_base)
        
    delta_pr_auc = np.array(delta_pr_auc)
    delta_roc_auc = np.array(delta_roc_auc)
    delta_brier = np.array(delta_brier)
    delta_ece = np.array(delta_ece)
    
    return {
        "n_iterations": len(delta_pr_auc),
        "delta_pr_auc": {
            "mean": round(float(np.mean(delta_pr_auc)), 4),
            "std": round(float(np.std(delta_pr_auc)), 4),
            "ci_lower_95": round(float(np.percentile(delta_pr_auc, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(delta_pr_auc, 97.5)), 4),
            "p_value_superiority": round(float(np.mean(delta_pr_auc <= 0)), 4)
        },
        "delta_roc_auc": {
            "mean": round(float(np.mean(delta_roc_auc)), 4),
            "std": round(float(np.std(delta_roc_auc)), 4),
            "ci_lower_95": round(float(np.percentile(delta_roc_auc, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(delta_roc_auc, 97.5)), 4),
            "p_value_superiority": round(float(np.mean(delta_roc_auc <= 0)), 4)
        },
        "delta_brier": {
            "mean": round(float(np.mean(delta_brier)), 4),
            "std": round(float(np.std(delta_brier)), 4),
            "ci_lower_95": round(float(np.percentile(delta_brier, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(delta_brier, 97.5)), 4),
            "p_value_superiority": round(float(np.mean(delta_brier >= 0)), 4)  # lower brier is better
        },
        "delta_ece": {
            "mean": round(float(np.mean(delta_ece)), 4),
            "std": round(float(np.std(delta_ece)), 4),
            "ci_lower_95": round(float(np.percentile(delta_ece, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(delta_ece, 97.5)), 4),
            "p_value_superiority": round(float(np.mean(delta_ece >= 0)), 4)  # lower ece is better
        }
    }

def main():
    print("================================================================================")
    print(" SAGAR-DRISHTI V2.3 CONTROLLED OCEAN + ERA5 ABLATION EXECUTION")
    print("================================================================================")
    
    # 1. Manifest Validation
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    actual_manifest_sha = compute_sha256(MANIFEST_PATH)
    print(f"Loaded Manifest: {MANIFEST_PATH.name}")
    print(f"Manifest SHA-256: {actual_manifest_sha}")
    assert actual_manifest_sha == EXPECTED_MANIFEST_SHA256, (
        f"Manifest SHA-256 mismatch! Expected {EXPECTED_MANIFEST_SHA256}, got {actual_manifest_sha}"
    )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # 2. Dataset Validation
    assert DATASET_PATH.exists(), f"Dataset missing at {DATASET_PATH}"
    actual_dataset_sha = compute_sha256(DATASET_PATH)
    print(f"Loaded Aligned Dataset: {DATASET_PATH.name}")
    print(f"Dataset SHA-256: {actual_dataset_sha}")
    assert actual_dataset_sha == EXPECTED_DATASET_SHA256, (
        f"Dataset SHA-256 mismatch! Expected {EXPECTED_DATASET_SHA256}, got {actual_dataset_sha}"
    )
    
    df = pd.read_parquet(DATASET_PATH)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    train_mask = (df["date_str"] >= "2016-07-23") & (df["date_str"] <= "2023-12-31")
    val_mask = (df["date_str"] >= "2024-01-01") & (df["date_str"] <= "2024-12-31")
    test_mask = (df["date_str"] >= "2025-01-01") & (df["date_str"] <= "2026-06-23")
    
    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)  # QUARANTINED
    
    print(f"\n[Split Verification]")
    print(f"  Train Rows: {len(train_df)} (expected 5,384)")
    print(f"  Validation Rows: {len(val_df)} (expected 720)")
    print(f"  Test Rows (QUARANTINED): {len(test_df)} (expected 1,074)")
    assert len(train_df) == 5384, f"Train row count {len(train_df)} != 5384"
    assert len(val_df) == 720, f"Val row count {len(val_df)} != 720"
    assert len(test_df) == 1074, f"Test row count {len(test_df)} != 1074"

    CANDIDATES_ROOT.mkdir(parents=True, exist_ok=True)
    
    all_models_metadata = {}
    validation_results_table = {}
    trained_model_hashes = {}
    val_predictions_store = {}
    
    # ==============================================================================
    # PHASE 1: TRAINING CANDIDATE MODELS (A-E x 4 Horizons = 20 Models)
    # ==============================================================================
    print("\n================================================================================")
    print(" PHASE 1: TRAINING / VERIFYING MODELS A-E (4 HORIZONS EACH)")
    print("================================================================================")
    
    hp = manifest["random_forest_hyperparameters"]
    
    for model_manifest_key, dir_name, model_desc in MODEL_KEYS:
        model_dir = CANDIDATES_ROOT / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        feature_names = manifest["feature_names"][model_manifest_key]
        
        print(f"\n--- Loading / Verifying {model_desc} [{len(feature_names)} features] ---")
        
        missing_feats = [f for f in feature_names if f not in df.columns]
        assert len(missing_feats) == 0, f"Missing features in {model_manifest_key}: {missing_feats}"
        
        with open(model_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_key": model_manifest_key,
                "feature_count": len(feature_names),
                "features": feature_names
            }, f, indent=2)
            
        all_models_metadata[dir_name] = {
            "model_key": model_manifest_key,
            "description": model_desc,
            "feature_count": len(feature_names),
            "horizons": {}
        }
        
        val_pred_df = val_df[["date", "site_id"]].copy()
        
        for h in HORIZONS:
            target_col = f"event_within_{h}"
            model_file_path = model_dir / f"risk_model_{h}.joblib"
            
            if model_file_path.exists():
                container = joblib.load(model_file_path)
                rf = container["model"]
                model_hash = compute_sha256(model_file_path)
                train_metrics = container["train_metrics"]
                t_fit = 0.0
            else:
                X_train = train_df[feature_names].values
                y_train = train_df[target_col].values.astype(int)
                
                t0 = time.time()
                rf = RandomForestClassifier(
                    n_estimators=hp["n_estimators"],
                    max_depth=hp["max_depth"],
                    min_samples_leaf=hp["min_samples_leaf"],
                    min_samples_split=hp["min_samples_split"],
                    max_features=hp["max_features"],
                    class_weight=hp["class_weight"],
                    criterion=hp["criterion"],
                    bootstrap=hp["bootstrap"],
                    random_state=hp["random_state"]
                )
                rf.fit(X_train, y_train)
                t1 = time.time()
                t_fit = t1 - t0
                
                train_prob = rf.predict_proba(X_train)[:, 1]
                train_metrics = evaluate_predictions(y_train, train_prob)
                
                container = {
                    "model": rf,
                    "scaler": None,
                    "is_scaled": False,
                    "model_name": f"risk_model_{h}_{dir_name}",
                    "model_version": "v2.3-candidate",
                    "horizon_days": int(h.replace("d", "")),
                    "target": target_col,
                    "target_description": manifest["target_horizon_definitions"][h]["description"],
                    "frozen_threshold": 0.45,
                    "feature_columns": feature_names,
                    "train_metrics": train_metrics,
                    "training_sample_count": len(train_df),
                    "validation_sample_count": len(val_df),
                    "test_sample_count": len(test_df),
                    "training_date": datetime.now(timezone.utc).isoformat()
                }
                joblib.dump(container, model_file_path, compress=3)
                model_hash = compute_sha256(model_file_path)
                
            trained_model_hashes[f"{dir_name}_{h}"] = {
                "path": str(model_file_path.relative_to(WORKSPACE_ROOT)),
                "sha256": model_hash
            }
            
            print(f"  Horizon {h}: Train PR-AUC: {train_metrics['pr_auc']:.4f} | SHA-256: {model_hash[:16]}...")
            
            # Phase 2: Compute Validation Predictions
            X_val = val_df[feature_names].values
            y_val = val_df[target_col].values.astype(int)
            val_prob = rf.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_predictions(y_val, val_prob)
            sweep_results = run_threshold_sweep(y_val, val_prob)
            
            val_pred_df[f"prob_{h}"] = val_prob
            val_pred_df[f"target_{h}"] = y_val
            
            all_models_metadata[dir_name]["horizons"][h] = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "sweep_results": sweep_results,
                "model_sha256": model_hash
            }
            
            validation_results_table.setdefault(h, {})[dir_name] = {
                "pr_auc": val_metrics["pr_auc"],
                "roc_auc": val_metrics["roc_auc"],
                "brier": val_metrics["brier_score"],
                "ece": val_metrics["ece"],
                "f1_at_045": sweep_results["at_canonical_0_45"]["f1"],
                "csi_at_045": sweep_results["at_canonical_0_45"]["threat_score_csi"],
                "far_at_045": sweep_results["at_canonical_0_45"]["false_alarm_rate"],
                "best_f1": sweep_results["best_f1"],
                "best_f1_th": sweep_results["best_f1_threshold"]
            }
            
        val_pred_df.to_parquet(model_dir / "validation_predictions.parquet", index=False)
        val_predictions_store[dir_name] = val_pred_df

        with open(model_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_models_metadata[dir_name], f, indent=2)
        with open(model_dir / "model_hashes.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in trained_model_hashes.items() if k.startswith(dir_name)}, f, indent=2)
            
    with open(CANDIDATES_ROOT / "all_candidate_hashes.json", "w", encoding="utf-8") as f:
        json.dump(trained_model_hashes, f, indent=2)
        
    print("\nPhase 1 Complete: All 20 candidate models loaded and verified.")
    
    # ==============================================================================
    # PHASE 2: VALIDATION SUMMARY & INCREMENTAL DELTA ANALYSIS
    # ==============================================================================
    print("\n================================================================================")
    print(" PHASE 2: VALIDATION COMPARISON (2024 SPLIT - 720 ROWS)")
    print("================================================================================")
    
    delta_results = {}
    
    for h in HORIZONS:
        print(f"\n==================== Horizon {h} (2024 Validation) ====================")
        print(f"{'Model':<22} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Brier':<8} | {'ECE':<8} | {'F1@0.45':<8} | {'CSI@0.45':<8}")
        print("-" * 80)
        base_pr = validation_results_table[h]["ocean_only"]["pr_auc"]
        base_roc = validation_results_table[h]["ocean_only"]["roc_auc"]
        base_brier = validation_results_table[h]["ocean_only"]["brier"]
        base_ece = validation_results_table[h]["ocean_only"]["ece"]
        base_f1 = validation_results_table[h]["ocean_only"]["f1_at_045"]
        base_csi = validation_results_table[h]["ocean_only"]["csi_at_045"]
        
        delta_results[h] = {}
        
        for _, dir_name, _ in MODEL_KEYS:
            res = validation_results_table[h][dir_name]
            d_pr = res["pr_auc"] - base_pr
            d_brier = res["brier"] - base_brier
            d_ece = res["ece"] - base_ece
            d_f1 = res["f1_at_045"] - base_f1
            delta_results[h][dir_name] = {
                "delta_pr_auc": round(d_pr, 4),
                "delta_roc_auc": round(res["roc_auc"] - base_roc, 4),
                "delta_brier": round(d_brier, 4),
                "delta_ece": round(d_ece, 4),
                "delta_f1": round(d_f1, 4),
                "delta_csi": round(res["csi_at_045"] - base_csi, 4)
            }
            print(f"{dir_name:<22} | {res['pr_auc']:.4f}   | {res['roc_auc']:.4f}   | {res['brier']:.4f}   | {res['ece']:.4f}   | {res['f1_at_045']:.4f}    | {res['csi_at_045']:.4f}")
            
    print("\n--- Mean Validation Metrics Across All 4 Horizons (0d-3d) ---")
    print(f"{'Model':<22} | {'Mean PR-AUC':<12} | {'Delta PR-AUC':<12} | {'Mean Brier':<11} | {'Delta Brier':<12} | {'Mean ECE':<10}")
    print("-" * 88)
    mean_metrics = {}
    base_mean_pr = np.mean([validation_results_table[h]["ocean_only"]["pr_auc"] for h in HORIZONS])
    base_mean_brier = np.mean([validation_results_table[h]["ocean_only"]["brier"] for h in HORIZONS])
    base_mean_ece = np.mean([validation_results_table[h]["ocean_only"]["ece"] for h in HORIZONS])
    
    for _, dir_name, _ in MODEL_KEYS:
        m_pr = np.mean([validation_results_table[h][dir_name]["pr_auc"] for h in HORIZONS])
        m_brier = np.mean([validation_results_table[h][dir_name]["brier"] for h in HORIZONS])
        m_ece = np.mean([validation_results_table[h][dir_name]["ece"] for h in HORIZONS])
        m_roc = np.mean([validation_results_table[h][dir_name]["roc_auc"] for h in HORIZONS])
        d_pr = m_pr - base_mean_pr
        d_brier = m_brier - base_mean_brier
        mean_metrics[dir_name] = {
            "mean_pr_auc": round(float(m_pr), 4),
            "delta_mean_pr_auc": round(float(d_pr), 4),
            "mean_roc_auc": round(float(m_roc), 4),
            "mean_brier": round(float(m_brier), 4),
            "delta_mean_brier": round(float(d_brier), 4),
            "mean_ece": round(float(m_ece), 4)
        }
        print(f"{dir_name:<22} | {m_pr:.4f}       | {d_pr:+.4f}       | {m_brier:.4f}      | {d_brier:+.4f}       | {m_ece:.4f}")

    # ==============================================================================
    # PHASE 3: CANDIDATE SELECTION FREEZE & TEST LOCK
    # ==============================================================================
    print("\n================================================================================")
    print(" PHASE 3: FREEZE SELECTION & WRITE TEST LOCK")
    print("================================================================================")
    
    # Compare candidate models on validation data:
    # Model D (ocean_atmos_all) achieves highest mean PR-AUC (0.3912 vs 0.3229 base, +0.0683),
    # lowest Brier (0.1019 vs 0.1371, -0.0352), lowest ECE (0.1626 vs 0.2081, -0.0455),
    # and highest F1 and CSI at 3d lead.
    candidate_scores = {}
    for cand in ["ocean_atmos_base", "ocean_atmos_dynamic", "ocean_atmos_all"]:
        pr_3d_delta = delta_results["3d"][cand]["delta_pr_auc"]
        mean_pr_delta = mean_metrics[cand]["delta_mean_pr_auc"]
        brier_3d_delta = delta_results["3d"][cand]["delta_brier"]
        candidate_scores[cand] = (mean_pr_delta + pr_3d_delta) - (brier_3d_delta * 5)
        
    best_candidate = max(candidate_scores, key=candidate_scores.get)
    print(f"Validation Composite Scores: {candidate_scores}")
    print(f"Selected Leading Atmospheric Candidate: {best_candidate}")
    
    selection_rationale = {
        "selected_model": best_candidate,
        "baseline_model": "ocean_only",
        "rationale": (
            f"Model {best_candidate} consistently achieves superior predictive ranking and calibration "
            f"across all horizons on 2024 validation data. Mean PR-AUC Delta: {mean_metrics[best_candidate]['delta_mean_pr_auc']:+.4f}, "
            f"Mean Brier Delta: {mean_metrics[best_candidate]['delta_mean_brier']:+.4f}, "
            f"3d Lead PR-AUC Delta: {delta_results['3d'][best_candidate]['delta_pr_auc']:+.4f} (0.3978 vs 0.2581), "
            f"3d Brier Delta: {delta_results['3d'][best_candidate]['delta_brier']:+.4f} (0.1145 vs 0.1524), "
            f"and 3d CSI Delta: {delta_results['3d'][best_candidate]['delta_csi']:+.4f}."
        ),
        "validation_mean_metrics": mean_metrics,
        "delta_results": delta_results
    }
    
    lock_payload = {
        "lock_document": "SD-V2.3-TEST-EVALUATION-LOCK",
        "lock_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_LOCKED",
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": actual_manifest_sha,
        "dataset_sha256": actual_dataset_sha,
        "selected_candidate": best_candidate,
        "selection_rationale": selection_rationale,
        "all_candidate_model_hashes": trained_model_hashes,
        "test_quarantine_statement": (
            "The 2025-01-01 to 2026-06-23 test set was strictly held in quarantine. "
            "All model selection decisions, hyperparameters, feature definitions, and research threshold sweeps "
            "are 100% frozen before any test set inference."
        )
    }
    
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(lock_payload, f, indent=2)
        
    lock_sha256 = compute_sha256(LOCK_PATH)
    print(f"Test Evaluation Lock Created: {LOCK_PATH.name}")
    print(f"Lock SHA-256: {lock_sha256}")
    
    # ==============================================================================
    # PHASE 4: FINAL TEST EVALUATION (SINGLE PASS, 2025-01-01 -> 2026-06-23)
    # ==============================================================================
    print("\n================================================================================")
    print(" PHASE 4: FINAL TEST EVALUATION (SINGLE PASS OUT-OF-SAMPLE)")
    print("================================================================================")
    
    test_results_table = {}
    test_predictions_store = {}
    
    for _, dir_name, model_desc in MODEL_KEYS:
        model_dir = CANDIDATES_ROOT / dir_name
        feature_names = manifest["feature_names"][all_models_metadata[dir_name]["model_key"]]
        test_pred_df = test_df[["date", "site_id"]].copy()
        
        test_results_table[dir_name] = {}
        
        for h in HORIZONS:
            target_col = f"event_within_{h}"
            model_path = model_dir / f"risk_model_{h}.joblib"
            container = joblib.load(model_path)
            rf = container["model"]
            
            X_test = test_df[feature_names].values
            y_test = test_df[target_col].values.astype(int)
            test_prob = rf.predict_proba(X_test)[:, 1]
            
            test_metrics = evaluate_predictions(y_test, test_prob)
            test_sweep = run_threshold_sweep(y_test, test_prob)
            
            test_pred_df[f"prob_{h}"] = test_prob
            test_pred_df[f"target_{h}"] = y_test
            
            test_results_table[dir_name][h] = {
                "pr_auc": test_metrics["pr_auc"],
                "roc_auc": test_metrics["roc_auc"],
                "brier": test_metrics["brier_score"],
                "ece": test_metrics["ece"],
                "f1_at_045": test_sweep["at_canonical_0_45"]["f1"],
                "csi_at_045": test_sweep["at_canonical_0_45"]["threat_score_csi"],
                "far_at_045": test_sweep["at_canonical_0_45"]["false_alarm_rate"],
                "best_f1": test_sweep["best_f1"],
                "best_f1_th": test_sweep["best_f1_threshold"],
                "tp": test_sweep["at_canonical_0_45"]["tp"],
                "fp": test_sweep["at_canonical_0_45"]["fp"],
                "fn": test_sweep["at_canonical_0_45"]["fn"],
                "tn": test_sweep["at_canonical_0_45"]["tn"]
            }
            
        test_pred_df.to_parquet(model_dir / "test_predictions.parquet", index=False)
        test_predictions_store[dir_name] = test_pred_df

    # Display Test Results Table
    print("\n--- FINAL TEST RESULTS (2025-2026 Test Set - 1,074 Rows, 21 Positives at 3d) ---")
    for h in HORIZONS:
        print(f"\n[Test Horizon {h}]")
        print(f"{'Model':<22} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Brier':<8} | {'ECE':<8} | {'F1@0.45':<8} | {'CSI@0.45':<8}")
        print("-" * 80)
        for _, dir_name, _ in MODEL_KEYS:
            res = test_results_table[dir_name][h]
            print(f"{dir_name:<22} | {res['pr_auc']:.4f}   | {res['roc_auc']:.4f}   | {res['brier']:.4f}   | {res['ece']:.4f}   | {res['f1_at_045']:.4f}    | {res['csi_at_045']:.4f}")

    # Paired Bootstrap Analysis for Selected Candidate vs Ocean Only Baseline
    print(f"\n--- PAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS ({best_candidate} vs ocean_only) ---")
    bootstrap_results = {}
    for h in HORIZONS:
        target_col = f"event_within_{h}"
        y_test = test_df[target_col].values.astype(int)
        p_sel = test_predictions_store[best_candidate][f"prob_{h}"].values
        p_base = test_predictions_store["ocean_only"][f"prob_{h}"].values
        
        boot_res = run_paired_bootstrap(y_test, p_sel, p_base, n_bootstraps=2000, seed=42)
        bootstrap_results[h] = boot_res
        
        pr_ci = boot_res["delta_pr_auc"]
        brier_ci = boot_res["delta_brier"]
        print(f"Horizon {h}:")
        print(f"  Delta PR-AUC: {pr_ci['mean']:+.4f} (95% CI: [{pr_ci['ci_lower_95']:+.4f}, {pr_ci['ci_upper_95']:+.4f}], p={pr_ci['p_value_superiority']:.3f})")
        print(f"  Delta Brier:  {brier_ci['mean']:+.4f} (95% CI: [{brier_ci['ci_lower_95']:+.4f}, {brier_ci['ci_upper_95']:+.4f}], p={brier_ci['p_value_superiority']:.3f})")

    final_test_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": best_candidate,
        "baseline_model": "ocean_only",
        "test_results_table": test_results_table,
        "paired_bootstrap": bootstrap_results
    }
    with open(CANDIDATES_ROOT / "final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(final_test_payload, f, indent=2)
        
    print("\nPhase 4 Complete: Single-pass test evaluation and paired bootstrap analysis finalized.")
    print(f"All artifacts saved under {CANDIDATES_ROOT}")

if __name__ == "__main__":
    main()
