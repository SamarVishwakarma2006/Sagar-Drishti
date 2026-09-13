"""
Unit and cryptographic regression tests for V2.3 Controlled Ocean + ERA5 Ablation Experiment.
"""

import json
import hashlib
from pathlib import Path
import pytest
import joblib

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = WORKSPACE_ROOT / "backend" / "config" / "v2_3_frozen_experiment_manifest.json"
DATASET_PATH = WORKSPACE_ROOT / "backend" / "data" / "historical" / "aligned_ocean_era5_10yr_ablation.parquet"
LOCK_PATH = WORKSPACE_ROOT / "backend" / "config" / "v2_3_test_evaluation_lock.json"
CANDIDATES_ROOT = WORKSPACE_ROOT / "backend" / "models" / "candidates" / "v2_3"

EXPECTED_INVARIANTS = {
    "backend/models/risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
    "backend/models/v2_10yr/risk_model_3d.joblib": "7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3",
    "backend/models/v2_10yr/model_metadata.json": "2fdaf9e57c6040a3a97142e2ea382e494352120f0ad9be0a7cf27fe6e391026f",
    "backend/config/frozen_alert_policy_v2.json": "6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6",
    "backend/data/historical/features_10yr.parquet": "cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867",
    "backend/data/historical/labeled_features_10yr_clean.parquet": "25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a",
    "backend/data/era5/features_atmosphere_10yr_daily.parquet": "551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864",
    "backend/config/v2_3_frozen_experiment_manifest.json": "73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc",
    "backend/data/historical/aligned_ocean_era5_10yr_ablation.parquet": "f3d39f3e73e64506be5872dd15ff014a2b9b9fa50e1068350209c49e99ebf4bf"
}

def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_01_baseline_invariants_unmodified():
    """Verify that all production, candidate v2_10yr, and validated datasets remain byte-for-byte unchanged."""
    for rel_path, expected_sha in EXPECTED_INVARIANTS.items():
        full_path = WORKSPACE_ROOT / rel_path
        assert full_path.exists(), f"Invariant file missing: {rel_path}"
        actual_sha = compute_sha256(full_path)
        assert actual_sha == expected_sha, f"SHA-256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"

def test_02_test_evaluation_lock_integrity():
    """Verify test evaluation lock exists, is frozen, and points to winning candidate."""
    assert LOCK_PATH.exists(), "Test evaluation lock missing"
    with open(LOCK_PATH, "r", encoding="utf-8") as f:
        lock = json.load(f)
    assert lock["status"] == "FROZEN_LOCKED"
    assert lock["manifest_sha256"] == EXPECTED_INVARIANTS["backend/config/v2_3_frozen_experiment_manifest.json"]
    assert lock["selected_candidate"] in ["ocean_atmos_all", "ocean_atmos_base", "ocean_atmos_dynamic"]
    assert len(lock["all_candidate_model_hashes"]) == 20

def test_03_all_20_candidate_models_valid():
    """Verify that all 20 candidate models exist, load cleanly, and contain expected container keys."""
    model_subdirs = ["ocean_only", "ocean_atmos_base", "ocean_atmos_dynamic", "ocean_atmos_all", "atmos_only"]
    horizons = ["0d", "1d", "2d", "3d"]
    
    for subdir in model_subdirs:
        model_dir = CANDIDATES_ROOT / subdir
        assert model_dir.exists(), f"Candidate directory missing: {subdir}"
        assert (model_dir / "feature_manifest.json").exists()
        assert (model_dir / "validation_predictions.parquet").exists()
        assert (model_dir / "test_predictions.parquet").exists()
        
        for h in horizons:
            model_path = model_dir / f"risk_model_{h}.joblib"
            assert model_path.exists(), f"Model file missing: {model_path}"
            container = joblib.load(model_path)
            assert isinstance(container, dict)
            assert "model" in container
            assert container["scaler"] is None
            assert container["is_scaled"] is False
            assert "train_metrics" in container
            assert len(container["feature_columns"]) > 0

def test_04_final_test_results_artifact():
    """Verify final test results artifact structure and paired bootstrap statistics."""
    results_path = CANDIDATES_ROOT / "final_test_results.json"
    assert results_path.exists(), "final_test_results.json missing"
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    assert "test_results_table" in res
    assert "paired_bootstrap" in res
    assert res["selected_candidate"] == "ocean_atmos_all"
    assert "3d" in res["paired_bootstrap"]
    boot_3d = res["paired_bootstrap"]["3d"]
    assert "delta_pr_auc" in boot_3d
    assert boot_3d["delta_pr_auc"]["mean"] > 0.0  # Statistically positive gain
