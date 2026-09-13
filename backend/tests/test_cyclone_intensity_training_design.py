"""
Sagar-Drishti — Cyclone Intensity ML Training & Experimental Design Gate V1 Tests
Enforces zero model fitting, code-level test set quarantine, storm isolation,
anti-leakage controls, authoritative 12 protected artifact hashes, and deterministic baselines.
"""

import os
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from app.services.intensity_data_manager import (
    CycloneIntensityDataManager,
    TestSetQuarantineViolationError,
    TargetLeakageError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_zero_model_fitting_occurs():
    """Item 25.1: Verifies NO model files or checkpoints exist in research or candidate intensity dirs."""
    forbidden_dirs = [
        PROJECT_ROOT / "research" / "cyclone_intensity",
        PROJECT_ROOT / "backend" / "models" / "candidates" / "intensity",
    ]
    for d in forbidden_dirs:
        if d.exists():
            for ext in ["*.joblib", "*.pkl", "*.pickle", "*.pt", "*.pth", "*.bin"]:
                matches = list(d.glob(ext))
                assert len(matches) == 0, f"Found forbidden trained model checkpoint: {matches}"


def test_code_level_test_set_quarantine():
    """Item 25.2 & Directive 1, 5, 6: Verifies code-level quarantine raises exception on unauthorized access."""
    dm = CycloneIntensityDataManager(PROJECT_ROOT / "research" / "cyclone_intensity" / "features_6hourly_candidate.parquet")

    # Accessing test partition without keycard MUST raise TestSetQuarantineViolationError
    with pytest.raises(TestSetQuarantineViolationError):
        dm.get_test_data()

    # Reject standalone authorization string
    with pytest.raises(TestSetQuarantineViolationError):
        dm.get_test_data(keycard_or_path="AUTHORIZE_TEST_EVALUATION")

    # Calculating baseline on test partition MUST raise TestSetQuarantineViolationError
    with pytest.raises(TestSetQuarantineViolationError):
        dm.compute_deterministic_baseline("persistence", "TEST")

    with pytest.raises(TestSetQuarantineViolationError):
        dm.compute_deterministic_baseline("damped_trend", "TEST")


def test_multi_factor_keycard_and_single_access_enforcement():
    """Directive 5 & 6: Tests multi-factor keycard cryptographic verification and single-access limit."""
    dm = CycloneIntensityDataManager(PROJECT_ROOT / "research" / "cyclone_intensity" / "features_6hourly_candidate.parquet")
    keycard_path = PROJECT_ROOT / "research" / "cyclone_intensity" / "test_keycard.json"
    assert keycard_path.exists()

    with open(keycard_path) as f:
        valid_card = json.load(f)

    # 1. Tampered hash MUST fail
    bad_card = dict(valid_card)
    bad_card["FINAL_MODEL_HASH"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with pytest.raises(TestSetQuarantineViolationError):
        dm.get_test_data(keycard_or_path=bad_card)

    # 2. TEST_ACCESS_COUNT != 0 MUST fail
    bad_card_access = dict(valid_card)
    bad_card_access["TEST_ACCESS_COUNT"] = 1
    with pytest.raises(TestSetQuarantineViolationError):
        dm.get_test_data(keycard_or_path=bad_card_access)

    # 3. Valid keycard unlocks on first attempt
    test_df, test_y = dm.get_test_data(keycard_or_path=keycard_path)
    assert len(test_df) == 396
    assert dm.test_access_count == 1

    # 4. Directive 6: Any second access attempt MUST fail immediately
    with pytest.raises(TestSetQuarantineViolationError):
        dm.get_test_data(keycard_or_path=keycard_path)

    # 5. Directive 8: Audit log exists and records all required fields
    audit_path = PROJECT_ROOT / "research" / "cyclone_intensity" / "test_access_audit.json"
    assert audit_path.exists()
    with open(audit_path) as f:
        audit_data = json.load(f)

    assert "unlock_timestamp_utc" in audit_data
    assert "keycard_hash" in audit_data
    assert "model_hash" in audit_data
    assert "dataset_hash" in audit_data
    assert audit_data["test_partition"] == "2024-2026"
    assert "evaluator_version" in audit_data
    assert audit_data["access_count"] == 1
    assert audit_data["read_only_enforced"] is True


def test_storm_ids_strictly_isolated():
    """Item 25.3: Verifies storm systems never cross partitions (zero overlap between Train, Val, Test)."""
    dataset_path = PROJECT_ROOT / "research" / "cyclone_intensity" / "features_6hourly_candidate.parquet"
    df = pd.read_parquet(dataset_path)

    train_storms = set(df[df["partition"] == "TRAIN"]["system_id"])
    val_storms = set(df[df["partition"] == "VALIDATION"]["system_id"])
    test_storms = set(df[df["partition"] == "TEST"]["system_id"])

    assert len(train_storms) == 64
    assert len(val_storms) == 24
    assert len(test_storms) == 29
    assert len(train_storms) + len(val_storms) + len(test_storms) == 117

    assert len(train_storms.intersection(val_storms)) == 0, "Storm ID crossover between Train and Validation!"
    assert len(train_storms.intersection(test_storms)) == 0, "Storm ID crossover between Train and Test!"
    assert len(val_storms.intersection(test_storms)) == 0, "Storm ID crossover between Validation and Test!"


def test_target_variables_never_leak_into_features():
    """Item 25.4: Verifies target columns never leak into input feature sets."""
    dm = CycloneIntensityDataManager(PROJECT_ROOT / "research" / "cyclone_intensity" / "features_6hourly_candidate.parquet")

    # Ingestion check: passing target column to feature selection raises TargetLeakageError
    with pytest.raises(TargetLeakageError):
        dm.get_training_data(feature_cols=["vmax_current", "target_vmax_24h"])

    with pytest.raises(TargetLeakageError):
        dm.get_validation_data(feature_cols=["vmax_current", "target_delta_vmax_24h"])

    # Feature groups check: feature_groups.json contains zero target columns
    with open(PROJECT_ROOT / "research" / "cyclone_intensity" / "feature_groups.json") as f:
        fg = json.load(f)

    for grp_key, grp_data in fg["feature_groups"].items():
        if "features" in grp_data:
            for feat in grp_data["features"]:
                assert not feat.startswith("target_"), f"Target column in feature group {grp_key}: {feat}"


def test_feature_groups_mutually_auditable_and_group_d_explicit():
    """Item 25.7 & Corrections 4 & 5: Verifies Group D explicit columns and Ablation E radial contrasts."""
    with open(PROJECT_ROOT / "research" / "cyclone_intensity" / "feature_groups.json") as f:
        fg = json.load(f)

    # Base groups
    assert fg["feature_groups"]["group_a_kinematics"]["feature_count"] == 9
    assert fg["feature_groups"]["group_b_atmosphere"]["feature_count"] == 11
    assert fg["feature_groups"]["group_c_ocean"]["feature_count"] == 4

    # Group D explicit core and environmental features
    grp_d = fg["feature_groups"]["group_d_storm_centered_spatial_structure"]
    assert grp_d["total_spatial_features"] == 15
    assert len(grp_d["core_features_0_100km"]) == 8
    assert len(grp_d["environmental_features_200_800km"]) == 7

    # Ablation E differentiation
    abl_e = fg["ablation_experiments"]["experiment_e"]
    assert abl_e["base_feature_count"] == 24
    assert abl_e["derived_contrast_count"] == 5
    assert abl_e["total_feature_count"] == 29
    assert len(abl_e["frozen_contrast_formulas"]) == 5


def test_deterministic_baselines_train_and_val_only():
    """Item 25.8 & Corrections 1, 2, 3: Verifies deterministic baselines on Train and Val with alpha=0.5."""
    dm = CycloneIntensityDataManager(PROJECT_ROOT / "research" / "cyclone_intensity" / "features_6hourly_candidate.parquet")

    # Train Persistence: 58 storms with valid 24h targets (out of 64 total train storms)
    train_pers = dm.compute_deterministic_baseline("persistence", "TRAIN")
    assert train_pers["storms_count"] == 58
    assert train_pers["rows_count"] == 1121
    assert train_pers["row_mae_kts"] == 15.16
    assert train_pers["storm_mae_kts"] == 10.26

    # Validation Persistence & Damped Trend: 22 storms with valid 24h targets (out of 24 total val storms)
    val_pers = dm.compute_deterministic_baseline("persistence", "VALIDATION")
    assert val_pers["storms_count"] == 22
    assert val_pers["rows_count"] == 382
    assert val_pers["row_mae_kts"] == 12.12
    assert val_pers["storm_mae_kts"] == 7.92

    val_trend = dm.compute_deterministic_baseline("damped_trend", "VALIDATION", alpha=0.5)
    assert val_trend["row_mae_kts"] == 11.32
    assert val_trend["storm_mae_kts"] == 7.67


def test_benchmark_marked_provisional():
    """Item 25.9: Verifies 11.5 kt benchmark is explicitly marked PROVISIONAL / UNPROVEN."""
    with open(PROJECT_ROOT / "research" / "cyclone_intensity" / "feature_contract.json") as f:
        contract = json.load(f)

    status = contract["benchmark_provenance"]["provenance_status"]
    assert "PROVISIONAL" in status.upper()
    assert "UNPROVEN" in status.upper()


def test_rapid_intensification_remains_data_not_ready():
    """Item 25.10: Verifies RI remains DATA_NOT_READY."""
    readme_path = PROJECT_ROOT / "research" / "cyclone_intensity" / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "DATA_NOT_READY" in content


def test_protected_artifacts_and_authoritative_manifest_12():
    """Item 25.11 & Correction 9: Verifies all 12 protected artifacts match authoritative manifest."""
    manifest_path = PROJECT_ROOT / "backend" / "config" / "protected_artifact_manifest.json"
    assert manifest_path.exists(), "protected_artifact_manifest.json does not exist!"

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_protected_artifacts"] == 12
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 12

    for rel_path, meta in artifacts.items():
        full_path = PROJECT_ROOT / rel_path
        assert full_path.exists(), f"Protected file missing: {rel_path}"
        with open(full_path, "rb") as f:
            live_hash = hashlib.sha256(f.read()).hexdigest()
        assert live_hash == meta["sha256"], f"Byte divergence in protected file {rel_path}: {live_hash} != {meta['sha256']}"
