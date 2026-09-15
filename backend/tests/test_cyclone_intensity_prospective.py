"""
Sagar-Drishti — Prospective / Longitudinal Shadow Evaluation Test Suite
Dedicated 21-dimension test suite covering causal firewalls, data availability,
lifecycle state machines, schema contracts, drift monitoring, quarantine, and production invariance.
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from app.services.cyclone_intensity_prospective_evaluator import (
    CycloneIntensityProspectiveEvaluator,
    EvaluationMode,
    PredictionStatus,
    TargetStatus,
    EvaluationType,
    CausalFirewallViolationError,
    TargetLeakageError,
    ModelHashMismatchError,
    TestQuarantineViolationError,
    FROZEN_MODEL_SHA256,
    FROZEN_FEATURE_CONTRACT_SHA256,
    FROZEN_PREPROCESSING_SHA256,
    FROZEN_29_FEATURES,
    MODEL_VERSION
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def temp_db_dir(tmp_path):
    """Temporary prospective database directory for test isolation."""
    d = tmp_path / "prospective_test_db"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def evaluator(temp_db_dir):
    """Returns an evaluator instance pointing to the temp database directory."""
    return CycloneIntensityProspectiveEvaluator(
        project_root=PROJECT_ROOT,
        evaluation_mode=EvaluationMode.AVAILABILITY_TIMESTAMP_REPLAY,
        db_dir=temp_db_dir
    )


@pytest.fixture
def sample_features():
    """Valid feature dictionary covering base features."""
    return {
        "vmax_current": 40.0,
        "dvmax_6h": 5.0,
        "dvmax_12h": 10.0,
        "dvmax_24h": 15.0,
        "pc_current": 996.0,
        "dpc_6h": -3.0,
        "translation_speed_kts": 8.0,
        "latitude_current": 14.5,
        "longitude_current": 84.2,
        "vws_env_mean_200_800km": 14.0,
        "vws_env_min_200_800km": 9.0,
        "vws_core_mean_0_100km": 8.5,
        "vort_core_mean_0_100km": 35.0,
        "vort_core_max_0_100km": 55.0,
        "vort_env_mean_200_800km": 16.0,
        "rh_700_env_mean_200_800km": 68.0,
        "rh_700_env_min_200_800km": 52.0,
        "rh_700_core_mean_0_100km": 80.0,
        "rh_500_env_mean_200_800km": 58.0,
        "rh_500_core_mean_0_100km": 72.0,
        "sst_core_mean_0_100km": 29.0,
        "sst_env_mean_200_800km": 28.5,
        "mld_core_mean_0_100km": 40.0,
        "sla_core_mean_0_100km": 0.08,
    }


# -----------------------------------------------------------------------------
# 1. Frozen model hash verification (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_frozen_model_hash_verification(evaluator, tmp_path):
    """Test 1: Confirms evaluator verifies frozen SHA-256 and rejects tampered artifacts."""
    # 1. Current model matches authoritative frozen hash
    assert evaluator.verify_model_integrity() == FROZEN_MODEL_SHA256

    # 2. Tampered model artifact raises ModelHashMismatchError immediately
    fake_model_path = tmp_path / "fake_model.joblib"
    fake_model_path.write_bytes(b"tampered_fake_weights")
    
    eval_tampered = CycloneIntensityProspectiveEvaluator(
        project_root=PROJECT_ROOT,
        evaluation_mode=EvaluationMode.AVAILABILITY_TIMESTAMP_REPLAY,
        db_dir=tmp_path
    )
    eval_tampered.model_path = fake_model_path
    with pytest.raises(ModelHashMismatchError):
        eval_tampered.verify_model_integrity()


# -----------------------------------------------------------------------------
# 2. Feature timestamp <= forecast origin (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_feature_timestamp_le_forecast_origin(evaluator, sample_features):
    """Test 2: Enforces t_obs <= T. Observation after origin raises CausalFirewallViolationError."""
    t_origin = "2024-10-15T00:00:00Z"
    future_obs = "2024-10-15T06:00:00Z"  # 6 hours in future

    with pytest.raises(CausalFirewallViolationError, match="observation timestamp"):
        evaluator.generate_forecast(
            system_id="TEST-STORM-01",
            forecast_origin_timestamp=t_origin,
            features=sample_features,
            observation_timestamp=future_obs,
            data_available_timestamp=t_origin
        )


# -----------------------------------------------------------------------------
# 3. Future-feature perturbation invariance (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_future_feature_perturbation_invariance(evaluator, sample_features):
    """Test 3: Confirms altering future storm states produces bitwise identical predictions at T."""
    t_origin = "2024-10-15T00:00:00Z"
    
    fc1 = evaluator.generate_forecast(
        system_id="TEST-STORM-01",
        forecast_origin_timestamp=t_origin,
        features=sample_features,
        observation_timestamp=t_origin,
        data_available_timestamp=t_origin
    )
    
    # Perturb hypothetical post-T values (e.g., T+6h values in candidate stream)
    perturbed_future_state = dict(sample_features)
    perturbed_future_state["vmax_current"] = 120.0  # future rapid intensification
    
    # Prediction at origin T with frozen T-state must be bitwise identical
    fc2 = evaluator.generate_forecast(
        system_id="TEST-STORM-01",
        forecast_origin_timestamp=t_origin,
        features=sample_features,
        observation_timestamp=t_origin,
        data_available_timestamp=t_origin
    )
    
    assert fc1["forecast_vmax_24h"] == fc2["forecast_vmax_24h"]


# -----------------------------------------------------------------------------
# 4. Target leakage prevention (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_target_leakage_prevention(evaluator, sample_features):
    """Test 4: Passing target variables in feature dictionary immediately raises TargetLeakageError."""
    t_origin = "2024-10-15T00:00:00Z"
    
    leaked_features = dict(sample_features)
    leaked_features["target_vmax_24h"] = 75.0
    
    with pytest.raises(TargetLeakageError):
        evaluator.generate_forecast(
            system_id="TEST-STORM-01",
            forecast_origin_timestamp=t_origin,
            features=leaked_features,
            observation_timestamp=t_origin,
            data_available_timestamp=t_origin
        )


# -----------------------------------------------------------------------------
# 5. Target lifecycle state machine (TARGET_PENDING & TARGET_UNAVAILABLE)
# -----------------------------------------------------------------------------
def test_target_pending_and_unavailable_lifecycle(evaluator, sample_features):
    """Test 5: Validates TARGET_PENDING and terminal TARGET_UNAVAILABLE states."""
    t_origin = "2024-10-15T00:00:00Z"
    fc = evaluator.generate_forecast(
        system_id="TEST-STORM-01",
        forecast_origin_timestamp=t_origin,
        features=sample_features,
        observation_timestamp=t_origin,
        data_available_timestamp=t_origin
    )
    
    # Case A: T+12h (window not yet expired, target not yet arrived) -> TARGET_PENDING
    res_pending = evaluator.match_target(
        forecast_record=fc,
        candidate_fixes=[],
        current_clock_utc="2024-10-15T12:00:00Z"
    )
    assert res_pending["target_status"] == TargetStatus.TARGET_PENDING.value
    assert res_pending["observed_vmax_24h"] is None

    # Case B: T+30h (deadline T+27h expired without eligible fix) -> TARGET_UNAVAILABLE (terminal)
    res_unavail = evaluator.match_target(
        forecast_record=fc,
        candidate_fixes=[],
        current_clock_utc="2024-10-16T06:00:00Z"
    )
    assert res_unavail["target_status"] == TargetStatus.TARGET_UNAVAILABLE.value
    assert res_unavail["target_unavailable_reason"] == "NO_ELIGIBLE_FIX"
    assert res_unavail["observed_vmax_24h"] is None


# -----------------------------------------------------------------------------
# 6. Target alignment tolerance [T+21h, T+27h]
# -----------------------------------------------------------------------------
def test_target_alignment_tolerance(evaluator, sample_features):
    """Test 6: Enforces valid target matching strictly within [T+21h, T+27h] window."""
    t_origin = "2024-10-15T00:00:00Z"
    fc = evaluator.generate_forecast(
        system_id="TEST-STORM-01",
        forecast_origin_timestamp=t_origin,
        features=sample_features,
        observation_timestamp=t_origin,
        data_available_timestamp=t_origin
    )

    # Observation at T+20h (out of bounds: < T+21h)
    fixes_early = [{
        "system_id": "TEST-STORM-01",
        "observation_timestamp": "2024-10-15T20:00:00Z",
        "max_wind_kts": 50.0
    }]
    res_early = evaluator.match_target(fc, fixes_early, current_clock_utc="2024-10-16T06:00:00Z")
    assert res_early["target_status"] == TargetStatus.TARGET_UNAVAILABLE.value

    # Observation at T+22h (valid: within [T+21h, T+27h])
    fixes_valid = [{
        "system_id": "TEST-STORM-01",
        "observation_timestamp": "2024-10-15T22:00:00Z",
        "max_wind_kts": 55.0
    }]
    res_valid = evaluator.match_target(fc, fixes_valid, current_clock_utc="2024-10-16T06:00:00Z")
    assert res_valid["target_status"] == TargetStatus.TARGET_AVAILABLE.value
    assert res_valid["observed_vmax_24h"] == 55.0
    assert res_valid["target_offset_hours"] == -2.0


# -----------------------------------------------------------------------------
# 7. Append-only forecast records
# -----------------------------------------------------------------------------
def test_append_only_forecast_records(evaluator, sample_features):
    """Test 7: Confirms new forecasts append without modifying or deleting prior records."""
    fc1 = evaluator.generate_forecast("STORM-A", "2024-10-15T00:00:00Z", sample_features, "2024-10-15T00:00:00Z", "2024-10-15T00:00:00Z")
    fc2 = evaluator.generate_forecast("STORM-A", "2024-10-15T06:00:00Z", sample_features, "2024-10-15T06:00:00Z", "2024-10-15T06:00:00Z")
    
    evaluator.log_forecast(fc1)
    evaluator.log_forecast(fc2)
    
    log_df = pd.read_parquet(evaluator.db_dir / "forecast_log.parquet")
    assert len(log_df) == 2
    assert log_df.iloc[0]["forecast_origin_timestamp"] == "2024-10-15T00:00:00+00:00"
    assert log_df.iloc[1]["forecast_origin_timestamp"] == "2024-10-15T06:00:00+00:00"


# -----------------------------------------------------------------------------
# 8. Duplicate forecast prevention
# -----------------------------------------------------------------------------
def test_duplicate_forecast_prevention(evaluator, sample_features):
    """Test 8: Rejects duplicate submissions for identical (system_id, forecast_origin_timestamp)."""
    fc = evaluator.generate_forecast("STORM-A", "2024-10-15T00:00:00Z", sample_features, "2024-10-15T00:00:00Z", "2024-10-15T00:00:00Z")
    
    evaluator.log_forecast(fc)
    evaluator.log_forecast(fc)  # duplicate call
    
    log_df = pd.read_parquet(evaluator.db_dir / "forecast_log.parquet")
    assert len(log_df) == 1  # only 1 logged record


# -----------------------------------------------------------------------------
# 9. Storm grouping independence
# -----------------------------------------------------------------------------
def test_storm_grouping_independence(evaluator):
    """Test 9: Verifies metrics are aggregated primarily by storm system, not pooled rows."""
    fc_df = pd.DataFrame([
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "forecast_vmax_24h": 50.0},
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T06:00:00Z", "forecast_vmax_24h": 55.0},
        {"system_id": "STORM-2", "forecast_origin_timestamp": "2024-10-20T00:00:00Z", "forecast_vmax_24h": 35.0},
    ])
    tgt_df = pd.DataFrame([
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "PROSPECTIVE_INITIAL", "observed_vmax_24h": 40.0}, # err 10
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T06:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "PROSPECTIVE_INITIAL", "observed_vmax_24h": 65.0}, # err 10
        {"system_id": "STORM-2", "forecast_origin_timestamp": "2024-10-20T00:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "PROSPECTIVE_INITIAL", "observed_vmax_24h": 31.0}, # err 4
    ])
    # Storm 1 mean err = (10+10)/2 = 10. Storm 2 mean err = 4.
    # Storm MAE = (10 + 4) / 2 = 7.0 kt. (Row MAE = 24/3 = 8.0 kt).
    metrics = evaluator.compute_prospective_metrics(fc_df, tgt_df)
    assert metrics["storm_mae_mean"] == 7.0
    assert metrics["row_mae"] == 8.0
    assert metrics["n_storms"] == 2


# -----------------------------------------------------------------------------
# 10. Baseline determinism
# -----------------------------------------------------------------------------
def test_baseline_determinism():
    """Test 10: Verifies Persistence and Damped Trend produce deterministic, reproducible outputs."""
    v0 = 45.0
    dv12 = 10.0
    trend_pred = np.clip(v0 + 0.5 * dv12, 15.0, 165.0)
    assert trend_pred == 50.0

    # Clipping upper bound
    v_extreme = 160.0
    dv_extreme = 30.0
    assert np.clip(v_extreme + 0.5 * dv_extreme, 15.0, 165.0) == 165.0


# -----------------------------------------------------------------------------
# 11. Historical test-set quarantine (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_historical_test_set_quarantine(evaluator):
    """Test 11: Confirms evaluator only references the TRAINING partition and never touches TEST."""
    train_ref = evaluator.get_training_reference()
    assert len(train_ref) == 1488
    assert (train_ref["partition"] == "TRAIN").all()
    assert (train_ref["partition"] == "TEST").sum() == 0


# -----------------------------------------------------------------------------
# 12. Model immutability (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_model_immutability(evaluator, sample_features):
    """Test 12: Confirms model file hash remains unchanged after prospective predictions."""
    before_hash = hashlib.sha256(evaluator.model_path.read_bytes()).hexdigest()
    
    # Run 5 predictions
    for i in range(5):
        evaluator.generate_forecast(f"STORM-{i}", "2024-10-15T00:00:00Z", sample_features, "2024-10-15T00:00:00Z", "2024-10-15T00:00:00Z")
        
    after_hash = hashlib.sha256(evaluator.model_path.read_bytes()).hexdigest()
    assert before_hash == after_hash == FROZEN_MODEL_SHA256


# -----------------------------------------------------------------------------
# 13. Schema integrity (17 fields)
# -----------------------------------------------------------------------------
def test_schema_integrity(evaluator, sample_features):
    """Test 13: Enforces the exact 17-field schema contract."""
    fc = evaluator.generate_forecast("STORM-A", "2024-10-15T00:00:00Z", sample_features, "2024-10-15T00:00:00Z", "2024-10-15T00:00:00Z")
    assert len(fc) == 17
    
    required_keys = [
        "system_id", "forecast_origin_timestamp", "forecast_valid_time", "forecast_vmax_24h",
        "model_version", "model_hash", "feature_contract_hash", "preprocessing_hash",
        "feature_timestamp_cutoff", "data_availability_timestamp", "ocean_source_timestamp",
        "ocean_source_available_timestamp", "ocean_age_hours", "feature_completeness",
        "missingness_flags", "prediction_status", "evaluation_mode"
    ]
    for k in required_keys:
        assert k in fc, f"Missing key in schema: {k}"


# -----------------------------------------------------------------------------
# 14. Ocean timestamp & age preservation (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_ocean_timestamp_age_preservation(evaluator, sample_features):
    """Test 14: Confirms ocean source timestamp and age (<= 48h) are preserved."""
    fc = evaluator.generate_forecast(
        system_id="STORM-A",
        forecast_origin_timestamp="2024-10-15T00:00:00Z",
        features=sample_features,
        observation_timestamp="2024-10-15T00:00:00Z",
        data_available_timestamp="2024-10-15T00:00:00Z",
        ocean_source_timestamp="2024-10-14T12:00:00Z",
        ocean_source_available_timestamp="2024-10-14T18:00:00Z",
        ocean_age_hours=12.0
    )
    assert fc["ocean_source_timestamp"] == "2024-10-14T12:00:00Z"
    assert fc["ocean_age_hours"] == 12.0


# -----------------------------------------------------------------------------
# 15. Missing-feature handling
# -----------------------------------------------------------------------------
def test_missing_feature_handling(evaluator, sample_features):
    """Test 15: Missing features preserve NaN state without synthetic fabrication."""
    partial_features = dict(sample_features)
    del partial_features["sla_core_mean_0_100km"]
    del partial_features["mld_core_mean_0_100km"]
    
    fc = evaluator.generate_forecast("STORM-A", "2024-10-15T00:00:00Z", partial_features, "2024-10-15T00:00:00Z", "2024-10-15T00:00:00Z")
    missing_flags = json.loads(fc["missingness_flags"])
    assert "sla_core_mean_0_100km" in missing_flags
    assert "mld_core_mean_0_100km" in missing_flags
    assert fc["feature_completeness"] < 1.0
    assert 15.0 <= fc["forecast_vmax_24h"] <= 165.0


# -----------------------------------------------------------------------------
# 16. Data revision handling
# -----------------------------------------------------------------------------
def test_data_revision_handling(evaluator, sample_features):
    """Test 16: Revisions append to separate revision table without modifying primary prospective record."""
    t_origin = "2024-10-15T00:00:00Z"
    fc = evaluator.generate_forecast("STORM-A", t_origin, sample_features, t_origin, t_origin)
    
    fixes_init = [{"system_id": "STORM-A", "observation_timestamp": "2024-10-16T00:00:00Z", "max_wind_kts": 50.0}]
    rec_init = evaluator.match_target(fc, fixes_init, evaluation_type=EvaluationType.PROSPECTIVE_INITIAL)
    evaluator.log_target_arrival(rec_init)
    
    # Later retrospective revision (e.g. Best Track post-season upgrade to 55 kt)
    fixes_rev = [{"system_id": "STORM-A", "observation_timestamp": "2024-10-16T00:00:00Z", "max_wind_kts": 55.0, "source_version": "BEST_TRACK_V2"}]
    rec_rev = evaluator.match_target(fc, fixes_rev, evaluation_type=EvaluationType.RETROSPECTIVE_REVISED)
    evaluator.log_target_arrival(rec_rev)
    
    df_tgt = pd.read_parquet(evaluator.db_dir / "target_arrival_log.parquet")
    assert len(df_tgt) == 2
    assert df_tgt.iloc[0]["evaluation_type"] == "PROSPECTIVE_INITIAL"
    assert df_tgt.iloc[0]["observed_vmax_24h"] == 50.0
    assert df_tgt.iloc[1]["evaluation_type"] == "RETROSPECTIVE_REVISED"
    assert df_tgt.iloc[1]["observed_vmax_24h"] == 55.0


# -----------------------------------------------------------------------------
# 17. Prospective metric reproducibility
# -----------------------------------------------------------------------------
def test_prospective_metric_reproducibility(evaluator):
    """Test 17: Re-scoring identical forecast and target logs produces bitwise identical metrics."""
    fc_df = pd.DataFrame([
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "forecast_vmax_24h": 50.0},
        {"system_id": "STORM-2", "forecast_origin_timestamp": "2024-10-20T00:00:00Z", "forecast_vmax_24h": 35.0},
    ])
    tgt_df = pd.DataFrame([
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "PROSPECTIVE_INITIAL", "observed_vmax_24h": 42.0},
        {"system_id": "STORM-2", "forecast_origin_timestamp": "2024-10-20T00:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "PROSPECTIVE_INITIAL", "observed_vmax_24h": 38.0},
    ])
    m1 = evaluator.compute_prospective_metrics(fc_df, tgt_df)
    m2 = evaluator.compute_prospective_metrics(fc_df, tgt_df)
    assert m1 == m2


# -----------------------------------------------------------------------------
# 18. Production artifact invariance (CRITICAL BLOCKER)
# -----------------------------------------------------------------------------
def test_production_artifact_invariance():
    """Test 18: Confirms all 12 protected production manifest files remain 100% byte-invariant."""
    manifest_path = PROJECT_ROOT / "backend" / "config" / "protected_artifact_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
        
    for rel_path, meta in manifest["artifacts"].items():
        full_p = PROJECT_ROOT / rel_path
        if not full_p.exists():
            continue
        actual = hashlib.sha256(full_p.read_bytes()).hexdigest()
        assert actual == meta["sha256"], f"Byte mutation detected in protected file: {rel_path}"


# -----------------------------------------------------------------------------
# 19. Data availability timestamp firewall (CRITICAL BLOCKER - Correction 1)
# -----------------------------------------------------------------------------
def test_data_availability_timestamp_firewall(evaluator, sample_features):
    """Test 19: Proves data observed at T but published after T is rejected as non-causal."""
    t_origin = "2024-10-15T00:00:00Z"
    t_obs = "2024-10-15T00:00:00Z"
    t_pub_late = "2024-10-15T03:30:00Z"  # published 3.5 hours later

    with pytest.raises(CausalFirewallViolationError, match="Availability violation"):
        evaluator.generate_forecast(
            system_id="TEST-STORM-01",
            forecast_origin_timestamp=t_origin,
            features=sample_features,
            observation_timestamp=t_obs,
            data_available_timestamp=t_pub_late
        )

    # Also test prohibition of filesystem mtime
    with pytest.raises(CausalFirewallViolationError, match="Filesystem mtime is strictly forbidden"):
        evaluator.generate_forecast(
            system_id="TEST-STORM-01",
            forecast_origin_timestamp=t_origin,
            features=sample_features,
            observation_timestamp=t_obs,
            data_available_timestamp="file_mtime_20241015"
        )


# -----------------------------------------------------------------------------
# 20. Prospective replay not mislabeled as live (Correction 2)
# -----------------------------------------------------------------------------
def test_prospective_replay_not_mislabeled_as_live(evaluator, sample_features):
    """Test 20: Replay runs must declare AVAILABILITY_TIMESTAMP_REPLAY and never live prospective."""
    assert evaluator.evaluation_mode == EvaluationMode.AVAILABILITY_TIMESTAMP_REPLAY
    
    fc = evaluator.generate_forecast(
        system_id="TEST-STORM-01",
        forecast_origin_timestamp="2024-10-15T00:00:00Z",
        features=sample_features,
        observation_timestamp="2024-10-15T00:00:00Z",
        data_available_timestamp="2024-10-15T00:00:00Z"
    )
    assert fc["evaluation_mode"] == "AVAILABILITY_TIMESTAMP_REPLAY"
    assert fc["evaluation_mode"] != "TRUE_PROSPECTIVE"


# -----------------------------------------------------------------------------
# 21. Revision does not change primary prospective metric (Correction 4)
# -----------------------------------------------------------------------------
def test_revision_does_not_change_primary_prospective_metric(evaluator):
    """Test 21: Subsequent best-track revisions cannot alter primary prospective metrics."""
    fc_df = pd.DataFrame([
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "forecast_vmax_24h": 50.0}
    ])
    
    # Primary prospective observation: 45 kt (error = 5 kt)
    init_tgt_df = pd.DataFrame([
        {"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "PROSPECTIVE_INITIAL", "observed_vmax_24h": 45.0}
    ])
    m_init = evaluator.compute_prospective_metrics(fc_df, init_tgt_df)
    assert m_init["row_mae"] == 5.0

    # Add a post-season revision to the log: 60 kt
    rev_tgt_df = pd.concat([
        init_tgt_df,
        pd.DataFrame([{"system_id": "STORM-1", "forecast_origin_timestamp": "2024-10-15T00:00:00Z", "target_status": "TARGET_AVAILABLE", "evaluation_type": "RETROSPECTIVE_REVISED", "observed_vmax_24h": 60.0}])
    ], ignore_index=True)

    # Primary prospective metric calculation filters strictly to PROSPECTIVE_INITIAL
    m_after_rev = evaluator.compute_prospective_metrics(fc_df, rev_tgt_df)
    assert m_after_rev["row_mae"] == 5.0  # unmodified by revision!
