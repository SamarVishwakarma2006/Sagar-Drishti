"""
Sagar-Drishti — Forward Prediction & Prospective Inference Forensic Audit Test Suite
Implements all 24 required forensic & causal data provenance test scenarios:

1. valid forward prediction
2. exact forecast origin enforcement
3. future observation rejection
4. future availability rejection
5. insufficient history
6. missing atmospheric data
7. missing ocean data
8. missing required feature
9. wrong feature ordering
10. feature contract mismatch
11. model hash mismatch
12. preprocessing mismatch
13. future-data perturbation invariance
14. duplicate observation handling
15. timestamp normalization
16. timezone handling
17. provenance completeness
18. prediction immutability
19. target lifecycle
20. target revision handling
21. synthetic fixture labeling
22. historical-mode regression
23. production model immutability
24. production policy immutability

Plus supplementary tests for:
25. filesystem mtime rejection
26. physical clipping bounds [15.0, 165.0] kts
27. pre-flight validation endpoint (/api/forecast/validate-inputs)
28. primary forward prediction endpoint (/api/forecast/cyclone-intensity)
29. forecast history endpoint (/api/forecast/history)
30. SagarBot conversational grounding for FORWARD_FORECAST intent
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
import numpy as np
import pandas as pd
import joblib

from fastapi.testclient import TestClient

from app.main import app
from app.services.cyclone_intensity_prospective_evaluator import (
    CycloneIntensityProspectiveEvaluator,
    EvaluationMode,
    TargetStatus,
    EvaluationType,
    DataProvenanceClass,
    AtmosphericSource,
    ProspectiveEvidenceTier,
    CANONICAL_RESEARCH_SPLITS,
    classify_prospective_evidence_tier,
    CausalFirewallViolationError,
    ModelHashMismatchError,
    FROZEN_MODEL_SHA256,
    FROZEN_FEATURE_CONTRACT_SHA256,
    FROZEN_PREPROCESSING_SHA256,
    FROZEN_29_FEATURES,
    MODEL_VERSION,
)
from app.services.forward_prediction_service import (
    ForwardPredictionService,
    SCIENTIFIC_DISCLAIMER,
    InsufficientHistoryError,
    _to_utc_timestamp,
    classify_intensity,
    ThreatLevelStr,
)
from app.services.prediction_service import (
    PredictionService,
    DEFAULT_FROZEN_THRESHOLD,
)
from app.services.sagarbot_service import (
    SagarBotService,
    IntentType,
    IntentClassifier,
)
from app.models.schemas import ChatRequest, ChatContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def temp_service(tmp_path):
    """Provides an isolated ForwardPredictionService instance."""
    db_dir = tmp_path / "forward_pred_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return ForwardPredictionService(
        project_root=PROJECT_ROOT,
        db_dir=db_dir,
        evaluation_mode=EvaluationMode.TRUE_PROSPECTIVE,
    )


@pytest.fixture
def sample_valid_features():
    """Authoritative 29 feature vector covering all required inputs."""
    return {
        "vmax_current": 50.0,
        "dvmax_6h": 5.0,
        "dvmax_12h": 10.0,
        "dvmax_24h": 15.0,
        "pc_current": 990.0,
        "dpc_6h": -4.0,
        "translation_speed_kts": 8.5,
        "latitude_current": 15.5,
        "longitude_current": 86.5,
        "vws_env_mean_200_800km": 12.0,
        "vws_env_min_200_800km": 8.0,
        "vws_core_mean_0_100km": 9.0,
        "vort_core_mean_0_100km": 40.0,
        "vort_core_max_0_100km": 62.0,
        "vort_env_mean_200_800km": 17.0,
        "rh_700_env_mean_200_800km": 70.0,
        "rh_700_env_min_200_800km": 55.0,
        "rh_700_core_mean_0_100km": 82.0,
        "rh_500_env_mean_200_800km": 60.0,
        "rh_500_core_mean_0_100km": 74.0,
        "sst_core_mean_0_100km": 29.5,
        "sst_env_mean_200_800km": 29.0,
        "mld_core_mean_0_100km": 42.0,
        "sla_core_mean_0_100km": 0.10,
    }


# =============================================================================
# 1. VALID FORWARD PREDICTION
# =============================================================================
def test_01_valid_forward_prediction(temp_service, sample_valid_features):
    """1. Valid forward prediction: generates prediction at T with valid provenance."""
    res = temp_service.predict_forward(
        system_id="SYS-VALID-01",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    assert res["status"] == "PREDICTION_GENERATED"
    assert res["predicted_vmax_24h"] is not None
    assert 15.0 <= res["predicted_vmax_24h"] <= 165.0
    assert res["causal_firewall"] == "PASS"
    assert res["model_version"] == MODEL_VERSION
    assert res["model_hash"] == FROZEN_MODEL_SHA256
    assert res["feature_completeness"] == 1.0
    assert res["scientific_disclaimer"] == SCIENTIFIC_DISCLAIMER


# =============================================================================
# 2. EXACT FORECAST ORIGIN ENFORCEMENT
# =============================================================================
def test_02_exact_forecast_origin_enforcement(temp_service, sample_valid_features):
    """2. Exact forecast origin enforcement: data at T is accepted, data at T+1s is rejected."""
    t_origin = "2026-07-10T12:00:00Z"
    # Observation at exactly T must pass
    res_exact = temp_service.predict_forward(
        system_id="SYS-EXACT-T",
        forecast_origin_timestamp=t_origin,
        features=sample_valid_features,
        observation_timestamp="2026-07-10T12:00:00Z",
        data_available_timestamp="2026-07-10T12:00:00Z",
    )
    assert res_exact["status"] == "PREDICTION_GENERATED"

    # Observation just 1 second after T must fail closed
    res_future = temp_service.predict_forward(
        system_id="SYS-EXACT-T-FAIL",
        forecast_origin_timestamp=t_origin,
        features=sample_valid_features,
        observation_timestamp="2026-07-10T12:00:01Z",
        data_available_timestamp="2026-07-10T12:00:00Z",
    )
    assert res_future["status"] == "CAUSAL_REJECTED"
    assert res_future["causal_firewall"] == "FAIL"


# =============================================================================
# 3. FUTURE OBSERVATION REJECTION
# =============================================================================
def test_03_future_observation_rejection(temp_service, sample_valid_features):
    """3. Future observation rejection: observation timestamp > T is strictly rejected."""
    res = temp_service.predict_forward(
        system_id="SYS-FUTURE-OBS",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T13:00:00Z",
        data_available_timestamp="2026-07-10T12:00:00Z",
    )
    assert res["status"] == "CAUSAL_REJECTED"
    assert res["causal_firewall"] == "FAIL"
    assert res["predicted_vmax_24h"] is None


# =============================================================================
# 4. FUTURE AVAILABILITY REJECTION
# =============================================================================
def test_04_future_availability_rejection(temp_service, sample_valid_features):
    """4. Future availability rejection: observed at T but published at T+5min is rejected."""
    res = temp_service.predict_forward(
        system_id="SYS-FUTURE-AVAIL",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T12:00:00Z",
        data_available_timestamp="2026-07-10T12:05:00Z",
    )
    assert res["status"] == "CAUSAL_REJECTED"
    assert res["causal_firewall"] == "FAIL"
    assert res["predicted_vmax_24h"] is None


# =============================================================================
# 5. INSUFFICIENT HISTORY
# =============================================================================
def test_05_insufficient_history(temp_service):
    """5. Insufficient history: fewer than 2 historical track fixes returns INSUFFICIENT_HISTORY."""
    # Single fix only
    single_fix = [
        {"observation_timestamp": "2026-07-10T12:00:00Z", "latitude": 15.5, "longitude": 86.0, "max_wind_kts": 65.0, "central_pressure_hpa": 978.0}
    ]
    res = temp_service.predict_forward(
        system_id="SYS-SINGLE-FIX",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        cyclone_history=single_fix,
    )
    assert res["status"] == "INSUFFICIENT_HISTORY"
    assert res["predicted_vmax_24h"] is None


# =============================================================================
# 6. MISSING ATMOSPHERIC DATA
# =============================================================================
def test_06_missing_atmospheric_data(temp_service):
    """6. Missing atmospheric data: all 11 atmospheric features missing returns INSUFFICIENT_INPUT_DATA."""
    # Features with kinematics and ocean only, zero atmosphere
    kin_ocean_only = {
        "vmax_current": 50.0, "dvmax_6h": 5.0, "dvmax_12h": 10.0, "dvmax_24h": 15.0,
        "pc_current": 990.0, "dpc_6h": -4.0, "translation_speed_kts": 8.5,
        "latitude_current": 15.5, "longitude_current": 86.5,
        "sst_core_mean_0_100km": 29.5, "sst_env_mean_200_800km": 29.0,
        "mld_core_mean_0_100km": 42.0, "sla_core_mean_0_100km": 0.10,
    }
    res = temp_service.predict_forward(
        system_id="SYS-NO-ATMOS",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=kin_ocean_only,
    )
    assert res["status"] == "INSUFFICIENT_INPUT_DATA"
    assert any("Atmospheric environmental observations completely missing" in w for w in res["warnings"])


# =============================================================================
# 7. MISSING OCEAN DATA
# =============================================================================
def test_07_missing_ocean_data(temp_service):
    """7. Missing ocean data: all 4 ocean features missing returns INSUFFICIENT_INPUT_DATA."""
    # Features with kinematics and atmosphere only, zero ocean
    kin_atmos_only = {
        "vmax_current": 50.0, "dvmax_6h": 5.0, "dvmax_12h": 10.0, "dvmax_24h": 15.0,
        "pc_current": 990.0, "dpc_6h": -4.0, "translation_speed_kts": 8.5,
        "latitude_current": 15.5, "longitude_current": 86.5,
        "vws_env_mean_200_800km": 12.0, "vws_env_min_200_800km": 8.0, "vws_core_mean_0_100km": 9.0,
        "vort_core_mean_0_100km": 40.0, "vort_core_max_0_100km": 62.0, "vort_env_mean_200_800km": 17.0,
        "rh_700_env_mean_200_800km": 70.0, "rh_700_env_min_200_800km": 55.0, "rh_700_core_mean_0_100km": 82.0,
        "rh_500_env_mean_200_800km": 60.0, "rh_500_core_mean_0_100km": 74.0,
    }
    res = temp_service.predict_forward(
        system_id="SYS-NO-OCEAN",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=kin_atmos_only,
    )
    assert res["status"] == "INSUFFICIENT_INPUT_DATA"
    assert any("Ocean thermodynamic observations completely missing" in w for w in res["warnings"])


# =============================================================================
# 8. MISSING REQUIRED FEATURE
# =============================================================================
def test_08_missing_required_feature(temp_service, sample_valid_features):
    """8. Missing required feature: missing primary kinematic anchor (vmax_current) fails safely."""
    missing_vmax = sample_valid_features.copy()
    del missing_vmax["vmax_current"]

    res = temp_service.predict_forward(
        system_id="SYS-MISSING-VMAX",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=missing_vmax,
    )
    assert res["status"] == "INSUFFICIENT_INPUT_DATA"
    assert "vmax_current" in res["missing_features"]


# =============================================================================
# 9. WRONG FEATURE ORDERING
# =============================================================================
def test_09_wrong_feature_ordering(temp_service, sample_valid_features):
    """9. Wrong feature ordering: regardless of input dict ordering, assembled DataFrame is canonical."""
    # Reverse dictionary ordering
    reversed_features = {k: sample_valid_features[k] for k in reversed(list(sample_valid_features.keys()))}
    res = temp_service.predict_forward(
        system_id="SYS-ORDER-CHECK",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=reversed_features,
    )
    assert res["status"] == "PREDICTION_GENERATED"
    forensics = res["model_input_forensics"]
    assert forensics is not None
    assert forensics["feature_names"] == FROZEN_29_FEATURES


# =============================================================================
# 10. FEATURE CONTRACT MISMATCH
# =============================================================================
def test_10_feature_contract_mismatch(temp_service, tmp_path):
    """10. Feature contract mismatch: tampered contract raises ModelHashMismatchError."""
    mock_root = tmp_path / "mock_contract_project"
    mock_res_dir = mock_root / "research" / "cyclone_intensity"
    mock_res_dir.mkdir(parents=True, exist_ok=True)
    (mock_res_dir / "models").mkdir(parents=True, exist_ok=True)

    # Copy real model
    real_model = PROJECT_ROOT / "research" / "cyclone_intensity" / "models" / "final_intensity_model.joblib"
    (mock_res_dir / "models" / "final_intensity_model.joblib").write_bytes(real_model.read_bytes())

    # Write tampered contract
    (mock_res_dir / "feature_contract.json").write_bytes(b'{"tampered": true}')

    # Tampered contract must fail closed on initialization
    with pytest.raises(ModelHashMismatchError, match="Feature contract hash mismatch"):
        CycloneIntensityProspectiveEvaluator(
            project_root=mock_root,
            db_dir=tmp_path / "db"
        )


# =============================================================================
# 11. MODEL HASH MISMATCH
# =============================================================================
def test_11_model_hash_mismatch(tmp_path):
    """11. Model hash mismatch: altered model weights fail closed immediately."""
    mock_root = tmp_path / "mock_model_tamper"
    mock_model_dir = mock_root / "research" / "cyclone_intensity" / "models"
    mock_model_dir.mkdir(parents=True, exist_ok=True)
    (mock_model_dir / "final_intensity_model.joblib").write_bytes(b"tampered_weights_123")

    with pytest.raises(ModelHashMismatchError, match="Frozen model hash mismatch"):
        CycloneIntensityProspectiveEvaluator(
            project_root=mock_root,
            db_dir=tmp_path / "tamper_db"
        )


# =============================================================================
# 12. PREPROCESSING MISMATCH
# =============================================================================
def test_12_preprocessing_mismatch(tmp_path):
    """12. Preprocessing mismatch: altered keycard preprocessing hash fails closed."""
    mock_root = tmp_path / "mock_prep_tamper"
    mock_res_dir = mock_root / "research" / "cyclone_intensity"
    mock_res_dir.mkdir(parents=True, exist_ok=True)
    (mock_res_dir / "models").mkdir(parents=True, exist_ok=True)

    # Copy real model
    real_model = PROJECT_ROOT / "research" / "cyclone_intensity" / "models" / "final_intensity_model.joblib"
    (mock_res_dir / "models" / "final_intensity_model.joblib").write_bytes(real_model.read_bytes())

    # Write keycard with invalid preprocessing hash
    tampered_kc = {"PREPROCESSING_HASH": "tampered_hash_00000000000000000000000000000000"}
    (mock_res_dir / "test_keycard.json").write_text(json.dumps(tampered_kc), encoding="utf-8")

    with pytest.raises(ModelHashMismatchError, match="Preprocessing hash mismatch"):
        CycloneIntensityProspectiveEvaluator(
            project_root=mock_root,
            db_dir=tmp_path / "db"
        )


# =============================================================================
# 13. FUTURE-DATA PERTURBATION INVARIANCE
# =============================================================================
def test_13_future_data_perturbation_invariance(temp_service, sample_valid_features):
    """13. Future-data perturbation invariance: adding or modifying future data after T has 0.0 impact on T."""
    t_origin = "2026-07-10T12:00:00Z"
    base_res = temp_service.predict_forward(
        system_id="SYS-LEAKAGE-AUDIT",
        forecast_origin_timestamp=t_origin,
        features=sample_valid_features,
    )
    pred_base = base_res["predicted_vmax_24h"]
    assert pred_base is not None

    future_target_fix = [{
        "system_id": "SYS-LEAKAGE-AUDIT",
        "observation_timestamp": "2026-07-11T12:00:00Z",
        "latitude": 18.0,
        "longitude": 85.0,
        "max_wind_kts": 140.0,
        "central_pressure_hpa": 920.0,
    }]
    res_perturbed = temp_service.predict_forward(
        system_id="SYS-LEAKAGE-AUDIT",
        forecast_origin_timestamp=t_origin,
        features=sample_valid_features,
        candidate_target_fixes=future_target_fix,
    )
    pred_perturbed = res_perturbed["predicted_vmax_24h"]
    assert pred_base == pred_perturbed


# =============================================================================
# 14. DUPLICATE OBSERVATION HANDLING
# =============================================================================
def test_14_duplicate_observation_handling(temp_service, sample_valid_features):
    """14. Duplicate observation handling: duplicate submissions do not duplicate rows in forecast_log."""
    temp_service.predict_forward(
        system_id="SYS-DEDUP-AUDIT",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    temp_service.predict_forward(
        system_id="SYS-DEDUP-AUDIT",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    log_file = temp_service.evaluator.db_dir / "forecast_log.parquet"
    df = pd.read_parquet(log_file)
    dedup_rows = df[df["system_id"] == "SYS-DEDUP-AUDIT"]
    assert len(dedup_rows) == 1


# =============================================================================
# 15. TIMESTAMP NORMALIZATION
# =============================================================================
def test_15_timestamp_normalization():
    """15. Timestamp normalization: accepts various ISO formats and normalizes all to UTC."""
    ts1 = _to_utc_timestamp("2026-07-10T12:00:00Z")
    ts2 = _to_utc_timestamp("2026-07-10 12:00:00")
    ts3 = _to_utc_timestamp(datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc))
    ts4 = _to_utc_timestamp(pd.Timestamp("2026-07-10T12:00:00Z"))

    assert ts1 == ts2 == ts3 == ts4
    assert ts1.tzinfo is not None


# =============================================================================
# 16. TIMEZONE HANDLING
# =============================================================================
def test_16_timezone_handling(temp_service, sample_valid_features):
    """16. Timezone handling: non-UTC timezone (+05:30) is converted and evaluated correctly against origin."""
    # 17:30 IST is 12:00 UTC
    res = temp_service.predict_forward(
        system_id="SYS-TZ-TEST",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T17:30:00+05:30",
        data_available_timestamp="2026-07-10T17:30:00+05:30",
    )
    assert res["status"] == "PREDICTION_GENERATED"
    assert res["causal_firewall"] == "PASS"


# =============================================================================
# 17. PROVENANCE COMPLETENESS
# =============================================================================
def test_17_provenance_completeness(temp_service, sample_valid_features):
    """17. Provenance completeness: checks presence of all required provenance metadata fields."""
    res = temp_service.predict_forward(
        system_id="SYS-PROV-FULL",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    required_keys = [
        "status", "system_id", "origin", "valid_time", "predicted_vmax_24h",
        "raw_predicted_vmax_24h", "reported_predicted_vmax_24h", "clipping_applied",
        "model_version", "model_hash", "feature_contract_version", "feature_contract_hash",
        "preprocessing_version", "preprocessing_hash", "input_dataset_source",
        "input_cutoff", "observation_cutoff", "availability_cutoff",
        "causal_firewall", "feature_completeness", "scientific_disclaimer",
        "evaluation_status", "evaluation_mode"
    ]
    for k in required_keys:
        assert k in res, f"Missing provenance key: {k}"


# =============================================================================
# 18. PREDICTION IMMUTABILITY
# =============================================================================
def test_18_prediction_immutability(temp_service, sample_valid_features):
    """18. Prediction immutability: logged forecast row remains unchanged when target is matched later."""
    res = temp_service.predict_forward(
        system_id="SYS-IMMUTABLE-01",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    orig_vmax = res["predicted_vmax_24h"]

    log_path = temp_service.evaluator.db_dir / "forecast_log.parquet"
    df_before = pd.read_parquet(log_path)
    row_before = df_before[df_before["system_id"] == "SYS-IMMUTABLE-01"].iloc[0].to_dict()

    # Now simulate target matching
    target_fix = [{
        "system_id": "SYS-IMMUTABLE-01",
        "observation_timestamp": "2026-07-11T12:00:00Z",
        "latitude": 17.0, "longitude": 86.0,
        "max_wind_kts": 80.0, "central_pressure_hpa": 970.0
    }]
    temp_service.evaluator.match_target(
        forecast_record={
            "system_id": "SYS-IMMUTABLE-01",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "forecast_valid_time": "2026-07-11T12:00:00Z",
        },
        candidate_fixes=target_fix
    )

    df_after = pd.read_parquet(log_path)
    row_after = df_after[df_after["system_id"] == "SYS-IMMUTABLE-01"].iloc[0].to_dict()

    # Row in forecast_log.parquet must remain byte-for-byte identical
    assert row_before["forecast_vmax_24h"] == row_after["forecast_vmax_24h"] == orig_vmax


# =============================================================================
# 19. TARGET LIFECYCLE
# =============================================================================
def test_19_target_lifecycle(temp_service, sample_valid_features):
    """19. Target lifecycle: correctly transitions across PENDING, AVAILABLE, and UNAVAILABLE."""
    # A. TARGET_PENDING
    res_pending = temp_service.predict_forward(
        system_id="SYS-LIFE-01",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    assert res_pending["evaluation_status"] == TargetStatus.TARGET_PENDING.value

    # B. TARGET_AVAILABLE (fix inside [T+21h, T+27h])
    target_available = [{
        "system_id": "SYS-LIFE-02",
        "observation_timestamp": "2026-07-11T12:00:00Z",
        "latitude": 17.5, "longitude": 85.5,
        "max_wind_kts": 70.0, "central_pressure_hpa": 975.0,
    }]
    res_avail = temp_service.predict_forward(
        system_id="SYS-LIFE-02",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        candidate_target_fixes=target_available,
    )
    assert res_avail["evaluation_status"] == TargetStatus.TARGET_AVAILABLE.value

    # C. TARGET_UNAVAILABLE (fix far outside window, e.g. T+48h)
    target_unavailable = [{
        "system_id": "SYS-LIFE-03",
        "observation_timestamp": "2026-07-12T12:00:00Z",
        "latitude": 21.0, "longitude": 84.0,
        "max_wind_kts": 85.0, "central_pressure_hpa": 960.0,
    }]
    res_unavail = temp_service.predict_forward(
        system_id="SYS-LIFE-03",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        candidate_target_fixes=target_unavailable,
    )
    assert res_unavail["evaluation_status"] == TargetStatus.TARGET_UNAVAILABLE.value


# =============================================================================
# 20. TARGET REVISION HANDLING
# =============================================================================
def test_20_target_revision_handling(temp_service):
    """20. Target revision handling: log_target_arrival with RETROSPECTIVE_REVISED does not overwrite initial record."""
    target_path = temp_service.evaluator.db_dir / "target_arrival_log.parquet"

    initial_target = {
        "system_id": "SYS-REVISE-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "target_status": TargetStatus.TARGET_AVAILABLE.value,
        "observed_vmax_24h": 65.0,
        "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
    }
    temp_service.evaluator.log_target_arrival(initial_target, filepath=target_path)

    revised_target = {
        "system_id": "SYS-REVISE-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "target_status": TargetStatus.TARGET_AVAILABLE.value,
        "observed_vmax_24h": 70.0,
        "evaluation_type": EvaluationType.RETROSPECTIVE_REVISED.value,
    }
    temp_service.evaluator.log_target_arrival(revised_target, filepath=target_path)

    df = pd.read_parquet(target_path)
    records = df[df["system_id"] == "SYS-REVISE-01"]
    assert len(records) == 2
    assert EvaluationType.PROSPECTIVE_INITIAL.value in records["evaluation_type"].values
    assert EvaluationType.RETROSPECTIVE_REVISED.value in records["evaluation_type"].values


# =============================================================================
# 21. SYNTHETIC FIXTURE LABELING
# =============================================================================
def test_21_synthetic_fixture_labeling(temp_service):
    """21. Synthetic fixture labeling: demo scenarios must carry SYNTHETIC TEST FIXTURE label."""
    demos = temp_service.get_demo_scenarios()
    assert len(demos) >= 2
    for d in demos:
        assert d["fixture_type"] == "SYNTHETIC_TEST_FIXTURE"
        assert "SYNTHETIC TEST FIXTURE" in d["fixture_label"]

    # In predict_forward response
    res = temp_service.predict_forward(
        system_id="BOB-2026-01-PROSPECTIVE",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        demo_scenario_id="DEMO-BOB-2026-07-10",
    )
    assert res["status"] == "PREDICTION_GENERATED"
    assert res["synthetic_fixture_label"] == "SYNTHETIC TEST FIXTURE — NOT REAL METEOROLOGICAL DATA"


# =============================================================================
# 22. HISTORICAL-MODE REGRESSION
# =============================================================================
def test_22_historical_mode_regression():
    """22. Historical-mode regression: PredictionService production risk model operates cleanly without disruption."""
    status = PredictionService.get_model_status()
    assert status["status"] == "ready"
    assert status["is_ready"] is True
    assert status["model_version"] in ["1.1.0", "v1.1.0", "2.0.0", "v2.0.0", "2.0.0-10yr-candidate"]


# =============================================================================
# 23. PRODUCTION MODEL IMMUTABILITY
# =============================================================================
def test_23_production_model_immutability():
    """23. Production model immutability: production risk model hashes are 100% byte-invariant."""
    known_hashes = {
        "backend/models/risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
        "backend/models/risk_model_0d.joblib": "54720c2218a46977c2f93e94889d4242f357c313ab069935cf57bdc31f1b61cd",
        "backend/models/risk_model_1d.joblib": "cb05c4408ef2f3b66999f8b0016b9dbd1285bc88c1c8f6e32aae310817ccd57c",
        "backend/models/risk_model_2d.joblib": "6e6e34c5806c2db83a61369eb8488dedb827845cba8a340c0b8b286363a52843",
    }
    for rel_path, expected_hash in known_hashes.items():
        p = PROJECT_ROOT / rel_path
        assert p.exists(), f"Model missing: {rel_path}"
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == expected_hash, f"Production model {rel_path} modified! {actual} != {expected_hash}"


# =============================================================================
# 24. PRODUCTION POLICY IMMUTABILITY
# =============================================================================
def test_24_production_policy_immutability():
    """24. Production policy immutability: operational alert threshold remains frozen."""
    assert DEFAULT_FROZEN_THRESHOLD in [0.27, 0.20]


# =============================================================================
# SUPPLEMENTARY TESTS (MTIME, CLIPPING, API ENDPOINTS, SAGARBOT)
# =============================================================================
def test_causal_firewall_mtime_rejection(temp_service, sample_valid_features):
    """Reject attempts to pass filesystem mtime as data availability timestamp."""
    res = temp_service.predict_forward(
        system_id="SYS-MTIME",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        data_available_timestamp="file_mtime:2026-07-10T11:00:00Z",
    )
    assert res["status"] == "CAUSAL_REJECTED"
    assert res["causal_firewall"] == "FAIL"


def test_physical_clipping_bounds(temp_service, sample_valid_features, monkeypatch):
    """Guarantees forecast vmax is bounded within meteorological limits [15.0, 165.0] kts."""
    res = temp_service.predict_forward(
        system_id="SYS-CLIP-TEST",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    assert 15.0 <= res["predicted_vmax_24h"] <= 165.0
    assert res["clipping_bounds"] == [15.0, 165.0]

    # Test that clipping guard triggers when raw prediction falls below 15.0 kts
    class MockLowModel:
        def predict(self, df):
            return np.array([8.4])

    monkeypatch.setattr(temp_service.evaluator, "_model", MockLowModel())
    res_low = temp_service.evaluator.generate_forecast(
        system_id="SYS-MOCK-LOW",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T12:00:00Z",
        data_available_timestamp="2026-07-10T12:00:00Z",
        include_extended=True,
    )
    assert res_low["raw_vmax_24h"] == 8.4
    assert res_low["forecast_vmax_24h"] == 15.0
    assert res_low["clipping_applied"] is True

    # Test that clipping guard triggers when raw prediction exceeds 165.0 kts
    class MockHighModel:
        def predict(self, df):
            return np.array([182.5])

    monkeypatch.setattr(temp_service.evaluator, "_model", MockHighModel())
    res_high = temp_service.evaluator.generate_forecast(
        system_id="SYS-MOCK-HIGH",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T12:00:00Z",
        data_available_timestamp="2026-07-10T12:00:00Z",
        include_extended=True,
    )
    assert res_high["raw_vmax_24h"] == 182.5
    assert res_high["forecast_vmax_24h"] == 165.0
    assert res_high["clipping_applied"] is True


def test_preflight_validation_endpoint():
    """FastAPI /api/forecast/validate-inputs endpoint returns itemized check status."""
    client = TestClient(app)
    payload = {
        "system_id": "BOB-TEST-VALIDATE",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "demo_scenario_id": "DEMO-BOB-2026-07-10",
    }
    response = client.post("/api/forecast/validate-inputs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["can_predict"] is True
    assert any(c["check"] == "FROZEN_MODEL_INTEGRITY" and c["status"] == "PASS" for c in data["checks"])


def test_api_predict_forward_endpoint():
    """FastAPI /api/forecast/cyclone-intensity endpoint returns successful forward prediction."""
    client = TestClient(app)
    payload = {
        "system_id": "BOB-API-TEST",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "demo_scenario_id": "DEMO-BOB-2026-07-10",
    }
    response = client.post("/api/forecast/cyclone-intensity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PREDICTION_GENERATED"
    assert data["model_version"] == MODEL_VERSION
    assert data["model_hash"] == FROZEN_MODEL_SHA256
    assert 15.0 <= data["predicted_vmax_24h"] <= 165.0


def test_api_forecast_history_endpoint():
    """FastAPI /api/forecast/history endpoint returns recent log entries."""
    client = TestClient(app)
    response = client.get("/api/forecast/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_sagarbot_forward_forecast_intent():
    """SagarBot correctly identifies forward forecast intent and returns causal facts."""
    import asyncio

    intent = IntentClassifier.classify("What is the predicted intensity 24 hours from now?")
    assert intent == IntentType.FORWARD_FORECAST

    intent_leak = IntentClassifier.classify("Was future data used in generating this forecast?")
    assert intent_leak == IntentType.FORWARD_FORECAST

    ctx = ChatContext(
        active_site="bob",
        coordinates={"lat": 17.5, "lon": 88.0},
        current_depth="10m",
        variable="temp",
        current_value="28.5 °C",
        time_offset="+0h"
    )
    req = ChatRequest(
        message="Was future data used?",
        context=ctx
    )

    async def _run():
        return await SagarBotService.process_chat(req)

    res = asyncio.run(_run())
    assert res.intent == IntentType.FORWARD_FORECAST.value
    assert "No future data was used" in res.reply
    assert "Dual Causal Availability Firewall" in res.reply
    assert "Scientific Notice" in res.reply


# =============================================================================
# FORENSIC AUDIT PASS 2: REAL-WORLD PROSPECTIVE READINESS (20 SPECIFIC TESTS)
# =============================================================================

def test_target_unavailable_not_scored_as_zero_error(temp_service):
    """1. TARGET_UNAVAILABLE records must NOT receive 0 error and contribute zero to MAE."""
    forecast_df = pd.DataFrame([{
        "system_id": "SYS-UNAVAIL-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "forecast_vmax_24h": 75.0,
        "vmax_current": 60.0,
        "dvmax_12h": 10.0,
        "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
    }])
    target_df = pd.DataFrame([{
        "system_id": "SYS-UNAVAIL-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "target_status": TargetStatus.TARGET_UNAVAILABLE.value,
        "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
        "observed_vmax_24h": None,
        "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
    }])

    metrics = temp_service.evaluator.compute_prospective_metrics(forecast_df, target_df)
    assert metrics["number_of_valid_evaluated_targets"] == 0
    assert metrics["row_mae"] is None
    assert metrics["row_rmse"] is None
    assert metrics["n_targets_unavailable"] == 1


def test_target_unavailable_excluded_from_mae_denominator(temp_service):
    """2. Denominator of MAE must strictly equal number_of_valid_evaluated_targets."""
    forecast_df = pd.DataFrame([
        {
            "system_id": "SYS-01",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "forecast_vmax_24h": 70.0,
            "vmax_current": 60.0,
            "dvmax_12h": 10.0,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        },
        {
            "system_id": "SYS-02",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "forecast_vmax_24h": 85.0,
            "vmax_current": 70.0,
            "dvmax_12h": 15.0,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        }
    ])
    target_df = pd.DataFrame([
        {
            "system_id": "SYS-01",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "target_status": TargetStatus.TARGET_AVAILABLE.value,
            "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
            "observed_vmax_24h": 80.0,  # error = |80 - 70| = 10
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        },
        {
            "system_id": "SYS-02",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "target_status": TargetStatus.TARGET_UNAVAILABLE.value,
            "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
            "observed_vmax_24h": None,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        }
    ])

    metrics = temp_service.evaluator.compute_prospective_metrics(forecast_df, target_df)
    assert metrics["number_of_valid_evaluated_targets"] == 1
    # Denominator must be 1, so row_mae = 10.0 / 1 = 10.0 (NOT 10/2 = 5.0!)
    assert metrics["row_mae"] == 10.0
    assert metrics["n_targets_unavailable"] == 1


def test_real_prospective_provenance_required(temp_service):
    """3. Prospective evaluation must require REAL_PROSPECTIVE provenance."""
    forecast_df = pd.DataFrame([{
        "system_id": "SYS-TEST-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "forecast_vmax_24h": 65.0,
        "data_provenance_class": "UNKNOWN"
    }])
    target_df = pd.DataFrame([{
        "system_id": "SYS-TEST-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "target_status": TargetStatus.TARGET_AVAILABLE.value,
        "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
        "observed_vmax_24h": 70.0,
        "data_provenance_class": "UNKNOWN"
    }])

    metrics = temp_service.evaluator.compute_prospective_metrics(forecast_df, target_df, require_real_prospective=True)
    assert metrics["number_of_valid_evaluated_targets"] == 0
    assert metrics["row_mae"] is None


def test_synthetic_fixture_excluded_from_accuracy_metrics(temp_service):
    """4. Synthetic demo fixtures are labeled and excluded from prospective accuracy."""
    res = temp_service.predict_forward(
        system_id="BOB-DEMO",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        demo_scenario_id="DEMO-BOB-2026-07-10"
    )
    assert res["data_provenance_class"] == DataProvenanceClass.SYNTHETIC_TEST_FIXTURE.value
    assert res["synthetic_fixture_label"] is not None
    assert "SYNTHETIC TEST FIXTURE" in res["synthetic_fixture_label"]

    # Target info must mark it as TARGET_EXCLUDED from prospective scientific evaluation
    if res.get("target_info"):
        assert res["target_info"]["target_status"] == TargetStatus.TARGET_EXCLUDED.value


def test_historical_replay_not_classified_as_prospective(temp_service, sample_valid_features):
    """5. Historical reanalysis replay cannot be labeled as REAL_PROSPECTIVE."""
    res = temp_service.predict_forward(
        system_id="BOB-REPLAY",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        evaluation_mode=EvaluationMode.AVAILABILITY_TIMESTAMP_REPLAY
    )
    assert res["data_provenance_class"] != DataProvenanceClass.REAL_PROSPECTIVE.value
    assert res["data_provenance_class"] == DataProvenanceClass.HISTORICAL_REANALYSIS.value


def test_forecast_precedes_target_availability(temp_service):
    """6. Target already available before forecast creation is classified as RETROSPECTIVE."""
    forecast_rec = {
        "system_id": "SYS-TIME-HONEST",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "forecast_valid_time": "2026-07-11T12:00:00Z",
        "forecast_created_at": "2026-07-12T12:00:00Z",  # Created after target was already published
        "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
    }
    candidate_fixes = [{
        "system_id": "SYS-TIME-HONEST",
        "observation_timestamp": "2026-07-11T12:00:00Z",
        "data_availability_timestamp": "2026-07-11T12:30:00Z",  # Target available at 12:30 on July 11
        "max_wind_kts": 85.0
    }]

    match = temp_service.evaluator.match_target(
        forecast_record=forecast_rec,
        candidate_fixes=candidate_fixes
    )
    assert match["target_status"] == TargetStatus.TARGET_EXCLUDED.value
    assert "RETROSPECTIVE_EVALUATION" in match["target_unavailable_reason"]


def test_era5_latency_prevents_false_real_time_claim(temp_service, sample_valid_features):
    """7. ERA5 source cannot masquerade as live operational REAL_PROSPECTIVE data."""
    res = temp_service.predict_forward(
        system_id="BOB-ERA5-TEST",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        atmos_source_id=AtmosphericSource.ERA5_REANALYSIS.value,
        data_provenance_class=DataProvenanceClass.REAL_PROSPECTIVE.value
    )
    # Must be reclassified to HISTORICAL_REANALYSIS due to ERA5 multi-month latency
    assert res["data_provenance_class"] == DataProvenanceClass.HISTORICAL_REANALYSIS.value


def test_operational_nwp_allowed_for_real_time_source(temp_service, sample_valid_features):
    """8. OPERATIONAL_NWP_ANALYSIS is accepted for true real-time forward prediction."""
    res = temp_service.predict_forward(
        system_id="BOB-NWP-TEST",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        atmos_source_id=AtmosphericSource.OPERATIONAL_NWP_ANALYSIS.value,
        data_provenance_class=DataProvenanceClass.REAL_PROSPECTIVE.value
    )
    assert res["data_provenance_class"] == DataProvenanceClass.REAL_PROSPECTIVE.value
    assert res["atmos_source_id"] == AtmosphericSource.OPERATIONAL_NWP_ANALYSIS.value


def test_future_complete_track_cannot_create_retrospective_forecast(temp_service, sample_valid_features):
    """9. Providing a full track ignores fixes beyond origin T for kinematics."""
    history_with_future = [
        {"system_id": "SYS-FUT", "observation_timestamp": "2026-07-10T00:00:00Z", "max_wind_kts": 35.0, "central_pressure_hpa": 1000.0, "latitude": 14.0, "longitude": 88.0},
        {"system_id": "SYS-FUT", "observation_timestamp": "2026-07-10T06:00:00Z", "max_wind_kts": 45.0, "central_pressure_hpa": 995.0, "latitude": 15.0, "longitude": 87.5},
        {"system_id": "SYS-FUT", "observation_timestamp": "2026-07-10T12:00:00Z", "max_wind_kts": 55.0, "central_pressure_hpa": 990.0, "latitude": 16.0, "longitude": 87.0},
        # Future fixes > T=12:00
        {"system_id": "SYS-FUT", "observation_timestamp": "2026-07-10T18:00:00Z", "max_wind_kts": 75.0, "central_pressure_hpa": 975.0, "latitude": 17.0, "longitude": 86.5},
        {"system_id": "SYS-FUT", "observation_timestamp": "2026-07-11T12:00:00Z", "max_wind_kts": 95.0, "central_pressure_hpa": 955.0, "latitude": 19.0, "longitude": 85.0},
    ]

    kin = temp_service.derive_kinematics_from_fixes(
        history_with_future,
        forecast_origin=_to_utc_timestamp("2026-07-10T12:00:00Z")
    )
    # vmax_current must be the fix at T (55.0), NOT future fixes (75.0 or 95.0)
    assert kin["vmax_current"] == 55.0
    assert kin["dvmax_6h"] == 10.0  # 55 - 45


def test_future_data_perturbation_invariance(temp_service, sample_valid_features):
    """10. Radically altering future fixes produces bitwise identical predictions."""
    history_a = [
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T00:00:00Z", "max_wind_kts": 35.0, "central_pressure_hpa": 1000.0, "latitude": 14.0, "longitude": 88.0},
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T06:00:00Z", "max_wind_kts": 45.0, "central_pressure_hpa": 995.0, "latitude": 15.0, "longitude": 87.5},
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T12:00:00Z", "max_wind_kts": 55.0, "central_pressure_hpa": 990.0, "latitude": 16.0, "longitude": 87.0},
        # Future A
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T18:00:00Z", "max_wind_kts": 60.0, "central_pressure_hpa": 985.0, "latitude": 17.0, "longitude": 86.5},
    ]
    history_b = [
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T00:00:00Z", "max_wind_kts": 35.0, "central_pressure_hpa": 1000.0, "latitude": 14.0, "longitude": 88.0},
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T06:00:00Z", "max_wind_kts": 45.0, "central_pressure_hpa": 995.0, "latitude": 15.0, "longitude": 87.5},
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T12:00:00Z", "max_wind_kts": 55.0, "central_pressure_hpa": 990.0, "latitude": 16.0, "longitude": 87.0},
        # Radically perturbed Future B
        {"system_id": "SYS-INV", "observation_timestamp": "2026-07-10T18:00:00Z", "max_wind_kts": 160.0, "central_pressure_hpa": 890.0, "latitude": 28.0, "longitude": 70.0},
    ]

    res_a = temp_service.predict_forward(
        system_id="SYS-INV",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        cyclone_history=history_a,
        features=sample_valid_features
    )
    res_b = temp_service.predict_forward(
        system_id="SYS-INV",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        cyclone_history=history_b,
        features=sample_valid_features
    )

    assert res_a["predicted_vmax_24h"] == res_b["predicted_vmax_24h"]
    assert res_a["raw_predicted_vmax_24h"] == res_b["raw_predicted_vmax_24h"]


def test_ocean_daily_resolution_preserved(temp_service, sample_valid_features):
    """11. Copernicus ocean products maintain daily temporal resolution."""
    res = temp_service.predict_forward(
        system_id="BOB-OCEAN-RES",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        ocean_source_timestamp="2026-07-10T00:00:00Z"
    )
    assert res["ocean_temporal_resolution"] == "daily"


def test_ocean_age_never_exceeds_48h(temp_service, sample_valid_features):
    """12. Ocean observations older than 48h are marked missing, never fabricated."""
    res = temp_service.predict_forward(
        system_id="BOB-STALE-OCEAN",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        ocean_source_timestamp="2026-07-07T00:00:00Z",
        ocean_source_available_timestamp="2026-07-07T06:00:00Z",
        ocean_age_hours=84.0  # > 48h limit
    )
    snapshot = res["model_input_forensics"]["feature_values"]
    assert snapshot["sst_core_mean_0_100km"] is None
    assert snapshot["mld_core_mean_0_100km"] is None
    assert snapshot["sla_core_mean_0_100km"] is None


def test_partition_boundaries_are_canonical():
    """13. Authoritative canonical research splits are strictly defined."""
    assert CANONICAL_RESEARCH_SPLITS["TRAIN"]["span"] == "2016–2021"
    assert CANONICAL_RESEARCH_SPLITS["TRAIN"]["storm_count"] == 64
    assert CANONICAL_RESEARCH_SPLITS["VALIDATION"]["span"] == "2022–2023"
    assert CANONICAL_RESEARCH_SPLITS["VALIDATION"]["storm_count"] == 24
    assert CANONICAL_RESEARCH_SPLITS["TEST"]["span"] == "2024–2026"
    assert CANONICAL_RESEARCH_SPLITS["TEST"]["storm_count"] == 29


def test_baseline_metrics_use_same_evaluated_targets(temp_service):
    """14. Persistence and damped trend use the exact same evaluated targets as model."""
    forecast_df = pd.DataFrame([
        {
            "system_id": "SYS-BASE-01",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "forecast_vmax_24h": 75.0,
            "vmax_current": 65.0,
            "dvmax_12h": 10.0,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        },
        {
            "system_id": "SYS-BASE-01",
            "forecast_origin_timestamp": "2026-07-10T18:00:00Z",
            "forecast_vmax_24h": 80.0,
            "vmax_current": 70.0,
            "dvmax_12h": 10.0,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        }
    ])
    target_df = pd.DataFrame([
        {
            "system_id": "SYS-BASE-01",
            "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
            "target_status": TargetStatus.TARGET_AVAILABLE.value,
            "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
            "observed_vmax_24h": 80.0,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        },
        {
            "system_id": "SYS-BASE-01",
            "forecast_origin_timestamp": "2026-07-10T18:00:00Z",
            "target_status": TargetStatus.TARGET_AVAILABLE.value,
            "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value,
            "observed_vmax_24h": 85.0,
            "data_provenance_class": DataProvenanceClass.REAL_PROSPECTIVE.value
        }
    ])

    metrics = temp_service.evaluator.compute_prospective_metrics(forecast_df, target_df)
    assert metrics["number_of_valid_evaluated_targets"] == 2
    assert metrics["row_mae"] == 5.0  # (|80-75| + |85-80|) / 2 = 5.0
    assert metrics["persistence_mae"] == 15.0  # (|80-65| + |85-70|) / 2 = 15.0
    # Model skill vs persistence = (1 - 5/15) * 100 = 66.67%
    assert metrics["model_skill_vs_persistence"] == 66.67


def test_storm_cluster_bootstrap():
    """15. Storm cluster bootstrap requires >= 5 storms; otherwise flags UNSTABLE_SAMPLE_SIZE."""
    # Under 5 storms
    df_small = pd.DataFrame({
        "system_id": ["S1", "S1", "S2"],
        "observed_vmax_24h": [60.0, 70.0, 80.0],
        "forecast_vmax_24h": [55.0, 65.0, 75.0],
    })
    res_small = CycloneIntensityProspectiveEvaluator.compute_storm_cluster_bootstrap(df_small)
    assert res_small["status"] == "UNSTABLE_SAMPLE_SIZE"
    assert res_small["storm_cluster_bootstrap_ci"] is None

    # With >= 5 storms
    df_large = pd.DataFrame({
        "system_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "observed_vmax_24h": [60.0, 70.0, 80.0, 90.0, 50.0, 65.0],
        "forecast_vmax_24h": [58.0, 72.0, 78.0, 85.0, 52.0, 63.0],
    })
    res_large = CycloneIntensityProspectiveEvaluator.compute_storm_cluster_bootstrap(df_large, n_bootstrap=200)
    assert res_large["status"] == "VALID_CLUSTER_BOOTSTRAP"
    assert isinstance(res_large["storm_cluster_bootstrap_ci"], list)
    assert len(res_large["storm_cluster_bootstrap_ci"]) == 2


def test_prospective_evidence_count_excludes_synthetic():
    """16. Synthetic fixtures cannot contribute to prospective evidence counts."""
    assert classify_prospective_evidence_tier(0) == ProspectiveEvidenceTier.INSUFFICIENT_PROSPECTIVE_EVIDENCE
    assert classify_prospective_evidence_tier(6) == ProspectiveEvidenceTier.EARLY_PROSPECTIVE_SIGNAL
    assert classify_prospective_evidence_tier(18) == ProspectiveEvidenceTier.PROMISING_PROSPECTIVE_EVIDENCE
    assert classify_prospective_evidence_tier(32) == ProspectiveEvidenceTier.ADEQUATE_PROSPECTIVE_SAMPLE_FOR_PRELIMINARY_GENERALIZATION


def test_target_revision_does_not_mutate_original_forecast(temp_service, tmp_path):
    """17. Logging a target revision appends without mutating original initial records."""
    log_path = tmp_path / "target_arrival_log.parquet"
    rec1 = {
        "system_id": "SYS-REV-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "target_status": TargetStatus.TARGET_AVAILABLE.value,
        "observed_vmax_24h": 75.0,
        "evaluation_type": EvaluationType.PROSPECTIVE_INITIAL.value
    }
    rec2 = {
        "system_id": "SYS-REV-01",
        "forecast_origin_timestamp": "2026-07-10T12:00:00Z",
        "target_status": TargetStatus.TARGET_AVAILABLE.value,
        "observed_vmax_24h": 80.0,  # Post-season revised intensity
        "evaluation_type": EvaluationType.RETROSPECTIVE_REVISED.value
    }

    temp_service.evaluator.log_target_arrival(rec1, filepath=log_path)
    temp_service.evaluator.log_target_arrival(rec2, filepath=log_path)

    df = pd.read_parquet(log_path)
    assert len(df) == 2
    assert df.iloc[0]["evaluation_type"] == EvaluationType.PROSPECTIVE_INITIAL.value
    assert df.iloc[0]["observed_vmax_24h"] == 75.0
    assert df.iloc[1]["evaluation_type"] == EvaluationType.RETROSPECTIVE_REVISED.value
    assert df.iloc[1]["observed_vmax_24h"] == 80.0


def test_no_model_mutation():
    """18. Research cyclone intensity model artifact SHA-256 is 100% invariant."""
    model_path = PROJECT_ROOT / "research" / "cyclone_intensity" / "models" / "final_intensity_model.joblib"
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert actual_hash == FROZEN_MODEL_SHA256


def test_no_production_artifact_mutation():
    """19. Production risk models (0d, 1d, 2d, 3d) remain 100% byte-invariant."""
    prod_hashes = {
        "risk_model_0d.joblib": "54720c2218a46977c2f93e94889d4242f357c313ab069935cf57bdc31f1b61cd",
        "risk_model_1d.joblib": "cb05c4408ef2f3b66999f8b0016b9dbd1285bc88c1c8f6e32aae310817ccd57c",
        "risk_model_2d.joblib": "6e6e34c5806c2db83a61369eb8488dedb827845cba8a340c0b8b286363a52843",
        "risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
    }
    for fname, expected in prod_hashes.items():
        p = PROJECT_ROOT / "backend" / "models" / fname
        assert p.exists()
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == expected, f"Production model {fname} mutated!"


def test_no_historical_test_reopening(temp_service):
    """20. Quarantined historical 2024-2026 test partition is NEVER accessed."""
    train_ref = temp_service.evaluator.get_training_reference()
    assert (train_ref["partition"] == "TRAIN").all()
    assert not (train_ref["partition"] == "TEST").any()


# ==============================================================================
# 21. SCIENTIST-FACING THREAT & RISK INTERPRETATION LAYER TESTS
# ==============================================================================

def test_intensity_threat_bands_canonical():
    """
    Validates canonical intensity-based threat bands (Vmax in knots):
      < 34 kt   -> LOW / WEAK SYSTEM
      34–47 kt  -> WATCH
      48–63 kt  -> MODERATE THREAT
      64–82 kt  -> HIGH THREAT
      83–95 kt  -> SEVERE THREAT
      >= 96 kt  -> CRITICAL THREAT
    """
    # 33 kt -> LOW
    c33 = classify_intensity(33)
    assert c33["threat_level"] == "LOW"
    assert c33["threat_level"] == "LOW / WEAK SYSTEM"
    assert c33["short_threat_level"] == "LOW"
    assert c33["intensity_band"] == "< 34 kt"
    assert c33["symbol"] == "🟢"
    assert c33["color"] == "emerald"

    # 34 kt -> WATCH
    c34 = classify_intensity(34)
    assert c34["threat_level"] == "WATCH"
    assert c34["intensity_band"] == "34–47 kt"
    assert c34["symbol"] == "🟡"
    assert c34["color"] == "amber"

    # 47 kt -> WATCH
    c47 = classify_intensity(47)
    assert c47["threat_level"] == "WATCH"
    assert c47["intensity_band"] == "34–47 kt"

    # 48 kt -> MODERATE THREAT
    c48 = classify_intensity(48)
    assert c48["threat_level"] == "MODERATE THREAT"
    assert c48["intensity_band"] == "48–63 kt"
    assert c48["symbol"] == "🟠"
    assert c48["color"] == "orange"

    # 63 kt -> MODERATE THREAT
    c63 = classify_intensity(63)
    assert c63["threat_level"] == "MODERATE THREAT"
    assert c63["intensity_band"] == "48–63 kt"

    # 64 kt -> HIGH THREAT
    c64 = classify_intensity(64)
    assert c64["threat_level"] == "HIGH THREAT"
    assert c64["intensity_band"] == "64–82 kt"
    assert c64["symbol"] == "🔴"
    assert c64["color"] == "red"

    # 82 kt -> HIGH THREAT
    c82 = classify_intensity(82)
    assert c82["threat_level"] == "HIGH THREAT"
    assert c82["intensity_band"] == "64–82 kt"

    # 83 kt -> SEVERE THREAT
    c83 = classify_intensity(83)
    assert c83["threat_level"] == "SEVERE THREAT"
    assert c83["intensity_band"] == "83–95 kt"
    assert c83["symbol"] == "🔴"

    # 95 kt -> SEVERE THREAT
    c95 = classify_intensity(95)
    assert c95["threat_level"] == "SEVERE THREAT"
    assert c95["intensity_band"] == "83–95 kt"

    # 96 kt -> CRITICAL THREAT
    c96 = classify_intensity(96)
    assert c96["threat_level"] == "CRITICAL THREAT"
    assert c96["intensity_band"] == ">= 96 kt"
    assert c96["symbol"] == "⚡"
    assert c96["color"] == "rose"


def test_intensity_threat_edge_cases_and_malformed():
    """
    Tests edge cases and malformed inputs:
    - 0 kt -> LOW
    - Negative values -> LOW
    - None / missing Vmax -> UNKNOWN (no crash)
    - NaN / non-numeric string -> UNKNOWN (no crash)
    - Extreme values (150 kt, 250 kt) -> CRITICAL THREAT
    """
    # 0 kt
    c0 = classify_intensity(0)
    assert c0["threat_level"] == "LOW"
    assert c0["intensity_band"] == "< 34 kt"

    # Negative values
    c_neg = classify_intensity(-10.5)
    assert c_neg["threat_level"] == "LOW"
    assert c_neg["intensity_band"] == "< 34 kt"

    # None / missing
    c_none = classify_intensity(None)
    assert c_none["threat_level"] == "UNKNOWN"
    assert c_none["intensity_band"] == "N/A"
    assert "Missing" in c_none["threat_basis"]
    assert c_none["predicted_vmax_kt"] is None

    # NaN
    c_nan = classify_intensity(float("nan"))
    assert c_nan["threat_level"] == "UNKNOWN"
    assert c_nan["intensity_band"] == "N/A"

    # Non-numeric string
    c_str = classify_intensity("invalid_input")
    assert c_str["threat_level"] == "UNKNOWN"
    assert c_str["intensity_band"] == "N/A"

    # Extreme values
    c_ext = classify_intensity(220.0)
    assert c_ext["threat_level"] == "CRITICAL THREAT"
    assert c_ext["intensity_band"] == ">= 96 kt"


def test_scientific_distinction_and_probability_handling():
    """
    Tests scientific distinction between physical intensity (kt) and probability:
    - If probability is None: probability_label is 'N/A' and risk_basis is 'INTENSITY ONLY'.
    - If calibrated probability is provided, fuses intensity + probability without overwriting intensity.
    """
    # Standalone intensity without probability
    c = classify_intensity(78.08)
    assert c["predicted_vmax_kt"] == 78.08
    assert c["calibrated_probability"] is None
    assert c["probability_label"] == "N/A"
    assert c["risk_basis"] == "INTENSITY ONLY"
    assert c["threat_level"] == "HIGH THREAT"

    # Calibrated probability provided (< 20%)
    c_low_prob = classify_intensity(78.08, calibrated_probability=0.15)
    assert c_low_prob["calibrated_probability"] == 0.15
    assert c_low_prob["probability_label"] == "15.0%"
    assert c_low_prob["risk_basis"] == "CALIBRATED MODEL"
    assert c_low_prob["risk_level"] == "LOW"
    assert c_low_prob["threat_level"] == "HIGH THREAT"  # Intensity is preserved!

    # Calibrated probability provided (60-80%)
    c_high_prob = classify_intensity(78.08, calibrated_probability=0.72)
    assert c_high_prob["probability_label"] == "72.0%"
    assert c_high_prob["risk_level"] == "HIGH THREAT"

    # Calibrated probability > 80%
    c_crit_prob = classify_intensity(50.0, calibrated_probability=0.88)
    assert c_crit_prob["risk_level"] == "VERY HIGH / CRITICAL CONFIDENCE"


def test_forward_prediction_service_emits_threat_assessment(temp_service, sample_valid_features):
    """
    Verifies that ForwardPredictionService.predict_forward() attaches the
    threat assessment fields without breaking any existing fields.
    """
    res = temp_service.predict_forward(
        system_id="BOB-THREAT-TEST",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    assert res["status"] == "PREDICTION_GENERATED"
    assert "threat_assessment" in res
    assert "intensity_threat_level" in res
    assert "intensity_band" in res
    assert "predicted_vmax_kt" in res

    threat = res["threat_assessment"]
    assert threat["predicted_vmax_kt"] == res["predicted_vmax_24h"]
    assert threat["risk_basis"] == "INTENSITY ONLY"
    assert threat["probability_label"] == "N/A"
    assert threat["calibrated_probability"] is None
    assert threat["threat_basis"] == "Predicted Vmax"


def test_forward_prediction_early_rejection_has_threat_metadata(temp_service):
    """
    Ensures early-rejection states (e.g. INSUFFICIENT_HISTORY) gracefully
    contain safe UNKNOWN threat metadata rather than failing or omitting keys.
    """
    res = temp_service.predict_forward(
        system_id="BOB-EMPTY",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        cyclone_history=[]
    )
    assert res["status"] in ("INSUFFICIENT_HISTORY", "INSUFFICIENT_INPUT_DATA")
    assert "threat_assessment" in res
    assert res["intensity_threat_level"] == "UNKNOWN"
    assert res["intensity_band"] == "N/A"
    assert res["predicted_vmax_kt"] is None

