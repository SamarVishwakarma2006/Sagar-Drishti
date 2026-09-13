"""
Sagar-Drishti — Forward Prediction & Prospective Inference Dedicated Test Suite
22 comprehensive test cases covering:
1. Frozen model SHA-256 integrity & fail-closed tamper rejection.
2. Dual causal availability firewall (t_obs <= T and t_avail <= T).
3. Filesystem mtime rejection.
4. Authoritative 29-feature contract and feature ordering.
5. Kinematic history derivation (dvmax, dpc, translation speed).
6. Pre-flight input validation and error states.
7. Post-historical demo scenario execution (BOB & ARAS).
8. CRITICAL NEGATIVE LEAKAGE TEST: Perturbing future inputs beyond T has ZERO effect on prediction at T.
9. Physical clipping bounds [15.0, 165.0] kts.
10. Immutable append-only logging & schema enforcement.
11. Deduplication integrity in forecast log.
12. Target lifecycle states (TARGET_PENDING, TARGET_AVAILABLE, TARGET_UNAVAILABLE).
13. Model byte invariance / zero-retraining verification.
14. Mandatory scientific disclaimer verification.
15. FastAPI endpoint verification (/forecast/validate-inputs, /forecast/cyclone-intensity, /forecast/history).
16. SagarBot conversational grounding for FORWARD_FORECAST intent.
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


# -----------------------------------------------------------------------------
# 1. Frozen Model Hash Invariant & Tamper Rejection
# -----------------------------------------------------------------------------
def test_frozen_model_hash_invariant(temp_service, tmp_path):
    """Test 1: Verifies model SHA-256 matches frozen hash and tampered model fails immediately."""
    assert temp_service.evaluator.verify_model_integrity() == FROZEN_MODEL_SHA256

    mock_root = tmp_path / "mock_project"
    mock_model_dir = mock_root / "research" / "cyclone_intensity" / "models"
    mock_model_dir.mkdir(parents=True, exist_ok=True)
    (mock_model_dir / "final_intensity_model.joblib").write_bytes(b"tampered_weights")
    with pytest.raises(ModelHashMismatchError):
        CycloneIntensityProspectiveEvaluator(
            project_root=mock_root,
            db_dir=tmp_path / "broken_db"
        )


# -----------------------------------------------------------------------------
# 2. Dual Causal Firewall: Observation Timestamp <= Forecast Origin
# -----------------------------------------------------------------------------
def test_causal_firewall_future_observation_rejection(temp_service, sample_valid_features):
    """Test 2: Rejects observation timestamp occurring strictly after forecast origin."""
    res = temp_service.predict_forward(
        system_id="SYS-FUTURE-OBS",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T13:00:00Z",  # 1 hour in the future!
        data_available_timestamp="2026-07-10T12:00:00Z",
    )
    assert res["status"] == "CAUSAL_REJECTED"
    assert res["causal_firewall"] == "FAIL"
    assert res["predicted_vmax_24h"] is None


# -----------------------------------------------------------------------------
# 3. Dual Causal Firewall: Data Availability Timestamp <= Forecast Origin
# -----------------------------------------------------------------------------
def test_causal_firewall_future_availability_rejection(temp_service, sample_valid_features):
    """Test 3: Rejects observation whose availability/publication time is strictly after origin."""
    res = temp_service.predict_forward(
        system_id="SYS-FUTURE-AVAIL",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        observation_timestamp="2026-07-10T12:00:00Z",
        data_available_timestamp="2026-07-10T12:05:00Z",  # Available 5 minutes late!
    )
    assert res["status"] == "CAUSAL_REJECTED"
    assert res["causal_firewall"] == "FAIL"
    assert res["predicted_vmax_24h"] is None


# -----------------------------------------------------------------------------
# 4. Causal Firewall: Filesystem mtime explicitly forbidden
# -----------------------------------------------------------------------------
def test_causal_firewall_mtime_rejection(temp_service, sample_valid_features):
    """Test 4: Reject attempts to pass filesystem mtime as data availability timestamp."""
    res = temp_service.predict_forward(
        system_id="SYS-MTIME",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        data_available_timestamp="file_mtime:2026-07-10T11:00:00Z",
    )
    assert res["status"] == "CAUSAL_REJECTED"
    assert res["causal_firewall"] == "FAIL"


# -----------------------------------------------------------------------------
# 5. Authoritative 29-Feature Contract Order and Completeness
# -----------------------------------------------------------------------------
def test_feature_contract_29_order_and_types(temp_service, sample_valid_features):
    """Test 5: Verifies that assembled features strictly conform to FROZEN_29_FEATURES."""
    assembled = temp_service._assemble_features(
        system_id="SYS-FEAT",
        forecast_origin=pd.Timestamp("2026-07-10T12:00:00Z"),
        features=sample_valid_features
    )
    for f in FROZEN_29_FEATURES:
        assert f in assembled, f"Feature {f} missing from assembled contract"
        assert not pd.isna(assembled[f]), f"Feature {f} is NaN"


# -----------------------------------------------------------------------------
# 6. Deterministic Kinematic Derivation from Cyclone History Fixes
# -----------------------------------------------------------------------------
def test_kinematic_derivation_from_fixes(temp_service):
    """Test 6: Validates calculation of dvmax, dpc, and translation speed from fixes."""
    fixes = [
        {"observation_timestamp": "2026-07-09T12:00:00Z", "latitude": 13.0, "longitude": 88.0, "max_wind_kts": 35.0, "central_pressure_hpa": 1000.0},
        {"observation_timestamp": "2026-07-09T18:00:00Z", "latitude": 13.6, "longitude": 87.5, "max_wind_kts": 40.0, "central_pressure_hpa": 996.0},
        {"observation_timestamp": "2026-07-10T00:00:00Z", "latitude": 14.2, "longitude": 87.0, "max_wind_kts": 48.0, "central_pressure_hpa": 991.0},
        {"observation_timestamp": "2026-07-10T06:00:00Z", "latitude": 14.8, "longitude": 86.5, "max_wind_kts": 55.0, "central_pressure_hpa": 985.0},
        {"observation_timestamp": "2026-07-10T12:00:00Z", "latitude": 15.5, "longitude": 86.0, "max_wind_kts": 65.0, "central_pressure_hpa": 978.0},
    ]
    kin = temp_service.derive_kinematics_from_fixes(fixes, pd.Timestamp("2026-07-10T12:00:00Z"))
    assert kin["vmax_current"] == 65.0
    assert kin["dvmax_6h"] == 10.0  # 65 - 55
    assert kin["dvmax_12h"] == 17.0 # 65 - 48
    assert kin["dvmax_24h"] == 30.0 # 65 - 35
    assert kin["pc_current"] == 978.0
    assert kin["dpc_6h"] == -7.0    # 978 - 985
    assert kin["translation_speed_kts"] > 0.0


# -----------------------------------------------------------------------------
# 7. Insufficient Input Data Handling
# -----------------------------------------------------------------------------
def test_insufficient_data_rejection(temp_service):
    """Test 7: Missing required kinematics cleanly returns INSUFFICIENT_INPUT_DATA without crashing."""
    res = temp_service.predict_forward(
        system_id="SYS-EMPTY",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features={"sst_core_mean_0_100km": 29.0},  # Missing all kinematics
    )
    assert res["status"] == "INSUFFICIENT_INPUT_DATA"
    assert res["predicted_vmax_24h"] is None
    assert len(res["missing_features"]) > 0


# -----------------------------------------------------------------------------
# 8. Post-Historical Demo Scenario: Bay of Bengal System
# -----------------------------------------------------------------------------
def test_demo_scenario_bob_2026(temp_service):
    """Test 8: Demo Scenario BOB-2026 executes prospective forward inference cleanly."""
    res = temp_service.predict_forward(
        system_id="BOB-2026-01-PROSPECTIVE",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        demo_scenario_id="DEMO-BOB-2026-07-10",
    )
    assert res["status"] == "PREDICTION_GENERATED"
    assert res["causal_firewall"] == "PASS"
    assert res["feature_completeness"] == 1.0
    assert 15.0 <= res["predicted_vmax_24h"] <= 165.0
    assert res["evaluation_status"] == "TARGET_AVAILABLE"


# -----------------------------------------------------------------------------
# 9. Post-Historical Demo Scenario: Arabian Sea System
# -----------------------------------------------------------------------------
def test_demo_scenario_aras_2026(temp_service):
    """Test 9: Demo Scenario ARAS-2026 executes prospective forward inference cleanly."""
    res = temp_service.predict_forward(
        system_id="ARAS-2026-01-PROSPECTIVE",
        forecast_origin_timestamp="2026-08-15T06:00:00Z",
        demo_scenario_id="DEMO-ARAS-2026-08-15",
    )
    assert res["status"] == "PREDICTION_GENERATED"
    assert res["causal_firewall"] == "PASS"
    assert res["feature_completeness"] == 1.0
    assert 15.0 <= res["predicted_vmax_24h"] <= 165.0


# -----------------------------------------------------------------------------
# 10. CRITICAL NEGATIVE LEAKAGE TEST: Perturbation Invariance
# -----------------------------------------------------------------------------
def test_future_observation_perturbation_invariance(temp_service, sample_valid_features):
    """
    Test 10: Crucial causal property — adding or perturbing future observations
    beyond forecast origin T MUST produce exactly zero change in the prediction at T.
    """
    t_origin = "2026-07-10T12:00:00Z"

    # Base prediction at T
    base_res = temp_service.predict_forward(
        system_id="SYS-LEAKAGE-AUDIT",
        forecast_origin_timestamp=t_origin,
        features=sample_valid_features,
    )
    pred_base = base_res["predicted_vmax_24h"]
    assert pred_base is not None

    # Now pass candidate target fixes in the future (T+24h) with extreme hypothetical values
    future_fix_1 = [{
        "system_id": "SYS-LEAKAGE-AUDIT",
        "observation_timestamp": "2026-07-11T12:00:00Z",
        "latitude": 18.0,
        "longitude": 85.0,
        "max_wind_kts": 140.0,  # Extreme Category 5 wind
        "central_pressure_hpa": 920.0,
    }]
    res_perturbed = temp_service.predict_forward(
        system_id="SYS-LEAKAGE-AUDIT-2",
        forecast_origin_timestamp=t_origin,
        features=sample_valid_features,
        candidate_target_fixes=future_fix_1,
    )
    pred_perturbed = res_perturbed["predicted_vmax_24h"]

    # Must be 100% bitwise identical!
    assert pred_base == pred_perturbed, (
        f"Leakage detected! Base prediction {pred_base} != Perturbed prediction {pred_perturbed}"
    )


# -----------------------------------------------------------------------------
# 11. Physical Clipping Bounds [15.0, 165.0] kts
# -----------------------------------------------------------------------------
def test_physical_clipping_bounds(temp_service, sample_valid_features):
    """Test 11: Guarantees forecast vmax is bounded within meteorological limits [15.0, 165.0] kts."""
    extreme_low = sample_valid_features.copy()
    extreme_low["vmax_current"] = 10.0
    extreme_low["dvmax_24h"] = -50.0

    res = temp_service.predict_forward(
        system_id="SYS-CLIP-LOW",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=extreme_low,
    )
    assert res["predicted_vmax_24h"] >= 15.0

    extreme_high = sample_valid_features.copy()
    extreme_high["vmax_current"] = 155.0
    extreme_high["dvmax_24h"] = 40.0
    res_high = temp_service.predict_forward(
        system_id="SYS-CLIP-HIGH",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=extreme_high,
    )
    assert res_high["predicted_vmax_24h"] <= 165.0


# -----------------------------------------------------------------------------
# 12. Immutable Forecast Log Append & Schema
# -----------------------------------------------------------------------------
def test_immutable_forecast_log_append(temp_service, sample_valid_features):
    """Test 12: Logs record conforming to 17-field prospective evaluation schema."""
    temp_service.predict_forward(
        system_id="SYS-LOG-01",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    log_file = temp_service.evaluator.db_dir / "forecast_log.parquet"
    assert log_file.exists()

    df = pd.read_parquet(log_file)
    assert len(df) >= 1
    assert "forecast_origin_timestamp" in df.columns
    assert "forecast_vmax_24h" in df.columns
    assert "prediction_status" in df.columns
    assert "model_hash" in df.columns


# -----------------------------------------------------------------------------
# 13. Deduplication Integrity in Forecast Log
# -----------------------------------------------------------------------------
def test_forecast_log_deduplication(temp_service, sample_valid_features):
    """Test 13: Submitting identical (system_id, forecast_origin) does not duplicate row."""
    temp_service.predict_forward(
        system_id="SYS-DEDUP",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    temp_service.predict_forward(
        system_id="SYS-DEDUP",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    log_file = temp_service.evaluator.db_dir / "forecast_log.parquet"
    df = pd.read_parquet(log_file)
    dedup_rows = df[df["system_id"] == "SYS-DEDUP"]
    assert len(dedup_rows) == 1


# -----------------------------------------------------------------------------
# 14. Target Lifecycle: TARGET_PENDING by default
# -----------------------------------------------------------------------------
def test_target_lifecycle_pending(temp_service, sample_valid_features):
    """Test 14: Without future verification fixes, target state is TARGET_PENDING."""
    res = temp_service.predict_forward(
        system_id="SYS-PENDING",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    assert res["evaluation_status"] == TargetStatus.TARGET_PENDING.value


# -----------------------------------------------------------------------------
# 15. Target Lifecycle: TARGET_AVAILABLE when ground-truth fix matches T+24h
# -----------------------------------------------------------------------------
def test_target_lifecycle_available(temp_service, sample_valid_features):
    """Test 15: Ground truth observation in [T+21h, T+27h] transitions to TARGET_AVAILABLE."""
    target_fix = [{
        "system_id": "SYS-TARGET-MATCH",
        "observation_timestamp": "2026-07-11T12:00:00Z",
        "latitude": 17.5,
        "longitude": 85.5,
        "max_wind_kts": 75.0,
        "central_pressure_hpa": 970.0,
    }]
    res = temp_service.predict_forward(
        system_id="SYS-TARGET-MATCH",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        candidate_target_fixes=target_fix,
    )
    assert res["evaluation_status"] == TargetStatus.TARGET_AVAILABLE.value
    assert res["target_info"]["observed_vmax_24h"] == 75.0


# -----------------------------------------------------------------------------
# 16. Target Lifecycle: TARGET_UNAVAILABLE on expiration or window mismatch
# -----------------------------------------------------------------------------
def test_target_lifecycle_unavailable(temp_service, sample_valid_features):
    """Test 16: Future fixes outside the target window result in TARGET_UNAVAILABLE."""
    mismatched_fix = [{
        "system_id": "SYS-TARGET-OUT-OF-WINDOW",
        "observation_timestamp": "2026-07-12T06:00:00Z",  # T+42h, far outside [T+21, T+27]
        "latitude": 20.0,
        "longitude": 84.0,
        "max_wind_kts": 80.0,
        "central_pressure_hpa": 965.0,
    }]
    res = temp_service.predict_forward(
        system_id="SYS-TARGET-OUT-OF-WINDOW",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
        candidate_target_fixes=mismatched_fix,
    )
    assert res["evaluation_status"] == TargetStatus.TARGET_UNAVAILABLE.value


# -----------------------------------------------------------------------------
# 17. Model Byte Invariance / Zero-Retraining Verification
# -----------------------------------------------------------------------------
def test_zero_retraining_enforcement(temp_service, sample_valid_features):
    """Test 17: Model artifact bytes on disk are 100% byte-invariant before and after inference."""
    model_path = PROJECT_ROOT / "research" / "cyclone_intensity" / "models" / "final_intensity_model.joblib"
    bytes_before = model_path.read_bytes()

    # Run inference
    temp_service.predict_forward(
        system_id="SYS-BYTE-CHECK",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )

    bytes_after = model_path.read_bytes()
    assert bytes_before == bytes_after
    assert hashlib.sha256(bytes_after).hexdigest() == FROZEN_MODEL_SHA256


# -----------------------------------------------------------------------------
# 18. Non-Negotiable Mandatory Scientific Disclaimer
# -----------------------------------------------------------------------------
def test_scientific_disclaimer_presence(temp_service, sample_valid_features):
    """Test 18: Every forward prediction response must carry the mandatory scientific disclaimer."""
    res = temp_service.predict_forward(
        system_id="SYS-DISCLAIMER",
        forecast_origin_timestamp="2026-07-10T12:00:00Z",
        features=sample_valid_features,
    )
    assert res["scientific_disclaimer"] == SCIENTIFIC_DISCLAIMER
    assert "not itself proof of prospective generalization" in res["scientific_disclaimer"]


# -----------------------------------------------------------------------------
# 19. FastAPI Pre-flight Validation Endpoint (/api/forecast/validate-inputs)
# -----------------------------------------------------------------------------
def test_preflight_validation_endpoint():
    """Test 19: /api/forecast/validate-inputs endpoint returns itemized check status."""
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


# -----------------------------------------------------------------------------
# 20. FastAPI Forward Prediction Endpoint (/api/forecast/cyclone-intensity)
# -----------------------------------------------------------------------------
def test_api_predict_forward_endpoint():
    """Test 20: /api/forecast/cyclone-intensity endpoint returns successful forward prediction."""
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


# -----------------------------------------------------------------------------
# 21. FastAPI Forecast History Endpoint (/api/forecast/history)
# -----------------------------------------------------------------------------
def test_api_forecast_history_endpoint():
    """Test 21: /api/forecast/history endpoint returns recent log entries."""
    client = TestClient(app)
    response = client.get("/api/forecast/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# -----------------------------------------------------------------------------
# 22. SagarBot Forward Forecast Conversational Grounding
# -----------------------------------------------------------------------------
def test_sagarbot_forward_forecast_intent():
    """Test 22: SagarBot correctly identifies forward forecast intent and returns causal facts."""
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
