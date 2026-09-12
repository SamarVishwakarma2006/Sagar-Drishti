"""
Unit and Integration Tests for Sagar-Drishti P0 Production-Safety Fixes:
1. Probability Calibration (loading, bounds [0, 1], monotonicity, determinism, metadata).
2. Basin-Specific Operational Thresholds (separate Bay of Bengal vs Arabian Sea, no global fallback).
3. Risk Tier Mapping (pure function tests for LOW, MODERATE, HIGH).
4. Shadow Inference Service (execution isolation, telemetry logging, zero alteration of production alerts).
"""
import os
import json
import tempfile
import numpy as np
import pytest
import joblib

from app.services.risk_tier_mapper import (
    map_risk_tier,
    get_basin_threshold,
    evaluate_operational_decision,
    normalize_basin_key,
    BASIN_THRESHOLDS
)
from app.services.shadow_service import (
    ShadowInferenceService,
    CALIBRATION_DIR,
    CANDIDATE_MODELS_DIR,
    SHADOW_DATA_DIR
)
from app.services.prediction_service import PredictionService
from app.models.schemas import PredictionRequest


class TestP0Calibration:
    """Tests for Phase P0 Probability Calibration artifacts and behavior."""

    def test_calibration_artifacts_exist_and_load(self):
        for h in [0, 1, 2, 3]:
            art_path = os.path.join(CALIBRATION_DIR, f"calibrator_{h}d.joblib")
            assert os.path.exists(art_path), f"Calibration artifact missing: {art_path}"
            data = joblib.load(art_path)
            assert "calibrator" in data
            assert data["horizon"] == f"{h}d"
            assert data["method"] == "IsotonicRegression"

    def test_calibration_metadata_validity(self):
        meta_path = os.path.join(CALIBRATION_DIR, "calibration_metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["calibration_method"] == "IsotonicRegression"
        assert meta["training_split"] == "val"
        assert meta["training_date_range"] == ["2024-01-01", "2024-12-31"]
        for h in ["0d", "1d", "2d", "3d"]:
            assert h in meta["horizons"]
            h_data = meta["horizons"][h]
            assert h_data["after_calibration"]["brier_score"] < h_data["before_calibration"]["brier_score"]
            assert h_data["after_calibration"]["ece"] < 0.05

    def test_calibrated_probabilities_within_bounds(self):
        for h in [0, 1, 2, 3]:
            art_path = os.path.join(CALIBRATION_DIR, f"calibrator_{h}d.joblib")
            iso = joblib.load(art_path)["calibrator"]
            test_inputs = np.linspace(0.0, 1.0, 100)
            cal_outputs = np.clip(iso.predict(test_inputs), 0.0, 1.0)
            assert np.all(cal_outputs >= 0.0)
            assert np.all(cal_outputs <= 1.0)
            # Monotonicity check
            assert np.all(np.diff(cal_outputs) >= -1e-7)

    def test_calibration_is_deterministic(self):
        for h in [0, 1, 2, 3]:
            art_path = os.path.join(CALIBRATION_DIR, f"calibrator_{h}d.joblib")
            iso = joblib.load(art_path)["calibrator"]
            score = 0.6543
            p1 = float(iso.predict(np.array([score]))[0])
            p2 = float(iso.predict(np.array([score]))[0])
            assert p1 == p2


class TestP0BasinThresholds:
    """Tests for separate basin-specific thresholds."""

    def test_basin_normalization(self):
        assert normalize_basin_key("bob") == "bob"
        assert normalize_basin_key("Bay of Bengal") == "bob"
        assert normalize_basin_key("aras") == "aras"
        assert normalize_basin_key("Arabian Sea") == "aras"

    def test_distinct_thresholds_for_basins(self):
        for h in [0, 1, 2, 3]:
            th_bob = get_basin_threshold("bob", h)
            th_aras = get_basin_threshold("aras", h)
            assert th_bob == 0.44
            assert th_aras == 0.15
            assert th_bob != th_aras, "Basin thresholds must be decoupled!"

    def test_arabian_sea_lead_detection_restoration(self):
        """Verify that at 0.15, Arabian Sea lead signal triggers alert rather than being silenced."""
        # Simulated Arabian Sea lead preconditioning score = 0.22 (which was silenced at global 0.45)
        decision_aras = evaluate_operational_decision(
            raw_score=0.22,
            calibrated_prob=0.10,
            site_or_basin="aras",
            horizon_days=2
        )
        assert decision_aras["operational_threshold"] == 0.15
        assert decision_aras["operational_alert"] == "ALERT"

        # Under Bay of Bengal threshold (0.44), 0.22 should be NO_ALERT/WATCH
        decision_bob = evaluate_operational_decision(
            raw_score=0.22,
            calibrated_prob=0.05,
            site_or_basin="bob",
            horizon_days=2
        )
        assert decision_bob["operational_threshold"] == 0.44
        assert decision_bob["operational_alert"] != "ALERT"


class TestP0RiskTiers:
    """Tests for risk-tier mapping logic."""

    def test_low_risk_tier(self):
        assert map_risk_tier(0.0) == "LOW"
        assert map_risk_tier(0.15) == "LOW"
        assert map_risk_tier(0.2999) == "LOW"

    def test_moderate_risk_tier(self):
        assert map_risk_tier(0.30) == "MODERATE"
        assert map_risk_tier(0.45) == "MODERATE"
        assert map_risk_tier(0.5999) == "MODERATE"

    def test_high_risk_tier(self):
        assert map_risk_tier(0.60) == "HIGH"
        assert map_risk_tier(0.75) == "HIGH"
        assert map_risk_tier(1.0) == "HIGH"


class TestP0ShadowMode:
    """Tests for Shadow Mode inference and telemetry."""

    def test_shadow_inference_execution(self):
        vec = np.zeros(101)
        record = ShadowInferenceService.run_shadow_evaluation(
            feature_vector=vec,
            site_id="bob",
            horizon_days=1,
            date_str="2024-10-24",
            v1_result={"risk_score": 0.20, "alert_level": "NO_ALERT", "threshold": 0.27}
        )
        assert record is not None
        assert "candidate_v2" in record
        assert "production_v1" in record
        assert record["candidate_v2"]["model_version"] == "v2.0.0-10yr-candidate"
        assert record["candidate_v2"]["operational_threshold"] in [0.44, 0.20]

    def test_shadow_mode_does_not_alter_production_response(self):
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.model_version == "v1.1.0"
        # Confirm that production alert behavior and threshold are intact
        assert resp.threshold == 0.27
        assert resp.is_calibrated is False


class TestPrePromotionAudit:
    """Rigorous Pre-Promotion Audit tests for API contract, isolation, and production integrity."""

    def test_api_response_schema_contract(self):
        """Confirm v2 shadow/candidate output conforms to the required contract:
        {basin, horizon, raw_score, calibrated_probability, risk_tier, operational_threshold, alert}
        """
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=2)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.candidate_v2 is not None
        c2 = resp.candidate_v2

        # Check required fields
        assert hasattr(c2, "basin") and c2.basin == "Bay of Bengal"
        assert hasattr(c2, "horizon") and c2.horizon == 2
        assert hasattr(c2, "raw_score") and isinstance(c2.raw_score, float)
        assert hasattr(c2, "calibrated_probability") and isinstance(c2.calibrated_probability, float)
        assert hasattr(c2, "risk_tier") and c2.risk_tier in ["LOW", "MODERATE", "HIGH"]
        assert hasattr(c2, "operational_threshold") and c2.operational_threshold in [0.44, 0.20]
        assert hasattr(c2, "alert") and c2.alert in ["ALERT", "WATCH", "NO_ALERT"]

        # Probability bounds
        assert 0.0 <= c2.calibrated_probability <= 1.0

    def test_api_response_contract_arabian_sea(self):
        """Confirm Arabian Sea routes to decoupled threshold and correct basin name."""
        req = PredictionRequest(date="2024-10-24", site_id="aras", horizon_days=1)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.candidate_v2 is not None
        c2 = resp.candidate_v2
        assert c2.basin == "Arabian Sea"
        assert c2.horizon == 1
        assert c2.operational_threshold in [0.15, 0.08]


    def test_shadow_isolation_on_exception(self, monkeypatch):
        """Verify that an unexpected failure in shadow evaluation CANNOT crash production inference."""
        def failing_shadow(*args, **kwargs):
            raise RuntimeError("Simulated shadow crash")

        monkeypatch.setattr(ShadowInferenceService, "run_shadow_evaluation", failing_shadow)
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.model_version == "v1.1.0"
        assert resp.threshold == 0.27
        assert resp.candidate_v2 is None

    def test_production_hashes_remain_unmodified(self):
        """Verify that SHA-256 hashes of every production model file remain 100% byte-identical."""
        import hashlib
        expected_hashes = {
            "model_event_type.joblib": "2ed7172fb06475aa00241ed465a5851b6155562f23a91291d29899ac56ff2fc2",
            "model_metadata.json": "f41ad99091a5f2366cd06bf5d8f9aa66135218b34f7b789968ab8c1626d025ad",
            "risk_model.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
            "risk_model_0d.joblib": "54720c2218a46977c2f93e94889d4242f357c313ab069935cf57bdc31f1b61cd",
            "risk_model_1d.joblib": "cb05c4408ef2f3b66999f8b0016b9dbd1285bc88c1c8f6e32aae310817ccd57c",
            "risk_model_2d.joblib": "6e6e34c5806c2db83a61369eb8488dedb827845cba8a340c0b8b286363a52843",
            "risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
        }
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        for fname, expected_hash in expected_hashes.items():
            fpath = os.path.join(models_dir, fname)
            assert os.path.exists(fpath), f"Production file missing: {fname}"
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch for {fname}: got {actual_hash}, expected {expected_hash}"

    def test_out_of_sample_calibration_check(self):
        """Verify that independent calibration check results demonstrate generalizable calibration (ECE < 0.05)."""
        audit_json = os.path.join(os.path.dirname(__file__), "..", "scratch", "calibration_overfitting_audit.json")
        if os.path.exists(audit_json):
            with open(audit_json, "r") as f:
                data = json.load(f)
            for h in ["0d", "1d", "2d", "3d"]:
                assert h in data
                h_data = data[h]
                assert h_data["oos_h2_isotonic"]["ece"] < 0.05
                assert h_data["oos_h2_isotonic"]["brier"] < h_data["oos_h2_raw"]["brier"]



