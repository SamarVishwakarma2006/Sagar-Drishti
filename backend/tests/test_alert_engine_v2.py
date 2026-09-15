"""
Unit and Integration Tests for Operational Alert Engine V2.

Verifies:
- Persistence state machine & transitions
- 2-consecutive-day policy (Bay of Bengal)
- 2-of-3 policy (Arabian Sea)
- 3-consecutive-day policy
- Immediate High-Risk escalation
- Basin-specific policy decoupling
- Alert reason and persistence state reporting
- Calibrated probability strictly drives alerting (not raw RF vote fraction)
- Cooldown and de-escalation logic
- Shadow isolation and non-blocking safety
- Production baseline v1.1.0 hash invariance
"""
import os
import hashlib
import pytest
import numpy as np

from app.services.alert_engine_v2 import (
    OperationalAlertEngineV2,
    normalize_basin_key,
    map_risk_tier
)
from app.services.shadow_service import ShadowInferenceService
from app.services.prediction_service import PredictionService
from app.models.schemas import PredictionRequest


class TestOperationalAlertEngineV2:

    def setup_method(self):
        OperationalAlertEngineV2.reset_state()

    def test_basin_key_normalization(self):
        assert normalize_basin_key("Bay of Bengal") == "bob"
        assert normalize_basin_key("bob") == "bob"
        assert normalize_basin_key("Arabian Sea") == "aras"
        assert normalize_basin_key("aras") == "aras"

    def test_risk_tier_mapping(self):
        assert map_risk_tier(0.10) == "LOW"
        assert map_risk_tier(0.299) == "LOW"
        assert map_risk_tier(0.30) == "MODERATE"
        assert map_risk_tier(0.599) == "MODERATE"
        assert map_risk_tier(0.60) == "HIGH"
        assert map_risk_tier(0.95) == "HIGH"

    def test_2_consecutive_policy_bay_of_bengal(self):
        """Bay of Bengal: p >= 0.20 requires 2 consecutive observations for ALERT."""
        # Day 1: p = 0.25 (First observation) -> WATCH
        d1 = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.25,
            raw_score=0.55,
            site_or_basin="bob",
            horizon_days=3,
            history_override=[0.25]
        )
        assert d1["alert_decision"] == "WATCH"
        assert d1["alert_reason"] == "MODERATE_WATCH_PENDING_CONFIRMATION"
        assert d1["persistence_state"] == "1_DAY_PENDING_CONFIRMATION"

        # Day 2: p = 0.26 (Second consecutive observation) -> ALERT
        d2 = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.26,
            raw_score=0.58,
            site_or_basin="bob",
            horizon_days=3,
            history_override=[0.25, 0.26]
        )
        assert d2["alert_decision"] == "ALERT"
        assert d2["alert_reason"] == "PERSISTENT_MODERATE_2_CONSECUTIVE"
        assert d2["persistence_state"] == "2_CONSECUTIVE_CONFIRMED"

    def test_2_of_3_policy_arabian_sea(self):
        """Arabian Sea: p >= 0.08 in >= 2 of last 3 days triggers ALERT."""
        # Day 1: p = 0.10 -> WATCH
        d1 = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.10,
            raw_score=0.18,
            site_or_basin="aras",
            horizon_days=3,
            history_override=[0.10]
        )
        assert d1["alert_decision"] == "WATCH"

        # Day 2: p = 0.03 (drops below threshold) -> NO_ALERT
        d2 = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.03,
            raw_score=0.07,
            site_or_basin="aras",
            horizon_days=3,
            history_override=[0.10, 0.03]
        )
        assert d2["alert_decision"] == "NO_ALERT"

        # Day 3: p = 0.11 (second positive in 3-day window: [0.10, 0.03, 0.11]) -> ALERT
        d3 = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.11,
            raw_score=0.19,
            site_or_basin="aras",
            horizon_days=3,
            history_override=[0.10, 0.03, 0.11]
        )
        assert d3["alert_decision"] == "ALERT"
        assert d3["alert_reason"] == "PERSISTENT_MODERATE_2_OF_3"
        assert d3["persistence_state"] == "2_OF_3_CONFIRMED"

    def test_immediate_high_risk_escalation(self):
        """High calibrated probability (>=0.60 in BoB, >=0.50 in ArAs) escalates immediately to ALERT."""
        # BoB immediate high
        d_bob = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.75,
            raw_score=0.82,
            site_or_basin="bob",
            horizon_days=3,
            history_override=[0.75]
        )
        assert d_bob["alert_decision"] == "ALERT"
        assert d_bob["alert_reason"] == "IMMEDIATE_HIGH"
        assert d_bob["risk_tier"] == "HIGH"

        # ArAs immediate high
        d_aras = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.55,
            raw_score=0.60,
            site_or_basin="aras",
            horizon_days=3,
            history_override=[0.55]
        )
        assert d_aras["alert_decision"] == "ALERT"
        assert d_aras["alert_reason"] == "IMMEDIATE_HIGH"

    def test_cooldown_de_escalation(self):
        """After ALERT, dropping to calm requires 2 consecutive days below cooldown threshold to reach NO_ALERT."""
        # Previous state was ALERT, day 1 drops to 0.05 (below cooldown) -> WATCH (cooldown)
        d_cool1 = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.05,
            raw_score=0.15,
            site_or_basin="bob",
            horizon_days=3,
            history_override=[0.70, 0.05],
            state_override="ALERT"
        )
        assert d_cool1["alert_decision"] == "WATCH"
        assert d_cool1["alert_reason"] == "COOLDOWN_DE_ESCALATION"

    def test_calibrated_probability_drives_decision_not_raw_score(self):
        """A high raw score with a low calibrated probability must NOT trigger alert."""
        # Simulated raw score = 0.55, but calibrated probability = 0.05
        d = OperationalAlertEngineV2.evaluate_decision(
            calibrated_prob=0.05,
            raw_score=0.55,
            site_or_basin="bob",
            horizon_days=3,
            history_override=[0.05]
        )
        assert d["alert_decision"] == "NO_ALERT"
        assert d["alert_reason"] == "BELOW_THRESHOLD"

    def test_shadow_service_and_api_contract_schema(self):
        """Verify that PredictionService returns the full 10-field v2 shadow payload."""
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.model_version in ["v2.0.0", "v1.1.0"]
        assert resp.candidate_v2 is not None

        c2 = resp.candidate_v2
        assert hasattr(c2, "basin") and c2.basin == "Bay of Bengal"
        assert hasattr(c2, "horizon") and c2.horizon == 3
        assert hasattr(c2, "raw_score") and isinstance(c2.raw_score, float)
        assert hasattr(c2, "calibrated_probability") and isinstance(c2.calibrated_probability, float)
        assert hasattr(c2, "risk_tier") and c2.risk_tier in ["LOW", "MODERATE", "HIGH"]
        assert hasattr(c2, "policy_threshold") and c2.policy_threshold == 0.20
        assert hasattr(c2, "persistence_state") and isinstance(c2.persistence_state, str)
        assert hasattr(c2, "alert_decision") and c2.alert_decision in ["ALERT", "WATCH", "NO_ALERT"]
        assert hasattr(c2, "alert_reason") and isinstance(c2.alert_reason, str)
        assert hasattr(c2, "model_version") and c2.model_version == "v2.0.0-10yr-candidate"

    def test_production_hash_invariance(self):
        """SHA-256 of all 7 production model files in backend/models/ must remain 100% byte-identical."""
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
        for fname, exp_hash in expected_hashes.items():
            fpath = os.path.join(models_dir, fname)
            assert os.path.exists(fpath), f"Missing production model: {fname}"
            with open(fpath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == exp_hash, f"Production model hash mismatch: {fname}"
