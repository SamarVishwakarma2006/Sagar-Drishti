"""
Unit and Integration Test Suite for Sagar-Drishti V2.3 Model D Shadow Evaluation.

Verifies:
1. Strict Observational Isolation: Zero authority to trigger alerts, alter risk scores, or affect production.
2. Production Response Equivalence: Identical PredictionResponse whether shadow is enabled or disabled.
3. Causal & Temporal Safety: Cutoff strictly at T 18:00:00 UTC; future data (T+1, T+2) rejected.
4. Exact 224-Feature Contract: 101 ocean + 123 atmos features, schema hash, and ordering validated.
5. Trusted Hash Validation: Artifact hashes match frozen manifest; schema hash matches trusted value.
6. 11 Failure-Isolation Cases:
   - Case 1: Model D loads successfully across all horizons.
   - Case 2: Model D missing -> fail-closed, production unaffected.
   - Case 3: Atmospheric data missing -> ATMOSPHERE_MISSING, production unaffected.
   - Case 4: Atmospheric data late -> ATMOSPHERE_LATE, production unaffected.
   - Case 5: Future/invalid timestamp -> INVALID_TIMESTAMP, production unaffected.
   - Case 6: Feature count incorrect -> FEATURE_SCHEMA_MISMATCH, production unaffected.
   - Case 7: Feature ordering / names mismatch -> FEATURE_SCHEMA_MISMATCH, production unaffected.
   - Case 8: NaN / Inf input -> fail-closed, production unaffected.
   - Case 9: Shadow inference exception -> caught, production unaffected.
   - Case 10: Telemetry storage failure -> caught, production unaffected.
   - Case 11: Production error unrelated to shadow -> handled by production as expected.
7. True Bounded Queue: Finite queue capacity, graceful drop under queue-full, no production blocking.
8. Safe Lazy Initialization: Startup and resource failures do not impact production.
9. Coverage Observability: Tracks submitted, executed, dropped, successes, failures, and latency.
10. Deterministic Inference: Identical inputs yield identical probability.
11. Baseline Invariant Immutability: SHA-256 hashes of production models and frozen policy unchanged.
"""

import os
import json
import time
import queue
import hashlib
import tempfile
import numpy as np
import pytest
import joblib

from app.services.shadow_v2_3_service import (
    ShadowV23Service,
    TRUSTED_MODEL_HASHES,
    TRUSTED_FEATURE_SCHEMA_HASH,
    TOTAL_FEATURES_EXPECTED,
    OCEAN_FEATURES_EXPECTED,
    ATMOS_FEATURES_EXPECTED,
    SPATIAL_MAPPING,
    STATUS_READY,
    STATUS_ATMOSPHERE_MISSING,
    STATUS_ATMOSPHERE_LATE,
    STATUS_FEATURE_SCHEMA_MISMATCH,
    STATUS_INVALID_TIMESTAMP,
    STATUS_INFERENCE_ERROR,
    COV_SHADOW_SUCCESS,
    COV_SHADOW_FAILED,
    COV_SHADOW_SKIPPED_MISSING_DATA,
    COV_SHADOW_SKIPPED_LATE_DATA,
    COV_SHADOW_NOT_INITIALIZED,
    COV_SHADOW_QUEUE_FULL,
    CANDIDATE_MODEL_D_DIR,
    FROZEN_MANIFEST_PATH,
    SHADOW_DATA_DIR,
)
from app.services.prediction_service import PredictionService
from app.models.schemas import PredictionRequest, CustomObservationRequest


@pytest.fixture(autouse=True)
def reset_shadow_service():
    """Ensure clean state before and after each test."""
    ShadowV23Service.set_enabled(True)
    ShadowV23Service.clear_cache()
    ShadowV23Service.reset_metrics()
    ShadowV23Service.set_overrides(None, None, None)
    yield
    ShadowV23Service.set_enabled(True)
    ShadowV23Service.clear_cache()
    ShadowV23Service.reset_metrics()
    ShadowV23Service.set_overrides(None, None, None)


class TestShadowContractAndHashes:
    """Tests for artifact hashes, schema hashes, and feature ordering."""

    def test_case_1_model_d_artifacts_load_successfully(self):
        """Case 1: All 4 horizons load cleanly and have 224 features."""
        assert ShadowV23Service._ensure_initialized() is True
        for h in [0, 1, 2, 3]:
            assert h in ShadowV23Service._models
            model = ShadowV23Service._models[h]
            assert hasattr(model, "predict_proba")
            assert model.n_features_in_ == TOTAL_FEATURES_EXPECTED

    def test_model_artifact_sha256_matches_trusted_manifest(self):
        """Verify each horizon's SHA-256 matches trusted expected hash."""
        for h in [0, 1, 2, 3]:
            art_path = os.path.join(CANDIDATE_MODEL_D_DIR, f"risk_model_{h}d.joblib")
            assert os.path.exists(art_path)
            actual_h = ShadowV23Service._compute_sha256(art_path)
            expected_h = TRUSTED_MODEL_HASHES[h]
            assert actual_h == expected_h, f"Horizon {h}d artifact hash mismatch!"

    def test_schema_hash_matches_trusted_manifest(self):
        """Verify the 224-feature schema hash matches trusted frozen value."""
        assert ShadowV23Service._ensure_initialized() is True
        assert ShadowV23Service._feature_schema_hash == TRUSTED_FEATURE_SCHEMA_HASH
        assert len(ShadowV23Service._expected_feature_names) == TOTAL_FEATURES_EXPECTED

    def test_feature_composition_and_zero_target_leakage(self):
        """Assert features derive exclusively from 101 ocean + 123 atmos; no targets/labels."""
        assert ShadowV23Service._ensure_initialized() is True
        feats = ShadowV23Service._expected_feature_names
        ocean_feats = [f for f in feats if not f.startswith("atmos_")]
        atmos_feats = [f for f in feats if f.startswith("atmos_")]

        assert len(ocean_feats) == OCEAN_FEATURES_EXPECTED
        assert len(atmos_feats) == ATMOS_FEATURES_EXPECTED
        assert len(ocean_feats) + len(atmos_feats) == TOTAL_FEATURES_EXPECTED

        # Zero target or label leakage
        forbidden_substrings = ["target", "label", "cyclone_id", "storm", "event_within", "lead_", "imd_grade"]
        for f in feats:
            for sub in forbidden_substrings:
                assert sub not in f.lower(), f"Potential target leakage in feature {f}"

    def test_hardcoded_spatial_mapping(self):
        """Assert exact spatial sub-basin mapping for Bay of Bengal and Arabian Sea."""
        assert SPATIAL_MAPPING["bob"]["atmos_north"] == "NBOB"
        assert SPATIAL_MAPPING["bob"]["atmos_central"] == "CBOB"
        assert SPATIAL_MAPPING["bob"]["atmos_south"] == "SBOB"

        assert SPATIAL_MAPPING["aras"]["atmos_north"] == "NAS"
        assert SPATIAL_MAPPING["aras"]["atmos_central"] == "CAS"
        assert SPATIAL_MAPPING["aras"]["atmos_south"] == "SAS"


class TestFailureIsolationCases:
    """Explicit tests for the required 11 failure isolation cases."""

    def test_case_2_model_d_missing_fails_closed_production_unaffected(self, tmp_path):
        """Case 2: If Model D artifact is missing, shadow fails closed, production succeeds."""
        # Point to empty directory
        ShadowV23Service.set_overrides(model_dir=str(tmp_path))
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
            v1_result={"risk_score": 0.35, "alert_level": "WATCH"},
        )
        assert rec["inference_status"] == "FAILED"
        assert rec["coverage_status"] == COV_SHADOW_NOT_INITIALIZED
        assert rec["shadow_probability"] is None

        # Production must continue normally
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.model_version == "v1.1.0"
        assert resp.probability > 0.0

    def test_case_3_atmospheric_data_missing(self):
        """Case 3: Date with missing atmospheric data in store -> ATMOSPHERE_MISSING."""
        ShadowV23Service._ensure_initialized()
        # 2020-01-01 is within historical range, but let's test a date intentionally absent or simulate missing
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2010-01-01",  # Prior to ERA5 dataset start (2016-06-24)
            v1_result={"risk_score": 0.20, "alert_level": "NO_ALERT"},
        )
        assert rec["inference_status"] == "SKIPPED"
        assert rec["data_quality_status"] == STATUS_ATMOSPHERE_MISSING
        assert rec["coverage_status"] == COV_SHADOW_SKIPPED_MISSING_DATA
        assert rec["shadow_probability"] is None

    def test_case_4_atmospheric_data_late(self):
        """Case 4: Date beyond latest available atmospheric data -> ATMOSPHERE_LATE."""
        ShadowV23Service._ensure_initialized()
        # Query date after max available date (2026-06-23)
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2026-06-24",
            v1_result={"risk_score": 0.20, "alert_level": "NO_ALERT"},
        )
        assert rec["inference_status"] == "SKIPPED"
        assert rec["data_quality_status"] == STATUS_ATMOSPHERE_LATE
        assert rec["coverage_status"] == COV_SHADOW_SKIPPED_LATE_DATA
        assert rec["shadow_probability"] is None

    def test_case_5_atmospheric_timestamp_in_future(self):
        """Case 5: Date in future relative to runtime now -> INVALID_TIMESTAMP."""
        ShadowV23Service._ensure_initialized()
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2035-01-01",
            v1_result={"risk_score": 0.20, "alert_level": "NO_ALERT"},
        )
        assert rec["inference_status"] == "FAILED"
        assert rec["data_quality_status"] == STATUS_INVALID_TIMESTAMP
        assert rec["coverage_status"] == COV_SHADOW_FAILED

    def test_case_6_feature_count_incorrect(self):
        """Case 6: Feature count != 101 -> FEATURE_SCHEMA_MISMATCH."""
        ShadowV23Service._ensure_initialized()
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(50),  # Wrong count!
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert rec["inference_status"] == "FAILED"
        assert rec["data_quality_status"] == STATUS_FEATURE_SCHEMA_MISMATCH
        assert rec["coverage_status"] == COV_SHADOW_FAILED

    def test_case_7_feature_ordering_incorrect(self):
        """Case 7: Feature dict missing required keys -> FEATURE_SCHEMA_MISMATCH."""
        ShadowV23Service._ensure_initialized()
        incomplete_dict = {"temp_current": 28.5}  # Missing 100 features
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=incomplete_dict,
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert rec["inference_status"] == "FAILED"
        assert rec["data_quality_status"] == STATUS_FEATURE_SCHEMA_MISMATCH
        assert rec["coverage_status"] == COV_SHADOW_FAILED

    def test_case_8_nan_or_inf_input(self):
        """Case 8: Input containing NaN or Inf -> FEATURE_SCHEMA_MISMATCH."""
        ShadowV23Service._ensure_initialized()
        ocean_nan = np.zeros(101)
        ocean_nan[10] = np.nan
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=ocean_nan,
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert rec["inference_status"] == "FAILED"
        assert rec["data_quality_status"] == STATUS_FEATURE_SCHEMA_MISMATCH
        assert "NaN or Inf" in rec["failure_reason"]

        ocean_inf = np.zeros(101)
        ocean_inf[15] = np.inf
        rec2 = ShadowV23Service.evaluate_shadow(
            ocean_features=ocean_inf,
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert rec2["inference_status"] == "FAILED"
        assert rec2["data_quality_status"] == STATUS_FEATURE_SCHEMA_MISMATCH

    def test_case_9_shadow_inference_exception(self, monkeypatch):
        """Case 9: Model inference exception caught and isolated."""
        ShadowV23Service._ensure_initialized()
        model = ShadowV23Service._models[3]

        def crash_predict(*args, **kwargs):
            raise RuntimeError("Hardware failure during matrix multiply")

        monkeypatch.setattr(model, "predict_proba", crash_predict)
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert rec["inference_status"] == "FAILED"
        assert rec["data_quality_status"] == STATUS_INFERENCE_ERROR
        assert "Hardware failure" in rec["failure_reason"]

        # Production still succeeds
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(req)
        assert resp.status == "success"

    def test_case_10_shadow_storage_failure(self, monkeypatch):
        """Case 10: Disk write failure caught and does not affect production."""
        ShadowV23Service._ensure_initialized()

        def fail_log(rec):
            raise IOError("Disk read-only filesystem error")

        monkeypatch.setattr(ShadowV23Service, "_log_telemetry", fail_log)
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(req)
        assert resp.status == "success"
        assert resp.probability > 0.0

    def test_case_11_production_error_unrelated_to_shadow(self):
        """Case 11: Production invalid input raises ValueError or returns insufficient_data normally; shadow does not mask it."""
        # 1. Invalid date format raises ValueError
        invalid_format_req = PredictionRequest(date="not-a-valid-date", site_id="bob", horizon_days=3)
        with pytest.raises(ValueError) as exc:
            PredictionService.predict(invalid_format_req)
        assert "Invalid date format" in str(exc.value)

        # 2. Out of coverage date returns status='insufficient_data'
        out_of_range_req = PredictionRequest(date="1990-01-01", site_id="bob", horizon_days=3)
        resp = PredictionService.predict(out_of_range_req)
        assert resp.status == "insufficient_data"
        assert "outside eligible ocean observation range" in resp.message


class TestProductionEquivalenceAndSafety:
    """Tests proving production responses are identical whether shadow is enabled or disabled."""

    def test_shadow_isolation_equivalence(self):
        """
        Run the exact same production prediction request twice:
        A) Shadow Disabled
        B) Shadow Enabled
        Assert every field in PredictionResponse is 100% identical.
        """
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)

        # Run A: Shadow Disabled
        ShadowV23Service.set_enabled(False)
        resp_disabled = PredictionService.predict(req)

        # Run B: Shadow Enabled
        ShadowV23Service.set_enabled(True)
        resp_enabled = PredictionService.predict(req)

        # Drain queue to allow shadow background evaluation to finish
        ShadowV23Service.drain_queue(timeout=5.0)

        # Assert full equivalence
        assert resp_disabled.status == resp_enabled.status == "success"
        assert resp_disabled.model_version == resp_enabled.model_version == "v1.1.0"
        assert resp_disabled.probability == resp_enabled.probability
        assert resp_disabled.warning_level == resp_enabled.warning_level
        assert resp_disabled.threshold == resp_enabled.threshold
        assert resp_disabled.target == resp_enabled.target
        assert resp_disabled.prediction == resp_enabled.prediction
        assert resp_disabled.event_type == resp_enabled.event_type
        assert resp_disabled.is_calibrated == resp_enabled.is_calibrated

        # Candidate v2 parity
        assert (resp_disabled.candidate_v2 is None) == (resp_enabled.candidate_v2 is None)
        if resp_disabled.candidate_v2:
            assert resp_disabled.candidate_v2.raw_score == resp_enabled.candidate_v2.raw_score
            assert resp_disabled.candidate_v2.calibrated_probability == resp_enabled.candidate_v2.calibrated_probability
            assert resp_disabled.candidate_v2.risk_tier == resp_enabled.candidate_v2.risk_tier

        # Top features & physical drivers
        assert len(resp_disabled.top_features) == len(resp_enabled.top_features)
        assert resp_disabled.top_features[0].feature == resp_enabled.top_features[0].feature
        assert resp_disabled.top_features[0].importance == resp_enabled.top_features[0].importance

    def test_causal_18z_cutoff_enforcement(self):
        """Verify that atmospheric features strictly respect the 18:00 UTC cutoff on date T."""
        ShadowV23Service._ensure_initialized()
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert rec["atmosphere_timestamp"] == "2024-10-24T18:00:00Z"
        assert rec["ocean_timestamp"] == "2024-10-24T00:00:00Z"

    def test_deterministic_inference(self):
        """Identical inputs produce identical probabilities across calls."""
        ShadowV23Service._ensure_initialized()
        ocean_input = np.linspace(-1.0, 1.0, 101)
        rec1 = ShadowV23Service.evaluate_shadow(
            ocean_features=ocean_input,
            site_id="bob",
            horizon_days=2,
            date_str="2024-10-24",
        )
        rec2 = ShadowV23Service.evaluate_shadow(
            ocean_features=ocean_input,
            site_id="bob",
            horizon_days=2,
            date_str="2024-10-24",
        )
        assert rec1["inference_status"] == rec2["inference_status"] == "SUCCESS"
        assert rec1["shadow_probability"] == rec2["shadow_probability"]
        assert rec1["shadow_alert_at_research_threshold"] == rec2["shadow_alert_at_research_threshold"]

    @pytest.mark.parametrize("failure_mode", [
        "normal",
        "model_missing",
        "inference_exception",
        "telemetry_failure",
        "queue_full",
        "era5_unavailable",
    ])
    def test_production_response_invariance_across_all_failure_modes(self, failure_mode, monkeypatch, tmp_path):
        """
        Gate 7: Verify production PredictionResponse is 100% invariant across all 6 shadow failure modes:
        1. Normal shadow inference
        2. Shadow model missing
        3. Shadow inference exception
        4. Shadow telemetry failure
        5. Queue full
        6. ERA5 unavailable
        """
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)

        # Baseline: Shadow Disabled
        ShadowV23Service.set_enabled(False)
        baseline_resp = PredictionService.predict(req)

        # Setup failure mode
        ShadowV23Service.set_enabled(True)
        if failure_mode == "model_missing":
            ShadowV23Service.set_overrides(model_dir=str(tmp_path))
        elif failure_mode == "inference_exception":
            ShadowV23Service._ensure_initialized()
            m = ShadowV23Service._models[3]
            monkeypatch.setattr(m, "predict_proba", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Crash")))
        elif failure_mode == "telemetry_failure":
            monkeypatch.setattr(ShadowV23Service, "_log_telemetry", lambda *args, **kwargs: (_ for _ in ()).throw(IOError("Disk Full")))
        elif failure_mode == "queue_full":
            small_q = queue.Queue(maxsize=1)
            small_q.put_nowait({"dummy": True})
            monkeypatch.setattr(ShadowV23Service, "_queue", small_q)
        elif failure_mode == "era5_unavailable":
            ShadowV23Service.set_overrides(era5_path=os.path.join(str(tmp_path), "nonexistent.parquet"))

        # Run with shadow enabled under failure mode
        test_resp = PredictionService.predict(req)
        ShadowV23Service.drain_queue(timeout=5.0)

        # Assert full equality
        assert test_resp.status == baseline_resp.status == "success"
        assert test_resp.model_version == baseline_resp.model_version == "v1.1.0"
        assert test_resp.probability == baseline_resp.probability
        assert test_resp.warning_level == baseline_resp.warning_level
        assert test_resp.threshold == baseline_resp.threshold
        assert test_resp.target == baseline_resp.target
        assert test_resp.prediction == baseline_resp.prediction
        assert test_resp.event_type == baseline_resp.event_type
        assert test_resp.is_calibrated == baseline_resp.is_calibrated

        if baseline_resp.candidate_v2:
            assert test_resp.candidate_v2.raw_score == baseline_resp.candidate_v2.raw_score
            assert test_resp.candidate_v2.calibrated_probability == baseline_resp.candidate_v2.calibrated_probability
            assert test_resp.candidate_v2.risk_tier == baseline_resp.candidate_v2.risk_tier

    def test_adversarial_future_atmospheric_timestamp_rejection(self):
        """
        Gate 4: Prove that even if atmospheric storage contains future records or newer data,
        the prediction cutoff T 18:00 UTC cannot be exceeded and future dates are rejected.
        """
        ShadowV23Service._ensure_initialized()
        # Request with a future date
        rec = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2030-01-01",
        )
        assert rec["data_quality_status"] == STATUS_INVALID_TIMESTAMP
        assert rec["inference_status"] == "FAILED"
        assert rec["shadow_probability"] is None

        # Request with explicit future prediction timestamp
        rec_future_ts = ShadowV23Service.evaluate_shadow(
            ocean_features=np.zeros(101),
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
            prediction_timestamp="2035-01-01T12:00:00Z",
        )
        assert rec_future_ts["data_quality_status"] == STATUS_INVALID_TIMESTAMP
        assert rec_future_ts["inference_status"] == "FAILED"

    def test_concurrent_telemetry_write_safety(self, tmp_path, monkeypatch):
        """
        Gate 1: Verify concurrent writes from multiple worker threads never corrupt JSONL lines.
        """
        import threading
        monkeypatch_dir = str(tmp_path)
        monkeypatch.setattr("app.services.shadow_v2_3_service.SHADOW_DATA_DIR", monkeypatch_dir)

        records_written = []
        errors = []

        def worker_write(idx):
            try:
                rec = {
                    "thread_id": idx,
                    "timestamp": f"2026-09-13T01:00:{idx:02d}Z",
                    "payload": f"payload_from_thread_{idx}" * 20,
                    "random_numbers": list(range(50)),
                }
                ShadowV23Service._log_telemetry(rec)
                records_written.append(idx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_write, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(records_written) == 40

        import glob
        files = glob.glob(os.path.join(monkeypatch_dir, "shadow_v2_3_telemetry_*.jsonl"))
        assert len(files) == 1
        with open(files[0], "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 40
        # Every single line must be strictly valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "thread_id" in parsed
            assert len(parsed["random_numbers"]) == 50


class TestBoundedQueueAndObservability:
    """Tests for queue capacity limits, job dropping, and metrics tracking."""

    def test_true_bounded_queue_drops_jobs_when_full(self, monkeypatch):
        """When queue reaches capacity, jobs are dropped without blocking production."""
        ShadowV23Service.reset_metrics()
        small_queue = queue.Queue(maxsize=3)
        monkeypatch.setattr(ShadowV23Service, "_queue", small_queue)

        dummy_ocean = np.zeros(101)

        # Fill the queue (3 items)
        for i in range(3):
            dispatched = ShadowV23Service.dispatch_shadow_evaluation(
                ocean_features=dummy_ocean,
                site_id="bob",
                horizon_days=3,
                date_str="2024-10-24",
            )
            assert dispatched is True

        # 4th item must be dropped immediately without blocking
        dispatched_4 = ShadowV23Service.dispatch_shadow_evaluation(
            ocean_features=dummy_ocean,
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
        )
        assert dispatched_4 is False

        metrics = ShadowV23Service.get_observability_metrics()
        assert metrics["jobs_submitted"] == 4
        assert metrics["jobs_dropped"] == 1
        assert metrics["queue_capacity"] == 3

    def test_observability_metrics_reporting(self):
        """Verify observability metrics track all required coverage states."""
        ShadowV23Service.reset_metrics()
        dummy_ocean = np.zeros(101)

        # 1 successful
        ShadowV23Service.evaluate_shadow(
            ocean_features=dummy_ocean,
            site_id="bob",
            horizon_days=3,
            date_str="2024-10-24",
            v1_result={"risk_score": 0.30},
        )

        # 1 missing
        ShadowV23Service.evaluate_shadow(
            ocean_features=dummy_ocean,
            site_id="bob",
            horizon_days=3,
            date_str="2010-01-01",
        )

        metrics = ShadowV23Service.get_observability_metrics()
        assert metrics["jobs_executed"] == 2
        assert metrics["successful_predictions"] == 1
        assert metrics["data_unavailable_predictions"] == 1
        assert metrics["missing_atmosphere_count"] == 1
        assert metrics["current_available_atmospheric_data_max_timestamp"] == "2026-06-23T18:00:00Z"
        assert metrics["predictions_by_site"]["bob"] == 1


class TestProductionIntegrityHashes:
    """Verify production baseline artifacts remain 100% byte-for-byte unmodified."""

    def test_production_artifacts_unmodified(self):
        expected_hashes = {
            "backend/models/risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
            "backend/models/v2_10yr/risk_model_3d.joblib": "7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3",
            "backend/config/frozen_alert_policy_v2.json": "6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6",
            "backend/data/historical/features_10yr.parquet": "cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867",
            "backend/data/historical/labeled_features_10yr_clean.parquet": "25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a",
            "backend/data/era5/features_atmosphere_10yr_daily.parquet": "551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864",
            "backend/config/v2_3_frozen_experiment_manifest.json": "73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc",
        }
        for rel_path, expected_h in expected_hashes.items():
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", rel_path)
            assert os.path.exists(full_path), f"Baseline file missing: {rel_path}"
            with open(full_path, "rb") as f:
                actual_h = hashlib.sha256(f.read()).hexdigest()
            assert actual_h == expected_h, f"Integrity violation on {rel_path}!"
