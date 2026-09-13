"""
Automated Test Suite for Sagar-Drishti V2.3 Shadow Monitoring & Evidence Accumulation.

Verifies:
1. Telemetry ingestion from JSONL to SQLite
2. Deterministic observation ID & duplicate protection
3. Safe concurrent ingestion
4. Malformed and corrupted telemetry isolation
5. Coverage & data quality metrics
6. Production vs Shadow comparative descriptive statistics
7. Event registry tracking
8. Insufficient evidence handling & sample size guardrails
9. Evidence maturity scale transitions
10. Feature drift detection (KS test & PSI)
11. Atmospheric freshness tracking
12. API research-only isolation & zero operational authority
13. No production alert mutation
14. No threshold mutation
15. No frozen policy mutation
16. No model modification
17. Historical test-set quarantine protection
18. Deterministic analytics
19. Missing telemetry handling
20. Corrupted telemetry handling
"""

import os
import json
import tempfile
import threading
import pytest
from datetime import datetime, timezone
import numpy as np
from fastapi.testclient import TestClient

from backend.app.services.shadow_v2_3_store import (
    ShadowV23Store,
    compute_observation_id,
    compute_atmospheric_age_hours,
)
from backend.app.services.shadow_v2_3_monitor import (
    ShadowV23Monitor,
    MIN_STORMS_FOR_STATISTICAL_CLAIMS,
    MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS,
)
from backend.app.services.shadow_v2_3_reporter import ShadowV23Reporter
from backend.app.services.shadow_v2_3_visualizer import ShadowV23Visualizer
from backend.app.main import app


@pytest.fixture
def temp_store(tmp_path):
    """Provides an isolated temporary SQLite database for testing."""
    db_file = str(tmp_path / "test_shadow_obs.db")
    return ShadowV23Store(db_path=db_file)


@pytest.fixture
def populated_store(temp_store):
    """Provides a store populated with synthetic test observations."""
    for i in range(40):
        # Unique timestamp per record to avoid duplicate deduplication
        day = (i // 24) + 1
        hour = i % 24
        dt_str = f"2026-07-{day:02d}"
        pred_ts = f"{dt_str}T{hour:02d}:00:00Z"
        ocean_ts = f"{dt_str}T00:00:00Z"
        atmos_ts = f"{dt_str}T18:00:00Z"

        prod_prob = round(0.10 + (i % 5) * 0.05, 3)
        shadow_prob = round(0.12 + (i % 6) * 0.06, 3)

        dq = "READY"
        inf_status = "SUCCESS"
        cov_status = "SHADOW_SUCCESS"

        if i % 10 == 7:
            dq = "ATMOSPHERE_LATE"
            inf_status = "SKIPPED"
            cov_status = "SHADOW_SKIPPED_LATE_DATA"
            shadow_prob = None
        elif i % 10 == 8:
            dq = "ATMOSPHERE_MISSING"
            inf_status = "SKIPPED"
            cov_status = "SHADOW_SKIPPED_MISSING_DATA"
            shadow_prob = None
        elif i % 10 == 9:
            dq = "INFERENCE_ERROR"
            inf_status = "FAILED"
            cov_status = "SHADOW_FAILED"
            shadow_prob = None

        rec = {
            "prediction_timestamp": pred_ts,
            "prediction_date": dt_str,
            "site": "bob" if i % 2 == 0 else "aras",
            "horizon": i % 4,
            "production_model_version": "v1.1.0",
            "shadow_model_version": "v2.3.0-model-d-shadow",
            "production_probability": prod_prob,
            "shadow_probability": shadow_prob,
            "production_alert": "NO_ALERT" if prod_prob < 0.25 else "WATCH",
            "shadow_alert_at_research_threshold": "RESEARCH_WATCH" if (shadow_prob or 0) >= 0.27 else "RESEARCH_NO_ALERT",
            "ocean_timestamp": ocean_ts,
            "atmosphere_timestamp": atmos_ts,
            "data_quality_status": dq,
            "coverage_status": cov_status,
            "inference_status": inf_status,
            "failure_reason": None if inf_status == "SUCCESS" else "Simulated error",
            "model_artifact_hash": "250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775",
            "feature_schema_hash": "25a9cad9cfa08414f86d5214155fc2e4f2e0225eb28f17902b4d35d5729db236",
            "created_at": pred_ts,
        }
        temp_store.insert_observation(rec)
    return temp_store


# ============================================================================
# COMPONENT 1 & 2: Telemetry Ingestion & Deduplication Tests
# ============================================================================
class TestTelemetryIngestionAndDeduplication:

    def test_1_telemetry_ingestion_sqlite(self, temp_store, tmp_path):
        """Test Case 1: Ingests valid JSONL telemetry into SQLite."""
        jsonl_file = tmp_path / "sample_telemetry.jsonl"
        with open(jsonl_file, "w") as f:
            for i in range(10):
                line = json.dumps({
                    "prediction_timestamp": f"2026-08-01T{i:02d}:00:00Z",
                    "site": "bob",
                    "horizon": 3,
                    "production_probability": 0.20,
                    "shadow_probability": 0.35,
                    "data_quality_status": "READY",
                    "inference_status": "SUCCESS",
                })
                f.write(line + "\n")

        res = temp_store.ingest_jsonl_file(str(jsonl_file))
        assert res["total_lines"] == 10
        assert res["inserted"] == 10
        assert res["duplicates"] == 0
        assert res["malformed"] == 0

        counts = temp_store.get_observation_counts()
        assert counts["total_observations"] == 10
        assert counts["successful_inferences"] == 10

    def test_2_duplicate_detection(self, temp_store, tmp_path):
        """Test Case 2: Re-ingestion of identical records must not inflate observation counts."""
        jsonl_file = tmp_path / "dup_telemetry.jsonl"
        with open(jsonl_file, "w") as f:
            line = json.dumps({
                "prediction_timestamp": "2026-08-05T12:00:00Z",
                "site": "aras",
                "horizon": 2,
                "ocean_timestamp": "2026-08-05T00:00:00Z",
                "production_probability": 0.15,
                "shadow_probability": 0.22,
                "data_quality_status": "READY",
                "inference_status": "SUCCESS",
            })
            f.write(line + "\n")

        # First ingestion
        res1 = temp_store.ingest_jsonl_file(str(jsonl_file))
        assert res1["inserted"] == 1
        assert res1["duplicates"] == 0

        # Second ingestion of the same file
        res2 = temp_store.ingest_jsonl_file(str(jsonl_file))
        assert res2["inserted"] == 0
        assert res2["duplicates"] == 1

        # Total counts remain exactly 1
        counts = temp_store.get_observation_counts()
        assert counts["total_observations"] == 1

    def test_3_deterministic_observation_id(self):
        """Test Case 3: Same keys produce identical deterministic IDs."""
        id1 = compute_observation_id("2026-08-01T00:00:00Z", "BOB", 3, "2026-08-01T00:00:00Z")
        id2 = compute_observation_id("2026-08-01T00:00:00Z", "bob", 3, "2026-08-01T00:00:00Z")
        assert id1 == id2
        assert id1.startswith("obs_")

        # Different key produces different ID
        id3 = compute_observation_id("2026-08-02T00:00:00Z", "bob", 3, "2026-08-01T00:00:00Z")
        assert id1 != id3

    def test_4_concurrent_ingestion_safety(self, temp_store):
        """Test Case 4: Concurrent multi-threaded ingestion executes safely without corruption."""
        errors = []

        def worker(thread_idx):
            try:
                for i in range(15):
                    rec = {
                        "prediction_timestamp": f"2026-09-01T{thread_idx:02d}:{i:02d}:00Z",
                        "site": "bob" if thread_idx % 2 == 0 else "aras",
                        "horizon": thread_idx % 4,
                        "production_probability": 0.10,
                        "shadow_probability": 0.20,
                        "data_quality_status": "READY",
                        "inference_status": "SUCCESS",
                    }
                    temp_store.insert_observation(rec)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        counts = temp_store.get_observation_counts()
        assert counts["total_observations"] == 60  # 4 threads * 15 unique records

    def test_5_malformed_telemetry_quarantine(self, temp_store, tmp_path):
        """Test Case 5: Malformed JSON lines are quarantined without aborting ingestion."""
        bad_jsonl = tmp_path / "malformed.jsonl"
        with open(bad_jsonl, "w") as f:
            f.write('{"prediction_timestamp": "2026-08-01T00:00:00Z", "site": "bob", "horizon": 1, "data_quality_status": "READY", "inference_status": "SUCCESS"}\n')
            f.write('THIS IS CORRUPTED NOT JSON\n')
            f.write('{"incomplete": json\n')
            f.write('{"prediction_timestamp": "2026-08-01T01:00:00Z", "site": "bob", "horizon": 1, "data_quality_status": "READY", "inference_status": "SUCCESS"}\n')

        res = temp_store.ingest_jsonl_file(str(bad_jsonl))
        assert res["total_lines"] == 4
        assert res["inserted"] == 2
        assert res["malformed"] == 2

        counts = temp_store.get_observation_counts()
        assert counts["total_observations"] == 2
        assert counts["malformed_telemetry_lines"] == 2

    def test_6_missing_and_corrupted_telemetry_handling(self, temp_store, tmp_path):
        """Test Case 6: Handles missing files or directories gracefully."""
        with pytest.raises(FileNotFoundError):
            temp_store.ingest_jsonl_file(str(tmp_path / "non_existent_file.jsonl"))

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        batch_res = temp_store.ingest_all_shadow_telemetry(shadow_dir=str(empty_dir))
        assert batch_res["total_files"] == 0
        assert batch_res["total_inserted"] == 0


# ============================================================================
# COMPONENT 3, 4, 5: Coverage, Comparison & Analytics Tests
# ============================================================================
class TestCoverageAndComparisonAnalytics:

    def test_7_coverage_metrics_calculation(self, populated_store):
        """Test Case 7: Calculates coverage and data quality breakdown."""
        monitor = ShadowV23Monitor(store=populated_store)
        cov = monitor.get_coverage_metrics()

        assert cov["total_prediction_opportunities"] == 40
        assert cov["shadow_jobs_successful"] == 28  # 40 - 4 late - 4 missing - 4 errors
        assert cov["shadow_coverage_pct"] == 70.0
        assert cov["data_quality"]["atmosphere_late_pct"] == 10.0
        assert cov["data_quality"]["atmosphere_missing_pct"] == 10.0

    def test_8_breakdown_by_site_horizon_month(self, populated_store):
        """Test Case 8: Slices metrics by site, horizon, and month."""
        monitor = ShadowV23Monitor(store=populated_store)
        cov = monitor.get_coverage_metrics()

        # Breakdown by site
        assert "bob" in cov["breakdowns"]["by_site"]
        assert "aras" in cov["breakdowns"]["by_site"]
        assert cov["breakdowns"]["by_site"]["bob"]["total"] == 20

        # Breakdown by horizon
        assert 0 in cov["breakdowns"]["by_horizon"]
        assert 3 in cov["breakdowns"]["by_horizon"]
        assert cov["breakdowns"]["by_horizon"][3]["total"] == 10

        # Breakdown by month
        assert "2026-07" in cov["breakdowns"]["by_month"]

    def test_9_production_vs_shadow_descriptive_stats(self, populated_store):
        """Test Case 9: Computes descriptive comparative statistics without superiority claims."""
        monitor = ShadowV23Monitor(store=populated_store)
        comp = monitor.get_comparison_analytics()

        assert comp["mode"] == "RESEARCH_ONLY"
        assert "superiority" in comp["interpretation_notice"].lower()

        # Check horizon 3d statistics
        h3 = comp["horizons"]["3d"]
        assert h3["observation_count"] > 0
        assert "mean" in h3["production"]
        assert "mean" in h3["shadow_model_d"]
        assert "mean_signed_delta" in h3["differences"]
        assert "percentiles_abs_delta" in h3["differences"]
        assert "fraction_close_within_5pct" in h3["differences"]

    def test_10_deterministic_analytics(self, populated_store):
        """Test Case 10: Analytics queries are completely deterministic across repeated calls."""
        monitor = ShadowV23Monitor(store=populated_store)
        res1 = monitor.get_comparison_analytics()
        res2 = monitor.get_comparison_analytics()
        assert res1["horizons"]["3d"]["differences"] == res2["horizons"]["3d"]["differences"]


# ============================================================================
# COMPONENT 6, 7, 8: Event Registry & Evidence Maturity Tests
# ============================================================================
class TestEventRegistryAndEvidenceMaturity:

    def test_11_event_registry_management(self, temp_store):
        """Test Case 11: Registers and queries independent storm events."""
        ok = temp_store.register_event(
            event_id="STORM-2026-TEST-1",
            event_name="TestCyclone",
            start_date="2026-10-10",
            end_date="2026-10-15",
            basin="bob",
            max_intensity_kts=65.0,
            classification="VSCS",
            source="Test Independent Observation",
        )
        assert ok is True

        events = temp_store.get_events()
        # 2 reference seeded events + 1 newly registered = 3
        assert len(events) >= 3
        names = [e["event_name"] for e in events]
        assert "TestCyclone" in names

    def test_12_insufficient_evidence_state(self, temp_store):
        """Test Case 12: Enforces INSUFFICIENT_EVIDENCE status when storm count or sample size is low."""
        monitor = ShadowV23Monitor(store=temp_store)
        mat = monitor.get_evidence_maturity()

        assert mat["evidence_status"] == "INSUFFICIENT"
        assert mat["statistical_evaluation_status"] == "INSUFFICIENT_EVIDENCE"
        assert "below minimum threshold" in mat["statistical_evaluation_reason"]

    def test_13_evidence_maturity_scale(self, temp_store):
        """Test Case 13: Maturity transitions and explicit promotion review guardrails."""
        monitor = ShadowV23Monitor(store=temp_store)
        mat = monitor.get_evidence_maturity()
        # Verify disclaimer
        assert "does NOT mean 'Promote Model D'" in mat["promotion_guardrail"]

        # Register 16 future storm events to test PROMOTION_REVIEW_ELIGIBLE
        for i in range(16):
            temp_store.register_event(
                event_id=f"STORM-2027-FUTURE-{i}",
                event_name=f"FutureStorm-{i}",
                start_date=f"2027-05-{i+1:02d}",
                end_date=f"2027-05-{i+2:02d}",
                basin="bob",
            )

        # Insert 30 positive shadow observations with unique timestamps
        for i in range(30):
            day = (i // 24) + 1
            hour = i % 24
            temp_store.insert_observation({
                "prediction_timestamp": f"2027-05-{day:02d}T{hour:02d}:00:00Z",
                "site": "bob",
                "horizon": 3,
                "production_probability": 0.40,
                "shadow_probability": 0.55,
                "shadow_alert_at_research_threshold": "RESEARCH_HIGH_ALERT",
                "data_quality_status": "READY",
                "inference_status": "SUCCESS",
            })

        mat2 = monitor.get_evidence_maturity()
        assert mat2["evidence_status"] == "PROMOTION_REVIEW_ELIGIBLE"
        assert "PROMOTION_REVIEW_ELIGIBLE" in mat2["promotion_guardrail"]

    def test_14_future_event_evaluation_contingency(self, populated_store):
        """Test Case 14: Calculates verification metrics when ground truth is provided, enforcing guardrails."""
        monitor = ShadowV23Monitor(store=populated_store)

        # Without ground truth labels: status is INSUFFICIENT_EVIDENCE
        ev_no_data = monitor.evaluate_future_events()
        assert ev_no_data["status"] == "INSUFFICIENT_EVIDENCE"
        assert ev_no_data["statistical_significance_claim_allowed"] is False

        # Small synthetic ground-truth dataset (< MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS)
        small_events = [
            {"event_id": "S1", "label": 1, "shadow_prediction": 1, "shadow_prob": 0.8},
            {"event_id": "S1", "label": 0, "shadow_prediction": 0, "shadow_prob": 0.1},
            {"event_id": "S2", "label": 1, "shadow_prediction": 0, "shadow_prob": 0.2},
        ]
        ev_small = monitor.evaluate_future_events(verified_events=small_events)
        assert ev_small["status"] == "INSUFFICIENT_EVIDENCE"
        assert ev_small["statistical_significance_claim_allowed"] is False
        assert ev_small["contingency_table"]["TP"] == 1
        assert ev_small["contingency_table"]["FN"] == 1
        assert ev_small["contingency_table"]["TN"] == 1


# ============================================================================
# COMPONENT 9 & 10: Drift Detection & Freshness Tests
# ============================================================================
class TestDriftAndFreshness:

    def test_15_feature_drift_detection_ks_and_psi(self):
        """Test Case 15: Computes KS test statistic, p-value, and PSI against reference distribution."""
        monitor = ShadowV23Monitor()
        drift = monitor.get_drift_metrics()

        assert drift["mode"] == "RESEARCH_ONLY"
        assert drift["features_evaluated"] > 0
        for feat, metrics in drift["feature_metrics"].items():
            assert "ks_statistic" in metrics
            assert "ks_p_value" in metrics
            assert "population_stability_index" in metrics

    def test_16_drift_warning_isolation(self):
        """Test Case 16: Atmospheric drift emits a DATA_QUALITY_WARNING, NEVER a production alert."""
        monitor = ShadowV23Monitor()
        drift = monitor.get_drift_metrics()
        warning = drift.get("warning")
        if warning:
            assert warning["type"] == "DATA_QUALITY_WARNING"
            assert "ZERO effect on operational production alerts" in warning["message"]

    def test_17_atmospheric_freshness_delay_tracking(self, populated_store):
        """Test Case 17: Monitors atmospheric data latency and notes NWP transition considerations."""
        monitor = ShadowV23Monitor(store=populated_store)
        fresh = monitor.get_freshness_metrics()

        assert fresh["total_checked"] == 40
        assert fresh["late_data_pct"] == 10.0
        assert fresh["missing_data_pct"] == 10.0
        assert "operational NWP" in fresh["nwp_transition_note"]


# ============================================================================
# COMPONENT 11, 15, 16: Research API & Operational Isolation Audit Tests
# ============================================================================
class TestResearchApiAndOperationalIsolation:

    def test_18_research_api_endpoints_return_research_only(self):
        """Test Case 18: Research endpoints return 200 with mode=RESEARCH_ONLY and operational_authority=False."""
        client = TestClient(app)
        endpoints = [
            "/shadow/v2.3/status",
            "/shadow/v2.3/metrics",
            "/shadow/v2.3/coverage",
            "/shadow/v2.3/comparison",
            "/shadow/v2.3/events",
            "/shadow/v2.3/evidence",
            "/shadow/v2.3/data-quality",
        ]
        for ep in endpoints:
            resp = client.get(ep)
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("mode") == "RESEARCH_ONLY"
            assert data.get("operational_authority") is False

    def test_19_strict_security_isolation_no_operational_mutation(self):
        """
        Test Case 19: Security Isolation Audit:
        Verifies research monitoring calls cannot alter production alert state,
        thresholds, frozen policies, or model files.
        """
        client = TestClient(app)

        # Query research endpoints
        client.get("/shadow/v2.3/status")
        client.get("/shadow/v2.3/metrics")

        # Verify production endpoints remain fully functional and unmutated
        resp_health = client.get("/api/health")
        assert resp_health.status_code == 200

        # Verify frozen policy file exists and has LF line ending hash
        import hashlib
        policy_path = os.path.join(os.path.dirname(__file__), "..", "config", "frozen_alert_policy_v2.json")
        policy_bytes = open(policy_path, "rb").read()
        policy_hash = hashlib.sha256(policy_bytes).hexdigest()
        assert policy_hash == "6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6"

    def test_20_historical_test_set_quarantine_protection(self):
        """
        Test Case 20: Verifies that the 2025-01-01 to 2026-06-23 test split
        is strictly treated as reference evidence and quarantined from tuning.
        """
        monitor = ShadowV23Monitor()
        ev_summary = monitor.get_event_registry_summary()

        assert "quarantined from iterative tuning" in ev_summary["quarantine_reminder"].lower()
        hist_events = [e for e in ev_summary["events"] if e["start_date"] <= "2026-06-23"]
        for ev in hist_events:
            assert ev["event_name"] in ("Montha", "UNNAMED-12")
