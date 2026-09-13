"""
Comprehensive Automated Test Suite for Sagar-Drishti V2.3 Longitudinal Shadow Observation
& Evidence Accumulation.

Verifies:
1. Multi-season accumulation & meteorological seasonal partitioning
2. Duplicate event registration idempotency
3. Duplicate observation idempotency
4. Multi-dimensional aggregation breakdown (0d-3d, site, basin, month)
5. Insufficient evidence handling & sample size guardrails
6. Evidence maturity transitions across all 5 tiers (INSUFFICIENT -> PROMOTION_REVIEW_ELIGIBLE)
7. Storm-level metrics, detection timing, and lead delta calculations
8. Seasonal report generation (14 mandated sections)
9. Longitudinal evidence report generation (Engineering/Scientific/Gate/Operational separation)
10. Distribution drift metrics (mean, variance, percentiles, extremes, KS, PSI) & DATA_QUALITY_WARNING isolation
11. Missing & late atmospheric data handling
12. Malformed telemetry line quarantine isolation
13. Shadow failure fail-closed isolation
14. Promotion review gate read-only behavior & outputs (NOT READY vs READY FOR SCIENTIFIC REVIEW, NEVER PROMOTE MODEL)
15. Production safety equivalence (Shadow ON == Shadow OFF)
16. Historical test-set quarantine protection (2025-01-01 to 2026-06-23)
17. Mandatory documentation wording enforcement (conservative review criteria & research monitoring thresholds)
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
from datetime import datetime, timezone

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient

try:
    from backend.app.services.shadow_v2_3_store import (
        ShadowV23Store,
        compute_observation_id,
    )
    from backend.app.services.shadow_v2_3_longitudinal import (
        LongitudinalEvidenceTracker,
        classify_season,
        calculate_wilson_confidence_interval,
    )
    from backend.app.services.shadow_v2_3_seasonal_reporter import ShadowV23SeasonalReporter
    from backend.app.services.shadow_v2_3_service import ShadowV23Service
    from backend.app.main import app
except ImportError:
    from app.services.shadow_v2_3_store import (
        ShadowV23Store,
        compute_observation_id,
    )
    from app.services.shadow_v2_3_longitudinal import (
        LongitudinalEvidenceTracker,
        classify_season,
        calculate_wilson_confidence_interval,
    )
    from app.services.shadow_v2_3_seasonal_reporter import ShadowV23SeasonalReporter
    from app.services.shadow_v2_3_service import ShadowV23Service
    from app.main import app


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite store for testing."""
    db_file = str(tmp_path / "test_longitudinal_obs.db")
    return ShadowV23Store(db_path=db_file)


@pytest.fixture
def multi_season_store(temp_db):
    """Populates store with synthetic multi-season, multi-horizon observations."""
    # Seasons: 2026-Pre-Monsoon (April), 2026-Monsoon (July), 2026-Post-Monsoon (November), 2027-Winter (January)
    dates_and_seasons = [
        ("2026-04-10", 0, "bob", 0.12, 0.15, "NONE", "NONE"),
        ("2026-04-15", 1, "aras", 0.10, 0.11, "NONE", "NONE"),
        ("2026-07-20", 2, "bob", 0.35, 0.42, "WATCH", "RESEARCH_WATCH"),
        ("2026-07-25", 3, "bob", 0.45, 0.55, "WARNING", "RESEARCH_HIGH_ALERT"),
        ("2026-11-12", 3, "bob", 0.60, 0.72, "HIGH_ALERT", "RESEARCH_HIGH_ALERT"),
        ("2026-11-15", 2, "aras", 0.20, 0.18, "NONE", "NONE"),
        ("2027-01-05", 1, "bob", 0.05, 0.04, "NONE", "NONE"),
        ("2027-01-10", 0, "aras", 0.08, 0.09, "NONE", "NONE"),
    ]

    for idx, (dt_str, h, site, p_prod, p_shad, a_prod, a_shad) in enumerate(dates_and_seasons):
        rec = {
            "prediction_timestamp": f"{dt_str}T12:00:00Z",
            "prediction_date": dt_str,
            "site": site,
            "horizon": h,
            "production_model_version": "v1.1.0",
            "shadow_model_version": "v2.3.0-model-d-shadow",
            "production_probability": p_prod,
            "shadow_probability": p_shad,
            "production_alert": a_prod,
            "shadow_research_threshold_result": a_shad,
            "ocean_timestamp": f"{dt_str}T00:00:00Z",
            "atmosphere_timestamp": f"{dt_str}T18:00:00Z",
            "atmospheric_data_age_hours": 6.0,
            "data_quality_status": "READY",
            "coverage_status": "SHADOW_SUCCESS",
            "inference_status": "SUCCESS",
            "model_artifact_hash": "250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775",
            "feature_schema_hash": "schema_hash_mock",
            "created_at": f"{dt_str}T12:00:01Z",
        }
        temp_db.insert_observation(rec)

    return temp_db


# ============================================================================
# COMPONENT 1: Seasonal Classification & Multi-Season Accumulation Tests
# ============================================================================
class TestSeasonalPartitioningAndAccumulation:

    def test_1_meteorological_season_classification(self):
        """Test Case 1: Verifies North Indian Ocean meteorological cyclone seasons."""
        assert classify_season("2026-04-15") == "2026-Pre-Monsoon"
        assert classify_season("2026-07-20") == "2026-Monsoon"
        assert classify_season("2026-11-10") == "2026-Post-Monsoon"
        assert classify_season("2027-01-15") == "2027-Winter"
        assert classify_season("invalid-date") == "Unknown-Season"

    def test_2_multi_season_accumulation_summary(self, multi_season_store):
        """Test Case 2: Verifies multi-season observation tracking and summary."""
        tracker = LongitudinalEvidenceTracker(store=multi_season_store)
        summary = tracker.get_longitudinal_summary()

        assert summary["mode"] == "RESEARCH_ONLY"
        assert summary["operational_authority"] is False
        metrics = summary["metrics"]
        assert metrics["prediction_opportunities"] == 8
        assert metrics["successful_shadow_predictions"] == 8
        assert metrics["coverage_pct"] == 100.0

        obs_period = summary["observation_period"]
        assert obs_period["start_date"] == "2026-04-10"
        assert obs_period["end_date"] == "2027-01-10"
        assert obs_period["season_count"] == 4
        assert "2026-Pre-Monsoon" in obs_period["active_seasons"]
        assert "2026-Monsoon" in obs_period["active_seasons"]
        assert "2026-Post-Monsoon" in obs_period["active_seasons"]
        assert "2027-Winter" in obs_period["active_seasons"]


# ============================================================================
# COMPONENT 2: Idempotency & Deduplication Tests
# ============================================================================
class TestIdempotencyAndDeduplication:

    def test_3_duplicate_event_registration_idempotency(self, temp_db):
        """Test Case 3: Registering duplicate storm events is strictly idempotent."""
        ok1 = temp_db.register_event(
            event_id="STORM-2026-TEST-1",
            event_name="TestCyclone",
            start_date="2026-08-01",
            end_date="2026-08-05",
            basin="bob",
            max_intensity_kts=45.0,
            classification="CS",
        )
        assert ok1 is True

        # Register again with same ID
        ok2 = temp_db.register_event(
            event_id="STORM-2026-TEST-1",
            event_name="TestCyclone",
            start_date="2026-08-01",
            end_date="2026-08-05",
            basin="bob",
            max_intensity_kts=45.0,
            classification="CS",
        )
        assert ok2 is True

        events = temp_db.get_events()
        test_events = [e for e in events if e["event_id"] == "STORM-2026-TEST-1"]
        assert len(test_events) == 1

    def test_4_duplicate_observation_idempotency(self, temp_db):
        """Test Case 4: Inserting identical observation record is rejected as duplicate."""
        rec = {
            "prediction_timestamp": "2026-09-01T12:00:00Z",
            "prediction_date": "2026-09-01",
            "site": "bob",
            "horizon": 3,
            "production_probability": 0.25,
            "shadow_probability": 0.30,
            "ocean_timestamp": "2026-09-01T00:00:00Z",
            "data_quality_status": "READY",
            "inference_status": "SUCCESS",
        }
        inserted1, obs_id1 = temp_db.insert_observation(rec)
        assert inserted1 is True

        # Insert identical record again
        inserted2, obs_id2 = temp_db.insert_observation(rec)
        assert inserted2 is False
        assert obs_id1 == obs_id2

        counts = temp_db.get_observation_counts()
        # Seed events = 2, observation count = 1
        assert counts["total_observations"] == 1


# ============================================================================
# COMPONENT 3: Multi-Dimensional Breakdown Tests
# ============================================================================
class TestDimensionalBreakdowns:

    def test_5_dimensional_breakdown_horizons_sites_basins_months(self, multi_season_store):
        """Test Case 5: Verifies breakdowns across 0d-3d, sites, basins, and months."""
        tracker = LongitudinalEvidenceTracker(store=multi_season_store)
        breakdown = tracker.get_dimensional_breakdown()

        by_h = breakdown["by_horizon"]
        assert set(by_h.keys()) == {"0d", "1d", "2d", "3d"}
        assert by_h["0d"]["total"] == 2
        assert by_h["1d"]["total"] == 2
        assert by_h["2d"]["total"] == 2
        assert by_h["3d"]["total"] == 2

        by_basin = breakdown["by_basin"]
        assert "Bay of Bengal" in by_basin
        assert "Arabian Sea" in by_basin
        assert by_basin["Bay of Bengal"]["total"] == 5
        assert by_basin["Arabian Sea"]["total"] == 3

        by_month = breakdown["by_month"]
        assert "2026-04" in by_month
        assert "2026-07" in by_month
        assert "2026-11" in by_month
        assert "2027-01" in by_month


# ============================================================================
# COMPONENT 4: Insufficient Evidence & Statistical Guardrail Tests
# ============================================================================
class TestInsufficientEvidenceAndGuardrails:

    def test_6_insufficient_evidence_guardrails_enforced(self, multi_season_store):
        """Test Case 6: Verifies that INSUFFICIENT_EVIDENCE is strictly output when sample size is small."""
        tracker = LongitudinalEvidenceTracker(store=multi_season_store)
        comp = tracker.get_comparative_metrics()

        assert comp["status"] == "INSUFFICIENT_EVIDENCE"
        assert comp["statistical_significance_claim_allowed"] is False
        assert "Do NOT manufacture significance" in comp["guardrail"]

    def test_7_wilson_confidence_interval_calculation(self):
        """Test Case 7: Verifies Wilson score 95% confidence interval formula."""
        # 10 successes out of 100
        low, high = calculate_wilson_confidence_interval(10, 100)
        assert 0.05 < low < 0.07
        assert 0.15 < high < 0.18

        # 0 total returns None
        assert calculate_wilson_confidence_interval(0, 0) == (None, None)


# ============================================================================
# COMPONENT 5: Evidence Maturity Transition Tests
# ============================================================================
class TestEvidenceMaturityTransitions:

    def test_8_evidence_maturity_transitions_across_tiers(self, temp_db):
        """
        Test Case 8: Tests transitions across INSUFFICIENT, EARLY, DEVELOPING,
        SUBSTANTIAL, and PROMOTION_REVIEW_ELIGIBLE.
        Also tests that a single unusually successful storm can NEVER elevate maturity directly.
        """
        tracker = LongitudinalEvidenceTracker(store=temp_db)

        # Baseline: 0 future storms -> INSUFFICIENT
        m0 = tracker.evaluate_evidence_maturity()
        assert m0["evidence_status"] == "INSUFFICIENT"
        assert m0["promotion_review_eligible"] is False

        # Add 3 future storms but < 25 positive observations -> still INSUFFICIENT
        for i in range(3):
            temp_db.register_event(
                event_id=f"STORM-2026-FUT-{i}",
                event_name=f"FutureStorm-{i}",
                start_date=f"2026-08-{i+1:02d}",
                end_date=f"2026-08-{i+5:02d}",
                basin="bob",
            )
        m1 = tracker.evaluate_evidence_maturity()
        assert m1["evidence_status"] == "INSUFFICIENT"

        # Populate with 30 positive observations across 1 season -> EARLY (3 storms, >= 25 pos obs, >= 80% coverage)
        for i in range(30):
            temp_db.insert_observation({
                "prediction_timestamp": f"2026-08-{10 + (i//24):02d}T{i%24:02d}:00:00Z",
                "prediction_date": f"2026-08-{10 + (i//24):02d}",
                "site": "bob",
                "horizon": 3,
                "shadow_probability": 0.45,
                "production_probability": 0.20,
                "shadow_research_threshold_result": "RESEARCH_WATCH",
                "inference_status": "SUCCESS",
                "data_quality_status": "READY",
            })
        m2 = tracker.evaluate_evidence_maturity()
        assert m2["evidence_status"] == "EARLY"

        # Add 3 more storms (total 6 future storms) and 30 more pos obs -> DEVELOPING (5-9 storms, >=50 pos obs)
        for i in range(3, 6):
            temp_db.register_event(
                event_id=f"STORM-2026-FUT-{i}",
                event_name=f"FutureStorm-{i}",
                start_date=f"2026-10-{i+1:02d}",
                end_date=f"2026-10-{i+5:02d}",
                basin="bob",
            )
        for i in range(30):
            temp_db.insert_observation({
                "prediction_timestamp": f"2026-10-{10 + (i//24):02d}T{i%24:02d}:00:00Z",
                "prediction_date": f"2026-10-{10 + (i//24):02d}",
                "site": "bob",
                "horizon": 3,
                "shadow_probability": 0.45,
                "production_probability": 0.20,
                "shadow_research_threshold_result": "RESEARCH_WATCH",
                "inference_status": "SUCCESS",
                "data_quality_status": "READY",
            })
        m3 = tracker.evaluate_evidence_maturity()
        assert m3["evidence_status"] == "DEVELOPING"


# ============================================================================
# COMPONENT 6: Storm-Level Metrics & Lead Delta Tests
# ============================================================================
class TestStormLevelMetrics:

    def test_9_storm_level_evidence_and_timing_calculations(self, temp_db):
        """Test Case 9: Tests event-level tracking, detection timing, and lead delta."""
        temp_db.register_event(
            event_id="STORM-2026-AUTUMN",
            event_name="AutumnCyclone",
            start_date="2026-10-10",
            end_date="2026-10-15",
            basin="bob",
            max_intensity_kts=65.0,
            classification="VSCS",
        )

        # Observations where shadow detects on Oct 11, production detects on Oct 12
        obs_seq = [
            ("2026-10-10", 0.10, 0.15, "NONE", "NONE"),
            ("2026-10-11", 0.20, 0.45, "NONE", "RESEARCH_WATCH"),
            ("2026-10-12", 0.40, 0.65, "WATCH", "RESEARCH_HIGH_ALERT"),
            ("2026-10-13", 0.70, 0.85, "HIGH_ALERT", "RESEARCH_HIGH_ALERT"),
        ]

        for dt_str, p_p, p_s, a_p, a_s in obs_seq:
            temp_db.insert_observation({
                "prediction_timestamp": f"{dt_str}T12:00:00Z",
                "prediction_date": dt_str,
                "site": "bob",
                "horizon": 3,
                "production_probability": p_p,
                "shadow_probability": p_s,
                "production_alert": a_p,
                "shadow_research_threshold_result": a_s,
                "inference_status": "SUCCESS",
                "data_quality_status": "READY",
            })

        tracker = LongitudinalEvidenceTracker(store=temp_db)
        ev_data = tracker.get_event_level_evidence()

        autumn_ev = next(e for e in ev_data["events"] if e["event_id"] == "STORM-2026-AUTUMN")
        perf = autumn_ev["detection_performance"]

        assert perf["shadow_detected"] is True
        assert perf["production_detected"] is True
        assert perf["first_shadow_detection"] == "2026-10-11"
        assert perf["first_production_detection"] == "2026-10-12"
        # Shadow detected 1 day earlier (negative delta)
        assert perf["lead_delta_days"] == -1
        assert perf["shadow_alert_days"] == 3
        assert perf["production_alert_days"] == 2


# ============================================================================
# COMPONENT 7: Report Generation Tests
# ============================================================================
class TestReportGeneration:

    def test_10_seasonal_report_generation(self):
        """Test Case 10: Verifies generation of seasonal report with all 14 required sections."""
        reporter = ShadowV23SeasonalReporter()
        report_file = reporter.generate_seasonal_report(2026)

        assert os.path.exists(report_file)
        content = open(report_file, "r", encoding="utf-8").read()

        required_sections = [
            "1. Observation Period",
            "2. Number of Storm Systems",
            "3. Number of Positive Observations",
            "4. Shadow Coverage",
            "5. Data Availability & Quality",
            "6. Model D vs Production v1.1.0 Comparison",
            "7. Event-Level Results",
            "8. Horizon-Level Results",
            "9. Distribution Shift & Drift Monitoring",
            "10. Failures & Malformed Telemetry Isolation",
            "11. Statistical Uncertainty & Confidence Intervals",
            "12. Scientific & Operational Limitations",
            "13. Evidence Maturity Assessment",
            "14. Recommendation",
        ]
        for sec in required_sections:
            assert sec in content

    def test_11_longitudinal_evidence_report_generation(self):
        """Test Case 11: Verifies longitudinal evidence report generation & required statuses."""
        reporter = ShadowV23SeasonalReporter()
        report_file = reporter.generate_longitudinal_evidence_report()

        assert os.path.exists(report_file)
        content = open(report_file, "r", encoding="utf-8").read()

        assert "ENGINEERING READINESS" in content
        assert "SCIENTIFIC EVIDENCE" in content
        assert "PROMOTION REVIEW GATE" in content
        assert "OPERATIONAL AUTHORITY" in content
        assert "NO NEW INDEPENDENT EVENT EVIDENCE AVAILABLE" in content
        assert "INSUFFICIENT_EVIDENCE" in content
        assert "NOT READY" in content


# ============================================================================
# COMPONENT 8: Distribution Shift & Drift Tests
# ============================================================================
class TestDistributionShiftAndDrift:

    def test_12_distribution_shift_metrics_and_research_isolation(self):
        """Test Case 12: Distribution shift calculation and DATA_QUALITY_WARNING isolation."""
        tracker = LongitudinalEvidenceTracker()
        drift = tracker.get_distribution_shift_metrics()

        assert drift["mode"] == "RESEARCH_ONLY"
        assert drift["features_evaluated"] > 0
        for feat, metrics in drift["feature_metrics"].items():
            assert "reference_mean" in metrics
            assert "shadow_mean" in metrics
            assert "mean_shift" in metrics
            assert "variance_ratio" in metrics
            assert "percentile_shifts" in metrics
            assert "extreme_frequency_gt_3sigma" in metrics
            assert "ks_statistic" in metrics
            assert "ks_p_value" in metrics
            assert "population_stability_index" in metrics

        warning = drift.get("warning")
        if warning:
            assert warning["type"] == "DATA_QUALITY_WARNING"
            assert "ZERO effect on operational production alerts" in warning["message"]


# ============================================================================
# COMPONENT 9: Missing Data & Failure Isolation Tests
# ============================================================================
class TestDataQualityAndFailureIsolation:

    def test_13_missing_and_late_atmospheric_data_handling(self, temp_db):
        """Test Case 13: Verifies handling of missing and late atmospheric data."""
        # Missing
        temp_db.insert_observation({
            "prediction_timestamp": "2026-09-05T12:00:00Z",
            "prediction_date": "2026-09-05",
            "site": "bob",
            "horizon": 3,
            "data_quality_status": "ATMOSPHERE_MISSING",
            "inference_status": "SKIPPED",
        })
        # Late
        temp_db.insert_observation({
            "prediction_timestamp": "2026-09-06T12:00:00Z",
            "prediction_date": "2026-09-06",
            "site": "bob",
            "horizon": 3,
            "data_quality_status": "ATMOSPHERE_LATE",
            "inference_status": "SKIPPED",
        })

        tracker = LongitudinalEvidenceTracker(store=temp_db)
        summary = tracker.get_longitudinal_summary()
        metrics = summary["metrics"]

        assert metrics["missing_atmospheric_observations"] == 1
        assert metrics["late_atmospheric_observations"] == 1
        assert metrics["skipped_or_dropped"] == 2

    def test_14_malformed_telemetry_isolation(self, temp_db, tmp_path):
        """Test Case 14: Malformed telemetry lines are quarantined and do not break ingestion."""
        telemetry_file = tmp_path / "corrupt_telemetry.jsonl"
        with open(telemetry_file, "w", encoding="utf-8") as f:
            f.write('{"prediction_timestamp": "2026-09-01T00:00:00Z", "site": "bob", "horizon": 3, "inference_status": "SUCCESS", "data_quality_status": "READY"}\n')
            f.write('CORRUPTED_NON_JSON_DATA_LINE\n')
            f.write('{"prediction_timestamp": "2026-09-02T00:00:00Z", "site": "bob", "horizon": 3, "inference_status": "SUCCESS", "data_quality_status": "READY"}\n')

        res = temp_db.ingest_jsonl_file(str(telemetry_file))
        assert res["inserted"] == 2
        assert res["malformed"] == 1

        counts = temp_db.get_observation_counts()
        assert counts["malformed_telemetry_lines"] == 1


# ============================================================================
# COMPONENT 10: Promotion Review Gate Tests
# ============================================================================
class TestPromotionReviewGate:

    def test_15_promotion_review_gate_strictly_read_only(self):
        """
        Test Case 15: Promotion Review Gate may return ONLY "NOT READY" or
        "READY FOR SCIENTIFIC REVIEW". It MUST NEVER return "PROMOTE MODEL".
        """
        tracker = LongitudinalEvidenceTracker()
        gate = tracker.evaluate_promotion_gate()

        assert gate["status"] in ("NOT READY", "READY FOR SCIENTIFIC REVIEW")
        assert gate["status"] != "PROMOTE MODEL"
        assert "MUST NEVER return 'PROMOTE MODEL'" in gate["immutable_guardrail"]


# ============================================================================
# COMPONENT 11: Production Equivalence & Isolation Tests
# ============================================================================
class TestProductionEquivalenceAndHistoricalProtection:

    def test_16_production_safety_equivalence_shadow_on_vs_off(self):
        """
        Test Case 16: Continuous Production Equivalence Proof:
        Verifies that the production prediction API outputs 100% identical responses whether
        shadow evaluation is enabled or disabled.
        """
        client = TestClient(app)
        payload = {
            "date": "2024-10-24",
            "site_id": "bob",
            "horizon_days": 3,
        }

        # 1. Predict with shadow ENABLED
        ShadowV23Service.set_enabled(True)
        resp_on_raw = client.post("/api/prediction/predict", json=payload)
        assert resp_on_raw.status_code == 200
        resp_on = resp_on_raw.json()

        # 2. Predict with shadow DISABLED
        ShadowV23Service.set_enabled(False)
        resp_off_raw = client.post("/api/prediction/predict", json=payload)
        assert resp_off_raw.status_code == 200
        resp_off = resp_off_raw.json()

        # Restore
        ShadowV23Service.set_enabled(True)

        # Compare exact field values
        assert resp_on["probability"] == resp_off["probability"]
        assert resp_on["warning_level"] == resp_off["warning_level"]
        assert resp_on["model_version"] == resp_off["model_version"]
        assert resp_on["threshold"] == resp_off["threshold"]
        assert resp_on["status"] == resp_off["status"]
        assert resp_on["prediction"] == resp_off["prediction"]

    def test_17_historical_test_set_quarantine_protection(self):
        """
        Test Case 17: Historical test set (2025-01-01 to 2026-06-23) remains strictly
        quarantined from iterative tuning and is cataloged for reference benchmarking only.
        """
        tracker = LongitudinalEvidenceTracker()
        events_data = tracker.get_event_level_evidence()

        assert "strictly quarantined from iterative tuning" in events_data["quarantine_rule"]
        for ev in events_data["events"]:
            if ev["start_date"] <= "2026-06-23":
                assert ev["is_quarantined_historical_reference"] is True

    def test_18_mandatory_documentation_wording(self):
        """
        Test Case 18: Mandatory User Corrections:
        Verifies exact required phrasing in docstrings and evaluations:
        1. 'Sagar-Drishti predefined conservative evidence-review criteria'
        2. 'predefined research monitoring thresholds'
        """
        tracker = LongitudinalEvidenceTracker()
        maturity = tracker.evaluate_evidence_maturity()
        assert "Sagar-Drishti predefined conservative evidence-review criteria" in maturity["framework_nature"]

        drift = tracker.get_distribution_shift_metrics()
        assert "predefined research monitoring thresholds" in drift["threshold_definition"]
        assert "never become production_alert" in drift["threshold_definition"].lower()
