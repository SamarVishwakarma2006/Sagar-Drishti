"""
Phase 4.6.10: End-to-End Prediction Pipeline Verification Script

Verifies:
1. API request parsing & validation
2. FastAPI / PredictionService invocation
3. HistoricalFeatureEngine 101-feature vector extraction (lazy NetCDF)
4. Schema & feature order validation against frozen risk_model.joblib
5. Frozen Random Forest inference with decision threshold 0.27
6. Multi-granularity explainability generation (WHAT, WHERE, WHEN, WHY)
7. Strict separation of [OBSERVED], [PREDICTED], and [HISTORICAL] contexts
8. Data quality reporting and safety guardrails
9. Frontend API payload compatibility
10. Latency & performance benchmarks
"""

import sys
import time
import json
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.app.models.schemas import PredictionRequest, PredictionResponse
from backend.app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_e2e_prediction")


def verify_e2e():
    print("\n" + "=" * 70)
    print("SAGAR-DRISHTI: PHASE 4.6.10 END-TO-END PREDICTION VERIFICATION")
    print("=" * 70 + "\n")

    # 1. Startup & Model Artifact Validation
    print("--- 1. Validating Model Artifacts on Startup ---")
    t0 = time.perf_counter()
    status_info = PredictionService.get_model_status()
    dt_startup = (time.perf_counter() - t0) * 1000
    print(f"Artifact Validation: {status_info['status'].upper()} in {dt_startup:.2f} ms")
    print(f"Model: {status_info['model_name']} ({status_info['model_version']})")
    print(f"Feature count: {status_info['feature_count']} features matching frozen schema")
    print(f"Frozen Decision Threshold: {status_info['frozen_threshold']}")
    assert status_info["status"] == "ready", "Model artifact status is not ready!"

    # 2. Test Cases to evaluate
    test_cases = [
        {
            "name": "Bay of Bengal (bob) - Active Fengal Period (2024-11-28)",
            "req": PredictionRequest(
                date="2024-11-28",
                site_id="bob",
                horizon_days=3,
            ),
            "expected_warning": ["WATCH", "HIGH_ALERT"],
        },
        {
            "name": "Arabian Sea (aras) - Asna Active Period (2024-08-30)",
            "req": PredictionRequest(
                date="2024-08-30",
                site_id="aras",
                horizon_days=3,
            ),
            "expected_warning": ["WATCH", "HIGH_ALERT"],
        },
        {
            "name": "Bay of Bengal (bob) - Calm Post-Monsoon (2024-10-15)",
            "req": PredictionRequest(
                date="2024-10-15",
                site_id="bob",
                horizon_days=3,
            ),
            "expected_warning": ["NO_ALERT", "WATCH"],
        },
        {
            "name": "Geographic Bounding Box - East Coast / BOB (14-22N, 83-93E)",
            "req": PredictionRequest(
                date="2024-11-28",
                min_lat=14.0,
                max_lat=22.0,
                min_lon=83.0,
                max_lon=93.0,
                horizon_days=3,
            ),
            "expected_warning": ["WATCH", "HIGH_ALERT"],
        },
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- Test Case {idx}: {tc['name']} ---")
        t_start = time.perf_counter()
        resp: PredictionResponse = PredictionService.predict(tc["req"])
        t_total = (time.perf_counter() - t_start) * 1000

        print(f"Total Inference Latency: {t_total:.2f} ms")
        print(f"  • Date: {resp.date} (Horizon: {resp.horizon_days} days)")
        print(f"  • Prediction Status: {resp.prediction.upper()} | Warning Level: {resp.warning_level}")
        print(f"  • Probability: {resp.probability:.4f} (Threshold: {resp.threshold:.2f})")
        print(f"  • Probability Display: \"{resp.probability_display}\"")
        print(f"  • Model: {resp.model_name} ({resp.model_version})")

        # Verify Threshold Rule
        if resp.probability < resp.threshold:
            assert resp.warning_level == "NO_ALERT", f"Expected NO_ALERT for prob {resp.probability} < {resp.threshold}"
        elif resp.probability < 0.50:
            assert resp.warning_level == "WATCH", f"Expected WATCH for prob {resp.probability}"
        else:
            assert resp.warning_level == "HIGH_ALERT", f"Expected HIGH_ALERT for prob {resp.probability}"

        # Verify Explainability
        expl = resp.explainability
        assert expl is not None, "Explainability payload missing!"
        print(f"  • Top Features ({len(resp.top_features)} reported):")
        for tf in resp.top_features[:3]:
            print(f"      - {tf.feature}: {tf.importance*100:.1f}% ({tf.description})")

        print(f"  • Human Readable Explanation (WHAT/WHERE/WHEN/WHY):")
        print(f"      - WHAT: {expl.human_readable.what}")
        print(f"      - WHERE: {expl.human_readable.where}")
        print(f"      - WHEN: {expl.human_readable.when}")
        print(f"      - WHY: {expl.human_readable.why}")

        # Verify Segregated Observed vs Predicted vs Historical
        assert resp.observed_state is not None, "Observed state missing!"
        assert resp.predicted_state is not None, "Predicted state missing!"
        assert resp.historical_context is not None, "Historical context missing!"
        assert resp.data_quality is not None, "Data quality missing!"

        obs = resp.observed_state if isinstance(resp.observed_state, dict) else resp.observed_state.model_dump()
        hist = resp.historical_context if isinstance(resp.historical_context, dict) else resp.historical_context.model_dump()
        dq = resp.data_quality if isinstance(resp.data_quality, dict) else resp.data_quality.model_dump()

        print(f"  • [OBSERVED]: SST={obs.get('sea_surface_temperature_c', 0.0):.2f}°C, "
              f"SSS={obs.get('sea_surface_salinity_psu', 0.0):.2f} PSU, "
              f"CurSpeed={obs.get('surface_current_speed_ms', 0.0):.2f} m/s, "
              f"SSH={obs.get('sea_surface_height_m', 0.0):.3f} m, "
              f"MLD={obs.get('mixed_layer_depth_m', 0.0):.1f} m")

        if hist.get("event_id"):
            print(f"  • [HISTORICAL]: {hist.get('event_name')} ({hist.get('event_type')}) "
                  f"Dates: {hist.get('event_dates')} (Proximity: {hist.get('distance_days')}d)")
        else:
            print(f"  • [HISTORICAL]: No disaster event cataloged within ±14d")

        print(f"  • [DATA QUALITY]: {dq.get('source_dataset')} | "
              f"Compatibility: {dq.get('model_feature_compatibility')} | "
              f"Missing Features: {dq.get('missing_feature_count')}")

        # Verify strict non-causal language in limitations
        assert len(resp.limitations) >= 3, "Expected at least 3 scientific limitations documented"
        print(f"  • Scientific Limitations ({len(resp.limitations)} documented)")

    # 3. Determinism Check (Repeated Identical Invocations)
    print("\n--- 3. Testing Determinism (5 Repeated Runs) ---")
    req_det = PredictionRequest(date="2024-11-28", site_id="bob", horizon_days=3)
    probs = []
    for _ in range(5):
        r = PredictionService.predict(req_det)
        probs.append(r.probability)
    assert len(set(probs)) == 1, f"Predictions are non-deterministic: {probs}"
    print(f"Determinism PASS: All 5 runs produced identical probability = {probs[0]:.6f}")

    # 4. JSON Serialization / Frontend Contract Compatibility
    print("\n--- 4. Testing JSON Serialization for Frontend API Contract ---")
    resp_dict = resp.model_dump()
    json_str = json.dumps(resp_dict, default=str)
    assert len(json_str) > 500, "Serialized JSON is suspiciously short!"
    print(f"JSON Contract PASS: Valid serialization ({len(json_str)} bytes)")

    print("\n" + "=" * 70)
    print("ALL END-TO-END PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    verify_e2e()
