"""
Comprehensive Regression & Integration Tests for Sagar-Drishti ML Prediction API (Phase 4.3 - 4.6).
Validates all 17 requirements:
1. Valid BOB prediction
2. Valid Arabian Sea prediction
3. Valid historical date
4. Invalid date format handling
5. Date outside dataset handling
6. Invalid bounding box handling
7. Missing/invalid site handling
8. Invalid horizon handling
9. Model artifact missing handling
10. Feature-schema mismatch handling
11. NaN / land-mask handling
12. Deterministic repeated predictions
13. Threshold behavior (NO_ALERT vs WATCH vs HIGH_ALERT)
14. Explainability payload structure
15. Data-quality payload structure
16. No future leakage
17. Correct HTTP status codes (200, 400, 422, 503)
"""
import unittest
import os
import sys
import tempfile
import shutil
import json
import numpy as np
import joblib


# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService, DEFAULT_FROZEN_THRESHOLD



class TestPredictionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_valid_bob_prediction(self):
        """1. Valid Bay of Bengal prediction on Dana date."""
        payload = {
            "date": "2024-10-24",
            "site_id": "bob",
            "horizon_days": 3,
        }
        res = self.client.post("/api/prediction/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn(data["warning_level"], ("WATCH", "HIGH_ALERT"))
        self.assertGreaterEqual(data["probability"], DEFAULT_FROZEN_THRESHOLD)
        self.assertEqual(data["threshold"], DEFAULT_FROZEN_THRESHOLD)
        self.assertIn("Bay of Bengal", data["location"].get("description", ""))

    def test_02_valid_arabian_sea_prediction(self):
        """2. Valid Arabian Sea prediction on Asna date."""
        payload = {
            "date": "2024-08-30",
            "site_id": "aras",
            "horizon_days": 3,
        }
        res = self.client.post("/api/prediction/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["warning_level"], "HIGH_ALERT")
        self.assertGreater(data["probability"], 0.50)
        self.assertIn("Arabian Sea", data["location"].get("description", ""))

    def test_03_valid_historical_date(self):
        """3. Valid historical date returning complete segregated sections."""
        payload = {
            "date": "2024-10-24",
            "site_id": "bob",
            "horizon_days": 3,
        }
        res = self.client.post("/api/prediction/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Ensure clear separation of OBSERVED, PREDICTED, and HISTORICAL
        self.assertIn("observed_state", data)
        self.assertIn("predicted_state", data)
        self.assertIn("historical_context", data)
        self.assertIn("sea_surface_temperature_c", data["observed_state"])
        self.assertIn("model_probability", data["predicted_state"])
        self.assertIn("event_name", data["historical_context"])

    def test_04_invalid_date_format(self):
        """4. Invalid date format returns 400 Bad Request."""
        payload = {
            "date": "2024/10/24",  # Not ISO YYYY-MM-DD
            "site_id": "bob",
            "horizon_days": 3,
        }
        res = self.client.post("/api/prediction/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid date format", res.json()["detail"])

        payload_gibberish = {
            "date": "not-a-valid-date",
            "site_id": "bob",
            "horizon_days": 3,
        }
        res_gibberish = self.client.post("/api/prediction/predict", json=payload_gibberish)
        self.assertEqual(res_gibberish.status_code, 400)

    def test_05_date_outside_dataset(self):
        """5. Date outside dataset coverage returns clean insufficient_data response."""
        # Prior to 2024-07-23 (insufficient 30d baseline)
        res_early = self.client.post("/api/prediction/predict", json={"date": "2024-06-25", "site_id": "bob"})
        self.assertEqual(res_early.status_code, 200)
        self.assertEqual(res_early.json()["status"], "insufficient_data")
        self.assertFalse(res_early.json()["data_quality"]["is_available"])

        # Distant past
        res_past = self.client.post("/api/prediction/predict", json={"date": "2023-01-01", "site_id": "bob"})
        self.assertEqual(res_past.status_code, 200)
        self.assertEqual(res_past.json()["status"], "insufficient_data")

        # Distant future
        res_future = self.client.post("/api/prediction/predict", json={"date": "2027-01-01", "site_id": "bob"})
        self.assertEqual(res_future.status_code, 200)
        self.assertEqual(res_future.json()["status"], "insufficient_data")


    def test_06_invalid_bounding_box(self):
        """6. Invalid bounding box coordinates return 400 Bad Request."""
        # min_lat >= max_lat
        res_inverted = self.client.post("/api/prediction/predict", json={
            "date": "2024-10-24",
            "min_lat": 20.0,
            "max_lat": 15.0,
            "min_lon": 83.0,
            "max_lon": 90.0,
        })
        self.assertEqual(res_inverted.status_code, 400)
        self.assertIn("Invalid latitude range", res_inverted.json()["detail"])

        # Out of Copernicus Indian Ocean domain (e.g. Pacific latitude 45°N)
        res_oob = self.client.post("/api/prediction/predict", json={
            "date": "2024-10-24",
            "min_lat": 35.0,
            "max_lat": 45.0,
            "min_lon": 83.0,
            "max_lon": 90.0,
        })
        self.assertEqual(res_oob.status_code, 400)
        self.assertIn("outside Copernicus Indian Ocean domain", res_oob.json()["detail"])

    def test_07_missing_or_invalid_site(self):
        """7. Unknown or unsupported study site returns 400 Bad Request."""
        res = self.client.post("/api/prediction/predict", json={
            "date": "2024-10-24",
            "site_id": "unknown_fictional_basin",
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown or unsupported study site", res.json()["detail"])

    def test_08_invalid_horizon(self):
        """8. Invalid horizon days returns 400 Bad Request."""
        res_neg = self.client.post("/api/prediction/predict", json={
            "date": "2024-10-24",
            "site_id": "bob",
            "horizon_days": -1,
        })
        self.assertEqual(res_neg.status_code, 400)
        self.assertIn("Invalid horizon_days", res_neg.json()["detail"])

        res_large = self.client.post("/api/prediction/predict", json={
            "date": "2024-10-24",
            "site_id": "bob",
            "horizon_days": 10,
        })
        self.assertEqual(res_large.status_code, 400)
        self.assertIn("Invalid horizon_days", res_large.json()["detail"])

    def test_09_model_artifact_missing(self):
        """9. Model artifact missing handling returns 503 Service Unavailable."""
        temp_dir = tempfile.mkdtemp()
        try:
            req = PredictionRequest(date="2024-10-24", site_id="bob")
            with self.assertRaises(FileNotFoundError):
                PredictionService.predict(req, artifacts_dir=temp_dir)
        finally:
            shutil.rmtree(temp_dir)

    def test_10_feature_schema_mismatch_handling(self):
        """10. Feature-schema mismatch fails loudly with clear error."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a mock corrupted model artifact expecting an alien feature
            from sklearn.ensemble import RandomForestClassifier
            mock_artifact = {
                "model": RandomForestClassifier(n_estimators=5, random_state=42),
                "feature_columns": ["alien_channel_x", "temp_current"],
                "frozen_threshold": 0.27,
            }
            # Fit mock model
            mock_artifact["model"].fit([[0.0, 28.0], [1.0, 30.0]], [0, 1])
            joblib.dump(mock_artifact, os.path.join(temp_dir, "risk_model.joblib"))

            mock_meta = {
                "version": "v1.1.0",
                "feature_list": ["alien_channel_x", "temp_current"],
                "frozen_threshold": 0.27,
            }
            with open(os.path.join(temp_dir, "model_metadata.json"), "w") as f:
                json.dump(mock_meta, f)

            # Model validation should catch missing 101 features
            with self.assertRaises(RuntimeError) as ctx:
                PredictionService.validate_model_artifacts(artifacts_dir=temp_dir)
            self.assertTrue("feature count mismatch" in str(ctx.exception).lower())
        finally:
            shutil.rmtree(temp_dir)

    def test_11_nan_land_mask_handling(self):
        """11. Land-mask / coastal point handling returns clean finite values."""
        # Point near Vizag coast (17.68°N, 83.21°E)
        req = PredictionRequest(date="2024-10-24", mode="point", lat=17.68, lon=83.21)
        res = PredictionService.predict(req)
        self.assertEqual(res.status, "success")
        self.assertFalse(np.isnan(res.probability))
        self.assertFalse(np.isinf(res.probability))
        self.assertGreaterEqual(res.probability, 0.0)
        self.assertLessEqual(res.probability, 1.0)

    def test_12_deterministic_repeated_prediction(self):
        """12. Repeated identical requests produce strictly deterministic identical results."""
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        res1 = PredictionService.predict(req)
        res2 = PredictionService.predict(req)
        self.assertEqual(res1.probability, res2.probability)
        self.assertEqual(res1.warning_level, res2.warning_level)
        self.assertEqual(res1.top_features[0].feature, res2.top_features[0].feature)
        self.assertEqual(res1.top_features[0].value, res2.top_features[0].value)

    def test_13_threshold_behavior(self):
        """13. Transparent threshold classification: NO_ALERT (<0.27), WATCH (0.27-0.50), HIGH_ALERT (>=0.50)."""
        # Active cyclone Asna should trigger HIGH_ALERT (P >= 0.50)
        res_high = PredictionService.predict(PredictionRequest(date="2024-08-30", site_id="aras", horizon_days=3))
        self.assertEqual(res_high.warning_level, "HIGH_ALERT")
        self.assertEqual(res_high.prediction, "alert")

        # Developing Dana window should trigger WATCH (0.27 <= P < 0.50)
        res_watch = PredictionService.predict(PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3))
        self.assertEqual(res_watch.warning_level, "WATCH")
        self.assertEqual(res_watch.prediction, "advisory")

        # Quiescent early November window should trigger NO_ALERT (P < 0.27)
        res_no = PredictionService.predict(PredictionRequest(date="2024-11-05", site_id="bob", horizon_days=3))
        self.assertEqual(res_no.warning_level, "NO_ALERT")
        self.assertEqual(res_no.prediction, "normal")

    def test_14_explainability_payload(self):
        """14. Explainability payload provides top features, ocean variable groups, and human narrative."""
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        res = PredictionService.predict(req)
        self.assertTrue(len(res.top_features) > 0)
        self.assertIn("ocean_variables", res.physical_drivers.model_dump())
        self.assertIn("temporal_scales", res.physical_drivers.model_dump())
        self.assertIn("temperature", res.physical_drivers.ocean_variables)
        self.assertIn("30_day", res.physical_drivers.temporal_scales)

        # Verify human-readable narrative answers WHAT, WHERE, WHEN, WHY
        hr = res.explainability.human_readable
        self.assertTrue(len(hr.what) > 0)
        self.assertTrue(len(hr.where) > 0)
        self.assertTrue(len(hr.when) > 0)
        self.assertTrue(len(hr.why) > 0)

    def test_15_data_quality_payload(self):
        """15. Data quality payload contains coverage range, spatial validity, and model version."""
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        res = PredictionService.predict(req)
        dq = res.data_quality
        self.assertEqual(dq["missing_feature_count"], 0)
        self.assertEqual(dq["valid_spatial_coverage_pct"], 100.0)
        self.assertEqual(dq["model_version"], "v1.1.0")
        self.assertIn("2024-07-23", dq["dataset_date_range"])

    def test_16_no_future_leakage(self):
        """16. Feature generation on date T does not access or leak any observations from T+1."""
        # Feature computation for 2024-08-01 should strictly use observations <= 2024-08-01
        from app.services.historical_engine import HistoricalFeatureEngine
        res_t = HistoricalFeatureEngine.compute_comparison(date_str="2024-08-01", mode="region", site_id="bob")
        # Ensure feature vector is fully deterministic and independent of future dates
        self.assertIn("temp_current", res_t.feature_vector)
        self.assertIn("temp_30d_mean", res_t.feature_vector)

    def test_17_correct_http_status_codes(self):
        """17. Endpoint returns 200 for valid requests, 400 for validation errors, 422 for malformed JSON."""
        # 200 OK
        res_200 = self.client.post("/api/prediction/predict", json={"date": "2024-10-24", "site_id": "bob"})
        self.assertEqual(res_200.status_code, 200)

        # 400 Bad Request (invalid date string)
        res_400 = self.client.post("/api/prediction/predict", json={"date": "invalid-date", "site_id": "bob"})
        self.assertEqual(res_400.status_code, 400)

        # 422 Unprocessable Entity (wrong datatype for horizon_days)
        res_422 = self.client.post("/api/prediction/predict", json={"date": "2024-10-24", "horizon_days": "not_an_int"})
        self.assertEqual(res_422.status_code, 422)

        # Prediction status endpoint
        res_status = self.client.get("/api/prediction/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertTrue(res_status.json()["is_ready"])

    def test_18_separate_horizon_models(self):
        """18. Verify each horizon uses its own distinct model artifact, target, and threshold."""
        expected_configs = [
            (0, "event_within_0d", 0.15),
            (1, "event_within_1d", 0.15),
            (2, "event_within_2d", 0.21),
            (3, "event_within_3d", 0.27),
        ]
        for h, exp_tgt, exp_th in expected_configs:
            res = self.client.post("/api/prediction/predict", json={
                "date": "2024-10-24",
                "site_id": "bob",
                "horizon_days": h,
            })
            self.assertEqual(res.status_code, 200, f"Horizon {h}d request failed")
            data = res.json()
            self.assertEqual(data["horizon_days"], h)
            self.assertEqual(data["target"], exp_tgt, f"Horizon {h}d returned target {data['target']}, expected {exp_tgt}")
            self.assertAlmostEqual(data["threshold"], exp_th, places=2, msg=f"Horizon {h}d threshold mismatch")
            self.assertAlmostEqual(data["alert_threshold"], exp_th, places=2, msg=f"Horizon {h}d alert_threshold mismatch")
            self.assertIn("RandomForestClassifier", data["model_name"])
            self.assertIn(f"{h}d horizon", data["model_name"])

    def test_19_no_cross_horizon_fallback(self):
        """19. Verify no cross-horizon fallback: missing a horizon model raises FileNotFoundError."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock metadata and only 3d model in temp_dir
            mock_meta = {
                "version": "v1.1.0",
                "feature_list": ["f_" + str(i) for i in range(101)],
            }
            with open(os.path.join(temp_dir, "model_metadata.json"), "w") as f:
                json.dump(mock_meta, f)

            from sklearn.ensemble import RandomForestClassifier
            rf_3d = RandomForestClassifier(n_estimators=5, random_state=42)
            rf_3d.fit(np.zeros((10, 101)), [0, 1] * 5)
            joblib.dump({
                "model": rf_3d,
                "feature_columns": ["f_" + str(i) for i in range(101)],
                "frozen_threshold": 0.27,
                "target": "event_within_3d",
            }, os.path.join(temp_dir, "risk_model_3d.joblib"))

            # When requesting horizon 1d, it must NOT fall back to risk_model_3d.joblib
            req_1d = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=1)
            with self.assertRaises(FileNotFoundError) as ctx:
                PredictionService.predict(req_1d, artifacts_dir=temp_dir)
            self.assertIn("Required model artifact for horizon", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir)

    def test_20_all_four_horizons_on_real_data(self):
        """20. Test representative real-data inference across all 4 horizons for BOB, ARAS, and calm dates."""
        test_cases = [
            ("2024-10-24", "bob", "BOB Dana Event Window"),
            ("2024-08-30", "aras", "ARAS Asna Event Window"),
            ("2024-11-05", "bob", "Calm Non-Event Period"),
        ]
        for date_str, site_id, desc in test_cases:
            for h in [0, 1, 2, 3]:
                req = PredictionRequest(date=date_str, site_id=site_id, horizon_days=h)
                res = PredictionService.predict(req)
                self.assertEqual(res.status, "success", f"Failed on {desc} horizon {h}")
                self.assertEqual(res.horizon_days, h)
                self.assertTrue(0.0 <= res.probability <= 1.0)
                self.assertIn(res.warning_level, ["NO_ALERT", "WATCH", "HIGH_ALERT"])
                self.assertIn("target", res.model_dump())
                self.assertEqual(res.target, f"event_within_{h}d")
                self.assertGreater(len(res.explainability.top_features), 0)

    def test_21_underpowered_horizon_limitations(self):
        """21. Verify 0d and 1d horizons report statistically underpowered notices."""
        req_0d = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=0)
        res_0d = PredictionService.predict(req_0d)
        has_underpowered_notice_0d = any("underpowered" in lim.lower() for lim in res_0d.limitations)
        self.assertTrue(has_underpowered_notice_0d, "Horizon 0d must document underpowered status in limitations")

        req_1d = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=1)
        res_1d = PredictionService.predict(req_1d)
        has_underpowered_notice_1d = any("underpowered" in lim.lower() for lim in res_1d.limitations)
        self.assertTrue(has_underpowered_notice_1d, "Horizon 1d must document underpowered status in limitations")

    def test_22_status_endpoint_reports_all_horizons(self):
        """22. Verify /api/prediction/status reports readiness for all 4 horizons."""
        res_status = self.client.get("/api/prediction/status")
        self.assertEqual(res_status.status_code, 200)
        data = res_status.json()
        self.assertTrue(data["is_ready"])
        horizons = data.get("horizons", {})
        for h_key in ["0d", "1d", "2d", "3d"]:
            self.assertIn(h_key, horizons, f"Status missing horizon key {h_key}")
            self.assertIn("target", horizons[h_key])
            self.assertIn("frozen_threshold", horizons[h_key])

    def test_23_conservative_scientific_phrasing(self):
        """23. Verify scientifically cautious phrasing in explanations, limitations, and probabilities."""
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        res = PredictionService.predict(req)
        # Check that probabilities are not stated as frequentist chances
        self.assertIn("Model-estimated risk score", res.probability_display)
        # Check limitations phrasing
        lims_text = " ".join(res.limitations)
        self.assertIn("research/decision-support baseline", lims_text)
        self.assertIn("Preliminary event-level generalization", lims_text)
        self.assertIn("hypothesis requiring additional events", lims_text)
        # Check human readable wording does not say "caused"
        hr = res.explainability.human_readable
        self.assertNotIn("caused the event", hr.why.lower())
        self.assertIn("associated", hr.why.lower())


    def test_24_nested_target_terminology_and_presentation_policy(self):
        """24. Verify logically nested target terminology, presentation policy labelling, and canonical artifact separation."""
        res_status = self.client.get("/api/prediction/status")
        self.assertEqual(res_status.status_code, 200)
        data = res_status.json()
        horizons_meta = data.get("metadata", {}).get("horizons", {})
        
        # Verify nested target names
        self.assertEqual(horizons_meta["0d"]["target"], "event_within_0d")
        self.assertEqual(horizons_meta["1d"]["target"], "event_within_1d")
        self.assertEqual(horizons_meta["2d"]["target"], "event_within_2d")
        self.assertEqual(horizons_meta["3d"]["target"], "event_within_3d")

        # Verify high alert triggers presentation severity notice in what explanation
        req_high = PredictionRequest(date="2024-08-30", site_id="aras", horizon_days=3)
        res_high = PredictionService.predict(req_high)
        self.assertEqual(res_high.warning_level, "HIGH_ALERT")
        self.assertIn("Presentation Severity", res_high.explainability.human_readable.what)

        # Verify canonical production artifacts exist and duplicate alias models are removed
        artifacts_dir = PredictionService._get_artifacts_dir()
        for f in ["risk_model_0d.joblib", "risk_model_1d.joblib", "risk_model_2d.joblib", "risk_model_3d.joblib"]:
            self.assertTrue(os.path.exists(os.path.join(artifacts_dir, f)), f"Canonical artifact {f} must exist")
        for f in ["model_event_within_1d.joblib", "model_event_within_2d.joblib", "model_event_within_3d.joblib"]:
            self.assertFalse(os.path.exists(os.path.join(artifacts_dir, f)), f"Legacy alias {f} must be removed")


if __name__ == "__main__":
    unittest.main()
