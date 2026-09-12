import unittest
import os
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.parsers.tabular_parser import TabularParser
from app.services.historical_engine import HistoricalFeatureEngine
from app.services.prediction_service import PredictionService
from app.models.schemas import CustomObservationRequest, PredictionResponse


class TestCustomObservationPipeline(unittest.TestCase):
    """
    Regression Test Suite for Custom New-Data + ML Prediction Pipeline (Tests A - L).
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_csv_bytes = (
            b"date,latitude,longitude,thetao,so,uo,vo,zos,mlotst\n"
            b"2026-06-24,17.8,88.2,30.20,33.80,1.15,0.85,0.24,18.5\n"
        )
        cls.sample_req = CustomObservationRequest(
            date="2026-06-24",
            lat=17.8,
            lon=88.2,
            horizon_days=3,
            thetao=30.20,
            so=33.80,
            uo=1.15,
            vo=0.85,
            zos=0.24,
            mlotst=18.5,
        )

    def test_a_single_row_csv_parsing(self):
        """Test A: TabularParser extracts all Copernicus columns and preserves exact date without corruption."""
        df, meta = TabularParser.parse_csv_from_bytes(self.sample_csv_bytes, "test_obs.csv")
        custom_obs = meta.get("custom_observation")
        self.assertIsNotNone(custom_obs, "custom_observation should be extracted from CSV")
        self.assertEqual(custom_obs["date"], "2026-06-24", "Date must be exactly 2026-06-24, not 2024-06-24")
        self.assertAlmostEqual(custom_obs["lat"], 17.8)
        self.assertAlmostEqual(custom_obs["lon"], 88.2)
        self.assertAlmostEqual(custom_obs["thetao"], 30.20)
        self.assertAlmostEqual(custom_obs["so"], 33.80)
        self.assertAlmostEqual(custom_obs["uo"], 1.15)
        self.assertAlmostEqual(custom_obs["vo"], 0.85)
        self.assertAlmostEqual(custom_obs["zos"], 0.24)
        self.assertAlmostEqual(custom_obs["mlotst"], 18.5)

    def test_b_upload_response_contains_custom_observation(self):
        """Test B: /api/upload endpoint returns custom_observation in UploadResponse."""
        response = self.client.post(
            "/api/upload",
            files={"file": ("test_obs.csv", self.sample_csv_bytes, "text/csv")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("custom_observation", data)
        obs = data["custom_observation"]
        self.assertIsNotNone(obs)
        self.assertEqual(obs["date"], "2026-06-24")
        self.assertEqual(obs["thetao"], 30.20)

    def test_c_historical_feature_engine_custom_current(self):
        """Test C: compute_comparison_with_custom_current generates exactly 101 features with custom current values."""
        custom_vals = {
            "thetao": 30.20,
            "so": 33.80,
            "uo": 1.15,
            "vo": 0.85,
            "zos": 0.24,
            "mlotst": 18.5,
        }
        res = HistoricalFeatureEngine.compute_comparison_with_custom_current(
            date_str="2026-06-24",
            custom_values=custom_vals,
            mode="point",
            lat=17.8,
            lon=88.2,
        )
        fv = res.feature_vector
        self.assertEqual(len(fv), 101, f"Expected 101 features, got {len(fv)}")
        self.assertAlmostEqual(fv["temp_current"], 30.20, places=2)
        self.assertAlmostEqual(fv["sal_current"], 33.80, places=2)
        self.assertAlmostEqual(fv["cur_u_current"], 1.15, places=2)
        self.assertAlmostEqual(fv["cur_v_current"], 0.85, places=2)
        self.assertAlmostEqual(fv["ssh_current"], 0.24, places=2)
        self.assertAlmostEqual(fv["mld_current"], 18.5, places=1)
        expected_speed = np.sqrt(1.15**2 + 0.85**2)
        self.assertAlmostEqual(fv["cur_current"], expected_speed, places=2)

    def test_d_rolling_windows_incorporate_custom_current(self):
        """Test D: 7d, 14d, 30d rolling windows incorporate custom current measurement."""
        custom_vals = {"thetao": 35.0}  # Noticeable heat spike
        res = HistoricalFeatureEngine.compute_comparison_with_custom_current(
            date_str="2026-06-24",
            custom_values=custom_vals,
            mode="point",
            lat=17.8,
            lon=88.2,
        )
        self.assertIn("7d", res.variables["temp"].windows)
        w7 = res.variables["temp"].windows["7d"]
        self.assertEqual(w7.window_end, "2026-06-24")
        self.assertEqual(w7.window_days, 7)
        self.assertGreater(w7.max, 34.0, "Spike value should be reflected in 7d max")

    def test_e_prediction_service_custom_observation(self):
        """Test E: PredictionService.predict_with_custom_observation succeeds on 2026-06-24."""
        pred = PredictionService.predict_with_custom_observation(self.sample_req)
        self.assertEqual(pred.status, "success")
        self.assertEqual(pred.date, "2026-06-24")
        self.assertIn(pred.warning_level, ["NO_ALERT", "WATCH", "HIGH_ALERT"])
        self.assertGreaterEqual(pred.probability, 0.0)
        self.assertLessEqual(pred.probability, 1.0)

    def test_f_observed_state_contains_custom_values(self):
        """Test F: PredictionResponse observed_state contains custom measurements, not Copernicus defaults."""
        pred = PredictionService.predict_with_custom_observation(self.sample_req)
        obs = pred.observed_state
        self.assertAlmostEqual(obs["sea_surface_temperature_c"], 30.20, places=1)
        self.assertAlmostEqual(obs["sea_surface_salinity_psu"], 33.80, places=1)
        self.assertAlmostEqual(obs["sea_surface_height_m"], 0.24, places=2)
        self.assertAlmostEqual(obs["mixed_layer_depth_m"], 18.5, places=1)
        self.assertEqual(obs["provenance"], "Custom In-situ Observation (User Upload)")

    def test_g_data_quality_source_dataset_and_compatibility(self):
        """Test G: data_quality accurately reports source dataset and 101/101 compatibility."""
        pred = PredictionService.predict_with_custom_observation(self.sample_req)
        dq = pred.data_quality
        self.assertTrue(dq["is_available"])
        self.assertEqual(dq["missing_feature_count"], 0)
        self.assertEqual(dq["source_dataset"], "Custom Observation + Copernicus Rolling Context")
        self.assertIn("101/101", dq["model_feature_compatibility"])

    def test_h_out_of_range_date_returns_insufficient_data(self):
        """Test H: Custom prediction for date out of range returns status='insufficient_data' without crashing."""
        req_future = CustomObservationRequest(
            date="2027-01-01",
            lat=17.8,
            lon=88.2,
            horizon_days=3,
            thetao=30.0,
        )
        pred = PredictionService.predict_with_custom_observation(req_future)
        self.assertEqual(pred.status, "insufficient_data")
        self.assertEqual(pred.probability, 0.0)
        self.assertEqual(pred.warning_level, "NO_ALERT")
        self.assertEqual(pred.data_quality["missing_feature_count"], 101)

    def test_i_api_predict_custom_endpoint(self):
        """Test I: POST /api/prediction/predict-custom HTTP endpoint works end-to-end."""
        response = self.client.post(
            "/api/prediction/predict-custom",
            json=self.sample_req.model_dump()
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["date"], "2026-06-24")
        self.assertEqual(data["observed_state"]["sea_surface_temperature_c"], 30.20)

    def test_j_date_preservation_no_corruption(self):
        """Test J: Custom date 2026-06-24 is preserved and never converted to 2024-06-24."""
        response = self.client.post(
            "/api/prediction/predict-custom",
            json=self.sample_req.model_dump()
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotEqual(data["date"], "2024-06-24", "Date must NOT be corrupted to 2024-06-24")
        self.assertEqual(data["date"], "2026-06-24")

    def test_k_no_historical_contamination(self):
        """Test K: Custom observation prediction is isolated from prior historical date selection."""
        # Request with 2026-06-24 returns 2026-06-24 regardless of any prior state
        pred = PredictionService.predict_with_custom_observation(self.sample_req)
        self.assertEqual(pred.date, "2026-06-24")
        self.assertIn("Custom Observation", pred.data_quality["source_dataset"])

    def test_l_all_four_horizons_with_custom_observation(self):
        """Test L: All 4 horizons (0d, 1d, 2d, 3d) evaluate successfully with custom observation."""
        for h in [0, 1, 2, 3]:
            req = CustomObservationRequest(
                date="2026-06-24",
                lat=17.8,
                lon=88.2,
                horizon_days=h,
                thetao=30.20,
                so=33.80,
                uo=1.15,
                vo=0.85,
                zos=0.24,
                mlotst=18.5,
            )
            pred = PredictionService.predict_with_custom_observation(req)
            self.assertEqual(pred.status, "success", f"Failed for horizon {h}d")
            self.assertEqual(pred.horizon_days, h)
            self.assertGreaterEqual(pred.probability, 0.0)


if __name__ == "__main__":
    unittest.main()
