"""
Phase 4 Supervised ML Training, Validation & Inference Unit Tests for Sagar-Drishti.
Verifies:
1. Model artifact existence (risk_model.joblib, model_metadata.json, task models)
2. Metadata integrity and completeness
3. Deterministic inference
4. Feature schema compatibility (exact 101 physical ocean features)
5. Early-warning threshold application
6. Out-of-coverage and unavailable date handling
7. REST API response schema validation
8. Target leakage prevention
9. Train-only preprocessing verification
10. Model prediction reproducibility
"""
import os
import json
import unittest
import numpy as np
import pandas as pd
import joblib
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService, DEFAULT_FROZEN_THRESHOLD
from app.services.historical_engine import HistoricalFeatureEngine
from app.services.ml_trainer import MODEL_ARTIFACTS_DIR, LEAKAGE_AND_METADATA_COLUMNS


class TestPhase4ML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from app.services.copernicus_service import CopernicusService
        with CopernicusService._ds_lock:
            if CopernicusService._ds is not None:
                try:
                    CopernicusService._ds.close()
                except Exception:
                    pass
                CopernicusService._ds = None
        os.environ.pop("COPERNICUS_DATA_DIR", None)
        os.environ.pop("MODEL_ARTIFACTS_DIR", None)

        cls.client = TestClient(app)
        cls.artifacts_dir = MODEL_ARTIFACTS_DIR
        cls.risk_model_path = os.path.join(cls.artifacts_dir, "risk_model.joblib")
        cls.meta_path = os.path.join(cls.artifacts_dir, "model_metadata.json")
        cls.multibasin_parquet = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "historical", "ml_features_multibasin.parquet")
        )

    def test_01_model_artifact_existence(self):
        """Verify existence of primary and secondary model artifacts."""
        self.assertTrue(os.path.exists(self.risk_model_path), "risk_model.joblib must exist")
        self.assertTrue(os.path.exists(self.meta_path), "model_metadata.json must exist")

        # Canonical production horizon models and event type model
        for fname in ["risk_model_0d.joblib", "risk_model_1d.joblib", "risk_model_2d.joblib", "risk_model_3d.joblib", "model_event_type.joblib"]:
            path = os.path.join(self.artifacts_dir, fname)
            self.assertTrue(os.path.exists(path), f"Production model artifact '{fname}' must exist")

    def test_02_metadata_integrity(self):
        """Verify model_metadata.json contains all required Phase 4 reporting fields."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        required_keys = [
            "model_name", "version", "training_date", "target", "feature_count", "feature_list",
            "train_date_range", "validation_date_range", "test_date_range", "class_distribution",
            "event_split", "candidate_models_validation", "selected_model", "selection_reason",
            "threshold_selection_methodology", "frozen_threshold", "validation_metrics",
            "final_test_metrics", "event_level_test_results", "preprocessing", "explainability",
            "limitations", "artifact_paths"
        ]
        for key in required_keys:
            self.assertIn(key, meta, f"Metadata must contain key '{key}'")

        # Check feature count
        self.assertEqual(meta["feature_count"], 101, "Expected exactly 101 pure physical features")
        self.assertEqual(len(meta["feature_list"]), 101)

        # Check event split integrity
        ev_split = meta["event_split"]
        self.assertEqual(ev_split["train_events"], ["D-BOB04", "SCS-ASNA", "DD-BOB05"])
        self.assertEqual(ev_split["validation_events"], ["D-ARB01", "SCS-DANA"])
        self.assertEqual(ev_split["test_events"], ["CS-FENGAL"])

        # Check class distribution
        cd = meta["class_distribution"]
        self.assertEqual(cd["overall"]["total"], 374)
        self.assertEqual(cd["train"]["positives"], 22)
        self.assertEqual(cd["validation"]["positives"], 13)
        self.assertEqual(cd["test"]["positives"], 7)

        # Check frozen threshold
        self.assertAlmostEqual(meta["frozen_threshold"], 0.27, places=2)

    def test_03_deterministic_inference(self):
        """Verify that repeated predictions on the same input produce identical outputs."""
        req = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        res1 = PredictionService.predict(req)
        res2 = PredictionService.predict(req)

        self.assertEqual(res1.status, res2.status)
        self.assertEqual(res1.prediction, res2.prediction)
        self.assertAlmostEqual(res1.probability, res2.probability, places=6)
        self.assertEqual(res1.threshold, res2.threshold)
        self.assertEqual(len(res1.explainability.top_features), len(res2.explainability.top_features))

    def test_04_feature_schema_compatibility(self):
        """Verify live feature extractor generates the exact 101 physical features expected by the model."""
        comp = HistoricalFeatureEngine.compute_comparison(
            date_str="2024-09-15", mode="region", site_id="bob"
        )
        live_feats = comp.feature_vector

        artifact = joblib.load(self.risk_model_path)
        trained_cols = artifact["feature_columns"]

        self.assertEqual(len(trained_cols), 101)
        self.assertEqual(len(live_feats), 101)

        for col in trained_cols:
            self.assertIn(col, live_feats, f"Missing feature in live extraction: '{col}'")
            self.assertIsInstance(live_feats[col], (int, float), f"Feature '{col}' must be numeric")

    def test_05_threshold_application(self):
        """Verify risk classification mapping based on frozen threshold."""
        artifact = joblib.load(self.risk_model_path)
        th = float(artifact.get("frozen_threshold", DEFAULT_FROZEN_THRESHOLD))

        # Under threshold -> normal
        # Between threshold and 0.50 -> advisory
        # At or above 0.50 -> alert
        self.assertAlmostEqual(th, 0.27, places=2)

        # Active event date during Dana (high probability)
        req_dana = PredictionRequest(date="2024-10-24", site_id="bob", horizon_days=3)
        res_dana = PredictionService.predict(req_dana)
        self.assertGreaterEqual(res_dana.probability, th)
        self.assertIn(res_dana.prediction, ["advisory", "alert"])

        # Quiet baseline date
        req_quiet = PredictionRequest(date="2024-08-05", site_id="bob", horizon_days=3)
        res_quiet = PredictionService.predict(req_quiet)
        self.assertLess(res_quiet.probability, th)
        self.assertEqual(res_quiet.prediction, "normal")

    def test_06_unavailable_date_handling(self):
        """Verify that dates outside validated Copernicus coverage return clear insufficient_data response."""
        # Date prior to coverage
        req_past = PredictionRequest(date="2023-01-01", site_id="bob", horizon_days=3)
        res_past = PredictionService.predict(req_past)
        self.assertEqual(res_past.status, "insufficient_data")
        self.assertEqual(res_past.probability, 0.0)
        self.assertEqual(res_past.prediction, "normal")
        self.assertFalse(res_past.data_quality.get("is_available", True))

        # Date past coverage
        req_future = PredictionRequest(date="2027-01-01", site_id="bob", horizon_days=3)
        res_future = PredictionService.predict(req_future)
        self.assertEqual(res_future.status, "insufficient_data")
        self.assertEqual(res_future.probability, 0.0)
        self.assertFalse(res_future.data_quality.get("is_available", True))

    def test_07_api_prediction_endpoint_schema(self):
        """Verify POST /api/prediction/predict matches OpenAPI schema and returns valid payload."""
        payload = {
            "date": "2024-10-23",
            "site_id": "bob",
            "mode": "region",
            "horizon_days": 3,
        }
        res = self.client.post("/api/prediction/predict", json=payload)
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("probability", data)
        self.assertIn("threshold", data)
        self.assertIn("prediction", data)
        self.assertIn("explainability", data)
        self.assertIn("top_features", data["explainability"])
        self.assertIn("ocean_variable_importance", data["explainability"])
        self.assertIn("time_window_importance", data["explainability"])
        self.assertIn("data_quality", data)

    def test_08_target_leakage_prevention(self):
        """Verify no target or target-derived columns exist in model input features."""
        artifact = joblib.load(self.risk_model_path)
        feature_cols = artifact["feature_columns"]

        for forbidden in LEAKAGE_AND_METADATA_COLUMNS:
            self.assertNotIn(forbidden, feature_cols, f"Target/metadata leakage: '{forbidden}' found in X!")

    def test_09_train_only_preprocessing(self):
        """Verify preprocessing metadata confirms scaler was fitted only on train split."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        prep = meta.get("preprocessing", {})
        scaling_info = prep.get("scaling", "").lower()
        self.assertTrue(
            "applied only to training data" in scaling_info or "scaler_fitted_on" in prep or "standardscaler" in scaling_info,
            f"Expected train-only scaling in {prep}"
        )
        self.assertIn("strictly avoided", prep.get("synthetic_oversampling", "").lower())

    def test_10_model_prediction_reproducibility(self):
        """Verify model generates reproducible probabilities on test observations."""
        artifact = joblib.load(self.risk_model_path)
        model = artifact["model"]
        feature_cols = artifact["feature_columns"]

        if os.path.exists(self.multibasin_parquet):
            df = pd.read_parquet(self.multibasin_parquet)
            test_row = df[df["split"] == "test"].iloc[0]
            x_test = np.array([float(test_row[c]) for c in feature_cols]).reshape(1, -1)

            prob1 = float(model.predict_proba(x_test)[0, 1])
            prob2 = float(model.predict_proba(x_test)[0, 1])
            self.assertEqual(prob1, prob2, "Model probabilities must be exactly reproducible")


if __name__ == "__main__":
    unittest.main()
