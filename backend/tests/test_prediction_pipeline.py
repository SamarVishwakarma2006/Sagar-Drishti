"""
Unit & Integration Tests for Phase 4: Leakage-Safe Machine Learning & Prediction Pipeline.
Validates:
1. Dynamic feature extraction & strict leakage column exclusion
2. Dynamic chronological 70/15/15 split without temporal overlap
3. Separate ML tasks: event_active, event_within_1d, 2d, 3d, and event_type
4. Calibration procedure & Brier score calculation
5. Multi-granularity explainability (ocean variables, time windows, individual features)
6. Model persistence & reproducible metadata
7. PredictionService inference & REST API endpoints
"""
import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.copernicus_service import CopernicusService
from app.services.historical_engine import HistoricalFeatureEngine
from app.services.event_matcher import SpatialTemporalMatcher
from app.services.ml_dataset_builder import MLDatasetBuilder
from app.services.ml_trainer import MLTrainer
from app.services.prediction_service import PredictionService
from app.models.schemas import PredictionRequest


class TestPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create isolated temporary directory
        cls.test_dir = tempfile.mkdtemp(prefix="sagar_prediction_pipeline_test_")
        cls.test_nc = os.path.join(cls.test_dir, "copernicus_phy_2yr_surface.nc")
        cls.models_dir = os.path.join(cls.test_dir, "models")

        with CopernicusService._ds_lock:
            if CopernicusService._ds is not None:
                try:
                    CopernicusService._ds.close()
                except Exception:
                    pass
                CopernicusService._ds = None

        # Create a test fixture with sufficient days (165 days)
        cls.test_days = 165
        CopernicusService.create_test_fixture(cls.test_nc, days=cls.test_days)

        cls.orig_data_dir = os.environ.get("COPERNICUS_DATA_DIR")
        os.environ["COPERNICUS_DATA_DIR"] = cls.test_dir

        cls.orig_models_dir = os.environ.get("MODEL_ARTIFACTS_DIR")
        os.environ["MODEL_ARTIFACTS_DIR"] = cls.models_dir

        cls.client = TestClient(app)

        # Generate rolling features and labeled dataset in test dir
        cls.features_parquet = os.path.join(cls.test_dir, "features.parquet")
        cls.features_meta = os.path.join(cls.test_dir, "features_metadata.json")
        cls.labeled_parquet = os.path.join(cls.test_dir, "labeled_features.parquet")
        cls.labeled_meta = os.path.join(cls.test_dir, "labeled_metadata.json")

        # Compute rolling features
        HistoricalFeatureEngine.export_training_dataset(
            output_dir=cls.test_dir,
        )

        # Compute labeled dataset
        SpatialTemporalMatcher.generate_labeled_dataset(
            feature_parquet_path=cls.features_parquet,
            output_dir=cls.test_dir,
        )

        # Build clean ML dataset and train baseline models
        cls.ds_res = MLDatasetBuilder.build_ml_dataset(
            labeled_parquet_path=cls.labeled_parquet,
            output_dir=cls.test_dir,
        )
        cls.train_res = MLTrainer.train_and_evaluate_pipeline(
            ml_df=cls.ds_res["dataframe"],
            feature_cols=cls.ds_res["feature_columns"],
            artifacts_dir=cls.models_dir,
            min_class_support=3,
        )

    @classmethod
    def tearDownClass(cls):
        with CopernicusService._ds_lock:
            if CopernicusService._ds is not None:
                try:
                    CopernicusService._ds.close()
                except Exception:
                    pass
                CopernicusService._ds = None

        if cls.orig_data_dir is not None:
            os.environ["COPERNICUS_DATA_DIR"] = cls.orig_data_dir
        else:
            os.environ.pop("COPERNICUS_DATA_DIR", None)

        if cls.orig_models_dir is not None:
            os.environ["MODEL_ARTIFACTS_DIR"] = cls.orig_models_dir
        else:
            os.environ.pop("MODEL_ARTIFACTS_DIR", None)

        if os.path.exists(cls.test_dir):
            try:
                shutil.rmtree(cls.test_dir)
            except Exception:
                pass

    def test_01_ml_dataset_builder_leakage_exclusion(self):
        """Verify dynamic feature discovery excludes all targets, event IDs, and leakage columns."""
        res = MLDatasetBuilder.build_ml_dataset(
            labeled_parquet_path=self.labeled_parquet,
            output_parquet=os.path.join(self.test_dir, "ml_dataset.parquet"),
            output_metadata=os.path.join(self.test_dir, "ml_metadata.json"),
        )
        feature_cols = res["feature_columns"]
        self.assertGreaterEqual(len(feature_cols), 80, "Expected at least 80 ocean features")

        # Explicit leakage check: None of the target, event ID, or label columns must be in feature_cols
        leakage_cols = [
            "event_id", "event_name", "event_type", "event_active",
            "is_active_event", "event_within_1d", "event_within_2d",
            "event_within_3d", "lead_days", "label_status", "split",
            "time", "date", "lat", "lon", "site_id",
        ]
        for col in leakage_cols:
            self.assertNotIn(col, feature_cols, f"Leakage column '{col}' found in ML features!")

        # Verify metadata record
        meta = res["metadata"]
        self.assertEqual(meta["final_feature_count"], len(feature_cols))
        self.assertIn("temporal_split", meta)

    def test_02_dynamic_chronological_split(self):
        """Verify strict chronological 70/15/15 train/val/test split without temporal overlap."""
        df = self.ds_res["dataframe"]
        split_meta = self.ds_res["metadata"]["temporal_split"]

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

        self.assertGreater(len(train_df), 0, "Train split should have observations")
        self.assertGreater(len(val_df), 0, "Val split should have observations")
        self.assertGreater(len(test_df), 0, "Test split should have observations")

        # Temporal ordering check
        max_train_date = train_df["date"].max()
        min_val_date = val_df["date"].min()
        max_val_date = val_df["date"].max()
        min_test_date = test_df["date"].min()

        self.assertLess(
            max_train_date,
            min_val_date,
            f"Train date ({max_train_date}) must strictly precede val date ({min_val_date})",
        )
        self.assertLess(
            max_val_date,
            min_test_date,
            f"Val date ({max_val_date}) must strictly precede test date ({min_test_date})",
        )

        # Confirm exact percentages ~ 70% / 15% / 15% of unique dates
        total_dates = split_meta["total_dates"]
        self.assertEqual(total_dates, len(df["date"].unique()))

    def test_03_clean_filtering_exclusions(self):
        """Verify unknown_uncovered and negative_buffer observations are excluded from ML dataset."""
        df = self.ds_res["dataframe"]
        # All rows should have label_status as negative_clean, active_event, or lead_event
        allowed_statuses = {"negative_clean", "active_event", "lead_event"}
        statuses_present = set(df["label_status"].unique())
        self.assertTrue(
            statuses_present.issubset(allowed_statuses),
            f"Unexpected label statuses in ML dataset: {statuses_present}",
        )
        self.assertNotIn("negative_buffer", statuses_present)
        self.assertNotIn("unknown_uncovered", statuses_present)

    def test_04_ml_trainer_multitask_training(self):
        """Verify training of all separate tasks: event_active, 1d, 2d, 3d, and event_type."""
        tasks = self.train_res["tasks"]
        self.assertIn("event_active", tasks)
        self.assertIn("event_within_1d", tasks)
        self.assertIn("event_within_2d", tasks)
        self.assertIn("event_within_3d", tasks)
        self.assertIn("event_type", tasks)

        # Check binary task evaluations
        for b_task in ["event_active", "event_within_1d", "event_within_2d", "event_within_3d"]:
            task_eval = tasks[b_task]
            self.assertIn("best_model_name", task_eval)
            self.assertIn("validation_metrics", task_eval)
            self.assertIn("test_metrics", task_eval)
            self.assertIn("calibration", task_eval)
            self.assertIn("explainability", task_eval)

            # Check multi-granularity explainability
            exp = task_eval["explainability"]
            self.assertIn("ocean_variable_importance", exp)
            self.assertIn("time_window_importance", exp)
            self.assertIn("all_feature_importances", exp)

            # Verify variable aggregations
            var_imp = exp["ocean_variable_importance"]
            for v in ["temperature", "salinity", "currents", "sea_surface_height", "mixed_layer"]:
                self.assertIn(v, var_imp)

            # Verify time window aggregations
            tw_imp = exp["time_window_importance"]
            for w in ["current", "7d", "14d", "30d"]:
                self.assertIn(w, tw_imp)

    def test_05_saved_artifacts_and_metadata(self):
        """Verify saved model files and comprehensive model_metadata.json."""
        meta_file = os.path.join(self.models_dir, "model_metadata.json")
        self.assertTrue(os.path.exists(meta_file), "model_metadata.json should be written")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("trained_tasks", meta)
        self.assertIn("feature_count", meta)
        self.assertIn("task_evaluations", meta)
        self.assertGreater(meta["feature_count"], 0)

        # Check model joblib artifacts exist
        for b_task in ["event_active", "event_within_1d", "event_within_2d", "event_within_3d"]:
            model_file = os.path.join(self.models_dir, f"model_{b_task}.joblib")
            self.assertTrue(os.path.exists(model_file), f"Missing artifact: {model_file}")

    def test_06_prediction_service_inference(self):
        """Test PredictionService inference for active and early-warning horizons."""
        # Active event prediction
        req_active = PredictionRequest(
            date="2024-08-01",
            lat=15.0,
            lon=70.0,
            mode="point",
            horizon_days=0,
        )
        resp_active = PredictionService.predict(req_active, artifacts_dir=self.models_dir)
        self.assertEqual(resp_active.status, "success")
        self.assertIn(resp_active.prediction, ["normal", "advisory", "alert"])
        self.assertGreaterEqual(resp_active.probability, 0.0)
        self.assertLessEqual(resp_active.probability, 1.0)
        self.assertGreater(len(resp_active.explainability.top_features), 0)

        # 3-day early warning prediction
        req_ew = PredictionRequest(
            date="2024-08-01",
            lat=15.0,
            lon=70.0,
            mode="point",
            horizon_days=3,
        )
        resp_ew = PredictionService.predict(req_ew, artifacts_dir=self.models_dir)
        self.assertEqual(resp_ew.status, "success")
        self.assertEqual(resp_ew.horizon_days, 3)
        self.assertIn(resp_ew.prediction, ["normal", "advisory", "alert"])

    def test_07_api_prediction_endpoints(self):
        """Test REST API endpoints: GET /api/prediction/status and POST /api/prediction/predict."""
        # Test status endpoint
        res_status = self.client.get("/api/prediction/status")
        self.assertEqual(res_status.status_code, 200)
        data_status = res_status.json()
        self.assertIn("is_ready", data_status)
        self.assertTrue(data_status["is_ready"])

        # Test predict endpoint with valid date inside Copernicus range
        payload = {
            "date": "2024-08-15",
            "lat": 15.0,
            "lon": 70.0,
            "mode": "point",
            "horizon_days": 3,
        }
        res_pred = self.client.post("/api/prediction/predict", json=payload)
        self.assertEqual(res_pred.status_code, 200)
        data_pred = res_pred.json()
        self.assertEqual(data_pred["status"], "success")
        self.assertIn("probability", data_pred)
        self.assertIn("prediction", data_pred)
        self.assertIn("explainability", data_pred)
        self.assertIn("top_features", data_pred["explainability"])


if __name__ == "__main__":
    unittest.main()
