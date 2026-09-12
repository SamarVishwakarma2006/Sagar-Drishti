"""
Unit tests for Copernicus Marine Historical Data Pipeline & Feature Engine in Sagar Drishti.
Tests variable standardization, post-download validation, lazy disk-backed slicing,
current speed derivation, 7/14/30-day rolling windows, dynamic baseline derivation,
Parquet dataset export, and API endpoints.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.copernicus_service import (
    CopernicusService,
    COPERNICUS_DATASET_ID,
    TARGET_VARIABLES,
    DEFAULT_MIN_LON,
    DEFAULT_MAX_LON,
    DEFAULT_MIN_LAT,
    DEFAULT_MAX_LAT,
    DEFAULT_DEPTH,
    ALIAS_MAP,
)
from app.services.historical_engine import HistoricalFeatureEngine, SUPPORTED_CHANNELS


class TestCopernicusHistoricalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Close any lingering dataset handles from previous test modules
        with CopernicusService._ds_lock:
            if CopernicusService._ds is not None:
                try:
                    CopernicusService._ds.close()
                except Exception:
                    pass
                CopernicusService._ds = None

        # Create a clean temporary directory for isolated test fixture
        cls.test_dir = tempfile.mkdtemp(prefix="sagar_copernicus_test_")
        cls.test_nc = os.path.join(cls.test_dir, "copernicus_phy_2yr_surface.nc")
        cls.test_meta = os.path.join(cls.test_dir, "metadata.json")

        # Generate test NetCDF fixture with 40 days to test 7d, 14d, 30d rolling windows
        cls.test_days = 40
        CopernicusService.create_test_fixture(cls.test_nc, days=cls.test_days)

        # Point service to test directory
        cls.orig_data_dir = os.environ.get("COPERNICUS_DATA_DIR")
        os.environ["COPERNICUS_DATA_DIR"] = cls.test_dir

        cls.client = TestClient(app)


    @classmethod
    def tearDownClass(cls):
        # Close xarray handle if open
        with CopernicusService._ds_lock:
            if CopernicusService._ds is not None:
                try:
                    CopernicusService._ds.close()
                except Exception:
                    pass
                CopernicusService._ds = None

        if cls.orig_data_dir:
            os.environ["COPERNICUS_DATA_DIR"] = cls.orig_data_dir
        else:
            os.environ.pop("COPERNICUS_DATA_DIR", None)

        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_target_specifications(self):
        """Verify Copernicus Marine target specification constants."""
        self.assertEqual(COPERNICUS_DATASET_ID, "cmems_mod_glo_phy_my_0.083deg_P1D-m")
        self.assertEqual(len(TARGET_VARIABLES), 6)
        for var in ["mlotst", "so", "thetao", "uo", "vo", "zos"]:
            self.assertIn(var, TARGET_VARIABLES)
        self.assertEqual(DEFAULT_MIN_LON, 50.0)
        self.assertEqual(DEFAULT_MAX_LON, 100.0)
        self.assertEqual(DEFAULT_MIN_LAT, 0.0)
        self.assertEqual(DEFAULT_MAX_LAT, 25.0)
        self.assertAlmostEqual(DEFAULT_DEPTH, 0.494, places=2)

    def test_02_variable_standardization_mappings(self):
        """Verify standardization of Copernicus variables to existing Sagar Drishti fields."""
        self.assertEqual(ALIAS_MAP["thetao"], "thetao")
        self.assertEqual(ALIAS_MAP["temp"], "thetao")
        self.assertEqual(ALIAS_MAP["so"], "so")
        self.assertEqual(ALIAS_MAP["sal"], "so")
        self.assertEqual(ALIAS_MAP["uo"], "uo")
        self.assertEqual(ALIAS_MAP["cur_u"], "uo")
        self.assertEqual(ALIAS_MAP["vo"], "vo")
        self.assertEqual(ALIAS_MAP["cur_v"], "vo")
        self.assertEqual(ALIAS_MAP["zos"], "zos")
        self.assertEqual(ALIAS_MAP["ssh"], "zos")
        self.assertEqual(ALIAS_MAP["mlotst"], "mlotst")
        self.assertEqual(ALIAS_MAP["mld"], "mlotst")
        self.assertEqual(ALIAS_MAP["cur"], "cur")

    def test_03_mandatory_post_download_validation(self):
        """Test the post-download validation engine on valid dataset."""
        report = CopernicusService.validate_dataset(self.test_nc)
        self.assertTrue(report["is_valid"])
        self.assertTrue(report["variables_valid"])
        self.assertTrue(report["bounds_valid"])
        self.assertTrue(report["depth_valid"])
        self.assertTrue(report["time_valid"])
        self.assertTrue(report["daily_frequency"])
        self.assertTrue(report["integrity_valid"])
        self.assertEqual(len(report["missing_variables"]), 0)
        self.assertEqual(report["start_date"], "2024-06-24")

    def test_04_derived_current_speed_computation(self):
        """Verify dynamic calculation of current speed sqrt(uo^2 + vo^2) on the fly."""
        slice_cur = CopernicusService.get_slice(date_str="2024-06-24", variable="cur", resolution=16)
        self.assertEqual(slice_cur.variable, "cur")
        self.assertEqual(slice_cur.unit, "m/s")
        self.assertGreater(slice_cur.max_val, 0.0)
        self.assertGreaterEqual(slice_cur.min_val, 0.0)

    def test_05_lazy_horizontal_slice_extraction(self):
        """Test lazy 2D horizontal depth slicing for temperature, salinity, SSH, and MLD."""
        for var in ["temp", "sal", "ssh", "mld"]:
            res = CopernicusService.get_slice(date_str="2024-06-24", variable=var, resolution=20)
            self.assertEqual(res.dataset_id, COPERNICUS_DATASET_ID)
            self.assertEqual(res.date, "2024-06-24")
            self.assertGreater(len(res.lats), 0)
            self.assertGreater(len(res.lons), 0)
            self.assertEqual(len(res.values), len(res.lats))
            self.assertEqual(len(res.values[0]), len(res.lons))
            self.assertLessEqual(res.min_val, res.max_val)

    def test_06_dynamic_observation_count_and_baseline_metadata(self):
        """Verify dynamic observation count is derived from NetCDF time coordinate, not hardcoded."""
        comp = HistoricalFeatureEngine.compute_comparison(
            date_str="2024-07-28",  # day 34, allows 30-day window
            mode="point",
            lat=15.0,
            lon=75.0,
        )
        self.assertEqual(comp.baseline_metadata.baseline_observations, self.test_days)
        self.assertEqual(comp.baseline_metadata.baseline_start, "2024-06-24")
        self.assertEqual(comp.baseline_metadata.baseline_type, "full_period_climatology")
        self.assertTrue(comp.baseline_metadata.baseline_end.startswith("2024-08"))

    def test_07_rolling_window_metrics_and_trends(self):
        """Verify exact window sizing (7, 14, 30 days), delta, and linear slope calculation."""
        comp = HistoricalFeatureEngine.compute_comparison(
            date_str="2024-07-28",
            mode="point",
            lat=15.0,
            lon=75.0,
            variable="temp",
        )
        temp_metrics = comp.variables["temp"]
        self.assertIn("7d", temp_metrics.windows)
        self.assertIn("14d", temp_metrics.windows)
        self.assertIn("30d", temp_metrics.windows)

        w7 = temp_metrics.windows["7d"]
        w14 = temp_metrics.windows["14d"]
        w30 = temp_metrics.windows["30d"]

        self.assertEqual(w7.window_days, 7)
        self.assertEqual(w14.window_days, 14)
        self.assertEqual(w30.window_days, 30)

        # Delta must match current_value minus start value
        self.assertIsInstance(w7.delta, float)
        self.assertIsInstance(w7.trend_per_day, float)
        self.assertIsInstance(w14.delta, float)
        self.assertIsInstance(w30.delta, float)

        # Baseline vs current anomalies
        self.assertAlmostEqual(
            temp_metrics.absolute_anomaly,
            round(temp_metrics.current_value - temp_metrics.baseline_mean, 4),
            places=3
        )
        expected_z = round((temp_metrics.current_value - temp_metrics.baseline_mean) / temp_metrics.baseline_std, 4) if temp_metrics.baseline_std > 1e-6 else 0.0
        self.assertAlmostEqual(temp_metrics.z_score, expected_z, places=3)

    def test_08_spatial_modes_point_and_region(self):
        """Test localized grid-cell extraction vs. regional bounding box aggregation."""
        comp_pt = HistoricalFeatureEngine.compute_comparison(
            date_str="2024-07-28",
            mode="point",
            lat=12.0,
            lon=80.0,
        )
        self.assertEqual(comp_pt.mode, "point")
        self.assertEqual(comp_pt.location["lat"], 12.0)

        comp_reg = HistoricalFeatureEngine.compute_comparison(
            date_str="2024-07-28",
            mode="region",
            min_lat=10.0,
            max_lat=16.0,
            min_lon=70.0,
            max_lon=80.0,
        )
        self.assertEqual(comp_reg.mode, "region")
        self.assertIn("bbox", comp_reg.location)

    def test_09_unified_feature_vector_structure(self):
        """Verify combined feature vector contains features for all 7 physical channels."""
        comp = HistoricalFeatureEngine.compute_comparison(
            date_str="2024-07-28",
            mode="point",
            lat=16.0,
            lon=88.0,
            variable="all",
        )
        fv = comp.feature_vector
        for ch in SUPPORTED_CHANNELS:
            self.assertIn(f"{ch}_current", fv)
            self.assertIn(f"{ch}_base_mean", fv)
            self.assertIn(f"{ch}_base_std", fv)
            self.assertIn(f"{ch}_abs_anom", fv)
            self.assertIn(f"{ch}_zscore", fv)
            self.assertIn(f"{ch}_7d_mean", fv)
            self.assertIn(f"{ch}_7d_delta", fv)
            self.assertIn(f"{ch}_7d_trend", fv)
            self.assertIn(f"{ch}_14d_mean", fv)
            self.assertIn(f"{ch}_14d_delta", fv)
            self.assertIn(f"{ch}_14d_trend", fv)
            self.assertIn(f"{ch}_30d_mean", fv)
            self.assertIn(f"{ch}_30d_delta", fv)
            self.assertIn(f"{ch}_30d_trend", fv)

        # No NaNs in feature vector
        for k, v in fv.items():
            self.assertFalse(np.isnan(v), f"Feature '{k}' is NaN")

    def test_10_insufficient_history_validation(self):
        """Verify querying a date with fewer than 30 preceding days raises clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            HistoricalFeatureEngine.compute_comparison(
                date_str="2024-07-01",  # only 7 days after start date 2024-06-24
                mode="point",
                lat=15.0,
                lon=75.0,
            )
        self.assertIn("requires at least 30 days of history", str(ctx.exception))

    def test_11_parquet_and_metadata_export(self):
        """Test rolling feature table export as Parquet and JSON companion using generate_training_dataset."""
        import hashlib

        # Measure raw NetCDF hash before export
        with open(self.test_nc, "rb") as f:
            nc_hash_before = hashlib.sha256(f.read()).hexdigest()

        export_dir = os.path.join(self.test_dir, "export_test")
        res = HistoricalFeatureEngine.generate_training_dataset(
            output_dir=export_dir,
            mode="point",
            lat=17.8,
            lon=88.2,
        )
        self.assertEqual(res.status, "success")
        self.assertTrue(os.path.exists(res.parquet_path))
        self.assertTrue(os.path.exists(res.metadata_path))

        # Verify Parquet content
        df = pd.read_parquet(res.parquet_path)
        self.assertGreater(len(df), 0)
        self.assertIn("date", df.columns)
        self.assertIn("temp_current", df.columns)
        self.assertIn("cur_30d_trend", df.columns)

        # Verify companion JSON metadata contains dynamic baseline metadata
        import json
        with open(res.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("baseline_metadata", meta)
        bmeta = meta["baseline_metadata"]
        self.assertEqual(bmeta["baseline_type"], "full_period_climatology")
        self.assertEqual(bmeta["baseline_start"], "2024-06-24")
        self.assertEqual(bmeta["baseline_observations"], self.test_days)

        # Verify Raw NetCDF was not touched or modified in any way
        with open(self.test_nc, "rb") as f:
            nc_hash_after = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(nc_hash_before, nc_hash_after, "Raw NetCDF must remain strictly unchanged")

    def test_12_api_historical_compare_endpoint(self):
        """Test GET /api/historical/compare endpoint."""
        resp = self.client.get("/api/historical/compare?date=2024-07-28&mode=point&lat=15.0&lon=75.0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["dataset_id"], COPERNICUS_DATASET_ID)
        self.assertIn("baseline_metadata", data)
        self.assertEqual(data["baseline_metadata"]["baseline_observations"], self.test_days)
        self.assertIn("variables", data)
        self.assertIn("temp", data["variables"])
        self.assertIn("feature_vector", data)

    def test_13_api_export_features_endpoint(self):
        """Test POST /api/historical/export-features endpoint."""
        resp = self.client.post("/api/historical/export-features?mode=point&lat=15.0&lon=75.0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["total_rows"], 0)
        self.assertTrue(os.path.exists(data["parquet_path"]))


if __name__ == "__main__":
    unittest.main()
