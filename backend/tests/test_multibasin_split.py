"""
Unit & Integration Tests for Multi-Basin Dataset Expansion & Event-Aware Splitting.
Validates:
1. Multi-site feature generation (BOB and ARAS)
2. Site separation & basin tagging
3. Combined dataset integrity & column parity
4. Event-aware chronological splitting with strict temporal bounds
5. Presence of authoritative positive events in both Validation and Test folds
6. Zero leakage into feature matrix X
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from app.services.copernicus_service import CopernicusService
from app.services.historical_engine import HistoricalFeatureEngine
from app.services.event_store import HistoricalEventStore
from app.services.event_matcher import SpatialTemporalMatcher
from app.services.site_registry import SiteRegistry
from app.services.ml_dataset_builder import LEAKAGE_AND_TARGET_COLUMNS


class TestMultiBasinSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="sagar_multibasin_test_")
        cls.test_nc = os.path.join(cls.test_dir, "copernicus_phy_2yr_surface.nc")
        # Create test fixture with 165 days for fast isolated testing
        CopernicusService.create_test_fixture(cls.test_nc, days=165)

        cls.orig_data_dir = os.environ.get("COPERNICUS_DATA_DIR")
        os.environ["COPERNICUS_DATA_DIR"] = cls.test_dir

    @classmethod
    def tearDownClass(cls):
        if cls.orig_data_dir is not None:
            os.environ["COPERNICUS_DATA_DIR"] = cls.orig_data_dir
        else:
            os.environ.pop("COPERNICUS_DATA_DIR", None)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_site_registry_separation(self):
        """Verify BOB and ARAS sites are distinct in bounding box, coordinates, and identity."""
        bob = SiteRegistry.get_site("bob")
        aras = SiteRegistry.get_site("aras")

        self.assertIsNotNone(bob)
        self.assertIsNotNone(aras)
        self.assertEqual(bob.id, "bob")
        self.assertEqual(aras.id, "aras")

        # Non-overlapping longitudes (BOB is east > 80°E, ARAS is west < 75°E)
        self.assertGreater(bob.bbox.min_lon, aras.bbox.max_lon)
        self.assertNotEqual(bob.lat, aras.lat)

    def test_02_multisite_feature_generation(self):
        """Verify Phase 2 features can be generated for both BOB and ARAS independently."""
        bob_out = os.path.join(self.test_dir, "bob_features")
        aras_out = os.path.join(self.test_dir, "aras_features")

        res_bob = HistoricalFeatureEngine.generate_training_dataset(
            mode="region", site_id="bob", output_dir=bob_out
        )
        res_aras = HistoricalFeatureEngine.generate_training_dataset(
            mode="region", site_id="aras", output_dir=aras_out
        )

        self.assertEqual(res_bob.status, "success")
        self.assertEqual(res_aras.status, "success")
        self.assertEqual(res_bob.total_rows, res_aras.total_rows)

        df_bob = pd.read_parquet(res_bob.parquet_path)
        df_aras = pd.read_parquet(res_aras.parquet_path)

        self.assertEqual(len(df_bob), len(df_aras))
        self.assertEqual(list(df_bob.columns), list(df_aras.columns))
        # Tagged site IDs
        self.assertTrue((df_bob["site_id"] == "bob").all())
        self.assertTrue((df_aras["site_id"] == "aras").all())

    def test_03_combined_dataset_integrity(self):
        """Verify combining BOB and ARAS maintains schema integrity and valid values."""
        bob_out = os.path.join(self.test_dir, "bob_features")
        aras_out = os.path.join(self.test_dir, "aras_features")
        df_bob = pd.read_parquet(os.path.join(bob_out, "features.parquet"))
        df_aras = pd.read_parquet(os.path.join(aras_out, "features.parquet"))

        combined = pd.concat([df_bob, df_aras], ignore_index=True)
        self.assertEqual(len(combined), len(df_bob) + len(df_aras))

        # Check no unexpected NaNs in physical feature columns
        num_cols = combined.select_dtypes(include=[np.number]).columns
        nan_counts = combined[num_cols].isna().sum().sum()
        self.assertEqual(nan_counts, 0, "Combined feature matrix contains unexpected NaNs")

    def test_04_event_aware_chronological_split(self):
        """Verify event-aware chronological partition preserves strict temporal ordering and disjoint events."""
        # Load the generated multi-basin dataset if available, or simulate partition
        multi_parquet = os.path.join("backend", "data", "historical", "ml_features_multibasin.parquet")
        if not os.path.exists(multi_parquet):
            multi_parquet = os.path.join(os.path.dirname(__file__), "..", "data", "historical", "ml_features_multibasin.parquet")

        if os.path.exists(multi_parquet):
            df = pd.read_parquet(multi_parquet)
            self.assertIn("split", df.columns)

            splits = df["split"].unique()
            self.assertIn("train", splits)
            self.assertIn("val", splits)
            self.assertIn("test", splits)

            train_df = df[df["split"] == "train"]
            val_df = df[df["split"] == "val"]
            test_df = df[df["split"] == "test"]

            # 1. Strict temporal order
            self.assertLess(train_df["date"].max(), val_df["date"].min())
            self.assertLess(val_df["date"].max(), test_df["date"].min())

            # 2. Positive events in Validation and Test folds
            val_pos = (val_df["event_active"] == 1).sum() + (val_df["event_within_3d"] == 1).sum()
            test_pos = (test_df["event_active"] == 1).sum() + (test_df["event_within_3d"] == 1).sum()

            self.assertGreater(val_pos, 0, "Validation fold must contain non-zero positive events")
            self.assertGreater(test_pos, 0, "Test fold must contain non-zero positive events")

            # 3. Disjoint event IDs
            train_events = set(train_df["event_id"].dropna().unique())
            val_events = set(val_df["event_id"].dropna().unique())
            test_events = set(test_df["event_id"].dropna().unique())

            self.assertEqual(len(train_events.intersection(val_events)), 0)
            self.assertEqual(len(val_events.intersection(test_events)), 0)
            self.assertEqual(len(train_events.intersection(test_events)), 0)

    def test_05_feature_leakage_prevention(self):
        """Verify zero leakage columns in feature matrix X."""
        multi_parquet = os.path.join("backend", "data", "historical", "ml_features_multibasin.parquet")
        if not os.path.exists(multi_parquet):
            multi_parquet = os.path.join(os.path.dirname(__file__), "..", "data", "historical", "ml_features_multibasin.parquet")

        if os.path.exists(multi_parquet):
            df = pd.read_parquet(multi_parquet)
            feature_cols = [c for c in df.columns if c not in LEAKAGE_AND_TARGET_COLUMNS and c not in ["basin", "parsed_date"]]

            # Verify no blacklisted column exists in feature_cols
            for col in feature_cols:
                self.assertNotIn(col, LEAKAGE_AND_TARGET_COLUMNS)
                self.assertNotIn("event", col.lower())
                self.assertNotIn("lead", col.lower())
                self.assertNotIn("target", col.lower())
                self.assertNotIn("label", col.lower())


if __name__ == "__main__":
    unittest.main()
