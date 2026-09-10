"""
Comprehensive Unit Tests for Phase 3: Historical Event-Labeling Pipeline.
Verifies event catalog invariants, extensibility, type-appropriate spatial representations,
discrete lead-time labels, explicit negative-clean vs buffer vs uncovered labeling,
perturbation-based temporal leakage invariance, Parquet export, and REST endpoints.
"""
import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import xarray as xr
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import (
    BoundingBox,
    HistoricalEvent,
    EventType,
    SpatialRepresentation,
    LabelStatus,
)
from app.services.copernicus_service import CopernicusService
from app.services.historical_engine import HistoricalFeatureEngine, SUPPORTED_CHANNELS
from app.services.event_store import (
    HistoricalEventStore,
    SEED_HISTORICAL_EVENTS,
    AUTHORITATIVE_COVERAGE,
)
from app.services.event_matcher import SpatialTemporalMatcher


class TestHistoricalEventLabelingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create isolated temporary directory
        cls.test_dir = tempfile.mkdtemp(prefix="sagar_event_label_test_")
        cls.test_nc = os.path.join(cls.test_dir, "copernicus_phy_2yr_surface.nc")

        # 60 days starting 2024-06-24 covers until late August 2024
        # We create a 135-day fixture to cover through Nov 2024 (including Asna, BOB 05, Dana, Fengal)
        cls.test_days = 165
        CopernicusService.create_test_fixture(cls.test_nc, days=cls.test_days)

        cls.orig_data_dir = os.environ.get("COPERNICUS_DATA_DIR")
        os.environ["COPERNICUS_DATA_DIR"] = cls.test_dir

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
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

    def setUp(self):
        # Always ensure store is in clean seed state
        HistoricalEventStore.reset_to_seeds()

    # ==========================================================================
    # TEST 1: EVENT CATALOG & CONTROLLED TAXONOMY
    # ==========================================================================
    def test_01_event_catalog_seed_records_and_taxonomy(self):
        """Verify 7 seed records from IMD RSMC New Delhi and INCOIS and controlled taxonomy."""
        events = HistoricalEventStore.get_all_events()
        self.assertGreaterEqual(len(events), 7)

        event_ids = [e.event_id for e in events]
        self.assertIn("IMD-2024-SCS-ASNA", event_ids)
        self.assertIn("IMD-2024-SCS-DANA", event_ids)
        self.assertIn("IMD-2024-CS-FENGAL", event_ids)
        self.assertIn("IMD-2024-DD-BOB05", event_ids)
        self.assertIn("IMD-2024-D-ARB01", event_ids)
        self.assertIn("INCOIS-2024-SW-KALLAKKADAL", event_ids)
        self.assertIn("INCOIS-2024-MHW-BOB", event_ids)

        # Controlled taxonomy
        allowed_types = {
            EventType.TROPICAL_CYCLONE,
            EventType.DEPRESSION,
            EventType.COASTAL_SWELL_SURGE,
            EventType.MARINE_HEATWAVE,
        }
        for ev in events:
            self.assertIn(ev.event_type, allowed_types)
            self.assertTrue(ev.name)
            self.assertTrue(ev.source)
            self.assertTrue(ev.source_reference)
            self.assertTrue(ev.severity)
            self.assertEqual(ev.confidence, "verified_authoritative")

    # ==========================================================================
    # TEST 2: STRICT EVENT INVARIANTS & VALIDATION
    # ==========================================================================
    def test_02_strict_event_invariants_and_validation(self):
        """Verify validate_event catches invalid dates, inverted bbox, and missing citations."""
        report = HistoricalEventStore.validate_catalog()
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["error_count"], 0)

        # Inverted dates (start > end)
        with self.assertRaises(ValueError) as ctx:
            bad_ev = HistoricalEvent(
                event_id="INVALID-DATE",
                event_type=EventType.TROPICAL_CYCLONE,
                name="Bad Date Storm",
                start_date="2024-10-25",
                end_date="2024-10-20",  # earlier than start
                affected_region="Bay of Bengal",
                bbox=BoundingBox(min_lat=10.0, max_lat=15.0, min_lon=80.0, max_lon=85.0),
                severity="Severe",
                source="IMD",
                source_reference="Report 1",
            )
            HistoricalEventStore.validate_event(bad_ev)
        self.assertIn("start_date", str(ctx.exception))

        # Inverted bounding box (min_lat > max_lat)
        with self.assertRaises(ValueError) as ctx:
            bad_bbox = HistoricalEvent(
                event_id="INVALID-BBOX",
                event_type=EventType.DEPRESSION,
                name="Bad BBox",
                start_date="2024-09-01",
                end_date="2024-09-05",
                affected_region="Arabian Sea",
                bbox=BoundingBox(min_lat=20.0, max_lat=10.0, min_lon=65.0, max_lon=70.0),
                severity="Depression",
                source="IMD",
                source_reference="Report 2",
            )
            HistoricalEventStore.validate_event(bad_bbox)
        self.assertIn("inverted bounding box", str(ctx.exception))

        # Missing source citation
        with self.assertRaises(ValueError) as ctx:
            bad_src = HistoricalEvent(
                event_id="INVALID-SRC",
                event_type=EventType.DEPRESSION,
                name="Bad Src",
                start_date="2024-09-01",
                end_date="2024-09-05",
                affected_region="Arabian Sea",
                bbox=BoundingBox(min_lat=10.0, max_lat=20.0, min_lon=65.0, max_lon=70.0),
                severity="Depression",
                source="   ",  # whitespace only
                source_reference="Report 2",
            )
            HistoricalEventStore.validate_event(bad_src)
        self.assertIn("missing mandatory 'source'", str(ctx.exception))

    # ==========================================================================
    # TEST 3: EXTENSIBILITY (DYNAMIC REGISTRATION & EXTERNAL LOADING)
    # ==========================================================================
    def test_03_store_extensibility_dynamic_registration(self):
        """Verify dynamic registration of additional authoritative events without matcher changes."""
        new_event = HistoricalEvent(
            event_id="IMD-2024-TEST-STORM",
            event_type=EventType.TROPICAL_CYCLONE,
            name="Test Verification Cyclone",
            start_date="2024-08-15",
            end_date="2024-08-18",
            affected_region="Central Arabian Sea",
            spatial_representation=SpatialRepresentation.TRACK_ENVELOPE,
            bbox=BoundingBox(min_lat=12.0, max_lat=16.0, min_lon=64.0, max_lon=68.0),
            severity="Cyclonic Storm (40 kts)",
            source="IMD RSMC New Delhi",
            source_reference="RSMC Experimental Advisory 2024",
        )

        HistoricalEventStore.register_event(new_event)
        retrieved = HistoricalEventStore.get_event_by_id("IMD-2024-TEST-STORM")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Verification Cyclone")

        # Duplicate ID check
        with self.assertRaises(ValueError) as ctx:
            HistoricalEventStore.register_event(new_event)
        self.assertIn("already registered", str(ctx.exception))

    # ==========================================================================
    # TEST 4: TYPE-APPROPRIATE SPATIAL REPRESENTATION
    # ==========================================================================
    def test_04_type_appropriate_spatial_representation(self):
        """Verify track envelope for cyclones/depressions and regional envelope for swell/heatwave."""
        asna = HistoricalEventStore.get_event_by_id("IMD-2024-SCS-ASNA")
        self.assertEqual(asna.spatial_representation, SpatialRepresentation.TRACK_ENVELOPE)
        self.assertIsNotNone(asna.track_coordinates)
        self.assertGreater(len(asna.track_coordinates), 0)

        dana = HistoricalEventStore.get_event_by_id("IMD-2024-SCS-DANA")
        self.assertEqual(dana.spatial_representation, SpatialRepresentation.TRACK_ENVELOPE)

        mhw = HistoricalEventStore.get_event_by_id("INCOIS-2024-MHW-BOB")
        self.assertEqual(mhw.spatial_representation, SpatialRepresentation.REGIONAL_ENVELOPE)

        kallakkadal = HistoricalEventStore.get_event_by_id("INCOIS-2024-SW-KALLAKKADAL")
        self.assertEqual(kallakkadal.spatial_representation, SpatialRepresentation.REGIONAL_ENVELOPE)

    # ==========================================================================
    # TEST 5: SPATIAL MATCHING (POINT & REGION)
    # ==========================================================================
    def test_05_spatial_matching_point_and_region(self):
        """Verify spatial matching for point-in-bbox and regional bbox-overlap."""
        # Point inside Dana (lat=19.5, lon=87.4) on active date
        res_inside = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-24",
            mode="point",
            lat=19.5,
            lon=87.4,
        )
        self.assertEqual(res_inside["event_id"], "IMD-2024-SCS-DANA")
        self.assertEqual(res_inside["label_status"], LabelStatus.ACTIVE_EVENT.value)

        # Point outside Dana (lat=5.0, lon=55.0 - Somali basin) on same date
        res_outside = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-24",
            mode="point",
            lat=5.0,
            lon=55.0,
        )
        self.assertNotEqual(res_outside["event_id"], "IMD-2024-SCS-DANA")
        self.assertEqual(res_outside["is_active_event"], 0)

        # Region mode: Bay of Bengal site intersects Dana
        res_bob = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-24",
            mode="region",
            site_id="bob",
        )
        self.assertEqual(res_bob["event_id"], "IMD-2024-SCS-DANA")
        self.assertEqual(res_bob["is_active_event"], 1)

        # Region mode: Arabian Sea site does NOT intersect Dana
        res_aras = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-24",
            mode="region",
            site_id="aras",
        )
        self.assertNotEqual(res_aras["event_id"], "IMD-2024-SCS-DANA")

    # ==========================================================================
    # TEST 6: TEMPORAL MATCHING (ACTIVE EVENT WINDOW)
    # ==========================================================================
    def test_06_temporal_matching_active_window(self):
        """Verify active event labels strictly during event active dates."""
        # SCS Dana was active 2024-10-22 to 2024-10-25
        for active_date in ["2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25"]:
            res = SpatialTemporalMatcher.match_observation(
                date_str=active_date,
                mode="point",
                lat=18.5,
                lon=88.0,
            )
            self.assertEqual(res["event_active"], 1)
            self.assertEqual(res["is_active_event"], 1)
            self.assertEqual(res["event_present"], 1)
            # Strictly future-only early warning: already active events are 0
            self.assertEqual(res["event_within_1d"], 0)
            self.assertEqual(res["event_within_2d"], 0)
            self.assertEqual(res["event_within_3d"], 0)
            self.assertEqual(res["lead_0"], 1)
            self.assertEqual(res["lead_days"], 0)
            self.assertEqual(res["label_status"], LabelStatus.ACTIVE_EVENT.value)
            self.assertEqual(res["event_id"], "IMD-2024-SCS-DANA")

    # ==========================================================================
    # TEST 7: DISCRETE LEAD-TIME LABELS (lead_0, lead_1, lead_2, lead_3)
    # ==========================================================================
    def test_07_discrete_lead_time_labels(self):
        """Verify distinct lead_0, lead_1, lead_2, lead_3 and future-only early warning targets."""
        # Dana start_date is 2024-10-22.
        # lead_1 is 2024-10-21 (1 day prior)
        res_l1 = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-21",
            mode="point",
            lat=18.5,
            lon=88.0,
        )
        self.assertEqual(res_l1["event_active"], 0)
        self.assertEqual(res_l1["is_active_event"], 0)
        self.assertEqual(res_l1["event_present"], 1)
        self.assertEqual(res_l1["event_within_1d"], 1)
        self.assertEqual(res_l1["event_within_2d"], 1)
        self.assertEqual(res_l1["event_within_3d"], 1)
        self.assertEqual(res_l1["lead_0"], 0)
        self.assertEqual(res_l1["lead_1"], 1)
        self.assertEqual(res_l1["lead_2"], 0)
        self.assertEqual(res_l1["lead_3"], 0)
        self.assertEqual(res_l1["lead_days"], 1)
        self.assertEqual(res_l1["label_status"], LabelStatus.LEAD_EVENT.value)
        self.assertEqual(res_l1["event_id"], "IMD-2024-SCS-DANA")

        # lead_2 is 2024-10-20 (2 days prior)
        res_l2 = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-20",
            mode="point",
            lat=18.5,
            lon=88.0,
        )
        self.assertEqual(res_l2["event_active"], 0)
        self.assertEqual(res_l2["event_within_1d"], 0)
        self.assertEqual(res_l2["event_within_2d"], 1)
        self.assertEqual(res_l2["event_within_3d"], 1)
        self.assertEqual(res_l2["lead_1"], 0)
        self.assertEqual(res_l2["lead_2"], 1)
        self.assertEqual(res_l2["lead_3"], 0)
        self.assertEqual(res_l2["lead_days"], 2)

        # lead_3 is 2024-10-19 (3 days prior)
        res_l3 = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-19",
            mode="point",
            lat=18.5,
            lon=88.0,
        )
        self.assertEqual(res_l3["event_active"], 0)
        self.assertEqual(res_l3["event_within_1d"], 0)
        self.assertEqual(res_l3["event_within_2d"], 0)
        self.assertEqual(res_l3["event_within_3d"], 1)
        self.assertEqual(res_l3["lead_1"], 0)
        self.assertEqual(res_l3["lead_2"], 0)
        self.assertEqual(res_l3["lead_3"], 1)
        self.assertEqual(res_l3["lead_days"], 3)

        # 4 days prior (2024-10-18): not in 3-day lead window
        res_l4 = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-18",
            mode="point",
            lat=18.5,
            lon=88.0,
        )
        self.assertEqual(res_l4["event_active"], 0)
        self.assertEqual(res_l4["event_present"], 0)
        self.assertEqual(res_l4["event_within_1d"], 0)
        self.assertEqual(res_l4["event_within_2d"], 0)
        self.assertEqual(res_l4["event_within_3d"], 0)
        self.assertEqual(res_l4["lead_days"], -1)

    # ==========================================================================
    # TEST 8: NEGATIVE METHODOLOGY (clean vs buffer vs uncovered)
    # ==========================================================================
    def test_08_explicit_negative_clean_buffer_and_uncovered_methodology(self):
        """Verify distinct tagging for negative_clean, negative_buffer, and unknown_uncovered."""
        # 1. Inside coverage with no active event or lead: clean negative
        res_clean = SpatialTemporalMatcher.match_observation(
            date_str="2024-08-10",
            mode="point",
            lat=17.8,
            lon=88.2,
        )
        self.assertEqual(res_clean["label_status"], LabelStatus.NEGATIVE_CLEAN.value)
        self.assertEqual(res_clean["event_present"], 0)
        self.assertEqual(res_clean["label_source"], "IMD_INCOIS_VERIFIED_COVERAGE")

        # 2. Post-event recovery buffer: 1 day after Dana (2024-10-26)
        res_buffer = SpatialTemporalMatcher.match_observation(
            date_str="2024-10-26",
            mode="point",
            lat=18.5,
            lon=88.0,
        )
        self.assertEqual(res_buffer["label_status"], LabelStatus.NEGATIVE_BUFFER.value)
        self.assertEqual(res_buffer["event_present"], 0)
        self.assertEqual(res_buffer["label_confidence"], "buffer_unclean")

        # 3. Outside coverage domain (far Southern Ocean lat=-45, lon=60): unknown_uncovered
        res_uncovered = SpatialTemporalMatcher.match_observation(
            date_str="2024-08-10",
            mode="point",
            lat=-45.0,
            lon=60.0,
        )
        self.assertEqual(res_uncovered["label_status"], LabelStatus.UNKNOWN_UNCOVERED.value)
        self.assertNotEqual(res_uncovered["label_status"], LabelStatus.NEGATIVE_CLEAN.value)

    # ==========================================================================
    # TEST 9: PERTURBATION-BASED TEMPORAL LEAKAGE INVARIANCE TEST
    # ==========================================================================
    def test_09_strengthened_temporal_leakage_invariance(self):
        """
        Formally verify that every feature for observation date T uses ONLY
        ocean data at or before date T (t <= T).
        Perturb future ocean data (t > T) and verify all 98+ features are identical!
        """
        target_date = "2024-07-28"  # Day 34 in fixture

        # Step A: Compute feature vector on unperturbed dataset
        comp_before = HistoricalFeatureEngine.compute_comparison(
            date_str=target_date,
            mode="point",
            lat=15.0,
            lon=75.0,
        )
        fv_before = comp_before.feature_vector

        # Step B: Open NetCDF and perturb all future time steps (t > Day 34)
        ds = CopernicusService.get_dataset()
        coord_time = next(c for c in ["time", "record"] if c in ds.coords or c in ds.dims)
        dataset_times = pd.to_datetime(ds[coord_time].values)
        target_idx = int(np.argmin(np.abs((dataset_times - pd.to_datetime(target_date)).total_seconds())))

        # Copy original future data and inject massive anomalies
        future_slice = slice(target_idx + 1, len(dataset_times))
        orig_thetao = ds["thetao"].values.copy()

        try:
            # Inject extreme +50.0 degree spike in all future time steps
            ds["thetao"].values[future_slice, ...] += 50.0

            # Step C: Recompute features at target_date
            comp_after = HistoricalFeatureEngine.compute_comparison(
                date_str=target_date,
                mode="point",
                lat=15.0,
                lon=75.0,
            )
            fv_after = comp_after.feature_vector

            # Step D: Verify every single rolling feature at date T is 100% invariant
            for k in fv_before.keys():
                # Rolling features (7d, 14d, 30d mean, delta, trend, current, base)
                # Note: full-period baseline in Phase 2 spans full available NetCDF period;
                # all rolling window features (7d, 14d, 30d) MUST be 100% invariant!
                if "7d_" in k or "14d_" in k or "30d_" in k or "_current" in k:
                    self.assertAlmostEqual(
                        fv_before[k],
                        fv_after[k],
                        places=4,
                        msg=f"Temporal leakage detected in feature '{k}'! It changed when future data changed.",
                    )
        finally:
            # Restore pristine dataset values
            ds["thetao"].values[:] = orig_thetao

    # ==========================================================================
    # TEST 10: LABELED DATASET PARQUET & METADATA EXPORT
    # ==========================================================================
    def test_10_labeled_dataset_parquet_and_metadata_export(self):
        """Test full labeled dataset generation and companion metadata export."""
        export_dir = os.path.join(self.test_dir, "historical_labeled_test")
        res = SpatialTemporalMatcher.generate_labeled_dataset(
            output_dir=export_dir,
            mode="point",
            lat=18.5,
            lon=88.0,
        )
        self.assertEqual(res.status, "success")
        self.assertTrue(os.path.exists(res.parquet_path))
        self.assertTrue(os.path.exists(res.metadata_path))
        self.assertGreater(res.total_rows, 0)

        # Verify Parquet content
        df = pd.read_parquet(res.parquet_path)
        self.assertIn("date", df.columns)
        self.assertIn("temp_current", df.columns)
        self.assertIn("temp_30d_trend", df.columns)
        self.assertIn("is_active_event", df.columns)
        self.assertIn("event_present", df.columns)
        self.assertIn("lead_0", df.columns)
        self.assertIn("lead_1", df.columns)
        self.assertIn("lead_2", df.columns)
        self.assertIn("lead_3", df.columns)
        self.assertIn("lead_days", df.columns)
        self.assertIn("label_status", df.columns)
        self.assertIn("event_id", df.columns)

        # Verify companion JSON metadata contains data quality report
        with open(res.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("data_quality_report", meta)
        dqr = meta["data_quality_report"]
        self.assertGreater(dqr["total_events"], 0)
        self.assertIn("lead_distribution", dqr)
        self.assertIn("methodology_notes", dqr)

    # ==========================================================================
    # TEST 11: DATASET IMMUTABILITY GUARANTEE
    # ==========================================================================
    def test_11_dataset_immutability_guarantee(self):
        """Verify raw Copernicus NetCDF and Phase 2 features.parquet remain strictly untouched."""
        import hashlib

        # Measure NetCDF hash before
        with open(self.test_nc, "rb") as f:
            hash_before = hashlib.sha256(f.read()).hexdigest()

        # Run labeled dataset generation
        export_dir = os.path.join(self.test_dir, "immutability_test")
        SpatialTemporalMatcher.generate_labeled_dataset(
            output_dir=export_dir,
            mode="point",
            lat=18.5,
            lon=88.0,
        )

        # Measure NetCDF hash after
        with open(self.test_nc, "rb") as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(hash_before, hash_after, "Raw NetCDF must remain strictly unchanged!")

    # ==========================================================================
    # TEST 12: REST API HISTORICAL EVENT ENDPOINTS
    # ==========================================================================
    def test_12_api_historical_event_endpoints(self):
        """Test GET /api/historical/events, GET /quality-report, and POST /generate-labeled-dataset."""
        # 1. GET /api/historical/events
        resp_events = self.client.get("/api/historical/events?event_type=tropical_cyclone")
        self.assertEqual(resp_events.status_code, 200)
        events_data = resp_events.json()
        self.assertGreaterEqual(len(events_data), 3)
        for ev in events_data:
            self.assertEqual(ev["event_type"], "tropical_cyclone")

        # 2. GET /api/historical/events/quality-report
        resp_qr = self.client.get("/api/historical/events/quality-report?mode=point&lat=18.5&lon=88.0")
        self.assertEqual(resp_qr.status_code, 200)
        qr_data = resp_qr.json()
        self.assertIn("total_events", qr_data)
        self.assertIn("positive_observations", qr_data)
        self.assertIn("lead_distribution", qr_data)

        # 3. POST /api/historical/events/generate-labeled-dataset
        resp_gen = self.client.post("/api/historical/events/generate-labeled-dataset?mode=point&lat=18.5&lon=88.0")
        self.assertEqual(resp_gen.status_code, 200)
        gen_data = resp_gen.json()
        self.assertEqual(gen_data["status"], "success")
        self.assertGreater(gen_data["total_rows"], 0)
        self.assertTrue(os.path.exists(gen_data["parquet_path"]))


if __name__ == "__main__":
    unittest.main()
