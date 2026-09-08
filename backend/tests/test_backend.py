"""Test suite for Sagar Drishti Backend."""
import os
import unittest
import numpy as np
from app.parsers.netcdf_parser import NetCDFParser
from app.parsers.tabular_parser import TabularParser
from app.services.site_registry import SiteRegistry
from app.services.slice_engine import SliceEngine
from app.services.ocean_ai import OceanAIService
from app.models.schemas import ChatContext, ChatRequest


class TestSagarDrishtiBackend(unittest.TestCase):

    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")
        self.sample_csv = os.path.join(self.sample_dir, "sample_argo_profiles.csv")
        self.sample_nc = os.path.join(self.sample_dir, "sample_ocean_grid.nc")

    def test_tabular_parser(self):
        """Test parsing of sample Argo CSV dataset."""
        with open(self.sample_csv, "rb") as f:
            content = f.read()

        df, meta = TabularParser.parse_csv_from_bytes(content, "sample_argo_profiles.csv")
        self.assertFalse(df.empty)
        self.assertIn("floats", meta)
        self.assertGreaterEqual(len(meta["floats"]), 3)
        self.assertIn("site_record", meta)
        self.assertEqual(meta["site_record"].isCustom, True)
        self.assertGreater(meta["site_record"].maxDepth, 100)

        # Test GeoJSON conversion
        geojson = TabularParser.to_geojson(meta["floats"])
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), len(meta["floats"]))

    def test_netcdf_parser(self):
        """Test parsing and metadata extraction from NetCDF file."""
        if not os.path.exists(self.sample_nc):
            self.skipTest("sample_ocean_grid.nc not found")

        with open(self.sample_nc, "rb") as f:
            content = f.read()

        ds, meta = NetCDFParser.parse_dataset_from_bytes(content, "sample_ocean_grid.nc")
        self.assertIn("site_record", meta)
        self.assertIn("bbox", meta)
        self.assertIn("variables", meta)
        self.assertIn("temp", meta["variables"])
        self.assertGreater(meta["bbox"].max_lat, meta["bbox"].min_lat)

        # Test slicing
        slice_data = NetCDFParser.slice_2d_grid(ds, var_key="temp", depth=10.0, time_idx=0)
        self.assertEqual(slice_data["variable"], "temp")
        self.assertGreater(len(slice_data["values"]), 0)

    def test_site_registry(self):
        """Test baseline sites and registration of custom uploaded sites."""
        sites = SiteRegistry.get_all_sites()
        self.assertGreaterEqual(len(sites), 8)
        bob = SiteRegistry.get_site("bob")
        self.assertIsNotNone(bob)
        self.assertEqual(bob.name, "Bay of Bengal — Fresh Plume")

    def test_slice_engine_synthetic(self):
        """Test synthetic slicing for baseline sites."""
        slice_res = SliceEngine.get_slice(site_id="bob", variable="temp", depth=50.0, time_offset=0.0)
        self.assertEqual(slice_res.dataset_id, "bob")
        self.assertEqual(slice_res.variable, "temp")
        self.assertEqual(len(slice_res.values), 48)
        self.assertEqual(len(slice_res.values[0]), 48)

    def test_offline_ocean_ai(self):
        """Test context grounding and offline oceanographic expert responses."""
        ctx = ChatContext(
            active_site="Bay of Bengal — Fresh Plume",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="120 m",
            variable="Salinity",
            current_value="34.8 PSU",
            time_offset="+0h",
            nearby_floats=["Argo 2903341"],
            custom_data=False
        )
        req = ChatRequest(message="What is the halocline barrier layer?", context=ctx, provider="offline")

        import asyncio
        resp = asyncio.run(OceanAIService.process_chat(req))
        self.assertIn("Salinity", resp.reply)
        self.assertIn("Barrier", resp.reply)
        self.assertEqual(resp.provider, "SagarBot Oceanographic Physics Engine (Offline)")

    def test_default_gemini_api_key(self):
        """Verify LLM API key environment integration is configured."""
        self.assertTrue(hasattr(OceanAIService, "DEFAULT_GEMINI_API_KEY"))
        self.assertIsInstance(OceanAIService.DEFAULT_GEMINI_API_KEY, str)
        self.assertTrue(hasattr(OceanAIService, "DEFAULT_GROQ_API_KEY"))
        self.assertTrue(hasattr(OceanAIService, "DEFAULT_OPENAI_API_KEY"))

    def test_rate_limiter(self):
        """Test sliding window rate limiter functionality."""
        from app.services.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=3, window_seconds=2.0)
        client_ip = "192.168.1.100"

        # First 3 requests must be allowed
        allowed1, retry1, rem1 = limiter.is_allowed(client_ip)
        allowed2, retry2, rem2 = limiter.is_allowed(client_ip)
        allowed3, retry3, rem3 = limiter.is_allowed(client_ip)

        self.assertTrue(allowed1)
        self.assertTrue(allowed2)
        self.assertTrue(allowed3)
        self.assertEqual(rem3, 0)

        # 4th request must be rejected
        allowed4, retry4, rem4 = limiter.is_allowed(client_ip)
        self.assertFalse(allowed4)
        self.assertGreater(retry4, 0)

        # Different client IP is not affected
        allowed_other, _, _ = limiter.is_allowed("192.168.1.200")
        self.assertTrue(allowed_other)

        # Reset clears history
        limiter.reset()
        allowed_after_reset, _, _ = limiter.is_allowed(client_ip)
        self.assertTrue(allowed_after_reset)


if __name__ == "__main__":
    unittest.main()
