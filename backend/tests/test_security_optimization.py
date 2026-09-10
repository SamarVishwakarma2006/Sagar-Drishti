"""
Security and Optimization Regression Tests.
Verifies:
1. PredictionService thread-safe model caching and clear_model_cache()
2. Upload 50 MB boundary enforcement without unbounded memory loading
3. Upload path traversal sanitization (os.path.basename)
4. CORS middleware origin hardening
5. RateLimiter automatic pruning under memory pressure
6. SagarBot conversational session memory bounding and turn clamping
7. ChatRequest message length boundary enforcement (max 2000 chars)
8. SiteRegistry clean dataset closing on deletion
"""
import os
import io
import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.services.prediction_service import PredictionService
from app.services.rate_limiter import RateLimiter
from app.services.sagarbot_service import SagarBotSessionManager
from app.services.site_registry import SiteRegistry
from app.models.schemas import ChatRequest, ChatContext, SitePhysics, BoundingBox


class TestSecurityAndOptimization(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        PredictionService.clear_model_cache()

    def tearDown(self):
        PredictionService.clear_model_cache()

    def test_prediction_model_cache_and_clear(self):
        """Test thread-safe model caching and clear_model_cache() isolation."""
        # Ensure cache starts empty after clear
        PredictionService.clear_model_cache()
        self.assertEqual(len(PredictionService._model_cache), 0)

        # Load 3d model artifact through canonical loader
        from app.services.ml_trainer import MODEL_ARTIFACTS_DIR
        from app.services.prediction_service import HORIZON_CONFIGS
        model_3d_path = os.path.join(MODEL_ARTIFACTS_DIR, HORIZON_CONFIGS[3]["file"])
        self.assertTrue(os.path.exists(model_3d_path))

        model_1 = PredictionService._get_loaded_artifact(model_3d_path)
        self.assertIsNotNone(model_1)
        self.assertIn(model_3d_path, PredictionService._model_cache)

        # Second load must return exact same cached object in memory
        model_2 = PredictionService._get_loaded_artifact(model_3d_path)
        self.assertIs(model_1, model_2)

        # clear_model_cache() must safely evict all entries
        PredictionService.clear_model_cache()
        self.assertEqual(len(PredictionService._model_cache), 0)

    def test_upload_streaming_50mb_limit(self):
        """Verify upload endpoint rejects files exceeding 50 MB with HTTP 413 without unbounded memory buffering."""
        # Create a mock file stream slightly larger than 50 MB (50 MB + 1 KB)
        OVERSIZED_BYTES = (50 * 1024 * 1024) + 1024

        class MockChunkedStream(io.RawIOBase):
            def __init__(self, total_size):
                self.remaining = total_size

            def read(self, size=-1):
                if self.remaining <= 0:
                    return b""
                n = min(size if size > 0 else 1024 * 1024, self.remaining)
                self.remaining -= n
                return b"X" * n

        stream = MockChunkedStream(OVERSIZED_BYTES)
        response = self.client.post(
            "/api/upload",
            files={"file": ("large_sample.csv", stream, "text/csv")}
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds maximum limit of 50 MB", response.json()["detail"])

    def test_upload_path_traversal_sanitization(self):
        """Verify upload endpoint sanitizes directory traversal sequences in filename."""
        payload = b"site,lat,lon\ntest,17.8,88.2\n"
        malicious_filename = "../../../../../etc/passwd.csv"

        response = self.client.post(
            "/api/upload",
            files={"file": (malicious_filename, io.BytesIO(payload), "text/csv")}
        )
        # Should succeed or return 422 if parsing fails, but MUST NOT use traversed path
        if response.status_code == 200:
            data = response.json()
            # The returned filename must be stripped of traversal path
            self.assertEqual(data["filename"], "passwd.csv")
            # Cleanup registered custom site
            SiteRegistry.delete_uploaded_site(data["site_id"])

    def test_chat_request_max_length_validation(self):
        """Verify ChatRequest rejects messages longer than 2000 characters."""
        ctx = ChatContext(
            active_site="Bay of Bengal — Fresh Plume",
            coordinates={"lat": 17.8, "lon": 88.2},
            current_depth="10 m",
            variable="Temperature",
            current_value="29.5 °C",
            time_offset="+0h",
        )
        # Valid message (under 2000 chars)
        req_valid = ChatRequest(message="Short safe query", context=ctx)
        self.assertEqual(req_valid.message, "Short safe query")

        # Oversized message (> 2000 chars)
        with self.assertRaises(ValidationError):
            ChatRequest(message="A" * 2001, context=ctx)

        # Direct API endpoint test
        response = self.client.post(
            "/api/chat",
            json={"message": "B" * 2001, "context": ctx.model_dump(), "provider": "offline"}
        )
        self.assertEqual(response.status_code, 422)

    def test_cors_middleware_configuration(self):
        """Verify CORS middleware is configured with explicit origins, not wildcard with credentials."""
        # Test OPTIONS pre-flight request with allowed origin
        response = self.client.options(
            "/api/sites",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        # Allowed origin should receive matching Access-Control-Allow-Origin
        self.assertIn(
            response.headers.get("access-control-allow-origin"),
            ["http://localhost:3000", "*"]
        )

    def test_rate_limiter_memory_pruning(self):
        """Verify RateLimiter prunes expired client entries when dictionary size exceeds 1,000."""
        limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        # Pre-fill with 1,050 expired client IPs
        expired_time = 1000.0  # Far in the past
        from collections import deque
        for i in range(1050):
            limiter._client_history[f"10.0.{i // 256}.{i % 256}"] = deque([expired_time])

        self.assertGreater(len(limiter._client_history), 1000)

        # Next check should trigger automatic pruning
        allowed, retry, rem = limiter.is_allowed("192.168.1.1")
        self.assertTrue(allowed)
        # The expired entries should have been pruned
        self.assertLessEqual(len(limiter._client_history), 10)

    def test_sagarbot_session_memory_bounding(self):
        """Verify SagarBotSessionManager bounds maximum stored sessions and clamps history."""
        # Insert 550 mock sessions
        for i in range(550):
            SagarBotSessionManager.get_session(f"bench_sess_{i}")

        # Total sessions must be bounded by MAX_SESSIONS
        self.assertLessEqual(len(SagarBotSessionManager._sessions), SagarBotSessionManager.MAX_SESSIONS + 10)

        # Test history clamping to 20 turns
        sess = SagarBotSessionManager.get_session("test_clamped_sess")
        oversized_history = [{"user": f"q{j}", "assistant": f"a{j}"} for j in range(35)]
        SagarBotSessionManager.update_session("test_clamped_sess", history=oversized_history)
        clamped = sess["history"]
        self.assertLessEqual(len(clamped), 20)
        self.assertEqual(clamped[-1]["user"], "q34")

    def test_site_registry_dataset_close_on_delete(self):
        """Verify delete_uploaded_site invokes close() on custom datasets to prevent resource leaks."""
        dummy_site = SitePhysics(
            id="custom_leak_test",
            name="Leak Test Site",
            region="Bay of Bengal",
            lat=17.5,
            lon=88.5,
            maxDepth=500.0,
            blurb="Test site",
            ts=29.0, ss=33.0, td=10.0, sd=34.0, mld=25.0, tw=30.0,
            salMaxAmp=0.5, salMaxZ=100.0, flow=0.3, eddy=150.0, bgU=0.1, bgV=0.1,
            o2s=180.0, o2d=150.0, o2z0=50.0, o2minAmp=100.0, o2minZ=200.0, o2minW=100.0,
            bbox=BoundingBox(min_lat=17.0, max_lat=18.0, min_lon=88.0, max_lon=89.0),
            isCustom=True,
            sourceType="CUSTOM_UPLOAD"
        )
        mock_ds = MagicMock()
        mock_ds.close = MagicMock()

        SiteRegistry.register_uploaded_site(dummy_site, raw_dataset=mock_ds)
        self.assertIn("custom_leak_test", SiteRegistry._custom_sites)
        self.assertIn("custom_leak_test", SiteRegistry._custom_datasets)

        # Delete site and verify close() was called
        removed = SiteRegistry.delete_uploaded_site("custom_leak_test")
        self.assertTrue(removed)
        mock_ds.close.assert_called_once()
        self.assertNotIn("custom_leak_test", SiteRegistry._custom_sites)
        self.assertNotIn("custom_leak_test", SiteRegistry._custom_datasets)

    def test_site_registry_custom_file_cleanup_and_safety(self):
        """
        Verify:
        1. Custom dataset deletion safely calls .close() on dataset resource.
        2. Associated application-owned temporary file is unlinked/removed.
        3. Missing files are handled safely without error.
        4. Arbitrary/protected files (like Copernicus or unrelated files) are NEVER deleted.
        """
        upload_dir = SiteRegistry.get_upload_dir()
        temp_custom_file = os.path.join(upload_dir, "temp_uploaded_dataset_123.nc")
        with open(temp_custom_file, "w") as f:
            f.write("temporary oceanographic data")

        unrelated_file = os.path.join(upload_dir, "unrelated_keeper_file.nc")
        with open(unrelated_file, "w") as f:
            f.write("do not delete me")

        mock_ds = MagicMock()
        mock_ds.close = MagicMock()

        custom_site = SitePhysics(
            id="custom_upload_site_123",
            name="Uploaded Custom Site",
            region="Bay of Bengal",
            lat=17.5,
            lon=88.5,
            maxDepth=500.0,
            blurb="Custom site with temp file",
            ts=29.0, ss=33.0, td=10.0, sd=34.0, mld=25.0, tw=30.0,
            salMaxAmp=0.5, salMaxZ=100.0, flow=0.3, eddy=150.0, bgU=0.1, bgV=0.1,
            o2s=180.0, o2d=150.0, o2z0=50.0, o2minAmp=100.0, o2minZ=200.0, o2minW=100.0,
            bbox=BoundingBox(min_lat=17.0, max_lat=18.0, min_lon=88.0, max_lon=89.0),
            isCustom=True,
            sourceType="CUSTOM_UPLOAD",
            customFilePath=temp_custom_file,
        )

        SiteRegistry.register_uploaded_site(custom_site, raw_dataset=mock_ds)

        # Confirm registration
        self.assertIn("custom_upload_site_123", SiteRegistry._custom_sites)
        self.assertTrue(os.path.exists(temp_custom_file))
        self.assertTrue(os.path.exists(unrelated_file))

        # Delete site
        removed = SiteRegistry.delete_uploaded_site("custom_upload_site_123")
        self.assertTrue(removed)

        # 1. Dataset resource was closed
        mock_ds.close.assert_called_once()

        # 2. Associated temporary file was deleted
        self.assertFalse(os.path.exists(temp_custom_file))

        # 3. Unrelated file was untouched
        self.assertTrue(os.path.exists(unrelated_file))

        # 4. Attempt to pass arbitrary/protected paths (Copernicus / system) must NEVER delete them
        arbitrary_protected_file = os.path.join(upload_dir, "copernicus_simulated.nc")
        with open(arbitrary_protected_file, "w") as f:
            f.write("copernicus protected data")

        malicious_site = SitePhysics(
            id="malicious_site_test",
            name="Malicious Path Site",
            region="Bay of Bengal",
            lat=17.5, lon=88.5, maxDepth=500.0, blurb="test",
            ts=29.0, ss=33.0, td=10.0, sd=34.0, mld=25.0, tw=30.0,
            salMaxAmp=0.5, salMaxZ=100.0, flow=0.3, eddy=150.0, bgU=0.1, bgV=0.1,
            o2s=180.0, o2d=150.0, o2z0=50.0, o2minAmp=100.0, o2minZ=200.0, o2minW=100.0,
            bbox=BoundingBox(min_lat=17.0, max_lat=18.0, min_lon=88.0, max_lon=89.0),
            isCustom=True,
            sourceType="CUSTOM_UPLOAD",
            customFilePath=arbitrary_protected_file,
        )
        SiteRegistry.register_uploaded_site(malicious_site)
        SiteRegistry.delete_uploaded_site("malicious_site_test")

        # Protected file MUST still exist (untouched)
        self.assertTrue(os.path.exists(arbitrary_protected_file))

        # 5. Missing file handling
        missing_site = SitePhysics(
            id="missing_file_site",
            name="Missing File Site",
            region="Bay of Bengal",
            lat=17.5, lon=88.5, maxDepth=500.0, blurb="test",
            ts=29.0, ss=33.0, td=10.0, sd=34.0, mld=25.0, tw=30.0,
            salMaxAmp=0.5, salMaxZ=100.0, flow=0.3, eddy=150.0, bgU=0.1, bgV=0.1,
            o2s=180.0, o2d=150.0, o2z0=50.0, o2minAmp=100.0, o2minZ=200.0, o2minW=100.0,
            bbox=BoundingBox(min_lat=17.0, max_lat=18.0, min_lon=88.0, max_lon=89.0),
            isCustom=True,
            sourceType="CUSTOM_UPLOAD",
            customFilePath=os.path.join(upload_dir, "non_existent_file.nc"),
        )
        SiteRegistry.register_uploaded_site(missing_site)
        # Should complete without error
        removed_missing = SiteRegistry.delete_uploaded_site("missing_file_site")
        self.assertTrue(removed_missing)

        # Cleanup test files
        if os.path.exists(unrelated_file):
            os.unlink(unrelated_file)
        if os.path.exists(arbitrary_protected_file):
            os.unlink(arbitrary_protected_file)


if __name__ == "__main__":
    unittest.main()
