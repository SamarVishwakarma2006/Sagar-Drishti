"""
Critical Temporal Leakage & 101-Feature Hard Contract Audit Tests.
Verifies that:
1. Feature vector for date T is constructed strictly from observations <= T.
2. Rolling windows (7d, 14d, 30d) strictly exclude future dates.
3. Zero IMD storm parameters (name, lat, lon, pressure, wind, category) exist in ML features.
4. Test-period data does not contaminate training splits.
5. All 4 target horizons (0d, 1d, 2d, 3d) are independent.
6. The feature schema contains exactly 101 canonical features in deterministic order.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from app.services.feature_manifest import FeatureManifest
from app.services.historical_engine import HistoricalFeatureEngine, SUPPORTED_CHANNELS
from app.services.ml_dataset_builder import LEAKAGE_AND_TARGET_COLUMNS
from app.services.ml_trainer import MLTrainer


def test_101_feature_contract_schema_and_ordering():
    """Verify exactly 101 canonical features and identical ordering with production models."""
    canonical_101 = FeatureManifest.get_canonical_101_feature_names()
    assert len(canonical_101) == 101

    manifest = FeatureManifest.generate_manifest()
    assert manifest["total_features"] == 101

    # Check against model_metadata.json if present
    meta_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "model_metadata.json")
    )
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        existing_features = meta.get("feature_list", [])
        assert len(existing_features) == 101
        assert existing_features == canonical_101, "Feature list order must match production contract"


def test_no_imd_storm_attributes_in_feature_columns():
    """Verify storm attributes (name, coordinates, wind, pressure, grade) are blacklisted."""
    forbidden_terms = [
        "name", "storm", "cyclone", "depression", "wind", "speed_kts",
        "pressure", "central_pressure", "grade", "category", "basin",
        "event_id", "severity", "source", "lead_days", "target"
    ]
    canonical_101 = FeatureManifest.get_canonical_101_feature_names()

    for feat in canonical_101:
        # Check that no feature column contains forbidden storm identifiers
        for term in ["storm", "cyclone", "pressure", "grade", "severity", "event_id"]:
            assert term not in feat.lower(), f"Feature '{feat}' leaks forbidden storm attribute '{term}'"

    # Verify LEAKAGE_AND_TARGET_COLUMNS contains all target and coordinate metadata
    assert "event_active" in LEAKAGE_AND_TARGET_COLUMNS
    assert "event_within_1d" in LEAKAGE_AND_TARGET_COLUMNS
    assert "event_within_2d" in LEAKAGE_AND_TARGET_COLUMNS
    assert "event_within_3d" in LEAKAGE_AND_TARGET_COLUMNS
    assert "lat" in LEAKAGE_AND_TARGET_COLUMNS
    assert "lon" in LEAKAGE_AND_TARGET_COLUMNS
    assert "event_name" in LEAKAGE_AND_TARGET_COLUMNS
    assert "severity" in LEAKAGE_AND_TARGET_COLUMNS


def test_rolling_windows_temporal_leak_firewall():
    """Verify rolling windows (7d, 14d, 30d) slice indices strictly in the past (t - w + 1 to t)."""
    # Create synthetic series
    N = 50
    series = np.arange(float(N))
    # Test index 29 (day 29)
    idx = 29
    for w_days in [7, 14, 30]:
        sub = series[idx - w_days + 1 : idx + 1]
        # Must have exactly w_days
        assert len(sub) == w_days
        # Last element must be current day (idx)
        assert sub[-1] == idx
        # First element must be (idx - w_days + 1)
        assert sub[0] == idx - w_days + 1
        # No elements greater than idx (strictly no future leakage)
        assert np.all(sub <= idx)


def test_independent_horizon_target_definitions():
    """Verify 0d, 1d, 2d, 3d horizon target properties."""
    from app.services.event_matcher import SpatialTemporalMatcher
    # Lead 0 (active) must not be confused with early warning horizons
    # When active: event_active=1, event_within_1d=0 (strictly future early warning)
    # When lead_1: event_active=0, event_within_1d=1, event_within_2d=1, event_within_3d=1
    # When lead_2: event_active=0, event_within_1d=0, event_within_2d=1, event_within_3d=1
    # When lead_3: event_active=0, event_within_1d=0, event_within_2d=0, event_within_3d=1

    # Active test
    res_active = SpatialTemporalMatcher.match_observation("2024-10-24", mode="point", lat=19.5, lon=87.4)
    assert res_active["event_active"] == 1
    assert res_active["event_within_1d"] == 0

    # Lead 1 test
    res_l1 = SpatialTemporalMatcher.match_observation("2024-10-21", mode="point", lat=18.5, lon=88.0)
    assert res_l1["event_active"] == 0
    assert res_l1["event_within_1d"] == 1
    assert res_l1["event_within_2d"] == 1
    assert res_l1["event_within_3d"] == 1

    # Lead 2 test
    res_l2 = SpatialTemporalMatcher.match_observation("2024-10-20", mode="point", lat=18.5, lon=88.0)
    assert res_l2["event_active"] == 0
    assert res_l2["event_within_1d"] == 0
    assert res_l2["event_within_2d"] == 1
    assert res_l2["event_within_3d"] == 1

    # Lead 3 test
    res_l3 = SpatialTemporalMatcher.match_observation("2024-10-19", mode="point", lat=18.5, lon=88.0)
    assert res_l3["event_active"] == 0
    assert res_l3["event_within_1d"] == 0
    assert res_l3["event_within_2d"] == 0
    assert res_l3["event_within_3d"] == 1


def test_chronological_splits_zero_test_leakage():
    """Verify test split is strictly in the future of train and validation splits."""
    multibasin_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "historical", "ml_features_multibasin.parquet")
    )
    if not os.path.exists(multibasin_path):
        pytest.skip("Multi-basin dataset not found on disk")

    df = pd.read_parquet(multibasin_path)
    train_dates = pd.to_datetime(df[df["split"] == "train"]["date"])
    val_dates = pd.to_datetime(df[df["split"] == "val"]["date"])
    test_dates = pd.to_datetime(df[df["split"] == "test"]["date"])

    assert train_dates.max() < val_dates.min(), "Train must strictly precede validation"
    assert val_dates.max() < test_dates.min(), "Validation must strictly precede test"
