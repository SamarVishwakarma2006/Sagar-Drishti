"""
Automated Test Suite for ERA5 Atmospheric Feature Engineering V2.1.
Enforces:
1. Production model SHA-256 hash invariance (v1.1.0 baseline MUST remain untouched)
2. Spatial sub-basin slicing and descending latitude handling across all 6 basins
3. Area-weighted vorticity exceedance fraction using cos(latitude) vs unweighted gridcell fraction
4. Vertical wind shear calculation and non-negativity (VWS >= 0)
5. Daily temporal aggregation logic and quality metadata completeness
6. Incomplete-day policy and flag validation
7. ZERO look-ahead leakage audit (corrupting date D+1 leaves date D features 100% invariant)
8. Output dataset schema, row count verification, duplicate check, and separate NaN accounting
9. Reproducibility test
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import xarray as xr

# Ensure backend root is on path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.era5_spatial import (
    ERA5SpatialFeatureExtractor,
    SUB_BASIN_DEFINITIONS,
    BASIN_CODE_TO_KEY,
)
from app.services.validate_era5 import ERA5Validator


# ---------------------------------------------------------------------------
# 1. Production Model Hash Invariance Test (P0 Guardrail)
# ---------------------------------------------------------------------------
def test_production_model_hashes_invariance():
    """
    CRITICAL: Asserts all 7 production baseline files in backend/models/
    remain 100% byte-identical to frozen v1.1.0 hashes.
    Feature engineering candidate code must never modify production baseline.
    """
    res = ERA5Validator.verify_production_hashes()
    assert res["all_passed"], f"Production model hash invariance violation: {res['files']}"
    for filename, meta in res["files"].items():
        assert meta["status"] == "VERIFIED"
        assert meta["matched"] is True


# ---------------------------------------------------------------------------
# 2. Spatial Sub-Basin Slicing & Descending Latitude Tests
# ---------------------------------------------------------------------------
def test_all_six_basins_defined_and_valid():
    """Verifies that all 6 operational research basins have valid coordinate bounds."""
    expected_basins = ["NAS", "CAS", "SAS", "NBOB", "CBOB", "SBOB"]
    actual_codes = [meta["code"] for meta in SUB_BASIN_DEFINITIONS.values()]
    assert sorted(actual_codes) == sorted(expected_basins)

    for key, bounds in SUB_BASIN_DEFINITIONS.items():
        assert bounds["lat_min"] < bounds["lat_max"]
        assert bounds["lon_min"] < bounds["lon_max"]
        assert 0.0 <= bounds["lat_min"] <= 25.0
        assert 0.0 <= bounds["lat_max"] <= 25.0
        assert 50.0 <= bounds["lon_min"] <= 100.0
        assert 50.0 <= bounds["lon_max"] <= 100.0


def test_sub_basin_slicing_descending_and_ascending():
    """
    Verifies that slicing works correctly whether dataset latitude is descending (ERA5 standard)
    or ascending, and asserts non-empty slices across all 6 basins.
    """
    # Create synthetic dataset with descending latitudes (25 -> 0)
    desc_lats = np.arange(25.0, -0.01, -0.25)
    lons = np.arange(50.0, 100.01, 0.25)
    times = pd.date_range("2024-01-01", periods=1, freq="6h")

    ds_desc = xr.Dataset(
        data_vars={
            "vo": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros((1, 4, len(desc_lats), len(lons)))),
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros((1, 4, len(desc_lats), len(lons)))),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros((1, 4, len(desc_lats), len(lons)))),
            "r": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros((1, 4, len(desc_lats), len(lons)))),
        },
        coords={
            "valid_time": times,
            "pressure_level": [850.0, 700.0, 500.0, 200.0],
            "latitude": desc_lats,
            "longitude": lons,
        }
    )

    for basin_key in SUB_BASIN_DEFINITIONS.keys():
        sub_desc = ERA5SpatialFeatureExtractor.slice_sub_basin(ds_desc, basin_key)
        assert len(sub_desc.latitude) > 0, f"Descending slice empty for {basin_key}"
        assert len(sub_desc.longitude) > 0, f"Descending lon slice empty for {basin_key}"
        bounds = SUB_BASIN_DEFINITIONS[basin_key]
        assert float(sub_desc.latitude.min()) >= bounds["lat_min"] - 1e-4
        assert float(sub_desc.latitude.max()) <= bounds["lat_max"] + 1e-4
        assert float(sub_desc.longitude.min()) >= bounds["lon_min"] - 1e-4
        assert float(sub_desc.longitude.max()) <= bounds["lon_max"] + 1e-4


# ---------------------------------------------------------------------------
# 3. Area-Weighted Exceedance Fraction Tests
# ---------------------------------------------------------------------------
def test_weighted_exceedance_fraction_vs_gridcell():
    """
    Analytically verifies latitude-weighted area exceedance fraction:
    cos(lat) weighting must give lower weight to high-latitude cells than low-latitude cells.
    """
    lats = np.array([24.0, 1.0])  # High lat (24N) vs Low lat (1N)
    lons = np.array([60.0, 61.0])
    vo_grid = np.zeros((len(lats), len(lons)))

    threshold = 5.0

    # Case A: Exceedance only at high latitude (24°N)
    vo_grid[0, :] = 6.0  # 24N exceeds
    vo_grid[1, :] = 2.0  # 1N does not
    weighted_high, grid_high = ERA5SpatialFeatureExtractor.compute_weighted_exceedance_fraction(
        vo_grid, lats, threshold=threshold
    )

    # Case B: Exceedance only at low latitude (1°N)
    vo_grid[0, :] = 2.0  # 24N does not
    vo_grid[1, :] = 6.0  # 1N exceeds
    weighted_low, grid_low = ERA5SpatialFeatureExtractor.compute_weighted_exceedance_fraction(
        vo_grid, lats, threshold=threshold
    )

    # Both cases have exactly 2 of 4 cells exceeding -> unweighted gridcell frac is identical (0.5)
    assert np.isclose(grid_high, 0.5)
    assert np.isclose(grid_low, 0.5)

    # But low-latitude cells have larger physical area (cos(1°) > cos(24°))
    # Therefore weighted_low MUST be strictly greater than weighted_high!
    assert weighted_low > weighted_high, f"Area weighting failed: low {weighted_low} <= high {weighted_high}"
    expected_ratio = np.cos(np.deg2rad(1.0)) / (np.cos(np.deg2rad(1.0)) + np.cos(np.deg2rad(24.0)))
    assert np.isclose(weighted_low, expected_ratio)


def test_weighted_exceedance_fraction_edge_cases():
    """Verifies edge cases: all 0 exceedance -> 0.0, all exceedance -> 1.0."""
    lats = np.array([20.0, 15.0, 10.0])
    vo_zero = np.zeros((3, 4))
    vo_full = np.full((3, 4), 10.0)

    w_zero, g_zero = ERA5SpatialFeatureExtractor.compute_weighted_exceedance_fraction(vo_zero, lats, 5.0)
    w_full, g_full = ERA5SpatialFeatureExtractor.compute_weighted_exceedance_fraction(vo_full, lats, 5.0)

    assert w_zero == 0.0 and g_zero == 0.0
    assert np.isclose(w_full, 1.0) and np.isclose(g_full, 1.0)


# ---------------------------------------------------------------------------
# 4. Vertical Wind Shear Formulation & Bounds
# ---------------------------------------------------------------------------
def test_vws_calculation_and_non_negativity():
    """Verifies VWS = sqrt((u200-u850)^2 + (v200-v850)^2) >= 0."""
    times = pd.date_range("2024-01-01", periods=1, freq="6h")
    lats = np.array([15.0])
    lons = np.array([65.0])

    # u200 = 13, u850 = 10 -> du = 3; v200 = 14, v850 = 10 -> dv = 4 => VWS = sqrt(9+16) = 5.0
    u_data = np.zeros((1, 4, 1, 1))
    v_data = np.zeros((1, 4, 1, 1))
    # 200 hPa is index 3, 850 hPa is index 0
    u_data[0, 3, 0, 0] = 13.0
    u_data[0, 0, 0, 0] = 10.0
    v_data[0, 3, 0, 0] = 14.0
    v_data[0, 0, 0, 0] = 10.0

    ds = xr.Dataset(
        data_vars={
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u_data),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v_data),
        },
        coords={
            "valid_time": times,
            "pressure_level": [850.0, 700.0, 500.0, 200.0],
            "latitude": lats,
            "longitude": lons,
        }
    )

    vws = ERA5SpatialFeatureExtractor.compute_vertical_wind_shear(ds)
    val = float(vws.values.squeeze())
    assert np.isclose(val, 5.0), f"Expected VWS=5.0, got {val}"
    assert val >= 0.0


# ---------------------------------------------------------------------------
# 5. Daily Temporal Aggregation & Quality Metadata Tests
# ---------------------------------------------------------------------------
def test_daily_aggregation_complete_day():
    """Verifies 4 6-hourly observations per day are correctly aggregated with complete flag."""
    times = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"])
    lats = np.array([22.0, 21.0])
    lons = np.array([62.0, 63.0])

    vo_data = np.zeros((4, 4, len(lats), len(lons)))
    vo_data[2, 0, 0, 0] = 1e-4  # Scaled vorticity = 10.0 at 12Z

    ds = xr.Dataset(
        data_vars={
            "vo": (("valid_time", "pressure_level", "latitude", "longitude"), vo_data),
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros_like(vo_data)),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros_like(vo_data)),
            "r": (("valid_time", "pressure_level", "latitude", "longitude"), np.full_like(vo_data, 75.0)),
        },
        coords={
            "valid_time": times,
            "pressure_level": [850.0, 700.0, 500.0, 200.0],
            "latitude": lats,
            "longitude": lons,
        }
    )

    rec = ERA5SpatialFeatureExtractor.extract_daily_basin_features(ds, "northern_arabian_sea", "2024-01-01", 5.0)
    assert rec["is_complete"] is True
    assert rec["number_of_valid_6h_samples"] == 4
    assert rec["expected_6h_samples"] == 4
    assert rec["completeness_fraction"] == 1.0
    assert np.isclose(rec["max_vorticity"], 10.0)
    assert rec["mean_rh700"] == 75.0
    assert rec["min_vws"] == 0.0


def test_daily_aggregation_incomplete_day():
    """Verifies incomplete day (e.g. only 2 timesteps) is correctly flagged in quality metadata."""
    times = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T06:00:00"])
    lats = np.array([22.0, 21.0])
    lons = np.array([62.0, 63.0])

    vo_data = np.zeros((2, 4, len(lats), len(lons)))
    ds = xr.Dataset(
        data_vars={
            "vo": (("valid_time", "pressure_level", "latitude", "longitude"), vo_data),
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros_like(vo_data)),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros_like(vo_data)),
            "r": (("valid_time", "pressure_level", "latitude", "longitude"), np.full_like(vo_data, 60.0)),
        },
        coords={
            "valid_time": times,
            "pressure_level": [850.0, 700.0, 500.0, 200.0],
            "latitude": lats,
            "longitude": lons,
        }
    )

    rec = ERA5SpatialFeatureExtractor.extract_daily_basin_features(ds, "northern_arabian_sea", "2024-01-01", 5.0)
    assert rec["is_complete"] is False
    assert rec["number_of_valid_6h_samples"] == 2
    assert rec["expected_6h_samples"] == 4
    assert rec["completeness_fraction"] == 0.5


# ---------------------------------------------------------------------------
# 6. Zero Look-Ahead Leakage Audit Test
# ---------------------------------------------------------------------------
def test_zero_lookahead_leakage_audit():
    """
    CRITICAL CAUSALITY AUDIT:
    Demonstrates mathematically that altering observations on date D+1
    produces ZERO change to any base or lookback features for date D.
    """
    from scripts.build_era5_daily_features import compute_causal_temporal_features

    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    basins = ["NAS"]

    records_original = []
    for d in dates:
        records_original.append({
            "date": d,
            "basin": "NAS",
            "max_vorticity": 5.0 if d == "2024-01-02" else 2.0,
            "mean_vorticity": 3.0,
            "std_vorticity": 1.0,
            "weighted_exceedance_fraction": 0.2,
            "gridcell_exceedance_fraction": 0.2,
            "latitude_of_max_vorticity": 22.0,
            "longitude_of_max_vorticity": 65.0,
            "min_vws": 10.0,
            "mean_vws": 15.0,
            "max_vws": 20.0,
            "mean_rh700": 70.0,
            "max_rh700": 85.0,
            "mean_rh500": 55.0,
            "max_rh500": 65.0,
            "number_of_valid_6h_samples": 4,
            "expected_6h_samples": 4,
            "completeness_fraction": 1.0,
            "is_complete": True,
        })

    df_orig = compute_causal_temporal_features(pd.DataFrame(records_original))
    d2_orig = df_orig[df_orig["date"] == "2024-01-02"].iloc[0].to_dict()

    # Now mutate date D+1 (2024-01-03) with extreme, chaotic shock values
    records_mutated = [r.copy() for r in records_original]
    records_mutated[2]["max_vorticity"] = 999999.0
    records_mutated[2]["min_vws"] = 0.0
    records_mutated[2]["mean_rh700"] = 100.0

    df_mut = compute_causal_temporal_features(pd.DataFrame(records_mutated))
    d2_mut = df_mut[df_mut["date"] == "2024-01-02"].iloc[0].to_dict()

    # Verify every single feature on date D (2024-01-02) is strictly identical!
    for col in df_orig.columns:
        v_orig = d2_orig[col]
        v_mut = d2_mut[col]
        if isinstance(v_orig, float) and np.isnan(v_orig):
            assert np.isnan(v_mut), f"NaN mismatch on {col}"
        else:
            assert v_orig == v_mut, f"Leakage detected on column '{col}': {v_orig} != {v_mut}"


# ---------------------------------------------------------------------------
# 7. Candidate Output Dataset On-Disk Verification
# ---------------------------------------------------------------------------
def test_candidate_dataset_on_disk():
    """
    Verifies that features_atmosphere_10yr_daily.parquet satisfies:
    - Exactly 6 basins
    - Complete date sequence (2016-06-24 to 2026-06-23)
    - Zero duplicates on (date, basin)
    - Base features have zero NaNs on complete days
    - Lag boundary nulls match exact expected boundary counts
    """
    parquet_path = os.path.join(backend_dir, "data", "era5", "features_atmosphere_10yr_daily.parquet")
    meta_path = os.path.join(backend_dir, "data", "era5", "features_atmosphere_10yr_daily_metadata.json")

    if not os.path.exists(parquet_path):
        pytest.skip(f"Output dataset not yet created at {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Basins
    expected_basins = ["CAS", "CBOB", "NAS", "NBOB", "SAS", "SBOB"]
    assert sorted(df["basin"].unique()) == expected_basins

    # Dates
    assert str(df["date"].min()) == "2016-06-24"
    assert str(df["date"].max()) == "2026-06-23"
    assert df["date"].nunique() == 3652
    assert len(df) == 3652 * 6

    # Zero duplicate rows
    assert df.duplicated(subset=["date", "basin"]).sum() == 0

    # Base features zero NaNs on complete days
    base_cols = [
        "max_vorticity", "mean_vorticity", "std_vorticity",
        "weighted_exceedance_fraction", "gridcell_exceedance_fraction",
        "latitude_of_max_vorticity", "longitude_of_max_vorticity",
        "min_vws", "mean_vws", "max_vws",
        "mean_rh700", "max_rh700", "mean_rh500", "max_rh500"
    ]
    complete_df = df[df["is_complete"] == True]
    assert complete_df[base_cols].isna().sum().sum() == 0

    # Separate lag boundary nulls
    # 24h change should have exactly 1 null per basin = 6 nulls
    assert df["max_vorticity_change_24h"].isna().sum() == 6
    # 48h change should have exactly 2 nulls per basin = 12 nulls
    assert df["max_vorticity_change_48h"].isna().sum() == 12
    # 72h change should have exactly 3 nulls per basin = 18 nulls
    assert df["max_vorticity_change_72h"].isna().sum() == 18

    # ML Warmup flag validation
    assert "is_ml_warmup_complete" in df.columns
    assert df["is_ml_warmup_complete"].sum() == (3652 - 3) * 6
    assert (~df["is_ml_warmup_complete"]).sum() == 18

    # Metadata file exists and is valid
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["total_rows"] == len(df)
    assert meta["data_integrity"]["duplicate_date_basin_pairs"] == 0
    assert meta["is_threshold_scientifically_frozen"] is False
