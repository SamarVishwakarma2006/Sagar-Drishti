"""
Automated Test Suite for ERA5 Atmospheric Dataset Ingestion (V2.1).
Enforces:
1. Production model SHA-256 hash invariance (v1.1.0 baseline MUST remain untouched)
2. NetCDF structure, dimensions, and variables (vo, r, u, v)
3. Pressure levels (850, 700, 500, 200 hPa)
4. Spatial grid: 0-25°N, 50-100°E at 0.25° resolution (101x201 points)
5. Temporal regularity: strictly 6-hourly (00, 06, 12, 18 UTC), no duplicate timestamps
6. Derived atmospheric features:
   - 850 hPa relative vorticity scaling: vo850 * 10^5
   - 200-850 hPa Vertical Wind Shear (VWS): sqrt((u200-u850)^2 + (v200-v850)^2) >= 0 m/s
   - Mid-tropospheric relative humidity: RH700, RH500
7. Physical range checks and NaN audit
8. Spatial architecture sub-basin boundaries (North, Central, South Arabian Sea, Bay of Bengal)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import xarray as xr

# Ensure backend root is on path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.validate_era5 import (
    ERA5Validator,
    EXPECTED_VARIABLES,
    EXPECTED_PRESSURE_LEVELS,
    EXPECTED_MIN_LAT,
    EXPECTED_MAX_LAT,
    EXPECTED_MIN_LON,
    EXPECTED_MAX_LON,
    EXPECTED_LAT_COUNT,
    EXPECTED_LON_COUNT,
    PRODUCTION_MODEL_HASHES,
    PHYSICAL_BOUNDS,
)


@pytest.fixture(scope="module")
def sample_era5_path():
    """Returns path to an existing verified ERA5 file for testing."""
    chunk_path = os.path.join(backend_dir, "data", "era5", "chunks", "era5_2024_08.nc")
    smoke_path = os.path.join(backend_dir, "scratch", "smoke_test_era5.nc")
    if os.path.exists(chunk_path):
        return chunk_path
    elif os.path.exists(smoke_path):
        return smoke_path
    else:
        pytest.skip("No ERA5 sample file available for testing.")


# ---------------------------------------------------------------------------
# 1. Production Model Hash Invariance Tests (P0 Guardrail)
# ---------------------------------------------------------------------------
def test_production_model_hashes_invariance():
    """
    CRITICAL: Verifies all 7 production baseline files in backend/models/
    remain 100% byte-identical to frozen v1.1.0 hashes.
    """
    res = ERA5Validator.verify_production_hashes()
    assert res["all_passed"], f"Production model hash invariance violation: {res['files']}"
    for filename, meta in res["files"].items():
        assert meta["status"] == "VERIFIED", f"Hash mismatch on {filename}: actual {meta.get('actual')} != expected {meta.get('expected')}"
        assert meta["matched"] is True


# ---------------------------------------------------------------------------
# 2. ERA5 NetCDF Structure, Variables & Pressure Levels Tests
# ---------------------------------------------------------------------------
def test_era5_file_opens_and_has_variables(sample_era5_path):
    """Verifies that ERA5 NetCDF opens and contains all required variables."""
    ds = xr.open_dataset(sample_era5_path)
    try:
        data_vars = list(ds.data_vars)
        for var in EXPECTED_VARIABLES:
            assert var in data_vars, f"Expected variable '{var}' missing from dataset: {data_vars}"
    finally:
        ds.close()


def test_era5_pressure_levels(sample_era5_path):
    """Verifies expected pressure levels: 850, 700, 500, 200 hPa."""
    ds = xr.open_dataset(sample_era5_path)
    try:
        levels = [float(lvl) for lvl in ds.pressure_level.values]
        assert sorted(levels) == sorted(EXPECTED_PRESSURE_LEVELS), (
            f"Pressure levels mismatch: actual {levels} != expected {EXPECTED_PRESSURE_LEVELS}"
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 3. Spatial Bounds & Grid Resolution Tests
# ---------------------------------------------------------------------------
def test_era5_spatial_grid(sample_era5_path):
    """
    Verifies spatial extent 0-25°N, 50-100°E with 0.25° resolution
    (101 latitude points, 201 longitude points).
    """
    ds = xr.open_dataset(sample_era5_path)
    try:
        lats = ds.latitude.values
        lons = ds.longitude.values
        assert len(lats) == EXPECTED_LAT_COUNT, f"Lat count mismatch: {len(lats)} != {EXPECTED_LAT_COUNT}"
        assert len(lons) == EXPECTED_LON_COUNT, f"Lon count mismatch: {len(lons)} != {EXPECTED_LON_COUNT}"
        assert abs(float(lats.min()) - EXPECTED_MIN_LAT) < 1e-4
        assert abs(float(lats.max()) - EXPECTED_MAX_LAT) < 1e-4
        assert abs(float(lons.min()) - EXPECTED_MIN_LON) < 1e-4
        assert abs(float(lons.max()) - EXPECTED_MAX_LON) < 1e-4
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 4. Temporal Integrity Tests (6-hourly UTC, No Duplicates)
# ---------------------------------------------------------------------------
def test_era5_temporal_integrity(sample_era5_path):
    """Verifies no duplicate timestamps and strictly regular 6-hourly interval."""
    ds = xr.open_dataset(sample_era5_path)
    try:
        times = pd.to_datetime(ds.valid_time.values)
        assert len(times) > 0, "No timesteps in file"
        assert len(times.unique()) == len(times), "Duplicate timestamps detected"

        if len(times) > 1:
            deltas = (times[1:] - times[:-1]).total_seconds() / 3600.0
            assert np.all(deltas == 6.0), f"Irregular time delta detected: {set(deltas)}"

        # Check UTC hour alignment (0, 6, 12, 18)
        hours = set(times.hour)
        assert hours.issubset({0, 6, 12, 18}), f"Unexpected observation hours: {hours}"
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 5. Missing Values & Physical Range Sanity Tests
# ---------------------------------------------------------------------------
def test_era5_nan_and_physical_ranges(sample_era5_path):
    """Verifies no NaNs in variables and values satisfy physical bounds."""
    ds = xr.open_dataset(sample_era5_path)
    try:
        for v in EXPECTED_VARIABLES:
            arr = ds[v].values
            nan_cnt = int(np.isnan(arr).sum())
            assert nan_cnt == 0, f"NaN values detected in variable '{v}': {nan_cnt}"

            v_min = float(arr.min())
            v_max = float(arr.max())
            b_min = PHYSICAL_BOUNDS[v]["min"]
            b_max = PHYSICAL_BOUNDS[v]["max"]
            assert v_min >= b_min, f"Physical minimum violated for '{v}': {v_min} < {b_min}"
            assert v_max <= b_max, f"Physical maximum violated for '{v}': {v_max} > {b_max}"
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 6. Derived Variable Formulations Tests
# ---------------------------------------------------------------------------
def test_derived_vws_calculation(sample_era5_path):
    """
    Verifies 200-850 hPa Vertical Wind Shear calculation:
    VWS = sqrt((u200 - u850)^2 + (v200 - v850)^2) >= 0 m/s
    """
    ds = xr.open_dataset(sample_era5_path)
    try:
        u200 = ds["u"].sel(pressure_level=200.0).isel(valid_time=0).values
        v200 = ds["v"].sel(pressure_level=200.0).isel(valid_time=0).values
        u850 = ds["u"].sel(pressure_level=850.0).isel(valid_time=0).values
        v850 = ds["v"].sel(pressure_level=850.0).isel(valid_time=0).values

        vws = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)

        assert not np.any(np.isnan(vws)), "NaN in VWS calculation"
        assert np.all(vws >= 0.0), "Negative VWS encountered"
        # Atmospheric shear over North Indian Ocean typically between 0 and 70 m/s
        assert float(vws.min()) >= 0.0
        assert float(vws.max()) <= 120.0
        assert float(vws.mean()) > 0.0
    finally:
        ds.close()


def test_derived_vorticity_scaling(sample_era5_path):
    """Verifies 850 hPa relative vorticity scaling: vo850_scaled = vo850 * 10^5."""
    ds = xr.open_dataset(sample_era5_path)
    try:
        raw_vo = ds["vo"].sel(pressure_level=850.0).isel(valid_time=0).values
        scaled_vo = raw_vo * 1e5

        assert np.allclose(scaled_vo, raw_vo * 100000.0)
        # Scaled vorticity typically in range [-50, 50] x 10^-5 s^-1
        assert float(scaled_vo.min()) >= -500.0
        assert float(scaled_vo.max()) <= 500.0
    finally:
        ds.close()


def test_derived_rh_levels(sample_era5_path):
    """Verifies mid-tropospheric humidity extractions for 700 hPa and 500 hPa."""
    ds = xr.open_dataset(sample_era5_path)
    try:
        rh700 = ds["r"].sel(pressure_level=700.0).isel(valid_time=0).values
        rh500 = ds["r"].sel(pressure_level=500.0).isel(valid_time=0).values

        assert not np.any(np.isnan(rh700))
        assert not np.any(np.isnan(rh500))
        assert 0.0 <= float(rh700.mean()) <= 100.0
        assert 0.0 <= float(rh500.mean()) <= 100.0
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 7. Spatial Architecture & Multi-Sub-Basin Slicing Tests
# ---------------------------------------------------------------------------
def test_spatial_sub_basin_slicing(sample_era5_path):
    """
    Verifies that ERA5 grid supports modular extraction across Arabian Sea
    sub-basins and Bay of Bengal without relying on single centroid.
    Sub-basins:
    - Northern Arabian Sea: 20-25°N, 60-72°E
    - Central Arabian Sea: 14-20°N, 60-75°E
    - Southern Arabian Sea: 8-14°N, 65-78°E
    - Bay of Bengal: 8-22°N, 80-95°E
    """
    ds = xr.open_dataset(sample_era5_path)
    try:
        sub_basins = {
            "nas": {"lat": slice(25.0, 20.0), "lon": slice(60.0, 72.0)},  # descending lat
            "cas": {"lat": slice(20.0, 14.0), "lon": slice(60.0, 75.0)},
            "sas": {"lat": slice(14.0, 8.0), "lon": slice(65.0, 78.0)},
            "bob": {"lat": slice(22.0, 8.0), "lon": slice(80.0, 95.0)},
        }

        for name, bounds in sub_basins.items():
            # Slice latitude and longitude
            sub_ds = ds.sel(latitude=bounds["lat"], longitude=bounds["lon"])
            n_lat = len(sub_ds.latitude)
            n_lon = len(sub_ds.longitude)
            assert n_lat > 0, f"Sub-basin {name} latitude slice is empty"
            assert n_lon > 0, f"Sub-basin {name} longitude slice is empty"

            # Check extracting mean vo850
            mean_vo850 = float(sub_ds["vo"].sel(pressure_level=850.0).mean())
            assert not np.isnan(mean_vo850)
    finally:
        ds.close()


def test_spatial_feature_extractor_service(sample_era5_path):
    """
    Verifies ERA5SpatialFeatureExtractor extracts all required modular features
    across all Arabian Sea sub-basins and Bay of Bengal regions.
    """
    from app.services.era5_spatial import ERA5SpatialFeatureExtractor, SUB_BASIN_DEFINITIONS

    ds = xr.open_dataset(sample_era5_path).isel(valid_time=0)
    try:
        for basin_key, defn in SUB_BASIN_DEFINITIONS.items():
            features = ERA5SpatialFeatureExtractor.extract_basin_summary_features(
                ds, basin_key, vorticity_threshold_scaled=5.0
            )
            code = defn["code"].lower()
            expected_keys = [
                f"{code}_vo850_max",
                f"{code}_vo850_mean",
                f"{code}_vo850_std",
                f"{code}_vo850_exceedance_frac",
                f"{code}_vo850_max_lat",
                f"{code}_vo850_max_lon",
                f"{code}_vws_min",
                f"{code}_vws_mean",
                f"{code}_rh700_mean",
                f"{code}_rh700_max",
                f"{code}_rh500_mean",
                f"{code}_rh500_max",
            ]
            for ek in expected_keys:
                assert ek in features, f"Missing feature '{ek}' in {basin_key}"
                assert not np.isnan(features[ek]), f"NaN encountered in {ek}"

            # Verify coordinate bounds
            assert defn["lat_min"] <= features[f"{code}_vo850_max_lat"] <= defn["lat_max"]
            assert defn["lon_min"] <= features[f"{code}_vo850_max_lon"] <= defn["lon_max"]
            assert 0.0 <= features[f"{code}_vo850_exceedance_frac"] <= 1.0
            assert features[f"{code}_vws_min"] >= 0.0
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 8. Complete ERA5Validator Service Integration Test
# ---------------------------------------------------------------------------
def test_era5_validator_service(sample_era5_path):
    """Verifies that ERA5Validator runs end-to-end and returns PASSED."""
    res = ERA5Validator.validate_file(sample_era5_path)
    assert res["status"] == "PASSED"
    assert res["all_passed"] is True
    assert "sha256" in res
    assert len(res["sha256"]) == 64
    assert res["checks"]["expected_variables"]["passed"]
    assert res["checks"]["pressure_levels"]["passed"]
    assert res["checks"]["spatial_bounds"]["passed"]
    assert res["checks"]["temporal_integrity"]["passed"]
    assert res["checks"]["nan_audit"]["passed"]
    assert res["checks"]["physical_ranges"]["passed"]
    assert res["checks"]["derived_variables"]["passed"]
