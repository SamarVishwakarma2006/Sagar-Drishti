"""
Authoritative Validation Engine for ERA5 Atmospheric Reanalysis Dataset (V2.1).
Enforces structural, temporal, spatial, physical, and derived-variable consistency checks
for individual chunk files, yearly files, and the canonical 10-year merged dataset.
Also guarantees byte-invariance of production baseline model artifacts.
"""

import os
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("sagar_drishti.validate_era5")

# Canonical ERA5 Grid & Variables Specification
EXPECTED_VARIABLES = ["vo", "r", "u", "v"]
EXPECTED_PRESSURE_LEVELS = [850.0, 700.0, 500.0, 200.0]
EXPECTED_MIN_LAT = 0.0
EXPECTED_MAX_LAT = 25.0
EXPECTED_MIN_LON = 50.0
EXPECTED_MAX_LON = 100.0
EXPECTED_LAT_COUNT = 101
EXPECTED_LON_COUNT = 201
EXPECTED_FREQUENCY_HOURS = 6

# Canonical 10-Year Temporal Bounds (matching Copernicus 2016-06-24 to 2026-06-23)
CANONICAL_START_STR = "2016-06-24T00:00:00"
CANONICAL_END_STR = "2026-06-23T18:00:00"
CANONICAL_TOTAL_TIMESTEPS = 14608  # 3652 daily periods * 4 observations/day

# Production Baseline SHA-256 Hashes (v1.1.0)
PRODUCTION_MODEL_HASHES = {
    "model_event_type.joblib": "2ed7172fb06475aa00241ed465a5851b6155562f23a91291d29899ac56ff2fc2",
    "model_metadata.json": "f41ad99091a5f2366cd06bf5d8f9aa66135218b34f7b789968ab8c1626d025ad",
    "risk_model.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
    "risk_model_0d.joblib": "54720c2218a46977c2f93e94889d4242f357c313ab069935cf57bdc31f1b61cd",
    "risk_model_1d.joblib": "cb05c4408ef2f3b66999f8b0016b9dbd1285bc88c1c8f6e32aae310817ccd57c",
    "risk_model_2d.joblib": "6e6e34c5806c2db83a61369eb8488dedb827845cba8a340c0b8b286363a52843",
    "risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
}

# Physical Sanity Bounds
PHYSICAL_BOUNDS = {
    "vo": {"min": -0.005, "max": 0.005, "units": "s**-1", "name": "Relative Vorticity"},
    "r": {"min": -20.0, "max": 220.0, "units": "%", "name": "Relative Humidity"},
    "u": {"min": -120.0, "max": 120.0, "units": "m s**-1", "name": "Eastward Wind"},
    "v": {"min": -120.0, "max": 120.0, "units": "m s**-1", "name": "Northward Wind"},
    "vws": {"min": 0.0, "max": 120.0, "units": "m s**-1", "name": "Vertical Wind Shear (200-850 hPa)"},
}


class ERA5Validator:
    """Validation engine for ERA5 atmospheric dataset."""

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Computes SHA-256 checksum of a file."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @classmethod
    def verify_production_hashes(cls, models_dir: Optional[str] = None) -> Dict[str, Any]:
        """Verifies byte-invariance of production baseline model artifacts."""
        if models_dir is None:
            models_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "models")
            )
        results = {}
        all_passed = True

        for filename, expected_hash in PRODUCTION_MODEL_HASHES.items():
            path = os.path.join(models_dir, filename)
            if not os.path.exists(path):
                results[filename] = {"status": "MISSING", "expected": expected_hash}
                all_passed = False
                continue
            actual_hash = cls.compute_sha256(path)
            matched = (actual_hash == expected_hash)
            results[filename] = {
                "status": "VERIFIED" if matched else "MISMATCH",
                "matched": matched,
                "expected": expected_hash,
                "actual": actual_hash,
            }
            if not matched:
                all_passed = False

        return {"all_passed": all_passed, "files": results}

    @classmethod
    def validate_file(
        cls,
        filepath: str,
        expected_start: Optional[str] = None,
        expected_end: Optional[str] = None,
        expected_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validates an ERA5 NetCDF file for structural, spatial, temporal,
        and physical integrity.
        """
        if not os.path.exists(filepath):
            return {"status": "FAILED", "error": f"File not found: {filepath}"}

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return {"status": "FAILED", "error": "File size is 0 bytes"}

        checks = {}
        try:
            ds = xr.open_dataset(filepath)
        except Exception as e:
            return {"status": "FAILED", "error": f"Failed to open NetCDF: {str(e)}"}

        try:
            # 1. Variables check
            available_vars = list(ds.data_vars)
            vars_ok = all(v in available_vars for v in EXPECTED_VARIABLES)
            checks["expected_variables"] = {
                "passed": vars_ok,
                "available": available_vars,
                "expected": EXPECTED_VARIABLES,
            }

            # 2. Coordinates & Pressure levels check
            levels = [float(lvl) for lvl in ds.pressure_level.values]
            levels_ok = sorted(levels) == sorted(EXPECTED_PRESSURE_LEVELS)
            checks["pressure_levels"] = {
                "passed": levels_ok,
                "actual": levels,
                "expected": EXPECTED_PRESSURE_LEVELS,
            }

            # 3. Spatial bounds & resolution check
            lats = ds.latitude.values
            lons = ds.longitude.values
            lat_ok = (
                len(lats) == EXPECTED_LAT_COUNT
                and abs(float(lats.min()) - EXPECTED_MIN_LAT) < 1e-4
                and abs(float(lats.max()) - EXPECTED_MAX_LAT) < 1e-4
            )
            lon_ok = (
                len(lons) == EXPECTED_LON_COUNT
                and abs(float(lons.min()) - EXPECTED_MIN_LON) < 1e-4
                and abs(float(lons.max()) - EXPECTED_MAX_LON) < 1e-4
            )
            checks["spatial_bounds"] = {
                "passed": lat_ok and lon_ok,
                "lat_range": [float(lats.min()), float(lats.max())],
                "lat_count": len(lats),
                "lon_range": [float(lons.min()), float(lons.max())],
                "lon_count": len(lons),
            }

            # 4. Temporal continuity and regularity
            times = pd.to_datetime(ds.valid_time.values)
            n_times = len(times)
            no_duplicates = bool(len(times.unique()) == n_times)

            # Check 6-hourly delta
            if n_times > 1:
                deltas = (times[1:] - times[:-1]).total_seconds() / 3600.0
                strictly_6hourly = bool(np.all(deltas == 6.0))
            else:
                strictly_6hourly = True

            start_time_str = times[0].strftime("%Y-%m-%dT%H:%M:%S")
            end_time_str = times[-1].strftime("%Y-%m-%dT%H:%M:%S")

            temporal_passed = no_duplicates and strictly_6hourly
            if expected_start:
                temporal_passed = temporal_passed and (start_time_str >= expected_start[:19])
            if expected_end:
                temporal_passed = temporal_passed and (end_time_str <= expected_end[:19])
            if expected_timesteps:
                temporal_passed = temporal_passed and (n_times == expected_timesteps)

            checks["temporal_integrity"] = {
                "passed": temporal_passed,
                "n_timesteps": n_times,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "no_duplicates": no_duplicates,
                "strictly_6hourly": strictly_6hourly,
            }

            # 5 & 6. Missing values / NaNs and physical ranges audit (single pass per variable, chunked over time)
            nan_counts = {}
            range_results = {}
            ranges_passed = True
            step = 1000

            for v in EXPECTED_VARIABLES:
                if v in ds:
                    total_nan = 0
                    v_min = float("inf")
                    v_max = float("-inf")
                    var = ds[v]
                    for i in range(0, n_times, step):
                        chunk_vals = var.isel(valid_time=slice(i, i + step)).values
                        nan_mask = np.isnan(chunk_vals)
                        total_nan += int(nan_mask.sum())
                        if not nan_mask.all():
                            c_min = float(np.nanmin(chunk_vals))
                            c_max = float(np.nanmax(chunk_vals))
                            if c_min < v_min:
                                v_min = c_min
                            if c_max > v_max:
                                v_max = c_max

                    nan_counts[v] = total_nan
                    b_min = PHYSICAL_BOUNDS[v]["min"]
                    b_max = PHYSICAL_BOUNDS[v]["max"]
                    in_bounds = (v_min >= b_min and v_max <= b_max)
                    range_results[v] = {
                        "min": v_min,
                        "max": v_max,
                        "in_bounds": in_bounds,
                    }
                    if not in_bounds:
                        ranges_passed = False

            no_nans = all(cnt == 0 for cnt in nan_counts.values())
            checks["nan_audit"] = {
                "passed": no_nans,
                "nan_counts": nan_counts,
            }
            checks["physical_ranges"] = {
                "passed": ranges_passed,
                "variables": range_results,
            }

            # 7. Derived variables calculation check
            # Sample first timestep to test derived feature computations
            vo850 = ds["vo"].sel(pressure_level=850.0).isel(valid_time=0).values * 1e5
            u850 = ds["u"].sel(pressure_level=850.0).isel(valid_time=0).values
            v850 = ds["v"].sel(pressure_level=850.0).isel(valid_time=0).values
            u200 = ds["u"].sel(pressure_level=200.0).isel(valid_time=0).values
            v200 = ds["v"].sel(pressure_level=200.0).isel(valid_time=0).values
            vws = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
            r700 = ds["r"].sel(pressure_level=700.0).isel(valid_time=0).values
            r500 = ds["r"].sel(pressure_level=500.0).isel(valid_time=0).values

            derived_ok = (
                not np.any(np.isnan(vws))
                and np.all(vws >= 0.0)
                and not np.any(np.isnan(vo850))
                and not np.any(np.isnan(r700))
                and not np.any(np.isnan(r500))
            )
            checks["derived_variables"] = {
                "passed": derived_ok,
                "sample_vws_min": float(vws.min()),
                "sample_vws_max": float(vws.max()),
                "sample_vo850_scaled_min": float(vo850.min()),
                "sample_vo850_scaled_max": float(vo850.max()),
                "sample_r700_mean": float(r700.mean()),
                "sample_r500_mean": float(r500.mean()),
            }

            all_passed = all(c.get("passed", False) for c in checks.values())

            return {
                "status": "PASSED" if all_passed else "FAILED",
                "all_passed": all_passed,
                "filepath": filepath,
                "file_size_bytes": file_size,
                "sha256": cls.compute_sha256(filepath),
                "checks": checks,
            }

        finally:
            ds.close()
