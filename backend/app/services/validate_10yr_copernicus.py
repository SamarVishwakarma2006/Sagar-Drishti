"""
Dedicated Copernicus Marine 10-Year Dataset Validation Engine.
Performs rigorous structural, temporal, spatial, physical, and cross-dataset consistency checks
comparing candidate 10-year physics datasets against the established 2-year baseline.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("sagar_drishti.copernicus_validator")

EXPECTED_VARIABLES = ["mlotst", "so", "thetao", "uo", "vo", "zos"]
EXPECTED_START = "2016-06-24"
EXPECTED_END = "2026-06-23"
EXPECTED_TIMESTEPS = 3652  # 10 years (365*10 + 2 leap days in 2020 & 2024)
TARGET_DEPTH = 0.49402499198913574
DEPTH_TOLERANCE = 0.05
EXPECTED_MIN_LAT = 0.0
EXPECTED_MAX_LAT = 25.0
EXPECTED_MIN_LON = 50.0
EXPECTED_MAX_LON = 100.0
EXPECTED_LAT_COUNT = 301
EXPECTED_LON_COUNT = 601

PHYSICAL_BOUNDS = {
    "thetao": {"min": 10.0, "max": 38.0, "unit": "°C", "name": "Potential Temperature"},
    "so": {"min": 0.0, "max": 45.0, "unit": "PSU", "name": "Salinity"},
    "uo": {"min": -4.0, "max": 4.0, "unit": "m/s", "name": "Eastward Velocity"},
    "vo": {"min": -4.0, "max": 4.0, "unit": "m/s", "name": "Northward Velocity"},
    "zos": {"min": -3.0, "max": 3.0, "unit": "m", "name": "Sea Surface Height"},
    "mlotst": {"min": 1.0, "max": 300.0, "unit": "m", "name": "Mixed Layer Depth"},
}


class Copernicus10YearValidator:
    """
    Validation engine enforcing non-negotiable checks on the 10-year Copernicus dataset.
    Does NOT eagerly load the entire dataset into RAM; uses chunked/lazy examination.
    """

    @classmethod
    def validate(
        cls,
        filepath: str,
        baseline_filepath: Optional[str] = None,
        sample_time_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete structural, temporal, spatial, and physical validation.
        Returns a structured validation dictionary with pass/fail status for each check.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"10-year Copernicus dataset not found at '{filepath}'")

        file_size_bytes = os.path.getsize(filepath)
        file_size_gb = round(file_size_bytes / (1024 ** 3), 3)

        checks: Dict[str, Any] = {}
        all_passed = True

        # Open lazily with xarray
        with xr.open_dataset(filepath) as ds:
            # 1. Variables check
            found_vars = list(ds.data_vars.keys())
            missing_vars = [v for v in EXPECTED_VARIABLES if v not in found_vars]
            extra_vars = [v for v in found_vars if v not in EXPECTED_VARIABLES]
            vars_ok = len(missing_vars) == 0
            checks["variables"] = {
                "passed": vars_ok,
                "found": found_vars,
                "missing": missing_vars,
                "extra": extra_vars,
            }
            if not vars_ok:
                all_passed = False

            # 2. Dimensions & Coordinates
            coords = list(ds.coords.keys())
            time_name = next((c for c in ["time", "record"] if c in coords or c in ds.dims), None)
            lat_name = next((c for c in ["latitude", "lat"] if c in coords or c in ds.dims), None)
            lon_name = next((c for c in ["longitude", "lon"] if c in coords or c in ds.dims), None)
            depth_name = next((c for c in ["depth", "lev"] if c in coords or c in ds.dims), None)

            checks["coordinates"] = {
                "time_name": time_name,
                "lat_name": lat_name,
                "lon_name": lon_name,
                "depth_name": depth_name,
            }

            # 3. Time dimension & continuity
            times = pd.to_datetime(ds[time_name].values)
            actual_count = len(times)
            start_date_str = times[0].strftime("%Y-%m-%d")
            end_date_str = times[-1].strftime("%Y-%m-%d")

            # Check duplicates
            duplicate_times = len(times) - len(np.unique(times))
            
            # Check daily continuity (expected exactly 1 day diff between consecutive steps)
            time_diffs = (times[1:] - times[:-1]).total_seconds() / 86400.0
            gaps_count = int(np.sum(time_diffs != 1.0))
            expected_dates = pd.date_range(start=EXPECTED_START, end=EXPECTED_END, freq="D")

            time_ok = (
                start_date_str == EXPECTED_START
                and end_date_str == EXPECTED_END
                and actual_count == EXPECTED_TIMESTEPS
                and duplicate_times == 0
                and gaps_count == 0
            )
            checks["temporal"] = {
                "passed": time_ok,
                "actual_timesteps": actual_count,
                "expected_timesteps": EXPECTED_TIMESTEPS,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "duplicate_timestamps": duplicate_times,
                "missing_date_gaps": gaps_count,
            }
            if not time_ok:
                all_passed = False

            # 4. Spatial grid & resolution
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            min_lat, max_lat = float(np.nanmin(lats)), float(np.nanmax(lats))
            min_lon, max_lon = float(np.nanmin(lons)), float(np.nanmax(lons))

            lat_monotonic = bool(np.all(np.diff(lats) > 0))
            lon_monotonic = bool(np.all(np.diff(lons) > 0))

            spatial_ok = (
                abs(min_lat - EXPECTED_MIN_LAT) < 0.1
                and abs(max_lat - EXPECTED_MAX_LAT) < 0.1
                and abs(min_lon - EXPECTED_MIN_LON) < 0.1
                and abs(max_lon - EXPECTED_MAX_LON) < 0.1
                and len(lats) == EXPECTED_LAT_COUNT
                and len(lons) == EXPECTED_LON_COUNT
                and lat_monotonic
                and lon_monotonic
            )
            checks["spatial"] = {
                "passed": spatial_ok,
                "lat_bounds": [round(min_lat, 4), round(max_lat, 4)],
                "lon_bounds": [round(min_lon, 4), round(max_lon, 4)],
                "lat_count": len(lats),
                "lon_count": len(lons),
                "lat_monotonic_increasing": lat_monotonic,
                "lon_monotonic_increasing": lon_monotonic,
                "resolution_deg": round(float(np.mean(np.diff(lats))), 4),
            }
            if not spatial_ok:
                all_passed = False

            # 5. Depth level verification (SURFACE ONLY Phase 1)
            depth_ok = False
            depth_val = None
            if depth_name and depth_name in ds:
                depth_vals = ds[depth_name].values
                if len(depth_vals) == 1:
                    depth_val = float(depth_vals[0])
                    depth_ok = abs(depth_val - TARGET_DEPTH) < DEPTH_TOLERANCE
            checks["depth"] = {
                "passed": depth_ok,
                "depth_levels_count": len(ds[depth_name].values) if depth_name and depth_name in ds else 0,
                "depth_level_m": depth_val,
                "target_depth_m": TARGET_DEPTH,
                "is_surface_only": depth_ok,
            }
            if not depth_ok:
                all_passed = False

            # 6. Physical value ranges & NaN audit on representative temporal samples
            if sample_time_indices is None:
                sample_time_indices = [0, actual_count // 4, actual_count // 2, (3 * actual_count) // 4, actual_count - 1]

            physics_summary: Dict[str, Any] = {}
            physics_ok = True
            for var in EXPECTED_VARIABLES:
                da = ds[var]
                sub = da.isel({time_name: sample_time_indices})
                if depth_name and depth_name in da.dims:
                    sub = sub.isel({depth_name: 0})
                
                vals = sub.values
                valid_mask = ~np.isnan(vals) & ~np.isinf(vals)
                nan_pct = round(float(np.mean(~valid_mask) * 100.0), 2)

                if np.sum(valid_mask) > 0:
                    v_min = float(np.min(vals[valid_mask]))
                    v_max = float(np.max(vals[valid_mask]))
                    v_mean = float(np.mean(vals[valid_mask]))
                else:
                    v_min, v_max, v_mean = 0.0, 0.0, 0.0

                expected_b = PHYSICAL_BOUNDS[var]
                in_bounds = (v_min >= expected_b["min"] - 1.0) and (v_max <= expected_b["max"] + 1.0)
                if not in_bounds or nan_pct > 65.0:  # Ocean land mask in 50-100E, 0-25N is typically ~30-50%
                    physics_ok = False

                physics_summary[var] = {
                    "valid_range": [round(v_min, 3), round(v_max, 3)],
                    "mean": round(v_mean, 3),
                    "nan_land_pct": nan_pct,
                    "in_physical_bounds": in_bounds,
                }

            checks["physical_ranges"] = {
                "passed": physics_ok,
                "variables": physics_summary,
            }
            if not physics_ok:
                all_passed = False

        # 7. Cross-comparison with baseline 2-year dataset
        baseline_comparison: Dict[str, Any] = {}
        if baseline_filepath and os.path.exists(baseline_filepath):
            with xr.open_dataset(baseline_filepath) as b_ds:
                baseline_comparison = {
                    "baseline_path": baseline_filepath,
                    "baseline_variables": list(b_ds.data_vars.keys()),
                    "baseline_timesteps": len(b_ds.time),
                    "baseline_start": str(pd.to_datetime(b_ds.time.values[0]).strftime("%Y-%m-%d")),
                    "baseline_end": str(pd.to_datetime(b_ds.time.values[-1]).strftime("%Y-%m-%d")),
                    "variables_match": set(found_vars) == set(b_ds.data_vars.keys()),
                    "spatial_resolution_match": (
                        len(lats) == len(b_ds.latitude.values) and len(lons) == len(b_ds.longitude.values)
                    ),
                    "depth_match": bool(
                        abs(float(b_ds.depth.values[0]) - (depth_val or 0.0)) < 1e-4
                    ),
                }

        return {
            "is_valid": all_passed,
            "filepath": os.path.abspath(filepath),
            "file_size_gb": file_size_gb,
            "checks": checks,
            "baseline_comparison": baseline_comparison,
            "validated_at": datetime.now().isoformat(),
        }
