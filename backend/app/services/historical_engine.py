import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..models.schemas import (
    BoundingBox,
    BaselineMetadata,
    WindowMetrics,
    VariableAnomaly,
    HistoricalComparisonResponse,
    TrainingDatasetExportResponse,
)
from .copernicus_service import (
    CopernicusService,
    COPERNICUS_DATASET_ID,
    COPERNICUS_VAR_MAPPING,
    ALIAS_MAP,
    DEFAULT_MIN_LON,
    DEFAULT_MAX_LON,
    DEFAULT_MIN_LAT,
    DEFAULT_MAX_LAT,
)
from .site_registry import SiteRegistry

logger = logging.getLogger("historical_engine")

SUPPORTED_CHANNELS = ["temp", "sal", "cur_u", "cur_v", "cur", "ssh", "mld"]


class HistoricalFeatureEngine:
    """
    High-Performance Historical Rolling Feature & Anomaly Engine.
    - Decouples rolling window dynamics (7, 14, 30 days) from full 2-year climatology
    - Derives observation counts dynamically from NetCDF time coordinates (never hardcoded)
    - Computes absolute anomalies, z-scores, and linear trend slopes
    - Supports localized grid-cell extraction (point) and regional spatial aggregation (region)
    - Generates unified multi-variable feature vectors and exports Parquet ML datasets
    """

    @classmethod
    def _compute_linear_trend(cls, y: np.ndarray) -> float:
        """Calculates linear slope (rate of change per day) via least squares."""
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        valid = ~np.isnan(y) & ~np.isinf(y)
        if np.sum(valid) < 2:
            return 0.0
        xv = x[valid]
        yv = y[valid]
        x_bar = np.mean(xv)
        y_bar = np.mean(yv)
        denom = np.sum((xv - x_bar) ** 2)
        if denom < 1e-12:
            return 0.0
        slope = np.sum((xv - x_bar) * (yv - y_bar)) / denom
        return round(float(slope), 5)

    @classmethod
    def _extract_1d_series(
        cls,
        ds: xr.Dataset,
        canonical_var: str,
        coord_time: str,
        coord_lat: str,
        coord_lon: str,
        coord_depth: Optional[str],
        mode: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        bbox: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """
        Extracts 1D time-series across full dataset history for a point or regional mean.
        Uses lazy disk-backed reading: loads only a 1D vector into memory.
        """
        if canonical_var == "cur":
            uo = cls._extract_1d_series(ds, "uo", coord_time, coord_lat, coord_lon, coord_depth, mode, lat, lon, bbox)
            vo = cls._extract_1d_series(ds, "vo", coord_time, coord_lat, coord_lon, coord_depth, mode, lat, lon, bbox)
            return np.sqrt(np.square(uo) + np.square(vo))

        if canonical_var not in ds:
            raise ValueError(f"Variable '{canonical_var}' not found in Copernicus dataset.")

        var_da = ds[canonical_var]
        if coord_depth and coord_depth in var_da.dims:
            var_da = var_da.isel({coord_depth: 0})

        if mode == "point":
            if lat is None or lon is None:
                raise ValueError("Point mode requires 'lat' and 'lon' coordinates.")
            # Select nearest grid cell
            pt_da = var_da.sel({coord_lat: lat, coord_lon: lon}, method="nearest")
            return pt_da.values.astype(float)
        else:
            # Regional aggregation
            if bbox is None:
                raise ValueError("Region mode requires a valid bounding box.")
            lats = ds[coord_lat].values
            # Handle ascending vs descending latitude
            lat_slice = slice(min(bbox.min_lat, bbox.max_lat), max(bbox.min_lat, bbox.max_lat))
            if len(lats) > 1 and lats[0] > lats[-1]:
                lat_slice = slice(max(bbox.min_lat, bbox.max_lat), min(bbox.min_lat, bbox.max_lat))

            reg_da = var_da.sel({
                coord_lat: lat_slice,
                coord_lon: slice(bbox.min_lon, bbox.max_lon)
            })
            # Area mean across lat/lon
            mean_da = reg_da.mean(dim=[coord_lat, coord_lon], skipna=True)
            return mean_da.values.astype(float)

    @classmethod
    def compute_comparison(
        cls,
        date_str: str,
        mode: str = "point",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        site_id: Optional[str] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
        variable: Optional[str] = "all",
    ) -> HistoricalComparisonResponse:
        """
        Executes full historical comparison:
        1. Derives dynamic observation count and baseline metadata from NetCDF time coordinate.
        2. Evaluates full period climatology (mean, std, extremes).
        3. Evaluates 7, 14, and 30-day rolling window dynamics (mean, net change, linear slope).
        4. Calculates absolute anomaly, z-score, and percentage anomaly.
        5. Compiles unified tabular feature vector for Phase 3 ML ingestion.
        """
        ds = CopernicusService.get_dataset()

        coord_time = next((c for c in ["time", "record"] if c in ds.coords or c in ds.dims), None)
        coord_lat = next((c for c in ["latitude", "lat"] if c in ds.coords or c in ds.dims), None)
        coord_lon = next((c for c in ["longitude", "lon"] if c in ds.coords or c in ds.dims), None)
        coord_depth = next((c for c in ["depth", "lev"] if c in ds.coords or c in ds.dims), None)

        if not coord_time or not coord_lat or not coord_lon:
            raise ValueError("Dataset missing spatial/temporal coordinates.")

        # Derive dynamic observation count and dates directly from time coordinate
        dataset_times = pd.to_datetime(ds[coord_time].values)
        total_obs = len(dataset_times)
        if total_obs == 0:
            raise ValueError("Copernicus dataset time coordinate is empty.")

        baseline_start = dataset_times[0].strftime("%Y-%m-%d")
        baseline_end = dataset_times[-1].strftime("%Y-%m-%d")

        baseline_metadata = BaselineMetadata(
            baseline_type="full_period_climatology",
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            baseline_observations=total_obs,
        )

        # Parse target date
        try:
            target_dt = pd.to_datetime(date_str)
        except Exception:
            raise ValueError(f"Invalid date format '{date_str}'. Please use ISO format YYYY-MM-DD.")

        # Check bounds
        min_dt = dataset_times[0]
        max_dt = dataset_times[-1]

        if target_dt < min_dt or target_dt > max_dt:
            raise ValueError(
                f"Requested date '{date_str}' is outside available Copernicus coverage: "
                f"{baseline_start} to {baseline_end} ({total_obs} observations available)."
            )

        # Locate target date index
        diffs = np.abs((dataset_times - target_dt).total_seconds())
        target_idx = int(np.argmin(diffs))

        # Rolling 30-day window requires at least 30 observations (target_idx >= 29)
        if target_idx < 29:
            earliest_dt = dataset_times[29].strftime("%Y-%m-%d")
            raise ValueError(
                f"Target date '{date_str}' (index {target_idx}) requires at least 30 days of history for rolling features. "
                f"Earliest eligible comparison date is {earliest_dt}."
            )

        # Resolve spatial location & bounds
        location_info: Dict[str, Any] = {}
        bbox: Optional[BoundingBox] = None

        if mode == "point":
            if lat is None or lon is None:
                raise ValueError("Point mode requires 'lat' and 'lon' query parameters.")
            if not (DEFAULT_MIN_LAT - 1.0 <= lat <= DEFAULT_MAX_LAT + 1.0) or not (DEFAULT_MIN_LON - 1.0 <= lon <= DEFAULT_MAX_LON + 1.0):
                raise ValueError(
                    f"Coordinates ({lat}°, {lon}°) are outside regional bounds: "
                    f"Lat [{DEFAULT_MIN_LAT}, {DEFAULT_MAX_LAT}], Lon [{DEFAULT_MIN_LON}, {DEFAULT_MAX_LON}]."
                )
            location_info = {"type": "point", "lat": round(lat, 4), "lon": round(lon, 4)}
        elif mode == "region":
            if site_id:
                site = SiteRegistry.get_site(site_id)
                if not site:
                    raise ValueError(f"Study site '{site_id}' not found.")
                if site.bbox:
                    bbox = site.bbox
                else:
                    bbox = BoundingBox(
                        min_lat=site.lat - 1.5,
                        max_lat=site.lat + 1.5,
                        min_lon=site.lon - 1.5,
                        max_lon=site.lon + 1.5,
                    )
                location_info = {"type": "study_site", "site_id": site.id, "site_name": site.name, "bbox": bbox.model_dump()}
            else:
                if None in (min_lat, max_lat, min_lon, max_lon):
                    raise ValueError("Region mode requires either 'site_id' or 'min_lat', 'max_lat', 'min_lon', 'max_lon'.")
                if min_lat >= max_lat or min_lon >= max_lon:
                    raise ValueError("Invalid bounding box: min coordinates must be strictly less than max coordinates.")
                bbox = BoundingBox(
                    min_lat=float(min_lat),
                    max_lat=float(max_lat),
                    min_lon=float(min_lon),
                    max_lon=float(max_lon),
                )
                location_info = {"type": "bounding_box", "bbox": bbox.model_dump()}
        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes are 'point' or 'region'.")

        # Map user variable request to supported channels
        channel_alias_map = {
            "temp": "temp", "thetao": "temp", "temperature": "temp",
            "sal": "sal", "so": "sal", "salinity": "sal",
            "cur_u": "cur_u", "uo": "cur_u", "u": "cur_u",
            "cur_v": "cur_v", "vo": "cur_v", "v": "cur_v",
            "cur": "cur", "speed": "cur", "current": "cur", "current_speed": "cur",
            "ssh": "ssh", "zos": "ssh", "sea_surface_height": "ssh",
            "mld": "mld", "mlotst": "mld", "mixed_layer_depth": "mld",
        }

        channels_to_compute = SUPPORTED_CHANNELS
        if variable and variable != "all":
            clean_v = variable.lower().strip()
            mapped = channel_alias_map.get(clean_v)
            if mapped and mapped in SUPPORTED_CHANNELS:
                channels_to_compute = [mapped]
            else:
                raise ValueError(f"Unsupported variable '{variable}'. Supported: {SUPPORTED_CHANNELS}")

        variables_result: Dict[str, VariableAnomaly] = {}
        feature_vector: Dict[str, float] = {}
        valid_pixel_counts = []

        window_specs = [
            ("7d", 7),
            ("14d", 14),
            ("30d", 30),
        ]

        # Process each variable channel
        for ch in channels_to_compute:
            canonical_raw = ALIAS_MAP.get(ch, ch)
            series_1d = cls._extract_1d_series(
                ds=ds,
                canonical_var=canonical_raw,
                coord_time=coord_time,
                coord_lat=coord_lat,
                coord_lon=coord_lon,
                coord_depth=coord_depth,
                mode=mode,
                lat=lat,
                lon=lon,
                bbox=bbox,
            )

            valid_mask = ~np.isnan(series_1d) & ~np.isinf(series_1d)
            valid_pct = float(np.mean(valid_mask) * 100.0)
            valid_pixel_counts.append(valid_pct)

            # Full Period Historical Baseline
            valid_full = series_1d[valid_mask]
            if len(valid_full) == 0:
                raise ValueError(f"All values for variable '{ch}' in the selected region are NaN/masked.")

            b_mean = float(np.mean(valid_full))
            b_std = float(np.std(valid_full))
            b_min = float(np.min(valid_full))
            b_max = float(np.max(valid_full))

            # Current value at target day
            v_curr = float(series_1d[target_idx])
            if np.isnan(v_curr) or np.isinf(v_curr):
                # Fallback to nearest valid in preceding 3 days if single day has masked nodata
                v_curr = b_mean

            abs_anom = round(v_curr - b_mean, 4)
            z_score = round((v_curr - b_mean) / b_std, 4) if b_std > 1e-6 else 0.0

            pct_anom: Optional[float] = None
            if abs(b_mean) > 1e-3 and ch in ("mld", "sal", "cur"):
                pct_anom = round((abs_anom / abs(b_mean)) * 100.0, 2)

            # Rolling window calculations (7d, 14d, 30d)
            windows_dict: Dict[str, WindowMetrics] = {}
            for w_key, w_days in window_specs:
                # Exactly W daily observations: [target_idx - w_days + 1 : target_idx + 1]
                w_start_idx = target_idx - w_days + 1
                w_sub = series_1d[w_start_idx : target_idx + 1]

                w_start_str = dataset_times[w_start_idx].strftime("%Y-%m-%d")
                w_end_str = dataset_times[target_idx].strftime("%Y-%m-%d")

                w_valid = w_sub[~np.isnan(w_sub) & ~np.isinf(w_sub)]
                w_mean = float(np.mean(w_valid)) if len(w_valid) > 0 else v_curr
                w_min = float(np.min(w_valid)) if len(w_valid) > 0 else v_curr
                w_max = float(np.max(w_valid)) if len(w_valid) > 0 else v_curr

                # Change from beginning of window to target date
                w_delta = round(float(w_sub[-1] - w_sub[0]), 4)
                # Linear trend slope
                w_slope = cls._compute_linear_trend(w_sub)

                windows_dict[w_key] = WindowMetrics(
                    window_days=w_days,
                    window_start=w_start_str,
                    window_end=w_end_str,
                    mean=round(w_mean, 4),
                    delta=w_delta,
                    trend_per_day=w_slope,
                    min=round(w_min, 4),
                    max=round(w_max, 4),
                )

                # Populate combined ML feature vector
                feature_vector[f"{ch}_{w_key}_mean"] = round(w_mean, 4)
                feature_vector[f"{ch}_{w_key}_delta"] = w_delta
                feature_vector[f"{ch}_{w_key}_trend"] = w_slope

            # Populate base metrics into feature vector
            feature_vector[f"{ch}_current"] = round(v_curr, 4)
            feature_vector[f"{ch}_base_mean"] = round(b_mean, 4)
            feature_vector[f"{ch}_base_std"] = round(b_std, 4)
            feature_vector[f"{ch}_abs_anom"] = abs_anom
            feature_vector[f"{ch}_zscore"] = z_score
            if pct_anom is not None:
                feature_vector[f"{ch}_pct_anom"] = pct_anom

            unit_str = COPERNICUS_VAR_MAPPING.get(canonical_raw, {}).get("unit", "")
            std_name = COPERNICUS_VAR_MAPPING.get(canonical_raw, {}).get("standard_name", ch)

            variables_result[ch] = VariableAnomaly(
                variable=ch,
                standard_name=std_name,
                unit=unit_str,
                current_value=round(v_curr, 4),
                baseline_mean=round(b_mean, 4),
                baseline_std=round(b_std, 4),
                baseline_min=round(b_min, 4),
                baseline_max=round(b_max, 4),
                absolute_anomaly=abs_anom,
                z_score=z_score,
                percentage_anomaly=pct_anom,
                windows=windows_dict,
            )

        avg_valid = float(np.mean(valid_pixel_counts)) if valid_pixel_counts else 100.0

        return HistoricalComparisonResponse(
            dataset_id=COPERNICUS_DATASET_ID,
            date=dataset_times[target_idx].strftime("%Y-%m-%d"),
            mode=mode,
            location=location_info,
            baseline_metadata=baseline_metadata,
            variables=variables_result,
            feature_vector=feature_vector,
            data_quality={
                "valid_data_percent": round(avg_valid, 2),
                "total_baseline_observations": total_obs,
                "window_coverage": "Complete (7, 14, 30 days)",
            },
            provenance="Copernicus Marine Global Ocean Physics Reanalysis (0.083° daily)",
        )

    @classmethod
    def compute_comparison_with_custom_current(
        cls,
        date_str: str,
        custom_values: Dict[str, Any],
        mode: str = "point",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        site_id: Optional[str] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
    ) -> HistoricalComparisonResponse:
        """
        Executes historical comparison for a custom observation:
        1. Uses Copernicus dataset for rolling context (up to target_dt - 1 day or nearest eligible day).
        2. Incorporates custom in-situ/model measurements for current day values:
           - thetao / temp
           - so / sal
           - uo / cur_u
           - vo / cur_v
           - zos / ssh
           - mlotst / mld
           - cur (calculated or provided)
        3. Integrates custom current values into rolling windows (7d, 14d, 30d).
        4. Calculates anomalies and compiles 101-feature vector with provenance="Custom Observation + Copernicus Context".
        """
        ds = CopernicusService.get_dataset()

        coord_time = next((c for c in ["time", "record"] if c in ds.coords or c in ds.dims), None)
        coord_lat = next((c for c in ["latitude", "lat"] if c in ds.coords or c in ds.dims), None)
        coord_lon = next((c for c in ["longitude", "lon"] if c in ds.coords or c in ds.dims), None)
        coord_depth = next((c for c in ["depth", "lev"] if c in ds.coords or c in ds.dims), None)

        if not coord_time or not coord_lat or not coord_lon:
            raise ValueError("Dataset missing spatial/temporal coordinates.")

        dataset_times = pd.to_datetime(ds[coord_time].values)
        total_obs = len(dataset_times)
        if total_obs == 0:
            raise ValueError("Copernicus dataset time coordinate is empty.")

        baseline_start = dataset_times[0].strftime("%Y-%m-%d")
        baseline_end = dataset_times[-1].strftime("%Y-%m-%d")

        baseline_metadata = BaselineMetadata(
            baseline_type="full_period_climatology",
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            baseline_observations=total_obs,
        )

        try:
            target_dt = pd.to_datetime(date_str)
        except Exception:
            raise ValueError(f"Invalid date format '{date_str}'. Please use ISO format YYYY-MM-DD.")

        min_dt = dataset_times[0]
        max_dt = dataset_times[-1]

        # Determine latest Copernicus context observation index
        # If target_dt is beyond max_dt, use max_dt as the latest context day
        if target_dt > max_dt:
            context_idx = total_obs - 1
        elif target_dt <= min_dt + pd.Timedelta(days=29):
            earliest_dt = (min_dt + pd.Timedelta(days=29)).strftime("%Y-%m-%d")
            raise ValueError(
                f"Custom observation date '{date_str}' requires at least 30 days of Copernicus history. "
                f"Earliest eligible comparison date is {earliest_dt}."
            )
        else:
            diffs = (dataset_times - target_dt).total_seconds()
            # Context index is the day immediately prior to target_dt if exact match, or the latest day <= target_dt
            preceding_indices = np.where(dataset_times < target_dt)[0]
            if len(preceding_indices) > 0:
                context_idx = int(preceding_indices[-1])
            else:
                context_idx = int(np.argmin(np.abs(diffs)))

        if context_idx < 29:
            earliest_dt = dataset_times[29].strftime("%Y-%m-%d")
            raise ValueError(
                f"Context day (index {context_idx}) requires at least 30 days of history for rolling features. "
                f"Earliest eligible date is {earliest_dt}."
            )

        # Resolve spatial location & bounds
        location_info: Dict[str, Any] = {}
        bbox: Optional[BoundingBox] = None

        if mode == "point":
            if lat is None or lon is None:
                raise ValueError("Point mode requires 'lat' and 'lon' query parameters.")
            if not (DEFAULT_MIN_LAT - 1.0 <= lat <= DEFAULT_MAX_LAT + 1.0) or not (DEFAULT_MIN_LON - 1.0 <= lon <= DEFAULT_MAX_LON + 1.0):
                raise ValueError(
                    f"Coordinates ({lat}°, {lon}°) are outside regional bounds: "
                    f"Lat [{DEFAULT_MIN_LAT}, {DEFAULT_MAX_LAT}], Lon [{DEFAULT_MIN_LON}, {DEFAULT_MAX_LON}]."
                )
            location_info = {"type": "point", "lat": round(lat, 4), "lon": round(lon, 4)}
        elif mode == "region":
            if site_id:
                site = SiteRegistry.get_site(site_id)
                if not site:
                    raise ValueError(f"Study site '{site_id}' not found.")
                if site.bbox:
                    bbox = site.bbox
                else:
                    bbox = BoundingBox(
                        min_lat=site.lat - 1.5,
                        max_lat=site.lat + 1.5,
                        min_lon=site.lon - 1.5,
                        max_lon=site.lon + 1.5,
                    )
                location_info = {"type": "study_site", "site_id": site.id, "site_name": site.name, "bbox": bbox.model_dump()}
            else:
                if None in (min_lat, max_lat, min_lon, max_lon):
                    raise ValueError("Region mode requires either 'site_id' or 'min_lat', 'max_lat', 'min_lon', 'max_lon'.")
                if min_lat >= max_lat or min_lon >= max_lon:
                    raise ValueError("Invalid bounding box: min coordinates must be strictly less than max coordinates.")
                bbox = BoundingBox(
                    min_lat=float(min_lat),
                    max_lat=float(max_lat),
                    min_lon=float(min_lon),
                    max_lon=float(max_lon),
                )
                location_info = {"type": "bounding_box", "bbox": bbox.model_dump()}
        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes are 'point' or 'region'.")

        # Map custom input values to channels
        val_map: Dict[str, float] = {}
        for k, v in custom_values.items():
            if v is not None:
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        val_map[str(k).lower().strip()] = fv
                except (ValueError, TypeError):
                    pass

        custom_channel_map = {
            "temp": val_map.get("thetao", val_map.get("temp", val_map.get("temperature"))),
            "sal": val_map.get("so", val_map.get("sal", val_map.get("salinity"))),
            "cur_u": val_map.get("uo", val_map.get("cur_u", val_map.get("u"))),
            "cur_v": val_map.get("vo", val_map.get("cur_v", val_map.get("v"))),
            "ssh": val_map.get("zos", val_map.get("ssh", val_map.get("sea_surface_height"))),
            "mld": val_map.get("mlotst", val_map.get("mld", val_map.get("mixed_layer_depth"))),
        }
        cur_val = val_map.get("cur", val_map.get("cur_speed", val_map.get("current_speed")))
        if cur_val is None:
            uo_val = custom_channel_map["cur_u"]
            vo_val = custom_channel_map["cur_v"]
            if uo_val is not None and vo_val is not None:
                cur_val = float(np.sqrt(uo_val**2 + vo_val**2))
        custom_channel_map["cur"] = cur_val

        variables_result: Dict[str, VariableAnomaly] = {}
        feature_vector: Dict[str, float] = {}
        valid_pixel_counts = []

        window_specs = [
            ("7d", 7),
            ("14d", 14),
            ("30d", 30),
        ]

        for ch in SUPPORTED_CHANNELS:
            canonical_raw = ALIAS_MAP.get(ch, ch)
            series_1d = cls._extract_1d_series(
                ds=ds,
                canonical_var=canonical_raw,
                coord_time=coord_time,
                coord_lat=coord_lat,
                coord_lon=coord_lon,
                coord_depth=coord_depth,
                mode=mode,
                lat=lat,
                lon=lon,
                bbox=bbox,
            )

            valid_mask = ~np.isnan(series_1d) & ~np.isinf(series_1d)
            valid_pct = float(np.mean(valid_mask) * 100.0)
            valid_pixel_counts.append(valid_pct)

            valid_full = series_1d[valid_mask]
            if len(valid_full) == 0:
                raise ValueError(f"All values for variable '{ch}' in the selected region are NaN/masked.")

            b_mean = float(np.mean(valid_full))
            b_std = float(np.std(valid_full))
            b_min = float(np.min(valid_full))
            b_max = float(np.max(valid_full))

            # Current value: user custom value if provided, else context day Copernicus value
            if custom_channel_map.get(ch) is not None:
                v_curr = float(custom_channel_map[ch])
            else:
                v_curr = float(series_1d[context_idx])
                if np.isnan(v_curr) or np.isinf(v_curr):
                    v_curr = b_mean

            abs_anom = round(v_curr - b_mean, 4)
            z_score = round((v_curr - b_mean) / b_std, 4) if b_std > 1e-6 else 0.0

            pct_anom: Optional[float] = None
            if abs(b_mean) > 1e-3 and ch in ("mld", "sal", "cur"):
                pct_anom = round((abs_anom / abs(b_mean)) * 100.0, 2)

            windows_dict: Dict[str, WindowMetrics] = {}
            for w_key, w_days in window_specs:
                # Preceding (w_days - 1) days from Copernicus history up to context_idx + custom current observation
                cop_sub = series_1d[context_idx - (w_days - 2) : context_idx + 1]
                w_sub = np.append(cop_sub, [v_curr])

                w_start_str = dataset_times[context_idx - (w_days - 2)].strftime("%Y-%m-%d")
                w_end_str = date_str

                w_valid = w_sub[~np.isnan(w_sub) & ~np.isinf(w_sub)]
                w_mean = float(np.mean(w_valid)) if len(w_valid) > 0 else v_curr
                w_min = float(np.min(w_valid)) if len(w_valid) > 0 else v_curr
                w_max = float(np.max(w_valid)) if len(w_valid) > 0 else v_curr

                w_delta = round(float(w_sub[-1] - w_sub[0]), 4)
                w_slope = cls._compute_linear_trend(w_sub)

                windows_dict[w_key] = WindowMetrics(
                    window_days=w_days,
                    window_start=w_start_str,
                    window_end=w_end_str,
                    mean=round(w_mean, 4),
                    delta=w_delta,
                    trend_per_day=w_slope,
                    min=round(w_min, 4),
                    max=round(w_max, 4),
                )

                feature_vector[f"{ch}_{w_key}_mean"] = round(w_mean, 4)
                feature_vector[f"{ch}_{w_key}_delta"] = w_delta
                feature_vector[f"{ch}_{w_key}_trend"] = w_slope

            feature_vector[f"{ch}_current"] = round(v_curr, 4)
            feature_vector[f"{ch}_base_mean"] = round(b_mean, 4)
            feature_vector[f"{ch}_base_std"] = round(b_std, 4)
            feature_vector[f"{ch}_abs_anom"] = abs_anom
            feature_vector[f"{ch}_zscore"] = z_score
            if pct_anom is not None:
                feature_vector[f"{ch}_pct_anom"] = pct_anom

            unit_str = COPERNICUS_VAR_MAPPING.get(canonical_raw, {}).get("unit", "")
            std_name = COPERNICUS_VAR_MAPPING.get(canonical_raw, {}).get("standard_name", ch)

            variables_result[ch] = VariableAnomaly(
                variable=ch,
                standard_name=std_name,
                unit=unit_str,
                current_value=round(v_curr, 4),
                baseline_mean=round(b_mean, 4),
                baseline_std=round(b_std, 4),
                baseline_min=round(b_min, 4),
                baseline_max=round(b_max, 4),
                absolute_anomaly=abs_anom,
                z_score=z_score,
                percentage_anomaly=pct_anom,
                windows=windows_dict,
            )

        avg_valid = float(np.mean(valid_pixel_counts)) if valid_pixel_counts else 100.0

        return HistoricalComparisonResponse(
            dataset_id=COPERNICUS_DATASET_ID,
            date=date_str,
            mode=mode,
            location=location_info,
            baseline_metadata=baseline_metadata,
            variables=variables_result,
            feature_vector=feature_vector,
            data_quality={
                "valid_data_percent": round(avg_valid, 2),
                "total_baseline_observations": total_obs,
                "window_coverage": "Complete (7, 14, 30 days) with Custom In-situ Current",
            },
            provenance="Custom Observation + Copernicus Context",
        )

    # ==========================================================================
    # ML DATASET EXPORTER (Parquet + Metadata JSON)
    # ==========================================================================
    @classmethod
    def export_training_dataset(
        cls,
        output_dir: Optional[str] = None,
        mode: str = "point",
        lat: float = 17.8,
        lon: float = 88.2,
        site_id: Optional[str] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
        nc_path: Optional[str] = None,
    ) -> TrainingDatasetExportResponse:
        """
        Rolls across all eligible days (t >= 29) of the available dataset,
        computes the unified feature vector for every day, and exports:
        1. features.parquet: Columnar feature table
        2. features_metadata.json: Detailed provenance, baseline metadata, and feature catalogue.
        Leaves raw Copernicus NetCDF completely untouched.
        """
        if nc_path and os.path.exists(nc_path):
            ds = xr.open_dataset(nc_path, engine="netcdf4")
        else:
            ds = CopernicusService.get_dataset()
        coord_time = next((c for c in ["time", "record"] if c in ds.coords or c in ds.dims), None)
        coord_lat = next((c for c in ["latitude", "lat"] if c in ds.coords or c in ds.dims), None)
        coord_lon = next((c for c in ["longitude", "lon"] if c in ds.coords or c in ds.dims), None)
        coord_depth = next((c for c in ["depth", "lev"] if c in ds.coords or c in ds.dims), None)

        dataset_times = pd.to_datetime(ds[coord_time].values)
        total_obs = len(dataset_times)

        if total_obs < 30:
            raise ValueError(f"Dataset has {total_obs} observations; at least 30 observations required.")

        baseline_metadata = BaselineMetadata(
            baseline_type="full_period_climatology",
            baseline_start=dataset_times[0].strftime("%Y-%m-%d"),
            baseline_end=dataset_times[-1].strftime("%Y-%m-%d"),
            baseline_observations=total_obs,
        )

        bbox = None
        if mode == "region":
            if site_id:
                site = SiteRegistry.get_site(site_id)
                if site and site.bbox:
                    bbox = site.bbox
            elif None not in (min_lat, max_lat, min_lon, max_lon):
                bbox = BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

        # Pre-extract all 7 channels 1D time-series
        series_dict: Dict[str, np.ndarray] = {}
        for ch in SUPPORTED_CHANNELS:
            canonical_raw = ALIAS_MAP.get(ch, ch)
            s = cls._extract_1d_series(
                ds=ds,
                canonical_var=canonical_raw,
                coord_time=coord_time,
                coord_lat=coord_lat,
                coord_lon=coord_lon,
                coord_depth=coord_depth,
                mode=mode,
                lat=lat,
                lon=lon,
                bbox=bbox,
            )
            series_dict[ch] = s

        # Compute full-period baseline statistics
        baseline_stats: Dict[str, Dict[str, float]] = {}
        for ch, s in series_dict.items():
            valid = s[~np.isnan(s) & ~np.isinf(s)]
            b_mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            b_std = float(np.std(valid)) if len(valid) > 0 else 1.0
            baseline_stats[ch] = {
                "mean": b_mean,
                "std": b_std if b_std > 1e-6 else 1.0,
                "min": float(np.min(valid)) if len(valid) > 0 else 0.0,
                "max": float(np.max(valid)) if len(valid) > 0 else 0.0,
            }

        rows: List[Dict[str, Any]] = []

        # Iterate over all eligible days: index 29 through total_obs - 1
        for idx in range(29, total_obs):
            row: Dict[str, Any] = {
                "date": dataset_times[idx].strftime("%Y-%m-%d"),
                "day_index": idx,
                "mode": mode,
            }
            if mode == "point":
                row["lat"] = lat
                row["lon"] = lon
            else:
                row["site_id"] = site_id or "custom_region"

            for ch in SUPPORTED_CHANNELS:
                s = series_dict[ch]
                bst = baseline_stats[ch]
                v_curr = float(s[idx])
                if np.isnan(v_curr) or np.isinf(v_curr):
                    v_curr = bst["mean"]

                abs_anom = round(v_curr - bst["mean"], 4)
                z_score = round((v_curr - bst["mean"]) / bst["std"], 4)

                row[f"{ch}_current"] = round(v_curr, 4)
                row[f"{ch}_base_mean"] = round(bst["mean"], 4)
                row[f"{ch}_base_std"] = round(bst["std"], 4)
                row[f"{ch}_abs_anom"] = abs_anom
                row[f"{ch}_zscore"] = z_score

                if abs(bst["mean"]) > 1e-3 and ch in ("mld", "sal", "cur"):
                    row[f"{ch}_pct_anom"] = round((abs_anom / abs(bst["mean"])) * 100.0, 2)

                for w_key, w_days in [("7d", 7), ("14d", 14), ("30d", 30)]:
                    w_sub = s[idx - w_days + 1 : idx + 1]
                    w_valid = w_sub[~np.isnan(w_sub) & ~np.isinf(w_sub)]
                    w_m = float(np.mean(w_valid)) if len(w_valid) > 0 else v_curr
                    w_delta = round(float(w_sub[-1] - w_sub[0]), 4)
                    w_slope = cls._compute_linear_trend(w_sub)

                    row[f"{ch}_{w_key}_mean"] = round(w_m, 4)
                    row[f"{ch}_{w_key}_delta"] = w_delta
                    row[f"{ch}_{w_key}_trend"] = w_slope

            rows.append(row)

        df = pd.DataFrame(rows)

        # Target export directory
        if not output_dir:
            output_dir = os.path.join(CopernicusService.get_data_dir(), "ml_features")
        os.makedirs(output_dir, exist_ok=True)

        parquet_path = os.path.join(output_dir, "features.parquet")
        metadata_path = os.path.join(output_dir, "features_metadata.json")

        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        # Save companion JSON metadata
        meta_payload = {
            "dataset_id": COPERNICUS_DATASET_ID,
            "export_type": "rolling_features_table",
            "format": "parquet",
            "total_rows": len(df),
            "total_features": len(df.columns),
            "feature_columns": list(df.columns),
            "date_range": {
                "start": df["date"].iloc[0] if len(df) > 0 else "",
                "end": df["date"].iloc[-1] if len(df) > 0 else "",
            },
            "baseline_metadata": baseline_metadata.model_dump(),
            "target_channels": SUPPORTED_CHANNELS,
            "rolling_windows": ["7d", "14d", "30d"],
            "exported_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        return TrainingDatasetExportResponse(
            status="success",
            parquet_path=os.path.abspath(parquet_path),
            metadata_path=os.path.abspath(metadata_path),
            total_rows=len(df),
            total_features=len(df.columns),
            date_range={"start": df["date"].iloc[0], "end": df["date"].iloc[-1]},
            baseline_metadata=baseline_metadata,
            message=f"Exported {len(df)} daily feature rows ({len(df.columns)} columns) to Parquet.",
        )

    # Explicit alias matching user request
    generate_training_dataset = export_training_dataset


def generate_training_dataset(*args, **kwargs) -> TrainingDatasetExportResponse:
    """Module-level convenience wrapper for generating and exporting the ML training dataset."""
    return HistoricalFeatureEngine.export_training_dataset(*args, **kwargs)

