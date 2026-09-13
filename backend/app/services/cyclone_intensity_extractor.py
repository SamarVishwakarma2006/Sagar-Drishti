"""
Sagar-Drishti — Cyclone Intensity 6-Hourly Storm-Centered Feature Extractor
RESEARCH-ONLY / ZERO ML TRAINING

Implements deterministic, strictly causal 6-hourly storm-centered feature extraction:
- 0–100 km Inner Core & 200–800 km Environmental Annulus spatial domains.
- Genuinely 6-hourly ERA5 atmospheric fields at T in {00Z, 06Z, 12Z, 18Z} (shear, vorticity, RH700, RH500).
- Strictly causal daily Copernicus ocean fields with ocean_source_timestamp <= T,
  explicit ocean_age_hours tracking, and 48-hour maximum age policy.
- Causal kinematic features and continuous 24h intensity targets (Vmax, Pc) with +/- 3h tolerance.
- Event-grouped chronological split assignment (64 Train, 24 Validation, 29 Test).
"""

import os
import re
import glob
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import xarray as xr


# Physical Constants
EARTH_RADIUS_KM = 6371.0088
MS_TO_KNOTS = 1.9438444924406
VORTICITY_SCALE = 1e5
MAX_OCEAN_AGE_HOURS = 48.0

# Partition Boundaries (Final Approved Event-Grouped Chronological Split)
TRAIN_MAX_YEAR = 2021
VAL_MAX_YEAR = 2023


def haversine_distance_matrix(
    lat_center: float,
    lon_center: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray
) -> np.ndarray:
    """
    Computes great-circle distances in km from storm center to all 2D grid coordinates.
    Handles latitude bounds and longitude wrapping across the antimeridian.
    """
    phi1 = np.radians(lat_center)
    phi2 = np.radians(lat_grid)
    dphi = phi2 - phi1

    # Longitude difference wrapped to [-180, 180]
    dlambda_deg = (lon_grid - lon_center + 180.0) % 360.0 - 180.0
    dlambda = np.radians(dlambda_deg)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class CycloneIntensityFeatureExtractor:
    """Deterministic, causal feature extractor for cyclone intensity research."""

    def __init__(
        self,
        era5_dir: str = "backend/data/era5",
        copernicus_path: str = "backend/data/copernicus/copernicus_phy_10yr_surface.nc",
        max_ocean_age_hours: float = MAX_OCEAN_AGE_HOURS
    ):
        self.era5_dir = era5_dir
        self.copernicus_path = copernicus_path
        self.max_ocean_age_hours = max_ocean_age_hours
        self._copernicus_ds: Optional[xr.Dataset] = None
        self._era5_cache: Dict[int, xr.Dataset] = {}

    def close(self) -> None:
        """Closes all cached datasets."""
        if self._copernicus_ds is not None:
            self._copernicus_ds.close()
            self._copernicus_ds = None
        for ds in self._era5_cache.values():
            ds.close()
        self._era5_cache.clear()

    def get_copernicus_dataset(self) -> xr.Dataset:
        """Loads and caches the 10-year Copernicus surface dataset."""
        if self._copernicus_ds is None:
            if not os.path.exists(self.copernicus_path):
                raise FileNotFoundError(f"Copernicus dataset missing: {self.copernicus_path}")
            self._copernicus_ds = xr.open_dataset(self.copernicus_path)
            # Verify required variables
            for v in ["thetao", "mlotst", "zos"]:
                if v not in self._copernicus_ds.data_vars:
                    raise KeyError(f"Required ocean variable {v} missing in Copernicus dataset.")
        return self._copernicus_ds

    def get_era5_dataset(self, year: int) -> xr.Dataset:
        """Loads and caches the yearly ERA5 pressure dataset."""
        if year not in self._era5_cache:
            path = os.path.join(self.era5_dir, f"era5_pressure_{year}.nc")
            if not os.path.exists(path):
                raise FileNotFoundError(f"ERA5 dataset for year {year} missing: {path}")
            ds = xr.open_dataset(path)
            for v in ["vo", "u", "v", "r"]:
                if v not in ds.data_vars:
                    raise KeyError(f"Required ERA5 variable {v} missing in {path}")
            self._era5_cache[year] = ds
        return self._era5_cache[year]

    def extract_atmospheric_features(
        self,
        t_forecast: datetime,
        lat_center: float,
        lon_center: float
    ) -> Dict[str, Any]:
        """
        Extracts 6-hourly storm-centered atmospheric features at exact synoptic hour t_forecast.
        Enforces strict causality: observation timestamp must equal t_forecast (<= t_forecast).
        """
        if hasattr(t_forecast, "tzinfo") and t_forecast.tzinfo is not None:
            t_forecast = t_forecast.astimezone(timezone.utc).replace(tzinfo=None)
        year = t_forecast.year
        ds = self.get_era5_dataset(year)

        # Coordinate arrays
        lats = ds.latitude.values
        lons = ds.longitude.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Great-circle distance
        dist_km = haversine_distance_matrix(lat_center, lon_center, lat_grid, lon_grid)
        mask_core = dist_km <= 100.0
        mask_env = (dist_km >= 200.0) & (dist_km <= 800.0)

        # Select exact 6-hourly time slice
        t_np = np.datetime64(t_forecast)
        if t_np not in ds.valid_time.values:
            raise KeyError(f"Synoptic timestamp {t_forecast.isoformat()} not found in ERA5 {year}.")

        slice_t = ds.sel(valid_time=t_np)

        # 1. Vertical Wind Shear (200 - 850 hPa)
        u200 = slice_t["u"].sel(pressure_level=200.0).values
        u850 = slice_t["u"].sel(pressure_level=850.0).values
        v200 = slice_t["v"].sel(pressure_level=200.0).values
        v850 = slice_t["v"].sel(pressure_level=850.0).values
        vws_ms = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        vws_kts = vws_ms * MS_TO_KNOTS

        # 2. Relative Vorticity (850 hPa)
        vo850 = slice_t["vo"].sel(pressure_level=850.0).values * VORTICITY_SCALE

        # 3. Relative Humidity (700 & 500 hPa)
        rh700 = slice_t["r"].sel(pressure_level=700.0).values
        rh500 = slice_t["r"].sel(pressure_level=500.0).values

        features = {
            "atmos_source_timestamp": t_forecast.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "atmos_temporal_resolution": "6-hourly",
            # Vertical Wind Shear (knots)
            "vws_env_mean_200_800km": float(np.nanmean(vws_kts[mask_env])) if mask_env.any() else np.nan,
            "vws_env_min_200_800km": float(np.nanmin(vws_kts[mask_env])) if mask_env.any() else np.nan,
            "vws_core_mean_0_100km": float(np.nanmean(vws_kts[mask_core])) if mask_core.any() else np.nan,
            # 850 hPa Relative Vorticity (10^-5 s^-1)
            "vort_core_mean_0_100km": float(np.nanmean(vo850[mask_core])) if mask_core.any() else np.nan,
            "vort_core_max_0_100km": float(np.nanmax(vo850[mask_core])) if mask_core.any() else np.nan,
            "vort_env_mean_200_800km": float(np.nanmean(vo850[mask_env])) if mask_env.any() else np.nan,
            # Relative Humidity (%)
            "rh_700_env_mean_200_800km": float(np.nanmean(rh700[mask_env])) if mask_env.any() else np.nan,
            "rh_700_env_min_200_800km": float(np.nanmin(rh700[mask_env])) if mask_env.any() else np.nan,
            "rh_700_core_mean_0_100km": float(np.nanmean(rh700[mask_core])) if mask_core.any() else np.nan,
            "rh_500_env_mean_200_800km": float(np.nanmean(rh500[mask_env])) if mask_env.any() else np.nan,
            "rh_500_core_mean_0_100km": float(np.nanmean(rh500[mask_core])) if mask_core.any() else np.nan,
        }
        return features

    def extract_ocean_features(
        self,
        t_forecast: datetime,
        lat_center: float,
        lon_center: float
    ) -> Dict[str, Any]:
        """
        Extracts strictly causal daily ocean features from Copernicus.
        Enforces: ocean_source_timestamp <= t_forecast, 0 <= ocean_age_hours <= max_ocean_age_hours.
        Never synthesizes or interpolates daily ocean data into 6-hourly values.
        """
        if hasattr(t_forecast, "tzinfo") and t_forecast.tzinfo is not None:
            t_forecast = t_forecast.astimezone(timezone.utc).replace(tzinfo=None)
        ds = self.get_copernicus_dataset()

        # Copernicus native time is daily at 00:00:00 UTC
        # Causal selection: find latest daily time <= t_forecast
        ocean_times = pd.to_datetime(ds.time.values)
        causal_times = ocean_times[ocean_times <= t_forecast]

        if len(causal_times) == 0:
            # Observation predates ocean dataset (e.g. ROANU May 2016)
            return {
                "ocean_source_timestamp": None,
                "ocean_age_hours": np.nan,
                "ocean_temporal_resolution": "daily",
                "sst_core_mean_0_100km": np.nan,
                "sst_env_mean_200_800km": np.nan,
                "mld_core_mean_0_100km": np.nan,
                "sla_core_mean_0_100km": np.nan,
                "ocean_land_fraction_core": 1.0,
                "ocean_data_status": "PRE_COPERNICUS_NO_SOURCE"
            }

        latest_t = causal_times[-1]
        age_hours = (t_forecast - latest_t.to_pydatetime()).total_seconds() / 3600.0

        if age_hours > self.max_ocean_age_hours:
            return {
                "ocean_source_timestamp": latest_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ocean_age_hours": float(age_hours),
                "ocean_temporal_resolution": "daily",
                "sst_core_mean_0_100km": np.nan,
                "sst_env_mean_200_800km": np.nan,
                "mld_core_mean_0_100km": np.nan,
                "sla_core_mean_0_100km": np.nan,
                "ocean_land_fraction_core": np.nan,
                "ocean_data_status": "ACTUAL_SOURCE_AGE_GT_48H"
            }

        # Select ocean time slice
        slice_t = ds.sel(time=latest_t)

        # Coordinate arrays (Copernicus 0.0833 deg)
        lats = ds.latitude.values
        lons = ds.longitude.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Great-circle distance
        dist_km = haversine_distance_matrix(lat_center, lon_center, lat_grid, lon_grid)
        mask_core = dist_km <= 100.0
        mask_env = (dist_km >= 200.0) & (dist_km <= 800.0)

        # Extract variables at depth index 0 (0.494 m surface layer)
        thetao = slice_t["thetao"].isel(depth=0).values  # SST (deg C)
        mlotst = slice_t["mlotst"].values                # MLD (m)
        zos = slice_t["zos"].values                      # SLA (m)

        # Land mask calculation in core (NaN indicates land in Copernicus)
        core_thetao = thetao[mask_core]
        valid_ocean_count = int(np.count_nonzero(~np.isnan(core_thetao)))
        total_core_count = int(len(core_thetao))
        land_fraction = 1.0 - (valid_ocean_count / total_core_count) if total_core_count > 0 else 1.0

        status = "OVERLAND_NO_MARINE_PIXELS" if valid_ocean_count == 0 else "VALID_CAUSAL_DAILY"

        features = {
            "ocean_source_timestamp": latest_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ocean_age_hours": float(age_hours),
            "ocean_temporal_resolution": "daily",
            "sst_core_mean_0_100km": float(np.nanmean(thetao[mask_core])) if valid_ocean_count > 0 else np.nan,
            "sst_env_mean_200_800km": float(np.nanmean(thetao[mask_env])) if np.count_nonzero(~np.isnan(thetao[mask_env])) > 0 else np.nan,
            "mld_core_mean_0_100km": float(np.nanmean(mlotst[mask_core])) if valid_ocean_count > 0 else np.nan,
            "sla_core_mean_0_100km": float(np.nanmean(zos[mask_core])) if valid_ocean_count > 0 else np.nan,
            "ocean_land_fraction_core": float(land_fraction),
            "ocean_data_status": status
        }
        return features

    @staticmethod
    def extract_kinematic_features(
        storm_history: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, Any]:
        """
        Extracts causal kinematic features from historical Best Track fixes strictly at or before T.
        """
        row_t = storm_history.iloc[current_idx]
        t0 = row_t["datetime"]
        v0 = float(row_t["max_wind_kts"])
        p0 = float(row_t["central_pressure_hpa"])

        # Look back up to current_idx
        history_before = storm_history.iloc[:current_idx]

        # Prior Deltas
        def get_delta_v(hours_back: float) -> float:
            target_t = t0 - timedelta(hours=hours_back)
            # Find fix closest to target_t within +/- 1.5h
            matches = history_before[
                (history_before["datetime"] >= target_t - timedelta(hours=1.5)) &
                (history_before["datetime"] <= target_t + timedelta(hours=1.5))
            ]
            if len(matches) > 0:
                v_prev = float(matches.iloc[-1]["max_wind_kts"])
                return v0 - v_prev
            return 0.0

        def get_delta_p(hours_back: float) -> float:
            target_t = t0 - timedelta(hours=hours_back)
            matches = history_before[
                (history_before["datetime"] >= target_t - timedelta(hours=1.5)) &
                (history_before["datetime"] <= target_t + timedelta(hours=1.5))
            ]
            if len(matches) > 0:
                p_prev = float(matches.iloc[-1]["central_pressure_hpa"])
                return p0 - p_prev
            return 0.0

        dv_6h = get_delta_v(6.0)
        dv_12h = get_delta_v(12.0) if len(history_before) >= 2 else (dv_6h * 2.0)
        dv_24h = get_delta_v(24.0) if len(history_before) >= 4 else (dv_6h * 4.0)
        dp_6h = get_delta_p(6.0)

        # Translation speed from previous 6h fix
        prev_fix = history_before.iloc[-1] if len(history_before) > 0 else None
        if prev_fix is not None:
            dt_h = (t0 - prev_fix["datetime"]).total_seconds() / 3600.0
            if dt_h > 0:
                dist_km = haversine_distance_matrix(
                    row_t["latitude"], row_t["longitude"],
                    np.array([prev_fix["latitude"]]), np.array([prev_fix["longitude"]])
                )[0]
                dist_nm = dist_km / 1.852
                trans_speed_kts = float(dist_nm / dt_h)
            else:
                trans_speed_kts = 0.0
        else:
            trans_speed_kts = 0.0

        return {
            "vmax_current": v0,
            "dvmax_6h": float(dv_6h),
            "dvmax_12h": float(dv_12h),
            "dvmax_24h": float(dv_24h),
            "pc_current": p0,
            "dpc_6h": float(dp_6h),
            "translation_speed_kts": float(trans_speed_kts),
            "latitude_current": float(row_t["latitude"]),
            "longitude_current": float(row_t["longitude"]),
            "basin_id": "ARAS" if float(row_t["longitude"]) < 80.0 else "BOB"
        }

    @staticmethod
    def match_intensity_targets(
        storm_future: pd.DataFrame,
        t_forecast: datetime
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Matches primary continuous target Vmax(T+24h) and secondary Pc(T+24h).
        Tolerance window: [T + 21h, T + 27h].
        Selection rule: Closest synoptic fix to exactly T + 24.0h.
        """
        t_target_ideal = t_forecast + timedelta(hours=24.0)
        window_start = t_forecast + timedelta(hours=21.0)
        window_end = t_forecast + timedelta(hours=27.0)

        candidates = storm_future[
            (storm_future["datetime"] >= window_start) &
            (storm_future["datetime"] <= window_end)
        ]

        if len(candidates) == 0:
            return None, "NO_VALID_24H_FUTURE_FIX_TERMINAL_DISSIPATION"

        # Selection rule: Pick candidate with minimum absolute time delta to T+24.0h
        candidates = candidates.copy()
        candidates["diff_h"] = (candidates["datetime"] - t_target_ideal).abs().dt.total_seconds() / 3600.0
        best_fix = candidates.sort_values("diff_h").iloc[0]

        target_info = {
            "target_timestamp": best_fix["datetime"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_time_diff_hours": float(best_fix["diff_h"]),
            "target_vmax_24h": float(best_fix["max_wind_kts"]),
            "target_pc_24h": float(best_fix["central_pressure_hpa"]),
            "target_grade_24h": str(best_fix.get("grade", "UNKNOWN"))
        }
        return target_info, None

    @staticmethod
    def assign_partition(system_id: str) -> str:
        """
        Assigns storm partition based strictly on approved event-grouped calendar years.
        TRAIN: <= 2021 (64 storms)
        VAL: 2022-2023 (24 storms)
        TEST: 2024-2026 (29 storms, quarantined)
        """
        m = re.search(r"IMD-(\d{4})", system_id)
        if not m:
            raise ValueError(f"Invalid system_id format: {system_id}")
        year = int(m.group(1))
        if year <= TRAIN_MAX_YEAR:
            return "TRAIN"
        elif year <= VAL_MAX_YEAR:
            return "VALIDATION"
        else:
            return "TEST"
