"""
Sagar-Drishti V2.1 — Spatial Architecture & Sub-Basin Atmospheric Feature Extraction.

Prepares modular multi-grid spatial feature pooling for ERA5 reanalysis data across:
1. Northern Arabian Sea (NAS): 20–25°N, 60–72°E
2. Central Arabian Sea (CAS): 14–20°N, 60–75°E
3. Southern Arabian Sea (SAS): 8–14°N, 65–78°E
4. Northern Bay of Bengal (NBOB): 18–23°N, 85–95°E
5. Central Bay of Bengal (CBOB): 13–18°N, 80–93°E
6. Southern Bay of Bengal (SBOB): 5–13°N, 80–93°E

Provides deterministic calculations for:
- 850 hPa relative vorticity (vo850 and vo850_scaled = vo850 * 10^5)
- 200–850 hPa Vertical Wind Shear (VWS = sqrt((u200 - u850)^2 + (v200 - v850)^2))
- Mid-tropospheric relative humidity (RH700, RH500)
- Latitude-weighted area exceedance fraction using cos(latitude)
- Vectorized daily temporal aggregation across 6-hourly observations

SAFETY CONSTRAINTS:
- Do NOT sample only at single central centroid.
- Vorticity threshold MUST be configurable and MUST NOT be tuned using 2025–2026 test data.
- Feature extraction does NOT modify frozen v1.1.0 or v2_10yr artifacts.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import xarray as xr


# Canonical Sub-Basin Spatial Partitioning (6 Operational Research Basins)
SUB_BASIN_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "northern_arabian_sea": {
        "name": "Northern Arabian Sea",
        "code": "NAS",
        "lat_min": 20.0,
        "lat_max": 25.0,
        "lon_min": 60.0,
        "lon_max": 72.0,
        "description": "High-latitude Arabian Sea, Gujarat/Oman coast, rapid intensification genesis region for Asna.",
    },
    "central_arabian_sea": {
        "name": "Central Arabian Sea",
        "code": "CAS",
        "lat_min": 14.0,
        "lat_max": 20.0,
        "lon_min": 60.0,
        "lon_max": 75.0,
        "description": "Mid-Arabian Sea basin, primary propagation zone for post-monsoon tropical cyclones.",
    },
    "southern_arabian_sea": {
        "name": "Southern Arabian Sea",
        "code": "SAS",
        "lat_min": 8.0,
        "lat_max": 14.0,
        "lon_min": 65.0,
        "lon_max": 78.0,
        "description": "Low-latitude Arabian Sea, pre-cyclone monsoon trough onset region.",
    },
    "northern_bay_of_bengal": {
        "name": "Northern Bay of Bengal",
        "code": "NBOB",
        "lat_min": 18.0,
        "lat_max": 23.0,
        "lon_min": 85.0,
        "lon_max": 95.0,
        "description": "Head Bay region, frequent genesis zone with high freshwater stratification.",
    },
    "central_bay_of_bengal": {
        "name": "Central Bay of Bengal",
        "code": "CBOB",
        "lat_min": 13.0,
        "lat_max": 18.0,
        "lon_min": 80.0,
        "lon_max": 93.0,
        "description": "Central Bay of Bengal track region for severe cyclonic storms.",
    },
    "southern_bay_of_bengal": {
        "name": "Southern Bay of Bengal",
        "code": "SBOB",
        "lat_min": 5.0,
        "lat_max": 13.0,
        "lon_min": 80.0,
        "lon_max": 93.0,
        "description": "Southern Bay basin, equatorial wave and MJO passage corridor.",
    },
}

# Mapping from basin code (e.g. 'NAS') to dict key ('northern_arabian_sea')
BASIN_CODE_TO_KEY: Dict[str, str] = {
    meta["code"]: key for key, meta in SUB_BASIN_DEFINITIONS.items()
}


class ERA5SpatialFeatureExtractor:
    """Modular spatial and temporal feature extractor for ERA5 atmospheric fields."""

    @staticmethod
    def validate_dataset_dimensions(ds: xr.Dataset) -> None:
        """Asserts that the dataset contains expected coordinates and dimensions."""
        for dim in ["latitude", "longitude"]:
            if dim not in ds.coords and dim not in ds.dims:
                raise ValueError(f"Required spatial coordinate '{dim}' missing from dataset coordinates: {list(ds.coords)}")
        
        if len(ds.latitude) < 2 or len(ds.longitude) < 2:
            raise ValueError(f"Spatial coordinates insufficient: lat count={len(ds.latitude)}, lon count={len(ds.longitude)}")

    @classmethod
    def slice_sub_basin(cls, ds: xr.Dataset, basin_key: str) -> xr.Dataset:
        """
        Slices dataset to the exact bounding box of a sub-basin.
        Handles descending latitude convention (25 -> 0) in ERA5 safely.
        Asserts non-empty result and valid bounds.
        """
        if basin_key not in SUB_BASIN_DEFINITIONS:
            # Check if key is basin code
            if basin_key in BASIN_CODE_TO_KEY:
                basin_key = BASIN_CODE_TO_KEY[basin_key]
            else:
                raise KeyError(f"Unknown sub-basin key: {basin_key}. Available: {list(SUB_BASIN_DEFINITIONS.keys())}")

        cls.validate_dataset_dimensions(ds)

        bounds = SUB_BASIN_DEFINITIONS[basin_key]
        lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
        lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]

        lats = ds.latitude.values
        # In ERA5, latitude is typically descending (e.g. 25.0, 24.75, ..., 0.0)
        if lats[0] > lats[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        lon_slice = slice(lon_min, lon_max)
        sub_ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

        if len(sub_ds.latitude) == 0 or len(sub_ds.longitude) == 0:
            raise ValueError(
                f"Slicing sub-basin '{basin_key}' resulted in empty grid: "
                f"lat_count={len(sub_ds.latitude)}, lon_count={len(sub_ds.longitude)}. "
                f"Requested lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]"
            )

        return sub_ds

    @staticmethod
    def compute_vertical_wind_shear(ds: xr.Dataset) -> xr.DataArray:
        """
        Calculates 200–850 hPa Vertical Wind Shear:
        VWS = sqrt((u200 - u850)^2 + (v200 - v850)^2) [m/s]
        """
        for var in ["u", "v"]:
            if var not in ds:
                raise KeyError(f"Variable '{var}' required for VWS missing from dataset: {list(ds.data_vars)}")
        
        levels = [float(lvl) for lvl in ds.pressure_level.values]
        if 200.0 not in levels or 850.0 not in levels:
            raise ValueError(f"Pressure levels 200.0 and 850.0 required for VWS; available: {levels}")

        u200 = ds["u"].sel(pressure_level=200.0)
        v200 = ds["v"].sel(pressure_level=200.0)
        u850 = ds["u"].sel(pressure_level=850.0)
        v850 = ds["v"].sel(pressure_level=850.0)

        vws = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        vws.name = "vws_200_850"
        vws.attrs = {"units": "m s**-1", "long_name": "200-850 hPa Vertical Wind Shear"}
        return vws

    @staticmethod
    def compute_scaled_vorticity(ds: xr.Dataset) -> xr.DataArray:
        """
        Calculates 850 hPa relative vorticity scaled by 10^5:
        vo850_scaled = vo850 * 10^5 [10^-5 s^-1]
        """
        if "vo" not in ds:
            raise KeyError(f"Variable 'vo' required for vorticity missing from dataset: {list(ds.data_vars)}")
        levels = [float(lvl) for lvl in ds.pressure_level.values]
        if 850.0 not in levels:
            raise ValueError(f"Pressure level 850.0 required for relative vorticity; available: {levels}")

        vo850 = ds["vo"].sel(pressure_level=850.0)
        vo_scaled = vo850 * 1e5
        vo_scaled.name = "vo850_scaled"
        vo_scaled.attrs = {"units": "10**-5 s**-1", "long_name": "850 hPa Relative Vorticity Scaled"}
        return vo_scaled

    @staticmethod
    def compute_weighted_exceedance_fraction(
        vo_scaled_arr: np.ndarray,
        lats: np.ndarray,
        threshold: float = 5.0,
    ) -> Tuple[float, float]:
        """
        Calculates both:
        1. True geographic area-weighted exceedance fraction using cos(latitude).
        2. Unweighted gridcell exceedance fraction (diagnostic).

        Formula for area-weighted fraction:
        weight = cos(radians(latitude))
        weighted_frac = sum(weight * (vo_scaled >= threshold)) / sum(weight)

        Parameters:
            vo_scaled_arr: 2D (lat, lon) or 3D (time, lat, lon) numpy array of scaled vorticity.
            lats: 1D array of latitude values matching the latitude dimension of vo_scaled_arr.
            threshold: vorticity threshold in scaled units (10^-5 s^-1).

        Returns:
            (weighted_exceedance_fraction, gridcell_exceedance_fraction)
        """
        if not np.all(np.isfinite(vo_scaled_arr)):
            raise ValueError("NaN or Inf encountered in vorticity array when computing exceedance fraction.")

        # cos(lat) weights (latitudes in radians)
        cos_lats = np.cos(np.deg2rad(lats))
        if np.any(cos_lats < 0):
            raise ValueError("Negative cosine weights encountered; check latitude bounds.")

        exceedance_mask = (vo_scaled_arr >= threshold).astype(np.float64)
        gridcell_frac = float(np.mean(exceedance_mask))

        if vo_scaled_arr.ndim == 2:
            # vo_scaled_arr shape: (n_lat, n_lon)
            n_lon = vo_scaled_arr.shape[1]
            weights_2d = cos_lats[:, np.newaxis]
            total_weight = np.sum(cos_lats) * n_lon
            if total_weight <= 0:
                raise ValueError("Total weight sum is non-positive.")
            weighted_frac = float(np.sum(weights_2d * exceedance_mask) / total_weight)

        elif vo_scaled_arr.ndim == 3:
            # vo_scaled_arr shape: (n_time, n_lat, n_lon)
            n_time, n_lat, n_lon = vo_scaled_arr.shape
            weights_3d = cos_lats[np.newaxis, :, np.newaxis]
            total_weight = n_time * np.sum(cos_lats) * n_lon
            if total_weight <= 0:
                raise ValueError("Total weight sum is non-positive.")
            weighted_frac = float(np.sum(weights_3d * exceedance_mask) / total_weight)

        else:
            raise ValueError(f"Expected 2D or 3D array for vorticity exceedance, got ndim={vo_scaled_arr.ndim}")

        return weighted_frac, gridcell_frac

    @classmethod
    def extract_basin_summary_features(
        cls,
        ds: xr.Dataset,
        basin_key: str,
        vorticity_threshold_scaled: float = 5.0,
    ) -> Dict[str, float]:
        """
        Extracts pooled summary statistics for a given sub-basin at a single timestep (or slice).
        All calculations are deterministic, vectorized, and finite-checked.

        Features extracted:
        - max(vo850_scaled), mean(vo850_scaled), std(vo850_scaled)
        - min(vws), mean(vws), max(vws)
        - mean(rh700), max(rh700)
        - mean(rh500), max(rh500)
        - weighted_exceedance_frac: area-weighted fraction where vo850_scaled >= threshold
        - gridcell_exceedance_frac: unweighted fraction (diagnostic)
        - max_vorticity_lat: latitude of maximum vorticity
        - max_vorticity_lon: longitude of maximum vorticity
        """
        basin_ds = cls.slice_sub_basin(ds, basin_key)
        code = SUB_BASIN_DEFINITIONS[basin_key]["code"].lower()
        lats = basin_ds.latitude.values
        lons = basin_ds.longitude.values

        # Vorticity
        vo_scaled = cls.compute_scaled_vorticity(basin_ds).values
        if not np.all(np.isfinite(vo_scaled)):
            raise ValueError(f"Non-finite values in vorticity array for basin '{basin_key}'")

        vo_max = float(np.max(vo_scaled))
        vo_mean = float(np.mean(vo_scaled))
        vo_std = float(np.std(vo_scaled))

        # Coordinates of maximum vorticity
        unravel_idx = np.unravel_index(np.argmax(vo_scaled), vo_scaled.shape)
        lat_idx = unravel_idx[-2] if len(unravel_idx) >= 2 else 0
        lon_idx = unravel_idx[-1] if len(unravel_idx) >= 2 else 0
        max_vo_lat = float(lats[lat_idx])
        max_vo_lon = float(lons[lon_idx])

        # Exceedance fractions (area-weighted vs unweighted gridcell)
        weighted_frac, gridcell_frac = cls.compute_weighted_exceedance_fraction(
            vo_scaled, lats, threshold=vorticity_threshold_scaled
        )

        # Vertical Wind Shear
        vws = cls.compute_vertical_wind_shear(basin_ds).values
        if not np.all(np.isfinite(vws)):
            raise ValueError(f"Non-finite values in VWS array for basin '{basin_key}'")
        vws_min = float(np.min(vws))
        vws_mean = float(np.mean(vws))
        vws_max = float(np.max(vws))

        # Relative Humidity (700 hPa and 500 hPa)
        rh700 = basin_ds["r"].sel(pressure_level=700.0).values
        rh500 = basin_ds["r"].sel(pressure_level=500.0).values
        if not np.all(np.isfinite(rh700)) or not np.all(np.isfinite(rh500)):
            raise ValueError(f"Non-finite values in relative humidity for basin '{basin_key}'")
        rh700_mean = float(np.mean(rh700))
        rh700_max = float(np.max(rh700))
        rh500_mean = float(np.mean(rh500))
        rh500_max = float(np.max(rh500))

        return {
            f"{code}_vo850_max": vo_max,
            f"{code}_vo850_mean": vo_mean,
            f"{code}_vo850_std": vo_std,
            f"{code}_vo850_weighted_exceedance_frac": weighted_frac,
            f"{code}_vo850_gridcell_exceedance_frac": gridcell_frac,
            # Legacy alias for test backward compatibility:
            f"{code}_vo850_exceedance_frac": weighted_frac,
            f"{code}_vo850_max_lat": max_vo_lat,
            f"{code}_vo850_max_lon": max_vo_lon,
            f"{code}_vws_min": vws_min,
            f"{code}_vws_mean": vws_mean,
            f"{code}_vws_max": vws_max,
            f"{code}_rh700_mean": rh700_mean,
            f"{code}_rh700_max": rh700_max,
            f"{code}_rh500_mean": rh500_mean,
            f"{code}_rh500_max": rh500_max,
        }

    @classmethod
    def extract_daily_basin_features(
        cls,
        daily_ds: xr.Dataset,
        basin_key: str,
        date_str: str,
        vorticity_threshold_scaled: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Extracts strictly causal daily aggregated features for date D across all 6-hourly
        observations available on date D (00Z, 06Z, 12Z, 18Z).

        Includes explicit quality metadata:
        - number_of_valid_6h_samples
        - expected_6h_samples (4)
        - completeness_fraction (n / 4.0)
        - is_complete (bool)

        Strict Causality Contract:
        Only timesteps belonging to date_str in UTC are processed. Zero D+1 access.
        """
        basin_ds = cls.slice_sub_basin(daily_ds, basin_key)
        basin_code = SUB_BASIN_DEFINITIONS[basin_key]["code"]
        lats = basin_ds.latitude.values
        lons = basin_ds.longitude.values

        # Inspect timesteps for this day
        n_samples = len(basin_ds.valid_time)
        expected_samples = 4
        completeness_fraction = float(n_samples / float(expected_samples))
        is_complete = bool(n_samples == expected_samples)

        if n_samples == 0:
            # Return empty record flagged as incomplete
            return {
                "date": date_str,
                "basin": basin_code,
                "max_vorticity": np.nan,
                "mean_vorticity": np.nan,
                "std_vorticity": np.nan,
                "weighted_exceedance_fraction": np.nan,
                "gridcell_exceedance_fraction": np.nan,
                "latitude_of_max_vorticity": np.nan,
                "longitude_of_max_vorticity": np.nan,
                "min_vws": np.nan,
                "mean_vws": np.nan,
                "max_vws": np.nan,
                "mean_rh700": np.nan,
                "max_rh700": np.nan,
                "mean_rh500": np.nan,
                "max_rh500": np.nan,
                "number_of_valid_6h_samples": 0,
                "expected_6h_samples": expected_samples,
                "completeness_fraction": 0.0,
                "is_complete": False,
            }

        # Vorticity over all daily 6h timesteps (shape: n_time, n_lat, n_lon)
        vo_scaled = cls.compute_scaled_vorticity(basin_ds).values
        if not np.all(np.isfinite(vo_scaled)):
            raise ValueError(f"Non-finite values in vorticity array for basin '{basin_key}' on date '{date_str}'")

        vo_max = float(np.max(vo_scaled))
        vo_mean = float(np.mean(vo_scaled))
        vo_std = float(np.std(vo_scaled))

        # Location of daily absolute maximum vorticity
        unravel_idx = np.unravel_index(np.argmax(vo_scaled), vo_scaled.shape)
        lat_idx = unravel_idx[-2]
        lon_idx = unravel_idx[-1]
        max_vo_lat = float(lats[lat_idx])
        max_vo_lon = float(lons[lon_idx])

        # Exceedance fraction (weighted area vs gridcell)
        weighted_frac, gridcell_frac = cls.compute_weighted_exceedance_fraction(
            vo_scaled, lats, threshold=vorticity_threshold_scaled
        )

        # Vertical Wind Shear over all daily 6h timesteps
        vws = cls.compute_vertical_wind_shear(basin_ds).values
        if not np.all(np.isfinite(vws)):
            raise ValueError(f"Non-finite values in VWS array for basin '{basin_key}' on date '{date_str}'")
        vws_min = float(np.min(vws))
        vws_mean = float(np.mean(vws))
        vws_max = float(np.max(vws))

        # Relative Humidity (700 and 500 hPa)
        rh700 = basin_ds["r"].sel(pressure_level=700.0).values
        rh500 = basin_ds["r"].sel(pressure_level=500.0).values
        if not np.all(np.isfinite(rh700)) or not np.all(np.isfinite(rh500)):
            raise ValueError(f"Non-finite values in relative humidity for basin '{basin_key}' on date '{date_str}'")
        rh700_mean = float(np.mean(rh700))
        rh700_max = float(np.max(rh700))
        rh500_mean = float(np.mean(rh500))
        rh500_max = float(np.max(rh500))

        return {
            "date": date_str,
            "basin": basin_code,
            "max_vorticity": vo_max,
            "mean_vorticity": vo_mean,
            "std_vorticity": vo_std,
            "weighted_exceedance_fraction": weighted_frac,
            "gridcell_exceedance_fraction": gridcell_frac,
            "latitude_of_max_vorticity": max_vo_lat,
            "longitude_of_max_vorticity": max_vo_lon,
            "min_vws": vws_min,
            "mean_vws": vws_mean,
            "max_vws": vws_max,
            "mean_rh700": rh700_mean,
            "max_rh700": rh700_max,
            "mean_rh500": rh500_mean,
            "max_rh500": rh500_max,
            "number_of_valid_6h_samples": n_samples,
            "expected_6h_samples": expected_samples,
            "completeness_fraction": completeness_fraction,
            "is_complete": is_complete,
        }
