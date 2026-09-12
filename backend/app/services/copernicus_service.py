import os
import json
import math
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..models.schemas import (
    BoundingBox,
    HistoricalStatusResponse,
    HistoricalSliceResponse,
    HistoricalPointResponse,
)

logger = logging.getLogger("copernicus_service")

# ==============================================================================
# DATASET CONFIGURATION & TARGET SPECIFICATION
# ==============================================================================
COPERNICUS_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
TARGET_VARIABLES = ["mlotst", "so", "thetao", "uo", "vo", "zos"]
DEFAULT_MIN_LON = 50.0
DEFAULT_MAX_LON = 100.0
DEFAULT_MIN_LAT = 0.0
DEFAULT_MAX_LAT = 25.0
DEFAULT_START_DATE = "2024-06-24T00:00:00"
DEFAULT_END_DATE = "2026-06-23T00:00:00"
DEFAULT_DEPTH = 0.49402499198913574

# Canonical variable mapping for Copernicus Marine Ocean Physics
COPERNICUS_VAR_MAPPING: Dict[str, Dict[str, Any]] = {
    "thetao": {
        "standard_key": "temp",
        "standard_name": "Sea Water Potential Temperature",
        "unit": "°C",
        "raw_names": ["thetao", "temp", "temperature"],
    },
    "so": {
        "standard_key": "sal",
        "standard_name": "Sea Water Practical Salinity",
        "unit": "PSU",
        "raw_names": ["so", "sal", "salinity"],
    },
    "uo": {
        "standard_key": "cur_u",
        "standard_name": "Eastward Sea Water Velocity",
        "unit": "m/s",
        "raw_names": ["uo", "u", "cur_u"],
    },
    "vo": {
        "standard_key": "cur_v",
        "standard_name": "Northward Sea Water Velocity",
        "unit": "m/s",
        "raw_names": ["vo", "v", "cur_v"],
    },
    "zos": {
        "standard_key": "ssh",
        "standard_name": "Sea Surface Height Above Geoid",
        "unit": "m",
        "raw_names": ["zos", "ssh", "sea_surface_height"],
    },
    "mlotst": {
        "standard_key": "mld",
        "standard_name": "Ocean Mixed Layer Thickness",
        "unit": "m",
        "raw_names": ["mlotst", "mld", "mixed_layer_depth"],
    },
    "cur": {
        "standard_key": "cur",
        "standard_name": "Ocean Surface Current Speed (Derived)",
        "unit": "m/s",
        "raw_names": ["cur", "current", "current_speed", "speed"],
        "is_derived": True,
    },
}

# Variable alias resolution
ALIAS_MAP: Dict[str, str] = {
    "temp": "thetao",
    "temperature": "thetao",
    "thetao": "thetao",
    "sal": "so",
    "salinity": "so",
    "so": "so",
    "cur_u": "uo",
    "uo": "uo",
    "u": "uo",
    "cur_v": "vo",
    "vo": "vo",
    "v": "vo",
    "ssh": "zos",
    "zos": "zos",
    "sea_surface_height": "zos",
    "mld": "mlotst",
    "mlotst": "mlotst",
    "mixed_layer_depth": "mlotst",
    "cur": "cur",
    "current": "cur",
    "current_speed": "cur",
    "speed": "cur",
}


class CopernicusService:
    """
    Dedicated Copernicus Marine Historical Data Service.
    - Manages background download of 2-year daily reanalysis via copernicusmarine.subset()
    - Enforces mandatory post-download validation (variables, bounds, dates, integrity)
    - Records detailed metadata & provenance
    - Employs lazy disk-backed xarray slicing (never loads 2 years into memory)
    - Resolves standardized variables and dynamically derives current speed from uo/vo
    """

    _ds: Optional[xr.Dataset] = None
    _ds_cache: Dict[str, xr.Dataset] = {}
    _ds_lock = threading.Lock()
    _ingestion_lock = threading.Lock()
    _status: str = "idle"  # "idle" | "in_progress" | "ready" | "failed"
    _error_message: Optional[str] = None
    _started_at: Optional[str] = None
    _task_id: Optional[str] = None

    @classmethod
    def get_data_dir(cls) -> str:
        """Returns the managed data directory path for Copernicus files."""
        configured = os.environ.get("COPERNICUS_DATA_DIR")
        if configured:
            if os.path.isabs(configured):
                base_dir = configured
            else:
                base_dir = os.path.join(os.path.dirname(__file__), "..", "..", configured)
        else:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "copernicus")

        os.makedirs(base_dir, exist_ok=True)
        return os.path.abspath(base_dir)

    @classmethod
    def get_2yr_netcdf_path(cls) -> str:
        """Returns target path for the 2-year Copernicus physics dataset NetCDF."""
        return os.path.join(cls.get_data_dir(), "copernicus_phy_2yr_surface.nc")

    @classmethod
    def get_10yr_netcdf_path(cls) -> str:
        """Returns target path for the 10-year Copernicus physics dataset NetCDF."""
        return os.path.join(cls.get_data_dir(), "copernicus_phy_10yr_surface.nc")

    @classmethod
    def get_netcdf_path(cls, version: Optional[str] = None) -> str:
        """
        Returns path for the Copernicus physics dataset NetCDF.
        Supports version='10yr', version='2yr', or environment variable COPERNICUS_NETCDF_PATH.
        Defaults to 2-year baseline file unless 10-year is requested or configured.
        """
        env_override = os.environ.get("COPERNICUS_NETCDF_PATH") or os.environ.get("COPERNICUS_NETCDF_FILE")
        if env_override:
            if os.path.isabs(env_override):
                return env_override
            return os.path.join(cls.get_data_dir(), env_override)

        env_version = os.environ.get("COPERNICUS_DATASET_VERSION", "").lower()
        target_version = (version or env_version).lower()

        if target_version in ("10yr", "10y", "10-year", "10_year", "v2"):
            path_10 = cls.get_10yr_netcdf_path()
            if os.path.exists(path_10):
                return path_10
            # If explicitly requested but not yet present, return path_10 for downloads/checks
            return path_10

        return cls.get_2yr_netcdf_path()

    @classmethod
    def get_metadata_path(cls, version: Optional[str] = None) -> str:
        """Returns path to metadata/provenance JSON record."""
        target_version = (version or os.environ.get("COPERNICUS_DATASET_VERSION", "")).lower()
        if target_version in ("10yr", "10y", "10-year", "10_year", "v2"):
            return os.path.join(cls.get_data_dir(), "metadata_10yr.json")
        return os.path.join(cls.get_data_dir(), "metadata.json")

    # ==========================================================================
    # DATASET VALIDATION & PROVENANCE
    # ==========================================================================
    @classmethod
    def validate_dataset(cls, filepath: str) -> Dict[str, Any]:
        """
        Mandatory post-download validation engine.
        Checks:
        1. File existence and non-zero size
        2. All 6 required variables: mlotst, so, thetao, uo, vo, zos
        3. Spatial bounds: Lon [50, 100], Lat [0, 25]
        4. Surface depth level: ~0.494m
        5. Temporal bounds and daily frequency
        6. Numeric integrity (not empty, valid non-NaN percentages)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Copernicus dataset file not found at '{filepath}'")

        file_size = os.path.getsize(filepath)
        if file_size < 1024:
            raise ValueError(f"Copernicus dataset file is suspiciously small ({file_size} bytes). Download may be corrupted.")

        # Open dataset with xarray for validation
        with xr.open_dataset(filepath) as ds:
            # 1. Variables check
            found_vars = list(ds.data_vars.keys())
            missing_vars = [v for v in TARGET_VARIABLES if v not in found_vars]
            variables_valid = len(missing_vars) == 0

            # 2. Coordinates & Spatial bounds
            coords = list(ds.coords.keys())
            lat_name = next((c for c in ["latitude", "lat", "LAT"] if c in coords), None)
            lon_name = next((c for c in ["longitude", "lon", "LON"] if c in coords), None)
            time_name = next((c for c in ["time", "record"] if c in coords), None)
            depth_name = next((c for c in ["depth", "lev", "level"] if c in coords), None)

            if not lat_name or not lon_name:
                raise ValueError(f"Could not identify latitude/longitude coordinates in Copernicus dataset. Found: {coords}")

            lats = ds[lat_name].values
            lons = ds[lon_name].values
            min_lat, max_lat = float(np.nanmin(lats)), float(np.nanmax(lats))
            min_lon, max_lon = float(np.nanmin(lons)), float(np.nanmax(lons))

            # Allow small float epsilon in bounds
            bounds_valid = (min_lat <= DEFAULT_MIN_LAT + 0.1 and max_lat >= DEFAULT_MAX_LAT - 0.1 and
                            min_lon <= DEFAULT_MIN_LON + 0.1 and max_lon >= DEFAULT_MAX_LON - 0.1)

            # 3. Depth check
            depth_val = None
            depth_valid = True
            if depth_name and depth_name in ds:
                depth_val = float(ds[depth_name].values[0]) if ds[depth_name].ndim > 0 else float(ds[depth_name].values)
                depth_valid = abs(depth_val - DEFAULT_DEPTH) < 1.0

            # 4. Dates & Daily frequency
            time_valid = False
            daily_frequency = False
            total_days = 0
            start_date_str = ""
            end_date_str = ""

            if time_name and time_name in ds:
                times = pd.to_datetime(ds[time_name].values)
                total_days = len(times)
                if total_days > 0:
                    start_date_str = times[0].strftime("%Y-%m-%d")
                    end_date_str = times[-1].strftime("%Y-%m-%d")
                    time_valid = total_days >= 1
                    if total_days > 1:
                        # Check median diff in days
                        diffs = (times[1:] - times[:-1]).total_seconds() / 86400.0
                        median_diff = float(np.median(diffs))
                        daily_frequency = 0.9 <= median_diff <= 1.1
                    else:
                        daily_frequency = True

            # 5. Data integrity (check first time slice of thetao for non-NaN values)
            integrity_valid = False
            sample_var = "thetao" if "thetao" in ds else (found_vars[0] if found_vars else None)
            if sample_var:
                slice_arr = ds[sample_var].isel({time_name: 0} if time_name else {}).values
                valid_count = int(np.sum(~np.isnan(slice_arr)))
                integrity_valid = valid_count > 0

            is_valid = (
                variables_valid and
                bounds_valid and
                depth_valid and
                time_valid and
                daily_frequency and
                integrity_valid
            )

            validation_report = {
                "is_valid": is_valid,
                "file_size_bytes": file_size,
                "file_size_human": f"{file_size / (1024 * 1024):.2f} MB",
                "variables_found": found_vars,
                "missing_variables": missing_vars,
                "variables_valid": variables_valid,
                "spatial_bounds": {
                    "min_lat": round(min_lat, 3),
                    "max_lat": round(max_lat, 3),
                    "min_lon": round(min_lon, 3),
                    "max_lon": round(max_lon, 3),
                },
                "bounds_valid": bounds_valid,
                "depth_m": depth_val,
                "depth_valid": depth_valid,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "total_days": total_days,
                "time_valid": time_valid,
                "daily_frequency": daily_frequency,
                "integrity_valid": integrity_valid,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }

            if not is_valid:
                reasons = []
                if not variables_valid:
                    reasons.append(f"Missing required variables: {missing_vars}")
                if not bounds_valid:
                    reasons.append(f"Bounds mismatch: expected [{DEFAULT_MIN_LAT}, {DEFAULT_MAX_LAT}] lat, [{DEFAULT_MIN_LON}, {DEFAULT_MAX_LON}] lon")
                if not daily_frequency:
                    reasons.append("Non-daily temporal frequency")
                if not integrity_valid:
                    reasons.append("Corrupt or all-NaN data slice")
                logger.warning(f"Copernicus validation failed: {'; '.join(reasons)}")

            return validation_report

    @classmethod
    def save_provenance(cls, validation_report: Dict[str, Any]):
        """Persists dataset metadata and provenance to metadata.json."""
        metadata = {
            "dataset_id": COPERNICUS_DATASET_ID,
            "source": "Copernicus Marine Service (E.U. Copernicus Programme)",
            "product": "Global Ocean Physics Reanalysis (GLORYS12V1 / Multi-Year)",
            "spatial_resolution": "0.083° (~9 km)",
            "temporal_resolution": "Daily (P1D)",
            "depth_level_m": DEFAULT_DEPTH,
            "target_variables": TARGET_VARIABLES,
            "derived_variables": ["cur (Surface Current Speed via sqrt(uo² + vo²))"],
            "validation": validation_report,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "file_path": cls.get_netcdf_path(),
        }
        with open(cls.get_metadata_path(), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # ==========================================================================
    # BACKGROUND INGESTION ENGINE
    # ==========================================================================
    @classmethod
    def trigger_ingest_background(cls) -> Tuple[bool, str, str]:
        """
        Triggers non-blocking Copernicus Marine ingestion in a background thread.
        Credentials MUST be provided via environment variables:
        COPERNICUSMARINE_SERVICE_USERNAME & COPERNICUSMARINE_SERVICE_PASSWORD
        or pre-configured via `copernicusmarine login`.
        Returns (started: bool, status_message: str, task_id: str).
        """
        if cls._ingestion_lock.locked():
            return False, "Ingestion is already in progress.", cls._task_id or "active_task"

        cls._task_id = f"copernicus_ingest_{int(datetime.now(timezone.utc).timestamp())}"
        cls._started_at = datetime.now(timezone.utc).isoformat()
        cls._status = "in_progress"
        cls._error_message = None

        thread = threading.Thread(target=cls._run_ingestion_task, daemon=True)
        thread.start()
        return True, "Copernicus Marine ingestion started in background.", cls._task_id

    @classmethod
    def _run_ingestion_task(cls):
        """Worker executing copernicusmarine.subset() with error handling."""
        with cls._ingestion_lock:
            try:
                import copernicusmarine
            except ImportError:
                cls._status = "failed"
                cls._error_message = "copernicusmarine package is not installed. Please install requirements."
                logger.error(cls._error_message)
                return

            out_dir = cls.get_data_dir()
            out_nc = cls.get_netcdf_path()

            # Check credentials from environment
            username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
            password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

            kwargs: Dict[str, Any] = {
                "dataset_id": COPERNICUS_DATASET_ID,
                "variables": TARGET_VARIABLES,
                "minimum_longitude": DEFAULT_MIN_LON,
                "maximum_longitude": DEFAULT_MAX_LON,
                "minimum_latitude": DEFAULT_MIN_LAT,
                "maximum_latitude": DEFAULT_MAX_LAT,
                "start_datetime": DEFAULT_START_DATE,
                "end_datetime": DEFAULT_END_DATE,
                "minimum_depth": DEFAULT_DEPTH,
                "maximum_depth": DEFAULT_DEPTH,
                "output_directory": out_dir,
                "output_filename": "copernicus_phy_2yr_surface.nc",
                "overwrite": True,
            }

            if username and password:
                kwargs["username"] = username
                kwargs["password"] = password

            logger.info(f"Starting Copernicus Marine subset download for {COPERNICUS_DATASET_ID}...")
            try:
                copernicusmarine.subset(**kwargs)

                # Locate downloaded file
                if not os.path.exists(out_nc):
                    candidates = [f for f in os.listdir(out_dir) if f.endswith(".nc") and f != "copernicus_phy_2yr_surface.nc"]
                    if candidates:
                        cand_path = os.path.join(out_dir, candidates[0])
                        os.replace(cand_path, out_nc)

                # Validate
                report = cls.validate_dataset(out_nc)
                if not report["is_valid"]:
                    cls._status = "failed"
                    cls._error_message = "Post-download validation failed: Dataset does not satisfy target specification."
                    logger.error(cls._error_message)
                    return

                cls.save_provenance(report)

                # Invalidate cached xarray handle so next read reopens fresh file
                with cls._ds_lock:
                    if cls._ds is not None:
                        try:
                            cls._ds.close()
                        except Exception:
                            pass
                        cls._ds = None

                cls._status = "ready"
                cls._error_message = None
                logger.info("Copernicus Marine dataset successfully ingested and validated.")

            except Exception as e:
                cls._status = "failed"
                cls._error_message = f"Copernicus ingestion failed: {str(e)}"
                logger.error(cls._error_message, exc_info=True)

    # ==========================================================================
    # DATA ACCESS & LAZY XARRAY SLICING
    # ==========================================================================
    @classmethod
    def get_dataset(cls, version: Optional[str] = None, filepath: Optional[str] = None) -> xr.Dataset:
        """
        Lazily opens the NetCDF dataset using disk-backed access.
        Never loads the entire multi-year volume into memory.
        """
        nc_path = filepath or cls.get_netcdf_path(version=version)
        if not os.path.exists(nc_path):
            raise FileNotFoundError(
                f"Copernicus Marine historical dataset is not available at '{nc_path}'. "
                "Please run ingestion first using POST /api/historical/ingest or python backend/scripts/ingest_copernicus.py."
            )

        with cls._ds_lock:
            if nc_path not in cls._ds_cache:
                cls._ds_cache[nc_path] = xr.open_dataset(nc_path, engine="netcdf4")
            # Maintain backward compatibility with _ds
            cls._ds = cls._ds_cache[nc_path]
            return cls._ds_cache[nc_path]

    @classmethod
    def get_status(cls) -> HistoricalStatusResponse:
        """Returns current ingestion status, validation details, and provenance."""
        nc_path = cls.get_netcdf_path()
        meta_path = cls.get_metadata_path()

        file_exists = os.path.exists(nc_path)
        meta_exists = os.path.exists(meta_path)

        provenance = None
        validation = None
        if meta_exists:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    provenance = json.load(f)
                    validation = provenance.get("validation")
            except Exception:
                pass

        # If file exists and valid, status is ready
        status = cls._status
        is_ready = False
        if file_exists and (status != "in_progress"):
            if validation and validation.get("is_valid"):
                status = "ready"
                is_ready = True
            elif not validation:
                try:
                    rep = cls.validate_dataset(nc_path)
                    if rep.get("is_valid"):
                        cls.save_provenance(rep)
                        validation = rep
                        status = "ready"
                        is_ready = True
                except Exception as e:
                    status = "failed"
                    cls._error_message = str(e)

        file_size = os.path.getsize(nc_path) if file_exists else 0
        file_size_human = f"{file_size / (1024 * 1024):.2f} MB" if file_exists else "0 MB"

        start_date = validation.get("start_date") if validation else None
        end_date = validation.get("end_date") if validation else None
        total_days = validation.get("total_days") if validation else 0
        depth_m = validation.get("depth_m") if validation else DEFAULT_DEPTH

        bbox = None
        if validation and "spatial_bounds" in validation:
            sb = validation["spatial_bounds"]
            bbox = BoundingBox(
                min_lat=sb["min_lat"],
                max_lat=sb["max_lat"],
                min_lon=sb["min_lon"],
                max_lon=sb["max_lon"],
            )

        msg = "Copernicus historical dataset is ready for historical queries." if is_ready else (
            "Ingestion is currently running in the background." if status == "in_progress" else (
                f"Ingestion failed: {cls._error_message}" if status == "failed" else
                "Dataset has not been downloaded yet. Run ingestion via CLI or API."
            )
        )

        return HistoricalStatusResponse(
            status=status,
            dataset_id=COPERNICUS_DATASET_ID,
            is_ready=is_ready,
            file_path=nc_path if file_exists else None,
            file_size_bytes=file_size if file_exists else None,
            file_size_human=file_size_human if file_exists else None,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            bounding_box=bbox,
            depth_m=depth_m,
            variables=["temp", "sal", "cur_u", "cur_v", "cur", "ssh", "mld"],
            validation=validation,
            provenance=provenance,
            message=msg,
        )

    @classmethod
    def get_slice(
        cls,
        date_str: str,
        variable: str = "temp",
        resolution: int = 48,
        bbox: Optional[BoundingBox] = None,
    ) -> HistoricalSliceResponse:
        """
        Slices a 2D horizontal depth slice at the given calendar date and variable.
        Uses lazy disk-backed indexing: reads only the target 2D slice from disk.
        Does NOT silently fall back to synthetic data.
        """
        ds = cls.get_dataset()

        # Resolve variable key
        raw_key = variable.lower().strip()
        canonical = ALIAS_MAP.get(raw_key)
        if not canonical:
            canonical = raw_key

        coord_time = next((c for c in ["time", "record"] if c in ds.coords or c in ds.dims), None)
        coord_lat = next((c for c in ["latitude", "lat"] if c in ds.coords or c in ds.dims), None)
        coord_lon = next((c for c in ["longitude", "lon"] if c in ds.coords or c in ds.dims), None)
        coord_depth = next((c for c in ["depth", "lev"] if c in ds.coords or c in ds.dims), None)

        if not coord_time or not coord_lat or not coord_lon:
            raise ValueError("Dataset missing spatial/temporal coordinates.")

        try:
            target_dt = pd.to_datetime(date_str)
        except Exception:
            raise ValueError(f"Invalid date format '{date_str}'. Use ISO format YYYY-MM-DD.")

        dataset_times = pd.to_datetime(ds[coord_time].values)
        min_dt = dataset_times[0]
        max_dt = dataset_times[-1]

        if target_dt < min_dt - pd.Timedelta(days=1) or target_dt > max_dt + pd.Timedelta(days=1):
            raise ValueError(
                f"Requested date '{date_str}' is outside available Copernicus coverage: "
                f"{min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}."
            )

        # Slice target variable
        # Handle derived current speed: sqrt(uo^2 + vo^2)
        if canonical == "cur":
            if "uo" not in ds or "vo" not in ds:
                raise ValueError("Current speed derivation requires 'uo' and 'vo' in dataset.")
            uo_arr = ds["uo"].sel({coord_time: target_dt}, method="nearest")
            vo_arr = ds["vo"].sel({coord_time: target_dt}, method="nearest")
            if coord_depth and coord_depth in uo_arr.dims:
                uo_arr = uo_arr.isel({coord_depth: 0})
                vo_arr = vo_arr.isel({coord_depth: 0})

            # Compute sqrt(uo^2 + vo^2) for the slice only
            speed_arr = np.sqrt(np.square(uo_arr.values) + np.square(vo_arr.values))
            data_values = speed_arr
            unit = "m/s"
            std_name = "Surface Current Velocity Speed"
        else:
            if canonical not in ds:
                raise ValueError(f"Variable '{variable}' ({canonical}) not present in Copernicus dataset.")
            var_slice = ds[canonical].sel({coord_time: target_dt}, method="nearest")
            if coord_depth and coord_depth in var_slice.dims:
                var_slice = var_slice.isel({coord_depth: 0})

            data_values = var_slice.values
            unit = str(var_slice.attrs.get("units", COPERNICUS_VAR_MAPPING.get(canonical, {}).get("unit", "")))
            std_name = str(var_slice.attrs.get("long_name", COPERNICUS_VAR_MAPPING.get(canonical, {}).get("standard_name", canonical)))

        if data_values.ndim > 2:
            data_values = data_values.reshape((data_values.shape[-2], data_values.shape[-1]))

        lats = ds[coord_lat].values
        lons = ds[coord_lon].values
        if lats.ndim > 1:
            lats = lats[:, 0]
        if lons.ndim > 1:
            lons = lons[0, :]

        # Downsample for JSON transmission
        ny, nx = data_values.shape
        step_y = max(1, ny // resolution)
        step_x = max(1, nx // resolution)

        sub_lats = [float(y) for y in lats[::step_y]]
        sub_lons = [float(x) for x in lons[::step_x]]
        sub_vals = data_values[::step_y, ::step_x]

        # Clean NaN/Inf
        clean_grid: List[List[Optional[float]]] = []
        valid_nums: List[float] = []
        for row in sub_vals:
            clean_row = []
            for v in row:
                if np.isnan(v) or np.isinf(v):
                    clean_row.append(None)
                else:
                    fv = round(float(v), 3)
                    clean_row.append(fv)
                    valid_nums.append(fv)
            clean_grid.append(clean_row)

        min_val = float(np.min(valid_nums)) if valid_nums else 0.0
        max_val = float(np.max(valid_nums)) if valid_nums else 1.0

        return HistoricalSliceResponse(
            dataset_id=COPERNICUS_DATASET_ID,
            date=target_dt.strftime("%Y-%m-%d"),
            variable=variable,
            standard_name=std_name,
            depth=DEFAULT_DEPTH,
            unit=unit,
            lats=sub_lats,
            lons=sub_lons,
            values=clean_grid,
            min_val=min_val,
            max_val=max_val,
            shape=[len(sub_lats), len(sub_lons)],
            provenance="Copernicus Marine Global Ocean Physics Reanalysis (0.083° daily)",
        )

    @classmethod
    def get_point(
        cls,
        date_str: str,
        lat: float,
        lon: float,
        variable: str = "temp"
    ) -> HistoricalPointResponse:
        """Queries single point value at specified coordinate and date."""
        ds = cls.get_dataset()

        raw_key = variable.lower().strip()
        canonical = ALIAS_MAP.get(raw_key, raw_key)

        coord_time = next((c for c in ["time", "record"] if c in ds.coords or c in ds.dims), None)
        coord_lat = next((c for c in ["latitude", "lat"] if c in ds.coords or c in ds.dims), None)
        coord_lon = next((c for c in ["longitude", "lon"] if c in ds.coords or c in ds.dims), None)
        coord_depth = next((c for c in ["depth", "lev"] if c in ds.coords or c in ds.dims), None)

        target_dt = pd.to_datetime(date_str)

        if canonical == "cur":
            uo_pt = ds["uo"].sel({coord_time: target_dt, coord_lat: lat, coord_lon: lon}, method="nearest")
            vo_pt = ds["vo"].sel({coord_time: target_dt, coord_lat: lat, coord_lon: lon}, method="nearest")
            if coord_depth and coord_depth in uo_pt.dims:
                uo_pt = uo_pt.isel({coord_depth: 0})
                vo_pt = vo_pt.isel({coord_depth: 0})
            val = float(np.sqrt(float(uo_pt.values)**2 + float(vo_pt.values)**2))
            unit = "m/s"
        else:
            var_pt = ds[canonical].sel({coord_time: target_dt, coord_lat: lat, coord_lon: lon}, method="nearest")
            if coord_depth and coord_depth in var_pt.dims:
                var_pt = var_pt.isel({coord_depth: 0})
            val = float(var_pt.values)
            unit = str(var_pt.attrs.get("units", ""))

        clean_val = None if (np.isnan(val) or np.isinf(val)) else round(val, 3)

        return HistoricalPointResponse(
            date=target_dt.strftime("%Y-%m-%d"),
            lat=lat,
            lon=lon,
            variable=variable,
            value=clean_val,
            unit=unit,
            depth=DEFAULT_DEPTH,
            provenance="Copernicus Marine Global Ocean Physics Reanalysis",
        )

    # ==========================================================================
    # TEST FIXTURE GENERATOR (Strictly for automated unit tests & offline CI)
    # ==========================================================================
    @classmethod
    def create_test_fixture(cls, output_path: str, days: int = 5) -> str:
        """
        Creates a valid CF-compliant NetCDF fixture matching Copernicus Marine
        coordinates and all 6 target variables. Strictly for automated test suites.
        """
        times = pd.date_range("2024-06-24", periods=days, freq="D")
        lats = np.linspace(DEFAULT_MIN_LAT, DEFAULT_MAX_LAT, 15)
        lons = np.linspace(DEFAULT_MIN_LON, DEFAULT_MAX_LON, 25)
        depths = np.array([DEFAULT_DEPTH])

        T, D, Y, X = np.meshgrid(np.arange(len(times)), depths, lats, lons, indexing="ij")

        # Realistic physics fields with spatial and temporal variation
        thetao = (28.0 - (Y - 12.0) * 0.3 + np.sin(X * 0.1) + 0.5 * np.cos(T * 0.1)).astype(np.float32)
        so = (34.5 + (Y * 0.05) - np.cos(X * 0.1) * 0.3 + 0.1 * np.sin(T * 0.15)).astype(np.float32)
        uo = (0.25 * np.cos(Y * 0.2) + 0.1 * np.sin(T * 0.2)).astype(np.float32)
        vo = (-0.15 * np.sin(X * 0.2) + 0.05 * np.cos(T * 0.2)).astype(np.float32)
        zos = (0.12 * np.sin(X * 0.1) * np.cos(Y * 0.1) + 0.02 * np.sin(T * 0.1)).astype(np.float32)
        mlotst = (35.0 + 10.0 * np.sin(Y * 0.3) + 2.0 * np.cos(T * 0.2)).astype(np.float32)

        ds = xr.Dataset(
            data_vars={
                "thetao": (["time", "depth", "latitude", "longitude"], thetao, {"units": "degrees_C", "long_name": "Sea Water Potential Temperature"}),
                "so": (["time", "depth", "latitude", "longitude"], so, {"units": "1e-3", "long_name": "Sea Water Practical Salinity"}),
                "uo": (["time", "depth", "latitude", "longitude"], uo, {"units": "m/s", "long_name": "Eastward Velocity"}),
                "vo": (["time", "depth", "latitude", "longitude"], vo, {"units": "m/s", "long_name": "Northward Velocity"}),
                "zos": (["time", "depth", "latitude", "longitude"], zos, {"units": "m", "long_name": "Sea Surface Height Above Geoid"}),
                "mlotst": (["time", "depth", "latitude", "longitude"], mlotst, {"units": "m", "long_name": "Ocean Mixed Layer Thickness"}),
            },
            coords={
                "time": times,
                "depth": depths,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "title": "Copernicus Marine Test Fixture",
                "dataset_id": COPERNICUS_DATASET_ID,
                "Conventions": "CF-1.8",
            },
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ds.to_netcdf(output_path)
        return output_path
