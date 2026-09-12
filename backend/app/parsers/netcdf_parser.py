import os
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import xarray as xr
from ..models.schemas import BoundingBox, VariableStats, SitePhysics


class NetCDFParser:
    """
    High-Performance NetCDF (.nc, .nc4) parser for oceanographic datasets.
    Extracts spatial coordinates, vertical levels, physical variables,
    calculates spatial bounding boxes, and generates 2D horizontal slices.
    """

    LAT_NAMES = ["lat", "latitude", "LAT", "nav_lat", "NAV_LAT", "y", "lat_rho", "lat_u", "lat_v"]
    LON_NAMES = ["lon", "longitude", "LON", "nav_lon", "NAV_LON", "x", "lon_rho", "lon_u", "lon_v"]
    DEPTH_NAMES = ["depth", "lev", "level", "depth_t", "depth_u", "depth_v", "z", "pres", "pressure", "s_rho"]
    TIME_NAMES = ["time", "time_counter", "record", "t", "ocean_time", "time_centered"]

    VAR_MAPPING = {
        "temp": ["temp", "temperature", "votemper", "thetao", "sst", "TEMP", "sea_water_temperature", "to", "theta"],
        "sal": ["sal", "salinity", "vosaline", "so", "sss", "PSAL", "sea_water_salinity", "sa", "salt"],
        "cur_u": ["u", "uo", "vozocrtx", "u_current", "cur_u", "current_u", "eastward_sea_water_velocity", "u_east"],
        "cur_v": ["v", "vo", "vomecrty", "v_current", "cur_v", "current_v", "northward_sea_water_velocity", "v_north"],
        "oxy": ["oxygen", "o2", "doxy", "dissolved_oxygen", "DOX2", "moles_of_oxygen_per_unit_mass_in_sea_water", "o2_sat"],
        "chl": ["chla", "chlorophyll", "chl", "CPHL", "mass_concentration_of_chlorophyll_a_in_sea_water"],
        "ssh": ["zos", "sea_surface_height", "ssh", "sea_surface_elevation", "zos_detrended"],
        "mld": ["mlotst", "mixed_layer_depth", "mld", "ocean_mixed_layer_thickness"]
    }

    @classmethod
    def _find_coord(cls, ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
        for name in candidates:
            if name in ds.coords or name in ds.dims or name in ds.variables:
                return name
        return None

    @classmethod
    def _map_variable(cls, var_name: str) -> Optional[str]:
        low = var_name.lower()
        for std_key, aliases in cls.VAR_MAPPING.items():
            if low in aliases or any(alias in low for alias in aliases):
                return std_key
        return None

    @classmethod
    def parse_dataset_from_bytes(cls, content: bytes, filename: str) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """
        Parses NetCDF bytes into an xarray.Dataset and extracts full oceanographic metadata.
        """
        # xarray requires a seekable file or file path for HDF5/NetCDF4
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            ds = xr.open_dataset(tmp_path, engine="netcdf4")
            metadata = cls.extract_metadata(ds, filename, tmp_path)
            return ds, metadata
        except Exception as e:
            # Try h5netcdf or default scipy engine if netcdf4 fails on 32-bit/classic NetCDF
            try:
                ds = xr.open_dataset(tmp_path)
                metadata = cls.extract_metadata(ds, filename, tmp_path)
                return ds, metadata
            except Exception as inner_e:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                raise ValueError(f"Failed to ingest NetCDF file '{filename}': {str(e)} | {str(inner_e)}")

    @classmethod
    def extract_metadata(cls, ds: xr.Dataset, filename: str, file_path: str = "") -> Dict[str, Any]:
        """
        Extracts dimensional extents, coordinates, bounding box, depth layers,
        and variable summary statistics.
        """
        lat_col = cls._find_coord(ds, cls.LAT_NAMES)
        lon_col = cls._find_coord(ds, cls.LON_NAMES)
        depth_col = cls._find_coord(ds, cls.DEPTH_NAMES)
        time_col = cls._find_coord(ds, cls.TIME_NAMES)

        if not lat_col or not lon_col:
            raise ValueError(f"Could not identify spatial coordinates (latitude/longitude) in '{filename}'. Found: {list(ds.coords.keys()) + list(ds.dims.keys())}")

        lats = ds[lat_col].values
        lons = ds[lon_col].values

        # Handle 2D grid coordinates (e.g. curvilinear grids)
        min_lat = float(np.nanmin(lats))
        max_lat = float(np.nanmax(lats))
        min_lon = float(np.nanmin(lons))
        max_lon = float(np.nanmax(lons))

        # Longitude normalization if 0..360
        if max_lon > 180 and min_lon >= 0:
            lons = np.where(lons > 180, lons - 360, lons)
            min_lon = float(np.nanmin(lons))
            max_lon = float(np.nanmax(lons))

        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        depth_levels: List[float] = [0.0]
        max_depth = 2000.0
        if depth_col:
            depth_vals = np.abs(ds[depth_col].values).astype(float)
            depth_levels = sorted([float(z) for z in depth_vals if not np.isnan(z)])
            if depth_levels:
                max_depth = float(depth_levels[-1])

        time_steps: List[str] = []
        if time_col:
            try:
                times = ds[time_col].values
                time_steps = [str(t)[:19] for t in times[:48]]
            except Exception:
                time_steps = ["Live/Current"]

        # Detect physical ocean variables
        detected_vars: Dict[str, VariableStats] = {}
        variable_keys: List[str] = []

        for var_name, data_var in ds.data_vars.items():
            canonical = cls._map_variable(str(var_name))
            if canonical or len(ds.data_vars) <= 6:
                key = canonical if canonical else str(var_name).lower()
                variable_keys.append(key)

                # Compute min/max/mean safely
                try:
                    arr = data_var.values
                    valid_arr = arr[~np.isnan(arr)]
                    if len(valid_arr) > 0:
                        vmin = float(np.min(valid_arr))
                        vmax = float(np.max(valid_arr))
                        vmean = float(np.mean(valid_arr))
                        vstd = float(np.std(valid_arr))
                    else:
                        vmin, vmax, vmean, vstd = 0.0, 1.0, 0.5, 0.1
                except Exception:
                    vmin, vmax, vmean, vstd = 0.0, 1.0, 0.5, 0.1

                unit = str(data_var.attrs.get("units", data_var.attrs.get("unit", "")))
                long_name = str(data_var.attrs.get("long_name", var_name))
                std_name = str(data_var.attrs.get("standard_name", var_name))

                detected_vars[key] = VariableStats(
                    name=var_name,
                    standard_name=std_name,
                    long_name=long_name,
                    unit=unit,
                    min=vmin,
                    max=vmax,
                    mean=vmean,
                    std=vstd
                )

        bbox = BoundingBox(
            min_lat=round(min_lat, 4),
            max_lat=round(max_lat, 4),
            min_lon=round(min_lon, 4),
            max_lon=round(max_lon, 4)
        )

        clean_id = f"custom_nc_{abs(hash(filename)) % 1000000}"
        clean_name = filename.replace(".nc4", "").replace(".nc", "").replace("_", " ").title()

        # Build SitePhysics object
        site_record = SitePhysics(
            id=clean_id,
            name=f"{clean_name} (NetCDF)",
            region=f"Coverage: {bbox.min_lat:.2f}° to {bbox.max_lat:.2f}°N, {bbox.min_lon:.2f}° to {bbox.max_lon:.2f}°E",
            lat=round(center_lat, 4),
            lon=round(center_lon, 4),
            maxDepth=max(50.0, round(max_depth, 1)),
            blurb=f"User-uploaded gridded model/satellite dataset '{filename}'. Contains {len(detected_vars)} variables across {len(depth_levels)} depth levels.",
            ts=detected_vars.get("temp", VariableStats(name="temp", min=20, max=30)).max,
            ss=detected_vars.get("sal", VariableStats(name="sal", min=32, max=36)).max,
            td=detected_vars.get("temp", VariableStats(name="temp", min=2, max=10)).min,
            sd=detected_vars.get("sal", VariableStats(name="sal", min=34, max=35)).min,
            mld=min(60.0, max(20.0, max_depth * 0.1)),
            tw=45.0,
            salMaxAmp=0.4,
            salMaxZ=100.0,
            flow=detected_vars.get("cur_u", VariableStats(name="cur_u", min=0, max=1.2)).max,
            eddy=220.0,
            bgU=0.2,
            bgV=0.1,
            o2s=detected_vars.get("oxy", VariableStats(name="oxy", min=150, max=220)).max,
            o2d=detected_vars.get("oxy", VariableStats(name="oxy", min=50, max=170)).min,
            o2z0=90.0,
            o2minAmp=120.0,
            o2minZ=min(400.0, max_depth * 0.4),
            o2minW=200.0,
            bbox=bbox,
            variables=list(detected_vars.keys()) if detected_vars else ["temp", "sal", "cur", "oxy"],
            isCustom=True,
            sourceType="NETCDF_GRIDDED",
            floats=[]
        )

        return {
            "site_id": clean_id,
            "filename": filename,
            "file_path": file_path,
            "bbox": bbox,
            "depth_levels": depth_levels,
            "time_steps": time_steps,
            "coord_names": {
                "lat": lat_col,
                "lon": lon_col,
                "depth": depth_col,
                "time": time_col
            },
            "variables": detected_vars,
            "site_record": site_record
        }

    @classmethod
    def slice_2d_grid(
        cls,
        ds: xr.Dataset,
        var_key: str,
        depth: float = 0.0,
        time_idx: int = 0,
        target_res: int = 64
    ) -> Dict[str, Any]:
        """
        Slices a 2D horizontal field (lat, lon) at depth level Z and time index T,
        downsampling if needed for efficient WebGL transmission.
        """
        lat_col = cls._find_coord(ds, cls.LAT_NAMES)
        lon_col = cls._find_coord(ds, cls.LON_NAMES)
        depth_col = cls._find_coord(ds, cls.DEPTH_NAMES)
        time_col = cls._find_coord(ds, cls.TIME_NAMES)

        # Locate target variable in dataset
        target_var = None
        for name in ds.data_vars:
            if cls._map_variable(name) == var_key or name.lower() == var_key.lower():
                target_var = name
                break
        if not target_var and len(ds.data_vars) > 0:
            target_var = list(ds.data_vars.keys())[0]

        data_array = ds[target_var]

        # Select time slice
        if time_col and time_col in data_array.dims:
            t_len = data_array.sizes[time_col]
            t_idx = min(time_idx, t_len - 1)
            data_array = data_array.isel({time_col: t_idx})

        # Select vertical depth slice (nearest depth)
        if depth_col and depth_col in data_array.dims:
            try:
                data_array = data_array.sel({depth_col: depth}, method="nearest")
            except Exception:
                data_array = data_array.isel({depth_col: 0})

        # Extract 2D array
        raw_vals = data_array.values
        if raw_vals.ndim > 2:
            raw_vals = raw_vals.reshape((raw_vals.shape[-2], raw_vals.shape[-1]))

        lats = ds[lat_col].values
        lons = ds[lon_col].values

        if lats.ndim > 1:
            lats = lats[:, 0]
        if lons.ndim > 1:
            lons = lons[0, :]

        # Downsample if grid is large
        ny, nx = raw_vals.shape
        step_y = max(1, ny // target_res)
        step_x = max(1, nx // target_res)

        sub_lats = [float(y) for y in lats[::step_y]]
        sub_lons = [float(x) for x in lons[::step_x]]
        sub_vals = raw_vals[::step_y, ::step_x]

        # Clean NaN/Inf to None for JSON serialization
        clean_grid: List[List[Optional[float]]] = []
        for row in sub_vals:
            clean_row = []
            for val in row:
                if np.isnan(val) or np.isinf(val):
                    clean_row.append(None)
                else:
                    clean_row.append(round(float(val), 3))
            clean_grid.append(clean_row)

        valid_nums = [v for row in clean_grid for v in row if v is not None]
        min_v = float(np.min(valid_nums)) if valid_nums else 0.0
        max_v = float(np.max(valid_nums)) if valid_nums else 1.0

        return {
            "variable": var_key,
            "depth": depth,
            "time_idx": time_idx,
            "lats": sub_lats,
            "lons": sub_lons,
            "values": clean_grid,
            "min_val": min_v,
            "max_val": max_v,
            "unit": str(data_array.attrs.get("units", "")),
            "shape": [len(sub_lats), len(sub_lons)]
        }
