import numpy as np
from typing import Dict, Any, List, Optional
import xarray as xr
from .site_registry import SiteRegistry
from ..models.schemas import DataSliceResponse
from ..parsers.netcdf_parser import NetCDFParser


class SliceEngine:
    """
    Slicing engine that extracts 2D horizontal depth slices from either
    uploaded NetCDF datasets or computed synthetic ocean grids.
    """

    @classmethod
    def get_slice(
        cls,
        site_id: str,
        variable: str = "temp",
        depth: float = 0.0,
        time_offset: float = 0.0,
        resolution: int = 48
    ) -> DataSliceResponse:
        site = SiteRegistry.get_site(site_id)
        if not site:
            raise ValueError(f"Site '{site_id}' not found.")

        dataset = SiteRegistry.get_dataset(site_id)

        # If we have an active xarray dataset for this custom site, slice from it
        if dataset is not None and isinstance(dataset, xr.Dataset):
            time_idx = int(np.clip((time_offset + 48) / 2, 0, 48))
            slice_data = NetCDFParser.slice_2d_grid(
                dataset,
                var_key=variable,
                depth=depth,
                time_idx=time_idx,
                target_res=resolution
            )
            return DataSliceResponse(
                dataset_id=site_id,
                variable=slice_data["variable"],
                depth=slice_data["depth"],
                time_offset=time_offset,
                lats=slice_data["lats"],
                lons=slice_data["lons"],
                values=slice_data["values"],
                min_val=slice_data["min_val"],
                max_val=slice_data["max_val"],
                unit=slice_data["unit"],
                shape=slice_data["shape"]
            )

        # Otherwise, synthesize a physically realistic 2D grid slice around the site
        return cls._generate_synthetic_slice(site, variable, depth, time_offset, resolution)

    @classmethod
    def _generate_synthetic_slice(
        cls,
        site: Any,
        variable: str,
        depth: float,
        time_offset: float,
        resolution: int = 48
    ) -> DataSliceResponse:
        lat_span = 3.0
        lon_span = 3.0
        lats = np.linspace(site.lat - lat_span / 2, site.lat + lat_span / 2, resolution)
        lons = np.linspace(site.lon - lon_span / 2, site.lon + lon_span / 2, resolution)

        grid_values: List[List[Optional[float]]] = []
        z = depth
        t = time_offset

        # Calculate base physics
        f_t = 1.0 / (1.0 + np.exp((z - (site.mld + 55.0)) / site.tw))
        base_temp = site.td + (site.ts - site.td) * f_t
        base_sal = site.sd + (site.ss - site.sd) * np.exp(-z / 95.0)
        if site.salMaxAmp > 0:
            base_sal += site.salMaxAmp * np.exp(-((z - site.salMaxZ) / 85.0) ** 2)
        base_o2 = site.o2d + (site.o2s - site.o2d) * np.exp(-z / site.o2z0)
        omz = site.o2minAmp * np.exp(-((z - site.o2minZ) / site.o2minW) ** 2)
        base_o2 = max(2.0, base_o2 - omz)
        base_cur = site.flow * (0.16 + 0.84 * np.exp(-z / 300.0))

        kx = 2.0 * np.pi / site.eddy
        ky = 2.0 * np.pi / (site.eddy * 0.72)

        for lat_val in lats:
            row: List[Optional[float]] = []
            dy = (lat_val - site.lat) * 110.54  # km
            for lon_val in lons:
                dx = (lon_val - site.lon) * 111.32 * np.cos(np.radians(site.lat))  # km
                eddy_mod = np.sin(kx * dx * 10 + t * 0.1) * np.cos(ky * dy * 10 + t * 0.07)

                if variable == "temp":
                    val = base_temp + 0.4 * eddy_mod
                    unit = "°C"
                elif variable == "sal":
                    val = base_sal + 0.15 * eddy_mod
                    unit = "PSU"
                elif variable in ["cur", "cur_u", "cur_v"]:
                    val = max(0.02, base_cur * (1.0 + 0.5 * eddy_mod))
                    unit = "m/s"
                elif variable in ["oxy", "o2"]:
                    val = max(1.0, base_o2 + 8.0 * eddy_mod)
                    unit = "µmol/kg"
                else:
                    val = base_temp
                    unit = ""

                row.append(round(float(val), 3))
            grid_values.append(row)

        all_vals = [v for r in grid_values for v in r if v is not None]
        min_v = float(np.min(all_vals)) if all_vals else 0.0
        max_v = float(np.max(all_vals)) if all_vals else 1.0

        return DataSliceResponse(
            dataset_id=site.id,
            variable=variable,
            depth=depth,
            time_offset=time_offset,
            lats=[round(float(y), 4) for y in lats],
            lons=[round(float(x), 4) for x in lons],
            values=grid_values,
            min_val=min_v,
            max_val=max_v,
            unit=unit,
            shape=[len(lats), len(lons)]
        )
