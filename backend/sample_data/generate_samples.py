"""
Utility script to generate sample oceanographic datasets for testing Sagar Drishti:
1. sample_argo_profiles.csv (Tabular Argo/CTD observations in Indian Ocean)
2. sample_ocean_grid.nc (Gridded NetCDF ocean model dataset)
"""
import os
import numpy as np
import pandas as pd


def generate_sample_csv(output_path: str):
    """Generates realistic Argo float profile observations around the Bay of Bengal & Arabian Sea."""
    records = []
    floats = [
        {"id": "2903341", "lat": 18.2, "lon": 88.5, "cycle": 104, "platform": "APEX-INCOIS"},
        {"id": "2903342", "lat": 16.8, "lon": 87.2, "cycle": 98, "platform": "PROVOR-INCOIS"},
        {"id": "2903343", "lat": 15.4, "lon": 89.1, "cycle": 112, "platform": "APEX-INCOIS"},
        {"id": "2903344", "lat": 17.5, "lon": 68.2, "cycle": 85, "platform": "NAVIS-INCOIS"},
        {"id": "2903345", "lat": 14.1, "lon": 66.8, "cycle": 91, "platform": "APEX-INCOIS"},
    ]

    depth_levels = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]

    for f in floats:
        base_lat = f["lat"]
        base_lon = f["lon"]
        cycle = f["cycle"]
        plat_id = f["id"]

        for d in depth_levels:
            # Physics formulas
            f_mld = 1.0 / (1.0 + np.exp((d - 75.0) / 40.0))
            temp = 2.8 + (29.2 - 2.8) * f_mld + (np.random.rand() - 0.5) * 0.2
            sal = 34.8 + (31.5 - 34.8) * np.exp(-d / 90.0) + (0.4 * np.exp(-((d - 110) / 80)**2) if "88" in str(base_lon) else 0)
            oxy = max(2.5, 170.0 * np.exp(-d / 80.0) - 150.0 * np.exp(-((d - 350) / 220)**2) + 40.0)
            speed = max(0.04, 0.45 * np.exp(-d / 280.0) + (np.random.rand() - 0.5) * 0.05)
            direction = (45.0 + d * 0.1) % 360.0

            records.append({
                "PLATFORM_NUMBER": plat_id,
                "CYCLE_NUMBER": cycle,
                "LATITUDE": base_lat + (np.random.rand() - 0.5) * 0.02,
                "LONGITUDE": base_lon + (np.random.rand() - 0.5) * 0.02,
                "DEPTH": d,
                "TEMP": round(temp, 2),
                "PSAL": round(sal, 2),
                "DOXY": round(oxy, 1),
                "CURRENT_SPEED": round(speed, 2),
                "CURRENT_DIR": round(direction, 1),
                "DATE_TIME": "2025-03-21T06:00:00Z"
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[+] Generated sample CSV: {output_path} ({len(df)} rows)")


def generate_sample_netcdf(output_path: str):
    """Generates synthetic CF-compliant NetCDF grid using xarray."""
    try:
        import xarray as xr
    except ImportError:
        print("[!] xarray not yet installed in global env. Sample NetCDF generator will run when dependencies are installed.")
        return

    lats = np.linspace(12.0, 22.0, 21)
    lons = np.linspace(80.0, 94.0, 29)
    depths = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    times = pd.date_range("2025-03-21", periods=3, freq="D")

    shape = (len(times), len(depths), len(lats), len(lons))

    # Create synthetic variable fields
    T, D, Y, X = np.meshgrid(np.arange(len(times)), depths, lats, lons, indexing="ij")

    temp_field = 3.0 + (28.5 - 3.0) / (1.0 + np.exp((D - 60.0) / 35.0)) + np.sin(Y * 0.5) * np.cos(X * 0.5)
    sal_field = 34.8 + (31.5 - 34.8) * np.exp(-D / 100.0) + 0.3 * np.exp(-((D - 120.0) / 70.0)**2)
    u_field = 0.35 * np.exp(-D / 250.0) * np.sin(Y * 0.8)
    v_field = 0.25 * np.exp(-D / 250.0) * np.cos(X * 0.8)
    oxy_field = np.maximum(5.0, 185.0 * np.exp(-D / 90.0) - 155.0 * np.exp(-((D - 360.0) / 200.0)**2))

    ds = xr.Dataset(
        data_vars={
            "temperature": (["time", "depth", "latitude", "longitude"], temp_field.astype(np.float32), {"units": "degC", "long_name": "Sea Water Temperature"}),
            "salinity": (["time", "depth", "latitude", "longitude"], sal_field.astype(np.float32), {"units": "PSU", "long_name": "Sea Water Practical Salinity"}),
            "u": (["time", "depth", "latitude", "longitude"], u_field.astype(np.float32), {"units": "m/s", "long_name": "Eastward Zonal Velocity"}),
            "v": (["time", "depth", "latitude", "longitude"], v_field.astype(np.float32), {"units": "m/s", "long_name": "Northward Meridional Velocity"}),
            "oxygen": (["time", "depth", "latitude", "longitude"], oxy_field.astype(np.float32), {"units": "umol/kg", "long_name": "Dissolved Oxygen Concentration"}),
        },
        coords={
            "time": times,
            "depth": depths,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "Sagar Drishti Bay of Bengal High-Resolution Reanalysis Grid",
            "source": "INCOIS INDOMOD / ROMS assimilation model",
            "Conventions": "CF-1.8"
        }
    )

    ds.to_netcdf(output_path)
    print(f"[+] Generated sample NetCDF: {output_path}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    base_dir = os.path.dirname(__file__)
    csv_file = os.path.join(base_dir, "sample_argo_profiles.csv")
    nc_file = os.path.join(base_dir, "sample_ocean_grid.nc")

    generate_sample_csv(csv_file)
    try:
        generate_sample_netcdf(nc_file)
    except Exception as e:
        print(f"Note: NetCDF sample generation: {e}")
