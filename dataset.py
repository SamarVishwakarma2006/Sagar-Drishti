import os
import copernicusmarine

# Target output directory for Sagar-Drishti Copernicus historical engine
output_dir = os.path.join(os.path.dirname(__file__), "backend", "data", "copernicus")
os.makedirs(output_dir, exist_ok=True)

kwargs = {
    "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "variables": ["mlotst", "so", "thetao", "uo", "vo", "zos"],
    "minimum_longitude": 50,
    "maximum_longitude": 100,
    "minimum_latitude": 0,
    "maximum_latitude": 25,
    "start_datetime": "2024-06-24T00:00:00",
    "end_datetime": "2026-06-23T00:00:00",
    "minimum_depth": 0.49402499198913574,
    "maximum_depth": 0.49402499198913574,
    "output_directory": output_dir,
    "output_filename": "copernicus_phy_2yr_surface.nc",
    "overwrite": True,
}

# Optional credentials from environment if configured
username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
if username and password:
    kwargs["username"] = username
    kwargs["password"] = password

print("[*] Starting Copernicus Marine subset extraction...")
print(f"[*] Destination: {os.path.join(output_dir, 'copernicus_phy_2yr_surface.nc')}")
print("[*] (If credentials are not yet saved, you will be prompted for your Copernicus Marine username & password)")

copernicusmarine.subset(**kwargs)
print("[+] Download complete!")