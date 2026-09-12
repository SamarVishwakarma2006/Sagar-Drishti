"""
Dedicated Copernicus Marine 10-Year Surface Physics Ingestion & Validation Script.
Dataset: cmems_mod_glo_phy_my_0.083deg_P1D-m
Variables: mlotst, so, thetao, uo, vo, zos
Spatial Domain: Lon [50.0, 100.0], Lat [0.0, 25.0]
Temporal Domain: 2016-06-24T00:00:00 to 2026-06-23T00:00:00 (3,652 daily steps)
Depth: Surface ~0.494025 m
Target File: backend/data/copernicus/copernicus_phy_10yr_surface.nc (~7.39 GB)
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.copernicus_service import CopernicusService
from app.services.validate_10yr_copernicus import Copernicus10YearValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_10yr_copernicus")

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ["mlotst", "so", "thetao", "uo", "vo", "zos"]
MIN_LON = 50.0
MAX_LON = 100.0
MIN_LAT = 0.0
MAX_LAT = 25.0
START_DATE = "2016-06-24T00:00:00"
END_DATE = "2026-06-23T00:00:00"
MIN_DEPTH = 0.49402499198913574
MAX_DEPTH = 0.494025
OUTPUT_FILENAME = "copernicus_phy_10yr_surface.nc"


def run_ingestion(dry_run: bool = False, validate_only: bool = False) -> int:
    output_dir = CopernicusService.get_data_dir()
    target_nc = os.path.join(output_dir, OUTPUT_FILENAME)

    print("=" * 80)
    print("SAGAR-DRISHTI: 10-YEAR COPERNICUS REANALYSIS INGESTION")
    print("=" * 80)
    print(f"Dataset ID:       {DATASET_ID}")
    print(f"Variables:        {', '.join(VARIABLES)}")
    print(f"Spatial Extent:   Lon [{MIN_LON}, {MAX_LON}], Lat [{MIN_LAT}, {MAX_LAT}]")
    print(f"Temporal Domain:  {START_DATE} to {END_DATE}")
    print(f"Depth:            {MIN_DEPTH} m (Surface level)")
    print(f"Target NetCDF:    {target_nc}")
    print(f"Dry Run:          {dry_run}")
    print(f"Validate Only:    {validate_only}")
    print("=" * 80)

    if validate_only:
        if not os.path.exists(target_nc):
            print(f"[!] Target NetCDF does not exist: {target_nc}")
            return 1
        return _run_validation(target_nc)

    try:
        import copernicusmarine
    except ImportError:
        print("[!] Error: 'copernicusmarine' package is not installed.")
        print("    Run: pip install copernicusmarine")
        return 1

    print("[*] Initiating copernicusmarine.subset() extraction...")
    try:
        res = copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=VARIABLES,
            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,
            start_datetime=START_DATE,
            end_datetime=END_DATE,
            minimum_depth=MIN_DEPTH,
            maximum_depth=MAX_DEPTH,
            output_filename=OUTPUT_FILENAME,
            output_directory=output_dir,
            dry_run=dry_run,
            overwrite=True,
        )
        print(f"[+] Copernicus Marine returned: {res}")

        if dry_run:
            print("[+] Dry run succeeded. Server is ready to serve the 10-year surface dataset.")
            return 0

        print(f"[+] Download completed. Running authoritative post-download validation on {target_nc}...")
        return _run_validation(target_nc)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


def _run_validation(nc_path: str) -> int:
    report = Copernicus10YearValidator.validate(nc_path)
    chk = report.get("checks", {})
    temp = chk.get("temporal", {})
    spat = chk.get("spatial", {})
    dep = chk.get("depth", {})
    vars_found = chk.get("variables", {}).get("found", [])

    print("\n--- VALIDATION REPORT ---")
    print(f"Valid:              {report.get('is_valid')}")
    print(f"Total Days:         {temp.get('actual_timesteps')} (Expected: {temp.get('expected_timesteps', 3652)})")
    print(f"File Size:          {report.get('file_size_gb')} GB ({os.path.getsize(nc_path)} bytes)")
    print(f"Variables:          {', '.join(vars_found)}")
    print(f"Spatial Domain:     Lat {spat.get('lat_bounds')}, Lon {spat.get('lon_bounds')}")
    print(f"Surface Depth:      {dep.get('depth_level_m')} m")

    if report.get("errors"):
        print("\n[!] Errors encountered:")
        for err in report["errors"]:
            print(f"    - {err}")

    if report.get("is_valid"):
        meta_path = CopernicusService.get_metadata_path(version="10yr")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_id": DATASET_ID,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "file_path": os.path.abspath(nc_path),
                "validation": report,
            }, f, indent=2)
        print(f"[+] Provenance metadata saved to {meta_path}")
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(description="Ingest and validate 10-year Copernicus physics dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Perform server-side request dry run without downloading.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing NetCDF file.")
    args = parser.parse_args()

    sys.exit(run_ingestion(dry_run=args.dry_run, validate_only=args.validate_only))


if __name__ == "__main__":
    main()
