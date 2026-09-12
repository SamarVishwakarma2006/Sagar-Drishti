"""
Standalone CLI script to trigger, track, or validate Copernicus Marine 2-Year
Global Ocean Physics Reanalysis ingestion for Sagar Drishti.

Usage:
  python backend/scripts/ingest_copernicus.py
  python backend/scripts/ingest_copernicus.py --status
  python backend/scripts/ingest_copernicus.py --validate-only
"""
import os
import sys
import argparse
import logging

# Ensure clean UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("copernicus_cli")

from app.services.copernicus_service import (
    CopernicusService,
    COPERNICUS_DATASET_ID,
    TARGET_VARIABLES,
    DEFAULT_MIN_LON,
    DEFAULT_MAX_LON,
    DEFAULT_MIN_LAT,
    DEFAULT_MAX_LAT,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_DEPTH,
)


def print_status():
    status = CopernicusService.get_status()
    print("\n=======================================================")
    print("[SAGAR DRISHTI] COPERNICUS MARINE HISTORICAL STATUS")
    print("=======================================================")
    print(f"Dataset ID:     {status.dataset_id}")
    print(f"Status:         {status.status.upper()}")
    print(f"Is Ready:       {status.is_ready}")
    print(f"NetCDF Path:    {status.file_path or 'Not found'}")
    print(f"File Size:      {status.file_size_human or '0 MB'}")
    print(f"Start Date:     {status.start_date or 'N/A'}")
    print(f"End Date:       {status.end_date or 'N/A'}")
    print(f"Total Days:     {status.total_days or 0}")
    print(f"Variables:      {', '.join(status.variables)}")
    print(f"Message:        {status.message}")
    print("=======================================================\n")
    return 0 if status.is_ready else 1


def validate_existing():
    nc_path = CopernicusService.get_netcdf_path()
    if not os.path.exists(nc_path):
        print(f"[!] No existing Copernicus dataset found at: {nc_path}")
        return 1

    print(f"[*] Validating existing dataset: {nc_path} ...")
    try:
        report = CopernicusService.validate_dataset(nc_path)
        print("\n--- VALIDATION REPORT ---")
        for k, v in report.items():
            print(f"  {k}: {v}")
        if report.get("is_valid"):
            print("\n[+] Dataset passed all validation criteria.")
            CopernicusService.save_provenance(report)
            return 0
        else:
            print("\n[!] Dataset failed validation.")
            return 1
    except Exception as e:
        print(f"[!] Validation error: {e}")
        return 1


def run_ingestion():
    print("\n=======================================================")
    print("[SAGAR DRISHTI] COPERNICUS MARINE INGESTION WORKER")
    print("=======================================================")
    print(f"Target Dataset:  {COPERNICUS_DATASET_ID}")
    print(f"Variables:       {', '.join(TARGET_VARIABLES)}")
    print(f"Spatial Extent:  Lon [{DEFAULT_MIN_LON}, {DEFAULT_MAX_LON}], Lat [{DEFAULT_MIN_LAT}, {DEFAULT_MAX_LAT}]")
    print(f"Depth:           {DEFAULT_DEPTH} m")
    print(f"Date Range:      {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")
    print(f"Destination:     {CopernicusService.get_netcdf_path()}")
    print("=======================================================\n")

    # Check copernicusmarine installation
    try:
        import copernicusmarine
    except ImportError:
        print("[!] Error: 'copernicusmarine' is not installed.")
        print("    Please run: pip install copernicusmarine")
        return 1

    # Check credentials
    username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if not username or not password:
        print("[*] Note: COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD")
        print("    are not set in environment. copernicusmarine will check local credentials store")
        print("    or interactive login session.")

    print("[*] Starting subset extraction...")
    started, msg, task_id = CopernicusService.trigger_ingest_background()
    print(f"[*] Task Status: {msg} (Task ID: {task_id})")

    # Wait for completion synchronously in CLI mode
    import time
    while True:
        status = CopernicusService.get_status()
        if status.status == "in_progress":
            print("    ... Ingestion in progress, waiting ...", flush=True)
            time.sleep(5)
        elif status.status == "ready":
            print("\n[+] Ingestion completed successfully!")
            print_status()
            return 0
        else:
            print(f"\n[!] Ingestion failed: {status.message}")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Copernicus Marine Historical Data Ingestion for Sagar Drishti")
    parser.add_argument("--status", action="store_true", help="Print current Copernicus dataset status")
    parser.add_argument("--validate-only", action="store_true", help="Run validation on existing NetCDF file")

    args = parser.parse_args()

    if args.status:
        sys.exit(print_status())
    elif args.validate_only:
        sys.exit(validate_existing())
    else:
        sys.exit(run_ingestion())


if __name__ == "__main__":
    main()
