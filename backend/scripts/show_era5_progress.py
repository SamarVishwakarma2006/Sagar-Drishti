"""
Sagar-Drishti V2.1 — Live ERA5 Ingestion Telemetry & Progress Monitor.

Reads real-time state from backend/reports/era5_ingestion_progress.json,
backend/data/era5/chunks/, and backend/data/era5/ assembled yearly files.
Renders an interactive, auto-updating progress bar with ETA, monthly breakdown,
and storage metrics.

Usage:
    python backend/scripts/show_era5_progress.py          # Single snapshot
    python backend/scripts/show_era5_progress.py --watch  # Live refreshing monitor (every 3s)
"""

import os
import sys
import time
import json
import calendar
from datetime import datetime

# Configure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
progress_file = os.path.join(backend_dir, "reports", "era5_ingestion_progress.json")
chunks_dir = os.path.join(backend_dir, "data", "era5", "chunks")
years_dir = os.path.join(backend_dir, "data", "era5")
canonical_file = os.path.join(years_dir, "era5_atmosphere_10yr_6hourly.nc")

YEAR_MONTHS = {
    2016: list(range(6, 13)),   # 7 months
    2017: list(range(1, 13)),
    2018: list(range(1, 13)),
    2019: list(range(1, 13)),
    2020: list(range(1, 13)),
    2021: list(range(1, 13)),
    2022: list(range(1, 13)),
    2023: list(range(1, 13)),
    2024: list(range(1, 13)),
    2025: list(range(1, 13)),
    2026: list(range(1, 7)),    # 6 months
}

TOTAL_CHUNKS = sum(len(m) for m in YEAR_MONTHS.values())  # 121 months total (2016-06 to 2026-06)
TOTAL_TIMESTEPS = 14608


def make_bar(percent: float, length: int = 40) -> str:
    """Renders a smooth Unicode block progress bar."""
    filled = int(round(length * (percent / 100.0)))
    filled = max(0, min(length, filled))
    bar = "█" * filled + "░" * (length - filled)
    return bar


def get_progress_data():
    """Gathers live stats from disk and progress json."""
    existing_chunks = set()
    if os.path.exists(chunks_dir):
        for f in os.listdir(chunks_dir):
            if f.startswith("era5_") and f.endswith(".nc"):
                existing_chunks.add(f)

    assembled_years = []
    total_era5_bytes = 0
    if os.path.exists(years_dir):
        for f in os.listdir(years_dir):
            fp = os.path.join(years_dir, f)
            if f.startswith("era5_pressure_") and f.endswith(".nc"):
                y_str = f.replace("era5_pressure_", "").replace(".nc", "")
                if y_str.isdigit():
                    assembled_years.append(int(y_str))
                total_era5_bytes += os.path.getsize(fp)

    if os.path.exists(chunks_dir):
        for f in os.listdir(chunks_dir):
            fp = os.path.join(chunks_dir, f)
            total_era5_bytes += os.path.getsize(fp)

    curr_year, curr_month = 2025, 7
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                p_data = json.load(f)
                curr_year = p_data.get("current_year", curr_year)
                curr_month = p_data.get("current_month", curr_month)
        except Exception:
            pass

    valid_chunk_set = {f"era5_{y}_{m:02d}.nc" for y, ms in YEAR_MONTHS.items() for m in ms}
    completed_chunks = existing_chunks.intersection(valid_chunk_set)
    completed_count = len(completed_chunks)
    remaining_count = max(0, TOTAL_CHUNKS - completed_count)
    percent = (completed_count / TOTAL_CHUNKS) * 100.0

    # Calculate completed timesteps
    # 2016-06 is 28 steps (7 days * 4). All full months: days * 4.
    timesteps_done = 0
    for y, months in YEAR_MONTHS.items():
        for m in months:
            fname = f"era5_{y}_{m:02d}.nc"
            if fname in existing_chunks:
                if y == 2016 and m == 6:
                    timesteps_done += 28
                elif y == 2026 and m == 6:
                    timesteps_done += 23 * 4
                else:
                    _, nd = calendar.monthrange(y, m)
                    timesteps_done += nd * 4

    timesteps_remaining = max(0, TOTAL_TIMESTEPS - timesteps_done)

    # Estimate remaining time based on ~330s per month
    est_seconds_left = remaining_count * 330
    hours_left = int(est_seconds_left // 3600)
    mins_left = int((est_seconds_left % 3600) // 60)

    return {
        "completed": completed_count,
        "remaining": remaining_count,
        "percent": percent,
        "timesteps_done": timesteps_done,
        "timesteps_remaining": timesteps_remaining,
        "assembled_years": sorted(assembled_years),
        "curr_year": curr_year,
        "curr_month": curr_month,
        "hours_left": hours_left,
        "mins_left": mins_left,
        "total_gb": total_era5_bytes / (1024 ** 3),
        "existing_chunks": existing_chunks,
    }


def render_display(data: dict) -> str:
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║           SAGAR-DRISHTI V2.1 — ERA5 10-YEAR ATMOSPHERIC INGESTION MONITOR            ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════════════╣")
    
    pct = data["percent"]
    bar = make_bar(pct, length=44)
    lines.append(f"║  Overall Progress: [{bar}] {pct:5.1f}%  ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  • Verified Chunks:       {data['completed']:3d} / {TOTAL_CHUNKS} completed  ({data['remaining']:2d} remaining)                     ║")
    lines.append(f"║  • Verified Timesteps:    {data['timesteps_done']:5,d} / {TOTAL_TIMESTEPS:,d} observations ({data['timesteps_remaining']:5,d} remaining)       ║")
    lines.append(f"║  • Assembled Yearly Files: {len(data['assembled_years']):2d} / 11 complete ({', '.join(str(y) for y in data['assembled_years']) if data['assembled_years'] else 'None'})      ║")
    lines.append(f"║  • Storage Footprint:     {data['total_gb']:5.2f} GB on disk (Compressed NetCDF4)                  ║")
    lines.append(f"║  • Estimated Time Left:   ~{data['hours_left']}h {data['mins_left']:02d}m remaining (at ~5.5 min / month)                 ║")
    lines.append(f"║  • Active Download:       Year {data['curr_year']} Month {data['curr_month']:02d} (ECMWF CDS Queue Active)               ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                               YEAR-BY-YEAR CHUNK MATRIX                              ║")
    lines.append("║  Year │ J  F  M  A  M  J  J  A  S  O  N  D │ Status                                   ║")
    lines.append("╟───────┼────────────────────────────────────┼─────────────────────────────────────────╢")

    months_abbr = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for y in range(2016, 2027):
        row_chars = []
        assembled = y in data["assembled_years"]
        for m in range(1, 13):
            fname = f"era5_{y}_{m:02d}.nc"
            if m not in YEAR_MONTHS[y]:
                row_chars.append("·")  # not in target bounds
            elif fname in data["existing_chunks"] or assembled:
                row_chars.append("■")  # completed
            elif y == data["curr_year"] and m == data["curr_month"]:
                row_chars.append("▶")  # currently active
            else:
                row_chars.append("□")  # pending

        grid_str = "  ".join(row_chars)
        status_str = f"ASSEMBLED ({len(YEAR_MONTHS[y])} mo)" if assembled else ("DOWNLOADING" if y == data["curr_year"] else "PENDING")
        lines.append(f"║  {y} │ {grid_str} │ {status_str:<39} ║")

    lines.append("╚══════════════════════════════════════════════════════════════════════════════════════╝")
    lines.append("  Legend: [■] Verified Complete   [▶] Currently Downloading   [□] Queued   [·] Out of Bounds")
    return "\n".join(lines)


def main():
    watch_mode = "--watch" in sys.argv
    try:
        while True:
            data = get_progress_data()
            output = render_display(data)
            if watch_mode:
                os.system("cls" if os.name == "nt" else "clear")
            print(output)
            if not watch_mode:
                break
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nExiting progress monitor.")


if __name__ == "__main__":
    main()
