"""
IMD RSMC New Delhi Best-Track Dataset Ingestion & Normalization Layer.
Parses multi-year sheets from official IMD Best Track workbook (1982-2026),
standardizes coordinate conventions, date/time timestamps, storm naming,
intensity/grade classifications, and identifies tentative near-real-time records.
"""

import os
import re
import math
import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple, Set

import pandas as pd
import numpy as np

logger = logging.getLogger("sagar_drishti.imd_parser")

# Standard IMD Cyclone Intensity Scale & Grade mapping
IMD_GRADE_MAPPING: Dict[str, Dict[str, Any]] = {
    "D": {"category": "depression", "name": "Depression", "min_wind_kts": 17, "max_wind_kts": 27},
    "DD": {"category": "deep_depression", "name": "Deep Depression", "min_wind_kts": 28, "max_wind_kts": 33},
    "CS": {"category": "cyclonic_storm", "name": "Cyclonic Storm", "min_wind_kts": 34, "max_wind_kts": 47},
    "SCS": {"category": "severe_cyclonic_storm", "name": "Severe Cyclonic Storm", "min_wind_kts": 48, "max_wind_kts": 63},
    "VSCS": {"category": "very_severe_cyclonic_storm", "name": "Very Severe Cyclonic Storm", "min_wind_kts": 64, "max_wind_kts": 89},
    "ESCS": {"category": "extremely_severe_cyclonic_storm", "name": "Extremely Severe Cyclonic Storm", "min_wind_kts": 90, "max_wind_kts": 119},
    "SUCS": {"category": "super_cyclonic_storm", "name": "Super Cyclonic Storm", "min_wind_kts": 120, "max_wind_kts": 200},
    "LOW": {"category": "low_pressure", "name": "Well Marked Low Pressure Area", "min_wind_kts": 10, "max_wind_kts": 16},
    "WML": {"category": "low_pressure", "name": "Well Marked Low Pressure Area", "min_wind_kts": 10, "max_wind_kts": 16},
}


class IMDBestTrackParser:
    """
    Robust ingestion parser for IMD Best-Track Excel workbooks.
    Handles varying header rows, varying date formats (DD-MM-YYYY vs DD/MM/YYYY vs Excel datetimes),
    and preserves complete provenance.
    """

    @classmethod
    def parse_coordinate(cls, val: Any) -> Optional[float]:
        """Parses latitude or longitude from numbers or text strings with N/S/E/W suffixes."""
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            if math.isnan(val):
                return None
            return float(val)
        
        s = str(val).strip().upper()
        if not s:
            return None
        # Remove deg symbols or trailing characters
        match = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        if not match:
            return None
        try:
            num = float(match.group(0))
            if "S" in s or "W" in s:
                num = -num
            return num
        except (ValueError, TypeError):
            return None

    @classmethod
    def parse_date(cls, val: Any, year_hint: Optional[int] = None) -> Optional[str]:
        """Parses calendar date from varied formats to ISO YYYY-MM-DD."""
        if pd.isna(val):
            return None
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.strftime("%Y-%m-%d")
        if isinstance(val, date):
            return val.strftime("%Y-%m-%d")
            
        s = str(val).strip()
        if not s:
            return None

        # Clean string
        s = s.split(" ")[0].strip()
        # Try common formats
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Handle two digit year
        for fmt in ("%d-%m-%y", "%d/%m/%y"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    @classmethod
    def parse_time_utc(cls, val: Any) -> Optional[str]:
        """Parses UTC time string, float (e.g. 300 for 03:00 UTC, 1200 for 12:00 UTC), or time object."""
        if pd.isna(val):
            return "00:00:00"
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.strftime("%H:%M:%S")
        if hasattr(val, "hour") and hasattr(val, "minute"):
            return f"{val.hour:02d}:{val.minute:02d}:00"
            
        try:
            if isinstance(val, (int, float)):
                iv = int(val)
                hh = iv // 100
                mm = iv % 100
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    return f"{hh:02d}:{mm:02d}:00"
            s = str(val).strip().replace(".", ":")
            if ":" in s:
                parts = s.split(":")
                hh = int(parts[0])
                mm = int(parts[1]) if len(parts) > 1 else 0
                return f"{hh:02d}:{mm:02d}:00"
            elif len(s) in (3, 4) and s.isdigit():
                iv = int(s)
                return f"{iv // 100:02d}:{iv % 100:02d}:00"
        except Exception:
            pass
        return "00:00:00"

    @classmethod
    def normalize_grade(cls, raw_grade: Any) -> Tuple[str, str]:
        """Normalizes grade code (e.g. VSCS, CS, DD, D) and returns (normalized_code, category_name)."""
        if pd.isna(raw_grade):
            return "UNKNOWN", "unknown"
        g = str(raw_grade).strip().upper().replace(".", "")
        # Match against known prefixes
        for code, info in IMD_GRADE_MAPPING.items():
            if g == code or g.startswith(code + " ") or g.endswith(" " + code):
                return code, info["category"]
        if "SUPER" in g or "SUCS" in g:
            return "SUCS", "super_cyclonic_storm"
        if "EXTREMELY" in g or "ESCS" in g:
            return "ESCS", "extremely_severe_cyclonic_storm"
        if "VERY SEVERE" in g or "VSCS" in g:
            return "VSCS", "very_severe_cyclonic_storm"
        if "SEVERE" in g or "SCS" in g:
            return "SCS", "severe_cyclonic_storm"
        if "CYCLONIC" in g or "CS" in g:
            return "CS", "cyclonic_storm"
        if "DEEP" in g or "DD" in g:
            return "DD", "deep_depression"
        if "DEPRESSION" in g or "D" in g:
            return "D", "depression"
        return g[:10], "tropical_cyclone"

    @classmethod
    def ingest_workbook(
        cls,
        filepath: str,
        start_year: int = 2016,
        end_year: int = 2026,
        include_tentative_2026: bool = True,
    ) -> Dict[str, Any]:
        """
        Parses all sheets between start_year and end_year from the IMD workbook.
        Returns normalized track points, system registry, and complete validation metrics.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"IMD Best-Track workbook not found at '{filepath}'")

        import openpyxl
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        all_sheets = wb.sheetnames

        target_sheets = [
            s for s in all_sheets
            if s.strip().isdigit() and start_year <= int(s.strip()) <= end_year
        ]

        logger.info(f"Discovered {len(target_sheets)} target sheets: {target_sheets}")

        track_records: List[Dict[str, Any]] = []
        parsing_failures = 0
        missing_coord_count = 0
        invalid_coord_count = 0
        duplicate_count = 0
        seen_points: Set[Tuple[str, str, str, float, float]] = set()

        grade_distribution: Dict[str, int] = {}
        systems_by_id: Dict[str, Dict[str, Any]] = {}

        for sheet_name in target_sheets:
            year_int = int(sheet_name.strip())
            is_tentative = (year_int == 2026)

            ws = wb[sheet_name]
            rows: List[Tuple[Any, ...]] = []
            empty_streak = 0
            for r in ws.iter_rows(values_only=True):
                if any(r):
                    rows.append(r)
                    empty_streak = 0
                else:
                    empty_streak += 1
                    if empty_streak >= 25 and len(rows) > 0:
                        break

            if not rows:
                continue

            # Detect header row dynamically
            header_idx = None
            for idx, r in enumerate(rows[:15]):
                row_str = " ".join([str(v) for v in r if v is not None]).lower()
                if "latitude" in row_str or "date" in row_str or "serial" in row_str:
                    header_idx = idx
                    break

            if header_idx is None:
                parsing_failures += 1
                logger.warning(f"Could not locate header row in sheet {sheet_name}")
                continue

            headers = [str(v).strip().replace("\n", " ") if v is not None else f"col_{i}" for i, v in enumerate(rows[header_idx])]
            df = pd.DataFrame(rows[header_idx + 1:], columns=headers)
            if df.empty:
                continue

            # Column identification via robust keyword matching
            serial_col = next((c for c in df.columns if "serial" in c.lower()), None)
            name_col = next((c for c in df.columns if "name" in c.lower()), None)
            basin_col = next((c for c in df.columns if "basin" in c.lower()), None)
            lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
            lon_col = next((c for c in df.columns if "long" in c.lower() or "lon" in c.lower()), None)
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            time_col = next((c for c in df.columns if "time" in c.lower()), None)
            grade_col = next((c for c in df.columns if "grade" in c.lower()), None)
            wind_col = next((c for c in df.columns if "wind" in c.lower()), None)
            pres_col = next((c for c in df.columns if "pressure" in c.lower() and "drop" not in c.lower()), None)
            drop_col = next((c for c in df.columns if "drop" in c.lower()), None)

            # Drop completely blank rows
            if lat_col and lon_col:
                df = df[df[lat_col].notna() | df[lon_col].notna()].copy()
            if df.empty:
                continue

            # Forward-fill system serial numbers and names for sub-track observations
            # Forward fill name and basin within serial group to prevent cross-storm contamination
            if serial_col:
                df[serial_col] = df[serial_col].ffill()
                if name_col:
                    df[name_col] = df.groupby(serial_col)[name_col].ffill()
                if basin_col:
                    df[basin_col] = df.groupby(serial_col)[basin_col].ffill()

            current_storm_serial = None
            last_valid_date_str: Optional[str] = None

            for _, row in df.iterrows():
                raw_lat = row[lat_col] if lat_col else None
                raw_lon = row[lon_col] if lon_col else None
                lat_val = cls.parse_coordinate(raw_lat)
                lon_val = cls.parse_coordinate(raw_lon)

                if lat_val is None or lon_val is None:
                    missing_coord_count += 1
                    continue

                # Coordinate validation (North Indian Ocean basin roughly -10 to 45°N, 30 to 120°E)
                if not (-10.0 <= lat_val <= 45.0 and 30.0 <= lon_val <= 120.0):
                    invalid_coord_count += 1
                    continue

                raw_serial = row[serial_col] if serial_col else "01"
                try:
                    serial_num = int(float(raw_serial))
                except (ValueError, TypeError):
                    serial_num = str(raw_serial).strip()

                # Check storm boundary: reset date forward-fill state when storm changes
                if serial_num != current_storm_serial:
                    current_storm_serial = serial_num
                    last_valid_date_str = None

                raw_date = row[date_col] if date_col else None
                parsed_date = cls.parse_date(raw_date, year_hint=year_int)

                if parsed_date is not None:
                    date_str = parsed_date
                    last_valid_date_str = parsed_date
                elif last_valid_date_str is not None:
                    # Inherit from most recent valid date in current storm
                    date_str = last_valid_date_str
                else:
                    # Neither a valid date nor a valid previous date exists for this storm
                    logger.warning(
                        f"Sheet {sheet_name}, Storm serial {serial_num}: row has no valid date "
                        f"and no preceding valid date to forward-fill (raw_date={raw_date}). Skipping row."
                    )
                    parsing_failures += 1
                    continue

                raw_time = row[time_col] if time_col else None
                time_str = cls.parse_time_utc(raw_time)

                raw_name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else ""
                if raw_name.lower() in ("nan", "none", ""):
                    raw_name = f"UNNAMED-{serial_num}"

                raw_basin = str(row[basin_col]).strip().upper() if basin_col and pd.notna(row[basin_col]) else "BOB"
                if "ARAB" in raw_basin or raw_basin == "AS" or raw_basin == "ARB":
                    basin_norm = "Arabian Sea"
                elif "BAY" in raw_basin or raw_basin == "BOB":
                    basin_norm = "Bay of Bengal"
                else:
                    basin_norm = "North Indian Ocean"

                norm_grade, cat_name = cls.normalize_grade(row[grade_col] if grade_col else None)
                grade_distribution[norm_grade] = grade_distribution.get(norm_grade, 0) + 1

                # Intensity parsing
                wind_kts = None
                if wind_col and pd.notna(row[wind_col]):
                    try:
                        wind_kts = float(row[wind_col])
                    except (ValueError, TypeError):
                        pass

                central_pres = None
                if pres_col and pd.notna(row[pres_col]):
                    try:
                        central_pres = float(row[pres_col])
                    except (ValueError, TypeError):
                        pass

                # Event system unique ID
                system_id = f"IMD-{year_int}-{serial_num}-{re.sub(r'[^A-Z0-9]', '', raw_name.upper())}"

                # Duplicate detection key: includes time_str to avoid dropping sub-daily observations
                dup_key = (system_id, date_str, time_str, round(lat_val, 3), round(lon_val, 3))
                if dup_key in seen_points:
                    duplicate_count += 1
                    continue
                seen_points.add(dup_key)

                point_record = {
                    "system_id": system_id,
                    "year": year_int,
                    "system_serial": serial_num,
                    "storm_name": raw_name,
                    "basin": basin_norm,
                    "date": date_str,
                    "time_utc": time_str,
                    "datetime_iso": f"{date_str}T{time_str}",
                    "latitude": round(lat_val, 4),
                    "longitude": round(lon_val, 4),
                    "grade": norm_grade,
                    "category": cat_name,
                    "max_wind_kts": wind_kts,
                    "central_pressure_hpa": central_pres,
                    "is_tentative_2026": is_tentative,
                    "source_sheet": sheet_name,
                    "source": "IMD RSMC New Delhi Best Track (1982-2026)",
                }
                track_records.append(point_record)

                # Aggregate system metadata
                if system_id not in systems_by_id:
                    systems_by_id[system_id] = {
                        "system_id": system_id,
                        "year": year_int,
                        "name": raw_name,
                        "basin": basin_norm,
                        "is_tentative_2026": is_tentative,
                        "dates": [date_str],
                        "lats": [lat_val],
                        "lons": [lon_val],
                        "max_grade": norm_grade,
                        "max_wind_kts": wind_kts or 0.0,
                        "min_pressure_hpa": central_pres or 1010.0,
                        "track_points_count": 1,
                    }
                else:
                    sys_entry = systems_by_id[system_id]
                    sys_entry["dates"].append(date_str)
                    sys_entry["lats"].append(lat_val)
                    sys_entry["lons"].append(lon_val)
                    sys_entry["track_points_count"] += 1
                    if wind_kts and wind_kts > sys_entry["max_wind_kts"]:
                        sys_entry["max_wind_kts"] = wind_kts
                        sys_entry["max_grade"] = norm_grade
                    if central_pres and central_pres < sys_entry["min_pressure_hpa"]:
                        sys_entry["min_pressure_hpa"] = central_pres

        wb.close()

        # Finalize system summaries
        systems_list = []
        for sid, sdata in systems_by_id.items():
            dates_sorted = sorted(sdata["dates"])
            systems_list.append({
                "system_id": sid,
                "year": sdata["year"],
                "name": sdata["name"],
                "basin": sdata["basin"],
                "start_date": dates_sorted[0],
                "end_date": dates_sorted[-1],
                "centroid_lat": round(float(np.mean(sdata["lats"])), 4),
                "centroid_lon": round(float(np.mean(sdata["lons"])), 4),
                "bbox": {
                    "min_lat": round(float(np.min(sdata["lats"])), 4),
                    "max_lat": round(float(np.max(sdata["lats"])), 4),
                    "min_lon": round(float(np.min(sdata["lons"])), 4),
                    "max_lon": round(float(np.max(sdata["lons"])), 4),
                },
                "max_grade": sdata["max_grade"],
                "peak_intensity_kts": sdata["max_wind_kts"],
                "track_points_count": sdata["track_points_count"],
                "is_tentative_2026": sdata["is_tentative_2026"],
            })

        track_df = pd.DataFrame(track_records)
        if not track_df.empty and "datetime_iso" in track_df.columns:
            track_df = track_df.sort_values(by=["system_id", "datetime_iso"]).reset_index(drop=True)
            track_records = track_df.to_dict("records")

        date_min = track_df["date"].min() if not track_df.empty else ""
        date_max = track_df["date"].max() if not track_df.empty else ""

        validation_report = {
            "workbook_path": os.path.abspath(filepath),
            "years_discovered": [int(s) for s in target_sheets],
            "total_systems_discovered": len(systems_list),
            "total_track_points": len(track_records),
            "date_range": {"start": date_min, "end": date_max},
            "missing_coordinate_count": missing_coord_count,
            "duplicate_count": duplicate_count,
            "invalid_coordinate_count": invalid_coord_count,
            "parsing_failures": parsing_failures,
            "category_grade_distribution": grade_distribution,
            "tentative_2026_records_count": sum(1 for r in track_records if r["is_tentative_2026"]),
            "finalized_historical_records_count": sum(1 for r in track_records if not r["is_tentative_2026"]),
        }

        return {
            "validation_report": validation_report,
            "track_dataframe": track_df,
            "systems": systems_list,
        }

    @classmethod
    def get_cached_or_parse(
        cls,
        filepath: str,
        cache_dir: Optional[str] = None,
        start_year: int = 2016,
        end_year: int = 2026,
        force_reparse: bool = False,
    ) -> Dict[str, Any]:
        """Loads normalized IMD records from fast disk cache, or parses and saves cache if missing."""
        import json
        if cache_dir is None:
            cache_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical")
            )
        os.makedirs(cache_dir, exist_ok=True)
        parquet_cache = os.path.join(cache_dir, f"imd_tracks_{start_year}_{end_year}.parquet")
        systems_cache = os.path.join(cache_dir, f"imd_systems_{start_year}_{end_year}.json")
        report_cache = os.path.join(cache_dir, f"imd_report_{start_year}_{end_year}.json")

        if not force_reparse and os.path.exists(parquet_cache) and os.path.exists(systems_cache) and os.path.exists(report_cache):
            track_df = pd.read_parquet(parquet_cache)
            with open(systems_cache, "r", encoding="utf-8") as f:
                systems = json.load(f)
            with open(report_cache, "r", encoding="utf-8") as f:
                report = json.load(f)
            return {
                "validation_report": report,
                "track_dataframe": track_df,
                "systems": systems,
            }

        res = cls.ingest_workbook(filepath, start_year=start_year, end_year=end_year)
        res["track_dataframe"].to_parquet(parquet_cache, index=False, engine="pyarrow")
        with open(systems_cache, "w", encoding="utf-8") as f:
            json.dump(res["systems"], f, indent=2)
        with open(report_cache, "w", encoding="utf-8") as f:
            json.dump(res["validation_report"], f, indent=2)

        return res

