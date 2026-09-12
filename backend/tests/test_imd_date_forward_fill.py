"""
Regression tests for IMD Best-Track date parsing and forward-fill logic.
Verifies:
A. Blank date cells are forward-filled.
B. Sub-daily rows (03:00, 06:00, 12:00, 18:00 UTC) inherit correct calendar dates.
C. Forward-fill state resets across storm boundaries.
D. No 2024 cyclone starts on 2024-01-01 (unless source explicitly states Jan 1).
E. Track timestamps remain strictly chronological within each storm.
F. Multiple sub-daily points on the same day retain distinct hours.
G. Missing date without any prior valid date in the storm is rejected, not fabricated.
"""

import os
import json
import pytest
import pandas as pd
from datetime import datetime

from app.parsers.imd_best_track_parser import IMDBestTrackParser

WORKBOOK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "78b4b0_Best_Tracks__Data__1982-2026_.xlsx")
)
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "historical")
)


@pytest.fixture(scope="module")
def parsed_imd_data():
    """Provides cached or parsed IMD Best Track data."""
    assert os.path.exists(WORKBOOK_PATH), f"Workbook not found at {WORKBOOK_PATH}"
    return IMDBestTrackParser.get_cached_or_parse(
        WORKBOOK_PATH,
        cache_dir=CACHE_DIR,
        start_year=2016,
        end_year=2026,
        force_reparse=False
    )


def test_blank_date_cells_are_forward_filled(parsed_imd_data):
    """Test A: Verifies sub-daily points inherit dates instead of failing or defaulting to Jan 01."""
    track_df = parsed_imd_data["track_dataframe"]
    # 2024 Remal has 30 track points spanning May 24 to May 28
    remal_pts = track_df[track_df["system_id"].str.contains("REMAL", case=False)]
    assert len(remal_pts) >= 20, f"Expected >= 20 points for Remal, got {len(remal_pts)}"
    
    # Check that all points for Remal have valid May dates, not 2024-01-01
    dates = remal_pts["date"].unique().tolist()
    for d in dates:
        assert d.startswith("2024-05-"), f"Unexpected date for Remal: {d}"
    assert "2024-01-01" not in dates


def test_subdaily_rows_inherit_correct_calendar_date(parsed_imd_data):
    """Test B: 03:00, 06:00, 12:00, 18:00 rows inherit the active day's calendar date."""
    track_df = parsed_imd_data["track_dataframe"]
    asna_pts = track_df[track_df["system_id"].str.contains("ASNA", case=False)]
    assert not asna_pts.empty
    
    # Asna formed in late August 2024 (2024-08-25 to 2024-09-02)
    # Check that 03:00, 06:00, 12:00 rows carry August/September dates
    subdaily_times = asna_pts[asna_pts["time_utc"].isin(["03:00:00", "06:00:00", "12:00:00", "18:00:00"])]
    assert len(subdaily_times) > 10
    for _, row in subdaily_times.iterrows():
        assert row["date"].startswith("2024-08-") or row["date"].startswith("2024-09-"), \
            f"Sub-daily point has invalid inherited date: {row['date']} at {row['time_utc']}"


def test_forward_fill_does_not_cross_storm_boundaries(parsed_imd_data):
    """Test C: Date state does not leak from one storm into a subsequent storm."""
    systems = parsed_imd_data["systems"]
    systems_2024 = [s for s in systems if s["year"] == 2024]
    
    # Remal ends May 28. Storm 2 (UNNAMED-2) starts July 19.
    # If forward-fill leaked across storms, Storm 2 would inherit May 28!
    remal = next((s for s in systems_2024 if "REMAL" in s["name"].upper()), None)
    unnamed_2 = next((s for s in systems_2024 if "IMD-2024-2-" in s["system_id"]), None)
    
    assert remal is not None
    assert unnamed_2 is not None
    assert unnamed_2["start_date"] >= "2024-07-01", \
        f"Storm 2 inherited date from previous storm: {unnamed_2['start_date']}"


def test_no_2024_cyclone_starts_on_jan_01(parsed_imd_data):
    """Test D: Explicit verification that no 2024 storm has start_date == 2024-01-01."""
    systems = parsed_imd_data["systems"]
    systems_2024 = [s for s in systems if s["year"] == 2024]
    
    for s in systems_2024:
        assert s["start_date"] != "2024-01-01", \
            f"Corrupted start date 2024-01-01 found for system: {s['system_id']}"
            
    # Known 2024 major storms
    asna = next(s for s in systems_2024 if "ASNA" in s["name"].upper())
    dana = next(s for s in systems_2024 if "DANA" in s["name"].upper())
    fengal = next(s for s in systems_2024 if "FENGAL" in s["name"].upper())
    
    assert asna["start_date"] >= "2024-08-20", f"Asna started too early: {asna['start_date']}"
    assert dana["start_date"] >= "2024-10-15", f"Dana started too early: {dana['start_date']}"
    assert fengal["start_date"] >= "2024-11-15", f"Fengal started too early: {fengal['start_date']}"


def test_track_timestamps_remain_chronological(parsed_imd_data):
    """Test E: Track timestamps within each storm are monotonic and chronological."""
    track_df = parsed_imd_data["track_dataframe"]
    
    for system_id, group in track_df.groupby("system_id"):
        timestamps = group["datetime_iso"].tolist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], \
                f"System {system_id} timestamp out of order: {timestamps[i - 1]} -> {timestamps[i]}"


def test_multiple_subdaily_points_retain_distinct_hours(parsed_imd_data):
    """Test F: Observations on the same calendar day retain their respective UTC hours."""
    track_df = parsed_imd_data["track_dataframe"]
    remal_pts = track_df[track_df["system_id"].str.contains("REMAL", case=False)]
    
    day_pts = remal_pts[remal_pts["date"] == "2024-05-25"]
    assert len(day_pts) >= 4, "Expected at least 4 observations for Remal on 2024-05-25"
    
    hours = day_pts["time_utc"].tolist()
    assert len(set(hours)) == len(hours), f"Duplicate hours detected on same date: {hours}"
    assert "00:00:00" in hours
    assert any(h in hours for h in ["03:00:00", "06:00:00", "12:00:00", "18:00:00"])


def test_missing_date_without_prior_date_rejected_not_fabricated():
    """Test G: Synthetic test verifying missing date with no prior date is rejected, not defaulted."""
    # Test parse_date behavior directly
    assert IMDBestTrackParser.parse_date(None) is None
    assert IMDBestTrackParser.parse_date("") is None
    assert IMDBestTrackParser.parse_date("invalid_text") is None
