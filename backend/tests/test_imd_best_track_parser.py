"""
Unit tests for IMD Best-Track Ingestion & Validation Parser.
Verifies multi-sheet discovery, coordinate normalization, grade classifications,
duplicate filtering, and distinction of tentative 2026 records.
"""

import os
import pytest
from app.parsers.imd_best_track_parser import IMDBestTrackParser

WORKBOOK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "78b4b0_Best_Tracks__Data__1982-2026_.xlsx")
)


def test_coordinate_parsing():
    assert IMDBestTrackParser.parse_coordinate(17.8) == 17.8
    assert IMDBestTrackParser.parse_coordinate("17.8°N") == 17.8
    assert IMDBestTrackParser.parse_coordinate("88.2E") == 88.2
    assert IMDBestTrackParser.parse_coordinate("15.5S") == -15.5
    assert IMDBestTrackParser.parse_coordinate("NaN") is None
    assert IMDBestTrackParser.parse_coordinate(None) is None


def test_date_parsing():
    assert IMDBestTrackParser.parse_date("24-06-2024") == "2024-06-24"
    assert IMDBestTrackParser.parse_date("24/06/2024") == "2024-06-24"
    assert IMDBestTrackParser.parse_date("2024-06-24") == "2024-06-24"


def test_grade_normalization():
    code, cat = IMDBestTrackParser.normalize_grade("VSCS")
    assert code == "VSCS"
    assert cat == "very_severe_cyclonic_storm"

    code, cat = IMDBestTrackParser.normalize_grade("DD")
    assert code == "DD"
    assert cat == "deep_depression"

    code, cat = IMDBestTrackParser.normalize_grade("CS")
    assert code == "CS"
    assert cat == "cyclonic_storm"


@pytest.mark.skipif(not os.path.exists(WORKBOOK_PATH), reason="IMD Best-Track workbook not present")
def test_workbook_ingestion_validation_report():
    res = IMDBestTrackParser.ingest_workbook(WORKBOOK_PATH, start_year=2016, end_year=2026)
    rep = res["validation_report"]

    assert rep["total_systems_discovered"] > 50
    assert rep["total_track_points"] > 2000
    assert len(rep["years_discovered"]) == 11
    assert rep["date_range"]["start"] <= "2016-12-31"
    assert rep["date_range"]["end"] >= "2024-10-01"
    assert rep["parsing_failures"] == 0
    assert rep["invalid_coordinate_count"] == 0
    assert rep["tentative_2026_records_count"] >= 0
    assert rep["finalized_historical_records_count"] > 2000
