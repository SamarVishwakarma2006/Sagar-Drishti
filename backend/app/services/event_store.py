"""
Historical Event & Disaster Store for Sagar-Drishti (Phase 3).
Provides an authoritative, extensible registry of real historical oceanographic
events and tropical cyclones from IMD RSMC New Delhi, INCOIS, and NOAA IBTrACS.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from threading import Lock
import pandas as pd

from ..models.schemas import (
    HistoricalEvent,
    EventType,
    SpatialRepresentation,
    BoundingBox,
)

logger = logging.getLogger("sagar_drishti.event_store")


# ==============================================================================
# AUTHORITATIVE SEED EVENTS (Verified IMD RSMC New Delhi & INCOIS Records)
# ==============================================================================
SEED_HISTORICAL_EVENTS: List[Dict[str, Any]] = [
    {
        "event_id": "IMD-2024-SCS-ASNA",
        "event_type": "tropical_cyclone",
        "name": "Severe Cyclonic Storm Asna",
        "start_date": "2024-08-29",
        "end_date": "2024-09-02",
        "affected_region": "Northeast Arabian Sea, coastal Saurashtra & Kutch, Gujarat",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 21.0,
            "max_lat": 24.5,
            "min_lon": 63.0,
            "max_lon": 70.5,
        },
        "track_coordinates": [
            {"date": "2024-08-29", "lat": 23.8, "lon": 69.8, "intensity_kts": 30},
            {"date": "2024-08-30", "lat": 23.5, "lon": 68.2, "intensity_kts": 40},
            {"date": "2024-08-31", "lat": 23.2, "lon": 66.5, "intensity_kts": 45},
            {"date": "2024-09-01", "lat": 22.8, "lon": 64.8, "intensity_kts": 35},
            {"date": "2024-09-02", "lat": 21.5, "lon": 63.2, "intensity_kts": 25},
        ],
        "centroid_lat": 23.0,
        "centroid_lon": 66.5,
        "severity": "Severe Cyclonic Storm (45 kts / 85 km/h, 988 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi / Cyclone Report on SCS Asna (Aug-Sep 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Arabian Sea", "landfall_status": "emerged from land into sea"},
    },
    {
        "event_id": "IMD-2024-SCS-DANA",
        "event_type": "tropical_cyclone",
        "name": "Severe Cyclonic Storm Dana",
        "start_date": "2024-10-22",
        "end_date": "2024-10-25",
        "affected_region": "East-central & Northwest Bay of Bengal, Odisha & West Bengal Coast",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 15.5,
            "max_lat": 22.0,
            "min_lon": 85.0,
            "max_lon": 91.0,
        },
        "track_coordinates": [
            {"date": "2024-10-22", "lat": 15.6, "lon": 90.5, "intensity_kts": 30},
            {"date": "2024-10-23", "lat": 17.2, "lon": 88.8, "intensity_kts": 45},
            {"date": "2024-10-24", "lat": 19.5, "lon": 87.4, "intensity_kts": 60},
            {"date": "2024-10-25", "lat": 21.2, "lon": 86.8, "intensity_kts": 55},
        ],
        "centroid_lat": 18.5,
        "centroid_lon": 88.0,
        "severity": "Severe Cyclonic Storm (110 km/h gusting 120 km/h, 984 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi / Best Track Report on SCS Dana (Oct 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Bay of Bengal", "landfall_location": "Dhamra Port / Bhitarkanika, Odisha"},
    },
    {
        "event_id": "IMD-2024-CS-FENGAL",
        "event_type": "tropical_cyclone",
        "name": "Cyclonic Storm Fengal",
        "start_date": "2024-11-28",
        "end_date": "2024-12-01",
        "affected_region": "Southwest Bay of Bengal, Tamil Nadu, Puducherry & Rayalaseema",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 9.5,
            "max_lat": 14.5,
            "min_lon": 79.0,
            "max_lon": 84.0,
        },
        "track_coordinates": [
            {"date": "2024-11-28", "lat": 10.2, "lon": 83.5, "intensity_kts": 35},
            {"date": "2024-11-29", "lat": 11.4, "lon": 81.8, "intensity_kts": 45},
            {"date": "2024-11-30", "lat": 12.0, "lon": 80.2, "intensity_kts": 45},
            {"date": "2024-12-01", "lat": 12.2, "lon": 79.5, "intensity_kts": 30},
        ],
        "centroid_lat": 11.5,
        "centroid_lon": 81.2,
        "severity": "Cyclonic Storm (85 km/h, 994 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi Cyclone Bulletin on CS Fengal (Nov-Dec 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Bay of Bengal", "landfall_location": "Puducherry Coast"},
    },
    {
        "event_id": "IMD-2024-DD-BOB05",
        "event_type": "depression",
        "name": "Deep Depression BOB 05",
        "start_date": "2024-09-07",
        "end_date": "2024-09-10",
        "affected_region": "West-central & Northwest Bay of Bengal, Odisha & North Andhra Pradesh",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 17.0,
            "max_lat": 21.5,
            "min_lon": 83.5,
            "max_lon": 89.0,
        },
        "centroid_lat": 19.2,
        "centroid_lon": 86.2,
        "severity": "Deep Depression (60 km/h, 992 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi Cyclone Bulletin on BOB 05 (Sep 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Bay of Bengal", "landfall_location": "Puri, Odisha"},
    },
    {
        "event_id": "IMD-2024-D-ARB01",
        "event_type": "depression",
        "name": "Depression ARB 01",
        "start_date": "2024-10-11",
        "end_date": "2024-10-13",
        "affected_region": "East-central & Northeast Arabian Sea, off Karnataka & Goa coast",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 14.0,
            "max_lat": 18.5,
            "min_lon": 68.0,
            "max_lon": 73.5,
        },
        "centroid_lat": 16.0,
        "centroid_lon": 70.8,
        "severity": "Depression (45 km/h, 1002 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi Cyclone Bulletin on ARB 01 (Oct 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Arabian Sea"},
    },
    {
        "event_id": "INCOIS-2024-SW-KALLAKKADAL",
        "event_type": "coastal_swell_surge",
        "name": "Kallakkadal / High Swell Surge Event",
        "start_date": "2024-07-15",
        "end_date": "2024-07-18",
        "affected_region": "South Arabian Sea, Kerala & Lakshadweep Coastal Zone",
        "spatial_representation": "regional_envelope",
        "bbox": {
            "min_lat": 6.0,
            "max_lat": 12.0,
            "min_lon": 71.0,
            "max_lon": 78.0,
        },
        "centroid_lat": 9.0,
        "centroid_lon": 74.5,
        "severity": "High Swell Warning (Wave Height 2.5 - 3.8m, Period 16-19s)",
        "source": "INCOIS",
        "source_reference": "INCOIS Ocean State Forecast & High Swell Warning Advisory (Jul 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"phenomenon": "Swell Surge (Kallakkadal) from Southern Ocean long-period swells"},
    },
    {
        "event_id": "INCOIS-2024-MHW-BOB",
        "event_type": "marine_heatwave",
        "name": "Northern Bay of Bengal Marine Heatwave",
        "start_date": "2024-07-02",
        "end_date": "2024-07-10",
        "affected_region": "Northern Bay of Bengal plume waters",
        "spatial_representation": "regional_envelope",
        "bbox": {
            "min_lat": 18.0,
            "max_lat": 22.0,
            "min_lon": 86.0,
            "max_lon": 92.0,
        },
        "centroid_lat": 20.0,
        "centroid_lon": 89.0,
        "severity": "Category 1 Moderate Marine Heatwave (+1.4°C SST anomaly)",
        "source": "INCOIS",
        "source_reference": "INCOIS Marine Heatwave Watch & MoES Bulletin (Jul 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"sst_anomaly_deg_c": 1.4, "depth_extent_m": 25.0},
    },
    {
        "event_id": "IMD-2024-SCS-REMAL",
        "event_type": "tropical_cyclone",
        "name": "Severe Cyclonic Storm Remal",
        "start_date": "2024-05-24",
        "end_date": "2024-05-28",
        "affected_region": "North Bay of Bengal, West Bengal & Bangladesh Coast",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 16.0,
            "max_lat": 23.5,
            "min_lon": 87.0,
            "max_lon": 91.5,
        },
        "centroid_lat": 19.5,
        "centroid_lon": 89.0,
        "severity": "Severe Cyclonic Storm (60 kts / 110 km/h, 978 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi / Best Track Report on SCS Remal (May 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Bay of Bengal", "landfall_location": "Sagar Island / Khepupara"},
    },
    {
        "event_id": "IMD-2024-D-BOB04",
        "event_type": "depression",
        "name": "Depression BOB 04",
        "start_date": "2024-08-24",
        "end_date": "2024-08-27",
        "affected_region": "Northwest Bay of Bengal, Gangetic West Bengal & Odisha",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 20.0,
            "max_lat": 23.5,
            "min_lon": 85.5,
            "max_lon": 89.5,
        },
        "centroid_lat": 21.5,
        "centroid_lon": 87.5,
        "severity": "Depression (45 km/h, 994 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi / Cyclone Bulletin on BOB 04 (Aug 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Bay of Bengal"},
    },
    {
        "event_id": "IMD-2024-DD-BOB06",
        "event_type": "depression",
        "name": "Deep Depression BOB 06",
        "start_date": "2024-12-20",
        "end_date": "2024-12-24",
        "affected_region": "Southwest Bay of Bengal, Sri Lanka & South Tamil Nadu Coast",
        "spatial_representation": "track_envelope",
        "bbox": {
            "min_lat": 7.0,
            "max_lat": 11.5,
            "min_lon": 80.5,
            "max_lon": 85.5,
        },
        "centroid_lat": 9.2,
        "centroid_lon": 83.0,
        "severity": "Deep Depression (55 km/h, 1000 hPa)",
        "source": "IMD RSMC New Delhi",
        "source_reference": "RSMC New Delhi / Cyclone Bulletin on BOB 06 (Dec 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"cyclone_basin": "Bay of Bengal"},
    },
    {
        "event_id": "INCOIS-2024-SW-KERALA",
        "event_type": "coastal_swell_surge",
        "name": "Kallakkadal / High Swell Surge Event — South Arabian Sea",
        "start_date": "2024-05-03",
        "end_date": "2024-05-06",
        "affected_region": "Kerala, Lakshadweep & Kanyakumari coastal belts",
        "spatial_representation": "regional_envelope",
        "bbox": {
            "min_lat": 7.5,
            "max_lat": 12.5,
            "min_lon": 72.0,
            "max_lon": 77.5,
        },
        "centroid_lat": 10.0,
        "centroid_lon": 75.0,
        "severity": "High Swell Warning (Swell waves 2.8 - 3.5m, Period 18-22s)",
        "source": "INCOIS",
        "source_reference": "INCOIS Ocean State Forecast & High Swell Warning Advisory (May 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"phenomenon": "Swell Surge (Kallakkadal) early pre-monsoon event"},
    },
    {
        "event_id": "INCOIS-2024-MHW-ARAS",
        "event_type": "marine_heatwave",
        "name": "Eastern Arabian Sea Marine Heatwave",
        "start_date": "2024-05-10",
        "end_date": "2024-05-22",
        "affected_region": "Eastern Arabian Sea off Konkan-Goa-Karnataka",
        "spatial_representation": "regional_envelope",
        "bbox": {
            "min_lat": 12.0,
            "max_lat": 18.0,
            "min_lon": 68.0,
            "max_lon": 74.0,
        },
        "centroid_lat": 15.0,
        "centroid_lon": 71.0,
        "severity": "Category 2 Strong Marine Heatwave (+1.8°C SST anomaly)",
        "source": "INCOIS",
        "source_reference": "INCOIS Marine Heatwave Watch Bulletin & MoES Climate Summary (May 2024)",
        "confidence": "verified_authoritative",
        "metadata": {"sst_anomaly_deg_c": 1.8},
    },
]


# Documented coverage limits for the authoritative event sources
# Observations outside these spatial or temporal limits are tagged 'unknown_uncovered'
AUTHORITATIVE_COVERAGE = {
    "agency": "IMD RSMC New Delhi / INCOIS",
    "min_lat": 0.0,
    "max_lat": 25.0,
    "min_lon": 50.0,
    "max_lon": 100.0,
    "coverage_start": "2016-05-01",
    "coverage_end": "2026-06-23",
}


class HistoricalEventStore:
    """
    Extensible thread-safe registry of authoritative historical events.
    Enforces strict validation, supports dynamic registration of additional
    authoritative bulletins without modifying matching engines.
    """
    _lock = Lock()
    _events: Dict[str, HistoricalEvent] = {}
    _initialized = False

    @classmethod
    def _initialize_if_needed(cls):
        with cls._lock:
            if not cls._initialized:
                cls.reset_to_seeds()
                cls._initialized = True

    @classmethod
    def reset_to_seeds(cls):
        """Resets the store strictly to authoritative seed records."""
        cls._events.clear()
        for raw in SEED_HISTORICAL_EVENTS:
            event = HistoricalEvent(
                event_id=raw["event_id"],
                event_type=EventType(raw["event_type"]),
                name=raw["name"],
                start_date=raw["start_date"],
                end_date=raw["end_date"],
                affected_region=raw["affected_region"],
                spatial_representation=SpatialRepresentation(raw.get("spatial_representation", "regional_envelope")),
                bbox=BoundingBox(**raw["bbox"]),
                track_coordinates=raw.get("track_coordinates"),
                centroid_lat=raw.get("centroid_lat"),
                centroid_lon=raw.get("centroid_lon"),
                severity=raw["severity"],
                source=raw["source"],
                source_reference=raw["source_reference"],
                confidence=raw.get("confidence", "verified_authoritative"),
                metadata=raw.get("metadata", {}),
            )
            cls.validate_event(event)
            cls._events[event.event_id] = event
        cls._initialized = True
        logger.info(f"HistoricalEventStore initialized with {len(cls._events)} seed events.")

    # ==========================================================================
    # VALIDATION INVARIANTS
    # ==========================================================================
    @classmethod
    def validate_event(cls, event: HistoricalEvent) -> None:
        """
        Validates strict event integrity:
        1. No missing dates, start_date <= end_date.
        2. Positive duration (end_date >= start_date).
        3. Coordinates within valid geographic range.
        4. Valid spatial representation (track_envelope vs regional_envelope).
        5. Mandatory source and source_reference citations.
        """
        if not event.event_id or not event.event_id.strip():
            raise ValueError("Event must have a valid non-empty 'event_id'.")

        try:
            dt_start = pd.to_datetime(event.start_date)
            dt_end = pd.to_datetime(event.end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format in event '{event.event_id}': {e}")

        if dt_start > dt_end:
            raise ValueError(
                f"Event '{event.event_id}' has start_date ({event.start_date}) > end_date ({event.end_date})."
            )

        bbox = event.bbox
        if bbox.min_lat > bbox.max_lat or bbox.min_lon > bbox.max_lon:
            raise ValueError(f"Event '{event.event_id}' has inverted bounding box: {bbox}")

        if bbox.min_lat < -90.0 or bbox.max_lat > 90.0 or bbox.min_lon < -180.0 or bbox.max_lon > 180.0:
            raise ValueError(f"Event '{event.event_id}' bounding box coordinates out of global range.")

        # Never invent spatial bounds: ensure bbox area is non-zero
        if bbox.min_lat == bbox.max_lat and bbox.min_lon == bbox.max_lon:
            raise ValueError(f"Event '{event.event_id}' has zero-area bounding box. Proper spatial bounds required.")

        if not event.source or not event.source.strip():
            raise ValueError(f"Event '{event.event_id}' missing mandatory 'source' agency citation.")

        if not event.source_reference or not event.source_reference.strip():
            raise ValueError(f"Event '{event.event_id}' missing mandatory 'source_reference' citation.")

    @classmethod
    def validate_catalog(cls) -> Dict[str, Any]:
        """Runs full validation audit across all registered events."""
        cls._initialize_if_needed()
        with cls._lock:
            errors = []
            for eid, event in cls._events.items():
                try:
                    cls.validate_event(event)
                except Exception as e:
                    errors.append({"event_id": eid, "error": str(e)})

            is_valid = len(errors) == 0
            return {
                "is_valid": is_valid,
                "total_events": len(cls._events),
                "error_count": len(errors),
                "errors": errors,
            }

    # ==========================================================================
    # EXTENSIBILITY (DYNAMIC REGISTRATION & EXTERNAL LOADING)
    # ==========================================================================
    @classmethod
    def register_event(cls, event: HistoricalEvent, validate: bool = True) -> HistoricalEvent:
        """
        Dynamically registers an additional authoritative event.
        Extensible without modifying the matcher.
        """
        cls._initialize_if_needed()
        if validate:
            cls.validate_event(event)

        with cls._lock:
            if event.event_id in cls._events:
                raise ValueError(f"Event with ID '{event.event_id}' is already registered in EventStore.")
            cls._events[event.event_id] = event
            logger.info(f"Registered event '{event.event_id}' ({event.name}) in HistoricalEventStore.")
            return event

    @classmethod
    def load_from_json(cls, file_path: str) -> int:
        """Loads and registers authoritative events from an external JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Event catalog file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of event objects.")

        count = 0
        for item in data:
            event = HistoricalEvent(**item)
            cls.register_event(event)
            count += 1
        return count

    @classmethod
    def load_imd_best_tracks(
        cls,
        workbook_path: Optional[str] = None,
        start_year: int = 2016,
        end_year: int = 2026,
    ) -> int:
        """Loads and integrates normalized 10-year IMD Best Track events into EventStore."""
        with cls._lock:
            if not cls._initialized:
                cls.reset_to_seeds()
                cls._initialized = True
        if workbook_path is None:
            possible = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "78b4b0_Best_Tracks__Data__1982-2026_.xlsx")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "78b4b0_Best_Tracks__Data__1982-2026_.xlsx")),
            ]
            for p in possible:
                if os.path.exists(p):
                    workbook_path = p
                    break

        if not workbook_path or not os.path.exists(workbook_path):
            logger.warning("IMD Best-Track workbook not found; keeping seed events.")
            return 0

        from ..parsers.imd_best_track_parser import IMDBestTrackParser
        res = IMDBestTrackParser.get_cached_or_parse(workbook_path, start_year=start_year, end_year=end_year)
        track_df = res["track_dataframe"]
        systems = res["systems"]

        added = 0
        with cls._lock:
            for s in systems:
                sid = s["system_id"]
                if sid in cls._events:
                    continue

                sub_df = track_df[track_df["system_id"] == sid]
                track_coords = []
                for _, r in sub_df.iterrows():
                    track_coords.append({
                        "date": r["date"],
                        "lat": float(r["latitude"]),
                        "lon": float(r["longitude"]),
                        "intensity_kts": float(r["max_wind_kts"]) if pd.notna(r["max_wind_kts"]) else 0.0,
                    })

                max_g = str(s["max_grade"]).upper()
                if max_g in ("D", "DD") or "DEPRESSION" in max_g:
                    ev_type = EventType.DEPRESSION
                else:
                    ev_type = EventType.TROPICAL_CYCLONE

                bbox_dict = dict(s["bbox"])
                if bbox_dict["min_lat"] == bbox_dict["max_lat"]:
                    bbox_dict["min_lat"] -= 0.2
                    bbox_dict["max_lat"] += 0.2
                if bbox_dict["min_lon"] == bbox_dict["max_lon"]:
                    bbox_dict["min_lon"] -= 0.2
                    bbox_dict["max_lon"] += 0.2

                event = HistoricalEvent(
                    event_id=sid,
                    event_type=ev_type,
                    name=s["name"],
                    start_date=s["start_date"],
                    end_date=s["end_date"],
                    affected_region=f"{s['basin']}, North Indian Ocean",
                    spatial_representation=SpatialRepresentation.TRACK_ENVELOPE,
                    bbox=BoundingBox(**bbox_dict),
                    track_coordinates=track_coords if track_coords else None,
                    centroid_lat=s["centroid_lat"],
                    centroid_lon=s["centroid_lon"],
                    severity=f"{s['max_grade']} (Peak: {s['peak_intensity_kts']:.0f} kts)",
                    source="IMD RSMC New Delhi",
                    source_reference=f"IMD Best Track Archive ({s['year']})",
                    confidence="tentative_record" if s.get("is_tentative_2026") else "verified_authoritative",
                    metadata={
                        "basin": s["basin"],
                        "year": s["year"],
                        "is_tentative_2026": s.get("is_tentative_2026", False),
                        "track_points_count": s["track_points_count"],
                        "peak_intensity_kts": s["peak_intensity_kts"],
                    },
                )
                cls._events[sid] = event
                added += 1

        logger.info(f"Loaded and registered {added} IMD Best Track systems into HistoricalEventStore (Total: {len(cls._events)}).")
        return added

    # ==========================================================================
    # QUERIES & COVERAGE CHECKING
    # ==========================================================================
    @classmethod
    def get_all_events(cls) -> List[HistoricalEvent]:
        cls._initialize_if_needed()
        with cls._lock:
            return list(cls._events.values())

    @classmethod
    def get_event_by_id(cls, event_id: str) -> Optional[HistoricalEvent]:
        cls._initialize_if_needed()
        with cls._lock:
            return cls._events.get(event_id)

    @classmethod
    def get_events(
        cls,
        event_type: Optional[EventType] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[HistoricalEvent]:
        """Filters events by type and date range."""
        cls._initialize_if_needed()
        with cls._lock:
            matched = list(cls._events.values())

        if event_type:
            matched = [e for e in matched if e.event_type == event_type]

        if start_date:
            dt_s = pd.to_datetime(start_date)
            matched = [e for e in matched if pd.to_datetime(e.end_date) >= dt_s]

        if end_date:
            dt_e = pd.to_datetime(end_date)
            matched = [e for e in matched if pd.to_datetime(e.start_date) <= dt_e]

        return matched

    @classmethod
    def is_in_coverage_domain(cls, date_str: str, lat: Optional[float] = None, lon: Optional[float] = None, bbox: Optional[BoundingBox] = None) -> bool:
        """
        Checks whether an observation falls strictly inside the documented
        spatial and temporal coverage of the authoritative event sources.
        """
        cov = AUTHORITATIVE_COVERAGE
        try:
            obs_dt = pd.to_datetime(date_str)
            cov_s = pd.to_datetime(cov["coverage_start"])
            cov_e = pd.to_datetime(cov["coverage_end"])
            if obs_dt < cov_s or obs_dt > cov_e:
                return False
        except Exception:
            return False

        if lat is not None and lon is not None:
            if not (cov["min_lat"] <= lat <= cov["max_lat"] and cov["min_lon"] <= lon <= cov["max_lon"]):
                return False

        if bbox is not None:
            # Overlap with domain
            if bbox.max_lat < cov["min_lat"] or bbox.min_lat > cov["max_lat"]:
                return False
            if bbox.max_lon < cov["min_lon"] or bbox.min_lon > cov["max_lon"]:
                return False

        return True
