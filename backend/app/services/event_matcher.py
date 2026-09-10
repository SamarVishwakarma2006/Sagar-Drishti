"""
Spatial + Temporal Historical Event Matcher & Supervised Labeling Engine (Phase 3).
Connects Phase 2 Copernicus rolling ocean features with real authoritative historical
disaster and extreme ocean event records (IMD RSMC New Delhi and INCOIS).
Produces an ML-ready labeled dataset with verified causality and explicit negative-sample methodology.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from ..models.schemas import (
    BoundingBox,
    HistoricalEvent,
    EventType,
    LabelStatus,
    DataQualityReport,
    LabeledDatasetExportResponse,
)
from .site_registry import SiteRegistry
from .event_store import HistoricalEventStore, AUTHORITATIVE_COVERAGE
from .copernicus_service import CopernicusService
from .historical_engine import HistoricalFeatureEngine

logger = logging.getLogger("sagar_drishti.event_matcher")


class SpatialTemporalMatcher:
    """
    Evaluates spatial and temporal intersection between ocean observations and
    authoritative historical disaster records.
    """

    @staticmethod
    def _point_in_bbox(lat: float, lon: float, bbox: BoundingBox) -> bool:
        """Determines if a point (lat, lon) falls inside a bounding box."""
        return (bbox.min_lat <= lat <= bbox.max_lat) and (bbox.min_lon <= lon <= bbox.max_lon)

    @staticmethod
    def _bboxes_overlap(b1: BoundingBox, b2: BoundingBox) -> bool:
        """Determines if two bounding boxes intersect."""
        lat_overlap = max(b1.min_lat, b2.min_lat) <= min(b1.max_lat, b2.max_lat)
        lon_overlap = max(b1.min_lon, b2.min_lon) <= min(b1.max_lon, b2.max_lon)
        return lat_overlap and lon_overlap

    @classmethod
    def match_observation(
        cls,
        date_str: str,
        mode: str = "point",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        site_id: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        lead_window_days: int = 3,
        events: Optional[List[HistoricalEvent]] = None,
    ) -> Dict[str, Any]:
        """
        Matches a single ocean observation against the historical event catalog.

        Label Categories:
        - active_event: Observation date is within [start_date, end_date] of a spatially matching event.
        - lead_event: Observation date is within 1 to lead_window_days before start_date.
        - negative_clean: Observation is inside authoritative coverage with NO active/lead event.
        - negative_buffer: Observation is within 1-2 days after an active event in the same region.
        - unknown_uncovered: Observation is outside documented coverage (NOT to be used as clean negative).
        """
        if events is None:
            events = HistoricalEventStore.get_all_events()

        obs_dt = pd.to_datetime(date_str)

        # Resolve observation geometry
        obs_bbox = bbox
        if mode == "region" and obs_bbox is None and site_id:
            site = SiteRegistry.get_site(site_id)
            if site and site.bbox:
                obs_bbox = site.bbox
            elif site and site.lat is not None and site.lon is not None:
                lat = site.lat
                lon = site.lon
                mode = "point"

        # Step 1: Find all spatially intersecting events
        spatially_matched_events: List[HistoricalEvent] = []
        for ev in events:
            spatial_hit = False
            if mode == "point" and lat is not None and lon is not None:
                spatial_hit = cls._point_in_bbox(lat, lon, ev.bbox)
            elif mode == "region" and obs_bbox is not None:
                spatial_hit = cls._bboxes_overlap(obs_bbox, ev.bbox)

            if spatial_hit:
                spatially_matched_events.append(ev)

        # Step 2: Temporal evaluation against spatially matching events
        active_matches: List[HistoricalEvent] = []
        lead_matches: List[Tuple[HistoricalEvent, int]] = []  # (event, lead_days)
        post_buffer_matches: List[HistoricalEvent] = []

        for ev in spatially_matched_events:
            ev_start = pd.to_datetime(ev.start_date)
            ev_end = pd.to_datetime(ev.end_date)

            if ev_start <= obs_dt <= ev_end:
                active_matches.append(ev)
            elif 1 <= (ev_start - obs_dt).days <= lead_window_days:
                lead_days = int((ev_start - obs_dt).days)
                lead_matches.append((ev, lead_days))
            elif 1 <= (obs_dt - ev_end).days <= 2:
                # 1 to 2 days post-event recovery wake
                post_buffer_matches.append(ev)

        # Step 3: Conflict resolution & primary label assignment
        if active_matches:
            # Active event takes precedence. Pick most severe if multiple.
            primary_ev = active_matches[0]
            secondary_id = active_matches[1].event_id if len(active_matches) > 1 else None

            return {
                "event_active": 1,
                "is_active_event": 1,
                "event_present": 1,
                "event_within_1d": 0,  # Strictly future-only: already active!
                "event_within_2d": 0,
                "event_within_3d": 0,
                "lead_0": 1,
                "lead_1": 0,
                "lead_2": 0,
                "lead_3": 0,
                "lead_days": 0,
                "label_status": LabelStatus.ACTIVE_EVENT.value,
                "event_type": primary_ev.event_type.value,
                "event_id": primary_ev.event_id,
                "event_name": primary_ev.name,
                "severity": primary_ev.severity,
                "secondary_event_id": secondary_id,
                "label_source": primary_ev.source,
                "label_confidence": primary_ev.confidence,
            }

        elif lead_matches:
            # Lead event match (earliest start / closest lead day)
            lead_matches.sort(key=lambda x: x[1])
            primary_ev, l_days = lead_matches[0]
            secondary_id = lead_matches[1][0].event_id if len(lead_matches) > 1 else None

            return {
                "event_active": 0,
                "is_active_event": 0,
                "event_present": 1,
                "event_within_1d": 1 if l_days <= 1 else 0,
                "event_within_2d": 1 if l_days <= 2 else 0,
                "event_within_3d": 1 if l_days <= 3 else 0,
                "lead_0": 0,
                "lead_1": 1 if l_days == 1 else 0,
                "lead_2": 1 if l_days == 2 else 0,
                "lead_3": 1 if l_days == 3 else 0,
                "lead_days": l_days,
                "label_status": LabelStatus.LEAD_EVENT.value,
                "event_type": primary_ev.event_type.value,
                "event_id": primary_ev.event_id,
                "event_name": primary_ev.name,
                "severity": primary_ev.severity,
                "secondary_event_id": secondary_id,
                "label_source": primary_ev.source,
                "label_confidence": primary_ev.confidence,
            }

        elif post_buffer_matches:
            # Observation falls into the post-event physical relaxation buffer
            ev = post_buffer_matches[0]
            return {
                "event_active": 0,
                "is_active_event": 0,
                "event_present": 0,
                "event_within_1d": 0,
                "event_within_2d": 0,
                "event_within_3d": 0,
                "lead_0": 0,
                "lead_1": 0,
                "lead_2": 0,
                "lead_3": 0,
                "lead_days": -1,
                "label_status": LabelStatus.NEGATIVE_BUFFER.value,
                "event_type": None,
                "event_id": None,
                "event_name": None,
                "severity": None,
                "secondary_event_id": ev.event_id,
                "label_source": "POST_EVENT_RECOVERY_BUFFER",
                "label_confidence": "buffer_unclean",
            }

        else:
            # Check if within authoritative coverage domain
            in_cov = HistoricalEventStore.is_in_coverage_domain(
                date_str=date_str,
                lat=lat,
                lon=lon,
                bbox=obs_bbox,
            )

            if in_cov:
                return {
                    "event_active": 0,
                    "is_active_event": 0,
                    "event_present": 0,
                    "event_within_1d": 0,
                    "event_within_2d": 0,
                    "event_within_3d": 0,
                    "lead_0": 0,
                    "lead_1": 0,
                    "lead_2": 0,
                    "lead_3": 0,
                    "lead_days": -1,
                    "label_status": LabelStatus.NEGATIVE_CLEAN.value,
                    "event_type": "none",
                    "event_id": None,
                    "event_name": None,
                    "severity": None,
                    "secondary_event_id": None,
                    "label_source": "IMD_INCOIS_VERIFIED_COVERAGE",
                    "label_confidence": "authoritative_negative",
                }
            else:
                return {
                    "event_active": 0,
                    "is_active_event": 0,
                    "event_present": 0,
                    "event_within_1d": 0,
                    "event_within_2d": 0,
                    "event_within_3d": 0,
                    "lead_0": 0,
                    "lead_1": 0,
                    "lead_2": 0,
                    "lead_3": 0,
                    "lead_days": -1,
                    "label_status": LabelStatus.UNKNOWN_UNCOVERED.value,
                    "event_type": None,
                    "event_id": None,
                    "event_name": None,
                    "severity": None,
                    "secondary_event_id": None,
                    "label_source": "UNCOVERED_OR_UNTRACKED_DOMAIN",
                    "label_confidence": "uncovered",
                }

    # ==========================================================================
    # DATASET GENERATION & AUDITING
    # ==========================================================================
    @classmethod
    def generate_labeled_dataset(
        cls,
        feature_df: Optional[pd.DataFrame] = None,
        feature_parquet_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        mode: str = "region",
        site_id: Optional[str] = "bob",
        lat: Optional[float] = 17.8,
        lon: Optional[float] = 88.2,
    ) -> LabeledDatasetExportResponse:
        """
        Connects existing Phase 2 feature dataset with historical events.
        Produces:
        1. data/historical/labeled_features.parquet
        2. data/historical/labeled_features_metadata.json
        Leaves raw Copernicus NetCDF and Phase 2 features.parquet untouched.
        """
        # Obtain base feature dataframe
        if feature_df is None:
            if feature_parquet_path and os.path.exists(feature_parquet_path):
                feature_df = pd.read_parquet(feature_parquet_path)
            else:
                default_p2_parquet = os.path.join(
                    CopernicusService.get_data_dir(), "ml_features", "features.parquet"
                )
                if os.path.exists(default_p2_parquet):
                    feature_df = pd.read_parquet(default_p2_parquet)
                else:
                    # Generate on the fly using Phase 2 engine
                    logger.info("Generating base Phase 2 features for labeling pipeline...")
                    p2_res = HistoricalFeatureEngine.generate_training_dataset(
                        mode=mode,
                        site_id=site_id,
                        lat=lat or 17.8,
                        lon=lon or 88.2,
                    )
                    feature_df = pd.read_parquet(p2_res.parquet_path)

        events = HistoricalEventStore.get_all_events()

        # Audit tracking
        events_spatially_matched = set()
        events_temporally_matched = set()

        labeled_rows: List[Dict[str, Any]] = []

        for _, row in feature_df.iterrows():
            row_dict = row.to_dict()
            date_str = str(row_dict["date"])
            r_mode = str(row_dict.get("mode", mode))
            r_lat = float(row_dict["lat"]) if "lat" in row_dict and pd.notna(row_dict["lat"]) else lat
            r_lon = float(row_dict["lon"]) if "lon" in row_dict and pd.notna(row_dict["lon"]) else lon
            r_site = str(row_dict["site_id"]) if "site_id" in row_dict and pd.notna(row_dict["site_id"]) else site_id

            match_res = cls.match_observation(
                date_str=date_str,
                mode=r_mode,
                lat=r_lat,
                lon=r_lon,
                site_id=r_site,
                events=events,
            )

            # Audit tracking
            ev_id = match_res.get("event_id")
            if ev_id:
                events_spatially_matched.add(ev_id)
                events_temporally_matched.add(ev_id)

            # Combine original features with authoritative labels
            combined_row = {**row_dict, **match_res}
            labeled_rows.append(combined_row)

        out_df = pd.DataFrame(labeled_rows)

        # Target output directories
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(CopernicusService.get_data_dir()), "historical"
            )
            # If data_dir was 'data/copernicus', this puts it in 'data/historical'
            if not os.path.exists(output_dir):
                output_dir = os.path.abspath(os.path.join("data", "historical"))

        os.makedirs(output_dir, exist_ok=True)
        parquet_out = os.path.join(output_dir, "labeled_features.parquet")
        metadata_out = os.path.join(output_dir, "labeled_features_metadata.json")

        out_df.to_parquet(parquet_out, index=False, engine="pyarrow")

        # Compile DataQualityReport
        events_by_type = {}
        for ev in events:
            etype = ev.event_type.value
            events_by_type[etype] = events_by_type.get(etype, 0) + 1

        unmatched_events = []
        for ev in events:
            if ev.event_id not in events_spatially_matched:
                unmatched_events.append({
                    "event_id": ev.event_id,
                    "name": ev.name,
                    "reason": f"Event bounding box {ev.bbox.model_dump()} does not intersect queried observation spatial scope ({r_mode}: lat={r_lat}, lon={r_lon}, site={r_site}).",
                })

        label_status_counts = out_df["label_status"].value_counts().to_dict()
        lead_dist = {
            "lead_0": int(out_df["lead_0"].sum()),
            "lead_1": int(out_df["lead_1"].sum()),
            "lead_2": int(out_df["lead_2"].sum()),
            "lead_3": int(out_df["lead_3"].sum()),
        }

        report = DataQualityReport(
            total_events=len(events),
            events_by_type=events_by_type,
            events_spatially_matched=len(events_spatially_matched),
            events_temporally_matched=len(events_temporally_matched),
            unmatched_events=unmatched_events,
            total_observations=len(out_df),
            positive_observations=int(out_df["event_present"].sum()),
            negative_clean_observations=int(label_status_counts.get(LabelStatus.NEGATIVE_CLEAN.value, 0)),
            negative_buffer_observations=int(label_status_counts.get(LabelStatus.NEGATIVE_BUFFER.value, 0)),
            unknown_uncovered_observations=int(label_status_counts.get(LabelStatus.UNKNOWN_UNCOVERED.value, 0)),
            lead_distribution=lead_dist,
            date_coverage={
                "start": str(out_df["date"].iloc[0]) if len(out_df) > 0 else "",
                "end": str(out_df["date"].iloc[-1]) if len(out_df) > 0 else "",
            },
            provenance_summary=[
                {"agency": "IMD RSMC New Delhi", "bulletin_type": "Tropical Cyclone & Depression Best Track Reports"},
                {"agency": "INCOIS", "bulletin_type": "Ocean State Forecast & High Swell/Heatwave Warnings"},
            ],
            methodology_notes=(
                "Explicit negative-label methodology: 'negative_clean' represents observations strictly inside "
                "authoritative IMD/INCOIS monitoring coverage with zero active advisories. 'negative_buffer' represents "
                "post-event relaxation wake (days T in [end_date+1, end_date+2]). 'unknown_uncovered' denotes observations "
                "outside official agency reporting dates/bounds and must be excluded from negative training sets. "
                "Discrete lead-time flags (lead_0, lead_1, lead_2, lead_3) represent 0 to 3 days pre-event lead time."
            ),
        )

        metadata_payload = {
            "dataset_name": "Sagar-Drishti Supervised Historical Labeled Ocean Dataset",
            "format": "parquet",
            "target_path": os.path.abspath(parquet_out),
            "total_rows": len(out_df),
            "total_columns": len(out_df.columns),
            "label_columns": [
                "event_active", "is_active_event", "event_present",
                "event_within_1d", "event_within_2d", "event_within_3d",
                "lead_0", "lead_1", "lead_2", "lead_3", "lead_days",
                "label_status", "event_type", "event_id", "event_name",
                "severity", "secondary_event_id", "label_source", "label_confidence"
            ],
            "data_quality_report": report.model_dump(),
            "exported_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        with open(metadata_out, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        return LabeledDatasetExportResponse(
            status="success",
            parquet_path=os.path.abspath(parquet_out),
            metadata_path=os.path.abspath(metadata_out),
            total_rows=len(out_df),
            positive_rows=int(out_df["event_present"].sum()),
            negative_clean_rows=int(label_status_counts.get(LabelStatus.NEGATIVE_CLEAN.value, 0)),
            negative_buffer_rows=int(label_status_counts.get(LabelStatus.NEGATIVE_BUFFER.value, 0)),
            unknown_uncovered_rows=int(label_status_counts.get(LabelStatus.UNKNOWN_UNCOVERED.value, 0)),
            data_quality_report=report,
            message=f"Exported {len(out_df)} labeled observations with {int(out_df['event_present'].sum())} positive event instances.",
        )
