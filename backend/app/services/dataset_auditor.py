"""
Dataset Readiness Auditor for Sagar-Drishti (Phase 3.5).
Generates an exhaustive dataset quality report covering class balance,
temporal/spatial coverage, missing values, duplicates, and before/after event expansion audits.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .event_store import HistoricalEventStore, AUTHORITATIVE_COVERAGE

logger = logging.getLogger("sagar_drishti.dataset_auditor")


class DatasetAuditor:
    """
    Computes scientific data readiness metrics across ocean features and labels.
    """

    @classmethod
    def audit_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data quality audit across an ocean observation/labeled DataFrame.
        """
        total_rows = len(df)
        if total_rows == 0:
            return {"error": "DataFrame is empty, cannot perform audit."}

        # Date parsing & sorting
        df = df.copy()
        if "date" in df.columns:
            df["parsed_date"] = pd.to_datetime(df["date"])
            df = df.sort_values("parsed_date").reset_index(drop=True)
            min_date = df["parsed_date"].min().strftime("%Y-%m-%d")
            max_date = df["parsed_date"].max().strftime("%Y-%m-%d")
            # Temporal gaps: check if consecutive daily dates exist
            date_diffs = (df["parsed_date"] - df["parsed_date"].shift(1)).dt.days
            temporal_gaps = int((date_diffs > 1).sum())
        else:
            min_date = "unknown"
            max_date = "unknown"
            temporal_gaps = 0

        # Feature vs Label column identification
        label_cols = {
            "is_active_event", "event_active", "event_present",
            "lead_0", "lead_1", "lead_2", "lead_3", "lead_days",
            "event_within_1d", "event_within_2d", "event_within_3d",
            "label_status", "event_type", "event_id", "event_name",
            "severity", "secondary_event_id", "label_source", "label_confidence"
        }
        meta_cols = {"date", "parsed_date", "day_index", "mode", "lat", "lon", "site_id"}

        feature_columns = [
            c for c in df.columns
            if c not in label_cols and c not in meta_cols and not c.startswith("target_")
        ]

        # Missing values & duplicates
        missing_counts = df[feature_columns].isnull().sum().to_dict()
        total_missing = int(sum(missing_counts.values()))
        duplicate_rows = int(df.duplicated(subset=["date", "lat", "lon"] if "lat" in df.columns and "lon" in df.columns else ["date"]).sum())

        # Label status distributions
        label_status_counts = (
            df["label_status"].value_counts().to_dict()
            if "label_status" in df.columns
            else {}
        )

        pos_count = int(df["event_present"].sum()) if "event_present" in df.columns else (
            int(df["event_active"].sum()) if "event_active" in df.columns else 0
        )
        neg_clean = int(label_status_counts.get("negative_clean", 0))
        neg_buffer = int(label_status_counts.get("negative_buffer", 0))
        uncovered = int(label_status_counts.get("unknown_uncovered", 0))

        # Event type breakdowns
        rows_by_type = (
            df[df["event_type"].notnull() & (df["event_type"] != "none")]["event_type"]
            .value_counts()
            .to_dict()
            if "event_type" in df.columns
            else {}
        )

        # Event ID breakdowns
        rows_by_id = (
            df[df["event_id"].notnull()]["event_id"].value_counts().to_dict()
            if "event_id" in df.columns
            else {}
        )

        # Year/Month distribution
        rows_by_year_month = {}
        if "parsed_date" in df.columns:
            df["ym"] = df["parsed_date"].dt.to_period("M").astype(str)
            rows_by_year_month = df["ym"].value_counts().sort_index().to_dict()

        # Region / Site distribution
        rows_by_region = {}
        if "site_id" in df.columns:
            rows_by_region = df["site_id"].fillna("point_coordinate").value_counts().to_dict()
        elif "mode" in df.columns:
            rows_by_region = df["mode"].value_counts().to_dict()

        # Class imbalance calculation
        imbalance_ratio = round(neg_clean / pos_count, 2) if pos_count > 0 else 0.0

        # Spatial gaps
        spatial_gaps = 0
        if "lat" in df.columns and "lon" in df.columns:
            # Check how many points are outside official coverage domain
            cov = AUTHORITATIVE_COVERAGE
            out_lat = (df["lat"] < cov["min_lat"]) | (df["lat"] > cov["max_lat"])
            out_lon = (df["lon"] < cov["min_lon"]) | (df["lon"] > cov["max_lon"])
            spatial_gaps = int((out_lat | out_lon).sum())

        audit_report = {
            "total_observations": total_rows,
            "total_feature_columns": len(feature_columns),
            "feature_column_names": feature_columns,
            "date_range": {"start": min_date, "end": max_date},
            "temporal_gaps_count": temporal_gaps,
            "spatial_out_of_domain_count": spatial_gaps,
            "missing_values_count": total_missing,
            "duplicate_rows_count": duplicate_rows,
            "label_summary": {
                "positive_event_rows": pos_count,
                "negative_clean_rows": neg_clean,
                "negative_buffer_rows": neg_buffer,
                "unknown_uncovered_rows": uncovered,
                "imbalance_ratio_clean_negative_to_positive": imbalance_ratio,
            },
            "rows_by_event_type": rows_by_type,
            "rows_by_event_id": rows_by_id,
            "rows_by_year_month": rows_by_year_month,
            "rows_by_region_or_site": rows_by_region,
            "event_coverage_percentage": round((pos_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0,
            "label_coverage_percentage": round(((total_rows - uncovered) / total_rows) * 100.0, 2) if total_rows > 0 else 0.0,
        }

        return audit_report

    @classmethod
    def compare_event_expansion(
        cls,
        events_before: List[str],
        events_after: List[str],
    ) -> Dict[str, Any]:
        """
        Generates before/after event coverage comparison report.
        """
        return {
            "before_expansion_event_count": len(events_before),
            "after_expansion_event_count": len(events_after),
            "events_added_count": len(events_after) - len(events_before),
            "events_before": events_before,
            "events_after": events_after,
            "newly_added_events": [e for e in events_after if e not in events_before],
        }
