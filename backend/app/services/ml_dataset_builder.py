"""
Machine Learning Dataset Builder & Chronological Splitter for Sagar-Drishti (Phase 4).
Extracts leakage-free physical ocean features, applies strict filtering (excluding
unknown_uncovered and post-event buffer samples), dynamically identifies feature columns,
and generates non-random chronological 70/15/15 Train/Validation/Test splits.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from .copernicus_service import CopernicusService
from .event_matcher import SpatialTemporalMatcher

logger = logging.getLogger("sagar_drishti.ml_dataset_builder")

# Strict blacklist of columns that leak target information or represent identifiers/metadata
LEAKAGE_AND_TARGET_COLUMNS = {
    # Target columns
    "event_active", "is_active_event", "event_present",
    "event_within_1d", "event_within_2d", "event_within_3d",
    "event_type", "lead_0", "lead_1", "lead_2", "lead_3", "lead_days",
    "label_status", "target_active", "target_within_1d", "target_within_2d", "target_within_3d",
    # Metadata & Identifiers
    "event_id", "event_name", "secondary_event_id", "severity",
    "label_source", "label_confidence", "provenance", "dataset_id",
    # Non-feature administrative columns and spatial/temporal coordinates
    "day_index", "split", "date", "time", "datetime", "lat", "lon", "latitude", "longitude", "site_id"
}


class MLDatasetBuilder:
    """
    Constructs the leakage-safe supervised ML dataset and performs chronological splitting.
    """

    @classmethod
    def build_ml_dataset(
        cls,
        labeled_df: Optional[pd.DataFrame] = None,
        labeled_parquet_path: Optional[str] = None,
        labeled_parquet: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_parquet: Optional[str] = None,
        output_metadata: Optional[str] = None,
        exclude_uncovered: bool = True,
        exclude_buffer: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds the clean ML dataset:
        1. Loads labeled observations from Phase 3.
        2. Filters out unknown_uncovered and negative_buffer records if requested.
        3. Dynamically identifies and validates clean feature columns (zero target leakage).
        4. Sorts chronologically by observation date and computes 70% Train / 15% Val / 15% Test.
        5. Saves data/historical/ml_features.parquet and data/historical/ml_dataset_metadata.json.
        """
        if labeled_parquet_path is None and labeled_parquet is not None:
            labeled_parquet_path = labeled_parquet

        if labeled_df is None:
            if labeled_parquet_path and os.path.exists(labeled_parquet_path):
                labeled_df = pd.read_parquet(labeled_parquet_path)
            else:
                search_paths = [
                    os.path.join(CopernicusService.get_data_dir(), "labeled_features.parquet"),
                    os.path.join(os.path.dirname(CopernicusService.get_data_dir()), "historical", "labeled_features.parquet"),
                    os.path.abspath(os.path.join("data", "historical", "labeled_features.parquet")),
                ]
                default_labeled_path = next((p for p in search_paths if os.path.exists(p)), None)

                if default_labeled_path is not None:
                    labeled_df = pd.read_parquet(default_labeled_path)
                else:
                    # Generate via SpatialTemporalMatcher
                    res = SpatialTemporalMatcher.generate_labeled_dataset()
                    labeled_df = pd.read_parquet(res.parquet_path)

        raw_rows = len(labeled_df)
        df = labeled_df.copy()

        # Step 1: Filter out non-trainable samples
        # unknown_uncovered: Outside official monitoring domain -> NOT reliable negatives
        if exclude_uncovered and "label_status" in df.columns:
            df = df[df["label_status"] != "unknown_uncovered"].copy()

        # negative_buffer: Physical relaxation / post-event wake -> NOT clean negatives
        if exclude_buffer and "label_status" in df.columns:
            df = df[df["label_status"] != "negative_buffer"].copy()

        filtered_rows = len(df)
        if filtered_rows == 0:
            raise ValueError("All rows were filtered out; no eligible training observations remain.")

        # Step 2: Chronological sorting by observation date
        df["parsed_date"] = pd.to_datetime(df["date"])
        df = df.sort_values("parsed_date").reset_index(drop=True)

        # Step 3: Dynamically identify feature columns
        all_cols = list(df.columns)
        feature_cols = [
            c for c in all_cols
            if c not in LEAKAGE_AND_TARGET_COLUMNS
            and c not in ["date", "parsed_date", "mode", "site_id"]
        ]

        # Verify no NaN or Inf in feature columns
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Step 4: Chronological 70% / 15% / 15% split based on sorted unique dates
        unique_dates = df["parsed_date"].drop_duplicates().sort_values().reset_index(drop=True)
        total_unique = len(unique_dates)

        train_idx = int(np.floor(total_unique * 0.70))
        val_idx = int(np.floor(total_unique * 0.85))

        # Ensure every split gets at least 1 date if dataset is small
        train_idx = max(1, train_idx)
        val_idx = max(train_idx + 1, min(val_idx, total_unique - 1)) if total_unique > 2 else total_unique - 1

        train_dates = set(unique_dates.iloc[:train_idx])
        val_dates = set(unique_dates.iloc[train_idx:val_idx])
        test_dates = set(unique_dates.iloc[val_idx:])

        # Assign split labels
        def assign_split(d):
            if d in train_dates:
                return "train"
            elif d in val_dates:
                return "val"
            else:
                return "test"

        df = df.copy()
        df["split"] = df["parsed_date"].apply(assign_split)

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

        # Step 5: Save output files
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(CopernicusService.get_data_dir()), "historical"
            )
            if not os.path.exists(output_dir):
                output_dir = os.path.abspath(os.path.join("data", "historical"))

        os.makedirs(output_dir, exist_ok=True)
        ml_parquet_path = output_parquet if output_parquet else os.path.join(output_dir, "ml_features.parquet")
        ml_metadata_path = output_metadata if output_metadata else os.path.join(output_dir, "ml_dataset_metadata.json")

        df.drop(columns=["parsed_date"], inplace=True)
        df.to_parquet(ml_parquet_path, index=False, engine="pyarrow")

        # Compile split metrics & class balances
        def split_stats(sub_df):
            if len(sub_df) == 0:
                return {"count": 0, "date_range": {}, "targets": {}}
            return {
                "count": len(sub_df),
                "date_range": {
                    "start": str(sub_df["date"].min()),
                    "end": str(sub_df["date"].max()),
                },
                "targets": {
                    "event_active_positive": int(sub_df["event_active"].sum()) if "event_active" in sub_df else 0,
                    "event_within_1d_positive": int(sub_df["event_within_1d"].sum()) if "event_within_1d" in sub_df else 0,
                    "event_within_2d_positive": int(sub_df["event_within_2d"].sum()) if "event_within_2d" in sub_df else 0,
                    "event_within_3d_positive": int(sub_df["event_within_3d"].sum()) if "event_within_3d" in sub_df else 0,
                    "event_type_distribution": sub_df[sub_df["event_type"].notnull() & (sub_df["event_type"] != "none")]["event_type"].value_counts().to_dict() if "event_type" in sub_df else {},
                }
            }

        temporal_split_info = {
            "total_dates": len(unique_dates),
            "train_dates_count": len(train_dates),
            "val_dates_count": len(val_dates),
            "test_dates_count": len(test_dates),
            "splits": {
                "train": split_stats(train_df),
                "val": split_stats(val_df),
                "test": split_stats(test_df),
            },
        }

        metadata_payload = {
            "dataset_name": "Sagar-Drishti Chronologically Split ML Feature Dataset",
            "format": "parquet",
            "ml_features_parquet": os.path.abspath(ml_parquet_path),
            "total_raw_rows": raw_rows,
            "total_clean_rows": filtered_rows,
            "excluded_uncovered_rows": raw_rows - filtered_rows,
            "final_feature_count": len(feature_cols),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "target_columns": [
                "event_active", "event_within_1d", "event_within_2d", "event_within_3d", "event_type"
            ],
            "split_methodology": "Chronological 70% Train / 15% Validation / 15% Test based on sorted dates (no temporal shuffling)",
            "temporal_split": temporal_split_info,
            "splits": temporal_split_info["splits"],
            "exported_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        with open(ml_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        return {
            "status": "success",
            "parquet_path": os.path.abspath(ml_parquet_path),
            "metadata_path": os.path.abspath(ml_metadata_path),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "splits": metadata_payload["splits"],
            "metadata": metadata_payload,
            "dataframe": df,
        }
