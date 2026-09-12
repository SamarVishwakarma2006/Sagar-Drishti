"""
Sagar-Drishti V2.1 — Atmospheric Feature Engineering Pipeline.
Extracts strictly CAUSAL daily atmospheric features from 6-hourly ERA5 pressure-level data
across the six operational research sub-basins of the North Indian Ocean.

STRICT OPERATIONAL SAFETY & SCIENTIFIC INTEGRITY:
- Candidate / Research pipeline ONLY.
- Does NOT modify or retrain production v1.1.0 or shadow v2_10yr models.
- Does NOT alter the 101-feature ocean ML contract or operational alert policy v2.0.0.
- Does NOT merge into features_10yr.parquet.
- Strictly causal: date D features use only observations up to 18:00 UTC of date D.
- True area-weighted vorticity exceedance fraction using cos(latitude).
- Base features and intentional lag-boundary nulls audited and reported separately.
- Incomplete days identified explicitly in quality metadata.
"""

import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xarray as xr

# Ensure backend root is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.era5_spatial import (
    ERA5SpatialFeatureExtractor,
    SUB_BASIN_DEFINITIONS,
)

DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "era5")
DEFAULT_OUTPUT_PARQUET = os.path.join(DEFAULT_DATA_DIR, "features_atmosphere_10yr_daily.parquet")
DEFAULT_METADATA_JSON = os.path.join(DEFAULT_DATA_DIR, "features_atmosphere_10yr_daily_metadata.json")


def verify_era5_source_coverage(data_dir: str) -> Dict[str, Any]:
    """
    Independently inspects all yearly ERA5 NetCDF files.
    Verifies chronological regularity, date bounds, and 6-hourly sample completeness.
    Does NOT assume theoretical 3,652 dates without empirical verification.
    """
    pattern = os.path.join(data_dir, "era5_pressure_*.nc")
    yearly_files = sorted(glob.glob(pattern))
    if not yearly_files:
        raise FileNotFoundError(f"No yearly ERA5 files found matching pattern: {pattern}")

    print(f"[*] Auditing {len(yearly_files)} source NetCDF files...")
    all_timestamps = []
    file_metadata = []

    for fpath in yearly_files:
        fname = os.path.basename(fpath)
        ds = xr.open_dataset(fpath)
        t_vals = pd.to_datetime(ds.valid_time.values)
        all_timestamps.extend(t_vals)
        file_metadata.append({
            "filename": fname,
            "timesteps": len(t_vals),
            "start": str(t_vals.min()),
            "end": str(t_vals.max()),
        })
        ds.close()

    time_idx = pd.DatetimeIndex(all_timestamps)
    is_monotonic = bool(time_idx.is_monotonic_increasing)
    unique_timesteps = len(time_idx.unique())
    total_timesteps = len(time_idx)

    if unique_timesteps != total_timesteps:
        raise ValueError(f"Duplicate timestamps detected in source files: {total_timesteps - unique_timesteps} duplicates")

    df_time = pd.DataFrame({"datetime": time_idx, "date": time_idx.date, "hour": time_idx.hour})
    daily_stats = df_time.groupby("date").agg(
        n_samples=("hour", "count"),
        hours=("hour", lambda h: sorted(list(h))),
        is_complete=("hour", lambda h: set(h) == {0, 6, 12, 18})
    )

    total_unique_dates = len(daily_stats)
    complete_days = int(daily_stats["is_complete"].sum())
    incomplete_days = total_unique_dates - complete_days
    first_date_str = str(daily_stats.index.min())
    last_date_str = str(daily_stats.index.max())

    audit_result = {
        "yearly_files_count": len(yearly_files),
        "files": file_metadata,
        "total_timesteps": total_timesteps,
        "is_monotonic_increasing": is_monotonic,
        "first_date": first_date_str,
        "last_date": last_date_str,
        "total_unique_dates": total_unique_dates,
        "complete_days": complete_days,
        "incomplete_days": incomplete_days,
        "incomplete_date_list": [str(d) for d in daily_stats[~daily_stats["is_complete"]].index],
    }

    print(f"[+] Source Audit: {total_unique_dates} dates ({first_date_str} to {last_date_str}), "
          f"{complete_days} complete, {incomplete_days} incomplete.")
    return audit_result


def extract_base_daily_features(
    data_dir: str,
    vorticity_threshold_scaled: float = 5.0,
) -> pd.DataFrame:
    """
    Phase 1: Extracts daily base atmospheric features from 6-hourly NetCDF files across all 6 basins.
    Processes year-by-year to maintain low memory footprint (<500 MB RAM).
    Strictly causal: Date D features use only timesteps belonging to Date D (00, 06, 12, 18 UTC).
    """
    raw_cache_path = os.path.join(data_dir, "features_atmosphere_base_raw.parquet")
    if os.path.exists(raw_cache_path):
        print(f"[*] Checking for existing raw base cache: {raw_cache_path}...", flush=True)
        df_cached = pd.read_parquet(raw_cache_path)
        if len(df_cached) == 21912:
            print(f"[+] Found valid base raw cache with {len(df_cached)} records. Using cache!", flush=True)
            return df_cached
        else:
            print(f"[!] Base raw cache incomplete ({len(df_cached)} != 21912). Re-extracting...", flush=True)

    yearly_files = sorted(glob.glob(os.path.join(data_dir, "era5_pressure_*.nc")))
    records: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx, fpath in enumerate(yearly_files, 1):
        fname = os.path.basename(fpath)
        t_year_start = time.time()
        print(f"[{idx}/{len(yearly_files)}] Processing {fname}...", flush=True)
        ds = xr.open_dataset(fpath)

        times = pd.to_datetime(ds.valid_time.values)
        unique_dates = sorted(list(set(times.date)))

        # Pre-slice and load all 6 basins into memory for this year
        # Drastically eliminates 1,000+ unbuffered disk seeks per file
        basin_datasets = {}
        for basin_key in SUB_BASIN_DEFINITIONS.keys():
            basin_datasets[basin_key] = ERA5SpatialFeatureExtractor.slice_sub_basin(ds, basin_key).load()

        for d in unique_dates:
            date_str = str(d)
            day_indices = np.where(times.date == d)[0]

            for basin_key, b_ds in basin_datasets.items():
                daily_b_ds = b_ds.isel(valid_time=day_indices)
                basin_record = ERA5SpatialFeatureExtractor.extract_daily_basin_features(
                    daily_ds=daily_b_ds,
                    basin_key=basin_key,
                    date_str=date_str,
                    vorticity_threshold_scaled=vorticity_threshold_scaled,
                )
                records.append(basin_record)

        del basin_datasets
        ds.close()
        t_year = time.time() - t_year_start
        print(f"    Completed {fname} ({len(unique_dates)} dates, {len(unique_dates)*len(SUB_BASIN_DEFINITIONS)} records) in {t_year:.1f}s", flush=True)

    df_base = pd.DataFrame.from_records(records)
    elapsed = time.time() - t_start
    print(f"[+] Phase 1 Complete: {len(df_base)} base records extracted in {elapsed:.1f}s.", flush=True)

    # Save raw base features cache to disk
    df_base.to_parquet(raw_cache_path, index=False)
    print(f"[+] Saved base raw cache to {raw_cache_path}.", flush=True)
    return df_base


def compute_causal_temporal_features(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Computes backward-looking causal temporal lookback features partitioned strictly per basin.
    
    Causal Contract:
    Feature for Date D must never access observations from Date D+1 or later.
    Only historical observations [D-k, ..., D] are used.

    Temporal features computed per basin:
    - 24h change: val(D) - val(D-1)
    - 48h change: val(D) - val(D-2)
    - 72h change: val(D) - val(D-3)
    - 3-day causal rolling mean: mean(D-2, D-1, D)
    - 3-day causal rolling max: max(D-2, D-1, D)
    - 3-day causal rolling min: min(D-2, D-1, D)
    - 3-day linear trend slope: (val(D) - val(D-2)) / 2.0
    """
    print("[*] Phase 2: Computing causal temporal lookback features per basin...", flush=True)
    df = df_base.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["basin", "date_dt"]).reset_index(drop=True)

    grouped = df.groupby("basin", group_keys=False)

    # 1. 850 hPa Relative Vorticity Lookbacks
    df["max_vorticity_change_24h"] = grouped["max_vorticity"].diff(1)
    df["max_vorticity_change_48h"] = grouped["max_vorticity"].diff(2)
    df["max_vorticity_change_72h"] = grouped["max_vorticity"].diff(3)
    df["max_vorticity_rolling_3d_mean"] = grouped["max_vorticity"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["max_vorticity_rolling_3d_max"] = grouped["max_vorticity"].transform(lambda s: s.rolling(3, min_periods=1).max())
    df["max_vorticity_rolling_3d_trend"] = df["max_vorticity_change_48h"] / 2.0

    df["mean_vorticity_change_24h"] = grouped["mean_vorticity"].diff(1)
    df["mean_vorticity_change_48h"] = grouped["mean_vorticity"].diff(2)
    df["mean_vorticity_rolling_3d_mean"] = grouped["mean_vorticity"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    df["weighted_exceedance_change_24h"] = grouped["weighted_exceedance_fraction"].diff(1)
    df["weighted_exceedance_change_48h"] = grouped["weighted_exceedance_fraction"].diff(2)
    df["weighted_exceedance_rolling_3d_mean"] = grouped["weighted_exceedance_fraction"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # 2. 200–850 hPa Vertical Wind Shear Lookbacks
    df["min_vws_change_24h"] = grouped["min_vws"].diff(1)
    df["min_vws_change_48h"] = grouped["min_vws"].diff(2)
    df["min_vws_change_72h"] = grouped["min_vws"].diff(3)
    df["min_vws_rolling_3d_mean"] = grouped["min_vws"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["min_vws_rolling_3d_min"] = grouped["min_vws"].transform(lambda s: s.rolling(3, min_periods=1).min())
    df["min_vws_rolling_3d_trend"] = df["min_vws_change_48h"] / 2.0

    df["mean_vws_change_24h"] = grouped["mean_vws"].diff(1)
    df["mean_vws_change_48h"] = grouped["mean_vws"].diff(2)
    df["mean_vws_rolling_3d_mean"] = grouped["mean_vws"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # 3. Mid-Tropospheric Relative Humidity Lookbacks (700 and 500 hPa)
    df["mean_rh700_change_24h"] = grouped["mean_rh700"].diff(1)
    df["mean_rh700_change_48h"] = grouped["mean_rh700"].diff(2)
    df["mean_rh700_rolling_3d_mean"] = grouped["mean_rh700"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    df["mean_rh500_change_24h"] = grouped["mean_rh500"].diff(1)
    df["mean_rh500_change_48h"] = grouped["mean_rh500"].diff(2)
    df["mean_rh500_rolling_3d_mean"] = grouped["mean_rh500"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # Explicit ML Warmup Validity Flag:
    # First 3 dates (2016-06-24, 2016-06-25, 2016-06-26) have boundary nulls for 72h deltas.
    # Dates >= 2016-06-27 have full 72h / 3-day causal history across all features.
    df["is_ml_warmup_complete"] = df["date"] >= "2016-06-27"

    # Drop temporary datetime column and ensure deterministic sorting
    df = df.drop(columns=["date_dt"])
    df = df.sort_values(by=["date", "basin"]).reset_index(drop=True)
    print(f"[+] Phase 2 Complete: {len(df.columns)} total columns.", flush=True)
    return df


def audit_and_save_dataset(
    df: pd.DataFrame,
    source_audit: Dict[str, Any],
    vorticity_threshold_scaled: float,
    output_parquet: str,
    metadata_json: str,
) -> Dict[str, Any]:
    """
    Audits numerical integrity, separates base NaNs from lag boundary nulls,
    verifies duplicate counts, saves parquet, and outputs machine-readable metadata.
    """
    print("[*] Auditing numerical integrity and separating base NaNs from boundary nulls...")

    base_feature_cols = [
        "max_vorticity", "mean_vorticity", "std_vorticity",
        "weighted_exceedance_fraction", "gridcell_exceedance_fraction",
        "latitude_of_max_vorticity", "longitude_of_max_vorticity",
        "min_vws", "mean_vws", "max_vws",
        "mean_rh700", "max_rh700", "mean_rh500", "max_rh500"
    ]

    quality_cols = [
        "number_of_valid_6h_samples", "expected_6h_samples",
        "completeness_fraction", "is_complete", "is_ml_warmup_complete"
    ]

    temporal_lookback_cols = [c for c in df.columns if c not in ["date", "basin"] + base_feature_cols + quality_cols]

    # 1. Base Feature Audit (Complete days must have 0 NaNs)
    complete_df = df[df["is_complete"] == True]
    base_nans = int(complete_df[base_feature_cols].isna().sum().sum())
    base_infs = int(np.isinf(complete_df[base_feature_cols].select_dtypes(include=np.number)).sum().sum())

    if base_nans > 0:
        raise ValueError(f"Violation: {base_nans} NaNs detected in base atmospheric features on complete days!")
    if base_infs > 0:
        raise ValueError(f"Violation: {base_infs} Inf values detected in base atmospheric features!")

    # 2. Intentional Temporal Boundary Null Audit
    boundary_nans_by_col = {col: int(df[col].isna().sum()) for col in temporal_lookback_cols}
    total_boundary_nans = sum(boundary_nans_by_col.values())

    # 3. Duplicate row audit
    duplicate_rows = int(df.duplicated(subset=["date", "basin"]).sum())
    if duplicate_rows > 0:
        raise ValueError(f"Violation: {duplicate_rows} duplicate (date, basin) rows found in dataset!")

    # 4. Save to parquet
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False, engine="pyarrow")
    file_size_bytes = os.path.getsize(output_parquet)

    # 5. Build Comprehensive Metadata
    feature_summary = {}
    for col in base_feature_cols + temporal_lookback_cols:
        series = df[col].dropna()
        feature_summary[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "null_count": int(df[col].isna().sum()),
        }

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "ERA5 Atmospheric Daily Causal Features (V2.1 Candidate)",
        "output_file": output_parquet,
        "file_size_bytes": file_size_bytes,
        "total_rows": len(df),
        "total_dates": int(df["date"].nunique()),
        "basins": sorted(list(df["basin"].unique())),
        "basin_count": int(df["basin"].nunique()),
        "start_date": str(df["date"].min()),
        "end_date": str(df["date"].max()),
        "vorticity_threshold_scaled": vorticity_threshold_scaled,
        "is_threshold_scientifically_frozen": False,
        "threshold_policy_note": "Initial engineering threshold (5.0 x 10^-5 s^-1); configurable research parameter.",
        "source_audit": source_audit,
        "data_integrity": {
            "base_feature_nans_on_complete_days": base_nans,
            "base_feature_infs": base_infs,
            "duplicate_date_basin_pairs": duplicate_rows,
            "total_boundary_lookback_nans": total_boundary_nans,
            "boundary_nans_by_feature": boundary_nans_by_col,
            "complete_days_count": int(df["is_complete"].sum() // df["basin"].nunique()),
            "incomplete_days_count": int((~df["is_complete"]).sum() // df["basin"].nunique()),
            "ml_ready_dates_count": int(df["is_ml_warmup_complete"].sum() // df["basin"].nunique()),
            "warmup_dates_count": int((~df["is_ml_warmup_complete"]).sum() // df["basin"].nunique()),
            "warmup_policy_note": "First 3 dates (2016-06-24 to 2016-06-26, 18 rows) lack full 72h history. Retained in candidate dataset with is_ml_warmup_complete=False; filterable for ML training.",
        },
        "columns": {
            "index_columns": ["date", "basin"],
            "base_feature_columns": base_feature_cols,
            "temporal_lookback_columns": temporal_lookback_cols,
            "quality_metadata_columns": quality_cols,
        },
        "feature_summary": feature_summary,
    }

    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Successfully saved candidate dataset: {output_parquet} ({file_size_bytes:,} bytes)")
    print(f"[+] Metadata written: {metadata_json}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build ERA5 V2.1 Daily Causal Atmospheric Features.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing yearly ERA5 files.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PARQUET, help="Output parquet path.")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_JSON, help="Output metadata JSON path.")
    parser.add_argument("--threshold", type=float, default=5.0, help="Vorticity threshold scaled (default: 5.0).")
    args = parser.parse_args()

    print("================================================================================")
    print("Sagar-Drishti V2.1 — Building Daily Causal Atmospheric Feature Dataset")
    print("================================================================================")
    print(f"Data directory: {args.data_dir}")
    print(f"Output parquet: {args.output}")
    print(f"Vorticity threshold: {args.threshold} (scaled 10^-5 s^-1)")

    # 1. Independent Source NetCDF Audit (Verification before assertion)
    source_audit = verify_era5_source_coverage(args.data_dir)

    # 2. Phase 1: Base Daily Features Extraction
    df_base = extract_base_daily_features(
        data_dir=args.data_dir,
        vorticity_threshold_scaled=args.threshold,
    )

    # 3. Phase 2: Backward-Looking Causal Temporal Features
    df_features = compute_causal_temporal_features(df_base)

    # 4. Integrity Verification & Artifact Storage
    metadata = audit_and_save_dataset(
        df=df_features,
        source_audit=source_audit,
        vorticity_threshold_scaled=args.threshold,
        output_parquet=args.output,
        metadata_json=args.metadata,
    )

    print("================================================================================")
    print("[+] Atmospheric feature generation completed successfully.")
    print("================================================================================")


if __name__ == "__main__":
    main()
