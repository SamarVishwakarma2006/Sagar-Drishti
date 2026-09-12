"""
Scientific Dataset & Model Validation Audit Script.
Traces every step from Copernicus NetCDF / test fixture through Phase 2 features,
Phase 3 labels, filtering, ML dataset, splits, calibration, explainability, and early warning semantics.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.copernicus_service import CopernicusService
from app.services.historical_engine import HistoricalFeatureEngine
from app.services.event_store import HistoricalEventStore, AUTHORITATIVE_COVERAGE, SEED_HISTORICAL_EVENTS
from app.services.event_matcher import SpatialTemporalMatcher
from app.services.dataset_auditor import DatasetAuditor
from app.services.ml_dataset_builder import MLDatasetBuilder
from app.services.ml_trainer import MLTrainer
import joblib


def run_audit():
    print("================================================================================")
    print("SCIENTIFIC DATASET AND MODEL VALIDATION AUDIT")
    print("================================================================================")

    # --------------------------------------------------------------------------
    # 1. NETCDF / COPERNICUS RAW DATA AUDIT
    # --------------------------------------------------------------------------
    print("\n--- 1. FULL DATA COVERAGE PIPELINE TRACE ---")
    data_dir = CopernicusService.get_data_dir()
    print(f"Copernicus data dir: {data_dir}")

    # Check for actual nc files
    search_dirs = [
        data_dir,
        os.path.join(os.path.dirname(data_dir), "copernicus"),
        os.path.join(os.path.dirname(data_dir), "sample_data"),
        os.path.abspath(os.path.join("data", "copernicus")),
        os.path.abspath(os.path.join("sample_data")),
    ]
    
    found_nc = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".nc"):
                    found_nc.append(os.path.join(d, f))

    print(f"NetCDF files discovered: {found_nc}")
    for active_nc in found_nc:
        print(f"\nNetCDF file details: {active_nc}")
        try:
            ds = xr.open_dataset(active_nc)
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Data variables: {list(ds.data_vars.keys())}")
            if "time" in ds.coords:
                times = pd.to_datetime(ds.coords["time"].values)
                print(f"  Start date: {times.min().strftime('%Y-%m-%d')}")
                print(f"  End date:   {times.max().strftime('%Y-%m-%d')}")
                print(f"  Total time steps (days): {len(times)}")
            if "latitude" in ds.coords:
                lats = ds.coords["latitude"].values
                print(f"  Latitude range: {lats.min()} to {lats.max()} (len {len(lats)})")
            if "longitude" in ds.coords:
                lons = ds.coords["longitude"].values
                print(f"  Longitude range: {lons.min()} to {lons.max()} (len {len(lons)})")
        except Exception as e:
            print(f"  Error reading {active_nc}: {e}")

    # --------------------------------------------------------------------------
    # 2. PHASE 2 FEATURE TABLE & PHASE 3 LABELS
    # --------------------------------------------------------------------------
    labeled_paths = [
        os.path.join(data_dir, "labeled_features.parquet"),
        os.path.join(os.path.dirname(data_dir), "historical", "labeled_features.parquet"),
        os.path.abspath(os.path.join("data", "historical", "labeled_features.parquet")),
        os.path.abspath(os.path.join("..", "data", "historical", "labeled_features.parquet")),
    ]
    labeled_parquet_path = next((p for p in labeled_paths if os.path.exists(p)), None)
    print(f"\nResolved labeled parquet: {labeled_parquet_path}")

    if labeled_parquet_path is None:
        print("ERROR: labeled_features.parquet not found!")
        return

    labeled_df = pd.read_parquet(labeled_parquet_path)
    print(f"Phase 3 Labeled DataFrame shape: {labeled_df.shape}")
    print(f"  Unique dates: {labeled_df['date'].nunique()}")
    print(f"  Date range: {labeled_df['date'].min()} to {labeled_df['date'].max()}")
    print(f"  Label status counts:\n{labeled_df['label_status'].value_counts().to_string()}")

    # --------------------------------------------------------------------------
    # 3. FILTERING & ML DATASET
    # --------------------------------------------------------------------------
    ml_paths = [
        os.path.join(data_dir, "ml_features.parquet"),
        os.path.join(os.path.dirname(data_dir), "historical", "ml_features.parquet"),
        os.path.abspath(os.path.join("data", "historical", "ml_features.parquet")),
        os.path.abspath(os.path.join("..", "data", "historical", "ml_features.parquet")),
    ]
    ml_parquet_path = next((p for p in ml_paths if os.path.exists(p)), None)
    print(f"\nResolved ML parquet: {ml_parquet_path}")

    if ml_parquet_path is None:
        print("ERROR: ml_features.parquet not found!")
        return

    ml_df = pd.read_parquet(ml_parquet_path)
    print(f"Phase 4 ML DataFrame shape: {ml_df.shape}")
    print(f"  Unique dates: {ml_df['date'].nunique()}")
    print(f"  Date range: {ml_df['date'].min()} to {ml_df['date'].max()}")
    print(f"  Split counts:\n{ml_df['split'].value_counts().to_string()}")
    print(f"  Label status counts:\n{ml_df['label_status'].value_counts().to_string()}")
    print(f"  Active event counts in ML df: {ml_df['event_active'].value_counts().to_dict()}")
    print(f"  Early warning (within 3d) counts in ML df: {ml_df['event_within_3d'].value_counts().to_dict()}")

    # Chronological breakdown
    for sp in ["train", "val", "test"]:
        sub = ml_df[ml_df["split"] == sp]
        dates = sub["date"].sort_values()
        print(f"\nSplit '{sp}':")
        print(f"  Observations: {len(sub)}")
        print(f"  Date range: {dates.iloc[0]} to {dates.iloc[-1]} ({dates.nunique()} unique dates)")
        print(f"  event_active (pos/neg): {sub['event_active'].sum()} / {len(sub) - sub['event_active'].sum()}")
        print(f"  event_within_1d (pos):  {sub['event_within_1d'].sum()}")
        print(f"  event_within_2d (pos):  {sub['event_within_2d'].sum()}")
        print(f"  event_within_3d (pos):  {sub['event_within_3d'].sum()}")
        print(f"  event_type distribution: {sub['event_type'].value_counts().to_dict()}")
        print(f"  event_ids: {sub[sub['event_id'].notnull()]['event_id'].unique().tolist()}")

    # --------------------------------------------------------------------------
    # 4. CALIBRATION & BRIER SCORE AUDIT
    # --------------------------------------------------------------------------
    print("\n--- 4. CALIBRATION & BRIER SCORE DEEP-DIVE ---")
    val_sub = ml_df[ml_df["split"] == "val"].copy()
    train_sub = ml_df[ml_df["split"] == "train"].copy()
    test_sub = ml_df[ml_df["split"] == "test"].copy()

    # Load artifacts and verify calibration behavior
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    for task in ["event_active", "event_within_1d", "event_within_2d", "event_within_3d"]:
        mod_path = os.path.join(artifacts_dir, f"model_{task}.joblib")
        if os.path.exists(mod_path):
            artifact = joblib.load(mod_path)
            cal_info = artifact.get("calibration", {})
            metrics = artifact.get("validation_metrics", {})
            print(f"\nTask: {task}")
            print(f"  Best model: {artifact.get('best_model_name')}")
            print(f"  Is calibrated: {artifact.get('is_calibrated')}")
            print(f"  Calibration info: {cal_info}")
            print(f"  Validation metrics: F1={metrics.get('f1')}, PR-AUC={metrics.get('pr_auc')}, ROC-AUC={metrics.get('roc_auc')}, CM={metrics.get('confusion_matrix')}")

    # --------------------------------------------------------------------------
    # 5. PERMUTATION IMPORTANCE STABILITY AUDIT
    # --------------------------------------------------------------------------
    active_path = os.path.join(artifacts_dir, "risk_model_0d.joblib")
    if not os.path.exists(active_path):
        active_path = os.path.join(artifacts_dir, "model_event_active.joblib")
    active_artifact = joblib.load(active_path)
    expl = active_artifact.get("explainability", {})
    print("Reported Ocean Variable Importance for event_active:")
    print(json.dumps(expl.get("ocean_variable_importance"), indent=2))
    print("Reported Top 10 Features:")
    for tf in expl.get("top_features", []):
        print(f"  {tf['feature']}: {tf['importance']}")

    # Stability test: Run permutation importance across different random seeds
    from sklearn.inspection import permutation_importance
    model_obj = active_artifact.get("model")
    feature_cols = active_artifact.get("feature_columns")
    X_val = val_sub[feature_cols].values
    y_val = val_sub["event_active"].values

    print("\nRunning Permutation Importance Stability Check (seeds 1, 42, 123, 999)...")
    for seed in [1, 42, 123, 999]:
        try:
            perm = permutation_importance(model_obj, X_val, y_val, n_repeats=5, random_state=seed)
            nz = [(feature_cols[i], perm.importances_mean[i]) for i in range(len(feature_cols)) if perm.importances_mean[i] > 1e-5]
            nz.sort(key=lambda x: x[1], reverse=True)
            print(f"  Seed {seed} non-zero features ({len(nz)} features): {nz[:3]}")
        except Exception as e:
            print(f"  Seed {seed} error: {e}")

    # --------------------------------------------------------------------------
    # 6. HISTORICAL EVENT COVERAGE & AUTHORITATIVE CATALOG AUDIT
    # --------------------------------------------------------------------------
    print("\n--- 6. HISTORICAL EVENT COVERAGE (2024 - 2026) ---")
    events = HistoricalEventStore.get_all_events()
    print(f"Total registered authoritative events: {len(events)}")
    for ev in events:
        print(f"  {ev.event_id} | {ev.name} | {ev.start_date} -> {ev.end_date} | {ev.event_type.value} | {ev.affected_region}")

    print("\nCoverage domain configuration:")
    print(json.dumps(AUTHORITATIVE_COVERAGE, indent=2))

    # --------------------------------------------------------------------------
    # 7. EARLY WARNING SEMANTICS AUDIT
    # --------------------------------------------------------------------------
    print("\n--- 7. EARLY WARNING TARGET SEMANTICS AUDIT ---")
    violations = 0
    mono_violations = 0
    for idx, row in ml_df.iterrows():
        is_act = row["event_active"]
        w1 = row["event_within_1d"]
        w2 = row["event_within_2d"]
        w3 = row["event_within_3d"]
        # Invariant: If is_active == 1, then event_within_1d/2d/3d MUST be 0
        if is_act == 1 and (w1 == 1 or w2 == 1 or w3 == 1):
            print(f"LEAKAGE VIOLATION at row {idx} date {row['date']}: active=1 but early warning is positive!")
            violations += 1
        # Invariant: w1 <= w2 <= w3 (cumulative future window)
        if not (w1 <= w2 <= w3):
            print(f"MONOTONICITY VIOLATION at row {idx} date {row['date']}: w1={w1}, w2={w2}, w3={w3}")
            mono_violations += 1

    if violations == 0 and mono_violations == 0:
        print("Early Warning Semantics Audit Complete: Verified that active events NEVER trigger early warning flags (violations=0), and cumulative monotonicity (w1 <= w2 <= w3) holds 100% strictly.")

    # --------------------------------------------------------------------------
    # 8. DATA SPLIT AUDIT
    # --------------------------------------------------------------------------
    print("\n--- 8. DATA SPLIT TEMPORAL BOUNDARY AUDIT ---")
    train_dates = set(train_sub["date"])
    val_dates = set(val_sub["date"])
    test_dates = set(test_sub["date"])

    train_val_overlap = train_dates.intersection(val_dates)
    val_test_overlap = val_dates.intersection(test_dates)
    train_test_overlap = train_dates.intersection(test_dates)

    print(f"Train/Val date overlap: {train_val_overlap} (count: {len(train_val_overlap)})")
    print(f"Val/Test date overlap: {val_test_overlap} (count: {len(val_test_overlap)})")
    print(f"Train/Test date overlap: {train_test_overlap} (count: {len(train_test_overlap)})")

    print(f"Train max date: {train_sub['date'].max()} < Val min date: {val_sub['date'].min()} -> {train_sub['date'].max() < val_sub['date'].min()}")
    print(f"Val max date: {val_sub['date'].max()} < Test min date: {test_sub['date'].min()} -> {val_sub['date'].max() < test_sub['date'].min()}")

    # Event leakage across splits
    train_events = set(train_sub[train_sub["event_id"].notnull()]["event_id"])
    val_events = set(val_sub[val_sub["event_id"].notnull()]["event_id"])
    test_events = set(test_sub[test_sub["event_id"].notnull()]["event_id"])

    print(f"Train distinct events: {train_events}")
    print(f"Val distinct events: {val_events}")
    print(f"Test distinct events: {test_events}")
    print(f"Events appearing in both Train and Val: {train_events.intersection(val_events)}")
    print(f"Events appearing in both Val and Test: {val_events.intersection(test_events)}")
    print(f"Events appearing in both Train and Test: {train_events.intersection(test_events)}")


if __name__ == "__main__":
    run_audit()
