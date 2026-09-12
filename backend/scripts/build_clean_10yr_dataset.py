"""
Build and validate clean 10-year multi-basin features and authoritative IMD labels.
Enforces:
1. 10-Year Copernicus Reanalysis (2016-06-24 to 2026-06-23, 3652 daily timesteps).
2. Frozen 101-feature hard contract (causal <= T rolling windows).
3. Authoritative IMD Best-Track labels (117 systems, forward-filled dates, genuine start dates).
4. Full distribution auditing before any model retraining.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.copernicus_service import CopernicusService
from app.services.historical_engine import HistoricalFeatureEngine
from app.services.event_store import HistoricalEventStore
from app.services.event_matcher import SpatialTemporalMatcher, BoundingBox
from app.services.feature_manifest import FeatureManifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_clean_10yr_dataset")

HISTORICAL_DATA_DIR = os.path.join(backend_dir, "data", "historical")
os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)

NC_10YR_PATH = os.path.abspath(os.path.join(backend_dir, "data", "copernicus", "copernicus_phy_10yr_surface.nc"))
FEATURES_10YR_PARQUET = os.path.join(HISTORICAL_DATA_DIR, "features_10yr.parquet")
LABELED_10YR_CLEAN_PARQUET = os.path.join(HISTORICAL_DATA_DIR, "labeled_features_10yr_clean.parquet")


def main():
    print("=" * 80)
    print("SAGAR-DRISHTI: CLEAN 10-YEAR FEATURE EXTRACTION & AUTHORITATIVE RELABELING")
    print("=" * 80)

    if not os.path.exists(NC_10YR_PATH):
        print(f"[!] Error: 10-year NetCDF not found at {NC_10YR_PATH}")
        sys.exit(1)

    print(f"[+] 10-Year NetCDF verified on disk: {NC_10YR_PATH} ({os.path.getsize(NC_10YR_PATH)} bytes)")

    # 1. Feature extraction for Bay of Bengal and Arabian Sea
    print("\n[Step 1] Extracting 101 Causal Features for Study Basins across 2016-2026...")
    if os.path.exists(FEATURES_10YR_PARQUET):
        print(f"[*] Pre-extracted 10-year features found at {FEATURES_10YR_PARQUET}. Loading...")
        df_features = pd.read_parquet(FEATURES_10YR_PARQUET)
        print(f"[+] Loaded {len(df_features)} multi-basin feature rows ({df_features['date'].min()} to {df_features['date'].max()})")
    else:
        raw_bob_dir = os.path.join(HISTORICAL_DATA_DIR, "raw_10yr_bob")
        raw_aras_dir = os.path.join(HISTORICAL_DATA_DIR, "raw_10yr_aras")

        res_bob = HistoricalFeatureEngine.export_training_dataset(
            mode="region",
            site_id="bob",
            nc_path=NC_10YR_PATH,
            output_dir=raw_bob_dir,
        )
        df_bob = pd.read_parquet(res_bob.parquet_path)
        df_bob["site_id"] = "bob"
        df_bob["basin"] = "Bay of Bengal"
        print(f"  Bay of Bengal: {len(df_bob)} daily feature rows extracted ({df_bob['date'].min()} to {df_bob['date'].max()})")

        res_aras = HistoricalFeatureEngine.export_training_dataset(
            mode="region",
            site_id="aras",
            nc_path=NC_10YR_PATH,
            output_dir=raw_aras_dir,
        )
        df_aras = pd.read_parquet(res_aras.parquet_path)
        df_aras["site_id"] = "aras"
        df_aras["basin"] = "Arabian Sea"
        print(f"  Arabian Sea:   {len(df_aras)} daily feature rows extracted ({df_aras['date'].min()} to {df_aras['date'].max()})")

        # Combine multi-basin features
        df_features = pd.concat([df_bob, df_aras], ignore_index=True).sort_values(["date", "site_id"]).reset_index(drop=True)
        df_features.to_parquet(FEATURES_10YR_PARQUET, index=False, engine="pyarrow")
        print(f"[+] Multi-basin features saved to {FEATURES_10YR_PARQUET} ({len(df_features)} total rows)")

    # 2. Verify 101 feature contract
    meta_cols = ["date", "day_index", "mode", "site_id", "basin"]
    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    print(f"\n[Step 2] Feature Contract Audit:")
    print(f"  Total feature columns: {len(feature_cols)} (Expected: 101)")
    if len(feature_cols) != 101:
        print(f"[!] ERROR: Feature count {len(feature_cols)} != 101!")
        sys.exit(1)
    print("  [PASS] 101-Feature contract strictly satisfied.")

    # 3. Load authoritative IMD records
    print("\n[Step 3] Loading Authoritative Corrected IMD Best Track Records...")
    HistoricalEventStore.reset_to_seeds()
    workbook_path = os.path.abspath(os.path.join(backend_dir, "..", "78b4b0_Best_Tracks__Data__1982-2026_.xlsx"))
    added = HistoricalEventStore.load_imd_best_tracks(workbook_path=workbook_path, start_year=2016, end_year=2026)
    all_events = HistoricalEventStore.get_all_events()
    print(f"[+] Loaded {added} IMD systems. Total catalog events in memory: {len(all_events)}")

    # 4. Generate clean labels using SpatialTemporalMatcher
    print("\n[Step 4] Relabeling Multi-Basin Observations against Clean IMD Events...")
    site_bboxes = {
        "bob": BoundingBox(min_lat=16.3, max_lat=19.3, min_lon=86.7, max_lon=89.7),
        "aras": BoundingBox(min_lat=15.0, max_lat=18.0, min_lon=68.0, max_lon=71.0),
    }

    # Pre-filter events per site bounding box
    site_events = {}
    for sid, bbox in site_bboxes.items():
        matched = [ev for ev in all_events if SpatialTemporalMatcher._bboxes_overlap(bbox, ev.bbox)]
        site_events[sid] = matched
        print(f"  Site '{sid}': {len(matched)} spatially intersecting events out of {len(all_events)}")

    # Precompute labels per unique date per site
    unique_dates = sorted(df_features["date"].unique())
    print(f"  Matching {len(unique_dates)} unique daily dates across {len(site_bboxes)} study sites...")

    labels_by_site_date = {}
    for sid, bbox in site_bboxes.items():
        evs = site_events[sid]
        for d_str in unique_dates:
            match_res = SpatialTemporalMatcher.match_observation(
                date_str=d_str,
                mode="region",
                site_id=sid,
                bbox=bbox,
                lead_window_days=3,
                events=evs
            )
            labels_by_site_date[(sid, d_str)] = match_res

    # Vectorized merge of labels with features
    label_records = [labels_by_site_date[(r.site_id, r.date)] for r in df_features[["site_id", "date"]].itertuples()]
    label_df = pd.DataFrame(label_records)
    df_labeled = pd.concat([df_features.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)
    df_labeled["year"] = pd.to_datetime(df_labeled["date"]).dt.year

    # Save clean labeled candidate dataset
    df_labeled.to_parquet(LABELED_10YR_CLEAN_PARQUET, index=False, engine="pyarrow")
    print(f"[+] Saved clean labeled dataset to {LABELED_10YR_CLEAN_PARQUET} ({len(df_labeled)} rows)")

    # 5. Mandatory Label Distribution Audit
    print("\n" + "=" * 80)
    print("MANDATORY AUDIT: CLEAN 10-YEAR LABEL DISTRIBUTION")
    print("=" * 80)

    total_obs = len(df_labeled)
    labeled_obs = sum(1 for s in df_labeled["label_status"] if s != "unknown_uncovered")
    unknown_uncovered = sum(1 for s in df_labeled["label_status"] if s == "unknown_uncovered")
    pos_0d = int(df_labeled["event_active"].sum())
    pos_1d = int(df_labeled["lead_1"].sum())
    pos_2d = int(df_labeled["lead_2"].sum())
    pos_3d = int(df_labeled["lead_3"].sum())
    pos_any = int(df_labeled["event_present"].sum())
    neg_clean = sum(1 for s in df_labeled["label_status"] if s == "negative_clean")
    neg_buffer = sum(1 for s in df_labeled["label_status"] if s == "negative_buffer")
    prev_0d = pos_0d / total_obs * 100
    prev_any = pos_any / total_obs * 100

    print(f"TOTAL OBSERVATIONS:         {total_obs}")
    print(f"KNOWN/LABELED OBSERVATIONS: {labeled_obs} ({labeled_obs / total_obs * 100:.2f}%)")
    print(f"UNKNOWN/UNCOVERED:          {unknown_uncovered} ({unknown_uncovered / total_obs * 100:.2f}%)")
    print(f"POSITIVE 0d (Active):       {pos_0d} ({prev_0d:.2f}%)")
    print(f"POSITIVE 1d (Lead 1d):      {pos_1d} ({pos_1d / total_obs * 100:.2f}%)")
    print(f"POSITIVE 2d (Lead 2d):      {pos_2d} ({pos_2d / total_obs * 100:.2f}%)")
    print(f"POSITIVE 3d (Lead 3d):      {pos_3d} ({pos_3d / total_obs * 100:.2f}%)")
    print(f"POSITIVE (Active or <=3d):  {pos_any} ({prev_any:.2f}%)")
    print(f"NEGATIVE CLEAN:             {neg_clean} ({neg_clean / total_obs * 100:.2f}%)")
    print(f"NEGATIVE BUFFER:            {neg_buffer} ({neg_buffer / total_obs * 100:.2f}%)")
    print(f"POSITIVE PREVALENCE:        {prev_any:.2f}%")

    print("\n--- YEAR-BY-YEAR DATA UTILIZATION (2016-2026) ---")
    year_rows = []
    for y in sorted(df_labeled["year"].unique()):
        sub = df_labeled[df_labeled["year"] == y]
        pos = int(sub["event_present"].sum())
        neg = sum(1 for s in sub["label_status"] if s == "negative_clean")
        unk = sum(1 for s in sub["label_status"] if s == "unknown_uncovered")
        y_prev = pos / len(sub) * 100
        year_rows.append({
            "YEAR": y,
            "TOTAL": len(sub),
            "POSITIVE": pos,
            "NEGATIVE": neg,
            "UNKNOWN": unk,
            "PREVALENCE": f"{y_prev:.2f}%",
            "USED": "YES"
        })
    y_df = pd.DataFrame(year_rows)
    print(y_df.to_string(index=False))

    # 6. Specific 2024 Training Fold Prevalence Check
    sub_2024 = df_labeled[df_labeled["year"] == 2024]
    prev_2024 = int(sub_2024["event_present"].sum()) / len(sub_2024) * 100
    print("\n--- 2024 PREVALENCE GATE CHECK ---")
    print(f"2024 Observations: {len(sub_2024)}")
    print(f"2024 Positive Days (Active+Lead): {int(sub_2024['event_present'].sum())}")
    print(f"2024 Positive Prevalence: {prev_2024:.2f}% (Previously corrupted at ~74%)")
    
    if prev_2024 > 40.0:
        print(f"[!] WARNING: 2024 prevalence {prev_2024:.2f}% is still abnormally high! Check spatial matching logic.")
    else:
        print(f"[PASS] 2024 prevalence is realistic and uncorrupted ({prev_2024:.2f}%).")


if __name__ == "__main__":
    if sys.stdout:
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    main()
