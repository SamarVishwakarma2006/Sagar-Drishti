"""
Multi-Basin Dataset Expansion & Event-Aware Chronological Split (BOB + ARAS).
Executes:
1. Phase 2 rolling feature generation for site 'aras' (Arabian Sea).
2. Phase 3 event matching & supervised labeling for site 'aras'.
3. Combined Multi-Basin dataset construction (BOB + ARAS).
4. Full Audit across BOB, ARAS, and Combined datasets.
5. Scientifically valid Event-Aware Chronological Split design.
6. Temporal leakage and causality verification.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.copernicus_service import CopernicusService
from app.services.historical_engine import HistoricalFeatureEngine, SUPPORTED_CHANNELS
from app.services.event_store import HistoricalEventStore, AUTHORITATIVE_COVERAGE
from app.services.event_matcher import SpatialTemporalMatcher
from app.services.ml_dataset_builder import MLDatasetBuilder, LEAKAGE_AND_TARGET_COLUMNS
from app.services.site_registry import SiteRegistry


def run_multibasin_pipeline():
    print("=" * 85)
    print("SAGAR-DRISHTI MULTI-BASIN EXPANSION & EVENT-AWARE SPLIT DESIGN (BOB + ARAS)")
    print("=" * 85)

    data_dir = CopernicusService.get_data_dir()
    historical_dir = os.path.join(os.path.dirname(data_dir), "historical")
    os.makedirs(historical_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # TASK 1: PHASE 2 FOR ARABIAN SEA (site_id='aras')
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TASK 1: GENERATING PHASE 2 ROLLING FEATURES FOR ARABIAN SEA ('aras')")
    print("=" * 85)
    aras_site = SiteRegistry.get_site("aras")
    print(f"Site 'aras' metadata:")
    print(f"  Name:   {aras_site.name}")
    print(f"  Bbox:   lat [{aras_site.bbox.min_lat}, {aras_site.bbox.max_lat}], lon [{aras_site.bbox.min_lon}, {aras_site.bbox.max_lon}]")
    print(f"  Center: lat {aras_site.lat}, lon {aras_site.lon}")

    aras_p2_out = os.path.join(data_dir, "ml_features_aras")
    p2_aras_res = HistoricalFeatureEngine.generate_training_dataset(
        mode="region",
        site_id="aras",
        output_dir=aras_p2_out,
    )
    print(f"[+] ARAS Phase 2 export status:  {p2_aras_res.status}")
    print(f"[+] ARAS Parquet:               {p2_aras_res.parquet_path}")
    print(f"[+] Total feature rows:         {p2_aras_res.total_rows}")
    print(f"[+] Total feature columns:      {p2_aras_res.total_features}")
    print(f"[+] Date range:                 {p2_aras_res.date_range['start']} to {p2_aras_res.date_range['end']}")

    df_p2_aras = pd.read_parquet(p2_aras_res.parquet_path)
    df_p2_aras["site_id"] = "aras"
    df_p2_aras["basin"] = "Arabian Sea"

    # Also load BOB Phase 2
    bob_p2_parquet = os.path.join(data_dir, "ml_features", "features.parquet")
    if not os.path.exists(bob_p2_parquet):
        # Generate BOB if missing
        p2_bob_res = HistoricalFeatureEngine.generate_training_dataset(
            mode="region",
            site_id="bob",
            output_dir=os.path.join(data_dir, "ml_features"),
        )
        bob_p2_parquet = p2_bob_res.parquet_path

    df_p2_bob = pd.read_parquet(bob_p2_parquet)
    df_p2_bob["site_id"] = "bob"
    df_p2_bob["basin"] = "Bay of Bengal"

    # --------------------------------------------------------------------------
    # TASK 2: PHASE 3 EVENT LABELING FOR ARAS & COMBINED DATASET
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TASK 2: APPLYING PHASE 3 HISTORICAL EVENT LABELING TO ARAS & COMBINING")
    print("=" * 85)

    aras_p3_out = os.path.join(historical_dir, "aras")
    p3_aras_res = SpatialTemporalMatcher.generate_labeled_dataset(
        feature_df=df_p2_aras,
        output_dir=aras_p3_out,
        mode="region",
        site_id="aras",
    )
    df_p3_aras = pd.read_parquet(p3_aras_res.parquet_path)
    df_p3_aras["site_id"] = "aras"
    df_p3_aras["basin"] = "Arabian Sea"

    # Load BOB Phase 3
    bob_p3_parquet = os.path.join(historical_dir, "labeled_features.parquet")
    df_p3_bob = pd.read_parquet(bob_p3_parquet)
    df_p3_bob["site_id"] = "bob"
    df_p3_bob["basin"] = "Bay of Bengal"

    # Combine BOB + ARAS into unified labeled dataset
    df_p3_combined = pd.concat([df_p3_bob, df_p3_aras], ignore_index=True)
    df_p3_combined = df_p3_combined.sort_values(["date", "site_id"]).reset_index(drop=True)

    combined_parquet = os.path.join(historical_dir, "labeled_features_combined.parquet")
    combined_meta = os.path.join(historical_dir, "labeled_features_combined_metadata.json")
    df_p3_combined.to_parquet(combined_parquet, index=False, engine="pyarrow")

    combined_summary = {
        "dataset_name": "Sagar-Drishti Multi-Basin Labeled Historical Dataset (BOB + ARAS)",
        "total_rows": len(df_p3_combined),
        "total_columns": len(df_p3_combined.columns),
        "date_range": {
            "start": df_p3_combined["date"].min(),
            "end": df_p3_combined["date"].max(),
        },
        "sites": ["bob", "aras"],
        "basins": ["Bay of Bengal", "Arabian Sea"],
        "rows_by_site": df_p3_combined["site_id"].value_counts().to_dict(),
        "label_distribution": df_p3_combined["label_status"].value_counts().to_dict(),
    }
    with open(combined_meta, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    print(f"[+] Saved combined labeled dataset: {combined_parquet}")
    print(f"[+] Total combined labeled rows:    {len(df_p3_combined)}")
    print(f"[+] Total columns:                 {len(df_p3_combined.columns)}")

    # --------------------------------------------------------------------------
    # TASK 3: FULL COMBINED DATASET AUDIT (BOB, ARAS, COMBINED)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TASK 3: FULL AUDIT SEPARATELY FOR BOB, ARAS, AND COMBINED")
    print("=" * 85)

    def print_basin_audit(name: str, df: pd.DataFrame):
        times = pd.to_datetime(df["date"])
        status_vc = df["label_status"].value_counts().to_dict()
        events_matched = df["event_id"].dropna().unique()
        event_types = df["event_type"].dropna().unique() if "event_type" in df.columns else []

        active_cnt = int((df["label_status"] == "active_event").sum())
        lead_cnt = int((df["label_status"] == "lead_event").sum())
        clean_cnt = int((df["label_status"] == "negative_clean").sum())
        buffer_cnt = int((df["label_status"] == "negative_buffer").sum())
        unknown_cnt = int((df["label_status"] == "unknown_uncovered").sum())

        missing_dates = pd.date_range(times.min(), times.max(), freq="D").difference(times)
        duplicates = len(df) - len(df["date"].unique()) if name != "Combined (BOB + ARAS)" else 0

        pos_density = ((active_cnt + lead_cnt) / len(df)) * 100.0

        print(f"\n--- {name.upper()} ---")
        print(f"  Feature row count:     {len(df)}")
        print(f"  Date range:            {times.min().strftime('%Y-%m-%d')} to {times.max().strftime('%Y-%m-%d')}")
        print(f"  Unique dates:          {len(times.unique())}")
        print(f"  Missing dates:         {len(missing_dates)}")
        print(f"  Duplicate dates:       {duplicates}")
        print(f"  Active event rows:     {active_cnt}")
        print(f"  Lead event rows:       {lead_cnt}")
        print(f"  Negative clean rows:   {clean_cnt}")
        print(f"  Negative buffer rows:  {buffer_cnt}")
        print(f"  Unknown rows:          {unknown_cnt}")
        print(f"  Total positive rows:   {active_cnt + lead_cnt} ({pos_density:.2f}% of total)")
        print(f"  Matched event IDs:     {list(events_matched)}")
        print(f"  Matched event types:   {list(event_types)}")

    print_basin_audit("Bay of Bengal (BOB)", df_p3_bob)
    print_basin_audit("Arabian Sea (ARAS)", df_p3_aras)
    print_basin_audit("Combined (BOB + ARAS)", df_p3_combined)

    # --------------------------------------------------------------------------
    # TASK 4: SCIENTIFICALLY VALID EVENT-AWARE CHRONOLOGICAL SPLIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TASK 4: SCIENTIFICALLY VALID EVENT-AWARE CHRONOLOGICAL SPLIT DESIGN")
    print("=" * 85)

    # Filter to clean ML rows (exclude unknown_uncovered and negative_buffer)
    clean_mask = (df_p3_combined["label_status"] != "unknown_uncovered") & (df_p3_combined["label_status"] != "negative_buffer")
    ml_combined = df_p3_combined[clean_mask].copy()
    ml_combined["parsed_date"] = pd.to_datetime(ml_combined["date"])

    print(f"Total clean multi-basin observations: {len(ml_combined)} rows ({len(ml_combined[ml_combined['site_id']=='bob'])} BOB, {len(ml_combined[ml_combined['site_id']=='aras'])} ARAS)")
    print(f"Clean date span: {ml_combined['parsed_date'].min().strftime('%Y-%m-%d')} to {ml_combined['parsed_date'].max().strftime('%Y-%m-%d')}")

    # Inspect all positive events and their exact active + lead time windows
    matched_events_info = []
    events = HistoricalEventStore.get_all_events()
    for ev in events:
        ev_obs = ml_combined[ml_combined["event_id"] == ev.event_id]
        if len(ev_obs) > 0:
            sites_present = list(ev_obs["site_id"].unique())
            min_d = ev_obs["date"].min()
            max_d = ev_obs["date"].max()
            act_n = int((ev_obs["label_status"] == "active_event").sum())
            lead_n = int((ev_obs["label_status"] == "lead_event").sum())
            matched_events_info.append({
                "event_id": ev.event_id,
                "name": ev.name,
                "type": ev.event_type.value,
                "sites": sites_present,
                "first_date": min_d,
                "last_date": max_d,
                "active_rows": act_n,
                "lead_rows": lead_n,
                "total_positive_rows": act_n + lead_n,
            })

    df_events = pd.DataFrame(matched_events_info).sort_values("first_date").reset_index(drop=True)
    print("\n--- CHRONOLOGICAL EVENT SEQUENCE ACROSS BOTH BASINS ---")
    for idx, r in df_events.iterrows():
        print(f"  [{idx+1}] {r['first_date']} to {r['last_date']} | {r['event_id']:20s} ({r['name']}) | Sites: {r['sites']} | Pos: {r['total_positive_rows']} ({r['active_rows']} act, {r['lead_rows']} lead)")

    # PROPOSED EVENT ALLOCATION:
    # 1. Train Fold: Earliest 3 events
    #    - IMD-2024-D-BOB04 (BOB, Aug 21-27)
    #    - IMD-2024-SCS-ASNA (ARAS, Aug 26-Sep 02)
    #    - IMD-2024-DD-BOB05 (BOB, Sep 04-10)
    #    Cutoff date: 2024-07-23 through 2024-09-30 (70 calendar days)
    #
    # 2. Validation Fold: Next 2 events
    #    - IMD-2024-D-ARB01 (ARAS, Oct 08-13)
    #    - IMD-2024-SCS-DANA (BOB, Oct 19-25)
    #    Cutoff date: 2024-10-01 through 2024-11-15 (46 calendar days)
    #
    # 3. Test Fold: Held-out final severe event + winter calm
    #    - IMD-2024-CS-FENGAL (BOB, Nov 25-Dec 01)
    #    Cutoff date: 2024-11-16 through 2025-01-31 (77 calendar days)

    train_cutoff_end = "2024-09-30"
    val_cutoff_end = "2024-11-15"

    def assign_event_aware_split(date_str: str) -> str:
        if date_str <= train_cutoff_end:
            return "train"
        elif date_str <= val_cutoff_end:
            return "val"
        else:
            return "test"

    ml_combined["split"] = ml_combined["date"].apply(assign_event_aware_split)

    print("\n--- RESULTING EVENT-AWARE CHRONOLOGICAL SPLIT COUNTS ---")
    split_report = {}
    for sp in ["train", "val", "test"]:
        sub = ml_combined[ml_combined["split"] == sp]
        tot = len(sub)
        act = int((sub["label_status"] == "active_event").sum())
        lead = int((sub["label_status"] == "lead_event").sum())
        clean = int((sub["label_status"] == "negative_clean").sum())
        evs = list(sub["event_id"].dropna().unique())
        sub_bob = len(sub[sub["site_id"] == "bob"])
        sub_aras = len(sub[sub["site_id"] == "aras"])

        split_report[sp] = {
            "total_rows": tot,
            "bob_rows": sub_bob,
            "aras_rows": sub_aras,
            "date_start": sub["date"].min(),
            "date_end": sub["date"].max(),
            "active_positives": act,
            "lead_positives": lead,
            "total_positives": act + lead,
            "clean_negatives": clean,
            "events_included": evs,
        }
        print(f"\n  Split '{sp.upper()}':")
        print(f"    Date span:        {sub['date'].min()} to {sub['date'].max()}")
        print(f"    Total rows:       {tot} (BOB: {sub_bob}, ARAS: {sub_aras})")
        print(f"    Active positives: {act}")
        print(f"    Lead positives:   {lead}")
        print(f"    Total positives:  {act + lead} ({((act + lead) / tot) * 100:.2f}%)")
        print(f"    Clean negatives:  {clean}")
        print(f"    Distinct events:  {evs}")

    # Save multi-basin ML features
    # Filter columns to pure physical features
    feature_cols = [c for c in ml_combined.columns if c not in LEAKAGE_AND_TARGET_COLUMNS and c not in ["basin", "parsed_date"]]
    print(f"\n[+] Total pure feature columns: {len(feature_cols)}")

    ml_multibasin_parquet = os.path.join(historical_dir, "ml_features_multibasin.parquet")
    ml_multibasin_meta = os.path.join(historical_dir, "ml_features_multibasin_metadata.json")

    ml_combined.to_parquet(ml_multibasin_parquet, index=False, engine="pyarrow")

    meta_payload = {
        "dataset_name": "Sagar-Drishti Multi-Basin Clean Supervised ML Dataset (BOB + ARAS)",
        "total_clean_rows": len(ml_combined),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "split_methodology": "Event-Aware Chronological Split (Train: 3 events, Val: 2 events, Test: 1 event)",
        "splits": split_report,
        "chronological_event_catalog": matched_events_info,
    }
    with open(ml_multibasin_meta, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"[+] Saved multi-basin ML Parquet:  {ml_multibasin_parquet}")
    print(f"[+] Saved multi-basin ML Metadata: {ml_multibasin_meta}")

    # --------------------------------------------------------------------------
    # TASK 5: LEAKAGE & CAUSALITY AUDIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TASK 5: RIGOROUS TEMPORAL LEAKAGE & EVENT SEPARATION AUDIT")
    print("=" * 85)

    # 1. Feature blacklist check
    leaking_in_X = [c for c in feature_cols if c in LEAKAGE_AND_TARGET_COLUMNS or "event" in c.lower() or "lead" in c.lower()]
    print(f"[1] Target/Identifier leakage blacklist check in X: {len(leaking_in_X)} violations")

    # 2. Chronological event separation
    train_max_date = split_report["train"]["date_end"]
    val_min_date = split_report["val"]["date_start"]
    val_max_date = split_report["val"]["date_end"]
    test_min_date = split_report["test"]["date_start"]

    event_sep_ok = (train_max_date < val_min_date) and (val_max_date < test_min_date)
    print(f"[2] Strict temporal separation (Train max {train_max_date} < Val min {val_min_date} < Val max {val_max_date} < Test min {test_min_date}): {event_sep_ok}")

    # 3. Disjoint event sets check
    train_evs = set(split_report["train"]["events_included"])
    val_evs = set(split_report["val"]["events_included"])
    test_evs = set(split_report["test"]["events_included"])

    ev_overlap_tv = train_evs.intersection(val_evs)
    ev_overlap_vt = val_evs.intersection(test_evs)
    ev_overlap_tt = train_evs.intersection(test_evs)
    disjoint_events_ok = (len(ev_overlap_tv) == 0) and (len(ev_overlap_vt) == 0) and (len(ev_overlap_tt) == 0)
    print(f"[3] Disjoint event separation across splits: {disjoint_events_ok}")
    print(f"    Train events:      {train_evs}")
    print(f"    Validation events: {val_evs}")
    print(f"    Test events:       {test_evs}")

    # 4. Monotonic chronological ordering within each basin
    bob_monotonic = ml_combined[ml_combined["site_id"]=="bob"]["parsed_date"].is_monotonic_increasing
    aras_monotonic = ml_combined[ml_combined["site_id"]=="aras"]["parsed_date"].is_monotonic_increasing
    print(f"[4] Monotonic chronological sorting (BOB: {bob_monotonic}, ARAS: {aras_monotonic})")

    # 5. Non-zero positive counts in Validation and Test
    val_pos_ok = split_report["val"]["total_positives"] > 0
    test_pos_ok = split_report["test"]["total_positives"] > 0
    print(f"[5] Non-zero positive observations in Validation fold: {val_pos_ok} ({split_report['val']['total_positives']} positives)")
    print(f"[6] Non-zero positive observations in Test fold:       {test_pos_ok} ({split_report['test']['total_positives']} positives)")

    leakage_verdict = (len(leaking_in_X) == 0) and event_sep_ok and disjoint_events_ok and bob_monotonic and aras_monotonic and val_pos_ok and test_pos_ok
    print(f"\n[+] LEAKAGE & EVENT-SPLIT VERDICT: {'PASS' if leakage_verdict else 'FAIL'}")

    return {
        "df_p3_bob": df_p3_bob,
        "df_p3_aras": df_p3_aras,
        "df_p3_combined": df_p3_combined,
        "ml_combined": ml_combined,
        "split_report": split_report,
        "matched_events": matched_events_info,
        "leakage_verdict": leakage_verdict,
    }


if __name__ == "__main__":
    run_multibasin_pipeline()
