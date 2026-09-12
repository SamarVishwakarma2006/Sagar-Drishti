import os
import sys
import pandas as pd
import numpy as np

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.event_store import HistoricalEventStore
from app.services.event_matcher import SpatialTemporalMatcher, BoundingBox
from app.services.copernicus_service import CopernicusService

print("=" * 80)
print("SAGAR-DRISHTI: CLEAN LABEL & SPATIAL MATCHING AUDIT")
print("=" * 80)

# Load event store and load corrected IMD systems
HistoricalEventStore.reset_to_seeds()
cache_dir = os.path.abspath(os.path.join(backend_dir, "data", "historical"))
imd_json = os.path.join(cache_dir, "imd_systems_2016_2026.json")
added = HistoricalEventStore.load_from_imd_json(imd_json)
print(f"HistoricalEventStore registered: {len(HistoricalEventStore.get_all_events())} events ({added} from IMD)")

# Test sites
sites = [
    {"site_id": "bob", "name": "Bay of Bengal", "bbox": BoundingBox(min_lat=16.3, max_lat=19.3, min_lon=86.7, max_lon=89.7)},
    {"site_id": "aras", "name": "Arabian Sea", "bbox": BoundingBox(min_lat=15.0, max_lat=18.0, min_lon=68.0, max_lon=71.0)},
]

# Generate date range for available 2-year NetCDF (2024-06-24 to 2026-06-23)
dates = pd.date_range("2024-06-24", "2026-06-23", freq="D").strftime("%Y-%m-%d").tolist()
print(f"Total dates in local Copernicus dataset: {len(dates)} days ({dates[0]} to {dates[-1]})")

records = []
for site in sites:
    site_id = site["site_id"]
    bbox = site["bbox"]
    for d in dates:
        res = SpatialTemporalMatcher.match_observation(
            date_str=d,
            mode="region",
            site_id=site_id,
            bbox=bbox,
            lead_window_days=3
        )
        res["site_id"] = site_id
        res["date"] = d
        res["year"] = int(d[:4])
        records.append(res)

df = pd.DataFrame(records)
print(f"\nTotal labeled observations generated: {len(df)}")

# Summary by label status
print("\n--- LABEL STATUS DISTRIBUTION ---")
print(df["label_status"].value_counts().to_string())

# Summary of Horizons
pos_0d = int(df["event_within_0d"].sum())
pos_1d = int(df["event_within_1d"].sum())
pos_2d = int(df["event_within_2d"].sum())
pos_3d = int(df["event_within_3d"].sum())
neg_clean = sum(1 for s in df["label_status"] if s == "negative_clean")
neg_buffer = sum(1 for s in df["label_status"] if s == "negative_buffer")
uncovered = sum(1 for s in df["label_status"] if s == "unknown_uncovered")

print("\n--- HORIZON TARGET TOTALS ---")
print(f"Total Observations:        {len(df)}")
print(f"Negative Clean:            {neg_clean}")
print(f"Negative Buffer:           {neg_buffer}")
print(f"Unknown / Uncovered:       {uncovered}")
print(f"Positive 0d (Active):      {pos_0d} ({pos_0d / len(df) * 100:.2f}%)")
print(f"Positive 1d (Lead <= 1d):  {pos_1d} ({pos_1d / len(df) * 100:.2f}%)")
print(f"Positive 2d (Lead <= 2d):  {pos_2d} ({pos_2d / len(df) * 100:.2f}%)")
print(f"Positive 3d (Lead <= 3d):  {pos_3d} ({pos_3d / len(df) * 100:.2f}%)")

# Year-by-year utilization
print("\n--- YEAR-BY-YEAR LABEL DISTRIBUTION (SITE: BAY OF BENGAL + ARABIAN SEA) ---")
year_rows = []
for y in sorted(df["year"].unique()):
    sub = df[df["year"] == y]
    pos = int(sub["event_within_3d"].sum())
    neg = sum(1 for s in sub["label_status"] if s == "negative_clean")
    unk = sum(1 for s in sub["label_status"] if s == "unknown_uncovered")
    prev = pos / len(sub) * 100
    year_rows.append({
        "YEAR": y,
        "TOTAL": len(sub),
        "POSITIVE (3d)": pos,
        "NEGATIVE": neg,
        "UNKNOWN": unk,
        "PREVALENCE": f"{prev:.2f}%",
        "USED": "YES"
    })
print(pd.DataFrame(year_rows).to_string(index=False))

# Section 14: Deep Audit of Bay of Bengal regional matching
print("\n" + "=" * 80)
print("SECTION 14: BAY OF BENGAL (site_id='bob') REGIONAL MATCHING AUDIT")
print("=" * 80)

bob_df = df[df["site_id"] == "bob"]
bob_pos = bob_df[bob_df["is_active_event"] == 1]
print(f"Total Bay of Bengal observations: {len(bob_df)}")
print(f"Active cyclone days matched in BoB: {len(bob_pos)} / {len(bob_df)} ({len(bob_pos)/len(bob_df)*100:.2f}%)")

if not bob_pos.empty:
    print("\nActive Events matched in Bay of Bengal:")
    for ev_id, group in bob_pos.groupby("event_id"):
        matched_dates = group["date"].tolist()
        ev = HistoricalEventStore.get_event_by_id(ev_id)
        ev_start = ev.start_date if ev else "N/A"
        ev_end = ev.end_date if ev else "N/A"
        ev_name = ev.name if ev else ev_id
        print(f"  {ev_id:<28} | Name: {ev_name:<20} | Real Range: {ev_start} to {ev_end} | Matched Days: {len(matched_dates)} ({matched_dates[0]} to {matched_dates[-1]})")
