import os
import sys
import json
import pandas as pd

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.parsers.imd_best_track_parser import IMDBestTrackParser

excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "78b4b0_Best_Tracks__Data__1982-2026_.xlsx"))
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "historical"))

print(f"Reparsing from: {excel_path}")
print(f"Target cache: {cache_dir}")

res = IMDBestTrackParser.get_cached_or_parse(excel_path, cache_dir=cache_dir, force_reparse=True)
report = res["validation_report"]
print("\nValidation report:")
print(json.dumps(report, indent=2))

systems = res["systems"]
track_df = res["track_dataframe"]

print(f"\nTotal systems: {len(systems)}")
print(f"Total track points: {len(track_df)}")
print(f"Date range: {track_df['date'].min()} to {track_df['date'].max()}")

print("\n=== 2024 SYSTEMS AND GENESIS DATES ===")
for s in sorted([s for s in systems if s["year"] == 2024], key=lambda x: x["start_date"]):
    print(f"  {s['system_id']:<28} | Name: {s['name']:<12} | Start: {s['start_date']} | End: {s['end_date']} | Points: {s['track_points_count']}")

print("\n=== YEARLY INVENTORY AUDIT (2016-2026) ===")
summary_rows = []
for y in range(2016, 2027):
    sub_df = track_df[track_df["year"] == y]
    sub_sys = [s for s in systems if s["year"] == y]
    min_d = sub_df["date"].min() if not sub_df.empty else "N/A"
    max_d = sub_df["date"].max() if not sub_df.empty else "N/A"
    # Check invalid timestamps
    invalid_ts = sum(1 for ts in sub_df["datetime_iso"] if len(str(ts)) < 19)
    summary_rows.append({
        "YEAR": y,
        "SYSTEMS": len(sub_sys),
        "TRACK POINTS": len(sub_df),
        "INVALID TIMESTAMPS": invalid_ts,
        "MIN DATE": min_d,
        "MAX DATE": max_d
    })

audit_df = pd.DataFrame(summary_rows)
print(audit_df.to_string(index=False))

# Verify specific named storms
targets = ["ASNA", "DANA", "FENGAL", "REMAL"]
print("\n=== SPECIFIC STORM VERIFICATION ===")
for name in targets:
    matches = [s for s in systems if name.upper() in s["name"].upper()]
    for m in matches:
        print(f"Storm {name}: {m['system_id']} -> Start: {m['start_date']}, End: {m['end_date']}, Peak Kts: {m['peak_intensity_kts']}")
