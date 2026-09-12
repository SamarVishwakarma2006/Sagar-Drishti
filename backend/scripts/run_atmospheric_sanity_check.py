"""
Sagar-Drishti V2.1 — Observational Atmospheric Precursor Sanity Check.
Inspects empirical observations for known historical cyclonic events:
1. Cyclone Asna (Aug–Sep 2024, Northern Arabian Sea)
2. Cyclone Remal (May 2024, Bay of Bengal)
3. Cyclone Biparjoy (June 2023, Arabian Sea)

SCIENTIFIC PRINCIPLE:
Strictly observational reporting — reports what the data ACTUALLY show.
Does NOT assume outcomes, force thresholds, or tune features.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_PATH = os.path.join(BACKEND_DIR, "data", "era5", "features_atmosphere_10yr_daily.parquet")
OUTPUT_JSON = os.path.join(BACKEND_DIR, "reports", "era5_storm_sanity_audit.json")


def audit_storm_window(
    df: pd.DataFrame,
    event_name: str,
    start_date: str,
    end_date: str,
    target_basins: list,
) -> dict:
    """Extracts observational feature time series for a specific event window and basins."""
    sub = df[(df["date"] >= start_date) & (df["date"] <= end_date) & (df["basin"].isin(target_basins))].copy()
    sub = sub.sort_values(by=["basin", "date"]).reset_index(drop=True)

    event_summary = {
        "event_name": event_name,
        "window_start": start_date,
        "window_end": end_date,
        "basins_analyzed": target_basins,
        "observations_count": len(sub),
        "basin_daily_observations": {},
    }

    for b in target_basins:
        b_df = sub[sub["basin"] == b]
        obs_list = []
        for _, row in b_df.iterrows():
            obs_list.append({
                "date": row["date"],
                "max_vorticity": round(float(row["max_vorticity"]), 3),
                "mean_vorticity": round(float(row["mean_vorticity"]), 3),
                "weighted_exceedance_fraction": round(float(row["weighted_exceedance_fraction"]), 4),
                "gridcell_exceedance_fraction": round(float(row["gridcell_exceedance_fraction"]), 4),
                "max_vorticity_loc": (round(float(row["latitude_of_max_vorticity"]), 2), round(float(row["longitude_of_max_vorticity"]), 2)),
                "min_vws": round(float(row["min_vws"]), 2),
                "mean_vws": round(float(row["mean_vws"]), 2),
                "mean_rh700": round(float(row["mean_rh700"]), 2),
                "mean_rh500": round(float(row["mean_rh500"]), 2),
                "max_vorticity_change_24h": round(float(row["max_vorticity_change_24h"]), 3) if pd.notna(row["max_vorticity_change_24h"]) else None,
                "min_vws_change_24h": round(float(row["min_vws_change_24h"]), 2) if pd.notna(row["min_vws_change_24h"]) else None,
            })

        # Key empirical metrics during the window
        peak_vo = float(b_df["max_vorticity"].max())
        peak_vo_date = str(b_df.loc[b_df["max_vorticity"].idxmax(), "date"])
        min_vws_val = float(b_df["min_vws"].min())
        min_vws_date = str(b_df.loc[b_df["min_vws"].idxmin(), "date"])
        max_exceed = float(b_df["weighted_exceedance_fraction"].max())
        max_exceed_date = str(b_df.loc[b_df["weighted_exceedance_fraction"].idxmax(), "date"])

        event_summary["basin_daily_observations"][b] = {
            "daily_records": obs_list,
            "empirical_highlights": {
                "peak_max_vorticity": peak_vo,
                "peak_max_vorticity_date": peak_vo_date,
                "minimum_vws": min_vws_val,
                "minimum_vws_date": min_vws_date,
                "max_weighted_exceedance_fraction": max_exceed,
                "max_weighted_exceedance_date": max_exceed_date,
            }
        }

    return event_summary


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: Candidate parquet file does not exist at {PARQUET_PATH}")
        sys.exit(1)

    print(f"[*] Loading candidate atmospheric dataset from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)

    results = {}

    # 1. Cyclone Asna (2024): Late August / Early September
    print("[*] Inspecting Cyclone Asna (2024) in NAS and CAS...")
    results["asna_2024"] = audit_storm_window(
        df=df,
        event_name="Cyclone Asna (2024)",
        start_date="2024-08-22",
        end_date="2024-09-04",
        target_basins=["NAS", "CAS"],
    )

    # 2. Cyclone Remal (2024): Late May
    print("[*] Inspecting Cyclone Remal (2024) in NBOB and CBOB...")
    results["remal_2024"] = audit_storm_window(
        df=df,
        event_name="Cyclone Remal (2024)",
        start_date="2024-05-18",
        end_date="2024-05-29",
        target_basins=["NBOB", "CBOB"],
    )

    # 3. Cyclone Biparjoy (2023): Early to Mid June
    print("[*] Inspecting Cyclone Biparjoy (2023) in CAS and NAS...")
    results["biparjoy_2023"] = audit_storm_window(
        df=df,
        event_name="Cyclone Biparjoy (2023)",
        start_date="2023-05-28",
        end_date="2023-06-16",
        target_basins=["CAS", "NAS"],
    )

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Observational audit completed and written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
