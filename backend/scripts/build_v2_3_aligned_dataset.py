"""
Sagar-Drishti V2.3 — Build Aligned Ocean + ERA5 Ablation Dataset.
Aligns 7,178 clean multi-basin ocean observations with 10-year ERA5 causal atmospheric features
using standardized regional sub-basin roles (north, central, south) to ensure zero NaNs and
zero Cartesian row expansion.

Safety: Candidate / Research dataset ONLY. Does not modify existing production or v2_10yr datasets.
"""

import os
import sys
import hashlib
import json
import pandas as pd
import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BACKEND_DIR, "data", "historical")
ERA5_DIR = os.path.join(BACKEND_DIR, "data", "era5")

OCEAN_ML_PATH = os.path.join(DATA_DIR, "ml_features_10yr_candidate.parquet")
ERA5_PATH = os.path.join(ERA5_DIR, "features_atmosphere_10yr_daily.parquet")
OUTPUT_ALIGNED_PATH = os.path.join(DATA_DIR, "aligned_ocean_era5_10yr_ablation.parquet")


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_aligned_dataset():
    print(f"[*] Loading Ocean ML candidate dataset from {OCEAN_ML_PATH}...")
    ocean_df = pd.read_parquet(OCEAN_ML_PATH)
    print(f"    Loaded {len(ocean_df)} clean rows. Splits: {ocean_df['split'].value_counts().to_dict()}")

    print(f"[*] Loading ERA5 candidate dataset from {ERA5_PATH}...")
    era5_df = pd.read_parquet(ERA5_PATH)
    print(f"    Loaded {len(era5_df)} rows across {era5_df['basin'].nunique()} basins.")

    ocean_df["date"] = pd.to_datetime(ocean_df["date"])
    era5_df["date"] = pd.to_datetime(era5_df["date"])

    meta_cols = [
        "expected_6h_samples", "number_of_valid_6h_samples",
        "completeness_fraction", "is_complete", "is_ml_warmup_complete"
    ]
    era5_feats = [c for c in era5_df.columns if c not in ["date", "basin"] + meta_cols]
    assert len(era5_feats) == 41, f"Expected 41 physical features per basin, found {len(era5_feats)}"

    basin_to_role = {
        "bob": {"NBOB": "atmos_north", "CBOB": "atmos_central", "SBOB": "atmos_south"},
        "aras": {"NAS": "atmos_north", "CAS": "atmos_central", "SAS": "atmos_south"}
    }

    dfs_aligned = []
    for site, role_map in basin_to_role.items():
        site_ml = ocean_df[ocean_df["site_id"] == site].copy()
        site_era5 = era5_df[era5_df["basin"].isin(role_map.keys())].copy()
        site_era5["role"] = site_era5["basin"].map(role_map)

        # Pivot to horizontal prefix format
        piv = site_era5.pivot(index="date", columns="role", values=era5_feats)
        piv.columns = [f"{role}_{feat}" for feat, role in piv.columns]
        piv = piv.reset_index()

        aligned = pd.merge(site_ml, piv, on="date", how="inner")
        dfs_aligned.append(aligned)

    final_df = pd.concat(dfs_aligned, ignore_index=True)
    # Ensure deterministic sort
    final_df = final_df.sort_values(by=["date", "site_id"]).reset_index(drop=True)

    # Verification checks
    assert len(final_df) == 7178, f"Row count mismatch: {len(final_df)} != 7178"
    assert final_df.duplicated(subset=["date", "site_id"]).sum() == 0, "Duplicate keys detected"

    atmos_cols = [c for c in final_df.columns if c.startswith("atmos_")]
    assert len(atmos_cols) == 123, f"Expected 123 atmospheric columns, found {len(atmos_cols)}"
    assert final_df[atmos_cols].isna().sum().sum() == 0, "NaNs found in atmospheric columns"

    # Save to parquet
    final_df.to_parquet(OUTPUT_ALIGNED_PATH, index=False, engine="pyarrow")
    sha256_val = compute_sha256(OUTPUT_ALIGNED_PATH)
    file_size = os.path.getsize(OUTPUT_ALIGNED_PATH)

    print(f"[+] Successfully exported aligned ablation dataset:")
    print(f"    Path:     {OUTPUT_ALIGNED_PATH}")
    print(f"    Rows:     {len(final_df)}")
    print(f"    Columns:  {len(final_df.columns)}")
    print(f"    Size:     {file_size:,} bytes")
    print(f"    SHA-256:  {sha256_val}")

    return OUTPUT_ALIGNED_PATH, sha256_val


if __name__ == "__main__":
    build_aligned_dataset()
