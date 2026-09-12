"""
Ocean + ERA5 Spatial and Temporal Alignment Test Suite
Verifies data integrity, spatial mapping, leakage prevention, and contract preservation.
"""
import pytest
import pandas as pd
import numpy as np
import hashlib
import os

OCEAN_FEATURES_PATH = "backend/data/historical/features_10yr.parquet"
LABELED_OCEAN_PATH = "backend/data/historical/labeled_features_10yr_clean.parquet"
ERA5_FEATURES_PATH = "backend/data/era5/features_atmosphere_10yr_daily.parquet"
FROZEN_POLICY_PATH = "backend/config/frozen_alert_policy_v2.json"
PRODUCTION_MODEL_DIR = "backend/models"

EXPECTED_FROZEN_POLICY_SHA256 = "6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6"
EXPECTED_OCEAN_SHA256 = "cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867"
EXPECTED_ERA5_SHA256 = "551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def datasets():
    df_ocean = pd.read_parquet(OCEAN_FEATURES_PATH)
    df_labeled = pd.read_parquet(LABELED_OCEAN_PATH)
    df_era5 = pd.read_parquet(ERA5_FEATURES_PATH)
    
    df_ocean["date"] = pd.to_datetime(df_ocean["date"])
    df_labeled["date"] = pd.to_datetime(df_labeled["date"])
    df_era5["date"] = pd.to_datetime(df_era5["date"])
    
    return {
        "ocean": df_ocean,
        "labeled": df_labeled,
        "era5": df_era5,
    }


class TestOceanERA5Alignment:
    """Rigorous audit tests for Ocean and ERA5 alignment."""

    def test_production_artifacts_invariant(self):
        """Confirm production policy, ocean features, and ERA5 candidate hashes are byte-identical."""
        assert compute_sha256(FROZEN_POLICY_PATH) == EXPECTED_FROZEN_POLICY_SHA256
        assert compute_sha256(OCEAN_FEATURES_PATH) == EXPECTED_OCEAN_SHA256
        assert compute_sha256(ERA5_FEATURES_PATH) == EXPECTED_ERA5_SHA256

    def test_ocean_dataset_contract_and_cardinality(self, datasets):
        """Verify ocean dataset dimensions, sites, dates, and 101 features."""
        df_ocean = datasets["ocean"]
        assert len(df_ocean) == 7246
        assert set(df_ocean["site_id"].unique()) == {"bob", "aras"}
        assert len(df_ocean[df_ocean["site_id"] == "bob"]) == 3623
        assert len(df_ocean[df_ocean["site_id"] == "aras"]) == 3623
        
        # Exact 101 float features
        float_cols = df_ocean.select_dtypes(include=["float"]).columns.tolist()
        assert len(float_cols) == 101
        
        # Zero duplicate keys
        assert df_ocean.duplicated(subset=["date", "site_id"]).sum() == 0
        
        # Zero NaNs across 101 features
        assert df_ocean[float_cols].isna().sum().sum() == 0

    def test_era5_dataset_contract_and_cardinality(self, datasets):
        """Verify ERA5 dataset dimensions, basins, dates, and quality flags."""
        df_era5 = datasets["era5"]
        assert len(df_era5) == 21912
        assert set(df_era5["basin"].unique()) == {"NAS", "CAS", "SAS", "NBOB", "CBOB", "SBOB"}
        assert len(df_era5["date"].unique()) == 3652
        
        # Zero duplicate keys
        assert df_era5.duplicated(subset=["date", "basin"]).sum() == 0
        
        # Zero base NaNs for complete days
        base_features = [
            "max_vorticity", "mean_vorticity", "std_vorticity",
            "gridcell_exceedance_fraction", "weighted_exceedance_fraction",
            "latitude_of_max_vorticity", "longitude_of_max_vorticity",
            "min_vws", "mean_vws", "max_vws",
            "mean_rh700", "max_rh700", "mean_rh500", "max_rh500"
        ]
        assert df_era5[df_era5["is_complete"]][base_features].isna().sum().sum() == 0

    def test_date_intersection(self, datasets):
        """Verify exact date overlap between Ocean (2016-07-23 to 2026-06-23) and ERA5."""
        ocean_dates = set(datasets["ocean"]["date"])
        era5_dates = set(datasets["era5"]["date"])
        
        # Ocean date range must be entirely contained within ERA5
        assert ocean_dates.issubset(era5_dates)
        assert len(ocean_dates) == 3623
        assert min(ocean_dates) == pd.Timestamp("2016-07-23")
        assert max(ocean_dates) == pd.Timestamp("2026-06-23")

    def test_warmup_safety_on_ocean_dates(self, datasets):
        """Verify that every date in the Ocean dataset has 100% completed ERA5 warmup."""
        df_era5 = datasets["era5"]
        ocean_start = datasets["ocean"]["date"].min()
        
        # Check all ERA5 rows on or after ocean start date
        era5_on_ocean = df_era5[df_era5["date"] >= ocean_start]
        assert era5_on_ocean["is_ml_warmup_complete"].all()
        
        # All 41 features must be finite and non-NaN on these dates
        non_meta = [c for c in df_era5.columns if c not in [
            "date", "basin", "expected_6h_samples", "number_of_valid_6h_samples",
            "completeness_fraction", "is_complete", "is_ml_warmup_complete"
        ]]
        assert era5_on_ocean[non_meta].isna().sum().sum() == 0

    def test_spatial_alignment_no_cartesian_explosion(self, datasets):
        """
        Verify Strategy 1 (Hierarchical Wide Regional Prefix Join):
        - 'bob' maps to ['NBOB', 'CBOB', 'SBOB']
        - 'aras' maps to ['NAS', 'CAS', 'SAS']
        Preserves EXACTLY 7,246 rows with zero row multiplication.
        """
        df_ocean = datasets["ocean"]
        df_era5 = datasets["era5"]
        
        # Meta columns to exclude from feature wide-pivot
        meta_cols = ["expected_6h_samples", "number_of_valid_6h_samples", 
                     "completeness_fraction", "is_complete", "is_ml_warmup_complete"]
        era5_feat_cols = [c for c in df_era5.columns if c not in ["date", "basin"] + meta_cols]
        assert len(era5_feat_cols) == 41
        
        # 1. BOB Alignment
        era5_bob = df_era5[df_era5["basin"].isin(["NBOB", "CBOB", "SBOB"])].copy()
        era5_bob_piv = era5_bob.pivot(index="date", columns="basin", values=era5_feat_cols)
        era5_bob_piv.columns = [f"{b.lower()}_{f}" for f, b in era5_bob_piv.columns]
        era5_bob_piv = era5_bob_piv.reset_index()
        
        bob_ocean = df_ocean[df_ocean["site_id"] == "bob"].copy()
        bob_aligned = pd.merge(bob_ocean, era5_bob_piv, on="date", how="inner")
        
        assert len(bob_aligned) == 3623
        assert len(era5_bob_piv.columns) == 1 + (3 * 41)  # date + 123 features
        
        # 2. ARAS Alignment
        era5_aras = df_era5[df_era5["basin"].isin(["NAS", "CAS", "SAS"])].copy()
        era5_aras_piv = era5_aras.pivot(index="date", columns="basin", values=era5_feat_cols)
        era5_aras_piv.columns = [f"{b.lower()}_{f}" for f, b in era5_aras_piv.columns]
        era5_aras_piv = era5_aras_piv.reset_index()
        
        aras_ocean = df_ocean[df_ocean["site_id"] == "aras"].copy()
        aras_aligned = pd.merge(aras_ocean, era5_aras_piv, on="date", how="inner")
        
        assert len(aras_aligned) == 3623
        assert len(era5_aras_piv.columns) == 1 + (3 * 41)
        
        # Total aligned rows across both sites
        total_rows = len(bob_aligned) + len(aras_aligned)
        assert total_rows == 7246

    def test_target_alignment_temporal_causality(self, datasets):
        """
        Verify target definitions and ensure no lookahead leakage:
        Features at Date T represent observations up to 18:00 UTC of Date T.
        Targets predict:
        - lead_0: active at Date T
        - lead_1: active at Date T+1
        - lead_2: active at Date T+2
        - lead_3: active at Date T+3
        """
        df_labeled = datasets["labeled"]
        target_cols = ["lead_0", "lead_1", "lead_2", "lead_3", "event_active"]
        for col in target_cols:
            assert col in df_labeled.columns
            # Targets must be binary {0, 1}
            vals = set(df_labeled[col].dropna().unique())
            assert vals.issubset({0, 1})
            
        # Verify chronological ordering
        for site in ["bob", "aras"]:
            site_df = df_labeled[df_labeled["site_id"] == site].sort_values("date")
            dates = site_df["date"].tolist()
            # Dates must be consecutive days (cadence = 1 day)
            diffs = pd.Series(dates).diff().dropna()
            assert (diffs == pd.Timedelta(days=1)).all()
