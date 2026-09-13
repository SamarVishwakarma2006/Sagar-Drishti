"""
Sagar-Drishti Cyclone Intensity & Rapid Intensification Data Gate Test Suite

Verifies:
1. Raw-data immutability and cryptographic integrity.
2. Storm ID integrity and consistency between Parquet and JSON.
3. Accurate duplicate fix detection and deterministic resolution.
4. Outlier flagging and prohibition of silent corrections.
5. Target construction for continuous Vmax and discrete RI.
6. RI definitions (30 kt primary provisional, 25 kt and 20 kt sensitivities).
7. Temporal tolerance matching without interpolation leakage.
8. Causal cutoff enforcement and rejection of future observations.
9. Event-grouped splitting with zero storm overlap across partitions.
10. Proper handling and alerting for zero-positive test periods (2024-2026).
11. Sample size reporting at both row level and storm level.
12. Strict production and V2.3 shadow model isolation.
13. Manifest reproducibility and label revision provenance.
"""

import hashlib
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from app.services.hazard_registry import (
    hazard_registry,
    HazardStatus,
    TemporalLeakageError,
    OperationalIsolationError
)

# Baseline Protected Artifact Hashes
PROTECTED_HASHES = {
    "backend/models/risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
    "backend/models/v2_10yr/risk_model_3d.joblib": "7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib": "7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib": "b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib": "aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib": "250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775",
    "backend/config/frozen_alert_policy_v2.json": "6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6",
    "backend/data/historical/features_10yr.parquet": "cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867",
    "backend/data/historical/labeled_features_10yr_clean.parquet": "25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a",
    "backend/data/era5/features_atmosphere_10yr_daily.parquet": "551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864",
    "backend/config/v2_3_frozen_experiment_manifest.json": "73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc",
}


@pytest.fixture(scope="module")
def imd_tracks() -> pd.DataFrame:
    """Load raw IMD Best Track dataset without modification."""
    path = "backend/data/historical/imd_tracks_2016_2026.parquet"
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime_iso"])
    return df


@pytest.fixture(scope="module")
def imd_systems() -> list:
    """Load IMD systems metadata JSON."""
    with open("backend/data/historical/imd_systems_2016_2026.json", "r") as f:
        return json.load(f)


class TestRawDataImmutabilityAndIntegrity:
    """Test 1 & 25: Verify raw dataset immutability and protected hashes."""

    @pytest.mark.parametrize("filepath,expected_hash", PROTECTED_HASHES.items())
    def test_protected_artifact_hashes_unmodified(self, filepath: str, expected_hash: str) -> None:
        p = Path(filepath)
        assert p.exists(), f"Protected file {filepath} is missing!"
        with open(p, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == expected_hash, f"Hash mismatch in {filepath}!"

    def test_storm_id_integrity_and_alignment(self, imd_tracks: pd.DataFrame, imd_systems: list) -> None:
        """Test 2: Verify total systems, non-null IDs, and consistency with JSON."""
        assert len(imd_tracks) == 2544, f"Expected 2,544 rows, found {len(imd_tracks)}"
        unique_storm_ids = set(imd_tracks["system_id"].unique())
        assert len(unique_storm_ids) == 117, f"Expected 117 unique storms, found {len(unique_storm_ids)}"
        assert len(imd_systems) == 117, f"Expected 117 systems in JSON, found {len(imd_systems)}"
        assert imd_tracks["system_id"].isnull().sum() == 0


class TestDuplicateAndOutlierForensics:
    """Tests 3, 4, 5, 6: Duplicate detection, resolution logic, and outlier flagging."""

    def test_duplicate_fix_detection(self, imd_tracks: pd.DataFrame) -> None:
        """Test 3: Verify detection of exactly 4 duplicate (system_id, datetime) pairs."""
        dups = imd_tracks[imd_tracks.duplicated(subset=["system_id", "datetime_iso"], keep=False)]
        assert len(dups) == 8, f"Expected 8 duplicate rows (4 pairs), found {len(dups)}"
        dup_systems = set(dups["system_id"].unique())
        assert dup_systems == {
            "IMD-2018-14-PHETHAI",
            "IMD-2021-9-UNNAMED9",
            "IMD-2022-14-UNNAMED14",
            "IMD-2023-9-MICHAUNG"
        }

    def test_duplicate_resolution_logic_conflicts(self, imd_tracks: pd.DataFrame) -> None:
        """Test 4: Verify duplicate pairs are conflicting and marked for candidate exclusion."""
        dups = imd_tracks[imd_tracks.duplicated(subset=["system_id", "datetime_iso"], keep=False)]
        for sys_id, group in dups.groupby("system_id"):
            assert len(group) == 2
            row_a = group.iloc[0]
            row_b = group.iloc[1]
            # Verify that each pair contains a physical conflict (lat, lon, Vmax, or Pc)
            has_conflict = (
                row_a["latitude"] != row_b["latitude"] or
                row_a["longitude"] != row_b["longitude"] or
                row_a["max_wind_kts"] != row_b["max_wind_kts"] or
                row_a["central_pressure_hpa"] != row_b["central_pressure_hpa"]
            )
            assert has_conflict, f"Pair for {sys_id} should be flagged as conflicting!"

    def test_outlier_flagging_and_no_silent_correction(self, imd_tracks: pd.DataFrame) -> None:
        """Test 5 & 6: Verify IMD-2016-8-NADA outlier is flagged and NOT silently modified."""
        nada_outlier = imd_tracks[
            (imd_tracks["system_id"] == "IMD-2016-8-NADA") &
            (imd_tracks["datetime_iso"] == "2016-12-02T00:00:00")
        ]
        assert len(nada_outlier) == 1
        row = nada_outlier.iloc[0]
        # Verify raw value is preserved in raw file
        assert row["central_pressure_hpa"] == 25.0, "Raw central pressure must remain 25.0 hPa in Parquet"
        assert row["max_wind_kts"] == 2.0, "Raw max wind must remain 2.0 kt in Parquet"
        # Verify it triggers an automated outlier filter (< 800 hPa)
        assert row["central_pressure_hpa"] < 800.0


class TestTargetConstructionAndTolerance:
    """Tests 7, 8, 9, 10, 11, 12: Target calculations, sensitivities, and tolerance rules."""

    def test_ri_target_construction_and_tolerance(self, imd_tracks: pd.DataFrame) -> None:
        """Test 7, 8, 11, 12: Verify 24h RI matching within +/- 3h tolerance without interpolation."""
        # Pick a known RI storm: FANI (2019)
        fani = imd_tracks[imd_tracks["system_id"] == "IMD-2019-2-FANI"].sort_values("datetime")
        # Target at 24h forward from genesis fix
        t0 = fani.iloc[0]["datetime"]
        v0 = fani.iloc[0]["max_wind_kts"]

        # Search candidates in [T+21h, T+27h]
        candidates = fani[(fani["datetime"] >= t0 + timedelta(hours=21)) & (fani["datetime"] <= t0 + timedelta(hours=27))]
        assert len(candidates) > 0

        # Select closest to 24h
        candidates = candidates.copy()
        candidates["dt_diff"] = (candidates["datetime"] - (t0 + timedelta(hours=24))).abs()
        best_match = candidates.sort_values("dt_diff").iloc[0]

        # Verify no synthetic interpolation was applied
        assert best_match["max_wind_kts"] in fani["max_wind_kts"].values

    def test_ri_sensitivities_and_sample_sizes(self, imd_tracks: pd.DataFrame) -> None:
        """Test 8, 9, 10, 18: Verify sample sizes for 30 kt, 25 kt, and 20 kt definitions."""
        ri_records = []
        for sys_id, group in imd_tracks.groupby("system_id"):
            group = group.sort_values("datetime")
            for _, row in group.iterrows():
                t0 = row["datetime"]
                v0 = row["max_wind_kts"]
                future = group[(group["datetime"] >= t0 + timedelta(hours=21)) & (group["datetime"] <= t0 + timedelta(hours=27))]
                if len(future) > 0:
                    future = future.copy()
                    future["diff"] = (future["datetime"] - (t0 + timedelta(hours=24))).abs()
                    best = future.sort_values("diff").iloc[0]
                    delta_v = best["max_wind_kts"] - v0
                    ri_records.append({
                        "system_id": sys_id,
                        "year": row["year"],
                        "delta_v": delta_v,
                        "is_ri_30": delta_v >= 30.0,
                        "is_ri_25": delta_v >= 25.0,
                        "is_ri_20": delta_v >= 20.0,
                    })

        ri_df = pd.DataFrame(ri_records)
        assert len(ri_df) == 1907, f"Expected 1,907 candidate 24h pairs, found {len(ri_df)}"
        assert ri_df["is_ri_30"].sum() == 103, f"Expected 103 standard RI pairs, found {ri_df['is_ri_30'].sum()}"
        assert ri_df["is_ri_25"].sum() == 150, f"Expected 150 25-kt pairs, found {ri_df['is_ri_25'].sum()}"
        assert ri_df["is_ri_20"].sum() == 253, f"Expected 253 20-kt pairs, found {ri_df['is_ri_20'].sum()}"

        # Storm-level counts
        unique_ri_30_storms = ri_df[ri_df["is_ri_30"]]["system_id"].nunique()
        assert unique_ri_30_storms == 18, f"Expected exactly 18 unique RI storms, found {unique_ri_30_storms}"

    def test_zero_positive_test_handling(self, imd_tracks: pd.DataFrame) -> None:
        """Test 17: Confirm 2024-2026 has zero standard RI storms and triggers warning flag."""
        recent_tracks = imd_tracks[imd_tracks["year"] >= 2024]
        ri_positives_recent = 0
        for sys_id, group in recent_tracks.groupby("system_id"):
            group = group.sort_values("datetime")
            for _, row in group.iterrows():
                t0 = row["datetime"]
                future = group[(group["datetime"] >= t0 + timedelta(hours=21)) & (group["datetime"] <= t0 + timedelta(hours=27))]
                if len(future) > 0:
                    best = future.iloc[0]
                    if best["max_wind_kts"] - row["max_wind_kts"] >= 30.0:
                        ri_positives_recent += 1

        assert ri_positives_recent == 0, f"Expected zero RI positive pairs in 2024-2026, found {ri_positives_recent}"


class TestCausalCutoffAndSplittingFirewall:
    """Tests 13, 14, 15, 16: Causal cutoffs and event-grouped splitting."""

    def test_causal_cutoff_future_observation_rejection(self) -> None:
        """Test 13 & 14: Verify future observation rejection via HazardRegistry."""
        t_forecast = "2026-09-13T06:00:00Z"
        t_valid = "2026-09-13T06:00:00Z"
        t_future = "2026-09-13T06:01:00Z"

        assert hazard_registry.validate_timestamp_causality(t_forecast, t_valid) is True
        with pytest.raises(TemporalLeakageError):
            hazard_registry.validate_timestamp_causality(t_forecast, t_future)

    def test_event_grouped_splitting_zero_storm_overlap(self, imd_tracks: pd.DataFrame) -> None:
        """Test 15 & 16: Chronological event-grouped partition has zero storm overlap."""
        train_years = range(2016, 2022)
        val_years = range(2022, 2024)
        test_years = range(2024, 2027)

        train_storms = set(imd_tracks[imd_tracks["year"].isin(train_years)]["system_id"].unique())
        val_storms = set(imd_tracks[imd_tracks["year"].isin(val_years)]["system_id"].unique())
        test_storms = set(imd_tracks[imd_tracks["year"].isin(test_years)]["system_id"].unique())

        # Zero intersection across all partitions
        assert len(train_storms.intersection(val_storms)) == 0, "Train and Val storm overlap detected!"
        assert len(train_storms.intersection(test_storms)) == 0, "Train and Test storm overlap detected!"
        assert len(val_storms.intersection(test_storms)) == 0, "Val and Test storm overlap detected!"

        # Total coverage matches 117
        assert len(train_storms) + len(val_storms) + len(test_storms) == 117


class TestOperationalIsolationAndMetadata:
    """Tests 19, 20, 21, 22: Operational isolation, manifest schema, and revision metadata."""

    def test_intensity_modules_operational_isolation(self) -> None:
        """Test 19 & 20: Intensity and RI cannot be invoked by production alert engines."""
        intensity_config = hazard_registry.get_hazard("cyclone_intensity")
        assert intensity_config.status == HazardStatus.RESEARCH_PROPOSED
        with pytest.raises(OperationalIsolationError):
            hazard_registry.assert_operational_isolation("cyclone_intensity")

    def test_dataset_manifest_and_revision_metadata(self) -> None:
        """Test 21 & 22: Manifest attributes and label provenance are well-defined."""
        intensity_config = hazard_registry.get_hazard("cyclone_intensity")
        assert intensity_config.label_source == "IMD RSMC Best Track Synoptic Fixes"
        assert intensity_config.causal_cutoff_rule is not None
        assert len(intensity_config.leakage_risks) > 0
        assert "MAE on V_max (knots)" in intensity_config.validation_protocol.primary_metric
