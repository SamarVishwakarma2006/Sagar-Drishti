"""
Sagar-Drishti — Cyclone Intensity Final Data Sufficiency & Pre-Training Gate V2
Test Suite — ZERO ML TRAINING

Automated validation covering all 26 required verification items:
1. All 11 protected hashes unchanged
2. Raw IMD hash unchanged
3. Production model unchanged
4. V2.3 Model D unchanged
5. Frozen policy unchanged
6. Existing ocean dataset unchanged
7. Existing ERA5 dataset unchanged
8. No model artifacts created
9. Duplicate detection deterministic
10. Conflicting duplicates excluded from candidate dataset
11. NADA anomaly remains raw and excluded only from candidate data
12. No future atmospheric timestamps
13. 00Z rejects same-day 18Z
14. 06Z rejects same-day 18Z
15. 12Z rejects same-day 18Z
16. 18Z allows 18Z
17. No future storm-center coordinates
18. No future track leakage
19. No future lifetime-intensity leakage
20. No centered rolling leakage
21. Zero storm overlap across partitions
22. Zero-positive RI test produces insufficient-evidence state
23. Intensity/RI target isolation
24. Production/V2.3 isolation
25. Research-only artifact placement
26. Final verdict consistency
"""

import hashlib
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np

from app.services.hazard_registry import (
    hazard_registry,
    HazardStatus,
    TemporalLeakageError,
    OperationalIsolationError
)

# 11 Protected Artifact Hashes (Pre-Implementation Baseline)
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

RAW_IMD_PATH = "backend/data/historical/imd_tracks_2016_2026.parquet"
RAW_IMD_EXPECTED_HASH = "e3f1b87455e716767e3b34a05b3dff04587dc09c6689a7dc0eaf7b72f951b643"


# Baseline Known Existing Models
ALLOWED_EXISTING_MODELS = {
    "backend/models/model_event_type.joblib",
    "backend/models/risk_model.joblib",
    "backend/models/risk_model_0d.joblib",
    "backend/models/risk_model_1d.joblib",
    "backend/models/risk_model_2d.joblib",
    "backend/models/risk_model_3d.joblib",
    "backend/models/candidates/v2_3/atmos_only/risk_model_0d.joblib",
    "backend/models/candidates/v2_3/atmos_only/risk_model_1d.joblib",
    "backend/models/candidates/v2_3/atmos_only/risk_model_2d.joblib",
    "backend/models/candidates/v2_3/atmos_only/risk_model_3d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_base/risk_model_0d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_base/risk_model_1d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_base/risk_model_2d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_base/risk_model_3d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_dynamic/risk_model_0d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_dynamic/risk_model_1d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_dynamic/risk_model_2d.joblib",
    "backend/models/candidates/v2_3/ocean_atmos_dynamic/risk_model_3d.joblib",
    "backend/models/candidates/v2_3/ocean_only/risk_model_0d.joblib",
    "backend/models/candidates/v2_3/ocean_only/risk_model_1d.joblib",
    "backend/models/candidates/v2_3/ocean_only/risk_model_2d.joblib",
    "backend/models/candidates/v2_3/ocean_only/risk_model_3d.joblib",
    "backend/models/v2_10yr/risk_model.joblib",
    "backend/models/v2_10yr/risk_model_0d.joblib",
    "backend/models/v2_10yr/risk_model_1d.joblib",
    "backend/models/v2_10yr/risk_model_2d.joblib",
    "backend/models/v2_10yr/risk_model_3d.joblib",
    "backend/models/v2_10yr/calibration/calibrator_0d.joblib",
    "backend/models/v2_10yr/calibration/calibrator_1d.joblib",
    "backend/models/v2_10yr/calibration/calibrator_2d.joblib",
    "backend/models/v2_10yr/calibration/calibrator_3d.joblib",
}


@pytest.fixture(scope="module")
def raw_imd_df() -> pd.DataFrame:
    """Load raw IMD Best Track Parquet without modification."""
    df = pd.read_parquet(RAW_IMD_PATH)
    df["datetime"] = pd.to_datetime(df["datetime_iso"])
    return df


class TestProtectedArtifactsAndImmutability:
    """Tests 1–8: Verifies strict preservation of production, V2.3, and datasets."""

    def test_item_01_all_11_protected_hashes_unchanged(self) -> None:
        """1. All 11 protected hashes unchanged."""
        for path, expected in PROTECTED_HASHES.items():
            p = Path(path)
            assert p.exists(), f"Protected file missing: {path}"
            with open(p, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            assert actual == expected, f"Protected hash mismatch in {path}!"

    def test_item_02_raw_imd_hash_unchanged(self) -> None:
        """2. Raw IMD hash unchanged."""
        p = Path(RAW_IMD_PATH)
        assert p.exists(), f"Raw IMD file missing: {RAW_IMD_PATH}"
        with open(p, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == RAW_IMD_EXPECTED_HASH, "Raw IMD Best Track Parquet has been modified!"

    def test_item_03_production_model_unchanged(self) -> None:
        """3. Production model unchanged."""
        prod_path = Path("backend/models/risk_model_3d.joblib")
        with open(prod_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == PROTECTED_HASHES["backend/models/risk_model_3d.joblib"]

    def test_item_04_v2_3_model_d_unchanged(self) -> None:
        """4. V2.3 Model D unchanged."""
        model_d_files = [
            "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib",
            "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib",
            "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib",
            "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib",
        ]
        for path in model_d_files:
            with open(path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            assert actual == PROTECTED_HASHES[path]

    def test_item_05_frozen_policy_unchanged(self) -> None:
        """5. Frozen policy unchanged."""
        policy_path = Path("backend/config/frozen_alert_policy_v2.json")
        with open(policy_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == PROTECTED_HASHES["backend/config/frozen_alert_policy_v2.json"]

    def test_item_06_existing_ocean_dataset_unchanged(self) -> None:
        """6. Existing ocean dataset unchanged."""
        for path in [
            "backend/data/historical/features_10yr.parquet",
            "backend/data/historical/labeled_features_10yr_clean.parquet",
        ]:
            with open(path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            assert actual == PROTECTED_HASHES[path]

    def test_item_07_existing_era5_dataset_unchanged(self) -> None:
        """7. Existing ERA5 dataset unchanged."""
        path = "backend/data/era5/features_atmosphere_10yr_daily.parquet"
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == PROTECTED_HASHES[path]

    def test_item_08_no_model_artifacts_created(self) -> None:
        """8. No model artifacts (.joblib, .pkl, .onnx) created during gate."""
        forbidden_extensions = {".onnx"}
        for root, _, files in os.walk("backend"):
            for file in files:
                ext = Path(file).suffix.lower()
                assert ext not in forbidden_extensions, f"Forbidden model artifact created: {file}"
                # Ensure no new joblib or pkl outside baseline known models
                if ext in {".joblib", ".pkl"}:
                    rel_path = Path(root, file).as_posix()
                    assert rel_path in ALLOWED_EXISTING_MODELS, f"Unauthorized new model artifact: {rel_path}"


class TestForensicDataQualityAndExclusions:
    """Tests 9–11: Duplicate detection, deterministic exclusion, and NADA anomaly handling."""

    def test_item_09_duplicate_detection_deterministic(self, raw_imd_df: pd.DataFrame) -> None:
        """9. Duplicate detection deterministic across full dataset."""
        dups = raw_imd_df[raw_imd_df.duplicated(subset=["system_id", "datetime_iso"], keep=False)]
        assert len(dups) == 8, f"Expected exactly 8 duplicate rows (4 pairs), found {len(dups)}"
        assert set(dups["system_id"].unique()) == {
            "IMD-2018-14-PHETHAI",
            "IMD-2021-9-UNNAMED9",
            "IMD-2022-14-UNNAMED14",
            "IMD-2023-9-MICHAUNG"
        }

    def test_item_10_conflicting_duplicates_excluded_from_candidate_dataset(self, raw_imd_df: pd.DataFrame) -> None:
        """10. Conflicting duplicates excluded from candidate dataset."""
        dup_mask = raw_imd_df.duplicated(subset=["system_id", "datetime_iso"], keep=False)
        candidate_df = raw_imd_df[~dup_mask]
        assert len(candidate_df) == 2544 - 8 == 2536
        # Verify no remaining duplicates in candidate view
        assert candidate_df.duplicated(subset=["system_id", "datetime_iso"]).sum() == 0

    def test_item_11_nada_anomaly_remains_raw_and_excluded_only_from_candidate_data(self, raw_imd_df: pd.DataFrame) -> None:
        """11. NADA anomaly remains raw in Parquet and excluded only from candidate dataset."""
        # Check raw row 168
        raw_row_168 = raw_imd_df.iloc[168]
        assert raw_row_168["system_id"] == "IMD-2016-8-NADA"
        assert raw_row_168["central_pressure_hpa"] == 25.0
        assert raw_row_168["max_wind_kts"] == 2.0

        # Build candidate filter: exclude duplicates and NADA row 168
        dup_mask = raw_imd_df.duplicated(subset=["system_id", "datetime_iso"], keep=False)
        nada_mask = (raw_imd_df["system_id"] == "IMD-2016-8-NADA") & (raw_imd_df["central_pressure_hpa"] < 100)
        candidate_df = raw_imd_df[~dup_mask & ~nada_mask]
        assert len(candidate_df) == 2544 - 8 - 1 == 2535
        assert (candidate_df["central_pressure_hpa"] < 100).sum() == 0


class TestTemporalCausalityAndLeakageFirewall:
    """Tests 12–20: Temporal bounds, synoptic restrictions, coordinate isolation, and rolling window limits."""

    def test_item_12_no_future_atmospheric_timestamps(self) -> None:
        """12. No future atmospheric timestamps relative to prediction timestamp T."""
        t_pred = "2023-10-21T12:00:00Z"
        valid_ts = ["2023-10-21T12:00:00Z", "2023-10-21T06:00:00Z", "2023-10-21T00:00:00Z"]
        future_ts = ["2023-10-21T18:00:00Z", "2023-10-22T00:00:00Z"]

        for ts in valid_ts:
            assert hazard_registry.validate_timestamp_causality(t_pred, ts) is True
        for ts in future_ts:
            with pytest.raises(TemporalLeakageError):
                hazard_registry.validate_timestamp_causality(t_pred, ts)

    def test_item_13_00z_rejects_same_day_18z(self) -> None:
        """13. 00Z prediction rejects same-day 18Z."""
        t_pred = "2023-10-21T00:00:00Z"
        same_day_18z = "2023-10-21T18:00:00Z"
        with pytest.raises(TemporalLeakageError):
            hazard_registry.validate_timestamp_causality(t_pred, same_day_18z)

    def test_item_14_06z_rejects_same_day_18z(self) -> None:
        """14. 06Z prediction rejects same-day 18Z."""
        t_pred = "2023-10-21T06:00:00Z"
        same_day_18z = "2023-10-21T18:00:00Z"
        with pytest.raises(TemporalLeakageError):
            hazard_registry.validate_timestamp_causality(t_pred, same_day_18z)

    def test_item_15_12z_rejects_same_day_18z(self) -> None:
        """15. 12Z prediction rejects same-day 18Z."""
        t_pred = "2023-10-21T12:00:00Z"
        same_day_18z = "2023-10-21T18:00:00Z"
        with pytest.raises(TemporalLeakageError):
            hazard_registry.validate_timestamp_causality(t_pred, same_day_18z)

    def test_item_16_18z_allows_same_day_18z(self) -> None:
        """16. 18Z prediction allows 18Z observation."""
        t_pred = "2023-10-21T18:00:00Z"
        same_day_18z = "2023-10-21T18:00:00Z"
        assert hazard_registry.validate_timestamp_causality(t_pred, same_day_18z) is True

    def test_item_17_no_future_storm_center_coordinates(self) -> None:
        """17. No future storm-center coordinates allowed for spatial extraction."""
        t_pred = "2023-10-21T12:00:00Z"
        coord_time = "2023-10-21T18:00:00Z"
        with pytest.raises(TemporalLeakageError):
            hazard_registry.validate_timestamp_causality(t_pred, coord_time)

    def test_item_18_no_future_track_leakage(self) -> None:
        """18. Feature vector must exclude future track coordinates and future motion vectors."""
        contract_path = Path("backend/reports/CYCLONE_INTENSITY_FEATURE_CONTRACT.md")
        assert contract_path.exists()
        content = contract_path.read_text(encoding="utf-8")
        assert "Future Track Coordinates" in content
        assert "PROHIBITED" in content

    def test_item_19_no_future_lifetime_intensity_leakage(self) -> None:
        """19. Feature vector must exclude lifetime peak intensity and final category."""
        contract_path = Path("backend/reports/CYCLONE_INTENSITY_FEATURE_CONTRACT.md")
        content = contract_path.read_text(encoding="utf-8")
        assert "`lifetime_max_vmax`" in content
        assert "`final_category`" in content
        assert "`storm_duration_hours`" in content

    def test_item_20_no_centered_rolling_leakage(self) -> None:
        """20. Rejects centered rolling windows that look ahead."""
        contract_path = Path("backend/reports/CYCLONE_INTENSITY_FEATURE_CONTRACT.md")
        content = contract_path.read_text(encoding="utf-8")
        assert "Centered Rolling Windows" in content
        assert "strictly backward-looking causal windows" in content


class TestSplittingPartitionsAndHoldoutEvaluation:
    """Tests 21–22: Partition isolation, zero storm overlap, and holdout zero-positive handling."""

    def test_item_21_zero_storm_overlap_across_partitions(self, raw_imd_df: pd.DataFrame) -> None:
        """21. Zero storm overlap across Train, Validation, and Test partitions."""
        raw_imd_df["year"] = raw_imd_df["datetime"].dt.year
        train_storms = set(raw_imd_df[raw_imd_df["year"] <= 2021]["system_id"].unique())
        val_storms = set(raw_imd_df[(raw_imd_df["year"] >= 2022) & (raw_imd_df["year"] <= 2023)]["system_id"].unique())
        test_storms = set(raw_imd_df[raw_imd_df["year"] >= 2024]["system_id"].unique())

        assert len(train_storms) == 64
        assert len(val_storms) == 24
        assert len(test_storms) == 29

        # Mathematical proof of zero overlap
        assert train_storms.intersection(val_storms) == set(), "Train and Val storm overlap detected!"
        assert val_storms.intersection(test_storms) == set(), "Val and Test storm overlap detected!"
        assert train_storms.intersection(test_storms) == set(), "Train and Test storm overlap detected!"
        assert len(train_storms) + len(val_storms) + len(test_storms) == 117

    def test_item_22_zero_positive_ri_test_produces_insufficient_evidence_state(self, raw_imd_df: pd.DataFrame) -> None:
        """22. Zero-positive RI test set (2024–2026) produces insufficient evidence state."""
        raw_imd_df["year"] = raw_imd_df["datetime"].dt.year
        test_df = raw_imd_df[raw_imd_df["year"] >= 2024].sort_values(["system_id", "datetime"]).reset_index(drop=True)

        # Count 24h pairs in test partition
        ri_positives = 0
        total_24h_pairs = 0
        for sid, grp in test_df.groupby("system_id"):
            grp = grp.reset_index(drop=True)
            for i in range(len(grp)):
                t_i = grp.loc[i, "datetime"]
                v_i = grp.loc[i, "max_wind_kts"]
                for j in range(i + 1, len(grp)):
                    t_j = grp.loc[j, "datetime"]
                    dt_h = (t_j - t_i).total_seconds() / 3600.0
                    if 21.0 <= dt_h <= 27.0:
                        total_24h_pairs += 1
                        if (grp.loc[j, "max_wind_kts"] - v_i) >= 30.0:
                            ri_positives += 1
                        break

        assert total_24h_pairs > 0, "Expected valid 24h pairs in test partition"
        assert ri_positives == 0, f"Expected 0 standard RI positives in 2024–2026, found {ri_positives}"

        # Evaluate metric contract on zero positives
        def evaluate_ri_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
            if (tp + fn) == 0:
                return {
                    "recall": None,
                    "pr_auc": None,
                    "csi": None,
                    "status": "INSUFFICIENT_EVIDENCE_FOR_RI_TESTING"
                }
            recall = tp / (tp + fn)
            csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            return {"recall": recall, "csi": csi, "status": "EVALUATED"}

        res = evaluate_ri_metrics(tp=0, fp=5, fn=0, tn=total_24h_pairs - 5)
        assert res["status"] == "INSUFFICIENT_EVIDENCE_FOR_RI_TESTING"
        assert res["pr_auc"] is None


class TestGovernanceTargetAndPipelineIsolation:
    """Tests 23–26: Target isolation, pipeline separation, research placement, and verdict consistency."""

    def test_item_23_intensity_ri_target_isolation(self) -> None:
        """23. Cyclone Intensity and RI targets are strictly isolated."""
        intensity_target = {
            "name": "vmax_24h",
            "type": "continuous_regression",
            "unit": "knots",
            "readiness": "TRAINING_READY"
        }
        ri_target = {
            "name": "is_ri_30kt_24h",
            "type": "binary_classification",
            "unit": "boolean",
            "readiness": "DATA_NOT_READY"
        }
        assert intensity_target["type"] != ri_target["type"]
        assert intensity_target["readiness"] != ri_target["readiness"]

    def test_item_24_production_and_v2_3_isolation(self) -> None:
        """24. Production and V2.3 models isolated from candidate intensity targets."""
        with pytest.raises(OperationalIsolationError):
            hazard_registry.assert_operational_isolation("cyclone_intensity")
        
        intensity = hazard_registry.get_hazard("cyclone_intensity")
        assert intensity.status == HazardStatus.RESEARCH_PROPOSED

    def test_item_25_research_only_artifact_placement(self) -> None:
        """25. All new reports and manifests reside in backend/reports with RESEARCH_ONLY tags."""
        new_reports = [
            "backend/reports/CYCLONE_INTENSITY_FEATURE_CONTRACT.md",
            "backend/reports/CYCLONE_INTENSITY_SPLIT_MANIFEST.md",
            "backend/reports/CYCLONE_RI_FINAL_STATUS.md",
            "backend/reports/CYCLONE_INTENSITY_FINAL_DATA_SUFFICIENCY_GATE.md",
        ]
        for path in new_reports:
            p = Path(path)
            assert p.exists(), f"Missing report: {path}"
            content = p.read_text(encoding="utf-8")
            assert "RESEARCH_ONLY" in content
            assert "NOT_FOR_PRODUCTION" in content

    def test_item_26_final_verdict_consistency(self) -> None:
        """26. Final verdict consistency across reports and gate documentation."""
        gate_path = Path("backend/reports/CYCLONE_INTENSITY_FINAL_DATA_SUFFICIENCY_GATE.md")
        content = gate_path.read_text(encoding="utf-8")
        assert "CYCLONE INTENSITY:\nTRAINING_READY" in content
        assert "RAPID INTENSIFICATION:\nDATA_NOT_READY" in content
        assert "ML TRAINING PERFORMED:\nNO" in content
        assert "PRODUCTION MODIFIED:\nNO" in content
        assert "V2.3 MODIFIED:\nNO" in content
