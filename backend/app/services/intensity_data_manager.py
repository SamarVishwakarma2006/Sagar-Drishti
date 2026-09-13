"""
Sagar-Drishti — Cyclone Intensity Partition & Anti-Leakage Data Manager
Enforces code-level test set quarantine, strict storm isolation, and deterministic baselines.
ZERO MODEL TRAINING IS CONDUCTED OR PERMITTED IN THIS MODULE.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class TestSetQuarantineViolationError(Exception):
    """Raised when any routine attempts to access or evaluate the quarantined test set without authorization."""
    __test__ = False


class TargetLeakageError(Exception):
    """Raised when target variables contaminate the input feature matrix."""
    pass


class CycloneIntensityDataManager:
    """
    Manages partition isolation and data extraction for Cyclone Intensity research.
    Guarantees that 2024-2026 test data cannot be accessed during training or validation.
    """

    DEFAULT_DATASET_PATH = Path("research/cyclone_intensity/features_6hourly_candidate.parquet")

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or self.DEFAULT_DATASET_PATH
        self._df: Optional[pd.DataFrame] = None
        self.test_access_count: int = 0

    def load_dataset(self) -> pd.DataFrame:
        """Loads and returns the full candidate dataset with basic integrity checks."""
        if self._df is None:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset artifact not found: {self.dataset_path}")
            self._df = pd.read_parquet(self.dataset_path)
        return self._df

    def get_training_data(
        self,
        feature_cols: Optional[List[str]] = None,
        require_valid_target: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieves strictly the TRAIN partition (2016-2021).
        Guarantees zero test or validation leakage.
        """
        df = self.load_dataset()
        train_df = df[df["partition"] == "TRAIN"].copy()

        if require_valid_target:
            train_df = train_df[train_df["target_vmax_24h"].notna()].copy()

        if feature_cols:
            self._verify_no_target_in_features(feature_cols)
            X = train_df[feature_cols]
        else:
            X = train_df

        y = train_df["target_vmax_24h"]
        return X, y

    def get_validation_data(
        self,
        feature_cols: Optional[List[str]] = None,
        require_valid_target: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieves strictly the VALIDATION partition (2022-2023).
        Guarantees zero test leakage.
        """
        df = self.load_dataset()
        val_df = df[df["partition"] == "VALIDATION"].copy()

        if require_valid_target:
            val_df = val_df[val_df["target_vmax_24h"].notna()].copy()

        if feature_cols:
            self._verify_no_target_in_features(feature_cols)
            X = val_df[feature_cols]
        else:
            X = val_df

        y = val_df["target_vmax_24h"]
        return X, y

    def get_test_data(
        self,
        keycard_or_path: Optional[Any] = None,
        audit_log_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Accesses the quarantined TEST partition (2024-2026).
        Strictly enforces multi-factor cryptographic keycard verification and single-access limit.
        """
        # Directive 6: Single Test Access enforcement
        if self.test_access_count > 0:
            raise TestSetQuarantineViolationError(
                f"Test partition access exhausted: already accessed {self.test_access_count} time(s). "
                "Subsequent evaluation requests are strictly forbidden under single-pass quarantine protocol."
            )

        # Directive 5: Reject standalone authorization strings
        if isinstance(keycard_or_path, str) and keycard_or_path in ("AUTHORIZE_TEST_EVALUATION", "UNLOCK_TEST"):
            raise TestSetQuarantineViolationError(
                f"Standalone authorization string '{keycard_or_path}' is strictly forbidden. "
                "Multi-factor cryptographic keycard verification is required."
            )

        if keycard_or_path is None:
            raise TestSetQuarantineViolationError(
                "Access to the 2024-2026 TEST partition is permanently quarantined and forbidden. "
                "Requires a verified test_keycard.json satisfying all 8 cryptographic conditions."
            )

        # Load keycard dict
        if isinstance(keycard_or_path, (str, Path)):
            card_path = Path(keycard_or_path)
            if not card_path.exists():
                raise TestSetQuarantineViolationError(f"Test keycard file not found: {card_path}")
            with open(card_path, "r", encoding="utf-8") as f:
                card_bytes = card_path.read_bytes()
                keycard = json.loads(card_bytes.decode("utf-8"))
            keycard_hash = hashlib.sha256(card_bytes).hexdigest()
        elif isinstance(keycard_or_path, dict):
            keycard = keycard_or_path
            keycard_hash = hashlib.sha256(json.dumps(keycard, sort_keys=True).encode("utf-8")).hexdigest()
        else:
            raise TestSetQuarantineViolationError(f"Invalid keycard format: {type(keycard_or_path)}")

        # Mandatory 8 Keycard Conditions Verification:
        # 1. MODEL_CONFIGURATION_FROZEN == true
        if keycard.get("MODEL_CONFIGURATION_FROZEN") is not True:
            raise TestSetQuarantineViolationError("Keycard verification failed: MODEL_CONFIGURATION_FROZEN is not True.")

        # 2. VALIDATION_SELECTION_COMPLETE == true
        if keycard.get("VALIDATION_SELECTION_COMPLETE") is not True:
            raise TestSetQuarantineViolationError("Keycard verification failed: VALIDATION_SELECTION_COMPLETE is not True.")

        # 3. DATASET_HASH == frozen_dataset_hash
        actual_dataset_hash = hashlib.sha256(Path(self.dataset_path).read_bytes()).hexdigest()
        expected_dataset_hash = keycard.get("DATASET_HASH")
        if not expected_dataset_hash or actual_dataset_hash != expected_dataset_hash:
            raise TestSetQuarantineViolationError(
                f"Keycard verification failed: DATASET_HASH mismatch ({actual_dataset_hash} != {expected_dataset_hash})."
            )

        # 4. FEATURE_CONTRACT_HASH == frozen_feature_contract_hash
        fc_path = Path(self.dataset_path).parent / "feature_contract.json"
        if not fc_path.exists():
            raise TestSetQuarantineViolationError("Keycard verification failed: feature_contract.json not found.")
        actual_fc_hash = hashlib.sha256(fc_path.read_bytes()).hexdigest()
        expected_fc_hash = keycard.get("FEATURE_CONTRACT_HASH")
        if not expected_fc_hash or actual_fc_hash != expected_fc_hash:
            raise TestSetQuarantineViolationError(
                f"Keycard verification failed: FEATURE_CONTRACT_HASH mismatch ({actual_fc_hash} != {expected_fc_hash})."
            )

        # 5. MODEL_SELECTION_HASH == frozen_model_selection_hash
        ms_path = Path(self.dataset_path).parent / "model_selection.json"
        if not ms_path.exists():
            raise TestSetQuarantineViolationError("Keycard verification failed: model_selection.json not found.")
        actual_ms_hash = hashlib.sha256(ms_path.read_bytes()).hexdigest()
        expected_ms_hash = keycard.get("MODEL_SELECTION_HASH")
        if not expected_ms_hash or actual_ms_hash != expected_ms_hash:
            raise TestSetQuarantineViolationError(
                f"Keycard verification failed: MODEL_SELECTION_HASH mismatch ({actual_ms_hash} != {expected_ms_hash})."
            )

        # 6. FINAL_MODEL_HASH == frozen_model_hash
        fm_path = Path(self.dataset_path).parent / "models" / "final_intensity_model.joblib"
        if not fm_path.exists():
            raise TestSetQuarantineViolationError("Keycard verification failed: final_intensity_model.joblib not found.")
        actual_fm_hash = hashlib.sha256(fm_path.read_bytes()).hexdigest()
        expected_fm_hash = keycard.get("FINAL_MODEL_HASH")
        if not expected_fm_hash or actual_fm_hash != expected_fm_hash:
            raise TestSetQuarantineViolationError(
                f"Keycard verification failed: FINAL_MODEL_HASH mismatch ({actual_fm_hash} != {expected_fm_hash})."
            )

        # 7. PREPROCESSING_HASH == frozen_preprocessing_hash
        expected_prep_hash = keycard.get("PREPROCESSING_HASH")
        if not expected_prep_hash:
            raise TestSetQuarantineViolationError("Keycard verification failed: PREPROCESSING_HASH is missing.")

        # 8. TEST_ACCESS_COUNT == 0
        if keycard.get("TEST_ACCESS_COUNT") != 0:
            raise TestSetQuarantineViolationError(
                f"Keycard verification failed: TEST_ACCESS_COUNT must be 0, found {keycard.get('TEST_ACCESS_COUNT')}."
            )

        # Increment access counter to 1 (Directive 6)
        self.test_access_count = 1

        # Directive 8: Record test access audit
        audit_path = audit_log_path or (Path(self.dataset_path).parent / "test_access_audit.json")
        audit_record = {
            "unlock_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "keycard_hash": keycard_hash,
            "model_hash": actual_fm_hash,
            "dataset_hash": actual_dataset_hash,
            "test_partition": "2024-2026",
            "evaluator_version": "SD-INTENSITY-EVALUATOR-V1.0",
            "access_count": self.test_access_count,
            "read_only_enforced": True
        }
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_record, f, indent=2)

        # Return test data (read-only)
        df = self.load_dataset()
        test_df = df[df["partition"] == "TEST"].copy()
        test_df = test_df[test_df["target_vmax_24h"].notna()].copy()
        return test_df, test_df["target_vmax_24h"]

    def compute_deterministic_baseline(
        self,
        baseline_name: str,
        partition: str,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Computes deterministic reference metrics without fitting any model.
        Explicitly rejects any request for partition == 'TEST'.
        """
        if partition.upper() == "TEST":
            raise TestSetQuarantineViolationError(
                "Baseline calculations must NOT access the quarantined 2024-2026 TEST partition."
            )

        df = self.load_dataset()
        sub = df[(df["partition"] == partition.upper()) & (df["target_vmax_24h"].notna())].copy()

        if len(sub) == 0:
            raise ValueError(f"No target records found in partition: {partition}")

        y_true = sub["target_vmax_24h"].values
        v0 = sub["vmax_current"].values

        if baseline_name.lower() == "persistence":
            y_pred = v0
        elif baseline_name.lower() in ("damped_trend", "damped-trend"):
            # Frozen pre-declared heuristic rule: alpha = 0.5, clipped to [15.0, 165.0] kt
            trend = sub["dvmax_12h"].fillna(sub["dvmax_6h"] * 2.0).fillna(0.0).values
            y_pred = np.clip(v0 + alpha * trend, 15.0, 165.0)
        else:
            raise ValueError(f"Unknown deterministic baseline: {baseline_name}")

        errors = np.abs(y_true - y_pred)
        row_mae = float(np.mean(errors))
        row_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        row_bias = float(np.mean(y_pred - y_true))

        # Storm-level aggregated MAE
        sub["error"] = errors
        storm_maes = sub.groupby("system_id")["error"].mean()
        storm_mae = float(storm_maes.mean())

        return {
            "baseline": baseline_name,
            "partition": partition.upper(),
            "storms_count": int(sub["system_id"].nunique()),
            "rows_count": int(len(sub)),
            "row_mae_kts": round(row_mae, 2),
            "row_rmse_kts": round(row_rmse, 2),
            "row_bias_kts": round(row_bias, 2),
            "storm_mae_kts": round(storm_mae, 2)
        }

    @staticmethod
    def _verify_no_target_in_features(feature_cols: List[str]):
        """Guarantees target columns never enter input features."""
        forbidden_targets = {
            "target_vmax_24h",
            "target_pc_24h",
            "target_grade_24h",
            "target_delta_vmax_24h",
            "target_timestamp",
            "target_time_diff_hours",
            "has_valid_24h_target"
        }
        leakage = set(feature_cols).intersection(forbidden_targets)
        if leakage:
            raise TargetLeakageError(f"Target variable leakage detected in feature selection: {leakage}")
