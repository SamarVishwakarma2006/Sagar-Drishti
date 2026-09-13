"""
Sagar-Drishti — Prospective / Longitudinal Shadow Evaluation Service for Cyclone Intensity V1
Evaluates the frozen research model (EXP-E XGBoost, 29 features) prospectively on cyclone observations.

CRITICAL SCIENTIFIC & CAUSAL CONTROLS:
1. Pure forward-looking evaluation: Zero model retraining, zero hyperparameter retuning, zero weight modification.
2. Dual Causal Availability Firewall: Enforces observation_timestamp <= T AND data_available_timestamp <= T.
   Filesystem mtime/ctime is strictly forbidden as a proxy for scientific data availability.
3. Three Explicit Evaluation Modes:
   - TRUE_PROSPECTIVE: Real-time forward generation on incoming observations.
   - AVAILABILITY_TIMESTAMP_REPLAY: Historical observations replayed strictly according to documented release times.
   - HISTORICAL_BACKTEST: Evaluation using assembled historical data without original release timings.
   Replay data must NEVER be described as prospective or live scientific validation.
4. Target Lifecycle:
   - PREDICTION_GENERATED -> TARGET_PENDING -> TARGET_AVAILABLE -> EVALUATED
   - Terminal branch: TARGET_PENDING -> TARGET_UNAVAILABLE (when target window expires without valid fix).
   - TARGET_UNAVAILABLE != MODEL FAILURE (zero contribution to MAE/RMSE/bias).
5. Immutable Forecast Logging: 17-field append-only schema in forecast_log.parquet.
6. Target Revision Isolation: Distinguishes PROSPECTIVE_INITIAL from RETROSPECTIVE_REVISED without mutating initial records.
7. Effect-Size Drift Monitoring: Compares prospective distributions against frozen TRAINING reference population (2016-2021).
8. Strict Quarantine: Historical 2024-2026 test set is never accessed.
"""

import os
import sys
import json
import hashlib
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("sagar_drishti.prospective_evaluator")

# -----------------------------------------------------------------------------
# CONSTANTS & FROZEN CRYPTOGRAPHIC HASHES
# -----------------------------------------------------------------------------

FROZEN_MODEL_SHA256 = "3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f"
FROZEN_FEATURE_CONTRACT_SHA256 = "258690562dc9a79589ddfabd6956feb4b36726f2a9bcb093177ea7e0dd26f896"
FROZEN_PREPROCESSING_SHA256 = "591143af5765a2680612b13572b3ec0a058d0e478547bbc0482f027f8f4cbe62"
FROZEN_DATASET_SHA256 = "8bcb247a5b5efb77005704b793c019ce2643c38852ab880881b20845dff25cd0"
MODEL_VERSION = "SD-INTENSITY-EXP-E-V1.0"

# Authoritative 29 features required by EXP-E (24 base + 5 spatial contrasts)
FROZEN_29_FEATURES = [
    "vmax_current", "dvmax_6h", "dvmax_12h", "dvmax_24h", "pc_current", "dpc_6h",
    "translation_speed_kts", "latitude_current", "longitude_current",
    "vws_env_mean_200_800km", "vws_env_min_200_800km", "vws_core_mean_0_100km",
    "vort_core_mean_0_100km", "vort_core_max_0_100km", "vort_env_mean_200_800km",
    "rh_700_env_mean_200_800km", "rh_700_env_min_200_800km", "rh_700_core_mean_0_100km",
    "rh_500_env_mean_200_800km", "rh_500_core_mean_0_100km",
    "sst_core_mean_0_100km", "sst_env_mean_200_800km", "mld_core_mean_0_100km", "sla_core_mean_0_100km",
    "delta_vws_core_minus_env", "delta_vort_core_minus_env", "delta_rh700_core_minus_env",
    "delta_rh500_core_minus_env", "delta_sst_core_minus_env"
]

MAX_OCEAN_AGE_HOURS = 48.0
TARGET_WINDOW_MIN_HOURS = 21.0
TARGET_WINDOW_MAX_HOURS = 27.0


# -----------------------------------------------------------------------------
# ENUMS & STATUS CODES
# -----------------------------------------------------------------------------

class EvaluationMode(str, Enum):
    TRUE_PROSPECTIVE = "TRUE_PROSPECTIVE"
    AVAILABILITY_TIMESTAMP_REPLAY = "AVAILABILITY_TIMESTAMP_REPLAY"
    HISTORICAL_BACKTEST = "HISTORICAL_BACKTEST"


class PredictionStatus(str, Enum):
    PREDICTION_GENERATED = "PREDICTION_GENERATED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    CAUSAL_REJECTED = "CAUSAL_REJECTED"


class TargetStatus(str, Enum):
    TARGET_PENDING = "TARGET_PENDING"
    TARGET_AVAILABLE = "TARGET_AVAILABLE"
    TARGET_UNAVAILABLE = "TARGET_UNAVAILABLE"


class EvaluationType(str, Enum):
    PROSPECTIVE_INITIAL = "PROSPECTIVE_INITIAL"
    RETROSPECTIVE_REVISED = "RETROSPECTIVE_REVISED"


class FrameworkStatus(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class ProspectiveEvidenceStatus(str, Enum):
    INSUFFICIENT_PROSPECTIVE_EVIDENCE = "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    PROMISING_PROSPECTIVE_EVIDENCE = "PROMISING_PROSPECTIVE_EVIDENCE"
    PROSPECTIVE_EVIDENCE_SUPPORTS_GENERALIZATION = "PROSPECTIVE_EVIDENCE_SUPPORTS_GENERALIZATION"
    PROSPECTIVE_EVIDENCE_INCONCLUSIVE = "PROSPECTIVE_EVIDENCE_INCONCLUSIVE"


# -----------------------------------------------------------------------------
# EXCEPTIONS
# -----------------------------------------------------------------------------

class CausalFirewallViolationError(Exception):
    """Raised when any predictor or observation violates temporal/availability causality."""
    pass


class TargetLeakageError(Exception):
    """Raised when target variables contaminate the prospective feature matrix."""
    pass


class ModelHashMismatchError(Exception):
    """Raised when frozen model artifact SHA-256 does not match authoritative manifest."""
    pass


class TestQuarantineViolationError(Exception):
    """Raised when any prospective routine queries the historical 2024-2026 test partition."""
    __test__ = False


class InvalidEvaluationModeError(Exception):
    """Raised when evaluation mode is invalid or mislabeled."""
    pass


# -----------------------------------------------------------------------------
# CORE PROSPECTIVE EVALUATION SERVICE
# -----------------------------------------------------------------------------

class CycloneIntensityProspectiveEvaluator:
    """
    Prospective shadow evaluator for the frozen research Cyclone Intensity model.
    Enforces dual-timestamp causal availability firewalls, append-only provenance logging,
    complete target lifecycle tracking, and effect-size drift monitoring.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.AVAILABILITY_TIMESTAMP_REPLAY,
        db_dir: Optional[Union[str, Path]] = None
    ):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
        self.evaluation_mode = evaluation_mode if isinstance(evaluation_mode, EvaluationMode) else EvaluationMode(evaluation_mode)
        
        self.model_path = self.project_root / "research" / "cyclone_intensity" / "models" / "final_intensity_model.joblib"
        self.contract_path = self.project_root / "research" / "cyclone_intensity" / "feature_contract.json"
        self.historical_dataset_path = self.project_root / "research" / "cyclone_intensity" / "features_6hourly_candidate.parquet"
        
        self.db_dir = Path(db_dir) if db_dir else self.project_root / "research" / "cyclone_intensity" / "prospective"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._training_reference_df = None
        self._audit_log_path = self.db_dir / "causal_audit.log"
        
        # Verify model artifact cryptographic hash upon initialization
        self.verify_model_integrity()

    def _log_causal_event(self, message: str, level: str = "INFO"):
        """Appends a timestamped causal audit entry."""
        ts = datetime.now(timezone.utc).isoformat()
        entry = f"{ts} [{level}] [{self.evaluation_mode.value}] {message}\n"
        with open(self._audit_log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def verify_model_integrity(self) -> str:
        """Verifies the exact byte SHA-256 hash of the frozen model artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Frozen model artifact not found at {self.model_path}")
        
        actual_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if actual_hash != FROZEN_MODEL_SHA256:
            msg = f"CRITICAL BLOCKER: Frozen model hash mismatch: {actual_hash} != {FROZEN_MODEL_SHA256}"
            self._log_causal_event(msg, "CRITICAL")
            raise ModelHashMismatchError(msg)
        
        # Also verify feature contract hash
        if self.contract_path.exists():
            fc_hash = hashlib.sha256(self.contract_path.read_bytes()).hexdigest()
            if fc_hash != FROZEN_FEATURE_CONTRACT_SHA256:
                raise ModelHashMismatchError(f"Feature contract hash mismatch: {fc_hash} != {FROZEN_FEATURE_CONTRACT_SHA256}")

        self._log_causal_event(f"Verified model SHA-256: {actual_hash} [MATCH]")
        return actual_hash

    @property
    def model(self):
        """Loads and returns the frozen model artifact in read-only mode."""
        if self._model is None:
            self.verify_model_integrity()
            self._model = joblib.load(self.model_path)
        return self._model

    def get_training_reference(self) -> pd.DataFrame:
        """Loads strictly the historical TRAINING partition (2016-2021) as drift reference."""
        if self._training_reference_df is None:
            if not self.historical_dataset_path.exists():
                raise FileNotFoundError(f"Candidate historical dataset missing: {self.historical_dataset_path}")
            df = pd.read_parquet(self.historical_dataset_path)
            # Strictly filter to TRAIN only. NEVER touch TEST.
            train_df = df[df["partition"] == "TRAIN"].copy()
            self._training_reference_df = train_df
        return self._training_reference_df

    def enforce_dual_timestamp_firewall(
        self,
        observation_timestamp: Union[str, pd.Timestamp, datetime],
        data_available_timestamp: Union[str, pd.Timestamp, datetime],
        forecast_origin_timestamp: Union[str, pd.Timestamp, datetime],
        source_name: str = "feature"
    ):
        """
        Correction 1: True Data Availability Firewall.
        Enforces BOTH:
          observation_timestamp <= forecast_origin_timestamp
          AND
          data_available_timestamp <= forecast_origin_timestamp
        Prohibits filesystem modification times (mtime, ctime).
        """
        def to_utc(ts):
            if isinstance(ts, str):
                # Detect forbidden filesystem time usage
                if ts.lower().startswith("file_mtime") or ts.lower().startswith("mtime"):
                    raise CausalFirewallViolationError("Filesystem mtime is strictly forbidden as data availability proof.")
                dt = pd.to_datetime(ts)
            elif isinstance(ts, pd.Timestamp):
                dt = ts
            elif isinstance(ts, datetime):
                dt = pd.Timestamp(ts)
            else:
                raise ValueError(f"Unsupported timestamp type: {type(ts)}")
            if dt.tz is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")
            return dt

        t_obs = to_utc(observation_timestamp)
        t_avail = to_utc(data_available_timestamp)
        t_origin = to_utc(forecast_origin_timestamp)

        # 1. Observation timestamp must not be in the future
        if t_obs > t_origin:
            msg = f"Causal violation: {source_name} observation timestamp {t_obs.isoformat()} > forecast origin {t_origin.isoformat()}"
            self._log_causal_event(msg, "ERROR")
            raise CausalFirewallViolationError(msg)

        # 2. Data available timestamp must not be in the future
        if t_avail > t_origin:
            msg = f"Availability violation: {source_name} was observed at {t_obs.isoformat()} but published at {t_avail.isoformat()} > origin {t_origin.isoformat()}"
            self._log_causal_event(msg, "ERROR")
            raise CausalFirewallViolationError(msg)

        return True

    def generate_forecast(
        self,
        system_id: str,
        forecast_origin_timestamp: str,
        features: Dict[str, Any],
        observation_timestamp: str,
        data_available_timestamp: str,
        ocean_source_timestamp: Optional[str] = None,
        ocean_source_available_timestamp: Optional[str] = None,
        ocean_age_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates a prospective intensity prediction at origin T using the frozen model.
        Enforces causal availability, schema contracts, and returns the 17-field record.
        """
        # 1. Check target leakage
        for k in features.keys():
            if k.startswith("target_"):
                raise TargetLeakageError(f"Target column '{k}' detected in input feature dictionary!")

        # 2. Enforce Dual Timestamp Availability Firewall (Correction 1)
        self.enforce_dual_timestamp_firewall(
            observation_timestamp=observation_timestamp,
            data_available_timestamp=data_available_timestamp,
            forecast_origin_timestamp=forecast_origin_timestamp,
            source_name="cyclone_observation"
        )

        # 3. Ocean availability verification (Correction 8)
        if ocean_source_timestamp and ocean_source_available_timestamp:
            self.enforce_dual_timestamp_firewall(
                observation_timestamp=ocean_source_timestamp,
                data_available_timestamp=ocean_source_available_timestamp,
                forecast_origin_timestamp=forecast_origin_timestamp,
                source_name="copernicus_ocean"
            )

        # Calculate forecast valid time = T + 24h
        t_origin = pd.to_datetime(forecast_origin_timestamp)
        if t_origin.tz is None:
            t_origin = t_origin.tz_localize("UTC")
        t_valid = t_origin + pd.Timedelta(hours=24)

        # Calculate radial contrasts if base features are provided
        feat_dict = dict(features)
        if "delta_vws_core_minus_env" not in feat_dict and "vws_core_mean_0_100km" in feat_dict and "vws_env_mean_200_800km" in feat_dict:
            feat_dict["delta_vws_core_minus_env"] = feat_dict["vws_core_mean_0_100km"] - feat_dict["vws_env_mean_200_800km"]
        if "delta_vort_core_minus_env" not in feat_dict and "vort_core_mean_0_100km" in feat_dict and "vort_env_mean_200_800km" in feat_dict:
            feat_dict["delta_vort_core_minus_env"] = feat_dict["vort_core_mean_0_100km"] - feat_dict["vort_env_mean_200_800km"]
        if "delta_rh700_core_minus_env" not in feat_dict and "rh_700_core_mean_0_100km" in feat_dict and "rh_700_env_mean_200_800km" in feat_dict:
            feat_dict["delta_rh700_core_minus_env"] = feat_dict["rh_700_core_mean_0_100km"] - feat_dict["rh_700_env_mean_200_800km"]
        if "delta_rh500_core_minus_env" not in feat_dict and "rh_500_core_mean_0_100km" in feat_dict and "rh_500_env_mean_200_800km" in feat_dict:
            feat_dict["delta_rh500_core_minus_env"] = feat_dict["rh_500_core_mean_0_100km"] - feat_dict["rh_500_env_mean_200_800km"]
        if "delta_sst_core_minus_env" not in feat_dict and "sst_core_mean_0_100km" in feat_dict and "sst_env_mean_200_800km" in feat_dict:
            feat_dict["delta_sst_core_minus_env"] = feat_dict["sst_core_mean_0_100km"] - feat_dict["sst_env_mean_200_800km"]

        # Build feature vector in exact frozen order
        missing_keys = [k for k in FROZEN_29_FEATURES if k not in feat_dict or pd.isna(feat_dict[k])]
        row_values = [feat_dict.get(k, np.nan) for k in FROZEN_29_FEATURES]
        feature_completeness = round(1.0 - (len(missing_keys) / len(FROZEN_29_FEATURES)), 4)
        
        df_x = pd.DataFrame([row_values], columns=FROZEN_29_FEATURES)
        
        # Frozen model inference (native NaN handling in XGBoost)
        pred_val = float(self.model.predict(df_x)[0])
        
        # Enforce physical clipping bounds [15.0, 165.0] kt
        pred_clipped = round(float(np.clip(pred_val, 15.0, 165.0)), 2)

        # Build 17-field forecast record (Correction 7 & Fix 1)
        record = {
            "system_id": system_id,
            "forecast_origin_timestamp": t_origin.isoformat(),
            "forecast_valid_time": t_valid.isoformat(),
            "forecast_vmax_24h": pred_clipped,
            "model_version": MODEL_VERSION,
            "model_hash": FROZEN_MODEL_SHA256,
            "feature_contract_hash": FROZEN_FEATURE_CONTRACT_SHA256,
            "preprocessing_hash": FROZEN_PREPROCESSING_SHA256,
            "feature_timestamp_cutoff": observation_timestamp,
            "data_availability_timestamp": data_available_timestamp,
            "ocean_source_timestamp": ocean_source_timestamp or "UNAVAILABLE",
            "ocean_source_available_timestamp": ocean_source_available_timestamp or "UNAVAILABLE",
            "ocean_age_hours": float(ocean_age_hours) if ocean_age_hours is not None else np.nan,
            "feature_completeness": feature_completeness,
            "missingness_flags": json.dumps(missing_keys),
            "prediction_status": PredictionStatus.PREDICTION_GENERATED.value,
            "evaluation_mode": self.evaluation_mode.value
        }

        return record

    def log_forecast(self, forecast_record: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """
        Appends the 17-field forecast record to forecast_log.parquet in an append-only,
        immutable fashion. Rejects duplicate submissions for (system_id, forecast_origin_timestamp).
        """
        target_path = filepath or (self.db_dir / "forecast_log.parquet")
        
        # Enforce 17-field schema contract
        expected_fields = [
            "system_id", "forecast_origin_timestamp", "forecast_valid_time", "forecast_vmax_24h",
            "model_version", "model_hash", "feature_contract_hash", "preprocessing_hash",
            "feature_timestamp_cutoff", "data_availability_timestamp", "ocean_source_timestamp",
            "ocean_source_available_timestamp", "ocean_age_hours", "feature_completeness",
            "missingness_flags", "prediction_status", "evaluation_mode"
        ]
        
        for field in expected_fields:
            if field not in forecast_record:
                raise ValueError(f"Forecast record missing required schema field: {field}")
        
        df_new = pd.DataFrame([forecast_record])
        
        if target_path.exists():
            existing_df = pd.read_parquet(target_path)
            # Duplicate prevention
            duplicate_mask = (
                (existing_df["system_id"] == forecast_record["system_id"]) &
                (existing_df["forecast_origin_timestamp"] == forecast_record["forecast_origin_timestamp"])
            )
            if duplicate_mask.any():
                logger.warning(f"Duplicate forecast detected for {forecast_record['system_id']} at {forecast_record['forecast_origin_timestamp']}. Skipping append.")
                return target_path
            
            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            combined_df = df_new
            
        combined_df.to_parquet(target_path, index=False)
        self._log_causal_event(f"Appended forecast record for {forecast_record['system_id']} at {forecast_record['forecast_origin_timestamp']} to {target_path.name}")
        return target_path

    def match_target(
        self,
        forecast_record: Dict[str, Any],
        candidate_fixes: List[Dict[str, Any]],
        evaluation_type: EvaluationType = EvaluationType.PROSPECTIVE_INITIAL,
        current_clock_utc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Matches an observation within [T+21h, T+27h] under the target lifecycle state machine.
        If deadline T+27h has passed and no fix is obtainable, enters TARGET_UNAVAILABLE (terminal).
        """
        t_origin = pd.to_datetime(forecast_record["forecast_origin_timestamp"])
        t_target_nominal = t_origin + pd.Timedelta(hours=24)
        t_min = t_origin + pd.Timedelta(hours=TARGET_WINDOW_MIN_HOURS)
        t_max = t_origin + pd.Timedelta(hours=TARGET_WINDOW_MAX_HOURS)

        now_utc = pd.to_datetime(current_clock_utc) if current_clock_utc else pd.Timestamp.now(tz="UTC")

        # Find eligible fixes within window [T+21h, T+27h]
        eligible_fixes = []
        for fix in candidate_fixes:
            if fix.get("system_id") != forecast_record["system_id"]:
                continue
            fix_time = pd.to_datetime(fix["observation_timestamp"])
            if fix_time.tz is None:
                fix_time = fix_time.tz_localize("UTC")
            if t_min <= fix_time <= t_max:
                vmax = fix.get("max_wind_kts") or fix.get("vmax_current")
                if vmax is not None and not pd.isna(vmax) and vmax >= 15.0:
                    delta_nominal = abs((fix_time - t_target_nominal).total_seconds())
                    eligible_fixes.append((delta_nominal, fix_time, float(vmax), fix))

        # Lifecycle resolution (Correction 3 & Fix 2)
        if eligible_fixes:
            # Pick closest fix to nominal T+24h
            eligible_fixes.sort(key=lambda x: x[0])
            best = eligible_fixes[0]
            target_fix_time = best[1]
            observed_vmax = best[2]
            fix_meta = best[3]
            offset_hours = round((target_fix_time - t_target_nominal).total_seconds() / 3600.0, 2)

            return {
                "system_id": forecast_record["system_id"],
                "forecast_origin_timestamp": forecast_record["forecast_origin_timestamp"],
                "forecast_valid_time": forecast_record["forecast_valid_time"],
                "target_status": TargetStatus.TARGET_AVAILABLE.value,
                "target_unavailable_reason": None,
                "actual_target_timestamp": target_fix_time.isoformat(),
                "target_offset_hours": offset_hours,
                "observed_vmax_24h": observed_vmax,
                "target_source": fix_meta.get("source", "IMD_SYNOPTIC"),
                "target_source_version": fix_meta.get("source_version", "OPERATIONAL_V1"),
                "target_available_timestamp": fix_meta.get("data_available_timestamp", target_fix_time.isoformat()),
                "target_retrieved_at": now_utc.isoformat(),
                "evaluation_type": evaluation_type.value,
                "evaluation_mode": self.evaluation_mode.value
            }
        
        # If no eligible fix found, check if deadline T+27h has passed
        if now_utc > t_max:
            # Terminal state TARGET_UNAVAILABLE
            return {
                "system_id": forecast_record["system_id"],
                "forecast_origin_timestamp": forecast_record["forecast_origin_timestamp"],
                "forecast_valid_time": forecast_record["forecast_valid_time"],
                "target_status": TargetStatus.TARGET_UNAVAILABLE.value,
                "target_unavailable_reason": "NO_ELIGIBLE_FIX",
                "actual_target_timestamp": None,
                "target_offset_hours": None,
                "observed_vmax_24h": None,
                "target_source": None,
                "target_source_version": None,
                "target_available_timestamp": None,
                "target_retrieved_at": now_utc.isoformat(),
                "evaluation_type": evaluation_type.value,
                "evaluation_mode": self.evaluation_mode.value
            }
        else:
            # Window still open: TARGET_PENDING
            return {
                "system_id": forecast_record["system_id"],
                "forecast_origin_timestamp": forecast_record["forecast_origin_timestamp"],
                "forecast_valid_time": forecast_record["forecast_valid_time"],
                "target_status": TargetStatus.TARGET_PENDING.value,
                "target_unavailable_reason": None,
                "actual_target_timestamp": None,
                "target_offset_hours": None,
                "observed_vmax_24h": None,
                "target_source": None,
                "target_source_version": None,
                "target_available_timestamp": None,
                "target_retrieved_at": now_utc.isoformat(),
                "evaluation_type": evaluation_type.value,
                "evaluation_mode": self.evaluation_mode.value
            }

    def log_target_arrival(self, target_record: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """
        Appends target arrival records to target_arrival_log.parquet.
        Distinguishes PROSPECTIVE_INITIAL from RETROSPECTIVE_REVISED (Correction 4).
        """
        target_path = filepath or (self.db_dir / "target_arrival_log.parquet")
        df_new = pd.DataFrame([target_record])

        if target_path.exists():
            existing_df = pd.read_parquet(target_path)
            # Revisions do not overwrite: they append with evaluation_type == RETROSPECTIVE_REVISED
            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            combined_df = df_new

        combined_df.to_parquet(target_path, index=False)
        return target_path

    def compute_prospective_metrics(
        self,
        forecast_df: pd.DataFrame,
        target_df: pd.DataFrame,
        baseline_v0_dict: Optional[Dict[str, float]] = None,
        baseline_trend_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates paired forecasts and targets.
        Strictly filters to target_status == 'TARGET_AVAILABLE' and evaluation_type == 'PROSPECTIVE_INITIAL'.
        TARGET_UNAVAILABLE records contribute zero error to MAE/RMSE/bias.
        """
        # Filter to available targets
        valid_targets = target_df[
            (target_df["target_status"] == TargetStatus.TARGET_AVAILABLE.value) &
            (target_df["evaluation_type"] == EvaluationType.PROSPECTIVE_INITIAL.value) &
            (target_df["observed_vmax_24h"].notna())
        ].copy()

        merged = pd.merge(
            forecast_df,
            valid_targets,
            on=["system_id", "forecast_origin_timestamp"],
            suffixes=("_fc", "_tgt")
        )

        n_forecasts_total = len(forecast_df)
        n_targets_available = len(merged)
        n_targets_pending = len(target_df[target_df["target_status"] == TargetStatus.TARGET_PENDING.value])
        n_targets_unavailable = len(target_df[target_df["target_status"] == TargetStatus.TARGET_UNAVAILABLE.value])
        n_storms = merged["system_id"].nunique()

        if n_targets_available == 0:
            return {
                "n_storms": n_storms,
                "n_forecasts_total": n_forecasts_total,
                "n_targets_available": 0,
                "n_targets_pending": n_targets_pending,
                "n_targets_unavailable": n_targets_unavailable,
                "row_mae": None,
                "row_rmse": None,
                "row_bias": None,
                "storm_mae_mean": None,
                "persistence_skill_pct": None,
                "trend_skill_pct": None,
                "evaluation_mode": self.evaluation_mode.value,
                "note": "Zero evaluated targets available."
            }

        y_true = merged["observed_vmax_24h"].values
        y_pred = merged["forecast_vmax_24h"].values
        system_ids = merged["system_id"].values

        # Absolute errors and bias: Bias = mean(y_pred - y_true)
        errors = np.abs(y_true - y_pred)
        row_mae = float(np.mean(errors))
        row_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        row_bias = float(np.mean(y_pred - y_true))

        # Storm-level aggregation
        df_eval = pd.DataFrame({"system_id": system_ids, "error": errors})
        storm_maes = df_eval.groupby("system_id")["error"].mean()
        storm_mae_mean = float(storm_maes.mean())

        # Baseline skills if reference baseline dicts are provided
        pers_skill = None
        trend_skill = None
        if baseline_v0_dict and baseline_trend_dict:
            pers_errors = []
            trend_errors = []
            for _, row in merged.iterrows():
                key = f"{row['system_id']}_{row['forecast_origin_timestamp']}"
                v0 = baseline_v0_dict.get(key, row["observed_vmax_24h"])
                trend = baseline_trend_dict.get(key, row["observed_vmax_24h"])
                pers_errors.append(abs(row["observed_vmax_24h"] - v0))
                trend_errors.append(abs(row["observed_vmax_24h"] - trend))
            
            pers_mae = float(np.mean(pers_errors)) if pers_errors else 0.0
            trend_mae = float(np.mean(trend_errors)) if trend_errors else 0.0
            
            pers_skill = round((1.0 - (row_mae / pers_mae)) * 100.0, 2) if pers_mae > 0 else 0.0
            trend_skill = round((1.0 - (row_mae / trend_mae)) * 100.0, 2) if trend_mae > 0 else 0.0

        return {
            "n_storms": n_storms,
            "n_forecasts_total": n_forecasts_total,
            "n_targets_available": n_targets_available,
            "n_targets_pending": n_targets_pending,
            "n_targets_unavailable": n_targets_unavailable,
            "row_mae": round(row_mae, 3),
            "row_rmse": round(row_rmse, 3),
            "row_bias": round(row_bias, 3),
            "storm_mae_mean": round(storm_mae_mean, 3),
            "persistence_skill_pct": pers_skill,
            "trend_skill_pct": trend_skill,
            "evaluation_mode": self.evaluation_mode.value
        }

    def compute_distribution_drift(self, prospective_features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Correction 5: Effect-size drift monitoring against the frozen TRAINING reference population.
        Reports Wasserstein distance, standardized mean difference (Cohen's d), and missingness shift.
        Does NOT rely solely on p-values. Sets FLAG_FOR_FUTURE_RESEARCH on notable shifts.
        """
        train_ref = self.get_training_reference()
        drift_records = []

        for col in FROZEN_29_FEATURES:
            if col not in prospective_features_df.columns:
                continue

            ref_series = train_ref[col].dropna()
            pros_series = prospective_features_df[col].dropna()

            n_ref = len(ref_series)
            n_pros = len(pros_series)

            if n_pros < 5:
                drift_records.append({
                    "variable": col,
                    "reference_sample_size": n_ref,
                    "prospective_sample_size": n_pros,
                    "effect_size_cohens_d": None,
                    "wasserstein_distance": None,
                    "missingness_rate_prospective": round(float(prospective_features_df[col].isna().mean()), 4),
                    "interpretation": "INSUFFICIENT_SAMPLE (< 5 fixes)",
                    "flag": "MONITORING_ONLY"
                })
                continue

            # Standardized mean difference (Cohen's d)
            mean_diff = float(pros_series.mean() - ref_series.mean())
            pooled_std = float(np.sqrt((ref_series.std() ** 2 + pros_series.std() ** 2) / 2.0))
            cohens_d = round(mean_diff / pooled_std, 3) if pooled_std > 0 else 0.0

            # Wasserstein distance (L1 earth mover's distance)
            u = np.sort(ref_series.values)
            v = np.sort(pros_series.values)
            # Approximate empirical wasserstein
            w_dist = round(float(np.mean(np.abs(np.percentile(u, np.linspace(0, 100, 50)) - np.percentile(v, np.linspace(0, 100, 50))))), 3)

            # Interpretation based on effect size
            if abs(cohens_d) > 0.8:
                interp = f"LARGE_DISTRIBUTION_SHIFT (Cohen's d = {cohens_d})"
                flag = "FLAG_FOR_FUTURE_RESEARCH"
            elif abs(cohens_d) > 0.4:
                interp = f"MODERATE_DISTRIBUTION_SHIFT (Cohen's d = {cohens_d})"
                flag = "FLAG_FOR_FUTURE_RESEARCH"
            else:
                interp = f"STABLE_DISTRIBUTION (Cohen's d = {cohens_d})"
                flag = "NORMAL"

            drift_records.append({
                "variable": col,
                "reference_sample_size": n_ref,
                "prospective_sample_size": n_pros,
                "effect_size_cohens_d": cohens_d,
                "wasserstein_distance": w_dist,
                "missingness_rate_prospective": round(float(prospective_features_df[col].isna().mean()), 4),
                "interpretation": interp,
                "flag": flag
            })

        return drift_records
