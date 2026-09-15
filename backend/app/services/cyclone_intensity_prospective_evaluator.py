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

# Authoritative Canonical Research Partition Boundaries
CANONICAL_RESEARCH_SPLITS = {
    "TRAIN": {"start_year": 2016, "end_year": 2021, "span": "2016–2021", "storm_count": 64},
    "VALIDATION": {"start_year": 2022, "end_year": 2023, "span": "2022–2023", "storm_count": 24},
    "TEST": {"start_year": 2024, "end_year": 2026, "span": "2024–2026", "storm_count": 29},
}


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
    TARGET_EXCLUDED = "TARGET_EXCLUDED"


class DataProvenanceClass(str, Enum):
    REAL_PROSPECTIVE = "REAL_PROSPECTIVE"
    SYNTHETIC_TEST_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
    HISTORICAL_REANALYSIS = "HISTORICAL_REANALYSIS"
    UNKNOWN = "UNKNOWN"


class AtmosphericSource(str, Enum):
    ERA5_REANALYSIS = "ERA5_REANALYSIS"
    OPERATIONAL_NWP_ANALYSIS = "OPERATIONAL_NWP_ANALYSIS"
    OTHER_VERIFIED_OPERATIONAL_SOURCE = "OTHER_VERIFIED_OPERATIONAL_SOURCE"


class EvaluationType(str, Enum):
    PROSPECTIVE_INITIAL = "PROSPECTIVE_INITIAL"
    RETROSPECTIVE_REVISED = "RETROSPECTIVE_REVISED"


class FrameworkStatus(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class ProspectiveEvidenceTier(str, Enum):
    INSUFFICIENT_PROSPECTIVE_EVIDENCE = "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    EARLY_PROSPECTIVE_SIGNAL = "EARLY_PROSPECTIVE_SIGNAL"
    PROMISING_PROSPECTIVE_EVIDENCE = "PROMISING_PROSPECTIVE_EVIDENCE"
    ADEQUATE_PROSPECTIVE_SAMPLE_FOR_PRELIMINARY_GENERALIZATION = "ADEQUATE_PROSPECTIVE_SAMPLE_FOR_PRELIMINARY_GENERALIZATION"


def classify_prospective_evidence_tier(genuine_storm_count: int) -> ProspectiveEvidenceTier:
    """Classifies prospective sample evidence according to formal scientific tiers."""
    if genuine_storm_count < 5:
        return ProspectiveEvidenceTier.INSUFFICIENT_PROSPECTIVE_EVIDENCE
    elif 5 <= genuine_storm_count <= 14:
        return ProspectiveEvidenceTier.EARLY_PROSPECTIVE_SIGNAL
    elif 15 <= genuine_storm_count <= 29:
        return ProspectiveEvidenceTier.PROMISING_PROSPECTIVE_EVIDENCE
    else:
        return ProspectiveEvidenceTier.ADEQUATE_PROSPECTIVE_SAMPLE_FOR_PRELIMINARY_GENERALIZATION


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

        # Also verify preprocessing hash
        keycard_path = self.project_root / "research" / "cyclone_intensity" / "test_keycard.json"
        if keycard_path.exists():
            with open(keycard_path, "r", encoding="utf-8") as f:
                kc = json.load(f)
                prep_hash = kc.get("PREPROCESSING_HASH")
                if prep_hash and prep_hash != FROZEN_PREPROCESSING_SHA256:
                    raise ModelHashMismatchError(f"Preprocessing hash mismatch: {prep_hash} != {FROZEN_PREPROCESSING_SHA256}")

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
        ocean_age_hours: Optional[float] = None,
        data_provenance_class: Union[DataProvenanceClass, str] = DataProvenanceClass.UNKNOWN,
        atmos_source_id: Union[AtmosphericSource, str] = AtmosphericSource.ERA5_REANALYSIS,
        forecast_created_at: Optional[str] = None,
        include_extended: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates a prospective intensity prediction at origin T using the frozen model.
        Enforces causal availability, schema contracts, and returns the auditable forecast record.
        """
        # 1. Check target leakage
        for k in features.keys():
            if k.startswith("target_"):
                raise TargetLeakageError(f"Target column '{k}' detected in input feature dictionary!")

        prov_class = data_provenance_class.value if isinstance(data_provenance_class, DataProvenanceClass) else str(data_provenance_class)
        atmos_src = atmos_source_id.value if isinstance(atmos_source_id, AtmosphericSource) else str(atmos_source_id)

        # Operational atmospheric source honesty: ERA5 has multi-month latency
        if atmos_src == AtmosphericSource.ERA5_REANALYSIS.value and prov_class == DataProvenanceClass.REAL_PROSPECTIVE.value:
            # ERA5 cannot be claimed as real-time operational prospective observation
            prov_class = DataProvenanceClass.HISTORICAL_REANALYSIS.value

        # 2. Enforce Dual Timestamp Availability Firewall
        self.enforce_dual_timestamp_firewall(
            observation_timestamp=observation_timestamp,
            data_available_timestamp=data_available_timestamp,
            forecast_origin_timestamp=forecast_origin_timestamp,
            source_name="cyclone_observation"
        )

        # 3. Ocean availability & freshness verification
        # Copernicus ocean source remains daily.
        # Require: ocean_source_timestamp <= T, ocean_source_available_timestamp <= T, ocean_age_hours <= 48
        # If conditions fail: ocean feature = missing. Do not fabricate a replacement.
        ocean_stale_or_invalid = False
        if ocean_source_timestamp:
            ocean_avail = ocean_source_available_timestamp or ocean_source_timestamp
            try:
                self.enforce_dual_timestamp_firewall(
                    observation_timestamp=ocean_source_timestamp,
                    data_available_timestamp=ocean_avail,
                    forecast_origin_timestamp=forecast_origin_timestamp,
                    source_name="copernicus_ocean"
                )
            except CausalFirewallViolationError:
                ocean_stale_or_invalid = True
                raise

            if ocean_age_hours is not None and ocean_age_hours > MAX_OCEAN_AGE_HOURS:
                ocean_stale_or_invalid = True

        # 4. Atmospheric availability verification
        atmos_obs_ts = features.get("atmos_observation_timestamp")
        atmos_avail_ts = features.get("atmos_availability_timestamp")
        if atmos_obs_ts or atmos_avail_ts:
            effective_atmos_obs = atmos_obs_ts or forecast_origin_timestamp
            effective_atmos_avail = atmos_avail_ts or effective_atmos_obs
            self.enforce_dual_timestamp_firewall(
                observation_timestamp=effective_atmos_obs,
                data_available_timestamp=effective_atmos_avail,
                forecast_origin_timestamp=forecast_origin_timestamp,
                source_name="atmosphere"
            )

        # Calculate forecast valid time = T + 24h
        t_origin = pd.to_datetime(forecast_origin_timestamp)
        if t_origin.tz is None:
            t_origin = t_origin.tz_localize("UTC")
        t_valid = t_origin + pd.Timedelta(hours=24)

        # Calculate radial contrasts if base features are provided
        feat_dict = dict(features)

        # If ocean data was stale (>48h), invalidate ocean features to prevent fabrication
        if ocean_stale_or_invalid:
            for ok in ["sst_core_mean_0_100km", "sst_env_mean_200_800km", "mld_core_mean_0_100km", "sla_core_mean_0_100km", "delta_sst_core_minus_env"]:
                feat_dict[ok] = np.nan

        if "delta_vws_core_minus_env" not in feat_dict and "vws_core_mean_0_100km" in feat_dict and "vws_env_mean_200_800km" in feat_dict:
            feat_dict["delta_vws_core_minus_env"] = feat_dict["vws_core_mean_0_100km"] - feat_dict["vws_env_mean_200_800km"]
        if "delta_vort_core_minus_env" not in feat_dict and "vort_core_mean_0_100km" in feat_dict and "vort_env_mean_200_800km" in feat_dict:
            feat_dict["delta_vort_core_minus_env"] = feat_dict["vort_core_mean_0_100km"] - feat_dict["vort_env_mean_200_800km"]
        if "delta_rh700_core_minus_env" not in feat_dict and "rh_700_core_mean_0_100km" in feat_dict and "rh_700_env_mean_200_800km" in feat_dict:
            feat_dict["delta_rh700_core_minus_env"] = feat_dict["rh_700_core_mean_0_100km"] - feat_dict["rh_700_env_mean_200_800km"]
        if "delta_rh500_core_minus_env" not in feat_dict and "rh_500_core_mean_0_100km" in feat_dict and "rh_500_env_mean_200_800km" in feat_dict:
            feat_dict["delta_rh500_core_minus_env"] = feat_dict["rh_500_core_mean_0_100km"] - feat_dict["rh_500_env_mean_200_800km"]
        if "delta_sst_core_minus_env" not in feat_dict and "sst_core_mean_0_100km" in feat_dict and "sst_env_mean_200_800km" in feat_dict:
            if not pd.isna(feat_dict["sst_core_mean_0_100km"]) and not pd.isna(feat_dict["sst_env_mean_200_800km"]):
                feat_dict["delta_sst_core_minus_env"] = feat_dict["sst_core_mean_0_100km"] - feat_dict["sst_env_mean_200_800km"]
            else:
                feat_dict["delta_sst_core_minus_env"] = np.nan

        # Build feature vector in exact frozen order
        missing_keys = [k for k in FROZEN_29_FEATURES if k not in feat_dict or pd.isna(feat_dict[k])]
        row_values = [feat_dict.get(k, np.nan) for k in FROZEN_29_FEATURES]
        feature_completeness = round(1.0 - (len(missing_keys) / len(FROZEN_29_FEATURES)), 4)
        
        df_x = pd.DataFrame([row_values], columns=FROZEN_29_FEATURES)
        
        # Frozen model inference (native NaN handling in XGBoost)
        pred_val = float(self.model.predict(df_x)[0])
        
        # Enforce physical clipping bounds [15.0, 165.0] kt (Operational validity guard, not model recalibration)
        clipping_applied = bool(pred_val < 15.0 or pred_val > 165.0)
        pred_clipped = round(float(np.clip(pred_val, 15.0, 165.0)), 2)

        created_at_str = forecast_created_at or datetime.now(timezone.utc).isoformat()

        # Model Input Forensics snapshot immediately before inference
        forensic_snapshot = {
            "forecast_origin_timestamp": t_origin.isoformat(),
            "feature_names": list(FROZEN_29_FEATURES),
            "feature_ordering": list(range(len(FROZEN_29_FEATURES))),
            "feature_values": {k: float(row_values[idx]) if not pd.isna(row_values[idx]) else None for idx, k in enumerate(FROZEN_29_FEATURES)},
            "dtypes": "float64",
            "missingness": {"missing_keys": missing_keys, "missing_count": len(missing_keys)},
            "preprocessing_state": "RAW_TABULAR_NATIVE_CONTINUOUS",
            "model_hash": FROZEN_MODEL_SHA256,
            "feature_contract_hash": FROZEN_FEATURE_CONTRACT_SHA256,
            "preprocessing_hash": FROZEN_PREPROCESSING_SHA256,
        }

        # Build auditable forecast record
        record = {
            "system_id": system_id,
            "forecast_origin_timestamp": t_origin.isoformat(),
            "forecast_valid_time": t_valid.isoformat(),
            "forecast_vmax_24h": pred_clipped,
            "raw_vmax_24h": round(pred_val, 2),
            "reported_vmax_24h": pred_clipped,
            "clipping_applied": clipping_applied,
            "model_version": MODEL_VERSION,
            "model_hash": FROZEN_MODEL_SHA256,
            "feature_contract_hash": FROZEN_FEATURE_CONTRACT_SHA256,
            "preprocessing_hash": FROZEN_PREPROCESSING_SHA256,
            "feature_timestamp_cutoff": observation_timestamp,
            "data_availability_timestamp": data_available_timestamp,
            "ocean_source_timestamp": ocean_source_timestamp or "UNAVAILABLE",
            "ocean_source_available_timestamp": ocean_source_available_timestamp or "UNAVAILABLE",
            "ocean_age_hours": float(ocean_age_hours) if ocean_age_hours is not None else np.nan,
            "ocean_temporal_resolution": "daily",
            "feature_completeness": feature_completeness,
            "missingness_flags": json.dumps(missing_keys),
            "prediction_status": PredictionStatus.PREDICTION_GENERATED.value,
            "evaluation_mode": self.evaluation_mode.value,
            "data_provenance_class": prov_class,
            "atmos_source_id": atmos_src,
            "forecast_created_at": created_at_str,
            "vmax_current": float(feat_dict.get("vmax_current", np.nan)) if not pd.isna(feat_dict.get("vmax_current")) else None,
            "dvmax_12h": float(feat_dict.get("dvmax_12h", np.nan)) if not pd.isna(feat_dict.get("dvmax_12h")) else None,
            "model_input_forensics": forensic_snapshot,
        }

        if not include_extended:
            canonical_17_keys = [
                "system_id", "forecast_origin_timestamp", "forecast_valid_time", "forecast_vmax_24h",
                "model_version", "model_hash", "feature_contract_hash", "preprocessing_hash",
                "feature_timestamp_cutoff", "data_availability_timestamp", "ocean_source_timestamp",
                "ocean_source_available_timestamp", "ocean_age_hours", "feature_completeness",
                "missingness_flags", "prediction_status", "evaluation_mode"
            ]
            return {k: record[k] for k in canonical_17_keys if k in record}

        return record

    def log_forecast(self, forecast_record: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """
        Appends the forecast record to forecast_log.parquet in an append-only,
        immutable fashion. Rejects duplicate submissions for (system_id, forecast_origin_timestamp).
        """
        target_path = filepath or (self.db_dir / "forecast_log.parquet")
        
        # Enforce schema contract
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
        
        # Exclude nested model_input_forensics dict from flat parquet file
        record_to_save = {k: v for k, v in forecast_record.items() if k != "model_input_forensics"}
        df_new = pd.DataFrame([record_to_save])
        
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
        current_clock_utc: Optional[str] = None,
        forecast_created_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Matches an observation within [T+21h, T+27h] under the target lifecycle state machine.
        Enforces:
        1. Temporal honesty: forecast_created_at < target_available_at. If violated, marked TARGET_EXCLUDED (RETROSPECTIVE).
        2. Provenance honesty: Non-real prospective data cannot masquerade as genuine prospective evaluation.
        3. If deadline T+27h has passed and no fix is obtainable, enters TARGET_UNAVAILABLE (terminal).
        TARGET_UNAVAILABLE records NEVER receive 0 error and are NEVER included in MAE/RMSE denominators.
        """
        t_origin = pd.to_datetime(forecast_record["forecast_origin_timestamp"])
        t_target_nominal = t_origin + pd.Timedelta(hours=24)
        t_min = t_origin + pd.Timedelta(hours=TARGET_WINDOW_MIN_HOURS)
        t_max = t_origin + pd.Timedelta(hours=TARGET_WINDOW_MAX_HOURS)

        now_utc = pd.to_datetime(current_clock_utc) if current_clock_utc else pd.Timestamp.now(tz="UTC")
        if now_utc.tz is None:
            now_utc = now_utc.tz_localize("UTC")

        # Temporal honesty check: prediction must genuinely precede knowledge of the outcome
        fc_created_str = forecast_created_at or forecast_record.get("forecast_created_at")
        t_created = pd.to_datetime(fc_created_str) if fc_created_str else now_utc
        if t_created.tz is None:
            t_created = t_created.tz_localize("UTC")

        prov_class = forecast_record.get("data_provenance_class", DataProvenanceClass.UNKNOWN.value)

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

        # Lifecycle resolution
        if eligible_fixes:
            # Pick closest fix to nominal T+24h
            eligible_fixes.sort(key=lambda x: x[0])
            best = eligible_fixes[0]
            target_fix_time = best[1]
            observed_vmax = best[2]
            fix_meta = best[3]
            offset_hours = round((target_fix_time - t_target_nominal).total_seconds() / 3600.0, 2)

            target_avail_str = fix_meta.get("data_availability_timestamp") or target_fix_time.isoformat()
            t_target_avail = pd.to_datetime(target_avail_str)
            if t_target_avail.tz is None:
                t_target_avail = t_target_avail.tz_localize("UTC")

            # Synthetic fixtures are strictly excluded from prospective scientific targets
            if prov_class == DataProvenanceClass.SYNTHETIC_TEST_FIXTURE.value:
                return {
                    "system_id": forecast_record["system_id"],
                    "forecast_origin_timestamp": forecast_record["forecast_origin_timestamp"],
                    "forecast_valid_time": forecast_record["forecast_valid_time"],
                    "target_status": TargetStatus.TARGET_EXCLUDED.value,
                    "target_unavailable_reason": "SYNTHETIC_FIXTURE_EXCLUDED_FROM_SCIENTIFIC_EVALUATION",
                    "actual_target_timestamp": target_fix_time.isoformat(),
                    "target_offset_hours": offset_hours,
                    "observed_vmax_24h": observed_vmax,
                    "target_source": fix_meta.get("source", "SYNTHETIC_FIXTURE"),
                    "target_source_version": fix_meta.get("source_version", "DEMO_V1"),
                    "target_available_timestamp": target_avail_str,
                    "target_retrieved_at": now_utc.isoformat(),
                    "forecast_created_at": t_created.isoformat(),
                    "data_provenance_class": prov_class,
                    "evaluation_type": evaluation_type.value,
                    "evaluation_mode": self.evaluation_mode.value,
                }

            # Temporal honesty rule: forecast_created_at < target_available_at
            # Applies when evaluating prospective forecasts in TRUE_PROSPECTIVE mode, or for REAL_PROSPECTIVE records
            if self.evaluation_mode == EvaluationMode.TRUE_PROSPECTIVE or prov_class == DataProvenanceClass.REAL_PROSPECTIVE.value:
                if t_created >= t_target_avail:
                    # Target was already available before forecast creation: RETROSPECTIVE EVALUATION
                    return {
                        "system_id": forecast_record["system_id"],
                        "forecast_origin_timestamp": forecast_record["forecast_origin_timestamp"],
                        "forecast_valid_time": forecast_record["forecast_valid_time"],
                        "target_status": TargetStatus.TARGET_EXCLUDED.value,
                        "target_unavailable_reason": "RETROSPECTIVE_EVALUATION_TARGET_AVAILABLE_BEFORE_FORECAST_CREATION",
                        "actual_target_timestamp": target_fix_time.isoformat(),
                        "target_offset_hours": offset_hours,
                        "observed_vmax_24h": observed_vmax,
                        "target_source": fix_meta.get("source", "IMD_SYNOPTIC"),
                        "target_source_version": fix_meta.get("source_version", "OPERATIONAL_V1"),
                        "target_available_timestamp": target_avail_str,
                        "target_retrieved_at": now_utc.isoformat(),
                        "forecast_created_at": t_created.isoformat(),
                        "data_provenance_class": prov_class,
                        "evaluation_type": EvaluationType.RETROSPECTIVE_REVISED.value,
                        "evaluation_mode": self.evaluation_mode.value,
                    }

                # If in TRUE_PROSPECTIVE mode, provenance must strictly be REAL_PROSPECTIVE
                if self.evaluation_mode == EvaluationMode.TRUE_PROSPECTIVE and prov_class != DataProvenanceClass.REAL_PROSPECTIVE.value:
                    return {
                        "system_id": forecast_record["system_id"],
                        "forecast_origin_timestamp": forecast_record["forecast_origin_timestamp"],
                        "forecast_valid_time": forecast_record["forecast_valid_time"],
                        "target_status": TargetStatus.TARGET_EXCLUDED.value,
                        "target_unavailable_reason": f"NON_PROSPECTIVE_PROVENANCE_CLASS_{prov_class}",
                        "actual_target_timestamp": target_fix_time.isoformat(),
                        "target_offset_hours": offset_hours,
                        "observed_vmax_24h": observed_vmax,
                        "target_source": fix_meta.get("source", "IMD_SYNOPTIC"),
                        "target_source_version": fix_meta.get("source_version", "OPERATIONAL_V1"),
                        "target_available_timestamp": target_avail_str,
                        "target_retrieved_at": now_utc.isoformat(),
                        "forecast_created_at": t_created.isoformat(),
                        "data_provenance_class": prov_class,
                        "evaluation_type": evaluation_type.value,
                        "evaluation_mode": self.evaluation_mode.value,
                    }

            # Valid genuine prospective target
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
                "target_available_timestamp": target_avail_str,
                "target_retrieved_at": now_utc.isoformat(),
                "forecast_created_at": t_created.isoformat(),
                "data_provenance_class": prov_class,
                "evaluation_type": evaluation_type.value,
                "evaluation_mode": self.evaluation_mode.value,
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
                "forecast_created_at": t_created.isoformat(),
                "data_provenance_class": prov_class,
                "evaluation_type": evaluation_type.value,
                "evaluation_mode": self.evaluation_mode.value,
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
                "forecast_created_at": t_created.isoformat(),
                "data_provenance_class": prov_class,
                "evaluation_type": evaluation_type.value,
                "evaluation_mode": self.evaluation_mode.value,
            }

    def log_target_arrival(self, target_record: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """
        Appends target arrival records to target_arrival_log.parquet.
        Distinguishes PROSPECTIVE_INITIAL from RETROSPECTIVE_REVISED.
        """
        target_path = filepath or (self.db_dir / "target_arrival_log.parquet")
        df_new = pd.DataFrame([target_record])

        if target_path.exists():
            existing_df = pd.read_parquet(target_path)
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
        baseline_trend_dict: Optional[Dict[str, float]] = None,
        require_real_prospective: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates paired forecasts and targets.
        CRITICAL SCIENTIFIC CONTROLS:
        1. Strictly filters to target_status == 'TARGET_AVAILABLE' and evaluation_type == 'PROSPECTIVE_INITIAL'.
        2. TARGET_UNAVAILABLE records NEVER receive 0 error and are NEVER included in MAE/RMSE denominators.
        3. All scientific accuracy metrics use strictly evaluated_targets as their denominator:
           MAE = sum(abs(pred - obs)) / number_of_valid_evaluated_targets.
        4. When require_real_prospective is True, only REAL_PROSPECTIVE records enter prospective accuracy metrics.
        5. Computes persistence and damped trend baselines on the exact same evaluated targets.
        """
        n_forecasts_total = len(forecast_df)
        n_targets_available = len(target_df[target_df["target_status"] == TargetStatus.TARGET_AVAILABLE.value])
        n_targets_pending = len(target_df[target_df["target_status"] == TargetStatus.TARGET_PENDING.value])
        n_targets_unavailable = len(target_df[target_df["target_status"] == TargetStatus.TARGET_UNAVAILABLE.value])
        n_targets_excluded = len(target_df[target_df["target_status"] == TargetStatus.TARGET_EXCLUDED.value])

        target_coverage_rate = round(n_targets_available / n_forecasts_total, 4) if n_forecasts_total > 0 else 0.0

        # Filter strictly to valid evaluated targets
        valid_targets = target_df[
            (target_df["target_status"] == TargetStatus.TARGET_AVAILABLE.value) &
            (target_df["evaluation_type"] == EvaluationType.PROSPECTIVE_INITIAL.value) &
            (target_df["observed_vmax_24h"].notna())
        ].copy()

        # If require_real_prospective is True, ensure non-prospective data does not enter prospective accuracy
        if require_real_prospective:
            if "data_provenance_class" in valid_targets.columns:
                valid_targets = valid_targets[valid_targets["data_provenance_class"] == DataProvenanceClass.REAL_PROSPECTIVE.value].copy()
            if "data_provenance_class" in forecast_df.columns:
                forecast_df = forecast_df[forecast_df["data_provenance_class"] == DataProvenanceClass.REAL_PROSPECTIVE.value].copy()

        merged = pd.merge(
            forecast_df,
            valid_targets,
            on=["system_id", "forecast_origin_timestamp"],
            suffixes=("_fc", "_tgt")
        )

        n_evaluated = len(merged)
        evaluated_target_rate = round(n_evaluated / n_forecasts_total, 4) if n_forecasts_total > 0 else 0.0
        n_storms = merged["system_id"].nunique() if n_evaluated > 0 else 0
        evidence_tier = classify_prospective_evidence_tier(n_storms)

        if n_evaluated == 0:
            return {
                "n_storms": n_storms,
                "n_forecasts_total": n_forecasts_total,
                "n_targets_available": n_targets_available,
                "n_targets_pending": n_targets_pending,
                "n_targets_unavailable": n_targets_unavailable,
                "n_targets_excluded": n_targets_excluded,
                "target_coverage_rate": target_coverage_rate,
                "evaluated_target_rate": evaluated_target_rate,
                "number_of_valid_evaluated_targets": 0,
                "row_mae": None,
                "row_rmse": None,
                "row_bias": None,
                "storm_mae_mean": None,
                "persistence_mae": None,
                "damped_trend_mae": None,
                "model_skill_vs_persistence": None,
                "model_skill_vs_damped_trend": None,
                "prospective_evidence_tier": evidence_tier.value,
                "evaluation_mode": self.evaluation_mode.value,
                "note": "Zero valid prospective evaluated targets."
            }

        y_true = merged["observed_vmax_24h"].values
        y_pred = merged["forecast_vmax_24h"].values
        system_ids = merged["system_id"].values

        # Absolute errors and bias: Denominator is strictly number_of_valid_evaluated_targets
        errors = np.abs(y_true - y_pred)
        row_mae = float(np.sum(errors) / n_evaluated)
        row_rmse = float(np.sqrt(np.sum((y_true - y_pred) ** 2) / n_evaluated))
        row_bias = float(np.sum(y_pred - y_true) / n_evaluated)

        # Storm-level aggregation
        df_eval = pd.DataFrame({"system_id": system_ids, "error": errors})
        storm_maes = df_eval.groupby("system_id")["error"].mean()
        storm_mae_mean = float(storm_maes.mean())

        # Baseline comparisons on the EXACT SAME evaluated targets
        pers_errors = []
        trend_errors = []
        for _, row in merged.iterrows():
            key = f"{row['system_id']}_{row['forecast_origin_timestamp']}"
            
            # Extract v0
            if baseline_v0_dict and key in baseline_v0_dict:
                v0 = baseline_v0_dict[key]
            elif "vmax_current" in row and not pd.isna(row["vmax_current"]):
                v0 = float(row["vmax_current"])
            else:
                v0 = float(row["observed_vmax_24h"])

            # Extract damped trend: clip(v0 + 0.5 * dv12h, 15, 165)
            if baseline_trend_dict and key in baseline_trend_dict:
                trend = baseline_trend_dict[key]
            elif "dvmax_12h" in row and not pd.isna(row["dvmax_12h"]):
                trend = float(np.clip(v0 + 0.5 * float(row["dvmax_12h"]), 15.0, 165.0))
            else:
                trend = v0

            pers_errors.append(abs(row["observed_vmax_24h"] - v0))
            trend_errors.append(abs(row["observed_vmax_24h"] - trend))

        pers_mae = float(np.sum(pers_errors) / n_evaluated) if n_evaluated > 0 else 0.0
        trend_mae = float(np.sum(trend_errors) / n_evaluated) if n_evaluated > 0 else 0.0

        pers_skill = round((1.0 - (row_mae / pers_mae)) * 100.0, 2) if pers_mae > 0 else 0.0
        trend_skill = round((1.0 - (row_mae / trend_mae)) * 100.0, 2) if trend_mae > 0 else 0.0

        return {
            "n_storms": n_storms,
            "n_forecasts_total": n_forecasts_total,
            "n_targets_available": n_targets_available,
            "n_targets_pending": n_targets_pending,
            "n_targets_unavailable": n_targets_unavailable,
            "n_targets_excluded": n_targets_excluded,
            "target_coverage_rate": target_coverage_rate,
            "evaluated_target_rate": evaluated_target_rate,
            "number_of_valid_evaluated_targets": n_evaluated,
            "row_mae": round(row_mae, 3),
            "row_rmse": round(row_rmse, 3),
            "row_bias": round(row_bias, 3),
            "storm_mae_mean": round(storm_mae_mean, 3),
            "persistence_mae": round(pers_mae, 3),
            "damped_trend_mae": round(trend_mae, 3),
            "model_skill_vs_persistence": pers_skill,
            "model_skill_vs_damped_trend": trend_skill,
            "prospective_evidence_tier": evidence_tier.value,
            "evaluation_mode": self.evaluation_mode.value
        }

    @staticmethod
    def compute_storm_cluster_bootstrap(
        merged_df: pd.DataFrame,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Correction 12: Clustered bootstrap resampling by storm (system_id), not individual fixes.
        Requires >= 5 storms; otherwise flags uncertainty as UNSTABLE_SAMPLE_SIZE.
        """
        if "system_id" not in merged_df.columns or len(merged_df) == 0:
            return {
                "n_storms": 0,
                "n_forecasts": 0,
                "bootstrap_iterations": n_bootstrap,
                "storm_cluster_bootstrap_ci": None,
                "status": "UNSTABLE_SAMPLE_SIZE",
                "detail": "Empty dataset."
            }

        unique_storms = merged_df["system_id"].unique()
        n_storms = len(unique_storms)
        if n_storms < 5:
            return {
                "n_storms": n_storms,
                "n_forecasts": len(merged_df),
                "bootstrap_iterations": n_bootstrap,
                "storm_cluster_bootstrap_ci": None,
                "status": "UNSTABLE_SAMPLE_SIZE",
                "detail": f"Sample size ({n_storms} storms) is insufficient for clustered bootstrap resampling (< 5 storms required)."
            }

        rng = np.random.RandomState(random_state)
        boot_means = []
        for _ in range(n_bootstrap):
            sampled_storms = rng.choice(unique_storms, size=n_storms, replace=True)
            boot_errors = []
            for s in sampled_storms:
                s_rows = merged_df[merged_df["system_id"] == s]
                s_err = np.mean(np.abs(s_rows["observed_vmax_24h"] - s_rows["forecast_vmax_24h"]))
                boot_errors.append(s_err)
            boot_means.append(float(np.mean(boot_errors)))

        alpha = (1.0 - confidence_level) / 2.0
        ci_lower = float(np.percentile(boot_means, alpha * 100))
        ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        return {
            "n_storms": n_storms,
            "n_forecasts": len(merged_df),
            "bootstrap_iterations": n_bootstrap,
            "storm_cluster_bootstrap_ci": [round(ci_lower, 2), round(ci_upper, 2)],
            "mean_bootstrap_mae": round(float(np.mean(boot_means)), 2),
            "status": "VALID_CLUSTER_BOOTSTRAP"
        }

    @staticmethod
    def compute_ocean_age_histogram(forecast_df: pd.DataFrame) -> Dict[str, int]:
        """
        Reports distribution of ocean observation freshness across forecasts.
        """
        hist = {
            "age_0h": 0,
            "age_6h": 0,
            "age_12h": 0,
            "age_18h": 0,
            "age_24h_plus": 0,
            "missing": 0,
        }
        if "ocean_age_hours" not in forecast_df.columns:
            hist["missing"] = len(forecast_df)
            return hist
        for val in forecast_df["ocean_age_hours"]:
            if pd.isna(val) or val is None:
                hist["missing"] += 1
            elif val <= 0.0:
                hist["age_0h"] += 1
            elif val <= 6.0:
                hist["age_6h"] += 1
            elif val <= 12.0:
                hist["age_12h"] += 1
            elif val <= 18.0:
                hist["age_18h"] += 1
            else:
                hist["age_24h_plus"] += 1
        return hist

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
