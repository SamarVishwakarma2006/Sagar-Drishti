"""
Isolated Runtime Shadow Evaluation Service for Sagar-Drishti V2.3 Model D.

CRITICAL SAFETY & OBSERVATIONAL SPECIFICATIONS:
1. Observational Only: Zero authority to trigger alerts, alter risk scores, change thresholds,
   send notifications, or modify operational state.
2. Complete Isolation: Bounded asynchronous background execution. Production responses are
   returned independently without latency impact. Shadow failures NEVER propagate to production.
3. Causal & Temporal Safety: Strictly enforces ERA5 cutoff of T 18:00:00 UTC. Never uses T+1/T+2/T+3.
4. Exact 224-Feature Contract: 101 ocean + 123 atmospheric features validated against trusted
   frozen schema and model hashes.
5. True Bounded Queue: Finite queue capacity (default 100). Drops jobs gracefully if queue is full.
6. Lazy & Safe Initialization: Model artifacts and ERA5 index load lazily; initialization failures
   keep production healthy and mark shadow as unavailable.
7. Append-Only Telemetry: Writes audit records to backend/data/shadow/shadow_v2_3_telemetry_{YYYYMM}.jsonl.
"""

import os
import json
import time
import queue
import logging
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("sagar_drishti.shadow_v2_3_service")

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CANDIDATE_MODEL_D_DIR = os.path.join(BASE_DIR, "models", "candidates", "v2_3", "ocean_atmos_all")
ERA5_DAILY_PARQUET_PATH = os.path.join(BASE_DIR, "data", "era5", "features_atmosphere_10yr_daily.parquet")
FROZEN_MANIFEST_PATH = os.path.join(BASE_DIR, "config", "v2_3_frozen_experiment_manifest.json")
SHADOW_DATA_DIR = os.path.join(BASE_DIR, "data", "shadow")

os.makedirs(SHADOW_DATA_DIR, exist_ok=True)

# Trusted expected artifact hashes from frozen manifest SD-V2.3-ABLATION-CONTROLLED
TRUSTED_MODEL_HASHES = {
    0: "7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15",
    1: "b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe",
    2: "aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c",
    3: "250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775",
}

# Trusted feature schema hash (SHA-256 of JSON-serialized list of 224 feature names)
TRUSTED_FEATURE_SCHEMA_HASH = "25a9cad9cfa08414f86d5214155fc2e4f2e0225eb28f17902b4d35d5729db236"
TOTAL_FEATURES_EXPECTED = 224
OCEAN_FEATURES_EXPECTED = 101
ATMOS_FEATURES_EXPECTED = 123
ATMOS_PHYSICAL_PER_ROLE = 41

# Fixed spatial mapping (Hardcoded V2.3 protocol - no runtime heuristics)
SPATIAL_MAPPING = {
    "bob": {"atmos_north": "NBOB", "atmos_central": "CBOB", "atmos_south": "SBOB"},
    "aras": {"atmos_north": "NAS", "atmos_central": "CAS", "atmos_south": "SAS"},
}

# Research thresholds (strictly for observational research evaluation, never operational)
RESEARCH_THRESHOLDS = {
    0: 0.15,
    1: 0.15,
    2: 0.21,
    3: 0.27,
}

# Coverage & Quality States
STATUS_READY = "READY"
STATUS_ATMOSPHERE_MISSING = "ATMOSPHERE_MISSING"
STATUS_ATMOSPHERE_LATE = "ATMOSPHERE_LATE"
STATUS_FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
STATUS_INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
STATUS_INFERENCE_ERROR = "INFERENCE_ERROR"

COV_SHADOW_SUCCESS = "SHADOW_SUCCESS"
COV_SHADOW_FAILED = "SHADOW_FAILED"
COV_SHADOW_SKIPPED_MISSING_DATA = "SHADOW_SKIPPED_MISSING_DATA"
COV_SHADOW_SKIPPED_LATE_DATA = "SHADOW_SKIPPED_LATE_DATA"
COV_SHADOW_QUEUE_FULL = "SHADOW_QUEUE_FULL"
COV_SHADOW_NOT_INITIALIZED = "SHADOW_NOT_INITIALIZED"

# True Bounded Queue parameters
DEFAULT_QUEUE_CAPACITY = 100
MAX_SHADOW_WORKERS = 2


class ShadowV23Service:
    """
    Completely isolated runtime shadow inference service for V2.3 Model D.
    Observational only. Zero authority. Never blocks or crashes production.
    """

    _lock = threading.Lock()
    _init_lock = threading.Lock()
    _file_write_lock = threading.Lock()

    _enabled: bool = True
    _initialized: bool = False
    _init_failed: bool = False
    _init_error_msg: Optional[str] = None

    _models: Dict[int, Any] = {}
    _model_hashes: Dict[int, str] = {}
    _expected_feature_names: List[str] = []
    _feature_schema_hash: str = ""

    # In-memory indexed ERA5 storage for sub-millisecond lookup
    # Key: (date_str, basin) -> dict of 41 physical features
    _era5_index: Dict[Tuple[str, str], Dict[str, float]] = {}
    _era5_feature_cols: List[str] = []
    _max_available_atmospheric_date: Optional[str] = None
    _max_available_atmospheric_timestamp: Optional[str] = None

    # Bounded Queue & Workers
    _queue: queue.Queue = queue.Queue(maxsize=DEFAULT_QUEUE_CAPACITY)
    _workers_started: bool = False
    _worker_threads: List[threading.Thread] = []

    # Observability metrics
    _metrics = {
        "jobs_submitted": 0,
        "jobs_executed": 0,
        "jobs_dropped": 0,
        "initialization_failures": 0,
        "successful_predictions": 0,
        "failed_predictions": 0,
        "data_unavailable_predictions": 0,
        "missing_atmosphere_count": 0,
        "late_atmosphere_count": 0,
        "schema_mismatch_count": 0,
        "total_latency_ms": 0.0,
        "predictions_by_site": {"bob": 0, "aras": 0},
        "predictions_by_horizon": {0: 0, 1: 0, 2: 0, 3: 0},
        "probability_deltas": [],
    }

    # Custom paths for testing overrides
    _override_model_dir: Optional[str] = None
    _override_era5_path: Optional[str] = None
    _override_manifest_path: Optional[str] = None

    @classmethod
    def set_enabled(cls, enabled: bool):
        cls._enabled = enabled

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def set_overrides(cls, model_dir: Optional[str] = None, era5_path: Optional[str] = None, manifest_path: Optional[str] = None):
        with cls._lock:
            cls._override_model_dir = model_dir
            cls._override_era5_path = era5_path
            cls._override_manifest_path = manifest_path

    @classmethod
    def clear_cache(cls):
        """Thread-safe cache clear and reset (useful for tests)."""
        with cls._lock:
            cls._models.clear()
            cls._model_hashes.clear()
            cls._expected_feature_names = []
            cls._feature_schema_hash = ""
            cls._era5_index.clear()
            cls._era5_feature_cols = []
            cls._max_available_atmospheric_date = None
            cls._max_available_atmospheric_timestamp = None
            cls._initialized = False
            cls._init_failed = False
            cls._init_error_msg = None

    @classmethod
    def reset_metrics(cls):
        with cls._lock:
            cls._metrics["jobs_submitted"] = 0
            cls._metrics["jobs_executed"] = 0
            cls._metrics["jobs_dropped"] = 0
            cls._metrics["initialization_failures"] = 0
            cls._metrics["successful_predictions"] = 0
            cls._metrics["failed_predictions"] = 0
            cls._metrics["data_unavailable_predictions"] = 0
            cls._metrics["missing_atmosphere_count"] = 0
            cls._metrics["late_atmosphere_count"] = 0
            cls._metrics["schema_mismatch_count"] = 0
            cls._metrics["total_latency_ms"] = 0.0
            cls._metrics["predictions_by_site"] = {"bob": 0, "aras": 0}
            cls._metrics["predictions_by_horizon"] = {0: 0, 1: 0, 2: 0, 3: 0}
            cls._metrics["probability_deltas"] = []

    @classmethod
    def get_observability_metrics(cls) -> Dict[str, Any]:
        with cls._lock:
            total_exec = cls._metrics["jobs_executed"]
            avg_lat = (cls._metrics["total_latency_ms"] / total_exec) if total_exec > 0 else 0.0
            deltas = cls._metrics["probability_deltas"]
            avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0

            return {
                "jobs_submitted": cls._metrics["jobs_submitted"],
                "jobs_executed": cls._metrics["jobs_executed"],
                "jobs_dropped": cls._metrics["jobs_dropped"],
                "queue_depth": cls._queue.qsize(),
                "queue_capacity": cls._queue.maxsize,
                "worker_threads_active": sum(1 for t in cls._worker_threads if t.is_alive()),
                "worker_threads_target": MAX_SHADOW_WORKERS,
                "initialization_failures": cls._metrics["initialization_failures"],
                "successful_predictions": cls._metrics["successful_predictions"],
                "failed_predictions": cls._metrics["failed_predictions"],
                "data_unavailable_predictions": cls._metrics["data_unavailable_predictions"],
                "missing_atmosphere_count": cls._metrics["missing_atmosphere_count"],
                "late_atmosphere_count": cls._metrics["late_atmosphere_count"],
                "schema_mismatch_count": cls._metrics["schema_mismatch_count"],
                "average_latency_ms": round(avg_lat, 2),
                "average_probability_delta": round(avg_delta, 4),
                "predictions_by_site": dict(cls._metrics["predictions_by_site"]),
                "predictions_by_horizon": dict(cls._metrics["predictions_by_horizon"]),
                "current_available_atmospheric_data_max_timestamp": cls._max_available_atmospheric_timestamp,
                "initialized": cls._initialized,
                "init_failed": cls._init_failed,
            }

    @classmethod
    def _start_workers(cls):
        """Ensure background worker threads are alive for bounded queue processing."""
        with cls._lock:
            # Self-healing: filter and respawn any terminated worker threads
            cls._worker_threads = [t for t in cls._worker_threads if t.is_alive()]
            while len(cls._worker_threads) < MAX_SHADOW_WORKERS:
                i = len(cls._worker_threads)
                t = threading.Thread(
                    target=cls._worker_loop,
                    name=f"sagar_shadow_v2_3_worker_{i}_{int(time.time()*1000)}",
                    daemon=True,
                )
                t.start()
                cls._worker_threads.append(t)
            cls._workers_started = True

    @classmethod
    def _worker_loop(cls):
        """Background worker loop pulling from bounded queue with guaranteed task_done()."""
        while True:
            try:
                task = cls._queue.get()
                if task is None:
                    break
                try:
                    cls.evaluate_shadow(**task)
                except Exception as e:
                    logger.warning("Shadow worker task error: %s", e)
                finally:
                    cls._queue.task_done()
            except Exception as e:
                logger.warning("Shadow worker loop non-fatal error: %s", e)

    @classmethod
    def drain_queue(cls, timeout: float = 10.0):
        """Helper to wait for pending queue items to complete (useful for tests)."""
        deadline = time.time() + timeout
        while cls._queue.unfinished_tasks > 0 and time.time() < deadline:
            time.sleep(0.05)

    @classmethod
    def _compute_sha256(cls, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _ensure_initialized(cls) -> bool:
        """
        Lazy, thread-safe initialization of Model D artifacts and ERA5 in-memory index.
        Safely isolates failures: if anything fails, sets _init_failed = True and returns False.
        Production is never disrupted.
        """
        if cls._initialized:
            return True
        if cls._init_failed:
            return False

        with cls._init_lock:
            if cls._initialized:
                return True
            if cls._init_failed:
                return False

            try:
                manifest_path = cls._override_manifest_path or FROZEN_MANIFEST_PATH
                if not os.path.exists(manifest_path):
                    raise FileNotFoundError(f"Frozen manifest not found at {manifest_path}")

                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                expected_features = manifest.get("feature_names", {}).get("model_d_ocean_plus_all_atmos", [])
                if len(expected_features) != TOTAL_FEATURES_EXPECTED:
                    raise ValueError(
                        f"Manifest Model D features count mismatch: expected {TOTAL_FEATURES_EXPECTED}, found {len(expected_features)}"
                    )

                computed_schema_hash = hashlib.sha256(json.dumps(expected_features).encode("utf-8")).hexdigest()
                if computed_schema_hash != TRUSTED_FEATURE_SCHEMA_HASH:
                    raise ValueError(
                        f"Manifest schema hash mismatch! Got {computed_schema_hash}, expected {TRUSTED_FEATURE_SCHEMA_HASH}"
                    )

                cls._expected_feature_names = expected_features
                cls._feature_schema_hash = computed_schema_hash

                # Load Model D candidate artifacts (Horizons 0d, 1d, 2d, 3d)
                model_dir = cls._override_model_dir or CANDIDATE_MODEL_D_DIR
                for h in [0, 1, 2, 3]:
                    model_path = os.path.join(model_dir, f"risk_model_{h}d.joblib")
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model D artifact missing for {h}d at {model_path}")

                    # Validate artifact SHA-256 against trusted expected hash
                    actual_hash = cls._compute_sha256(model_path)
                    expected_hash = TRUSTED_MODEL_HASHES.get(h)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"Model D artifact hash mismatch for {h}d! Actual: {actual_hash}, Expected: {expected_hash}"
                        )

                    cls._model_hashes[h] = actual_hash

                    loaded_obj = joblib.load(model_path)
                    model_obj = loaded_obj.get("model") if isinstance(loaded_obj, dict) else loaded_obj
                    if model_obj is None:
                        raise ValueError(f"Corrupt Model D joblib for horizon {h}d: missing 'model'")

                    # Validate feature_columns stored inside artifact
                    artifact_cols = loaded_obj.get("feature_columns", [])
                    if artifact_cols != cls._expected_feature_names:
                        raise ValueError(f"Artifact feature_columns mismatch for horizon {h}d!")

                    cls._models[h] = model_obj

                # Load and index ERA5 Daily features
                era5_path = cls._override_era5_path or ERA5_DAILY_PARQUET_PATH
                if not os.path.exists(era5_path):
                    raise FileNotFoundError(f"ERA5 daily parquet missing at {era5_path}")

                df = pd.read_parquet(era5_path)
                meta_cols = [
                    "expected_6h_samples",
                    "number_of_valid_6h_samples",
                    "completeness_fraction",
                    "is_complete",
                    "is_ml_warmup_complete",
                ]
                cls._era5_feature_cols = [c for c in df.columns if c not in ["date", "basin"] + meta_cols]
                if len(cls._era5_feature_cols) != ATMOS_PHYSICAL_PER_ROLE:
                    raise ValueError(
                        f"Expected {ATMOS_PHYSICAL_PER_ROLE} ERA5 physical features, found {len(cls._era5_feature_cols)}"
                    )

                df["date_str"] = df["date"].astype(str).str[:10]
                # Determine dynamic available ceiling
                max_d = str(df["date_str"].max())
                cls._max_available_atmospheric_date = max_d
                cls._max_available_atmospheric_timestamp = f"{max_d}T18:00:00Z"

                # Fast in-memory indexing: (date_str, basin) -> dict of feature values
                cls._era5_index = {}
                records = df[["date_str", "basin"] + cls._era5_feature_cols].to_dict(orient="records")
                for rec in records:
                    d = rec["date_str"]
                    b = rec["basin"]
                    feat_dict = {col: rec[col] for col in cls._era5_feature_cols}
                    cls._era5_index[(d, b)] = feat_dict

                cls._initialized = True
                cls._init_failed = False
                cls._init_error_msg = None
                logger.info(
                    "ShadowV23Service initialized successfully. Max available atmosphere: %s",
                    cls._max_available_atmospheric_timestamp,
                )
                return True

            except Exception as e:
                cls._initialized = False
                cls._init_failed = True
                cls._init_error_msg = str(e)
                with cls._lock:
                    cls._metrics["initialization_failures"] += 1
                logger.warning("ShadowV23Service initialization failed (non-fatal to production): %s", e)
                return False

    @classmethod
    def dispatch_shadow_evaluation(
        cls,
        ocean_features: Any,
        site_id: str,
        horizon_days: int,
        date_str: str,
        v1_result: Optional[Dict[str, Any]] = None,
        prediction_timestamp: Optional[str] = None,
    ) -> bool:
        """
        Non-blocking dispatch of shadow evaluation to bounded queue.
        Guarantees:
        1. Never blocks production request latency.
        2. Drops job if queue is full (preserving memory safety).
        3. Never raises an exception to caller.
        Returns True if queued, False if dropped or disabled.
        """
        if not cls._enabled:
            return False

        cls._start_workers()

        with cls._lock:
            cls._metrics["jobs_submitted"] += 1

        task = {
            "ocean_features": ocean_features,
            "site_id": site_id,
            "horizon_days": horizon_days,
            "date_str": date_str,
            "v1_result": v1_result,
            "prediction_timestamp": prediction_timestamp,
        }

        try:
            cls._queue.put_nowait(task)
            return True
        except queue.Full:
            with cls._lock:
                cls._metrics["jobs_dropped"] += 1
            logger.warning(
                "SHADOW_QUEUE_FULL: Dropped shadow job for site=%s, date=%s (queue capacity=%d)",
                site_id,
                date_str,
                cls._queue.maxsize,
            )
            return False
        except Exception as e:
            logger.debug("Failed to dispatch shadow evaluation: %s", e)
            return False

    @classmethod
    def evaluate_shadow(
        cls,
        ocean_features: Any,
        site_id: str,
        horizon_days: int,
        date_str: str,
        v1_result: Optional[Dict[str, Any]] = None,
        prediction_timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes isolated Model D shadow prediction.
        Observational only. Returns telemetry record and logs append-only JSONL.
        Never raises exceptions.
        """
        start_time = time.time()
        now_utc = datetime.now(timezone.utc)
        created_at_iso = now_utc.isoformat()
        pred_ts = prediction_timestamp or created_at_iso

        v1_res = v1_result or {}
        v1_prob = v1_res.get("risk_score")
        v1_alert = v1_res.get("alert_level", "NO_ALERT")
        v1_version = "v1.1.0"
        shadow_version = "v2.3.0-model-d-shadow"

        # Baseline record template
        telemetry_record: Dict[str, Any] = {
            "timestamp": created_at_iso,
            "prediction_timestamp": pred_ts,
            "site": site_id,
            "horizon": int(horizon_days),
            "production_model_version": v1_version,
            "shadow_model_version": shadow_version,
            "production_probability": v1_prob,
            "shadow_probability": None,
            "production_alert": v1_alert,
            "shadow_alert_at_research_threshold": None,
            "ocean_timestamp": f"{date_str}T00:00:00Z",
            "atmosphere_timestamp": None,
            "data_quality_status": STATUS_INFERENCE_ERROR,
            "coverage_status": COV_SHADOW_FAILED,
            "feature_schema_version": "v2.3.0-224-features",
            "feature_schema_hash": TRUSTED_FEATURE_SCHEMA_HASH,
            "model_artifact_hash": TRUSTED_MODEL_HASHES.get(int(horizon_days)),
            "inference_status": "FAILED",
            "failure_reason": None,
            "created_at": created_at_iso,
        }

        try:
            # 1. Check Initialization
            if not cls._ensure_initialized():
                telemetry_record["failure_reason"] = cls._init_error_msg or "SHADOW_NOT_INITIALIZED"
                telemetry_record["coverage_status"] = COV_SHADOW_NOT_INITIALIZED
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            # 2. Causal / Temporal Safety Validation
            # Rule A: Prediction timestamp must not be in the future relative to runtime now
            try:
                # Compare dates
                req_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if req_date > now_utc.date():
                    telemetry_record["data_quality_status"] = STATUS_INVALID_TIMESTAMP
                    telemetry_record["failure_reason"] = f"Prediction date {date_str} is in the future relative to runtime"
                    telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                    cls._finalize_failure(telemetry_record, start_time)
                    return telemetry_record
                if prediction_timestamp:
                    try:
                        p_ts_norm = prediction_timestamp.replace("Z", "+00:00")
                        p_dt = datetime.fromisoformat(p_ts_norm)
                        if p_dt > now_utc:
                            telemetry_record["data_quality_status"] = STATUS_INVALID_TIMESTAMP
                            telemetry_record["failure_reason"] = f"Prediction timestamp {prediction_timestamp} is in the future relative to runtime"
                            telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                            cls._finalize_failure(telemetry_record, start_time)
                            return telemetry_record
                    except Exception:
                        pass
            except ValueError as e:
                telemetry_record["data_quality_status"] = STATUS_INVALID_TIMESTAMP
                telemetry_record["failure_reason"] = f"Invalid date format: {date_str}"
                telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            # Rule B: Exact Causal Cutoff Invariant
            # atmospheric_timestamp <= date_str 18:00:00 UTC
            cutoff_timestamp = f"{date_str}T18:00:00Z"
            telemetry_record["atmosphere_timestamp"] = cutoff_timestamp

            # 3. Spatial Basin Mapping
            norm_site = site_id.lower().strip()
            if norm_site in ["bay of bengal", "bob", "vizag", "chennai", "paradip"]:
                basin_key = "bob"
            elif norm_site in ["arabian sea", "aras", "kochi", "mumbai", "goa"]:
                basin_key = "aras"
            else:
                basin_key = "bob" if "bob" in norm_site else "aras"

            role_mapping = SPATIAL_MAPPING.get(basin_key)
            if not role_mapping:
                telemetry_record["data_quality_status"] = STATUS_FEATURE_SCHEMA_MISMATCH
                telemetry_record["failure_reason"] = f"Unknown spatial basin for site {site_id}"
                telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            # 4. Check Atmospheric Availability
            # All 3 sub-basins must be present for date_str
            missing_subbasins = []
            subbasin_data = {}
            for role_prefix, subbasin in role_mapping.items():
                data = cls._era5_index.get((date_str, subbasin))
                if data is None:
                    missing_subbasins.append(subbasin)
                else:
                    subbasin_data[role_prefix] = data

            if missing_subbasins:
                # Distinguish missing vs late
                max_d = cls._max_available_atmospheric_date
                elapsed_ms = (time.time() - start_time) * 1000.0
                if max_d and date_str > max_d:
                    telemetry_record["data_quality_status"] = STATUS_ATMOSPHERE_LATE
                    telemetry_record["coverage_status"] = COV_SHADOW_SKIPPED_LATE_DATA
                    telemetry_record["failure_reason"] = (
                        f"Atmospheric data late/pending: {date_str} > latest available {max_d}"
                    )
                    with cls._lock:
                        cls._metrics["jobs_executed"] += 1
                        cls._metrics["total_latency_ms"] += elapsed_ms
                        cls._metrics["late_atmosphere_count"] += 1
                        cls._metrics["data_unavailable_predictions"] += 1
                else:
                    telemetry_record["data_quality_status"] = STATUS_ATMOSPHERE_MISSING
                    telemetry_record["coverage_status"] = COV_SHADOW_SKIPPED_MISSING_DATA
                    telemetry_record["failure_reason"] = (
                        f"Atmospheric data missing for {missing_subbasins} on {date_str}"
                    )
                    with cls._lock:
                        cls._metrics["jobs_executed"] += 1
                        cls._metrics["total_latency_ms"] += elapsed_ms
                        cls._metrics["missing_atmosphere_count"] += 1
                        cls._metrics["data_unavailable_predictions"] += 1

                telemetry_record["inference_status"] = "SKIPPED"
                cls._log_telemetry(telemetry_record)
                return telemetry_record

            # 5. Assemble and Validate 224-Feature Vector
            # First 101 ocean features + 123 atmospheric features
            ocean_dict: Dict[str, float] = {}
            if isinstance(ocean_features, dict):
                ocean_dict = ocean_features
            elif isinstance(ocean_features, (list, np.ndarray)):
                expected_ocean_names = cls._expected_feature_names[:OCEAN_FEATURES_EXPECTED]
                if len(ocean_features) != OCEAN_FEATURES_EXPECTED:
                    telemetry_record["data_quality_status"] = STATUS_FEATURE_SCHEMA_MISMATCH
                    telemetry_record["failure_reason"] = (
                        f"Ocean feature count mismatch: expected {OCEAN_FEATURES_EXPECTED}, found {len(ocean_features)}"
                    )
                    telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                    with cls._lock:
                        cls._metrics["schema_mismatch_count"] += 1
                    cls._finalize_failure(telemetry_record, start_time)
                    return telemetry_record
                ocean_dict = {name: float(val) for name, val in zip(expected_ocean_names, ocean_features)}
            else:
                telemetry_record["data_quality_status"] = STATUS_FEATURE_SCHEMA_MISMATCH
                telemetry_record["failure_reason"] = f"Unsupported ocean features type: {type(ocean_features)}"
                telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                with cls._lock:
                    cls._metrics["schema_mismatch_count"] += 1
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            # Build full dictionary
            assembled_dict = dict(ocean_dict)
            for role_prefix, feat_map in subbasin_data.items():
                for col_name, val in feat_map.items():
                    assembled_dict[f"{role_prefix}_{col_name}"] = float(val)

            # Strict Contract Validation: 224 features matching expected order
            missing_cols = [col for col in cls._expected_feature_names if col not in assembled_dict]
            if missing_cols:
                telemetry_record["data_quality_status"] = STATUS_FEATURE_SCHEMA_MISMATCH
                telemetry_record["failure_reason"] = f"Missing {len(missing_cols)} expected features: {missing_cols[:5]}"
                telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                with cls._lock:
                    cls._metrics["schema_mismatch_count"] += 1
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            # Construct ordered vector
            vector_224 = np.array([assembled_dict[col] for col in cls._expected_feature_names], dtype=float)

            # Check NaNs / Infs
            if np.isnan(vector_224).any() or np.isinf(vector_224).any():
                telemetry_record["data_quality_status"] = STATUS_FEATURE_SCHEMA_MISMATCH
                telemetry_record["failure_reason"] = "NaN or Inf detected in 224-feature vector"
                telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                with cls._lock:
                    cls._metrics["schema_mismatch_count"] += 1
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            # 6. Model Inference (Random Forest Model D)
            h = int(horizon_days)
            model = cls._models.get(h)
            if model is None:
                telemetry_record["data_quality_status"] = STATUS_INFERENCE_ERROR
                telemetry_record["failure_reason"] = f"Model D not found for horizon {h}d"
                telemetry_record["coverage_status"] = COV_SHADOW_FAILED
                cls._finalize_failure(telemetry_record, start_time)
                return telemetry_record

            X = vector_224.reshape(1, -1)
            raw_prob = float(model.predict_proba(X)[0, 1])

            # Research Threshold Comparison (Strictly Observational / Research Only)
            res_thresh = RESEARCH_THRESHOLDS.get(h, 0.27)
            if raw_prob >= 0.50:
                shadow_research_alert = "RESEARCH_HIGH_ALERT"
            elif raw_prob >= res_thresh:
                shadow_research_alert = "RESEARCH_WATCH"
            else:
                shadow_research_alert = "RESEARCH_NO_ALERT"

            # 7. Finalize Success
            elapsed_ms = (time.time() - start_time) * 1000.0
            telemetry_record["data_quality_status"] = STATUS_READY
            telemetry_record["inference_status"] = "SUCCESS"
            telemetry_record["coverage_status"] = COV_SHADOW_SUCCESS
            telemetry_record["shadow_probability"] = round(raw_prob, 6)
            telemetry_record["shadow_alert_at_research_threshold"] = shadow_research_alert

            with cls._lock:
                cls._metrics["jobs_executed"] += 1
                cls._metrics["successful_predictions"] += 1
                cls._metrics["total_latency_ms"] += elapsed_ms
                if basin_key in cls._metrics["predictions_by_site"]:
                    cls._metrics["predictions_by_site"][basin_key] += 1
                if h in cls._metrics["predictions_by_horizon"]:
                    cls._metrics["predictions_by_horizon"][h] += 1
                if v1_prob is not None:
                    cls._metrics["probability_deltas"].append(abs(raw_prob - float(v1_prob)))

            cls._log_telemetry(telemetry_record)
            return telemetry_record

        except Exception as e:
            logger.warning("Shadow evaluation exception: %s", e)
            telemetry_record["data_quality_status"] = STATUS_INFERENCE_ERROR
            telemetry_record["inference_status"] = "FAILED"
            telemetry_record["coverage_status"] = COV_SHADOW_FAILED
            telemetry_record["failure_reason"] = str(e)
            cls._finalize_failure(telemetry_record, start_time)
            return telemetry_record

    @classmethod
    def _finalize_failure(cls, record: Dict[str, Any], start_time: float):
        elapsed_ms = (time.time() - start_time) * 1000.0
        with cls._lock:
            cls._metrics["jobs_executed"] += 1
            cls._metrics["failed_predictions"] += 1
            cls._metrics["total_latency_ms"] += elapsed_ms
        cls._log_telemetry(record)

    @classmethod
    def _log_telemetry(cls, record: Dict[str, Any]):
        """Append-only thread-safe shadow telemetry logging."""
        try:
            now = datetime.now(timezone.utc)
            filename = f"shadow_v2_3_telemetry_{now.strftime('%Y%m')}.jsonl"
            log_path = os.path.join(SHADOW_DATA_DIR, filename)
            line = json.dumps(record) + "\n"
            with cls._file_write_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
        except Exception as e:
            logger.debug("Failed to append shadow telemetry: %s", e)
