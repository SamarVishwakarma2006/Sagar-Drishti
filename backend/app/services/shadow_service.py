"""
Shadow Inference Service for Sagar-Drishti v2_10yr Candidate Model.

Executes non-blocking, completely isolated shadow predictions alongside the v1.1.0 production baseline:
1. Evaluates candidate v2_10yr Random Forest ensemble on the identical 101 physical feature vector.
2. Applies post-hoc Isotonic probability calibration.
3. Evaluates decoupled basin-specific operational thresholds and risk tiers.
4. Streams telemetry records to `backend/data/shadow/shadow_telemetry_{YYYYMM}.jsonl`.
5. Guarantees ZERO impact on production responses, status codes, or disaster alerts.
"""
import os
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import numpy as np
import joblib

from .risk_tier_mapper import evaluate_operational_decision, normalize_basin_key

logger = logging.getLogger("sagar_drishti.shadow_service")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CANDIDATE_MODELS_DIR = os.path.join(BASE_DIR, "models", "v2_10yr")
CALIBRATION_DIR = os.path.join(CANDIDATE_MODELS_DIR, "calibration")
SHADOW_DATA_DIR = os.path.join(BASE_DIR, "data", "shadow")

os.makedirs(SHADOW_DATA_DIR, exist_ok=True)

class ShadowInferenceService:
    _lock = threading.Lock()
    _candidate_models: Dict[int, Any] = {}
    _calibrators: Dict[int, Any] = {}
    _initialized: bool = False
    _candidate_version: str = "v2.0.0-10yr-candidate"

    @classmethod
    def _ensure_loaded(cls) -> bool:
        if cls._initialized:
            return True
        with cls._lock:
            if cls._initialized:
                return True
            try:
                for h in [0, 1, 2, 3]:
                    model_path = os.path.join(CANDIDATE_MODELS_DIR, f"risk_model_{h}d.joblib")
                    calib_path = os.path.join(CALIBRATION_DIR, f"calibrator_{h}d.joblib")
                    
                    if os.path.exists(model_path):
                        m_obj = joblib.load(model_path)
                        cls._candidate_models[h] = m_obj["model"] if isinstance(m_obj, dict) and "model" in m_obj else m_obj
                    else:
                        logger.warning(f"Shadow model missing: {model_path}")
                        
                    if os.path.exists(calib_path):
                        c_obj = joblib.load(calib_path)
                        cls._calibrators[h] = c_obj["calibrator"] if isinstance(c_obj, dict) and "calibrator" in c_obj else c_obj
                    else:
                        logger.warning(f"Shadow calibrator missing: {calib_path}")
                
                cls._initialized = True
                return True
            except Exception as e:
                logger.error(f"Failed to load shadow model artifacts: {e}")
                return False

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._candidate_models.clear()
            cls._calibrators.clear()
            cls._initialized = False

    @classmethod
    def run_shadow_evaluation(
        cls,
        feature_vector: np.ndarray,
        site_id: str,
        horizon_days: int,
        date_str: str,
        v1_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute shadow prediction for v2_10yr.
        Strictly isolated: returns payload and logs telemetry.
        Never raises exceptions to callers.
        """
        try:
            if not cls._ensure_loaded():
                return None
                
            h = int(horizon_days)
            model = cls._candidate_models.get(h)
            calibrator = cls._calibrators.get(h)
            
            if model is None:
                return None
                
            # Compute raw prediction score
            X = np.asarray(feature_vector, dtype=float).reshape(1, -1)
            raw_score = float(model.predict_proba(X)[0, 1])
            
            # Compute calibrated probability
            if calibrator is not None:
                calibrated_prob = float(np.clip(calibrator.predict(np.array([raw_score]))[0], 0.0, 1.0))
            else:
                calibrated_prob = raw_score
                
            # Evaluate operational alert policy (persistence, thresholds, escalation)
            from .alert_engine_v2 import OperationalAlertEngineV2
            engine_decision = OperationalAlertEngineV2.evaluate_decision(
                calibrated_prob=calibrated_prob,
                raw_score=raw_score,
                site_or_basin=site_id,
                horizon_days=h
            )
            
            # Construct shadow telemetry record
            telemetry_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_date": date_str,
                "site_id": site_id,
                "basin": engine_decision["basin"],
                "basin_key": engine_decision["basin_key"],
                "horizon": h,
                "production_v1": {
                    "model_version": "v1.1.0",
                    "risk_score": v1_result.get("risk_score") if v1_result else None,
                    "alert_level": v1_result.get("alert_level") if v1_result else None,
                    "threshold": v1_result.get("threshold") if v1_result else None,
                } if v1_result else None,
                "candidate_v2": {
                    "model_version": engine_decision["model_version"],
                    "basin": engine_decision["basin"],
                    "horizon": h,
                    "raw_score": engine_decision["raw_score"],
                    "calibrated_probability": engine_decision["calibrated_probability"],
                    "risk_tier": engine_decision["risk_tier"],
                    "policy_threshold": engine_decision["policy_threshold"],
                    "persistence_state": engine_decision["persistence_state"],
                    "alert_decision": engine_decision["alert_decision"],
                    "alert_reason": engine_decision["alert_reason"],
                    # Backward compatibility aliases
                    "operational_threshold": engine_decision["policy_threshold"],
                    "operational_alert": engine_decision["alert_decision"],
                    "alert": engine_decision["alert_decision"],
                }
            }
            
            # Non-blocking log to disk
            cls._log_telemetry(telemetry_record)
            return telemetry_record

            
        except Exception as e:
            logger.warning(f"Shadow inference encountered non-fatal error: {e}")
            return None

    @classmethod
    def _log_telemetry(cls, record: Dict[str, Any]):
        """Append shadow telemetry record to JSONL log file."""
        try:
            now = datetime.now(timezone.utc)
            filename = f"shadow_telemetry_{now.strftime('%Y%m')}.jsonl"
            log_path = os.path.join(SHADOW_DATA_DIR, filename)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"Failed to append shadow telemetry: {e}")
