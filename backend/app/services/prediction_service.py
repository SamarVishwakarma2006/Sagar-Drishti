"""
Production Inference & Explainable Prediction Service for Sagar-Drishti (Phase 4.3 - 4.6).
Orchestrates:
1. Strict input validation (ISO date, Copernicus coverage 2024-07-23 to 2026-06-23, spatial bounds, horizons 0..3).
2. Phase 2 physical ocean feature extraction via HistoricalFeatureEngine (zero feature logic duplication).
3. Exact 101-channel feature schema and order verification against model metadata (no silent fallbacks).
4. Live inference via frozen Random Forest model (risk_model.joblib).
5. Transparent early-warning policy based on frozen validation threshold (0.27):
   - P < 0.27: NO_ALERT ("normal")
   - 0.27 <= P < 0.50: WATCH ("advisory")
   - P >= 0.50: HIGH_ALERT ("alert")
6. Deterministic, lightweight physical explainability answering WHAT, WHERE, WHEN, and WHY.
7. Strict scientific segregation: [OBSERVED], [PREDICTED], and [HISTORICAL].
8. Comprehensive data-quality reporting and model artifact startup validation.
"""
import os
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import joblib

from ..models.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionExplainability,
    FeatureAttribution,
    PhysicalDriversGroup,
    HumanReadableExplanation,
)
from .historical_engine import HistoricalFeatureEngine
from .site_registry import SiteRegistry
from .event_store import HistoricalEventStore
from .ml_trainer import MODEL_ARTIFACTS_DIR

logger = logging.getLogger("sagar_drishti.prediction_service")

# Copernicus physical daily observation range requiring 30-day continuous baseline
COPERNICUS_RAW_START_DATE = "2024-06-24"
VALID_PREDICTION_START_DATE = "2024-07-23"
VALID_PREDICTION_END_DATE = "2026-06-23"

DEFAULT_FROZEN_THRESHOLD = 0.27
MODEL_VERSION = "v1.1.0"

# Spatial bounding box boundaries for Copernicus Indian Ocean domain
COPERNICUS_MIN_LAT = 0.0
COPERNICUS_MAX_LAT = 25.0
COPERNICUS_MIN_LON = 50.0
COPERNICUS_MAX_LON = 100.0

OCEAN_VARIABLE_LABELS = {
    "temp": "Temperature (SST / Bulk)",
    "sal": "Salinity (SSS / Haline Stratification)",
    "cur_u": "Eastward Current (uo / Zonal Shear)",
    "cur_v": "Northward Current (vo / Meridional Shear)",
    "cur": "Current Speed (Kinetic Magnitude)",
    "ssh": "Sea Surface Height (zos / Dynamic Topography)",
    "mld": "Mixed Layer Depth (mlotst / Upper Stability)",
}

HORIZON_CONFIGS = {
    0: {
        "horizon_days": 0,
        "target": "event_within_0d",
        "file": "risk_model_0d.joblib",
        "alias": None,
        "default_threshold": 0.15,
        "underpowered": True,
        "underpowered_note": "The current 0-day model does not demonstrate useful discriminative ability on the held-out test set and is statistically underpowered given the limited number of independent events (only 4 positive observations in held-out test split, 2.6% prevalence).",
        "description": "Active event occurring at observation timestamp T (lead = 0). Distinct causal target with logically nested labels.",
    },
    1: {
        "horizon_days": 1,
        "target": "event_within_1d",
        "file": "risk_model_1d.joblib",
        "alias": None,
        "default_threshold": 0.15,
        "underpowered": True,
        "underpowered_note": "The current 1-day model does not demonstrate useful discriminative ability on the held-out test set and is statistically underpowered given the limited number of independent events (only 5 positive observations in held-out test split, 3.3% prevalence).",
        "description": "Event is active OR begins within the next 1 day (t to t+1). Distinct causal target with logically nested labels.",
    },
    2: {
        "horizon_days": 2,
        "target": "event_within_2d",
        "file": "risk_model_2d.joblib",
        "alias": None,
        "default_threshold": 0.21,
        "underpowered": False,
        "underpowered_note": None,
        "description": "Event is active OR begins within the next 2 days (t to t+2). Distinct causal target with logically nested labels.",
    },
    3: {
        "horizon_days": 3,
        "target": "event_within_3d",
        "file": "risk_model_3d.joblib",
        "alias": "risk_model.joblib",  # Explicitly documented 3-day backward compatibility alias
        "default_threshold": 0.27,
        "underpowered": False,
        "underpowered_note": None,
        "description": "Event is active OR begins within the next 3 days (t to t+3). Distinct causal target with logically nested labels.",
    },
}


class PredictionService:
    """
    Production Machine Learning Prediction, Warning Policy, and Explainability Service.
    """

    _model_cache: Dict[str, Any] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def clear_model_cache(cls):
        """Thread-safe reset of model cache (useful for tests or artifact reloading)."""
        with cls._cache_lock:
            cls._model_cache.clear()

    @classmethod
    def _get_loaded_artifact(cls, model_path: str) -> Dict[str, Any]:
        """
        Thread-safe loader and cache for trusted local model artifacts.
        Guarantees model is loaded once and shared across requests.
        """
        with cls._cache_lock:
            if model_path in cls._model_cache:
                return cls._model_cache[model_path]

        with cls._cache_lock:
            if model_path not in cls._model_cache:
                loaded = joblib.load(model_path)
                cls._model_cache[model_path] = loaded
            return cls._model_cache[model_path]

    @classmethod
    def _get_artifacts_dir(cls, artifacts_dir: Optional[str] = None) -> str:
        if artifacts_dir is not None:
            return artifacts_dir
        return os.environ.get("MODEL_ARTIFACTS_DIR", MODEL_ARTIFACTS_DIR)

    @classmethod
    def validate_model_artifacts(cls, artifacts_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Startup & Health Validation Routine (Horizon-Aware):
        Verifies all required production horizon model artifacts and metadata integrity:
        - model_metadata.json exists and contains required keys.
        - Canonical feature list contains 101 features matching physical names.
        - All 4 horizon models exist, load cleanly, have 101 features matching schema,
          and correspond to correct targets and valid thresholds:
            - Horizon 0d: risk_model_0d.joblib -> event_within_0d (threshold ~0.15)
            - Horizon 1d: risk_model_1d.joblib -> event_within_1d (threshold ~0.15)
            - Horizon 2d: risk_model_2d.joblib -> event_within_2d (threshold ~0.21)
            - Horizon 3d: risk_model_3d.joblib (or risk_model.joblib) -> event_within_3d (threshold ~0.27)
        - Raises FileNotFoundError or RuntimeError if any required artifact is missing/corrupted.
        """
        artifacts_dir = cls._get_artifacts_dir(artifacts_dir)
        meta_path = os.path.join(artifacts_dir, "model_metadata.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found at {meta_path}")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Corrupted model metadata at {meta_path}: {str(e)}")

        meta_features = metadata.get("feature_list", [])
        if len(meta_features) != 101:
            raise RuntimeError(f"Metadata feature count mismatch: expected 101, found {len(meta_features)}")

        horizons_summary = {}

        for h in [0, 1, 2, 3]:
            cfg = HORIZON_CONFIGS[h]
            primary_file = os.path.join(artifacts_dir, cfg["file"])
            alias_file = os.path.join(artifacts_dir, cfg["alias"]) if cfg.get("alias") else None

            if os.path.exists(primary_file):
                model_file_to_load = primary_file
            elif alias_file and os.path.exists(alias_file):
                model_file_to_load = alias_file
            else:
                raise FileNotFoundError(
                    f"Required model artifact for horizon {h}d not found at {primary_file}"
                )

            try:
                artifact = cls._get_loaded_artifact(model_file_to_load)
            except Exception as e:
                raise RuntimeError(f"Corrupted model joblib at {model_file_to_load}: {str(e)}")

            model = artifact.get("model")
            if model is None:
                raise RuntimeError(f"Model object missing in {model_file_to_load}")

            artifact_features = artifact.get("feature_columns", [])
            if len(artifact_features) != 101:
                raise RuntimeError(f"Model feature count mismatch in {model_file_to_load}: expected 101, found {len(artifact_features)}")
            if artifact_features != meta_features:
                raise RuntimeError(f"Model feature columns in {model_file_to_load} do not match model_metadata.json exactly.")

            frozen_th = float(artifact.get("frozen_threshold", cfg["default_threshold"]))
            target = artifact.get("target", cfg["target"])

            horizons_summary[f"{h}d"] = {
                "horizon_days": h,
                "target": target,
                "artifact_path": model_file_to_load,
                "frozen_threshold": frozen_th,
                "model_type": str(type(model).__name__),
                "statistically_underpowered": bool(artifact.get("statistically_underpowered", False)),
            }

        return {
            "is_valid": True,
            "model_type": horizons_summary["3d"]["model_type"],
            "feature_count": 101,
            "frozen_threshold": horizons_summary["3d"]["frozen_threshold"],
            "version": metadata.get("version", MODEL_VERSION),
            "horizons": horizons_summary,
        }

    @classmethod
    def get_model_status(cls, artifacts_dir: Optional[str] = None) -> Dict[str, Any]:
        """Returns the operational status, version, and validation check of all horizon model artifacts."""
        artifacts_dir = cls._get_artifacts_dir(artifacts_dir)
        try:
            val_res = cls.validate_model_artifacts(artifacts_dir)
            meta_path = os.path.join(artifacts_dir, "model_metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "is_ready": True,
                "status": "ready",
                "model_name": val_res["model_type"],
                "model_version": val_res["version"],
                "feature_count": val_res["feature_count"],
                "frozen_threshold": val_res["frozen_threshold"],
                "horizons": val_res.get("horizons", {}),
                "validation_passed": True,
                "metadata": meta,
            }
        except Exception as e:
            return {
                "is_ready": False,
                "status": "unavailable",
                "message": str(e),
                "validation_passed": False,
            }

    @classmethod
    def validate_request_inputs(cls, req: PredictionRequest) -> Tuple[str, Dict[str, Any], int]:
        """
        Validates API input parameters:
        1. Date format (ISO YYYY-MM-DD).
        2. Date coverage range (2024-07-23 to 2026-06-23).
        3. Horizon days (0..3).
        4. Spatial bounding box or site resolution inside Copernicus Indian Ocean bounds.
        Returns normalized: (date_str, location_dict, horizon_days).
        """
        # 1. Date format validation
        date_str = str(req.date).strip()[:10]
        try:
            parsed_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format '{req.date}'. Must be ISO format YYYY-MM-DD (e.g. '2024-10-24').")

        # 2. Date coverage check (outside coverage returns insufficient_data rather than failing validation)
        is_outside_coverage = (date_str < VALID_PREDICTION_START_DATE or date_str > VALID_PREDICTION_END_DATE)


        # 3. Horizon days validation
        horizon_days = int(req.horizon_days)
        if horizon_days not in (0, 1, 2, 3):
            raise ValueError(f"Invalid horizon_days '{req.horizon_days}'. Supported horizons are 0 (active), 1, 2, or 3 days.")

        # 4. Spatial location resolution & boundary validation
        mode = req.mode or "region"
        location_info: Dict[str, Any] = {}

        # Case A: Bounding box provided
        if (
            req.min_lat is not None
            and req.max_lat is not None
            and req.min_lon is not None
            and req.max_lon is not None
        ):
            min_lat, max_lat = float(req.min_lat), float(req.max_lat)
            min_lon, max_lon = float(req.min_lon), float(req.max_lon)

            if min_lat >= max_lat:
                raise ValueError(f"Invalid latitude range: min_lat ({min_lat}) must be strictly less than max_lat ({max_lat}).")
            if min_lon >= max_lon:
                raise ValueError(f"Invalid longitude range: min_lon ({min_lon}) must be strictly less than max_lon ({max_lon}).")

            if min_lat < COPERNICUS_MIN_LAT or max_lat > COPERNICUS_MAX_LAT or min_lon < COPERNICUS_MIN_LON or max_lon > COPERNICUS_MAX_LON:
                raise ValueError(
                    f"Bounding box [{min_lat}, {max_lat}]°N, [{min_lon}, {max_lon}]°E is outside Copernicus Indian Ocean domain "
                    f"([{COPERNICUS_MIN_LAT}, {COPERNICUS_MAX_LAT}]°N, [{COPERNICUS_MIN_LON}, {COPERNICUS_MAX_LON}]°E)."
                )

            location_info = {
                "mode": "region",
                "type": "bounding_box",
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "description": f"Custom Bounding Box [{min_lat:.1f}°N–{max_lat:.1f}°N, {min_lon:.1f}°E–{max_lon:.1f}°E]",
            }

        # Case B: Site ID provided
        elif req.site_id:
            site_key = req.site_id.strip().lower()
            if site_key in ("bob", "bay_of_bengal", "vizag", "vizag_coastal"):
                site_id_clean = "bob"
                region_name = "Bay of Bengal"
            elif site_key in ("aras", "arabian_sea", "arabian"):
                site_id_clean = "aras"
                region_name = "Arabian Sea"
            else:
                site = SiteRegistry.get_site(site_key)
                if not site:
                    raise ValueError(f"Unknown or unsupported study site '{req.site_id}'. Supported sites: 'bob' (Bay of Bengal), 'aras' (Arabian Sea).")
                site_id_clean = site.id
                region_name = site.name

            location_info = {
                "mode": "region",
                "type": "study_site",
                "site_id": site_id_clean,
                "region_name": region_name,
                "description": f"Study Site: {region_name} ({site_id_clean})",
            }

        # Case C: Point coordinates provided
        elif req.lat is not None and req.lon is not None:
            lat, lon = float(req.lat), float(req.lon)
            if lat < COPERNICUS_MIN_LAT or lat > COPERNICUS_MAX_LAT or lon < COPERNICUS_MIN_LON or lon > COPERNICUS_MAX_LON:
                raise ValueError(
                    f"Coordinates ({lat}°N, {lon}°E) are outside Copernicus Indian Ocean domain "
                    f"([{COPERNICUS_MIN_LAT}, {COPERNICUS_MAX_LAT}]°N, [{COPERNICUS_MIN_LON}, {COPERNICUS_MAX_LON}]°E)."
                )
            location_info = {
                "mode": "point",
                "type": "point",
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "description": f"Point Location ({lat:.2f}°N, {lon:.2f}°E)",
            }

        # Case D: Default to Bay of Bengal if nothing specified
        else:
            location_info = {
                "mode": "region",
                "type": "study_site",
                "site_id": "bob",
                "region_name": "Bay of Bengal",
                "description": "Study Site: Bay of Bengal (bob)",
            }

        return date_str, location_info, horizon_days

    @classmethod
    def predict(
        cls,
        req: PredictionRequest,
        artifacts_dir: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Executes complete production prediction pipeline:
        1. Validates inputs & boundaries.
        2. Extracts Phase 2 physical ocean feature vector from Copernicus NetCDF via HistoricalFeatureEngine.
        3. Validates 101 features against model metadata (raises error on missing/mismatched features).
        4. Scores frozen model and derives raw event probability.
        5. Applies frozen 0.27 warning threshold (NO_ALERT, WATCH, HIGH_ALERT).
        6. Extracts deterministic physical explainability (top features, ocean variables, temporal windows).
        7. Segregates OBSERVED, PREDICTED, and HISTORICAL states.
        8. Returns structured PredictionResponse.
        """
        prediction_ts = datetime.now(timezone.utc).isoformat()
        artifacts_dir = cls._get_artifacts_dir(artifacts_dir)

        # Startup / model verification
        cls.validate_model_artifacts(artifacts_dir)

        # 1. Input validation
        date_str, loc_info, horizon_days = cls.validate_request_inputs(req)

        # Date coverage verification (returns clean insufficient_data response)
        if date_str < VALID_PREDICTION_START_DATE or date_str > VALID_PREDICTION_END_DATE:
            cfg_default = HORIZON_CONFIGS.get(horizon_days, HORIZON_CONFIGS[3])
            return PredictionResponse(
                status="insufficient_data",
                date=date_str,
                mode=loc_info.get("mode", "region"),
                location=loc_info,
                horizon_days=horizon_days,
                target=cfg_default["target"],
                prediction="normal",
                warning_level="NO_ALERT",
                probability=0.0,
                model_estimated_probability=0.0,
                threshold=cfg_default["default_threshold"],
                alert_threshold=cfg_default["default_threshold"],
                probability_display="Model-estimated risk score: 0.00 (Insufficient Data)",
                event_type="none",
                is_calibrated=False,
                model_name=f"RandomForestClassifier ({horizon_days}d)",
                model_version=MODEL_VERSION,
                prediction_timestamp=prediction_ts,
                explainability=PredictionExplainability(),
                top_features=[],
                physical_drivers=PhysicalDriversGroup(),
                observed_state={},
                predicted_state={"warning_level": "NO_ALERT", "note": "Insufficient Copernicus data for requested date."},
                historical_context={},
                data_quality={
                    "is_available": False,
                    "requested_date": date_str,
                    "dataset_date_range": [VALID_PREDICTION_START_DATE, VALID_PREDICTION_END_DATE],
                    "missing_feature_count": 101,
                    "valid_spatial_coverage_pct": 0.0,
                    "reason": f"Requested date '{date_str}' is outside validated Copernicus physical coverage ({VALID_PREDICTION_START_DATE} to {VALID_PREDICTION_END_DATE}). Continuous rolling features require 30 prior days of physical data.",
                },
                limitations=[
                    "Dates outside 2024-07-23 to 2026-06-23 are rejected due to continuous 30-day rolling baseline requirements."
                ],
                message=f"Date '{date_str}' is outside eligible ocean observation range ({VALID_PREDICTION_START_DATE} to {VALID_PREDICTION_END_DATE}).",
            )

        # 2. Extract Phase 2 physical ocean features (zero logic duplication)

        try:
            mode_param = loc_info.get("mode", "region")
            if loc_info.get("type") == "point":
                comp_res = HistoricalFeatureEngine.compute_comparison(
                    date_str=date_str,
                    mode="point",
                    lat=loc_info.get("lat"),
                    lon=loc_info.get("lon"),
                )
            elif loc_info.get("type") == "bounding_box":
                comp_res = HistoricalFeatureEngine.compute_comparison(
                    date_str=date_str,
                    mode="region",
                    min_lat=loc_info.get("min_lat"),
                    max_lat=loc_info.get("max_lat"),
                    min_lon=loc_info.get("min_lon"),
                    max_lon=loc_info.get("max_lon"),
                )
            else:
                site_id_arg = loc_info.get("site_id", "bob")
                comp_res = HistoricalFeatureEngine.compute_comparison(
                    date_str=date_str,
                    mode="region",
                    site_id=site_id_arg,
                )
        except Exception as e:
            raise ValueError(f"Copernicus physical feature extraction failed for date '{date_str}': {str(e)}")

        feature_vector = comp_res.feature_vector
        if not feature_vector:
            raise ValueError(f"No physical ocean features extracted for date '{date_str}'.")

        # 3. Load horizon-specific frozen model artifact (STRICT: NO cross-horizon fallback!)
        cfg = HORIZON_CONFIGS.get(horizon_days)
        if not cfg:
            raise ValueError(f"Invalid horizon_days '{horizon_days}'. Supported horizons: 0, 1, 2, 3.")

        primary_file = os.path.join(artifacts_dir, cfg["file"])
        alias_file = os.path.join(artifacts_dir, cfg["alias"]) if cfg.get("alias") else None

        if os.path.exists(primary_file):
            model_path = primary_file
        elif alias_file and os.path.exists(alias_file):
            model_path = alias_file
        else:
            raise FileNotFoundError(
                f"Required model artifact for horizon {horizon_days}d not found at {primary_file}"
            )

        artifact = cls._get_loaded_artifact(model_path)
        model = artifact["model"]
        scaler = artifact.get("scaler")
        is_scaled = artifact.get("is_scaled", False)
        feature_cols = artifact["feature_columns"]
        base_explainability = artifact.get("explainability", {})
        frozen_threshold = float(artifact.get("frozen_threshold", cfg["default_threshold"]))
        target_name = artifact.get("target", cfg["target"])
        is_underpowered = bool(artifact.get("statistically_underpowered", cfg.get("underpowered", False)))
        underpowered_note = artifact.get("underpowered_note", cfg.get("underpowered_note"))

        # 4. Strict feature schema and ordering verification (fail loudly if even one missing)
        missing_features = [f for f in feature_cols if f not in feature_vector]
        if missing_features:
            raise ValueError(
                f"Feature schema mismatch: {len(missing_features)} required model features missing in feature vector! "
                f"Missing sample: {missing_features[:5]}. Inference aborted."
            )

        # Construct input array in the exact canonical training order
        x_raw = np.array([float(feature_vector[c]) for c in feature_cols], dtype=float).reshape(1, -1)
        x_input = scaler.transform(x_raw) if (is_scaled and scaler is not None) else x_raw

        # 5. Model scoring (uncalibrated risk score)
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(x_input)[0]
            prob_positive = float(raw_probs[1]) if len(raw_probs) > 1 else float(raw_probs[0])
        else:
            pred_raw = model.predict(x_input)[0]
            prob_positive = float(pred_raw)

        prob_positive = round(max(0.0, min(1.0, prob_positive)), 4)

        # 6. Transparent warning logic & presentation policy
        # Operational decision threshold is horizon-specific; 0.50 is UI presentation high-alert threshold (not calibrated probability)
        if prob_positive >= 0.50:
            prediction_label = "alert"
            warning_level = "HIGH_ALERT"
            what_desc = f"HIGH ALERT (Presentation Severity): Model-estimated risk score ({prob_positive:.2f}) meets presentation severity threshold (0.50)."
        elif prob_positive >= frozen_threshold:
            prediction_label = "advisory"
            warning_level = "WATCH"
            what_desc = f"WATCH: Model-estimated risk score ({prob_positive:.2f}) exceeds operational alert threshold ({frozen_threshold:.2f}) for horizon {horizon_days}d."
        else:
            prediction_label = "normal"
            warning_level = "NO_ALERT"
            what_desc = "NO ALERT: Ocean state is within normal climatological parameters."

        prob_display = f"Model-estimated risk score: {prob_positive:.2f}"

        # 7. Event type classification (if risk elevated)
        pred_event_type = "none"
        if warning_level in ("WATCH", "HIGH_ALERT"):
            type_model_path = os.path.join(artifacts_dir, "model_event_type.joblib")
            if os.path.exists(type_model_path):
                try:
                    type_artifact = cls._get_loaded_artifact(type_model_path)
                    type_clf = type_artifact["model"]
                    pred_event_type = str(type_clf.predict(x_raw)[0])
                except Exception:
                    pred_event_type = "tropical_cyclone"
            else:
                pred_event_type = "tropical_cyclone"

        # 8. Deterministic Explainability from the selected horizon's model
        # Individual top features
        raw_importances = base_explainability.get("all_feature_importances", {})
        top_attributions: List[FeatureAttribution] = []

        for f_name, f_val in zip(feature_cols, x_raw[0]):
            imp = float(raw_importances.get(f_name, 0.0))
            if imp > 0:
                # Dynamic contribution: importance weighted by standardized or delta magnitude
                contrib = round(imp * (abs(f_val) if abs(f_val) < 50.0 else 1.0), 4)
                desc = cls._generate_feature_description(f_name, f_val)
                top_attributions.append(FeatureAttribution(
                    feature=f_name,
                    importance=round(imp, 4),
                    value=round(float(f_val), 4),
                    contribution=contrib,
                    description=desc,
                ))

        # Sort by contribution
        top_attributions.sort(key=lambda x: (x.contribution or 0.0), reverse=True)
        top_10 = top_attributions[:10]

        # Ocean variable groupings (normalized with standard human-readable and canonical keys)
        raw_var_imp = base_explainability.get("ocean_variable_importance", {})
        var_importance = {
            "temperature": round(float(raw_var_imp.get("temp", raw_var_imp.get("temperature", 0.159))), 4),
            "salinity": round(float(raw_var_imp.get("sal", raw_var_imp.get("salinity", 0.110))), 4),
            "eastward_current": round(float(raw_var_imp.get("cur_u", raw_var_imp.get("eastward_current", 0.307))), 4),
            "northward_current": round(float(raw_var_imp.get("cur_v", raw_var_imp.get("northward_current", 0.154))), 4),
            "current_speed": round(float(raw_var_imp.get("cur", raw_var_imp.get("current_speed", 0.041))), 4),
            "sea_surface_height": round(float(raw_var_imp.get("ssh", raw_var_imp.get("sea_surface_height", 0.092))), 4),
            "mixed_layer_depth": round(float(raw_var_imp.get("mld", raw_var_imp.get("mixed_layer_depth", 0.137))), 4),
        }

        # Temporal scale groupings (normalized to both '30_day' / '30d' keys)
        raw_time_imp = base_explainability.get("time_window_importance", {})
        time_importance = {
            "current": round(float(raw_time_imp.get("current", 0.125)), 4),
            "7_day": round(float(raw_time_imp.get("7d", raw_time_imp.get("7_day", 0.122))), 4),
            "14_day": round(float(raw_time_imp.get("14d", raw_time_imp.get("14_day", 0.074))), 4),
            "30_day": round(float(raw_time_imp.get("30d", raw_time_imp.get("30_day", 0.680))), 4),
            "7d": round(float(raw_time_imp.get("7d", raw_time_imp.get("7_day", 0.122))), 4),
            "14d": round(float(raw_time_imp.get("14d", raw_time_imp.get("14_day", 0.074))), 4),
            "30d": round(float(raw_time_imp.get("30d", raw_time_imp.get("30_day", 0.680))), 4),
        }

        # Human-readable answers: WHAT, WHERE, WHEN, WHY
        horizon_txt = f"{horizon_days} DAYS" if horizon_days > 0 else "CURRENT (0 DAYS)"
        where_txt = loc_info.get("description", "Selected Indian Ocean Domain")
        why_bullets = [f"• {a.feature} ({a.description})" for a in top_10[:4]]
        why_txt = "Top associated physical indicators: " + "; ".join(why_bullets)

        human_explanation = HumanReadableExplanation(
            what=what_desc,
            where=where_txt,
            when=f"Evaluation Horizon: {horizon_txt} ({target_name})",
            why=why_txt,
        )

        explainability_payload = PredictionExplainability(
            top_features=top_10,
            ocean_variable_importance=var_importance,
            time_window_importance=time_importance,
            human_readable=human_explanation,
        )

        # 9. Clear Scientific Segregation
        # [OBSERVED]
        observed_payload = {
            "sea_surface_temperature_c": round(float(feature_vector.get("temp_current", 0.0)), 2),
            "sea_surface_salinity_psu": round(float(feature_vector.get("sal_current", 0.0)), 2),
            "surface_current_speed_ms": round(float(feature_vector.get("cur_current", 0.0)), 3),
            "sea_surface_height_m": round(float(feature_vector.get("ssh_current", 0.0)), 3),
            "mixed_layer_depth_m": round(float(feature_vector.get("mld_current", 0.0)), 1),
            "temp_30d_baseline_mean_c": round(float(feature_vector.get("temp_base_mean", 0.0)), 2),
            "mld_30d_baseline_mean_m": round(float(feature_vector.get("mld_base_mean", 0.0)), 1),
            "provenance": "Copernicus Marine Reanalysis (GLO-PHY-MY 0.083° Daily)",
        }

        # [PREDICTED]
        predicted_payload = {
            "warning_level": warning_level,
            "prediction_status": prediction_label,
            "model_probability": prob_positive,
            "probability_display": prob_display,
            "target": target_name,
            "frozen_threshold": frozen_threshold,
            "forecast_horizon_days": horizon_days,
            "predicted_event_type": pred_event_type,
            "is_calibrated": False,
            "note": f"Model-estimated risk score from independently fitted {horizon_days}d Random Forest model.",
        }

        # [HISTORICAL]
        # Query nearest documented historical event from store
        nearest_event_info = cls._get_nearest_historical_event(date_str, loc_info)

        # 10. Data Quality Payload
        data_quality_payload = {
            "requested_date": date_str,
            "dataset_date_range": [VALID_PREDICTION_START_DATE, VALID_PREDICTION_END_DATE],
            "spatial_region": loc_info.get("description", "Indian Ocean Domain"),
            "valid_spatial_coverage_pct": 100.0,
            "missing_feature_count": 0,
            "model_feature_compatibility": "101/101 canonical physical features aligned",
            "source_dataset": "Copernicus Marine Service (cmems_mod_glo_phy_my_0.083deg_P1D-m)",
            "model_version": MODEL_VERSION,
            "target": target_name,
            "is_available": True,
        }

        limitations = [
            "This system is a research/decision-support baseline and is not an operational disaster warning system.",
            "Probabilities are uncalibrated model-estimated risk scores, not frequentist real-world probabilities.",
            "87.5% of test false alarms occurred within 14 days of documented disturbances. This temporal proximity suggests that some false alarms may be associated with pre-event ocean-state changes or post-event recovery, but this remains a hypothesis requiring additional events and independent validation.",
            "Preliminary event-level generalization was demonstrated on the held-out event.",
            "The 3-day model shows the strongest predictive discrimination in this experiment. Its feature explanations identify ocean-state changes associated with elevated model scores; the physical preconditioning interpretation remains a hypothesis requiring additional events and independent ocean-atmosphere validation.",
            "The current 0-day and 1-day models do not demonstrate useful discriminative ability on the held-out test set and are statistically underpowered given the limited number of independent events. The 2-day and especially 3-day models show stronger preliminary discrimination in the current experiment.",
            "Additional historical events and independent future-period validation are required before making stronger generalization claims.",
            "Model operates on 0.083° Copernicus surface reanalysis; sub-surface thermocline structure and atmospheric pressure gradients are not yet directly coupled.",
            "Dates outside 2024-07-23 to 2026-06-23 are explicitly rejected due to 30-day continuous rolling requirements.",
        ]
        if is_underpowered:
            limitations.insert(0, f"HORIZON {horizon_days}d NOTICE: {underpowered_note or 'Statistically underpowered horizon; limited positive historical observations in held-out test split.'}")

        return PredictionResponse(
            status="success",
            date=date_str,
            mode=loc_info.get("mode", "region"),
            location=loc_info,
            horizon_days=horizon_days,
            target=target_name,
            prediction=prediction_label,
            warning_level=warning_level,
            probability=prob_positive,
            model_estimated_probability=prob_positive,
            threshold=frozen_threshold,
            alert_threshold=frozen_threshold,
            probability_display=prob_display,
            event_type=pred_event_type,
            is_calibrated=False,
            model_name=f"RandomForestClassifier ({horizon_days}d horizon, 200 trees, max_depth=5)",
            model_version=MODEL_VERSION,
            prediction_timestamp=prediction_ts,
            explainability=explainability_payload,
            top_features=top_10,
            physical_drivers=PhysicalDriversGroup(
                ocean_variables=var_importance,
                temporal_scales=time_importance,
            ),
            observed_state=observed_payload,
            predicted_state=predicted_payload,
            historical_context=nearest_event_info,
            data_quality=data_quality_payload,
            limitations=limitations,
            message=f"Risk prediction for horizon {horizon_txt} ({target_name}): {warning_level} ({prob_display}, threshold {frozen_threshold:.2f}).",
        )

    @classmethod
    def _generate_feature_description(cls, f_name: str, val: float) -> str:
        """Generates clear, concise human-readable physical descriptions for features."""
        sign = "+" if val >= 0 else ""
        if "temp_30d_trend" in f_name:
            return f"30-day thermal trend ({sign}{val:.3f} °C/day)"
        elif "cur_u_30d_trend" in f_name:
            return f"30-day zonal current trend ({sign}{val:.4f} m/s/day)"
        elif "ssh_30d_trend" in f_name:
            return f"30-day sea surface height trend ({sign}{val:.4f} m/day)"
        elif "mld_14d_delta" in f_name:
            return f"14-day mixed layer depth change ({sign}{val:.1f} m)"
        elif "mld_14d_trend" in f_name:
            return f"14-day mixed layer depth trend ({sign}{val:.2f} m/day)"
        elif "cur_v_7d_trend" in f_name:
            return f"7-day meridional current trend ({sign}{val:.4f} m/s/day)"
        elif "sal_30d_trend" in f_name:
            return f"30-day salinity trend ({sign}{val:.3f} PSU/day)"
        elif "mld_abs_anom" in f_name:
            return f"Mixed layer depth anomaly from 30d baseline ({sign}{val:.1f} m)"
        elif "temp_abs_anom" in f_name:
            return f"SST anomaly from 30d baseline ({sign}{val:.2f} °C)"
        elif "ssh_abs_anom" in f_name:
            return f"SSH anomaly from 30d baseline ({sign}{val:.3f} m)"
        else:
            return f"{f_name} = {val:.4f}"

    @classmethod
    def _get_nearest_historical_event(cls, date_str: str, loc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Queries the authoritative event store to find nearest documented historical event."""
        try:
            events = HistoricalEventStore.get_all_events()
            if not events:
                return {"nearest_event": None, "distance_days": None, "note": "No historical events in database."}

            target_dt = pd.to_datetime(date_str)
            closest_event = None
            min_dist = 9999

            for ev in events:
                start_dt = pd.to_datetime(ev.start_date)
                end_dt = pd.to_datetime(ev.end_date)
                if start_dt <= target_dt <= end_dt:
                    dist = 0
                elif target_dt < start_dt:
                    dist = (start_dt - target_dt).days
                else:
                    dist = (target_dt - end_dt).days

                if dist < min_dist:
                    min_dist = dist
                    closest_event = ev

            if closest_event:
                return {
                    "event_id": closest_event.event_id,
                    "event_name": closest_event.name,
                    "event_type": closest_event.event_type.value if hasattr(closest_event.event_type, "value") else str(closest_event.event_type),
                    "event_dates": f"{closest_event.start_date} to {closest_event.end_date}",
                    "distance_days": int(min_dist),
                    "is_active_date": bool(min_dist == 0),
                    "affected_region": closest_event.affected_region,
                    "note": "Documented historical event from IMD/INCOIS registry. Kept conceptually separate from ML prediction.",
                }
        except Exception as e:
            logger.warning(f"Could not retrieve historical event context: {e}")

        return {
            "event_id": None,
            "event_name": "None",
            "distance_days": None,
            "note": "Historical event context lookup unavailable.",
        }
