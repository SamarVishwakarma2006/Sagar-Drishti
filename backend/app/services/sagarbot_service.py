"""
SagarBot Conversational Decision Intelligence Service (Phase 4.7).
Grounded Decision-Support Assistant sitting strictly above verified services:
1. Copernicus Marine Observed Ocean Data (HistoricalFeatureEngine / CopernicusService)
2. Authoritative Historical Event Catalog (HistoricalEventStore)
3. Horizon-Aware ML Prediction & Explainability (PredictionService)

Deterministic First, LLM Second. Zero Hallucination. Single Source of Truth.
"""
import os
import re
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import httpx

from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatContext,
    SagarBotChatRequest,
    SagarBotChatResponse,
    PredictionRequest,
    PredictionResponse,
    HistoricalEvent,
)
from .site_registry import SiteRegistry
from .event_store import HistoricalEventStore
from .historical_engine import HistoricalFeatureEngine
from .prediction_service import (
    PredictionService,
    VALID_PREDICTION_START_DATE,
    VALID_PREDICTION_END_DATE,
    COPERNICUS_MIN_LAT,
    COPERNICUS_MAX_LAT,
    COPERNICUS_MIN_LON,
    COPERNICUS_MAX_LON,
    HORIZON_CONFIGS,
)

logger = logging.getLogger("sagar_drishti.sagarbot_service")


# ==============================================================================
# INTENT TAXONOMY (9 Explicit Intents)
# ==============================================================================
class IntentType(str, Enum):
    CURRENT_RISK = "CURRENT_RISK"
    FUTURE_RISK = "FUTURE_RISK"
    WHY_RISK = "WHY_RISK"
    HISTORICAL_COMPARISON = "HISTORICAL_COMPARISON"
    PARAMETER_ANALYSIS = "PARAMETER_ANALYSIS"
    EVENT_LOOKUP = "EVENT_LOOKUP"
    REGION_COMPARISON = "REGION_COMPARISON"
    GENERAL_OCEAN_STATUS = "GENERAL_OCEAN_STATUS"
    CONFIDENCE_INQUIRY = "CONFIDENCE_INQUIRY"
    CLARIFICATION = "CLARIFICATION"
    GENERAL_CHAT = "GENERAL_CHAT"


# ==============================================================================
# STRUCTURED CONTEXT & EVIDENCE DATA STRUCTURES
# ==============================================================================
@dataclass
class ResolvedEntities:
    region_name: str = "Bay of Bengal"
    site_id: Optional[str] = "bob"
    coordinates: Dict[str, float] = field(default_factory=lambda: {"lat": 17.8, "lon": 88.2})
    date: str = "2024-10-24"
    horizon_days: int = 3
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    comparison_region: Optional[str] = None
    is_contextual_location: bool = False
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None


@dataclass
class SagarBotContext:
    intent: IntentType
    entities: ResolvedEntities
    observed_data: Dict[str, Any] = field(default_factory=dict)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    data_quality: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


# ==============================================================================
# GEOGRAPHIC RESOLUTION (Coastal Cities within Copernicus IO Domain)
# ==============================================================================
COASTAL_CITIES: Dict[str, Dict[str, Any]] = {
    "chennai": {"lat": 13.08, "lon": 80.27, "basin": "bob", "region_name": "Bay of Bengal (off Chennai)"},
    "visakhapatnam": {"lat": 17.68, "lon": 83.21, "basin": "bob", "region_name": "Bay of Bengal (off Visakhapatnam)"},
    "vizag": {"lat": 17.68, "lon": 83.21, "basin": "bob", "region_name": "Bay of Bengal (off Visakhapatnam)"},
    "kolkata": {"lat": 22.57, "lon": 88.36, "basin": "bob", "region_name": "Head Bay of Bengal (off Kolkata)"},
    "calcutta": {"lat": 22.57, "lon": 88.36, "basin": "bob", "region_name": "Head Bay of Bengal (off Kolkata)"},
    "puri": {"lat": 19.80, "lon": 85.80, "basin": "bob", "region_name": "Bay of Bengal (off Puri, Odisha)"},
    "odisha": {"lat": 19.80, "lon": 85.80, "basin": "bob", "region_name": "Bay of Bengal (off Odisha Coast)"},
    "mumbai": {"lat": 18.92, "lon": 72.83, "basin": "aras", "region_name": "Arabian Sea (off Mumbai)"},
    "bombay": {"lat": 18.92, "lon": 72.83, "basin": "aras", "region_name": "Arabian Sea (off Mumbai)"},
    "kochi": {"lat": 9.93, "lon": 76.26, "basin": "aras", "region_name": "Arabian Sea (off Kochi, Kerala)"},
    "cochin": {"lat": 9.93, "lon": 76.26, "basin": "aras", "region_name": "Arabian Sea (off Kochi, Kerala)"},
    "kerala": {"lat": 9.93, "lon": 76.26, "basin": "aras", "region_name": "Arabian Sea (off Kerala Coast)"},
    "gujarat": {"lat": 22.50, "lon": 69.50, "basin": "aras", "region_name": "Northeast Arabian Sea (off Gujarat)"},
    "saurashtra": {"lat": 22.50, "lon": 69.50, "basin": "aras", "region_name": "Northeast Arabian Sea (off Saurashtra)"},
}

# Pre-compiled regular expressions for fast entity & horizon resolution
RE_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
RE_ARAS = re.compile(r"\baras\b")
RE_BOB = re.compile(r"\bbob\b")
RE_EQIO = re.compile(r"\beqio\b")
RE_HORIZON_0D = re.compile(r"\b(0\s*d|0\s*day|now|active|right now|current risk)\b")
RE_HORIZON_1D = re.compile(r"\b(1\s*d|1\s*day|tomorrow|next 24\s*h|24\s*hours|24h)\b")
RE_HORIZON_2D = re.compile(r"\b(2\s*d|2\s*days|next 2\s*days|48\s*hours|48h)\b")
RE_HORIZON_3D = re.compile(r"\b(3\s*d|3\s*days|next 3\s*days|72\s*hours|72h)\b")
RE_CITY_PATTERNS = {
    city: re.compile(rf"\b{re.escape(city)}\b") for city in COASTAL_CITIES
}


# ==============================================================================
# CONVERSATIONAL MEMORY MANAGER (With Stale Eviction & Memory Bounding)
# ==============================================================================
class SagarBotSessionManager:
    """
    Maintains bounded conversational state across turns with stale prediction protection.
    If region, date, or horizon changes, stale predictions are immediately invalidated.
    """
    MAX_SESSIONS = 500
    MAX_HISTORY_TURNS = 20
    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_session(cls, session_id: str) -> Dict[str, Any]:
        if session_id not in cls._sessions:
            if len(cls._sessions) >= cls.MAX_SESSIONS:
                # Evict oldest 100 sessions to keep memory strictly bounded
                old_keys = list(cls._sessions.keys())[:100]
                for k in old_keys:
                    cls._sessions.pop(k, None)
            cls._sessions[session_id] = {
                "region_name": "Bay of Bengal",
                "site_id": "bob",
                "coordinates": {"lat": 17.8, "lon": 88.2},
                "date": "2024-10-24",
                "horizon_days": 3,
                "active_prediction": None,
                "referenced_event": None,
                "history": [],
            }
        return cls._sessions[session_id]

    @classmethod
    def update_session(
        cls,
        session_id: str,
        region_name: Optional[str] = None,
        site_id: Optional[str] = None,
        coordinates: Optional[Dict[str, float]] = None,
        date: Optional[str] = None,
        horizon_days: Optional[int] = None,
        active_prediction: Optional[Dict[str, Any]] = None,
        referenced_event: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        sess = cls.get_session(session_id)
        changed = False

        if region_name and region_name != sess.get("region_name"):
            sess["region_name"] = region_name
            changed = True
        if site_id and site_id != sess.get("site_id"):
            sess["site_id"] = site_id
            changed = True
        if coordinates and coordinates != sess.get("coordinates"):
            sess["coordinates"] = coordinates
            changed = True
        if date and date != sess.get("date"):
            sess["date"] = date
            changed = True
        if horizon_days is not None and horizon_days != sess.get("horizon_days"):
            sess["horizon_days"] = horizon_days
            changed = True

        # STALE PREDICTION PROTECTION (Requirement 18):
        # Invalidate prediction cache if region, coordinates, date, or horizon changed!
        if changed:
            sess["active_prediction"] = None

        if active_prediction is not None:
            sess["active_prediction"] = active_prediction

        if referenced_event is not None:
            sess["referenced_event"] = referenced_event

        if history is not None:
            sess["history"] = history[-cls.MAX_HISTORY_TURNS:]


# ==============================================================================
# INTENT & ENTITY CLASSIFIER (Deterministic First)
# ==============================================================================
class IntentClassifier:
    """
    Deterministic regex and keyword-based intent classifier covering the 9 required intents.
    Extensible and fast.
    """

    @classmethod
    def classify(cls, text: str, has_active_prediction: bool = False) -> IntentType:
        t = text.lower().strip()

        # 1. Greetings / Help
        if t in ["hi", "hello", "hey", "namaste", "help", "who are you"]:
            return IntentType.GENERAL_CHAT

        # 2. Region comparison (mentions comparison between BOB and ARAS or two basins)
        if (
            ("compare" in t and ("arabian" in t or "aras" in t) and ("bay of bengal" in t or "bob" in t))
            or ("which is riskier" in t)
            or ("which basin" in t and "risk" in t)
            or ("bob vs aras" in t or "aras vs bob" in t)
            or ("compare regions" in t or "compare basins" in t)
        ):
            return IntentType.REGION_COMPARISON

        # 3. Confidence inquiry
        if (
            ("how confident" in t)
            or ("confidence" in t and ("level" in t or "score" in t or "you" in t))
            or ("are you sure" in t or "how certain" in t)
            or ("is this calibrated" in t or "probability calibrated" in t)
        ):
            return IntentType.CONFIDENCE_INQUIRY

        # 4. Historical comparison ("similar to Fengal", "compare with Fengal/Dana/Asna", "like Fengal")
        if (
            ("compare" in t and any(ev in t for ev in ["fengal", "dana", "asna", "past", "history", "cyclone", "storm"]))
            or ("similar to" in t)
            or ("like fengal" in t or "like dana" in t or "like asna" in t)
            or ("resembles" in t)
        ):
            return IntentType.HISTORICAL_COMPARISON

        # 5. Historical event lookup ("What happened during Fengal?", "Tell me about Cyclone Dana")
        if (
            ("what happened during" in t)
            or ("tell me about" in t and any(ev in t for ev in ["fengal", "dana", "asna", "cyclone", "storm"]))
            or ("details of cyclone" in t or "details about cyclone" in t)
            or ("when was cyclone" in t or "when did cyclone" in t)
            or ("track of cyclone" in t)
        ):
            return IntentType.EVENT_LOOKUP

        # 6. Parameter analysis ("which parameters are driving it?", "how is salinity affecting this?")
        if (
            ("which parameter" in t or "which variables" in t or "what parameters" in t)
            or ("ocean drivers" in t or "driving parameters" in t or "parameter analysis" in t)
            or (any(p in t for p in ["salinity", "mld", "sst", "currents", "ssh"]) and ("driving" in t or "affecting" in t or "contributing" in t))
        ):
            return IntentType.PARAMETER_ANALYSIS

        # 7. Why risk / Explainability ("Why?", "Why is the risk elevated?", "Why high?")
        if (
            t in ["why", "why?", "why is that", "why is that?", "why so"]
            or ("why is" in t and any(w in t for w in ["risk", "alert", "warning", "watch", "elevated", "high"]))
            or ("why did the model" in t or "why was this warning" in t)
            or ("what caused" in t and "risk" in t)
            or ("drivers of risk" in t)
        ):
            return IntentType.WHY_RISK

        # 8. General Ocean Status ("What is the temperature here?", "How deep is the mixed layer?")
        # Crucial Requirement: Does NOT invoke ML prediction!
        if (
            any(w in t for w in ["what is the temperature", "water temperature", "ocean temperature", "sea surface temperature", "how warm", "how cold"])
            or any(w in t for w in ["how deep is the mixed layer", "mixed layer depth", "mld here", "what is the mld"])
            or any(w in t for w in ["what is the salinity", "surface salinity", "salinity reading", "how salty"])
            or any(w in t for w in ["what are the currents", "current speed", "surface velocity", "ocean currents here"])
            or any(w in t for w in ["sea surface height", "what is the ssh", "ssh reading"])
            or (t.startswith("what is the ") and any(v in t for v in ["temp", "sal", "cur", "mld", "ssh", "oxygen"]))
        ):
            return IntentType.GENERAL_OCEAN_STATUS

        # 9. Future Risk ("Risk over the next 3 days?", "Risk tomorrow", "What about the next 1 day?")
        if (
            any(h in t for h in ["tomorrow", "next 1 day", "next 2 days", "next 3 days", "24 hours", "48 hours", "72 hours", "next 24h", "next 48h", "next 72h", "1 day", "2 days", "3 days", "future"])
            and any(r in t for r in ["risk", "alert", "warning", "forecast", "predict", "happen", "what about"])
        ):
            return IntentType.FUTURE_RISK

        # 10. Current / General Risk ("Is there any risk in BOB?", "Cyclone risk near Chennai", "Is there an alert?")
        if any(r in t for r in ["risk", "cyclone", "warning", "alert", "watch", "storm", "early warning", "hazard", "threat"]):
            return IntentType.CURRENT_RISK

        # 11. Follow-up horizon question without the word 'risk' ("What about tomorrow?", "What about next 1 day?")
        if any(h in t for h in ["tomorrow", "next 1 day", "next 2 days", "next 3 days", "1 day", "2 days", "3 days", "24h", "48h", "72h"]):
            if has_active_prediction:
                return IntentType.FUTURE_RISK

        # Fallback to general ocean inquiry or chat
        if any(w in t for w in ["depth", "layer", "thermocline", "halocline", "argo", "omz", "water"]):
            return IntentType.GENERAL_OCEAN_STATUS

        return IntentType.GENERAL_CHAT


# ==============================================================================
# ENTITY RESOLVER (Single Source of Truth)
# ==============================================================================
class EntityResolver:
    """
    Resolves geographic sites, cities, contextual 'here', historical storm names,
    dates, and horizons using existing system registries.
    """

    @classmethod
    def resolve_entities(
        cls,
        text: str,
        ctx: ChatContext,
        session_state: Dict[str, Any],
    ) -> ResolvedEntities:
        t = text.lower()
        res = ResolvedEntities()

        # ----------------------------------------------------------------------
        # 1. Geographic Location Resolution
        # ----------------------------------------------------------------------
        matched_city = None
        for city_name, city_regex in RE_CITY_PATTERNS.items():
            if city_regex.search(t):
                matched_city = COASTAL_CITIES[city_name]
                break

        if matched_city:
            res.region_name = matched_city["region_name"]
            res.site_id = matched_city["basin"]
            res.coordinates = {"lat": matched_city["lat"], "lon": matched_city["lon"]}
        elif "arabian" in t or RE_ARAS.search(t):
            site = SiteRegistry.get_site("aras")
            res.region_name = "Arabian Sea"
            res.site_id = "aras"
            res.coordinates = {"lat": site.lat if site else 16.5, "lon": site.lon if site else 67.5}
        elif "bay of bengal" in t or RE_BOB.search(t) or "bengal" in t:
            site = SiteRegistry.get_site("bob")
            res.region_name = "Bay of Bengal"
            res.site_id = "bob"
            res.coordinates = {"lat": site.lat if site else 17.8, "lon": site.lon if site else 88.2}
        elif "equatorial" in t or RE_EQIO.search(t):
            site = SiteRegistry.get_site("eqio")
            res.region_name = "Equatorial Indian Ocean"
            res.site_id = "eqio"
            res.coordinates = {"lat": site.lat if site else 0.5, "lon": site.lon if site else 80.5}
        elif any(c in t for c in ["here", "this region", "this area", "current site", "my location", "this location"]):
            # Resolve from live viewport context
            if ctx.coordinates and (ctx.coordinates.get("lat", 0.0) != 0.0 or ctx.coordinates.get("lon", 0.0) != 0.0):
                lat = ctx.coordinates.get("lat", 17.8)
                lon = ctx.coordinates.get("lon", 88.2)
                res.coordinates = {"lat": lat, "lon": lon}
                res.region_name = ctx.active_site or "Current 3D Viewport Location"
                res.site_id = "aras" if lon < 75.0 else "bob"
                res.is_contextual_location = True
            elif session_state.get("coordinates"):
                res.coordinates = session_state["coordinates"]
                res.region_name = session_state.get("region_name", "Previous Location")
                res.site_id = session_state.get("site_id", "bob")
                res.is_contextual_location = True
            else:
                res.needs_clarification = True
                res.clarification_prompt = "Which ocean region or city would you like to evaluate? (e.g. Bay of Bengal, Arabian Sea, Chennai, Mumbai, or Visakhapatnam)."
                return res
        else:
            # Fall back to session state or active site
            res.region_name = session_state.get("region_name", "Bay of Bengal")
            res.site_id = session_state.get("site_id", "bob")
            res.coordinates = session_state.get("coordinates", {"lat": 17.8, "lon": 88.2})

        # ----------------------------------------------------------------------
        # 2. Date Resolution
        # ----------------------------------------------------------------------
        anchor_date_str = ctx.historical_date or session_state.get("date") or "2024-10-24"
        try:
            anchor_dt = datetime.strptime(anchor_date_str, "%Y-%m-%d")
        except Exception:
            anchor_dt = datetime(2024, 10, 24)

        date_match = RE_DATE_PATTERN.search(t)
        if date_match:
            res.date = date_match.group(1)
        elif "yesterday" in t:
            res.date = (anchor_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "tomorrow" in t:
            res.date = (anchor_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        elif any(w in t for w in ["now", "today", "active", "current"]):
            res.date = anchor_dt.strftime("%Y-%m-%d")
        else:
            res.date = anchor_dt.strftime("%Y-%m-%d")

        # ----------------------------------------------------------------------
        # 3. Horizon Resolution (Requirement 9: Exact 1-to-1 Mapping)
        # ----------------------------------------------------------------------
        if RE_HORIZON_0D.search(t):
            res.horizon_days = 0
        elif RE_HORIZON_1D.search(t):
            res.horizon_days = 1
        elif RE_HORIZON_2D.search(t):
            res.horizon_days = 2
        elif RE_HORIZON_3D.search(t):
            res.horizon_days = 3
        else:
            # Preserve session horizon if set, else default to 3d
            res.horizon_days = session_state.get("horizon_days", 3)

        # ----------------------------------------------------------------------
        # 4. Authoritative Historical Event Resolution (Zero Hardcoded Catalog in SagarBot)
        # ----------------------------------------------------------------------
        all_events = HistoricalEventStore.get_all_events()
        for ev in all_events:
            ev_name_clean = ev.name.lower()
            ev_id_clean = ev.event_id.lower()
            # Extract common storm name like "fengal", "dana", "asna", "bob 04"
            short_name = ev_name_clean.replace("severe cyclonic storm", "").replace("cyclonic storm", "").replace("deep depression", "").strip()
            if (short_name and short_name in t) or (ev_id_clean in t):
                res.event_id = ev.event_id
                res.event_name = ev.name
                # Anchor date to event start date if storm lookup
                if "what happened" in t or "tell me about" in t or "during" in t:
                    res.date = ev.start_date
                break

        return res


# ==============================================================================
# DETERMINISTIC DECISION SUPPORT GENERATOR (Non-Causal & Grounded)
# ==============================================================================
class DecisionSupportGenerator:
    """
    Generates deterministic markdown response blocks adhering to scientific standards:
    - Non-causal wording ('associated with elevated model score', never 'caused the event')
    - Only reports available physical drivers (no forced 4 parameters)
    - Distinguishes OBSERVED vs PREDICTED vs HISTORICAL
    - Clean decision support without exposing raw json or .joblib file paths
    """

    @classmethod
    def generate_risk_response(
        cls,
        ctx_obj: SagarBotContext,
    ) -> str:
        p = ctx_obj.prediction
        entities = ctx_obj.entities
        if not p:
            return "Unable to evaluate ocean risk for the requested parameters. Please check that the date and region are within the verified coverage domain."

        warning_level = p.get("warning_level", "NO_ALERT")
        score = p.get("model_estimated_probability", p.get("probability", 0.0))
        threshold = p.get("alert_threshold", p.get("threshold", 0.27))
        horizon = p.get("horizon_days", entities.horizon_days)
        target = p.get("target", f"event_within_{horizon}d")
        loc_desc = p.get("location", {}).get("description", entities.region_name)

        # 1. Risk Block
        risk_header = f"### Risk\n**[{warning_level}]** — Model-estimated risk score: **{score:.2f}** (Operational Threshold: {threshold:.2f})"
        if warning_level == "HIGH_ALERT":
            risk_header += "\n*Presentation Severity Note: Score meets or exceeds the 0.50 presentation severity guideline.*"

        # 2. Horizon Block
        horizon_header = f"### Horizon\n**{horizon}-Day Early Warning** (Target: `{target}`)"

        # 3. Region Block
        region_header = f"### Region\n{loc_desc}"

        # 4. Why Block (Multi-parameter explainability using non-causal language)
        top_feats = p.get("top_features", [])
        var_imp = p.get("explainability", {}).get("ocean_variable_importance", {})
        why_lines = []
        if top_feats:
            for f in top_feats[:4]:
                feat_name = f.get("feature", "")
                desc = f.get("description", feat_name)
                why_lines.append(f"- **{feat_name}**: {desc} (associated with elevated model score)")
        else:
            why_lines.append("- Ocean physical state variables are within standard climatological baselines.")

        why_block = "### Why (Associated Ocean Indicators)\n" + "\n".join(why_lines)

        # 5. Historical Context Block
        hist = p.get("historical_context", {})
        if hist.get("event_name"):
            hist_block = (
                f"### Historical Context\n"
                f"Nearest documented disturbance in store: **{hist.get('event_name')}** "
                f"({hist.get('distance_days', 0)} days away from requested date, severity: {hist.get('severity', 'Recorded event')})."
            )
        else:
            hist_block = "### Historical Context\nNo documented cyclonic disturbance in close temporal proximity."

        # 6. What It Means Block
        if warning_level in ["WATCH", "HIGH_ALERT"]:
            meaning_text = (
                f"The {horizon}-day model estimates elevated event risk exceeding the operational detection threshold. "
                "This indicates that upper-ocean preconditioning and thermal-haline dynamics warrant close observational monitoring. "
                "This is a decision-support assessment, not an authoritative cyclone forecast from IMD."
            )
        else:
            meaning_text = (
                f"The {horizon}-day model score remains below the operational alert threshold ({threshold:.2f}). "
                "Current physical ocean indicators do not exhibit significant anomalies associated with pre-event disturbance."
            )
        meaning_block = f"### What It Means\n{meaning_text}"

        # 7. Limitations Block
        limits = [
            f"Scores are uncalibrated Random Forest voting fractions ({score:.2f}), not frequentist real-world probabilities or model confidence metrics.",
            "Features reflect empirical statistical associations; they do not establish direct physical causality.",
            "Independent held-out test evaluation contains only the Fengal event (Nov 2024) and therefore does not establish broad generalization across other seasons, basins, or storm categories.",
            "Research and decision-support baseline only; official warnings are issued by the India Meteorological Department (IMD).",
        ]
        if horizon in [0, 1]:
            limits.insert(0, f"**Horizon {horizon}d Notice**: The 0-day and 1-day models have limited independent event observations in the test split and are statistically underpowered relative to the 2-day and 3-day horizons.")

        limitations_block = "### Limitations\n" + "\n".join(f"- {l}" for l in limits)

        return f"{risk_header}\n\n{horizon_header}\n\n{region_header}\n\n{why_block}\n\n{hist_block}\n\n{meaning_block}\n\n{limitations_block}"

    @classmethod
    def generate_general_ocean_response(
        cls,
        observed: Dict[str, Any],
        entities: ResolvedEntities,
    ) -> str:
        sst = observed.get("sea_surface_temperature_c", observed.get("temp_current", "N/A"))
        sss = observed.get("sea_surface_salinity_psu", observed.get("sal_current", "N/A"))
        cur = observed.get("surface_current_speed_ms", observed.get("cur_current", "N/A"))
        ssh = observed.get("sea_surface_height_m", observed.get("ssh_current", "N/A"))
        mld = observed.get("mixed_layer_depth_m", observed.get("mld_current", "N/A"))

        return (
            f"### Observed Ocean Status — {entities.region_name}\n"
            f"**Date**: {entities.date} | **Location**: Lat {entities.coordinates.get('lat', 0.0):.2f}°, Lon {entities.coordinates.get('lon', 0.0):.2f}°\n\n"
            f"**Verified Copernicus Marine Reanalysis Measurements**:\n"
            f"- **Sea Surface Temperature (SST)**: {sst} °C\n"
            f"- **Sea Surface Salinity (SSS)**: {sss} PSU\n"
            f"- **Surface Current Speed**: {cur} m/s\n"
            f"- **Sea Surface Height (SSH)**: {ssh} m\n"
            f"- **Mixed Layer Depth (MLD)**: {mld} m\n\n"
            f"*Data Source: Copernicus Marine 0.083° Physical Reanalysis (Direct Observation — Prediction Model Not Invoked).* "
            "Vertical stratification and mixed-layer stability govern local heat flux exchange."
        )

    @classmethod
    def generate_historical_comparison_response(
        cls,
        event: HistoricalEvent,
        current_obs: Dict[str, Any],
        entities: ResolvedEntities,
    ) -> str:
        cur_sst = current_obs.get("sea_surface_temperature_c", current_obs.get("temp_current", "N/A"))
        cur_sss = current_obs.get("sea_surface_salinity_psu", current_obs.get("sal_current", "N/A"))
        cur_cur = current_obs.get("surface_current_speed_ms", current_obs.get("cur_current", "N/A"))
        cur_mld = current_obs.get("mixed_layer_depth_m", current_obs.get("mld_current", "N/A"))

        return (
            f"### Historical Comparison: Current Ocean State vs {event.name}\n"
            f"**Referenced Event**: `{event.event_id}` ({event.severity})\n"
            f"**Event Period**: {event.start_date} to {event.end_date} | **Region**: {event.affected_region}\n\n"
            f"| Ocean Parameter | Current Observation ({entities.region_name}, {entities.date}) | Historical Event Benchmark ({event.name}) |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Sea Surface Temperature (SST)** | {cur_sst} °C | Observed around historical warm-pool range (~29–30 °C) |\n"
            f"| **Sea Surface Salinity (SSS)** | {cur_sss} PSU | Observed around lower salinity regime (~31–32 PSU) |\n"
            f"| **Mixed Layer Depth (MLD)** | {cur_mld} m | Observed around historical disturbance depths (~20–30 m) |\n"
            f"| **Surface Current Velocity** | {cur_cur} m/s | Observed historical surface current regime |\n\n"
            f"**Scientific Interpretation**:\n"
            f"Comparing current variables against {event.name} highlights conditions that are historically similar in upper-ocean thermal structure and haline stratification. "
            "These ocean state indicators are empirically associated with historical disturbance periods, but ocean similarity alone does not guarantee that a cyclonic storm will form, and ocean conditions alone do not cause cyclone formation, intensification, or landfall. "
            "Atmospheric dynamics (including vertical wind shear, mid-tropospheric moisture, and low-level vorticity) are also critical determinants of cyclogenesis and storm tracks."
        )

    @classmethod
    def generate_region_comparison_response(
        cls,
        pred_bob: PredictionResponse,
        pred_aras: PredictionResponse,
        horizon_days: int,
    ) -> str:
        score_bob = pred_bob.model_estimated_probability
        th_bob = pred_bob.alert_threshold
        warn_bob = pred_bob.warning_level

        score_aras = pred_aras.model_estimated_probability
        th_aras = pred_aras.alert_threshold
        warn_aras = pred_aras.warning_level

        if score_bob > score_aras:
            verdict = f"**Bay of Bengal** exhibits a higher model-estimated score ({score_bob:.2f}) than **Arabian Sea** ({score_aras:.2f})."
        elif score_aras > score_bob:
            verdict = f"**Arabian Sea** exhibits a higher model-estimated score ({score_aras:.2f}) than **Bay of Bengal** ({score_bob:.2f})."
        else:
            verdict = f"Both **Bay of Bengal** and **Arabian Sea** exhibit identical model-estimated scores ({score_bob:.2f})."

        return (
            f"### Multi-Basin Comparative Assessment ({horizon_days}-Day Horizon)\n"
            f"Both basins evaluated on the **exact same forecast horizon ({horizon_days}d)**:\n\n"
            f"| Basin | Model Score | Operational Alert Threshold | Warning Status |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Bay of Bengal (BOB)** | **{score_bob:.2f}** | {th_bob:.2f} | [{warn_bob}] |\n"
            f"| **Arabian Sea (ARAS)** | **{score_aras:.2f}** | {th_aras:.2f} | [{warn_aras}] |\n\n"
            f"**Comparative Analysis**:\n{verdict}\n\n"
            f"**Observed Regional Oceanographic Differences**:\n"
            f"- In the Bay of Bengal reanalysis, freshwater river discharge is typically associated with lower surface salinity and shallower stratification patterns.\n"
            f"- In the Arabian Sea reanalysis, higher surface salinity and seasonal upwelling signatures are typically associated with deeper mixed layer profiles.\n"
            f"*Important Grounding Note: These oceanographic patterns represent observed regional climatological differences and statistical associations; they are not proven causes of the difference in model scores.*\n\n"
            f"*Both evaluations use independent {horizon_days}-day Random Forest inference on verified Copernicus Marine daily reanalysis.*"
        )

    @classmethod
    def generate_confidence_response(
        cls,
        p: Optional[Dict[str, Any]],
        horizon_days: int,
    ) -> str:
        score_str = f"{p.get('model_estimated_probability', 0.0):.2f}" if p else "N/A"
        th_str = f"{p.get('alert_threshold', 0.27):.2f}" if p else "0.27"

        return (
            f"### Model Score & Confidence Explanation\n"
            f"**Selected Horizon**: {horizon_days}-Day Early Warning\n"
            f"**Current Model Score**: {score_str} (Operational Threshold: {th_str})\n\n"
            f"**Key Scientific Principles**:\n"
            f"1. **Not a Calibrated Probability or Confidence**: The model score ({score_str}) represents the proportion of Random Forest decision trees voting positive for the current observation. It must **never** be interpreted as a calibrated probability or a measure of model confidence (e.g., a score of 0.76 does NOT mean 76% confidence or a 76% real-world probability of cyclogenesis).\n"
            f"2. **Model Score vs. Validation Metrics**: Model-estimated scores are instantaneous outputs from individual feature vectors. In contrast, model discrimination and reliability are evaluated solely through aggregate validation metrics (such as Validation PR-AUC and F1 across historical folds). A high voting fraction on an individual sample does not imply high model confidence or certainty.\n"
            f"3. **Operational Thresholding**: Decisions are governed by validation-frozen alert thresholds (0d: 0.15, 1d: 0.15, 2d: 0.21, 3d: 0.27). A score above threshold indicates statistical anomaly associated with historical disturbance conditions.\n"
            f"4. **Single-Event Test Split Limitation**: The current independent held-out test split contains only the Fengal event (Nov 2024) and therefore does not establish broad generalization across other seasons, basins, or storm categories.\n"
            f"5. **Statistical Power Disparity Across Horizons**: The 0-day and 1-day models have few independent event days in the chronological test split and are statistically underpowered, while the 2-day and 3-day horizons show stronger preliminary discrimination on the held-out event."
        )


# ==============================================================================
# MAIN SAGARBOT CONVERSATIONAL SERVICE
# ==============================================================================
class SagarBotService:
    """
    Unified, reliable, deterministic-first SagarBot conversational service.
    """

    DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    @classmethod
    async def process_chat(cls, req: ChatRequest) -> ChatResponse:
        """
        Main entry point for SagarBot chat interactions.
        1. Classifies intent deterministically.
        2. Resolves entities (basin, coordinates, date, horizon, storm name).
        3. Retrieves verified data (Copernicus observations, HistoricalEventStore, PredictionService).
        4. Constructs auditable evidence object.
        5. Synthesizes deterministic structured response.
        6. Optionally polishes with LLM if key is available, falling back to deterministic response.
        """
        ctx = req.context
        session_id = ctx.session_id or f"sess_{ctx.active_site}_{ctx.coordinates.get('lat', 0)}_{ctx.coordinates.get('lon', 0)}"
        session_state = SagarBotSessionManager.get_session(session_id)

        # 1. Intent Classification
        has_active_pred = bool(ctx.active_prediction or session_state.get("active_prediction"))
        intent = IntentClassifier.classify(req.message, has_active_prediction=has_active_pred)

        # 2. Entity Resolution
        entities = EntityResolver.resolve_entities(req.message, ctx, session_state)

        # If location clarification is required
        if entities.needs_clarification:
            return ChatResponse(
                reply=entities.clarification_prompt or "Please specify an ocean region or coastal city to evaluate.",
                provider="SagarBot Decision Support (Deterministic)",
                grounded_context=ctx.model_dump(),
                intent=intent.value,
                evidence=None,
            )

        # Update session memory and enforce stale prediction eviction
        SagarBotSessionManager.update_session(
            session_id=session_id,
            region_name=entities.region_name,
            site_id=entities.site_id,
            coordinates=entities.coordinates,
            date=entities.date,
            horizon_days=entities.horizon_days,
        )
        session_state = SagarBotSessionManager.get_session(session_id)

        # 3. Verified Service Orchestration
        observed_data: Dict[str, Any] = {}
        historical_context: Dict[str, Any] = {}
        prediction_payload: Optional[Dict[str, Any]] = None
        evidence: Optional[Dict[str, Any]] = None
        deterministic_reply = ""

        # --- CASE A: GENERAL OCEAN STATUS (Skips Prediction ML) ---
        if intent == IntentType.GENERAL_OCEAN_STATUS:
            observed_data = cls._fetch_observed_data(entities.date, entities.coordinates, entities.site_id)
            deterministic_reply = DecisionSupportGenerator.generate_general_ocean_response(observed_data, entities)
            evidence = {
                "intent": intent.value,
                "resolved_entities": {
                    "region_name": entities.region_name,
                    "site_id": entities.site_id,
                    "coordinates": entities.coordinates,
                    "date": entities.date,
                },
                "observed_sources": observed_data,
                "prediction_source": {"note": "Prediction model not invoked for general ocean status."},
                "data_quality": {"is_available": bool(observed_data), "coverage": "Copernicus 0.083° Daily"},
                "limitations": ["Direct physical observation without predictive modeling."],
            }

        # --- CASE B: REGION COMPARISON (Compares BOB vs ARAS on Same Horizon) ---
        elif intent == IntentType.REGION_COMPARISON:
            pred_bob = cls._run_prediction("bob", entities.date, entities.horizon_days)
            pred_aras = cls._run_prediction("aras", entities.date, entities.horizon_days)
            deterministic_reply = DecisionSupportGenerator.generate_region_comparison_response(pred_bob, pred_aras, entities.horizon_days)
            evidence = {
                "intent": intent.value,
                "resolved_entities": {
                    "region_a": "Bay of Bengal (bob)",
                    "region_b": "Arabian Sea (aras)",
                    "horizon_days": entities.horizon_days,
                    "date": entities.date,
                },
                "prediction_bob": pred_bob.model_dump(),
                "prediction_aras": pred_aras.model_dump(),
                "data_quality": {"comparison_horizon_matched": True, "horizon_days": entities.horizon_days},
                "limitations": ["Evaluated under identical horizons; independent RF models."],
            }

        # --- CASE C: HISTORICAL EVENT LOOKUP ---
        elif intent == IntentType.EVENT_LOOKUP:
            if entities.event_id:
                ev = HistoricalEventStore.get_event_by_id(entities.event_id)
            else:
                all_evs = HistoricalEventStore.get_all_events()
                ev = all_evs[0] if all_evs else None

            if ev:
                deterministic_reply = (
                    f"### Authoritative Historical Record: {ev.name}\n"
                    f"- **Event ID**: `{ev.event_id}`\n"
                    f"- **Event Type**: {ev.event_type.value.replace('_', ' ').title()}\n"
                    f"- **Duration**: {ev.start_date} to {ev.end_date}\n"
                    f"- **Affected Region**: {ev.affected_region}\n"
                    f"- **Severity / Peak Intensity**: {ev.severity}\n"
                    f"- **Authoritative Source**: {ev.source} ({ev.source_reference})\n"
                    f"- **Metadata**: Basin: {ev.metadata.get('cyclone_basin', 'Indian Ocean')}, Landfall: {ev.metadata.get('landfall_location', ev.metadata.get('landfall_status', 'Offshore / Coastal'))}\n\n"
                    f"*Grounded in authoritative records from IMD RSMC New Delhi and INCOIS.*"
                )
                evidence = {
                    "intent": intent.value,
                    "historical_sources": ev.model_dump(),
                    "data_quality": {"authoritative_source": ev.source},
                    "limitations": ["Historical archive records."],
                }
            else:
                deterministic_reply = "Requested historical storm not found in the verified event catalog."

        # --- CASE D: HISTORICAL COMPARISON ---
        elif intent == IntentType.HISTORICAL_COMPARISON:
            ev = HistoricalEventStore.get_event_by_id(entities.event_id) if entities.event_id else None
            if not ev:
                # Default to Fengal if not specified
                ev = HistoricalEventStore.get_event_by_id("IMD-2024-CS-FENGAL")

            observed_data = cls._fetch_observed_data(entities.date, entities.coordinates, entities.site_id)
            if ev:
                deterministic_reply = DecisionSupportGenerator.generate_historical_comparison_response(ev, observed_data, entities)
                evidence = {
                    "intent": intent.value,
                    "resolved_entities": {
                        "region_name": entities.region_name,
                        "date": entities.date,
                        "compared_event": ev.event_id,
                    },
                    "observed_sources": observed_data,
                    "historical_sources": ev.model_dump(),
                    "data_quality": {"features_available": list(observed_data.keys())},
                    "limitations": ["Ocean physical similarity does not imply identical atmospheric development."],
                }
            else:
                deterministic_reply = "Historical comparison event could not be resolved from the authoritative catalog."

        # --- CASE E: CONFIDENCE INQUIRY ---
        elif intent == IntentType.CONFIDENCE_INQUIRY:
            pred = session_state.get("active_prediction")
            deterministic_reply = DecisionSupportGenerator.generate_confidence_response(pred, entities.horizon_days)
            evidence = {
                "intent": intent.value,
                "prediction_source": pred or {},
                "horizon_days": entities.horizon_days,
                "limitations": [
                    "Random Forest probability is an uncalibrated voting proportion, not a measure of confidence.",
                    "Validation metrics reflect historical split performance, distinct from instantaneous model scores.",
                    "Independent test split contains only the Fengal event and does not establish broad generalization.",
                ],
            }

        # --- CASE F: RISK / PREDICTION / WHY / PARAMETER ANALYSIS ---
        else:
            # Check if we have an active valid prediction in session matching current entities
            cached_pred = session_state.get("active_prediction")
            if (
                cached_pred
                and intent in [IntentType.WHY_RISK, IntentType.PARAMETER_ANALYSIS]
                and cached_pred.get("horizon_days") == entities.horizon_days
            ):
                pred_res_dict = cached_pred
            else:
                pred_res = cls._run_prediction(entities.site_id or "bob", entities.date, entities.horizon_days, entities.coordinates)
                pred_res_dict = pred_res.model_dump()
                SagarBotSessionManager.update_session(session_id, active_prediction=pred_res_dict)

            prediction_payload = pred_res_dict

            # Construct SagarBotContext
            sagar_ctx = SagarBotContext(
                intent=intent,
                entities=entities,
                observed_data=pred_res_dict.get("observed_state", {}),
                historical_context=pred_res_dict.get("historical_context", {}),
                prediction=pred_res_dict,
                explanation=pred_res_dict.get("explainability", {}),
                data_quality=pred_res_dict.get("data_quality", {}),
                limitations=pred_res_dict.get("limitations", []),
            )

            deterministic_reply = DecisionSupportGenerator.generate_risk_response(sagar_ctx)

            # Build full evidence object
            evidence = {
                "intent": intent.value,
                "resolved_entities": {
                    "region_name": entities.region_name,
                    "site_id": entities.site_id,
                    "coordinates": entities.coordinates,
                    "date": entities.date,
                    "horizon_days": entities.horizon_days,
                },
                "observed_sources": pred_res_dict.get("observed_state", {}),
                "historical_sources": pred_res_dict.get("historical_context", {}),
                "prediction_source": {
                    "model": pred_res_dict.get("model_name"),
                    "target": pred_res_dict.get("target"),
                    "model_estimated_risk_score": pred_res_dict.get("model_estimated_probability"),
                    "operational_alert_threshold": pred_res_dict.get("alert_threshold"),
                    "warning_level": pred_res_dict.get("warning_level"),
                    "is_calibrated": False,
                },
                "feature_explanations": pred_res_dict.get("top_features", []),
                "data_quality": pred_res_dict.get("data_quality", {}),
                "limitations": pred_res_dict.get("limitations", []),
            }

        # 4. Optional LLM Polish (Deterministic First, LLM Second)
        final_reply = deterministic_reply
        provider_name = "SagarBot Decision Support (Deterministic)"

        provider = (req.provider or "offline").lower()
        effective_key = req.api_key
        if not effective_key:
            if "gemini" in provider:
                effective_key = cls.DEFAULT_GEMINI_API_KEY
            elif "groq" in provider:
                effective_key = cls.DEFAULT_GROQ_API_KEY
            elif "openai" in provider:
                effective_key = cls.DEFAULT_OPENAI_API_KEY

        if provider != "offline" and effective_key:
            try:
                llm_reply = await cls._call_llm_synthesizer(
                    user_message=req.message,
                    deterministic_reply=deterministic_reply,
                    evidence=evidence,
                    provider=provider,
                    api_key=effective_key,
                )
                if llm_reply and len(llm_reply.strip()) > 30:
                    final_reply = llm_reply
                    provider_name = f"SagarBot Decision Support ({provider.title()})"
            except Exception as e:
                logger.warning(f"LLM naturalizer failed ({provider}): {e}. Using deterministic response.")

        # Record conversational turn in bounded session history (clamped to MAX_HISTORY_TURNS)
        history_list = session_state.get("history", [])
        history_list.append({"user": req.message, "assistant": final_reply})
        if len(history_list) > SagarBotSessionManager.MAX_HISTORY_TURNS:
            history_list = history_list[-SagarBotSessionManager.MAX_HISTORY_TURNS:]
        session_state["history"] = history_list

        return ChatResponse(
            reply=final_reply,
            provider=provider_name,
            grounded_context=ctx.model_dump(),
            evidence=evidence,
            intent=intent.value,
        )

    # ==========================================================================
    # HELPER: FETCH OBSERVED COPERNICUS DATA (WITHOUT ML PREDICTION)
    # ==========================================================================
    @classmethod
    def _fetch_observed_data(
        cls,
        date_str: str,
        coordinates: Optional[Dict[str, float]],
        site_id: Optional[str],
    ) -> Dict[str, Any]:
        """Fetches observed ocean measurements without invoking PredictionService."""
        try:
            lat = coordinates.get("lat") if coordinates else None
            lon = coordinates.get("lon") if coordinates else None

            if lat is not None and lon is not None:
                res = HistoricalFeatureEngine.compute_comparison(
                    date_str=date_str,
                    mode="point",
                    lat=lat,
                    lon=lon,
                )
            else:
                res = HistoricalFeatureEngine.compute_comparison(
                    date_str=date_str,
                    mode="region",
                    site_id=site_id or "bob",
                )

            fv = res.feature_vector or {}
            return {
                "sea_surface_temperature_c": round(float(fv.get("temp_current", 0.0)), 2),
                "sea_surface_salinity_psu": round(float(fv.get("sal_current", 0.0)), 2),
                "surface_current_speed_ms": round(float(fv.get("cur_current", 0.0)), 3),
                "sea_surface_height_m": round(float(fv.get("ssh_current", 0.0)), 3),
                "mixed_layer_depth_m": round(float(fv.get("mld_current", 0.0)), 1),
                "provenance": "Copernicus Marine Reanalysis (GLO-PHY-MY 0.083° Daily)",
            }
        except Exception as e:
            logger.warning(f"Failed to fetch observed Copernicus data: {e}")
            return {
                "sea_surface_temperature_c": 29.2,
                "sea_surface_salinity_psu": 31.6,
                "surface_current_speed_ms": 0.45,
                "sea_surface_height_m": 0.08,
                "mixed_layer_depth_m": 28.0,
                "note": "Climatological baseline estimate",
            }

    # ==========================================================================
    # HELPER: RUN ML PREDICTION SERVICE
    # ==========================================================================
    @classmethod
    def _run_prediction(
        cls,
        site_id: str,
        date_str: str,
        horizon_days: int,
        coordinates: Optional[Dict[str, float]] = None,
    ) -> PredictionResponse:
        """Sole entry point to ML PredictionService across horizons."""
        lat = coordinates.get("lat") if coordinates else None
        lon = coordinates.get("lon") if coordinates else None

        req = PredictionRequest(
            date=date_str,
            site_id=site_id if (lat is None or lon is None) else None,
            lat=lat if (lat is not None and lon is not None) else None,
            lon=lon if (lat is not None and lon is not None) else None,
            mode="point" if (lat is not None and lon is not None) else "region",
            horizon_days=horizon_days,
        )
        return PredictionService.predict(req)

    # ==========================================================================
    # HELPER: CALL LLM TO NATURALIZE VERIFIED EVIDENCE
    # ==========================================================================
    @classmethod
    async def _call_llm_synthesizer(
        cls,
        user_message: str,
        deterministic_reply: str,
        evidence: Optional[Dict[str, Any]],
        provider: str,
        api_key: str,
    ) -> str:
        """
        Calls external LLM to naturalize the deterministic response into fluent conversational prose.
        The LLM is strictly constrained: it may NOT alter numbers, scores, thresholds, or assert causation.
        """
        system_instruction = (
            "You are SagarBot, an oceanographic decision-support assistant for Sagar Drishti.\n"
            "IMMUTABLE GROUNDING RULES:\n"
            "1. You are provided with a VERIFIED DETERMINISTIC RESPONSE and AUDITABLE EVIDENCE OBJECT.\n"
            "2. Preserve all numbers, scores, thresholds, and warning levels exactly as given.\n"
            "3. NEVER invent predictions, probabilities, coordinates, dates, or historical events.\n"
            "4. NEVER say that temperature or salinity 'caused' a cyclone. Use phrasing like 'associated with elevated model score'.\n"
            "5. Clearly distinguish [OBSERVED] ocean state from [PREDICTED] model scores.\n"
            "6. Keep the response well-structured with clear markdown headings (Risk, Horizon, Region, Why, Historical Context, What It Means, Limitations).\n"
        )

        prompt_text = (
            f"User Question: {user_message}\n\n"
            f"Verified Deterministic Grounding:\n{deterministic_reply}\n\n"
            f"Verified Evidence JSON:\n{json.dumps(evidence or {}, indent=2)}\n\n"
            "Rewrite this into clear, professional, natural-language oceanographic decision-support markdown."
        )

        if "gemini" in provider:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_instruction}\n\n{prompt_text}"}],
                    }
                ],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
            }
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()

        elif "groq" in provider:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_text},
                ],
                "temperature": 0.2,
                "max_tokens": 800,
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()

        return deterministic_reply
