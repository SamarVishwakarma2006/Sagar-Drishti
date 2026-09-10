import os
import re
from typing import Dict, Any, Optional
from ..models.schemas import ChatContext, ChatRequest, ChatResponse


class OceanAIService:
    """
    Context-aware Oceanographic AI service for SagarBot.
    Embeds live 3D viewport state into system grounding prompts and uses
    offline physics-based expert reasoning (deterministic mode).
    """

    SYSTEM_PROMPT_TEMPLATE = """You are SagarBot, an expert oceanographic AI assistant embedded inside "Sagar Drishti" (Immersive Ocean Observatory).
You have real-time telemetry from the user's active 3D visualization viewport.

IMMUTABLE GROUNDING CONTEXT:
- Active Study Site: {active_site}
- Spatial Coordinates: Lat {lat}°, Lon {lon}°
- Current Depth: {current_depth}
- Active Variable: {variable}
- Live Value at Observer: {current_value}
- Temporal Offset: {time_offset}
- In-situ Argo Floats in Sector: {nearby_floats}
- Dataset Mode: {data_mode}

CORE INSTRUCTIONS:
1. Always ground your explanation in physical oceanography (thermodynamics, hydrodynamics, Coriolis deflection, mixed layer dynamics, thermocline, halocline barrier layers, oxygen minimum zones, and geostrophic balance).
2. Directly reference the observer's live depth ({current_depth}) and variable ({variable} = {current_value}) in your answers.
3. If asked about data provenance, mention Argo float delayed-mode QC, INCOIS INDOMOD reanalysis, or user-uploaded NetCDF/CSV datasets.
4. Keep answers concise, highly informative, and scientifically rigorous.
"""

    # LLM providers disabled — deterministic offline mode only.
    DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


    @classmethod
    async def process_chat(cls, req: ChatRequest) -> ChatResponse:
        ctx = req.context
        lat = ctx.coordinates.get("lat", 0.0)
        lon = ctx.coordinates.get("lon", 0.0)
        data_mode = "User Uploaded Dataset" if ctx.custom_data else "INCOIS Baseline Model / Argo"
        nearby_str = ", ".join(ctx.nearby_floats) if ctx.nearby_floats else "None currently in range"
        pred_summary = "None requested / normal background conditions."
        if ctx.active_prediction:
            p = ctx.active_prediction
            h_days = p.get("horizon_days", 3)
            tgt = p.get("target", f"event_within_{h_days}d")
            score = p.get("model_estimated_probability", p.get("probability", 0.0))
            th = p.get("alert_threshold", p.get("threshold", 0.27))
            top_feats = [f.get("feature", "") for f in p.get("top_features", [])[:4]]
            observed = p.get("observed_state", {})
            historical = p.get("historical_context", {})
            obs_summary = f"SST {observed.get('sea_surface_temperature_c', 'N/A')}°C, SSS {observed.get('sea_surface_salinity_psu', 'N/A')} PSU, Current {observed.get('surface_current_speed_ms', 'N/A')} m/s, MLD {observed.get('mixed_layer_depth_m', 'N/A')} m" if observed else "Observed conditions segregated"
            hist_summary = f"Nearest: {historical.get('event_name', 'None')} ({historical.get('distance_days', 'N/A')} days away)" if historical.get("event_name") else "No nearby historical disaster"

            pred_summary = (
                f"Selected Horizon: {h_days}d (Target Model: {tgt}), "
                f"Warning Level: {p.get('warning_level', 'NO_ALERT')}, "
                f"Model-Estimated Risk Score: {score:.3f} (Operational Horizon Threshold: {th:.2f}), "
                f"Event Type: {p.get('event_type', 'none')}, "
                f"Top Associated Physical Drivers: {', '.join(top_feats) if top_feats else 'None'}, "
                f"[OBSERVED]: {obs_summary}, "
                "SCIENTIFIC GUARDRAILS: Each horizon has a distinct causal target definition, with logically nested labels. 0d and 1d models do not demonstrate useful discriminative ability and are statistically underpowered; 2d and especially 3d models show stronger preliminary discrimination. Risk scores are uncalibrated empirical associations. Do NOT assert causation."
            )

        # Always use the offline deterministic physics engine.
        # External LLM providers (Gemini, Groq, OpenAI) are disabled.
        expert_reply = cls._offline_ocean_expert(req.message, ctx)
        return ChatResponse(
            reply=expert_reply,
            provider="SagarBot Oceanographic Physics Engine (Offline)",
            grounded_context=ctx.model_dump()
        )



    @classmethod
    async def _call_groq(cls, message: str, system_prompt: str, api_key: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    @classmethod
    async def _call_openai(cls, message: str, system_prompt: str, api_key: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    @classmethod
    def _offline_ocean_expert(cls, text: str, ctx: ChatContext) -> str:
        """
        Deterministic, offline expert engine grounded in oceanographic water column physics.
        """
        t = text.lower()
        vl = ctx.variable.lower()
        d_str = ctx.current_depth
        site_name = ctx.active_site
        val_str = ctx.current_value

        # Extract numeric depth if possible
        d_num = 60.0
        m = re.search(r"(\d+)", d_str)
        if m:
            d_num = float(m.group(1))

        if any(w in t for w in ["hi", "hello", "hey", "namaste", "greet", "who are you"]):
            return (
                f"Namaste! I am SagarBot, your oceanographic AI co-pilot. I am monitoring your live telemetry at "
                f"{d_str} in {site_name}. You are currently visualizing {ctx.variable} (current value: {val_str}). "
                f"Ask me anything about thermocline gradients, salinity barrier layers, oxygen minimum zones (OMZs), "
                f"or in-situ Argo profiling floats!"
            )

        if any(w in t for w in ["temp", "temperature", "warm", "cold", "thermocline", "heat"]):
            if d_num < 45:
                regime = "inside the sunlit, wind-stirred Mixed Layer where temperature is quasi-uniform"
            elif d_num < 600:
                regime = "within the permanent Thermocline where solar heating drops off exponentially and rapid vertical stratification occurs"
            else:
                regime = "in the deep ocean interior, characterized by cold, dense Antarctic/North Atlantic deep water (~2–4 °C)"
            return (
                f"Temperature Analysis for {site_name}:\n"
                f"• Viewport Depth: {d_str}\n"
                f"• Observed Reading: {val_str}\n"
                f"• Physical Regime: You are {regime}.\n"
                f"Vertical temperature gradients here govern buoyancy frequency (Brunt–Väisälä) and suppress vertical turbulent mixing."
            )

        if any(w in t for w in ["sal", "salinity", "halocline", "psu", "fresh", "barrier"]):
            return (
                f"Salinity Analysis for {site_name}:\n"
                f"• Viewport Depth: {d_str}\n"
                f"• Active Value: {val_str}\n"
                f"• Layer Physics: In the Bay of Bengal and monsoon-impacted basins, river runoff produces a sharp Halocline (salt stratification). "
                f"This creates a 'Barrier Layer' between the mixed layer base and isothermal layer, trapping heat and fueling tropical cyclogenesis."
            )

        if any(w in t for w in ["cur", "current", "flow", "velocity", "stream", "speed", "drift", "eddy"]):
            return (
                f"Hydrodynamics & Velocity Field for {site_name}:\n"
                f"• Active Current Value: {val_str} at depth {d_str}\n"
                f"• Flow Mechanics: Upper layer currents are driven by Ekman transport and wind stress curl, whereas subsurface currents are "
                f"in geostrophic balance (pressure gradient balancing the Coriolis acceleration $f = 2\\Omega \\sin\\phi$). "
                f"Mesoscale eddies (100–300 km diameter) visible in the flow field transport heat and nutrients across the basin."
            )

        if any(w in t for w in ["oxy", "oxygen", "o2", "omz", "suboxic", "hypoxia", "anoxia"]):
            is_omz = 150 <= d_num <= 800
            status = "INSIDE the intense Oxygen Minimum Zone (OMZ)" if is_omz else "outside the core OMZ depth range"
            return (
                f"Dissolved Oxygen Analysis for {site_name}:\n"
                f"• O₂ at Observer: {val_str} at depth {d_str}\n"
                f"• OMZ Status: You are currently {status}.\n"
                f"• Biogeochemistry: High biological surface export combined with sluggish intermediate ventilation drives heavy bacterial respiration, "
                f"depleting dissolved O₂ and inducing active denitrification at mid-depths (200–800 m)."
            )

        if any(w in t for w in ["argo", "float", "instrument", "glider", "ctd", "platform", "sensor"]):
            floats_info = f"Active floats nearby: {nearby_str}." if ctx.nearby_floats else "No active float directly intersecting current cone."
            return (
                f"In-Situ Instrument Telemetry:\n"
                f"• Sector: {site_name} (Lat {lat:.2f}°, Lon {lon:.2f}°)\n"
                f"• {floats_info}\n"
                f"• Argo Cycle Dynamics: APEX/PROVOR floats drift at a 1,000 m parking depth for 10 days, descend to 2,000 m, and profile "
                f"CTD parameters to the surface, transmitting real-time data via Iridium satellite."
            )

        if any(w in t for w in ["risk", "cyclone", "storm", "warning", "early warning", "predict", "forecast", "alert", "advisory", "extreme", "heatwave"]):
            pred_info = None
            if ctx.active_prediction:
                pred_info = ctx.active_prediction
            else:
                try:
                    from .prediction_service import PredictionService
                    from ..models.schemas import PredictionRequest
                    p_date = ctx.historical_date or "2024-08-01"
                    req_p = PredictionRequest(
                        date=p_date,
                        lat=lat,
                        lon=lon,
                        mode="point",
                        horizon_days=3,
                    )
                    pred_res = PredictionService.predict(req_p)
                    pred_info = pred_res.model_dump()
                except Exception:
                    pred_info = None

            if pred_info:
                prob = pred_info.get("model_estimated_probability", pred_info.get("probability", 0.0))
                thresh = pred_info.get("alert_threshold", pred_info.get("threshold", 0.27))
                warning_lvl = pred_info.get("warning_level", "NO_ALERT")
                ev_type = str(pred_info.get("event_type", "cyclone / severe weather")).replace("_", " ").title()
                horizon = pred_info.get("horizon_days", 3)
                target_name = pred_info.get("target", f"event_within_{horizon}d")
                top_feats = [f"{f['feature']} ({round(f.get('importance', 0)*100, 1)}%)" for f in pred_info.get("top_features", pred_info.get("explainability", {}).get("top_features", []))[:3]]
                top_str = ", ".join(top_feats) if top_feats else "current velocity change and SST anomalies"

                return (
                    f"AI Early Warning Assessment for {site_name}:\n"
                    f"• Forecast Horizon: {horizon} Days (Model Target: {target_name})\n"
                    f"• Warning Status: [{warning_lvl}] (Operational Threshold: {thresh:.2f})\n"
                    f"• Model-Estimated Risk Score: {prob:.2f} ({prob*100:.1f}% uncalibrated RF score)\n"
                    f"• Event Archetype: {ev_type}\n"
                    f"• Top Associated Physical Indicators: {top_str}\n"
                    f"• Data Source: Copernicus Marine 0.083° Physical Reanalysis\n"
                    f"• Scientific Guardrails: Each horizon has a distinct causal target definition, with logically nested labels. The current 0-day and 1-day models do not demonstrate useful discriminative ability on the held-out test set and are statistically underpowered given the limited number of independent events; the 2-day and especially 3-day models show stronger preliminary discrimination in the current experiment. Risk scores are uncalibrated empirical associations. Feature importance does not prove physical causality. Research/decision-support baseline only, not an operational warning system."
                )

        if any(w in t for w in ["data", "upload", "netcdf", "csv", "source", "provenance", "grid"]):
            return (
                f"Data Provenance & Ingestion Status:\n"
                f"• Active Data Mode: {data_mode}\n"
                f"• Target Site: {site_name}\n"
                f"• Slicing Engine: 2D horizontal slices and vertical profile series are ingested via xarray/pandas in real time, "
                f"standardized to OGC GeoJSON and WebGL color lookup tables."
            )

        # Default comprehensive grounding summary
        return (
            f"Observation Summary at {site_name} (Depth {d_str}, Time: {ctx.time_offset}):\n"
            f"• Live Active Variable: {ctx.variable} = {val_str}\n"
            f"• Coordinates: Lat {lat:.2f}°, Lon {lon:.2f}°\n"
            f"• Vertical Structure: The water column is governed by solar radiation attenuation, seasonal monsoon forcing, and pycnocline stratification. "
            f"Select any variable (Temperature, Salinity, Current, O₂) or an Argo float marker to inspect depth-series profiles."
        )
