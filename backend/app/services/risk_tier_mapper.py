"""
Centralized Risk Tier Mapper and Operational Threshold Service.
Enforces P0 Production-Safety Requirements for Sagar-Drishti:
1. Pure function mapping of calibrated probability to user-facing risk tiers:
   - LOW:      p < 0.30
   - MODERATE: 0.30 <= p < 0.60
   - HIGH:     p >= 0.60
2. Decoupled basin-specific operational thresholds:
   - Bay of Bengal (bob):   theta = 0.44
   - Arabian Sea (aras):    theta = 0.15
3. Never exposes raw uncalibrated Random Forest tree scores as literal probability percentages.
"""
from typing import Dict, Any, Optional

BASIN_THRESHOLDS: Dict[str, Dict[int, float]] = {
    "bob": {0: 0.44, 1: 0.44, 2: 0.44, 3: 0.44},
    "aras": {0: 0.15, 1: 0.15, 2: 0.15, 3: 0.15},
}

DEFAULT_FALLBACK_THRESHOLD = 0.27

def normalize_basin_key(site_or_basin: str) -> str:
    """Normalize site_id or basin string into 'bob' or 'aras'."""
    s = (site_or_basin or "").strip().lower()
    if "arab" in s or "aras" in s:
        return "aras"
    if "bengal" in s or "bob" in s:
        return "bob"
    return "bob"

def get_basin_threshold(site_or_basin: str, horizon_days: int) -> float:
    """Retrieve the validation-derived operational threshold for a given basin and horizon."""
    b_key = normalize_basin_key(site_or_basin)
    h_map = BASIN_THRESHOLDS.get(b_key, {})
    return h_map.get(int(horizon_days), DEFAULT_FALLBACK_THRESHOLD)

def map_risk_tier(calibrated_prob: float) -> str:
    """
    Map a calibrated probability to a user-facing risk tier.
    - LOW:      p < 0.30
    - MODERATE: 0.30 <= p < 0.60
    - HIGH:     p >= 0.60
    """
    p = float(calibrated_prob)
    if p < 0.30:
        return "LOW"
    elif p < 0.60:
        return "MODERATE"
    else:
        return "HIGH"

def evaluate_operational_decision(
    raw_score: float,
    calibrated_prob: float,
    site_or_basin: str,
    horizon_days: int
) -> Dict[str, Any]:
    """
    Evaluate full decision payload distinguishing raw score, calibrated probability,
    basin, horizon, operational threshold, risk tier, and alert decision.
    """
    b_key = normalize_basin_key(site_or_basin)
    basin_name = "Bay of Bengal" if b_key == "bob" else "Arabian Sea"
    threshold = get_basin_threshold(b_key, horizon_days)
    risk_tier = map_risk_tier(calibrated_prob)
    
    # Alert logic: triggered if raw score meets or exceeds the basin operational threshold,
    # or if calibrated probability is in the HIGH tier (>= 0.60).
    is_alert = (raw_score >= threshold) or (calibrated_prob >= 0.60)
    is_watch = (not is_alert) and (calibrated_prob >= 0.30 or raw_score >= (threshold * 0.75))
    
    if is_alert:
        alert_decision = "ALERT"
    elif is_watch:
        alert_decision = "WATCH"
    else:
        alert_decision = "NO_ALERT"
        
    return {
        "raw_score": round(float(raw_score), 4),
        "calibrated_probability": round(float(calibrated_prob), 4),
        "risk_tier": risk_tier,
        "operational_alert": alert_decision,
        "horizon_days": int(horizon_days),
        "basin": basin_name,
        "basin_key": b_key,
        "operational_threshold": round(float(threshold), 4),
    }
