"""
Operational Alert Engine V2 for Sagar-Drishti.

Implements the locked, frozen operational alert policy:
1. Operates strictly on post-hoc calibrated probability (not raw RF vote fractions).
2. Enforces temporal persistence rules to suppress transient single-day false alarms.
3. Implements risk-tier escalation:
   - Bay of Bengal:
     * IMMEDIATE_HIGH: p >= 0.60 triggers ALERT immediately on single day.
     * 2_CONSECUTIVE_MODERATE: p >= 0.20 for 2 consecutive days triggers ALERT.
     * Single day of p >= 0.20 triggers WATCH (advisory pending confirmation).
     * p < 0.20: NO_ALERT.
   - Arabian Sea:
     * IMMEDIATE_HIGH: p >= 0.50 triggers ALERT immediately on single day.
     * 2_OF_3_MODERATE: p >= 0.08 in >= 2 of last 3 days triggers ALERT.
     * Single day of p >= 0.08 triggers WATCH (advisory pending confirmation).
     * p < 0.08: NO_ALERT.
4. Cooldown de-escalation: requires 2 consecutive calm days to reset from ALERT to NO_ALERT.
5. Thread-safe in-memory rolling history for live operational and shadow scoring.
"""
import os
import json
import logging
import threading
from typing import Dict, Any, List, Optional
import numpy as np


logger = logging.getLogger("sagar_drishti.alert_engine_v2")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "frozen_alert_policy_v2.json")

# Default in-code fallback identical to frozen specification
FROZEN_POLICY_SPEC = {
    "policy_version": "v2.0.0-operational-alert-policy",
    "basin_policies": {
        "bob": {
            "basin_name": "Bay of Bengal",
            "thresholds": {
                "high_immediate": 0.60,
                "moderate_persistent": 0.20,
                "low_cutoff": 0.20,
                "cooldown_reset": 0.15
            },
            "persistence": {
                "rule": "2_consecutive",
                "window_size_days": 2,
                "min_positive_days": 2
            },
            "cooldown": {
                "reset_consecutive_days": 2
            }
        },
        "aras": {
            "basin_name": "Arabian Sea",
            "thresholds": {
                "high_immediate": 0.50,
                "moderate_persistent": 0.08,
                "low_cutoff": 0.08,
                "cooldown_reset": 0.05
            },
            "persistence": {
                "rule": "2_of_3",
                "window_size_days": 3,
                "min_positive_days": 2
            },
            "cooldown": {
                "reset_consecutive_days": 2
            }
        }
    }
}

def normalize_basin_key(site_or_basin: str) -> str:
    """Normalize site_id or basin string into 'bob' or 'aras'."""
    s = (site_or_basin or "").strip().lower()
    if "arab" in s or "aras" in s:
        return "aras"
    if "bengal" in s or "bob" in s:
        return "bob"
    return "bob"

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

class OperationalAlertEngineV2:
    """State-machine and decision engine for Operational Alert Engine V2."""
    _lock = threading.Lock()
    _site_histories: Dict[str, List[float]] = {}
    _site_states: Dict[str, Dict[str, Any]] = {}
    _policy: Dict[str, Any] = FROZEN_POLICY_SPEC

    @classmethod
    def load_policy(cls) -> Dict[str, Any]:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cls._policy = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load {CONFIG_PATH}, using built-in frozen spec: {e}")
        return cls._policy

    @classmethod
    def reset_state(cls):
        """Clears rolling site histories and states (used in tests/simulations)."""
        with cls._lock:
            cls._site_histories.clear()
            cls._site_states.clear()

    @classmethod
    def evaluate_decision(
        cls,
        calibrated_prob: float,
        raw_score: float,
        site_or_basin: str,
        horizon_days: int = 3,
        history_override: Optional[List[float]] = None,
        state_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pure and stateful decision evaluator.
        If history_override is provided, evaluates deterministically on that sequence.
        Otherwise, appends to the internal rolling history for site_or_basin.
        """
        p = float(np.clip(calibrated_prob, 0.0, 1.0))
        r = float(raw_score)
        h = int(horizon_days)
        b_key = normalize_basin_key(site_or_basin)
        
        cfg = cls._policy["basin_policies"].get(b_key, cls._policy["basin_policies"]["bob"])
        basin_name = cfg.get("basin_name", "Bay of Bengal")
        th = cfg["thresholds"]
        pers_cfg = cfg["persistence"]
        rule = pers_cfg["rule"]
        
        high_th = th["high_immediate"]
        mod_th = th["moderate_persistent"]
        cooldown_th = th["cooldown_reset"]
        
        risk_tier = map_risk_tier(p)
        
        # Resolve history
        site_key = f"{b_key}_{h}"
        with cls._lock:
            if history_override is not None:
                history = list(history_override)
                if not history or history[-1] != p:
                    history.append(p)
                prev_state = state_override or "NO_ALERT"
                below_count = 0
            else:
                if site_key not in cls._site_histories:
                    cls._site_histories[site_key] = []
                    cls._site_states[site_key] = {"state": "NO_ALERT", "below_count": 0}
                cls._site_histories[site_key].append(p)
                # Cap history at 14 observations
                if len(cls._site_histories[site_key]) > 14:
                    cls._site_histories[site_key].pop(0)
                history = cls._site_histories[site_key]
                prev_state = cls._site_states[site_key]["state"]
                below_count = cls._site_states[site_key]["below_count"]

        # Evaluate Persistence
        is_high = (p >= high_th)
        is_mod = (p >= mod_th)
        
        mod_persisted = False
        pers_state = "NO_PERSISTENCE"
        
        if rule == "2_consecutive":
            if len(history) >= 2 and history[-1] >= mod_th and history[-2] >= mod_th:
                mod_persisted = True
                pers_state = "2_CONSECUTIVE_CONFIRMED"
            elif is_mod:
                pers_state = "1_DAY_PENDING_CONFIRMATION"
        elif rule == "2_of_3":
            window = history[-3:]
            count = sum(1 for x in window if x >= mod_th)
            if count >= 2:
                mod_persisted = True
                pers_state = "2_OF_3_CONFIRMED"
            elif is_mod:
                pers_state = "1_OF_3_PENDING_CONFIRMATION"
        else:
            mod_persisted = is_mod
            pers_state = "SINGLE_DAY"

        alert_decision = "NO_ALERT"
        alert_reason = "BELOW_THRESHOLD"
        
        if is_high:
            alert_decision = "ALERT"
            alert_reason = "IMMEDIATE_HIGH"
            new_below_count = 0
        elif mod_persisted:
            alert_decision = "ALERT"
            alert_reason = f"PERSISTENT_MODERATE_{rule.upper()}"
            new_below_count = 0
        elif is_mod:
            alert_decision = "WATCH"
            alert_reason = "MODERATE_WATCH_PENDING_CONFIRMATION"
            new_below_count = 0
        else:
            if p < cooldown_th:
                new_below_count = below_count + 1
            else:
                new_below_count = below_count
                
            if prev_state == "ALERT" and new_below_count < 2:
                alert_decision = "WATCH"
                alert_reason = "COOLDOWN_DE_ESCALATION"
            else:
                alert_decision = "NO_ALERT"
                alert_reason = "BELOW_THRESHOLD"

        # Update in-memory state if stateful
        if history_override is None:
            with cls._lock:
                cls._site_states[site_key] = {
                    "state": alert_decision,
                    "below_count": new_below_count
                }

        return {
            "basin": basin_name,
            "basin_key": b_key,
            "horizon": h,
            "raw_score": round(r, 4),
            "calibrated_probability": round(p, 4),
            "risk_tier": risk_tier,
            "policy_threshold": round(mod_th, 4),
            "persistence_state": pers_state,
            "alert_decision": alert_decision,
            "alert_reason": alert_reason,
            "model_version": "v2.0.0-10yr-candidate",
            "operational_threshold": round(mod_th, 4),
            "operational_alert": alert_decision,
            "alert": alert_decision,
        }
