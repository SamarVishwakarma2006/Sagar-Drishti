"""
Research-Only API Router for Sagar-Drishti V2.3 Shadow Evaluation.

CRITICAL SECURITY & OPERATIONAL ISOLATION INVARIANTS:
1. RESEARCH ONLY: Zero authority to trigger operational alerts, alter risk levels,
   modify thresholds, send notifications, or change production states.
2. Read-Only Operations: All endpoints perform observational analytics without state mutation.
3. Complete Production Isolation: Production endpoints and models remain 100% unaffected.
"""

from typing import Optional
from fastapi import APIRouter, Query
try:
    from ..services.shadow_v2_3_monitor import ShadowV23Monitor
except (ImportError, ValueError):
    from backend.app.services.shadow_v2_3_monitor import ShadowV23Monitor

router = APIRouter(tags=["V2.3 Shadow Research"])
_monitor = ShadowV23Monitor()


@router.get("/status")
def get_shadow_status():
    """
    Returns high-level status of the V2.3 Model D shadow evaluation system.
    RESEARCH ONLY - Zero operational authority.
    """
    return _monitor.get_status()


@router.get("/metrics")
def get_shadow_metrics():
    """
    Returns aggregated observational telemetry metrics.
    RESEARCH ONLY - Observational comparison only.
    """
    coverage = _monitor.get_coverage_metrics()
    comparison = _monitor.get_comparison_analytics()
    maturity = _monitor.get_evidence_maturity()
    freshness = _monitor.get_freshness_metrics()
    return {
        "mode": "RESEARCH_ONLY",
        "operational_authority": False,
        "coverage": coverage,
        "comparison_summary": comparison,
        "evidence_maturity": maturity,
        "atmospheric_freshness": freshness,
    }


@router.get("/coverage")
def get_shadow_coverage(
    site: Optional[str] = Query(None, description="Site filter ('bob' or 'aras')"),
    horizon: Optional[int] = Query(None, description="Horizon filter (0, 1, 2, 3)"),
    month: Optional[str] = Query(None, description="Month filter (YYYY-MM)"),
):
    """
    Returns shadow evaluation coverage and data quality breakdown.
    RESEARCH ONLY - Does not affect production coverage.
    """
    cov = _monitor.get_coverage_metrics(site=site, horizon=horizon, month=month)
    return {
        "mode": "RESEARCH_ONLY",
        "operational_authority": False,
        **cov,
    }


@router.get("/comparison")
def get_shadow_comparison(
    horizon: Optional[int] = Query(None, description="Forecast lead horizon in days (0, 1, 2, 3)"),
    site: Optional[str] = Query(None, description="Site filter ('bob' or 'aras')"),
):
    """
    Returns descriptive statistics comparing Production v1.1.0 vs Shadow Model D.
    RESEARCH ONLY - Probability differences are not proof of superiority.
    """
    comp = _monitor.get_comparison_analytics(horizon=horizon, site=site)
    return {
        "mode": "RESEARCH_ONLY",
        "operational_authority": False,
        **comp,
    }


@router.get("/events")
def get_shadow_events():
    """
    Returns registered storm events and accumulated shadow telemetry.
    RESEARCH ONLY - Historical test events remain quarantined from iterative tuning.
    """
    ev = _monitor.get_event_registry_summary()
    return {
        "mode": "RESEARCH_ONLY",
        "operational_authority": False,
        **ev,
    }


@router.get("/evidence")
def get_shadow_evidence():
    """
    Returns evidence maturity level and statistical sample size readiness.
    RESEARCH ONLY - Promotion Review Eligible does NOT mean promote Model D.
    """
    ev_mat = _monitor.get_evidence_maturity()
    return {
        "mode": "RESEARCH_ONLY",
        "operational_authority": False,
        **ev_mat,
    }


@router.get("/data-quality")
def get_shadow_data_quality():
    """
    Returns atmospheric data quality, freshness, and feature drift detection metrics.
    RESEARCH ONLY - Drift emits DATA_QUALITY_WARNING, NEVER an operational alert.
    """
    drift = _monitor.get_drift_metrics()
    freshness = _monitor.get_freshness_metrics()
    return {
        "mode": "RESEARCH_ONLY",
        "operational_authority": False,
        "feature_drift": drift,
        "freshness_latency": freshness,
    }
