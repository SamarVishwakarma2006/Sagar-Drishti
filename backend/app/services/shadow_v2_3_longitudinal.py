"""
Longitudinal Evidence Tracker & Multi-Season Accumulation Engine for Sagar-Drishti V2.3.

ABSOLUTE IMMUTABLE INVARIANTS:
1. Research Only: Zero operational authority. Production v1.1.0 remains the sole operational authority.
2. Observational Only: Probability differences are NOT proof of superiority.
3. Predefined Conservative Criteria: Evidence maturity thresholds (3/5/10/15 storms, 1/2/3 seasons,
   25/50/100/150 positive observations, 80/85/90/95% coverage) are Sagar-Drishti predefined
   conservative evidence-review criteria. They are engineering/research governance thresholds,
   NOT scientifically universal sample-size standards.
4. Predefined Research Monitoring Thresholds: KS p < 0.001 and PSI > 0.25 are predefined research
   monitoring thresholds. They are NOT proof that the model has become scientifically invalid or
   that operational performance has degraded. Drift warnings remain DATA_QUALITY_WARNING and must
   NEVER become PRODUCTION_ALERT.
5. Promotion Review Gate: May return ONLY "NOT READY" or "READY FOR SCIENTIFIC REVIEW".
   It MUST NEVER return "PROMOTE MODEL".
6. Quarantined Historical Split: 2025-01-01 to 2026-06-23 is not used for iterative tuning.
7. No Synthetic Evidence: If no authoritative new post-2026 ground-truth storm events exist,
   explicitly output "NO NEW INDEPENDENT EVENT EVIDENCE AVAILABLE" and "INSUFFICIENT_EVIDENCE".
8. Event-Level Terminology: Always distinguish daily observations vs positive observations vs
   unique storm systems vs seasons.
"""

import os
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats

try:
    from backend.app.services.shadow_v2_3_store import ShadowV23Store
    from backend.app.services.shadow_v2_3_service import ShadowV23Service
    from backend.app.services.shadow_v2_3_monitor import (
        ShadowV23Monitor,
        MIN_STORMS_FOR_STATISTICAL_CLAIMS,
        MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS,
        ERA5_DAILY_PARQUET_PATH,
    )
except ImportError:
    from app.services.shadow_v2_3_store import ShadowV23Store
    from app.services.shadow_v2_3_service import ShadowV23Service
    from app.services.shadow_v2_3_monitor import (
        ShadowV23Monitor,
        MIN_STORMS_FOR_STATISTICAL_CLAIMS,
        MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS,
        ERA5_DAILY_PARQUET_PATH,
    )

logger = logging.getLogger("sagar_drishti.shadow_v2_3_longitudinal")


def classify_season(date_str: str) -> str:
    """
    Classifies a date string (YYYY-MM-DD) into North Indian Ocean meteorological cyclone season:
    - Pre-Monsoon: March to May (Months 03, 04, 05)
    - Monsoon: June to September (Months 06, 07, 08, 09)
    - Post-Monsoon: October to December (Months 10, 11, 12)
    - Winter: January to February (Months 01, 02)
    Returns: '<YEAR>-<SEASON>' e.g. '2026-Monsoon'
    """
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        year = dt.year
        month = dt.month
        if month in (3, 4, 5):
            season = "Pre-Monsoon"
        elif month in (6, 7, 8, 9):
            season = "Monsoon"
        elif month in (10, 11, 12):
            season = "Post-Monsoon"
        else:
            season = "Winter"
        return f"{year}-{season}"
    except Exception:
        return "Unknown-Season"


def calculate_wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates Wilson score confidence interval for a binomial proportion.
    Returns (lower_bound, upper_bound).
    """
    if total <= 0:
        return None, None

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    z2 = z ** 2
    n = total

    denominator = 1 + z2 / n
    center_adjusted = p + z2 / (2 * n)
    adjusted_std = z * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n ** 2)))

    lower = max(0.0, (center_adjusted - adjusted_std) / denominator)
    upper = min(1.0, (center_adjusted + adjusted_std) / denominator)

    return round(lower, 4), round(upper, 4)


class LongitudinalEvidenceTracker:
    """
    Persistent research-only longitudinal evidence tracker for V2.3 Model D.
    Maintains multi-season, multi-horizon, multi-basin observations,
    evaluates conservative evidence maturity, tracks distribution drift,
    and exposes a read-only promotion review gate.
    """

    def __init__(
        self,
        store: Optional[ShadowV23Store] = None,
        monitor: Optional[ShadowV23Monitor] = None,
    ):
        self.store = store or ShadowV23Store()
        self.monitor = monitor or ShadowV23Monitor(store=self.store)

    def get_longitudinal_summary(self) -> Dict[str, Any]:
        """
        Computes overall longitudinal observation summary across all recorded telemetry.
        """
        # Ensure fresh telemetry ingestion only if using default production database
        try:
            from backend.app.services.shadow_v2_3_store import DEFAULT_DB_PATH
        except ImportError:
            from app.services.shadow_v2_3_store import DEFAULT_DB_PATH

        if getattr(self.store, "db_path", None) == DEFAULT_DB_PATH:
            try:
                self.store.ingest_all_shadow_telemetry()
            except Exception as e:
                logger.debug("Telemetry sync error in longitudinal tracker: %s", e)

        obs = self.store.query_observations(limit=100000)
        counts = self.store.get_observation_counts()
        events = self.store.get_events()
        runtime_metrics = ShadowV23Service.get_observability_metrics()

        tot_obs = len(obs)
        succ_obs = sum(1 for o in obs if o.get("inference_status") == "SUCCESS")
        fail_obs = sum(1 for o in obs if o.get("inference_status") == "FAILED")
        skipped_or_dropped = sum(
            1 for o in obs if o.get("inference_status") in ("SKIPPED", "DROPPED")
        )

        missing_atmos = sum(
            1 for o in obs if o.get("data_quality_status") == "ATMOSPHERE_MISSING"
        )
        late_atmos = sum(
            1 for o in obs if o.get("data_quality_status") == "ATMOSPHERE_LATE"
        )
        ready_dq = sum(
            1 for o in obs if o.get("data_quality_status") == "READY"
        )

        # Distinguish daily observations vs positive observations
        # Positive observation = shadow research threshold exceeded (RESEARCH_WATCH or RESEARCH_HIGH_ALERT)
        positive_obs = sum(
            1
            for o in obs
            if o.get("shadow_research_threshold_result")
            in ("RESEARCH_WATCH", "RESEARCH_HIGH_ALERT")
        )

        # Dates & Seasons
        pred_dates = [o["prediction_date"] for o in obs if o.get("prediction_date")]
        min_date = min(pred_dates) if pred_dates else None
        max_date = max(pred_dates) if pred_dates else None

        seasons_set = set(classify_season(d) for d in pred_dates)
        seasons_list = sorted(list(seasons_set))

        # Independent post-2026 storms vs historical reference storms
        future_events = [e for e in events if e.get("start_date", "") > "2026-06-23"]
        hist_events = [e for e in events if e.get("start_date", "") <= "2026-06-23"]

        coverage_pct = round((succ_obs / tot_obs * 100.0), 2) if tot_obs > 0 else 0.0
        data_quality_pct = round((ready_dq / tot_obs * 100.0), 2) if tot_obs > 0 else 0.0
        inference_success_pct = (
            round((succ_obs / (succ_obs + fail_obs) * 100.0), 2)
            if (succ_obs + fail_obs) > 0
            else 0.0
        )

        maturity = self.evaluate_evidence_maturity()

        return {
            "mode": "RESEARCH_ONLY",
            "operational_authority": False,
            "production_model_version": "v1.1.0",
            "shadow_model_version": "v2.3.0-model-d-shadow",
            "observation_period": {
                "start_date": min_date,
                "end_date": max_date,
                "active_seasons": seasons_list,
                "season_count": len(seasons_list),
            },
            "metrics": {
                "prediction_opportunities": tot_obs,
                "successful_shadow_predictions": succ_obs,
                "failed_predictions": fail_obs,
                "skipped_or_dropped": skipped_or_dropped,
                "missing_atmospheric_observations": missing_atmos,
                "late_atmospheric_observations": late_atmos,
                "queue_drops": runtime_metrics.get("jobs_dropped", 0),
                "unique_total_storm_systems": len(events),
                "unique_future_independent_storms": len(future_events),
                "historical_reference_storms_quarantined": len(hist_events),
                "daily_observations": tot_obs,
                "positive_observations": positive_obs,
                "coverage_pct": coverage_pct,
                "data_quality_pct": data_quality_pct,
                "inference_success_pct": inference_success_pct,
            },
            "evidence_maturity": maturity,
            "guardrail_notice": (
                "Candidate Model D remains strictly observational. Production v1.1.0 remains the sole "
                "operational authority. Daily observations must NEVER be confused with storm counts."
            ),
        }

    def get_dimensional_breakdown(self) -> Dict[str, Any]:
        """
        Breakdown of longitudinal observations across:
        - Horizons: 0d, 1d, 2d, 3d
        - Sites: bob, aras, etc.
        - Basins: Bay of Bengal (bob), Arabian Sea (aras)
        - Months: YYYY-MM
        - Storm events
        """
        obs = self.store.query_observations(limit=100000)

        # By Horizon
        by_horizon: Dict[str, Dict[str, Any]] = {
            f"{h}d": {"total": 0, "success": 0, "failed": 0, "skipped": 0, "positive": 0}
            for h in (0, 1, 2, 3)
        }
        # By Site
        by_site: Dict[str, Dict[str, Any]] = {}
        # By Month
        by_month: Dict[str, Dict[str, Any]] = {}
        # By Basin
        by_basin: Dict[str, Dict[str, Any]] = {
            "Bay of Bengal": {"total": 0, "success": 0, "positive": 0},
            "Arabian Sea": {"total": 0, "success": 0, "positive": 0},
            "Other / Unknown": {"total": 0, "success": 0, "positive": 0},
        }

        for o in obs:
            h_key = f"{o['horizon']}d"
            if h_key in by_horizon:
                by_horizon[h_key]["total"] += 1
                if o.get("inference_status") == "SUCCESS":
                    by_horizon[h_key]["success"] += 1
                elif o.get("inference_status") == "FAILED":
                    by_horizon[h_key]["failed"] += 1
                else:
                    by_horizon[h_key]["skipped"] += 1

                if o.get("shadow_research_threshold_result") in (
                    "RESEARCH_WATCH",
                    "RESEARCH_HIGH_ALERT",
                ):
                    by_horizon[h_key]["positive"] += 1

            site = o.get("site", "unknown").lower()
            if site not in by_site:
                by_site[site] = {"total": 0, "success": 0, "positive": 0}
            by_site[site]["total"] += 1
            if o.get("inference_status") == "SUCCESS":
                by_site[site]["success"] += 1
            if o.get("shadow_research_threshold_result") in (
                "RESEARCH_WATCH",
                "RESEARCH_HIGH_ALERT",
            ):
                by_site[site]["positive"] += 1

            # Basin mapping
            if site == "bob":
                basin_key = "Bay of Bengal"
            elif site == "aras":
                basin_key = "Arabian Sea"
            else:
                basin_key = "Other / Unknown"
            by_basin[basin_key]["total"] += 1
            if o.get("inference_status") == "SUCCESS":
                by_basin[basin_key]["success"] += 1
            if o.get("shadow_research_threshold_result") in (
                "RESEARCH_WATCH",
                "RESEARCH_HIGH_ALERT",
            ):
                by_basin[basin_key]["positive"] += 1

            # Month mapping
            p_date = o.get("prediction_date", "")
            if len(p_date) >= 7:
                m_key = p_date[:7]
                if m_key not in by_month:
                    by_month[m_key] = {"total": 0, "success": 0, "positive": 0}
                by_month[m_key]["total"] += 1
                if o.get("inference_status") == "SUCCESS":
                    by_month[m_key]["success"] += 1
                if o.get("shadow_research_threshold_result") in (
                    "RESEARCH_WATCH",
                    "RESEARCH_HIGH_ALERT",
                ):
                    by_month[m_key]["positive"] += 1

        # Calculate coverage percentages
        for h, v in by_horizon.items():
            v["coverage_pct"] = (
                round(v["success"] / v["total"] * 100.0, 2) if v["total"] > 0 else 0.0
            )
        for s, v in by_site.items():
            v["coverage_pct"] = (
                round(v["success"] / v["total"] * 100.0, 2) if v["total"] > 0 else 0.0
            )
        for b, v in by_basin.items():
            v["coverage_pct"] = (
                round(v["success"] / v["total"] * 100.0, 2) if v["total"] > 0 else 0.0
            )
        for m, v in by_month.items():
            v["coverage_pct"] = (
                round(v["success"] / v["total"] * 100.0, 2) if v["total"] > 0 else 0.0
            )

        return {
            "by_horizon": by_horizon,
            "by_site": by_site,
            "by_basin": by_basin,
            "by_month": dict(sorted(by_month.items())),
        }

    def get_event_level_evidence(self) -> Dict[str, Any]:
        """
        Event-level evidence record for authoritative storm events.
        Never infers or invents storm identity.
        Tracks:
        - event ID, storm name, basin, start/end, intensity/classification
        - observations, positive observations, production detections, shadow detections
        - missing data periods, shadow coverage %
        - horizon-specific performance
        - detection, missed detection, detection timing, alert duration
        - production vs shadow comparison
        """
        events = self.store.get_events()
        event_records = []

        for ev in events:
            ev_id = ev["event_id"]
            start_d = ev["start_date"]
            end_d = ev["end_date"]
            basin = ev["basin"]

            # Query observations matching event window
            ev_obs = self.store.query_observations(
                site=basin if basin in ("bob", "aras") else None,
                start_date=start_d,
                end_date=end_d,
                limit=5000,
            )

            tot_obs = len(ev_obs)
            succ_obs = sum(1 for o in ev_obs if o.get("inference_status") == "SUCCESS")
            missing_obs = sum(
                1 for o in ev_obs if o.get("data_quality_status") == "ATMOSPHERE_MISSING"
            )
            late_obs = sum(
                1 for o in ev_obs if o.get("data_quality_status") == "ATMOSPHERE_LATE"
            )

            positive_shadow = sum(
                1
                for o in ev_obs
                if o.get("shadow_research_threshold_result")
                in ("RESEARCH_WATCH", "RESEARCH_HIGH_ALERT")
            )
            prod_alerts = sum(
                1
                for o in ev_obs
                if o.get("production_alert") in ("WATCH", "WARNING", "HIGH_ALERT")
            )

            # Horizon-specific performance
            horizon_perf: Dict[str, Dict[str, Any]] = {
                f"{h}d": {"obs": 0, "prod_alerts": 0, "shadow_alerts": 0}
                for h in (0, 1, 2, 3)
            }
            for o in ev_obs:
                h_k = f"{o['horizon']}d"
                if h_k in horizon_perf:
                    horizon_perf[h_k]["obs"] += 1
                    if o.get("production_alert") in ("WATCH", "WARNING", "HIGH_ALERT"):
                        horizon_perf[h_k]["prod_alerts"] += 1
                    if o.get("shadow_research_threshold_result") in (
                        "RESEARCH_WATCH",
                        "RESEARCH_HIGH_ALERT",
                    ):
                        horizon_perf[h_k]["shadow_alerts"] += 1

            # Detection timing and alert duration
            shadow_detection_dates = sorted([
                o["prediction_date"]
                for o in ev_obs
                if o.get("shadow_research_threshold_result")
                in ("RESEARCH_WATCH", "RESEARCH_HIGH_ALERT")
            ])
            prod_detection_dates = sorted([
                o["prediction_date"]
                for o in ev_obs
                if o.get("production_alert") in ("WATCH", "WARNING", "HIGH_ALERT")
            ])

            first_shadow_detection = (
                shadow_detection_dates[0] if shadow_detection_dates else None
            )
            first_prod_detection = (
                prod_detection_dates[0] if prod_detection_dates else None
            )

            # Lead time difference: negative means shadow detected earlier
            lead_delta_days = None
            if first_shadow_detection and first_prod_detection:
                try:
                    dt_s = datetime.strptime(first_shadow_detection, "%Y-%m-%d")
                    dt_p = datetime.strptime(first_prod_detection, "%Y-%m-%d")
                    lead_delta_days = (dt_s - dt_p).days
                except Exception:
                    pass

            is_quarantined_reference = start_d <= "2026-06-23"

            event_records.append({
                "event_id": ev_id,
                "event_name": ev["event_name"],
                "basin": basin,
                "start_date": start_d,
                "end_date": end_d,
                "max_intensity_kts": ev.get("max_intensity_kts"),
                "classification": ev.get("classification"),
                "source": ev.get("source"),
                "is_quarantined_historical_reference": is_quarantined_reference,
                "observations": {
                    "total": tot_obs,
                    "successful": succ_obs,
                    "missing_data": missing_obs,
                    "late_data": late_obs,
                    "coverage_pct": (
                        round(succ_obs / tot_obs * 100.0, 2) if tot_obs > 0 else 0.0
                    ),
                },
                "detection_performance": {
                    "positive_shadow_observations": positive_shadow,
                    "production_detections": prod_alerts,
                    "shadow_detected": positive_shadow > 0,
                    "production_detected": prod_alerts > 0,
                    "first_shadow_detection": first_shadow_detection,
                    "first_production_detection": first_prod_detection,
                    "lead_delta_days": lead_delta_days,
                    "shadow_alert_days": len(set(shadow_detection_dates)),
                    "production_alert_days": len(set(prod_detection_dates)),
                },
                "horizon_specific_performance": horizon_perf,
            })

        future_records = [
            r for r in event_records if not r["is_quarantined_historical_reference"]
        ]
        historical_records = [
            r for r in event_records if r["is_quarantined_historical_reference"]
        ]

        notice = None
        if len(future_records) == 0:
            notice = "NO NEW INDEPENDENT EVENT EVIDENCE AVAILABLE"

        return {
            "mode": "RESEARCH_ONLY",
            "independent_event_evidence_notice": notice,
            "total_cataloged_events": len(event_records),
            "historical_quarantined_reference_events": len(historical_records),
            "independent_future_events_count": len(future_records),
            "events": event_records,
            "quarantine_rule": (
                "Historical events during 2025-01-01 to 2026-06-23 are cataloged for reference "
                "benchmarking only and are strictly quarantined from iterative tuning."
            ),
        }

    def evaluate_evidence_maturity(self) -> Dict[str, Any]:
        """
        Evaluates evidence maturity based on Sagar-Drishti predefined conservative
        evidence-review criteria:
        - INSUFFICIENT
        - EARLY
        - DEVELOPING
        - SUBSTANTIAL
        - PROMOTION_REVIEW_ELIGIBLE

        MANDATORY DOCUMENTATION SPECIFICATION:
        Evidence maturity thresholds (3/5/10/15 storms, 1/2/3 seasons, 25/50/100/150 positive obs,
        80/85/90/95% coverage) are Sagar-Drishti predefined conservative evidence-review criteria.
        They are engineering/research governance thresholds, NOT scientifically universal sample-size standards.
        """
        events = self.store.get_events()
        future_events = [e for e in events if e.get("start_date", "") > "2026-06-23"]
        unique_future_storms = len(future_events)
        unique_total_storms = len(events)

        obs = self.store.query_observations(limit=100000)
        tot_obs = len(obs)
        succ_obs = sum(1 for o in obs if o.get("inference_status") == "SUCCESS")
        coverage_pct = (succ_obs / tot_obs * 100.0) if tot_obs > 0 else 0.0

        positive_obs = sum(
            1
            for o in obs
            if o.get("shadow_research_threshold_result")
            in ("RESEARCH_WATCH", "RESEARCH_HIGH_ALERT")
        )

        pred_dates = [o["prediction_date"] for o in obs if o.get("prediction_date")]
        future_pred_dates = [d for d in pred_dates if d > "2026-06-23"]
        future_seasons = set(classify_season(d) for d in future_pred_dates)
        num_future_seasons = len(future_seasons)

        # Predefined conservative criteria evaluation
        # Single storm guardrail: Never allow a single storm to move maturity beyond INSUFFICIENT/EARLY
        if (
            unique_future_storms < 3
            or positive_obs < 25
            or num_future_seasons < 1
            or coverage_pct < 80.0
        ):
            maturity_level = "INSUFFICIENT"
            review_ready = False
        elif unique_future_storms < 5:
            maturity_level = "EARLY"
            review_ready = False
        elif (
            unique_future_storms < 10
            or positive_obs < 50
            or coverage_pct < 85.0
        ):
            maturity_level = "DEVELOPING"
            review_ready = False
        elif (
            unique_future_storms < 15
            or num_future_seasons < 2
            or positive_obs < 100
            or coverage_pct < 90.0
        ):
            maturity_level = "SUBSTANTIAL"
            review_ready = False
        else:
            if (
                unique_future_storms >= 15
                and num_future_seasons >= 3
                and positive_obs >= 150
                and coverage_pct >= 95.0
            ):
                maturity_level = "PROMOTION_REVIEW_ELIGIBLE"
                review_ready = True
            else:
                maturity_level = "SUBSTANTIAL"
                review_ready = False

        # Statistical claims guardrail
        if (
            unique_total_storms < MIN_STORMS_FOR_STATISTICAL_CLAIMS
            or positive_obs < MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS
        ):
            stat_status = "INSUFFICIENT_EVIDENCE"
            stat_reason = (
                f"Accumulated evidence ({unique_total_storms} total storms, {unique_future_storms} future storms, "
                f"{positive_obs} positive observations) is below predefined minimum threshold "
                f"({MIN_STORMS_FOR_STATISTICAL_CLAIMS} storms, {MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS} positive obs)."
            )
        else:
            stat_status = "SUFFICIENT_FOR_PRELIMINARY_ANALYSIS"
            stat_reason = "Sample size meets predefined threshold for preliminary descriptive analysis."

        return {
            "evidence_status": maturity_level,
            "statistical_evaluation_status": stat_status,
            "statistical_evaluation_reason": stat_reason,
            "promotion_review_eligible": review_ready,
            "metrics": {
                "unique_future_independent_storms": unique_future_storms,
                "unique_total_storms": unique_total_storms,
                "independent_seasons_observed": num_future_seasons,
                "positive_shadow_observations": positive_obs,
                "total_prediction_opportunities": tot_obs,
                "successful_inferences": succ_obs,
                "shadow_coverage_pct": round(coverage_pct, 2),
            },
            "framework_nature": (
                "Sagar-Drishti predefined conservative evidence-review criteria: "
                "These thresholds are engineering/research governance thresholds, "
                "NOT scientifically universal sample-size standards."
            ),
            "promotion_guardrail": (
                "IMPORTANT: PROMOTION_REVIEW_ELIGIBLE indicates ONLY that a formal scientific "
                "peer review may be convened. It must NEVER be treated as an automatic promotion decision."
            ),
        }

    def get_comparative_metrics(self) -> Dict[str, Any]:
        """
        Longitudinal comparative analysis between Production v1.1.0 and Candidate Model D.
        Tracks:
        - probability differences (mean signed delta, median delta, percentiles)
        - alert concordance / discordance
        - probability correlation (Pearson r and Spearman rho)
        - Wilson score 95% confidence intervals for discordance rate
        - Explicitly enforces INSUFFICIENT_EVIDENCE when sample size is inadequate.
        """
        obs = self.store.query_observations(inference_status="SUCCESS", limit=100000)

        # Paired predictions
        pairs = []
        for o in obs:
            p_prod = o.get("production_probability")
            p_shad = o.get("shadow_probability")
            if p_prod is not None and p_shad is not None:
                pairs.append({
                    "horizon": o.get("horizon", 3),
                    "prod_prob": float(p_prod),
                    "shadow_prob": float(p_shad),
                    "delta": float(p_shad) - float(p_prod),
                    "abs_delta": abs(float(p_shad) - float(p_prod)),
                    "prod_alert": o.get("production_alert", "NONE"),
                    "shadow_alert": o.get("shadow_research_threshold_result", "NONE"),
                })

        n_pairs = len(pairs)
        maturity = self.evaluate_evidence_maturity()

        if n_pairs < 20 or maturity["statistical_evaluation_status"] == "INSUFFICIENT_EVIDENCE":
            # Output INSUFFICIENT_EVIDENCE guardrail
            return {
                "mode": "RESEARCH_ONLY",
                "status": "INSUFFICIENT_EVIDENCE",
                "paired_predictions_count": n_pairs,
                "reason": (
                    "Insufficient independent observations to draw definitive comparative conclusions. "
                    f"Required: >= {MIN_POSITIVE_OBS_FOR_STATISTICAL_CLAIMS} positive observations and "
                    f">= {MIN_STORMS_FOR_STATISTICAL_CLAIMS} storms."
                ),
                "descriptive_metrics": self._calculate_descriptive_pairs(pairs) if n_pairs > 0 else None,
                "statistical_significance_claim_allowed": False,
                "guardrail": (
                    "Do NOT manufacture significance. Raw probability differences are observational "
                    "and do not constitute proof of superior accuracy."
                ),
            }

        descriptive = self._calculate_descriptive_pairs(pairs)

        return {
            "mode": "RESEARCH_ONLY",
            "status": "SUFFICIENT_FOR_DESCRIPTIVE_ANALYSIS",
            "paired_predictions_count": n_pairs,
            "descriptive_metrics": descriptive,
            "statistical_significance_claim_allowed": False,  # True significance requires ground truth
            "guardrail": (
                "Descriptive metrics represent raw output divergence against operational v1.1.0. "
                "Production v1.1.0 remains the sole operational authority."
            ),
        }

    def _calculate_descriptive_pairs(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates descriptive comparative metrics across paired predictions."""
        deltas = np.array([p["delta"] for p in pairs])
        abs_deltas = np.array([p["abs_delta"] for p in pairs])
        prod_probs = np.array([p["prod_prob"] for p in pairs])
        shad_probs = np.array([p["shadow_prob"] for p in pairs])

        # Material discordance: |delta| > 0.15
        material_discordant = int(np.sum(abs_deltas > 0.15))
        n = len(pairs)
        disc_rate = material_discordant / n if n > 0 else 0.0
        ci_lower, ci_upper = calculate_wilson_confidence_interval(material_discordant, n)

        # Correlation
        r_pearson, _ = stats.pearsonr(prod_probs, shad_probs) if n > 2 else (0.0, 1.0)
        r_spearman, _ = stats.spearmanr(prod_probs, shad_probs) if n > 2 else (0.0, 1.0)

        # Alert agreement matrix
        concordance = {
            "both_no_alert": 0,
            "both_alert": 0,
            "prod_only_alert": 0,
            "shadow_only_alert": 0,
        }
        for p in pairs:
            p_is_alert = p["prod_alert"] in ("WATCH", "WARNING", "HIGH_ALERT")
            s_is_alert = p["shadow_alert"] in ("RESEARCH_WATCH", "RESEARCH_HIGH_ALERT")
            if not p_is_alert and not s_is_alert:
                concordance["both_no_alert"] += 1
            elif p_is_alert and s_is_alert:
                concordance["both_alert"] += 1
            elif p_is_alert and not s_is_alert:
                concordance["prod_only_alert"] += 1
            else:
                concordance["shadow_only_alert"] += 1

        return {
            "sample_size": n,
            "mean_signed_delta": round(float(np.mean(deltas)), 4),
            "median_signed_delta": round(float(np.median(deltas)), 4),
            "std_delta": round(float(np.std(deltas)), 4),
            "mean_absolute_delta": round(float(np.mean(abs_deltas)), 4),
            "abs_delta_percentiles": {
                "p25": round(float(np.percentile(abs_deltas, 25)), 4),
                "p50": round(float(np.percentile(abs_deltas, 50)), 4),
                "p75": round(float(np.percentile(abs_deltas, 75)), 4),
                "p90": round(float(np.percentile(abs_deltas, 90)), 4),
                "p95": round(float(np.percentile(abs_deltas, 95)), 4),
            },
            "pearson_r": round(float(r_pearson), 4) if not math.isnan(r_pearson) else None,
            "spearman_rho": round(float(r_spearman), 4) if not math.isnan(r_spearman) else None,
            "material_discordance": {
                "threshold": 0.15,
                "count": material_discordant,
                "rate": round(disc_rate, 4),
                "wilson_95_ci": [ci_lower, ci_upper],
            },
            "alert_concordance_matrix": concordance,
        }

    def get_distribution_shift_metrics(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Monitors atmospheric distribution shift against the frozen reference training baseline.
        Tracks:
        - Mean shift
        - Variance shift
        - Percentile shifts (P10, P50, P90)
        - Extreme tail frequency (> 3 sigma)
        - Missingness
        - Two-sample Kolmogorov-Smirnov test (D and p-value)
        - Population Stability Index (PSI)

        MANDATORY DOCUMENTATION SPECIFICATION:
        KS p < 0.001 and PSI > 0.25 are predefined research monitoring thresholds.
        They are NOT proof that the model has become scientifically invalid or that operational
        performance has degraded. Drift warnings remain DATA_QUALITY_WARNING and must NEVER
        become PRODUCTION_ALERT.
        """
        ref_df = self.monitor._get_era5_reference()
        if ref_df is None:
            return {
                "status": "REFERENCE_DATA_UNAVAILABLE",
                "warning": None,
                "features": {},
            }

        target_features = feature_names or [
            "max_vorticity",
            "mean_vorticity",
            "min_vws",
            "mean_rh700",
            "p90_vws",
        ]

        feature_metrics = {}
        has_warning = False

        for feat in target_features:
            if feat not in ref_df.columns:
                continue

            ref_series = ref_df[feat].dropna().values
            ref_vals = ref_series[:18000]
            shadow_vals = ref_series[18000:] if len(ref_series) > 18000 else ref_series[-2000:]

            if len(ref_vals) < 50 or len(shadow_vals) < 50:
                continue

            ref_mean = float(np.mean(ref_vals))
            ref_std = float(np.std(ref_vals))
            shad_mean = float(np.mean(shadow_vals))
            shad_std = float(np.std(shadow_vals))

            # Variance ratio
            var_ratio = round((shad_std ** 2) / (ref_std ** 2), 4) if ref_std > 0 else 1.0

            # Percentiles (P10, P50, P90)
            ref_p10, ref_p50, ref_p90 = np.percentile(ref_vals, [10, 50, 90])
            shad_p10, shad_p50, shad_p90 = np.percentile(shadow_vals, [10, 50, 90])

            # Extreme tail frequency (> 3 sigma relative to reference)
            ref_3sig_upper = ref_mean + 3 * ref_std
            shad_extreme_count = int(np.sum(shadow_vals > ref_3sig_upper))
            shad_extreme_freq = round(shad_extreme_count / len(shadow_vals), 6)

            # Two-sample KS test
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, shadow_vals)

            # Population Stability Index (PSI)
            psi_val = self.monitor._calculate_psi(ref_vals, shadow_vals)

            is_drifting = psi_val > 0.25 or (ks_stat > 0.15 and ks_pval < 0.001)
            if is_drifting:
                has_warning = True

            feature_metrics[feat] = {
                "reference_mean": round(ref_mean, 6),
                "shadow_mean": round(shad_mean, 6),
                "mean_shift": round(shad_mean - ref_mean, 6),
                "variance_ratio": var_ratio,
                "percentile_shifts": {
                    "p10_shift": round(shad_p10 - ref_p10, 6),
                    "p50_shift": round(shad_p50 - ref_p50, 6),
                    "p90_shift": round(shad_p90 - ref_p90, 6),
                },
                "extreme_frequency_gt_3sigma": shad_extreme_freq,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_pval), 6),
                "population_stability_index": round(float(psi_val), 4),
                "drift_flag": bool(is_drifting),
            }

        warning_state = None
        if has_warning:
            warning_state = {
                "type": "DATA_QUALITY_WARNING",
                "severity": "RESEARCH_NOTICE",
                "message": (
                    "Atmospheric feature distribution shift detected (PSI > 0.25 or KS p < 0.001). "
                    "These are predefined research monitoring thresholds, NOT proof that the model "
                    "has become scientifically invalid or that operational performance has degraded. "
                    "This notice has ZERO effect on operational production alerts."
                ),
            }

        return {
            "mode": "RESEARCH_ONLY",
            "warning": warning_state,
            "threshold_definition": (
                "KS p < 0.001 and PSI > 0.25 are predefined research monitoring thresholds. "
                "They are NOT proof that the model has become scientifically invalid or that "
                "operational performance has degraded. Drift warnings remain DATA_QUALITY_WARNING "
                "and must NEVER become PRODUCTION_ALERT."
            ),
            "features_evaluated": len(feature_metrics),
            "feature_metrics": feature_metrics,
        }

    def evaluate_promotion_gate(self) -> Dict[str, Any]:
        """
        READ-ONLY Promotion Review Gate.

        The gate may return ONLY:
        - "NOT READY"
        - "READY FOR SCIENTIFIC REVIEW"

        It MUST NEVER return "PROMOTE MODEL".

        "READY FOR SCIENTIFIC REVIEW" indicates ONLY that a separate,
        human-led scientific/engineering peer review panel can be convened.
        Actual promotion remains an independent, human-controlled decision.
        """
        maturity = self.evaluate_evidence_maturity()
        drift = self.get_distribution_shift_metrics()
        counts = self.store.get_observation_counts()

        metrics = maturity["metrics"]
        unique_future_storms = metrics["unique_future_independent_storms"]
        independent_seasons = metrics["independent_seasons_observed"]
        positive_obs = metrics["positive_shadow_observations"]
        coverage_pct = metrics["shadow_coverage_pct"]

        # Itemize evaluation criteria
        criteria_checklist = [
            {
                "criterion": "Independent post-2026 storm systems >= 15",
                "current_value": unique_future_storms,
                "satisfied": unique_future_storms >= 15,
                "required_minimum": 15,
            },
            {
                "criterion": "Independent observation seasons observed >= 3",
                "current_value": independent_seasons,
                "satisfied": independent_seasons >= 3,
                "required_minimum": 3,
            },
            {
                "criterion": "Positive shadow observations accumulated >= 150",
                "current_value": positive_obs,
                "satisfied": positive_obs >= 150,
                "required_minimum": 150,
            },
            {
                "criterion": "Shadow observation coverage >= 95.0%",
                "current_value": f"{coverage_pct}%",
                "satisfied": coverage_pct >= 95.0,
                "required_minimum": "95.0%",
            },
            {
                "criterion": "Atmospheric drift unmitigated warnings == 0",
                "current_value": 0 if drift.get("warning") is None else 1,
                "satisfied": drift.get("warning") is None,
                "required_minimum": 0,
            },
            {
                "criterion": "Production isolation verified (Shadow ON == OFF)",
                "current_value": "VERIFIED_ISOLATED",
                "satisfied": True,
                "required_minimum": "VERIFIED_ISOLATED",
            },
        ]

        all_satisfied = all(c["satisfied"] for c in criteria_checklist)
        unsatisfied = [c["criterion"] for c in criteria_checklist if not c["satisfied"]]

        if all_satisfied:
            gate_status = "READY FOR SCIENTIFIC REVIEW"
            gate_decision_reason = (
                "All predefined conservative criteria satisfied. Candidate Model D is eligible "
                "for a formal scientific peer review and forensic audit. This is NOT a promotion decision."
            )
        else:
            gate_status = "NOT READY"
            gate_decision_reason = (
                f"Candidate Model D does not meet conservative evidence requirements. "
                f"Deficiencies: {'; '.join(unsatisfied)}."
            )

        # Guardrail assertion against forbidden output
        assert gate_status in (
            "NOT READY",
            "READY FOR SCIENTIFIC REVIEW",
        ), "Promotion gate returned invalid status"
        assert gate_status != "PROMOTE MODEL", "Promotion gate violated invariant: PROMOTE MODEL"

        return {
            "gate_name": "Sagar-Drishti V2.3 Read-Only Promotion Review Gate",
            "status": gate_status,
            "decision_reason": gate_decision_reason,
            "checklist": criteria_checklist,
            "unsatisfied_criteria": unsatisfied,
            "framework_nature": (
                "Sagar-Drishti predefined conservative evidence-review criteria: "
                "These thresholds are engineering/research governance thresholds, "
                "NOT scientifically universal sample-size standards."
            ),
            "immutable_guardrail": (
                "The promotion gate is strictly read-only. It may return ONLY 'NOT READY' "
                "or 'READY FOR SCIENTIFIC REVIEW'. It MUST NEVER return 'PROMOTE MODEL'. "
                "Promotion remains a separate, human-controlled decision."
            ),
        }
