"""
Research Visualizations Generator for Sagar-Drishti V2.3 Shadow Evaluation.

CRITICAL GUARDRAIL:
All generated plots are strictly watermarked and labeled:
"V2.3 SHADOW — RESEARCH ONLY (ZERO OPERATIONAL AUTHORITY)"
Never alters production UI or operational alerts.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from backend.app.services.shadow_v2_3_monitor import ShadowV23Monitor

logger = logging.getLogger("sagar_drishti.shadow_v2_3_visualizer")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


class ShadowV23Visualizer:
    """
    Generates research-only publication-grade diagnostic plots for shadow evaluation.
    """

    def __init__(self, monitor: Optional[ShadowV23Monitor] = None):
        self.monitor = monitor or ShadowV23Monitor()

    def generate_all_figures(self) -> Dict[str, str]:
        """
        Generates all standard research visualization charts and returns file paths.
        """
        results = {}
        results["prod_vs_shadow_time_series"] = self.plot_probability_time_series()
        results["probability_deltas"] = self.plot_probability_deltas()
        results["coverage_breakdown"] = self.plot_coverage_breakdown()
        results["atmospheric_freshness"] = self.plot_atmospheric_freshness()
        results["horizon_comparison"] = self.plot_horizon_comparison()
        return results

    def plot_probability_time_series(self, filename: str = "v2_3_shadow_prod_vs_shadow_probs.png") -> str:
        """
        Time-series plot of Production v1.1.0 vs Model D probabilities.
        """
        series = self.monitor.get_temporal_series(horizon=3, limit=150)
        out_path = os.path.join(FIGURES_DIR, filename)

        # Filter out records with missing probabilities
        valid_series = [
            s for s in series
            if s.get("production_probability") is not None
            and s.get("shadow_probability") is not None
            and s.get("probability_delta") is not None
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

        if not valid_series:
            ax1.text(0.5, 0.5, "No valid paired observations recorded", ha="center", va="center")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_path

        indices = list(range(len(valid_series)))
        prod_probs = [float(s["production_probability"]) for s in valid_series]
        shadow_probs = [float(s["shadow_probability"]) for s in valid_series]
        deltas = [float(s["probability_delta"]) for s in valid_series]

        ax1.plot(indices, prod_probs, label="Production v1.1.0 (Operational)", color="#1f77b4", lw=2, marker="o", markersize=3, alpha=0.85)
        ax1.plot(indices, shadow_probs, label="Candidate Model D (Shadow / Research Only)", color="#ff7f0e", lw=2, linestyle="--", marker="s", markersize=3, alpha=0.85)
        ax1.axhline(0.27, color="#d62728", linestyle=":", label="Shadow Research Threshold (0.27)")
        ax1.set_ylabel("Predicted Probability", fontsize=11, fontweight="bold")
        ax1.set_title("V2.3 Model D vs Production v1.1.0 — Probability Tracking (Horizon 3d)\n[RESEARCH ONLY — ZERO OPERATIONAL AUTHORITY]", fontsize=12, fontweight="bold", pad=12)
        ax1.legend(loc="upper left", framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # Delta subplot
        bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax2.bar(indices, deltas, color=bar_colors, alpha=0.7, width=0.8)
        ax2.axhline(0.0, color="black", lw=1)
        ax2.axhline(0.15, color="orange", linestyle="--", alpha=0.5, label="Material Difference (|Δ| > 0.15)")
        ax2.axhline(-0.15, color="orange", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Observation Index (Sequential Telemetry)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Δ (Shadow - Prod)", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.8, 0.8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_probability_deltas(self, filename: str = "v2_3_shadow_probability_deltas.png") -> str:
        """
        Distribution and percentile boxplot of probability deltas.
        """
        obs = self.monitor.store.query_observations(horizon=3, inference_status="SUCCESS", limit=50000)
        deltas = [
            float(o["probability_delta"])
            for o in obs
            if o.get("probability_delta") is not None
        ]

        out_path = os.path.join(FIGURES_DIR, filename)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if not deltas:
            ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_path

        # Histogram
        ax1.hist(deltas, bins=25, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
        ax1.axvline(0, color="black", linestyle="-", lw=1.5)
        ax1.axvline(np.mean(deltas), color="#d62728", linestyle="--", lw=2, label=f"Mean Δ ({np.mean(deltas):+.3f})")
        ax1.axvline(np.median(deltas), color="#ff7f0e", linestyle=":", lw=2, label=f"Median Δ ({np.median(deltas):+.3f})")
        ax1.set_xlabel("Probability Delta (Shadow - Production)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax1.set_title("Probability Delta Distribution (Horizon 3d)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Boxplot
        ax2.boxplot(deltas, vert=True, patch_artist=True, boxprops=dict(facecolor="#8bb8e8", color="black"), medianprops=dict(color="red", lw=2))
        ax2.set_xticklabels(["Horizon 3d"], fontsize=11, fontweight="bold")
        ax2.set_ylabel("Probability Delta (Shadow - Production)", fontsize=11, fontweight="bold")
        ax2.set_title("Distribution Spread & Outliers", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("V2.3 Shadow Observation — Probability Divergence Diagnostics\n[RESEARCH ONLY — ZERO OPERATIONAL AUTHORITY]", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_coverage_breakdown(self, filename: str = "v2_3_shadow_coverage_breakdown.png") -> str:
        """
        Pie chart and category breakdown of shadow execution and data quality statuses.
        """
        cov = self.monitor.get_coverage_metrics()
        out_path = os.path.join(FIGURES_DIR, filename)

        labels = ["Successful", "Late Atmosphere", "Missing Atmosphere", "Schema Mismatch", "Inference Error"]
        counts = [
            cov["shadow_jobs_successful"],
            int(cov["total_prediction_opportunities"] * cov["data_quality"]["atmosphere_late_pct"] / 100.0),
            int(cov["total_prediction_opportunities"] * cov["data_quality"]["atmosphere_missing_pct"] / 100.0),
            int(cov["total_prediction_opportunities"] * cov["data_quality"]["feature_schema_mismatch_pct"] / 100.0),
            cov["shadow_jobs_failed"],
        ]
        # Filter non-zero
        filtered = [(l, c) for l, c in zip(labels, counts) if c > 0]
        if not filtered:
            filtered = [("No Data", 1)]

        f_labels = [f[0] for f in filtered]
        f_counts = [f[1] for f in filtered]
        colors = ["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728", "#9467bd"][: len(filtered)]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(f_counts, labels=f_labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=[0.05] * len(filtered))
        ax.set_title(
            f"V2.3 Shadow Evaluation Coverage Breakdown (Total Opportunities = {cov['total_prediction_opportunities']})\n[RESEARCH ONLY — ZERO OPERATIONAL AUTHORITY]",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_atmospheric_freshness(self, filename: str = "v2_3_shadow_atmospheric_freshness.png") -> str:
        """
        Histogram of atmospheric data latency.
        """
        obs = self.monitor.store.query_observations(limit=50000)
        ages = [
            float(o["atmospheric_data_age_hours"])
            for o in obs
            if o.get("atmospheric_data_age_hours") is not None
        ]
        out_path = os.path.join(FIGURES_DIR, filename)
        fig, ax = plt.subplots(figsize=(10, 5))

        if not ages:
            ax.text(0.5, 0.5, "No freshness data recorded", ha="center", va="center")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_path

        ax.hist(ages, bins=20, color="#17becf", edgecolor="black", alpha=0.75)
        ax.axvline(np.median(ages), color="red", linestyle="--", lw=2, label=f"Median Age: {np.median(ages):.1f} hrs")
        ax.set_xlabel("Atmospheric Data Age at Prediction Time (Hours)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count", fontsize=11, fontweight="bold")
        ax.set_title(
            "Atmospheric Data Freshness Latency Distribution\n[Pre-Operational Research Telemetry — ERA5 Cutoff T 18:00 UTC]",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_horizon_comparison(self, filename: str = "v2_3_shadow_horizon_comparison.png") -> str:
        """
        Comparative bar chart of Production v1.1.0 vs Model D mean probability across all lead horizons.
        """
        comp = self.monitor.get_comparison_analytics()
        out_path = os.path.join(FIGURES_DIR, filename)

        horizons = ["0d", "1d", "2d", "3d"]
        prod_means = []
        shadow_means = []

        for h in horizons:
            h_data = comp.get("horizons", {}).get(h, {})
            if h_data and "production" in h_data:
                prod_means.append(h_data["production"]["mean"])
                shadow_means.append(h_data["shadow_model_d"]["mean"])
            else:
                prod_means.append(0.0)
                shadow_means.append(0.0)

        x = np.arange(len(horizons))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, prod_means, width, label="Production v1.1.0 (Operational)", color="#1f77b4", alpha=0.85)
        ax.bar(x + width / 2, shadow_means, width, label="Candidate Model D (Shadow Research)", color="#ff7f0e", alpha=0.85)

        ax.set_xlabel("Forecast Lead Horizon", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
        ax.set_title("Mean Risk Probability Comparison across Forecast Horizons\n[RESEARCH ONLY — ZERO OPERATIONAL AUTHORITY]", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Horizon {h}" for h in horizons], fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
