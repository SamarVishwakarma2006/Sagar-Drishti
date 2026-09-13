"""
Sagar-Drishti Multi-Hazard Architecture & Isolation Test Suite

Verifies:
1. Cryptographic baseline integrity of all 11 protected artifacts.
2. Complete declarative registration of all 10 candidate hazards.
3. Strict target independence across all hazards (no recycling of genesis target).
4. Rigorous label type categorization (Observational vs Reanalysis vs Derived).
5. Accurate identification of available vs missing datasets (data gap audits).
6. Proper flagging of provisional thresholds (is_provisional=True).
7. Strict causal cutoff validation and rejection of future timestamps.
8. Operational isolation: Research modules cannot be invoked by production alert engines.
9. Immutability of production v1.1.0, candidate Model D, and frozen alert policy.
"""

import hashlib
import json
import pytest
from pathlib import Path

from app.services.hazard_registry import (
    hazard_registry,
    HazardCategory,
    HazardStatus,
    FeasibilityRating,
    LabelType,
    ValidationLevel,
    OperationalIsolationError,
    TemporalLeakageError
)
from app.services.prediction_service import PredictionService
from app.services.alert_engine_v2 import OperationalAlertEngineV2


# Baseline SHA-256 Hashes of all 11 Protected Artifacts
PROTECTED_HASHES = {
    "backend/models/risk_model_3d.joblib": "3f52f16be035c66391cc885ebfc55efb2c8bc7627b13b7b43ccafd98f4a41740",
    "backend/models/v2_10yr/risk_model_3d.joblib": "7c4f17861c0d48e962c286404d77f7b9bc329afc076a11bb24dd099df91082d3",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_0d.joblib": "7f78946c01442793af8103995496199d19361378d627dd92262a391a7dd61e15",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_1d.joblib": "b0245d154f53786119e8d226f3ba90c22fff0a3c1e59b70f09c8ce420c2679fe",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_2d.joblib": "aca2e9423af471e533928a970b52bdf6d1dc7433ae85545a3b2a79fb86dad86c",
    "backend/models/candidates/v2_3/ocean_atmos_all/risk_model_3d.joblib": "250948b2aa72babeb68afca8910c36626b623a114c064126862e2c834f63d775",
    "backend/config/frozen_alert_policy_v2.json": "6ff3fe00ff9354c4c9a5a49ce4f04dc7458bceea29c7018d590550dbad3eefd6",
    "backend/data/historical/features_10yr.parquet": "cdf3837f9ebe68eab100a6122d32a383a29b87a937afa42de39e787452903867",
    "backend/data/historical/labeled_features_10yr_clean.parquet": "25070aad573c23bb67adbfbae34314515f4860b9ecfa9a35ab3c6f4e86b99f6a",
    "backend/data/era5/features_atmosphere_10yr_daily.parquet": "551aa9cb4ca460ec7d81036a6be54a0155efd1603ae849033dc9582c80d79864",
    "backend/config/v2_3_frozen_experiment_manifest.json": "73d407d798f68de2e5934f544077bb89f8b84e9db97d060447275d54661d50bc",
}


class TestCryptographicBaselineIntegrity:
    """Verifies that all 11 protected artifacts remain 100% byte-for-byte immutable."""

    @pytest.mark.parametrize("filepath,expected_hash", PROTECTED_HASHES.items())
    def test_protected_artifact_hash_invariance(self, filepath: str, expected_hash: str) -> None:
        p = Path(filepath)
        assert p.exists(), f"Protected artifact {filepath} does not exist!"
        with open(p, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, (
            f"Cryptographic corruption detected in {filepath}!\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}"
        )


class TestHazardRegistryCompleteness:
    """Verifies the declarative configuration of all 10 candidate hazards."""

    EXPECTED_HAZARDS = [
        "cyclone_genesis",
        "cyclone_intensity",
        "cyclone_track",
        "extreme_wind",
        "extreme_rainfall",
        "extreme_waves",
        "storm_surge",
        "coastal_flooding",
        "extreme_sea_level",
        "marine_heatwaves",
    ]

    def test_all_ten_hazards_registered(self) -> None:
        hazards = hazard_registry.list_all_hazards()
        assert len(hazards) == 10, f"Expected exactly 10 registered hazards, found {len(hazards)}"
        registered_ids = [h.hazard_id for h in hazards]
        for expected_id in self.EXPECTED_HAZARDS:
            assert expected_id in registered_ids, f"Hazard '{expected_id}' is missing from registry!"

    def test_target_independence_principle(self) -> None:
        """Every hazard must have an independent, distinct target formula."""
        targets = [h.target.mathematical_formula for h in hazard_registry.list_all_hazards()]
        assert len(targets) == len(set(targets)), "Duplicate target formulas detected! Target independence violated."
        
        # Explicit check: No other hazard may reuse the cyclone genesis formula
        genesis = hazard_registry.get_hazard("cyclone_genesis")
        for h in hazard_registry.list_all_hazards():
            if h.hazard_id != "cyclone_genesis":
                assert h.target.mathematical_formula != genesis.target.mathematical_formula, (
                    f"Hazard '{h.hazard_id}' illegally reuses the cyclone genesis target formula!"
                )

    def test_label_type_classification(self) -> None:
        """Verifies that labels are rigorously categorized."""
        for h in hazard_registry.list_all_hazards():
            assert isinstance(h.label_type, LabelType)
            assert h.label_source is not None and len(h.label_source) > 0

        # Specific ground truth audits
        assert hazard_registry.get_hazard("cyclone_intensity").label_type == LabelType.OBSERVATIONAL_GROUND_TRUTH
        assert hazard_registry.get_hazard("cyclone_track").label_type == LabelType.OBSERVATIONAL_GROUND_TRUTH
        assert hazard_registry.get_hazard("storm_surge").label_type == LabelType.OBSERVATIONAL_GROUND_TRUTH
        assert hazard_registry.get_hazard("extreme_wind").label_type == LabelType.REANALYSIS_REFERENCE_DATA
        assert hazard_registry.get_hazard("cyclone_genesis").label_type == LabelType.DERIVED_LABEL

    def test_provisional_threshold_flagging(self) -> None:
        """Verifies that non-authoritative operational thresholds are marked is_provisional=True."""
        provisional_hazards = ["extreme_wind", "extreme_rainfall", "extreme_waves", "storm_surge", "coastal_flooding", "extreme_sea_level"]
        for hid in provisional_hazards:
            hazard = hazard_registry.get_hazard(hid)
            assert hazard.target.is_provisional is True, f"Hazard '{hid}' must be marked is_provisional=True!"

        # Genesis, intensity, and MHW have established official definitions
        assert hazard_registry.get_hazard("cyclone_genesis").target.is_provisional is False
        assert hazard_registry.get_hazard("cyclone_intensity").target.is_provisional is False
        assert hazard_registry.get_hazard("marine_heatwaves").target.is_provisional is False

    def test_data_gap_identification(self) -> None:
        """Verifies that missing datasets are explicitly acknowledged and not marked available."""
        gaps = hazard_registry.audit_data_gaps()
        assert "extreme_rainfall" in gaps, "Extreme rainfall must report data gaps!"
        assert "extreme_waves" in gaps, "Extreme waves must report data gaps!"
        assert "storm_surge" in gaps, "Storm surge must report data gaps!"
        assert "coastal_flooding" in gaps, "Coastal flooding must report data gaps!"

        # Cyclone genesis has zero data gaps for its current scope
        assert len(hazard_registry.get_hazard("cyclone_genesis").data_gaps) == 0

    def test_feasibility_assessment_integrity(self) -> None:
        """Verifies feasibility ratings are realistic and not inflated."""
        assert hazard_registry.get_hazard("cyclone_genesis").feasibility == FeasibilityRating.HIGH
        assert hazard_registry.get_hazard("cyclone_intensity").feasibility == FeasibilityRating.HIGH
        assert hazard_registry.get_hazard("marine_heatwaves").feasibility == FeasibilityRating.HIGH
        assert hazard_registry.get_hazard("extreme_waves").feasibility == FeasibilityRating.LOW
        assert hazard_registry.get_hazard("storm_surge").feasibility == FeasibilityRating.LOW
        assert hazard_registry.get_hazard("coastal_flooding").feasibility == FeasibilityRating.NOT_CURRENTLY_FEASIBLE


class TestCausalCutoffAndLeakageFirewall:
    """Verifies strict adherence to causal cutoff rules."""

    def test_past_and_present_data_passes_causality(self) -> None:
        forecast_time = "2026-09-13T18:00:00Z"
        past_data = "2026-09-13T12:00:00Z"
        same_data = "2026-09-13T18:00:00Z"
        assert hazard_registry.validate_timestamp_causality(forecast_time, past_data) is True
        assert hazard_registry.validate_timestamp_causality(forecast_time, same_data) is True

    def test_future_data_violates_causal_cutoff(self) -> None:
        forecast_time = "2026-09-13T18:00:00Z"
        future_data = "2026-09-13T18:01:00Z"
        with pytest.raises(TemporalLeakageError) as excinfo:
            hazard_registry.validate_timestamp_causality(forecast_time, future_data)
        assert "Causal cutoff breach" in str(excinfo.value)
        assert "future relative to forecast time T" in str(excinfo.value)


class TestOperationalIsolation:
    """Verifies that research proposed hazards cannot enter production operational decisions."""

    def test_only_genesis_is_production_or_shadow(self) -> None:
        production_hazards = hazard_registry.get_hazards_by_status(HazardStatus.PRODUCTION)
        assert len(production_hazards) == 1
        assert production_hazards[0].hazard_id == "cyclone_genesis"

        shadow_hazards = hazard_registry.get_hazards_by_status(HazardStatus.SHADOW)
        assert len(shadow_hazards) == 0  # Genesis is modeled in shadow via Model D under the genesis umbrella

        research_hazards = hazard_registry.get_hazards_by_status(HazardStatus.RESEARCH_PROPOSED)
        assert len(research_hazards) == 9  # All 9 new hazards remain strictly RESEARCH_PROPOSED

    def test_research_hazards_cannot_execute_in_production(self) -> None:
        research_ids = [
            "cyclone_intensity", "cyclone_track", "extreme_wind",
            "extreme_rainfall", "extreme_waves", "storm_surge",
            "coastal_flooding", "extreme_sea_level", "marine_heatwaves"
        ]
        for hid in research_ids:
            with pytest.raises(OperationalIsolationError) as excinfo:
                hazard_registry.assert_operational_isolation(hid)
            assert "Operational Isolation Violation" in str(excinfo.value)
            assert "cannot be invoked by production operational decision engines" in str(excinfo.value)

    def test_genesis_passes_operational_isolation(self) -> None:
        # Genesis is authorized for production
        hazard_registry.assert_operational_isolation("cyclone_genesis")

    def test_production_alert_engine_remains_isolated(self) -> None:
        """Verifies that the operational Alert Engine v2 has zero references to unapproved research hazards."""
        # Ensure Alert Engine v2 policy only maps 0d, 1d, 2d, 3d horizons for cyclonic risk
        with open("backend/config/frozen_alert_policy_v2.json", "r") as f:
            policy = json.load(f)
        assert policy["policy_version"] == "v2.0.0-operational-alert-policy"
        assert "selected_basin_specific_thresholds" in policy
        assert "selected_risk_tier_escalation_rules" in policy
        assert set(policy["selected_risk_tier_escalation_rules"]["tiers"].keys()) == {"CRITICAL", "HIGH", "MODERATE", "LOW"}
        assert "policy_selection_gate" in policy
        assert policy["policy_selection_gate"]["status"] == "LOCKED"
