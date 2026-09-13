"""
Sagar-Drishti Multi-Hazard Registry & Architecture Specification Service

Document ID: SD-SERVICE-2026-HAZARD-REGISTRY
Phase: Pre-Implementation Research & Architecture Phase
Classification: Research Specification & Modular Registry

Absolute Guardrails:
- Pure metadata, configuration, and architectural specification layer.
- Zero model training, fine-tuning, or inference modification.
- Strict operational isolation: RESEARCH_PROPOSED modules cannot be invoked
  by production operational decision authorities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class HazardCategory(str, Enum):
    CYCLONE_INTELLIGENCE = "CYCLONE_INTELLIGENCE"
    EXTREME_WEATHER = "EXTREME_WEATHER"
    MARINE_OCEAN_HAZARDS = "MARINE_OCEAN_HAZARDS"
    COASTAL_IMPACT = "COASTAL_IMPACT"


class HazardStatus(str, Enum):
    PRODUCTION = "PRODUCTION"                  # Sole operational decision authority (v1.1.0)
    SHADOW = "SHADOW"                          # Background shadow observation only (V2.3 Model D)
    RESEARCH_PROPOSED = "RESEARCH_PROPOSED"    # Pre-implementation / research architecture only


class FeasibilityRating(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_CURRENTLY_FEASIBLE = "NOT_CURRENTLY_FEASIBLE"


class LabelType(str, Enum):
    OBSERVATIONAL_GROUND_TRUTH = "OBSERVATIONAL_GROUND_TRUTH"
    REANALYSIS_REFERENCE_DATA = "REANALYSIS_REFERENCE_DATA"
    OPERATIONAL_FORECAST_DATA = "OPERATIONAL_FORECAST_DATA"
    DERIVED_LABEL = "DERIVED_LABEL"


class ValidationLevel(str, Enum):
    ROW_LEVEL = "ROW_LEVEL"
    EVENT_LEVEL = "EVENT_LEVEL"
    TRACK_LEVEL = "TRACK_LEVEL"
    CONTINUOUS = "CONTINUOUS"
    PROBABILISTIC = "PROBABILISTIC"
    SPATIAL_PIXEL = "SPATIAL_PIXEL"


class OperationalIsolationError(RuntimeError):
    """Raised when an attempt is made to execute or promote a research hazard into operational production."""
    pass


class TemporalLeakageError(ValueError):
    """Raised when data timestamp violates the strict causal cutoff at forecast time T."""
    pass


@dataclass(frozen=True)
class HazardTargetDefinition:
    """Explicit mathematical and domain definition of a hazard target."""
    target_id: str
    mathematical_formula: str
    units: str
    target_type: str                            # binary_classification, regression, vector_displacement, etc.
    prediction_horizons: List[str]             # e.g., ["0d", "1d", "2d", "3d"] or ["24h", "48h"]
    is_provisional: bool                        # True if threshold is provisional / pending national calibration
    threshold_value: Optional[float] = None
    threshold_description: Optional[str] = None


@dataclass(frozen=True)
class HazardDataRequirement:
    """Data prerequisites and availability status for a hazard."""
    dataset_name: str
    provider: str
    variables: List[str]
    is_currently_available: bool
    data_tier: LabelType
    spatial_resolution: str
    temporal_resolution: str
    latency_description: str
    licensing: str
    known_limitations: str


@dataclass(frozen=True)
class HazardValidationProtocol:
    """Validation standards, split strategies, and metrics."""
    evaluation_level: ValidationLevel
    primary_metric: str
    secondary_metrics: List[str]
    split_strategy: str                         # e.g., "Chronological 7yr train, 2yr val, 1yr test"
    requires_event_aggregation: bool
    target_benchmark: str


@dataclass(frozen=True)
class HazardModuleConfig:
    """Complete declarative specification for an individual hazard module."""
    hazard_id: str
    name: str
    category: HazardCategory
    status: HazardStatus
    target: HazardTargetDefinition
    data_requirements: List[HazardDataRequirement]
    label_source: str
    label_type: LabelType
    causal_cutoff_rule: str
    leakage_risks: List[str]
    validation_protocol: HazardValidationProtocol
    feasibility: FeasibilityRating
    data_gaps: List[str]
    provisional_notes: Optional[str] = None


class HazardRegistry:
    """
    Central Declarative Architecture Registry for Sagar-Drishti Multi-Hazard Platform.
    
    Maintains all 10 hazard configurations with strict operational isolation
    and causal cutoff verification.
    """

    def __init__(self) -> None:
        self._hazards: Dict[str, HazardModuleConfig] = {}
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Populate the declarative registry with all 10 hazard specifications."""

        # ---------------------------------------------------------------------
        # 1. Tropical Cyclone Genesis (CURRENT - Production / Shadow)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="cyclone_genesis",
            name="Tropical Cyclone Genesis Risk",
            category=HazardCategory.CYCLONE_INTELLIGENCE,
            status=HazardStatus.PRODUCTION,  # Operational v1.1.0 & Shadow V2.3 Model D
            target=HazardTargetDefinition(
                target_id="p_genesis_within_h",
                mathematical_formula="Y_genesis(T+H) = 1 if cyclone fix within 500km at T+H, else 0 (with 48h buffer)",
                units="probability [0, 1]",
                target_type="binary_classification",
                prediction_horizons=["0d", "1d", "2d", "3d"],
                is_provisional=False,
                threshold_value=0.5,
                threshold_description="IMD RSMC standard cyclonic disturbance criteria"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="features_10yr.parquet",
                    provider="Copernicus Marine (GLORYS12V1)",
                    variables=["SST", "MLD", "Salinity", "SLA", "Currents"],
                    is_currently_available=True,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="0.083 deg point & regional averages",
                    temporal_resolution="Daily",
                    latency_description="Delayed reanalysis; NRT 24h",
                    licensing="Copernicus Open Access",
                    known_limitations="Coarse nearshore representation"
                ),
                HazardDataRequirement(
                    dataset_name="features_atmosphere_10yr_daily.parquet",
                    provider="ECMWF (ERA5)",
                    variables=["Vorticity", "VWS", "RH", "Divergence", "MSLP"],
                    is_currently_available=True,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="0.25 deg basin summaries",
                    temporal_resolution="Daily (18Z cutoff)",
                    latency_description="ERA5 2-3 months; ERA5T 5 days",
                    licensing="Copernicus C3S",
                    known_limitations="Smoothed peak wind speeds"
                )
            ],
            label_source="IMD RSMC Official Best Track (1982-2026)",
            label_type=LabelType.DERIVED_LABEL,
            causal_cutoff_rule="Data timestamp <= forecast time T (strict 18:00 UTC cutoff)",
            leakage_risks=["Post-season track revisions", "Lookahead rolling windows"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.EVENT_LEVEL,
                primary_metric="PR-AUC",
                secondary_metrics=["Brier Score", "ECE", "Precision", "Recall", "CSI"],
                split_strategy="Chronological (2016-2024 train, 2024-2025 val, 2025-2026 test)",
                requires_event_aggregation=True,
                target_benchmark="Climatology baseline"
            ),
            feasibility=FeasibilityRating.HIGH,
            data_gaps=[]
        ))

        # ---------------------------------------------------------------------
        # 2. Cyclone Intensity (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="cyclone_intensity",
            name="Cyclone Intensity & Rapid Intensification",
            category=HazardCategory.CYCLONE_INTELLIGENCE,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="v_max_and_ri",
                mathematical_formula="V_max(T+H) in knots (continuous) and 1(V_max(T+24) - V_max(T) >= 30 kt) (RI)",
                units="knots & binary RI",
                target_type="multi_task_regression_and_classification",
                prediction_horizons=["12h", "24h", "36h", "48h"],
                is_provisional=False,
                threshold_value=30.0,
                threshold_description="WMO/IMD standard for Rapid Intensification (>=30 kt in 24h)"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="imd_tracks_2016_2026.parquet",
                    provider="IMD RSMC New Delhi",
                    variables=["max_wind_kts", "central_pressure_hpa", "pressure_drop_hpa"],
                    is_currently_available=True,
                    data_tier=LabelType.OBSERVATIONAL_GROUND_TRUTH,
                    spatial_resolution="Point synoptic fixes",
                    temporal_resolution="6-hourly",
                    latency_description="Official post-season review",
                    licensing="IMD Official Meteorological Archive",
                    known_limitations="Open ocean Dvorak subjective estimates"
                )
            ],
            label_source="IMD RSMC Best Track Synoptic Fixes",
            label_type=LabelType.OBSERVATIONAL_GROUND_TRUTH,
            causal_cutoff_rule="Storm intensity at or prior to T; no future fixes permitted",
            leakage_risks=["Post-event intensity re-analysis", "Future environmental shear"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.TRACK_LEVEL,
                primary_metric="MAE on V_max (knots)",
                secondary_metrics=["RMSE", "Bias", "RI Brier Score", "RI CSI"],
                split_strategy="Leave-one-season-out chronological cross-validation",
                requires_event_aggregation=True,
                target_benchmark="SHIPS / CLIPER benchmark"
            ),
            feasibility=FeasibilityRating.HIGH,
            data_gaps=["Operational real-time satellite infrared brightness temperature (INSAT-3D)"]
        ))

        # ---------------------------------------------------------------------
        # 3. Cyclone Track Displacement (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="cyclone_track",
            name="Cyclone Track & Trajectory Displacement",
            category=HazardCategory.CYCLONE_INTELLIGENCE,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="delta_trajectory",
                mathematical_formula="Delta_pos(T+H) = [lat(T+H) - lat(T), lon(T+H) - lon(T)] in degrees",
                units="degrees displacement / km error",
                target_type="vector_trajectory_regression",
                prediction_horizons=["24h", "48h", "72h"],
                is_provisional=False,
                threshold_description="Great-circle along-track and cross-track error"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="imd_tracks_2016_2026.parquet",
                    provider="IMD RSMC New Delhi",
                    variables=["latitude", "longitude", "basin"],
                    is_currently_available=True,
                    data_tier=LabelType.OBSERVATIONAL_GROUND_TRUTH,
                    spatial_resolution="0.1 degree fix precision",
                    temporal_resolution="6-hourly",
                    latency_description="Official post-season review",
                    licensing="IMD Official Meteorological Archive",
                    known_limitations="6-hour temporal coarseness"
                )
            ],
            label_source="IMD RSMC Synoptic Fix Coordinates",
            label_type=LabelType.OBSERVATIONAL_GROUND_TRUTH,
            causal_cutoff_rule="Storm positions tau <= T; no post-initialization fixes",
            leakage_risks=["Future steering winds", "Post-season track smoothing"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.TRACK_LEVEL,
                primary_metric="Mean Track Error (MTE, km)",
                secondary_metrics=["Along-Track Error (ATE)", "Cross-Track Error (CTE)"],
                split_strategy="Chronological season splits",
                requires_event_aggregation=True,
                target_benchmark="CLIPER / NWP Consensus"
            ),
            feasibility=FeasibilityRating.MEDIUM,
            data_gaps=["Dynamic NWP ensemble steering flow forecasts at runtime"]
        ))

        # ---------------------------------------------------------------------
        # 4. Extreme Wind (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="extreme_wind",
            name="Extreme Synoptic & Monsoonal Wind",
            category=HazardCategory.EXTREME_WEATHER,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="p_wind_exceedance",
                mathematical_formula="P(max_{t in [T, T+H]} U_10(t) >= 34.0 kt)",
                units="probability [0, 1]",
                target_type="probabilistic_classification",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=True,
                threshold_value=34.0,
                threshold_description="Provisional: Beaufort Force 8 / Gale Force threshold (17.2 m/s)"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="features_atmosphere_10yr_daily.parquet",
                    provider="ECMWF (ERA5)",
                    variables=["u10", "v10", "msl"],
                    is_currently_available=True,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="0.25 deg grid",
                    temporal_resolution="Daily maxima",
                    latency_description="ERA5 delayed; ERA5T preliminary",
                    licensing="Copernicus C3S",
                    known_limitations="Underestimates convective gusts"
                )
            ],
            label_source="ERA5 10m Wind Fields (Reference) / Coastal Anemometers (Missing)",
            label_type=LabelType.REANALYSIS_REFERENCE_DATA,
            causal_cutoff_rule="Surface observations tau <= T",
            leakage_risks=["ERA5 assimilation window lookahead", "Nearshore boundary smoothing"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.ROW_LEVEL,
                primary_metric="Brier Score",
                secondary_metrics=["PR-AUC", "ECE", "Critical Success Index (CSI)", "FAR"],
                split_strategy="Chronological multi-year train/val/test splits",
                requires_event_aggregation=False,
                target_benchmark="Persistence / Climatology"
            ),
            feasibility=FeasibilityRating.HIGH,
            data_gaps=["IMD Coastal AWS surface anemometer observational records"],
            provisional_notes="34.0 kt threshold is provisional pending coastal station calibration."
        ))

        # ---------------------------------------------------------------------
        # 5. Extreme Rainfall (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="extreme_rainfall",
            name="Extreme Daily Precipitation",
            category=HazardCategory.EXTREME_WEATHER,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="p_heavy_rain",
                mathematical_formula="P(R_24(T+H) >= 64.5 mm/day)",
                units="probability [0, 1]",
                target_type="probabilistic_classification",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=True,
                threshold_value=64.5,
                threshold_description="Provisional: IMD 'Heavy Rainfall' classification (64.5 - 115.5 mm/day)"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="imd_gridded_rain_025 (MISSING)",
                    provider="IMD National Climate Centre",
                    variables=["rainfall_mm_24h"],
                    is_currently_available=False,
                    data_tier=LabelType.OBSERVATIONAL_GROUND_TRUTH,
                    spatial_resolution="0.25 deg gridded",
                    temporal_resolution="Daily (08:30 IST / 03:00 UTC)",
                    latency_description="Daily 24h delayed",
                    licensing="IMD Open Research Access",
                    known_limitations="Over-land only; zero oceanic rain gauges"
                )
            ],
            label_source="IMD 0.25 deg Daily Gridded Rainfall / NASA GPM IMERG",
            label_type=LabelType.OBSERVATIONAL_GROUND_TRUTH,
            causal_cutoff_rule="Rainfall accumulation tau <= T (03:00 UTC previous day)",
            leakage_risks=["Rain gauge reporting lag", "24-hour accumulation window offset"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.EVENT_LEVEL,
                primary_metric="Critical Success Index (CSI)",
                secondary_metrics=["Frequency Bias", "ETS", "F1 Score", "Brier Score"],
                split_strategy="Spatially and chronologically blocked cross-validation",
                requires_event_aggregation=True,
                target_benchmark="Numerical Weather Prediction (NCMRWF) raw rain"
            ),
            feasibility=FeasibilityRating.MEDIUM,
            data_gaps=["IMD 0.25 deg gridded rainfall dataset", "NASA GPM IMERG satellite precipitation"],
            provisional_notes="64.5 mm/day threshold is provisional pending spatial verification."
        ))

        # ---------------------------------------------------------------------
        # 6. Extreme Waves / Dangerous Sea State (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="extreme_waves",
            name="Extreme Waves & Dangerous Sea State",
            category=HazardCategory.MARINE_OCEAN_HAZARDS,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="p_high_sea_state",
                mathematical_formula="P(max_{t in [T, T+H]} H_s(t) >= 4.0 m)",
                units="probability [0, 1]",
                target_type="probabilistic_classification_and_regression",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=True,
                threshold_value=4.0,
                threshold_description="Provisional: WMO Douglas Sea State Code 6 (Very Rough Sea, 4-6m)"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="copernicus_waverys (MISSING)",
                    provider="Copernicus Marine (CMEMS)",
                    variables=["VHM0 (Hs)", "VTPK (Tp)", "VMDR (Dir)", "VHM0_SW1 (Swell)"],
                    is_currently_available=False,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="0.2 deg grid",
                    temporal_resolution="3-hourly",
                    latency_description="Delayed reanalysis",
                    licensing="Copernicus Open Access",
                    known_limitations="Coarse nearshore bathymetry"
                )
            ],
            label_source="Copernicus WAVERYS / INCOIS Wave Rider Buoys",
            label_type=LabelType.REANALYSIS_REFERENCE_DATA,
            causal_cutoff_rule="Wave spectra and wind forcing tau <= T",
            leakage_risks=["Remote swell propagation latency", "Altimeter assimilation lookahead"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.CONTINUOUS,
                primary_metric="RMSE on H_s (meters)",
                secondary_metrics=["Scatter Index (SI)", "Peak Bias", "PR-AUC (Hs >= 4m)"],
                split_strategy="Chronological train/val/test splits",
                requires_event_aggregation=False,
                target_benchmark="INCOIS WaveWatch III operational forecasts"
            ),
            feasibility=FeasibilityRating.LOW,
            data_gaps=["Copernicus WAVERYS wave reanalysis", "INCOIS Wave Rider Buoy records"],
            provisional_notes="4.0m Hs threshold is provisional pending buoy validation."
        ))

        # ---------------------------------------------------------------------
        # 7. Storm Surge (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="storm_surge",
            name="Coastal Storm Surge Residual",
            category=HazardCategory.COASTAL_IMPACT,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="eta_peak_surge",
                mathematical_formula="P(max_{t in [T, T+H]} [h(t) - zeta_tide(t)] >= 1.0 m)",
                units="meters surge residual",
                target_type="regression_and_classification",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=True,
                threshold_value=1.0,
                threshold_description="Provisional: 1.0 meter coastal surge residual above astronomical tide"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="incois_tide_gauges (MISSING)",
                    provider="INCOIS / Survey of India",
                    variables=["water_level", "surge_residual"],
                    is_currently_available=False,
                    data_tier=LabelType.OBSERVATIONAL_GROUND_TRUTH,
                    spatial_resolution="Point coastal tide gauge stations",
                    temporal_resolution="Hourly / 6-minute",
                    latency_description="Real-time telemetry",
                    licensing="Restricted Government Access",
                    known_limitations="Gauge siltation and storm cutoff failures"
                )
            ],
            label_source="INCOIS Coastal Tide Gauge Network",
            label_type=LabelType.OBSERVATIONAL_GROUND_TRUTH,
            causal_cutoff_rule="Atmospheric forcing and water level tau <= T",
            leakage_risks=["Post-event datum realignment", "Surge-tide non-linear interaction"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.EVENT_LEVEL,
                primary_metric="Peak Surge Height MAE (meters)",
                secondary_metrics=["Arrival Timing Error (hours)", "CSI (Surge >= 1m)"],
                split_strategy="Storm-by-storm leave-one-out cross-validation",
                requires_event_aggregation=True,
                target_benchmark="IIT Delhi Storm Surge Model / INCOIS ADCIRC"
            ),
            feasibility=FeasibilityRating.LOW,
            data_gaps=["INCOIS tide gauge records", "GEBCO 15 arc-sec coastal bathymetry", "TPXO tidal constituents"],
            provisional_notes="1.0m surge threshold is provisional pending gauge network integration."
        ))

        # ---------------------------------------------------------------------
        # 8. Coastal Flooding (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="coastal_flooding",
            name="Coastal Compound Inundation & Flooding",
            category=HazardCategory.COASTAL_IMPACT,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="flood_extent_and_depth",
                mathematical_formula="A_flood(T+H) in km^2 & 1(d_flood(x, y, T+H) >= 0.5 m)",
                units="inundation depth (m) and area (km2)",
                target_type="spatial_segmentation_and_depth_regression",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=True,
                threshold_value=0.5,
                threshold_description="Provisional: Inundation water depth >= 0.5 meters"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="copernicus_dem_30m (MISSING)",
                    provider="Copernicus GLO-30 / ISRO",
                    variables=["elevation_m", "slope", "roughness"],
                    is_currently_available=False,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="30 meter grid",
                    temporal_resolution="Static topographic baseline",
                    latency_description="Static",
                    licensing="Copernicus Open Access",
                    known_limitations="Vegetation canopy errors in mangrove deltas"
                )
            ],
            label_source="Sentinel-1 SAR Satellite Flood Maps / Field Inundation Surveys",
            label_type=LabelType.OBSERVATIONAL_GROUND_TRUTH,
            causal_cutoff_rule="Topography, tide, surge, and rainfall inputs tau <= T",
            leakage_risks=["Post-event damage survey leaks", "Pluvial-fluvial runoff delay"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.SPATIAL_PIXEL,
                primary_metric="Spatial Intersection over Union (IoU)",
                secondary_metrics=["Critical Success Index", "Mean Depth Error (m)"],
                split_strategy="Geographically and historically holdout landfalls",
                requires_event_aggregation=True,
                target_benchmark="Hydrodynamic 2D Inundation Model (LFP / Delft3D)"
            ),
            feasibility=FeasibilityRating.NOT_CURRENTLY_FEASIBLE,
            data_gaps=["30m Coastal DEM", "River discharge (CWC)", "Drainage network GIS layers"],
            provisional_notes="0.5m depth threshold is provisional; module currently not feasible."
        ))

        # ---------------------------------------------------------------------
        # 9. Extreme Sea-Level Events (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="extreme_sea_level",
            name="Extreme Non-Cyclonic Sea-Level Events",
            category=HazardCategory.MARINE_OCEAN_HAZARDS,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="p_extreme_total_water_level",
                mathematical_formula="P(h(T+H) >= h_crit,99.5)",
                units="probability [0, 1]",
                target_type="extreme_value_probabilistic_classification",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=True,
                threshold_description="Provisional: Local 99.5th percentile total coastal water level"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="features_10yr.parquet (SLA offshore only)",
                    provider="Copernicus Marine",
                    variables=["sla"],
                    is_currently_available=True,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="0.083 deg point extractions",
                    temporal_resolution="Daily",
                    latency_description="Delayed mode",
                    licensing="Copernicus Open Access",
                    known_limitations="Altimetry SLA is offshore; misses coastal wave setup & tide"
                )
            ],
            label_source="Survey of India Coastal Tide Gauge Network",
            label_type=LabelType.OBSERVATIONAL_GROUND_TRUTH,
            causal_cutoff_rule="Water level and tidal constituents tau <= T",
            leakage_risks=["Datum shifts", "Secular sea level rise non-stationarity"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.ROW_LEVEL,
                primary_metric="Quantile Score (99.5th percentile)",
                secondary_metrics=["Brier Score", "Return Period Error (years)"],
                split_strategy="Chronological split",
                requires_event_aggregation=False,
                target_benchmark="Harmonic Astronomical Tide + Climatological SLA"
            ),
            feasibility=FeasibilityRating.LOW,
            data_gaps=["Long-term coastal tide gauge records", "Astronomical tidal constituents"],
            provisional_notes="Local 99.5th percentile threshold is provisional."
        ))

        # ---------------------------------------------------------------------
        # 10. Marine / Ocean Anomalies (Marine Heatwaves) (RESEARCH_PROPOSED)
        # ---------------------------------------------------------------------
        self._register(HazardModuleConfig(
            hazard_id="marine_heatwaves",
            name="Marine Heatwaves & Thermal Anomalies",
            category=HazardCategory.MARINE_OCEAN_HAZARDS,
            status=HazardStatus.RESEARCH_PROPOSED,
            target=HazardTargetDefinition(
                target_id="p_mhw_event",
                mathematical_formula="1(SST(t) >= SST_90th(doy) for >= 5 consecutive days within [T, T+H])",
                units="probability [0, 1] & degree-days",
                target_type="probabilistic_classification_and_intensity",
                prediction_horizons=["1d", "2d", "3d"],
                is_provisional=False,
                threshold_value=90.0,
                threshold_description="Hobday et al. (2016) International MHW Standard: >=90th percentile for >=5 days"
            ),
            data_requirements=[
                HazardDataRequirement(
                    dataset_name="features_10yr.parquet",
                    provider="Copernicus Marine",
                    variables=["temp_current", "temp_base_mean", "temp_abs_anom", "temp_zscore"],
                    is_currently_available=True,
                    data_tier=LabelType.REANALYSIS_REFERENCE_DATA,
                    spatial_resolution="0.083 deg extractions",
                    temporal_resolution="Daily",
                    latency_description="Daily analysis",
                    licensing="Copernicus Open Access",
                    known_limitations="Nearshore coastal satellite bias"
                )
            ],
            label_source="Copernicus GLORYS SST / NOAA OISST Daily Climatology",
            label_type=LabelType.DERIVED_LABEL,
            causal_cutoff_rule="SST time series tau <= T; no forward-looking climatological windows",
            leakage_risks=["Climatology calculated including target year (target leakage)"],
            validation_protocol=HazardValidationProtocol(
                evaluation_level=ValidationLevel.EVENT_LEVEL,
                primary_metric="MHW Day F1 Score",
                secondary_metrics=["Duration MAE (days)", "Peak Intensity Error (deg C)", "ECE"],
                split_strategy="Chronological multi-year holdout",
                requires_event_aggregation=True,
                target_benchmark="SST Climatological Persistence"
            ),
            feasibility=FeasibilityRating.HIGH,
            data_gaps=["Frozen 30-year daily baseline climatology (1982-2011)"],
            provisional_notes="Standard follows Hobday et al. (2016); 30-year climatology baseline to freeze."
        ))

    def _register(self, config: HazardModuleConfig) -> None:
        """Register a hazard module configuration."""
        self._hazards[config.hazard_id] = config

    def get_hazard(self, hazard_id: str) -> HazardModuleConfig:
        """Retrieve a specific hazard configuration by ID."""
        if hazard_id not in self._hazards:
            raise KeyError(f"Hazard '{hazard_id}' is not registered in Sagar-Drishti.")
        return self._hazards[hazard_id]

    def list_all_hazards(self) -> List[HazardModuleConfig]:
        """Return all registered hazard configurations."""
        return list(self._hazards.values())

    def get_hazards_by_status(self, status: HazardStatus) -> List[HazardModuleConfig]:
        """Filter hazards by operational status."""
        return [h for h in self._hazards.values() if h.status == status]

    def get_hazards_by_feasibility(self, feasibility: FeasibilityRating) -> List[HazardModuleConfig]:
        """Filter hazards by physical feasibility rating."""
        return [h for h in self._hazards.values() if h.feasibility == feasibility]

    def audit_data_gaps(self) -> Dict[str, List[str]]:
        """Audit all data gaps across registered hazards."""
        return {h.hazard_id: h.data_gaps for h in self._hazards.values() if h.data_gaps}

    def validate_timestamp_causality(self, forecast_time_iso: str, data_timestamp_iso: str) -> bool:
        """
        Enforce the strict causal cutoff firewall:
        Only information available at or before forecast initialization time T may enter prediction.
        """
        t_forecast = datetime.fromisoformat(forecast_time_iso.replace("Z", "+00:00"))
        t_data = datetime.fromisoformat(data_timestamp_iso.replace("Z", "+00:00"))
        if t_data > t_forecast:
            raise TemporalLeakageError(
                f"Causal cutoff breach: Data timestamp ({data_timestamp_iso}) is in the future "
                f"relative to forecast time T ({forecast_time_iso}). Temporal leakage prohibited."
            )
        return True

    def assert_operational_isolation(self, hazard_id: str) -> None:
        """
        Verify that a given hazard is authorized for operational execution.
        Raises OperationalIsolationError if a RESEARCH_PROPOSED hazard attempts production execution.
        """
        hazard = self.get_hazard(hazard_id)
        if hazard.status == HazardStatus.RESEARCH_PROPOSED:
            raise OperationalIsolationError(
                f"Operational Isolation Violation: Hazard '{hazard_id}' is classified as "
                f"RESEARCH_PROPOSED and cannot be invoked by production operational decision engines. "
                f"Production v1.1.0 remains the sole operational decision authority."
            )


# Global singleton instance of HazardRegistry
hazard_registry = HazardRegistry()
