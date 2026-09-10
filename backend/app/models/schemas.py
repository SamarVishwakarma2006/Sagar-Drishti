from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    min_lat: float = Field(..., description="Minimum latitude in degrees (-90 to 90)")
    max_lat: float = Field(..., description="Maximum latitude in degrees (-90 to 90)")
    min_lon: float = Field(..., description="Minimum longitude in degrees (-180 to 180)")
    max_lon: float = Field(..., description="Maximum longitude in degrees (-180 to 180)")


class VariableStats(BaseModel):
    name: str
    standard_name: Optional[str] = None
    long_name: Optional[str] = None
    unit: str = ""
    min: float
    max: float
    mean: Optional[float] = None
    std: Optional[float] = None


class ProfilePoint(BaseModel):
    depth: float
    temperature: float
    salinity: float
    currentSpeed: float
    currentDir: float
    oxygen: float


class ProfileResult(BaseModel):
    zmax: float
    points: List[ProfilePoint]


class FloatRecord(BaseModel):
    id: str
    platform: str
    lat: float
    lon: float
    cycle: int
    lastReportOffset: float
    parkingDepth: float
    profile: ProfileResult
    source: str


class SitePhysics(BaseModel):
    id: str
    name: str
    region: str
    lat: float
    lon: float
    maxDepth: float
    blurb: str
    ts: float
    ss: float
    td: float
    sd: float
    mld: float
    tw: float
    salMaxAmp: float
    salMaxZ: float
    flow: float
    eddy: float
    bgU: float
    bgV: float
    o2s: float
    o2d: float
    o2z0: float
    o2minAmp: float
    o2minZ: float
    o2minW: float
    bbox: Optional[BoundingBox] = None
    variables: Optional[List[str]] = Field(default_factory=lambda: ["temp", "sal", "cur", "oxy"])
    isCustom: bool = False
    sourceType: str = "INCOIS_BASELINE"
    floats: Optional[List[FloatRecord]] = None
    customFilePath: Optional[str] = None


class DataSliceRequest(BaseModel):
    dataset_id: str
    variable: str = "temp"
    depth: float = 0.0
    time_offset: float = 0.0
    resolution: Optional[int] = 64


class DataSliceResponse(BaseModel):
    dataset_id: str
    variable: str
    depth: float
    time_offset: float
    lats: List[float]
    lons: List[float]
    values: List[List[Optional[float]]]
    min_val: float
    max_val: float
    unit: str
    shape: List[int]


class ChatContext(BaseModel):
    active_site: str
    coordinates: Dict[str, float]
    current_depth: str
    variable: str
    current_value: str
    time_offset: str
    nearby_floats: List[str] = Field(default_factory=list)
    custom_data: bool = False
    historical_date: Optional[str] = None
    historical_mode: Optional[bool] = False
    active_prediction: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000, description="User message (max 2000 characters)")
    context: ChatContext
    api_key: Optional[str] = None
    provider: Optional[str] = "offline"  # "gemini" | "groq" | "openai" | "offline"


class ChatResponse(BaseModel):
    reply: str
    provider: str
    grounded_context: Dict[str, Any]
    evidence: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None


class SagarBotChatRequest(ChatRequest):
    """Explicit SagarBot request model extending ChatRequest."""
    pass


class SagarBotChatResponse(ChatResponse):
    """Explicit SagarBot response model extending ChatResponse."""
    pass


class UploadResponse(BaseModel):
    site_id: str
    name: str
    filename: str
    file_type: str
    bounding_box: BoundingBox
    depth_range: Dict[str, float]
    variables: List[str]
    float_count: int
    site_record: SitePhysics
    message: str
    # When a CSV contains Copernicus-style physical columns, this carries the
    # extracted single-day observation for custom prediction routing.
    custom_observation: Optional[Dict[str, Any]] = None



class HistoricalStatusResponse(BaseModel):
    status: str = Field(..., description="'idle' | 'in_progress' | 'ready' | 'failed'")
    dataset_id: str = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    is_ready: bool = False
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_size_human: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_days: Optional[int] = None
    bounding_box: Optional[BoundingBox] = None
    depth_m: Optional[float] = None
    variables: List[str] = Field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
    message: str = ""


class HistoricalIngestResponse(BaseModel):
    status: str
    task_id: str
    message: str
    started_at: str


class HistoricalSliceResponse(BaseModel):
    dataset_id: str
    date: str
    variable: str
    standard_name: str
    depth: float
    unit: str
    lats: List[float]
    lons: List[float]
    values: List[List[Optional[float]]]
    min_val: float
    max_val: float
    shape: List[int]
    provenance: str = "Copernicus Marine Global Ocean Physics Reanalysis"


class HistoricalPointResponse(BaseModel):
    date: str
    lat: float
    lon: float
    variable: str
    value: Optional[float]
    unit: str
    depth: float
    provenance: str = "Copernicus Marine Global Ocean Physics Reanalysis"


class BaselineMetadata(BaseModel):
    baseline_type: str = "full_period_climatology"
    baseline_start: str
    baseline_end: str
    baseline_observations: int


class WindowMetrics(BaseModel):
    window_days: int
    window_start: str
    window_end: str
    mean: float
    delta: float
    trend_per_day: float
    min: float
    max: float


class VariableAnomaly(BaseModel):
    variable: str
    standard_name: str
    unit: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    baseline_min: float
    baseline_max: float
    absolute_anomaly: float
    z_score: float
    percentage_anomaly: Optional[float] = None
    windows: Dict[str, WindowMetrics] = Field(default_factory=dict)


class HistoricalComparisonResponse(BaseModel):
    dataset_id: str = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    date: str
    mode: str = "point"  # "point" | "region"
    location: Dict[str, Any] = Field(default_factory=dict)
    baseline_metadata: BaselineMetadata
    variables: Dict[str, VariableAnomaly] = Field(default_factory=dict)
    feature_vector: Dict[str, float] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    provenance: str = "Copernicus Marine Global Ocean Physics Reanalysis (0.083° daily)"


class TrainingDatasetExportResponse(BaseModel):
    status: str
    parquet_path: str
    metadata_path: str
    total_rows: int
    total_features: int
    date_range: Dict[str, str]
    baseline_metadata: BaselineMetadata
    message: str


# ==============================================================================
# PHASE 3: HISTORICAL EVENT LABELING & ML DATASET SCHEMAS
# ==============================================================================

class EventType(str, Enum):
    TROPICAL_CYCLONE = "tropical_cyclone"
    DEPRESSION = "depression"
    COASTAL_SWELL_SURGE = "coastal_swell_surge"
    MARINE_HEATWAVE = "marine_heatwave"


class SpatialRepresentation(str, Enum):
    TRACK_ENVELOPE = "track_envelope"
    REGIONAL_ENVELOPE = "regional_envelope"


class LabelStatus(str, Enum):
    ACTIVE_EVENT = "active_event"
    LEAD_EVENT = "lead_event"
    NEGATIVE_CLEAN = "negative_clean"
    NEGATIVE_BUFFER = "negative_buffer"
    UNKNOWN_UNCOVERED = "unknown_uncovered"


class HistoricalEvent(BaseModel):
    event_id: str
    event_type: EventType
    name: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    affected_region: str
    spatial_representation: SpatialRepresentation = SpatialRepresentation.REGIONAL_ENVELOPE
    bbox: BoundingBox
    track_coordinates: Optional[List[Dict[str, Any]]] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    severity: str
    source: str
    source_reference: str
    confidence: str = "verified_authoritative"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LabeledObservation(BaseModel):
    observation_date: str
    day_index: int
    mode: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    site_id: Optional[str] = None
    event_active: int
    is_active_event: int
    event_present: int
    event_within_1d: int
    event_within_2d: int
    event_within_3d: int
    lead_0: int
    lead_1: int
    lead_2: int
    lead_3: int
    lead_days: int
    label_status: LabelStatus
    event_type: Optional[str] = None
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    severity: Optional[str] = None
    label_source: str
    label_confidence: str
    features: Dict[str, float] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    total_events: int
    events_by_type: Dict[str, int]
    events_spatially_matched: int
    events_temporally_matched: int
    unmatched_events: List[Dict[str, Any]]
    total_observations: int
    positive_observations: int
    negative_clean_observations: int
    negative_buffer_observations: int
    unknown_uncovered_observations: int
    lead_distribution: Dict[str, int]
    date_coverage: Dict[str, str]
    provenance_summary: List[Dict[str, Any]]
    methodology_notes: str


class LabeledDatasetExportResponse(BaseModel):
    status: str
    parquet_path: str
    metadata_path: str
    total_rows: int
    positive_rows: int
    negative_clean_rows: int
    negative_buffer_rows: int
    unknown_uncovered_rows: int
    data_quality_report: DataQualityReport
    message: str


# ==============================================================================
# ==============================================================================
# PHASE 4: ML PREDICTION & EXPLAINABILITY SCHEMAS
# ==============================================================================

class PredictionRequest(BaseModel):
    date: str
    mode: Optional[str] = "region"  # "point" | "region"
    site_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    horizon_days: int = 3  # 0 for active disturbance, 1..3 for early warning


class CustomObservationRequest(BaseModel):
    """
    One-shot custom observation request.
    Carries a single new in-situ/model observation with Copernicus-standard physical fields.
    The backend uses Copernicus history (preceding days) as rolling context and overrides
    the *_current features with the user-supplied values.
    """
    date: str = Field(..., description="Observation date ISO YYYY-MM-DD")
    lat: float = Field(..., description="Latitude (°N, 0–25)")
    lon: float = Field(..., description="Longitude (°E, 50–100)")
    horizon_days: int = Field(3, description="Forecast horizon 0–3 days")
    # Copernicus physical variable names (aliases accepted by HistoricalFeatureEngine)
    thetao: Optional[float] = Field(None, description="Sea water potential temperature (°C)")
    so: Optional[float] = Field(None, description="Sea water salinity (PSU)")
    uo: Optional[float] = Field(None, description="Eastward sea water velocity (m/s)")
    vo: Optional[float] = Field(None, description="Northward sea water velocity (m/s)")
    zos: Optional[float] = Field(None, description="Sea surface height above geoid (m)")
    mlotst: Optional[float] = Field(None, description="Mixed layer thickness (m)")



class FeatureAttribution(BaseModel):
    feature: str
    importance: float
    value: Optional[float] = None
    contribution: Optional[float] = None
    description: Optional[str] = None


class PhysicalDriversGroup(BaseModel):
    ocean_variables: Dict[str, float] = Field(default_factory=dict)
    temporal_scales: Dict[str, float] = Field(default_factory=dict)


class HumanReadableExplanation(BaseModel):
    what: str = ""
    where: str = ""
    when: str = ""
    why: str = ""


class PredictionExplainability(BaseModel):
    top_features: List[FeatureAttribution] = Field(default_factory=list)
    ocean_variable_importance: Dict[str, float] = Field(default_factory=dict)
    time_window_importance: Dict[str, float] = Field(default_factory=dict)
    human_readable: HumanReadableExplanation = Field(default_factory=HumanReadableExplanation)


class PredictionResponse(BaseModel):
    status: str  # "success" | "insufficient_data" | "error"
    date: str
    mode: str
    location: Dict[str, Any]
    horizon_days: int
    target: str = "event_within_3d"  # "event_within_0d" | "event_within_1d" | "event_within_2d" | "event_within_3d"
    prediction: str  # "alert" | "advisory" | "normal"
    warning_level: str = "NO_ALERT"  # "NO_ALERT" | "WATCH" | "HIGH_ALERT"
    probability: float  # Uncalibrated model-estimated risk score
    model_estimated_probability: Optional[float] = None  # Explicit uncalibrated score alias
    threshold: float = 0.27  # Horizon-specific operational decision threshold
    alert_threshold: Optional[float] = None  # Explicit horizon-specific decision threshold alias
    probability_display: str = ""
    event_type: str = "none"  # "tropical_cyclone", "depression", "none"
    is_calibrated: bool = False
    model_name: str = "RandomForestClassifier"
    model_version: str = "v1.1.0"
    prediction_timestamp: str = ""
    explainability: PredictionExplainability = Field(default_factory=PredictionExplainability)
    top_features: List[FeatureAttribution] = Field(default_factory=list)
    physical_drivers: PhysicalDriversGroup = Field(default_factory=PhysicalDriversGroup)
    observed_state: Dict[str, Any] = Field(default_factory=dict)
    predicted_state: Dict[str, Any] = Field(default_factory=dict)
    historical_context: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    message: str = ""



