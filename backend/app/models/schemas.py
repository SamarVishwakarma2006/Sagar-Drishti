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


class ChatRequest(BaseModel):
    message: str
    context: ChatContext
    api_key: Optional[str] = None
    provider: Optional[str] = "offline"  # "gemini" | "groq" | "openai" | "offline"


class ChatResponse(BaseModel):
    reply: str
    provider: str
    grounded_context: Dict[str, Any]


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
