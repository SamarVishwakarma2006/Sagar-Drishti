import os
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request, Response, status

from datetime import datetime, timezone
from ..models.schemas import (
    SitePhysics,
    FloatRecord,
    DataSliceResponse,
    ChatRequest,
    ChatResponse,
    UploadResponse,
    HistoricalStatusResponse,
    HistoricalIngestResponse,
    HistoricalSliceResponse,
    HistoricalPointResponse,
    HistoricalComparisonResponse,
    TrainingDatasetExportResponse,
    HistoricalEvent,
    EventType,
    LabeledDatasetExportResponse,
    DataQualityReport,
    PredictionRequest,
    PredictionResponse,
    CustomObservationRequest,
)
from ..parsers.netcdf_parser import NetCDFParser
from ..parsers.tabular_parser import TabularParser
from ..services.site_registry import SiteRegistry
from ..services.slice_engine import SliceEngine
from ..services.rate_limiter import chat_rate_limiter
from ..services.copernicus_service import CopernicusService
from ..services.historical_engine import HistoricalFeatureEngine
from ..services.event_store import HistoricalEventStore
from ..services.event_matcher import SpatialTemporalMatcher
from ..services.prediction_service import PredictionService


router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Sagar Drishti Ocean Ingestion & Visualization Engine",
        "version": "2.0.0",
        "active_sites_count": len(SiteRegistry.get_all_sites())
    }


@router.get("/sites", response_model=List[SitePhysics], tags=["Study Sites"])
async def list_study_sites():
    """
    Returns all active oceanographic study sites (INCOIS baseline + dynamically ingested datasets).
    """
    return SiteRegistry.get_all_sites()


@router.get("/sites/{site_id}", response_model=SitePhysics, tags=["Study Sites"])
async def get_study_site(site_id: str):
    """
    Returns detailed metadata for a specific study site.
    """
    site = SiteRegistry.get_site(site_id)
    if not site:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Study site '{site_id}' not found."
        )
    return site


@router.delete("/sites/{site_id}", tags=["Study Sites"])
async def delete_custom_site(site_id: str):
    """
    Deletes an uploaded dataset study site from the registry.
    """
    site = SiteRegistry.get_site(site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site not found.")
    if not site.isCustom:
        raise HTTPException(status_code=400, detail="Cannot delete default INCOIS baseline sites.")

    SiteRegistry.delete_uploaded_site(site_id)
    return {"status": "deleted", "site_id": site_id, "message": f"Dataset '{site.name}' unloaded successfully."}


@router.get("/floats/{site_id}", tags=["Instruments"])
async def get_site_floats(site_id: str, format: Optional[str] = "json"):
    """
    Returns float profiles for a specific site. Supports standard JSON or OGC GeoJSON.
    """
    floats = SiteRegistry.get_floats(site_id)
    if format.lower() == "geojson":
        return TabularParser.to_geojson(floats)
    return floats


@router.get("/slice", response_model=DataSliceResponse, tags=["Slicing Engine"])
async def get_ocean_slice(
    site_id: str = Query(..., description="ID of the active site"),
    variable: str = Query("temp", description="Physical variable (temp, sal, cur, oxy)"),
    depth: float = Query(0.0, description="Target depth in meters (0 to maxDepth)"),
    time_offset: float = Query(0.0, description="Time offset in hours (-48 to +48)"),
    resolution: int = Query(48, description="Target horizontal grid resolution (e.g. 48, 64)")
):
    """
    Slices a 2D horizontal depth slice (lat x lon) at given depth Z and time T.
    """
    try:
        return SliceEngine.get_slice(
            site_id=site_id,
            variable=variable,
            depth=depth,
            time_offset=time_offset,
            resolution=resolution
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Slicing error: {str(e)}")


@router.post("/upload", response_model=UploadResponse, tags=["Ingestion Pipeline"])
async def upload_dataset(
    file: UploadFile = File(..., description="Oceanographic dataset (.nc, .nc4, .csv, .txt)")
):
    """
    Handles multi-part upload of gridded NetCDF models or tabular Argo/CTD datasets.
    Extracts spatial bounding boxes, vertical levels, registers a new Study Site,
    and returns rich dataset metadata.
    """
    # Sanitize filename against directory traversal
    raw_filename = file.filename or "uploaded_dataset"
    filename = os.path.basename(raw_filename)
    ext = filename.lower().split(".")[-1]

    if ext not in ["nc", "nc4", "csv", "txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Sagar Drishti accepts NetCDF (.nc, .nc4) and Tabular (.csv, .txt) files."
        )

    # Enforce 50 MB boundary without loading unbounded requests into memory
    MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
    CHUNK_SIZE = 1024 * 1024  # 1 MB

    try:
        chunks = []
        total_read = 0
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Uploaded file exceeds maximum limit of 50 MB (received >{total_read // (1024 * 1024)} MB). Upload rejected."
                )
            chunks.append(chunk)

        content = b"".join(chunks)

        if ext in ["nc", "nc4"]:
            ds, meta = NetCDFParser.parse_dataset_from_bytes(content, filename)
            site_record = meta["site_record"]
            SiteRegistry.register_uploaded_site(site_record, raw_dataset=ds)

            return UploadResponse(
                site_id=site_record.id,
                name=site_record.name,
                filename=filename,
                file_type="NetCDF Gridded Ocean Model",
                bounding_box=meta["bbox"],
                depth_range={"min": 0.0, "max": site_record.maxDepth},
                variables=site_record.variables or ["temp", "sal", "cur", "oxy"],
                float_count=0,
                site_record=site_record,
                message=f"Successfully ingested NetCDF dataset '{filename}'. Spatial bounding box and depth levels computed."
            )
        else:
            df, meta = TabularParser.parse_csv_from_bytes(content, filename)
            site_record = meta["site_record"]
            SiteRegistry.register_uploaded_site(site_record, raw_dataset=df)

            return UploadResponse(
                site_id=site_record.id,
                name=site_record.name,
                filename=filename,
                file_type="Tabular Argo / CTD Profiles",
                bounding_box=meta["bbox"],
                depth_range=meta["depth_range"],
                variables=site_record.variables or ["temp", "sal", "cur", "oxy"],
                float_count=meta["float_count"],
                site_record=site_record,
                message=f"Successfully ingested tabular dataset '{filename}' with {meta['float_count']} profiles.",
                custom_observation=meta.get("custom_observation"),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse oceanographic dataset '{filename}': {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse, tags=["SagarBot AI"])
async def chat_with_sagarbot(req: ChatRequest, request: Request, response: Response):
    """
    Backend proxy for SagarBot with live viewport grounding context injection,
    Gemini 2.5 Flash / multi-provider LLM support, and sliding-window rate limiting.
    Delegates to SagarBotService (Phase 4.7 conversational decision intelligence).
    """
    # Determine client IP for rate limiting
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown_client"

    # Enforce rate limiter
    allowed, retry_after, remaining = chat_rate_limiter.is_allowed(client_ip)
    response.headers["X-RateLimit-Limit"] = str(chat_rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. SagarBot accepts up to {chat_rate_limiter.max_requests} queries per minute. Please wait {retry_after} second(s) before sending another query.",
            headers={"Retry-After": str(retry_after), "X-RateLimit-Reset": str(retry_after)}
        )

    try:
        from ..services.sagarbot_service import SagarBotService
        return await SagarBotService.process_chat(req)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SagarBot processing error: {str(e)}"
        )


@router.post("/sagarbot/chat", response_model=ChatResponse, tags=["SagarBot AI"])
async def sagarbot_conversational_chat(req: ChatRequest, request: Request, response: Response):
    """
    Dedicated Phase 4.7 SagarBot Conversational Decision Intelligence endpoint.
    Returns deterministic risk blocks, available physical explainability drivers,
    historical comparisons, and structured auditable evidence.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown_client"

    allowed, retry_after, remaining = chat_rate_limiter.is_allowed(client_ip)
    response.headers["X-RateLimit-Limit"] = str(chat_rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. SagarBot accepts up to {chat_rate_limiter.max_requests} queries per minute. Please wait {retry_after} second(s) before sending another query.",
            headers={"Retry-After": str(retry_after), "X-RateLimit-Reset": str(retry_after)}
        )

    try:
        from ..services.sagarbot_service import SagarBotService
        return await SagarBotService.process_chat(req)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SagarBot processing error: {str(e)}"
        )



# ==============================================================================
# HISTORICAL COPERNICUS MARINE DATA PIPELINE
# ==============================================================================

@router.get("/historical/status", response_model=HistoricalStatusResponse, tags=["Historical Copernicus"])
async def get_historical_status():
    """
    Returns current Copernicus Marine historical dataset status, date bounds,
    variables, file size, and post-download validation report.
    """
    return CopernicusService.get_status()


@router.post("/historical/ingest", response_model=HistoricalIngestResponse, tags=["Historical Copernicus"])
async def trigger_historical_ingest():
    """
    Triggers asynchronous, non-blocking ingestion of 2-year daily Copernicus Marine
    dataset (cmems_mod_glo_phy_my_0.083deg_P1D-m).
    Credentials MUST be set in environment variables (COPERNICUSMARINE_SERVICE_USERNAME / PASSWORD)
    or pre-configured via 'copernicusmarine login'.
    """
    started, message, task_id = CopernicusService.trigger_ingest_background()
    return HistoricalIngestResponse(
        status="in_progress" if started else "busy",
        task_id=task_id,
        message=message,
        started_at=datetime.now(timezone.utc).isoformat()
    )


@router.get("/historical/slice", response_model=HistoricalSliceResponse, tags=["Historical Copernicus"])
async def get_historical_slice(
    date: str = Query(..., description="Calendar date in ISO format YYYY-MM-DD (e.g. 2024-07-01)"),
    variable: str = Query("temp", description="Physical variable (temp, sal, cur, cur_u, cur_v, ssh, mld)"),
    resolution: int = Query(48, description="Target horizontal grid resolution (e.g. 48, 64)"),
):
    """
    Queries a 2D horizontal depth slice from the Copernicus 2-year daily reanalysis.
    Lazy disk-backed read: only extracts the requested date and variable.
    Does NOT silently fall back to synthetic data.
    """
    try:
        return CopernicusService.get_slice(
            date_str=date,
            variable=variable,
            resolution=resolution
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical dataset not available: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Historical slicing error: {str(e)}"
        )


@router.get("/historical/point", response_model=HistoricalPointResponse, tags=["Historical Copernicus"])
async def get_historical_point(
    date: str = Query(..., description="Calendar date in ISO format YYYY-MM-DD"),
    lat: float = Query(..., description="Latitude in degrees (0 to 25)"),
    lon: float = Query(..., description="Longitude in degrees (50 to 100)"),
    variable: str = Query("temp", description="Physical variable (temp, sal, cur, cur_u, cur_v, ssh, mld)"),
):
    """
    Queries a point observation at a specific geographic coordinate and date from the Copernicus dataset.
    """
    try:
        return CopernicusService.get_point(
            date_str=date,
            lat=lat,
            lon=lon,
            variable=variable
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical dataset not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Historical point query error: {str(e)}"
        )


@router.get("/historical/compare", response_model=HistoricalComparisonResponse, tags=["Historical Copernicus"])
async def compare_historical_dynamics(
    date: str = Query(..., description="Target comparison date in ISO format YYYY-MM-DD"),
    mode: str = Query("point", description="Evaluation spatial mode: 'point' or 'region'"),
    lat: Optional[float] = Query(None, description="Latitude for point mode"),
    lon: Optional[float] = Query(None, description="Longitude for point mode"),
    site_id: Optional[str] = Query(None, description="Study site ID for region mode (e.g. 'bob')"),
    min_lat: Optional[float] = Query(None, description="Bounding box min latitude for custom region"),
    max_lat: Optional[float] = Query(None, description="Bounding box max latitude for custom region"),
    min_lon: Optional[float] = Query(None, description="Bounding box min longitude for custom region"),
    max_lon: Optional[float] = Query(None, description="Bounding box max longitude for custom region"),
    variable: Optional[str] = Query("all", description="Target variable or 'all' for complete feature vector"),
):
    """
    Historical Feature & Anomaly Engine:
    - Decouples 7, 14, 30-day rolling window dynamics from full 2-year climatological baseline.
    - Observation count derived dynamically from NetCDF time coordinate.
    - Computes current value, baseline mean/std, absolute anomaly, z-score, window net change, and linear trend.
    - Supports localized grid-cell extraction (point) and regional spatial aggregation (region).
    - Produces a unified feature vector ready for downstream Phase 3 ML risk modeling.
    """
    try:
        return HistoricalFeatureEngine.compute_comparison(
            date_str=date,
            mode=mode,
            lat=lat,
            lon=lon,
            site_id=site_id,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            variable=variable,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical dataset not available: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Historical comparison error: {str(e)}"
        )


@router.post("/historical/export-features", response_model=TrainingDatasetExportResponse, tags=["Historical Copernicus"])
async def export_historical_features(
    mode: str = Query("point", description="'point' or 'region'"),
    lat: float = Query(17.8, description="Latitude for point mode"),
    lon: float = Query(88.2, description="Longitude for point mode"),
    site_id: Optional[str] = Query(None, description="Study site ID for region mode"),
):
    """
    Rolls across all eligible days (t >= 29) of the available 2-year Copernicus dataset,
    computes the unified rolling feature vector for every day, and exports:
    1. features.parquet: Columnar feature table for ML training
    2. features_metadata.json: Baseline metadata, feature catalogue, and provenance
    Raw Copernicus NetCDF is preserved completely unchanged.
    """
    try:
        return HistoricalFeatureEngine.export_training_dataset(
            mode=mode,
            lat=lat,
            lon=lon,
            site_id=site_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical dataset not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature export error: {str(e)}"
        )


# ==============================================================================
# PHASE 3: HISTORICAL EVENTS & LABELED ML DATASET ENDPOINTS
# ==============================================================================

@router.get("/historical/events", response_model=List[HistoricalEvent], tags=["Historical Events"])
async def list_historical_events(
    event_type: Optional[EventType] = Query(None, description="Filter by event type"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """
    Returns normalized, authoritative historical disaster/event records
    from IMD RSMC New Delhi and INCOIS (cyclones, depressions, swell surges, marine heatwaves).
    """
    return HistoricalEventStore.get_events(
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/historical/events/quality-report", response_model=DataQualityReport, tags=["Historical Events"])
async def get_event_data_quality_report(
    mode: str = Query("region", description="'point' or 'region'"),
    site_id: Optional[str] = Query("bob", description="Study site ID"),
    lat: Optional[float] = Query(17.8, description="Latitude"),
    lon: Optional[float] = Query(88.2, description="Longitude"),
):
    """
    Generates a full data-quality audit report for historical event matching:
    total events, spatial/temporal matches, unmatched events with reasons,
    positive vs. clean-negative counts, discrete lead distributions, and provenance.
    """
    try:
        res = SpatialTemporalMatcher.generate_labeled_dataset(
            mode=mode,
            site_id=site_id,
            lat=lat,
            lon=lon,
        )
        return res.data_quality_report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating data quality report: {str(e)}"
        )


@router.post("/historical/events/generate-labeled-dataset", response_model=LabeledDatasetExportResponse, tags=["Historical Events"])
async def generate_labeled_dataset(
    mode: str = Query("region", description="'point' or 'region'"),
    site_id: Optional[str] = Query("bob", description="Study site ID"),
    lat: Optional[float] = Query(17.8, description="Latitude"),
    lon: Optional[float] = Query(88.2, description="Longitude"),
):
    """
    Connects Phase 2 rolling ocean features with real authoritative disaster records,
    generating:
    1. data/historical/labeled_features.parquet
    2. data/historical/labeled_features_metadata.json
    Leaves raw Copernicus NetCDF and Phase 2 features.parquet untouched.
    """
    try:
        return SpatialTemporalMatcher.generate_labeled_dataset(
            mode=mode,
            site_id=site_id,
            lat=lat,
            lon=lon,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Required dataset not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate labeled dataset: {str(e)}"
        )


# ==============================================================================
# PHASE 4.6: MACHINE LEARNING RISK PREDICTION API
# ==============================================================================

@router.get("/prediction/status", tags=["ML Prediction"])
async def get_prediction_model_status():
    """
    Returns the readiness status, version, and trained tasks of the ML prediction models.
    """
    return PredictionService.get_model_status()


@router.post("/prediction/predict", response_model=PredictionResponse, tags=["ML Prediction"])
async def predict_ocean_risk(req: PredictionRequest):
    """
    Live Machine Learning Risk Inference:
    Evaluates ocean observations at date T, extracts rolling Phase 2 physical features,
    and returns:
    - risk level (normal, advisory, alert)
    - calibrated event probability
    - early-warning horizon (0d active to 3d early-warning)
    - multi-granularity explainability (top contributing features, ocean variable & time window importance)
    """
    try:
        return PredictionService.predict(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact or dataset unavailable: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@router.post("/prediction/predict-custom", response_model=PredictionResponse, tags=["ML Prediction"])
async def predict_custom_ocean_risk(req: CustomObservationRequest):
    """
    Live Machine Learning Risk Inference for Custom In-Situ / Single-Day Observation:
    Accepts physical measurements (thetao, so, uo, vo, zos, mlotst) for a specific date and location,
    computes rolling context from Copernicus reanalysis history, overrides current features with
    user-supplied values, and runs the frozen 101-feature risk model.
    """
    try:
        return PredictionService.predict_with_custom_observation(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact or dataset unavailable: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom prediction error: {str(e)}"
        )



