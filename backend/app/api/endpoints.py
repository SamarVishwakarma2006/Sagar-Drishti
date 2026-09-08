import os
import io
import tempfile
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from ..models.schemas import (
    SitePhysics,
    FloatRecord,
    DataSliceResponse,
    ChatRequest,
    ChatResponse,
    UploadResponse,
)
from ..parsers.netcdf_parser import NetCDFParser
from ..parsers.tabular_parser import TabularParser
from ..services.site_registry import SiteRegistry
from ..services.slice_engine import SliceEngine
from ..services.ocean_ai import OceanAIService
from ..services.rate_limiter import chat_rate_limiter

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
    filename = file.filename or "uploaded_dataset"
    ext = filename.lower().split(".")[-1]

    if ext not in ["nc", "nc4", "csv", "txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Sagar Drishti accepts NetCDF (.nc, .nc4) and Tabular (.csv, .txt) files."
        )

    try:
        content = await file.read()

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
                message=f"Successfully ingested tabular dataset '{filename}' with {meta['float_count']} profiles."
            )

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
        return await OceanAIService.process_chat(req)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SagarBot processing error: {str(e)}"
        )
