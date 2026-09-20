import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router as api_router
from .services.site_registry import SiteRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize baseline data on startup
    SiteRegistry.initialize_baseline_floats()
    # Validate ML model artifacts on startup (Phase 4.6.5)
    try:
        from .services.prediction_service import PredictionService
        PredictionService.validate_model_artifacts()
    except Exception as e:
        import logging
        logging.getLogger("sagar_drishti").warning(f"ML Model startup check: {e}")
    yield



app = FastAPI(
    title="Sagar Drishti — Oceanographic Ingestion & Slicing API",
    description="Production-grade high-performance ocean data ingestion and 3D visualization engine.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration: allow local development origins, production Vercel frontend, and environment-configured origins
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://sagar-drishti-two.vercel.app",
]

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    env_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
    allowed_origins = list(dict.fromkeys(default_origins + env_origins))
else:
    allowed_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/sagar-drishti[a-zA-Z0-9-]*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router, prefix="/api")

# Mount V2.3 Shadow Research API router (Research Only - Zero Operational Authority)
from .api.shadow_v2_3_routes import router as shadow_v2_3_router
app.include_router(shadow_v2_3_router, prefix="/shadow/v2.3")
app.include_router(shadow_v2_3_router, prefix="/api/shadow/v2.3")


@app.get("/")
def root():
    return {
        "title": "Sagar Drishti Oceanographic Ingestion Engine",
        "docs": "/docs",
        "health": "/api/health",
        "sites": "/api/sites"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
