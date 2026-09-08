from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router as api_router
from .services.site_registry import SiteRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize baseline data on startup
    SiteRegistry.initialize_baseline_floats()
    yield


app = FastAPI(
    title="Sagar Drishti — Oceanographic Ingestion & Slicing API",
    description="Production-grade high-performance ocean data ingestion and 3D visualization engine.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router, prefix="/api")


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
