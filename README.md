# 🌊 SAGAR DRISHTI (सागर दृष्टि)
### 3D Immersive Ocean Visualization & High-Performance Geospatial Ingestion Engine

---

## 🌟 Overview
**Sagar Drishti** is a production-grade full-stack oceanographic observatory platform. It decouples high-performance Python data ingestion (`xarray`, `netCDF4`, `pandas`) from an immersive WebGL visualization frontend powered by **CesiumJS** (planetary globe) and **Three.js** (volumetric underwater particle flows & layer slicing).

---

## 🏗️ Architecture

```
sagar-drishti/
├── backend/                             # High-Performance FastAPI Ingestion Engine
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py             # /api/upload, /api/sites, /api/slice, /api/chat
│   │   ├── parsers/
│   │   │   ├── netcdf_parser.py         # xarray & netCDF4 coordinate/variable extraction
│   │   │   └── tabular_parser.py        # Argo/CTD CSV header detection & GeoJSON conversion
│   │   ├── services/
│   │   │   ├── site_registry.py         # INCOIS baseline sites & dynamic upload registry
│   │   │   ├── slice_engine.py          # 2D/3D horizontal & vertical array slicing
│   │   │   └── ocean_ai.py              # SagarBot AI proxy & grounding physics engine
│   │   ├── models/
│   │   │   └── schemas.py               # Pydantic V2 typed data models
│   │   └── main.py                      # FastAPI application & CORS configuration
│   ├── sample_data/
│   │   ├── generate_samples.py          # Synthetic .nc and .csv dataset generator
│   │   ├── sample_argo_profiles.csv     # Test tabular Argo profile dataset
│   │   └── sample_ocean_grid.nc         # Test 4D NetCDF ocean model grid
│   ├── tests/
│   │   └── test_backend.py              # Python unittest test suite
│   ├── requirements.txt                 # Backend dependencies
│   └── run_backend.py                   # One-click backend startup script
│
├── frontend/                            # Modular React + Vite + TypeScript Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── TopBar.tsx               # Header with variable switchers & AI config
│   │   │   ├── SiteExplorer.tsx         # Dynamic study sites sidebar (auto-flies to bounds)
│   │   │   ├── SitePanel.tsx            # Floating site statistics & DIVE action
│   │   │   ├── DataIngestionPanel.tsx   # Drag-and-drop dropzone for .nc and .csv
│   │   │   ├── Inspector.tsx            # Argo profile depth charts ($0$ to $Z_{max}$) & Copy JSON
│   │   │   ├── ColorbarControls.tsx     # Palette switcher, min/max bounds, opacity, 3D exag
│   │   │   ├── SagarBot.tsx             # Context-grounded AI chatbot drawer
│   │   │   ├── DepthControl.tsx         # Vertical depth slider with float parking pips
│   │   │   ├── Timeline.tsx             # -48h to +48h temporal scrubber & playback
│   │   │   ├── CompassHud.tsx           # 3D yaw compass heading & reset button
│   │   │   ├── CoordReadout.tsx         # Live coordinate mouse tracking
│   │   │   ├── Veil.tsx                 # Dive / ascend transition overlay
│   │   │   ├── Boot.tsx                 # Loading splash screen
│   │   │   ├── GlobeViewer.tsx          # CesiumJS 3D Globe component
│   │   │   └── UnderwaterViewer.tsx     # Three.js volumetric rendering component
│   │   ├── engines/
│   │   │   ├── GlobeEngine.ts           # CesiumJS camera, entities, and flight paths
│   │   │   └── UnderwaterEngine.ts      # Three.js particles, flow lines, snow, floor
│   │   ├── services/
│   │   │   ├── api.ts                   # Backend REST client with offline fallback
│   │   │   ├── syntheticOcean.ts        # INCOIS baseline physics & cmocean colormaps
│   │   │   ├── clientNetcdfParser.ts    # Browser NetCDF parser fallback
│   │   │   └── clientCsvParser.ts       # Browser CSV/TXT Argo parser fallback
│   │   ├── store/
│   │   │   └── oceanStore.ts            # Central reactive store & external subscriptions
│   │   ├── types/
│   │   │   └── ocean.ts                 # TypeScript type definitions
│   │   ├── App.tsx                      # Master application coordinator
│   │   ├── main.tsx                     # React root
│   │   └── index.css                    # Design tokens & glassmorphic styling
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
```

---

## 🚀 Quick Start Guide

### 1. Start Backend (FastAPI)
```powershell
# 1. Navigate to backend
cd backend

# 2. (Optional) Install Python dependencies
pip install -r requirements.txt

# 3. Generate sample datasets
python sample_data/generate_samples.py

# 4. Run automated test suite
python -m unittest discover -s tests -t .

# 5. Start FastAPI server (runs on http://localhost:8000)
python run_backend.py
```
API Documentation will be live at `http://localhost:8000/docs`.

### 2. Start Frontend (React + Vite)
```powershell
# 1. Navigate to frontend
cd frontend

# 2. Install NPM dependencies
npm.cmd install

# 3. Start development server (runs on http://localhost:3000)
npm.cmd run dev
```

---

## 🔬 Core Features & Workflows

### 1. Ingestion Pipeline & Dynamic Study Sites
- Drop `.nc` or `.csv` files into the **Ingestion Pipeline** dropzone.
- The backend automatically detects coordinates (`lat`, `lon`, `depth`, `time`), variable channels (`temperature`, `salinity`, `u`, `v`, `oxygen`), and generates spatial bounding boxes.
- The **Site Explorer** immediately lists the uploaded dataset.
- Clicking the dataset automatically flies the Cesium camera directly to its bounding box and recalibrates the depth slider to the detected $Z_{max}$.

### 2. 3D Volumetric Flow & Colorbar Controls
- Transition from planetary orbit into true 3D underwater exploration using the **DIVE** button.
- Particles react dynamically to velocity vectors ($u, v, w$).
- Open the **Colorbar Controls** panel to switch palettes (`thermal`, `haline`, `speed`, `oxy`, `viridis`, `deep`, `curl`), set custom min/max numerical bounds, toggle logarithmic scaling, and adjust vertical depth exaggeration ($1\times$ to $25\times$).

### 3. Context-Aware SagarBot AI
- Real-time telemetry grounding: Every question packages the live 3D depth, active variable, coordinate bounds, and nearby Argo float IDs.
- Seamless multi-LLM integration powered by Google Gemini, Groq, or OpenAI via environment variables.
- Automatic graceful fallback to a deterministic oceanographic physics engine when offline.

---

## ⚖️ License
MIT License. All rights reserved.
