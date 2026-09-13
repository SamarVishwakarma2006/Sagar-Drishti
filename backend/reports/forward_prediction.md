# Forward Prediction / Prospective Inference Engine — Architectural Specification

## 1. Overview
The Forward Prediction Engine enables Sagar-Drishti to execute prospective inference on new cyclone observations post-dating the historical training partition (post-2026-06-23) up to forecast origin $T$.

## 2. Core Constraints
1. **Model Immutability**: Uses the frozen research model `research/cyclone_intensity/models/final_intensity_model.joblib` (SHA-256: `3abf49bc7b167c96f91425cf97d12afd606bb025f5310e774648cd52a3a1423f`).
2. **Dual Causal Firewall**: Requires $t_{\text{obs}} \le T_{\text{origin}}$ and $t_{\text{avail}} \le T_{\text{origin}}$. Filesystem `mtime` is explicitly rejected.
3. **29-Feature Contract**: Assembles kinematic lookbacks, environmental atmospheric variables, ocean variables, and radial contrast terms causally.
4. **Append-Only Provenance**: All inferences are written to `forecast_log.parquet`.
5. **Mandatory Scientific Disclaimer**:
   *"Forward inference uses the frozen research model on user-provided or newly available observations. This forecast is not itself proof of prospective generalization; future outcomes are evaluated separately through the prospective validation framework."*

## 3. Endpoints
- `POST /api/forecast/cyclone-intensity`
- `POST /api/forecast/validate-inputs`
- `GET /api/forecast/demo-scenarios`
- `GET /api/forecast/history`

## 4. UI Integration
- Accessible through the **FORECAST MODE** radar toggle in the top navigation bar.
- Highlights real-time causal firewall checks, category classifications, provenance metadata, and 3D Cesium beacon visualization.
