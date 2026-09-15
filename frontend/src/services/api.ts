import { SitePhysics, IngestionMetadata, ViewportContext, PredictionResponse, PredictionRequest, CustomPredictionRequest } from '../types/ocean';
import { ClientCsvParser } from './clientCsvParser';
import { ClientNetcdfParser } from './clientNetcdfParser';
import { Ocean } from './syntheticOcean';

const API_BASE = '/api';

export const OceanAPI = {
  /**
   * Fetches all active study sites from the backend (with fallback to baseline sites).
   */
  async getSites(): Promise<SitePhysics[]> {
    try {
      const res = await fetch(`${API_BASE}/sites`);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      // Backend not running, use synthetic baseline
    }
    return Ocean.listSites();
  },

  /**
   * Uploads an ocean dataset (.nc, .nc4, .csv, .txt) via multi-part upload.
   * If backend is offline, parses directly on the client side without failure.
   */
  async uploadDataset(file: File): Promise<IngestionMetadata> {
    const ext = file.name.toLowerCase().split('.').pop() || '';
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        return {
          site_id: data.site_id,
          name: data.name,
          filename: data.filename,
          file_type: data.file_type,
          file_size: `${(file.size / 1024).toFixed(1)} KB`,
          bounding_box: data.bounding_box,
          depth_range: data.depth_range,
          variables: data.variables,
          float_count: data.float_count,
          site_record: data.site_record,
          custom_observation: data.custom_observation,
        };
      }
    } catch (e) {
      console.warn('Backend upload unreachable, falling back to client-side parser:', e);
    }

    // Client-side fallback parsing
    if (ext === 'csv' || ext === 'txt') {
      const text = await file.text();
      const { site, floats, customObservation } = ClientCsvParser.parseCsv(text, file.name);
      return {
        site_id: site.id,
        name: site.name,
        filename: file.name,
        file_type: 'Tabular Argo / CTD Profiles (Client-Parsed)',
        file_size: `${(file.size / 1024).toFixed(1)} KB`,
        bounding_box: site.bbox || { min_lat: site.lat - 2, max_lat: site.lat + 2, min_lon: site.lon - 2, max_lon: site.lon + 2 },
        depth_range: { min: 0, max: site.maxDepth },
        variables: site.variables || ['temp', 'sal', 'cur', 'oxy'],
        float_count: floats.length,
        site_record: site,
        custom_observation: customObservation || site.custom_observation || null,
      };
    } else {
      const buffer = await file.arrayBuffer();
      const { site } = ClientNetcdfParser.parseNetcdfBuffer(buffer, file.name);
      return {
        site_id: site.id,
        name: site.name,
        filename: file.name,
        file_type: 'NetCDF Gridded Ocean Model (Client-Parsed)',
        file_size: `${(file.size / 1024).toFixed(1)} KB`,
        bounding_box: site.bbox || { min_lat: site.lat - 3, max_lat: site.lat + 3, min_lon: site.lon - 3, max_lon: site.lon + 3 },
        depth_range: { min: 0, max: site.maxDepth },
        variables: site.variables || ['temp', 'sal', 'cur', 'oxy'],
        float_count: 0,
        site_record: site,
      };
    }
  },

  /**
   * Sends user question and live viewport grounding context to SagarBot.
   */
  async chatWithBot(
    message: string,
    context: ViewportContext,
    apiKey?: string,
    provider?: string
  ): Promise<{ reply: string; provider: string; intent?: string; evidence?: Record<string, any> }> {
    try {
      const res = await fetch(`${API_BASE}/sagarbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          context,
          api_key: apiKey || undefined,
          provider: provider || 'offline',
        }),
      });

      if (res.ok) {
        const data = await res.json();
        return {
          reply: data.reply,
          provider: data.provider,
          intent: data.intent,
          evidence: data.evidence,
        };
      }

      if (res.status === 429) {
        const err = await res.json().catch(() => ({ detail: 'Rate limit exceeded.' }));
        throw new Error(err.detail || 'Rate limit exceeded. Please wait a few seconds before asking another question.');
      }
    } catch (e: any) {
      if (e?.message?.includes('Rate limit')) {
        throw e;
      }
      console.warn('Backend chat unreachable, falling back to local ocean expert reasoning:', e);
    }


    // Client fallback expert engine
    const dStr = context.current_depth;
    const siteName = context.active_site;
    const val = context.current_value;
    const v = context.variable;

    let fallbackReply = `At ${dStr} in ${siteName}: Active ${v} = ${val}. The vertical stratification here shows steep gradients in the upper 300 m. Select any variable to view the profile curve.`;
    const t = message.toLowerCase();

    if (t.includes('temp')) {
      fallbackReply = `Temperature Analysis at ${siteName}: Currently ${val} at depth ${dStr}. You are observing the permanent thermocline transition where solar irradiance attenuates rapidly.`;
    } else if (t.includes('sal')) {
      fallbackReply = `Salinity Structure at ${siteName}: Salinity reads ${val} at depth ${dStr}. Halocline barrier layers prevent vertical turbulent mixing and trap heat in the upper water column.`;
    } else if (t.includes('oxy') || t.includes('o2')) {
      fallbackReply = `Dissolved Oxygen at ${siteName}: O₂ is ${val} at ${dStr}. Mid-depth respiration by bacteria creates an oxygen minimum zone between 200m and 800m.`;
    } else if (t.includes('cur') || t.includes('flow')) {
      fallbackReply = `Currents at ${siteName}: Velocity is ${val} at ${dStr}. The flow field follows geostrophic balance with mesoscale eddy rings advecting water masses.`;
    }

    return {
      reply: fallbackReply,
      provider: 'SagarBot Oceanographic Physics Engine (Client Offline)',
    };
  },
};

export const HistoricalAPI = {
  /**
   * Fetches Copernicus Marine historical dataset status and validation report.
   */
  async getStatus() {
    const res = await fetch(`${API_BASE}/historical/status`);
    if (!res.ok) {
      throw new Error(`Failed to fetch historical status: ${res.statusText}`);
    }
    return await res.json();
  },

  /**
   * Triggers background Copernicus Marine reanalysis ingestion.
   */
  async triggerIngest() {
    const res = await fetch(`${API_BASE}/historical/ingest`, {
      method: 'POST',
    });
    if (!res.ok) {
      throw new Error(`Failed to trigger ingestion: ${res.statusText}`);
    }
    return await res.json();
  },

  /**
   * Slices 2D horizontal field from the Copernicus 2-year daily reanalysis.
   */
  async getSlice(date: string, variable: string = 'temp', resolution: number = 48) {
    const params = new URLSearchParams({
      date,
      variable,
      resolution: String(resolution),
    });
    const res = await fetch(`${API_BASE}/historical/slice?${params.toString()}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to fetch historical slice');
    }
    return await res.json();
  },

  /**
   * Queries point observation at specific geographic coordinate and date.
   */
  async getPoint(date: string, lat: number, lon: number, variable: string = 'temp') {
    const params = new URLSearchParams({
      date,
      lat: String(lat),
      lon: String(lon),
      variable,
    });
    const res = await fetch(`${API_BASE}/historical/point?${params.toString()}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to fetch historical point');
    }
    return await res.json();
  },

  /**
   * Compares recent rolling window behavior (7d, 14d, 30d) against full 2-year climatological baseline.
   */
  async compare(params: {
    date: string;
    mode?: 'point' | 'region';
    lat?: number;
    lon?: number;
    site_id?: string;
    min_lat?: number;
    max_lat?: number;
    min_lon?: number;
    max_lon?: number;
    variable?: string;
  }) {
    const q = new URLSearchParams();
    q.set('date', params.date);
    if (params.mode) q.set('mode', params.mode);
    if (params.lat !== undefined) q.set('lat', String(params.lat));
    if (params.lon !== undefined) q.set('lon', String(params.lon));
    if (params.site_id) q.set('site_id', params.site_id);
    if (params.min_lat !== undefined) q.set('min_lat', String(params.min_lat));
    if (params.max_lat !== undefined) q.set('max_lat', String(params.max_lat));
    if (params.min_lon !== undefined) q.set('min_lon', String(params.min_lon));
    if (params.max_lon !== undefined) q.set('max_lon', String(params.max_lon));
    if (params.variable) q.set('variable', params.variable);

    const res = await fetch(`${API_BASE}/historical/compare?${q.toString()}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to fetch historical comparison');
    }
    return await res.json();
  },
};

export const PredictionAPI = {
  /**
   * Fetches status of ML models and prediction engine.
   */
  async getStatus(): Promise<{ is_ready: boolean; metadata?: Record<string, any>; message?: string }> {
    const res = await fetch(`${API_BASE}/prediction/status`);
    if (!res.ok) throw new Error('Failed to fetch prediction status');
    return await res.json();
  },

  /**
   * Generates live ML risk prediction with multi-granularity explainability.
   */
  async predict(params: PredictionRequest): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/prediction/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        date: params.date,
        mode: params.mode || (params.site_id || params.min_lat !== undefined ? 'region' : 'point'),
        lat: params.lat,
        lon: params.lon,
        site_id: params.site_id,
        min_lat: params.min_lat,
        max_lat: params.max_lat,
        min_lon: params.min_lon,
        max_lon: params.max_lon,
        horizon_days: params.horizon_days ?? 3,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Prediction request failed');
    }
    return await res.json();
  },

  /**
   * Generates live ML risk prediction for custom in-situ observation.
   */
  async predictCustom(params: CustomPredictionRequest): Promise<PredictionResponse> {
    const res = await fetch(`${API_BASE}/prediction/predict-custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        date: params.date,
        lat: params.lat,
        lon: params.lon,
        horizon_days: params.horizon_days ?? 3,
        thetao: params.thetao,
        so: params.so,
        uo: params.uo,
        vo: params.vo,
        zos: params.zos,
        mlotst: params.mlotst,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Custom prediction request failed');
    }
    return await res.json();
  },
};

export interface ForwardPredictionReq {
  system_id: string;
  forecast_origin_timestamp: string;
  cyclone_history?: any[];
  observations?: any[];
  features?: Record<string, number>;
  demo_scenario_id?: string;
  observation_timestamp?: string;
  data_availability_timestamp?: string;
  ocean_source_timestamp?: string;
  ocean_source_available_timestamp?: string;
  ocean_age_hours?: number;
  data_provenance_class?: string;
  atmos_source_id?: string;
  candidate_target_fixes?: any[];
}

export interface ThreatAssessment {
  threat_level: string;
  short_threat_level: string;
  intensity_band: string;
  threat_basis: string;
  symbol: string;
  color: string;
  description: string;
  predicted_vmax_kt: number | null;
  calibrated_probability: number | null;
  probability_label: string;
  risk_basis: string;
  risk_level: string | null;
}

export interface ForwardPredictionRes {
  status: string;
  system_id: string;
  origin: string;
  valid_time: string;
  predicted_vmax_24h: number | null;
  raw_predicted_vmax_24h?: number | null;
  reported_predicted_vmax_24h?: number | null;
  clipping_applied?: boolean;
  clipping_bounds?: number[];
  model_version: string;
  model_hash: string;
  feature_contract_hash: string;
  preprocessing_hash: string;
  input_dataset_source?: string;
  synthetic_fixture_label?: string | null;
  data_provenance_class?: string;
  atmos_source_id?: string;
  forecast_created_at?: string;
  ocean_temporal_resolution?: string;
  causal_firewall: string;
  feature_completeness: number;
  missing_features: string[];
  latitude?: number | null;
  longitude?: number | null;
  scientific_disclaimer: string;
  evaluation_status: string;
  evaluation_mode?: string;
  target_info?: any;
  warnings: string[];
  // Threat & Risk Interpretation Layer
  threat_assessment?: ThreatAssessment;
  intensity_threat_level?: string;
  intensity_band?: string;
  predicted_vmax_kt?: number | null;
}

export interface ValidationRes {
  is_valid: boolean;
  system_id: string;
  forecast_origin_timestamp: string;
  checks: Array<{ check: string; status: string; detail: string }>;
  available_features: string[];
  missing_features: string[];
  feature_completeness: number;
  can_predict: boolean;
  message: string;
}

export interface DemoScenarioItem {
  scenario_id: string;
  system_id: string;
  system_name: string;
  description: string;
  forecast_origin_timestamp: string;
  latitude: number;
  longitude: number;
  basin: string;
  fix_count: number;
  latest_observed_vmax: number;
}

export const ForecastAPI = {
  async getDemoScenarios(): Promise<DemoScenarioItem[]> {
    const res = await fetch(`${API_BASE}/forecast/demo-scenarios`);
    if (!res.ok) {
      throw new Error('Failed to fetch prospective demo scenarios');
    }
    return await res.json();
  },

  async validateInputs(req: ForwardPredictionReq): Promise<ValidationRes> {
    const res = await fetch(`${API_BASE}/forecast/validate-inputs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Validation request failed');
    }
    return await res.json();
  },

  async predictForward(req: ForwardPredictionReq): Promise<ForwardPredictionRes> {
    const res = await fetch(`${API_BASE}/forecast/cyclone-intensity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Forward prediction request failed');
    }
    return await res.json();
  },

  async getHistory(limit: number = 50): Promise<any[]> {
    const res = await fetch(`${API_BASE}/forecast/history?limit=${limit}`);
    if (!res.ok) {
      return [];
    }
    return await res.json();
  },
};

