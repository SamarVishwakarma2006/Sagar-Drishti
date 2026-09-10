export type VariableKey = 'temp' | 'sal' | 'cur' | 'oxy' | 'ssh' | 'mld';
export type Provenance = 'observed' | 'interpolated' | 'modelled' | 'predicted' | 'historical';
export type PaletteKey = 'thermal' | 'haline' | 'speed' | 'oxy' | 'viridis' | 'deep' | 'curl';
export type Phase = 'boot' | 'globe' | 'diving' | 'underwater' | 'ascending';
export type LLMProvider = 'offline';

export type Selection =
  | { kind: 'float'; id: string }
  | { kind: 'sample'; lat: number; lon: number; depth: number };

export interface LatLon {
  lat: number;
  lon: number;
}

export interface BoundingBox {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

export interface ProfilePoint {
  depth: number;
  temperature: number;
  salinity: number;
  currentSpeed: number;
  currentDir: number;
  oxygen: number;
  ssh?: number;
  mld?: number;
}

export interface ProfileResult {
  zmax: number;
  points: ProfilePoint[];
}

export interface Sample {
  depth: number;
  temperature: number;
  salinity: number;
  currentSpeed: number;
  currentDir: number;
  oxygen: number;
  ssh?: number;
  mld?: number;
}

export interface ProvInfo {
  provenance: Provenance;
  source: string;
  label: string;
}

export interface DataPoint extends Sample {
  lat: number;
  lon: number;
  prov: ProvInfo;
  timestamp: string;
}

export interface FloatRecord {
  id: string;
  platform: string;
  lat: number;
  lon: number;
  cycle: number;
  lastReportOffset: number;
  parkingDepth: number;
  profile: ProfileResult;
  source: string;
}

export interface CurrentVector {
  u: number;
  v: number;
  w: number;
  speed: number;
  dirDeg: number;
}

export interface SitePhysics extends LatLon {
  id: string;
  name: string;
  region: string;
  maxDepth: number;
  blurb: string;
  ts: number;
  ss: number;
  td: number;
  sd: number;
  mld: number;
  tw: number;
  salMaxAmp: number;
  salMaxZ: number;
  flow: number;
  eddy: number;
  bgU: number;
  bgV: number;
  o2s: number;
  o2d: number;
  o2z0: number;
  o2minAmp: number;
  o2minZ: number;
  o2minW: number;
  bbox?: BoundingBox;
  variables?: string[];
  isCustom?: boolean;
  sourceType?: string;
  floats?: FloatRecord[];
}

export interface ColorbarSettings {
  palette: PaletteKey;
  customMin: number | null;
  customMax: number | null;
  isLogScale: boolean;
  opacity: number;
  verticalExaggeration: number;
}

export interface ViewportContext {
  active_site: string;
  coordinates: { lat: number; lon: number };
  current_depth: string;
  variable: string;
  current_value: string;
  time_offset: string;
  nearby_floats: string[];
  custom_data: boolean;
  historical_date?: string;
  historical_mode?: boolean;
  active_prediction?: any;
  history?: Array<{ role: 'user' | 'bot'; text: string }>;
  session_id?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'bot';
  text: string;
  timestamp: string;
  provider?: string;
  intent?: string;
  evidence?: Record<string, any>;
}

export interface ChatResponse {
  reply: string;
  provider: string;
  grounded_context?: Record<string, any>;
  intent?: string;
  evidence?: Record<string, any>;
}


export interface CustomObservation {
  date: string;
  lat: number;
  lon: number;
  thetao?: number;
  so?: number;
  uo?: number;
  vo?: number;
  zos?: number;
  mlotst?: number;
}

export interface IngestionMetadata {
  site_id: string;
  name: string;
  filename: string;
  file_type: string;
  file_size?: string;
  bounding_box: BoundingBox;
  depth_range: { min: number; max: number };
  variables: string[];
  float_count: number;
  site_record: SitePhysics;
  custom_observation?: CustomObservation | null;
}

export interface HistoricalStatus {
  status: 'idle' | 'in_progress' | 'ready' | 'failed';
  dataset_id: string;
  is_ready: boolean;
  file_path?: string | null;
  file_size_human?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  total_days?: number | null;
  bounding_box?: BoundingBox | null;
  depth_m?: number | null;
  variables: string[];
  message: string;
  validation?: Record<string, any> | null;
  provenance?: Record<string, any> | null;
}

export interface HistoricalSlice {
  dataset_id: string;
  date: string;
  variable: string;
  standard_name: string;
  depth: number;
  unit: string;
  lats: number[];
  lons: number[];
  values: (number | null)[][];
  min_val: number;
  max_val: number;
  shape: [number, number];
  provenance: string;
}

export interface BaselineMetadata {
  baseline_type: string;
  baseline_start: string;
  baseline_end: string;
  baseline_observations: number;
}

export interface WindowMetrics {
  window_days: number;
  window_start: string;
  window_end: string;
  mean: number;
  delta: number;
  trend_per_day: number;
  min: number;
  max: number;
}

export interface VariableAnomaly {
  variable: string;
  standard_name: string;
  unit: string;
  current_value: number;
  baseline_mean: number;
  baseline_std: number;
  baseline_min: number;
  baseline_max: number;
  absolute_anomaly: number;
  z_score: number;
  percentage_anomaly?: number | null;
  windows: Record<string, WindowMetrics>;
}

export interface HistoricalComparisonResponse {
  dataset_id: string;
  date: string;
  mode: 'point' | 'region';
  location: Record<string, any>;
  baseline_metadata: BaselineMetadata;
  variables: Record<string, VariableAnomaly>;
  feature_vector: Record<string, number>;
  data_quality: Record<string, any>;
  provenance: string;
}

export interface AppState {
  phase: Phase;
  site: SitePhysics | null;
  variable: VariableKey;
  depth: number;
  timeOffset: number;
  playing: boolean;
  selection: Selection | null;
  toast: { msg: string; id: number } | null;
  customDataMode: boolean;
  activeUpload: IngestionMetadata | null;
  uploadedSites: SitePhysics[];
  colorbar: ColorbarSettings;
  llmProvider: LLMProvider;
  apiKey: string;
  historicalMode: boolean;
  historicalDate: string;
  historicalStatus: HistoricalStatus | null;
  activePrediction?: PredictionResponse | null;
  customObservation?: CustomObservation | null;
  activeHazardZone?: any | null;
}

export interface FeatureAttribution {
  feature: string;
  importance: number;
  value?: number;
  contribution?: number;
  description?: string;
}

export interface PhysicalDriversGroup {
  ocean_variables: Record<string, number>;
  temporal_scales: Record<string, number>;
}

export interface HumanReadableExplanation {
  what: string;
  where: string;
  when: string;
  why: string;
}

export interface PredictionExplainability {
  top_features: FeatureAttribution[];
  ocean_variable_importance: Record<string, number>;
  time_window_importance: Record<string, number>;
  human_readable?: HumanReadableExplanation;
}

export interface ObservedOceanState {
  sea_surface_temperature_c?: number;
  sea_surface_salinity_psu?: number;
  surface_current_speed_ms?: number;
  sea_surface_height_m?: number;
  mixed_layer_depth_m?: number;
  temp_30d_baseline_mean_c?: number;
  mld_30d_baseline_mean_m?: number;
  provenance?: string;
  [key: string]: any;
}

export interface PredictedRiskState {
  warning_level: 'NO_ALERT' | 'WATCH' | 'HIGH_ALERT';
  prediction_status: 'normal' | 'advisory' | 'alert';
  model_probability: number;
  probability_display: string;
  frozen_threshold: number;
  forecast_horizon_days: number;
  predicted_event_type: string;
  is_calibrated: boolean;
  note?: string;
}

export interface HistoricalEventContext {
  event_id?: string | null;
  event_name?: string;
  event_type?: string;
  event_dates?: string;
  distance_days?: number | null;
  is_active_date?: boolean;
  affected_region?: string;
  severity?: string;
  bbox?: BoundingBox | null;
  centroid_lat?: number | null;
  centroid_lon?: number | null;
  track_coordinates?: Array<{ date: string; lat: number; lon: number; intensity_kts?: number }>;
  similarity_score?: number | null;
  matched_basin?: string;
  parameter_comparison?: Record<string, { observed: number; historical: number; delta: number; unit: string; match_pct: number }>;
  analog_assessment?: string;
  note?: string;
}

export interface PredictionDataQuality {
  requested_date?: string;
  dataset_date_range?: [string, string];
  spatial_region?: string;
  valid_spatial_coverage_pct?: number;
  missing_feature_count?: number;
  model_feature_compatibility?: string;
  source_dataset?: string;
  model_version?: string;
  is_available: boolean;
  reason?: string;
}

export interface PredictionResponse {
  status: 'success' | 'insufficient_data' | 'error';
  date: string;
  mode: 'point' | 'region';
  location: Record<string, any>;
  horizon_days: number;
  target?: string;
  prediction: 'normal' | 'advisory' | 'alert';
  warning_level: 'NO_ALERT' | 'WATCH' | 'HIGH_ALERT';
  probability: number;
  model_estimated_probability?: number;
  threshold: number;
  alert_threshold?: number;
  probability_display?: string;
  event_type: string;
  is_calibrated: boolean;
  model_name: string;
  model_version: string;
  prediction_timestamp?: string;
  explainability: PredictionExplainability;
  top_features: FeatureAttribution[];
  physical_drivers: PhysicalDriversGroup;
  observed_state: ObservedOceanState;
  predicted_state: PredictedRiskState;
  historical_context: HistoricalEventContext;
  data_quality: PredictionDataQuality;
  limitations: string[];
  message: string;
}

export interface PredictionRequest {
  date: string;
  mode?: 'point' | 'region';
  lat?: number;
  lon?: number;
  site_id?: string;
  min_lat?: number;
  max_lat?: number;
  min_lon?: number;
  max_lon?: number;
  horizon_days?: number;
}

export interface CustomPredictionRequest {
  date: string;
  lat: number;
  lon: number;
  horizon_days?: number;
  thetao?: number;
  so?: number;
  uo?: number;
  vo?: number;
  zos?: number;
  mlotst?: number;
}

