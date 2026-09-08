export type VariableKey = 'temp' | 'sal' | 'cur' | 'oxy';
export type PaletteKey = 'thermal' | 'haline' | 'speed' | 'oxy' | 'viridis' | 'deep' | 'curl';
export type Provenance = 'observed' | 'interpolated' | 'modelled' | 'predicted';
export type Phase = 'boot' | 'globe' | 'diving' | 'underwater' | 'ascending';
export type LLMProvider = 'offline' | 'gemini' | 'groq' | 'openai';

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
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'bot';
  text: string;
  timestamp: string;
  provider?: string;
}

export interface ChatResponse {
  reply: string;
  provider: string;
  grounded_context?: Record<string, any>;
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
}
