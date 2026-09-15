import { useSyncExternalStore } from 'react';
import {
  AppState,
  IngestionMetadata,
  ColorbarSettings,
  LLMProvider,
} from '../types/ocean';
import { SITES } from '../services/syntheticOcean';
import type { GlobeEngine } from '../engines/GlobeEngine';
import type { UnderwaterEngine } from '../engines/UnderwaterEngine';

const STORAGE_API_KEY = 'sagar_drishti_api_key';
const STORAGE_PROVIDER = 'sagar_drishti_llm_provider';

const savedApiKey = '';
const savedProvider: LLMProvider = 'offline';

const initialColorbar: ColorbarSettings = {
  palette: 'thermal',
  customMin: null,
  customMax: null,
  isLogScale: false,
  opacity: 0.55,
  verticalExaggeration: 1.0,
};

let state: AppState = {
  phase: 'boot',
  site: null,
  variable: 'temp',
  depth: 60,
  timeOffset: 0,
  playing: false,
  selection: null,
  toast: null,
  customDataMode: false,
  activeUpload: null,
  uploadedSites: [],
  colorbar: initialColorbar,
  llmProvider: savedProvider,
  apiKey: savedApiKey,
  historicalMode: false,
  historicalDate: '2024-06-24',
  historicalStatus: null,
  activePrediction: null,
  customObservation: null,
  activeHazardZone: null,
};

const listeners = new Set<() => void>();

export const store = {
  get: () => state,
  set(patch: Partial<AppState>) {
    state = { ...state, ...patch };
    listeners.forEach((listener) => listener());
  },
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },

  // Colorbar actions
  setColorbar(patch: Partial<ColorbarSettings>) {
    state = {
      ...state,
      colorbar: { ...state.colorbar, ...patch },
    };
    listeners.forEach((listener) => listener());
  },

  // API Key & Provider configuration
  setApiKey(key: string, provider: LLMProvider = 'offline') {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_API_KEY, key);
      localStorage.setItem(STORAGE_PROVIDER, provider);
    }
    state = { ...state, apiKey: key, llmProvider: provider };
    listeners.forEach((listener) => listener());
  },

  // Ingestion and study sites actions
  addUploadedSite(meta: IngestionMetadata) {
    const obs = meta.custom_observation || meta.site_record.custom_observation || null;
    const exists = state.uploadedSites.some((s) => s.id === meta.site_id);
    const updatedSite = {
      ...meta.site_record,
      custom_observation: obs,
    };
    const updated = exists
      ? state.uploadedSites.map((s) => (s.id === meta.site_id ? updatedSite : s))
      : [updatedSite, ...state.uploadedSites];

    state = {
      ...state,
      uploadedSites: updated,
      activeUpload: meta,
      customDataMode: true,
      customObservation: obs,
      historicalMode: false,
      historicalDate: obs?.date || state.historicalDate,
      site: updatedSite,
      selection: null,
      depth: Math.min(60, updatedSite.maxDepth * 0.2),
    };
    listeners.forEach((listener) => listener());
  },

  removeUploadedSite(siteId: string) {
    const updated = state.uploadedSites.filter((s) => s.id !== siteId);
    const isCurrent = state.site?.id === siteId;
    state = {
      ...state,
      uploadedSites: updated,
      activeUpload: null,
      customDataMode: updated.length > 0,
      customObservation: updated.length > 0 ? state.customObservation : null,
      site: isCurrent ? (updated[0] || SITES[0]) : state.site,
      selection: null,
    };
    listeners.forEach((listener) => listener());
  },

  setCustomObservation(obs: any) {
    state = { ...state, customObservation: obs };
    listeners.forEach((listener) => listener());
  },

  // Historical Copernicus mode actions
  setHistoricalMode(active: boolean) {
    state = { ...state, historicalMode: active };
    listeners.forEach((listener) => listener());
  },

  setHistoricalDate(date: string) {
    state = { ...state, historicalDate: date };
    listeners.forEach((listener) => listener());
  },

  setHistoricalStatus(status: any) {
    state = { ...state, historicalStatus: status };
    listeners.forEach((listener) => listener());
  },

  setActivePrediction(pred: any) {
    state = { ...state, activePrediction: pred };
    listeners.forEach((listener) => listener());
  },

  setActiveHazardZone(zone: any) {
    state = { ...state, activeHazardZone: zone };
    listeners.forEach((listener) => listener());
  },
};

export const useApp = () => useSyncExternalStore(store.subscribe, store.get);

export const toast = (msg: string) => {
  store.set({ toast: { msg, id: Date.now() + Math.random() } });
};

export const globeRegistry: { g: GlobeEngine | null } = { g: null };
export const uwRegistry: { e: UnderwaterEngine | null } = { e: null };
