import { useSyncExternalStore } from 'react';
import {
  AppState,
  SitePhysics,
  VariableKey,
  PaletteKey,
  Selection,
  IngestionMetadata,
  ColorbarSettings,
  LLMProvider,
} from '../types/ocean';
import { SITES } from '../services/syntheticOcean';
import type { GlobeEngine } from '../engines/GlobeEngine';
import type { UnderwaterEngine } from '../engines/UnderwaterEngine';

const STORAGE_API_KEY = 'sagar_drishti_api_key';
const STORAGE_PROVIDER = 'sagar_drishti_llm_provider';
const envApiKey = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_GEMINI_API_KEY) || '';

const savedApiKey = typeof window !== 'undefined' ? (localStorage.getItem(STORAGE_API_KEY) || envApiKey) : envApiKey;
const savedProvider = (typeof window !== 'undefined' ? (localStorage.getItem(STORAGE_PROVIDER) as LLMProvider) : 'gemini') || 'gemini';

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
  setApiKey(key: string, provider: LLMProvider = 'gemini') {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_API_KEY, key);
      localStorage.setItem(STORAGE_PROVIDER, provider);
    }
    state = { ...state, apiKey: key, llmProvider: provider };
    listeners.forEach((listener) => listener());
  },

  // Ingestion and study sites actions
  addUploadedSite(meta: IngestionMetadata) {
    const exists = state.uploadedSites.some((s) => s.id === meta.site_id);
    const updated = exists
      ? state.uploadedSites.map((s) => (s.id === meta.site_id ? meta.site_record : s))
      : [meta.site_record, ...state.uploadedSites];

    state = {
      ...state,
      uploadedSites: updated,
      activeUpload: meta,
      customDataMode: true,
      site: meta.site_record,
      selection: null,
      depth: Math.min(60, meta.site_record.maxDepth * 0.2),
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
      site: isCurrent ? (updated[0] || SITES[0]) : state.site,
      selection: null,
    };
    listeners.forEach((listener) => listener());
  },
};

export const useApp = () => useSyncExternalStore(store.subscribe, store.get);

export const toast = (msg: string) => {
  store.set({ toast: { msg, id: Date.now() + Math.random() } });
};

export const globeRegistry: { g: GlobeEngine | null } = { g: null };
export const uwRegistry: { e: UnderwaterEngine | null } = { e: null };
