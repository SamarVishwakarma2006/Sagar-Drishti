import { SitePhysics, IngestionMetadata, ViewportContext, ChatResponse } from '../types/ocean';
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
        };
      }
    } catch (e) {
      console.warn('Backend upload unreachable, falling back to client-side parser:', e);
    }

    // Client-side fallback parsing
    if (ext === 'csv' || ext === 'txt') {
      const text = await file.text();
      const { site, floats } = ClientCsvParser.parseCsv(text, file.name);
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
  ): Promise<{ reply: string; provider: string }> {
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          context,
          api_key: apiKey || undefined,
          provider: provider || 'gemini',
        }),
      });

      if (res.ok) {
        const data = await res.json();
        return { reply: data.reply, provider: data.provider };
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
