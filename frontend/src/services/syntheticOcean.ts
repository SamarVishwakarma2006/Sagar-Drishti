import {
  SitePhysics,
  VariableKey,
  PaletteKey,
  Provenance,
  ProvInfo,
  ProfileResult,
  ProfilePoint,
  Sample,
  DataPoint,
  FloatRecord,
  CurrentVector,
} from '../types/ocean';

/* ============================================================================
 * MATH & UTILITIES
 * ==========================================================================*/
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const mod = (a: number, n: number) => ((a % n) + n) % n;
export const pad2 = (n: number) => String(n).padStart(2, '0');
export const isMobile = typeof window !== 'undefined'
  ? (window.matchMedia('(pointer: coarse)').matches || window.innerWidth < 768)
  : false;

export function hashStr(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const fmtLat = (v: number, d = 2) => `${Math.abs(v).toFixed(d)}°${v >= 0 ? 'N' : 'S'}`;
export const fmtLon = (v: number, d = 2) => `${Math.abs(v).toFixed(d)}°${v >= 0 ? 'E' : 'W'}`;
export const fmtDepth = (d: number) =>
  d >= 1000 ? `${(d / 1000).toFixed(d >= 10000 ? 1 : 2)} km` : `${Math.round(d)} m`;

/* ============================================================================
 * COLORMAP PALETTES (cmocean, Thyng et al. 2016 + viridis, deep, curl)
 * ==========================================================================*/
export const PALETTES: Record<PaletteKey, { label: string; ref: string; stops: [number, string][] }> = {
  thermal: {
    label: 'Thermal',
    ref: 'cmocean·thermal',
    stops: [[0, '#1c0f45'], [0.22, '#551d6e'], [0.42, '#a22c6d'], [0.62, '#e04e63'], [0.80, '#f5925a'], [1, '#f6e8a4']],
  },
  haline: {
    label: 'Haline',
    ref: 'cmocean·haline',
    stops: [[0, '#0e1f38'], [0.28, '#1d5087'], [0.52, '#2e93bd'], [0.75, '#7fd0d2'], [1, '#e3f5ef']],
  },
  speed: {
    label: 'Speed',
    ref: 'cmocean·speed',
    stops: [[0, '#e9f2f6'], [0.25, '#9fbcd3'], [0.50, '#6b83b6'], [0.72, '#7f5ba6'], [0.88, '#c04a70'], [1, '#ef5a54']],
  },
  oxy: {
    label: 'Dissolved O₂',
    ref: 'cmocean·oxy',
    stops: [[0, '#f2e4dc'], [0.25, '#c9a7cd'], [0.50, '#7b8abc'], [0.75, '#39679a'], [1, '#132b45']],
  },
  viridis: {
    label: 'Viridis',
    ref: 'matplotlib·viridis',
    stops: [[0, '#440154'], [0.25, '#3b528b'], [0.5, '#21918c'], [0.75, '#5ec962'], [1, '#fde725']],
  },
  deep: {
    label: 'Deep',
    ref: 'cmocean·deep',
    stops: [[0, '#fbe3d5'], [0.25, '#c57876'], [0.5, '#853268'], [0.75, '#3f1747'], [1, '#06020c']],
  },
  curl: {
    label: 'Curl (Diverging)',
    ref: 'cmocean·curl',
    stops: [[0, '#1d5087'], [0.25, '#7fd0d2'], [0.5, '#ffffff'], [0.75, '#f5925a'], [1, '#a22c6d']],
  },
};

export const CMAPS: Record<VariableKey, { label: string; ref: string; stops: [number, string][] }> = {
  temp: PALETTES.thermal,
  sal: PALETTES.haline,
  cur: PALETTES.speed,
  oxy: PALETTES.oxy,
  ssh: PALETTES.curl,
  mld: PALETTES.deep,
};

export const VAR_COLORS: Record<VariableKey, string> = {
  temp: '#f2a15f',
  sal: '#7fd0d2',
  cur: '#a9c0d6',
  oxy: '#c9a7cd',
  ssh: '#56d4e2',
  mld: '#e2b45a',
};

export const FIELD: Record<VariableKey, 'temperature' | 'salinity' | 'currentSpeed' | 'oxygen' | 'ssh' | 'mld'> = {
  temp: 'temperature',
  sal: 'salinity',
  cur: 'currentSpeed',
  oxy: 'oxygen',
  ssh: 'ssh',
  mld: 'mld',
};

export const VARIABLES: { key: VariableKey; label: string; unit: string; icon: string; digits: number }[] = [
  { key: 'temp', label: 'TEMP', unit: '°C', icon: 'Thermometer', digits: 2 },
  { key: 'sal', label: 'SAL', unit: 'PSU', icon: 'Droplets', digits: 2 },
  { key: 'cur', label: 'CURRENT', unit: 'm/s', icon: 'Waves', digits: 2 },
  { key: 'oxy', label: 'O₂', unit: 'µmol/kg', icon: 'Wind', digits: 1 },
];

export const PROV: Record<Provenance, { label: string; color: string; dashed?: boolean }> = {
  observed: { label: 'Observed', color: '#56d4e2' },
  interpolated: { label: 'Interpolated', color: '#9db0ba', dashed: true },
  modelled: { label: 'Modelled', color: '#8fb0c9' },
  predicted: { label: 'Predicted', color: '#e2b45a' },
  historical: { label: 'Historical Reanalysis', color: '#f59e0b' },
};

export const PROV_SRC: Record<Provenance, string> = {
  observed: 'in-situ (Argo / CTD) · delayed-mode QC',
  interpolated: 'analysis L4 · multi-mission altimetry (0.25°)',
  modelled: 'INDOMOD reanalysis · INCOIS / MoES',
  predicted: 'NCUM-O coupled forecast · 48 h horizon',
  historical: 'Copernicus Marine Global Ocean Physics Reanalysis (0.083° daily)',
};

export function hex2rgb(h: string): [number, number, number] {
  return [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ];
}

export function sampleStops(stops: [number, string][], t: number): [number, number, number] {
  t = clamp(t, 0, 1);
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const f = (t - stops[i - 1][0]) / Math.max(1e-6, stops[i][0] - stops[i - 1][0]);
      const a = hex2rgb(stops[i - 1][1]);
      const b = hex2rgb(stops[i][1]);
      return [lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f)];
    }
  }
  return hex2rgb(stops[stops.length - 1][1]);
}

export function buildLUT(stops: [number, string][]): Float32Array {
  const lut = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = sampleStops(stops, i / 255);
    lut[i * 3] = c[0] / 255;
    lut[i * 3 + 1] = c[1] / 255;
    lut[i * 3 + 2] = c[2] / 255;
  }
  return lut;
}

export const cssGradient = (stops: [number, string][]) =>
  `linear-gradient(90deg, ${stops.map(([p, c]) => `${c} ${Math.round(p * 100)}%`).join(', ')})`;

/* ============================================================================
 * BASELINE STUDY SITES
 * ==========================================================================*/
export const REF_MS = Date.UTC(2025, 2, 21, 6, 0, 0);

export const provFor = (t: number): ProvInfo =>
  t < -2
    ? { provenance: 'modelled', label: 'Reanalysis', source: 'INDOMOD reanalysis · INCOIS/MoES — assimilates Argo profiles & altimetry' }
    : t <= 2
    ? { provenance: 'interpolated', label: 'Analysis', source: 'L4 blended analysis · multi-mission altimetry + in-situ gridding (0.25°)' }
    : { provenance: 'predicted', label: 'Forecast', source: 'NCUM-O coupled forecast · 48 h horizon · 0.25° grid' };

export const timeISO = (t: number) => new Date(REF_MS + t * 3600000).toISOString().slice(0, 19) + 'Z';
export const timeUTC = (t: number) => {
  const d = new Date(REF_MS + t * 3600000);
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())} UTC`;
};

export const SITES: SitePhysics[] = [
  {
    id: 'bob',
    name: 'Bay of Bengal — Fresh Plume',
    region: 'N. Indian Ocean',
    lat: 17.8,
    lon: 88.2,
    maxDepth: 2600,
    blurb: 'Ganga–Brahmaputra runoff caps the bay with a low-salinity lid. The strong halocline traps heat and feeds one of Earth’s most intense oxygen-minimum zones.',
    ts: 29.2, ss: 31.6, td: 2.8, sd: 34.86, mld: 28, tw: 34,
    salMaxAmp: 0.55, salMaxZ: 120,
    flow: 0.45, eddy: 190, bgU: 0.18, bgV: 0.05,
    o2s: 188, o2d: 168, o2z0: 80, o2minAmp: 160, o2minZ: 380, o2minW: 250,
  },
  {
    id: 'eqio',
    name: 'Equatorial Indian Ocean',
    region: 'Indian Ocean',
    lat: 0.5,
    lon: 80.5,
    maxDepth: 4300,
    blurb: 'Wyrtki jets race east along the equator; a shallow thermocline makes this the fastest-responding region of the Indian Ocean to climate modes.',
    ts: 29.3, ss: 34.9, td: 3.4, sd: 34.72, mld: 40, tw: 45,
    salMaxAmp: 0.12, salMaxZ: 150,
    flow: 0.55, eddy: 240, bgU: 0.32, bgV: 0,
    o2s: 182, o2d: 170, o2z0: 90, o2minAmp: 120, o2minZ: 520, o2minW: 300,
  },
  {
    id: 'aras',
    name: 'Arabian Sea — OMZ Core',
    region: 'N. Indian Ocean',
    lat: 16.5,
    lon: 67.5,
    maxDepth: 3700,
    blurb: 'Intense monsoon-driven productivity and poor ventilation produce one of the most severe oxygen-minimum zones in the global ocean.',
    ts: 27.4, ss: 36.5, td: 2.6, sd: 34.8, mld: 55, tw: 55,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 0.40, eddy: 220, bgU: 0.12, bgV: -0.08,
    o2s: 178, o2d: 170, o2z0: 70, o2minAmp: 176, o2minZ: 260, o2minW: 220,
  },
  {
    id: 'agul',
    name: 'Agulhas Current',
    region: 'SW Indian Ocean',
    lat: -34.2,
    lon: 26.8,
    maxDepth: 3100,
    blurb: 'The strongest western boundary current of the Indian Ocean, famous for its retroflection eddies that leak warm water into the Atlantic.',
    ts: 22.6, ss: 35.5, td: 3.6, sd: 34.8, mld: 65, tw: 70,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 1.50, eddy: 260, bgU: -1.0, bgV: -0.8,
    o2s: 196, o2d: 185, o2z0: 90, o2minAmp: 70, o2minZ: 600, o2minW: 350,
  },
  {
    id: 'guls',
    name: 'Gulf Stream',
    region: 'N. Atlantic',
    lat: 36.4,
    lon: -73.6,
    maxDepth: 3300,
    blurb: 'A swift, meandering western boundary current transporting warm subtropical water northward and shedding energetic rings.',
    ts: 23.6, ss: 36.1, td: 3.8, sd: 34.94, mld: 70, tw: 80,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 1.30, eddy: 280, bgU: 0.25, bgV: 1.05,
    o2s: 210, o2d: 190, o2z0: 110, o2minAmp: 95, o2minZ: 650, o2minW: 380,
  },
  {
    id: 'kuro',
    name: 'Kuroshio Extension',
    region: 'N.W. Pacific',
    lat: 35.2,
    lon: 145.4,
    maxDepth: 5500,
    blurb: 'Where the Kuroshio frees itself from the coast — a basin-scale eddy nursery and a sharp front between subtropical and subpolar water.',
    ts: 21.8, ss: 34.7, td: 2.2, sd: 34.5, mld: 90, tw: 95,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 1.15, eddy: 300, bgU: 0.9, bgV: 0.2,
    o2s: 220, o2d: 160, o2z0: 100, o2minAmp: 130, o2minZ: 800, o2minW: 420,
  },
  {
    id: 'drak',
    name: 'Drake Passage',
    region: 'Southern Ocean',
    lat: -58.5,
    lon: -63.0,
    maxDepth: 3900,
    blurb: 'The Antarctic Circumpolar Current squeezes through the tightest gap in the world ocean — cold, well-ventilated, relentlessly stormy.',
    ts: 3.4, ss: 33.9, td: 1.2, sd: 34.72, mld: 120, tw: 160,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 0.80, eddy: 320, bgU: 0.55, bgV: 0.3,
    o2s: 330, o2d: 200, o2z0: 150, o2minAmp: 35, o2minZ: 700, o2minW: 500,
  },
  {
    id: 'chal',
    name: 'Challenger Deep',
    region: 'Mariana Trench',
    lat: 11.37,
    lon: 142.59,
    maxDepth: 10935,
    blurb: 'The deepest seafloor on Earth. Hadal waters below 6,000 m are near-freezing, near-still, and pressed by more than a thousand atmospheres.',
    ts: 29.1, ss: 34.55, td: 1.4, sd: 34.68, mld: 50, tw: 60,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 0.06, eddy: 500, bgU: 0.02, bgV: 0.015,
    o2s: 195, o2d: 152, o2z0: 110, o2minAmp: 55, o2minZ: 750, o2minW: 450,
  },
  {
    id: 'peru',
    name: 'Peru Upwelling',
    region: 'S.E. Pacific',
    lat: -12.5,
    lon: -80.5,
    maxDepth: 3900,
    blurb: 'Trade winds drive cold, nutrient-rich water to the surface — the engine of the Humboldt Current system and the El Niño heartbeat.',
    ts: 18.4, ss: 34.95, td: 2.4, sd: 34.6, mld: 25, tw: 30,
    salMaxAmp: 0, salMaxZ: 0,
    flow: 0.35, eddy: 180, bgU: 0.25, bgV: 0.1,
    o2s: 180, o2d: 150, o2z0: 60, o2minAmp: 165, o2minZ: 200, o2minW: 180,
  },
];

/* ============================================================================
 * SYNTHETIC OCEAN DATA SERVICE
 * ==========================================================================*/
export const Ocean = {
  _rangeMemo: new Map<string, { min: number; max: number }>(),
  _floatMemo: new Map<string, FloatRecord[]>(),
  _phaseMemo: new Map<string, number[]>(),

  listSites() {
    return SITES;
  },

  latLonToOffset(site: SitePhysics, lat: number, lon: number) {
    return {
      x: (lon - site.lon) * 111320 * Math.cos((site.lat * Math.PI) / 180),
      y: (lat - site.lat) * 110540,
    };
  },

  offsetToLatLon(site: SitePhysics, x: number, y: number) {
    return {
      lat: site.lat + y / 110540,
      lon: site.lon + x / (111320 * Math.cos((site.lat * Math.PI) / 180)),
    };
  },

  siteForPoint(lat: number, lon: number): SitePhysics {
    let best = SITES[0];
    let bd = 1e9;
    SITES.forEach((s) => {
      const d = (s.lat - lat) * (s.lat - lat) + (s.lon - lon) * (s.lon - lon);
      if (d < bd) {
        bd = d;
        best = s;
      }
    });
    return {
      ...best,
      id: `drop:${lat.toFixed(2)},${lon.toFixed(2)}`,
      name: 'User Ocean Drop Point',
      region: `${fmtLat(lat)} ${fmtLon(lon)}`,
      lat,
      lon,
      ts: clamp(best.ts + (best.lat - lat) * 0.34, -1.8, 31.5),
      ss: clamp(best.ss + (lat - best.lat) * 0.012, 30.5, 37.5),
      flow: best.flow * 0.85,
      bgU: best.bgU * 0.7,
      bgV: best.bgV * 0.7,
      blurb: 'User-defined study point — physics inherited from the nearest climatic regime and adjusted for latitude.',
      isCustom: true,
    };
  },

  tempAt(site: SitePhysics, z: number, t: number, jit = 0) {
    const f = 1 / (1 + Math.exp((z - (site.mld + 55)) / site.tw));
    const di = 0.28 * Math.exp(-z / 28) * Math.sin(((mod(6 + t, 24) / 24) * 6.2832) - 2.4);
    return site.td + (site.ts - site.td) * f + di + jit;
  },

  salAt(site: SitePhysics, z: number, jit = 0) {
    const b = site.sd + (site.ss - site.sd) * Math.exp(-z / 95);
    const sub = site.salMaxAmp > 0 ? site.salMaxAmp * Math.exp(-Math.pow((z - site.salMaxZ) / 85, 2)) : 0;
    return b + sub + jit;
  },

  oxyAt(site: SitePhysics, z: number, jit = 0) {
    const s = site.o2d + (site.o2s - site.o2d) * Math.exp(-z / site.o2z0);
    const omz = site.o2minAmp * Math.exp(-Math.pow((z - site.o2minZ) / site.o2minW, 2));
    return Math.max(2, s - omz + jit);
  },

  _phases(site: SitePhysics) {
    let m = this._phaseMemo.get(site.id);
    if (m) return m;
    const r = mulberry32(hashStr(site.id));
    m = [r() * 6.28, r() * 6.28, r() * 6.28, r() * 6.28];
    this._phaseMemo.set(site.id, m);
    return m;
  },

  vectorAt(site: SitePhysics, x: number, y: number, z: number, t: number): CurrentVector {
    const P = this._phases(site);
    const kx = 6.2832 / site.eddy;
    const ky = 6.2832 / (site.eddy * 0.72);
    const A = site.flow * (0.16 + 0.84 * Math.exp(-z / 300)) * (site.eddy / 6.2832);
    const w1 = t * 0.10 + P[0];
    const w2 = t * 0.07 + P[1];
    const w3 = t * 0.13 + P[2];
    const w4 = t * 0.05 + P[3];
    const u1 = -A * Math.sin(kx * x + w1) * ky * Math.sin(ky * y + w2);
    const v1 = -A * Math.cos(ky * y + w2) * kx * Math.cos(kx * x + w1);
    const u2 = 0.42 * A * Math.sin(kx * 1.9 * x - w3 + P[0]) * ky * 2.3 * Math.cos(ky * 2.3 * y + w4 + P[1]);
    const v2 = -0.42 * A * kx * 1.9 * Math.cos(kx * 1.9 * x - w3 + P[0]) * Math.sin(ky * 2.3 * y + w4 + P[1]);
    const u = u1 + u2 + site.bgU;
    const v = v1 + v2 + site.bgV;
    const w = 0.05 * A * Math.sin(kx * 1.3 * x + ky * 0.6 * y + t * 0.2 + P[2]);
    return {
      u,
      v,
      w,
      speed: Math.hypot(u, v),
      dirDeg: mod(Math.atan2(u, v) * 57.2958, 360),
    };
  },

  variableAt(site: SitePhysics, x: number, y: number, z: number, t: number, key: VariableKey, jit = 0) {
    if (key === 'cur') return this.vectorAt(site, x, y, z, t).speed;
    if (key === 'temp') return this.tempAt(site, z, t, jit * 0.6);
    if (key === 'sal') return this.salAt(site, z, jit * 0.05);
    return this.oxyAt(site, z, jit * 3);
  },

  profileAt(site: SitePhysics, lat: number, lon: number, t: number): ProfileResult {
    const o = this.latLonToOffset(site, lat, lon);
    const zmax = Math.min(2000, site.maxDepth);
    const mT = 0.35 * Math.sin(o.x * 0.0021) * Math.cos(o.y * 0.0017);
    const pts: ProfilePoint[] = [];
    for (let i = 0; i <= 40; i++) {
      const z = i <= 8 ? i * 10 : 80 + ((i - 8) * (zmax - 80)) / 32;
      const dm = 0.7 + 0.3 * Math.exp(-z / 300);
      const v = this.vectorAt(site, o.x, o.y, z, t);
      pts.push({
        depth: z,
        temperature: this.tempAt(site, z, t) + mT * dm,
        salinity: this.salAt(site, z),
        currentSpeed: v.speed,
        currentDir: v.dirDeg,
        oxygen: this.oxyAt(site, z),
      });
    }
    return { zmax, points: pts };
  },

  sampleAt(site: SitePhysics, lat: number, lon: number, depth: number, t: number): DataPoint {
    const pr = this.profileAt(site, lat, lon, t);
    const p = pr.points.reduce((a, b) => (Math.abs(b.depth - depth) < Math.abs(a.depth - depth) ? b : a));
    const o = this.latLonToOffset(site, lat, lon);
    return {
      depth,
      temperature: p.temperature,
      salinity: p.salinity,
      currentSpeed: p.currentSpeed,
      currentDir: this.vectorAt(site, o.x, o.y, depth, t).dirDeg,
      oxygen: p.oxygen,
      lat,
      lon,
      prov: provFor(t),
      timestamp: timeISO(t),
    };
  },

  sampleFromFloat(f: FloatRecord, depth: number): Sample {
    const pts = f.profile.points;
    let i = 0;
    while (i < pts.length - 2 && pts[i + 1].depth < depth) i++;
    const a = pts[i];
    const b = pts[i + 1] || pts[i];
    const L = clamp((depth - a.depth) / Math.max(1e-6, b.depth - a.depth), 0, 1);
    return {
      depth,
      temperature: lerp(a.temperature, b.temperature, L),
      salinity: lerp(a.salinity, b.salinity, L),
      currentSpeed: lerp(a.currentSpeed, b.currentSpeed, L),
      currentDir: a.currentDir,
      oxygen: lerp(a.oxygen, b.oxygen, L),
    };
  },

  floatsAt(site: SitePhysics): FloatRecord[] {
    if (site.floats && site.floats.length > 0) return site.floats;
    const m = this._floatMemo.get(site.id);
    if (m) return m;
    const r = mulberry32(hashStr(site.id) + 13);
    const park = [26, 85, 1000, 1000, 2000];
    const out: FloatRecord[] = [];
    for (let i = 0; i < 5; i++) {
      const lat = site.lat + (r() - 0.5) * 1.5;
      const lon = site.lon + (r() - 0.5) * 1.7;
      const pd = clamp(park[i] + (r() - 0.5) * 60, 15, site.maxDepth - 15);
      const last = -(1 + r() * 66);
      out.push({
        id: `290${Math.floor(1500 + r() * 900)}`,
        platform: 'Argo float · APEX (INCOIS)',
        lat,
        lon,
        cycle: Math.floor(70 + r() * 140),
        lastReportOffset: last,
        parkingDepth: pd,
        profile: this.profileAt(site, lat, lon, last),
        source: 'Argo GDAC · INCOIS / Coriolis — delayed-mode QC',
      });
    }
    this._floatMemo.set(site.id, out);
    return out;
  },

  rangeAt(site: SitePhysics, key: VariableKey): { min: number; max: number } {
    const k = site.id + ':' + key;
    const m = this._rangeMemo.get(k);
    if (m) return m;
    let min = 1e9;
    let max = -1e9;
    for (let z = 0; z <= 2000; z += 20) {
      const v =
        key === 'cur'
          ? this.vectorAt(site, 0, 0, z, 0).speed
          : key === 'temp'
          ? this.tempAt(site, z, 0)
          : key === 'sal'
          ? this.salAt(site, z)
          : this.oxyAt(site, z);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (key === 'cur') {
      min = 0;
      max = max * 1.08;
    } else {
      const p = (max - min) * 0.05;
      min -= p;
      max += p;
    }
    const r = { min, max };
    this._rangeMemo.set(k, r);
    return r;
  },

  surfaceStats(site: SitePhysics, t: number) {
    return {
      sst: this.tempAt(site, 3, t),
      sss: this.salAt(site, 3),
      cur: this.vectorAt(site, 0, 0, 6, t),
    };
  },

  interpretNote(s: Sample, depth: number, site: SitePhysics) {
    const notes: string[] = [];
    const gT = (this.tempAt(site, Math.max(1, depth - 25), 0) - this.tempAt(site, depth + 25, 0)) / 50;
    const gS = (this.salAt(site, Math.max(1, depth - 25)) - this.salAt(site, depth + 25)) / 50;
    if (depth < site.mld * 1.25)
      notes.push('Inside the wind-stirred surface mixed layer — properties nearly uniform with depth.');
    if (gT > 0.06) notes.push('Thermocline — temperature falls rapidly with depth.');
    if (gS > 0.008) notes.push('Halocline — salinity rises sharply with depth (barrier layer).');
    if (s.oxygen < 20 && depth > 120 && depth < 1300)
      notes.push('Within the oxygen-minimum zone — near-suboxic waters.');
    if (s.currentSpeed > 0.8) notes.push('Swift geostrophic flow — an energetic boundary current.');
    if (depth > site.maxDepth - 130) notes.push('Near the seafloor — benthic boundary layer.');
    if (!notes.length) notes.push('Open-ocean interior — weak vertical gradients at this level.');
    return notes.slice(0, 2).join(' ');
  },
};

/* ============================================================================
 * BATHYMETRIC MASKING & LAND-SEA CLASSIFICATION
 * ==========================================================================*/
function pointInPoly(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = ((yi > py) !== (yj > py)) &&
      (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

// Major continent polygon outlines (lon, lat)
const LAND_POLYGONS: [number, number][][] = [
  // Indian Subcontinent & Central/North Asia
  [
    [68.5, 24.0], [70.0, 21.0], [72.8, 19.0], [74.5, 15.0], [76.0, 11.0], [77.5, 8.0],
    [78.5, 9.5], [80.3, 13.0], [83.0, 17.5], [86.0, 20.0], [89.0, 22.0], [92.0, 21.0],
    [94.0, 26.0], [97.0, 28.0], [105.0, 22.0], [108.0, 15.0], [104.0, 10.0], [101.0, 3.0],
    [104.0, 1.2], [108.0, 4.0], [118.0, 24.0], [122.0, 31.0], [129.0, 35.0], [140.0, 45.0],
    [170.0, 65.0], [180.0, 70.0], [-170.0, 66.0], [60.0, 75.0], [30.0, 70.0], [10.0, 58.0],
    [-5.0, 48.0], [-9.0, 38.0], [0.0, 36.0], [35.0, 32.0], [50.0, 30.0], [60.0, 25.0],
    [68.5, 24.0]
  ],
  // Arabian Peninsula
  [
    [35.0, 28.0], [43.0, 13.0], [52.0, 13.0], [59.5, 22.5], [56.0, 26.0], [48.0, 30.0],
    [35.0, 32.0], [35.0, 28.0]
  ],
  // Africa
  [
    [-17.5, 14.8], [-14.0, 28.0], [-5.0, 36.0], [11.0, 37.0], [32.0, 31.5], [43.0, 12.0],
    [51.0, 10.5], [40.0, -10.0], [33.0, -28.0], [18.5, -34.8], [12.0, -17.0], [9.0, 4.5],
    [-8.0, 4.5], [-17.5, 14.8]
  ],
  // North America
  [
    [-168.0, 66.0], [-130.0, 55.0], [-124.0, 38.0], [-117.0, 30.0], [-105.0, 20.0],
    [-87.0, 13.5], [-77.0, 8.0], [-80.0, 25.0], [-75.0, 35.0], [-65.0, 44.0],
    [-55.0, 50.0], [-64.0, 60.0], [-85.0, 70.0], [-120.0, 75.0], [-168.0, 66.0]
  ],
  // South America
  [
    [-77.0, 8.0], [-80.0, 0.0], [-81.0, -5.0], [-72.0, -38.0], [-74.0, -52.0],
    [-65.0, -55.0], [-50.0, -34.0], [-35.0, -6.0], [-50.0, 0.0], [-62.0, 10.0],
    [-77.0, 8.0]
  ],
  // Australia
  [
    [114.0, -22.0], [115.0, -34.0], [138.0, -35.0], [147.0, -43.0], [153.0, -28.0],
    [142.0, -11.0], [130.0, -12.0], [122.0, -17.0], [114.0, -22.0]
  ],
  // Greenland
  [
    [-52.0, 60.0], [-40.0, 60.0], [-20.0, 70.0], [-20.0, 82.0], [-60.0, 82.0],
    [-55.0, 70.0], [-52.0, 60.0]
  ],
  // Antarctica interior
  [
    [-180.0, -66.0], [-90.0, -70.0], [0.0, -68.0], [90.0, -66.0], [180.0, -66.0],
    [180.0, -90.0], [-180.0, -90.0], [-180.0, -66.0]
  ]
];

/**
 * Checks if a given Lat/Lon coordinate lies strictly within the global ocean.
 * Returns true for ocean/sea waters, false for landmasses and continents.
 */
export function isOcean(lat: number, lon: number): boolean {
  // Normalize longitude to -180 .. 180
  let nLon = lon;
  while (nLon > 180) nLon -= 360;
  while (nLon < -180) nLon += 360;

  // Polar ice cap beyond 85°N is arctic ocean, beyond 68°S is antarctic landmass
  if (lat < -68.0) return false;
  if (lat > 84.0) return true;

  // Check land polygons
  for (let i = 0; i < LAND_POLYGONS.length; i++) {
    if (pointInPoly(nLon, lat, LAND_POLYGONS[i])) {
      return false;
    }
  }

  return true;
}

/**
 * Estimates bathymetric depth at a given Lat/Lon coordinate.
 * Positive depth values represent ocean water depth in meters.
 * Negative depth values represent elevation above sea level (land).
 */
export function getBathymetry(lat: number, lon: number): number {
  if (!isOcean(lat, lon)) {
    return -100.0; // Land elevation
  }

  // Deep oceanic trenches vs continental shelf
  const distEquator = Math.abs(lat);
  if (lat > 10 && lat < 12 && lon > 141 && lon < 144) {
    return 10920; // Mariana / Challenger deep
  }
  if (lat > 12 && lat < 22 && lon > 82 && lon < 93) {
    return 2600 + Math.sin(lat * 0.4) * 400; // Bay of Bengal basin
  }
  if (lat > 10 && lat < 24 && lon > 60 && lon < 74) {
    return 3600 + Math.cos(lon * 0.3) * 500; // Arabian Sea basin
  }

  return Math.max(120, 3800 - distEquator * 20);
}

