import { SitePhysics, FloatRecord, ProfilePoint, ProfileResult, BoundingBox } from '../types/ocean';

export class ClientCsvParser {
  static parseCsv(text: string, filename: string): { site: SitePhysics; floats: FloatRecord[] } {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) {
      throw new Error('CSV file contains insufficient data');
    }

    // Skip comment lines
    let headerIdx = 0;
    while (headerIdx < lines.length && lines[headerIdx].trim().startsWith('#')) {
      headerIdx++;
    }

    const headerLine = lines[headerIdx];
    const sep = headerLine.includes('\t') ? '\t' : headerLine.includes(';') ? ';' : ',';
    const headers = headerLine.split(sep).map((h) => h.trim().toLowerCase().replace(/['"]/g, ''));

    const findIdx = (patterns: RegExp[]) =>
      headers.findIndex((h) => patterns.some((p) => p.test(h)));

    const latIdx = findIdx([/^lat/, /^latitude/, /^y$/]);
    const lonIdx = findIdx([/^lon/, /^long/, /^longitude/, /^x$/]);
    const depthIdx = findIdx([/^depth/, /^pres/, /^pressure/, /^z$/]);
    const tempIdx = findIdx([/^temp/, /^temperature/, /^sst/]);
    const salIdx = findIdx([/^sal/, /^salinity/, /^psal/, /^sss/]);
    const oxyIdx = findIdx([/^oxy/, /^oxygen/, /^doxy/, /^o2/]);
    const spdIdx = findIdx([/^cur/, /^current/, /^speed/, /^velocity/]);
    const dirIdx = findIdx([/^dir/, /^direction/, /^heading/]);
    const idIdx = findIdx([/^platform/, /^float/, /^id/, /^wmo/]);
    const cycleIdx = findIdx([/^cycle/, /^cast/, /^profile/]);

    if (latIdx === -1 || lonIdx === -1) {
      throw new Error('Could not find Latitude / Longitude headers in CSV');
    }

    const rows: any[] = [];
    for (let i = headerIdx + 1; i < lines.length; i++) {
      const parts = lines[i].split(sep).map((p) => p.trim().replace(/['"]/g, ''));
      if (parts.length <= Math.max(latIdx, lonIdx)) continue;

      const lat = parseFloat(parts[latIdx]);
      const lon = parseFloat(parts[lonIdx]);
      if (isNaN(lat) || isNaN(lon)) continue;

      const depth = depthIdx !== -1 ? Math.abs(parseFloat(parts[depthIdx]) || 0) : 0;
      const temp = tempIdx !== -1 ? parseFloat(parts[tempIdx]) : 24.0 - depth * 0.012;
      const sal = salIdx !== -1 ? parseFloat(parts[salIdx]) : 34.5 + depth * 0.001;
      const oxy = oxyIdx !== -1 ? parseFloat(parts[oxyIdx]) : Math.max(10, 180 - depth * 0.2);
      const cur = spdIdx !== -1 ? parseFloat(parts[spdIdx]) : Math.max(0.05, 0.45 - depth * 0.0003);
      const dir = dirIdx !== -1 ? parseFloat(parts[dirIdx]) : 45.0;
      const id = idIdx !== -1 ? parts[idIdx] : `Float_${Math.floor(lat * 10)}_${Math.floor(lon * 10)}`;
      const cycle = cycleIdx !== -1 ? parseInt(parts[cycleIdx], 10) || 1 : 1;

      rows.push({ lat, lon, depth, temp, sal, oxy, cur, dir, id, cycle });
    }

    if (rows.length === 0) {
      throw new Error('No valid numeric observation rows found in CSV');
    }

    // Group into distinct float records
    const groups = new Map<string, typeof rows>();
    rows.forEach((r) => {
      const key = `${r.id}_${r.cycle}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(r);
    });

    const floats: FloatRecord[] = [];
    let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180, maxD = 0;

    groups.forEach((pts, key) => {
      pts.sort((a, b) => a.depth - b.depth);
      const first = pts[0];
      minLat = Math.min(minLat, first.lat);
      maxLat = Math.max(maxLat, first.lat);
      minLon = Math.min(minLon, first.lon);
      maxLon = Math.max(maxLon, first.lon);

      const profilePoints: ProfilePoint[] = pts.map((p) => {
        maxD = Math.max(maxD, p.depth);
        return {
          depth: p.depth,
          temperature: isNaN(p.temp) ? 20.0 : p.temp,
          salinity: isNaN(p.sal) ? 34.5 : p.sal,
          currentSpeed: isNaN(p.cur) ? 0.3 : p.cur,
          currentDir: isNaN(p.dir) ? 0 : p.dir,
          oxygen: isNaN(p.oxy) ? 150 : p.oxy,
        };
      });

      const zmax = profilePoints.length > 0 ? profilePoints[profilePoints.length - 1].depth : 1000;

      floats.push({
        id: first.id.startsWith('290') ? first.id : `Float_${floats.length + 1}`,
        platform: `In-situ Float (${filename})`,
        lat: first.lat,
        lon: first.lon,
        cycle: first.cycle,
        lastReportOffset: -12.0,
        parkingDepth: Math.min(zmax * 0.5, 1000),
        profile: { zmax: Math.max(zmax, 200), points: profilePoints },
        source: `Client-parsed · ${filename}`,
      });
    });

    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;
    const siteMaxDepth = Math.max(maxD, 1200);

    const cleanId = `upload_csv_${Date.now()}`;
    const cleanName = filename.replace(/\.(csv|txt)$/i, '').replace(/_/g, ' ');

    const bbox: BoundingBox = {
      min_lat: minLat,
      max_lat: maxLat,
      min_lon: minLon,
      max_lon: maxLon,
    };

    const site: SitePhysics = {
      id: cleanId,
      name: `${cleanName} (Tabular Upload)`,
      region: `Lat ${minLat.toFixed(1)}°–${maxLat.toFixed(1)}°, Lon ${minLon.toFixed(1)}°–${maxLon.toFixed(1)}°`,
      lat: centerLat,
      lon: centerLon,
      maxDepth: siteMaxDepth,
      blurb: `User dataset '${filename}' parsed with ${floats.length} float stations and ${rows.length} depth observations.`,
      ts: rows[0]?.temp || 28.0,
      ss: rows[0]?.sal || 34.5,
      td: 2.8,
      sd: 34.8,
      mld: 40.0,
      tw: 45.0,
      salMaxAmp: 0.3,
      salMaxZ: 100.0,
      flow: 0.4,
      eddy: 200.0,
      bgU: 0.15,
      bgV: 0.08,
      o2s: rows[0]?.oxy || 180.0,
      o2d: 150.0,
      o2z0: 90.0,
      o2minAmp: 120.0,
      o2minZ: 350.0,
      o2minW: 200.0,
      bbox,
      variables: ['temp', 'sal', 'cur', 'oxy'],
      isCustom: true,
      sourceType: 'TABULAR_OBSERVATION',
      floats,
    };

    return { site, floats };
  }
}
