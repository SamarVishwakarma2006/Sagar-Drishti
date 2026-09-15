import { SitePhysics, FloatRecord, ProfilePoint, BoundingBox } from '../types/ocean';

export class ClientCsvParser {
  static parseCsv(text: string, filename: string): { site: SitePhysics; floats: FloatRecord[]; customObservation?: any } {
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

    const timeIdx = findIdx([/^time/, /^date/, /^datetime/, /^timestamp/, /^juld/]);
    const latIdx = findIdx([/^lat/, /^latitude/, /^y$/]);
    const lonIdx = findIdx([/^lon/, /^long/, /^longitude/, /^x$/]);
    const depthIdx = findIdx([/^depth/, /^pres/, /^pressure/, /^z$/]);
    const tempIdx = findIdx([/^thetao$/, /^temp/, /^temperature/, /^sst/]);
    const salIdx = findIdx([/^so$/, /^sal/, /^salinity/, /^psal/, /^sss/]);
    const uoIdx = findIdx([/^uo$/, /^cur_u/, /^u_velocity/, /^u$/]);
    const voIdx = findIdx([/^vo$/, /^cur_v/, /^v_velocity/, /^v$/]);
    const spdIdx = findIdx([/^cur/, /^current/, /^speed/, /^velocity/]);
    const sshIdx = findIdx([/^zos$/, /^ssh/, /^sea_surface_height/]);
    const mldIdx = findIdx([/^mlotst$/, /^mld/, /^mixed_layer/]);
    const oxyIdx = findIdx([/^oxy/, /^oxygen/, /^doxy/, /^o2/]);
    const dirIdx = findIdx([/^dir/, /^direction/, /^heading/]);
    const idIdx = findIdx([/^platform/, /^float/, /^id/, /^wmo/, /^station/]);
    const cycleIdx = findIdx([/^cycle/, /^cast/, /^profile/]);

    if (latIdx === -1 || lonIdx === -1) {
      throw new Error('Could not find Latitude / Longitude headers in CSV');
    }

    const rows: any[] = [];
    let firstDate: string | undefined = undefined;

    for (let i = headerIdx + 1; i < lines.length; i++) {
      const parts = lines[i].split(sep).map((p) => p.trim().replace(/['"]/g, ''));
      if (parts.length <= Math.max(latIdx, lonIdx)) continue;

      const lat = parseFloat(parts[latIdx]);
      const lon = parseFloat(parts[lonIdx]);
      if (isNaN(lat) || isNaN(lon)) continue;

      if (!firstDate && timeIdx !== -1 && parts[timeIdx]) {
        firstDate = parts[timeIdx].split('T')[0].split(' ')[0];
      }

      const depth = depthIdx !== -1 ? Math.abs(parseFloat(parts[depthIdx]) || 0) : 0;
      const temp = tempIdx !== -1 && !isNaN(parseFloat(parts[tempIdx])) ? parseFloat(parts[tempIdx]) : 24.0 - depth * 0.012;
      const sal = salIdx !== -1 && !isNaN(parseFloat(parts[salIdx])) ? parseFloat(parts[salIdx]) : 34.5 + depth * 0.001;
      const uo = uoIdx !== -1 && !isNaN(parseFloat(parts[uoIdx])) ? parseFloat(parts[uoIdx]) : undefined;
      const vo = voIdx !== -1 && !isNaN(parseFloat(parts[voIdx])) ? parseFloat(parts[voIdx]) : undefined;
      const cur = spdIdx !== -1 && !isNaN(parseFloat(parts[spdIdx]))
        ? parseFloat(parts[spdIdx])
        : (uo !== undefined && vo !== undefined ? Math.sqrt(uo * uo + vo * vo) : Math.max(0.05, 0.45 - depth * 0.0003));
      const zos = sshIdx !== -1 && !isNaN(parseFloat(parts[sshIdx])) ? parseFloat(parts[sshIdx]) : undefined;
      const mld = mldIdx !== -1 && !isNaN(parseFloat(parts[mldIdx])) ? parseFloat(parts[mldIdx]) : undefined;
      const oxy = oxyIdx !== -1 && !isNaN(parseFloat(parts[oxyIdx])) ? parseFloat(parts[oxyIdx]) : Math.max(10, 180 - depth * 0.2);
      const dir = dirIdx !== -1 && !isNaN(parseFloat(parts[dirIdx])) ? parseFloat(parts[dirIdx]) : 45.0;
      const id = idIdx !== -1 ? parts[idIdx] : `Float_${Math.floor(lat * 10)}_${Math.floor(lon * 10)}`;
      const cycle = cycleIdx !== -1 ? parseInt(parts[cycleIdx], 10) || 1 : 1;

      rows.push({ lat, lon, depth, temp, sal, oxy, cur, dir, id, cycle, uo, vo, zos, mld });
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

    let customObs: any = null;
    if (rows.length > 0 && (tempIdx !== -1 || salIdx !== -1 || uoIdx !== -1 || voIdx !== -1 || sshIdx !== -1 || mldIdx !== -1)) {
      const r0 = rows[0];
      const obsDate = firstDate || new Date().toISOString().split('T')[0];
      customObs = {
        date: obsDate,
        lat: r0.lat,
        lon: r0.lon,
        thetao: tempIdx !== -1 ? r0.temp : undefined,
        so: salIdx !== -1 ? r0.sal : undefined,
        uo: r0.uo,
        vo: r0.vo,
        zos: r0.zos,
        mlotst: r0.mld,
      };
    }

    const site: SitePhysics = {
      id: cleanId,
      name: `${cleanName} (Tabular Upload)`,
      region: `Lat ${minLat.toFixed(1)}°–${maxLat.toFixed(1)}°, Lon ${minLon.toFixed(1)}°–${maxLon.toFixed(1)}°`,
      lat: centerLat,
      lon: centerLon,
      maxDepth: siteMaxDepth,
      blurb: `User dataset '${filename}' parsed with ${floats.length} float stations and ${rows.length} depth observations.`,
      ts: customObs?.thetao ?? (rows[0]?.temp || 28.0),
      ss: customObs?.so ?? (rows[0]?.sal || 34.5),
      td: 2.8,
      sd: 34.8,
      mld: customObs?.mlotst ?? 40.0,
      tw: 45.0,
      salMaxAmp: 0.3,
      salMaxZ: 100.0,
      flow: 0.4,
      eddy: 200.0,
      bgU: customObs?.uo ?? 0.15,
      bgV: customObs?.vo ?? 0.08,
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
      custom_observation: customObs,
    };

    return { site, floats, customObservation: customObs };
  }
}
