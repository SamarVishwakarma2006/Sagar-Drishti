import React, { useMemo, useState, useEffect } from 'react';
import { useApp, store, toast } from '../store/oceanStore';
import {
  Ocean,
  VAR_COLORS,
  FIELD,
  provFor,
  fmtLat,
  fmtLon,
  fmtDepth,
  timeISO,
  clamp,
} from '../services/syntheticOcean';
import { VariableKey, FloatRecord, ProfileResult, ProvInfo, LatLon } from '../types/ocean';
import { Radio, Crosshair, X, Info, Copy } from 'lucide-react';
import { EarlyWarningCard } from './EarlyWarningCard';

const ProfileChart: React.FC<{
  prof: ProfileResult;
  depth: number;
  active: VariableKey;
}> = ({ prof, depth, active }) => {
  const W = 260;
  const H = 140;
  const L = 34;
  const R = 8;
  const T = 8;
  const B = 8;
  const site = store.get().site!;
  const zmax = Math.max(prof.zmax, 1);
  const y = (z: number) => T + (clamp(z, 0, zmax) / zmax) * (H - T - B);
  const rgA = Ocean.rangeAt(site, active);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full block bg-black/20 rounded p-1">
      {[200, 500, 1000, 1500, 2000]
        .filter((z) => z <= zmax)
        .map((z) => (
          <g key={z}>
            <line x1={L} y1={y(z)} x2={W - R} y2={y(z)} stroke="rgba(126,196,207,0.12)" />
            <text
              x={W - R}
              y={y(z) + 3}
              textAnchor="end"
              fontSize="7"
              fill="#667f8a"
              fontFamily="IBM Plex Mono"
            >
              {z >= 1000 ? z / 1000 + 'k' : z}
            </text>
          </g>
        ))}

      {(['temp', 'sal', 'cur', 'oxy'] as VariableKey[]).map((k) => {
        const rg = Ocean.rangeAt(site, k);
        const d = prof.points
          .map(
            (p, i) =>
              `${i ? 'L' : 'M'}${(
                L +
                clamp(((p[FIELD[k]] ?? 0) - rg.min) / Math.max(1e-6, rg.max - rg.min), 0, 1) *
                  (W - L - R)
              ).toFixed(1)} ${y(p.depth).toFixed(1)}`
          )
          .join(' ');
        return (
          <path
            key={k}
            d={d}
            fill="none"
            stroke={VAR_COLORS[k]}
            strokeWidth={k === active ? 2.2 : 0.9}
            opacity={k === active ? 1 : 0.25}
          />
        );
      })}

      <line
        x1={L}
        y1={y(depth)}
        x2={W - R}
        y2={y(depth)}
        stroke="#56d4e2"
        strokeDasharray="3 2"
        strokeWidth="0.8"
      />
      <text
        x={L - 4}
        y={y(depth) + 3}
        textAnchor="end"
        fontSize="7.5"
        fill="#56d4e2"
        fontFamily="IBM Plex Mono"
      >
        {Math.round(depth)}m
      </text>
      <text x={2} y={T + 6} fontSize="7" fill="#8fa6b0" fontFamily="IBM Plex Mono">
        {rgA.max.toFixed(1)}
      </text>
      <text x={2} y={H - B} fontSize="7" fill="#8fa6b0" fontFamily="IBM Plex Mono">
        {rgA.min.toFixed(1)}
      </text>
    </svg>
  );
};

export const Inspector: React.FC = () => {
  const s = useApp();
  const site = s.site;
  const sel = s.selection;

  if (!site || !sel) return null;

  const isFloat = sel.kind === 'float';
  const f: FloatRecord | undefined = isFloat
    ? Ocean.floatsAt(site).find((x) => x.id === sel.id)
    : undefined;

  const tOff = isFloat ? (f ? f.lastReportOffset : 0) : s.timeOffset;
  const prov: ProvInfo = isFloat
    ? { provenance: 'observed', source: f ? f.source : 'Argo GDAC · INCOIS', label: 'Observation' }
    : provFor(s.timeOffset);

  const depth = isFloat
    ? clamp(s.depth, 2, 2000)
    : clamp((sel as any).depth, 2, site.maxDepth);

  const pos: LatLon =
    isFloat && f ? { lat: f.lat, lon: f.lon } : { lat: (sel as any).lat, lon: (sel as any).lon };

  const sample = useMemo(
    () =>
      isFloat && f
        ? Ocean.sampleFromFloat(f, depth)
        : Ocean.sampleAt(site, pos.lat, pos.lon, depth, s.timeOffset),
    [site.id, isFloat, f?.id, depth, s.timeOffset]
  );

  const prof = useMemo(
    () =>
      isFloat && f
        ? f.profile
        : Ocean.profileAt(site, pos.lat, pos.lon, tOff),
    [site.id, isFloat, f?.id, pos.lat.toFixed(3), pos.lon.toFixed(3), Math.round(tOff)]
  );

  const note = Ocean.interpretNote(sample, depth, site);

  const copy = () => {
    try {
      const data = {
        platform: isFloat ? `Argo ${f?.id}` : 'virtual sample',
        time: timeISO(tOff),
        position: { lat: +pos.lat.toFixed(4), lon: +pos.lon.toFixed(4) },
        depth_m: Math.round(depth),
        provenance: isFloat ? 'observed' : s.historicalMode ? 'historical' : prov.provenance,
        source: isFloat ? (f ? f.source : 'Argo GDAC') : s.historicalMode ? 'Copernicus Marine PHY Reanalysis' : prov.source,
        temperature_C: +sample.temperature.toFixed(3),
        salinity_PSU: +sample.salinity.toFixed(3),
        current_mps: +sample.currentSpeed.toFixed(3),
        current_dir_deg: +sample.currentDir.toFixed(1),
        oxygen_umolkg: +sample.oxygen.toFixed(2),
        prediction: s.activePrediction
          ? {
              horizon_days: s.activePrediction.horizon_days,
              risk_level: s.activePrediction.prediction,
              event_type: s.activePrediction.event_type,
              probability: s.activePrediction.probability,
            }
          : null,
      };
      navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      toast('Inspector telemetry copied as JSON');
    } catch (e) {
      toast('Clipboard unavailable in this browser');
    }
  };

  return (
    <div className="absolute z-30 right-3 md:right-4 top-14 bottom-4 max-md:left-3 max-md:top-auto max-md:bottom-3 max-md:max-h-[58vh] w-[calc(100vw-24px)] md:w-[320px] glass flex flex-col overflow-hidden border border-line shadow-2xl">
      {/* Header */}
      <div className="flex items-center gap-2 px-3.5 py-3 border-b border-line bg-accent/[0.04]">
        <span className="text-accent">
          {isFloat ? <Radio size={15} /> : <Crosshair size={15} />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[12.5px] font-medium text-mist truncate">
            {isFloat ? `Argo ${f?.id}` : 'Virtual Water Probe'}
          </div>
          <div className="font-mono text-[8.5px] text-dim tracking-wider">
            {isFloat ? 'IN-SITU OBSERVATION' : s.historicalMode ? 'COPERNICUS HISTORICAL REANALYSIS' : 'MODEL FIELD SAMPLER'}
          </div>
        </div>
        <button
          onClick={() => store.set({ selection: null })}
          className="text-dim hover:text-mist p-1"
        >
          <X size={14} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-3.5 py-3 space-y-3">
        {/* Coordinate & Provenance Info */}
        <div className="space-y-1.5 font-mono text-[10px] bg-white/[0.02] p-2.5 rounded border border-line">
          <div className="flex justify-between items-center pb-1 border-b border-line/40">
            <span className="text-dim text-[8.5px]">DATA PROVENANCE:</span>
            <div className="flex items-center gap-1.5">
              {isFloat ? (
                <span className="px-1.5 py-0.5 rounded text-[8px] font-mono tracking-wider font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                  OBSERVED
                </span>
              ) : s.historicalMode ? (
                <span className="px-1.5 py-0.5 rounded text-[8px] font-mono tracking-wider font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  HISTORICAL
                </span>
              ) : (
                <span className="px-1.5 py-0.5 rounded text-[8px] font-mono tracking-wider font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                  MODELLED
                </span>
              )}
              {s.activePrediction && (
                <span className="px-1.5 py-0.5 rounded text-[8px] font-mono tracking-wider font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/40">
                  PREDICTED
                </span>
              )}
            </div>
          </div>
          <div className="flex justify-between">
            <span className="text-dim">COORDINATES:</span>
            <span className="text-mist">
              {fmtLat(pos.lat, 3)} {fmtLon(pos.lon, 3)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-dim">DEPTH:</span>
            <span className="text-accent font-semibold">{fmtDepth(depth)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dim">TIMESTAMP:</span>
            <span className="text-mist">{s.historicalMode ? s.historicalDate : timeISO(tOff)}</span>
          </div>
          {isFloat && f && (
            <div className="flex justify-between">
              <span className="text-dim">CYCLE:</span>
              <span className="text-mist">
                #{f.cycle} · reported {Math.round(Math.abs(f.lastReportOffset))}h ago
              </span>
            </div>
          )}
        </div>

        {/* Dedicated ML Early Warning Card */}
        <EarlyWarningCard
          lat={pos.lat}
          lon={pos.lon}
          regionName={site.name}
          compact={true}
          customObservation={site.custom_observation || s.customObservation || s.activeUpload?.custom_observation}
        />

        {/* Physical readings */}
        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 rounded bg-white/[0.03] border border-line">
            <div className="font-mono text-[8px] tracking-[0.2em]" style={{ color: VAR_COLORS.temp }}>
              TEMP
            </div>
            <div className="font-mono text-[14px] text-mist mt-0.5">
              {sample.temperature.toFixed(2)}
            </div>
            <div className="font-mono text-[8px] text-dim">°C</div>
          </div>

          <div className="p-2 rounded bg-white/[0.03] border border-line">
            <div className="font-mono text-[8px] tracking-[0.2em]" style={{ color: VAR_COLORS.sal }}>
              SAL
            </div>
            <div className="font-mono text-[14px] text-mist mt-0.5">
              {sample.salinity.toFixed(2)}
            </div>
            <div className="font-mono text-[8px] text-dim">PSU</div>
          </div>

          <div className="p-2 rounded bg-white/[0.03] border border-line">
            <div className="font-mono text-[8px] tracking-[0.2em]" style={{ color: VAR_COLORS.cur }}>
              CURRENT
            </div>
            <div className="font-mono text-[14px] text-mist mt-0.5">
              {sample.currentSpeed.toFixed(2)}
            </div>
            <div className="font-mono text-[8px] text-dim">
              m/s @ {Math.round(sample.currentDir)}°
            </div>
          </div>

          <div className="p-2 rounded bg-white/[0.03] border border-line">
            <div className="font-mono text-[8px] tracking-[0.2em]" style={{ color: VAR_COLORS.oxy }}>
              DISSOLVED O₂
            </div>
            <div className="font-mono text-[14px] text-mist mt-0.5">
              {sample.oxygen.toFixed(1)}
            </div>
            <div className="font-mono text-[8px] text-dim">µmol/kg</div>
          </div>
        </div>

        {/* Profile Chart */}
        <div>
          <div className="text-[8.5px] font-mono text-dim tracking-[0.2em] mb-1.5">
            VERTICAL PROFILE · 0–{Math.round(prof.zmax)} m
          </div>
          <ProfileChart prof={prof} depth={depth} active={s.variable} />
        </div>

        {/* Note */}
        <div className="flex gap-2 items-start text-dim bg-accent/[0.03] p-2 rounded border border-line">
          <Info size={13} className="text-accent shrink-0 mt-0.5" />
          <p className="text-[9.5px] leading-relaxed text-mist/90">{note}</p>
        </div>

        <button
          onClick={copy}
          className="w-full flex items-center justify-center gap-2 border border-line hover:border-accent/40 text-dim hover:text-mist rounded py-2 text-[10px] font-mono tracking-widest transition-colors"
        >
          <Copy size={12} />
          COPY JSON TELEMETRY
        </button>
      </div>
    </div>
  );
};
