import React, { useMemo, useState } from 'react';
import { useApp, store, globeRegistry } from '../store/oceanStore';
import { Ocean, provFor, fmtLat, fmtLon, fmtDepth } from '../services/syntheticOcean';
import { Provenance } from '../types/ocean';
import { X, LocateFixed, ArrowDownToLine, Waves, Activity, BarChart2 } from 'lucide-react';
import { EarlyWarningCard } from './EarlyWarningCard';

export const SitePanel: React.FC<{ onDive: () => void }> = ({ onDive }) => {
  const s = useApp();
  const site = s.site;
  const [viewMode, setViewMode] = useState<'telemetry' | 'early_warning'>('telemetry');

  if (!site) return null;

  const stats = useMemo(
    () => Ocean.surfaceStats(site, s.timeOffset),
    [site.id, Math.round(s.timeOffset * 2)]
  );

  const pv = provFor(s.timeOffset);

  return (
    <div className="absolute bottom-4 md:bottom-6 left-1/2 -translate-x-1/2 z-20 w-[min(620px,calc(100vw-24px))] glass px-4 md:px-5 py-3.5 border border-line shadow-2xl">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-[14.5px] font-semibold text-mist truncate flex items-center gap-2">
              {site.name}
              {site.isCustom && (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-accent/20 text-accent border border-accent/40">
                  CUSTOM
                </span>
              )}
            </h2>
            <span className="font-mono text-[10px] text-accent">
              {fmtLat(site.lat)} {fmtLon(site.lon)}
            </span>
          </div>
          <p className="text-[10.5px] text-dim mt-1 leading-snug line-clamp-2">
            {site.blurb}
          </p>
        </div>

        <button
          onClick={() => {
            store.set({ site: null });
            globeRegistry.g?.clearSite();
          }}
          className="text-dim hover:text-mist p-1"
        >
          <X size={15} />
        </button>
      </div>

      {/* Mode Switcher Tabs */}
      <div className="flex items-center gap-2 mt-3 border-b border-line/60 pb-1.5">
        <button
          onClick={() => setViewMode('telemetry')}
          className={`flex items-center gap-1.5 text-[9.5px] font-mono tracking-wider px-2.5 py-1 rounded transition-colors ${
            viewMode === 'telemetry'
              ? 'bg-accent/20 text-accent border border-accent/40 font-semibold'
              : 'text-dim hover:text-mist'
          }`}
        >
          <BarChart2 size={12} />
          <span>SURFACE TELEMETRY</span>
        </button>

        <button
          onClick={() => setViewMode('early_warning')}
          className={`flex items-center gap-1.5 text-[9.5px] font-mono tracking-wider px-2.5 py-1 rounded transition-colors ${
            viewMode === 'early_warning'
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-semibold'
              : 'text-dim hover:text-mist'
          }`}
        >
          <Activity size={12} className="text-rose-400" />
          <span>AI EARLY WARNING (ML RISK)</span>
        </button>

        {site.isCustom ? (
          <span className="ml-auto font-mono text-[8.5px] text-accent bg-accent/10 border border-accent/30 rounded px-1.5 py-0.5">
            CUSTOM {site.custom_observation?.date || s.customObservation?.date || 'OBSERVATION'}
          </span>
        ) : s.historicalMode ? (
          <span className="ml-auto font-mono text-[8.5px] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-1.5 py-0.5">
            COPERNICUS {s.historicalDate}
          </span>
        ) : null}
      </div>

      {viewMode === 'telemetry' ? (
        <React.Fragment>
          <div className="grid grid-cols-4 gap-2 mt-3">
            {[
              ['SST', stats.sst.toFixed(1), '°C'],
              ['SSS', stats.sss.toFixed(2), 'PSU'],
              ['SURF CURRENT', stats.cur.speed.toFixed(2), 'm/s'],
              ['SEAFLOOR', fmtDepth(site.maxDepth), ''],
            ].map((r) => (
              <div key={r[0]} className="p-1.5 rounded bg-white/[0.03] border border-line">
                <div className="text-[7.5px] font-mono tracking-[0.18em] text-dim">{r[0]}</div>
                <div className="font-mono text-[13px] text-mist mt-0.5">
                  {r[1]} <span className="text-[8px] text-dim">{r[2]}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between mt-3.5 pt-3 border-t border-line">
            <div className="flex items-center gap-1.5 font-mono text-[9.5px]">
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse"></span>
              <span className="text-dim">{pv.label} · {pv.source.split('·')[0]}</span>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => globeRegistry.g?.flyToSite(site)}
                className="flex gap-1.5 items-center text-dim hover:text-mist border border-line hover:border-accent/40 rounded px-2.5 py-1.5 text-[10px] font-mono tracking-widest"
              >
                <LocateFixed size={12} />
                RECENTER
              </button>
              <button
                onClick={onDive}
                className="bg-accent hover:bg-[#7ce0ec] text-[#03222a] text-[11px] font-semibold tracking-wider px-4 py-1.5 rounded flex gap-1.5 items-center shadow-lg transition-transform hover:scale-[1.02]"
              >
                <ArrowDownToLine size={13} />
                DIVE
              </button>
            </div>
          </div>
        </React.Fragment>
      ) : (
        <div className="mt-3 max-h-[380px] overflow-y-auto pr-1">
          <EarlyWarningCard
            siteId={site.id === 'bob' || site.id === 'aras' ? site.id : undefined}
            lat={site.lat}
            lon={site.lon}
            regionName={site.name}
            compact={true}
            customObservation={site.custom_observation || s.customObservation || s.activeUpload?.custom_observation}
          />
        </div>
      )}
    </div>
  );
};

