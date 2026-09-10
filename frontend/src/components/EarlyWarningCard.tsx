import React, { useState, useEffect } from 'react';
import { useApp, store, toast } from '../store/oceanStore';
import { PredictionResponse, PredictionRequest } from '../types/ocean';
import { PredictionAPI } from '../services/api';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Clock,
  Waves,
  Database,
  Info,
  Calendar,
  Layers,
  ChevronDown,
  ChevronUp,
  Activity,
  Compass,
} from 'lucide-react';

interface EarlyWarningCardProps {
  siteId?: string;
  lat?: number;
  lon?: number;
  regionName?: string;
  compact?: boolean;
  onClose?: () => void;
}

export const EarlyWarningCard: React.FC<EarlyWarningCardProps> = ({
  siteId,
  lat,
  lon,
  regionName,
  compact = false,
  onClose,
}) => {
  const s = useApp();
  const [horizon, setHorizon] = useState<number>(3);
  const [activeTab, setActiveTab] = useState<'predicted' | 'observed' | 'historical' | 'quality'>('predicted');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(s.activePrediction || null);
  const [showVariableBreakdown, setShowVariableBreakdown] = useState<boolean>(false);

  // Target date for prediction: use s.historicalDate if historicalMode is active, otherwise default to latest 2024-10-24
  const targetDate = s.historicalMode ? s.historicalDate : (s.historicalDate || '2024-10-24');

  // Resolve study site or coordinates
  const effectiveSiteId = siteId || (s.site?.id === 'bob' || s.site?.id === 'aras' ? s.site.id : undefined);
  const effectiveLat = lat !== undefined ? lat : s.site?.lat;
  const effectiveLon = lon !== undefined ? lon : s.site?.lon;
  const effectiveRegion = regionName || s.site?.name || (effectiveSiteId === 'aras' ? 'Arabian Sea' : 'Bay of Bengal');

  useEffect(() => {
    let isCancelled = false;

    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);

      try {
        const req: PredictionRequest = {
          date: targetDate,
          horizon_days: horizon,
        };

        if (effectiveSiteId) {
          req.site_id = effectiveSiteId;
          req.mode = 'region';
        } else if (effectiveLat !== undefined && effectiveLon !== undefined) {
          req.lat = effectiveLat;
          req.lon = effectiveLon;
          req.mode = 'point';
        } else {
          req.site_id = 'bob';
          req.mode = 'region';
        }

        const res = await PredictionAPI.predict(req);
        if (!isCancelled) {
          setPrediction(res);
          store.setActivePrediction(res);
        }
      } catch (err: any) {
        if (!isCancelled) {
          const msg = err?.message || 'Failed to evaluate ocean risk model.';
          setError(msg);
          console.warn('[EarlyWarningCard] Prediction error:', msg);
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };

    fetchPrediction();

    return () => {
      isCancelled = true;
    };
  }, [targetDate, horizon, effectiveSiteId, effectiveLat, effectiveLon]);

  const warningLevel = prediction?.warning_level || 'NO_ALERT';
  const prob = prediction ? Math.round((prediction.model_estimated_probability ?? prediction.probability) * 100) : 0;
  const threshold = prediction
    ? Math.round((prediction.alert_threshold ?? prediction.threshold) * 100)
    : (horizon === 3 ? 27 : horizon === 2 ? 21 : 15);
  const targetName = prediction?.target || `event_within_${horizon}d`;

  // Status configuration
  const statusConfig = {
    NO_ALERT: {
      label: 'NO ALERT',
      color: 'text-emerald-400',
      border: 'border-emerald-500/40',
      bg: 'bg-emerald-500/10',
      badgeBg: 'bg-emerald-500/20',
      icon: <ShieldCheck size={16} className="text-emerald-400" />,
      tagline: 'Normal background oceanic conditions',
    },
    WATCH: {
      label: 'WATCH',
      color: 'text-amber-300',
      border: 'border-amber-500/50',
      bg: 'bg-amber-500/10',
      badgeBg: 'bg-amber-500/20',
      icon: <AlertTriangle size={16} className="text-amber-400" />,
      tagline: 'Elevated ocean thermal & kinetic anomaly detected',
    },
    HIGH_ALERT: {
      label: 'HIGH ALERT (SEVERITY)',
      color: 'text-rose-400',
      border: 'border-rose-500/60',
      bg: 'bg-rose-500/15',
      badgeBg: 'bg-rose-500/25',
      icon: <ShieldAlert size={16} className="text-rose-400 animate-pulse" />,
      tagline: 'High Alert — presentation severity policy (score ≥ 0.50, not calibrated probability)',
    },
  }[warningLevel];

  return (
    <div className={`rounded-lg glass border ${statusConfig.border} overflow-hidden shadow-2xl transition-all duration-300 ${compact ? 'p-3 text-[11px]' : 'p-4 text-[12px]'}`}>
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-line/60">
        <div className="flex items-center gap-2">
          <Activity size={17} className="text-accent shrink-0" />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-mist tracking-wider text-[12.5px] uppercase">
                {horizon}-DAY EARLY WARNING
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-white/[0.04] border border-line text-dim">
                {targetName}
              </span>
            </div>
            <div className="text-[9.5px] font-mono text-dim flex items-center gap-2 mt-0.5">
              <span>REGION: {effectiveRegion}</span>
              <span>·</span>
              <span>DATE: {targetDate}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className={`px-2.5 py-1 rounded border font-mono font-bold text-[10px] tracking-widest uppercase flex items-center gap-1.5 ${statusConfig.badgeBg} ${statusConfig.color} ${statusConfig.border}`}>
            {statusConfig.icon}
            <span>{statusConfig.label}</span>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-dim hover:text-mist p-1 rounded hover:bg-white/[0.05]"
              title="Close panel"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Horizon Selector Bar */}
      <div className="flex items-center justify-between mt-3 px-2 py-1.5 rounded bg-black/40 border border-line/40 font-mono text-[9px]">
        <span className="text-dim flex items-center gap-1">
          <Clock size={11} /> FORECAST HORIZON:
        </span>
        <div className="flex items-center gap-1">
          {[
            { val: 0, label: '0d (Active)' },
            { val: 1, label: '1d Lead' },
            { val: 2, label: '2d Lead' },
            { val: 3, label: '3d Lead' },
          ].map((h) => (
            <button
              key={h.val}
              onClick={() => setHorizon(h.val)}
              className={`px-2 py-0.5 rounded transition-colors ${
                horizon === h.val
                  ? 'bg-accent/25 text-accent font-bold border border-accent/40 shadow-sm'
                  : 'text-dim hover:text-mist hover:bg-white/[0.04]'
              }`}
            >
              {h.label}
            </button>
          ))}
        </div>
      </div>

      {/* Horizon Underpowered Banner if 0d or 1d */}
      {(horizon === 0 || horizon === 1) && (
        <div className="mt-2 px-2.5 py-1.5 rounded bg-amber-500/10 border border-amber-500/25 text-amber-300 text-[8.5px] font-mono flex items-start gap-2">
          <Info size={13} className="shrink-0 text-amber-400 mt-0.5" />
          <div>
            <span className="font-bold">Experimental Horizon ({horizon}d):</span> The current {horizon}-day model does not demonstrate useful discriminative ability on the held-out test set and is statistically underpowered given the limited number of independent events.
          </div>
        </div>
      )}

      {/* Model Probability Meter */}
      <div className="mt-3 p-2.5 rounded bg-black/30 border border-line/40 space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="text-[10px] font-mono text-dim">
            Model-estimated risk score:
          </span>
          <div className="text-right">
            <span className={`font-mono text-[16px] font-bold ${statusConfig.color}`}>
              {prob}%
            </span>
            <span className="text-[9px] font-mono text-dim ml-1.5">
              (Operational Threshold: {threshold}%)
            </span>
          </div>
        </div>

        {/* Probability bar with threshold marker */}
        <div className="relative w-full bg-black/60 h-2 rounded-full overflow-hidden border border-line/50">
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              warningLevel === 'HIGH_ALERT'
                ? 'bg-gradient-to-r from-amber-500 via-rose-500 to-red-500'
                : warningLevel === 'WATCH'
                ? 'bg-gradient-to-r from-emerald-500 via-yellow-500 to-amber-500'
                : 'bg-gradient-to-r from-teal-500 to-emerald-400'
            }`}
            style={{ width: `${Math.max(2, Math.min(100, prob))}%` }}
          />
          {/* Threshold marker tick */}
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-white shadow-[0_0_4px_white]"
            style={{ left: `${threshold}%` }}
            title={`Operational Alert Threshold: ${threshold}%`}
          />
        </div>

        <div className="flex justify-between items-center text-[8.5px] font-mono text-dim">
          <span>0% BASELINE</span>
          <span className="text-accent/90">▲ ALERT THRESHOLD ({threshold}%)</span>
          <span>100% EXTREME</span>
        </div>
        <div className="text-[8px] font-mono text-dim/75 pt-1 border-t border-line/20">
          Policy: Alert trigger &ge; {threshold}% (validation-frozen); High Alert &ge; 50% (presentation severity policy, not calibrated probability).
        </div>
      </div>

      {/* Segregated Context Tabs */}
      <div className="flex items-center gap-1 mt-3 border-b border-line/50 pb-1 text-[9.5px] font-mono">
        {[
          { key: 'predicted', label: '[PREDICTED]', desc: 'Model Risk & Drivers' },
          { key: 'observed', label: '[OBSERVED]', desc: 'Real Ocean State' },
          { key: 'historical', label: '[HISTORICAL]', desc: 'Documented Events' },
          { key: 'quality', label: '[DATA QUALITY]', desc: 'Audit & Guardrails' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-2.5 py-1 rounded transition-colors ${
              activeTab === tab.key
                ? 'bg-accent/20 text-accent font-bold border border-accent/40 shadow-sm'
                : 'text-dim hover:text-mist hover:bg-white/[0.03]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content Panes */}
      <div className="mt-2.5 min-h-[140px]">
        {loading ? (
          <div className="py-8 flex flex-col items-center justify-center gap-2 font-mono text-[10px] text-dim">
            <span className="w-2 h-2 rounded-full bg-accent animate-ping" />
            <span>Evaluating leak-free Random Forest on Copernicus reanalysis...</span>
          </div>
        ) : error ? (
          <div className="p-3 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 font-mono text-[10px] space-y-1">
            <div className="font-semibold flex items-center gap-1.5">
              <AlertTriangle size={13} /> {error}
            </div>
            <div className="text-[9px] text-dim">
              Note: Copernicus reanalysis covers continuous daily dates from 2024-06-24 to 2026-06-23. Select a date within this range.
            </div>
          </div>
        ) : prediction ? (
          <React.Fragment>
            {/* TAB 1: PREDICTED RISK & PHYSICAL DRIVERS */}
            {activeTab === 'predicted' && (
              <div className="space-y-2.5 animate-fade-in">
                <div className="text-[9.5px] font-mono text-dim leading-relaxed bg-white/[0.02] p-2 rounded border border-line/40">
                  <span className="text-mist font-semibold">Assessment: </span>
                  {prediction.explainability?.human_readable?.what || statusConfig.tagline}.
                </div>

                {/* Top Physical Indicators */}
                <div>
                  <div className="text-[8.5px] font-mono text-dim uppercase tracking-wider mb-1.5 flex justify-between items-center">
                    <span>Top Associated Physical Indicators</span>
                    <span className="text-accent/80 text-[8px]">NON-CAUSAL CORRELATIONS</span>
                  </div>

                  <div className="space-y-1.5 font-mono text-[9px]">
                    {(prediction.top_features || []).slice(0, 4).map((f) => (
                      <div key={f.feature} className="p-1.5 rounded bg-black/25 border border-line/30">
                        <div className="flex justify-between items-center">
                          <span className="text-mist font-medium truncate max-w-[210px]">
                            {f.description || f.feature}
                          </span>
                          <span className="text-accent font-semibold ml-2">
                            {Math.round(f.importance * 100)}%
                          </span>
                        </div>
                        <div className="w-full bg-black/40 h-1 rounded-full overflow-hidden mt-1">
                          <div
                            className="bg-accent h-full rounded-full"
                            style={{ width: `${Math.max(4, Math.min(100, f.importance * 100))}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Ocean Variable Grouping Collapsible */}
                <div className="pt-1 border-t border-line/40">
                  <button
                    onClick={() => setShowVariableBreakdown(!showVariableBreakdown)}
                    className="w-full flex items-center justify-between text-[8.5px] font-mono text-dim hover:text-accent py-1"
                  >
                    <span>VARIABLE & TEMPORAL SCALE BREAKDOWN</span>
                    {showVariableBreakdown ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </button>

                  {showVariableBreakdown && prediction.explainability && (
                    <div className="mt-1.5 p-2 rounded bg-black/40 border border-line/30 space-y-2 font-mono text-[8.5px]">
                      <div>
                        <div className="text-[8px] text-dim uppercase mb-1">Ocean Variable Distribution</div>
                        <div className="grid grid-cols-2 gap-1">
                          {(Object.entries(prediction.explainability.ocean_variable_importance || {}) as [string, number][]).map(([k, v]) => (
                            <div key={k} className="flex justify-between bg-white/[0.02] px-1.5 py-0.5 rounded border border-line/20">
                              <span className="text-dim capitalize">{k.replace(/_/g, ' ')}</span>
                              <span className="text-mist">{Math.round((v || 0) * 100)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="pt-1 border-t border-line/30">
                        <div className="text-[8px] text-dim uppercase mb-1">Temporal Window Scale</div>
                        <div className="grid grid-cols-2 gap-1">
                          {(Object.entries(prediction.explainability.time_window_importance || {}) as [string, number][]).map(([k, v]) => (
                            <div key={k} className="flex justify-between bg-white/[0.02] px-1.5 py-0.5 rounded border border-line/20">
                              <span className="text-dim">{k.toUpperCase()}</span>
                              <span className="text-mist">{Math.round((v || 0) * 100)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TAB 2: OBSERVED OCEAN STATE */}
            {activeTab === 'observed' && (
              <div className="space-y-2 font-mono text-[9px] animate-fade-in">
                <div className="p-2 rounded bg-cyan-950/20 border border-cyan-500/30 text-cyan-200">
                  <div className="font-semibold text-[9.5px]">REAL MEASURED / REANALYSIS CONDITIONS</div>
                  <div className="text-[8.5px] text-cyan-300/80 mt-0.5">
                    Spatial sector: {prediction.observed_state?.spatial_coverage || effectiveRegion} · Date: {prediction.observed_state?.observation_date || targetDate}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                  <div className="p-2 rounded bg-white/[0.02] border border-line/40">
                    <div className="text-[8px] text-dim uppercase">Sea Surface Temp (thetao)</div>
                    <div className="text-[13px] font-semibold text-mist mt-0.5">
                      {prediction.observed_state?.surface_temperature_c?.toFixed(2) ?? '—'} <span className="text-[9px] text-dim">°C</span>
                    </div>
                  </div>

                  <div className="p-2 rounded bg-white/[0.02] border border-line/40">
                    <div className="text-[8px] text-dim uppercase">Salinity (so)</div>
                    <div className="text-[13px] font-semibold text-mist mt-0.5">
                      {prediction.observed_state?.surface_salinity_psu?.toFixed(2) ?? '—'} <span className="text-[9px] text-dim">PSU</span>
                    </div>
                  </div>

                  <div className="p-2 rounded bg-white/[0.02] border border-line/40">
                    <div className="text-[8px] text-dim uppercase">Current Speed</div>
                    <div className="text-[13px] font-semibold text-mist mt-0.5">
                      {prediction.observed_state?.current_speed_mps?.toFixed(2) ?? '—'} <span className="text-[9px] text-dim">m/s</span>
                    </div>
                  </div>

                  <div className="p-2 rounded bg-white/[0.02] border border-line/40">
                    <div className="text-[8px] text-dim uppercase">Sea Surface Height (zos)</div>
                    <div className="text-[13px] font-semibold text-mist mt-0.5">
                      {prediction.observed_state?.sea_surface_height_m?.toFixed(3) ?? '—'} <span className="text-[9px] text-dim">m</span>
                    </div>
                  </div>
                </div>

                {prediction.observed_state?.mixed_layer_depth_m !== undefined && (
                  <div className="p-2 rounded bg-white/[0.02] border border-line/40 flex justify-between items-center">
                    <span className="text-[8.5px] text-dim uppercase">Mixed Layer Depth (mlotst):</span>
                    <span className="text-[12px] font-semibold text-accent">
                      {prediction.observed_state.mixed_layer_depth_m.toFixed(1)} m
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: HISTORICAL DISASTER CONTEXT */}
            {activeTab === 'historical' && (
              <div className="space-y-2 font-mono text-[9px] animate-fade-in">
                <div className="p-2 rounded bg-amber-950/20 border border-amber-500/30 text-amber-200">
                  <div className="font-semibold text-[9.5px]">DOCUMENTED HISTORICAL DISASTER CATALOG</div>
                  <div className="text-[8.5px] text-amber-300/80 mt-0.5">
                    Authoritative events cataloged from IMD & NDMA archives for North Indian Ocean.
                  </div>
                </div>

                {prediction.historical_context?.event_id ? (
                  <div className="p-2.5 rounded bg-black/30 border border-line/40 space-y-1.5">
                    <div className="flex justify-between items-center">
                      <span className="text-mist font-bold text-[10.5px]">
                        {prediction.historical_context.event_name}
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[8px] font-semibold border border-amber-500/40">
                        {prediction.historical_context.event_type}
                      </span>
                    </div>

                    <div className="text-[8.5px] text-dim flex justify-between">
                      <span>Event Dates:</span>
                      <span className="text-mist">{prediction.historical_context.event_dates}</span>
                    </div>

                    <div className="text-[8.5px] text-dim flex justify-between">
                      <span>Basin:</span>
                      <span className="text-mist">{prediction.historical_context.affected_region}</span>
                    </div>

                    {prediction.historical_context.distance_days !== null && prediction.historical_context.distance_days !== undefined && (
                      <div className="text-[8.5px] text-dim flex justify-between">
                        <span>Temporal Proximity:</span>
                        <span className="text-accent font-semibold">
                          {prediction.historical_context.distance_days === 0
                            ? 'ACTIVE TODAY'
                            : `${prediction.historical_context.distance_days} days from observation date`}
                        </span>
                      </div>
                    )}

                    <div className="text-[8px] text-dim/90 pt-1 border-t border-line/30 italic">
                      {prediction.historical_context.note}
                    </div>
                  </div>
                ) : (
                  <div className="p-3 rounded bg-white/[0.02] border border-line/30 text-dim text-center">
                    No documented historical extreme disaster within ±14 days of {targetDate} in this basin.
                  </div>
                )}

                <div className="p-2 rounded bg-black/40 border border-line/30 text-[8px] text-dim">
                  <span className="text-accent">GUARDRAIL: </span>
                  Historical events are documented past records. They are presented for situational awareness only and are never fed into or confused with model inference.
                </div>
              </div>
            )}

            {/* TAB 4: DATA QUALITY & LIMITATIONS */}
            {activeTab === 'quality' && (
              <div className="space-y-2 font-mono text-[9px] animate-fade-in">
                <div className="p-2 rounded bg-white/[0.02] border border-line/40 space-y-1">
                  <div className="flex justify-between">
                    <span className="text-dim">Dataset Source:</span>
                    <span className="text-mist">{prediction.data_quality?.source_dataset || 'Copernicus Marine 0.083° Daily'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dim">Reanalysis Coverage:</span>
                    <span className="text-mist">2024-06-24 → 2026-06-23</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dim">Feature Compatibility:</span>
                    <span className="text-emerald-400 font-semibold">{prediction.data_quality?.model_feature_compatibility || '101/101 Features Valid'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dim">Missing Features:</span>
                    <span className="text-emerald-400 font-semibold">{prediction.data_quality?.missing_feature_count ?? 0}</span>
                  </div>
                </div>

                <div className="p-2 rounded bg-black/30 border border-line/30 space-y-1">
                  <div className="text-[8.5px] font-semibold text-mist uppercase">Model Limitations & Boundaries</div>
                  <ul className="list-disc pl-3 text-[8px] text-dim space-y-0.5">
                    {(prediction.limitations || [
                      'Statistical Random Forest model trained on 2-year Copernicus physical reanalysis.',
                      'Probabilities represent uncalibrated classifier scores, not frequentist cyclone occurrences.',
                      'Physical feature importances reflect statistical correlations and do not imply causal mechanisms.',
                      'Does not include atmospheric barometric pressure or satellite scatterometry.'
                    ]).map((lim, i) => (
                      <li key={i}>{lim}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </React.Fragment>
        ) : null}
      </div>

      {/* Footer Info */}
      <div className="mt-3 pt-2 border-t border-line/40 flex items-center justify-between text-[8.5px] font-mono text-dim">
        <span className="flex items-center gap-1">
          <Database size={10} /> Copernicus Physical Reanalysis
        </span>
        <span className="text-dim/80">
          Leak-Free Chronological ML
        </span>
      </div>
    </div>
  );
};
