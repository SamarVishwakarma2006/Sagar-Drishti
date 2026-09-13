import React, { useState, useEffect } from 'react';
import { ForecastAPI, DemoScenarioItem, ForwardPredictionRes, ValidationRes } from '../services/api';
import { globeRegistry, toast } from '../store/oceanStore';
import {
  Zap,
  ShieldCheck,
  AlertTriangle,
  Compass,
  CheckCircle2,
  XCircle,
  Clock,
  Globe2,
  FileText,
  HelpCircle,
  X,
  Sparkles,
  Info,
} from 'lucide-react';

interface ForwardPredictionPanelProps {
  onClose: () => void;
}

export const ForwardPredictionPanel: React.FC<ForwardPredictionPanelProps> = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState<'demo' | 'custom' | 'history'>('demo');
  const [scenarios, setScenarios] = useState<DemoScenarioItem[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('DEMO-BOB-2026-07-10');
  const [customOrigin, setCustomOrigin] = useState<string>('2026-07-10T12:00:00Z');
  const [customSystemId, setCustomSystemId] = useState<string>('CUSTOM-BOB-01');
  const [customJson, setCustomJson] = useState<string>(
    JSON.stringify(
      {
        vmax_current: 55.0,
        dvmax_6h: 7.0,
        dvmax_12h: 12.0,
        dvmax_24h: 18.0,
        pc_current: 984.0,
        dpc_6h: -4.0,
        translation_speed_kts: 9.5,
        latitude_current: 16.2,
        longitude_current: 87.5,
        vws_env_mean_200_800km: 11.5,
        vws_env_min_200_800km: 6.8,
        vws_core_mean_0_100km: 8.4,
        vort_core_mean_0_100km: 44.0,
        vort_core_max_0_100km: 70.0,
        vort_env_mean_200_800km: 17.0,
        rh_700_env_mean_200_800km: 74.0,
        rh_700_env_min_200_800km: 60.0,
        rh_700_core_mean_0_100km: 86.0,
        rh_500_env_mean_200_800km: 66.0,
        rh_500_core_mean_0_100km: 78.0,
        sst_core_mean_0_100km: 30.0,
        sst_env_mean_200_800km: 29.4,
        mld_core_mean_0_100km: 40.0,
        sla_core_mean_0_100km: 0.14,
      },
      null,
      2
    )
  );

  const [validation, setValidation] = useState<ValidationRes | null>(null);
  const [prediction, setPrediction] = useState<ForwardPredictionRes | null>(null);
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);
  const [history, setHistory] = useState<any[]>([]);

  // Load demo scenarios on mount
  useEffect(() => {
    ForecastAPI.getDemoScenarios()
      .then((data) => {
        setScenarios(data);
        if (data.length > 0) {
          setSelectedScenarioId(data[0].scenario_id);
        }
      })
      .catch((err) => {
        console.warn('Failed to load demo scenarios:', err);
      });

    ForecastAPI.getHistory(20)
      .then((data) => setHistory(data))
      .catch(() => {});
  }, []);

  // Pre-flight validate whenever scenario or custom changes
  const runValidation = async () => {
    setValidating(true);
    try {
      if (activeTab === 'demo') {
        const res = await ForecastAPI.validateInputs({
          system_id: selectedScenarioId,
          forecast_origin_timestamp:
            scenarios.find((s) => s.scenario_id === selectedScenarioId)?.forecast_origin_timestamp ||
            '2026-07-10T12:00:00Z',
          demo_scenario_id: selectedScenarioId,
        });
        setValidation(res);
      } else {
        const parsedFeatures = JSON.parse(customJson);
        const res = await ForecastAPI.validateInputs({
          system_id: customSystemId,
          forecast_origin_timestamp: customOrigin,
          features: parsedFeatures,
        });
        setValidation(res);
      }
    } catch (e: any) {
      toast(`Validation error: ${e.message}`);
    } finally {
      setValidating(false);
    }
  };

  useEffect(() => {
    runValidation();
  }, [selectedScenarioId, activeTab]);

  const handlePredict = async () => {
    setLoading(true);
    try {
      let res: ForwardPredictionRes;
      if (activeTab === 'demo') {
        const scen = scenarios.find((s) => s.scenario_id === selectedScenarioId);
        res = await ForecastAPI.predictForward({
          system_id: scen?.system_id || selectedScenarioId,
          forecast_origin_timestamp: scen?.forecast_origin_timestamp || '2026-07-10T12:00:00Z',
          demo_scenario_id: selectedScenarioId,
        });
      } else {
        const parsedFeatures = JSON.parse(customJson);
        res = await ForecastAPI.predictForward({
          system_id: customSystemId,
          forecast_origin_timestamp: customOrigin,
          features: parsedFeatures,
        });
      }

      setPrediction(res);
      toast(`Forward Prediction Generated: ${res.predicted_vmax_24h} kts`);

      // Refresh history
      ForecastAPI.getHistory(20).then(setHistory).catch(() => {});

      // Plot to Cesium 3D Globe if coordinate present
      if (res.latitude && res.longitude && globeRegistry.g) {
        globeRegistry.g.markForwardPrediction({
          system_id: res.system_id,
          origin: res.origin,
          valid_time: res.valid_time,
          predicted_vmax_24h: res.predicted_vmax_24h ?? 0,
          lat: res.latitude,
          lon: res.longitude,
        });
      }
    } catch (e: any) {
      toast(`Prediction failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleViewOnGlobe = () => {
    if (!prediction || !prediction.latitude || !prediction.longitude || !globeRegistry.g) {
      toast('No coordinates associated with this prediction.');
      return;
    }
    globeRegistry.g.markForwardPrediction({
      system_id: prediction.system_id,
      origin: prediction.origin,
      valid_time: prediction.valid_time,
      predicted_vmax_24h: prediction.predicted_vmax_24h ?? 0,
      lat: prediction.latitude,
      lon: prediction.longitude,
    });
    toast('Camera centered on Forecast Origin Beacon.');
  };

  return (
    <div className="absolute top-14 right-4 z-40 w-[420px] max-h-[calc(100vh-80px)] flex flex-col bg-[#040c16]/95 backdrop-blur-xl border border-cyan-500/30 rounded-xl shadow-2xl overflow-hidden animate-fade-in font-disp">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-cyan-500/20 bg-cyan-950/30">
        <div className="flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 animate-pulse">
            <Zap size={16} />
          </span>
          <div>
            <div className="text-[13px] font-semibold tracking-wider text-cyan-100 flex items-center gap-2">
              FORECAST MODE
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-400/30">
                PROSPECTIVE
              </span>
            </div>
            <div className="text-[9px] font-mono text-cyan-400/70">
              FROZEN MODEL INFERENCE (T &gt; 2026-06-23)
            </div>
          </div>
        </div>
        <button
          onClick={() => {
            globeRegistry.g?.clearForwardPrediction();
            onClose();
          }}
          className="text-dim hover:text-mist p-1 rounded hover:bg-white/10 transition-colors"
          title="Close Forecast Mode"
        >
          <X size={16} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-line bg-black/40 text-[10px] font-mono">
        <button
          onClick={() => setActiveTab('demo')}
          className={`flex-1 py-2 text-center transition-colors border-b-2 ${
            activeTab === 'demo'
              ? 'border-cyan-400 text-cyan-200 bg-cyan-950/20 font-semibold'
              : 'border-transparent text-dim hover:text-mist'
          }`}
        >
          DEMO SCENARIOS
        </button>
        <button
          onClick={() => setActiveTab('custom')}
          className={`flex-1 py-2 text-center transition-colors border-b-2 ${
            activeTab === 'custom'
              ? 'border-cyan-400 text-cyan-200 bg-cyan-950/20 font-semibold'
              : 'border-transparent text-dim hover:text-mist'
          }`}
        >
          CUSTOM INGESTION
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`flex-1 py-2 text-center transition-colors border-b-2 ${
            activeTab === 'history'
              ? 'border-cyan-400 text-cyan-200 bg-cyan-950/20 font-semibold'
              : 'border-transparent text-dim hover:text-mist'
          }`}
        >
          AUDIT LOG ({history.length})
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-[11px]">
        {/* TAB 1: DEMO SCENARIOS */}
        {activeTab === 'demo' && (
          <div className="space-y-3">
            <label className="block text-[10px] font-mono tracking-wider text-cyan-300">
              SELECT POST-HISTORICAL TEST FIXTURE:
            </label>
            <div className="grid grid-cols-1 gap-2">
              {scenarios.map((s) => (
                <button
                  key={s.scenario_id}
                  onClick={() => setSelectedScenarioId(s.scenario_id)}
                  className={`text-left p-2.5 rounded-lg border transition-all ${
                    selectedScenarioId === s.scenario_id
                      ? 'border-cyan-400 bg-cyan-950/40 text-cyan-100 shadow-md ring-1 ring-cyan-400/40'
                      : 'border-line/60 bg-white/[0.02] text-dim hover:border-line hover:text-mist'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[11.5px] text-cyan-200">
                      {s.system_name}
                    </span>
                    <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300">
                      {s.basin}
                    </span>
                  </div>
                  <div className="text-[10px] text-dim mt-1 line-clamp-2">{s.description}</div>
                  <div className="flex items-center gap-3 mt-2 font-mono text-[9px] text-cyan-400/80">
                    <span>Origin: {s.forecast_origin_timestamp.replace('T', ' ').replace('Z', '')}</span>
                    <span>Vmax(T): {s.latest_observed_vmax} kt</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* TAB 2: CUSTOM INGESTION */}
        {activeTab === 'custom' && (
          <div className="space-y-3">
            <div>
              <label className="block text-[9px] font-mono text-cyan-300 mb-1">
                SYSTEM IDENTIFIER &amp; ORIGIN TIMESTAMP (T):
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customSystemId}
                  onChange={(e) => setCustomSystemId(e.target.value)}
                  className="flex-1 bg-black/60 border border-line rounded px-2 py-1 text-[11px] font-mono text-cyan-100 outline-none focus:border-cyan-400"
                  placeholder="e.g. INCOIS-NEW-2026"
                />
                <input
                  type="text"
                  value={customOrigin}
                  onChange={(e) => setCustomOrigin(e.target.value)}
                  className="flex-1 bg-black/60 border border-line rounded px-2 py-1 text-[11px] font-mono text-cyan-100 outline-none focus:border-cyan-400"
                  placeholder="YYYY-MM-DDTHH:MM:SSZ"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-[9px] font-mono text-cyan-300">
                  FEATURE VECTOR / OBSERVATION JSON:
                </label>
                <button
                  onClick={runValidation}
                  className="text-[9px] font-mono text-cyan-400 hover:underline"
                >
                  Verify Contract
                </button>
              </div>
              <textarea
                rows={8}
                value={customJson}
                onChange={(e) => setCustomJson(e.target.value)}
                className="w-full bg-black/70 border border-line rounded p-2 text-[10px] font-mono text-cyan-200 outline-none focus:border-cyan-400 resize-none"
              />
            </div>
          </div>
        )}

        {/* TAB 3: HISTORY */}
        {activeTab === 'history' && (
          <div className="space-y-2">
            <div className="text-[10px] font-mono text-dim mb-1">
              IMMUTABLE APPEND-ONLY LOG (forecast_log.parquet):
            </div>
            {history.length === 0 ? (
              <div className="text-dim text-center py-6 font-mono">No predictions logged yet.</div>
            ) : (
              history.map((h, i) => (
                <div
                  key={i}
                  className="p-2 rounded border border-line/60 bg-white/[0.02] flex items-center justify-between"
                >
                  <div>
                    <div className="font-semibold text-cyan-200">{h.system_id}</div>
                    <div className="text-[9px] font-mono text-dim">
                      Origin: {h.forecast_origin_timestamp}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-bold text-cyan-300 text-[12px]">
                      {h.forecast_vmax_24h} kt
                    </div>
                    <span className="text-[8px] font-mono px-1 rounded bg-cyan-500/20 text-cyan-400">
                      T+24h
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* PRE-FLIGHT CAUSAL CHECKLIST */}
        {activeTab !== 'history' && validation && (
          <div className="p-3 rounded-lg bg-black/50 border border-cyan-500/20 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] font-semibold text-cyan-300 flex items-center gap-1.5">
                <ShieldCheck size={13} />
                PRE-FLIGHT CAUSAL CHECKLIST
              </span>
              <span
                className={`font-mono text-[9px] px-1.5 py-0.5 rounded ${
                  validation.can_predict
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                }`}
              >
                {validation.can_predict ? 'READY FOR INFERENCE' : 'BLOCKED'}
              </span>
            </div>

            <div className="space-y-1.5 pt-1">
              {validation.checks.map((c, i) => (
                <div key={i} className="flex items-start gap-2 text-[9.5px]">
                  {c.status === 'PASS' ? (
                    <CheckCircle2 size={12} className="text-emerald-400 shrink-0 mt-0.5" />
                  ) : c.status === 'WARNING' ? (
                    <AlertTriangle size={12} className="text-amber-400 shrink-0 mt-0.5" />
                  ) : (
                    <XCircle size={12} className="text-rose-400 shrink-0 mt-0.5" />
                  )}
                  <div className="leading-tight">
                    <span className="font-mono text-cyan-200">{c.check}: </span>
                    <span className="text-dim">{c.detail}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="pt-2 flex items-center justify-between text-[9px] font-mono text-dim border-t border-line/40">
              <span>Feature Completeness: {(validation.feature_completeness * 100).toFixed(0)}%</span>
              <span>Available: {validation.available_features.length}/29</span>
            </div>
          </div>
        )}

        {/* INFERENCE ACTION */}
        {activeTab !== 'history' && (
          <button
            onClick={handlePredict}
            disabled={loading || (validation !== null && !validation.can_predict)}
            className={`w-full py-2.5 rounded-lg font-mono text-[11px] font-semibold tracking-wider flex items-center justify-center gap-2 shadow-lg transition-all ${
              loading || (validation !== null && !validation.can_predict)
                ? 'bg-line/40 text-dim cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-900/40 active:scale-[0.99]'
            }`}
          >
            {loading ? (
              <React.Fragment>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>EVALUATING FROZEN MODEL...</span>
              </React.Fragment>
            ) : (
              <React.Fragment>
                <Zap size={14} />
                <span>GENERATE FORWARD PREDICTION</span>
              </React.Fragment>
            )}
          </button>
        )}

        {/* PREDICTION RESULTS CARD */}
        {prediction && (
          <div className="p-3.5 rounded-xl bg-gradient-to-b from-cyan-950/40 to-black/60 border border-cyan-400/40 space-y-3 animate-fade-in">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] text-cyan-300 font-semibold tracking-wider">
                ⚡ T+24H FORECAST RESULT
              </span>
              <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-400/30">
                {prediction.evaluation_status}
              </span>
            </div>

            <div className="flex items-baseline justify-between bg-black/40 p-2.5 rounded-lg border border-cyan-500/20">
              <div>
                <div className="text-[10px] font-mono text-dim">PREDICTED INTENSITY (Vmax)</div>
                <div className="text-[26px] font-mono font-bold text-cyan-200 leading-tight">
                  {prediction.predicted_vmax_24h}{' '}
                  <span className="text-[14px] text-cyan-400">kts</span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-[9px] font-mono text-dim">HORIZON</div>
                <div className="font-mono font-bold text-cyan-400">T + 24.0h</div>
              </div>
            </div>

            <div className="space-y-1 text-[9.5px] font-mono text-dim">
              <div className="flex justify-between">
                <span>System ID:</span>
                <span className="text-cyan-200">{prediction.system_id}</span>
              </div>
              <div className="flex justify-between">
                <span>Forecast Origin (T):</span>
                <span className="text-cyan-200">{prediction.origin}</span>
              </div>
              <div className="flex justify-between">
                <span>Valid Time (T+24h):</span>
                <span className="text-cyan-200">{prediction.valid_time}</span>
              </div>
              <div className="flex justify-between">
                <span>Model Version:</span>
                <span className="text-cyan-200">{prediction.model_version}</span>
              </div>
              <div className="flex justify-between">
                <span>Model SHA-256:</span>
                <span className="text-cyan-200">{prediction.model_hash.slice(0, 12)}...</span>
              </div>
            </div>

            {/* Target Match Outcome if available */}
            {prediction.target_info && (
              <div className="p-2 rounded bg-emerald-950/20 border border-emerald-500/30 text-[9.5px] font-mono text-emerald-200">
                <div className="font-semibold mb-0.5">GROUND TRUTH FIX MATCHED:</div>
                <div>Observed Vmax: {prediction.target_info.observed_vmax_24h} kts</div>
                <div>Actual Time: {prediction.target_info.actual_target_timestamp}</div>
              </div>
            )}

            {/* View on Globe Button */}
            {prediction.latitude && prediction.longitude && (
              <button
                onClick={handleViewOnGlobe}
                className="w-full py-1.5 rounded bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-200 border border-cyan-400/30 font-mono text-[10px] flex items-center justify-center gap-1.5 transition-colors"
              >
                <Globe2 size={12} />
                <span>CENTER ON 3D GLOBE (BEACON ACTIVE)</span>
              </button>
            )}
          </div>
        )}

        {/* SCIENTIFIC DISCLAIMER (MANDATORY NON-NEGOTIABLE) */}
        <div className="p-2.5 rounded-lg bg-white/[0.02] border border-line/60 flex items-start gap-2 text-[9px] text-dim leading-relaxed">
          <Info size={14} className="text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-mist">Scientific Notice: </span>
            Forward inference uses the frozen research model on user-provided or newly available
            observations. This forecast is not itself proof of prospective generalization; future
            outcomes are evaluated separately through the prospective validation framework.
          </div>
        </div>
      </div>
    </div>
  );
};
