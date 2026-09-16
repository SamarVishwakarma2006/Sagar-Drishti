import React, { useState, useEffect } from 'react';
import { ForecastAPI, DemoScenarioItem, ForwardPredictionRes, ValidationRes, ThreatAssessment } from '../services/api';
import { globeRegistry, toast } from '../store/oceanStore';
import {
  Zap,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Globe2,
  X,
  Info,
  Copy,
} from 'lucide-react';

function getClientThreatAssessment(vmax: number | null | undefined, threatAssessment?: ThreatAssessment): ThreatAssessment {
  if (threatAssessment) {
    return threatAssessment;
  }
  if (vmax === null || vmax === undefined || isNaN(vmax)) {
    return {
      threat_level: 'UNKNOWN',
      short_threat_level: 'UNKNOWN',
      intensity_band: 'N/A',
      threat_basis: 'Predicted Vmax (Missing)',
      symbol: '⚪',
      color: 'gray',
      description: 'Intensity value unavailable for threat classification.',
      predicted_vmax_kt: null,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'UNKNOWN',
    };
  }

  if (vmax < 34.0) {
    return {
      threat_level: 'LOW / WEAK SYSTEM',
      short_threat_level: 'LOW',
      intensity_band: '< 34 kt',
      threat_basis: 'Predicted Vmax',
      symbol: '🟢',
      color: 'emerald',
      description: 'Sub-cyclonic or weak tropical system (< 34 kt). Low operational threat potential.',
      predicted_vmax_kt: vmax,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'LOW',
    };
  } else if (vmax < 48.0) {
    return {
      threat_level: 'WATCH',
      short_threat_level: 'WATCH',
      intensity_band: '34–47 kt',
      threat_basis: 'Predicted Vmax',
      symbol: '🟡',
      color: 'amber',
      description: 'Depression to Deep Depression intensity (34–47 kt). System under meteorological watch.',
      predicted_vmax_kt: vmax,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'WATCH',
    };
  } else if (vmax < 64.0) {
    return {
      threat_level: 'MODERATE THREAT',
      short_threat_level: 'MODERATE THREAT',
      intensity_band: '48–63 kt',
      threat_basis: 'Predicted Vmax',
      symbol: '🟠',
      color: 'orange',
      description: 'Cyclonic Storm intensity (48–63 kt). Moderate wind and maritime threat.',
      predicted_vmax_kt: vmax,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'MODERATE THREAT',
    };
  } else if (vmax < 83.0) {
    return {
      threat_level: 'HIGH THREAT',
      short_threat_level: 'HIGH THREAT',
      intensity_band: '64–82 kt',
      threat_basis: 'Predicted Vmax',
      symbol: '🔴',
      color: 'red',
      description: 'Severe to Very Severe Cyclonic Storm (64–82 kt). High structural and coastal threat.',
      predicted_vmax_kt: vmax,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'HIGH THREAT',
    };
  } else if (vmax < 96.0) {
    return {
      threat_level: 'SEVERE THREAT',
      short_threat_level: 'SEVERE THREAT',
      intensity_band: '83–95 kt',
      threat_basis: 'Predicted Vmax',
      symbol: '🔴',
      color: 'purple',
      description: 'Extremely Severe Cyclonic Storm (83–95 kt). Severe life-threatening conditions.',
      predicted_vmax_kt: vmax,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'SEVERE THREAT',
    };
  } else {
    return {
      threat_level: 'CRITICAL THREAT',
      short_threat_level: 'CRITICAL THREAT',
      intensity_band: '>= 96 kt',
      threat_basis: 'Predicted Vmax',
      symbol: '⚡',
      color: 'rose',
      description: 'Super Cyclonic Storm (>= 96 kt). Critical, catastrophic wind and surge threat.',
      predicted_vmax_kt: vmax,
      calibrated_probability: null,
      probability_label: 'N/A',
      risk_basis: 'INTENSITY ONLY',
      risk_level: 'CRITICAL THREAT',
    };
  }
}

interface ForwardPredictionPanelProps {
  onClose: () => void;
}

export const ForwardPredictionPanel: React.FC<ForwardPredictionPanelProps> = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState<'demo' | 'custom'>('demo');
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
                  <div className="mt-1.5 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-[8.5px] font-mono text-amber-300 font-semibold tracking-wider">
                    SYNTHETIC TEST FIXTURE — NOT REAL METEOROLOGICAL DATA
                  </div>
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
                OBSERVATION ID &amp; TIME (T):
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

        {/* PRE-FLIGHT CAUSAL CHECKLIST */}
        {validation && (
          <div className="p-3 rounded-lg bg-black/50 border border-cyan-500/20 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] font-semibold text-cyan-300 flex items-center gap-1.5">
                <ShieldCheck size={13} />
                FORECAST VALIDATION CHECK
              </span>
              <span
                className={`font-mono text-[9px] px-1.5 py-0.5 rounded ${
                  validation.can_predict
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                }`}
              >
                {validating
                  ? 'VALIDATING...'
                  : validation.can_predict
                  ? 'READY TO FORECAST'
                  : 'BLOCKED'}
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
              <span>GENERATE FORECAST</span>
            </React.Fragment>
          )}
        </button>

        {/* PREDICTION RESULTS CARD */}
        {prediction && (
          <div className="p-3.5 rounded-xl bg-gradient-to-b from-cyan-950/40 to-black/60 border border-cyan-400/40 space-y-3 animate-fade-in">
            {/* 1. FORECAST HEADER */}
            <div className="flex items-center justify-between pb-2 border-b border-cyan-500/20">
              <span className="font-mono text-[10.5px] text-cyan-300 font-bold tracking-wider flex items-center gap-1.5">
                <Zap size={13} className="text-cyan-400" />
                T+24H FORECAST RESULT
              </span>
              <span className="font-mono text-[9px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-200 border border-cyan-400/40 font-semibold tracking-wider">
                {prediction.evaluation_status || 'TARGET_EXCLUDED'}
              </span>
            </div>

            {/* 2. SYNTHETIC TEST FIXTURE WARNING BANNER */}
            {(prediction.input_dataset_source === 'DEMO_FIXTURE' || prediction.synthetic_fixture_label) && (
              <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/40 text-amber-300 font-mono">
                <div className="font-bold text-[9.5px] flex items-center gap-1.5 tracking-wide">
                  <AlertTriangle size={13} className="text-amber-400 shrink-0" />
                  <span>SYNTHETIC TEST FIXTURE — NOT REAL METEOROLOGICAL DATA</span>
                </div>
                <div className="text-[8.5px] text-amber-200/90 mt-1 leading-relaxed">
                  Pipeline &amp; causal testing only. Zero prospective accuracy or real-world forecasting skill is claimed.
                </div>
              </div>
            )}

            {/* 3. PRIMARY FORECAST SUMMARY AREA (DOMINANT VMAX + HORIZON) */}
            <div className="grid grid-cols-2 gap-3 bg-black/50 p-3 rounded-lg border border-cyan-500/30">
              <div>
                <div className="text-[9.5px] font-mono tracking-wider text-dim">PREDICTED INTENSITY (Vmax)</div>
                <div className="text-[28px] font-mono font-bold text-cyan-100 leading-tight mt-0.5">
                  {prediction.predicted_vmax_24h}{' '}
                  <span className="text-[14px] text-cyan-400 font-normal">kt</span>
                </div>
                {prediction.clipping_applied && (
                  <div className="text-[8px] font-mono text-amber-300/90 mt-1">
                    Physical Guard Applied [15–165 kt] (Raw: {prediction.raw_predicted_vmax_24h} kt)
                  </div>
                )}
              </div>
              <div className="text-right flex flex-col justify-between">
                <div className="text-[9.5px] font-mono tracking-wider text-dim">HORIZON</div>
                <div className="font-mono text-[20px] font-bold text-cyan-300 leading-tight">
                  T + 24.0h
                </div>
                <div className="text-[8px] font-mono text-dim mt-1">
                  Valid: {prediction.valid_time?.replace('T', ' ').replace('Z', '')}
                </div>
              </div>
            </div>

            {/* 4. FORECAST THREAT ASSESSMENT (REASONING & BASIS) */}
            {(() => {
              const threat = getClientThreatAssessment(prediction.predicted_vmax_24h, prediction.threat_assessment);
              const isSynthetic = prediction.input_dataset_source === 'DEMO_FIXTURE' || Boolean(prediction.synthetic_fixture_label);
              
              const colorStyles: Record<string, { badge: string; border: string; bg: string; text: string }> = {
                emerald: {
                  badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
                  border: 'border-emerald-500/30',
                  bg: 'from-emerald-950/30 to-black/40',
                  text: 'text-emerald-400',
                },
                amber: {
                  badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
                  border: 'border-amber-500/30',
                  bg: 'from-amber-950/30 to-black/40',
                  text: 'text-amber-400',
                },
                orange: {
                  badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
                  border: 'border-orange-500/30',
                  bg: 'from-orange-950/30 to-black/40',
                  text: 'text-orange-400',
                },
                red: {
                  badge: 'bg-red-500/20 text-red-300 border-red-500/40 shadow-red-950/50',
                  border: 'border-red-500/30',
                  bg: 'from-red-950/30 to-black/40',
                  text: 'text-red-400',
                },
                purple: {
                  badge: 'bg-rose-500/25 text-rose-200 border-rose-500/50 shadow-rose-950/50',
                  border: 'border-rose-500/40',
                  bg: 'from-rose-950/35 to-black/40',
                  text: 'text-rose-300',
                },
                rose: {
                  badge: 'bg-rose-600/30 text-rose-100 border-rose-400 animate-pulse',
                  border: 'border-rose-500/50',
                  bg: 'from-rose-950/40 to-black/50',
                  text: 'text-rose-400',
                },
                gray: {
                  badge: 'bg-gray-500/20 text-gray-300 border-gray-500/40',
                  border: 'border-gray-500/30',
                  bg: 'from-gray-950/30 to-black/40',
                  text: 'text-gray-400',
                },
              };

              const style = colorStyles[threat.color] || colorStyles.gray;

              return (
                <div className={`p-3 rounded-lg bg-gradient-to-b ${style.bg} border ${style.border} space-y-2.5`}>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono text-mist font-bold tracking-wider flex items-center gap-1.5">
                      <ShieldAlert size={13} className={style.text} />
                      FORECAST THREAT ASSESSMENT
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${style.badge} flex items-center gap-1`}>
                      <span>{threat.symbol}</span>
                      <span>{threat.threat_level}</span>
                    </span>
                  </div>

                  {/* Subordinate test fixture notice if synthetic */}
                  {isSynthetic && (
                    <div className="text-[8px] font-mono text-amber-300/90 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/30 flex items-center gap-1">
                      <span>⚡ SIMULATED THREAT INTERPRETATION — TEST FIXTURE ONLY</span>
                    </div>
                  )}

                  {/* Threat Reasoning Grid (Clear & Uncluttered) */}
                  <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-[9px] border-t border-white/10">
                    <div className="p-1.5 rounded bg-black/30 border border-white/5">
                      <div className="text-[8px] text-dim uppercase tracking-wider">Intensity Band</div>
                      <div className="text-mist font-semibold mt-0.5">{threat.intensity_band}</div>
                    </div>
                    <div className="p-1.5 rounded bg-black/30 border border-white/5">
                      <div className="text-[8px] text-dim uppercase tracking-wider">Threat Basis</div>
                      <div className="text-cyan-300 font-semibold mt-0.5">{threat.threat_basis}</div>
                    </div>
                    <div className="p-1.5 rounded bg-black/30 border border-white/5">
                      <div className="text-[8px] text-dim uppercase tracking-wider">Probability</div>
                      <div className={`mt-0.5 ${threat.calibrated_probability !== null ? 'text-cyan-200 font-semibold' : 'text-dim'}`}>
                        {threat.calibrated_probability !== null ? `${(threat.calibrated_probability * 100).toFixed(1)}%` : (threat.probability_label || 'N/A')}
                      </div>
                    </div>
                    <div className="p-1.5 rounded bg-black/30 border border-white/5">
                      <div className="text-[8px] text-dim uppercase tracking-wider">Risk Assessment</div>
                      <div className="text-amber-200/90 font-semibold mt-0.5">{threat.risk_basis === 'INTENSITY ONLY' ? 'BASED ON INTENSITY' : (threat.risk_basis || 'BASED ON INTENSITY')}</div>
                    </div>
                  </div>

                  {threat.description && (
                    <div className="text-[8.5px] font-mono text-dim/90 leading-snug pt-0.5">
                      {threat.description}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* 5. FORECAST METADATA (PROVENANCE & REPRODUCIBILITY) */}
            <div className="p-2.5 rounded-lg bg-black/40 border border-line/50 space-y-1.5">
              <div className="text-[9px] font-mono font-bold tracking-wider text-cyan-300 flex items-center justify-between pb-1 border-b border-line/40">
                <span>FORECAST METADATA</span>
                <span className="text-[8px] font-normal text-dim">PROVENANCE &amp; FORENSICS</span>
              </div>
              <div className="space-y-1 text-[9px] font-mono">
                <div className="flex justify-between items-center">
                  <span className="text-dim">System ID:</span>
                  <span className="text-cyan-200 font-medium">{prediction.system_id}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dim">Forecast Origin (T):</span>
                  <span className="text-cyan-200 font-medium">{prediction.origin}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dim">Valid Time (T+24h):</span>
                  <span className="text-cyan-200 font-medium">{prediction.valid_time}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dim">Data Provenance:</span>
                  <span className={`px-1.5 py-0.2 rounded text-[8px] font-semibold tracking-wider ${
                    prediction.data_provenance_class === 'REAL_PROSPECTIVE'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  }`}>
                    {prediction.data_provenance_class || 'UNKNOWN'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dim">Atmos Source:</span>
                  <span className="text-cyan-200 font-medium">{prediction.atmos_source_id || 'OPERATIONAL_NWP'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-dim">Model Version:</span>
                  <span className="text-cyan-200 font-medium">{prediction.model_version}</span>
                </div>
                <div className="pt-1 border-t border-line/30 flex flex-col gap-0.5">
                  <div className="flex justify-between items-center">
                    <span className="text-dim">Model SHA-256:</span>
                    <button
                      onClick={() => {
                        try {
                          navigator.clipboard.writeText(prediction.model_hash);
                          toast('Copied full Model SHA-256 hash');
                        } catch {
                          toast('Clipboard unavailable');
                        }
                      }}
                      title="Click to copy full Model SHA-256 hash"
                      className="text-cyan-400 hover:text-cyan-200 flex items-center gap-1 text-[8px] transition-colors"
                    >
                      <Copy size={10} />
                      <span>COPY HASH</span>
                    </button>
                  </div>
                  <span
                    className="text-[8px] text-cyan-200/90 font-mono break-all select-all bg-black/50 p-1.5 rounded border border-white/5 leading-tight"
                    title={prediction.model_hash}
                  >
                    {prediction.model_hash}
                  </span>
                </div>
              </div>
            </div>

            {/* 6. VALIDATION / GROUND TRUTH (SCIENTIFICALLY SEGREGATED) */}
            {prediction.target_info && (
              <div className="p-2.5 rounded-lg bg-emerald-950/25 border border-emerald-500/40 space-y-1 font-mono text-[9px]">
                <div className="flex items-center justify-between pb-1 border-b border-emerald-500/20">
                  <span className="font-bold text-emerald-300 flex items-center gap-1">
                    <CheckCircle2 size={12} className="text-emerald-400" />
                    VALIDATION / GROUND TRUTH
                  </span>
                  <span className="px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[8px] font-semibold">
                    OBSERVED EVENT MATCHED
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1">
                  <div>
                    <span className="text-dim">Observed Vmax:</span>{' '}
                    <span className="text-emerald-200 font-bold text-[11px] ml-1">
                      {prediction.target_info.observed_vmax_24h} kt
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-dim">Actual Time:</span>{' '}
                    <span className="text-emerald-200 font-medium ml-1">
                      {prediction.target_info.actual_target_timestamp?.replace('T', ' ').replace('Z', '')}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* 7. CONTROLS / 3D GLOBE INTEGRATION */}
            {prediction.latitude && prediction.longitude && (
              <button
                onClick={handleViewOnGlobe}
                className="w-full py-2 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-200 border border-cyan-400/30 font-mono text-[10px] font-semibold flex items-center justify-center gap-2 transition-colors shadow-sm"
              >
                <Globe2 size={13} />
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
