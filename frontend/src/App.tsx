import React, { useState, useEffect, useRef } from 'react';
import { useApp, store, globeRegistry, uwRegistry, toast } from './store/oceanStore';
import { TopBar } from './components/TopBar';
import { SiteExplorer } from './components/SiteExplorer';
import { SitePanel } from './components/SitePanel';
import { DataIngestionPanel } from './components/DataIngestionPanel';
import { HistoricalDisasterSelector } from './components/HistoricalDisasterSelector';
import { ForwardPredictionPanel } from './components/ForwardPredictionPanel';
import { Inspector } from './components/Inspector';
import { ColorbarControls } from './components/ColorbarControls';
import { SagarBot } from './components/SagarBot';
import { DepthControl } from './components/DepthControl';
import { Timeline } from './components/Timeline';
import { CompassHud } from './components/CompassHud';
import { CoordReadout } from './components/CoordReadout';
import { Veil } from './components/Veil';
import { Boot } from './components/Boot';
import { GlobeViewer } from './components/GlobeViewer';
import { UnderwaterViewer } from './components/UnderwaterViewer';
import { clamp, fmtLat, fmtLon, fmtDepth } from './services/syntheticOcean';
import { ArrowUpToLine, RotateCcw } from 'lucide-react';

export const App: React.FC = () => {
  const s = useApp();
  const [veilBg, setVeilBg] = useState(false);
  const [veilText, setVeilText] = useState('');
  const [bootMsg, setBootMsg] = useState('INITIALIZING SAGAR DRISHTI OBSERVATORY...');
  const [showColorbar, setShowColorbar] = useState(false);
  const [showDisasters, setShowDisasters] = useState(false);
  const [showForecast, setShowForecast] = useState(false);
  const skipRef = useRef<() => void>(() => {});

  // Dive orchestration: Globe -> Dive -> Underwater
  useEffect(() => {
    if (s.phase !== 'diving' || !s.site) return;
    const g = globeRegistry.g;
    const site = s.site;
    if (g) {
      g.diveTo(site, () => {});
    }

    setVeilText(`DESCENDING · ${site.name.toUpperCase()}`);
    const t1 = setTimeout(() => setVeilBg(true), 3300);

    const complete = () => {
      if (store.get().phase !== 'diving') return;
      store.set({ phase: 'underwater', depth: 60, selection: null });
      g?.setPaused(true);
      setVeilBg(false);
    };

    skipRef.current = complete;
    const t2 = setTimeout(complete, 5400);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [s.phase, s.site?.id]);

  // Ascend orchestration: Underwater -> Ascend -> Globe
  const handleAscend = () => {
    if (!s.site) return;
    store.set({ phase: 'ascending', selection: null });
    setVeilText('SURFACING TO PLANETARY ORBIT');
  };

  useEffect(() => {
    if (s.phase !== 'ascending') return;
    setVeilBg(true);

    const t1 = setTimeout(() => {
      const g = globeRegistry.g;
      if (g) {
        g.setPaused(false);
        g.ascendFrom(store.get().site!);
      }
      store.set({ phase: 'globe' });
    }, 750);

    const t2 = setTimeout(() => setVeilBg(false), 3400);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [s.phase]);

  // Global Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const st = store.get();
      if (e.key === 'Escape') {
        store.set({ selection: null });
        setShowColorbar(false);
      }
      if (st.phase === 'underwater' && st.site) {
        if (e.key === 'ArrowUp') {
          store.set({
            depth: clamp(st.depth + (e.shiftKey ? 100 : 10), 8, st.site.maxDepth),
          });
        }
        if (e.key === 'ArrowDown') {
          store.set({
            depth: clamp(st.depth - (e.shiftKey ? 100 : 10), 8, st.site.maxDepth),
          });
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Toast auto-clear
  useEffect(() => {
    if (!s.toast) return;
    const t = setTimeout(() => store.set({ toast: null }), 2800);
    return () => clearTimeout(t);
  }, [s.toast?.id]);

  return (
    <div className="fixed inset-0 bg-abyss text-mist font-disp overflow-hidden">
      {/* 3D Global & Volumetric Viewers */}
      <GlobeViewer
        onReady={() => store.set({ phase: 'globe' })}
        onError={(err) => setBootMsg(err)}
      />
      <UnderwaterViewer />

      {/* Boot Splash */}
      {s.phase === 'boot' && <Boot msg={bootMsg} />}

      {/* Navigation TopBar */}
      {s.phase !== 'boot' && s.phase !== 'diving' && s.phase !== 'ascending' && (
        <TopBar
          showColorbar={showColorbar}
          onToggleColorbar={() => setShowColorbar(!showColorbar)}
          showDisasters={showDisasters}
          onToggleDisasters={() => setShowDisasters(!showDisasters)}
          showForecast={showForecast}
          onToggleForecast={() => setShowForecast(!showForecast)}
        />
      )}

      {/* GLOBE PHASE HUD */}
      {s.phase === 'globe' && (
        <React.Fragment>
          <SiteExplorer />
          {s.site && (
            <SitePanel onDive={() => store.set({ phase: 'diving', selection: null })} />
          )}
          <DataIngestionPanel />
          {showDisasters && (
            <HistoricalDisasterSelector onClose={() => setShowDisasters(false)} />
          )}
          {showForecast && (
            <ForwardPredictionPanel onClose={() => setShowForecast(false)} />
          )}

          <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-2 items-end">
            <button
              onClick={() => globeRegistry.g?.resetView()}
              title="Reset planetary camera view"
              className="glass w-8 h-8 flex items-center justify-center text-dim hover:text-accent border border-line hover:border-accent/40 shadow-lg"
            >
              <RotateCcw size={14} />
            </button>
            <CoordReadout />
            <span className="font-mono text-[8px] text-dim/80 text-right max-w-[240px] leading-relaxed">
              CesiumJS 3D Globe · Natural Earth II · INCOIS Live Ingestion Engine
            </span>
          </div>

          <div className="absolute top-16 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
            <span className="font-mono text-[9px] tracking-[0.22em] text-dim bg-abyss/80 border border-line rounded px-3 py-1.5 shadow-lg">
              DRAG TO ORBIT · SCROLL TO ZOOM · SELECT STUDY SITE TO DIVE
            </span>
          </div>
        </React.Fragment>
      )}

      {/* UNDERWATER PHASE HUD */}
      {s.phase === 'underwater' && s.site && (
        <React.Fragment>
          {/* Surface Button & Site Header */}
          <div className="absolute top-14 left-3 md:left-4 z-20 flex items-center gap-2.5">
            <button
              onClick={handleAscend}
              className="glass flex items-center gap-2 px-3 py-2 text-[10px] font-mono tracking-widest text-mist hover:text-accent hover:border-accent/40 border border-line shadow-lg transition-colors"
            >
              <ArrowUpToLine size={13} />
              SURFACE
            </button>
            <div className="leading-tight">
              <div className="text-[12.5px] font-medium text-mist flex items-center gap-2">
                {s.site.name}
              </div>
              <div className="font-mono text-[9px] text-dim">
                {fmtLat(s.site.lat)} {fmtLon(s.site.lon)} · floor {fmtDepth(s.site.maxDepth)}
              </div>
            </div>
          </div>

          <DepthControl />
          <Timeline />
          <CompassHud />

          {/* Colorbar & 3D Layer Controls */}
          {showColorbar && (
            <ColorbarControls onClose={() => setShowColorbar(false)} />
          )}

          {/* Inspector */}
          {s.selection && <Inspector />}

          {/* Context-Aware SagarBot */}
          <SagarBot />

          <div className="absolute bottom-[200px] right-4 hidden sm:block pointer-events-none z-10">
            <span className="font-mono text-[8.5px] tracking-[0.2em] text-dim bg-abyss/80 border border-line rounded px-2.5 py-1 shadow-lg">
              DRAG TO LOOK · SCROLL FOR DEPTH · CLICK FLOATS TO INSPECT
            </span>
          </div>
        </React.Fragment>
      )}

      {/* Dive / Ascend Veil Transition */}
      <Veil
        show={s.phase === 'diving' || s.phase === 'ascending'}
        bgOn={veilBg}
        text={veilText}
        showSkip={s.phase === 'diving'}
        onSkip={() => skipRef.current()}
      />

      {/* Toast Notifications */}
      {s.toast && (
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-50 glass px-4 py-2 text-[11px] font-mono text-mist border border-accent/40 shadow-2xl flex items-center gap-2 animate-fade-in">
          <span className="w-1.5 h-1.5 rounded-full bg-accent"></span>
          {s.toast.msg}
        </div>
      )}
    </div>
  );
};
