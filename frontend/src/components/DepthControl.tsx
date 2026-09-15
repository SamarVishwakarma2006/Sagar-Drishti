import React, { useRef, useMemo, useState } from 'react';
import { useApp, store, toast } from '../store/oceanStore';
import { Ocean, clamp, fmtDepth } from '../services/syntheticOcean';
import { ChevronUp, ChevronDown } from 'lucide-react';

function getStratum(depth: number): { name: string; zone: string } {
  if (depth < 200) return { name: 'EPIPELAGIC', zone: 'Sunlit' };
  if (depth < 1000) return { name: 'MESOPELAGIC', zone: 'Twilight' };
  if (depth < 4000) return { name: 'BATHYPELAGIC', zone: 'Midnight' };
  return { name: 'ABYSSAL', zone: 'Abyss' };
}

export const DepthControl: React.FC = () => {
  const s = useApp();
  const site = s.site;
  if (!site) return null;

  const dmax = site.maxDepth;
  const trackRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);

  // Power-law non-linear scaling (f in [0, 1])
  const frac = Math.pow(clamp(s.depth / dmax, 0, 1), 1 / 3.2);
  const pf = (d: number) => Math.pow(clamp(d / dmax, 0, 1), 1 / 3.2) * 100;

  const setFrom = (e: React.PointerEvent) => {
    if (!trackRef.current) return;
    const r = trackRef.current.getBoundingClientRect();
    const f = clamp((e.clientY - r.top) / r.height, 0, 1);
    const target = Math.max(8, Math.round(dmax * Math.pow(f, 3.2)));
    store.set({ depth: target });
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const step = e.shiftKey ? 100 : (s.depth < 200 ? 10 : 50);
    const delta = e.deltaY > 0 ? step : -step;
    store.set({ depth: clamp(s.depth + delta, 8, dmax) });
  };

  const floats = useMemo(() => Ocean.floatsAt(site), [site.id]);
  const ticks = [50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000].filter((t) => t < dmax * 0.94);
  const stratum = getStratum(s.depth);

  return (
    <div
      className="absolute left-3 md:left-4 top-1/2 -translate-y-1/2 z-20 flex flex-col items-center gap-1.5 select-none"
      onWheel={handleWheel}
    >
      {/* Current Depth Badge & Stratum Readout */}
      <div className="glass px-2.5 py-1.5 font-mono text-center border border-accent/40 shadow-xl min-w-[76px]">
        <div className="text-[11px] font-semibold text-accent tracking-wide flex items-center justify-center gap-1">
          <span>{Math.round(s.depth)}</span>
          <span className="text-[9px] text-accent/70">m</span>
        </div>
        <div className="text-[7.5px] text-dim font-medium tracking-wider text-center uppercase">
          {stratum.zone}
        </div>
      </div>

      {/* Vertical Track Rail Container */}
      <div className="glass relative w-14 h-[40vh] max-h-[380px] min-h-[220px] px-1 py-3 border border-line shadow-2xl flex flex-col items-center justify-between">
        {/* Surface Quick-Jump */}
        <button
          onClick={() => store.set({ depth: 8 })}
          title="Jump to Surface (8m)"
          className="font-mono text-[7px] text-dim hover:text-accent tracking-widest uppercase transition-colors px-1 py-0.5 rounded hover:bg-white/5"
        >
          SURF
        </button>

        {/* Interactive Track Area */}
        <div
          ref={trackRef}
          className="relative w-full flex-1 my-1 cursor-ns-resize touch-none"
          onPointerDown={(e) => {
            dragRef.current = true;
            setIsDragging(true);
            (e.currentTarget as any).setPointerCapture(e.pointerId);
            setFrom(e);
          }}
          onPointerMove={(e) => {
            if (dragRef.current) setFrom(e);
          }}
          onPointerUp={() => {
            dragRef.current = false;
            setIsDragging(false);
          }}
          onPointerCancel={() => {
            dragRef.current = false;
            setIsDragging(false);
          }}
        >
          {/* Central Vertical Track Line */}
          <div className="absolute top-0 bottom-0 left-[14px] w-[3px] bg-white/10 rounded-full">
            {/* Active Depth Progress */}
            <div
              className="absolute top-0 left-0 w-full bg-gradient-to-b from-accent/90 to-accent/60 rounded-full shadow-[0_0_8px_rgba(86,212,226,0.6)]"
              style={{ height: `${frac * 100}%` }}
            />

            {/* Float Parking Depth Markers */}
            {floats.map((f, i) => (
              <button
                key={i}
                title={`Argo ${f.id} · Parking Depth: ${Math.round(f.parkingDepth)}m`}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  store.set({ depth: clamp(Math.round(f.parkingDepth), 8, dmax) });
                  toast(`Argo ${f.id} parking depth: ${Math.round(f.parkingDepth)} m`);
                }}
                className="group/float absolute -left-[4px] w-[11px] h-[11px] rounded-full border border-accent bg-abyss hover:bg-accent transition-all hover:scale-125 z-10"
                style={{ top: `calc(${pf(clamp(f.parkingDepth, 15, dmax))}% - 5.5px)` }}
              >
                <span className="opacity-0 group-hover/float:opacity-100 pointer-events-none absolute left-4 -top-2 whitespace-nowrap font-mono text-[8.5px] bg-abyss/95 border border-accent/50 px-2 py-0.5 rounded shadow-xl text-accent z-30 transition-opacity">
                  Argo {f.id} ({Math.round(f.parkingDepth)}m)
                </span>
              </button>
            ))}

            {/* Slider Thumb Handle */}
            <div
              className={`absolute -left-[7.5px] w-[18px] h-[18px] rounded-full bg-[#06121a] border-2 border-accent shadow-[0_0_12px_rgba(86,212,226,0.9)] z-20 flex items-center justify-center transition-transform ${
                isDragging ? 'scale-125 cursor-grabbing' : 'hover:scale-110 cursor-grab'
              }`}
              style={{ top: `calc(${frac * 100}% - 9px)` }}
            >
              <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            </div>
          </div>

          {/* Depth Ticks & Labels */}
          {ticks.map((t) => {
            const topPct = pf(t);
            return (
              <button
                key={t}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  store.set({ depth: t });
                }}
                title={`Seek to ${t}m`}
                className="group/tick absolute left-[14px] -translate-y-1/2 flex items-center cursor-pointer transition-colors"
                style={{ top: `${topPct}%` }}
              >
                {/* Horizontal Notch */}
                <span className="w-2 h-[1px] bg-white/20 group-hover/tick:bg-accent group-hover/tick:w-3 transition-all" />
                {/* Tick Depth Label */}
                <span className="font-mono text-[8px] text-dim/80 group-hover/tick:text-accent pl-1.5 select-none drop-shadow-[0_1px_2px_rgba(0,0,0,0.9)]">
                  {t >= 1000 ? `${t / 1000}k` : t}
                </span>
              </button>
            );
          })}
        </div>

        {/* Floor Quick-Jump & Depth Display */}
        <button
          onClick={() => store.set({ depth: dmax })}
          title={`Jump to Seabed Floor (${fmtDepth(dmax)})`}
          className="font-mono text-[7px] text-dim hover:text-accent tracking-widest uppercase transition-colors px-1 py-0.5 rounded hover:bg-white/5"
        >
          {fmtDepth(dmax)}
        </button>
      </div>

      {/* Up / Down Micro-Step Buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => store.set({ depth: clamp(s.depth - (s.depth < 200 ? 10 : 50), 8, dmax) })}
          title="Ascend (-50m)"
          className="glass w-6 h-6 flex items-center justify-center text-dim hover:text-accent border border-line hover:border-accent/40 rounded transition-colors"
        >
          <ChevronUp size={13} />
        </button>
        <button
          onClick={() => store.set({ depth: clamp(s.depth + (s.depth < 200 ? 10 : 50), 8, dmax) })}
          title="Descend (+50m)"
          className="glass w-6 h-6 flex items-center justify-center text-dim hover:text-accent border border-line hover:border-accent/40 rounded transition-colors"
        >
          <ChevronDown size={13} />
        </button>
      </div>
    </div>
  );
};
