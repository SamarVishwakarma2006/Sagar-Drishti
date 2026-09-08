import React, { useEffect } from 'react';
import { useApp, store } from '../store/oceanStore';
import { provFor, timeUTC } from '../services/syntheticOcean';
import { Play, Pause, RotateCcw } from 'lucide-react';

export const Timeline: React.FC = () => {
  const s = useApp();

  useEffect(() => {
    if (!s.playing) return;
    const iv = setInterval(() => {
      const t = store.get().timeOffset;
      if (t >= 48) {
        store.set({ timeOffset: 48, playing: false });
      } else {
        store.set({ timeOffset: Math.min(48, t + 0.5) });
      }
    }, 160);
    return () => clearInterval(iv);
  }, [s.playing]);

  const pv = provFor(s.timeOffset);
  const pct = (t: number) => ((t + 48) / 96) * 100;

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 w-[calc(100vw-124px)] sm:w-[min(560px,calc(100vw-380px))] select-none">
      <div className="glass px-3 pt-2.5 pb-2 border border-line shadow-2xl">
        <div className="flex items-center gap-2.5">
          <button
            onClick={() =>
              store.set({
                playing: !s.playing,
                timeOffset: s.timeOffset >= 48 ? -48 : s.timeOffset,
              })
            }
            className="text-accent hover:text-mist w-7 h-7 flex items-center justify-center border border-line rounded hover:bg-white/[0.04]"
          >
            {s.playing ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <button
            onClick={() => store.set({ timeOffset: 0, playing: false })}
            title="Reset to analysis baseline (NOW)"
            className="text-dim hover:text-mist w-7 h-7 flex items-center justify-center border border-line rounded hover:bg-white/[0.04]"
          >
            <RotateCcw size={12} />
          </button>

          <span className="font-mono text-[10.5px] text-mist tracking-wide truncate">
            {timeUTC(s.timeOffset)}
          </span>

          <span className="ml-auto font-mono text-[9px] px-2 py-0.5 rounded border border-line text-dim">
            {pv.label}
          </span>
        </div>

        <div className="relative mt-2.5 h-4">
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-[3px] rounded-sm overflow-hidden flex">
            <div style={{ width: `${pct(-2)}%` }} className="bg-[#8fb0c9]/40" />
            <div style={{ width: `${pct(2) - pct(-2)}%` }} className="bg-[#9db0ba]/50" />
            <div style={{ width: `${100 - pct(2)}%` }} className="hatch" />
          </div>
          <input
            type="range"
            min={-48}
            max={48}
            step={0.25}
            value={s.timeOffset}
            onChange={(e) => store.set({ timeOffset: parseFloat(e.target.value) })}
            className="tl-range absolute inset-0 w-full"
          />
        </div>

        <div className="flex justify-between mt-1 font-mono text-[8.5px] text-dim">
          <span>-48 h (Reanalysis)</span>
          <span className="hidden sm:inline">-24</span>
          <span className="text-accent font-semibold">NOW</span>
          <span className="hidden sm:inline">+24</span>
          <span>+48 h (Forecast)</span>
        </div>
      </div>
    </div>
  );
};
