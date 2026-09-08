import React from 'react';
import { useApp, store } from '../store/oceanStore';
import { VARIABLES, timeUTC } from '../services/syntheticOcean';
import { VariableKey } from '../types/ocean';
import { Waves, Thermometer, Droplets, Wind, Sliders } from 'lucide-react';

const iconMap: Record<string, React.ReactNode> = {
  Thermometer: <Thermometer size={13} />,
  Droplets: <Droplets size={13} />,
  Waves: <Waves size={13} />,
  Wind: <Wind size={13} />,
};

interface TopBarProps {
  onToggleColorbar?: () => void;
  showColorbar?: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({ onToggleColorbar, showColorbar }) => {
  const s = useApp();
  const uw = s.phase === 'underwater';

  return (
    <header className="absolute top-0 left-0 right-0 h-12 z-30 flex items-center px-3 md:px-4 border-b border-line bg-[#040a10]/80 backdrop-blur-md">
      <div className="flex items-center gap-2.5">
        <span className="text-accent">
          <Waves size={19} />
        </span>
        <div className="leading-none">
          <div className="text-[13px] font-semibold tracking-[0.22em] text-mist flex items-center gap-2">
            SAGAR DRISHTI
            {s.customDataMode && (
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-accent/15 text-accent border border-accent/30 tracking-wider">
                CUSTOM DATASET LIVE
              </span>
            )}
          </div>
          <div className="hidden md:block text-[8px] tracking-[0.3em] text-dim mt-1">
            IMMERSIVE OCEAN OBSERVATORY
          </div>
        </div>
      </div>

      {/* Underwater Variable Switcher */}
      {uw && (
        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-0.5 bg-white/[0.04] border border-line rounded-md p-0.5 shadow-inner">
          {VARIABLES.map((v) => (
            <button
              key={v.key}
              onClick={() => store.set({ variable: v.key as VariableKey })}
              className={`flex items-center gap-1.5 px-2 md:px-2.5 py-1.5 rounded text-[10px] font-mono tracking-widest transition-all ${
                s.variable === v.key
                  ? 'text-accent bg-accent/15 font-semibold shadow-sm'
                  : 'text-dim hover:text-mist hover:bg-white/[0.04]'
              }`}
            >
              {iconMap[v.icon]}
              <span className="hidden md:inline">{v.label}</span>
            </button>
          ))}
        </div>
      )}

      <div className="ml-auto flex items-center gap-2">
        {uw && onToggleColorbar && (
          <button
            onClick={onToggleColorbar}
            title="Colorbar & Layer Controls"
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-[10px] font-mono border transition-colors ${
              showColorbar
                ? 'border-accent text-accent bg-accent/10'
                : 'border-line text-dim hover:text-mist hover:border-line'
            }`}
          >
            <Sliders size={12} />
            <span className="hidden sm:inline">COLORBAR</span>
          </button>
        )}

        {uw && (
          <span className="font-mono text-[10px] text-dim hidden lg:inline border border-line rounded px-2.5 py-1">
            {timeUTC(s.timeOffset)}
          </span>
        )}
      </div>
    </header>
  );
};
