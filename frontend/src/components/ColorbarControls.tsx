import React from 'react';
import { useApp, store, uwRegistry } from '../store/oceanStore';
import { PALETTES, VARIABLES, Ocean, cssGradient } from '../services/syntheticOcean';
import { PaletteKey } from '../types/ocean';
import { Sliders, RotateCcw, X } from 'lucide-react';

export const ColorbarControls: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
  const s = useApp();
  const site = s.site;
  const cb = s.colorbar;
  const autoRange = site ? Ocean.rangeAt(site, s.variable) : { min: 0, max: 30 };
  const currentMin = cb.customMin !== null ? cb.customMin : autoRange.min;
  const currentMax = cb.customMax !== null ? cb.customMax : autoRange.max;
  const curVar = VARIABLES.find((v) => v.key === s.variable);

  const setPalette = (palette: PaletteKey) => {
    store.setColorbar({ palette });
    uwRegistry.e?.setColorbarSettings({ ...cb, palette });
  };

  const handleMinChange = (v: number) => {
    store.setColorbar({ customMin: v });
    uwRegistry.e?.setColorbarSettings({ ...cb, customMin: v });
  };

  const handleMaxChange = (v: number) => {
    store.setColorbar({ customMax: v });
    uwRegistry.e?.setColorbarSettings({ ...cb, customMax: v });
  };

  const resetBounds = () => {
    store.setColorbar({ customMin: null, customMax: null });
    uwRegistry.e?.setColorbarSettings({ ...cb, customMin: null, customMax: null });
  };

  const toggleLog = () => {
    const nextLog = !cb.isLogScale;
    store.setColorbar({ isLogScale: nextLog });
    uwRegistry.e?.setColorbarSettings({ ...cb, isLogScale: nextLog });
  };

  const handleOpacity = (opacity: number) => {
    store.setColorbar({ opacity });
    uwRegistry.e?.setColorbarSettings({ ...cb, opacity });
  };

  const handleExaggeration = (verticalExaggeration: number) => {
    store.setColorbar({ verticalExaggeration });
    uwRegistry.e?.setColorbarSettings({ ...cb, verticalExaggeration });
  };

  return (
    <div className="absolute top-14 left-3 md:left-4 z-30 w-[290px] glass p-3.5 flex flex-col gap-3.5 border border-line shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-line pb-2.5">
        <div className="flex items-center gap-2 text-mist font-medium text-[12px]">
          <Sliders size={14} className="text-accent" />
          Colorbar & 3D Layer Controls
        </div>
        {onClose && (
          <button onClick={onClose} className="text-dim hover:text-mist">
            <X size={14} />
          </button>
        )}
      </div>

      {/* Palette Switcher */}
      <div>
        <label className="block text-[9.5px] font-mono text-dim tracking-wider mb-1.5">
          PALETTE ({cb.palette.toUpperCase()})
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          {(Object.keys(PALETTES) as PaletteKey[]).map((pkey) => {
            const p = PALETTES[pkey];
            const active = cb.palette === pkey;
            return (
              <button
                key={pkey}
                onClick={() => setPalette(pkey)}
                className={`p-1.5 rounded text-left border transition-all ${
                  active
                    ? 'border-accent bg-accent/15'
                    : 'border-line hover:border-white/20 bg-white/[0.02]'
                }`}
              >
                <div className="flex justify-between text-[9px] font-mono mb-1">
                  <span className={active ? 'text-accent font-semibold' : 'text-mist'}>
                    {p.label}
                  </span>
                </div>
                <div
                  className="h-2 rounded-sm border border-black/30"
                  style={{ background: cssGradient(p.stops) }}
                />
              </button>
            );
          })}
        </div>
      </div>

      {/* Scale & Dynamic Min/Max Range */}
      <div className="space-y-2 border-t border-line pt-2.5">
        <div className="flex items-center justify-between">
          <span className="text-[9.5px] font-mono text-dim tracking-wider">RANGE BOUNDS</span>
          <div className="flex items-center gap-1.5">
            <button
              onClick={toggleLog}
              className={`px-1.5 py-0.5 rounded text-[8.5px] font-mono border ${
                cb.isLogScale
                  ? 'border-accent text-accent bg-accent/10'
                  : 'border-line text-dim hover:text-mist'
              }`}
            >
              {cb.isLogScale ? 'LOG' : 'LINEAR'}
            </button>
            <button
              onClick={resetBounds}
              title="Reset to auto min/max"
              className="p-1 rounded text-dim hover:text-accent border border-line"
            >
              <RotateCcw size={10} />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-[8px] font-mono text-dim">MIN ({curVar?.unit})</span>
            <input
              type="number"
              step="0.1"
              value={currentMin.toFixed(1)}
              onChange={(e) => handleMinChange(parseFloat(e.target.value) || 0)}
              className="w-full bg-[#06121a] border border-line rounded px-2 py-1 text-[11px] font-mono text-mist outline-none focus:border-accent"
            />
          </div>
          <div>
            <span className="text-[8px] font-mono text-dim">MAX ({curVar?.unit})</span>
            <input
              type="number"
              step="0.1"
              value={currentMax.toFixed(1)}
              onChange={(e) => handleMaxChange(parseFloat(e.target.value) || 1)}
              className="w-full bg-[#06121a] border border-line rounded px-2 py-1 text-[11px] font-mono text-mist outline-none focus:border-accent"
            />
          </div>
        </div>
      </div>

      {/* Opacity & Vertical Exaggeration Sliders */}
      <div className="space-y-2.5 border-t border-line pt-2.5">
        <div>
          <div className="flex justify-between text-[9px] font-mono text-dim mb-1">
            <span>LAYER OPACITY</span>
            <span className="text-accent">{Math.round(cb.opacity * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={cb.opacity}
            onChange={(e) => handleOpacity(parseFloat(e.target.value))}
            className="w-full accent-accent cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-[9px] font-mono text-dim mb-1">
            <span>VERTICAL EXAGGERATION (3D)</span>
            <span className="text-accent">{cb.verticalExaggeration.toFixed(1)}×</span>
          </div>
          <input
            type="range"
            min="1.0"
            max="25.0"
            step="0.5"
            value={cb.verticalExaggeration}
            onChange={(e) => handleExaggeration(parseFloat(e.target.value))}
            className="w-full accent-accent cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};
