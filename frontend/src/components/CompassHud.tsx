import React, { useState, useEffect } from 'react';
import { useApp, uwRegistry, toast } from '../store/oceanStore';
import { fmtLat, fmtLon, mod } from '../services/syntheticOcean';
import { RotateCcw } from 'lucide-react';

export const CompassHud: React.FC = () => {
  const s = useApp();
  const site = s.site;
  const [yaw, setYaw] = useState(0);

  useEffect(() => {
    const iv = setInterval(() => {
      setYaw(uwRegistry.e?.getYaw() ?? 0);
    }, 120);
    return () => clearInterval(iv);
  }, []);

  if (!site) return null;

  const deg = mod((-yaw * 180) / Math.PI, 360);

  return (
    <div className="absolute bottom-4 right-3 md:right-4 z-20 flex flex-col items-end gap-2 select-none">
      <div className="glass px-2.5 py-2 flex flex-col items-center gap-1.5 border border-line shadow-xl">
        <svg width="50" height="50" viewBox="0 0 52 52">
          <circle cx="26" cy="26" r="22" fill="none" stroke="rgba(126,196,207,0.25)" />
          {Array.from({ length: 8 }).map((_, i) => {
            const a = (i * Math.PI) / 4;
            return (
              <line
                key={i}
                x1={26 + Math.sin(a) * 18}
                y1={26 - Math.cos(a) * 18}
                x2={26 + Math.sin(a) * 22}
                y2={26 - Math.cos(a) * 22}
                stroke="rgba(126,196,207,0.35)"
                strokeWidth={i % 2 ? 0.7 : 1.2}
              />
            );
          })}
          <g transform={`rotate(${deg} 26 26)`}>
            <path d="M26 4 L30 12 L26 9.5 L22 12 Z" fill="#56d4e2" />
            <line x1="26" y1="12" x2="26" y2="20" stroke="#56d4e2" strokeWidth="1.4" />
          </g>
          <circle cx="26" cy="26" r="1.8" fill="#d9e7ec" />
        </svg>
        <span className="font-mono text-[9px] text-dim font-medium">
          {String(Math.round(deg)).padStart(3, '0')}°
        </span>
      </div>

      <button
        title="Reset 3D camera heading"
        onClick={() => {
          uwRegistry.e?.resetView();
          toast('3D Camera angle reset');
        }}
        className="glass w-8 h-8 flex items-center justify-center text-dim hover:text-accent border border-line hover:border-accent/40"
      >
        <RotateCcw size={13} />
      </button>

      <div className="glass px-2.5 py-1.5 font-mono text-[9px] text-dim text-right leading-relaxed border border-line">
        {fmtLat(site.lat)} {fmtLon(site.lon)}
        <br />
        <span className="text-accent font-semibold">DEPTH {Math.round(s.depth)} m</span>
      </div>
    </div>
  );
};
