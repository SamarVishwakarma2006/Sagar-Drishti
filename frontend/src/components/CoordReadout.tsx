import React, { useState, useEffect } from 'react';
import { globeRegistry } from '../store/oceanStore';
import { fmtLat, fmtLon, getBathymetry } from '../services/syntheticOcean';
import { LatLon } from '../types/ocean';

export const CoordReadout: React.FC = () => {
  const [c, setC] = useState<LatLon | null>(null);

  useEffect(() => {
    const iv = setInterval(() => {
      setC(globeRegistry.g?.getHover() ?? null);
    }, 100);
    return () => clearInterval(iv);
  }, []);

  const bathy = c ? Math.round(getBathymetry(c.lat, c.lon)) : 0;

  return (
    <div className="glass px-2.5 py-1.5 font-mono text-[9.5px] text-dim min-w-[170px] text-center border border-line flex items-center justify-center gap-2">
      {c ? (
        <>
          <span className="text-cyan font-medium">{fmtLat(c.lat, 3)}</span>
          <span className="text-cyan font-medium">{fmtLon(c.lon, 3)}</span>
          <span className="text-accent/80 text-[8.5px]">(-{bathy}m)</span>
        </>
      ) : (
        <span className="text-dim/60">OCEAN PROBE ACTIVE</span>
      )}
    </div>
  );
};

