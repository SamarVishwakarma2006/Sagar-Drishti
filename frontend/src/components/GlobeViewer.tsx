import React, { useEffect, useRef } from 'react';
import { GlobeEngine } from '../engines/GlobeEngine';
import { globeRegistry, store } from '../store/oceanStore';
import { Ocean } from '../services/syntheticOcean';

export const GlobeViewer: React.FC<{
  onReady: () => void;
  onError: (msg: string) => void;
}> = ({ onReady, onError }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const g = new GlobeEngine();
    globeRegistry.g = g;

    g.init(containerRef.current, {
      onReady: () => {
        onReady();
      },
      onError: (err) => {
        onError(err);
      },
      onPickPoint: (lat, lon) => {
        if (store.get().phase !== 'globe') return;
        const site = Ocean.siteForPoint(lat, lon);
        store.set({ site, selection: null });
        g.showSite(site);
        g.flyToSite(site);
      },
      onPickFloat: (id) => {
        store.set({ selection: { kind: 'float', id } });
      },
    });

    return () => {
      g.destroy();
      globeRegistry.g = null;
    };
  }, []);

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" />;
};
