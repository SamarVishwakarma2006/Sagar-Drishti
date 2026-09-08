import React, { useEffect, useRef } from 'react';
import { UnderwaterEngine } from '../engines/UnderwaterEngine';
import { useApp, store, uwRegistry, toast, globeRegistry } from '../store/oceanStore';
import { clamp } from '../services/syntheticOcean';

export const UnderwaterViewer: React.FC = () => {
  const s = useApp();
  const containerRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<UnderwaterEngine | null>(null);

  useEffect(() => {
    if (s.phase !== 'underwater' || engineRef.current || !containerRef.current) return;

    const st = store.get();
    const site = st.site;
    if (!site) return;

    const uw = new UnderwaterEngine();
    try {
      uw.mount(containerRef.current, site, {
        onSelect: (sel) => store.set({ selection: sel }),
        onDepthDelta: (d) => {
          const cur = store.get();
          if (!cur.site) return;
          store.set({ depth: clamp(cur.depth + d, 8, cur.site.maxDepth) });
        },
      });

      uw.setParams({
        depth: st.depth,
        variable: st.variable,
        timeOffset: st.timeOffset,
      });

      uw.setColorbarSettings(st.colorbar);

      engineRef.current = uw;
      uwRegistry.e = uw;
    } catch (err: any) {
      uw.dispose();
      toast('UNDERWATER RENDERER FAILED — CHECK WEBGL SETTINGS');
      globeRegistry.g?.setPaused(false);
      store.set({ phase: 'globe' });
    }

    return () => {
      if (engineRef.current) {
        engineRef.current.dispose();
        engineRef.current = null;
        uwRegistry.e = null;
      }
    };
  }, [s.phase, s.site?.id]);

  // Live state synchronization into Three.js engine
  useEffect(() => {
    engineRef.current?.setParams({
      depth: s.depth,
      variable: s.variable,
      timeOffset: s.timeOffset,
    });
  }, [s.depth, s.variable, s.timeOffset]);

  useEffect(() => {
    engineRef.current?.setColorbarSettings(s.colorbar);
  }, [s.colorbar]);

  useEffect(() => {
    engineRef.current?.setSelection(s.selection);
  }, [s.selection]);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 w-full h-full ${s.phase === 'underwater' ? '' : 'hidden'}`}
    />
  );
};
