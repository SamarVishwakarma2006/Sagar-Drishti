import React, { useState, useMemo } from 'react';
import { useApp, store, globeRegistry, toast } from '../store/oceanStore';
import { SITES, fmtLat, fmtLon, fmtDepth, pad2 } from '../services/syntheticOcean';
import { SitePhysics } from '../types/ocean';
import { Search, X, Globe, Navigation } from 'lucide-react';

export const SiteExplorer: React.FC = () => {
  const s = useApp();
  const [q, setQ] = useState('');

  // Combine uploaded custom datasets + default INCOIS baseline sites
  const allSites: SitePhysics[] = useMemo(() => {
    return [...s.uploadedSites, ...SITES];
  }, [s.uploadedSites]);

  const filtered = useMemo(() => {
    const query = q.toLowerCase().trim();
    if (!query) return allSites;
    return allSites.filter(
      (x) =>
        x.name.toLowerCase().includes(query) ||
        x.region.toLowerCase().includes(query) ||
        (x.sourceType && x.sourceType.toLowerCase().includes(query))
    );
  }, [allSites, q]);

  const selectSite = (site: SitePhysics) => {
    const isCustom = !!site.isCustom;
    const obs = site.custom_observation || (isCustom ? s.customObservation : null);

    store.set({
      site,
      selection: null,
      customDataMode: isCustom,
      customObservation: obs,
      historicalMode: isCustom ? false : s.historicalMode,
      historicalDate: isCustom && obs?.date ? obs.date : s.historicalDate,
      depth: Math.min(60, site.maxDepth * 0.2),
    });

    if (globeRegistry.g) {
      globeRegistry.g.showSite(site);
      globeRegistry.g.flyToSite(site);
    }
    toast(`Navigation target: ${site.name}`);
  };

  return (
    <aside
      className={`absolute left-3 md:left-4 top-16 w-[calc(100vw-24px)] sm:w-[320px] z-20 flex flex-col glass overflow-hidden transition-all duration-300 ${
        s.site ? 'bottom-[250px] md:bottom-[236px]' : 'bottom-16 md:bottom-6'
      }`}
    >
      {/* Header */}
      <div className="px-3.5 pt-3 pb-2 flex items-baseline justify-between border-b border-line">
        <span className="flex items-center gap-1.5 text-[10px] font-mono tracking-[0.25em] text-dim">
          <Globe size={12} className="text-accent" />
          STUDY SITES REGISTRY
        </span>
        <span className="text-[10px] font-mono text-accent">
          {filtered.length}/{allSites.length}
        </span>
      </div>

      {/* Search Input */}
      <div className="px-3 py-2.5 border-b border-line">
        <div className="flex items-center gap-2 bg-white/[0.04] border border-line rounded px-2.5 py-1.5 focus-within:border-accent/50">
          <Search size={13} className="text-dim shrink-0" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search ocean basins or uploads..."
            className="bg-transparent outline-none text-[11.5px] text-mist placeholder:text-dim/70 w-full"
          />
          {q && (
            <button onClick={() => setQ('')} className="text-dim hover:text-mist">
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Sites List */}
      <div className="flex-1 overflow-y-auto px-1.5 py-2 space-y-1">
        {filtered.map((site, i) => {
          const active = s.site?.id === site.id;
          const isUploaded = site.isCustom;

          return (
            <button
              key={site.id}
              onClick={() => selectSite(site)}
              className={`w-full text-left px-2.5 py-2 rounded-md flex gap-2.5 items-start border-l-2 transition-all ${
                active
                  ? 'border-accent bg-accent/[0.12] shadow-sm'
                  : 'border-transparent hover:bg-white/[0.04]'
              }`}
            >
              <span
                className={`font-mono text-[10px] pt-0.5 shrink-0 ${
                  active ? 'text-accent font-semibold' : 'text-dim'
                }`}
              >
                {pad2(i + 1)}
              </span>

              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-1">
                  <span
                    className={`block text-[12px] font-medium truncate ${
                      active ? 'text-accent' : 'text-mist'
                    }`}
                  >
                    {site.name}
                  </span>
                  {isUploaded && (
                    <span className="shrink-0 text-[8px] font-mono px-1 py-0.2 rounded bg-accent/20 text-accent border border-accent/40 uppercase">
                      UPLOAD
                    </span>
                  )}
                </div>

                <div className="font-mono text-[9px] text-dim mt-0.5 truncate">
                  {fmtLat(site.lat)} {fmtLon(site.lon)} · floor {fmtDepth(site.maxDepth)}
                </div>

                <div className="font-mono text-[8.5px] text-dim/80 mt-0.5 truncate flex items-center gap-1.5">
                  <span className="truncate">{site.region}</span>
                </div>
              </div>
            </button>
          );
        })}

        {filtered.length === 0 && (
          <div className="px-3 py-6 text-[11px] text-dim font-mono text-center">
            No matching ocean study site.
          </div>
        )}
      </div>

      {/* Footer Instructions */}
      <div className="px-3.5 py-2 border-t border-line text-[9px] font-mono text-dim flex items-center gap-1.5">
        <Navigation size={11} className="text-accent shrink-0" />
        <span>Click anywhere on globe to drop study point.</span>
      </div>
    </aside>
  );
};
