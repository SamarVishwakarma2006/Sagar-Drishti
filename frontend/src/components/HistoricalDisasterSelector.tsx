import React, { useState } from 'react';
import { useApp, store, globeRegistry, toast } from '../store/oceanStore';
import { HistoricalEventContext, BoundingBox } from '../types/ocean';
import {
  AlertTriangle,
  Compass,
  Calendar,
  Layers,
  Eye,
  EyeOff,
  LocateFixed,
  ChevronDown,
  ChevronUp,
  X,
  Flame,
  Waves,
  Wind,
} from 'lucide-react';

interface PreloadedEvent {
  id: string;
  name: string;
  type: string;
  dateRange: string;
  peakDate: string;
  region: string;
  basin: string;
  severity: string;
  icon: 'cyclone' | 'swell' | 'heatwave' | 'depression';
  bbox: BoundingBox;
  centroid: { lat: number; lon: number };
  track?: Array<{ date: string; lat: number; lon: number; intensity_kts?: number }>;
  analogAssessment: string;
}

const AUTHORITATIVE_DISASTERS: PreloadedEvent[] = [
  {
    id: 'IMD-2024-SCS-DANA',
    name: 'Severe Cyclonic Storm Dana',
    type: 'Tropical Cyclone',
    dateRange: '2024-10-22 to 2024-10-25',
    peakDate: '2024-10-24',
    region: 'Odisha & West Bengal Coast',
    basin: 'Bay of Bengal',
    severity: 'Severe Cyclonic Storm (110 km/h, 984 hPa)',
    icon: 'cyclone',
    bbox: { min_lat: 15.5, max_lat: 22.0, min_lon: 85.0, max_lon: 91.0 },
    centroid: { lat: 18.5, lon: 88.0 },
    track: [
      { date: '2024-10-22', lat: 15.6, lon: 90.5, intensity_kts: 30 },
      { date: '2024-10-23', lat: 17.2, lon: 88.8, intensity_kts: 45 },
      { date: '2024-10-24', lat: 19.5, lon: 87.4, intensity_kts: 60 },
      { date: '2024-10-25', lat: 21.2, lon: 86.8, intensity_kts: 55 },
    ],
    analogAssessment: 'High SST (30.1°C), low salinity freshwater cap, and thin MLD fueled rapid cyclonic intensification.',
  },
  {
    id: 'IMD-2024-SCS-ASNA',
    name: 'Severe Cyclonic Storm Asna',
    type: 'Tropical Cyclone',
    dateRange: '2024-08-29 to 2024-09-02',
    peakDate: '2024-08-31',
    region: 'Northeast Arabian Sea & Gujarat',
    basin: 'Arabian Sea',
    severity: 'Severe Cyclonic Storm (45 kts / 85 km/h, 988 hPa)',
    icon: 'cyclone',
    bbox: { min_lat: 21.0, max_lat: 24.5, min_lon: 63.0, max_lon: 70.5 },
    centroid: { lat: 23.0, lon: 66.5 },
    track: [
      { date: '2024-08-29', lat: 23.8, lon: 69.8, intensity_kts: 30 },
      { date: '2024-08-30', lat: 23.5, lon: 68.2, intensity_kts: 40 },
      { date: '2024-08-31', lat: 23.2, lon: 66.5, intensity_kts: 45 },
      { date: '2024-09-01', lat: 22.8, lon: 64.8, intensity_kts: 35 },
      { date: '2024-09-02', lat: 21.5, lon: 63.2, intensity_kts: 25 },
    ],
    analogAssessment: 'Rare land-originating storm that emerged into Arabian Sea with elevated monsoon current shear.',
  },
  {
    id: 'IMD-2024-CS-FENGAL',
    name: 'Cyclonic Storm Fengal',
    type: 'Tropical Cyclone',
    dateRange: '2024-11-28 to 2024-12-01',
    peakDate: '2024-11-30',
    region: 'SW Bay of Bengal & Puducherry',
    basin: 'Bay of Bengal',
    severity: 'Cyclonic Storm (85 km/h, 994 hPa)',
    icon: 'cyclone',
    bbox: { min_lat: 9.5, max_lat: 14.5, min_lon: 79.0, max_lon: 84.0 },
    centroid: { lat: 11.5, lon: 81.2 },
    track: [
      { date: '2024-11-28', lat: 10.2, lon: 83.5, intensity_kts: 35 },
      { date: '2024-11-29', lat: 11.4, lon: 81.8, intensity_kts: 45 },
      { date: '2024-11-30', lat: 12.0, lon: 80.2, intensity_kts: 45 },
      { date: '2024-12-01', lat: 12.2, lon: 79.5, intensity_kts: 30 },
    ],
    analogAssessment: 'Late-season post-monsoon cyclogenesis with moderate SST (29.2°C) causing torrential coastal flooding.',
  },
  {
    id: 'IMD-2024-DD-BOB05',
    name: 'Deep Depression BOB 05',
    type: 'Deep Depression',
    dateRange: '2024-09-07 to 2024-09-10',
    peakDate: '2024-09-09',
    region: 'Odisha & North Andhra Pradesh',
    basin: 'Bay of Bengal',
    severity: 'Deep Depression (60 km/h, 992 hPa)',
    icon: 'depression',
    bbox: { min_lat: 17.0, max_lat: 21.5, min_lon: 83.5, max_lon: 89.0 },
    centroid: { lat: 19.2, lon: 86.2 },
    analogAssessment: 'Monsoon depression fueled by massive runoff plume and strong surface current convergence.',
  },
  {
    id: 'INCOIS-2024-SW-KALLAKKADAL',
    name: 'Kallakkadal Swell Surge',
    type: 'High Swell Surge',
    dateRange: '2024-07-15 to 2024-07-18',
    peakDate: '2024-07-16',
    region: 'Kerala & Lakshadweep Coast',
    basin: 'South Arabian Sea',
    severity: 'High Swell (Wave Ht 2.5-3.8m, Period 16-19s)',
    icon: 'swell',
    bbox: { min_lat: 6.0, max_lat: 12.0, min_lon: 71.0, max_lon: 78.0 },
    centroid: { lat: 9.0, lon: 74.5 },
    analogAssessment: 'Long-period Southern Ocean swell propagation causing sudden unannounced coastal inundation.',
  },
  {
    id: 'INCOIS-2024-MHW-BOB',
    name: 'Northern BoB Marine Heatwave',
    type: 'Marine Heatwave',
    dateRange: '2024-07-02 to 2024-07-10',
    peakDate: '2024-07-05',
    region: 'Northern Bay of Bengal Plume',
    basin: 'Bay of Bengal',
    severity: 'Cat 1 Moderate Heatwave (+1.4°C Anomaly)',
    icon: 'heatwave',
    bbox: { min_lat: 18.0, max_lat: 22.0, min_lon: 86.0, max_lon: 92.0 },
    centroid: { lat: 20.0, lon: 89.0 },
    analogAssessment: 'Persistent atmospheric blocking and thin MLD suppressing vertical cooling mixing.',
  },
];

export const HistoricalDisasterSelector: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const s = useApp();
  const [selectedEventId, setSelectedEventId] = useState<string | null>(
    s.activeHazardZone?.event_id || null
  );
  const [customDate, setCustomDate] = useState<string>(s.historicalDate || '2024-10-24');

  const handleSelectEvent = (event: PreloadedEvent) => {
    setSelectedEventId(event.id);

    // Switch store historical date to peak date
    store.setHistoricalMode(true);
    store.setHistoricalDate(event.peakDate);

    // Prepare hazard zone metadata
    const hazardZone: HistoricalEventContext = {
      event_id: event.id,
      event_name: event.name,
      event_type: event.type,
      event_dates: event.dateRange,
      severity: event.severity,
      affected_region: event.region,
      matched_basin: event.basin,
      bbox: event.bbox,
      centroid_lat: event.centroid.lat,
      centroid_lon: event.centroid.lon,
      track_coordinates: event.track,
      analog_assessment: event.analogAssessment,
    };

    store.setActiveHazardZone(hazardZone);

    // Mark on 3D Cesium Globe
    if (globeRegistry.g) {
      globeRegistry.g.markDisasterHazardArea({
        name: event.name,
        type: event.type,
        date: event.peakDate,
        severity: event.severity,
        bbox: event.bbox,
        centroid_lat: event.centroid.lat,
        centroid_lon: event.centroid.lon,
        track_coordinates: event.track,
        analog_assessment: event.analogAssessment,
      });

      globeRegistry.g.flyToDisasterArea(event.bbox, event.centroid);
    }

    toast(`Marked ${event.name} hazard zone on 3D globe`);
  };

  const handleCustomDateSubmit = () => {
    store.setHistoricalMode(true);
    store.setHistoricalDate(customDate);

    // Check if there is an exact or near matching disaster for this date
    const matched = AUTHORITATIVE_DISASTERS.find(
      (ev) => customDate >= ev.dateRange.split(' to ')[0] && customDate <= ev.dateRange.split(' to ')[1]
    );

    if (matched) {
      handleSelectEvent(matched);
    } else {
      toast(`Historical date set to ${customDate} (Copernicus 2-year reanalysis)`);
    }
  };

  const handleClearHazard = () => {
    setSelectedEventId(null);
    store.setActiveHazardZone(null);
    if (globeRegistry.g) {
      globeRegistry.g.clearDisasterHazardArea();
    }
    toast('Disaster hazard markings cleared from globe');
  };

  const handleFlyToActive = () => {
    if (s.activeHazardZone && globeRegistry.g) {
      globeRegistry.g.flyToDisasterArea(
        s.activeHazardZone.bbox,
        s.activeHazardZone.centroid_lat && s.activeHazardZone.centroid_lon
          ? { lat: s.activeHazardZone.centroid_lat, lon: s.activeHazardZone.centroid_lon }
          : null
      );
    }
  };

  return (
    <div className="absolute top-16 left-3 md:left-4 z-30 w-[min(380px,calc(100vw-24px))] glass border border-amber-500/40 shadow-2xl overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between px-3.5 py-2.5 bg-amber-950/40 border-b border-amber-500/30">
        <div className="flex items-center gap-2">
          <AlertTriangle size={15} className="text-amber-400" />
          <div>
            <div className="text-[11px] font-semibold text-amber-200 tracking-wider">
              HISTORICAL DISASTER & REANALYSIS
            </div>
            <div className="text-[8px] font-mono text-amber-300/70 tracking-widest">
              3D GLOBE HAZARD ZONE MARKER
            </div>
          </div>
        </div>

        <button
          onClick={onClose}
          className="text-dim hover:text-mist p-1 rounded transition-colors"
          title="Close selector"
        >
          <X size={14} />
        </button>
      </div>

      <div className="p-3 space-y-3 max-h-[calc(100vh-140px)] overflow-y-auto">
        {/* Active Hazard Zone Bar */}
        {s.activeHazardZone && (
          <div className="p-2 rounded bg-rose-950/40 border border-rose-500/40 flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
                <span className="text-[10px] font-bold text-rose-200 truncate">
                  {s.activeHazardZone.event_name}
                </span>
              </div>
              <div className="text-[8px] font-mono text-rose-300/80 truncate">
                {s.activeHazardZone.affected_region}
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={handleFlyToActive}
                className="p-1 rounded bg-white/[0.06] hover:bg-white/[0.12] text-dim hover:text-rose-200 text-[8px] font-mono border border-line"
                title="Fly camera to hazard area"
              >
                <LocateFixed size={12} />
              </button>
              <button
                onClick={handleClearHazard}
                className="p-1 rounded bg-red-500/20 hover:bg-red-500/30 text-red-300 text-[8px] font-mono border border-red-500/40"
                title="Remove hazard marking from globe"
              >
                CLEAR
              </button>
            </div>
          </div>
        )}

        {/* Date Selector for Copernicus Reanalysis */}
        <div className="p-2.5 rounded bg-black/30 border border-line/40 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-dim tracking-wider flex items-center gap-1.5">
              <Calendar size={12} className="text-accent" />
              SELECT REANALYSIS DATE
            </span>
            <span className="text-[8px] font-mono text-dim/70">2024-06-24 → 2026-06-23</span>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="date"
              min="2024-06-24"
              max="2026-06-23"
              value={customDate}
              onChange={(e) => setCustomDate(e.target.value)}
              className="flex-1 bg-black/60 border border-line focus:border-amber-500/60 rounded px-2 py-1.5 text-[10.5px] font-mono text-amber-200 outline-none"
            />
            <button
              onClick={handleCustomDateSubmit}
              className="px-2.5 py-1.5 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-[9.5px] font-mono tracking-wider transition-colors"
            >
              APPLY
            </button>
          </div>
          <div className="text-[8px] font-mono text-dim/80">
            Select any historical date to evaluate real Copernicus physics and match against documented ocean disasters.
          </div>
        </div>

        {/* Notable Historical Disasters List */}
        <div className="space-y-1.5">
          <div className="text-[8.5px] font-mono text-dim tracking-wider uppercase flex items-center justify-between">
            <span>Verified Historical Disasters</span>
            <span className="text-amber-400/90 text-[8px]">IMD & INCOIS</span>
          </div>

          {AUTHORITATIVE_DISASTERS.map((ev) => {
            const isSelected = selectedEventId === ev.id;
            return (
              <div
                key={ev.id}
                className={`p-2.5 rounded transition-all border ${
                  isSelected
                    ? 'bg-rose-950/30 border-rose-500/50 shadow-md'
                    : 'bg-black/25 border-line/30 hover:border-amber-500/40 hover:bg-white/[0.02]'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="text-[10.5px] font-semibold text-mist flex items-center gap-1.5">
                      {ev.icon === 'cyclone' && <Wind size={12} className="text-rose-400" />}
                      {ev.icon === 'swell' && <Waves size={12} className="text-cyan-400" />}
                      {ev.icon === 'heatwave' && <Flame size={12} className="text-amber-400" />}
                      {ev.icon === 'depression' && <AlertTriangle size={12} className="text-yellow-400" />}
                      <span>{ev.name}</span>
                    </div>
                    <div className="text-[8.5px] font-mono text-dim mt-0.5">
                      {ev.dateRange} · <span className="text-amber-300/90">{ev.basin}</span>
                    </div>
                  </div>

                  <button
                    onClick={() => handleSelectEvent(ev)}
                    className={`shrink-0 px-2 py-1 rounded text-[8.5px] font-mono tracking-wider transition-colors ${
                      isSelected
                        ? 'bg-rose-500 text-[#03151c] font-bold'
                        : 'bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/40'
                    }`}
                  >
                    {isSelected ? 'MARKED' : 'MARK GLOBE'}
                  </button>
                </div>

                <div className="mt-1.5 text-[8px] font-mono text-dim/90 flex justify-between items-center border-t border-line/20 pt-1">
                  <span className="truncate max-w-[210px]">{ev.region}</span>
                  <span className="text-accent font-semibold">{ev.severity.split('(')[0]}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
