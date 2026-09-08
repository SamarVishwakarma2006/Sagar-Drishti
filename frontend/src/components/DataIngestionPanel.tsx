import React, { useState, useRef } from 'react';
import { useApp, store, globeRegistry, toast } from '../store/oceanStore';
import { OceanAPI } from '../services/api';
import { UploadCloud, Database, FileUp, Loader, ChevronDown, ChevronUp, Trash2, CheckCircle } from 'lucide-react';

export const DataIngestionPanel: React.FC = () => {
  const s = useApp();
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const activeMeta = s.activeUpload;

  const handleUploadFile = async (file: File) => {
    const ext = file.name.toLowerCase().split('.').pop() || '';
    if (!['nc', 'nc4', 'csv', 'txt'].includes(ext)) {
      toast(`Unsupported format '.${ext}' — only .nc, .nc4, .csv, and .txt are supported.`);
      return;
    }

    setLoading(true);
    toast(`Ingesting ${file.name}...`);

    try {
      const meta = await OceanAPI.uploadDataset(file);
      store.addUploadedSite(meta);

      // Immediately fly Cesium camera to the extracted bounding box
      if (globeRegistry.g) {
        globeRegistry.g.showSite(meta.site_record);
        globeRegistry.g.flyToSite(meta.site_record);
      }

      toast(`Successfully ingested ${file.name} — now active`);
    } catch (e: any) {
      toast(`Ingestion error: ${e?.message || 'Failed to parse dataset'}`);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUploadFile(file);
  };

  if (s.customDataMode && activeMeta) {
    return (
      <div className="absolute top-16 right-3 md:right-4 z-20 w-[270px] glass overflow-hidden border border-line shadow-xl">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-between px-3 py-2.5 bg-accent/[0.04] border-b border-line"
        >
          <span className="flex items-center gap-2">
            <Database size={13} className="text-accent" />
            <span className="text-[10px] font-mono tracking-[0.25em] text-dim">
              INGESTION PIPELINE
            </span>
          </span>
          <span className="text-dim">
            {expanded ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
          </span>
        </button>

        {expanded && (
          <div className="px-3 py-3 space-y-2.5">
            <div className="flex items-center gap-2 p-2 rounded bg-accent/[0.08] border border-accent/30">
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse"></span>
              <div className="min-w-0 flex-1">
                <div className="font-mono text-[10.5px] text-accent font-medium truncate">
                  {activeMeta.filename}
                </div>
                <div className="font-mono text-[8.5px] text-dim">
                  {activeMeta.file_type} · {activeMeta.file_size}
                </div>
              </div>
            </div>

            <div className="font-mono text-[9px] text-dim space-y-1 bg-white/[0.02] p-2 rounded border border-line">
              <div className="flex justify-between">
                <span>COVERAGE:</span>
                <span className="text-mist">
                  {activeMeta.bounding_box.min_lat.toFixed(1)}° to {activeMeta.bounding_box.max_lat.toFixed(1)}°N
                </span>
              </div>
              <div className="flex justify-between">
                <span>MAX DEPTH:</span>
                <span className="text-mist">{activeMeta.depth_range.max} m</span>
              </div>
              {activeMeta.float_count > 0 && (
                <div className="flex justify-between">
                  <span>FLOAT PROFILES:</span>
                  <span className="text-accent font-semibold">{activeMeta.float_count} parsed</span>
                </div>
              )}
            </div>

            <button
              onClick={() => {
                store.removeUploadedSite(activeMeta.site_id);
                toast('Custom dataset unloaded — baseline INCOIS active');
              }}
              className="w-full flex items-center justify-center gap-2 border border-line hover:border-red-400/50 hover:bg-red-500/10 text-dim hover:text-red-300 rounded py-2 text-[10px] font-mono tracking-widest transition-colors"
            >
              <Trash2 size={12} />
              UNLOAD DATASET
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="absolute top-16 right-3 md:right-4 z-20 w-[270px] glass overflow-hidden border border-line shadow-xl">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-accent/[0.04] border-b border-line"
      >
        <span className="flex items-center gap-2">
          <UploadCloud size={13} className="text-accent" />
          <span className="text-[10px] font-mono tracking-[0.25em] text-dim">
            INGESTION PIPELINE
          </span>
        </span>
        <span className="text-dim">
          {expanded ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
        </span>
      </button>

      {expanded && (
        <div className="px-3 py-3">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (!loading) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !loading && inputRef.current?.click()}
            className={`relative cursor-pointer rounded-md border-2 border-dashed transition-all flex flex-col items-center justify-center gap-2 py-6 px-3 text-center ${
              dragOver
                ? 'border-accent bg-accent/[0.12] scale-[1.01]'
                : 'border-line hover:border-accent/40 bg-white/[0.02]'
            }`}
          >
            {loading ? (
              <React.Fragment>
                <Loader size={22} className="animate-spin text-accent" />
                <div className="font-mono text-[10px] text-mist tracking-wider">
                  PARSING DATASET...
                </div>
                <div className="w-full h-[2px] bg-white/10 overflow-hidden rounded mt-1">
                  <div className="h-full w-1/3 bg-accent boot-bar"></div>
                </div>
              </React.Fragment>
            ) : (
              <React.Fragment>
                <FileUp size={22} className={dragOver ? 'text-accent' : 'text-dim'} />
                <div className="text-[11px] text-mist font-medium">
                  {dragOver ? 'Release to upload' : 'Drop dataset here'}
                </div>
                <div className="font-mono text-[8.5px] text-dim leading-relaxed">
                  .nc / .nc4 (NetCDF) · .csv / .txt<br />or click to browse
                </div>
              </React.Fragment>
            )}

            <input
              ref={inputRef}
              type="file"
              accept=".nc,.nc4,.csv,.txt"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUploadFile(f);
                e.currentTarget.value = '';
              }}
            />
          </div>

          <div className="mt-2.5 font-mono text-[8.5px] text-dim leading-relaxed">
            Ingest custom ocean models, Argo CTDs, or gliders. System auto-extracts spatial bounds and layers.
          </div>
        </div>
      )}
    </div>
  );
};
