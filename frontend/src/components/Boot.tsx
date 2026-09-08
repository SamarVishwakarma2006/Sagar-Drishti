import React from 'react';
import { Waves } from 'lucide-react';

export const Boot: React.FC<{ msg: string }> = ({ msg }) => {
  return (
    <div className="absolute inset-0 z-50 bg-abyss flex flex-col items-center justify-center gap-6 select-none">
      <div className="p-3 rounded-full bg-accent/10 border border-accent/30 text-accent animate-pulse">
        <Waves size={36} />
      </div>

      <div className="text-center">
        <div className="text-[18px] font-semibold tracking-[0.35em] text-mist">
          SAGAR DRISHTI
        </div>
        <div className="text-[9px] tracking-[0.32em] text-dim mt-2">
          IMMERSIVE OCEAN VISUALIZATION OBSERVATORY
        </div>
      </div>

      <div className="w-48 h-[2.5px] bg-white/[0.08] overflow-hidden rounded">
        <div className="h-full w-1/3 bg-accent boot-bar" />
      </div>

      <div className="font-mono text-[9.5px] tracking-[0.2em] text-dim">{msg}</div>
    </div>
  );
};
