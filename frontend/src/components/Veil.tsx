import React from 'react';

export const Veil: React.FC<{
  show: boolean;
  bgOn: boolean;
  text: string;
  showSkip: boolean;
  onSkip: () => void;
}> = ({ show, bgOn, text, showSkip, onSkip }) => {
  return (
    <div
      className={`absolute inset-0 z-40 pointer-events-none transition-opacity duration-700 ${
        show ? 'opacity-100' : 'opacity-0'
      }`}
    >
      <div
        className={`absolute inset-0 bg-[#0d4257] transition-opacity ease-in ${
          bgOn ? 'opacity-100' : 'opacity-0'
        }`}
        style={{ transitionDuration: bgOn ? '1500ms' : '500ms' }}
      />
      <div
        className={`absolute bottom-24 left-1/2 -translate-x-1/2 transition-opacity duration-700 ${
          show ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <div className="font-mono text-[10.5px] tracking-[0.3em] text-[#a8dbe6] text-center whitespace-nowrap drop-shadow-md">
          {text}
        </div>
        {showSkip && (
          <div className="flex justify-center mt-3 pointer-events-auto">
            <button
              onClick={onSkip}
              className="border border-[#56d4e2]/40 text-[#7cc9d6] hover:bg-[#56d4e2]/15 bg-[#040a10]/60 rounded px-3 py-1 text-[9px] font-mono tracking-[0.25em] transition-all"
            >
              SKIP DIVE ANIMATION
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
