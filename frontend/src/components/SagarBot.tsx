import React, { useState, useEffect, useRef } from 'react';
import { useApp } from '../store/oceanStore';
import { Ocean, CMAPS, VARIABLES, fmtDepth, timeUTC } from '../services/syntheticOcean';
import { ViewportContext, ChatMessage } from '../types/ocean';
import { OceanAPI } from '../services/api';
import { Bot, Send, X, Sparkles, Loader2 } from 'lucide-react';

export const SagarBot: React.FC = () => {
  const s = useApp();
  const site = s.site;
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init',
      role: 'bot',
      text: 'SagarBot online. I am actively grounded in your 3D viewport telemetry — ask me about water column stratification, thermoclines, oxygen minimum zones, or in-situ Argo floats!',
      timestamp: 'Now',
      provider: 'Oceanographic Co-Pilot',
    },
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-collapse when Inspector opens to avoid spatial overlap
  useEffect(() => {
    if (s.selection) setOpen(false);
  }, [s.selection]);

  // Cooldown countdown timer
  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setInterval(() => {
      setCooldown((c) => Math.max(0, c - 1));
    }, 1000);
    return () => clearInterval(timer);
  }, [cooldown]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, typing]);

  if (!site) return null;

  const curVar = VARIABLES.find((v) => v.key === s.variable)!;
  const curVal = Ocean.variableAt(site, 0, 0, s.depth, s.timeOffset, s.variable, 0);
  const variableLabel = CMAPS[s.variable].label;
  const nearbyFloats = Ocean.floatsAt(site).slice(0, 3).map((f) => `Argo ${f.id} (${Math.round(f.parkingDepth)}m)`);

  // Build live viewport telemetry context
  const buildLiveContext = (): ViewportContext => ({
    active_site: site.name,
    coordinates: { lat: site.lat, lon: site.lon },
    current_depth: fmtDepth(s.depth),
    variable: variableLabel,
    current_value: `${curVal.toFixed(curVar.digits)} ${curVar.unit}`,
    time_offset: `${s.timeOffset >= 0 ? '+' : ''}${s.timeOffset}h (${timeUTC(s.timeOffset)})`,
    nearby_floats: nearbyFloats,
    custom_data: s.customDataMode,
  });

  const handleSend = async () => {
    const text = input.trim();
    if (!text || typing || cooldown > 0) return;

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setTyping(true);
    setCooldown(3); // 3-second client rate-limiting cooldown

    const context = buildLiveContext();

    try {
      const resp = await OceanAPI.chatWithBot(text, context, s.apiKey, s.llmProvider);
      setMessages((prev) => [
        ...prev,
        {
          id: `bot_${Date.now()}`,
          role: 'bot',
          text: resp.reply,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          provider: resp.provider,
        },
      ]);
    } catch (e: any) {
      const isRateLimit = e?.message?.toLowerCase().includes('rate limit');
      setMessages((prev) => [
        ...prev,
        {
          id: `bot_${Date.now()}`,
          role: 'bot',
          text: isRateLimit
            ? `⚠️ ${e.message}`
            : `Telemetry grounding note: ${e?.message || 'Failed to generate response'}.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          provider: isRateLimit ? 'Rate Limiter' : 'Offline Physics Engine',
        },
      ]);
    } finally {
      setTyping(false);
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        title="Open SagarBot AI Ocean Assistant"
        className="absolute bottom-4 right-[106px] md:right-[114px] z-30 h-9 px-3 rounded-full glass flex items-center gap-2 text-accent hover:bg-accent/15 hover:scale-105 transition-all shadow-xl border border-accent/40 group"
      >
        <Bot size={15} className="text-accent" />
        <span className="font-mono text-[9.5px] tracking-wider text-mist group-hover:text-accent font-medium hidden sm:inline">
          SAGARBOT
        </span>
        <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse"></span>
      </button>
    );
  }

  return (
    <div className="absolute right-3 md:right-4 top-14 bottom-4 z-40 w-[320px] max-md:w-[calc(100vw-24px)] glass flex flex-col overflow-hidden border border-line shadow-2xl">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-line bg-accent/[0.04]">
        <Sparkles size={15} className="text-accent" />
        <div className="flex-1 min-w-0">
          <div className="text-[12px] font-semibold text-mist flex items-center gap-1.5">
            SagarBot
            <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-accent/20 text-accent font-semibold uppercase tracking-wider">
              ONLINE
            </span>
          </div>
          <div className="font-mono text-[8px] text-dim tracking-wider">
            LIVE OCEANOGRAPHIC CO-PILOT
          </div>
        </div>

        <button
          onClick={() => setOpen(false)}
          className="text-dim hover:text-mist p-1"
          title="Close SagarBot"
        >
          <X size={14} />
        </button>
      </div>

      {/* Live Viewport Grounding HUD */}
      <div className="px-3 py-2 border-b border-line bg-accent/[0.03]">
        <div className="flex items-center gap-1.5 mb-1">
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse"></span>
          <span className="font-mono text-[8.5px] tracking-[0.25em] text-dim">
            LIVE VIEWPORT GROUNDING
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono text-[8.5px]">
          <div>
            <span className="text-dim">DEPTH:</span>{' '}
            <span className="text-accent">{fmtDepth(s.depth)}</span>
          </div>
          <div>
            <span className="text-dim">OBSERVED:</span>{' '}
            <span className="text-accent">{curVal.toFixed(1)} {curVar.unit}</span>
          </div>
          <div>
            <span className="text-dim">TIME:</span>{' '}
            <span className="text-mist">{s.timeOffset >= 0 ? `+${s.timeOffset}` : s.timeOffset}h</span>
          </div>
          <div>
            <span className="text-dim">DATA:</span>{' '}
            <span className="text-mist">{s.customDataMode ? 'CUSTOM' : 'BASELINE'}</span>
          </div>
          <div className="col-span-2 truncate">
            <span className="text-dim">SITE:</span>{' '}
            <span className="text-mist">{site.name}</span>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2.5 space-y-2.5">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[88%] px-2.5 py-1.5 rounded text-[11px] leading-relaxed ${
                m.role === 'user'
                  ? 'bg-accent/15 text-mist border border-accent/30'
                  : 'bg-white/[0.04] text-mist/95 border border-line'
              }`}
            >
              {m.text}
            </div>
            {m.provider && (
              <span className="text-[7.5px] font-mono text-dim/70 mt-0.5 px-1">
                via {m.provider}
              </span>
            )}
          </div>
        ))}

        {typing && (
          <div className="flex items-center gap-1.5 px-2.5 py-2 rounded bg-white/[0.04] border border-line w-fit">
            <Loader2 size={12} className="animate-spin text-accent" />
            <span className="font-mono text-[9px] text-dim">Grounding in water physics...</span>
          </div>
        )}
      </div>

      {/* Quick Prompt Chips */}
      <div className="px-3 pt-1.5 pb-1 flex gap-1 flex-wrap border-t border-line">
        {[
          'Thermocline',
          'Halocline Barrier Layer',
          'OMZ Core',
          'Currents & Eddies',
          'Argo Floats',
        ].map((p) => (
          <button
            key={p}
            onClick={() => setInput(`Explain the ${p} at this depth and region.`)}
            className="text-[8px] font-mono text-dim hover:text-accent border border-line hover:border-accent/40 rounded px-1.5 py-0.5 transition-colors"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Message Input */}
      <div className="px-2.5 py-2 border-t border-line bg-black/20">
        <div className="flex items-center gap-2 bg-white/[0.04] border border-line rounded px-2 py-1.5 focus-within:border-accent/50">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSend();
            }}
            placeholder={cooldown > 0 ? `Rate limit cooldown (${cooldown}s)...` : "Ask about this ocean depth..."}
            className="bg-transparent outline-none text-[11px] text-mist placeholder:text-dim/70 w-full"
          />
          {cooldown > 0 ? (
            <span className="text-[9px] font-mono text-accent bg-accent/15 px-1.5 py-0.5 rounded border border-accent/30 animate-pulse">
              {cooldown}s
            </span>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim() || typing || cooldown > 0}
              className="text-accent hover:text-mist disabled:opacity-30 disabled:cursor-not-allowed p-0.5"
            >
              <Send size={13} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
