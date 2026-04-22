import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState, useRef } from "react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useSimulation } from "@/hooks/useSimulation";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { ArrowLeft, Maximize2, Minimize2 } from "lucide-react";

export const Route = createFileRoute("/pitch")({
  head: () => ({
    meta: [
      { title: "Pitch Mode · TRIAGE" },
      { name: "description", content: "Fullscreen presenter mode for the TRIAGE hackathon pitch." },
      { name: "robots", content: "noindex" },
    ],
  }),
  component: PitchMode,
});

interface ScriptEvent {
  time: number;
  label?: string;
  narrative: string;
  reward?: number;
  episodeBadge?: string;
}

const PITCH_SCRIPT: ScriptEvent[] = [
  { time: 0, label: "Episode 1 — Baseline (untrained model)", narrative: "27 patients. 4 ICU beds. Agents have never trained before.", reward: 0, episodeBadge: "EP 1 · BASELINE" },
  { time: 8, narrative: "Watch — ER agent makes a critical triage error on #PT-0019.", reward: 24, episodeBadge: "EP 1 · BASELINE" },
  { time: 18, narrative: "CMO oversight misses it. Patient deteriorates.", reward: 32, episodeBadge: "EP 1 · BASELINE" },
  { time: 28, narrative: "Episode ends. Reward: 47.3. Three patients lost.", reward: 47.3, episodeBadge: "EP 1 · BASELINE" },
  { time: 38, label: "After DPO Training — StrategyMemory active", narrative: "Same crisis. Same agents. After training.", reward: 47.3, episodeBadge: "EP 2 · TRAINED" },
  { time: 48, narrative: "Same violation injected — ER triage error on #PT-0031.", reward: 60, episodeBadge: "EP 2 · TRAINED" },
  { time: 54, narrative: "CMO catches it immediately. Corrects in 2 steps.", reward: 72, episodeBadge: "EP 2 · TRAINED" },
  { time: 70, narrative: "Reward: 84.7 — that's +37.4 improvement. All critical patients stable.", reward: 84.7, episodeBadge: "EP 2 · TRAINED" },
  { time: 82, narrative: "The improvement curve across 10 episodes — we'll see it next.", reward: 87, episodeBadge: "EP 2 · TRAINED" },
];

function PitchMode() {
  const sim = useSimulation();
  const [tick, setTick] = useState(0);
  const [paused, setPaused] = useState(false);
  const [eventIdx, setEventIdx] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  async function toggleFullscreen() {
    const node = containerRef.current;
    if (!node) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen?.().catch(() => {});
      return;
    }
    await node.requestFullscreen?.().catch(() => {});
  }

  // Time advance
  useEffect(() => {
    if (paused) return;
    const t = setInterval(() => setTick((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [paused]);

  // Advance script
  useEffect(() => {
    const next = PITCH_SCRIPT.findIndex((e, i) => i > eventIdx && tick >= e.time);
    if (next > -1) setEventIdx(next);
  }, [tick, eventIdx]);

  const cur = PITCH_SCRIPT[eventIdx];
  const target = cur.reward ?? 0;

  // Keyboard controls
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === " ") {
        e.preventDefault();
        setPaused((p) => !p);
      }
      if (e.key === "ArrowRight") {
        setEventIdx((i) => Math.min(PITCH_SCRIPT.length - 1, i + 1));
      }
      if (e.key === "ArrowLeft") {
        setEventIdx((i) => Math.max(0, i - 1));
      }
      if (e.key === "r" || e.key === "R") {
        setTick(0);
        setEventIdx(0);
      }
      if (e.key === "f" || e.key === "F") {
        void toggleFullscreen();
      }
      if (e.key === "Escape") {
        document.exitFullscreen?.().catch(() => {});
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    function onFullscreenChange() {
      setIsFullscreen(Boolean(document.fullscreenElement));
    }
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  const survival = 61 + Math.min(32, (target / 87) * 32);
  const compliance = 70 + Math.min(20, (target / 87) * 20);
  const oversight = 55 + Math.min(40, (target / 87) * 40);

  return (
    <div ref={containerRef} className="fixed inset-0 flex flex-col bg-background text-text-primary">
      {/* Top crisis bar */}
      <div
        className="flex items-center justify-between border-b border-border bg-surface px-8 py-4"
        style={{ borderLeft: "6px solid var(--emergency-red)" }}
      >
        <div className="flex items-center gap-4">
          <span className="h-3 w-3 rounded-full bg-emergency pulse-dot" />
          <span className="font-mono text-[14px] uppercase tracking-widest text-emergency">
            MASS CASUALTY EVENT — ACTIVE
          </span>
        </div>
        <div className="font-mono text-[14px] uppercase tracking-wider text-text-secondary">
          {cur.episodeBadge}
        </div>
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={() => void toggleFullscreen()}
            className="inline-flex items-center gap-2 border border-border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-text-secondary hover:text-text-primary"
          >
            {isFullscreen ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
            {isFullscreen ? "Windowed" : "Fullscreen"}
          </button>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-text-muted hover:text-text-primary"
          >
            <ArrowLeft className="h-3 w-3" /> Exit
          </Link>
        </div>
      </div>

      {/* Center split */}
      <div className="grid flex-1 grid-cols-[45%_55%] overflow-hidden">
        <PitchPatientBoard patients={sim.patients} />
        <PitchMessageFeed messages={sim.messages} />
      </div>

      {/* Bottom reward strip */}
      <div className="border-t border-border bg-surface px-8 py-5">
        <div className="grid grid-cols-[1fr_auto] items-center gap-8">
          <div>
            <div className="flex items-baseline gap-3">
              <div className="font-mono text-[12px] uppercase tracking-widest text-text-muted">
                Reward
              </div>
              <PitchRewardCounter target={target} />
              <div className="font-mono text-[14px] text-text-muted">/ 100</div>
            </div>
            <div className="mt-2 h-3 w-full overflow-hidden bg-surface-2" style={{ borderRadius: 2 }}>
              <motion.div
                className="h-full"
                initial={{ width: 0 }}
                animate={{ width: `${target}%` }}
                transition={{ duration: 1.2, ease: "easeOut" }}
                style={{
                  background:
                    target > 70
                      ? "var(--stable-green)"
                      : target > 40
                        ? "var(--warning-amber)"
                        : "var(--emergency-red)",
                }}
              />
            </div>
            <div className="mt-3 flex gap-8 font-mono text-[12px]">
              <Stat label="SURVIVAL" value={`${survival.toFixed(0)}%`} />
              <Stat label="COMPLIANCE" value={`${compliance.toFixed(0)}%`} />
              <Stat label="OVERSIGHT" value={`${oversight.toFixed(0)}%`} />
            </div>
          </div>
          <div className="text-right font-mono text-[10px] uppercase tracking-widest text-text-muted">
            <div>[SPACE Pause] [→ Next] [R Reset] [F Fullscreen] [ESC Exit]</div>
            <div className="mt-1 text-text-secondary">
              {paused ? "PAUSED" : `Tick ${tick}s · Event ${eventIdx + 1}/${PITCH_SCRIPT.length}`}
            </div>
          </div>
        </div>
      </div>

      <PitchNarrative key={eventIdx} text={cur.narrative} label={cur.label} />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-text-muted">{label}</div>
      <div className="text-2xl text-text-primary">{value}</div>
    </div>
  );
}

function PitchRewardCounter({ target }: { target: number }) {
  const v = useMotionValue(0);
  const display = useTransform(v, (latest) => latest.toFixed(1));
  useEffect(() => {
    const c = animate(v, target, { duration: 1.2, ease: "easeOut" });
    return c.stop;
  }, [target, v]);
  const color =
    target > 70 ? "var(--stable-green)" : target > 40 ? "var(--warning-amber)" : "var(--emergency-red)";
  return (
    <motion.span className="font-mono text-[72px] leading-none" style={{ color }}>
      {display}
    </motion.span>
  );
}

function PitchNarrative({ text, label }: { text: string; label?: string }) {
  const [shown, setShown] = useState("");
  const idx = useRef(0);
  useEffect(() => {
    setShown("");
    idx.current = 0;
    const t = setInterval(() => {
      if (idx.current >= text.length) {
        clearInterval(t);
        return;
      }
      setShown(text.slice(0, ++idx.current));
    }, 25);
    return () => clearInterval(t);
  }, [text]);
  return (
    <div className="absolute inset-x-0 bottom-32 flex justify-center">
      <div
        className="max-w-3xl border border-border bg-surface px-6 py-3 text-center shadow-lg"
        style={{ borderRadius: 6, borderLeft: "4px solid var(--clinical-blue)" }}
      >
        {label && (
          <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-primary">
            {label}
          </div>
        )}
        <div className="font-display text-[20px] italic text-text-primary">{shown}</div>
      </div>
    </div>
  );
}

function PitchPatientBoard({ patients }: { patients: ReturnType<typeof useSimulation>["patients"] }) {
  const sorted = [...patients].sort((a, b) => b.triageScore - a.triageScore).slice(0, 14);
  return (
    <div className="border-r border-border bg-surface">
      <div className="border-b border-border px-6 py-3 font-mono text-[12px] uppercase tracking-widest text-text-muted">
        Patient Board · top {sorted.length}
      </div>
      <div className="divide-y divide-border">
        {sorted.map((p) => (
          <div
            key={p.id}
            className="grid grid-cols-[100px_1fr_140px_60px] items-center gap-3 px-6 py-2.5"
          >
            <div className="font-mono text-[14px] text-text-primary">{p.id}</div>
            <div className="text-[14px] text-text-primary truncate">{p.name} · {p.condition}</div>
            <StatusBadge status={p.status} />
            <div className="text-right font-mono text-[14px]" style={{ color: p.triageScore >= 8 ? "var(--emergency-red)" : p.triageScore >= 5 ? "var(--warning-amber)" : "var(--stable-green)" }}>
              {p.triageScore}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PitchMessageFeed({ messages }: { messages: ReturnType<typeof useSimulation>["messages"] }) {
  const recent = messages.slice(0, 12);
  const typeColor: Record<string, string> = {
    OVERSIGHT: "var(--agent-purple)",
    ALERT: "var(--emergency-red)",
    HANDOFF: "var(--stable-green)",
    ACTION: "var(--clinical-blue)",
    REQUEST: "var(--text-muted)",
  };
  return (
    <div className="bg-surface">
      <div className="border-b border-border px-6 py-3 font-mono text-[12px] uppercase tracking-widest text-text-muted">
        Agent Message Feed
      </div>
      <div className="space-y-2 p-3 overflow-auto">
        {recent.map((m) => (
          <motion.div
            key={m.id}
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            className="border border-border bg-surface px-4 py-2.5"
            style={{ borderRadius: 4, borderLeft: `4px solid ${typeColor[m.type]}` }}
          >
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <span style={{ color: typeColor[m.type] }}>{m.type}</span>
              <span className="text-text-primary">{m.from}</span>
              <span className="text-text-muted">→</span>
              <span className="text-text-primary">{m.to}</span>
            </div>
            <div className="mt-1 text-[14px] text-text-primary">{m.content}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
