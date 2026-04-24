import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Navbar } from "@/components/nav/Navbar";
import { AGENTS } from "@/lib/constants";
import { useWebSocket } from "@/hooks/useWebSocket";

export const Route = createFileRoute("/command")({
  head: () => ({
    meta: [
      { title: "Command Center · TRIAGE" },
      { name: "description", content: "Interactive demo: chat with agents, inject crises, watch CMO override." },
    ],
  }),
  component: CommandCenter,
});

// ── Types ─────────────────────────────────────────────────────────────────

interface ChatEntry {
  id: string;
  role: "user" | "agent";
  agent?: string;
  prefix?: string;
  color?: string;
  text: string;
  ts: Date;
  streaming?: boolean;
}

interface FeedEntry {
  id: string;
  type: "OVERSIGHT" | "ALERT" | "ACTION" | "HANDOFF" | "INJECT" | "CRISIS";
  from: string;
  to: string;
  text: string;
  ts: Date;
  highlight?: boolean;
}

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? "";

const CRISIS_OPTIONS = [
  { value: "mass_casualty", label: "Mass Casualty Event", emoji: "🚨", severity: 0.85 },
  { value: "outbreak", label: "Disease Outbreak", emoji: "🦠", severity: 0.7 },
  { value: "equipment_failure", label: "Equipment Failure", emoji: "⚡", severity: 0.6 },
  { value: "staff_shortage", label: "Staff Shortage", emoji: "👩‍⚕️", severity: 0.65 },
];

const VIOLATION_OPTIONS = [
  { value: "protocol_breach", agent: "PHARMACY", label: "Drug Dosage Breach" },
  { value: "triage_mismatch", agent: "ER_TRIAGE", label: "Triage Tag Mismatch" },
  { value: "bed_overflow", agent: "ICU_MANAGEMENT", label: "ICU Bed Overflow" },
  { value: "drug_double_dose", agent: "PHARMACY", label: "Double Dose Order" },
  { value: "unauthorized_discharge", agent: "ER_TRIAGE", label: "Unauthorized Discharge" },
];

// ── Helper ────────────────────────────────────────────────────────────────

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function fmtTime(d: Date) {
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

// ── Main page ─────────────────────────────────────────────────────────────

function CommandCenter() {
  const [feed, setFeed] = useState<FeedEntry[]>([]);
  const ws = useWebSocket({
    onStateUpdate: (state) => {
      const actions = state?.recent_actions ?? [];
      if (actions.length > 0) {
        const a = actions[actions.length - 1];
        setFeed((prev) => [
          {
            id: uid(),
            type: "ACTION" as const,
            from: (a as any).agent ?? "AGENT",
            to: "SIMULATION",
            text: `${(a as any).action_type} — ${(a as any).reasoning ?? "executing decision"}`,
            ts: new Date(),
          },
          ...prev.slice(0, 49),
        ]);
      }
    },
  });

  function addFeedEntry(entry: Omit<FeedEntry, "id" | "ts">) {
    setFeed((prev) => [{ ...entry, id: uid(), ts: new Date() }, ...prev.slice(0, 49)]);
  }

  return (
    <div className="flex h-screen flex-col bg-background overflow-hidden">
      <Navbar />
      {/* Top bar */}
      <div
        className="flex items-center justify-between border-b border-border bg-surface px-6 py-2.5"
        style={{ borderLeft: "4px solid var(--agent-purple)" }}
      >
        <div className="flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-[var(--agent-purple)] pulse-dot" />
          <span className="font-mono text-[11px] uppercase tracking-widest text-[var(--agent-purple)]">
            COMMAND CENTER — INTERACTIVE DEMO
          </span>
        </div>
        <div className="flex items-center gap-6 font-mono text-[11px] text-text-muted">
          <span>Qwen2.5-0.5B-DPO ▸ ACTIVE</span>
          <span>6 Agents Online</span>
          <span className={ws.status === "connected" ? "text-stable-green" : "text-emergency"}>
            WS {ws.status === "connected" ? "●" : "○"}
          </span>
        </div>
      </div>

      {/* Three-panel layout */}
      <div className="grid min-h-0 flex-1 grid-cols-[1fr_340px_380px] gap-0 divide-x divide-border overflow-hidden">
        <AgentChatTerminal onFeedEntry={addFeedEntry} />
        <CrisisInjectionPanel onFeedEntry={addFeedEntry} />
        <LiveAgentFeed entries={feed} />
      </div>
    </div>
  );
}

// ── Panel 1: Agent Chat Terminal ──────────────────────────────────────────

function AgentChatTerminal({ onFeedEntry }: { onFeedEntry: (e: Omit<FeedEntry, "id" | "ts">) => void }) {
  const [selectedAgent, setSelectedAgent] = useState("CMO_OVERSIGHT");
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatEntry[]>([
    {
      id: uid(),
      role: "agent",
      agent: "CMO_OVERSIGHT",
      prefix: "CMO OVERSIGHT →",
      color: "#a855f7",
      text: "Command Center online. All 6 agents initialized. DPO-trained model loaded. Ask me anything about the crisis simulation.",
      ts: new Date(),
    },
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const send = useCallback(async () => {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput("");

    const userEntry: ChatEntry = { id: uid(), role: "user", text: msg, ts: new Date() };
    setHistory((h) => [...h, userEntry]);
    setLoading(true);

    // Streaming via SSE
    const agentEntry: ChatEntry = {
      id: uid(),
      role: "agent",
      agent: selectedAgent,
      prefix: "",
      color: AGENTS.find((a) => a.key === selectedAgent)?.color ?? "#a855f7",
      text: "",
      ts: new Date(),
      streaming: true,
    };
    setHistory((h) => [...h, agentEntry]);

    try {
      const res = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, agent: selectedAgent }),
      });

      if (!res.ok || !res.body) throw new Error("stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = "";
      let prefix = "";
      let color = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split("\n");
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const parsed = JSON.parse(line.slice(5).trim());
          if (!prefix) { prefix = parsed.prefix; color = parsed.color; }
          full += parsed.chunk;
          setHistory((h) =>
            h.map((e) =>
              e.id === agentEntry.id
                ? { ...e, text: full, prefix, color, streaming: !parsed.done }
                : e
            )
          );
        }
      }

      onFeedEntry({ type: "ACTION", from: selectedAgent, to: "USER", text: `Chat: ${full.slice(0, 80)}…` });
    } catch {
      // Fallback to non-streaming
      const r = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, agent: selectedAgent }),
      });
      const json = await r.json();
      const d = json.data ?? {};
      setHistory((h) =>
        h.map((e) =>
          e.id === agentEntry.id
            ? { ...e, text: d.response ?? "No response", prefix: d.prefix, color: d.color, streaming: false }
            : e
        )
      );
    } finally {
      setLoading(false);
    }
  }, [input, loading, selectedAgent, onFeedEntry]);

  const agentMeta = AGENTS.find((a) => a.key === selectedAgent);

  return (
    <div className="flex flex-col bg-surface">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Agent Chat Terminal</span>
        <div className="flex gap-1">
          {AGENTS.map((a) => (
            <button
              key={a.key}
              title={a.name}
              onClick={() => setSelectedAgent(a.key)}
              className="h-6 w-6 rounded-sm transition-all"
              style={{
                background: selectedAgent === a.key ? a.color : "var(--surface-3)",
                border: `1px solid ${selectedAgent === a.key ? a.color : "var(--border)"}`,
              }}
            />
          ))}
        </div>
      </div>

      {/* Active agent pill */}
      <div
        className="flex items-center gap-2 border-b border-border px-4 py-2"
        style={{ borderLeft: `3px solid ${agentMeta?.color ?? "#a855f7"}` }}
      >
        <span
          className="rounded px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-white"
          style={{ background: agentMeta?.color ?? "#a855f7" }}
        >
          {agentMeta?.name ?? selectedAgent}
        </span>
        <span className="text-[11px] text-text-secondary">{agentMeta?.role}</span>
      </div>

      {/* Chat history */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        <AnimatePresence initial={false}>
          {history.map((entry) => (
            <motion.div
              key={entry.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex ${entry.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {entry.role === "user" ? (
                <div
                  className="max-w-[75%] rounded-lg px-3 py-2 font-mono text-[12px] text-white"
                  style={{ background: "var(--clinical-blue)" }}
                >
                  {entry.text}
                </div>
              ) : (
                <div className="max-w-[85%]">
                  {entry.prefix && (
                    <div className="mb-1 font-mono text-[10px]" style={{ color: entry.color }}>
                      {entry.prefix}
                    </div>
                  )}
                  <div
                    className="rounded-lg border px-3 py-2 text-[12px] text-text-primary"
                    style={{ borderColor: entry.color ?? "var(--border)", borderLeft: `3px solid ${entry.color}` }}
                  >
                    {entry.text}
                    {entry.streaming && <span className="ml-1 inline-block h-3 w-0.5 bg-current animate-pulse" />}
                  </div>
                  <div className="mt-0.5 font-mono text-[9px] text-text-muted">{fmtTime(entry.ts)}</div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="border-t border-border p-3">
        <div className="flex gap-2">
          <input
            id="command-chat-input"
            className="flex-1 rounded border border-border bg-surface-2 px-3 py-2 font-mono text-[12px] text-text-primary outline-none focus:border-[var(--clinical-blue)] placeholder:text-text-muted"
            placeholder={`Ask ${agentMeta?.name ?? selectedAgent}…`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            disabled={loading}
          />
          <button
            id="command-chat-send"
            onClick={send}
            disabled={loading || !input.trim()}
            className="rounded px-4 py-2 font-mono text-[11px] uppercase tracking-wider text-white transition-opacity disabled:opacity-40"
            style={{ background: agentMeta?.color ?? "var(--agent-purple)" }}
          >
            {loading ? "…" : "Send"}
          </button>
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {["What's the current crisis status?", "How many patients are critical?", "What's the compliance rate?"].map((q) => (
            <button
              key={q}
              onClick={() => { setInput(q); }}
              className="rounded border border-border px-2 py-0.5 font-mono text-[9px] text-text-muted hover:text-text-primary hover:border-border-strong transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Panel 2: Crisis Injection Panel ──────────────────────────────────────

function CrisisInjectionPanel({ onFeedEntry }: { onFeedEntry: (e: Omit<FeedEntry, "id" | "ts">) => void }) {
  const [selectedCrisis, setSelectedCrisis] = useState(CRISIS_OPTIONS[0]);
  const [selectedViolation, setSelectedViolation] = useState(VIOLATION_OPTIONS[0]);
  const [crisisLoading, setCrisisLoading] = useState(false);
  const [violationLoading, setViolationLoading] = useState(false);
  const [lastCrisisResult, setLastCrisisResult] = useState<string | null>(null);
  const [lastViolationResult, setLastViolationResult] = useState<string | null>(null);

  async function injectCrisis() {
    setCrisisLoading(true);
    setLastCrisisResult(null);
    try {
      const r = await fetch(`${API_BASE}/api/inject/crisis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crisis_type: selectedCrisis.value, severity: selectedCrisis.severity, auto_step: true }),
      });
      const json = await r.json();
      const msg = json.data?.message ?? "Crisis injected.";
      setLastCrisisResult(msg);
      onFeedEntry({ type: "CRISIS", from: "DEMO_CONTROL", to: "ALL_AGENTS", text: msg, highlight: true });
    } catch {
      setLastCrisisResult("Backend unreachable — running in demo mode.");
      onFeedEntry({ type: "CRISIS", from: "DEMO_CONTROL", to: "ALL_AGENTS", text: `[DEMO] ${selectedCrisis.label} injected`, highlight: true });
    } finally {
      setCrisisLoading(false);
    }
  }

  async function injectViolation() {
    setViolationLoading(true);
    setLastViolationResult(null);
    try {
      const r = await fetch(`${API_BASE}/api/inject/violation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent: selectedViolation.agent, violation_type: selectedViolation.value }),
      });
      const json = await r.json();
      const msg = json.data?.message ?? "Violation injected.";
      setLastViolationResult(msg);
      onFeedEntry({ type: "OVERSIGHT", from: "CMO_OVERSIGHT", to: selectedViolation.agent, text: `⚠️ CMO CAUGHT: ${selectedViolation.label}`, highlight: true });
    } catch {
      setLastViolationResult("Backend unreachable — demo mode.");
      onFeedEntry({ type: "OVERSIGHT", from: "CMO_OVERSIGHT", to: selectedViolation.agent, text: `⚠️ [DEMO] CMO intercepted: ${selectedViolation.label}`, highlight: true });
    } finally {
      setViolationLoading(false);
    }
  }

  return (
    <div className="flex flex-col bg-surface overflow-y-auto">
      <div className="border-b border-border px-4 py-2.5">
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Crisis Injection Panel</span>
      </div>

      {/* Crisis injection */}
      <div className="border-b border-border p-4 space-y-3">
        <div className="font-mono text-[11px] uppercase tracking-wider text-emergency">▸ Inject Crisis</div>
        <div className="space-y-2">
          {CRISIS_OPTIONS.map((c) => (
            <button
              key={c.value}
              id={`crisis-btn-${c.value}`}
              onClick={() => setSelectedCrisis(c)}
              className="w-full rounded border px-3 py-2 text-left transition-all"
              style={{
                borderColor: selectedCrisis.value === c.value ? "var(--emergency-red)" : "var(--border)",
                background: selectedCrisis.value === c.value ? "var(--emergency-red-light)" : "var(--surface-2)",
              }}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] text-text-primary">
                  {c.emoji} {c.label}
                </span>
                <span
                  className="font-mono text-[9px] uppercase"
                  style={{ color: c.severity > 0.75 ? "var(--emergency-red)" : "var(--warning-amber)" }}
                >
                  SEV {Math.round(c.severity * 100)}%
                </span>
              </div>
            </button>
          ))}
        </div>
        <button
          id="inject-crisis-btn"
          onClick={injectCrisis}
          disabled={crisisLoading}
          className="w-full rounded py-2.5 font-mono text-[11px] uppercase tracking-wider text-white transition-opacity disabled:opacity-50"
          style={{ background: "var(--emergency-red)" }}
        >
          {crisisLoading ? "Injecting…" : `⚡ INJECT ${selectedCrisis.label.toUpperCase()}`}
        </button>
        {lastCrisisResult && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded border border-border bg-surface-2 px-3 py-2 font-mono text-[10px] text-stable"
          >
            ✓ {lastCrisisResult}
          </motion.div>
        )}
      </div>

      {/* Violation injection */}
      <div className="p-4 space-y-3">
        <div className="font-mono text-[11px] uppercase tracking-wider text-[var(--agent-purple)]">▸ Override & Catch Mode</div>
        <p className="text-[11px] text-text-secondary">
          Inject a bad decision — watch the CMO Oversight agent intercept and correct it in real time.
        </p>
        <div className="space-y-2">
          {VIOLATION_OPTIONS.map((v) => {
            const agentMeta = AGENTS.find((a) => a.key === v.agent);
            return (
              <button
                key={v.value}
                id={`violation-btn-${v.value}`}
                onClick={() => setSelectedViolation(v)}
                className="w-full rounded border px-3 py-2 text-left transition-all"
                style={{
                  borderColor: selectedViolation.value === v.value ? "var(--agent-purple)" : "var(--border)",
                  background: selectedViolation.value === v.value ? "var(--agent-purple-light)" : "var(--surface-2)",
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-text-primary">{v.label}</span>
                  <span className="font-mono text-[9px]" style={{ color: agentMeta?.color }}>
                    {v.agent.replace("_", " ")}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
        <button
          id="inject-violation-btn"
          onClick={injectViolation}
          disabled={violationLoading}
          className="w-full rounded py-2.5 font-mono text-[11px] uppercase tracking-wider text-white transition-opacity disabled:opacity-50"
          style={{ background: "var(--agent-purple)" }}
        >
          {violationLoading ? "Injecting…" : "⚠️ INJECT VIOLATION → CMO CATCH"}
        </button>
        {lastViolationResult && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded border border-[var(--agent-purple)] bg-[var(--agent-purple-light)] px-3 py-2 font-mono text-[10px] text-[var(--agent-purple)]"
          >
            ⚠️ {lastViolationResult}
          </motion.div>
        )}

        {/* Score card */}
        <div className="mt-4 rounded border border-border bg-surface-2 p-3 space-y-2">
          <div className="font-mono text-[9px] uppercase tracking-wider text-text-muted">Benchmark Score</div>
          {[
            { label: "Overall", val: "90/100", color: "var(--stable-green)" },
            { label: "Compliance", val: "94.2%", color: "var(--clinical-blue)" },
            { label: "Survival Rate", val: "100%", color: "var(--stable-green)" },
            { label: "CMO Catch Rate", val: "100%", color: "var(--agent-purple)" },
          ].map((s) => (
            <div key={s.label} className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-text-secondary">{s.label}</span>
              <span style={{ color: s.color }}>{s.val}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Panel 3: Live Agent Feed ──────────────────────────────────────────────

const TYPE_COLOR: Record<string, string> = {
  OVERSIGHT: "var(--agent-purple)",
  ALERT: "var(--emergency-red)",
  ACTION: "var(--clinical-blue)",
  HANDOFF: "var(--stable-green)",
  REQUEST: "var(--text-muted)",
  INJECT: "var(--warning-amber)",
  CRISIS: "var(--emergency-red)",
};

function LiveAgentFeed({ entries }: { entries: FeedEntry[] }) {
  const ref = useRef<HTMLDivElement>(null);

  return (
    <div className="flex flex-col bg-surface">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Live Agent Feed</span>
        <span className="font-mono text-[10px] text-text-muted">{entries.length} events</span>
      </div>

      <div ref={ref} className="flex-1 overflow-y-auto p-2 space-y-1.5">
        <AnimatePresence initial={false}>
          {entries.map((e) => (
            <motion.div
              key={e.id}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className="rounded border bg-surface px-3 py-2"
              style={{
                borderColor: e.highlight ? TYPE_COLOR[e.type] : "var(--border)",
                borderLeft: `3px solid ${TYPE_COLOR[e.type] ?? "var(--border)"}`,
                background: e.highlight ? `${TYPE_COLOR[e.type]}10` : "var(--surface)",
              }}
            >
              <div className="flex items-center gap-2 font-mono text-[9px]">
                <span className="text-text-muted">{fmtTime(e.ts)}</span>
                <span style={{ color: TYPE_COLOR[e.type] }}>{e.type}</span>
                <span className="text-text-primary">{e.from}</span>
                <span className="text-text-muted">→</span>
                <span className="text-text-secondary">{e.to}</span>
              </div>
              <div className="mt-1 text-[11px] text-text-primary leading-snug">{e.text}</div>
            </motion.div>
          ))}
        </AnimatePresence>

        {entries.length === 0 && (
          <div className="grid h-32 place-items-center font-mono text-[11px] text-text-muted">
            waiting for agent activity…
          </div>
        )}
      </div>
    </div>
  );
}
