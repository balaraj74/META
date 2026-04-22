/**
 * useSimulation
 *
 * Primary data source for the dashboard.
 *
 * Strategy
 * ────────
 * 1. On mount, probe /api/health.
 * 2. If backend is UP → open WebSocket, start a simulation, receive live
 *    state via WS updates, and project it into the frontend type models.
 * 3. If backend is DOWN (or WS falls back to mock) → drive with the existing
 *    seedable local simulation (unchanged behaviour for demos without a server).
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import type {
  AgentMessage,
  AgentState,
  DecisionLogEntry,
  EpisodeMetrics,
  Patient,
  PatientStatus,
  Resources,
} from "@/lib/types";
import {
  initialAgents,
  initialMetrics,
  initialPatients,
  initialResources,
  makeDecision,
  makeMessage,
  tickAgents,
  tickPatients,
  tickResources,
} from "@/lib/simulation";
import {
  pingBackend,
  startSimulation,
  stopSimulation,
} from "@/lib/api";
import type { BackendStateUpdate } from "@/lib/api";
import { useWebSocket } from "./useWebSocket";

// ── Adapter: translate backend patient → frontend Patient ─────────────────────

function adaptPatient(bp: BackendStateUpdate["patients"][number], idx: number): Patient {
  const severityToStatus: Record<string, PatientStatus> = {
    CRITICAL: "CRITICAL",
    SERIOUS: "SERIOUS",
    STABLE: "STABLE",
    DISCHARGED: "DISCHARGED",
    DECEASED: "DECEASED",
  };
  const tagToStatus: Record<string, PatientStatus> = {
    RED: "CRITICAL",
    YELLOW: "SERIOUS",
    GREEN: "STABLE",
    BLACK: "DECEASED",
  };

  const status: PatientStatus = !bp.is_alive
    ? "DECEASED"
    : bp.is_discharged
      ? "DISCHARGED"
      : (severityToStatus[bp.severity?.toUpperCase?.()] ??
          tagToStatus[bp.triage_tag?.toUpperCase?.()] ??
          "STABLE");

  // Derive a triage score from severity
  const scoreMap: Record<string, number> = {
    CRITICAL: 9,
    SERIOUS: 6,
    STABLE: 3,
    DISCHARGED: 1,
    DECEASED: 0,
  };

  return {
    id: bp.id,
    name: bp.name,
    age: bp.age,
    condition: bp.condition,
    status,
    ward: bp.location ?? "ER",
    triageScore: scoreMap[status] ?? 3,
    assignedAgent: "CMO OVERSIGHT",
    admittedAt: new Date(),
    lastUpdated: new Date(),
  };
}

// ── Adapter: backend resources → frontend Resources ───────────────────────────

function adaptResources(br: BackendStateUpdate["resources"]): Resources {
  return {
    icuBeds: { used: br.icu_beds_used ?? 0, total: br.icu_beds_total ?? 20 },
    ventilators: {
      used: (br.ventilators_total ?? 15) - (br.ventilators_available ?? 6),
      total: br.ventilators_total ?? 15,
    },
    bloodSupply: Math.round(((br.drug_inventory?.["blood"] ?? 78) / 100) * 100),
    staffOnDuty: {
      used: (br.staff_total ?? 45) - (br.staff_available ?? 7),
      total: br.staff_total ?? 45,
    },
  };
}

// ── Adapter: backend agents → frontend AgentState[] ───────────────────────────

function adaptAgents(backendAgents: BackendStateUpdate["agents"]): AgentState[] {
  const nameMap: Record<string, string> = {
    cmo_oversight: "CMO OVERSIGHT",
    er_triage: "ER TRIAGE",
    icu_management: "ICU MANAGEMENT",
    pharmacy: "PHARMACY",
    hr_rostering: "HR ROSTERING",
    it_systems: "IT SYSTEMS",
  };
  const keyMap: Record<string, AgentState["key"]> = {
    cmo_oversight: "CMO_OVERSIGHT",
    er_triage: "ER_TRIAGE",
    icu_management: "ICU_MANAGEMENT",
    pharmacy: "PHARMACY",
    hr_rostering: "HR_ROSTERING",
    it_systems: "IT_SYSTEMS",
  };

  return backendAgents.map((ba) => ({
    key: keyMap[ba.agent_type] ?? ("CMO_OVERSIGHT" as AgentState["key"]),
    name: nameMap[ba.agent_type] ?? ba.agent_type.replace(/_/g, " ").toUpperCase(),
    status: ba.is_active ? "ACTIVE" : "WAITING",
    currentAction: ba.last_action ?? "Monitoring queue",
    messagesSent: ba.actions_taken,
  }));
}

// ── Hook ───────────────────────────────────────────────────────────────────────

export function useSimulation() {
  // ── state ──────────────────────────────────────────────────────────────────
  const [patients, setPatients] = useState<Patient[]>(() => initialPatients(24));
  const [agents, setAgents] = useState<AgentState[]>(() => initialAgents());
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [metrics] = useState<EpisodeMetrics[]>(() => initialMetrics());
  const [resources, setResources] = useState<Resources>(() => initialResources());
  const [decisions, setDecisions] = useState<DecisionLogEntry[]>([]);
  const [flashed, setFlashed] = useState<Set<string>>(new Set());
  const [isRunning, setIsRunning] = useState(true);
  const [episode, setEpisode] = useState(7);
  const [step, setStep] = useState(247);
  const [elapsed, setElapsed] = useState(6153);
  const [speed, setSpeed] = useState(1);

  // ── backend mode flag ──────────────────────────────────────────────────────
  const [isLive, setIsLive] = useState(false);
  const liveRef = useRef(false);

  // ── mock timers ────────────────────────────────────────────────────────────
  const ptTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const msgTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearMock = () => {
    if (ptTimer.current) clearInterval(ptTimer.current);
    if (msgTimer.current) clearInterval(msgTimer.current);
    if (stepTimer.current) clearInterval(stepTimer.current);
  };

  // ── backend state handler ──────────────────────────────────────────────────
  const handleBackendState = useCallback((bs: BackendStateUpdate) => {
    if (!liveRef.current) return;

    const adapted = bs.patients?.map((p, i) => adaptPatient(p, i)) ?? [];
    setPatients(adapted);

    if (bs.resources) setResources(adaptResources(bs.resources));
    if (bs.agents?.length) setAgents(adaptAgents(bs.agents));

    setStep(bs.step ?? 0);
    setElapsed((e) => e + 1);
  }, []);

  // ── WebSocket ──────────────────────────────────────────────────────────────
  const ws = useWebSocket({ onStateUpdate: handleBackendState });

  // ── Probe backend on mount ─────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    (async () => {
      const alive = await pingBackend();
      if (!mounted) return;
      if (alive) {
        liveRef.current = true;
        setIsLive(true);
        // Start the simulation on the backend
        await startSimulation({ auto_step: true, step_delay_ms: 400 });
      }
    })();
    return () => {
      mounted = false;
      // Stop backend sim when component unmounts
      if (liveRef.current) {
        stopSimulation().catch(() => {/* ignore */});
      }
    };
  }, []);

  useEffect(() => {
    if (!isLive) return;
    if (ws.status !== "fallback" && ws.status !== "disconnected") return;
    liveRef.current = false;
    setIsLive(false);
    stopSimulation().catch(() => {/* ignore */});
  }, [isLive, ws.status]);

  // ── Mock simulation (runs only when NOT in live mode) ──────────────────────
  useEffect(() => {
    if (isLive || !isRunning) { clearMock(); return; }

    ptTimer.current = setInterval(() => {
      setPatients((prev) => {
        const { next, flashed: f } = tickPatients(prev);
        setFlashed(new Set(f));
        return next;
      });
      setResources((r) => tickResources(r));
    }, 2000 / speed);

    msgTimer.current = setInterval(() => {
      setPatients((curP) => {
        const m = makeMessage(curP);
        setMessages((prev) => [m, ...prev].slice(0, 200));
        setAgents((a) => tickAgents(a, m));
        if (Math.random() < 0.5) {
          const d = makeDecision(curP);
          setDecisions((prev) => [d, ...prev].slice(0, 80));
        }
        return curP;
      });
    }, 1500 / speed);

    stepTimer.current = setInterval(() => {
      setStep((s) => s + 1);
      setElapsed((e) => e + 1);
    }, 1000 / speed);

    return clearMock;
  }, [isRunning, speed, isLive]);

  // ── Controls ───────────────────────────────────────────────────────────────

  const toggleSimulation = useCallback(() => setIsRunning((v) => !v), []);

  const resetSimulation = useCallback(() => {
    setPatients(initialPatients(24));
    setAgents(initialAgents());
    setMessages([]);
    setResources(initialResources());
    setDecisions([]);
    setStep(0);
    setElapsed(0);
    setEpisode(1);
  }, []);

  const stepForward = useCallback(() => {
    if (isLive) {
      ws.sendCommand("step");
    } else {
      setPatients((prev) => {
        const { next, flashed: f } = tickPatients(prev);
        setFlashed(new Set(f));
        const m = makeMessage(next);
        setMessages((p) => [m, ...p].slice(0, 200));
        setAgents((a) => tickAgents(a, m));
        return next;
      });
      setStep((s) => s + 1);
    }
  }, [isLive, ws]);

  const currentReward = metrics[episode - 1]?.rewardScore ?? 87;

  return {
    patients,
    agents,
    messages,
    metrics,
    resources,
    decisions,
    flashed,
    isRunning,
    episode,
    setEpisode,
    step,
    elapsed,
    currentReward,
    speed,
    setSpeed,
    toggleSimulation,
    resetSimulation,
    stepForward,
    // Expose connection status for the navbar badge
    wsStatus: ws.status,
    wsLatency: ws.latencyMs,
    isLive: isLive && !ws.isMockMode,
  };
}
