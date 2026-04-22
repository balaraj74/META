/**
 * TRIAGE API Client
 *
 * All REST and WebSocket communication with the FastAPI backend runs through
 * this module.  The frontend proxies /api → http://localhost:8000 and
 * /ws     → ws://localhost:8000  via the Vite dev-server proxy (vite.config.ts).
 *
 * In production, set VITE_API_BASE_URL to your deployed backend URL.
 */

// ── Base URLs ──────────────────────────────────────────────────────────────────

const API_BASE =
  (typeof import.meta !== "undefined" && (import.meta as unknown as Record<string, unknown>).env &&
    ((import.meta as unknown as Record<string, unknown>).env as Record<string, string>).VITE_API_BASE_URL) ||
  "";  // empty → relative → vite proxy

const WS_BASE =
  (typeof import.meta !== "undefined" && (import.meta as unknown as Record<string, unknown>).env &&
    ((import.meta as unknown as Record<string, unknown>).env as Record<string, string>).VITE_WS_URL) ||
  (typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`
    : "ws://localhost:8000");

export const ENDPOINTS = {
  // Health
  health: `${API_BASE}/api/health`,

  // Simulation
  simulationStart: `${API_BASE}/api/simulation/start`,
  simulationStep: `${API_BASE}/api/simulation/step`,
  simulationStop: `${API_BASE}/api/simulation/stop`,
  simulationState: `${API_BASE}/api/simulation/state`,
  simulationHistory: `${API_BASE}/api/simulation/history`,

  // Training
  trainingStart: `${API_BASE}/api/training/start`,
  trainingStatus: `${API_BASE}/api/training/status`,

  // Metrics
  rewardCurve: `${API_BASE}/api/metrics/reward-curve`,
  rewardBreakdown: `${API_BASE}/api/metrics/reward-breakdown`,
  comparison: `${API_BASE}/api/metrics/comparison`,

  // WebSocket
  ws: `${WS_BASE}/ws/simulation`,
} as const;

// ── Types mirrored from backend schemas.py ─────────────────────────────────────

export interface BackendPatient {
  id: string;
  name: string;
  age: number;
  condition: string;
  severity: string;          // from env (CRITICAL | SERIOUS | STABLE …)
  triage_tag: string;        // RED | YELLOW | GREEN | BLACK
  vital_signs: Record<string, number>;
  treatments: string[];
  location: string;
  is_alive: boolean;
  is_discharged: boolean;
}

export interface BackendResources {
  icu_beds_total: number;
  icu_beds_used: number;
  er_beds_total: number;
  er_beds_used: number;
  staff_available: number;
  staff_total: number;
  ventilators_available: number;
  ventilators_total: number;
  drug_inventory: Record<string, number>;
}

export interface BackendAgent {
  agent_type: string;
  role: string;
  actions_taken: number;
  total_tokens: number;
  last_action: string | null;
  is_active: boolean;
}

export interface BackendMetrics {
  survival_rate: number;
  deceased_count: number;
  discharged_count: number;
  critical_count: number;
  alive_count: number;
  icu_occupancy: number;
  total_reward: number;
  violations_caught: number;
  violations_injected: number;
  compliance_rate: number;
}

export interface BackendStateUpdate {
  status: string;
  step: number;
  max_steps: number;
  crisis_type: string;
  difficulty: number;
  patients: BackendPatient[];
  resources: BackendResources;
  agents: BackendAgent[];
  metrics: BackendMetrics;
  recent_actions: BackendStepData[];
}

export interface BackendStepData {
  step: number;
  action: Record<string, unknown>;
  reward: number;
  breakdown: Record<string, number>;
  terminated: boolean;
  drift_events: unknown[];
}

export interface BackendTrainingStatus {
  phase: "not_started" | "collecting" | "training" | "completed" | "error";
  progress: number;
  current_episode?: number;
  total_episodes?: number;
  metrics?: Record<string, unknown>;
  error?: string;
}

export interface BackendComparisonMetrics {
  // Per-episode arrays for the bar chart
  baseline_rewards: number[];
  trained_rewards: number[];
  // Aggregate scalars
  baseline_mean_reward: number;
  trained_mean_reward: number;
  improvement: number;
  dpo_accuracy: number;
  reward_margin: number;
  step: number;
  progress: number;
  // Legacy fields kept for backward compat
  reward_delta?: number;
  baseline_mean_survival?: number;
  trained_mean_survival?: number;
  survival_delta?: number;
  episode_counts?: Record<string, number>;
}

export interface BackendRewardCurve {
  curve: { step: number; reward: number; cumulative: number }[];
}

export interface SimulationStartConfig {
  crisis_type?: "mass_casualty" | "outbreak" | "equipment_failure" | "staff_shortage";
  difficulty?: number;
  max_steps?: number;
  mock_llm?: boolean;
  seed?: number;
  auto_step?: boolean;
  step_delay_ms?: number;
}

export interface TrainingStartConfig {
  n_episodes?: number;
  difficulty?: number;
  mock_llm?: boolean;
  mock_training?: boolean;
  model_name?: string;
  learning_rate?: number;
  num_epochs?: number;
}

// ── HTTP helpers ───────────────────────────────────────────────────────────────

async function request<T>(
  url: string,
  options: RequestInit = {},
  timeoutMs = 8000,
): Promise<T> {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(tid);
  }
}

function post<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function get<T>(url: string): Promise<T> {
  return request<T>(url, { method: "GET" });
}

// ── API functions ──────────────────────────────────────────────────────────────

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

/**
 * Ping the backend health endpoint.
 * Resolves `true` if backend is reachable, `false` otherwise.
 */
export async function pingBackend(): Promise<boolean> {
  try {
    const r = await request<{ status: string }>(ENDPOINTS.health, {}, 3000);
    return r.status === "healthy";
  } catch {
    return false;
  }
}

export async function startSimulation(
  config: SimulationStartConfig = {},
): Promise<ApiResponse<{ status: string; crisis_type: string }>> {
  return post(ENDPOINTS.simulationStart, {
    crisis_type: "mass_casualty",
    difficulty: 0.5,
    max_steps: 200,
    mock_llm: true,
    auto_step: true,
    step_delay_ms: 400,
    ...config,
  });
}

export async function stepSimulation(): Promise<ApiResponse<BackendStepData>> {
  return post(ENDPOINTS.simulationStep, {});
}

export async function stopSimulation(): Promise<ApiResponse<unknown>> {
  return post(ENDPOINTS.simulationStop, {});
}

export async function getSimulationState(): Promise<ApiResponse<BackendStateUpdate>> {
  return get(ENDPOINTS.simulationState);
}

export async function startTraining(
  config: TrainingStartConfig = {},
): Promise<ApiResponse<{ status: string }>> {
  return post(ENDPOINTS.trainingStart, {
    n_episodes: 10,
    mock_llm: true,
    mock_training: true,
    ...config,
  });
}

export async function getTrainingStatus(): Promise<ApiResponse<BackendTrainingStatus>> {
  return get(ENDPOINTS.trainingStatus);
}

export async function getRewardCurve(): Promise<ApiResponse<BackendRewardCurve>> {
  return get(ENDPOINTS.rewardCurve);
}

export async function getRewardBreakdown(): Promise<
  ApiResponse<{ breakdown: Record<string, number> }>
> {
  return get(ENDPOINTS.rewardBreakdown);
}

export async function getComparisonMetrics(): Promise<ApiResponse<BackendComparisonMetrics>> {
  return get(ENDPOINTS.comparison);
}
