export type PatientStatus = "CRITICAL" | "SERIOUS" | "STABLE" | "DISCHARGED" | "DECEASED";
export type AgentStatus = "ACTIVE" | "PROCESSING" | "WAITING" | "BLOCKED";
export type CrisisType = "MASS_CASUALTY" | "OUTBREAK" | "EQUIPMENT_FAILURE" | "STAFF_SHORTAGE";
export type MessageType = "ACTION" | "ALERT" | "HANDOFF" | "OVERSIGHT" | "REQUEST";
export type AgentKey =
  | "CMO_OVERSIGHT"
  | "ER_TRIAGE"
  | "ICU_MANAGEMENT"
  | "PHARMACY"
  | "HR_ROSTERING"
  | "IT_SYSTEMS";

export interface Patient {
  id: string;
  name: string;
  age: number;
  condition: string;
  status: PatientStatus;
  ward: string;
  triageScore: number;
  assignedAgent: string;
  admittedAt: Date;
  lastUpdated: Date;
}

export interface AgentMessage {
  id: string;
  from: string;
  to: string;
  content: string;
  type: MessageType;
  timestamp: Date;
  patientId?: string;
}

export interface AgentState {
  key: AgentKey;
  name: string;
  status: AgentStatus;
  currentAction: string;
  messagesSent: number;
}

export interface EpisodeMetrics {
  episode: number;
  rewardScore: number;
  baselineScore: number;
  survivalRate: number;
  complianceScore: number;
  stepsToResolution: number;
}

export interface Resources {
  icuBeds: { used: number; total: number };
  ventilators: { used: number; total: number };
  bloodSupply: number;
  staffOnDuty: { used: number; total: number };
}

export interface DecisionLogEntry {
  id: string;
  agent: string;
  action: string;
  patientId: string;
  outcome: "OPTIMAL" | "SUBOPTIMAL" | "ERROR";
  timestamp: Date;
}

export interface MemoryLesson {
  id: string;
  episode: number;
  agentType: AgentKey;
  pattern: string;
  correction: string;
  confidence: number;
  rewardDelta: number;
  timesApplied: number;
  successCount: number;
}

export interface KeyMoment {
  step: number;
  type:
    | "violation"
    | "oversight_catch"
    | "drug_shortage"
    | "icu_full"
    | "crisis_resolved"
    | "patient_death"
    | "drift_event";
  description: string;
  agentInvolved: AgentKey;
  rewardDelta: number;
}

export interface ReplayEpisode {
  id: string;
  episode: number;
  totalSteps: number;
  finalReward: number;
  survivalRate: number;
  keyMoments: KeyMoment[];
  label: string;
}

export interface SponsorEntry {
  sponsor: string;
  theme: string;
  requirement: string;
  coverage: string;
  status: "direct" | "partial" | "missed";
  dashboardHighlight: string;
  rewardComponent: string;
  bonusPrize: string;
  fixNote?: string;
}

export type WSStatus = "connecting" | "connected" | "disconnected" | "fallback";

export interface TrainingStatus {
  isTraining: boolean;
  phase: "idle" | "collecting" | "labeling" | "training" | "complete" | "error";
  epoch: number;
  totalEpochs: number;
  step: number;
  totalSteps: number;
  trainLoss: number[];
  evalLoss: number[];
  lossSteps: number[];
  etaMinutes: number;
  gpuUtilization: number;
  baselineRewards: number[];
  trainedRewards: number[];
  rewardMargin: number;
  dpoAccuracy: number;
  comparisonStep: number;
  nPreferencePairs: number;
  modelPushedToHub: boolean;
  hubUrl: string | null;
}
