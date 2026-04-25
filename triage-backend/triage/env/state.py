"""
EnvironmentState — the complete world model for the hospital simulation.

Contains all patients, resources, agent states, policies, and episode history.
Provides serialization to numpy (for RL), JSON (for API), and ASCII (for terminal).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


# ─── Enums ───────────────────────────────────────────────────


class PatientStatus(str, Enum):
    INCOMING = "INCOMING"
    CRITICAL = "CRITICAL"
    SERIOUS = "SERIOUS"
    STABLE = "STABLE"
    DISCHARGED = "DISCHARGED"
    DECEASED = "DECEASED"


class WardType(str, Enum):
    ER = "ER"
    ICU = "ICU"
    WARD_A = "WARD_A"
    WARD_B = "WARD_B"
    OR = "OR"
    PHARMACY = "PHARMACY"
    TRIAGE = "TRIAGE"


class IsolationStatus(str, Enum):
    NONE = "none"
    DROPLET = "droplet"
    AIRBORNE = "airborne"
    CONTACT = "contact"
    FULL = "full_isolation"


class AgentType(str, Enum):
    AMBULANCE_DISPATCH = "ambulance_dispatch"
    CMO_OVERSIGHT = "cmo_oversight"
    ER_TRIAGE = "er_triage"
    INFECTION_CONTROL = "infection_control"
    ICU_MANAGEMENT = "icu_management"
    PHARMACY = "pharmacy"
    HR_ROSTERING = "hr_rostering"
    IT_SYSTEMS = "it_systems"
    BLOOD_BANK = "blood_bank"
    ETHICS_COMMITTEE = "ethics_committee"


class EthicalFramework(str, Enum):
    UTILITARIAN = "utilitarian"
    CLINICAL_PRIORITY = "clinical"
    FIRST_COME_FIRST_SERVED = "fcfs"
    EQUITY = "equity"


class CrisisType(str, Enum):
    MASS_CASUALTY = "mass_casualty"
    OUTBREAK = "outbreak"
    EQUIPMENT_FAILURE = "equipment_failure"
    STAFF_SHORTAGE = "staff_shortage"


class AmbulanceStatus(str, Enum):
    AVAILABLE = "available"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RETURNING = "returning"
    OFFLINE = "offline"


class SafetyViolationType(str, Enum):
    CRITICAL_PATIENT_DISCHARGE    = "critical_patient_discharge"
    DRUG_INTERACTION              = "drug_interaction"
    ZERO_ICU_STAFF                = "zero_icu_staff"
    VENTILATOR_OVER_ALLOCATION    = "ventilator_over_allocation"
    BLOOD_TYPE_MISMATCH           = "blood_type_mismatch"
    UNAUTHORIZED_CMO_OVERRIDE     = "unauthorized_cmo_override"
    TREATMENT_WITHOUT_TRIAGE      = "treatment_without_triage"
    ICU_TRANSFER_NO_BED           = "icu_transfer_no_bed"
    MEDICATION_WITHOUT_DIAGNOSIS  = "medication_without_diagnosis"
    DUPLICATE_CRITICAL_ACTION     = "duplicate_critical_action"



class ActionType(int, Enum):
    TRIAGE_PATIENT = 0
    TRANSFER_TO_ICU = 1
    TRANSFER_TO_WARD = 2
    ASSIGN_TREATMENT = 3
    ORDER_MEDICATION = 4
    REQUEST_BLOOD = 5
    ACTIVATE_PROTOCOL = 6
    REQUEST_STAFF = 7
    ESCALATE_TO_CMO = 8
    DISCHARGE_PATIENT = 9
    FLAG_POLICY_VIOLATION = 10
    UPDATE_EHR = 11
    VERIFY_INSURANCE = 12
    ALLOCATE_EQUIPMENT = 13
    SEND_MESSAGE = 14
    OVERRIDE_DECISION = 15
    REQUEST_SPECIALIST = 16
    ACTIVATE_OVERFLOW = 17
    UPDATE_TREATMENT_PLAN = 18
    CLOSE_CASE = 19


class MessageType(str, Enum):
    ACTION = "ACTION"
    ALERT = "ALERT"
    HANDOFF = "HANDOFF"
    OVERSIGHT = "OVERSIGHT"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    BROADCAST = "BROADCAST"
    EXPERT = "EXPERT"


class MessagePriority(int, Enum):
    CRITICAL = 10
    HIGH = 7
    NORMAL = 4
    LOW = 1


# ─── Data Classes ────────────────────────────────────────────


@dataclass
class PatientEvent:
    timestamp: datetime
    event_type: str
    description: str
    agent: AgentType | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Patient:
    id: str
    name: str
    age: int
    condition: str
    status: PatientStatus = PatientStatus.INCOMING
    ward: WardType = WardType.TRIAGE
    triage_score: int = 0
    assigned_agent: AgentType = AgentType.ER_TRIAGE
    treatment_plan: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    insurance_verified: bool = False
    insurance_plan: str | None = None
    icu_required: bool = False
    admitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: list[PatientEvent] = field(default_factory=list)
    deterioration_rate: float = 0.0  # per-step probability of getting worse

    @property
    def acuity_score(self) -> int:
        return self.triage_score

    @acuity_score.setter
    def acuity_score(self, value: int) -> None:
        self.triage_score = value

    def add_event(self, event_type: str, description: str, agent: AgentType | None = None) -> None:
        self.history.append(
            PatientEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=event_type,
                description=description,
                agent=agent,
            )
        )
        self.last_updated = datetime.now(timezone.utc)

    def to_vector(self) -> np.ndarray:
        """Convert patient to 12-dim feature vector for observation space."""
        status_map = {s: i for i, s in enumerate(PatientStatus)}
        status_onehot = np.zeros(5, dtype=np.float32)
        idx = status_map.get(self.status, 0)
        if idx < 5:
            status_onehot[idx] = 1.0

        ward_map = {w: i for i, w in enumerate(WardType)}
        agent_map = {a: i for i, a in enumerate(AgentType)}

        return np.array([
            *status_onehot,
            self.triage_score / 10.0,
            ward_map.get(self.ward, 0) / len(WardType),
            min(len(self.history), 100) / 100.0,  # time proxy
            agent_map.get(self.assigned_agent, 0) / len(AgentType),
            float(self.icu_required),
            float(len(self.treatment_plan) > 0),
            float(self.insurance_verified),
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "condition": self.condition,
            "status": self.status.value,
            "ward": self.ward.value,
            "triage_score": self.triage_score,
            "assigned_agent": self.assigned_agent.value,
            "treatment_plan": self.treatment_plan,
            "medications": self.medications,
            "allergies": self.allergies,
            "insurance_verified": self.insurance_verified,
            "insurance_plan": self.insurance_plan,
            "icu_required": self.icu_required,
            "admitted_at": self.admitted_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "history": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "description": e.description,
                    "agent": e.agent.value if e.agent else None,
                }
                for e in self.history
            ],
        }


@dataclass
class Ambulance:
    unit_id: str
    status: AmbulanceStatus
    patient_id: str | None
    eta_steps: int
    acuity_estimate: int
    incident_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status.value if isinstance(self.status, AmbulanceStatus) else str(self.status),
            "patient_id": self.patient_id,
            "eta_steps": self.eta_steps,
            "acuity_estimate": self.acuity_estimate,
            "incident_type": self.incident_type,
        }


@dataclass
class IncomingPatient:
    patient_id: str
    acuity_estimate: int
    incident_type: str
    eta_steps: int
    ambulance_id: str
    pre_alert_sent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "acuity_estimate": self.acuity_estimate,
            "incident_type": self.incident_type,
            "eta_steps": self.eta_steps,
            "ambulance_id": self.ambulance_id,
            "pre_alert_sent": self.pre_alert_sent,
        }


@dataclass
class ResourceState:
    icu_beds_total: int = 20
    icu_beds_occupied: int = 0
    ventilators_total: int = 15
    ventilators_in_use: int = 0
    blood_supply_ab: float = 1.0    # 0.0-1.0 normalized
    blood_supply_oneg: float = 1.0
    staff_ratio: float = 1.0        # actual / required
    pharmacy_stock: float = 1.0
    equipment_status: float = 1.0   # fraction operational
    it_uptime: float = 1.0
    blood_inventory: dict[str, int] = field(default_factory=lambda: {
        "O+": 20, "O-": 10, "A+": 15, "A-": 8,
        "B+": 12, "B-": 6, "AB+": 5, "AB-": 3
    })

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.icu_beds_occupied / max(self.icu_beds_total, 1),
            self.ventilators_in_use / max(self.ventilators_total, 1),
            self.blood_supply_ab,
            self.blood_supply_oneg,
            self.staff_ratio,
            self.pharmacy_stock,
            self.equipment_status,
            self.it_uptime,
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "icu_beds_total": self.icu_beds_total,
            "icu_beds_occupied": self.icu_beds_occupied,
            "icu_beds_available": self.icu_beds_total - self.icu_beds_occupied,
            "ventilators_total": self.ventilators_total,
            "ventilators_in_use": self.ventilators_in_use,
            "ventilators_available": self.ventilators_total - self.ventilators_in_use,
            "blood_supply_ab": self.blood_supply_ab,
            "blood_supply_oneg": self.blood_supply_oneg,
            "staff_ratio": self.staff_ratio,
            "pharmacy_stock": self.pharmacy_stock,
            "equipment_status": self.equipment_status,
            "it_uptime": self.it_uptime,
            "blood_inventory": self.blood_inventory,
        }


@dataclass
class AgentState:
    agent_type: AgentType
    is_active: bool = True
    current_action: str = "idle"
    patients_assigned: int = 0
    actions_taken: int = 0
    violations_count: int = 0
    messages_sent: int = 0
    token_usage: int = 0
    idle_steps: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.is_active),
            min(self.patients_assigned, 20) / 20.0,
            min(self.actions_taken, 100) / 100.0,
            min(self.violations_count, 10) / 10.0,
            min(self.messages_sent, 50) / 50.0,
            min(self.token_usage, 10000) / 10000.0,
            min(self.idle_steps, 20) / 20.0,
            0.0,  # reserved
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "is_active": self.is_active,
            "current_action": self.current_action,
            "patients_assigned": self.patients_assigned,
            "actions_taken": self.actions_taken,
            "violations_count": self.violations_count,
            "messages_sent": self.messages_sent,
            "token_usage": self.token_usage,
        }


@dataclass
class Crisis:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: CrisisType = CrisisType.MASS_CASUALTY
    name: str = "Mass Casualty Event"
    severity: str = "critical"
    patient_count: int = 25
    incoming_rate: int = 3
    typical_conditions: list[str] = field(default_factory=list)
    special_rules: list[str] = field(default_factory=list)
    patient_list: list[Patient] = field(default_factory=list)
    drug_inventory: dict[str, int] = field(default_factory=dict)
    blood_inventory: dict[str, int] = field(default_factory=dict)
    staff_roster: dict[str, Any] = field(default_factory=dict)
    icu_config: dict[str, Any] = field(default_factory=dict)
    insurance_policies: dict[str, Any] = field(default_factory=dict)
    initial_incoming_patients: list[IncomingPatient] = field(default_factory=list)
    offline_ambulance_count: int = 0
    staff_reduction: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "severity": self.severity,
            "patient_count": self.patient_count,
            "incoming_rate": self.incoming_rate,
            "typical_conditions": self.typical_conditions,
            "special_rules": self.special_rules,
        }


@dataclass
class Policy:
    id: str
    name: str
    version: str
    rules: list[str]
    effective_from: int  # episode number
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "rules": self.rules,
            "effective_from": self.effective_from,
            "is_active": self.is_active,
        }


@dataclass
class PolicyChange:
    policy_id: str
    change_type: str  # "added" | "modified" | "removed"
    old_value: str | None
    new_value: str | None
    episode: int


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: AgentType = AgentType.CMO_OVERSIGHT
    to_agent: AgentType | str = "ALL"
    content: str = ""
    msg_type: MessageType = MessageType.ACTION
    priority: int = 4  # Default to MessagePriority.NORMAL.value
    patient_id: str | None = None
    action_id: str | None = None
    request_type: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    authorization_id: str | None = None
    status: str = "pending"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    requires_response: bool = False
    response_deadline: datetime | None = None
    deadline: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_agent": self.from_agent.value if isinstance(self.from_agent, AgentType) else self.from_agent,
            "to_agent": self.to_agent.value if isinstance(self.to_agent, AgentType) else str(self.to_agent),
            "content": self.content,
            "msg_type": self.msg_type.value,
            "priority": self.priority,
            "patient_id": self.patient_id,
            "action_id": self.action_id,
            "request_type": self.request_type,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "authorization_id": self.authorization_id,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "requires_response": self.requires_response,
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }


@dataclass
class AgentAction:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_type: AgentType = AgentType.ER_TRIAGE
    action_type: ActionType = ActionType.TRIAGE_PATIENT
    target_id: int = 0
    priority: int = 0
    reasoning: str = ""
    reasoning_tokens: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_env_action(self) -> dict[str, int]:
        return {
            "agent_id": list(AgentType).index(self.agent_type),
            "action_type": self.action_type.value,
            "target_id": self.target_id,
            "priority": self.priority,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_type": self.agent_type.value,
            "action_type": self.action_type.name,
            "target_id": self.target_id,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "reasoning_tokens": self.reasoning_tokens,
            "timestamp": self.timestamp.isoformat(),
        }

@dataclass
class RationingDecision:
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: str = ""
    candidates: list[str] = field(default_factory=list)
    selected_patient_id: str = ""
    rejected_patient_ids: list[str] = field(default_factory=list)
    framework_used: EthicalFramework = EthicalFramework.UTILITARIAN
    justification: str = ""
    step: int = 0
    overridden_by_cmo: bool = False
    override_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "resource_type": self.resource_type,
            "candidates": self.candidates,
            "selected_patient_id": self.selected_patient_id,
            "rejected_patient_ids": self.rejected_patient_ids,
            "framework_used": self.framework_used.value if isinstance(self.framework_used, EthicalFramework) else self.framework_used,
            "justification": self.justification,
            "step": self.step,
            "overridden_by_cmo": self.overridden_by_cmo,
            "override_reason": self.override_reason,
        }

@dataclass
class SafetyBlock:
    block_id: str                      # uuid4
    step: int
    agent_type: str                    # which agent produced the blocked action
    violation_type: SafetyViolationType
    blocked_action: AgentAction        # the original unsafe action (preserved)
    fallback_action: AgentAction       # what replaced it
    reason: str                        # human-readable explanation
    patient_id: str | None             # affected patient if applicable
    severity: int                      # 1-10, how dangerous was this

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "step": self.step,
            "agent_type": self.agent_type,
            "violation_type": self.violation_type.value if isinstance(self.violation_type, SafetyViolationType) else self.violation_type,
            "blocked_action": self.blocked_action.to_dict() if hasattr(self.blocked_action, "to_dict") else self.blocked_action,
            "fallback_action": self.fallback_action.to_dict() if hasattr(self.fallback_action, "to_dict") else self.fallback_action,
            "reason": self.reason,
            "patient_id": self.patient_id,
            "severity": self.severity,
        }


@dataclass
class InfectionEvent:
    event_id: str
    step: int
    source_patient_id: str
    infected_patient_id: str
    ward: str
    pathogen: str
    prevented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "step": self.step,
            "source_patient_id": self.source_patient_id,
            "infected_patient_id": self.infected_patient_id,
            "ward": self.ward,
            "pathogen": self.pathogen,
            "prevented": self.prevented,
        }

# ─── Main State ──────────────────────────────────────────────



@dataclass
class AuthorizationToken:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    approver: AgentType = AgentType.CMO_OVERSIGHT
    scope: str = "emergency_override"
    patient_id: str | None = None
    reason: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_after_step: int | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "approver": self.approver.value,
            "scope": self.scope,
            "patient_id": self.patient_id,
            "reason": self.reason,
            "issued_at": self.issued_at.isoformat(),
            "expires_after_step": self.expires_after_step,
            "active": self.active,
        }


@dataclass
class AppAuditEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    app: str = ""
    tool_name: str = ""
    requester: AgentType = AgentType.CMO_OVERSIGHT
    patient_id: str | None = None
    status: str = "approved"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    authorization_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "app": self.app,
            "tool_name": self.tool_name,
            "requester": self.requester.value,
            "patient_id": self.patient_id,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "authorization_id": self.authorization_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EnvironmentState:
    """Complete world state for the hospital simulation."""

    crisis: Crisis
    episode: int
    patients: list[Patient] = field(default_factory=list)
    resources: ResourceState = field(default_factory=ResourceState)
    agent_states: dict[AgentType, AgentState] = field(default_factory=dict)
    active_policies: dict[str, Policy] = field(default_factory=dict)
    expert_signals: dict[str, float] = field(default_factory=dict)
    contract_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    regulatory_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    drift_history: list[dict[str, Any]] = field(default_factory=list)
    message_history: list[AgentMessage] = field(default_factory=list)
    action_history: list[AgentAction] = field(default_factory=list)
    app_audit_log: list[AppAuditEvent] = field(default_factory=list)
    override_tokens: dict[str, AuthorizationToken] = field(default_factory=dict)
    pending_patients: list[Patient] = field(default_factory=list)  # incoming queue
    rationing_decisions: list[RationingDecision] = field(default_factory=list)
    infection_events: list[InfectionEvent] = field(default_factory=list)
    ward_lockdowns: dict[str, bool] = field(default_factory=dict)
    active_pathogens: list[str] = field(default_factory=list)
    ambulances: list[Ambulance] = field(
        default_factory=lambda: [
            Ambulance(f"AMB-{i:02d}", AmbulanceStatus.AVAILABLE, None, 0, 0, "")
            for i in range(1, 6)
        ]
    )
    incoming_patients: list[IncomingPatient] = field(default_factory=list)
    diverted_count: int = 0
    dispatch_events: list[dict[str, Any]] = field(default_factory=list)
    violations_injected: int = 0
    violations_caught: int = 0
    step_count: int = 0
    safety_blocks: list[SafetyBlock] = field(default_factory=list)
    constitution_active: bool = True

    def __post_init__(self) -> None:
        if not self.agent_states:
            for at in AgentType:
                self.agent_states[at] = AgentState(agent_type=at)
        if not self.expert_signals:
            self.expert_signals = {
                "cost_weight": 0.3,
                "quality_weight": 0.5,
                "speed_weight": 0.2,
            }
        if not self.contract_constraints:
            self.contract_constraints = {
                "insurance_portal": {
                    "schema_version": "v1",
                    "member_id_field": "member_id",
                    "coverage_field": "coverage_percent",
                    "authorization_mode": "waived_for_emergency",
                    "requires_portal_reference": False,
                },
            }
        if not self.regulatory_constraints:
            self.regulatory_constraints = {
                "hipaa": {
                    "max_access_window_minutes": 15,
                    "require_break_glass_justification": False,
                },
                "medication_safety": {
                    "dual_signoff_required": True,
                    "verbal_order_timeout_minutes": 60,
                },
            }
        if not self.ward_lockdowns:
            self.ward_lockdowns = {"ER": False, "ICU": False, "WARD": False, "ISOLATION": False}

    # ── Computed Properties ──────────────────────────────

    @property
    def alive_count(self) -> int:
        return sum(
            1 for p in self.patients
            if p.status not in (PatientStatus.DECEASED, PatientStatus.DISCHARGED)
        )

    @property
    def deceased_count(self) -> int:
        return sum(1 for p in self.patients if p.status == PatientStatus.DECEASED)

    @property
    def discharged_count(self) -> int:
        return sum(1 for p in self.patients if p.status == PatientStatus.DISCHARGED)

    @property
    def critical_count(self) -> int:
        return sum(1 for p in self.patients if p.status == PatientStatus.CRITICAL)

    @property
    def total_patients(self) -> int:
        return len(self.patients)

    @property
    def icu_occupancy(self) -> float:
        if self.resources.icu_beds_total == 0:
            return 1.0
        return self.resources.icu_beds_occupied / self.resources.icu_beds_total

    @property
    def survival_rate(self) -> float:
        total = self.total_patients
        if total == 0:
            return 1.0
        return 1.0 - (self.deceased_count / total)

    @property
    def all_patients_processed(self) -> bool:
        return all(
            p.status in (PatientStatus.DISCHARGED, PatientStatus.DECEASED, PatientStatus.STABLE)
            for p in self.patients
        ) and len(self.pending_patients) == 0

    @property
    def crisis_resolved(self) -> bool:
        if not self.patients:
            return False
        critical = self.critical_count
        incoming = len(self.pending_patients)
        return critical == 0 and incoming == 0 and self.all_patients_processed

    @property
    def catastrophic_failure(self) -> bool:
        if self.total_patients == 0:
            return False
        return (self.deceased_count / self.total_patients) > 0.5

    # ── Mutation ─────────────────────────────────────────

    def update(self, action_result: dict[str, Any]) -> None:
        """Apply action result to world state. Called by HospitalEnv.step()."""
        self.step_count += 1
        self._expire_override_tokens()

        # Admit pending patients based on incoming rate
        rate = self.crisis.incoming_rate
        admitted = 0
        while self.pending_patients and admitted < rate:
            patient = self.pending_patients.pop(0)
            patient.status = PatientStatus.INCOMING
            patient.add_event("ADMITTED", f"Patient admitted to ER — step {self.step_count}")
            self.patients.append(patient)
            admitted += 1

        # Natural deterioration of untreated critical patients
        for p in self.patients:
            if p.status == PatientStatus.CRITICAL and not p.treatment_plan:
                if np.random.random() < p.deterioration_rate:
                    p.status = PatientStatus.DECEASED
                    p.add_event("DECEASED", "Patient deteriorated without treatment")
            elif p.status == PatientStatus.SERIOUS:
                if np.random.random() < (p.deterioration_rate * 0.3):
                    p.status = PatientStatus.CRITICAL
                    p.icu_required = True
                    p.add_event("DETERIORATED", "Condition worsened to CRITICAL")

        # Apply action result side effects
        for effect in action_result.get("side_effects", []):
            self._apply_side_effect(effect)

    def _apply_side_effect(self, effect: dict[str, Any]) -> None:
        effect_type = effect.get("type", "")
        if effect_type == "resource_change":
            attr = effect.get("resource")
            delta = effect.get("delta", 0)
            if hasattr(self.resources, attr):
                current = getattr(self.resources, attr)
                setattr(self.resources, attr, max(0, current + delta))

    # ── Serialization ────────────────────────────────────

    def to_observation(self) -> dict[str, np.ndarray]:
        """Convert to observation space format (numpy arrays)."""
        # Patients: 50 slots × 12 features
        patient_matrix = np.zeros((50, 12), dtype=np.float32)
        for i, p in enumerate(self.patients[:50]):
            patient_matrix[i] = p.to_vector()

        # Resources: 8 features
        resources_vec = self.resources.to_vector()

        # Agent states: N × 8
        agent_matrix = np.zeros((len(AgentType), 8), dtype=np.float32)
        for i, at in enumerate(AgentType):
            if at in self.agent_states:
                agent_matrix[i] = self.agent_states[at].to_vector()

        # Crisis state: 10 features
        crisis_vec = np.zeros(10, dtype=np.float32)
        crisis_type_idx = list(CrisisType).index(self.crisis.type)
        if crisis_type_idx < 5:
            crisis_vec[crisis_type_idx] = 1.0
        severity_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        crisis_vec[5] = severity_map.get(self.crisis.severity, 0.5)
        crisis_vec[6] = min(self.step_count, 500) / 500.0
        crisis_vec[7] = len(self.pending_patients) / max(self.crisis.patient_count, 1)
        crisis_vec[8] = float(self.violations_injected > 0)
        crisis_vec[9] = float(bool(self.drift_history))

        # Policy state: 20 features
        policy_vec = np.zeros(20, dtype=np.float32)
        for i, (_, policy) in enumerate(list(self.active_policies.items())[:20]):
            policy_vec[i] = float(policy.is_active)

        # Expert signals: 6 features
        expert_vec = np.zeros(6, dtype=np.float32)
        signal_keys = ["cost_weight", "quality_weight", "speed_weight"]
        for i, key in enumerate(signal_keys):
            expert_vec[i] = self.expert_signals.get(key, 0.0)

        return {
            "patients": patient_matrix,
            "resources": resources_vec,
            "agent_states": agent_matrix,
            "crisis_state": crisis_vec,
            "policy_state": policy_vec,
            "expert_signals": expert_vec,
        }

    def to_json(self) -> dict[str, Any]:
        """Full JSON serialization for API responses."""
        return {
            "episode": self.episode,
            "step": self.step_count,
            "crisis": self.crisis.to_dict(),
            "patients": [p.to_dict() for p in self.patients],
            "pending_patients": len(self.pending_patients),
            "resources": self.resources.to_dict(),
            "agent_states": {k.value: v.to_dict() for k, v in self.agent_states.items()},
            "active_policies": {k: v.to_dict() for k, v in self.active_policies.items()},
            "expert_signals": self.expert_signals,
            "contract_constraints": self.contract_constraints,
            "regulatory_constraints": self.regulatory_constraints,
            "recent_drift_events": self.drift_history[-10:],
            "app_audit_log": [event.to_dict() for event in self.app_audit_log[-25:]],
            "override_tokens": [token.to_dict() for token in self.override_tokens.values() if token.active],
            "infection_events": [event.to_dict() for event in self.infection_events[-25:]],
            "ward_lockdowns": self.ward_lockdowns,
            "active_pathogens": self.active_pathogens,
            "ambulances": [ambulance.to_dict() for ambulance in self.ambulances],
            "incoming_patients": [patient.to_dict() for patient in self.incoming_patients],
            "diverted_count": self.diverted_count,
            "dispatch_events": self.dispatch_events[-25:],
            "stats": {
                "alive_count": self.alive_count,
                "deceased_count": self.deceased_count,
                "discharged_count": self.discharged_count,
                "critical_count": self.critical_count,
                "total_patients": self.total_patients,
                "icu_occupancy": self.icu_occupancy,
                "survival_rate": self.survival_rate,
                "violations_injected": self.violations_injected,
                "violations_caught": self.violations_caught,
            },
        }

    def add_app_audit(self, event: AppAuditEvent) -> None:
        self.app_audit_log.append(event)

    def add_drift_event(self, event: dict[str, Any]) -> None:
        self.drift_history.append(event)

    def issue_override_token(
        self,
        scope: str,
        reason: str,
        patient_id: str | None = None,
        approver: AgentType = AgentType.CMO_OVERSIGHT,
        expires_in_steps: int = 2,
    ) -> AuthorizationToken:
        expires_after_step = self.step_count + expires_in_steps if expires_in_steps is not None else None
        token = AuthorizationToken(
            approver=approver,
            scope=scope,
            patient_id=patient_id,
            reason=reason,
            expires_after_step=expires_after_step,
        )
        self.override_tokens[token.id] = token
        return token

    def validate_override_token(
        self,
        token_id: str | None,
        scope: str,
        patient_id: str | None = None,
        consume: bool = False,
    ) -> bool:
        if not token_id:
            return False
        token = self.override_tokens.get(token_id)
        if token is None or not token.active:
            return False
        if token.scope != scope:
            return False
        if token.patient_id is not None and patient_id is not None and token.patient_id != patient_id:
            return False
        if token.expires_after_step is not None and token.expires_after_step < self.step_count:
            token.active = False
            return False
        if consume:
            token.active = False
        return True

    def find_active_override_token(
        self,
        scope: str,
        patient_id: str | None = None,
    ) -> str | None:
        for token in self.override_tokens.values():
            if not token.active or token.scope != scope:
                continue
            if patient_id is not None and token.patient_id not in (None, patient_id):
                continue
            if token.expires_after_step is not None and token.expires_after_step < self.step_count:
                token.active = False
                continue
            return token.id
        return None

    def _expire_override_tokens(self) -> None:
        for token in self.override_tokens.values():
            if token.active and token.expires_after_step is not None and token.expires_after_step < self.step_count:
                token.active = False

    def render_ascii(self) -> str:
        """Terminal-friendly state render."""
        lines = [
            f"╔══════════════════════════════════════════════════════╗",
            f"║  TRIAGE — Episode {self.episode:>3d}  |  Step {self.step_count:>4d}              ║",
            f"║  Crisis: {self.crisis.name:<30s}         ║",
            f"╠══════════════════════════════════════════════════════╣",
            f"║  Patients: {self.total_patients:>3d}  |  "
            f"Critical: {self.critical_count:>3d}  |  "
            f"Dead: {self.deceased_count:>3d}    ║",
            f"║  ICU: {self.resources.icu_beds_occupied}/{self.resources.icu_beds_total}  |  "
            f"Vents: {self.resources.ventilators_in_use}/{self.resources.ventilators_total}  |  "
            f"Survival: {self.survival_rate:.1%}  ║",
            f"╠══════════════════════════════════════════════════════╣",
        ]
        for at, state in self.agent_states.items():
            status = "●" if state.is_active else "○"
            lines.append(
                f"║  {status} {at.value:<18s}  actions={state.actions_taken:<4d}  "
                f"violations={state.violations_count:<2d}  ║"
            )
        lines.append(f"╚══════════════════════════════════════════════════════╝")
        return "\n".join(lines)
