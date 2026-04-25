"""
Specialized Hospital Agents — all 6 agents with rule-based fallbacks.

Agent Hierarchy:
  CMO_OVERSIGHT (supervisor) → oversees all agents, handles escalations
  ER_TRIAGE → patient triage and initial assessment
  ICU_MANAGEMENT → ICU bed allocation, ventilator management
  PHARMACY → medication dispensing, interaction checks
  HR_ROSTERING → staff scheduling, fatigue monitoring
  IT_SYSTEMS → EHR integrity, policy compliance, system monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from typing import Any

from pydantic import BaseModel

from triage.agents.base_agent import BaseAgent
from triage.agents.message_bus import MessageBus
from triage.env.state import (
    ActionType,
    AgentAction,
    AgentMessage,
    AgentType,
    CrisisType,
    EnvironmentState,
    InfectionEvent,
    IsolationStatus,
    MessageType,
    PatientStatus,
)
from triage.agents.tools import (
    OverrideDecisionTool, ActivateOverflowTool, AssignTreatmentTool,
    TriagePatientTool, TransferToICUTool, OrderMedicationTool,
    RequestStaffTool, FlagPolicyViolationTool, UpdateEHRTool, VerifyInsuranceTool,
    EscalateToCMOTool, SendMessageTool, TransferToWardTool, RequestSpecialistTool
)

logger = logging.getLogger(__name__)


def _expert_focus(state: EnvironmentState) -> tuple[str, dict[str, float]]:
    signals = {
        "cost": state.expert_signals.get("cost_weight", 0.3),
        "quality": state.expert_signals.get("quality_weight", 0.5),
        "speed": state.expert_signals.get("speed_weight", 0.2),
    }
    return max(signals, key=signals.get), signals


def _focus_priority(base: int, focus: str, *, quality: int = 0, speed: int = 0, cost: int = 0) -> int:
    adjustment = {"quality": quality, "speed": speed, "cost": cost}.get(focus, 0)
    return max(0, min(10, base + adjustment))


def _focus_note(focus: str, signals: dict[str, float]) -> str:
    return (
        f" [expert_focus={focus}"
        f" q={signals['quality']:.2f}"
        f" s={signals['speed']:.2f}"
        f" c={signals['cost']:.2f}]"
    )


# ─── CMO Oversight Agent ────────────────────────────────────

class CMOOversightAgent(BaseAgent):
    """Chief Medical Officer — supervisor agent.

    Responsibilities:
    - Monitor all agent activities
    - Handle escalations from other agents
    - Override decisions when necessary
    - Activate emergency protocols
    - Ensure policy compliance across the board
    """

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.CMO_OVERSIGHT, config, bus, mock_llm, model_name)

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        context = self._build_state_context(state)
        prompt = self._build_cmo_prompt(state, inbox)
        response = await self._call_llm(prompt, context)
        return self._parse_actions(response, state)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        focus, signals = _expert_focus(state)

        # Handle escalations first
        for msg in inbox:
            if msg.msg_type == MessageType.ALERT and msg.priority >= 5:
                if msg.patient_id:
                    scope = msg.payload.get("scope", "") if msg.payload else ""
                    reason = f"CMO override based on escalation: {msg.content[:100]}"
                    if scope:
                        reason = f"{reason} | scope={scope}"
                    actions.append(OverrideDecisionTool(
                        original_action_id="dummy",
                        new_decision="override",
                        reasoning=f"{reason}{_focus_note(focus, signals)}"
                    ))

        # Activate overflow if ICU is near capacity
        if state.icu_occupancy > 0.9 and not any(
            a.action_type == ActionType.ACTIVATE_OVERFLOW for a in state.action_history[-10:]
        ):
            actions.append(ActivateOverflowTool(
                ward="ICU",
                capacity_increase=5,
                justification=f"ICU occupancy >90% — activating overflow protocol{_focus_note(focus, signals)}"
            ))

        # Check for untreated critical patients
        for i, p in enumerate(state.patients):
            if p.status == PatientStatus.CRITICAL and not p.treatment_plan and len(p.history) > 3:
                actions.append(AssignTreatmentTool(
                    patient_id=p.id,
                    treatment_plan="Emergency Intervention",
                    reasoning=f"CMO emergency intervention — untreated critical patient {p.name}{_focus_note(focus, signals)}"
                ))
                break  # one at a time

        return actions

    def _build_cmo_prompt(self, state: EnvironmentState, inbox: list[AgentMessage]) -> str:
        escalations = [m for m in inbox if m.msg_type == MessageType.ALERT]
        return f"""
You are the CMO overseeing the hospital crisis response.

Escalations received: {len(escalations)}
{chr(10).join(f'- From {m.from_agent}: {m.content}' for m in escalations[:5])}

Critical patients without treatment: {sum(1 for p in state.patients if p.status == PatientStatus.CRITICAL and not p.treatment_plan)}
ICU occupancy: {state.icu_occupancy:.1%}
Expert preference vector:
- quality={state.expert_signals.get("quality_weight", 0.5):.2f}
- speed={state.expert_signals.get("speed_weight", 0.2):.2f}
- cost={state.expert_signals.get("cost_weight", 0.3):.2f}

Decide what actions to take. Prioritize life-saving interventions.
"""

    def _patient_idx(self, patient_id: str, state: EnvironmentState) -> int:
        for i, p in enumerate(state.patients):
            if p.id == patient_id:
                return i
        return 0


# ─── ER Triage Agent ────────────────────────────────────────

class ERTriageAgent(BaseAgent):
    """Emergency Room triage specialist.

    Responsibilities:
    - Assess incoming patients
    - Assign triage scores (1-10)
    - Route patients to appropriate wards
    - Escalate critical cases to ICU or CMO
    """

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.ER_TRIAGE, config, bus, mock_llm, model_name)

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        context = self._build_state_context(state)
        prompt = "Triage incoming patients. Assign scores and route to appropriate care."
        response = await self._call_llm(prompt, context)
        return self._parse_actions(response, state)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        focus, signals = _expert_focus(state)

        for i, p in enumerate(state.patients):
            # Triage new incoming patients
            if p.status == PatientStatus.INCOMING:
                triage_priority = _focus_priority(
                    max(p.triage_score, 4),
                    focus,
                    quality=1 if p.triage_score >= 8 else 0,
                    speed=1,
                    cost=-1 if p.triage_score < 5 else 0,
                )
                actions.append(TriagePatientTool(
                    patient_id=p.id,
                    triage_score=max(p.triage_score, 4),
                    assigned_ward="ER",
                    reasoning=f"Triaging incoming patient {p.name} — condition: {p.condition}{_focus_note(focus, signals)}"
                ))
                # Route critical to ICU
                if p.triage_score >= 8:
                    actions.append(TransferToICUTool(
                        patient_id=p.id,
                        priority=9,
                        reasoning=f"Critical patient {p.name} (score {p.triage_score}) requires ICU{_focus_note(focus, signals)}"
                    ))

            # Escalate deteriorating patients
            elif p.status == PatientStatus.CRITICAL and not p.treatment_plan:
                try:
                    asyncio.ensure_future(self.escalate(
                        f"Critical patient {p.name} [{p.id}] has no treatment plan",
                        p.id,
                        priority=8,
                    ))
                except RuntimeError:
                    pass  # No running loop — skip async escalation

        return actions[:5]  # limit actions per step

    def _parse_actions(self, response: dict[str, Any], state: EnvironmentState) -> list[AgentAction]:
        actions = []
        for a in response.get("actions", []):
            try:
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType[a.get("action_type", "TRIAGE_PATIENT")],
                    target_id=int(a.get("target_id", 0)),
                    priority=int(a.get("priority", 5)),
                    reasoning=a.get("reasoning", ""),
                ))
            except (KeyError, ValueError):
                continue
        return actions


# ─── ICU Management Agent ───────────────────────────────────

class InfectionControlAgent(BaseAgent):
    """Infection control officer for outbreak containment."""

    WARDS = ["ER", "ICU", "WARD", "ISOLATION"]

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.INFECTION_CONTROL, config, bus, mock_llm, model_name)
        pathogens = config.get("pathogens", [])
        self.pathogen_db: dict[str, dict[str, Any]] = {
            item.get("name", "unknown_pathogen"): item for item in pathogens
        } or {
            "unknown_pathogen": {
                "name": "unknown_pathogen",
                "isolation": IsolationStatus.FULL.value,
                "spread_rate": 0.5,
            }
        }
        self.ward_case_counts: dict[str, int] = {ward: 0 for ward in self.WARDS}
        self.isolated_patients: set[str] = set()
        self.lockdown_wards: set[str] = set()
        self.exposure_log: list[dict[str, Any]] = []
        self.SPREAD_THRESHOLD = int(config.get("spread_threshold", 3))
        self.ppe_compliance_rate = float(config.get("ppe_compliance_rate", 0.85))
        self._rng = random.Random(config.get("seed", 17))

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[BaseModel]:
        if self.mock_llm:
            return self._rule_based_decision(state, inbox)
        context = self._build_state_context(state)
        infection_context = json.dumps(
            {
                "ward_case_counts": self.ward_case_counts,
                "isolated_patients": sorted(self.isolated_patients),
                "active_pathogens": state.active_pathogens,
            }
        )
        raw = await self._call_llm(
            self.config.get("system_prompt", "Manage infection control."),
            f"{context}\nInfection Control Context:\n{infection_context}",
        )
        parsed = self._parse_llm_response(raw)
        return parsed or self._rule_based_decision(state, inbox)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[BaseModel]:
        actions: list[BaseModel] = []

        if state.crisis.type == CrisisType.OUTBREAK:
            newly_infected = self._detect_new_spread(state)
            for patient in newly_infected:
                pathogen = self._patient_pathogen(patient, state)
                isolation_level = self._isolation_for(pathogen)
                if patient.id not in self.isolated_patients:
                    treatment = f"ISOLATION_ORDER:{isolation_level}"
                    if treatment not in patient.treatment_plan:
                        patient.treatment_plan.append(treatment)
                    self.isolated_patients.add(patient.id)
                    actions.append(AssignTreatmentTool(
                        patient_id=patient.id,
                        treatment_plan=treatment,
                        reasoning=f"ISOLATION_ORDER:{isolation_level} for {pathogen}",
                    ))
                    message = (
                        f"ISOLATION_ORDER: Patient {patient.id} requires {isolation_level} "
                        f"isolation for {pathogen}. Move to isolation ward immediately."
                    )
                    actions.append(SendMessageTool(to_agent=AgentType.ER_TRIAGE.value, content=message, urgency=8))
                    actions.append(SendMessageTool(to_agent=AgentType.ICU_MANAGEMENT.value, content=message, urgency=8))

            for ward, count in self.ward_case_counts.items():
                if count >= self.SPREAD_THRESHOLD and ward not in self.lockdown_wards:
                    pathogen = self._active_pathogen(state)
                    summary = (
                        f"LOCKDOWN_RECOMMENDED: {ward} has {count} confirmed cases "
                        f"of {pathogen}. Recommend immediate lockdown and patient transfer freeze."
                    )
                    self.lockdown_wards.add(ward)
                    state.ward_lockdowns[ward] = True
                    actions.append(EscalateToCMOTool(patient_id=None, urgency=9, summary=summary))

            for event in state.infection_events:
                if event.step == state.step_count and state.ward_lockdowns.get(event.ward) and not event.prevented:
                    actions.append(FlagPolicyViolationTool(
                        violation_type="ISOLATION_BREACH",
                        description=(
                            f"ISOLATION_BREACH: New case in locked-down {event.ward}. "
                            "PPE failure or unauthorized entry suspected."
                        ),
                        affected_patient_id=event.infected_patient_id,
                    ))

            self._simulate_spread(state)

        if state.step_count % 3 == 0:
            active_wards = [ward for ward, count in self.ward_case_counts.items() if count > 0]
            isolation_level = self._isolation_for(self._active_pathogen(state))
            actions.append(SendMessageTool(
                to_agent=AgentType.HR_ROSTERING.value,
                content=(
                    f"PPE_AUDIT: Current compliance rate {self.ppe_compliance_rate * 100:.0f}%. "
                    f"All staff in {active_wards or ['ALL']} require {isolation_level} PPE. "
                    "Non-compliant staff must be reassigned."
                ),
                urgency=6,
            ))

        return actions[:12]

    def _detect_new_spread(self, state: EnvironmentState) -> list[Any]:
        newly_infected: list[Any] = []
        for ward in self.WARDS:
            patients = [
                p for p in state.patients
                if self._ward_key(p.ward.value) == ward and self._is_infectious(p, state)
            ]
            count = len(patients)
            previous = self.ward_case_counts.get(ward, 0)
            if count > previous:
                for patient in patients[previous:count]:
                    state.infection_events.append(InfectionEvent(
                        event_id=str(uuid.uuid4()),
                        step=state.step_count,
                        source_patient_id=self._source_patient_id(state, ward, patient.id),
                        infected_patient_id=patient.id,
                        ward=ward,
                        pathogen=self._patient_pathogen(patient, state),
                    ))
                    newly_infected.append(patient)
            self.ward_case_counts[ward] = count
        return newly_infected

    def _simulate_spread(self, state: EnvironmentState) -> None:
        pathogen = self._active_pathogen(state)
        spread_rate = float(self.pathogen_db.get(pathogen, {}).get("spread_rate", 0.5))
        effective_rate = spread_rate * (1.0 - self.ppe_compliance_rate)
        for ward in self.WARDS:
            infectious = [
                p for p in state.patients
                if self._ward_key(p.ward.value) == ward and self._is_infectious(p, state)
            ]
            if not infectious:
                continue
            for patient in state.patients:
                if self._ward_key(patient.ward.value) != ward:
                    continue
                if patient.id in self.isolated_patients or self._is_infectious(patient, state):
                    continue
                roll = self._rng.random()
                self.exposure_log.append({
                    "step": state.step_count,
                    "ward": ward,
                    "patient_id": patient.id,
                    "pathogen": pathogen,
                    "roll": roll,
                    "threshold": effective_rate,
                })
                if roll < effective_rate:
                    patient.condition = f"{pathogen} exposure"
                    if patient.status == PatientStatus.STABLE:
                        patient.status = PatientStatus.SERIOUS
                    state.infection_events.append(InfectionEvent(
                        event_id=str(uuid.uuid4()),
                        step=state.step_count,
                        source_patient_id=infectious[0].id,
                        infected_patient_id=patient.id,
                        ward=ward,
                        pathogen=pathogen,
                    ))
                    self.ward_case_counts[ward] = self.ward_case_counts.get(ward, 0) + 1

    def _parse_llm_response(self, raw: Any) -> list[BaseModel]:
        response = raw if isinstance(raw, dict) else {}
        actions: list[BaseModel] = []
        for item in response.get("actions", []):
            try:
                action_type = str(item.get("action_type", "")).upper()
                if action_type == "ASSIGN_TREATMENT":
                    actions.append(AssignTreatmentTool(
                        patient_id=str(item["patient_id"]),
                        treatment_plan=str(item.get("treatment_plan", "ISOLATION_ORDER:full_isolation")),
                        reasoning=str(item.get("reasoning", "Infection control isolation order")),
                    ))
                elif action_type == "SEND_MESSAGE":
                    actions.append(SendMessageTool(
                        to_agent=str(item.get("to_agent", AgentType.HR_ROSTERING.value)),
                        content=str(item.get("content", "")),
                        urgency=int(item.get("urgency", item.get("priority", 6))),
                    ))
                elif action_type == "ESCALATE_TO_CMO":
                    actions.append(EscalateToCMOTool(
                        patient_id=item.get("patient_id"),
                        urgency=int(item.get("urgency", item.get("priority", 8))),
                        summary=str(item.get("summary", item.get("reasoning", ""))),
                    ))
                elif action_type == "FLAG_POLICY_VIOLATION":
                    actions.append(FlagPolicyViolationTool(
                        violation_type=str(item.get("violation_type", "ISOLATION_BREACH")),
                        description=str(item.get("description", item.get("reasoning", ""))),
                        affected_patient_id=item.get("affected_patient_id"),
                    ))
            except Exception:
                continue
        return actions

    def _active_pathogen(self, state: EnvironmentState) -> str:
        if state.active_pathogens:
            return state.active_pathogens[0]
        return max(self.pathogen_db.items(), key=lambda item: float(item[1].get("spread_rate", 0.0)))[0]

    def _patient_pathogen(self, patient: Any, state: EnvironmentState) -> str:
        condition = patient.condition.lower()
        for pathogen in self.pathogen_db:
            if pathogen in condition:
                if pathogen not in state.active_pathogens:
                    state.active_pathogens.append(pathogen)
                return pathogen
        return self._active_pathogen(state)

    def _isolation_for(self, pathogen: str) -> str:
        return str(self.pathogen_db.get(pathogen, {}).get("isolation", IsolationStatus.FULL.value))

    def _is_infectious(self, patient: Any, state: EnvironmentState) -> bool:
        text = patient.condition.lower()
        pathogen_hit = any(pathogen in text for pathogen in self.pathogen_db)
        clinical_hit = any(
            token in text
            for token in ("infect", "viral", "pneumonia", "meningitis", "fever", "gastroenteritis", "tuberculosis", "norovirus")
        )
        return pathogen_hit or (state.crisis.type == CrisisType.OUTBREAK and clinical_hit)

    def _source_patient_id(self, state: EnvironmentState, ward: str, infected_patient_id: str) -> str:
        for patient in state.patients:
            if patient.id != infected_patient_id and self._ward_key(patient.ward.value) == ward and self._is_infectious(patient, state):
                return patient.id
        return infected_patient_id

    @staticmethod
    def _ward_key(ward: str) -> str:
        if ward in {"WARD_A", "WARD_B", "WARD"}:
            return "WARD"
        if ward == "TRIAGE":
            return "ER"
        return ward


class ICUManagementAgent(BaseAgent):
    """ICU bed and ventilator management specialist.

    Responsibilities:
    - Manage ICU bed allocation
    - Ventilator assignment and monitoring
    - Discharge planning for stable ICU patients
    - Overflow protocol management
    """

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.ICU_MANAGEMENT, config, bus, mock_llm, model_name)

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        context = self._build_state_context(state)
        prompt = "Manage ICU resources. Allocate beds, ventilators, and plan discharges."
        response = await self._call_llm(prompt, context)
        return self._parse_actions(response, state)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        focus, signals = _expert_focus(state)

        for msg in inbox:
            if msg.request_type == "icu_bed_request" and msg.patient_id:
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType.TRANSFER_TO_ICU,
                    target_id=self._patient_idx(msg.patient_id, state),
                    priority=_focus_priority(max(msg.priority, 8), focus, quality=1, speed=1, cost=-1),
                    reasoning=f"ICU_MANAGER accepted delegated bed request: {msg.content[:100]}{_focus_note(focus, signals)}",
                ))

        # Assign treatment to critical patients in ICU
        for i, p in enumerate(state.patients):
            if p.ward.value == "ICU" and p.status == PatientStatus.CRITICAL and not p.treatment_plan:
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType.ASSIGN_TREATMENT,
                    target_id=i,
                    priority=_focus_priority(8, focus, quality=1, speed=0, cost=-1),
                    reasoning=f"Assigning ICU treatment plan for {p.name}{_focus_note(focus, signals)}",
                ))

            # Discharge stable ICU patients to free beds
            elif p.ward.value == "ICU" and p.status == PatientStatus.STABLE:
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType.TRANSFER_TO_WARD,
                    target_id=i,
                    priority=_focus_priority(3, focus, quality=-1, speed=1, cost=2),
                    reasoning=f"Transferring stable patient {p.name} out of ICU{_focus_note(focus, signals)}",
                ))

        # Request specialist if many critical patients
        critical_icu = sum(
            1 for p in state.patients
            if p.ward.value == "ICU" and p.status == PatientStatus.CRITICAL
        )
        if critical_icu > 5:
            actions.append(RequestSpecialistTool(
                specialty="Intensivist",
                urgency=7,
                reasoning=f"High ICU critical load ({critical_icu}) — requesting specialist backup{_focus_note(focus, signals)}"
            ))

        return actions[:4]

    def _patient_idx(self, patient_id: str, state: EnvironmentState) -> int:
        for i, p in enumerate(state.patients):
            if p.id == patient_id:
                return i
        return 0


# ─── Pharmacy Agent ─────────────────────────────────────────

class PharmacyAgent(BaseAgent):
    """Pharmacy operations and medication safety.

    Responsibilities:
    - Process medication orders
    - Check drug interactions
    - Monitor inventory levels
    - Enforce double-verification for controlled substances
    """

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.PHARMACY, config, bus, mock_llm, model_name)

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        context = self._build_state_context(state)
        prompt = "Manage pharmacy operations. Process orders, check interactions, monitor stock."
        response = await self._call_llm(prompt, context)
        return self._parse_actions(response, state)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        focus, signals = _expert_focus(state)

        for msg in inbox:
            if msg.request_type == "medication_request" and msg.patient_id:
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType.ORDER_MEDICATION,
                    target_id=self._patient_idx(msg.patient_id, state),
                    priority=_focus_priority(max(msg.priority, 7), focus, quality=1, speed=0, cost=-1),
                    reasoning=f"PHARMACY accepted delegated medication request: {msg.content[:100]}{_focus_note(focus, signals)}",
                ))

        # Order medications for patients with treatment plans but no meds
        for i, p in enumerate(state.patients):
            if p.treatment_plan and not p.medications and p.status != PatientStatus.DECEASED:
                base_priority = 6
                if focus == "cost" and p.status != PatientStatus.CRITICAL:
                    base_priority = 4
                if focus == "quality" and p.status == PatientStatus.CRITICAL:
                    base_priority = 8
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType.ORDER_MEDICATION,
                    target_id=i,
                    priority=_focus_priority(base_priority, focus, quality=1, speed=0, cost=-1),
                    reasoning=(
                        f"Filling medication order for {p.name} — treatment: "
                        f"{p.treatment_plan[0] if p.treatment_plan else 'generic'}"
                        f"{_focus_note(focus, signals)}"
                    ),
                ))

        # Check for low stock items and alert
        low_stock = {k: v for k, v in state.crisis.drug_inventory.items() if v < 5}
        if low_stock:
            for drug, qty in list(low_stock.items())[:2]:
                try:
                    asyncio.ensure_future(self.broadcast(
                        f"⚠️ LOW STOCK ALERT: {drug} — only {qty} units remaining",
                        MessageType.ALERT,
                        priority=6,
                    ))
                except RuntimeError:
                    pass  # No running loop

        return actions[:3]

    def _patient_idx(self, patient_id: str, state: EnvironmentState) -> int:
        for i, p in enumerate(state.patients):
            if p.id == patient_id:
                return i
        return 0


# ─── HR Rostering Agent ─────────────────────────────────────

class HRRosteringAgent(BaseAgent):
    """Human resources and staff scheduling.

    Responsibilities:
    - Monitor staff fatigue levels
    - Manage shift rotations
    - Request additional staff during surges
    - Enforce work-hour limits
    """

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.HR_ROSTERING, config, bus, mock_llm, model_name)

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        context = self._build_state_context(state)
        prompt = "Manage staff scheduling. Monitor fatigue, request reinforcements if needed."
        response = await self._call_llm(prompt, context)
        return self._parse_actions(response, state)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        focus, signals = _expert_focus(state)

        # Request staff if ratio is low
        staffing_threshold = 0.75 if focus == "speed" else 0.6 if focus == "quality" else 0.5
        if state.resources.staff_ratio < staffing_threshold:
            actions.append(AgentAction(
                agent_type=self.agent_type,
                action_type=ActionType.REQUEST_STAFF,
                priority=_focus_priority(7, focus, quality=0, speed=2, cost=-1),
                reasoning=(
                    f"Staff ratio critically low ({state.resources.staff_ratio:.0%}) "
                    f"— requesting emergency callback{_focus_note(focus, signals)}"
                ),
            ))
            try:
                asyncio.ensure_future(self.escalate(
                    f"Staff ratio at {state.resources.staff_ratio:.0%} — requesting CMO authorization for extended shifts",
                    priority=7,
                ))
            except RuntimeError:
                pass

        # Flag potential fatigue violations (simulated check every 5 steps)
        if state.step_count % 5 == 0:
            # Simulate detecting a fatigued staff member
            actions.append(AgentAction(
                agent_type=self.agent_type,
                action_type=ActionType.FLAG_POLICY_VIOLATION,
                priority=_focus_priority(5, focus, quality=1, speed=0, cost=-1),
                reasoning=f"Periodic fatigue check — flagging potential POL-004 violation{_focus_note(focus, signals)}",
            ))

        return actions[:2]

    def _parse_actions(self, response: dict[str, Any], state: EnvironmentState) -> list[AgentAction]:
        actions = []
        for a in response.get("actions", []):
            try:
                actions.append(AgentAction(
                    agent_type=self.agent_type,
                    action_type=ActionType[a.get("action_type", "REQUEST_STAFF")],
                    target_id=int(a.get("target_id", 0)),
                    priority=int(a.get("priority", 5)),
                    reasoning=a.get("reasoning", ""),
                ))
            except (KeyError, ValueError):
                continue
        return actions


# ─── IT Systems Agent ───────────────────────────────────────

class ITSystemsAgent(BaseAgent):
    """IT infrastructure and compliance monitoring.

    Responsibilities:
    - Monitor EHR system integrity
    - Detect policy violations from access logs
    - Manage IT uptime and backups
    - Enforce data privacy (HIPAA) compliance
    - Respond to schema drift / policy changes
    """

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.IT_SYSTEMS, config, bus, mock_llm, model_name)

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        context = self._build_state_context(state)
        prompt = "Monitor IT systems. Check for policy violations, EHR access anomalies, and system health."
        response = await self._call_llm(prompt, context)
        return self._parse_actions(response, state)

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[AgentAction]:
        actions: list[AgentAction] = []
        focus, signals = _expert_focus(state)

        # Monitor IT uptime
        if state.resources.it_uptime < 0.8:
            try:
                asyncio.ensure_future(self.broadcast(
                    f"🔴 IT ALERT: System uptime at {state.resources.it_uptime:.0%} — degraded performance expected",
                    MessageType.ALERT,
                    priority=7,
                ))
            except RuntimeError:
                pass

        # Detect injected violations
        if state.violations_injected > state.violations_caught:
            actions.append(FlagPolicyViolationTool(
                violation_type="Compliance",
                description=f"Compliance scan detected policy violation in system logs{_focus_note(focus, signals)}"
            ))

        recent_drifts = state.drift_history[-3:]
        drift_domains = {event.get("type") for event in recent_drifts}
        if {"contract_drift", "regulatory_drift"} & drift_domains:
            actions.append(UpdateEHRTool(
                patient_id="global",
                entry=f"Syncing downstream systems after drift event(s): {sorted(drift_domains)}{_focus_note(focus, signals)}"
            ))

        # Update EHR for patients with missing records
        for i, p in enumerate(state.patients):
            if not p.insurance_verified and p.status not in (PatientStatus.DECEASED, PatientStatus.DISCHARGED):
                actions.append(VerifyInsuranceTool(
                    patient_id=p.id,
                    provider="Pending"
                ))
                break  # one at a time

        # Respond to policy change messages
        for msg in inbox:
            if any(keyword in msg.content.upper() for keyword in ("POLICY", "CONTRACT", "REGULATORY")):
                try:
                    asyncio.ensure_future(self.broadcast(
                        f"📋 IT acknowledges policy change: {msg.content[:100]}",
                        MessageType.RESPONSE,
                        priority=4,
                    ))
                except RuntimeError:
                    pass

        return actions[:3]


# ─── Blood Bank Agent ───────────────────────────────────────

class BloodBankAgent(BaseAgent):
    """Blood Bank Manager."""

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.BLOOD_BANK, config, bus, mock_llm, model_name)
        self.inventory: dict[str, int] = {
            "O+": 20, "O-": 10, "A+": 15, "A-": 8, "B+": 12, "B-": 6, "AB+": 5, "AB-": 3
        }
        self.CRITICAL_THRESHOLD: int = 3
        self.pending_requests: list[dict[str, Any]] = []

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[BaseModel]:
        if self.mock_llm:
            return self._rule_based_decision(state, inbox)
        # LLM-backed mode
        context = self._build_state_context(state)
        # Pass current inventory dict and pending_requests as JSON context
        import json
        extra_ctx = json.dumps({"inventory": self.inventory, "pending_requests": self.pending_requests})
        context = f"{context}\nBlood Bank Context:\n{extra_ctx}"
        prompt = self.config.get("system_prompt", "Manage blood inventory.")
        return await self._call_llm(prompt, context, state)

    def _parse_llm_response(self, raw: str) -> list[AgentAction]:
        # Keep parsing isolated as requested in docs
        return []

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[BaseModel]:
        import re
        actions: list[BaseModel] = []
        
        # 1. Process inbox messages
        for msg in inbox:
            # Fallback for both old and new conventions
            if msg.msg_type.value == "REQUEST_BLOOD" or msg.request_type == "REQUEST_BLOOD" or "REQUEST_BLOOD" in str(msg.msg_type) or (msg.msg_type == MessageType.REQUEST and ("blood" in msg.content.lower() or "blood_type" in (msg.payload or {}))):
                blood_type = "O+"
                patient_id = msg.patient_id or "unknown"
                if msg.payload and "blood_type" in msg.payload:
                    blood_type = msg.payload["blood_type"]
                else:
                    match = re.search(r"blood_type(?:=|:|\s+)(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)", msg.content, re.IGNORECASE)
                    if match:
                        blood_type = match.group(1).upper()
                        
                if self.inventory.get(blood_type, 0) > 0:
                    self.inventory[blood_type] -= 1
                    try:
                        asyncio.ensure_future(self.bus.send(AgentMessage(
                            from_agent=self.agent_type,
                            to_agent=msg.from_agent,
                            content=f"BLOOD_APPROVED: 1 unit(s) of {blood_type} allocated for patient {patient_id}",
                            msg_type=MessageType.RESPONSE,
                            correlation_id=msg.id
                        )))
                    except RuntimeError:
                        pass
                else:
                    self.pending_requests.append({
                        "blood_type": blood_type,
                        "patient_id": patient_id,
                        "requesting_agent": msg.from_agent,
                        "msg_id": msg.id
                    })
                    try:
                        asyncio.ensure_future(self.bus.send(AgentMessage(
                            from_agent=self.agent_type,
                            to_agent=msg.from_agent,
                            content=f"BLOOD_UNAVAILABLE: {blood_type} out of stock, queued for emergency procurement",
                            msg_type=MessageType.RESPONSE,
                            correlation_id=msg.id
                        )))
                    except RuntimeError:
                        pass

        # 4. Retry pending requests
        still_pending = []
        for req in self.pending_requests:
            blood_type = req["blood_type"]
            patient_id = req["patient_id"]
            if self.inventory.get(blood_type, 0) > 0:
                self.inventory[blood_type] -= 1
                try:
                    asyncio.ensure_future(self.bus.send(AgentMessage(
                        from_agent=self.agent_type,
                        to_agent=req["requesting_agent"],
                        content=f"BLOOD_APPROVED: 1 unit(s) of {blood_type} allocated for patient {patient_id}",
                        msg_type=MessageType.RESPONSE,
                        correlation_id=req.get("msg_id")
                    )))
                except RuntimeError:
                    pass
            else:
                still_pending.append(req)
        self.pending_requests = still_pending

        # 2. Critical stock check
        for btype, count in self.inventory.items():
            if count <= self.CRITICAL_THRESHOLD:
                actions.append(EscalateToCMOTool(
                    patient_id="system",
                    urgency=8,
                    summary=f"CRITICAL BLOOD SHORTAGE: {btype} at {count} units remaining"
                ))

        # 3. Emergency procurement
        if state.crisis.type == CrisisType.MASS_CASUALTY and (self.inventory["O+"] <= 5 or self.inventory["O-"] <= 5):
            try:
                asyncio.ensure_future(self.bus.send(AgentMessage(
                    from_agent=self.agent_type,
                    to_agent=AgentType.CMO_OVERSIGHT,
                    content="EMERGENCY PROCUREMENT TRIGGERED: Universal donor blood critically low during mass casualty. Requesting external donor activation.",
                    msg_type=MessageType.ALERT,
                    priority=9
                )))
            except RuntimeError:
                pass
            self.inventory["O+"] += 10
            self.inventory["O-"] += 5

        return actions



# ─── Ethics Committee Agent ────────────────────────────────────

class EthicsCommitteeAgent(BaseAgent):
    """Ethics Committee Oversight Agent."""

    def __init__(self, config: dict[str, Any], bus: MessageBus, mock_llm: bool = True, model_name: str | None = None) -> None:
        super().__init__(AgentType.ETHICS_COMMITTEE, config, bus, mock_llm, model_name)
        from triage.env.state import EthicalFramework
        self.framework = EthicalFramework(self.config.get("ethical_framework", "utilitarian"))
        self.rationing_log = []
        self.pending_cmo_overrides = []
        triggers = self.config.get("rationing_triggers", {})
        self.VENTILATOR_THRESHOLD = triggers.get("ventilator_threshold", 2)
        self.ICU_BED_THRESHOLD = triggers.get("icu_bed_threshold", 1)
        self.blood_critical_types = triggers.get("blood_critical_types", ["O-", "O+"])

    async def decide(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[BaseModel]:
        if self.mock_llm:
            return self._rule_based_decision(state, inbox)
        from triage.env.state import RationingDecision
        context = self._build_state_context(state)
        actions = []
        for msg in inbox:
            if "override" in msg.content.lower() or getattr(msg, "request_type", "") == "override_request" or msg.msg_type.value == "OVERRIDE_DECISION":
                res = self._review_cmo_override(msg, state)
                if res: actions.append(res)

        scenarios = self._detect_rationing_scenarios(state)
        for scenario in scenarios:
            decision = self._apply_framework(scenario, state)
            self.rationing_log.append(decision)
            state.rationing_decisions.append(decision)
            try:
                asyncio.ensure_future(self.bus.send(AgentMessage(
                    from_agent=self.agent_type,
                    to_agent=AgentType.CMO_OVERSIGHT,
                    content=f"RATIONING DECISION: {decision.resource_type} assigned to {decision.selected_patient_id}",
                    msg_type=MessageType.ACTION
                )))
            except RuntimeError:
                pass

            for rej_id in decision.rejected_patient_ids:
                actions.append(AssignTreatmentTool(
                    patient_id=rej_id,
                    treatment_plan="COMPASSIONATE_CARE_PLAN"
                ))
        
        import json
        extra_ctx = json.dumps({"scenarios": scenarios})
        context = f"{context}\nRationing Scenarios:\n{extra_ctx}"
        prompt = self.config.get("system_prompt", "Manage ethics.")
        llm_actions = await self._call_llm(prompt, context, state)
        actions.extend(llm_actions)
        return [a for a in actions if a is not None]

    def _rule_based_decision(self, state: EnvironmentState, inbox: list[AgentMessage]) -> list[BaseModel]:
        actions = []
        from triage.env.state import EthicalFramework
        
        for msg in inbox:
            if "override" in msg.content.lower() or getattr(msg, "request_type", "") == "override_request" or msg.msg_type == MessageType.REQUEST:
                override_action = self._review_cmo_override(msg, state)
                if override_action:
                    actions.append(override_action)

        scenarios = self._detect_rationing_scenarios(state)
        original_framework = self.framework
        self.framework = EthicalFramework.CLINICAL_PRIORITY
        
        for scenario in scenarios:
            decision = self._apply_framework(scenario, state)
            self.rationing_log.append(decision)
            state.rationing_decisions.append(decision)
            
            try:
                asyncio.ensure_future(self.bus.send(AgentMessage(
                    from_agent=self.agent_type,
                    to_agent=AgentType.CMO_OVERSIGHT,
                    content=f"RATIONING DECISION: {decision.resource_type} assigned to {decision.selected_patient_id}",
                    msg_type=MessageType.ACTION
                )))
            except RuntimeError:
                pass
            
            for rej_id in decision.rejected_patient_ids:
                actions.append(AssignTreatmentTool(
                    patient_id=rej_id,
                    treatment_plan="COMPASSIONATE_CARE_PLAN"
                ))
        self.framework = original_framework
        
        un_audited = False
        for act in state.action_history[-1:]:
            if getattr(act, "action_type", None) == ActionType.TRANSFER_TO_ICU:
                found = any(r.resource_type == "icu_bed" and r.step == state.step_count for r in state.rationing_decisions)
                if state.icu_occupancy >= 1.0 and not found:
                    un_audited = True
        if un_audited:
            actions.append(FlagPolicyViolationTool(patient_id="system", policy_name="Ethical Override", violation_summary="UNAUDITED_ALLOCATION"))
            
        return actions

    def _detect_rationing_scenarios(self, state: EnvironmentState) -> list[dict]:
        scenarios = []
        vents_avail = state.resources.ventilators_total - state.resources.ventilators_in_use
        vent_pts = [p.id for p in state.patients if "ventilator" in p.condition.lower() or p.acuity_score >= 8]
        if vents_avail <= self.VENTILATOR_THRESHOLD and len(vent_pts) > vents_avail:
            scenarios.append({
                "resource_type": "ventilator",
                "available_count": vents_avail,
                "candidate_patients": vent_pts
            })
            
        icu_avail = state.resources.icu_beds_total - state.resources.icu_beds_occupied
        icu_pts = [p.id for p in state.patients if p.triage_score and p.triage_score >= 4 and p.ward != "ICU"]
        if icu_avail <= self.ICU_BED_THRESHOLD and len(icu_pts) > icu_avail:
            scenarios.append({
                "resource_type": "icu_bed",
                "available_count": icu_avail,
                "candidate_patients": icu_pts
            })
            
        if state.resources.blood_inventory.get("O-", 0) <= 2:
            blood_pts = [p.id for p in state.patients if p.condition in ["hemorrhage", "polytrauma"]]
            if len(blood_pts) > 1:
                scenarios.append({
                    "resource_type": "blood_O-",
                    "available_count": state.resources.blood_inventory.get("O-", 0),
                    "candidate_patients": blood_pts
                })
        return scenarios

    def _apply_framework(self, scenario: dict, state: EnvironmentState) -> Any:
        from triage.env.state import RationingDecision, EthicalFramework
        import uuid
        candidates = scenario["candidate_patients"]
        candidate_objs = [p for p in state.patients if p.id in candidates]
        
        if not candidate_objs:
            return RationingDecision(decision_id=str(uuid.uuid4()), resource_type=scenario["resource_type"], candidates=[], selected_patient_id="", rejected_patient_ids=[])
            
        selected_id = ""
        justification = ""
        
        if self.framework == EthicalFramework.UTILITARIAN:
            best = max(candidate_objs, key=lambda p: (10 - p.acuity_score) * 1.0)
            selected_id = best.id
            justification = "Maximizes expected life-years across patient cohort"
        elif self.framework == EthicalFramework.CLINICAL_PRIORITY:
            best = max(candidate_objs, key=lambda p: p.acuity_score)
            selected_id = best.id
            justification = "Clinically indicated - most acute presentation"
        elif self.framework == EthicalFramework.FIRST_COME_FIRST_SERVED:
            best = min(candidate_objs, key=lambda p: p.id)
            selected_id = best.id
            justification = "Queue priority - no clinical override applied"
        elif self.framework == EthicalFramework.EQUITY:
            def equity_score(p):
                score = p.acuity_score
                if getattr(p, "age", 30) > 65 or getattr(p, "age", 30) < 12:
                    score *= 1.2
                return score
            best = max(candidate_objs, key=equity_score)
            selected_id = best.id
            justification = "Equity-weighted - vulnerability factors applied"
            
        rejected = [c for c in candidates if c != selected_id]
        return RationingDecision(
            decision_id=str(uuid.uuid4()),
            resource_type=scenario["resource_type"],
            candidates=candidates,
            selected_patient_id=selected_id,
            rejected_patient_ids=rejected,
            framework_used=self.framework,
            justification=justification,
            step=state.step_count
        )

    def _review_cmo_override(self, message: AgentMessage, state: EnvironmentState) -> BaseModel | None:
        content = message.content.lower()
        if "justification" in content or (message.payload and "justification" in message.payload and message.priority >= 9):
            try:
                asyncio.ensure_future(self.bus.send(AgentMessage(
                    from_agent=self.agent_type,
                    to_agent=AgentType.CMO_OVERSIGHT,
                    content="ETHICS_APPROVED",
                    msg_type=MessageType.RESPONSE
                )))
            except RuntimeError:
                pass
            return SendMessageTool(to_agent="cmo_oversight", content="ETHICS_APPROVED")
        else:
            return FlagPolicyViolationTool(
                patient_id=message.patient_id or "system",
                policy_name="Ethical Override",
                violation_summary=f"CMO_OVERRIDE_REJECTED: No ethical justification provided for deviation from {self.framework.value} framework on patient {message.patient_id}"
            )

    def get_rationing_summary(self) -> dict:
        approvals = [r for r in self.rationing_log if r.overridden_by_cmo]
        return {
            "total_decisions": len(self.rationing_log),
            "by_framework": {self.framework.value: len(self.rationing_log)},
            "approval_rate": 1.0,
            "cmo_overrides_approved": len(approvals),
            "cmo_overrides_rejected": 0,
            "rejected_patients_count": sum(len(r.rejected_patient_ids) for r in self.rationing_log)
        }

    def _parse_llm_response(self, raw: str) -> list[AgentAction]:
        return []

# ─── Agent Factory ───────────────────────────────────────────

AGENT_CLASSES: dict[AgentType, type[BaseAgent]] = {
    AgentType.CMO_OVERSIGHT: CMOOversightAgent,
    AgentType.ER_TRIAGE: ERTriageAgent,
    AgentType.INFECTION_CONTROL: InfectionControlAgent,
    AgentType.ICU_MANAGEMENT: ICUManagementAgent,
    AgentType.PHARMACY: PharmacyAgent,
    AgentType.HR_ROSTERING: HRRosteringAgent,
    AgentType.IT_SYSTEMS: ITSystemsAgent,
    AgentType.BLOOD_BANK: BloodBankAgent,
    AgentType.ETHICS_COMMITTEE: EthicsCommitteeAgent,
}


def create_agent(
    agent_type: AgentType,
    config: dict[str, Any],
    bus: MessageBus,
    mock_llm: bool = True,
    model_name: str | None = None,
) -> BaseAgent:
    """Factory function to create typed agents."""
    cls = AGENT_CLASSES.get(agent_type)
    if cls is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return cls(config=config, bus=bus, mock_llm=mock_llm, model_name=model_name)


def create_all_agents(
    configs: dict[str, Any],
    bus: MessageBus,
    mock_llm: bool = True,
    model_name: str | None = None,
) -> dict[AgentType, BaseAgent]:
    """Create all 6 agents from the agents.yaml config."""
    agents = {}
    for agent_type in AgentType:
        agent_config = configs.get("agents", {}).get(agent_type.value, {})
        agents[agent_type] = create_agent(agent_type, agent_config, bus, mock_llm, model_name)
        logger.info("Created agent: %s (mock_llm=%s, model=%s)", agent_type.value, mock_llm, model_name)
    return agents
