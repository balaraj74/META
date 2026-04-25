"""TRL GRPO environment_factory adapter for live HospitalEnv rollouts."""

from __future__ import annotations

import asyncio
import os
from typing import Any


def _run(coro):
    """Run a HospitalEnv coroutine from TRL's synchronous tool interface."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


class HospitalGRPOEnvironment:
    """Environment-factory class consumed directly by TRL GRPOTrainer."""

    def __init__(self):
        self.current_state = None
        self.step_count = 0
        self._last_reward = 0.0
        self._done = False
        self._use_http = os.getenv("GRPO_TRAINING_MODE", "").lower() == "true"
        self._base_url = os.getenv("TRIAGE_ENV_SERVER_URL", "http://localhost:8001")
        self._client = None
        self.env = None

        if self._use_http:
            try:
                import httpx

                self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
                self._client.get("/health").raise_for_status()
                return
            except Exception:
                self._client = None
                self._use_http = False

        from triage.env.hospital_env import HospitalEnv

        self.env = HospitalEnv(max_steps=int(os.getenv("GRPO_ENV_MAX_STEPS", "50")))

    def reset(self, crisis_type: str = "mass_casualty", **kwargs) -> str:
        """Reset environment for a new episode."""
        difficulty = kwargs.get("difficulty", 0.5)
        if self._client is not None:
            payload = {"crisis_type": crisis_type, "difficulty": difficulty}
            response = self._client.post("/reset", json=payload)
            response.raise_for_status()
            self.current_state = response.json()["state"]
        else:
            _run(
                self.env.reset(
                    scenario={"crisis_type": crisis_type, "difficulty": difficulty}
                )
            )
            self.current_state = self.env.state

        self.step_count = 0
        self._last_reward = 0.0
        self._done = False
        return self._format_observation(self.current_state)

    def triage_patient(self, patient_id: str, acuity_score: int, assigned_ward: str) -> str:
        """
        Triage a patient and assign initial acuity level.
        Args:
            patient_id: UUID of the patient to triage
            acuity_score: Severity score 1-10, where 10 is most critical
            assigned_ward: Target ward - one of: ER, ICU, WARD, ISOLATION
        """
        if self._client is not None:
            return self._remote_tool(
                "triage_patient",
                {
                    "patient_id": patient_id,
                    "acuity_score": acuity_score,
                    "assigned_ward": assigned_ward,
                },
            )

        patient, target_id = self._find_patient(patient_id)
        if patient is not None:
            from triage.env.state import PatientStatus

            patient.triage_score = max(1, min(10, int(acuity_score)))
            patient.ward = self._ward(assigned_ward)
            if patient.triage_score >= 8:
                patient.status = PatientStatus.CRITICAL
                patient.icu_required = True
            elif patient.triage_score >= 5:
                patient.status = PatientStatus.SERIOUS
            else:
                patient.status = PatientStatus.STABLE
            patient.add_event("TRIAGE_UPDATE", f"Acuity updated to {patient.triage_score}")
        return self._step("TRIAGE_PATIENT", "er_triage", target_id, acuity_score, "Initial triage")

    def transfer_to_icu(self, patient_id: str, reason: str) -> str:
        """
        Transfer a patient to the ICU.
        Args:
            patient_id: UUID of patient to transfer
            reason: Clinical justification for ICU transfer
        """
        return self._step("TRANSFER_TO_ICU", "icu_management", patient_id, 9, reason)

    def order_medication(
        self, patient_id: str, drug_name: str, dose_mg: float, reason: str
    ) -> str:
        """
        Order medication for a patient.
        Args:
            patient_id: UUID of target patient
            drug_name: Name of the medication to prescribe
            dose_mg: Dosage in milligrams
            reason: Clinical reason for this prescription
        """
        patient, _ = self._find_patient(patient_id)
        if patient is not None:
            patient.medications.append(f"{drug_name} {dose_mg:g}mg")
            patient.add_event("MEDICATION_ORDER", f"{drug_name} {dose_mg:g}mg: {reason}")
        return self._step("ORDER_MEDICATION", "pharmacy", patient_id, 7, reason)

    def request_blood(self, patient_id: str, blood_type: str, units: int) -> str:
        """
        Request blood product for a patient.
        Args:
            patient_id: UUID of patient needing blood
            blood_type: ABO/Rh type e.g. O-, A+, AB+
            units: Number of units requested (1-6)
        """
        units = max(1, min(6, int(units)))
        if self._client is None and self.current_state is not None:
            inventory = self.current_state.resources.blood_inventory
            inventory[blood_type] = max(0, int(inventory.get(blood_type, 0)) - units)
        return self._step(
            "REQUEST_BLOOD",
            "blood_bank",
            patient_id,
            min(10, units + 4),
            f"Request {units} units {blood_type}",
        )

    def escalate_to_cmo(self, patient_id: str, urgency: int, summary: str) -> str:
        """
        Escalate a patient situation to CMO oversight.
        Args:
            patient_id: UUID of the patient being escalated
            urgency: Urgency level 1-10
            summary: Brief clinical summary for CMO
        """
        return self._step("ESCALATE_TO_CMO", "er_triage", patient_id, urgency, summary)

    def discharge_patient(self, patient_id: str, discharge_notes: str) -> str:
        """
        Discharge a patient from the hospital.
        Args:
            patient_id: UUID of patient to discharge
            discharge_notes: Summary of care and follow-up instructions
        """
        return self._step("DISCHARGE_PATIENT", "er_triage", patient_id, 3, discharge_notes)

    def allocate_equipment(self, equipment_type: str, patient_id: str) -> str:
        """
        Allocate critical equipment to a patient.
        Args:
            equipment_type: Type of equipment - ventilator, monitor, pump
            patient_id: UUID of patient receiving equipment
        """
        return self._step(
            "ALLOCATE_EQUIPMENT",
            "icu_management",
            patient_id,
            8,
            f"Allocate {equipment_type}",
        )

    def activate_protocol(self, protocol_name: str, justification: str) -> str:
        """
        Activate a hospital emergency protocol.
        Args:
            protocol_name: Protocol to activate - MASS_CASUALTY, OUTBREAK,
                           LOCKDOWN, OVERFLOW, CODE_RED
            justification: Clinical or operational justification
        """
        action = "ACTIVATE_OVERFLOW" if protocol_name.upper() == "OVERFLOW" else "ACTIVATE_PROTOCOL"
        return self._step(action, "cmo_oversight", 0, 9, f"{protocol_name}: {justification}")

    def _format_observation(self, state) -> str:
        """Convert EnvironmentState to rich text for LLM consumption."""
        if isinstance(state, dict):
            return self._format_dict_observation(state)

        resources = state.resources
        blood = resources.blood_inventory
        patients = sorted(
            state.patients,
            key=lambda p: (p.status.value == "CRITICAL", p.triage_score),
            reverse=True,
        )[:5]
        patient_lines = [
            f"- {p.id}: acuity {p.triage_score}, {p.condition}, {p.ward.value}"
            for p in patients
        ] or ["- none"]
        messages = [
            f"- {m.from_agent.value if hasattr(m.from_agent, 'value') else m.from_agent}: {m.content[:120]}"
            for m in state.message_history[-3:]
        ] or ["- none"]
        return "\n".join(
            [
                f"Episode {state.episode} step {state.step_count}.",
                f"Crisis: {state.crisis.type.value} severity {state.crisis.severity}.",
                "Top critical patients:",
                *patient_lines,
                (
                    "Resources: "
                    f"ICU beds free {resources.icu_beds_total - resources.icu_beds_occupied}; "
                    f"ventilators {resources.ventilators_total - resources.ventilators_in_use}; "
                    f"blood O+ {blood.get('O+', 0)} O- {blood.get('O-', 0)}."
                ),
                f"Active safety blocks: {len(state.safety_blocks)}.",
                "Last agent messages:",
                *messages,
            ]
        )

    def _get_terminal_reward(self) -> float:
        """Called at episode end. Returns scalar reward for this episode."""
        if self._client is not None:
            response = self._client.get("/terminal_reward")
            response.raise_for_status()
            return float(response.json()["reward"])

        if self.current_state is None:
            return 0.0
        state = self.current_state
        survival = state.survival_rate
        resources = state.resources
        icu_efficiency = 1.0 - min(1.0, max(0.0, state.icu_occupancy - 0.85) / 0.15)
        ventilator_efficiency = 1.0 - min(
            1.0,
            resources.ventilators_in_use / max(resources.ventilators_total, 1),
        )
        no_safety_violations = 1.0 if not state.safety_blocks else 0.0
        reward = survival * 0.55 + icu_efficiency * 0.2 + ventilator_efficiency * 0.1
        reward += no_safety_violations * 0.15
        return max(-1.0, min(1.0, (reward * 2.0) - 1.0))

    def _step(
        self,
        action_type: str,
        agent_type: str,
        patient_id: str | int,
        priority: int,
        reasoning: str,
    ) -> str:
        if self._client is not None:
            return self._remote_tool(
                "_step",
                {
                    "action_type": action_type,
                    "agent_type": agent_type,
                    "patient_id": patient_id,
                    "priority": priority,
                    "reasoning": reasoning,
                },
            )

        from triage.env.state import ActionType, AgentType

        _, target_id = self._find_patient(patient_id)
        action = {
            "agent_id": list(AgentType).index(AgentType(agent_type)),
            "action_type": ActionType[action_type].value,
            "target_id": target_id,
            "priority": max(1, min(10, int(priority))),
            "reasoning": str(reasoning)[:500],
            "reasoning_tokens": max(1, len(str(reasoning).split())),
        }
        _, reward, done, _ = _run(self.env.step(action))
        self._last_reward = float(reward)
        self._done = bool(done)
        self.current_state = self.env.state
        self.step_count = self.current_state.step_count
        return self._format_observation(self.current_state)

    def _remote_tool(self, tool_name: str, payload: dict[str, Any]) -> str:
        response = self._client.post(f"/tool/{tool_name}", json=payload)
        response.raise_for_status()
        data = response.json()
        self.current_state = data["state"]
        self.step_count = int(data.get("step", self.step_count + 1))
        self._last_reward = float(data.get("reward", 0.0))
        self._done = bool(data.get("done", False))
        return data["observation"]

    def _find_patient(self, patient_id: str | int):
        if self.current_state is None:
            return None, 0
        patients = self.current_state.get("patients", []) if isinstance(self.current_state, dict) else self.current_state.patients
        if isinstance(patient_id, int):
            idx = max(0, min(len(patients) - 1, patient_id)) if patients else 0
            return (patients[idx] if patients else None), idx
        patient_id = str(patient_id)
        for idx, patient in enumerate(patients):
            pid = patient.get("id") if isinstance(patient, dict) else patient.id
            if str(pid) == patient_id:
                return patient, idx
        if patient_id.isdigit():
            idx = max(0, min(len(patients) - 1, int(patient_id))) if patients else 0
            return (patients[idx] if patients else None), idx
        return (patients[0] if patients else None), 0

    @staticmethod
    def _ward(name: str):
        from triage.env.state import WardType

        mapping = {
            "ER": WardType.ER,
            "ICU": WardType.ICU,
            "WARD": WardType.WARD_A,
            "WARD_A": WardType.WARD_A,
            "WARD_B": WardType.WARD_B,
            "ISOLATION": WardType.WARD_B,
            "TRIAGE": WardType.TRIAGE,
        }
        return mapping.get(str(name).upper(), WardType.TRIAGE)

    @staticmethod
    def _format_dict_observation(state: dict[str, Any]) -> str:
        crisis = state.get("crisis", {})
        resources = state.get("resources", {})
        stats = state.get("stats", {})
        patients = sorted(
            state.get("patients", []),
            key=lambda p: (p.get("status") == "CRITICAL", p.get("triage_score", 0)),
            reverse=True,
        )[:5]
        blood = resources.get("blood_inventory", {})
        patient_lines = [
            f"- {p.get('id')}: acuity {p.get('triage_score', 0)}, {p.get('condition')}, {p.get('ward')}"
            for p in patients
        ] or ["- none"]
        messages = state.get("message_history", [])[-3:] or []
        message_lines = [
            f"- {m.get('from_agent', 'agent')}: {str(m.get('content', ''))[:120]}"
            for m in messages
        ] or ["- none"]
        return "\n".join(
            [
                f"Episode {state.get('episode')} step {state.get('step')}.",
                f"Crisis: {crisis.get('type')} severity {crisis.get('severity')}.",
                "Top critical patients:",
                *patient_lines,
                (
                    "Resources: "
                    f"ICU beds free {resources.get('icu_beds_available', 0)}; "
                    f"ventilators {resources.get('ventilators_available', 0)}; "
                    f"blood O+ {blood.get('O+', 0)} O- {blood.get('O-', 0)}."
                ),
                f"Active safety blocks: {len(state.get('safety_blocks', []))}.",
                f"Patients alive {stats.get('alive_count', 0)} critical {stats.get('critical_count', 0)}.",
                "Last agent messages:",
                *message_lines,
            ]
        )
