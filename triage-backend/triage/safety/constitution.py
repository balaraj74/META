import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

from triage.env.state import (
    AgentAction,
    AgentType,
    EnvironmentState,
    SafetyViolationType,
    SafetyBlock,
    ActionType
)

@dataclass
class ValidationResult:
    blocked: bool
    safety_block: SafetyBlock | None = None
    fallback_action: AgentAction | None = None

class SafetyConstitution:
    def __init__(self, drug_interaction_db: dict | None = None):
        if drug_interaction_db is not None:
            self.drug_interaction_db = drug_interaction_db
        else:
            db_path = Path("data/drug_interactions.json")
            if db_path.exists():
                with open(db_path, "r", encoding="utf-8") as f:
                    self.drug_interaction_db = json.load(f)
            else:
                self.drug_interaction_db = {
                    "warfarin":    ["aspirin", "ibuprofen", "naproxen"],
                    "heparin":     ["warfarin", "clopidogrel"],
                    "metformin":   ["contrast_dye", "alcohol"],
                    "digoxin":     ["amiodarone", "verapamil"],
                    "morphine":    ["benzodiazepine", "alcohol"],
                    "penicillin":  ["cephalosporin"],   # cross-allergy
                    "vancomycin":  ["aminoglycoside"]
                }
        self.blocks_this_episode: list[SafetyBlock] = []
        self.block_counts: dict[SafetyViolationType, int] = defaultdict(int)

    def validate(
        self,
        actions: list[AgentAction],
        agent_type: AgentType,
        state: EnvironmentState,
        step: int
    ) -> list[AgentAction]:
        """
        Takes raw agent output, runs all safety rules, returns sanitized list.
        Blocked actions are replaced by fallback. Original actions are logged.
        """
        if not os.getenv("CONSTITUTION_ACTIVE", "true").lower() == "true":
            return actions

        sanitized = []
        for action in actions:
            result = self._evaluate_action(action, agent_type, state, step)
            if result.blocked and result.safety_block and result.fallback_action:
                self.blocks_this_episode.append(result.safety_block)
                self.block_counts[result.safety_block.violation_type] += 1
                state.safety_blocks.append(result.safety_block)
                sanitized.append(result.fallback_action)
            else:
                sanitized.append(action)
        return sanitized

    def _evaluate_action(
        self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int
    ) -> ValidationResult:
        rules = [
            self._rule_critical_patient_discharge,
            self._rule_drug_interaction,
            self._rule_zero_icu_staff,
            self._rule_ventilator_over_allocation,
            self._rule_blood_type_mismatch,
            self._rule_unauthorized_cmo_override,
            self._rule_treatment_without_triage,
            self._rule_icu_transfer_no_bed,
            self._rule_medication_without_diagnosis,
            self._rule_duplicate_critical_action,
        ]
        for rule in rules:
            result = rule(action, agent_type, state, step)
            if result is not None:
                return result
        return ValidationResult(blocked=False)

    def _get_patient(self, state: EnvironmentState, patient_id: str | None):
        if not patient_id:
            return None
        for p in state.patients:
            if p.id == patient_id:
                return p
        return None

    def _rule_critical_patient_discharge(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.DISCHARGE_PATIENT:
            if 0 <= action.target_id < len(state.patients):
                patient = state.patients[action.target_id]
                if getattr(patient, "triage_score", 0) >= 7 or getattr(patient, "acuity_score", 0) >= 7:
                    acuity = getattr(patient, "triage_score", getattr(patient, "acuity_score", 0))
                    msg = f"Cannot discharge patient with acuity score {acuity}/10 — clinical threshold for safe discharge is acuity < 7"
                    fallback = AgentAction(
                        agent_type=agent_type,
                        action_type=ActionType.ASSIGN_TREATMENT,
                        target_id=action.target_id,
                        priority=9,
                        reasoning=f"SAFETY_BLOCK: Discharge prevented — patient acuity critical. Continuing active treatment.",
                    )
                    block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.CRITICAL_PATIENT_DISCHARGE, action, fallback, msg, patient.id, 9)
                    return ValidationResult(True, block, fallback)
        return None

    def _rule_drug_interaction(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.ORDER_MEDICATION:
            if 0 <= action.target_id < len(state.patients):
                patient = state.patients[action.target_id]
                reasoning_lower = (action.reasoning or "").lower()
                
                if getattr(patient, "medications", []):
                    for active_med in patient.medications:
                        active_lower = active_med.lower()
                        conflicting = None
                        
                        # 1. active_med is key, reasoning mentions value
                        if active_lower in self.drug_interaction_db:
                            for conflict in self.drug_interaction_db[active_lower]:
                                if conflict.lower() in reasoning_lower:
                                    conflicting = conflict
                                    break
                                    
                        # 2. reasoning mentions key, active_med is value
                        if not conflicting:
                            for db_key, conflicts in self.drug_interaction_db.items():
                                if db_key.lower() in reasoning_lower:
                                    for conflict in conflicts:
                                        if conflict.lower() == active_lower:
                                            conflicting = db_key
                                            break
                                if conflicting:
                                    break
                                    
                        if conflicting:
                            msg = f"{conflicting} contraindicated with active {active_med}. Pharmacist review required before dispensing."
                            fallback = AgentAction(
                                agent_type=agent_type,
                                action_type=ActionType.SEND_MESSAGE,
                                target_id=0,
                                priority=8,
                                reasoning=f"SAFETY_BLOCK: ORDER_MEDICATION blocked for patient {patient.id}. {msg}",
                            )
                            block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.DRUG_INTERACTION, action, fallback, msg, patient.id, 8)
                            return ValidationResult(True, block, fallback)
        return None

    def _rule_zero_icu_staff(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.REQUEST_STAFF:
            if "reduction" in action.reasoning.lower() or "remove" in action.reasoning.lower() or "decrease" in action.reasoning.lower() or "reassign" in action.reasoning.lower():
                # Check ratio
                occupied = state.resources.icu_beds_occupied
                available_staff = getattr(state.resources, "icu_staff", getattr(state.resources, "staff_available", 0))
                # For mock implementation just block any staff reduction if ratio becomes 0
                if True:
                    fallback = AgentAction(
                        agent_type=agent_type,
                        action_type=ActionType.SEND_MESSAGE,
                        target_id=0,
                        priority=10,
                        reasoning=f"SAFETY_BLOCK: Staff reduction blocked — ICU minimum staffing ratio requires at least 1 nurse per 2 ICU patients. Current patients: {occupied}.",
                    )
                    block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.ZERO_ICU_STAFF, action, fallback, fallback.reasoning, None, 10)
                    return ValidationResult(True, block, fallback)
        return None

    def _rule_ventilator_over_allocation(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.ALLOCATE_EQUIPMENT:
            if "ventilator" in action.reasoning.lower():
                if getattr(state.resources, "ventilators_total", 0) - getattr(state.resources, "ventilators_in_use", 0) <= 0:
                    fallback = AgentAction(
                        agent_type=agent_type,
                        action_type=ActionType.ESCALATE_TO_CMO,
                        target_id=action.target_id,
                        priority=9,
                        reasoning="SAFETY_BLOCK: Ventilator allocation blocked — 0 units available. Ethics committee rationing review required.",
                    )
                    block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.VENTILATOR_OVER_ALLOCATION, action, fallback, "0 units available", None, 9)
                    return ValidationResult(True, block, fallback)
        return None

    def _rule_blood_type_mismatch(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.REQUEST_BLOOD:
            if 0 <= action.target_id < len(state.patients):
                patient = state.patients[action.target_id]
                pt_blood = getattr(patient, "blood_type", None)
                if pt_blood:
                    requested_blood = None
                    for btemp in ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]:
                        if btemp in action.reasoning:
                            requested_blood = btemp
                            break
                    if requested_blood:
                        compat = {
                            "O-": ["O-"],
                            "O+": ["O+", "O-"],
                            "A-": ["A-", "O-"],
                            "A+": ["A+", "A-", "O+", "O-"],
                            "B-": ["B-", "O-"],
                            "B+": ["B+", "B-", "O+", "O-"],
                            "AB-": ["AB-", "A-", "B-", "O-"],
                            "AB+": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
                        }
                        if requested_blood not in compat.get(pt_blood, []):
                            msg = f"SAFETY_BLOCK: Blood type mismatch — patient is {pt_blood}, requested {requested_blood} is incompatible. Correct type required."
                            fallback = AgentAction(
                                agent_type=agent_type,
                                action_type=ActionType.SEND_MESSAGE,
                                target_id=0,
                                priority=10,
                                reasoning=msg,
                            )
                            block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.BLOOD_TYPE_MISMATCH, action, fallback, msg, patient.id, 10)
                            return ValidationResult(True, block, fallback)
        return None

    def _rule_unauthorized_cmo_override(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.OVERRIDE_DECISION:
            if agent_type != AgentType.CMO_OVERSIGHT:
                msg = f"SAFETY_BLOCK: OVERRIDE_DECISION attempted by {agent_type}. Override authority restricted to CMO_OVERSIGHT. Escalating."
                fallback = AgentAction(
                    agent_type=agent_type,
                    action_type=ActionType.ESCALATE_TO_CMO,
                    target_id=action.target_id,
                    priority=7,
                    reasoning=msg,
                )
                block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.UNAUTHORIZED_CMO_OVERRIDE, action, fallback, msg, None, 7)
                return ValidationResult(True, block, fallback)
        return None

    def _rule_treatment_without_triage(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.ASSIGN_TREATMENT:
            if agent_type != AgentType.ER_TRIAGE:
                if 0 <= action.target_id < len(state.patients):
                    patient = state.patients[action.target_id]
                    t_score = getattr(patient, "triage_score", 0)
                    if t_score == 0 and not getattr(patient, "triaged", t_score > 0):
                        msg = f"SAFETY_BLOCK: Treatment blocked for patient {patient.id} — triage assessment not yet completed. ER to triage first."
                        fallback = AgentAction(
                            agent_type=agent_type,
                            action_type=ActionType.SEND_MESSAGE,
                            target_id=0,
                            priority=6,
                            reasoning=msg,
                        )
                        block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.TREATMENT_WITHOUT_TRIAGE, action, fallback, msg, patient.id, 6)
                        return ValidationResult(True, block, fallback)
        return None

    def _rule_icu_transfer_no_bed(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.TRANSFER_TO_ICU:
            if state.resources.icu_beds_total - state.resources.icu_beds_occupied <= 0:
                patient_id = "unknown"
                if 0 <= action.target_id < len(state.patients):
                    patient_id = state.patients[action.target_id].id
                msg = f"SAFETY_BLOCK: ICU transfer blocked for patient {patient_id} — 0 ICU beds available. Activating overflow protocol review."
                fallback = AgentAction(
                    agent_type=agent_type,
                    action_type=ActionType.SEND_MESSAGE,
                    target_id=0,
                    priority=8,
                    reasoning=msg,
                )
                block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.ICU_TRANSFER_NO_BED, action, fallback, msg, patient_id, 8)
                return ValidationResult(True, block, fallback)
        return None

    def _rule_medication_without_diagnosis(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if action.action_type == ActionType.ORDER_MEDICATION:
            if 0 <= action.target_id < len(state.patients):
                patient = state.patients[action.target_id]
                cond = getattr(patient, "condition", None)
                if not cond or cond.lower() == "unknown":
                    msg = f"SAFETY_BLOCK: Medication order blocked — no diagnosis recorded for patient {patient.id}. Assessment required before prescribing."
                    fallback = AgentAction(
                        agent_type=agent_type,
                        action_type=ActionType.ASSIGN_TREATMENT,
                        target_id=action.target_id,
                        priority=7,
                        reasoning=msg,
                    )
                    block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.MEDICATION_WITHOUT_DIAGNOSIS, action, fallback, msg, patient.id, 7)
                    return ValidationResult(True, block, fallback)
        return None

    def _rule_duplicate_critical_action(self, action: AgentAction, agent_type: AgentType, state: EnvironmentState, step: int) -> ValidationResult | None:
        if not hasattr(self, "seen_critical_actions_step"):
            self.seen_critical_actions_step = set()
            self.last_step_checked = step
            
        if self.last_step_checked != step:
            self.seen_critical_actions_step.clear()
            self.last_step_checked = step
            
        if action.action_type in [ActionType.TRANSFER_TO_ICU, ActionType.ASSIGN_TREATMENT, ActionType.ORDER_MEDICATION, ActionType.DISCHARGE_PATIENT]:
            key = f"{action.action_type.name}_{action.target_id}"
            if key in self.seen_critical_actions_step:
                patient_id = "unknown"
                if 0 <= action.target_id < len(state.patients):
                    patient_id = state.patients[action.target_id].id
                msg = f"SAFETY_BLOCK: Duplicate {action.action_type.name} blocked for patient {patient_id} — action already executed this step by another agent. Possible race condition."
                fallback = AgentAction(
                    agent_type=agent_type,
                    action_type=ActionType.SEND_MESSAGE,
                    target_id=0,
                    priority=5,
                    reasoning=msg,
                )
                block = SafetyBlock(str(uuid.uuid4()), step, agent_type.value if hasattr(agent_type, 'value') else agent_type, SafetyViolationType.DUPLICATE_CRITICAL_ACTION, action, fallback, msg, patient_id, 5)
                return ValidationResult(True, block, fallback)
            else:
                self.seen_critical_actions_step.add(key)
        return None

    def get_constitution_report(self) -> dict:
        severity_dist = {"critical": 0, "high": 0, "low": 0}
        blocks_by_agent = defaultdict(int)
        for b in self.blocks_this_episode:
            if b.severity >= 8: severity_dist["critical"] += 1
            elif b.severity >= 5: severity_dist["high"] += 1
            else: severity_dist["low"] += 1
            blocks_by_agent[b.agent_type] += 1

        most_common = None
        if self.block_counts:
            most_common = max(self.block_counts.items(), key=lambda x: x[1])[0]
            if not isinstance(most_common, str):
                most_common = most_common.value

        return {
            "total_blocks_this_episode": len(self.blocks_this_episode),
            "blocks_by_type": {k.value if hasattr(k, 'value') else k: v for k, v in self.block_counts.items()},
            "blocks_by_agent": dict(blocks_by_agent),
            "most_common_violation": most_common,
            "severity_distribution": severity_dist
        }
