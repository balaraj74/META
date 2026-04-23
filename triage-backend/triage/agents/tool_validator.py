from typing import Any, Optional, Dict, Union
from dataclasses import dataclass
from triage.env.state import EnvironmentState, ActionType
from triage.agents.tools import (
    AGENT_TOOLS,
    TriagePatientTool, TransferToICUTool, TransferToWardTool, AssignTreatmentTool,
    OrderMedicationTool, RequestBloodTool, ActivateProtocolTool, RequestStaffTool,
    EscalateToCMOTool, DischargePatientTool, FlagPolicyViolationTool, UpdateEHRTool,
    VerifyInsuranceTool, AllocateEquipmentTool, SendMessageTool, OverrideDecisionTool,
    RequestSpecialistTool, ActivateOverflowTool, UpdateTreatmentPlanTool, CloseCaseTool
)

@dataclass
class ValidatedAction:
    action_type: ActionType
    kwargs: Dict[str, Any]
    target_id: int
    priority: int
    reasoning: str

@dataclass
class ValidationError:
    reason: str

class ToolValidationLayer:
    def __init__(self):
        pass

    def validate(self, tool_name: str, tool_kwargs: Dict[str, Any], state: EnvironmentState) -> Union[ValidatedAction, ValidationError]:
        """Validates tool arguments against current EnvironmentState."""
        try:
            if tool_name == "TriagePatientTool":
                tool = TriagePatientTool(**tool_kwargs)
                action_type = ActionType.TRIAGE_PATIENT
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found in state.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=tool.triage_score, reasoning=tool.reasoning)

            elif tool_name == "TransferToICUTool":
                tool = TransferToICUTool(**tool_kwargs)
                action_type = ActionType.TRANSFER_TO_ICU
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                if state.resources.icu_beds_occupied >= state.resources.icu_beds_total:
                    return ValidationError("Cannot transfer to ICU. ICU beds capacity reached (0 available).")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=tool.priority, reasoning=tool.reasoning)

            elif tool_name == "TransferToWardTool":
                tool = TransferToWardTool(**tool_kwargs)
                action_type = ActionType.TRANSFER_TO_WARD
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "AssignTreatmentTool":
                tool = AssignTreatmentTool(**tool_kwargs)
                action_type = ActionType.ASSIGN_TREATMENT
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "OrderMedicationTool":
                tool = OrderMedicationTool(**tool_kwargs)
                action_type = ActionType.ORDER_MEDICATION
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "RequestBloodTool":
                tool = RequestBloodTool(**tool_kwargs)
                action_type = ActionType.REQUEST_BLOOD
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "ActivateProtocolTool":
                tool = ActivateProtocolTool(**tool_kwargs)
                action_type = ActionType.ACTIVATE_PROTOCOL
                return ValidatedAction(action_type, tool_kwargs, target_id=0, priority=8, reasoning=tool.justification)

            elif tool_name == "RequestStaffTool":
                tool = RequestStaffTool(**tool_kwargs)
                action_type = ActionType.REQUEST_STAFF
                return ValidatedAction(action_type, tool_kwargs, target_id=0, priority=tool.urgency, reasoning=tool.reasoning)

            elif tool_name == "EscalateToCMOTool":
                tool = EscalateToCMOTool(**tool_kwargs)
                action_type = ActionType.ESCALATE_TO_CMO
                target_id = self._get_patient_idx(tool.patient_id, state) if tool.patient_id else 0
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=tool.urgency, reasoning=tool.summary)

            elif tool_name == "DischargePatientTool":
                tool = DischargePatientTool(**tool_kwargs)
                action_type = ActionType.DISCHARGE_PATIENT
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "FlagPolicyViolationTool":
                tool = FlagPolicyViolationTool(**tool_kwargs)
                action_type = ActionType.FLAG_POLICY_VIOLATION
                target_id = self._get_patient_idx(tool.affected_patient_id, state) if tool.affected_patient_id else 0
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.description)

            elif tool_name == "UpdateEHRTool":
                tool = UpdateEHRTool(**tool_kwargs)
                action_type = ActionType.UPDATE_EHR
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=f"Update EHR: {tool.entry}")

            elif tool_name == "VerifyInsuranceTool":
                tool = VerifyInsuranceTool(**tool_kwargs)
                action_type = ActionType.VERIFY_INSURANCE
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=f"Verify insurance {tool.provider}")

            elif tool_name == "AllocateEquipmentTool":
                tool = AllocateEquipmentTool(**tool_kwargs)
                action_type = ActionType.ALLOCATE_EQUIPMENT
                target_id = self._get_patient_idx(tool.patient_id, state) if tool.patient_id else 0
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "SendMessageTool":
                tool = SendMessageTool(**tool_kwargs)
                action_type = ActionType.SEND_MESSAGE
                return ValidatedAction(action_type, tool_kwargs, target_id=0, priority=tool.urgency, reasoning=tool.content)

            elif tool_name == "OverrideDecisionTool":
                tool = OverrideDecisionTool(**tool_kwargs)
                action_type = ActionType.OVERRIDE_DECISION
                return ValidatedAction(action_type, tool_kwargs, target_id=0, priority=8, reasoning=tool.reasoning)

            elif tool_name == "RequestSpecialistTool":
                tool = RequestSpecialistTool(**tool_kwargs)
                action_type = ActionType.REQUEST_SPECIALIST
                target_id = self._get_patient_idx(tool.patient_id, state) if tool.patient_id else 0
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=tool.urgency, reasoning=f"Request {tool.specialty} specialist")

            elif tool_name == "ActivateOverflowTool":
                tool = ActivateOverflowTool(**tool_kwargs)
                action_type = ActionType.ACTIVATE_OVERFLOW
                return ValidatedAction(action_type, tool_kwargs, target_id=0, priority=8, reasoning=tool.justification)

            elif tool_name == "UpdateTreatmentPlanTool":
                tool = UpdateTreatmentPlanTool(**tool_kwargs)
                action_type = ActionType.UPDATE_TREATMENT_PLAN
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.reasoning)

            elif tool_name == "CloseCaseTool":
                tool = CloseCaseTool(**tool_kwargs)
                action_type = ActionType.CLOSE_CASE
                target_id = self._get_patient_idx(tool.patient_id, state)
                if target_id == -1:
                    return ValidationError(f"Patient ID {tool.patient_id} not found.")
                return ValidatedAction(action_type, tool_kwargs, target_id, priority=5, reasoning=tool.resolution_summary)

            else:
                return ValidationError(f"Unknown tool name: {tool_name}")

        except Exception as e:
            return ValidationError(f"Tool validation failed: {str(e)}")

    def _get_patient_idx(self, patient_id: str, state: EnvironmentState) -> int:
        for i, p in enumerate(state.patients):
            if p.id == patient_id:
                return i
        return -1
