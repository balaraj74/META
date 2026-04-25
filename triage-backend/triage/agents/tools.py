from pydantic import BaseModel, Field
from typing import Optional, List

class TriagePatientTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient being triaged")
    triage_score: int = Field(..., description="Triage score from 1 to 10")
    reasoning: str = Field(..., description="Justification for the triage score")

class TransferToICUTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient to transfer to ICU")
    priority: int = Field(..., description="Priority of the transfer (1-10)")
    reasoning: str = Field(..., description="Medical justification for ICU transfer")

class TransferToWardTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient to transfer")
    ward: str = Field(..., description="Name of the ward to transfer to")
    reasoning: str = Field(..., description="Reason for ward transfer")

class AssignTreatmentTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    treatment_plan: str = Field(..., description="Description of the core treatment plan")
    reasoning: str = Field(..., description="Medical reasoning")

class OrderMedicationTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    drug_name: str = Field(..., description="Name of the medication to order")
    dose_mg: float = Field(..., description="Dose in milligrams")
    reasoning: str = Field(..., description="Clinical indication")

class RequestBloodTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    blood_type: str = Field(..., description="Type of blood required")
    units: int = Field(..., description="Number of units needed")
    reasoning: str = Field(..., description="Reason for blood request")

class ActivateProtocolTool(BaseModel):
    protocol_name: str = Field(..., description="Name of the emergency protocol")
    justification: str = Field(..., description="Justification for activating protocol")

class RequestStaffTool(BaseModel):
    role: str = Field(..., description="Role of staff requested")
    count: int = Field(..., description="Number of staff members")
    urgency: int = Field(..., description="Urgency from 1 to 10")
    reasoning: str = Field(..., description="Reason for request")

class EscalateToCMOTool(BaseModel):
    patient_id: Optional[str] = Field(None, description="Patient ID if patient-specific")
    urgency: int = Field(..., description="Urgency of escalation (1-10)")
    summary: str = Field(..., description="Summary of the issue")

class DischargePatientTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient to discharge")
    destination: str = Field(..., description="Discharge destination")
    reasoning: str = Field(..., description="Medical justification")

class FlagPolicyViolationTool(BaseModel):
    violation_type: str = Field(..., description="Type of policy violated")
    description: str = Field(..., description="Details of the violation")
    affected_patient_id: Optional[str] = Field(None, description="Affected patient ID")

class UpdateEHRTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    entry: str = Field(..., description="Content to add to the EHR")

class VerifyInsuranceTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    provider: str = Field(..., description="Insurance provider name")

class AllocateEquipmentTool(BaseModel):
    equipment_type: str = Field(..., description="Type of equipment to allocate")
    patient_id: Optional[str] = Field(None, description="Patient ID if allocated to a patient")
    reasoning: str = Field(..., description="Reasoning for allocation")

class SendMessageTool(BaseModel):
    to_agent: str = Field(..., description="Agent to send message to")
    content: str = Field(..., description="Message content")
    urgency: int = Field(..., description="Urgency (1-10)")

class OverrideDecisionTool(BaseModel):
    original_action_id: str = Field(..., description="ID of the action being overridden")
    new_decision: str = Field(..., description="The overriding decision")
    reasoning: str = Field(..., description="Justification")

class RequestSpecialistTool(BaseModel):
    specialty: str = Field(..., description="Medical specialty requested")
    patient_id: Optional[str] = Field(None, description="Patient ID if patient-specific")
    urgency: int = Field(..., description="Urgency (1-10)")

class ActivateOverflowTool(BaseModel):
    ward: str = Field(..., description="Ward to activate overflow for")
    capacity_increase: int = Field(..., description="Additional capacity required")
    justification: str = Field(..., description="Justification")

class UpdateTreatmentPlanTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    modifications: str = Field(..., description="Changes to the treatment plan")
    reasoning: str = Field(..., description="Medical reasoning")

class CloseCaseTool(BaseModel):
    patient_id: str = Field(..., description="ID of the patient")
    resolution_summary: str = Field(..., description="Summary of case resolution")

from triage.env.state import AgentType

AGENT_TOOLS = {
    AgentType.CMO_OVERSIGHT: [EscalateToCMOTool, FlagPolicyViolationTool, OverrideDecisionTool, SendMessageTool, UpdateEHRTool],
    AgentType.ER_TRIAGE: [TriagePatientTool, TransferToICUTool, TransferToWardTool, RequestSpecialistTool, SendMessageTool, UpdateEHRTool],
    AgentType.INFECTION_CONTROL: [SendMessageTool, EscalateToCMOTool, FlagPolicyViolationTool, AssignTreatmentTool, ActivateProtocolTool],
    AgentType.ICU_MANAGEMENT: [TransferToICUTool, AssignTreatmentTool, RequestBloodTool, TransferToWardTool, SendMessageTool, UpdateEHRTool],
    AgentType.PHARMACY: [OrderMedicationTool, FlagPolicyViolationTool, SendMessageTool, UpdateEHRTool],
    AgentType.HR_ROSTERING: [RequestStaffTool, SendMessageTool, FlagPolicyViolationTool, UpdateEHRTool],
    AgentType.IT_SYSTEMS: [AllocateEquipmentTool, UpdateEHRTool, SendMessageTool, FlagPolicyViolationTool],
}
