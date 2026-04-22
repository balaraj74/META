"""Insurance portal simulator with contract/schema drift awareness."""

from __future__ import annotations

import uuid
from typing import Any

from triage.env.state import AgentType, AppAuditEvent, EnvironmentState, Patient


class InsurancePortalSystem:
    """Coverage verification and authorization workflow simulator."""

    _PLAN_DB = {
        "PPO_GOLD": {"coverage_percent": 0.90, "icu_covered": True, "pharmacy_covered": True},
        "HMO_BASIC": {"coverage_percent": 0.70, "icu_covered": True, "pharmacy_covered": False},
        "MEDICAID": {"coverage_percent": 0.95, "icu_covered": True, "pharmacy_covered": True},
        "UNINSURED": {"coverage_percent": 0.00, "icu_covered": False, "pharmacy_covered": False},
        "EMERGENCY_ONLY": {"coverage_percent": 0.50, "icu_covered": True, "pharmacy_covered": False},
    }

    def verify_patient(
        self,
        patient_id: str,
        state: EnvironmentState,
        requester: AgentType = AgentType.IT_SYSTEMS,
    ) -> dict[str, Any]:
        patient = self._find_patient(patient_id, state)
        if patient is None:
            return self._audit(
                state,
                requester,
                "verify_insurance",
                patient_id,
                "rejected_unknown_tool",
                f"Patient {patient_id} not found",
            )

        plan_name = patient.insurance_plan or self._deterministic_plan(patient_id)
        contract = state.contract_constraints.get("insurance_portal", {})
        coverage_field = contract.get("coverage_field", "coverage_percent")
        member_id_field = contract.get("member_id_field", "member_id")
        requires_portal_ref = contract.get("requires_portal_reference", False)

        patient.insurance_verified = True
        patient.insurance_plan = plan_name
        patient.add_event("INSURANCE", f"Verified: {plan_name}", requester)

        plan = self._PLAN_DB[plan_name]
        details = {
            "plan": plan_name,
            "schema_version": contract.get("schema_version", "v1"),
            coverage_field: plan["coverage_percent"],
            member_id_field: f"MBR-{patient.id[-4:].upper()}",
            "icu_covered": plan["icu_covered"],
            "pharmacy_covered": plan["pharmacy_covered"],
            "authorization_mode": contract.get("authorization_mode", "waived_for_emergency"),
            "requires_portal_reference": requires_portal_ref,
            "authorization_number": str(uuid.uuid4())[:12].upper(),
        }
        if requires_portal_ref:
            details["portal_reference"] = f"PORTAL-{patient.id[-6:].upper()}"
        return self._audit(
            state,
            requester,
            "verify_insurance",
            patient_id,
            "approved",
            "Insurance verified",
            details=details,
        )

    def check_authorization(
        self,
        patient_id: str,
        procedure: str,
        state: EnvironmentState,
        requester: AgentType = AgentType.IT_SYSTEMS,
    ) -> dict[str, Any]:
        patient = self._find_patient(patient_id, state)
        if patient is None:
            return self._audit(
                state,
                requester,
                "check_authorization",
                patient_id,
                "rejected_unknown_tool",
                f"Patient {patient_id} not found",
            )

        contract = state.contract_constraints.get("insurance_portal", {})
        emergency = procedure.lower() in {"emergency", "icu_transfer", "life_saving"}
        auth_mode = contract.get("authorization_mode", "waived_for_emergency")
        pre_authorized = emergency or auth_mode == "waived_for_emergency"
        return self._audit(
            state,
            requester,
            "check_authorization",
            patient_id,
            "approved" if pre_authorized else "needs_override",
            "Authorization evaluated",
            details={
                "procedure": procedure,
                "pre_authorized": pre_authorized,
                "authorization_mode": auth_mode,
                "schema_version": contract.get("schema_version", "v1"),
                "patient_plan": patient.insurance_plan,
            },
        )

    def _deterministic_plan(self, patient_id: str) -> str:
        plans = list(self._PLAN_DB.keys())
        return plans[sum(ord(char) for char in patient_id) % len(plans)]

    def _find_patient(self, patient_id: str, state: EnvironmentState) -> Patient | None:
        for patient in state.patients:
            if patient.id == patient_id:
                return patient
        return None

    def _audit(
        self,
        state: EnvironmentState,
        requester: AgentType,
        tool_name: str,
        patient_id: str | None,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "message": message,
            "patient_id": patient_id,
            "details": details or {},
        }
        state.add_app_audit(
            AppAuditEvent(
                app="insurance_portal",
                tool_name=tool_name,
                requester=requester,
                patient_id=patient_id,
                status=status,
                message=message,
                details=details or {},
            )
        )
        return payload


class Insurance(InsurancePortalSystem):
    """Backward-compatible alias."""
