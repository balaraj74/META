"""
Enterprise app registry and supporting hospital systems.

Pharmacy and ICU logic live in dedicated submodules so they can own the
workflow complexity required by the hackathon demo.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any

from triage.env.enterprise_apps.hris import HRISSystem
from triage.env.enterprise_apps.icu_manager import ICUManagerSystem
from triage.env.enterprise_apps.insurance import InsurancePortalSystem
from triage.env.enterprise_apps.it_systems import ITTrackerSystem
from triage.env.enterprise_apps.pharmacy import PharmacySystem
from triage.env.state import AgentType, EnvironmentState, Patient, PatientStatus, WardType


class EHRSystem:
    """Electronic Health Record simulator."""

    def __init__(self) -> None:
        self._access_log: list[dict[str, Any]] = []

    def lookup_patient(self, patient_id: str, state: EnvironmentState, requester: AgentType) -> dict[str, Any]:
        self._log_access(patient_id, requester, "lookup")
        patient = self._find_patient(patient_id, state)
        if not patient:
            return {"status": "rejected_unknown_tool", "message": f"Patient {patient_id} not found in EHR"}
        return {
            "status": "approved",
            "patient_id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "condition": patient.condition,
            "status_label": patient.status.value,
            "ward": patient.ward.value,
            "triage_score": patient.triage_score,
            "treatment_plan": patient.treatment_plan,
            "medications": patient.medications,
            "allergies": patient.allergies,
            "insurance_verified": patient.insurance_verified,
            "insurance_plan": patient.insurance_plan,
            "history_count": len(patient.history),
            "admitted_at": patient.admitted_at.isoformat(),
        }

    def update_record(
        self,
        patient_id: str,
        updates: dict[str, Any],
        state: EnvironmentState,
        requester: AgentType,
    ) -> dict[str, Any]:
        self._log_access(patient_id, requester, "update")
        patient = self._find_patient(patient_id, state)
        if not patient:
            return {"status": "rejected_unknown_tool", "message": f"Patient {patient_id} not found"}

        applied: list[str] = []
        if "status" in updates:
            try:
                patient.status = PatientStatus(updates["status"])
                applied.append("status")
            except ValueError:
                pass
        if "ward" in updates:
            try:
                patient.ward = WardType(updates["ward"])
                applied.append("ward")
            except ValueError:
                pass
        if "triage_score" in updates:
            patient.triage_score = int(updates["triage_score"])
            applied.append("triage_score")
        if "treatment_plan" in updates:
            patient.treatment_plan = updates["treatment_plan"]
            applied.append("treatment_plan")
        if "medications" in updates:
            patient.medications = updates["medications"]
            applied.append("medications")
        if "insurance_verified" in updates:
            patient.insurance_verified = bool(updates["insurance_verified"])
            applied.append("insurance_verified")
        if "insurance_plan" in updates:
            patient.insurance_plan = updates["insurance_plan"]
            applied.append("insurance_plan")
        if "icu_required" in updates:
            patient.icu_required = bool(updates["icu_required"])
            applied.append("icu_required")
        if "allergies" in updates:
            patient.allergies = list(updates["allergies"])
            applied.append("allergies")

        patient.add_event("EHR_UPDATE", f"Fields updated: {', '.join(applied)}", requester)
        return {"status": "approved", "success": True, "fields_updated": applied}

    def list_patients(self, state: EnvironmentState, ward: str | None = None) -> list[dict[str, Any]]:
        patients = state.patients
        if ward:
            patients = [patient for patient in patients if patient.ward.value == ward]
        return [
            {
                "id": patient.id,
                "name": patient.name,
                "status": patient.status.value,
                "ward": patient.ward.value,
                "triage_score": patient.triage_score,
                "condition": patient.condition,
            }
            for patient in patients
        ]

    def _find_patient(self, patient_id: str, state: EnvironmentState) -> Patient | None:
        for patient in state.patients:
            if patient.id == patient_id:
                return patient
        return None

    def _log_access(self, patient_id: str, requester: AgentType, action: str) -> None:
        self._access_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patient_id": patient_id,
                "requester": requester.value,
                "action": action,
            }
        )


class SchedulingSystem:
    """Staff scheduling and fatigue tracking."""

    def __init__(self) -> None:
        self._fatigue_alerts: list[dict[str, Any]] = []

    def get_roster(self, state: EnvironmentState) -> dict[str, Any]:
        roster = state.crisis.staff_roster
        return {
            "status": "approved",
            "roster": dict(roster),
            "total_staff": sum(roster.values()),
            "staff_reduction": state.crisis.staff_reduction,
            "fatigue_alerts": list(self._fatigue_alerts),
        }

    def check_staff_fatigue(self, staff_role: str, hours_worked: float) -> dict[str, Any]:
        is_fatigued = hours_worked > 16
        needs_break = hours_worked > 6
        result = {
            "status": "approved",
            "staff_role": staff_role,
            "hours_worked": hours_worked,
            "is_fatigued": is_fatigued,
            "needs_break": needs_break,
            "recommendation": "",
        }
        if is_fatigued:
            result["recommendation"] = "CRITICAL: Staff member must be relieved immediately (POL-004)"
            self._fatigue_alerts.append(
                {
                    "role": staff_role,
                    "hours": hours_worked,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        elif needs_break:
            result["recommendation"] = "Schedule 30-minute break at next opportunity"
        return result

    def request_additional_staff(self, role: str, count: int, state: EnvironmentState) -> dict[str, Any]:
        roster = state.crisis.staff_roster
        current = roster.get(role, 0)
        available = random.randint(0, count)
        roster[role] = current + available
        return {
            "status": "approved",
            "role": role,
            "requested": count,
            "fulfilled": available,
            "new_total": roster[role],
            "note": "Full callback typically takes 45-90 minutes" if available < count else "Request fulfilled",
        }


class InsuranceVerifier:
    """Insurance verification simulator."""

    _PLAN_DB = {
        "PPO_GOLD": {"coverage": 0.90, "icu_covered": True, "pharmacy_covered": True},
        "HMO_BASIC": {"coverage": 0.70, "icu_covered": True, "pharmacy_covered": False},
        "MEDICAID": {"coverage": 0.95, "icu_covered": True, "pharmacy_covered": True},
        "UNINSURED": {"coverage": 0.00, "icu_covered": False, "pharmacy_covered": False},
        "EMERGENCY_ONLY": {"coverage": 0.50, "icu_covered": True, "pharmacy_covered": False},
    }

    def verify_patient(self, patient_id: str, state: EnvironmentState) -> dict[str, Any]:
        for patient in state.patients:
            if patient.id == patient_id:
                plan_name = patient.insurance_plan or random.choice(list(self._PLAN_DB.keys()))
                plan = self._PLAN_DB[plan_name]
                patient.insurance_verified = True
                patient.insurance_plan = plan_name
                patient.add_event("INSURANCE", f"Verified: {plan_name}")
                return {
                    "status": "approved",
                    "patient_id": patient_id,
                    "verified": True,
                    "plan": plan_name,
                    "coverage_percent": plan["coverage"],
                    "icu_covered": plan["icu_covered"],
                    "pharmacy_covered": plan["pharmacy_covered"],
                    "authorization_number": str(uuid.uuid4())[:12].upper(),
                }
        return {"status": "rejected_unknown_tool", "message": f"Patient {patient_id} not found"}

    def check_authorization(self, patient_id: str, procedure: str) -> dict[str, Any]:
        return {
            "status": "approved",
            "patient_id": patient_id,
            "procedure": procedure,
            "pre_authorized": True,
            "reason": "Emergency protocol — pre-authorization waived",
        }


class EquipmentTracker:
    """Medical equipment status and allocation."""

    def __init__(self) -> None:
        self._allocations: dict[str, str] = {}

    def get_status(self, state: EnvironmentState) -> dict[str, Any]:
        return {
            "status": "approved",
            "ventilators": {
                "total": state.resources.ventilators_total,
                "in_use": state.resources.ventilators_in_use,
                "available": state.resources.ventilators_total - state.resources.ventilators_in_use,
            },
            "icu_beds": {
                "total": state.resources.icu_beds_total,
                "occupied": state.resources.icu_beds_occupied,
                "available": state.resources.icu_beds_total - state.resources.icu_beds_occupied,
            },
            "equipment_status": state.resources.equipment_status,
            "it_uptime": state.resources.it_uptime,
            "allocations": dict(self._allocations),
        }

    def allocate_ventilator(self, patient_id: str, state: EnvironmentState) -> dict[str, Any]:
        available = state.resources.ventilators_total - state.resources.ventilators_in_use
        if available <= 0:
            return {
                "status": "blocked",
                "success": False,
                "message": "No ventilators available",
                "suggestion": "Consider non-invasive ventilation or patient transfer",
            }
        state.resources.ventilators_in_use += 1
        equipment_id = f"VENT-{state.resources.ventilators_in_use:03d}"
        self._allocations[equipment_id] = patient_id
        return {
            "status": "approved",
            "success": True,
            "equipment_id": equipment_id,
            "patient_id": patient_id,
            "remaining_ventilators": available - 1,
        }

    def release_equipment(self, equipment_id: str, state: EnvironmentState) -> dict[str, Any]:
        if equipment_id not in self._allocations:
            return {"status": "blocked", "message": f"Equipment {equipment_id} not found in allocations"}
        patient_id = self._allocations.pop(equipment_id)
        if equipment_id.startswith("VENT"):
            state.resources.ventilators_in_use = max(0, state.resources.ventilators_in_use - 1)
        return {
            "status": "approved",
            "success": True,
            "equipment_id": equipment_id,
            "released_from_patient": patient_id,
        }


class EnterpriseAppRegistry:
    """Central registry for all enterprise app simulators."""

    def __init__(self) -> None:
        self.ehr = EHRSystem()
        self.pharmacy = PharmacySystem()
        self.hris = HRISSystem()
        self.scheduling = self.hris
        self.insurance = InsurancePortalSystem()
        self.it_tracker = ITTrackerSystem()
        self.equipment = self.it_tracker
        self.icu_manager = ICUManagerSystem()

    def reset(self) -> None:
        self.__init__()

    def execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        state: EnvironmentState,
        requester: AgentType,
    ) -> dict[str, Any]:
        tool_map: dict[str, Any] = {
            "lookup_patient": lambda: self._lookup_patient(params["patient_id"], state, requester),
            "update_record": lambda: self.ehr.update_record(params["patient_id"], params.get("updates", {}), state, requester),
            "list_patients": lambda: self.ehr.list_patients(state, params.get("ward")),
            "check_inventory": lambda: self.pharmacy.check_inventory(state),
            "dispense_medication": lambda: self.pharmacy.dispense_medication(
                params["patient_id"],
                params["medication"],
                params.get("dose", "standard"),
                state,
                requester,
                params.get("double_verified", False),
                params.get("emergency", False),
                params.get("authorization_id"),
            ),
            "check_interactions": lambda: self.pharmacy.check_interactions(
                params["patient_id"],
                params["medication"],
                state,
                requester,
            ),
            "get_roster": lambda: self.hris.get_roster(state, requester),
            "check_staff_fatigue": lambda: self.hris.check_staff_fatigue(
                params["role"],
                params.get("hours_worked", 0),
                state,
                requester,
            ),
            "request_staff": lambda: self.hris.request_additional_staff(
                params["role"],
                params.get("count", 1),
                state,
                requester,
            ),
            "verify_insurance": lambda: self.insurance.verify_patient(params["patient_id"], state, requester),
            "check_authorization": lambda: self.insurance.check_authorization(
                params["patient_id"],
                params.get("procedure", "emergency"),
                state,
                requester,
            ),
            "get_equipment_status": lambda: self.it_tracker.get_status(state, requester),
            "allocate_ventilator": lambda: self.it_tracker.allocate_ventilator(params["patient_id"], state, requester),
            "release_equipment": lambda: self.it_tracker.release_equipment(params["equipment_id"], state, requester),
            "query_icu_capacity": lambda: self.icu_manager.query_capacity(state, requester, params.get("patient_id")),
            "allocate_icu_bed": lambda: self.icu_manager.allocate_bed(
                params["patient_id"],
                state,
                requester,
                params.get("authorization_id"),
            ),
            "release_icu_bed": lambda: self.icu_manager.release_bed(params["bed_id"], state, requester),
        }
        handler = tool_map.get(tool_name)
        if not handler:
            return {
                "status": "rejected_unknown_tool",
                "message": f"Unknown tool: {tool_name}",
                "available_tools": list(tool_map.keys()),
            }
        try:
            return handler()
        except Exception as exc:
            return {"status": "blocked", "message": f"Tool execution failed: {exc!s}", "tool": tool_name}

    def _lookup_patient(
        self,
        patient_id: str,
        state: EnvironmentState,
        requester: AgentType,
    ) -> dict[str, Any]:
        result = self.ehr.lookup_patient(patient_id, state, requester)
        if result.get("status") == "approved":
            self.pharmacy.register_patient_lookup(patient_id, requester, state)
        return result
