"""IT tracker simulator for equipment, uptime, and drift-linked incidents."""

from __future__ import annotations

from typing import Any

from triage.env.state import AgentType, AppAuditEvent, EnvironmentState


class ITTrackerSystem:
    """Tracks equipment allocations and operational incidents."""

    def __init__(self) -> None:
        self._allocations: dict[str, str] = {}
        self._incident_ids: set[str] = set()
        self._incidents: list[dict[str, Any]] = []

    def get_status(
        self,
        state: EnvironmentState,
        requester: AgentType = AgentType.IT_SYSTEMS,
    ) -> dict[str, Any]:
        self._sync_incidents(state)
        return self._audit(
            state,
            requester,
            "get_equipment_status",
            None,
            "approved",
            "IT systems status retrieved",
            details={
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
                "open_incidents": list(self._incidents[-5:]),
            },
        )

    def allocate_ventilator(
        self,
        patient_id: str,
        state: EnvironmentState,
        requester: AgentType = AgentType.ICU_MANAGEMENT,
    ) -> dict[str, Any]:
        available = state.resources.ventilators_total - state.resources.ventilators_in_use
        if available <= 0:
            return self._audit(
                state,
                requester,
                "allocate_ventilator",
                patient_id,
                "blocked",
                "No ventilators available",
                details={"suggestion": "Use non-invasive support or transfer"},
            )
        state.resources.ventilators_in_use += 1
        equipment_id = f"VENT-{state.resources.ventilators_in_use:03d}"
        self._allocations[equipment_id] = patient_id
        return self._audit(
            state,
            requester,
            "allocate_ventilator",
            patient_id,
            "approved",
            f"Allocated ventilator {equipment_id}",
            details={
                "equipment_id": equipment_id,
                "remaining_ventilators": available - 1,
            },
        )

    def release_equipment(
        self,
        equipment_id: str,
        state: EnvironmentState,
        requester: AgentType = AgentType.IT_SYSTEMS,
    ) -> dict[str, Any]:
        patient_id = self._allocations.pop(equipment_id, None)
        if patient_id is None:
            return self._audit(
                state,
                requester,
                "release_equipment",
                None,
                "blocked",
                f"Equipment {equipment_id} not found",
            )
        if equipment_id.startswith("VENT"):
            state.resources.ventilators_in_use = max(0, state.resources.ventilators_in_use - 1)
        return self._audit(
            state,
            requester,
            "release_equipment",
            patient_id,
            "approved",
            f"Released equipment {equipment_id}",
            details={"equipment_id": equipment_id},
        )

    def _sync_incidents(self, state: EnvironmentState) -> None:
        for event in state.drift_history[-5:]:
            event_id = f"{event.get('type')}:{event.get('step', 'na')}"
            if event_id in self._incident_ids:
                continue
            if event.get("type") not in {"contract_drift", "regulatory_drift", "resource_shock"}:
                continue
            self._incident_ids.add(event_id)
            self._incidents.append(
                {
                    "id": event_id,
                    "severity": "high" if event.get("type") == "resource_shock" else "medium",
                    "summary": event.get("message", "Drift event detected"),
                }
            )

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
                app="it_tracker",
                tool_name=tool_name,
                requester=requester,
                patient_id=patient_id,
                status=status,
                message=message,
                details=details or {},
            )
        )
        return payload


class ITSystems(ITTrackerSystem):
    """Backward-compatible alias."""
