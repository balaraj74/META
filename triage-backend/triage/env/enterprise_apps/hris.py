"""HRIS workforce simulator with fatigue and staffing workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from triage.env.state import AgentType, AppAuditEvent, EnvironmentState


class HRISSystem:
    """Staff roster and fatigue workflow simulator."""

    def __init__(self) -> None:
        self._fatigue_alerts: list[dict[str, Any]] = []
        self._callback_requests: list[dict[str, Any]] = []

    def get_roster(
        self,
        state: EnvironmentState,
        requester: AgentType = AgentType.HR_ROSTERING,
    ) -> dict[str, Any]:
        roster = dict(state.crisis.staff_roster)
        return self._audit(
            state,
            requester,
            "get_roster",
            "approved",
            "HRIS roster retrieved",
            details={
                "roster": roster,
                "total_staff": sum(roster.values()),
                "staff_ratio": round(state.resources.staff_ratio, 3),
                "fatigue_alerts": list(self._fatigue_alerts[-5:]),
                "callback_requests": list(self._callback_requests[-5:]),
            },
        )

    def check_staff_fatigue(
        self,
        staff_role: str,
        hours_worked: float,
        state: EnvironmentState,
        requester: AgentType = AgentType.HR_ROSTERING,
    ) -> dict[str, Any]:
        is_fatigued = hours_worked >= 16
        needs_break = hours_worked >= 6
        recommendation = "continue_monitoring"
        if is_fatigued:
            recommendation = "relieve_immediately"
            self._fatigue_alerts.append(
                {
                    "role": staff_role,
                    "hours_worked": hours_worked,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        elif needs_break:
            recommendation = "schedule_break"
        return self._audit(
            state,
            requester,
            "check_staff_fatigue",
            "approved",
            "Staff fatigue evaluated",
            details={
                "staff_role": staff_role,
                "hours_worked": hours_worked,
                "is_fatigued": is_fatigued,
                "needs_break": needs_break,
                "recommendation": recommendation,
            },
        )

    def request_additional_staff(
        self,
        role: str,
        count: int,
        state: EnvironmentState,
        requester: AgentType = AgentType.HR_ROSTERING,
    ) -> dict[str, Any]:
        roster = state.crisis.staff_roster
        current = int(roster.get(role, 0))
        fulfillment_ratio = min(1.0, max(0.35, 1.0 - state.resources.staff_ratio / 2))
        fulfilled = max(1, min(count, int(round(count * fulfillment_ratio))))
        roster[role] = current + fulfilled
        state.resources.staff_ratio = min(1.2, state.resources.staff_ratio + 0.04 * fulfilled)
        callback = {
            "role": role,
            "requested": count,
            "fulfilled": fulfilled,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._callback_requests.append(callback)
        return self._audit(
            state,
            requester,
            "request_staff",
            "approved",
            "Staff callback processed",
            details={
                "role": role,
                "requested": count,
                "fulfilled": fulfilled,
                "new_total": roster[role],
                "staff_ratio": round(state.resources.staff_ratio, 3),
            },
        )

    def _audit(
        self,
        state: EnvironmentState,
        requester: AgentType,
        tool_name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {"status": status, "message": message, "details": details or {}}
        state.add_app_audit(
            AppAuditEvent(
                app="hris",
                tool_name=tool_name,
                requester=requester,
                status=status,
                message=message,
                details=details or {},
            )
        )
        return payload


class HRIS(HRISSystem):
    """Backward-compatible alias."""
