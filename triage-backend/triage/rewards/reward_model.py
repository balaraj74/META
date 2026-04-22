"""
RewardModel — prompt-aligned multi-component reward system.

This keeps the existing `triage.rewards` import path as the runtime source of
truth while delegating component calculations to the modular calculators under
`triage.reward.components`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from triage.env.state import AgentAction, EnvironmentState
from triage.reward.components import (
    AdaptationReward,
    ComplianceReward,
    CoordinationReward,
    DepthReward,
    ExpertAlignmentReward,
    OversightReward,
    SurvivalReward,
)


@dataclass
class RewardBreakdown:
    """Detailed reward decomposition for interpretability."""

    survival: float = 0.0
    compliance: float = 0.0
    coordination: float = 0.0
    oversight: float = 0.0
    depth: float = 0.0
    adaptation: float = 0.0
    expert_alignment: float = 0.0
    penalties: float = 0.0
    workflow_bonus: float = 0.0
    terminal_bonus: float = 0.0
    total: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base = {
            "survival": round(self.survival, 4),
            "compliance": round(self.compliance, 4),
            "coordination": round(self.coordination, 4),
            "oversight": round(self.oversight, 4),
            "depth": round(self.depth, 4),
            "adaptation": round(self.adaptation, 4),
            "expert_alignment": round(self.expert_alignment, 4),
            "penalties": round(self.penalties, 4),
            "workflow_bonus": round(self.workflow_bonus, 4),
            "terminal_bonus": round(self.terminal_bonus, 4),
            "total": round(self.total, 4),
            "weights": {k: round(v, 3) for k, v in self.weights.items()},
            "details": self.details,
        }
        # Legacy aliases retained so the existing demo and any old consumers keep working.
        base.update(
            {
                "patient_outcomes": base["survival"],
                "resource_efficiency": base["coordination"],
                "communication_quality": base["coordination"],
                "compliance_adherence": base["compliance"],
                "drift_adaptation": base["adaptation"],
                "token_economy": base["depth"],
            }
        )
        return base


class RewardModel:
    """Production reward model with prompt-aligned component names."""

    # Default weights — can be overridden by expert signals
    DEFAULT_WEIGHTS = {
        "survival": 0.30,
        "compliance": 0.20,
        "coordination": 0.15,
        "oversight": 0.10,
        "depth": 0.10,
        "adaptation": 0.10,
        "expert_alignment": 0.05,
    }

    def __init__(self, custom_weights: dict[str, float] | None = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if custom_weights:
            self.weights.update(custom_weights)
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {key: value / total_weight for key, value in self.weights.items()}

        self.survival_component = SurvivalReward()
        self.compliance_component = ComplianceReward()
        self.coordination_component = CoordinationReward()
        self.oversight_component = OversightReward()
        self.depth_component = DepthReward()
        self.adaptation_component = AdaptationReward()
        self.expert_component = ExpertAlignmentReward()

    def compute(
        self,
        state: EnvironmentState,
        actions: list[AgentAction],
        drift_events: list[dict[str, Any]] | None = None,
        action_result: dict[str, Any] | None = None,
        messages: list[Any] | None = None,
        app_audits: list[Any] | None = None,
    ) -> RewardBreakdown:
        """Compute the full reward breakdown for the current step."""
        effective_weights = self._effective_weights(state)
        breakdown = RewardBreakdown(weights=dict(effective_weights))
        drift_events = drift_events or []
        messages = messages or state.message_history
        app_audits = app_audits or state.app_audit_log
        action_result = action_result or {}

        breakdown.survival = self.survival_component.compute(state)
        breakdown.compliance = self.compliance_component.compute(state)
        breakdown.coordination = self.coordination_component.compute(state)
        breakdown.oversight = self.oversight_component.compute(state)
        breakdown.depth = self.depth_component.compute(actions)
        breakdown.adaptation = self.adaptation_component.compute(state, drift_events)
        breakdown.expert_alignment = self.expert_component.compute(state)
        workflow_penalties, penalty_details = self._workflow_penalties(app_audits, action_result)
        breakdown.penalties = self._penalties(state) + workflow_penalties
        breakdown.workflow_bonus = self._workflow_bonus(messages, app_audits, action_result)
        breakdown.terminal_bonus = self._terminal_bonus(state)

        weighted_base = (
            effective_weights["survival"] * breakdown.survival
            + effective_weights["compliance"] * breakdown.compliance
            + effective_weights["coordination"] * breakdown.coordination
            + effective_weights["oversight"] * breakdown.oversight
            + effective_weights["depth"] * breakdown.depth
            + effective_weights["adaptation"] * breakdown.adaptation
            + effective_weights["expert_alignment"] * breakdown.expert_alignment
        )
        breakdown.total = (
            weighted_base
            + breakdown.penalties
            + breakdown.workflow_bonus
            + breakdown.terminal_bonus
        )
        breakdown.details = {
            "step": state.step_count,
            "alive_count": state.alive_count,
            "critical_count": state.critical_count,
            "violations": {
                "caught": state.violations_caught,
                "injected": state.violations_injected,
            },
            "drift_events": len(drift_events),
            "drift_types": [event.get("type", "unknown") for event in drift_events],
            "expert_profile": {
                "dominant_signal": self._dominant_signal(state),
                "signals": {k: round(v, 3) for k, v in state.expert_signals.items()},
            },
            "workflow": {
                "penalties": penalty_details,
                "recent_audits": [self._audit_to_dict(event) for event in app_audits[-5:]],
                "recent_requests": [
                    self._message_to_dict(message)
                    for message in messages[-5:]
                    if getattr(message, "request_type", None)
                ],
            },
        }
        return breakdown

    def _effective_weights(self, state: EnvironmentState) -> dict[str, float]:
        weights = dict(self.weights)
        signals = state.expert_signals or {}
        cost = signals.get("cost_weight", 0.3)
        quality = signals.get("quality_weight", 0.5)
        speed = signals.get("speed_weight", 0.2)

        weights["survival"] *= 1.0 + quality * 0.65
        weights["compliance"] *= 1.0 + quality * 0.25
        weights["oversight"] *= 1.0 + quality * 0.25
        weights["coordination"] *= 1.0 + speed * 0.55
        weights["adaptation"] *= 1.0 + speed * 0.35
        weights["depth"] *= max(0.35, 1.0 + quality * 0.15 - cost * 0.55)
        weights["expert_alignment"] *= 1.0 + max(cost, quality, speed) * 0.5

        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

    def _dominant_signal(self, state: EnvironmentState) -> str:
        signals = state.expert_signals or {}
        if not signals:
            return "quality"
        dominant = max(signals, key=signals.get)
        return dominant.replace("_weight", "")

    def compute_episode_reward(
        self,
        state: EnvironmentState,
    ) -> dict[str, float]:
        """Compute final episode-level reward summary."""
        survival = state.survival_rate
        compliance = (
            state.violations_caught / max(state.violations_injected, 1)
            if state.violations_injected
            else 1.0
        )
        throughput = state.discharged_count / max(state.total_patients, 1)
        oversight = compliance

        total = (
            survival * 0.40
            + compliance * 0.20
            + throughput * 0.20
            + oversight * 0.20
        )

        return {
            "survival": round(survival, 4),
            "compliance": round(compliance, 4),
            "throughput": round(throughput, 4),
            "oversight": round(oversight, 4),
            "total": round(total, 4),
        }

    def _penalties(self, state: EnvironmentState) -> float:
        idle_agents = sum(1 for agent in state.agent_states.values() if agent.idle_steps > 10)
        untreated_critical = sum(
            1
            for patient in state.patients
            if patient.status.value == "CRITICAL" and not patient.treatment_plan
        )
        return -(idle_agents * 0.02 + untreated_critical * 0.01)

    def _workflow_penalties(
        self,
        app_audits: list[Any],
        action_result: dict[str, Any],
    ) -> tuple[float, dict[str, int]]:
        recent = app_audits[-10:]
        unknown_tool = sum(1 for event in recent if getattr(event, "status", "") == "rejected_unknown_tool")
        missing_precheck = sum(1 for event in recent if getattr(event, "status", "") == "missing_precheck")
        needs_override = sum(1 for event in recent if getattr(event, "status", "") == "needs_override")
        bypass = sum(
            1
            for event in recent
            if getattr(event, "details", {}).get("workflow_violation") == "bypass_chain_of_command"
        )
        failed_action = 1 if action_result and action_result.get("success") is False else 0
        total = -(
            unknown_tool * 0.08
            + bypass * 0.10
            + missing_precheck * 0.06
            + needs_override * 0.04
            + failed_action * 0.03
        )
        return total, {
            "hallucinated_api_calls": unknown_tool,
            "chain_of_command_bypasses": bypass,
            "missing_prechecks": missing_precheck,
            "override_required_blocks": needs_override,
            "failed_actions": failed_action,
        }

    def _workflow_bonus(
        self,
        messages: list[Any],
        app_audits: list[Any],
        action_result: dict[str, Any],
    ) -> float:
        recent_messages = messages[-10:]
        recent_audits = app_audits[-10:]
        delegated = sum(
            1
            for message in recent_messages
            if getattr(message, "request_type", None) in {"icu_bed_request", "medication_request", "override_request"}
        )
        approved_enterprise_actions = sum(
            1
            for event in recent_audits
            if getattr(event, "status", "") == "approved"
            and getattr(event, "tool_name", "") in {"allocate_icu_bed", "dispense_medication", "query_icu_capacity", "check_interactions"}
        )
        override_issued = 1 if action_result.get("authorization_id") else 0
        return min(0.18, delegated * 0.025 + approved_enterprise_actions * 0.02 + override_issued * 0.03)

    def _audit_to_dict(self, event: Any) -> dict[str, Any]:
        if hasattr(event, "to_dict"):
            return event.to_dict()
        return dict(event) if isinstance(event, dict) else {}

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        if hasattr(message, "to_dict"):
            return message.to_dict()
        return dict(message) if isinstance(message, dict) else {}

    def _terminal_bonus(self, state: EnvironmentState) -> float:
        if not state.crisis_resolved:
            return 0.0
        bonus = 0.0
        if state.critical_count == 0:
            bonus += 0.1
        if state.violations_caught >= state.violations_injected:
            bonus += 0.05
        return bonus
