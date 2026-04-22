"""Schema-drift adaptation reward component."""

from __future__ import annotations

from triage.env.state import ActionType, EnvironmentState


class AdaptationReward:
    """Measures responsiveness to policy and schema drift events."""

    def compute(self, state: EnvironmentState, drift_events: list[dict]) -> float:
        if not drift_events:
            return 0.5

        drift_related_actions = [
            action
            for action in state.action_history
            if action.action_type in (
                ActionType.FLAG_POLICY_VIOLATION,
                ActionType.UPDATE_EHR,
                ActionType.ACTIVATE_PROTOCOL,
                ActionType.VERIFY_INSURANCE,
            )
        ]
        drift_messages = [
            message
            for message in state.message_history
            if any(
                keyword in message.content.lower()
                for keyword in ("policy", "protocol", "contract", "schema", "regulatory", "compliance")
            )
        ]
        domain_coverage = len({event.get("type", "unknown") for event in drift_events})
        score = min(
            (len(drift_related_actions) + len(drift_messages) + domain_coverage)
            / max(len(drift_events) * 2, 1),
            1.0,
        )
        return max(-1.0, min(1.0, score))
