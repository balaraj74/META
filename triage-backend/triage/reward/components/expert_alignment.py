"""Expert-alignment reward component."""

from __future__ import annotations

from triage.env.state import EnvironmentState


class ExpertAlignmentReward:
    """Aligns outcomes with the simulated expert preferences."""

    def compute(self, state: EnvironmentState) -> float:
        signals = state.expert_signals
        cost_weight = signals.get("cost_weight", 0.33)
        quality_weight = signals.get("quality_weight", 0.33)
        speed_weight = signals.get("speed_weight", 0.33)

        untreated_critical = sum(
            1
            for patient in state.patients
            if patient.status.value == "CRITICAL" and not patient.treatment_plan
        )
        critical_pool = max(state.critical_count, 1)
        compliance_actual = (
            state.violations_caught / max(state.violations_injected, 1)
            if state.violations_injected
            else 1.0
        )
        quality_actual = min(
            1.0,
            state.survival_rate * 0.7
            + (1.0 - untreated_critical / critical_pool) * 0.2
            + compliance_actual * 0.1,
        )

        patient_flow = state.discharged_count / max(state.total_patients, 1)
        pending_pressure = len(state.pending_patients) / max(state.crisis.patient_count, 1)
        speed_actual = max(
            0.0,
            min(
                1.0,
                (1.0 - min(state.step_count / 220, 1.0)) * 0.5
                + patient_flow * 0.3
                + (1.0 - pending_pressure) * 0.2,
            ),
        )

        token_usage = sum(agent.token_usage for agent in state.agent_states.values())
        avg_tokens_per_step = token_usage / max(state.step_count, 1)
        cost_actual = max(
            0.0,
            min(
                1.0,
                (1.0 - min(avg_tokens_per_step / 1200, 1.0)) * 0.7
                + (1.0 - min(state.icu_occupancy, 1.0)) * 0.3,
            ),
        )

        score = (
            quality_weight * quality_actual
            + speed_weight * speed_actual
            + cost_weight * cost_actual
        )
        return max(-1.0, min(1.0, score))
