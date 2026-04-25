"""Episode orchestrator built on top of the current async env and agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from triage.agents.message_bus import MessageBus
from triage.agents.specialized import create_all_agents
from triage.env.hospital_env import HospitalEnv
from triage.env.state import (
    ActionType,
    AgentAction,
    AgentMessage,
    AgentType,
    CrisisType,
    EnvironmentState,
    MessageType,
)
from triage.rewards.reward_model import RewardModel
from triage.safety.constitution import SafetyConstitution


@dataclass
class OrchestratorStepResult:
    """Single step execution result."""

    step: int
    action: dict[str, Any]
    reward: float
    breakdown: dict[str, Any]
    terminated: bool
    drift_events: list[dict[str, Any]] = field(default_factory=list)
    action_result: dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """Coordinates the current env, message bus, and agent set."""

    def __init__(
        self,
        env: HospitalEnv | None = None,
        agents_config_path: str | None = None,
        mock_llm: bool = True,
        token_budget: int = 50_000,
        seed: int | None = None,
        max_steps: int = 200,
        difficulty: float = 0.5,
    ) -> None:
        self.env = env or HospitalEnv(seed=seed, max_steps=max_steps, difficulty=difficulty)
        self.bus = MessageBus(token_budget=token_budget)
        self.reward_model = RewardModel()
        self.constitution = SafetyConstitution()
        self.mock_llm = mock_llm
        config_path = Path(agents_config_path or "./config/agents.yaml")
        with open(config_path, encoding="utf-8") as handle:
            self.agent_configs = yaml.safe_load(handle)
        self.agents = create_all_agents(self.agent_configs, self.bus, mock_llm)
        self._step_history: list[dict[str, Any]] = []
        self._total_reward = 0.0

    @property
    def state(self) -> EnvironmentState:
        return self.env.state

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def step_history(self) -> list[dict[str, Any]]:
        return list(self._step_history)

    async def reset(self, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
        for agent in self.agents.values():
            agent.reset()
        self.bus.reset()
        self._step_history = []
        self._total_reward = 0.0
        return await self.env.reset(scenario)

    async def step(self, action: dict[str, Any] | None = None) -> OrchestratorStepResult:
        state_before = self.state
        all_actions: list[AgentAction] = []

        if action is None:
            for agent_type in AgentType:
                agent = self.agents.get(agent_type)
                if agent is None:
                    continue
                raw_actions = await agent.act(state_before)
                safe_actions = self.constitution.validate(
                    raw_actions, agent_type, state_before, state_before.step_count
                )
                all_actions.extend(safe_actions)

            if all_actions:
                all_actions.sort(key=lambda item: item.priority, reverse=True)
                primary = await self._select_executable_action(state_before, all_actions)
                if primary is None:
                    primary = all_actions[0]
                action = primary.to_env_action()
                action["reasoning"] = primary.reasoning
                action["reasoning_tokens"] = primary.reasoning_tokens
            else:
                action = self.env.action_space.sample()

        _, env_reward, terminated, info = await self.env.step(action)
        await self.bus.tick()
        state_after = self.state
        breakdown = self.reward_model.compute(
            state_after,
            all_actions,
            info.get("drift_events", []),
            action_result=info.get("action_result", {}),
            messages=self.bus.history,
            app_audits=state_after.app_audit_log,
        )
        self._total_reward += breakdown.total

        result = OrchestratorStepResult(
            step=state_after.step_count,
            action=action,
            reward=float(env_reward),
            breakdown=breakdown.to_dict(),
            terminated=terminated,
            drift_events=info.get("drift_events", []),
            action_result=info.get("action_result", {}),
        )
        self._step_history.append(
            {
                "step": result.step,
                "action": result.action,
                "reward": result.reward,
                "breakdown": result.breakdown,
                "terminated": result.terminated,
                "drift_events": result.drift_events,
                "action_result": result.action_result,
            }
        )
        return result

    async def run(
        self,
        delay_ms: int = 0,
        max_steps: int | None = None,
    ) -> list[dict[str, Any]]:
        step_cap = max_steps or self.env.max_steps
        while not self.env.is_terminal and self.state.step_count < step_cap:
            result = await self.step()
            if result.terminated:
                break
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
        return self.step_history

    async def manual_override(
        self,
        agent_type: AgentType,
        action_type: str,
        target_id: int,
        priority: int,
        reasoning: str,
        reasoning_tokens: int,
    ) -> OrchestratorStepResult:
        action_enum = ActionType[action_type]
        action = AgentAction(
            agent_type=agent_type,
            action_type=action_enum,
            target_id=target_id,
            priority=priority,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
        ).to_env_action()
        action["reasoning"] = reasoning
        action["reasoning_tokens"] = reasoning_tokens
        return await self.step(action)

    def get_agent_stats(self) -> list[dict[str, Any]]:
        return [agent.get_stats() for agent in self.agents.values()]

    def get_agent_messages(self, agent_type: AgentType, limit: int = 50) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.bus.get_messages_for(agent_type, limit=limit)]

    def build_state_snapshot(self) -> dict[str, Any]:
        state = self.state
        return {
            "episode_num": state.episode,
            "status": "completed" if self.env.is_terminal else "running",
            "step": state.step_count,
            "max_steps": self.env.max_steps,
            "crisis_type": state.crisis.type.value,
            "difficulty": self.env.difficulty,
            "patients": [patient.to_dict() for patient in state.patients],
            "resources": state.resources.to_dict(),
            "agents": self.get_agent_stats(),
            "metrics": {
                "survival_rate": state.survival_rate,
                "deceased_count": state.deceased_count,
                "discharged_count": state.discharged_count,
                "critical_count": state.critical_count,
                "alive_count": state.alive_count,
                "icu_occupancy": state.icu_occupancy,
                "total_reward": self._total_reward,
                "violations_caught": state.violations_caught,
                "violations_injected": state.violations_injected,
                "compliance_rate": (
                    state.violations_caught / max(state.violations_injected, 1)
                    if state.violations_injected
                    else 1.0
                ),
            },
            "recent_actions": self._step_history[-10:],
            "drift_events": getattr(self.env, "_drift_events", []),
        }

    async def _select_executable_action(
        self,
        state: EnvironmentState,
        actions: list[AgentAction],
    ) -> AgentAction | None:
        for action in actions:
            prepared = await self._prepare_action(state, action)
            if prepared:
                return action
        return None

    async def _prepare_action(
        self,
        state: EnvironmentState,
        action: AgentAction,
    ) -> bool:
        patient = None
        if 0 <= action.target_id < len(state.patients):
            patient = state.patients[action.target_id]

        if action.action_type == ActionType.TRANSFER_TO_ICU and patient is not None:
            if action.agent_type != AgentType.ICU_MANAGEMENT:
                await self._dispatch_message(
                    AgentMessage(
                        from_agent=action.agent_type,
                        to_agent=AgentType.ICU_MANAGEMENT,
                        content=f"Request ICU allocation for {patient.name}",
                        msg_type=MessageType.REQUEST,
                        priority=action.priority,
                        patient_id=patient.id,
                        request_type="icu_bed_request",
                        payload={"patient_id": patient.id, "reason": action.reasoning},
                        correlation_id=action.id,
                    )
                )
                if state.icu_occupancy >= 0.9:
                    await self._dispatch_message(
                        AgentMessage(
                            from_agent=action.agent_type,
                            to_agent=AgentType.CMO_OVERSIGHT,
                            content=f"ICU allocation for {patient.name} may need override",
                            msg_type=MessageType.ALERT,
                            priority=max(action.priority, 8),
                            patient_id=patient.id,
                            request_type="override_request",
                            payload={"scope": "icu_override", "patient_id": patient.id, "reason": action.reasoning},
                            correlation_id=action.id,
                        )
                    )
                return False
            await self.env.execute_tool(
                "query_icu_capacity",
                {"patient_id": patient.id},
                action.agent_type,
            )
            return True

        if action.action_type == ActionType.ORDER_MEDICATION and patient is not None:
            if action.agent_type != AgentType.PHARMACY:
                await self._dispatch_message(
                    AgentMessage(
                        from_agent=action.agent_type,
                        to_agent=AgentType.PHARMACY,
                        content=f"Medication review requested for {patient.name}",
                        msg_type=MessageType.REQUEST,
                        priority=action.priority,
                        patient_id=patient.id,
                        request_type="medication_request",
                        payload={"patient_id": patient.id, "reason": action.reasoning},
                        correlation_id=action.id,
                    )
                )
                return False
            medication = self.env._select_medication(patient)
            await self.env.execute_tool("lookup_patient", {"patient_id": patient.id}, action.agent_type)
            if patient.status.value != "CRITICAL":
                await self.env.execute_tool("verify_insurance", {"patient_id": patient.id}, action.agent_type)
            await self.env.execute_tool(
                "check_interactions",
                {"patient_id": patient.id, "medication": medication},
                action.agent_type,
            )
            return True

        if action.action_type == ActionType.OVERRIDE_DECISION and action.agent_type != AgentType.CMO_OVERSIGHT:
            if patient is not None:
                await self._dispatch_message(
                    AgentMessage(
                        from_agent=action.agent_type,
                        to_agent=AgentType.CMO_OVERSIGHT,
                        content=f"Override requested for {patient.name}",
                        msg_type=MessageType.ALERT,
                        priority=action.priority,
                        patient_id=patient.id,
                        request_type="override_request",
                        payload={"patient_id": patient.id, "reason": action.reasoning},
                        correlation_id=action.id,
                    )
                )
            return False

        return True

    async def _dispatch_message(self, message: AgentMessage) -> None:
        await self.bus.send(message)
        await self.env.send_message(message)

    @staticmethod
    def normalize_crisis_type(value: str | None) -> CrisisType | None:
        if value is None:
            return None
        return CrisisType(value)
