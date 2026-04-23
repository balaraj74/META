"""
HospitalEnv — OpenEnv-compatible hospital crisis simulation.

Provides both the async OpenEnv API (reset/step/state) and a Gymnasium-style
observation/action space shim for RL training.

Episode lifecycle:
  1. reset(scenario?) → generates crisis, initializes state
  2. step(action) → applies agent action, advances world, returns reward
  3. state() → returns current EnvironmentState
  4. Repeats until terminated (crisis resolved or catastrophic failure)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from triage.env.crisis_generator import CrisisGenerator
from triage.env.enterprise_registry import EnterpriseAppRegistry
from triage.env.schema_drift import SchemaDrift
from triage.env.state import (
    ActionType,
    AgentAction,
    AgentMessage,
    AgentState,
    AppAuditEvent,
    AgentType,
    CrisisType,
    EnvironmentState,
    MessageType,
    Patient,
    PatientStatus,
    ResourceState,
    WardType,
)

logger = logging.getLogger(__name__)


# ─── Observation & Action Spaces (Gymnasium shim) ───────────

class ObservationSpace:
    """Defines the shape of the observation returned by the environment."""

    PATIENTS_SHAPE = (50, 12)
    RESOURCES_SHAPE = (8,)
    AGENTS_SHAPE = (6, 8)
    CRISIS_SHAPE = (10,)
    POLICY_SHAPE = (20,)
    EXPERT_SHAPE = (6,)

    @classmethod
    def sample(cls) -> dict[str, np.ndarray]:
        return {
            "patients": np.random.rand(*cls.PATIENTS_SHAPE).astype(np.float32),
            "resources": np.random.rand(*cls.RESOURCES_SHAPE).astype(np.float32),
            "agent_states": np.random.rand(*cls.AGENTS_SHAPE).astype(np.float32),
            "crisis_state": np.random.rand(*cls.CRISIS_SHAPE).astype(np.float32),
            "policy_state": np.random.rand(*cls.POLICY_SHAPE).astype(np.float32),
            "expert_signals": np.random.rand(*cls.EXPERT_SHAPE).astype(np.float32),
        }


class ActionSpace:
    """Defines the discrete action space:
    [agent_id(6), action_type(20), target_id(50), priority(5), reasoning_tokens(1)]
    """

    N_AGENTS = len(AgentType)
    N_ACTIONS = len(ActionType)
    N_TARGETS = 50
    N_PRIORITY = 5

    @classmethod
    def sample(cls) -> dict[str, int]:
        return {
            "agent_id": np.random.randint(0, cls.N_AGENTS),
            "action_type": np.random.randint(0, cls.N_ACTIONS),
            "target_id": np.random.randint(0, cls.N_TARGETS),
            "priority": np.random.randint(0, cls.N_PRIORITY),
            "reasoning_tokens": np.random.randint(50, 500),
        }


# ─── Main Environment ───────────────────────────────────────

class HospitalEnv:
    """OpenEnv-compatible hospital crisis simulation environment.

    Attributes:
        observation_space: Shape definitions for numpy observations.
        action_space: Discrete action space definitions.
        state: Current EnvironmentState (world model).
        apps: Enterprise application simulators.
    """

    observation_space = ObservationSpace
    action_space = ActionSpace

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int = 200,
        difficulty: float = 0.5,
    ) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.difficulty = difficulty

        self._crisis_gen = CrisisGenerator(seed=seed)
        self._schema_drift = SchemaDrift(seed=seed)
        self.apps = EnterpriseAppRegistry()

        self._state: EnvironmentState | None = None
        self._episode_count = 0
        self._step_rewards: list[float] = []
        self._step_times: list[float] = []
        self._drift_events: list[dict[str, Any]] = []

    # ── Properties ───────────────────────────────────────

    @property
    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment not initialized — call reset() first")
        return self._state

    @property
    def is_terminal(self) -> bool:
        if self._state is None:
            return True
        return (
            self._state.crisis_resolved
            or self._state.catastrophic_failure
            or self._state.step_count >= self.max_steps
        )

    @property
    def episode_stats(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        return {
            "episode": self._episode_count,
            "steps": self._state.step_count,
            "total_reward": sum(self._step_rewards),
            "mean_step_reward": np.mean(self._step_rewards) if self._step_rewards else 0.0,
            "survival_rate": self._state.survival_rate,
            "patients_total": self._state.total_patients,
            "deceased": self._state.deceased_count,
            "discharged": self._state.discharged_count,
            "violations_injected": self._state.violations_injected,
            "violations_caught": self._state.violations_caught,
            "drift_events": len(self._drift_events),
            "mean_step_time_ms": np.mean(self._step_times) * 1000 if self._step_times else 0.0,
            "terminated_reason": self._termination_reason(),
        }

    # ── OpenEnv API ──────────────────────────────────────

    async def reset(
        self,
        scenario: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the environment for a new episode.

        Args:
            scenario: Optional crisis configuration override.

        Returns:
            Initial observation dict.
        """
        self._episode_count += 1
        self._step_rewards.clear()
        self._step_times.clear()
        self._drift_events.clear()

        # Generate crisis
        crisis_type = None
        difficulty = self.difficulty
        if scenario:
            ct = scenario.get("crisis_type")
            if ct:
                crisis_type = CrisisType(ct)
            difficulty = scenario.get("difficulty", self.difficulty)

        crisis, policies = self._crisis_gen.generate(
            crisis_type=crisis_type,
            episode=self._episode_count,
            difficulty=difficulty,
        )

        # Initialize resources based on crisis
        resources = ResourceState(
            icu_beds_total=crisis.icu_config.get("beds", 20),
            ventilators_total=crisis.icu_config.get("ventilators", 15),
            staff_ratio=crisis.staff_reduction,
        )
        if crisis.blood_inventory:
            resources.blood_inventory = crisis.blood_inventory

        # Build pending patients queue (patients not yet admitted)
        pending = crisis.patient_list.copy()
        immediate = pending[:max(3, int(len(pending) * 0.3))]
        pending_queue = pending[len(immediate):]

        self._state = EnvironmentState(
            crisis=crisis,
            episode=self._episode_count,
            patients=list(immediate),
            resources=resources,
            active_policies=policies,
            pending_patients=pending_queue,
        )

        # Plan schema drifts for this episode
        self._schema_drift.plan_drifts(self.max_steps, difficulty)

        # Reset enterprise apps
        self.apps.reset()

        logger.info(
            "Episode %d started — crisis=%s patients=%d difficulty=%.1f",
            self._episode_count,
            crisis.type.value,
            crisis.patient_count,
            difficulty,
        )

        return self._build_observation()

    async def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute one action in the environment.

        Args:
            action: Dict with agent_id, action_type, target_id, priority, reasoning_tokens.

        Returns:
            Tuple of (observation, reward, terminated, info).
        """
        t0 = time.perf_counter()

        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        if self.is_terminal:
            return self._build_observation(), 0.0, True, {"reason": "already_terminal"}

        # Parse action
        agent_action = self._parse_action(action)

        # Execute action on state
        action_result = self._execute_action(agent_action)

        # Apply schema drifts
        drift_events = self._schema_drift.apply_drifts(self._state)
        self._drift_events.extend(drift_events)
        for event in drift_events:
            self._state.add_drift_event(event)

        # Inject violations periodically
        if self._state.step_count % 10 == 0 and self.difficulty > 0.3:
            violation = self._crisis_gen.inject_violation(self._state.crisis)
            self._state.violations_injected += 1
            action_result["violation_injected"] = violation

        # Update world state (natural deterioration, patient arrivals)
        self._state.update(action_result)

        # Record action in history
        self._state.action_history.append(agent_action)

        # Compute reward (basic — full reward model is in triage/rewards/)
        reward = self._compute_step_reward(agent_action, action_result)
        self._step_rewards.append(reward)

        # Timing
        elapsed = time.perf_counter() - t0
        self._step_times.append(elapsed)

        terminated = self.is_terminal
        info = {
            "action_result": action_result,
            "drift_events": drift_events,
            "step": self._state.step_count,
            "reward_breakdown": self._reward_breakdown(agent_action),
        }

        if terminated:
            info["episode_stats"] = self.episode_stats
            logger.info(
                "Episode %d ended — reason=%s survival=%.1f%% steps=%d total_reward=%.2f",
                self._episode_count,
                self._termination_reason(),
                self._state.survival_rate * 100,
                self._state.step_count,
                sum(self._step_rewards),
            )

        return self._build_observation(), reward, terminated, info

    async def get_state(self) -> dict[str, Any]:
        """Return current state as JSON (for API/WebSocket)."""
        return self.state.to_json()

    def render(self, mode: str = "ascii") -> str | None:
        """Render current state."""
        if mode == "ascii":
            return self.state.render_ascii()
        return None

    # ── Tool Execution (for agents) ──────────────────────

    async def execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        requester: AgentType,
    ) -> dict[str, Any]:
        """Execute an enterprise app tool call.

        This is the primary interface for LLM agents to interact with EHR,
        Pharmacy, Scheduling, Insurance, and Equipment systems.
        """
        if self._state is None:
            return {"error": "Environment not initialized"}

        result = self.apps.execute_tool(tool_name, params, self._state, requester)

        # Track agent activity
        if requester in self._state.agent_states:
            self._state.agent_states[requester].actions_taken += 1

        return result

    # ── Message Bus ──────────────────────────────────────

    async def send_message(self, message: AgentMessage) -> None:
        """Route a message between agents."""
        if self._state is None:
            return
        self._state.message_history.append(message)

        if isinstance(message.from_agent, AgentType):
            agent_state = self._state.agent_states.get(message.from_agent)
            if agent_state:
                agent_state.messages_sent += 1
                agent_state.token_usage += message.token_count

    # ── Internal Logic ───────────────────────────────────

    def _parse_action(self, action: dict[str, Any]) -> AgentAction:
        agent_types = list(AgentType)
        agent_idx = action.get("agent_id", 0) % len(agent_types)
        action_type_idx = action.get("action_type", 0) % len(ActionType)

        return AgentAction(
            agent_type=agent_types[agent_idx],
            action_type=ActionType(action_type_idx),
            target_id=action.get("target_id", 0),
            priority=action.get("priority", 0),
            reasoning=action.get("reasoning", ""),
            reasoning_tokens=action.get("reasoning_tokens", 0),
        )

    def _execute_action(self, action: AgentAction) -> dict[str, Any]:
        """Execute agent action on state, return result with side effects."""
        result: dict[str, Any] = {
            "success": True,
            "action": action.to_dict(),
            "side_effects": [],
        }

        # Update agent state
        agent_state = self._state.agent_states.get(action.agent_type)
        if agent_state:
            agent_state.actions_taken += 1
            agent_state.current_action = action.action_type.name
            agent_state.token_usage += action.reasoning_tokens
            agent_state.idle_steps = 0

        # Increment idle counter for agents that didn't act
        for at, ast in self._state.agent_states.items():
            if at != action.agent_type:
                ast.idle_steps += 1

        # Action-specific logic
        target_patient = self._get_target_patient(action.target_id)

        if action.action_type == ActionType.TRIAGE_PATIENT and target_patient:
            target_patient.add_event("TRIAGED", f"Triage score assigned: {target_patient.triage_score}", action.agent_type)

        elif action.action_type == ActionType.TRANSFER_TO_ICU and target_patient:
            if action.agent_type != AgentType.ICU_MANAGEMENT:
                result["success"] = False
                result["error"] = "Bypassed ICU chain of command"
                self._state.add_app_audit(
                    AppAuditEvent(
                        app="icu_manager",
                        tool_name="allocate_icu_bed",
                        requester=action.agent_type,
                        patient_id=target_patient.id,
                        status="blocked",
                        message="Only ICU_MANAGEMENT can allocate ICU beds",
                        details={"workflow_violation": "bypass_chain_of_command"},
                    )
                )
            else:
                authorization_id = self._state.find_active_override_token("icu_override", target_patient.id)
                tool_result = self.apps.execute_tool(
                    "allocate_icu_bed",
                    {
                        "patient_id": target_patient.id,
                        "authorization_id": authorization_id,
                    },
                    self._state,
                    action.agent_type,
                )
                result["tool_result"] = tool_result
                result["success"] = tool_result.get("status") == "approved"
                if not result["success"]:
                    result["error"] = tool_result.get("message", "ICU allocation blocked")

        elif action.action_type == ActionType.ASSIGN_TREATMENT and target_patient:
            target_patient.treatment_plan.append(f"Treatment-{self._state.step_count}")
            target_patient.add_event("TREATMENT", "Treatment plan assigned", action.agent_type)
            # Reduce deterioration rate with treatment
            target_patient.deterioration_rate *= 0.5

        elif action.action_type == ActionType.ORDER_MEDICATION and target_patient:
            if action.agent_type != AgentType.PHARMACY:
                result["success"] = False
                result["error"] = "Bypassed pharmacy chain of command"
                self._state.add_app_audit(
                    AppAuditEvent(
                        app="pharmacy",
                        tool_name="dispense_medication",
                        requester=action.agent_type,
                        patient_id=target_patient.id,
                        status="blocked",
                        message="Only PHARMACY can clear medication fulfillment",
                        details={"workflow_violation": "bypass_chain_of_command"},
                    )
                )
            else:
                medication = self._select_medication(target_patient)
                authorization_id = self._state.find_active_override_token(
                    "pharmacy_override",
                    target_patient.id,
                )
                tool_result = self.apps.execute_tool(
                    "dispense_medication",
                    {
                        "patient_id": target_patient.id,
                        "medication": medication,
                        "dose": "standard",
                        "double_verified": medication in {"morphine", "fentanyl", "ketamine", "midazolam", "propofol"},
                        "emergency": target_patient.status == PatientStatus.CRITICAL,
                        "authorization_id": authorization_id,
                    },
                    self._state,
                    action.agent_type,
                )
                result["tool_result"] = tool_result
                result["success"] = tool_result.get("status") == "approved"
                if not result["success"]:
                    result["error"] = tool_result.get("message", "Medication blocked")

        elif action.action_type == ActionType.DISCHARGE_PATIENT and target_patient:
            if target_patient.status in (PatientStatus.STABLE, PatientStatus.SERIOUS):
                target_patient.status = PatientStatus.DISCHARGED
                target_patient.add_event("DISCHARGED", "Patient discharged", action.agent_type)
                if target_patient.ward == WardType.ICU:
                    self._state.resources.icu_beds_occupied = max(0, self._state.resources.icu_beds_occupied - 1)

        elif action.action_type == ActionType.ESCALATE_TO_CMO:
            msg = AgentMessage(
                from_agent=action.agent_type,
                to_agent=AgentType.CMO_OVERSIGHT,
                content=f"Escalation: {action.reasoning}",
                msg_type=MessageType.ALERT,
                priority=action.priority,
                patient_id=target_patient.id if target_patient else None,
                request_type="override_request",
                payload={"action_type": action.action_type.name, "reasoning": action.reasoning},
            )
            self._state.message_history.append(msg)

        elif action.action_type == ActionType.FLAG_POLICY_VIOLATION:
            if agent_state:
                agent_state.violations_count += 1  # tracks violations detected
            self._state.violations_caught += 1

        elif action.action_type == ActionType.OVERRIDE_DECISION:
            scope = self._override_scope_from_reasoning(action.reasoning)
            token = self._state.issue_override_token(
                scope=scope,
                reason=action.reasoning or "CMO override approval",
                patient_id=target_patient.id if target_patient else None,
            )
            result["authorization_id"] = token.id
            result["side_effects"].append(
                {
                    "type": "override_token",
                    "scope": scope,
                    "patient_id": target_patient.id if target_patient else None,
                }
            )

        elif action.action_type == ActionType.ACTIVATE_OVERFLOW:
            self._state.resources.icu_beds_total += 5
            result["side_effects"].append({
                "type": "resource_change",
                "resource": "icu_beds_total",
                "delta": 5,
            })

        return result

    def _get_target_patient(self, target_id: int) -> Patient | None:
        if 0 <= target_id < len(self._state.patients):
            return self._state.patients[target_id]
        return None

    def _select_medication(self, patient: Patient) -> str:
        text = " ".join(patient.treatment_plan).lower()
        if "sedation" in text or "vent" in text:
            return "midazolam"
        if patient.condition in {"hemorrhage", "postpartum hemorrhage"}:
            return "blood_thinners"
        if patient.condition in {"sepsis", "pneumonia", "meningitis"}:
            return "antibiotics_broad"
        if patient.status == PatientStatus.CRITICAL:
            return "morphine"
        return "antibiotics_broad"

    def _override_scope_from_reasoning(self, reasoning: str) -> str:
        lowered = reasoning.lower()
        if any(token in lowered for token in ["pharmacy", "med", "drug", "allergy", "interaction"]):
            return "pharmacy_override"
        return "icu_override"

    def _build_observation(self) -> dict[str, Any]:
        obs = self.state.to_observation()
        return {
            "numpy": obs,
            "json": self.state.to_json(),
        }

    def _compute_step_reward(self, action: AgentAction, result: dict[str, Any]) -> float:
        """Basic reward computation. Full reward model is in triage/rewards/."""
        reward = 0.0

        # Survival bonus
        reward += self._state.survival_rate * 0.3

        # Action success bonus
        if result.get("success"):
            reward += 0.1

        # Penalty for deaths this step
        reward -= self._state.deceased_count * 0.05

        # Critical patients without treatment penalty
        untreated_critical = sum(
            1 for p in self._state.patients
            if p.status == PatientStatus.CRITICAL and not p.treatment_plan
        )
        reward -= untreated_critical * 0.02

        # Efficiency bonus — reasoning tokens cost
        if action.reasoning_tokens > 300:
            reward -= 0.01  # penalty for verbose reasoning

        # Violation detection bonus
        if action.action_type == ActionType.FLAG_POLICY_VIOLATION:
            reward += 0.2

        # ICU overload penalty
        if self._state.icu_occupancy > 0.95:
            reward -= 0.1

        return round(reward, 4)

    def _reward_breakdown(self, action: AgentAction) -> dict[str, float]:
        return {
            "survival": self._state.survival_rate * 0.3,
            "action_quality": 0.1,
            "death_penalty": -self._state.deceased_count * 0.05,
            "token_efficiency": -0.01 if action.reasoning_tokens > 300 else 0.0,
            "violation_bonus": 0.2 if action.action_type == ActionType.FLAG_POLICY_VIOLATION else 0.0,
        }

    def _termination_reason(self) -> str:
        if self._state is None:
            return "not_started"
        if self._state.catastrophic_failure:
            return "catastrophic_failure"
        if self._state.crisis_resolved:
            return "crisis_resolved"
        if self._state.step_count >= self.max_steps:
            return "max_steps_reached"
        return "in_progress"
