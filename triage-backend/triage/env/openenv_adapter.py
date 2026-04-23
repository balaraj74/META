"""
openenv_adapter.py — OpenEnv-compliant wrapper around HospitalEnv.

Exposes HospitalEnv via the openenv-core BaseEnv interface so that:
  - OpenEnv trainers can discover and talk to this environment
  - HuggingFace Spaces can serve it as a standard RL environment
  - GRPOTrainer rollout functions can call reset()/step() synchronously

Usage:
    from triage.env.openenv_adapter import TriageOpenEnv
    env = TriageOpenEnv()
    obs = env.reset()
    obs, reward, done, info = env.step({"action_type": "TRIAGE_PATIENT", ...})
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from triage.env.hospital_env import HospitalEnv
from triage.env.state import ActionType, AgentType, CrisisType

logger = logging.getLogger(__name__)

# ── OpenEnv metadata schema (matches openenv-core spec) ─────────────────────
ENV_INFO = {
    "name": "triage-hospital-crisis",
    "version": "1.0.0",
    "description": (
        "Multi-agent hospital crisis simulation. 6 specialized AI agents "
        "(CMO, ER Triage, ICU, Pharmacy, HR, IT) coordinate to manage "
        "mass-casualty events, disease outbreaks, equipment failures, and "
        "staff shortages. Designed for GRPO/RLVR training."
    ),
    "author": "TRIAGE Team",
    "action_space": "discrete",
    "observation_space": "dict",
    "reward_range": [-1.0, 1.5],
    "max_episode_steps": 200,
    "crisis_types": [c.value for c in CrisisType],
    "agent_types": [a.value for a in AgentType],
    "action_types": [a.value for a in ActionType],
    "tags": ["healthcare", "multi-agent", "crisis-management", "rl"],
}


# ── Sync helper — run async env in sync context ───────────────────────────────
def _run(coro):
    """Run an async coroutine from sync context safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from an already-running loop (e.g. Jupyter / FastAPI),
            # use a new thread to avoid deadlock.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── TriageOpenEnv ─────────────────────────────────────────────────────────────
class TriageOpenEnv:
    """
    OpenEnv-compliant wrapper around HospitalEnv.

    Provides both sync and async interfaces:
      Sync:  env.reset() / env.step(action)
      Async: await env.async_reset() / await env.async_step(action)

    Compatible with:
      - openenv-core BaseEnv spec
      - TRL GRPOTrainer rollout functions
      - FastAPI /env/* routes
      - Gradio Spaces demo
    """

    # ── Class-level metadata ──────────────────────────────────────────────────
    metadata = ENV_INFO

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 50,
        difficulty: float = 0.5,
        crisis_type: str | None = None,
    ) -> None:
        self._env = HospitalEnv(
            seed=seed,
            max_steps=max_steps,
            difficulty=difficulty,
        )
        self._crisis_type = crisis_type
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_obs: dict[str, Any] = {}
        self._seed = seed

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def info(self) -> dict[str, Any]:
        """Return OpenEnv environment info (matches /env/info endpoint)."""
        return dict(ENV_INFO)

    @property
    def is_done(self) -> bool:
        return self._env.is_terminal or self._step_count >= self._env.max_steps

    # ── Sync API (for GRPOTrainer rollout functions) ──────────────────────────

    def reset(self, scenario: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        """
        Reset the environment. Returns initial observation dict.

        Args:
            scenario: optional dict with keys like {"crisis_type": "mass_casualty",
                                                    "difficulty": 0.7}
        """
        self._step_count = 0
        self._episode_reward = 0.0
        obs_raw = _run(self._env.reset(scenario=scenario or self._build_scenario()))
        self._last_obs = self._make_observation(obs_raw)
        return self._last_obs

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Apply action and advance one step.

        Args:
            action: dict with keys:
                - agent_type: str (e.g. "er_triage")
                - action_type: str (e.g. "TRIAGE_PATIENT")
                - target_id: int
                - priority: int (1-10)
                - reasoning: str

        Returns:
            (observation, reward, done, info)
        """
        safe_action = self._validate_and_normalize_action(action)
        obs_raw, reward, terminated, info = _run(self._env.step(safe_action))
        self._step_count += 1
        self._episode_reward += float(reward)
        done = terminated or self.is_done
        obs = self._make_observation(obs_raw)
        self._last_obs = obs
        info["episode_reward"] = self._episode_reward
        info["step"] = self._step_count
        return obs, float(reward), done, info

    def observation(self) -> dict[str, Any]:
        """Return the current observation without advancing the environment."""
        return self._last_obs

    def render(self) -> str:
        """Return a human-readable text representation of current state."""
        state = self._env.state
        lines = [
            f"Step {state.step_count} | Crisis: {state.crisis.type.value}",
            f"Patients: {state.alive_count} alive, {state.critical_count} critical",
            f"ICU: {state.icu_occupancy:.0%} occupancy",
            f"Violations: {state.violations_caught}/{state.violations_injected} caught",
            f"Survival: {state.survival_rate:.1%}",
        ]
        return "\n".join(lines)

    # ── Async API (for FastAPI routes and internal use) ───────────────────────

    async def async_reset(self, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
        self._step_count = 0
        self._episode_reward = 0.0
        obs_raw = await self._env.reset(scenario=scenario or self._build_scenario())
        self._last_obs = self._make_observation(obs_raw)
        return self._last_obs

    async def async_step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        safe_action = self._validate_and_normalize_action(action)
        obs_raw, reward, terminated, info = await self._env.step(safe_action)
        self._step_count += 1
        self._episode_reward += float(reward)
        done = terminated or self.is_done
        obs = self._make_observation(obs_raw)
        self._last_obs = obs
        info["episode_reward"] = self._episode_reward
        info["step"] = self._step_count
        return obs, float(reward), done, info

    # ── Prompt generation (for GRPO dataset building) ─────────────────────────

    def state_to_prompt(self, agent_type: str = "er_triage") -> str:
        """
        Convert current environment state into an LLM prompt for GRPO training.

        The model must output a valid action JSON in its completion.
        """
        state = self._env.state
        critical_patients = [
            p for p in state.patients if p.status.value == "CRITICAL"
        ][:5]

        patient_lines = []
        for p in critical_patients:
            patient_lines.append(
                f"  - P-{p.id}: {p.name}, age {p.age}, "
                f"status={p.status.value}, "
                f"condition={p.condition}, "
                f"triage={p.triage_score}, deterioration={p.deterioration_rate:.2f}"
            )
        patients_str = "\n".join(patient_lines) if patient_lines else "  (none)"

        prompt = f"""You are the {agent_type.upper()} agent in a hospital crisis simulation.

CRISIS: {state.crisis.type.value.upper()}
STEP: {state.step_count}/{self._env.max_steps}
ICU OCCUPANCY: {state.icu_occupancy:.0%} ({state.resources.icu_beds_occupied}/{state.resources.icu_beds_total} beds)
CRITICAL PATIENTS ({state.critical_count} total — top 5):
{patients_str}
VIOLATIONS INJECTED: {state.violations_injected} | CAUGHT: {state.violations_caught}
SURVIVAL RATE: {state.survival_rate:.1%}

Your role: {self._agent_role(agent_type)}

Decide the single most important action right now. Respond with ONLY valid JSON:
{{
  "action_type": "<one of: {self._valid_actions_for(agent_type)}>",
  "target_id": <patient ID integer or 0 if not patient-specific>,
  "priority": <integer 1-10, where 1=highest>,
  "reasoning": "<1-2 sentences citing specific patient data or metrics>"
}}"""
        return prompt

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_scenario(self) -> dict[str, Any] | None:
        if self._crisis_type:
            return {"crisis_type": self._crisis_type}
        return None

    def _make_observation(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Normalise the raw env observation into a clean OpenEnv observation."""
        state = self._env.state
        return {
            "step": state.step_count,
            "max_steps": self._env.max_steps,
            "crisis_type": state.crisis.type.value,
            "difficulty": self._env.difficulty,
            "alive_count": state.alive_count,
            "critical_count": state.critical_count,
            "deceased_count": state.deceased_count,
            "discharged_count": state.discharged_count,
            "icu_occupancy": round(state.icu_occupancy, 4),
            "survival_rate": round(state.survival_rate, 4),
            "violations_injected": state.violations_injected,
            "violations_caught": state.violations_caught,
            "resources": state.resources.to_dict() if hasattr(state.resources, "to_dict") else {},
            "patients_summary": [
                {
                    "id": p.id,
                    "status": p.status.value,
                    "age": p.age,
                    "has_treatment": bool(p.treatment_plan),
                    "condition": p.condition,
                    "triage_score": p.triage_score,
                }
                for p in state.patients[:20]  # cap at 20 for token efficiency
            ],
        }

    def _validate_and_normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Validate action dict and fill defaults. Prevents env crashes."""
        # Normalise agent_type
        agent_type_raw = action.get("agent_type", "er_triage")
        try:
            agent_type = AgentType(str(agent_type_raw).lower())
        except ValueError:
            agent_type = AgentType.ER_TRIAGE

        # Normalise action_type
        action_type_raw = action.get("action_type", "TRIAGE_PATIENT")
        try:
            action_type = ActionType[str(action_type_raw).upper()]
        except KeyError:
            action_type = ActionType.TRIAGE_PATIENT

        return {
            "agent_type": agent_type.value,
            "action_type": action_type.value,
            "target_id": int(action.get("target_id", 0)),
            "priority": max(1, min(10, int(action.get("priority", 5)))),
            "reasoning": str(action.get("reasoning", ""))[:500],
            "reasoning_tokens": int(action.get("reasoning_tokens", 50)),
        }

    @staticmethod
    def _agent_role(agent_type: str) -> str:
        roles = {
            "cmo_oversight":  "Chief Medical Officer — escalation authority, hospital-wide crisis governance",
            "er_triage":      "Emergency Room Triage — patient severity classification, START protocol",
            "icu_management": "ICU Management — bed allocation, ventilator management, overflow protocols",
            "pharmacy":       "Pharmacy — medication dispensing, drug interaction verification, order validation",
            "hr_rostering":   "HR Rostering — emergency staff scheduling, fatigue monitoring, agency call-ins",
            "it_systems":     "IT Systems — EHR integrity, backup protocols, policy compliance monitoring",
        }
        return roles.get(agent_type, "Hospital staff — support patient care")

    @staticmethod
    def _valid_actions_for(agent_type: str) -> str:
        primary = {
            "er_triage":      "TRIAGE_PATIENT, UPDATE_EHR, ASSIGN_TREATMENT",
            "icu_management": "TRANSFER_TO_ICU, TRANSFER_TO_WARD, ACTIVATE_OVERFLOW",
            "pharmacy":       "ORDER_MEDICATION, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE",
            "hr_rostering":   "REQUEST_STAFF, FLAG_POLICY_VIOLATION",
            "cmo_oversight":  "OVERRIDE_DECISION, ACTIVATE_OVERFLOW, ASSIGN_TREATMENT",
            "it_systems":     "UPDATE_EHR, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE",
        }
        return primary.get(agent_type, "TRIAGE_PATIENT, ASSIGN_TREATMENT, UPDATE_EHR")


# ── Module-level factory (openenv-core discovers this) ───────────────────────
def make_env(**kwargs) -> TriageOpenEnv:
    """Factory function called by openenv-core when loading this environment."""
    return TriageOpenEnv(**kwargs)
