"""
EpisodeCollector — runs episodes, collects (state, action, reward) tuples
for DPO preference pair generation.

Produces JSONL datasets compatible with HF TRL DPOTrainer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from triage.agents.message_bus import MessageBus
from triage.agents.specialized import create_all_agents
from triage.agents.strategy_memory import StrategyMemory
from triage.env.hospital_env import HospitalEnv
from triage.env.state import AgentType, CrisisType
from triage.rewards.reward_model import RewardModel

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Summary of a completed episode."""
    episode_id: int
    crisis_type: str
    steps: int
    total_reward: float
    survival_rate: float
    deceased: int
    discharged: int
    violations_caught: int
    violations_injected: int
    drift_events: int
    duration_seconds: float
    trajectory: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "crisis_type": self.crisis_type,
            "steps": self.steps,
            "total_reward": round(self.total_reward, 4),
            "survival_rate": round(self.survival_rate, 4),
            "deceased": self.deceased,
            "discharged": self.discharged,
            "violations_caught": self.violations_caught,
            "violations_injected": self.violations_injected,
            "drift_events": self.drift_events,
            "duration_seconds": round(self.duration_seconds, 2),
        }



class EpisodeCollector:
    """Runs simulation episodes and collects training data.

    Outputs:
      - JSONL trajectory files for DPO training
      - Episode summaries for metrics dashboards
      - Preference pairs (chosen/rejected) for DPO
    """

    def __init__(
        self,
        agents_config_path: str = "./config/agents.yaml",
        output_dir: str = "./data/episodes",
        mock_llm: bool = True,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mock_llm = mock_llm
        self.seed = seed

        # Load agent configs
        with open(agents_config_path) as f:
            self.agent_configs = yaml.safe_load(f)

        self.reward_model = RewardModel()
        self.strategy_memory = StrategyMemory()
        self._episode_results: list[EpisodeResult] = []

    async def collect_episode(
        self,
        crisis_type: CrisisType | None = None,
        difficulty: float = 0.5,
        max_steps: int = 200,
    ) -> EpisodeResult:
        """Run a single episode and collect trajectory data."""
        t0 = time.perf_counter()

        # Initialize environment
        env = HospitalEnv(seed=self.seed, max_steps=max_steps, difficulty=difficulty)

        # Initialize message bus and agents
        bus = MessageBus(token_budget=50_000)
        agents = create_all_agents(self.agent_configs, bus, self.mock_llm)

        # Reset environment
        scenario = {}
        if crisis_type:
            scenario["crisis_type"] = crisis_type.value
            scenario["difficulty"] = difficulty

        obs = await env.reset(scenario if scenario else None)
        state = env.state

        # Strategy memory is now injected directly by BaseAgent._call_llm

        trajectory: list[dict[str, Any]] = []
        step_rewards: list[float] = []

        # Episode loop
        step = 0
        while not env.is_terminal and step < max_steps:
            step += 1

            # Each agent makes decisions
            all_actions = []
            for agent_type in AgentType:
                agent = agents[agent_type]
                actions = await agent.act(state)
                all_actions.extend(actions)

            # Execute first valid action (simple scheduler)
            if all_actions:
                # Sort by priority descending
                all_actions.sort(key=lambda a: a.priority, reverse=True)
                primary_action = all_actions[0]
                action_dict = primary_action.to_env_action()
                action_dict["reasoning"] = primary_action.reasoning
                action_dict["reasoning_tokens"] = primary_action.reasoning_tokens

                obs, reward, terminated, info = await env.step(action_dict)

                # Compute detailed reward
                breakdown = self.reward_model.compute(
                    env.state,
                    all_actions,
                    info.get("drift_events", []),
                    action_result=info.get("action_result", {}),
                    messages=bus.history,
                    app_audits=env.state.app_audit_log,
                )

                trajectory.append({
                    "step": step,
                    "state_summary": {
                        "alive": state.alive_count,
                        "critical": state.critical_count,
                        "deceased": state.deceased_count,
                        "icu_occupancy": round(state.icu_occupancy, 3),
                    },
                    "action": primary_action.to_dict(),
                    "all_actions": [a.to_dict() for a in all_actions[:5]],
                    "reward": round(reward, 4),
                    "reward_breakdown": breakdown.to_dict(),
                    "drift_events": info.get("drift_events", []),
                    "terminated": terminated,
                })
                step_rewards.append(breakdown.total)

                state = env.state
                if terminated:
                    break
            else:
                # No actions — skip step
                await env.step(env.action_space.sample())
                state = env.state

        # Episode complete
        elapsed = time.perf_counter() - t0
        stats = env.episode_stats

        result = EpisodeResult(
            episode_id=stats.get("episode", 0),
            crisis_type=state.crisis.type.value,
            steps=state.step_count,
            total_reward=sum(step_rewards),
            survival_rate=state.survival_rate,
            deceased=state.deceased_count,
            discharged=state.discharged_count,
            violations_caught=state.violations_caught,
            violations_injected=state.violations_injected,
            drift_events=len(info.get("drift_events", [])) if 'info' in dir() else 0,
            duration_seconds=elapsed,
            trajectory=trajectory,
        )

        # Record strategies
        for agent_type, agent in agents.items():
            lesson = {
                "context": f"Crisis: {state.crisis.type.value}, Difficulty: {state.crisis.severity}",
                "action_taken": f"{agent.actions_taken} actions taken during episode",
                "outcome": f"Survival: {result.survival_rate:.1f}%",
                "reward_delta": float(result.total_reward),
                "crisis_type": state.crisis.type.value,
                "step": result.steps,
            }
            self.strategy_memory.add_lesson(agent_type.value, lesson)

        self._episode_results.append(result)

        # Save trajectory
        self._save_trajectory(result)

        logger.info(
            "Episode %d complete — survival=%.1f%% reward=%.2f steps=%d time=%.1fs",
            result.episode_id,
            result.survival_rate * 100,
            result.total_reward,
            result.steps,
            elapsed,
        )

        return result

    async def collect_batch(
        self,
        n_episodes: int = 10,
        crisis_types: list[CrisisType] | None = None,
        difficulty: float = 0.5,
    ) -> list[EpisodeResult]:
        """Collect multiple episodes."""
        results = []
        types = crisis_types or list(CrisisType)

        for i in range(n_episodes):
            ct = types[i % len(types)]
            self.seed += 1
            result = await self.collect_episode(
                crisis_type=ct,
                difficulty=difficulty,
            )
            results.append(result)

        # Generate DPO preference pairs
        self._generate_preference_pairs(results)

        return results

    def _save_trajectory(self, result: EpisodeResult) -> None:
        """Save trajectory as JSONL."""
        path = self.output_dir / f"episode_{result.episode_id:04d}.jsonl"
        with open(path, "w") as f:
            for step_data in result.trajectory:
                f.write(json.dumps(step_data) + "\n")

        # Also save summary
        summary_path = self.output_dir / f"episode_{result.episode_id:04d}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _generate_preference_pairs(self, results: list[EpisodeResult]) -> None:
        """Generate DPO preference pairs from episode results.

        Pairs episodes by crisis type, marking higher-reward episodes as 'chosen'
        and lower-reward episodes as 'rejected'.
        """
        pairs: list[dict[str, Any]] = []

        # Group by crisis type
        by_type: dict[str, list[EpisodeResult]] = {}
        for r in results:
            by_type.setdefault(r.crisis_type, []).append(r)

        for crisis_type, episodes in by_type.items():
            if len(episodes) < 2:
                continue
            # Sort by total reward
            sorted_eps = sorted(episodes, key=lambda e: e.total_reward, reverse=True)

            for i in range(len(sorted_eps) - 1):
                chosen = sorted_eps[i]
                rejected = sorted_eps[i + 1]

                # Build prompt from the shared crisis scenario
                prompt = f"Crisis: {crisis_type}\n"

                # Build chosen/rejected from trajectories
                chosen_text = json.dumps([
                    {"step": t["step"], "action": t["action"], "reward": t["reward"]}
                    for t in chosen.trajectory[:10]
                ])
                rejected_text = json.dumps([
                    {"step": t["step"], "action": t["action"], "reward": t["reward"]}
                    for t in rejected.trajectory[:10]
                ])

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "chosen_reward": chosen.total_reward,
                    "rejected_reward": rejected.total_reward,
                })

        # Save
        if pairs:
            path = self.output_dir / "dpo_pairs.jsonl"
            with open(path, "a") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
            logger.info("Generated %d DPO preference pairs", len(pairs))

    def get_summary(self) -> dict[str, Any]:
        """Summary of all collected episodes."""
        if not self._episode_results:
            return {"episodes": 0}

        rewards = [r.total_reward for r in self._episode_results]
        survivals = [r.survival_rate for r in self._episode_results]
        return {
            "episodes": len(self._episode_results),
            "mean_reward": round(float(np.mean(rewards)), 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "mean_survival": round(float(np.mean(survivals)), 4),
            "best_reward": round(float(max(rewards)), 4),
            "worst_reward": round(float(min(rewards)), 4),
        }
