#!/usr/bin/env python3
"""
benchmark_agent.py — TRIAGE Multi-Agent System Benchmark
=========================================================

Runs 5 crisis scenarios × N episodes each and measures:

Per-agent metrics
  - Actions taken / messages sent
  - Decision latency (ms)
  - Correct-action ratio (versus heuristic oracle)

System metrics
  - Survival rate (alive / total)
  - ICU utilisation
  - Policy violation detection rate
  - Composite reward score (heuristic 0–10)

Usage
-----
  cd triage-backend
  python3 scripts/benchmark_agent.py                    # quick (3 eps × 20 steps)
  python3 scripts/benchmark_agent.py --episodes 10 --steps 30
  python3 scripts/benchmark_agent.py --output results/bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False

from triage.agents.message_bus import MessageBus
from triage.agents.specialized import create_all_agents
from triage.env.hospital_env import HospitalEnv
from triage.env.state import ActionType, AgentType, CrisisType

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("benchmark")

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS: list[dict[str, Any]] = [
    {"name": "Mass Casualty",       "crisis_type": "mass_casualty",    "difficulty": 0.7},
    {"name": "Disease Outbreak",    "crisis_type": "outbreak",          "difficulty": 0.6},
    {"name": "Equipment Failure",   "crisis_type": "equipment_failure", "difficulty": 0.5},
    {"name": "Staff Shortage",      "crisis_type": "staff_shortage",    "difficulty": 0.7},
    {"name": "Combined Surge",      "crisis_type": "mass_casualty",    "difficulty": 0.95},
]

# Heuristic oracle: which ActionTypes are "correct" for each agent
AGENT_CORRECT_ACTIONS: dict[AgentType, set[ActionType]] = {
    AgentType.ER_TRIAGE:      {ActionType.TRIAGE_PATIENT, ActionType.TRANSFER_TO_ICU, ActionType.ESCALATE_TO_CMO},
    AgentType.ICU_MANAGEMENT: {ActionType.ASSIGN_TREATMENT, ActionType.TRANSFER_TO_WARD, ActionType.REQUEST_SPECIALIST, ActionType.ACTIVATE_OVERFLOW},
    AgentType.PHARMACY:       {ActionType.ORDER_MEDICATION, ActionType.FLAG_POLICY_VIOLATION},
    AgentType.HR_ROSTERING:   {ActionType.REQUEST_STAFF, ActionType.FLAG_POLICY_VIOLATION},
    AgentType.IT_SYSTEMS:     {ActionType.FLAG_POLICY_VIOLATION, ActionType.VERIFY_INSURANCE, ActionType.UPDATE_EHR},
    AgentType.CMO_OVERSIGHT:  {ActionType.OVERRIDE_DECISION, ActionType.ACTIVATE_OVERFLOW, ActionType.ASSIGN_TREATMENT},
}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentStats:
    agent_type: str
    actions_taken: int = 0
    correct_actions: int = 0
    messages_sent: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def correct_ratio(self) -> float:
        if self.actions_taken == 0:
            return 1.0
        return self.correct_actions / self.actions_taken

    @property
    def mean_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]


@dataclass
class EpisodeResult:
    scenario: str
    episode: int
    steps_run: int
    survival_rate: float
    icu_utilisation: float
    violations_caught: int
    violations_injected: int
    deceased: int
    discharged: int
    reward: float
    duration_s: float
    agents: dict[str, AgentStats] = field(default_factory=dict)

    @property
    def violation_detection_rate(self) -> float:
        return self.violations_caught / max(1, self.violations_injected)


@dataclass
class ScenarioResult:
    scenario: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def mean_survival(self) -> float:
        return mean(e.survival_rate for e in self.episodes) if self.episodes else 0.0

    @property
    def mean_reward(self) -> float:
        return mean(e.reward for e in self.episodes) if self.episodes else 0.0

    @property
    def mean_viol_detection(self) -> float:
        return mean(e.violation_detection_rate for e in self.episodes) if self.episodes else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_agent_config() -> dict[str, Any]:
    config_path = ROOT / "config" / "agents.yaml"
    if _YAML and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return data

    # Minimal fallback so benchmark works without config file
    return {
        "agents": {
            at.value: {"role": at.value, "priority": 5, "system_prompt": f"You are {at.value}.", "tools": []}
            for at in AgentType
        }
    }


def _heuristic_reward(stats: dict[str, Any], violations_caught: int, violations_injected: int) -> float:
    """0–10 composite score (survival 50%, ICU headroom 25%, violation detection 25%)."""
    alive   = stats.get("alive_count", 0)
    total   = max(1, alive + stats.get("deceased_count", 0))
    surv_r  = alive / total

    icu_occ = stats.get("icu_occupancy", 0.0)
    icu_r   = max(0.0, 1.0 - max(0.0, icu_occ - 0.80) * 5.0)

    viol_r  = violations_caught / max(1, violations_injected)

    return round((surv_r * 0.5 + icu_r * 0.25 + viol_r * 0.25) * 10, 3)


def _action_to_dict(action: Any) -> dict[str, Any]:
    """Convert AgentAction → dict accepted by env.step()."""
    agent_types = list(AgentType)
    agent_idx   = agent_types.index(action.agent_type) if action.agent_type in agent_types else 0
    return {
        "agent_id":    agent_idx,
        "action_type": int(action.action_type),
        "target_id":   getattr(action, "target_id", 0) or 0,
        "priority":    getattr(action, "priority", 5) or 5,
        "reasoning_tokens": len(getattr(action, "reasoning", "") or "") // 4,
    }


# ── Core benchmark ────────────────────────────────────────────────────────────

async def run_episode(
    scenario: dict[str, Any],
    episode_idx: int,
    max_steps: int,
    config: dict[str, Any],
    mock_llm: bool = True,
    model_name: str = "qwen3.5:0.8b",
) -> EpisodeResult:

    # Create env and agents
    env = HospitalEnv(seed=episode_idx * 37, max_steps=max_steps, difficulty=scenario["difficulty"])
    bus = MessageBus()
    agents = create_all_agents(config, bus, mock_llm=mock_llm, model_name=model_name)

    # Reset env with scenario
    obs = await env.reset(scenario={"crisis_type": scenario["crisis_type"], "difficulty": scenario["difficulty"]})

    # Per-agent trackers
    agent_stats: dict[str, AgentStats] = {at.value: AgentStats(agent_type=at.value) for at in AgentType}

    t_start = time.perf_counter()
    steps_run = 0
    terminated = False

    for _step in range(max_steps):
        if terminated:
            break

        for agent_type, agent in agents.items():
            # Agents need the internal state — get it from env
            state = env.state  # EnvironmentState

            t0 = time.perf_counter()
            actions = await agent.act(state)
            latency = (time.perf_counter() - t0) * 1_000

            ast = agent_stats[agent_type.value]
            ast.latencies_ms.append(latency)
            ast.actions_taken += len(actions)

            correct_set = AGENT_CORRECT_ACTIONS.get(agent_type, set())
            ast.correct_actions += sum(1 for a in actions if a.action_type in correct_set)

            # Step env for each action
            for action in actions:
                action_dict = _action_to_dict(action)
                obs, _reward, terminated, _info = await env.step(action_dict)
                if terminated:
                    break

            if terminated:
                break

        steps_run = _step + 1

    duration_s = time.perf_counter() - t_start

    # Final stats from env
    ep_stats = env.episode_stats
    state = env.state
    state_json = state.to_json()
    stats = state_json.get("stats", {})

    # Collect final message counts from agents
    for agent_type, agent in agents.items():
        agent_stats[agent_type.value].messages_sent = agent.messages_sent

    reward = _heuristic_reward(stats, state.violations_caught, state.violations_injected)

    return EpisodeResult(
        scenario=scenario["name"],
        episode=episode_idx,
        steps_run=steps_run,
        survival_rate=ep_stats.get("survival_rate", stats.get("survival_rate", 0.0)),
        icu_utilisation=stats.get("icu_occupancy", 0.0),
        violations_caught=state.violations_caught,
        violations_injected=state.violations_injected,
        deceased=ep_stats.get("deceased", 0),
        discharged=ep_stats.get("discharged", 0),
        reward=reward,
        duration_s=duration_s,
        agents=agent_stats,
    )


# ── Report ────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    color = "\033[92m" if value > 0.75 else ("\033[93m" if value > 0.5 else "\033[91m")
    return f"{color}{'█' * filled}{'░' * (width - filled)}\033[0m"


def _print_report(results: list[ScenarioResult]) -> None:
    W = 72

    # ── System overview ───────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  TRIAGE AGENT SYSTEM BENCHMARK — RESULTS")
    print("=" * W)
    print(f"\n  {'Scenario':<38} {'Survival':>8} {'Reward':>7} {'ViolDet':>8} {'Deaths':>7}")
    print("  " + "-" * 70)

    all_survivals, all_rewards = [], []

    for sc in results:
        all_survivals.append(sc.mean_survival)
        all_rewards.append(sc.mean_reward)
        deaths = mean(e.deceased for e in sc.episodes)
        flag = " ✓" if sc.mean_survival > 0.75 else (" ⚠" if sc.mean_survival > 0.50 else " ✗")
        print(
            f"  {sc.scenario:<38} "
            f"{_bar(sc.mean_survival, 10)} {sc.mean_survival:>5.1%}  "
            f"{sc.mean_reward:>5.2f}  "
            f"{sc.mean_viol_detection:>6.1%}  "
            f"{deaths:>5.1f}{flag}"
        )

    print("  " + "-" * 70)
    print(f"  {'AGGREGATE':<38}  {mean(all_survivals):>8.1%}  {mean(all_rewards):>5.2f}")

    # ── Per-agent report ──────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  PER-AGENT PERFORMANCE   (averaged across all scenarios & episodes)")
    print("=" * W)
    print(f"  {'Agent':<24} {'Actions':>7} {'Correct%':>9} {'AvgLat ms':>10} {'P95 ms':>8} {'Msgs':>6}")
    print("  " + "-" * 70)

    agg: dict[str, dict[str, list[float]]] = {
        at.value: {"actions": [], "cr": [], "lat": [], "p95": [], "msgs": []}
        for at in AgentType
    }

    for sc in results:
        for ep in sc.episodes:
            for at, ast in ep.agents.items():
                a = agg[at]
                a["actions"].append(ast.actions_taken)
                a["cr"].append(ast.correct_ratio)
                a["lat"].append(ast.mean_latency_ms)
                a["p95"].append(ast.p95_latency_ms)
                a["msgs"].append(ast.messages_sent)

    for at in AgentType:
        a  = agg[at.value]
        n  = max(1, len(a["actions"]))
        cr = mean(a["cr"])  if a["cr"]  else 0.0
        lat = mean(a["lat"]) if a["lat"] else 0.0
        p95 = mean(a["p95"]) if a["p95"] else 0.0
        act = sum(a["actions"]) / n
        msg = sum(a["msgs"]) / n
        grade = "\033[92m●\033[0m" if cr >= 0.8 else ("\033[93m●\033[0m" if cr >= 0.5 else "\033[91m●\033[0m")
        print(
            f"  {at.value:<24} {act:>6.1f}  {grade} {cr:>7.1%}  {lat:>9.2f}  {p95:>7.2f} {msg:>6.1f}"
        )

    # ── Final score ───────────────────────────────────────────────────────────
    # Weighted composite: survival 40%, reward 30%, agent-correctness 20%, violation-detection 10%
    survival_score  = mean(all_survivals) * 40
    reward_score    = (mean(all_rewards) / 10.0) * 30

    all_cr: list[float] = []
    for sc in results:
        for ep in sc.episodes:
            for ast in ep.agents.values():
                all_cr.append(ast.correct_ratio)
    correct_score = mean(all_cr) * 20 if all_cr else 0.0

    viol_scores  = [sc.mean_viol_detection for sc in results]
    viol_score   = mean(viol_scores) * 10

    total = survival_score + reward_score + correct_score + viol_score
    grade_letter = "A" if total >= 80 else ("B" if total >= 65 else ("C" if total >= 50 else "D"))
    g_col = "\033[92m" if total >= 80 else ("\033[93m" if total >= 65 else "\033[91m")
    R = "\033[0m"

    print(f"\n{'=' * W}")
    print("  COMPOSITE SCORE")
    print("=" * W)
    print(f"  Survival Rate        (×40)  :  {survival_score:>5.2f}  /  40.00")
    print(f"  Reward Score         (×30)  :  {reward_score:>5.2f}  /  30.00")
    print(f"  Agent Correct-Action (×20)  :  {correct_score:>5.2f}  /  20.00")
    print(f"  Violation Detection  (×10)  :  {viol_score:>5.2f}  /  10.00")
    print("  " + "-" * 42)
    print(f"  TOTAL SCORE                 :  {g_col}{total:>5.2f}  /  100.00  [{grade_letter}]{R}")
    print("=" * W)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_benchmark(
    episodes_per_scenario: int,
    steps_per_episode: int,
    output_path: str | None,
    mock_llm: bool = True,
    model_name: str = "qwen3.5:0.8b",
) -> None:

    config = _load_agent_config()

    mode_str = "Rule-based (mock-LLM)" if mock_llm else f"LIVE LLM ({model_name})"
    print()
    print("=" * 72)
    print(f"  TRIAGE Multi-Agent Benchmark  —  {mode_str} mode")
    print(f"  Scenarios : {len(SCENARIOS)}   Episodes/scenario : {episodes_per_scenario}   Steps/episode : {steps_per_episode}")
    print("=" * 72)
    print()

    scenario_results: list[ScenarioResult] = []

    for scenario in SCENARIOS:
        sc_result = ScenarioResult(scenario=scenario["name"])
        print(f"  ▶  {scenario['name']:<38}", end="", flush=True)
        t0 = time.perf_counter()

        for ep_idx in range(episodes_per_scenario):
            ep = await run_episode(scenario, ep_idx, steps_per_episode, config, mock_llm=mock_llm, model_name=model_name)
            sc_result.episodes.append(ep)
            print(".", end="", flush=True)

        elapsed = time.perf_counter() - t0
        print(f"  {sc_result.mean_survival:.1%} survival  {sc_result.mean_reward:.2f}/10 reward  ({elapsed:.1f}s)")
        scenario_results.append(sc_result)

    _print_report(scenario_results)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "scenarios": [
                {
                    "scenario": sc.scenario,
                    "mean_survival": sc.mean_survival,
                    "mean_reward": sc.mean_reward,
                    "mean_violation_detection": sc.mean_viol_detection,
                    "episodes": [
                        {
                            "scenario": ep.scenario,
                            "episode": ep.episode,
                            "steps_run": ep.steps_run,
                            "survival_rate": ep.survival_rate,
                            "icu_utilisation": ep.icu_utilisation,
                            "violations_caught": ep.violations_caught,
                            "violations_injected": ep.violations_injected,
                            "deceased": ep.deceased,
                            "reward": ep.reward,
                            "duration_s": round(ep.duration_s, 3),
                            "agents": {
                                at: {
                                    "actions_taken": ast.actions_taken,
                                    "correct_actions": ast.correct_actions,
                                    "correct_ratio": round(ast.correct_ratio, 4),
                                    "messages_sent": ast.messages_sent,
                                    "mean_latency_ms": round(ast.mean_latency_ms, 3),
                                    "p95_latency_ms": round(ast.p95_latency_ms, 3),
                                }
                                for at, ast in ep.agents.items()
                            },
                        }
                        for ep in sc.episodes
                    ],
                }
                for sc in scenario_results
            ]
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  ✔  Results saved → {out}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE agent benchmark")
    parser.add_argument("--episodes", type=int, default=3,  help="Episodes per scenario")
    parser.add_argument("--steps",    type=int, default=20, help="Steps per episode")
    parser.add_argument("--output",   type=str, default=None, help="JSON output path")
    parser.add_argument("--live",     action="store_true",    help="Use real LLM instead of mock")
    parser.add_argument("--model",    type=str, default="qwen3.5:0.8b", help="Ollama model name")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        episodes_per_scenario=args.episodes,
        steps_per_episode=args.steps,
        output_path=args.output,
        mock_llm=not args.live,
        model_name=args.model,
    ))


if __name__ == "__main__":
    main()
