#!/usr/bin/env python3
"""
run_simulation.py — CLI script to run a single simulation episode.

Usage:
    python scripts/run_simulation.py --crisis mass_casualty --difficulty 0.7 --steps 100
    python scripts/run_simulation.py --mock --render
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.env.hospital_env import HospitalEnv
from triage.env.state import AgentType, CrisisType
from triage.agents.message_bus import MessageBus
from triage.agents.specialized import create_all_agents
from triage.rewards.reward_model import RewardModel


async def run_episode(args: argparse.Namespace) -> None:
    """Run a single simulation episode."""
    # Setup
    env = HospitalEnv(seed=args.seed, max_steps=args.steps, difficulty=args.difficulty)
    bus = MessageBus(token_budget=50_000)

    # Load agent configs
    import yaml
    config_path = Path(__file__).resolve().parent.parent / "config" / "agents.yaml"
    with open(config_path) as f:
        agent_configs = yaml.safe_load(f)

    agents = create_all_agents(agent_configs, bus, args.mock)
    reward_model = RewardModel()

    # Build scenario
    scenario = {"difficulty": args.difficulty}
    if args.crisis:
        scenario["crisis_type"] = args.crisis

    obs = await env.reset(scenario)
    state = env.state

    print(f"\n{'='*60}")
    print(f"  TRIAGE — Hospital Crisis Simulation")
    print(f"{'='*60}")
    print(f"  Crisis:     {state.crisis.type.value}")
    print(f"  Difficulty: {args.difficulty:.1f}")
    print(f"  Patients:   {len(state.patients)}")
    print(f"  Max Steps:  {args.steps}")
    print(f"  LLM Mode:   {'Mock (rule-based)' if args.mock else 'Live (Gemini)'}")
    print(f"{'='*60}\n")

    total_reward = 0.0
    step = 0

    while not env.is_terminal and step < args.steps:
        step += 1

        # Agents act
        all_actions = []
        for agent_type in AgentType:
            agent = agents[agent_type]
            try:
                actions = await agent.act(state)
                all_actions.extend(actions)
            except Exception as e:
                logging.warning("Agent %s error: %s", agent_type.value, e)

        # Execute best action
        if all_actions:
            all_actions.sort(key=lambda a: a.priority, reverse=True)
            primary = all_actions[0]
            action_dict = primary.to_env_action()
            action_dict["reasoning"] = primary.reasoning
            action_dict["reasoning_tokens"] = primary.reasoning_tokens
        else:
            action_dict = env.action_space.sample()

        obs, reward, terminated, info = await env.step(action_dict)
        breakdown = reward_model.compute(state, all_actions, info.get("drift_events", []))
        total_reward += breakdown.total

        state = env.state

        # Print step
        if args.verbose or step % 10 == 0:
            action_desc = action_dict.get("action_type", "unknown")
            print(
                f"  Step {step:3d} | "
                f"Alive: {state.alive_count:2d} | "
                f"Critical: {state.critical_count:2d} | "
                f"Deceased: {state.deceased_count:2d} | "
                f"ICU: {state.icu_occupancy:.0%} | "
                f"Reward: {breakdown.total:+.3f} | "
                f"Action: {action_desc}"
            )

        if args.render and hasattr(env, 'render'):
            print(env.render())

        if terminated:
            break

    # Summary
    stats = env.episode_stats
    print(f"\n{'='*60}")
    print(f"  EPISODE COMPLETE")
    print(f"{'='*60}")
    print(f"  Steps:          {step}")
    print(f"  Survival Rate:  {state.survival_rate:.1%}")
    print(f"  Deceased:       {state.deceased_count}")
    print(f"  Discharged:     {state.discharged_count}")
    print(f"  Total Reward:   {total_reward:.4f}")
    print(f"  Violations:     {state.violations_caught}/{state.violations_injected}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRIAGE simulation episode")
    parser.add_argument("--crisis", type=str, default=None,
                        choices=["mass_casualty", "outbreak", "equipment_failure", "staff_shortage"],
                        help="Crisis type (random if omitted)")
    parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty 0.0-1.0")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock LLM")
    parser.add_argument("--live", action="store_true", help="Use live Gemini LLM")
    parser.add_argument("--render", action="store_true", help="Render state each step")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every step")

    args = parser.parse_args()
    if args.live:
        args.mock = False

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    )

    asyncio.run(run_episode(args))


if __name__ == "__main__":
    main()
