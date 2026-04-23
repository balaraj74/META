#!/usr/bin/env python3
"""
build_grpo_dataset.py — Generate prompt dataset for GRPO training.

GRPO needs prompts, not input/output pairs. Each prompt is a crisis state
snapshot that the model must respond to with a valid action JSON.

This script:
  1. Creates a TriageOpenEnv
  2. Rolls out N episodes in mock mode (no LLM needed)
  3. At each step, captures the state and builds a prompt
  4. Saves the prompt dataset as JSONL

Usage:
    python scripts/build_grpo_dataset.py --episodes 100 --output data/grpo/train.jsonl
    python scripts/build_grpo_dataset.py --episodes 20 --output data/grpo/eval.jsonl --seed 999

Output format (one JSON per line):
    {
        "prompt": "You are the ER_TRIAGE agent...",
        "crisis_type": "mass_casualty",
        "difficulty": 0.6,
        "step": 12,
        "state": { ... }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.env.openenv_adapter import TriageOpenEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Agent types to generate prompts for
AGENT_TYPES = [
    "er_triage",
    "icu_management",
    "pharmacy",
    "cmo_oversight",
    "hr_rostering",
    "it_systems",
]

# Crisis types and difficulty tiers
CRISIS_TYPES = ["mass_casualty", "outbreak", "equipment_failure", "staff_shortage"]
DIFFICULTY_TIERS = [0.2, 0.4, 0.6, 0.8, 1.0]


def build_dataset(
    n_episodes: int,
    output_path: Path,
    seed: int = 42,
    max_steps_per_episode: int = 30,
    agents_per_step: int = 2,
) -> int:
    """
    Generate prompt dataset by rolling out episodes.

    Returns:
        Total number of prompts generated.
    """
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_prompts = 0

    with open(output_path, "w") as f:
        for ep_idx in range(n_episodes):
            crisis = rng.choice(CRISIS_TYPES)
            difficulty = rng.choice(DIFFICULTY_TIERS)

            env = TriageOpenEnv(
                seed=seed + ep_idx,
                max_steps=max_steps_per_episode,
                difficulty=difficulty,
                crisis_type=crisis,
            )

            try:
                obs = env.reset()
            except Exception as exc:
                logger.warning("Episode %d reset failed: %s", ep_idx, exc)
                continue

            for step_idx in range(max_steps_per_episode):
                # Pick random agents to generate prompts for
                selected_agents = rng.sample(
                    AGENT_TYPES,
                    min(agents_per_step, len(AGENT_TYPES)),
                )

                for agent_type in selected_agents:
                    try:
                        prompt = env.state_to_prompt(agent_type)
                    except Exception:
                        continue

                    record = {
                        "prompt": prompt,
                        "crisis_type": crisis,
                        "difficulty": difficulty,
                        "step": step_idx,
                        "episode": ep_idx,
                        "agent_type": agent_type,
                        "state": obs,
                    }
                    f.write(json.dumps(record) + "\n")
                    total_prompts += 1

                # Take a random action to advance the environment
                action = _random_action(agent_type, obs, rng)
                try:
                    obs, reward, done, info = env.step(action)
                except Exception:
                    break

                if done:
                    break

            if (ep_idx + 1) % 10 == 0:
                logger.info(
                    "Episode %d/%d  |  %d prompts  |  crisis=%s  diff=%.1f",
                    ep_idx + 1, n_episodes, total_prompts, crisis, difficulty,
                )

    logger.info("Dataset saved to %s  (%d prompts)", output_path, total_prompts)
    return total_prompts


def _random_action(
    agent_type: str,
    obs: dict,
    rng: random.Random,
) -> dict:
    """Generate a plausible random action for environment advancement."""
    action_map = {
        "er_triage": ["TRIAGE_PATIENT", "ASSIGN_TREATMENT", "UPDATE_EHR"],
        "icu_management": ["TRANSFER_TO_ICU", "TRANSFER_TO_WARD", "ACTIVATE_OVERFLOW"],
        "pharmacy": ["ORDER_MEDICATION", "FLAG_POLICY_VIOLATION"],
        "cmo_oversight": ["OVERRIDE_DECISION", "ACTIVATE_OVERFLOW"],
        "hr_rostering": ["REQUEST_STAFF", "FLAG_POLICY_VIOLATION"],
        "it_systems": ["UPDATE_EHR", "FLAG_POLICY_VIOLATION", "VERIFY_INSURANCE"],
    }

    actions = action_map.get(agent_type, ["TRIAGE_PATIENT"])
    patients = obs.get("patients_summary", [])
    target_id = patients[0]["id"] if patients else 0

    return {
        "agent_type": agent_type,
        "action_type": rng.choice(actions),
        "target_id": target_id,
        "priority": rng.randint(1, 5),
        "reasoning": "Random action for dataset generation.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build GRPO prompt dataset from environment rollouts"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes to roll out (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default="data/grpo/train.jsonl",
        help="Output JSONL file path (default: data/grpo/train.jsonl)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max steps per episode (default: 30)"
    )
    parser.add_argument(
        "--agents-per-step", type=int, default=2,
        help="Number of agent prompts per step (default: 2)"
    )

    args = parser.parse_args()
    output = Path(args.output)

    total = build_dataset(
        n_episodes=args.episodes,
        output_path=output,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
        agents_per_step=args.agents_per_step,
    )

    print(f"\n{'═' * 50}")
    print(f"  GRPO Dataset Ready")
    print(f"  Prompts: {total}")
    print(f"  Output:  {output}")
    print(f"  Size:    {output.stat().st_size / 1024:.1f} KB")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
