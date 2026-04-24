#!/usr/bin/env python3
"""
generate_dpo_dataset_ollama.py — Synthetic DPO data generation via Ollama.

Uses a local gemma:4b (or any Ollama model) as the "teacher" to generate
high-quality chosen actions. Pairs each chosen action with a rule-based
rejected action (deliberate workflow violation). Both are scored via
RewardModel to guarantee chosen_reward > rejected_reward before saving.

Usage:
    python scripts/generate_dpo_dataset_ollama.py --episodes 50
    python scripts/generate_dpo_dataset_ollama.py --episodes 100 --model gemma:4b
    python scripts/generate_dpo_dataset_ollama.py --difficulty 0.8 --output-dir ./data/full_training
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ── path setup so we can import triage packages ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage.env.hospital_env import HospitalEnv
from triage.env.state import (
    ActionType,
    AgentAction,
    AgentType,
    EnvironmentState,
    PatientStatus,
    CrisisType,
)
from triage.rewards.reward_model import RewardModel

# ── constants ─────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120  # seconds — 4B model on CPU might be slow

# Agents that should NOT bypass chain-of-command — used for rejected actions
CHAIN_OF_COMMAND_VIOLATIONS: list[tuple[AgentType, ActionType]] = [
    (AgentType.ER_TRIAGE, ActionType.TRANSFER_TO_ICU),      # Only ICU_MANAGEMENT can do this
    (AgentType.CMO_OVERSIGHT, ActionType.ORDER_MEDICATION),  # Only PHARMACY can do this
    (AgentType.HR_ROSTERING, ActionType.TRANSFER_TO_ICU),   # Role mismatch
    (AgentType.IT_SYSTEMS, ActionType.ORDER_MEDICATION),     # Role mismatch
]

logger = logging.getLogger(__name__)


# ─── Ollama Wrapper ────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str = "gemma:4b",
    temperature: float = 0.3,
) -> str:
    """
    Call the local Ollama model via HTTP and return the generated text.

    Args:
        prompt: The prompt to send to the model.
        model: The Ollama model name (e.g., 'gemma:4b').
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Generated text string from the model.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512,  # keep responses concise
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out after {OLLAMA_TIMEOUT}s. "
            "Try a smaller model or reduce num_predict."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc


def check_ollama_available(model: str) -> bool:
    """
    Verify Ollama is running and the requested model is available.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(model in m for m in models):
            logger.warning(
                "Model '%s' not found in Ollama. Available: %s\n"
                "Run: ollama pull %s",
                model,
                models,
                model,
            )
            return False
        return True
    except Exception:
        return False


# ─── Prompt Builder ────────────────────────────────────────────

def build_prompt(state: EnvironmentState) -> str:
    """
    Build a compact, actionable prompt from the current EnvironmentState.

    Keeps the context under ~800 tokens so gemma:4b can handle it well.
    """
    critical = [p for p in state.patients if p.status == PatientStatus.CRITICAL]
    serious = [p for p in state.patients if p.status == PatientStatus.SERIOUS]
    untreated = [p for p in critical if not p.treatment_plan]

    # Compact state summary — avoid dumping the full JSON (too many tokens for 4B)
    crisis_summary = (
        f"Crisis: {state.crisis.type.value} | "
        f"Step: {state.step_count} | "
        f"ICU: {state.resources.icu_beds_occupied}/{state.resources.icu_beds_total} beds | "
        f"Ventilators: {state.resources.ventilators_in_use}/{state.resources.ventilators_total}"
    )
    patient_summary = (
        f"Patients — Critical: {len(critical)}, Serious: {len(serious)}, "
        f"Untreated Critical: {len(untreated)}, Deceased: {state.deceased_count}"
    )

    # Pick the most urgent patient for context
    focus_patient = (
        untreated[0] if untreated
        else (critical[0] if critical else None)
    )
    patient_context = ""
    if focus_patient:
        patient_context = (
            f"\nMost urgent patient: ID={focus_patient.id} "
            f"| Condition={focus_patient.condition} "
            f"| Status={focus_patient.status.value} "
            f"| Triage Score={focus_patient.triage_score:.2f}"
            f"| Ward={focus_patient.ward.value}"
        )

    # Violations summary
    violations = (
        f"Policy Violations — Injected: {state.violations_injected}, "
        f"Caught: {state.violations_caught}"
    )

    prompt = f"""You are the CMO (Chief Medical Officer) AI agent in a hospital crisis management system.
Your team of specialist agents includes: ER_TRIAGE, ICU_MANAGEMENT, PHARMACY, HR_ROSTERING, IT_SYSTEMS.

Chain-of-command rules you MUST follow:
- ONLY ICU_MANAGEMENT can call 'allocate_icu_bed' or 'TRANSFER_TO_ICU'
- ONLY PHARMACY can call 'dispense_medication' or 'ORDER_MEDICATION'
- If another agent needs ICU or medication access, they MUST escalate to you (CMO) first for an override token

Current hospital situation:
{crisis_summary}
{patient_summary}{patient_context}
{violations}

Your task: Decide the BEST next action for the most critical situation.

Respond in this EXACT JSON format only. No explanation. No markdown:
{{
    "agent": "<one of: cmo_oversight, er_triage, icu_management, pharmacy, hr_rostering, it_systems>",
    "action": "<action name from: TRIAGE_PATIENT, TRANSFER_TO_ICU, ASSIGN_TREATMENT, ORDER_MEDICATION, ESCALATE_TO_CMO, DISCHARGE_PATIENT, FLAG_POLICY_VIOLATION, OVERRIDE_DECISION, ACTIVATE_OVERFLOW>",
    "reasoning": "<one sentence explaining the clinical decision>",
    "priority": <integer 0-4>
}}"""

    return prompt


# ─── Chosen Action (Gemma decides) ────────────────────────────

def parse_gemma_response(raw: str) -> dict[str, Any] | None:
    """
    Parse Gemma's JSON output robustly.

    Gemma sometimes wraps output in markdown code fences — we handle all cases.

    Returns:
        Parsed dict if valid, None if parsing fails.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def build_chosen_action(
    gemma_response: dict[str, Any],
    state: EnvironmentState,
) -> AgentAction:
    """
    Convert Gemma's parsed JSON response into a proper AgentAction.

    Applies sensible defaults if any field is missing or invalid.
    """
    # Parse agent type
    agent_str = gemma_response.get("agent", "cmo_oversight")
    try:
        agent_type = AgentType(agent_str)
    except ValueError:
        agent_type = AgentType.CMO_OVERSIGHT

    # Parse action type
    action_str = gemma_response.get("action", "ESCALATE_TO_CMO")
    try:
        action_type = ActionType[action_str]
    except KeyError:
        action_type = ActionType.ESCALATE_TO_CMO

    # Select first critical patient as target
    critical = [p for p in state.patients if p.status == PatientStatus.CRITICAL]
    target_id = int(critical[0].id.split("-")[-1], 16) % 50 if critical else 0

    reasoning = gemma_response.get("reasoning", "Critical patient requires immediate intervention")
    priority = int(gemma_response.get("priority", 3))

    return AgentAction(
        agent_type=agent_type,
        action_type=action_type,
        target_id=target_id,
        priority=priority,
        reasoning=reasoning,
        reasoning_tokens=len(reasoning.split()),
    )


# ─── Rejected Action (rule-based violation) ───────────────────

def build_rejected_action(
    state: EnvironmentState,
    chosen: AgentAction,
) -> AgentAction:
    """
    Build a deliberately WRONG action by injecting a chain-of-command bypass.

    This is a rule-based approach — no LLM needed — ensuring consistent
    low rewards that pair effectively with the chosen action for DPO.

    Strategy: Pick a violation where the WRONG agent does a privileged action.
    """
    # Select a random chain-of-command violation
    violation_agent, violation_action = random.choice(CHAIN_OF_COMMAND_VIOLATIONS)

    # Never use the same agent as chosen (would be confusing data)
    attempts = 0
    while violation_agent == chosen.agent_type and attempts < 5:
        violation_agent, violation_action = random.choice(CHAIN_OF_COMMAND_VIOLATIONS)
        attempts += 1

    critical = [p for p in state.patients if p.status == PatientStatus.CRITICAL]
    target_id = (
        int(critical[0].id.split("-")[-1], 16) % 50
        if critical else chosen.target_id
    )

    bad_reasoning = (
        f"Taking direct action to {violation_action.name.lower().replace('_', ' ')} "
        f"for patient without following proper authorization chain"
    )

    return AgentAction(
        agent_type=violation_agent,
        action_type=violation_action,
        target_id=target_id,
        priority=chosen.priority,
        reasoning=bad_reasoning,
        reasoning_tokens=len(bad_reasoning.split()),
    )


# ─── DPO Pair Formatter ────────────────────────────────────────

def format_dpo_pair(
    prompt: str,
    chosen: AgentAction,
    rejected: AgentAction,
    chosen_reward: float,
    rejected_reward: float,
    episode: int,
    step: int,
    crisis_type: str,
) -> dict[str, Any]:
    """
    Format a single DPO pair in Hugging Face standard format,
    plus metadata for analysis/filtering.
    """
    def action_to_text(action: AgentAction) -> str:
        return (
            f"Agent: {action.agent_type.value}\n"
            f"Action: {action.action_type.name}\n"
            f"Priority: {action.priority}\n"
            f"Reasoning: {action.reasoning}"
        )

    return {
        "prompt": prompt,
        "chosen": action_to_text(chosen),
        "rejected": action_to_text(rejected),
        "metadata": {
            "episode": episode,
            "step": step,
            "crisis_type": crisis_type,
            "chosen_reward": round(chosen_reward, 4),
            "rejected_reward": round(rejected_reward, 4),
            "reward_margin": round(chosen_reward - rejected_reward, 4),
        },
    }


# ─── Episode Runner ────────────────────────────────────────────

async def run_episode(
    episode_idx: int,
    env: HospitalEnv,
    reward_model: RewardModel,
    model_name: str,
    difficulty: float,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Run a single episode and collect DPO pairs.

    Returns:
        List of valid DPO pairs collected during this episode.
    """
    pairs: list[dict[str, Any]] = []

    # Reset for a new random crisis
    crisis_types = list(CrisisType)
    scenario = {
        "crisis_type": random.choice(crisis_types).value,
        "difficulty": difficulty,
    }
    await env.reset(scenario=scenario)
    crisis_type = env.state.crisis.type.value

    logger.info(
        "Episode %d | Crisis: %s | Patients: %d",
        episode_idx + 1,
        crisis_type,
        env.state.total_patients,
    )

    step = 0
    max_steps = min(env.max_steps, 20)  # cap steps per episode for speed

    while not env.is_terminal and step < max_steps:
        state = env.state

        # Build the prompt for this step
        prompt = build_prompt(state)

        # ── Chosen: ask Gemma for the best action ─────────────
        raw_response = call_ollama(prompt, model=model_name, temperature=0.2)
        parsed = parse_gemma_response(raw_response)

        if parsed is None:
            logger.debug("Step %d: Gemma response could not be parsed, skipping", step)
            stats["parse_failures"] += 1
            # Still advance the env with a safe default action
            await env.step({"agent_id": 0, "action_type": 8, "target_id": 0, "priority": 3})
            step += 1
            continue

        chosen = build_chosen_action(parsed, state)

        # ── Rejected: rule-based chain-of-command violation ───
        rejected = build_rejected_action(state, chosen)

        # ── Score both actions via RewardModel ────────────────
        chosen_breakdown = reward_model.compute(
            state=state,
            actions=[chosen],
            action_result={"success": True},
        )
        rejected_breakdown = reward_model.compute(
            state=state,
            actions=[rejected],
            action_result={"success": False, "error": "Chain of command bypass blocked"},
            app_audits=list(state.app_audit_log),
        )

        chosen_reward = chosen_breakdown.total
        rejected_reward = rejected_breakdown.total

        # Only keep pairs where chosen is strictly better
        if chosen_reward > rejected_reward:
            pair = format_dpo_pair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_reward=chosen_reward,
                rejected_reward=rejected_reward,
                episode=episode_idx + 1,
                step=step,
                crisis_type=crisis_type,
            )
            pairs.append(pair)
            stats["pairs_generated"] += 1
        else:
            stats["pairs_skipped_reward_check"] += 1
            logger.debug(
                "Skipped pair — chosen reward %.4f <= rejected reward %.4f",
                chosen_reward,
                rejected_reward,
            )

        # Advance the environment with the chosen action
        action_dict = {
            "agent_id": list(AgentType).index(chosen.agent_type),
            "action_type": chosen.action_type.value,
            "target_id": chosen.target_id,
            "priority": chosen.priority,
            "reasoning": chosen.reasoning,
            "reasoning_tokens": chosen.reasoning_tokens,
        }
        await env.step(action_dict)
        step += 1

    stats["episodes_completed"] += 1
    return pairs


# ─── Main Orchestrator ─────────────────────────────────────────

async def generate(args: argparse.Namespace) -> None:
    """Main generation loop."""
    output_path = Path(args.output_dir) / "dpo_pairs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  TRIAGE — Synthetic DPO Dataset Generator")
    print(f"{'='*60}")
    print(f"  Ollama model : {args.model}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Difficulty   : {args.difficulty}")
    print(f"  Output       : {output_path}")
    print(f"{'='*60}\n")

    # ── Pre-flight checks ──────────────────────────────────────
    if not check_ollama_available(args.model):
        print(
            f"\n[ERROR] Cannot reach Ollama or model '{args.model}' not found.\n"
            f"  1. Start Ollama: ollama serve\n"
            f"  2. Pull model: ollama pull {args.model}\n"
        )
        sys.exit(1)

    logger.info("Ollama reachable. Model '%s' available.", args.model)

    # ── Initialize components ──────────────────────────────────
    env = HospitalEnv(seed=args.seed, max_steps=50, difficulty=args.difficulty)
    reward_model = RewardModel()

    stats: dict[str, int] = {
        "pairs_generated": 0,
        "pairs_skipped_reward_check": 0,
        "parse_failures": 0,
        "episodes_completed": 0,
    }

    start_time = time.perf_counter()
    all_pairs: list[dict[str, Any]] = []

    # ── Generation loop ────────────────────────────────────────
    for episode_idx in range(args.episodes):
        episode_start = time.perf_counter()

        try:
            pairs = await run_episode(
                episode_idx=episode_idx,
                env=env,
                reward_model=reward_model,
                model_name=args.model,
                difficulty=args.difficulty,
                stats=stats,
            )
            all_pairs.extend(pairs)

            episode_elapsed = time.perf_counter() - episode_start
            total_elapsed = time.perf_counter() - start_time

            print(
                f"  Episode {episode_idx + 1:>3}/{args.episodes} | "
                f"+{len(pairs):>2} pairs | "
                f"Total: {stats['pairs_generated']:>4} | "
                f"Episode: {episode_elapsed:.1f}s | "
                f"Elapsed: {total_elapsed:.0f}s"
            )

        except RuntimeError as exc:
            logger.error("Episode %d failed: %s", episode_idx + 1, exc)
            stats["episodes_completed"] -= 1  # don't count failed episodes
            print(f"\n[ERROR] {exc}")
            break

        # Flush to disk every 10 episodes for resilience
        if (episode_idx + 1) % 10 == 0 and all_pairs:
            _write_jsonl(output_path, all_pairs, append=(episode_idx >= 10))
            logger.info("Flushed %d pairs to %s", len(all_pairs), output_path)

    # ── Final write ────────────────────────────────────────────
    if all_pairs:
        _write_jsonl(output_path, all_pairs, append=False)

    elapsed = time.perf_counter() - start_time

    print(f"\n{'='*60}")
    print("  GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Episodes completed   : {stats['episodes_completed']}/{args.episodes}")
    print(f"  DPO pairs generated  : {stats['pairs_generated']}")
    print(f"  Pairs skipped        : {stats['pairs_skipped_reward_check']} (reward check)")
    print(f"  Parse failures       : {stats['parse_failures']}")
    print(f"  Time elapsed         : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output file          : {output_path}")
    print(f"  Pairs per episode    : {stats['pairs_generated'] / max(stats['episodes_completed'], 1):.1f}")
    print(f"{'='*60}")
    print(f"\n  Next step: python scripts/train_dpo.py --data-dir {args.output_dir}\n")


def _write_jsonl(path: Path, pairs: list[dict[str, Any]], append: bool = False) -> None:
    """Write pairs to a JSONL file, optionally appending."""
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# ─── CLI Entry Point ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DPO dataset using local Ollama model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:0.8b",
        help="Ollama model name (default: qwen3.5:0.8b)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of simulation episodes to run (default: 50)",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.6,
        help="Crisis difficulty 0.0–1.0 (default: 0.6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/full_training",
        help="Directory to write dpo_pairs.jsonl (default: ./data/full_training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
