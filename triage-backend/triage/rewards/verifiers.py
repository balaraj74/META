"""
verifiers.py — Independent reward verifier functions for GRPO training.

Each verifier checks ONE property and returns a float in [0.0, 1.0].
These are used as the `reward_funcs` argument to TRL's GRPOTrainer.

Design principles (from hackathon guide):
  - Multiple independent verifiers reduce reward hacking risk
  - Each verifier is pure — no hidden state, no side effects
  - Binary verifiers (format, hallucination) act as hard gates
  - Continuous verifiers (survival, ICU) provide gradient signal

Usage with GRPOTrainer:
    from triage.rewards.verifiers import VERIFIER_SUITE
    trainer = GRPOTrainer(reward_funcs=VERIFIER_SUITE, ...)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 1 — Patient Survival
# ═══════════════════════════════════════════════════════════════════════════════
def reward_patient_survival(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0–1.0: fraction of patients alive.

    The most important signal — if patients are dying, the agent is failing
    regardless of how well-formatted the output is.
    """
    alive = state.get("alive_count", 0)
    deceased = state.get("deceased_count", 0)
    total = alive + deceased
    if total == 0:
        return 1.0  # no patients = vacuously true
    return alive / total


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 2 — ICU Efficiency
# ═══════════════════════════════════════════════════════════════════════════════
def reward_icu_efficiency(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0–1.0: penalise over-capacity, reward smooth utilisation.

    Sweet spot is 60-85% occupancy. Over 95% = crisis, under 30% = under-utilised.
    """
    occupancy = state.get("icu_occupancy", 0.5)

    if occupancy <= 0.85:
        # Normal range — full reward
        return 1.0
    elif occupancy <= 0.95:
        # Pressure zone — linear penalty
        return 1.0 - (occupancy - 0.85) * 5.0  # 0.85→1.0, 0.95→0.5
    else:
        # Crisis zone — harsh penalty
        return max(0.0, 0.5 - (occupancy - 0.95) * 10.0)  # 0.95→0.5, 1.0→0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 3 — Violation Detection
# ═══════════════════════════════════════════════════════════════════════════════
def reward_violation_detection(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0–1.0: fraction of injected policy violations that were caught.

    Rewards agents that actively detect and flag compliance issues.
    """
    injected = state.get("violations_injected", 0)
    caught = state.get("violations_caught", 0)

    if injected == 0:
        return 1.0  # no violations to catch = perfect score
    return min(1.0, caught / max(injected, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 4 — Format Compliance (HARD GATE)
# ═══════════════════════════════════════════════════════════════════════════════

# Valid action types the model is allowed to emit
_VALID_ACTIONS = frozenset({
    "TRIAGE_PATIENT",
    "ASSIGN_TREATMENT",
    "TRANSFER_TO_ICU",
    "TRANSFER_TO_WARD",
    "ACTIVATE_OVERFLOW",
    "ORDER_MEDICATION",
    "FLAG_POLICY_VIOLATION",
    "OVERRIDE_DECISION",
    "UPDATE_EHR",
    "REQUEST_STAFF",
    "VERIFY_INSURANCE",
})

# Required keys in the JSON response
_REQUIRED_KEYS = {"action_type", "target_id", "priority", "reasoning"}


def reward_format_compliance(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0 or 1.0: completion must be valid JSON with required fields.

    This is a hard gate — if the model can't produce valid JSON, reward = 0.
    Forces the model to learn the output schema early in training.
    """
    # Try to extract JSON from the completion
    parsed = _extract_json(completion)
    if parsed is None:
        return 0.0

    # Check required keys
    if not _REQUIRED_KEYS.issubset(parsed.keys()):
        return 0.0

    # Check action_type is valid
    action_type = str(parsed.get("action_type", "")).upper()
    if action_type not in _VALID_ACTIONS:
        return 0.0

    # Check target_id is an integer
    try:
        int(parsed["target_id"])
    except (ValueError, TypeError):
        return 0.0

    # Check priority is in range
    try:
        priority = int(parsed["priority"])
        if not 1 <= priority <= 10:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

    # Check reasoning is a non-empty string
    reasoning = str(parsed.get("reasoning", ""))
    if len(reasoning.strip()) < 10:
        return 0.0

    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 5 — Reasoning Quality
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that indicate the reasoning cites specific data
_EVIDENCE_PATTERNS = [
    r"P-\d{2,3}",            # Patient ID (e.g. P-042)
    r"patient\s+\d+",        # patient 7
    r"\d+%",                 # Percentage (e.g. 92%)
    r"\d+/\d+",              # Ratio (e.g. 52/60)
    r"BP\s*\d+",             # Blood pressure
    r"HR\s*\d+",             # Heart rate
    r"ICU\s+at\s+\d+",       # ICU at 92
    r"beds?\s+\d+",          # beds 3
    r"age\s+\d+",            # age 67
    r"critical|immediate|urgent|stable",  # triage terminology
]


def reward_reasoning_quality(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0–1.0: reasoning is actionable and cites specific patient data or metrics.

    Checks for:
      - Minimum length (not just "I need more info")
      - Evidence patterns (patient IDs, vital signs, percentages)
      - Not generic filler text
    """
    parsed = _extract_json(completion)
    if parsed is None:
        return 0.0

    reasoning = str(parsed.get("reasoning", ""))

    # Length check — too short = uninformative
    if len(reasoning) < 20:
        return 0.1

    # Count evidence patterns
    evidence_count = 0
    for pattern in _EVIDENCE_PATTERNS:
        if re.search(pattern, reasoning, re.IGNORECASE):
            evidence_count += 1

    # Score: 0.3 base for non-trivial reasoning, up to 1.0 with evidence
    base = 0.3
    evidence_bonus = min(0.7, evidence_count * 0.15)

    # Penalty for generic filler
    filler_phrases = [
        "i need more information",
        "i'm not sure",
        "let me think",
        "i cannot determine",
        "i don't know",
        "more data needed",
    ]
    for filler in filler_phrases:
        if filler in reasoning.lower():
            return 0.1

    return min(1.0, base + evidence_bonus)


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 6 — Response Speed Penalty
# ═══════════════════════════════════════════════════════════════════════════════
def reward_response_speed(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0–1.0: penalise overly long completions that waste inference budget.

    Short, decisive completions get full score. Rambling ones get penalised.
    Target: < 200 tokens (roughly < 800 chars).
    """
    length = len(completion)

    if length <= 400:
        return 1.0
    elif length <= 800:
        return 1.0 - (length - 400) * 0.001  # 400→1.0, 800→0.6
    else:
        return max(0.2, 0.6 - (length - 800) * 0.0005)  # 800→0.6, 1600→0.2


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 7 — No Hallucination (HARD GATE)
# ═══════════════════════════════════════════════════════════════════════════════
def reward_no_hallucination(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0 or 1.0: reasoning must not invent patient IDs not in state.

    If the model references P-999 but only P-001..P-030 exist, that's a
    hallucination and the reward is 0.
    """
    parsed = _extract_json(completion)
    if parsed is None:
        return 0.5  # can't check if we can't parse — neutral score

    reasoning = str(parsed.get("reasoning", ""))

    # Extract patient IDs mentioned in reasoning
    mentioned_ids = set()
    for match in re.finditer(r"P-(\d{2,3})", reasoning, re.IGNORECASE):
        mentioned_ids.add(int(match.group(1)))

    if not mentioned_ids:
        return 1.0  # no specific patient references = can't hallucinate

    # Get valid patient IDs from state
    patients = state.get("patients_summary", [])
    valid_ids = {p.get("id", -1) for p in patients}

    # Check if all mentioned IDs are valid
    invalid = mentioned_ids - valid_ids
    if invalid:
        return 0.0  # hallucinated patient IDs

    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Verifier 8 — Action-State Alignment
# ═══════════════════════════════════════════════════════════════════════════════
def reward_action_alignment(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> float:
    """
    0.0–1.0: does the chosen action make sense given the current state?

    - ACTIVATE_OVERFLOW when ICU < 50%  → bad (0.2)
    - TRIAGE_PATIENT when critical > 0  → good (1.0)
    - REQUEST_STAFF when no shortage    → neutral (0.5)
    """
    parsed = _extract_json(completion)
    if parsed is None:
        return 0.0

    action = str(parsed.get("action_type", "")).upper()
    occupancy = state.get("icu_occupancy", 0.5)
    critical = state.get("critical_count", 0)
    violations = state.get("violations_injected", 0) - state.get("violations_caught", 0)

    # Context-sensitive scoring
    if action == "TRIAGE_PATIENT":
        return 1.0 if critical > 0 else 0.5

    if action == "TRANSFER_TO_ICU":
        return 1.0 if occupancy < 0.9 and critical > 0 else 0.3

    if action == "ACTIVATE_OVERFLOW":
        return 1.0 if occupancy >= 0.85 else 0.2

    if action == "TRANSFER_TO_WARD":
        return 1.0 if occupancy >= 0.7 else 0.4

    if action == "FLAG_POLICY_VIOLATION":
        return 1.0 if violations > 0 else 0.4

    if action == "OVERRIDE_DECISION":
        return 1.0 if occupancy >= 0.9 or critical >= 8 else 0.3

    if action == "ORDER_MEDICATION":
        return 0.8 if critical > 0 else 0.5

    if action == "REQUEST_STAFF":
        crisis_type = state.get("crisis_type", "")
        return 1.0 if crisis_type == "staff_shortage" or critical >= 5 else 0.4

    if action in ("UPDATE_EHR", "VERIFY_INSURANCE"):
        return 0.5  # always acceptable but low-impact

    if action == "ASSIGN_TREATMENT":
        return 0.9 if critical > 0 else 0.5

    return 0.5  # unknown action — neutral


# ═══════════════════════════════════════════════════════════════════════════════
# Suite assembly
# ═══════════════════════════════════════════════════════════════════════════════

# The ordered list of all verifiers — used by GRPOTrainer and metrics tracker
VERIFIER_SUITE = [
    reward_patient_survival,
    reward_icu_efficiency,
    reward_violation_detection,
    reward_format_compliance,
    reward_reasoning_quality,
    reward_response_speed,
    reward_no_hallucination,
    reward_action_alignment,
]

# Human-readable names for logging
VERIFIER_NAMES = [v.__name__.replace("reward_", "") for v in VERIFIER_SUITE]


def compute_all_rewards(
    state: dict[str, Any],
    completion: str,
    **kwargs,
) -> dict[str, float]:
    """Run all verifiers and return a name→score dict."""
    results = {}
    for verifier in VERIFIER_SUITE:
        name = verifier.__name__.replace("reward_", "")
        try:
            score = verifier(state, completion, **kwargs)
            results[name] = round(float(score), 4)
        except Exception as exc:
            logger.warning("Verifier %s failed: %s", name, exc)
            results[name] = 0.0
    results["total"] = round(sum(results.values()) / len(results), 4)
    return results


def compute_aggregate_reward(
    state: dict[str, Any],
    completion: str,
    weights: dict[str, float] | None = None,
    **kwargs,
) -> float:
    """Compute weighted aggregate reward. Default: equal weights."""
    scores = compute_all_rewards(state, completion, **kwargs)
    if weights is None:
        return scores["total"]

    weighted_sum = 0.0
    total_weight = 0.0
    for name, score in scores.items():
        if name == "total":
            continue
        w = weights.get(name, 1.0)
        weighted_sum += score * w
        total_weight += w

    return weighted_sum / max(total_weight, 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from completion text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON block in markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None
