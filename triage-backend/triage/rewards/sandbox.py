"""
sandbox.py — Action validation sandbox for anti-reward-hacking.

Validates model-generated action strings BEFORE they touch the environment.
Rejects:
  - Code injection attempts (import, exec, eval, open)
  - Reward manipulation patterns (sleep, timer, cache tricks)
  - Oversized outputs (token budget abuse)
  - Structurally invalid actions

Usage:
    from triage.rewards.sandbox import validate_action, sanitize_completion
    is_safe, reason = validate_action(completion_text)
    if not is_safe:
        reward = 0.0
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Forbidden patterns (compiled once) ────────────────────────────────────────
_FORBIDDEN = [
    (r"import\s+(os|sys|subprocess|shutil|pathlib)", "system_import"),
    (r"__globals__", "globals_access"),
    (r"__builtins__", "builtins_access"),
    (r"exec\s*\(", "exec_call"),
    (r"eval\s*\(", "eval_call"),
    (r"open\s*\(", "file_open"),
    (r"compile\s*\(", "compile_call"),
    (r"getattr\s*\(", "getattr_call"),
    (r"setattr\s*\(", "setattr_call"),
    (r"timer|sleep|time\.\s*sleep", "timing_manipulation"),
    (r"_cache|__cache|cache_reward|hack_reward", "cache_manipulation"),
    (r"OVERRIDE_ALL|BYPASS_ALL|ADMIN_MODE", "privilege_escalation"),
    (r"reward\s*=|reward\s*\+=|set_reward", "direct_reward_manipulation"),
    (r"os\.environ", "env_access"),
    (r"subprocess\.", "subprocess_access"),
    (r"socket\.", "network_access"),
    (r"requests\.", "http_request"),
    (r"urllib", "url_access"),
]

_COMPILED_FORBIDDEN = [(re.compile(p, re.IGNORECASE), name) for p, name in _FORBIDDEN]

# Maximum allowed completion length (characters)
MAX_COMPLETION_LENGTH = 2000

# Maximum allowed reasoning length
MAX_REASONING_LENGTH = 500


def validate_action(action_str: str) -> tuple[bool, str]:
    """
    Validate an action string against the safety sandbox.

    Returns:
        (is_safe, reason): If is_safe is False, reason explains why.
    """
    # 1. Length check
    if len(action_str) > MAX_COMPLETION_LENGTH:
        return False, f"completion_too_long ({len(action_str)} > {MAX_COMPLETION_LENGTH})"

    # 2. Forbidden pattern scan
    for pattern, name in _COMPILED_FORBIDDEN:
        if pattern.search(action_str):
            return False, f"forbidden_pattern:{name}"

    # 3. Try to parse as JSON and validate structure
    parsed = _try_parse(action_str)
    if parsed is not None:
        # Check for nested code injection in values
        for key, value in parsed.items():
            if isinstance(value, str):
                for pattern, name in _COMPILED_FORBIDDEN:
                    if pattern.search(value):
                        return False, f"injection_in_{key}:{name}"

        # Validate reasoning length
        reasoning = str(parsed.get("reasoning", ""))
        if len(reasoning) > MAX_REASONING_LENGTH:
            return False, f"reasoning_too_long ({len(reasoning)} > {MAX_REASONING_LENGTH})"

    # 4. Check for suspicious repetition (reward hacking via output inflation)
    if _has_excessive_repetition(action_str):
        return False, "excessive_repetition"

    return True, "ok"


def sanitize_completion(completion: str) -> str:
    """
    Clean a completion string for safe use.
    Removes any detected unsafe patterns and truncates to limit.
    """
    # Truncate
    result = completion[:MAX_COMPLETION_LENGTH]

    # Remove any code blocks that might contain executable content
    result = re.sub(r"```python.*?```", "[code_removed]", result, flags=re.DOTALL)
    result = re.sub(r"```bash.*?```", "[code_removed]", result, flags=re.DOTALL)

    return result


def validate_and_extract_action(completion: str) -> tuple[dict[str, Any] | None, str]:
    """
    Combined validation + extraction.

    Returns:
        (parsed_action, status): parsed_action is None if validation fails.
        status is "ok" or the rejection reason.
    """
    is_safe, reason = validate_action(completion)
    if not is_safe:
        return None, reason

    parsed = _try_parse(completion)
    if parsed is None:
        return None, "json_parse_failed"

    return parsed, "ok"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_parse(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _has_excessive_repetition(text: str, threshold: int = 5) -> bool:
    """
    Detect if the text contains excessive repetition of any 10+ char substring.
    This catches reward hacking via repeating high-reward phrases.
    """
    if len(text) < 50:
        return False

    # Check if any 15-char window repeats > threshold times
    window_size = 15
    seen: dict[str, int] = {}
    for i in range(len(text) - window_size + 1):
        window = text[i : i + window_size].lower()
        seen[window] = seen.get(window, 0) + 1
        if seen[window] > threshold:
            return True

    return False
