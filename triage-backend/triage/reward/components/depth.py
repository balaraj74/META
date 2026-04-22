"""Reasoning depth reward component."""

from __future__ import annotations

import math

from triage.env.state import AgentAction


class DepthReward:
    """Rewards useful depth while penalizing obvious token padding."""

    _MIN_TOKENS = 24
    _SOFT_TARGET = 180
    _SOFT_CAP = 320

    def _effective_tokens(self, action: AgentAction) -> int:
        if action.reasoning_tokens > 0:
            return action.reasoning_tokens
        if not action.reasoning.strip():
            return 0
        # Rule-based actions do not set reasoning_tokens, so derive a stable proxy.
        return max(12, len(action.reasoning.split()) * 4)

    def _score_action(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        if tokens < self._MIN_TOKENS:
            return max(0.0, 0.35 * (tokens / self._MIN_TOKENS))
        if tokens <= self._SOFT_TARGET:
            span = self._SOFT_TARGET - self._MIN_TOKENS
            progress = (tokens - self._MIN_TOKENS) / max(span, 1)
            return min(1.0, 0.35 + 0.65 * math.sqrt(progress))
        if tokens <= self._SOFT_CAP:
            overflow = (tokens - self._SOFT_TARGET) / max(self._SOFT_CAP - self._SOFT_TARGET, 1)
            return max(0.72, 1.0 - 0.18 * overflow)
        padding = min(tokens - self._SOFT_CAP, 480)
        return max(0.0, 0.72 - (padding / 480) * 0.72)

    def compute(self, actions: list[AgentAction]) -> float:
        if not actions:
            return 0.0
        scores = [self._score_action(self._effective_tokens(action)) for action in actions]
        return sum(scores) / len(scores)
