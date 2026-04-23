"""
curriculum.py — Difficulty curriculum scheduler for GRPO training.

RL only works when the probability of getting a good answer is > 0.
If we start with max-difficulty mass casualty events, the model gets zero
reward and learns nothing. This scheduler ramps difficulty based on
observed performance.

Stages:
  1. warm_up       → equipment_failure at 0.2 difficulty (easy)
  2. easy          → outbreak at 0.4 (some complexity)
  3. medium        → staff_shortage at 0.6 (coordination needed)
  4. hard          → mass_casualty at 0.8 (full pressure)
  5. full_spectrum → all crisis types at 1.0 (exam mode)

Usage:
    scheduler = CurriculumScheduler()
    stage = scheduler.current_stage
    env = TriageOpenEnv(difficulty=stage.difficulty, crisis_type=stage.crisis_type)
    ...
    scheduler.report_reward(0.72)
    if scheduler.should_advance():
        scheduler.advance()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    """One curriculum stage with target difficulty and crisis type."""

    name: str
    difficulty: float
    crisis_types: list[str]
    min_reward_to_advance: float
    min_episodes_at_stage: int = 10

    @property
    def crisis_type(self) -> str | None:
        """Return a single crisis type for environments that need it, or None for random."""
        if len(self.crisis_types) == 1:
            return self.crisis_types[0]
        return None  # None = random from the list


# ── Default curriculum stages ─────────────────────────────────────────────────

DEFAULT_STAGES = [
    Stage(
        name="warm_up",
        difficulty=0.2,
        crisis_types=["equipment_failure"],
        min_reward_to_advance=0.40,
        min_episodes_at_stage=5,
    ),
    Stage(
        name="easy",
        difficulty=0.4,
        crisis_types=["outbreak"],
        min_reward_to_advance=0.50,
        min_episodes_at_stage=8,
    ),
    Stage(
        name="medium",
        difficulty=0.6,
        crisis_types=["staff_shortage"],
        min_reward_to_advance=0.55,
        min_episodes_at_stage=10,
    ),
    Stage(
        name="hard",
        difficulty=0.8,
        crisis_types=["mass_casualty"],
        min_reward_to_advance=0.60,
        min_episodes_at_stage=10,
    ),
    Stage(
        name="full_spectrum",
        difficulty=1.0,
        crisis_types=["mass_casualty", "outbreak", "equipment_failure", "staff_shortage"],
        min_reward_to_advance=0.0,  # terminal stage — never advances
        min_episodes_at_stage=999999,
    ),
]


class CurriculumScheduler:
    """
    Manages difficulty progression during GRPO training.

    The scheduler tracks recent rewards and advances to the next stage
    when performance consistently exceeds the current stage's threshold.
    """

    def __init__(
        self,
        stages: list[Stage] | None = None,
        window_size: int = 10,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.stages = stages or list(DEFAULT_STAGES)
        self._stage_idx = 0
        self._window_size = window_size
        self._reward_history: list[float] = []
        self._total_episodes = 0
        self._episodes_at_stage = 0
        self._advancement_log: list[dict[str, Any]] = []
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_stage(self) -> Stage:
        return self.stages[self._stage_idx]

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def is_final_stage(self) -> bool:
        return self._stage_idx >= len(self.stages) - 1

    @property
    def recent_mean_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        window = self._reward_history[-self._window_size :]
        return sum(window) / len(window)

    # ── Core API ──────────────────────────────────────────────────────────────

    def report_reward(self, reward: float) -> None:
        """Record a reward from a completed episode."""
        self._reward_history.append(float(reward))
        self._total_episodes += 1
        self._episodes_at_stage += 1

    def should_advance(self) -> bool:
        """Check if the scheduler should move to the next stage."""
        if self.is_final_stage:
            return False

        stage = self.current_stage
        if self._episodes_at_stage < stage.min_episodes_at_stage:
            return False

        # Check if recent rewards exceed the threshold
        if len(self._reward_history) < self._window_size:
            return False

        return self.recent_mean_reward >= stage.min_reward_to_advance

    def advance(self) -> Stage:
        """Advance to the next stage. Returns the new stage."""
        if self.is_final_stage:
            logger.info("Already at final stage: %s", self.current_stage.name)
            return self.current_stage

        old_stage = self.current_stage
        self._advancement_log.append({
            "from_stage": old_stage.name,
            "to_stage": self.stages[self._stage_idx + 1].name,
            "episode": self._total_episodes,
            "mean_reward": round(self.recent_mean_reward, 4),
            "episodes_at_stage": self._episodes_at_stage,
        })

        self._stage_idx += 1
        self._episodes_at_stage = 0
        new_stage = self.current_stage

        logger.info(
            "Curriculum advanced: %s → %s (episode %d, mean_reward=%.3f)",
            old_stage.name,
            new_stage.name,
            self._total_episodes,
            self.recent_mean_reward,
        )

        if self._checkpoint_dir:
            self.save_checkpoint()

        return new_stage

    def step(self, reward: float) -> Stage:
        """Convenience: report reward + auto-advance if ready. Returns current stage."""
        self.report_reward(reward)
        if self.should_advance():
            self.advance()
        return self.current_stage

    # ── Env config for current stage ──────────────────────────────────────────

    def env_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to TriageOpenEnv for the current stage."""
        import random
        stage = self.current_stage
        crisis = stage.crisis_type or random.choice(stage.crisis_types)
        return {
            "difficulty": stage.difficulty,
            "crisis_type": crisis,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path | None = None) -> None:
        out = path or (self._checkpoint_dir / "curriculum_state.json" if self._checkpoint_dir else None)
        if out is None:
            return
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "stage_idx": self._stage_idx,
            "total_episodes": self._total_episodes,
            "episodes_at_stage": self._episodes_at_stage,
            "reward_history": self._reward_history[-100:],  # keep last 100
            "advancement_log": self._advancement_log,
        }
        out.write_text(json.dumps(state, indent=2))

    def load_checkpoint(self, path: str | Path | None = None) -> None:
        src = path or (self._checkpoint_dir / "curriculum_state.json" if self._checkpoint_dir else None)
        if src is None or not Path(src).exists():
            return
        state = json.loads(Path(src).read_text())
        self._stage_idx = min(state.get("stage_idx", 0), len(self.stages) - 1)
        self._total_episodes = state.get("total_episodes", 0)
        self._episodes_at_stage = state.get("episodes_at_stage", 0)
        self._reward_history = state.get("reward_history", [])
        self._advancement_log = state.get("advancement_log", [])
        logger.info("Loaded curriculum checkpoint: stage=%s, episodes=%d", self.current_stage.name, self._total_episodes)

    # ── Status display ────────────────────────────────────────────────────────

    def status(self) -> str:
        """Return a human-readable status string."""
        stage = self.current_stage
        lines = [
            f"Stage: {stage.name} ({self._stage_idx + 1}/{len(self.stages)})",
            f"Difficulty: {stage.difficulty}",
            f"Crisis: {', '.join(stage.crisis_types)}",
            f"Episodes at stage: {self._episodes_at_stage}/{stage.min_episodes_at_stage}",
            f"Recent mean reward: {self.recent_mean_reward:.3f} (target: {stage.min_reward_to_advance:.2f})",
            f"Total episodes: {self._total_episodes}",
        ]
        if self._advancement_log:
            lines.append(f"Advancements: {len(self._advancement_log)}")
        return "\n".join(lines)
