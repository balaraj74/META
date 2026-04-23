"""
metrics_tracker.py — Per-verifier reward tracking and export for GRPO training.

Tracks each verifier's score independently across training steps, enabling:
  - Per-verifier reward curves (not just aggregate)
  - Stage-by-stage performance comparison
  - Export to JSON for the Gradio demo's "Training Metrics" tab
  - ASCII terminal sparklines during training

Usage:
    tracker = MetricsTracker(verifier_names=VERIFIER_NAMES)
    tracker.log_step(step=10, rewards={"patient_survival": 0.9, ...})
    tracker.export_json("metrics.json")
    print(tracker.summary())
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track per-verifier rewards across GRPO training.

    Stores a rolling buffer of metrics per step, supports checkpoint/resume,
    and exports to JSON for visualization.
    """

    def __init__(
        self,
        verifier_names: list[str] | None = None,
        max_history: int = 10000,
    ) -> None:
        self._verifier_names = verifier_names or []
        self._max_history = max_history
        self._steps: list[dict[str, Any]] = []
        self._start_time = time.time()
        self._epoch = 0
        self._stage = "warm_up"

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        rewards: dict[str, float],
        stage: str | None = None,
        epoch: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Log one training step's verifier rewards.

        Args:
            step: global training step number
            rewards: dict mapping verifier_name → score
            stage: curriculum stage name (optional)
            epoch: training epoch (optional)
            extra: additional metadata (e.g. crisis_type, completion text)
        """
        if stage is not None:
            self._stage = stage
        if epoch is not None:
            self._epoch = epoch

        entry = {
            "step": step,
            "epoch": self._epoch,
            "stage": self._stage,
            "timestamp": time.time() - self._start_time,
            "rewards": {k: round(v, 4) for k, v in rewards.items()},
        }
        if extra:
            entry["extra"] = extra

        self._steps.append(entry)

        # Rolling buffer
        if len(self._steps) > self._max_history:
            self._steps = self._steps[-self._max_history:]

    # ── Aggregation ───────────────────────────────────────────────────────────

    def mean_rewards(self, last_n: int = 50) -> dict[str, float]:
        """Compute mean reward per verifier over last N steps."""
        if not self._steps:
            return {}

        recent = self._steps[-last_n:]
        accum: dict[str, list[float]] = defaultdict(list)
        for entry in recent:
            for name, score in entry["rewards"].items():
                accum[name].append(score)

        return {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in accum.items()
        }

    def success_rate(self, threshold: float = 0.5, last_n: int = 50) -> float:
        """Fraction of recent steps where total reward > threshold."""
        if not self._steps:
            return 0.0
        recent = self._steps[-last_n:]
        successes = sum(
            1 for entry in recent
            if entry["rewards"].get("total", 0.0) > threshold
        )
        return round(successes / len(recent), 4)

    def per_stage_summary(self) -> dict[str, dict[str, float]]:
        """Compute mean rewards broken down by curriculum stage."""
        stage_accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for entry in self._steps:
            stage = entry.get("stage", "unknown")
            for name, score in entry["rewards"].items():
                stage_accum[stage][name].append(score)

        return {
            stage: {
                name: round(sum(scores) / len(scores), 4)
                for name, scores in verifiers.items()
            }
            for stage, verifiers in stage_accum.items()
        }

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self, last_n: int = 50) -> str:
        """Return a formatted summary for terminal display."""
        means = self.mean_rewards(last_n)
        if not means:
            return "No data yet."

        lines = [
            f"╔══ Training Metrics (last {min(last_n, len(self._steps))} steps) ══╗",
            f"  Stage: {self._stage}  |  Epoch: {self._epoch}  |  Total steps: {len(self._steps)}",
            f"  Success rate: {self.success_rate(last_n=last_n):.1%}",
            "  ──────────────────────────────────────",
        ]

        for name in sorted(means.keys()):
            if name == "total":
                continue
            score = means[name]
            bar = self._bar(score)
            lines.append(f"  {name:<25s} {score:.3f}  {bar}")

        total = means.get("total", 0.0)
        lines.append("  ──────────────────────────────────────")
        lines.append(f"  {'TOTAL':<25s} {total:.3f}  {self._bar(total)}")
        lines.append("╚════════════════════════════════════════╝")

        return "\n".join(lines)

    def sparkline(self, verifier_name: str = "total", last_n: int = 40) -> str:
        """Return an ASCII sparkline for a single verifier."""
        if not self._steps:
            return ""

        recent = self._steps[-last_n:]
        values = [entry["rewards"].get(verifier_name, 0.0) for entry in recent]
        blocks = " ▁▂▃▄▅▆▇█"
        if not values:
            return ""

        v_min, v_max = min(values), max(values)
        v_range = v_max - v_min if v_max > v_min else 1.0

        return "".join(
            blocks[min(len(blocks) - 1, int((v - v_min) / v_range * (len(blocks) - 1)))]
            for v in values
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self, path: str | Path) -> None:
        """Export full metrics history to JSON for the Gradio demo tab."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "total_steps": len(self._steps),
            "elapsed_seconds": round(time.time() - self._start_time, 1),
            "current_stage": self._stage,
            "current_epoch": self._epoch,
            "mean_rewards_last_50": self.mean_rewards(50),
            "success_rate": self.success_rate(),
            "per_stage": self.per_stage_summary(),
            "history": self._steps[-500:],  # keep last 500 for Gradio plots
        }

        out.write_text(json.dumps(payload, indent=2))
        logger.info("Exported metrics to %s (%d steps)", out, len(self._steps))

    def load_json(self, path: str | Path) -> None:
        """Load metrics from a previous export."""
        src = Path(path)
        if not src.exists():
            return
        payload = json.loads(src.read_text())
        self._steps = payload.get("history", [])
        self._stage = payload.get("current_stage", "warm_up")
        self._epoch = payload.get("current_epoch", 0)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bar(value: float, width: int = 20) -> str:
        """Generate a simple ASCII progress bar."""
        filled = int(value * width)
        return f"[{'█' * filled}{'░' * (width - filled)}]"
