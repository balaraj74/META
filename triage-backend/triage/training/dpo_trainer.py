"""
DPOTrainingPipeline — HF TRL + Unsloth fine-tuning pipeline.

Implements:
  1. Dataset loading from collected JSONL preference pairs
  2. Model loading (with Unsloth 4-bit quantization when available)
  3. DPO training with configurable hyperparameters
  4. Evaluation and model saving

Requires GPU with ~16GB VRAM for actual training.
Falls back to mock mode on CPU for development.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "4b_reliable": {
        "model_name": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_length": 2048,
        "lora_r": 16,
    },
    "8b_showcase": {
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_length": 3072,
        "lora_r": 16,
    },
}


@dataclass
class DPOConfig:
    """DPO training hyperparameters."""
    preset: str = "4b_reliable"
    model_name: str = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
    output_dir: str = "./models/dpo_output"
    data_dir: str = "./data/episodes"
    learning_rate: float = 5e-7
    beta: float = 0.1
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    max_prompt_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    use_unsloth: bool = True
    mock_mode: bool = False  # set True for CPU dev

    def __post_init__(self) -> None:
        preset = MODEL_PRESETS.get(self.preset)
        if not preset:
            return
        for key, value in preset.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DPOTrainingPipeline:
    """End-to-end DPO training pipeline.

    Usage:
        pipeline = DPOTrainingPipeline(config)
        metrics = await pipeline.train()
    """

    def __init__(self, config: DPOConfig | None = None) -> None:
        self.config = config or DPOConfig()
        self._metrics: list[dict[str, float]] = []

    async def train(self) -> dict[str, Any]:
        """Run the full training pipeline."""
        if self.config.mock_mode:
            return self._mock_training()

        return self._real_training()

    def _real_training(self) -> dict[str, Any]:
        """Actual DPO training with TRL + Unsloth."""
        try:
            from datasets import Dataset
            from trl import DPOConfig as TRLDPOConfig, DPOTrainer

            # Load preference pairs
            pairs = self._load_preference_pairs()
            if not pairs:
                return {"error": "No preference pairs found", "data_dir": self.config.data_dir}

            dataset = Dataset.from_list(pairs)
            logger.info("Loaded %d preference pairs for DPO training", len(pairs))

            # Load model
            model, tokenizer = self._load_model()

            # Configure DPO training
            training_args = TRLDPOConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                beta=self.config.beta,
                max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                seed=self.config.seed,
                logging_steps=10,
                save_steps=100,
                fp16=True,
                remove_unused_columns=False,
            )

            trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )

            # Train
            logger.info("Starting DPO training...")
            train_result = trainer.train()
            self._export_training_curves(trainer)

            # Save
            trainer.save_model(self.config.output_dir)
            tokenizer.save_pretrained(self.config.output_dir)

            metrics = {
                "status": "completed",
                "preset": self.config.preset,
                "train_loss": train_result.training_loss,
                "train_steps": train_result.global_step,
                "model_saved_to": self.config.output_dir,
                "dataset_size": len(pairs),
            }
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "training_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info("DPO training complete: %s", metrics)
            return metrics

        except ImportError as e:
            logger.warning("Training dependencies not available: %s", e)
            return {"error": f"Missing dependency: {e}", "fallback": "mock_mode"}
        except Exception as e:
            logger.exception("Training failed")
            return {"error": str(e)}

    def _export_training_curves(self, trainer: Any) -> None:
        """Export training loss and reward curves from trainer logs."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot export")
            return

        log_history = getattr(getattr(trainer, "state", None), "log_history", []) or []
        loss_steps: list[float] = []
        loss_values: list[float] = []
        reward_episodes: list[float] = []
        reward_values: list[float] = []

        def _numeric(value: Any) -> float | None:
            if isinstance(value, (int, float)):
                return float(value)
            return None

        for index, entry in enumerate(log_history, start=1):
            if not isinstance(entry, dict):
                continue

            step_value = _numeric(entry.get("step")) or float(index)
            loss_value = _numeric(entry.get("loss"))
            if loss_value is not None:
                loss_steps.append(step_value)
                loss_values.append(loss_value)

            reward_value = None
            if "total_reward" in entry:
                reward_value = _numeric(entry.get("total_reward"))
            if reward_value is None:
                for key in entry:
                    if "reward" in key.lower():
                        reward_value = _numeric(entry.get(key))
                        if reward_value is not None:
                            break
            if reward_value is not None:
                episode_value = _numeric(entry.get("epoch")) or float(len(reward_episodes) + 1)
                reward_episodes.append(episode_value)
                reward_values.append(reward_value)

        if not loss_values and not reward_values:
            return

        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "training_curves.png"

        fig, (ax_loss, ax_reward) = plt.subplots(1, 2, figsize=(12, 5))

        if loss_values:
            ax_loss.plot(loss_steps, loss_values, marker="o", linewidth=1.5)
        ax_loss.set_title("DPO Training Loss")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, alpha=0.3)

        if reward_values:
            ax_reward.plot(reward_episodes, reward_values, marker="o", linewidth=1.5)
        ax_reward.set_title("Reward Curve")
        ax_reward.set_xlabel("episode")
        ax_reward.set_ylabel("total_reward")
        ax_reward.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _load_model(self) -> tuple[Any, Any]:
        """Load model with Unsloth or standard HF transformers."""
        if self.config.use_unsloth:
            try:
                from unsloth import FastLanguageModel

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.model_name,
                    max_seq_length=self.config.max_length,
                    load_in_4bit=True,
                )
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=self.config.lora_r,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    use_gradient_checkpointing="unsloth",
                )
                logger.info("Model loaded with Unsloth 4-bit quantization")
                return model, tokenizer
            except ImportError:
                logger.warning("Unsloth not available, falling back to transformers")

        # Standard HF transformers fallback
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        return model, tokenizer

    def _load_preference_pairs(self) -> list[dict[str, str]]:
        """Load DPO preference pairs from JSONL."""
        pairs_path = Path(self.config.data_dir) / "dpo_pairs.jsonl"
        if pairs_path.exists():
            pairs = []
            with open(pairs_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        pairs.append({
                            "prompt": data["prompt"],
                            "chosen": data["chosen"],
                            "rejected": data["rejected"],
                        })
            return pairs

        dataset_path = Path(self.config.data_dir) / "preference_dataset.json"
        if dataset_path.exists():
            with open(dataset_path) as f:
                dataset = json.load(f)
            return [
                {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
                for item in dataset.get("train", [])
                if {"prompt", "chosen", "rejected"}.issubset(item)
            ]

        logger.warning("No DPO pairs file found at %s", pairs_path)
        return []

    def _mock_training(self) -> dict[str, Any]:
        """Simulated training for development/demo."""
        import numpy as np

        logger.info("Running mock DPO training (CPU mode)")
        rng = np.random.default_rng(self.config.seed)

        pairs = self._load_preference_pairs()
        n_steps = max(len(pairs) * self.config.num_epochs, 30)

        # Simulate training loss curve
        losses = []
        for step in range(n_steps):
            t = step / n_steps
            loss = 0.7 * np.exp(-3 * t) + 0.1 + rng.normal(0, 0.02)
            losses.append(round(float(loss), 4))

        metrics = {
            "status": "completed_mock",
            "preset": self.config.preset,
            "train_loss_final": losses[-1],
            "train_loss_curve": losses,
            "train_steps": n_steps,
            "dataset_size": len(pairs),
            "model_saved_to": self.config.output_dir,
            "note": "Mock training — no actual model weights updated",
        }

        # Save mock metrics
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Mock training complete: final_loss=%.4f steps=%d", losses[-1], n_steps)
        return metrics

    def get_metrics(self) -> dict[str, Any]:
        """Get training metrics if available."""
        metrics_path = Path(self.config.output_dir) / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {"status": "not_trained"}


TrainingConfig = DPOConfig


class TRIAGEDPOTrainer:
    """Compatibility wrapper exposing the planned trainer interface."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.pipeline = DPOTrainingPipeline(config or TrainingConfig())

    def setup_model(self) -> tuple[Any, Any]:
        """Best-effort model setup for notebook and trainer flows."""
        try:
            return self.pipeline._load_model()
        except Exception:
            return None, None

    async def train(
        self,
        model: Any = None,
        tokenizer: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
    ) -> dict[str, Any]:
        """Run training via the existing pipeline."""
        return await self.pipeline.train()

    def evaluate_improvement(
        self,
        baseline_rewards: list[float],
        trained_rewards: list[float],
    ) -> dict[str, Any]:
        """Compute simple improvement metrics for demo charts."""
        if not baseline_rewards or not trained_rewards:
            return {
                "avg_improvement": 0.0,
                "best_improvement": 0.0,
                "convergence_episode": None,
            }

        deltas = [trained - base for base, trained in zip(baseline_rewards, trained_rewards)]
        avg_improvement = sum(deltas) / len(deltas)
        best_improvement = max(deltas)
        convergence_episode = next(
            (index + 1 for index, delta in enumerate(deltas) if delta > 0),
            None,
        )
        return {
            "avg_improvement": round(avg_improvement, 4),
            "best_improvement": round(best_improvement, 4),
            "convergence_episode": convergence_episode,
        }
