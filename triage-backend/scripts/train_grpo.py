#!/usr/bin/env python3
"""Live GRPO training for TRIAGE using HospitalEnv rollouts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from scripts.build_grpo_dataset import build_crisis_prompt_dataset
from triage.env.grpo_env_adapter import HospitalGRPOEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_grpo")

MODEL_NAME = "Qwen/Qwen3-27B"
DATASET_DIR = Path("data/grpo_crisis_prompts")
OUTPUT_DIR = "./models/grpo_qwen3_27b"


def _terminal_rewards(**kwargs: Any) -> list[float] | None:
    rewards = kwargs.get("terminal_rewards") or kwargs.get("rewards")
    if rewards is None:
        return None
    return [float(r) for r in rewards]


def survival_reward(completions=None, prompts=None, **kwargs) -> list[float]:
    terminal = _terminal_rewards(**kwargs)
    if terminal is not None:
        return terminal
    states = kwargs.get("states") or kwargs.get("state_dicts") or []
    rewards = []
    for state in states:
        stats = state.get("stats", {}) if isinstance(state, dict) else {}
        rewards.append(float(stats.get("survival_rate", 0.0)) * 2.0 - 1.0)
    return rewards or [0.0 for _ in (completions or [])]


def safety_reward(completions=None, prompts=None, **kwargs) -> list[float]:
    terminal = _terminal_rewards(**kwargs)
    if terminal is not None:
        return terminal
    states = kwargs.get("states") or kwargs.get("state_dicts") or []
    rewards = []
    for state in states:
        blocks = state.get("safety_blocks", []) if isinstance(state, dict) else []
        rewards.append(1.0 if not blocks else -1.0)
    return rewards or [0.0 for _ in (completions or [])]


def resource_reward(completions=None, prompts=None, **kwargs) -> list[float]:
    terminal = _terminal_rewards(**kwargs)
    if terminal is not None:
        return terminal
    states = kwargs.get("states") or kwargs.get("state_dicts") or []
    rewards = []
    for state in states:
        stats = state.get("stats", {}) if isinstance(state, dict) else {}
        occupancy = float(stats.get("icu_occupancy", 0.0))
        rewards.append(1.0 - min(1.0, max(0.0, occupancy - 0.85) / 0.15) * 2.0)
    return rewards or [0.0 for _ in (completions or [])]


def ethics_reward(completions=None, prompts=None, **kwargs) -> list[float]:
    terminal = _terminal_rewards(**kwargs)
    if terminal is not None:
        return terminal
    states = kwargs.get("states") or kwargs.get("state_dicts") or []
    rewards = []
    for state in states:
        stats = state.get("stats", {}) if isinstance(state, dict) else {}
        injected = int(stats.get("violations_injected", 0))
        caught = int(stats.get("violations_caught", 0))
        rewards.append(
            1.0 if injected == 0 else min(1.0, caught / max(injected, 1)) * 2.0 - 1.0
        )
    return rewards or [0.0 for _ in (completions or [])]


def load_or_build_dataset(path: Path):
    if not path.exists():
        logger.info("Building crisis prompt dataset at %s", path)
        return build_crisis_prompt_dataset(output_dir=path)
    return load_from_disk(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-27B with live HospitalEnv GRPO")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Use a small dataset slice for smoke testing")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.resume or MODEL_NAME,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    if args.resume is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            random_state=42,
        )

    crisis_dataset = load_or_build_dataset(args.dataset)
    if args.quick:
        crisis_dataset = crisis_dataset.select(range(min(32, len(crisis_dataset))))

    config = GRPOConfig(
        num_generations=16,
        max_prompt_length=2048,
        max_completion_length=2048,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1 if args.quick else 3,
        max_steps=2 if args.quick else -1,
        max_grad_norm=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="wandb",
        logging_steps=1,
        save_steps=50,
        output_dir=args.output_dir,
        bf16=True,
        save_total_limit=3,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=crisis_dataset,
        reward_funcs=[
            survival_reward,
            safety_reward,
            resource_reward,
            ethics_reward,
        ],
        environment_factory=HospitalGRPOEnvironment,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
