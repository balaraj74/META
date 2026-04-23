#!/usr/bin/env python3
"""
train_grpo.py — GRPO/RLVR training pipeline for TRIAGE hospital agents.

This is the CORE hackathon deliverable. It:
  1. Loads Qwen2.5-0.5B-Instruct in 4-bit via Unsloth
  2. Applies LoRA adapters for parameter-efficient training
  3. Builds prompt dataset from hospital crisis environment rollouts
  4. Runs GRPO training with 8 independent reward verifiers
  5. Tracks per-verifier metrics with curriculum scheduling
  6. Saves LoRA adapters and merged model

Hardware target: 4GB RTX 2050
  - load_in_4bit=True
  - num_generations=4 (minimum viable GRPO group size)
  - gradient_checkpointing="unsloth"
  - per_device_train_batch_size=1 + gradient_accumulation_steps=8

Usage:
    # Full training run
    python scripts/train_grpo.py

    # Quick validation (10 steps)
    python scripts/train_grpo.py --quick

    # Resume from checkpoint
    python scripts/train_grpo.py --resume ./models/grpo_output/checkpoint-100

    # Custom dataset
    python scripts/train_grpo.py --dataset data/grpo/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_grpo")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Model
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# GRPO
NUM_GENERATIONS = 4          # G in GRPO — group size for reward comparison
MAX_COMPLETION_LENGTH = 128  # agent decisions are short
TEMPERATURE = 0.9            # exploration during training
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 1
GRADIENT_ACCUM = 8           # effective batch = 8
LEARNING_RATE = 5e-5
LOGGING_STEPS = 5
SAVE_STEPS = 50

# Paths
OUTPUT_DIR = "./models/grpo_output"
DATASET_PATH = "./data/grpo/train.jsonl"
METRICS_PATH = "./data/grpo/training_metrics.json"
MERGED_DIR = "./models/grpo_merged"

# ═══════════════════════════════════════════════════════════════════════════════
# Reward function wrapper for GRPOTrainer
# ═══════════════════════════════════════════════════════════════════════════════

def _build_reward_function():
    """
    Build a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls: reward_func(completions: list[str], prompts: list[str]) -> list[float]

    We extract the state from the prompt and run all 8 verifiers.
    """
    from triage.rewards.verifiers import compute_all_rewards
    from triage.rewards.sandbox import validate_action

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """Compute aggregate reward for each completion."""
        rewards = []
        prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))

        for i, completion in enumerate(completions):
            # 1. Sandbox check — reject unsafe completions immediately
            is_safe, reason = validate_action(completion)
            if not is_safe:
                rewards.append(0.0)
                continue

            # 2. Build a synthetic state from the prompt
            prompt = prompts[i] if i < len(prompts) else ""
            state = _extract_state_from_prompt(prompt)

            # 3. Run all verifiers
            scores = compute_all_rewards(state, completion)
            rewards.append(scores.get("total", 0.0))

        return rewards

    return reward_fn


def _extract_state_from_prompt(prompt: str) -> dict:
    """
    Extract state variables from the structured prompt text.

    The prompt format (from openenv_adapter.state_to_prompt) contains
    embedded state values that we parse back into a dict for verifiers.
    """
    import re

    state = {
        "alive_count": 20,
        "deceased_count": 0,
        "critical_count": 0,
        "icu_occupancy": 0.5,
        "violations_injected": 0,
        "violations_caught": 0,
        "survival_rate": 1.0,
        "crisis_type": "mass_casualty",
        "patients_summary": [],
    }

    # Extract ICU occupancy
    match = re.search(r"ICU OCCUPANCY:\s*(\d+)%", prompt)
    if match:
        state["icu_occupancy"] = int(match.group(1)) / 100.0

    # Extract critical count
    match = re.search(r"CRITICAL PATIENTS\s*\((\d+)", prompt)
    if match:
        state["critical_count"] = int(match.group(1))

    # Extract violations
    match = re.search(r"VIOLATIONS INJECTED:\s*(\d+)\s*\|\s*CAUGHT:\s*(\d+)", prompt)
    if match:
        state["violations_injected"] = int(match.group(1))
        state["violations_caught"] = int(match.group(2))

    # Extract survival rate
    match = re.search(r"SURVIVAL RATE:\s*(\d+\.?\d*)%", prompt)
    if match:
        state["survival_rate"] = float(match.group(1)) / 100.0

    # Extract crisis type
    match = re.search(r"CRISIS:\s*(\w+)", prompt)
    if match:
        state["crisis_type"] = match.group(1).lower()

    # Extract patient IDs from the prompt for hallucination checking
    patient_ids = []
    for match in re.finditer(r"P-(\d{2,3})", prompt):
        patient_ids.append({"id": int(match.group(1)), "status": "CRITICAL"})
    state["patients_summary"] = patient_ids

    # Compute alive/deceased from survival rate
    total = 20  # approximate
    state["alive_count"] = int(state["survival_rate"] * total)
    state["deceased_count"] = total - state["alive_count"]

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_prompt_dataset(path: str) -> list[dict]:
    """Load JSONL prompt dataset. Returns list of {"prompt": "...", ...}."""
    dataset = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                dataset.append(record)
    logger.info("Loaded %d prompts from %s", len(dataset), path)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GRPO training for TRIAGE agents")
    parser.add_argument("--quick", action="store_true", help="Quick validation (10 steps)")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to JSONL prompt dataset")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--no-merge", action="store_true", help="Skip model merge after training")
    parser.add_argument("--build-dataset", action="store_true", help="Build dataset before training")
    args = parser.parse_args()

    start = time.time()

    # ── Step 0: Build dataset if needed ───────────────────────────────────────
    dataset_path = Path(args.dataset)
    if args.build_dataset or not dataset_path.exists():
        logger.info("Building prompt dataset...")
        from scripts.build_grpo_dataset import build_dataset
        build_dataset(
            n_episodes=50 if args.quick else 100,
            output_path=dataset_path,
            seed=42,
        )

    # ── Step 1: Load model with Unsloth ───────────────────────────────────────
    logger.info("Loading model: %s (4-bit=%s)", MODEL_NAME, LOAD_IN_4BIT)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "Unsloth not installed. Install with:\n"
            "  pip install unsloth\n"
            "Or for Colab:\n"
            "  pip install 'unsloth[colab-new]'"
        )
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.resume or MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,  # auto
    )

    # ── Step 2: Apply LoRA ────────────────────────────────────────────────────
    if args.resume is None:
        logger.info("Applying LoRA (r=%d, alpha=%d)", LORA_R, LORA_ALPHA)
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # ── Step 3: Load dataset ─────────────────────────────────────────────────
    raw_data = load_prompt_dataset(str(dataset_path))
    prompts = [record["prompt"] for record in raw_data]

    # TRL expects a Dataset with a "prompt" column
    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    if args.quick:
        dataset = dataset.select(range(min(40, len(dataset))))
        logger.info("Quick mode: using %d prompts", len(dataset))

    # ── Step 4: Build reward function ─────────────────────────────────────────
    reward_fn = _build_reward_function()

    # ── Step 5: Configure GRPOTrainer ─────────────────────────────────────────
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        logger.error(
            "TRL not installed. Install with:\n"
            "  pip install trl>=0.12"
        )
        sys.exit(1)

    max_steps = 10 if args.quick else 0  # 0 = use num_train_epochs

    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1 if args.quick else NUM_EPOCHS,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        temperature=TEMPERATURE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS if not args.quick else 999999,
        save_total_limit=3,
        report_to="none",
        bf16=False,          # RTX 2050 supports fp16, not bf16
        fp16=True,
        seed=42,
        log_level="info",
    )

    logger.info("Initializing GRPOTrainer (G=%d, batch=%d×%d=%d)",
                NUM_GENERATIONS, PER_DEVICE_BATCH, GRADIENT_ACCUM,
                PER_DEVICE_BATCH * GRADIENT_ACCUM)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
    )

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    logger.info("Starting GRPO training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume)

    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    logger.info("Saving LoRA adapters to %s", args.output)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # ── Step 8: Merge (optional) ──────────────────────────────────────────────
    if not args.no_merge:
        merge_dir = Path(MERGED_DIR)
        logger.info("Merging LoRA → full model at %s", merge_dir)
        try:
            model.save_pretrained_merged(
                str(merge_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            logger.info("Merged model saved successfully.")
        except Exception as exc:
            logger.warning("Merge failed (expected on 4-bit): %s. LoRA adapters saved separately.", exc)

    # ── Step 9: Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"  GRPO Training Complete")
    print(f"  Time:       {elapsed / 60:.1f} min")
    print(f"  Steps:      {train_result.global_step}")
    print(f"  Loss:       {train_result.training_loss:.4f}")
    print(f"  Adapters:   {args.output}")
    if not args.no_merge:
        print(f"  Merged:     {MERGED_DIR}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
