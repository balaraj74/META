#!/usr/bin/env python3
"""Train Qwen3-14B for TRIAGE with GRPO on a single 24GB GPU.

This entrypoint is tuned for AWS g5.2xlarge spot instances:
1x NVIDIA A10G, 24GB VRAM, 4-bit base model, LoRA adapters, frequent checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("train_grpo_qwen3_14b")

DEFAULT_MODEL = "Qwen/Qwen3-14B-Instruct"
DEFAULT_DATASET = "data/grpo_crisis_prompts"
DEFAULT_OUTPUT_DIR = "models/grpo_qwen3_14b"
DEFAULT_HUB_MODEL_ID = "balarajr/triage-qwen3-14b-grpo"

VALID_ACTIONS = frozenset(
    {
        "TRIAGE_PATIENT",
        "TRANSFER_TO_ICU",
        "TRANSFER_TO_WARD",
        "ACTIVATE_OVERFLOW",
        "ORDER_MEDICATION",
        "REQUEST_STAFF",
        "FLAG_POLICY_VIOLATION",
        "OVERRIDE_DECISION",
        "UPDATE_EHR",
        "ASSIGN_TREATMENT",
        "DISCHARGE_PATIENT",
        "VERIFY_INSURANCE",
    }
)


def _completion_text(completion: Any) -> str:
    """Normalize TRL completion payloads across versions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or completion)
    if isinstance(completion, list):
        if not completion:
            return ""
        if isinstance(completion[0], dict):
            return str(completion[0].get("content") or completion[0].get("text") or "")
        return "\n".join(str(item) for item in completion)
    return str(completion)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

    block = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if block:
        try:
            parsed = json.loads(block.group(0))
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _action_value(parsed: dict[str, Any]) -> str:
    return str(parsed.get("action_type") or parsed.get("action") or "").upper()


def reward_format_compliance(prompts, completions, **_kwargs) -> list[float]:
    """Reward valid TRIAGE action JSON."""
    scores = []
    for completion in completions:
        parsed = _extract_json(_completion_text(completion))
        if parsed is None:
            scores.append(0.0)
            continue

        action = _action_value(parsed)
        priority = parsed.get("priority")
        has_target = any(key in parsed for key in ("target_id", "patient_id"))
        reasoning = str(parsed.get("reasoning", parsed.get("rationale", ""))).strip()
        has_reasoning = len(reasoning) >= 10

        try:
            priority_ok = 1 <= int(priority) <= 10
        except (TypeError, ValueError):
            priority_ok = False

        scores.append(
            1.0
            if action in VALID_ACTIONS and has_target and priority_ok and has_reasoning
            else 0.0
        )
    return scores


def reward_reasoning_quality(prompts, completions, **_kwargs) -> list[float]:
    """Reward completions that cite patient/context evidence instead of generic text."""
    clinical_terms = (
        "triage",
        "priority",
        "critical",
        "icu",
        "transfer",
        "protocol",
        "capacity",
        "medication",
        "overflow",
        "vitals",
        "acuity",
    )
    scores = []
    for prompt, completion in zip(prompts, completions, strict=False):
        text = _completion_text(completion)
        prompt_ids = set(
            re.findall(r"\b(?:PT|P)-?\d+\b", str(prompt), re.IGNORECASE)
        )
        cited_ids = {pid for pid in prompt_ids if pid.lower() in text.lower()}
        id_score = 1.0 if not prompt_ids else len(cited_ids) / max(len(prompt_ids), 1)
        term_score = min(1.0, sum(term in text.lower() for term in clinical_terms) / 3.0)
        has_evidence = re.search(
            r"\d+%|\bHR\s*\d+|\bBP\s*\d+|\bICU\b",
            text,
            re.I,
        )
        evidence_score = 1.0 if has_evidence else 0.0
        scores.append((id_score * 0.35) + (term_score * 0.45) + (evidence_score * 0.20))
    return scores


def reward_no_hallucination(prompts, completions, **_kwargs) -> list[float]:
    """Penalize invented patient IDs."""
    scores = []
    for prompt, completion in zip(prompts, completions, strict=False):
        real_ids = set(re.findall(r"\b(?:PT|P)-?\d+\b", str(prompt), re.IGNORECASE))
        cited_ids = set(
            re.findall(
                r"\b(?:PT|P)-?\d+\b",
                _completion_text(completion),
                re.IGNORECASE,
            )
        )
        scores.append(0.0 if cited_ids - real_ids else 1.0)
    return scores


def reward_action_alignment(prompts, completions, **_kwargs) -> list[float]:
    """Lightweight context/action alignment signal for crisis prompts."""
    scores = []
    for prompt, completion in zip(prompts, completions, strict=False):
        prompt_text = str(prompt).lower()
        completion_text = _completion_text(completion)
        parsed = _extract_json(completion_text) or {}
        action = _action_value(parsed)
        action_or_text = action or completion_text.upper()

        score = 0.5
        if "mass_casualty" in prompt_text and "TRIAGE" in action_or_text:
            score = 1.0
        elif "equipment_failure" in prompt_text and (
            "ACTIVATE_OVERFLOW" in action_or_text or "REQUEST_STAFF" in action_or_text
        ):
            score = 0.8
        elif "staff_shortage" in prompt_text and "REQUEST_STAFF" in action_or_text:
            score = 1.0
        elif "outbreak" in prompt_text and (
            "FLAG_POLICY_VIOLATION" in action_or_text or "ASSIGN_TREATMENT" in action_or_text
        ):
            score = 0.8

        if action in {"WAIT", "NO_OP"} or not completion_text.strip():
            score = 0.0
        scores.append(score)
    return scores


def reward_response_speed(prompts, completions, **_kwargs) -> list[float]:
    """Prefer concise clinical decisions."""
    scores = []
    for completion in completions:
        tokens = len(_completion_text(completion).split())
        if 35 <= tokens <= 180:
            score = 1.0
        elif tokens < 10:
            score = 0.0
        elif tokens > 300:
            score = max(0.0, 1.0 - ((tokens - 180) / 300.0))
        else:
            score = 0.8
        scores.append(score)
    return scores


REWARD_FUNCS = [
    reward_format_compliance,
    reward_reasoning_quality,
    reward_no_hallucination,
    reward_action_alignment,
    reward_response_speed,
]


def load_prompt_dataset(dataset_ref: str) -> Any:
    """Load a JSONL file, saved Arrow dataset, or HF Hub dataset with a prompt column."""
    from datasets import Dataset, load_dataset, load_from_disk

    path = Path(dataset_ref)
    if path.exists() and path.is_dir():
        dataset = load_from_disk(str(path))
    elif path.exists():
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "prompt" not in record:
                    raise ValueError(f"Dataset row is missing `prompt`: {record}")
                rows.append(record)
        dataset = Dataset.from_list(rows)
    elif dataset_ref == DEFAULT_DATASET:
        from scripts.build_grpo_dataset import build_crisis_prompt_dataset

        logger.info("Dataset %s is missing; building crisis prompts", dataset_ref)
        dataset = build_crisis_prompt_dataset(output_dir=path)
    else:
        dataset = load_dataset(dataset_ref, split="train")

    if "prompt" not in dataset.column_names:
        raise ValueError(f"Dataset must contain a `prompt` column, found {dataset.column_names}")
    return dataset


def latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None

    checkpoints = []
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            checkpoints.append((int(child.name.rsplit("-", 1)[1]), child))
        except ValueError:
            continue

    if not checkpoints:
        return None
    return str(sorted(checkpoints, key=lambda item: item[0])[-1][1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--no-push-to-hub", action="store_true")
    parser.add_argument("--merge-16bit", action="store_true")
    parser.add_argument("--merged-dir", default="models/triage-qwen3-14b-grpo-merged")
    parser.add_argument("--smoke-rewards", action="store_true")
    return parser


def smoke_rewards() -> None:
    prompts = [
        "Crisis: mass_casualty. Patients: PT-001 HR 132, ICU at 92%.",
        "Crisis: staff_shortage. Patients: PT-002 stable.",
    ]
    completions = [
        json.dumps(
            {
                "action_type": "TRIAGE_PATIENT",
                "target_id": 1,
                "priority": 9,
                "reasoning": "PT-001 has critical vitals and ICU capacity pressure.",
            }
        ),
        json.dumps(
            {
                "action_type": "REQUEST_STAFF",
                "target_id": 2,
                "priority": 6,
                "reasoning": "PT-002 can wait while staff coverage is restored.",
            }
        ),
    ]
    for reward_func in REWARD_FUNCS:
        scores = reward_func(prompts, completions)
        if len(scores) != len(completions):
            raise AssertionError(f"{reward_func.__name__} returned {scores}")
        logger.info("%s: %s", reward_func.__name__, scores)


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_rewards:
        smoke_rewards()
        return

    import torch
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    logger.info("Loading dataset: %s", args.dataset)
    dataset = load_prompt_dataset(args.dataset)
    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    logger.info("Dataset ready: %d train / %d eval", len(split["train"]), len(split["test"]))

    logger.info("Loading model %s in 4-bit", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    logger.info("Model loaded. CUDA allocated: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=0.9,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        beta=0.04,
        report_to="none",
        push_to_hub=not args.no_push_to_hub,
        hub_model_id=args.hub_model_id if not args.no_push_to_hub else None,
        seed=args.seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "reward_funcs": REWARD_FUNCS,
    }
    try:
        trainer = GRPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    checkpoint = args.resume_from_checkpoint
    if args.resume and checkpoint is None:
        checkpoint = latest_checkpoint(args.output_dir)
    if checkpoint:
        logger.info("Resuming from checkpoint: %s", checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_16bit:
        logger.info("Saving merged 16-bit model to %s", args.merged_dir)
        model.save_pretrained_merged(
            args.merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        if not args.no_push_to_hub:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required to push the merged model")
            model.push_to_hub_merged(
                args.hub_model_id,
                tokenizer,
                save_method="merged_16bit",
                token=token,
            )

    logger.info("GRPO training complete")


if __name__ == "__main__":
    main()
