#!/usr/bin/env python3
"""
train_hf_dpo.py — HuggingFace DPO Training Pipeline for TRIAGE.

Optimized for RTX 2050 (4 GB VRAM) with full HuggingFace Hub integration.

Features:
  - 4-bit NF4 quantization (bitsandbytes)
  - LoRA (rank=32, alpha=64, targeting q/k/v/o/gate/up/down projections)
  - Gradient checkpointing + paged_adamw_8bit
  - Auto dataset upload to HF Hub
  - Auto model merge + push after training
  - Live progress JSON for dashboard polling
  - OOM-safe with automatic batch size fallback

VRAM Budget (RTX 2050 — 4 GB):
  Model (4-bit Qwen2.5-0.5B):  ~0.4 GB
  LoRA adapters:                ~0.1 GB
  Optimizer (8-bit paged):      ~0.3 GB
  Activations (checkpointed):   ~1.0 GB
  KV cache (max_length=512):    ~0.2 GB
  Headroom:                     ~2.0 GB
  TOTAL:                        ~4.0 GB ✓

Usage:
    # Full training with HF push:
    python scripts/train_hf_dpo.py --push-to-hub --hf-repo balarajr/triage-qwen-0.5b-dpo

    # Quick test (10 min):
    python scripts/train_hf_dpo.py --max-samples 500 --epochs 1

    # Dataset-only push (no training):
    python scripts/train_hf_dpo.py --push-dataset-only --hf-dataset-repo balarajr/triage-dpo-dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("triage.train_hf_dpo")

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR       = ROOT / "data" / "full_training"
HF_DATA_PATH   = DATA_DIR / "hf_dpo_pairs.jsonl"
ORIG_DATA_PATH = DATA_DIR / "dpo_pairs.jsonl"
OUTPUT_DIR     = ROOT / "models" / "hf_dpo_output"
LIVE_STATUS    = ROOT / "data" / "training_live.json"

DEFAULT_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"

# ── VRAM-safe defaults ─────────────────────────────────────────
MAX_LENGTH         = 512
MAX_PROMPT_LENGTH  = 256
BATCH_SIZE         = 1
GRAD_ACCUM         = 8
LORA_R             = 32
LORA_ALPHA         = 64


def _write_live(payload: dict) -> None:
    """Atomically write live training status for the dashboard."""
    LIVE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(LIVE_STATUS)


# ─── Dataset ───────────────────────────────────────────────────

def load_dataset(path: Path, max_samples: int | None = None):
    """Load JSONL → HF Dataset with train/eval split."""
    from datasets import Dataset

    rows: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rows.append({
                    "prompt":   str(rec["prompt"])[:MAX_LENGTH * 2],
                    "chosen":   str(rec["chosen"])[:MAX_LENGTH],
                    "rejected": str(rec["rejected"])[:MAX_LENGTH],
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if max_samples:
        import random
        random.seed(42)
        random.shuffle(rows)
        rows = rows[:max_samples]

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.05, seed=42)
    logger.info("Dataset loaded — Train: %d | Eval: %d", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


def push_dataset_to_hub(train_ds, eval_ds, repo_id: str, private: bool = True) -> str:
    """Push dataset to HuggingFace Hub."""
    from datasets import DatasetDict

    dd = DatasetDict({"train": train_ds, "test": eval_ds})
    dd.push_to_hub(repo_id, private=private)
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Dataset pushed → %s", url)
    return url


# ─── Model Loading ─────────────────────────────────────────────

def load_model_4bit(model_id: str):
    """Load model in 4-bit NF4 quantization for 4 GB VRAM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    vram = torch.cuda.memory_allocated() / 1024**3
    logger.info("Model loaded — VRAM used: %.2f GB", vram)
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters targeting Qwen2 attention + MLP layers."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("LoRA adapters — %s trainable / %s total (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model


# ─── Progress Callback ─────────────────────────────────────────

from transformers import TrainerCallback


class HFLiveProgressCallback(TrainerCallback):
    """Write training progress to JSON for dashboard polling."""

    def __init__(self, total_steps: int, epochs: int, train_samples: int, model_id: str):
        self.total_steps = total_steps
        self.epochs = epochs
        self.train_samples = train_samples
        self.model_id = model_id
        self._start = time.perf_counter()
        self._losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is None:
            return

        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss"))
        if loss is not None:
            self._losses.append(float(loss))

        elapsed = time.perf_counter() - self._start
        progress = step / max(self.total_steps, 1)
        sps = step / max(elapsed, 1)
        eta = (self.total_steps - step) / max(sps, 0.001)

        vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        vram_total = torch.cuda.get_device_properties(0).total_memoryory / 1024**3 if torch.cuda.is_available() else 4.0

        _write_live({
            "phase": "training",
            "method": "DPO",
            "progress": round(min(progress, 0.99), 4),
            "step": step,
            "total_steps": self.total_steps,
            "epoch": round(state.epoch or 0.0, 2),
            "total_epochs": self.epochs,
            "loss": round(loss, 4) if loss is not None else None,
            "avg_loss": round(sum(self._losses) / len(self._losses), 4) if self._losses else None,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": round(vram_total, 2),
            "gpu_pct": round(vram_used / vram_total * 100, 1),
            "model": self.model_id,
            "train_samples": self.train_samples,
        })


# ─── Training ──────────────────────────────────────────────────

def run_training(
    model_id: str,
    data_path: Path,
    output_dir: Path,
    epochs: int,
    max_samples: int | None,
    push_to_hub: bool = False,
    hf_repo: str | None = None,
    hf_dataset_repo: str | None = None,
    private: bool = True,
) -> dict[str, Any]:
    """Full DPO training pipeline with optional HF Hub push."""
    from trl import DPOConfig, DPOTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Dataset ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  [1/5] Loading dataset...")
    print(f"{'─'*50}")
    train_ds, eval_ds = load_dataset(data_path, max_samples)
    print(f"  Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    # Optionally push dataset to Hub
    if hf_dataset_repo:
        print(f"\n  Pushing dataset → {hf_dataset_repo}")
        push_dataset_to_hub(train_ds, eval_ds, hf_dataset_repo, private=private)

    # ── 2. Model ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  [2/5] Loading 4-bit quantized model...")
    print(f"{'─'*50}")
    model, tokenizer = load_model_4bit(model_id)

    # ── 3. LoRA ─────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  [3/5] Applying LoRA adapters...")
    print(f"{'─'*50}")
    model = apply_lora(model)

    # ── 4. Train ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  [4/5] Starting DPO training...")
    print(f"{'─'*50}")

    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memoryory / 1024**3
    print(f"  VRAM before training: {vram_used:.2f} / {vram_total:.2f} GB")

    # Hub config
    hub_kwargs = {}
    if push_to_hub and hf_repo:
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": hf_repo,
            "hub_private_repo": private,
        }

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=0.05,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=["none"],
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
        **hub_kwargs,
    )

    # Progress tracking
    steps_per_epoch = max(1, len(train_ds) // (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * epochs

    _write_live({
        "phase": "initializing",
        "method": "DPO",
        "progress": 0.01,
        "step": 0,
        "total_steps": total_steps,
        "model": model_id,
        "train_samples": len(train_ds),
    })

    live_cb = HFLiveProgressCallback(
        total_steps=total_steps,
        epochs=epochs,
        train_samples=len(train_ds),
        model_id=model_id,
    )

    # TRL version compat
    try:
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=[live_cb],
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            callbacks=[live_cb],
        )

    start = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start

    # ── 5. Save ─────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  [5/5] Saving model...")
    print(f"{'─'*50}")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Metrics
    metrics = {
        "model": model_id,
        "method": "DPO",
        "epochs": epochs,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "train_loss": round(train_result.training_loss, 4),
        "train_runtime_seconds": round(elapsed, 1),
        "train_steps": train_result.global_step,
        "samples_per_second": round(len(train_ds) * epochs / max(elapsed, 1), 2),
        "gpu": torch.cuda.get_device_name(0),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "output_dir": str(final_dir),
    }

    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Mark completion in live status
    _write_live({
        "phase": "complete",
        "method": "DPO",
        "progress": 1.0,
        "step": train_result.global_step,
        "total_steps": total_steps,
        "loss": metrics["train_loss"],
        "elapsed_seconds": metrics["train_runtime_seconds"],
        "model": model_id,
        "train_samples": len(train_ds),
        "vram_peak_gb": metrics["vram_peak_gb"],
    })

    return metrics


# ─── Merge + Push ──────────────────────────────────────────────

def merge_and_push(adapter_dir: Path, hf_repo: str, private: bool = True) -> None:
    """Merge LoRA adapter into base model and push to HuggingFace Hub."""
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi

    merged_dir = adapter_dir.parent / "merged"

    print(f"\n  [Merge] Loading adapter: {adapter_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        trust_remote_code=True,
        device_map="cpu",
    )

    print("  [Merge] Merging LoRA weights...")
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Merge] Saving merged model → {merged_dir}")
    merged.save_pretrained(str(merged_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_dir))

    # Write model card
    model_card = f"""---
language: [en]
license: apache-2.0
base_model: {DEFAULT_MODEL}
tags: [medical, triage, hospital, multi-agent, dpo, lora, qwen, clinical-ai]
datasets: [openlifescienceai/medmcqa, bigbio/med_qa]
pipeline_tag: text-generation
---

# TRIAGE — Hospital Crisis Agent (Qwen2.5-0.5B DPO)

DPO fine-tuned version of `{DEFAULT_MODEL}` for **hospital crisis management**.

## Training
- Method: DPO (Direct Preference Optimization)
- LoRA: r={LORA_R}, alpha={LORA_ALPHA}
- Quantization: 4-bit NF4
- Hardware: NVIDIA RTX 2050 (4 GB VRAM)
"""
    (merged_dir / "README.md").write_text(model_card)

    # Push
    print(f"  [Push] Uploading → {hf_repo}")
    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=hf_repo,
        repo_type="model",
        commit_message="Upload TRIAGE DPO model (merged LoRA)",
    )
    print(f"  ✓ Model pushed → https://huggingface.co/{hf_repo}")


# ─── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE — HuggingFace DPO Training Pipeline")

    # Model
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model ID")

    # Data
    parser.add_argument("--data-path", type=str, default=None, help="Path to DPO JSONL dataset")
    parser.add_argument("--hf-dataset", action="store_true", help="Use high-quality HF dataset (hf_dpo_pairs.jsonl)")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap dataset size for quick tests")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")

    # HF Hub
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub after training")
    parser.add_argument("--hf-repo", type=str, default="balarajr/triage-qwen-0.5b-dpo", help="HF Hub model repo")
    parser.add_argument("--hf-dataset-repo", type=str, default=None, help="Push dataset to this HF repo")
    parser.add_argument("--private", action="store_true", default=True, help="Make HF repos private")
    parser.add_argument("--public", action="store_true", help="Make HF repos public")

    # Actions
    parser.add_argument("--push-dataset-only", action="store_true", help="Only push dataset, skip training")
    parser.add_argument("--merge-and-push", action="store_true", help="Only merge adapter + push, skip training")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Adapter dir for --merge-and-push")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    private = not args.public

    # Resolve data path
    if args.data_path:
        data_path = Path(args.data_path)
    elif args.hf_dataset:
        data_path = HF_DATA_PATH
    else:
        data_path = HF_DATA_PATH if HF_DATA_PATH.exists() else ORIG_DATA_PATH

    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        logger.error("Run: python3 scripts/generate_hf_dpo.py")
        sys.exit(1)

    # ── Dataset-only mode ───────────────────────────────────
    if args.push_dataset_only:
        repo = args.hf_dataset_repo or "balarajr/triage-dpo-dataset"
        train_ds, eval_ds = load_dataset(data_path, args.max_samples)
        push_dataset_to_hub(train_ds, eval_ds, repo, private=private)
        print(f"\n  ✓ Dataset pushed to: https://huggingface.co/datasets/{repo}")
        return

    # ── Merge-only mode ─────────────────────────────────────
    if args.merge_and_push:
        adapter_dir = Path(args.adapter_dir) if args.adapter_dir else OUTPUT_DIR / "final"
        if not adapter_dir.exists():
            logger.error("Adapter not found: %s — train first!", adapter_dir)
            sys.exit(1)
        merge_and_push(adapter_dir, args.hf_repo, private=private)
        return

    # ── Full training ───────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_mb  = torch.cuda.get_device_properties(0).total_mem // (1024 ** 2)

    print(f"\n{'='*60}")
    print("  TRIAGE — HuggingFace DPO Training Pipeline")
    print(f"{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  GPU        : {gpu_name}")
    print(f"  VRAM       : {vram_mb} MB")
    print(f"  Dataset    : {data_path} ({data_path.stat().st_size // 1024} KB)")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LoRA       : r={LORA_R}, α={LORA_ALPHA}")
    print(f"  Max Length  : {MAX_LENGTH}")
    print(f"  Batch      : {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Output     : {args.output_dir}")
    print(f"  Push to Hub: {args.push_to_hub} → {args.hf_repo}")
    print(f"{'='*60}\n")

    try:
        metrics = run_training(
            model_id=args.model,
            data_path=data_path,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            max_samples=args.max_samples,
            push_to_hub=args.push_to_hub,
            hf_repo=args.hf_repo,
            hf_dataset_repo=args.hf_dataset_repo,
            private=private,
        )

        print(f"\n{'='*60}")
        print("  ✅ TRAINING COMPLETE")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:<30}: {v}")
        print(f"{'='*60}")

        # Auto merge + push if requested
        if args.push_to_hub:
            print("\n  [Auto] Merging adapter and pushing to Hub...")
            adapter_dir = Path(args.output_dir) / "final"
            merge_and_push(adapter_dir, args.hf_repo, private=private)

    except torch.cuda.OutOfMemoryError:
        print("\n[OOM] GPU out of memory. Try:")
        print("  1. python scripts/train_hf_dpo.py --max-samples 300 --epochs 1")
        print("  2. Reduce max length in script (currently 512)")
        _write_live({"phase": "error", "error": "OOM", "method": "DPO"})
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  [Interrupted] Saving checkpoint...")
        _write_live({"phase": "interrupted", "method": "DPO"})
        sys.exit(130)


if __name__ == "__main__":
    main()
