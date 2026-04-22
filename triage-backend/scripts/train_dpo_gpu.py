#!/usr/bin/env python3
"""
train_dpo_gpu.py — Real DPO fine-tuning on RTX 2050 (4 GB VRAM).

Uses TRL DPOTrainer + PEFT LoRA + 4-bit quantization (bitsandbytes)
to fine-tune Qwen2.5-0.5B-Instruct on the TRIAGE hospital workflow dataset.

Model: Qwen/Qwen2.5-0.5B-Instruct  (~500M params, ~1 GB download, ~0.5 GB in 4-bit)
       Perfect fit for 4 GB VRAM — leaves plenty of headroom for gradients.
LoRA: rank=16, alpha=32, attention + MLP layers
DPO beta: 0.1 (standard)
Batch: 1 + gradient_accumulation=8 (effective batch=8)

Usage:
    python scripts/train_dpo_gpu.py
    python scripts/train_dpo_gpu.py --epochs 3
    python scripts/train_dpo_gpu.py --epochs 1 --max-samples 500  (quick test)
    python scripts/train_dpo_gpu.py --model Qwen/Qwen2.5-1.5B-Instruct (larger)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Live progress file polled by the dashboard API
LIVE_STATUS_FILE = ROOT / "data" / "training_live.json"


def _write_live_status(payload: dict) -> None:
    """Atomically write live status so the API never reads a partial file."""
    LIVE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(LIVE_STATUS_FILE)

# ─── VRAM Budget for RTX 2050 (4 GB) ─────────────────────────
# Model (4-bit):  ~0.4 GB  (Qwen2.5-0.5B params)
# LoRA adapters:  ~0.1 GB
# Optimizer:      ~0.3 GB
# KV cache:       ~0.2 GB
# Activations:    ~1.0 GB  (long sequences)
# Safety margin:  ~2.0 GB
# TOTAL:          ~4.0 GB ✓ (with 2 GB to spare for seq2seq)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = ROOT / "data" / "full_training" / "healthcare_dpo.jsonl"
OUTPUT_DIR = ROOT / "models" / "dpo_output_gpu"


# ─── Dataset Loader ───────────────────────────────────────────

def load_dpo_dataset(data_path: Path, max_samples: int | None = None):
    """Load JSONL into HuggingFace Dataset format for DPOTrainer."""
    from datasets import Dataset  # type: ignore[import-untyped]

    pairs: list[dict[str, str]] = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # DPOTrainer expects: prompt, chosen, rejected (all str)
                pairs.append({
                    "prompt": str(row["prompt"])[:1024],   # cap to avoid OOM
                    "chosen": str(row["chosen"])[:512],
                    "rejected": str(row["rejected"])[:512],
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if max_samples:
        import random
        random.shuffle(pairs)
        pairs = pairs[:max_samples]

    dataset = Dataset.from_list(pairs)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


# ─── Model Loader (4-bit quantized) ───────────────────────────

def load_model_4bit(model_id: str):
    """Load model in 4-bit quantization for 4 GB VRAM."""
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",  # required for DPO
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quant saves ~0.4 GB
        bnb_4bit_quant_type="nf4",        # best quality 4-bit format
    )

    print(f"  Loading model (4-bit NF4 + nested quant)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # scaled dot product — works without flash-attn
    )

    return model, tokenizer


# ─── LoRA Config ──────────────────────────────────────────────

def apply_lora(model):
    """Apply LoRA adapters — only the key attention + MLP layers."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-untyped]

    # Prepare for k-bit training (enables gradient checkpointing)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,                           # rank
        lora_alpha=32,                  # scaling = alpha/r = 2
        target_modules=[               # Qwen2 attention + MLP layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    print(f"  LoRA adapters: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")
    return model


# ─── Live Progress Callback ───────────────────────────────────

from transformers import TrainerCallback  # type: ignore[import-untyped]

class LiveProgressCallback(TrainerCallback):
    """Write progress metrics to disk every step so the dashboard can display them."""

    def __init__(self, total_steps: int, epochs: int, train_samples: int, model_id: str) -> None:
        self.total_steps = total_steps
        self.epochs = epochs
        self.train_samples = train_samples
        self.model_id = model_id
        self._start = time.perf_counter()
        self._losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # noqa: ANN001
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss", 0.0))
        if loss:
            self._losses.append(float(loss))
        elapsed = time.perf_counter() - self._start
        progress = step / max(self.total_steps, 1)
        sps = step / max(elapsed, 1)
        eta = (self.total_steps - step) / max(sps, 0.001)
        epoch = state.epoch or 0.0
        vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 4.0
        _write_live_status({
            "phase": "training",
            "progress": round(min(progress, 0.99), 4),
            "step": step,
            "total_steps": self.total_steps,
            "epoch": round(epoch, 2),
            "total_epochs": self.epochs,
            "loss": round(loss, 4) if loss else None,
            "avg_loss": round(sum(self._losses) / len(self._losses), 4) if self._losses else None,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": round(vram_total, 2),
            "gpu_pct": round(vram_used / vram_total * 100, 1),
            "model": self.model_id,
            "train_samples": self.train_samples,
        })


# ─── Training ─────────────────────────────────────────────────

def train(
    model_id: str,
    data_path: Path,
    output_dir: Path,
    epochs: int,
    max_samples: int | None,
) -> dict[str, Any]:
    """Full DPO training pipeline."""
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-untyped]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    train_ds, eval_ds = load_dpo_dataset(data_path, max_samples)
    print(f"  Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    # ── Model ───────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model, tokenizer = load_model_4bit(model_id)

    print("\n[3/4] Applying LoRA...")
    model = apply_lora(model)

    # ── Training Config ─────────────────────────────────────
    print("\n[4/4] Starting DPO training...")
    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM before training: {vram_used:.2f} / {vram_total:.2f} GB")

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,        # 4 GB constraint
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,        # effective batch = 8
        gradient_checkpointing=True,          # trade speed for VRAM
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=0.1,                             # DPO temperature
        max_length=384,                       # fits within 4 GB VRAM
        max_prompt_length=192,                # half of max_length
        fp16=False,
        bf16=True,                            # RTX 2050 supports bf16
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",                     # no wandb needed
        optim="paged_adamw_8bit",            # 8-bit optimizer saves VRAM
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )

    # Compute total steps for progress tracking
    steps_per_epoch = len(train_ds) // (dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs

    # Write initial status so dashboard shows 'training' immediately
    _write_live_status({
        "phase": "training",
        "progress": 0.01,
        "step": 0,
        "total_steps": total_steps,
        "epoch": 0,
        "total_epochs": epochs,
        "loss": None,
        "model": model_id,
        "train_samples": len(train_ds),
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "gpu_pct": round(vram_used / vram_total * 100, 1),
    })

    live_cb = LiveProgressCallback(
        total_steps=total_steps,
        epochs=epochs,
        train_samples=len(train_ds),
        model_id=model_id,
    )

    # TRL 0.11.x uses 'tokenizer'; newer TRL (0.12+) renamed it 'processing_class'
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

    # ── Save ────────────────────────────────────────────────
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save training metrics
    metrics = {
        "model": model_id,
        "epochs": epochs,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "train_loss": round(train_result.training_loss, 4),
        "train_runtime_seconds": round(elapsed, 1),
        "train_steps": train_result.global_step,
        "samples_per_second": round(len(train_ds) * epochs / elapsed, 2),
        "gpu": torch.cuda.get_device_name(0),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "output_dir": str(output_dir / "final"),
    }

    with (output_dir / "gpu_training_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real DPO GPU training for TRIAGE hospital agents"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL}). "
             f"Other options: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "data" / "full_training"),
        help="Directory containing dpo_pairs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Where to save fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2 — safe for 4 GB VRAM)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap dataset size (e.g. 500 for a quick 10-min test)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    data_path = Path(args.data_dir) / "dpo_pairs.jsonl"
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        print("  Run: python scripts/generate_dpo_fast.py first")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. This script requires a GPU.")
        print("  Use: python scripts/train_dpo.py  (CPU mock training)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  TRIAGE — Real DPO GPU Training")
    print(f"{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Dataset  : {data_path}")
    print(f"  Output   : {args.output_dir}")
    print(f"{'='*60}\n")

    try:
        metrics = train(
            model_id=args.model,
            data_path=data_path,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            max_samples=args.max_samples,
        )

        print(f"\n{'='*60}")
        print("  TRAINING COMPLETE ✓")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:<30}: {v}")
        print(f"{'='*60}\n")

    except torch.cuda.OutOfMemoryError:
        print("\n[OOM] GPU out of memory. Try:")
        print("  1. python scripts/train_dpo_gpu.py --max-samples 300")
        print("  2. Reduce epochs: --epochs 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
