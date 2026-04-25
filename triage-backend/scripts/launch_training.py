#!/usr/bin/env python3
"""
launch_training.py — Launch GRPO training on HuggingFace A100 infrastructure.

This script:
1. Creates a HF Space with Docker SDK + A100 GPU
2. Uploads the training script, requirements, and Dockerfile
3. Sets secrets (HF_TOKEN, WANDB_API_KEY)
4. The Space auto-builds and starts training

Budget: $35 @ ~$4.13/hr (A100-small) = ~8.5 hours
        $35 @ ~$1.10/hr (L40S) = ~31 hours

Usage:
    python scripts/launch_training.py --gpu a100-small
    python scripts/launch_training.py --gpu l40s --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

# ── GPU Options ──────────────────────────────────────────────────────────────
GPU_MAP = {
    "a100-large": SpaceHardware.A100_LARGE,   # 80GB — ~$6.67/hr
    "l40s":       SpaceHardware.L40SX1,        # 48GB — ~$1.10/hr
    "a10g-large": SpaceHardware.A10G_LARGE,    # 24GB — ~$3.15/hr
    "a10g-small": SpaceHardware.A10G_SMALL,    # 24GB — ~$1.05/hr
    "l4":         SpaceHardware.L4X1,           # 24GB — ~$0.80/hr
    "t4-medium":  SpaceHardware.T4_MEDIUM,     # 16GB — ~$0.60/hr
}

SPACE_ID = "balarajr/triage-grpo-training"
SCRIPTS_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Launch GRPO training on HF")
    parser.add_argument("--gpu", default="a100-large", choices=GPU_MAP.keys(),
                        help="GPU tier (default: a100-small)")
    parser.add_argument("--space-id", default=SPACE_ID,
                        help="HF Space ID for training")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare files but don't create Space")
    parser.add_argument("--model", default="unsloth/Qwen3.6-27B-bnb-4bit",
                        help="Base model to train")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip HF dataset augmentation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (10 steps)")
    args = parser.parse_args()

    api = HfApi()
    # Retry whoami with backoff (rate-limited after heavy uploads)
    for attempt in range(5):
        try:
            who = api.whoami()
            break
        except Exception as exc:
            if attempt < 4:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, retrying in {wait}s... ({exc})")
                import time; time.sleep(wait)
            else:
                print(f"✗ Cannot authenticate: {exc}")
                sys.exit(1)
    print(f"✓ Authenticated as: {who['name']}")

    # ── Prepare staging directory ────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="triage_training_"))
    print(f"  Staging dir: {staging}")

    # Copy training script
    train_script = SCRIPTS_DIR / "train_grpo_hf.py"
    if not train_script.exists():
        print(f"✗ Training script not found: {train_script}")
        sys.exit(1)
    shutil.copy2(train_script, staging / "train_grpo_hf.py")

    # Copy requirements
    req_file = SCRIPTS_DIR / "hf_requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, staging / "requirements.txt")
    else:
        print("✗ hf_requirements.txt not found")
        sys.exit(1)

    # Generate Dockerfile with correct CMD args
    cmd_parts = [
        "python3", "train_grpo_hf.py",
        "--model", args.model,
        "--dataset", "balarajr/triage-grpo",
        "--hub-model", "balarajr/triage-agent-27b",
        "--push",
        "--epochs", str(args.epochs),
        "--batch-size", "1",
        "--grad-accum", "4",
        "--num-gen", "4",
        "--lr", "5e-6",
    ]
    if not args.no_augment:
        cmd_parts.extend(["--augment-hf", "--augment-max", "200"])
    if args.quick:
        cmd_parts.append("--quick")

    cmd_json = ", ".join(f'"{p}"' for p in cmd_parts)

    dockerfile = f"""FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-dev git curl && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (standard PyPI packages)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Install unsloth from git (needs separate step — PEP 508 git URL)
RUN pip3 install --no-cache-dir --break-system-packages \\
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Copy training script
COPY train_grpo_hf.py .

# Training command
CMD [{cmd_json}]
"""
    (staging / "Dockerfile").write_text(dockerfile)

    # README for Space metadata
    readme = f"""---
title: TRIAGE GRPO Training
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# TRIAGE Agent GRPO Training

Training {args.model} on `balarajr/triage-grpo` dataset.
GPU: {args.gpu} | Epochs: {args.epochs}
"""
    (staging / "README.md").write_text(readme)

    print(f"\n  Files staged:")
    for f in sorted(staging.iterdir()):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    if args.dry_run:
        print(f"\n  --dry-run: Files prepared in {staging}")
        print("  To upload manually:")
        print(f"    huggingface-cli upload {args.space_id} {staging} . --repo-type space")
        return

    # ── Create Space ─────────────────────────────────────────────────────────
    print(f"\n  Creating Space: {args.space_id}")
    try:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"  ✓ Space created/exists")
    except Exception as exc:
        print(f"  ✗ Failed to create Space: {exc}")
        sys.exit(1)

    # ── Set secrets ──────────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        api.add_space_secret(args.space_id, "HF_TOKEN", hf_token)
        print("  ✓ HF_TOKEN secret set")
    else:
        print("  ⚠ HF_TOKEN not in env — set it manually in Space settings")

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        api.add_space_secret(args.space_id, "WANDB_API_KEY", wandb_key)
        print("  ✓ WANDB_API_KEY secret set")
    else:
        print("  ℹ No WANDB_API_KEY — training will log locally only")

    # ── Upload files ─────────────────────────────────────────────────────────
    print(f"\n  Uploading files to {args.space_id}...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.space_id,
        repo_type="space",
    )
    print("  ✓ Files uploaded")

    # ── Set GPU hardware ─────────────────────────────────────────────────────
    hw = GPU_MAP[args.gpu]
    print(f"\n  Requesting GPU: {args.gpu} ({hw})")
    try:
        api.request_space_hardware(args.space_id, hw)
        print(f"  ✓ GPU requested — Space will build and start training")
    except Exception as exc:
        print(f"  ✗ GPU request failed: {exc}")
        print(f"  → Set GPU manually: https://huggingface.co/spaces/{args.space_id}/settings")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"""
{'═' * 60}
  TRAINING JOB SUBMITTED
  ──────────────────────────────────────────────────────
  Space:    https://huggingface.co/spaces/{args.space_id}
  GPU:      {args.gpu}
  Model:    {args.model}
  Dataset:  balarajr/triage-grpo (3,000 prompts)
  Augment:  {'Yes (200/source × 10 sources)' if not args.no_augment else 'No'}
  Epochs:   {args.epochs}
  
  MONITOR:
    → Space logs: https://huggingface.co/spaces/{args.space_id}/logs
    → Output model: https://huggingface.co/balarajr/triage-agent-27b
  
  BUDGET:
    → Estimated: ~6-8 hours on A100-small (~$25-33)
    → Auto-pause: Space pauses when Docker exits
    → STOP EARLY: https://huggingface.co/spaces/{args.space_id}/settings
{'═' * 60}
""")

    # Cleanup staging
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
