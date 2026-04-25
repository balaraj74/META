#!/usr/bin/env python3
"""
launch_training_v2.py — Launch V2 GRPO training on second HuggingFace account.

Differences from V1:
  - Uses train_grpo_hf_v2.py (expanded datasets, different seed)
  - Targets a DIFFERENT Space ID and model repo
  - Higher augment-max (300 vs 200) for more diverse data
  - Saves raw LoRA adapter separately for merge

Before running:
  1. Log into your second HF account:
     huggingface-cli logout
     huggingface-cli login  # <-- enter second account token

  2. Set environment variables:
     export HF_TOKEN=hf_YOUR_SECOND_ACCOUNT_TOKEN
     export ACCOUNT2_SPACE="<username2>/triage-grpo-training-v2"
     export ACCOUNT2_MODEL="<username2>/triage-agent-27b-v2"

  3. Run:
     python scripts/launch_training_v2.py --gpu l40s
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware

GPU_MAP = {
    "a100-large": SpaceHardware.A100_LARGE,
    "l40s":       SpaceHardware.L40SX1,
    "a10g-large": SpaceHardware.A10G_LARGE,
    "l4":         SpaceHardware.L4X1,
}

# These MUST be set for your second account
SPACE_ID = os.environ.get("ACCOUNT2_SPACE", "CHANGE_ME/triage-grpo-training-v2")
HUB_MODEL = os.environ.get("ACCOUNT2_MODEL", "CHANGE_ME/triage-agent-27b-v2")
SCRIPTS_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Launch V2 GRPO training on Account 2")
    parser.add_argument("--gpu", default="l40s", choices=GPU_MAP.keys())
    parser.add_argument("--space-id", default=SPACE_ID)
    parser.add_argument("--hub-model", default=HUB_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if "CHANGE_ME" in args.space_id or "CHANGE_ME" in args.hub_model:
        print("✗ Please set ACCOUNT2_SPACE and ACCOUNT2_MODEL environment variables")
        print("  Example:")
        print("    export ACCOUNT2_SPACE='myuser2/triage-grpo-training-v2'")
        print("    export ACCOUNT2_MODEL='myuser2/triage-agent-27b-v2'")
        sys.exit(1)

    api = HfApi()
    for attempt in range(5):
        try:
            who = api.whoami()
            break
        except Exception as exc:
            if attempt < 4:
                import time; time.sleep(2 ** (attempt + 1))
            else:
                print(f"✗ Cannot authenticate: {exc}")
                sys.exit(1)
    print(f"✓ Authenticated as: {who['name']}")

    # ── Staging ───────────────────────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="triage_v2_"))
    print(f"  Staging dir: {staging}")

    # Copy V2 training script
    v2_script = SCRIPTS_DIR / "train_grpo_hf_v2.py"
    if not v2_script.exists():
        print(f"✗ V2 script not found: {v2_script}")
        sys.exit(1)
    shutil.copy2(v2_script, staging / "train_grpo_hf.py")  # Rename to same name for Dockerfile

    # Requirements
    req_file = SCRIPTS_DIR / "hf_requirements.txt"
    if req_file.exists():
        shutil.copy2(req_file, staging / "requirements.txt")
    else:
        print("✗ hf_requirements.txt not found")
        sys.exit(1)

    # Dockerfile — same structure as V1 but with V2 CMD
    cmd_parts = [
        "python3", "train_grpo_hf.py",
        "--model", "unsloth/Qwen3.5-27B",
        "--dataset", "balarajr/triage-grpo",
        "--hub-model", args.hub_model,
        "--push",
        "--epochs", str(args.epochs),
        "--batch-size", "1",
        "--grad-accum", "1",
        "--num-gen", "2",
        "--lr", "5e-6",
        "--augment-hf",
        "--augment-max", "300",  # More augmentation for V2
    ]
    if args.quick:
        cmd_parts.append("--quick")

    cmd_json = ", ".join(f'"{p}"' for p in cmd_parts)

    dockerfile = f"""FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential && \\
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch
RUN pip install --no-cache-dir torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unsloth
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# V2 training script
COPY train_grpo_hf.py .

# V2 training command
CMD [{cmd_json}]
"""
    (staging / "Dockerfile").write_text(dockerfile)

    readme = f"""---
title: TRIAGE GRPO Training V2
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# TRIAGE Agent GRPO Training — V2 (Account 2)

Training Qwen3.5-27B on expanded medical dataset (V2).
**This is the second training run for LoRA adapter merging.**

GPU: {args.gpu} | Epochs: {args.epochs}
"""
    (staging / "README.md").write_text(readme)

    print(f"\n  Files staged:")
    for f in sorted(staging.iterdir()):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    if args.dry_run:
        print(f"\n  --dry-run: Files prepared in {staging}")
        return

    # ── Create Space + Upload ─────────────────────────────────────────────────
    print(f"\n  Creating Space: {args.space_id}")
    try:
        api.create_repo(args.space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        print("  ✓ Space created")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        api.add_space_secret(args.space_id, "HF_TOKEN", hf_token)
        print("  ✓ HF_TOKEN set")

    print(f"\n  Uploading to {args.space_id}...")
    api.upload_folder(folder_path=str(staging), repo_id=args.space_id, repo_type="space")
    print("  ✓ Uploaded")

    hw = GPU_MAP[args.gpu]
    print(f"\n  Requesting GPU: {args.gpu}")
    try:
        api.request_space_hardware(args.space_id, hw)
        print(f"  ✓ GPU requested")
    except Exception as exc:
        print(f"  ✗ GPU request failed: {exc}")

    print(f"""
{'═' * 60}
  V2 TRAINING JOB SUBMITTED (Account 2)
  ──────────────────────────────────────────────────────
  Space:    https://huggingface.co/spaces/{args.space_id}
  GPU:      {args.gpu}
  Model:    unsloth/Qwen3.5-27B
  Output:   https://huggingface.co/{args.hub_model}
  Strategy: V2 — expanded medical + ethics datasets

  AFTER BOTH TRAININGS COMPLETE:
    python scripts/merge_lora_adapters.py \\
      --base-model unsloth/Qwen3.5-27B \\
      --adapter1 balarajr/triage-agent-27b-lora \\
      --adapter2 {args.hub_model}-lora \\
      --output ./models/merged_triage_27b \\
      --strategy ties \\
      --push --hub-model balarajr/triage-agent-27b-merged
{'═' * 60}
""")

    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
