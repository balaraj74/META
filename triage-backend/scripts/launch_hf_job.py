#!/usr/bin/env python3
"""
launch_hf_job.py — Submit GRPO training to HuggingFace Compute (Spaces/Jobs).

This script automates the entire cloud training workflow:
  1. Verifies the dataset exists on HF Hub
  2. Uploads the training script
  3. Creates & starts a HF Space for training (A100 GPU)
  4. Monitors progress

Prerequisites:
  - HF_TOKEN environment variable set
  - Dataset already pushed (see push_dataset_to_hub.py)
  - huggingface_hub installed

Usage:
    python scripts/launch_hf_job.py \\
        --dataset balarajr/triage-grpo \\
        --model unsloth/Qwen3.6-27B-bnb-4bit \\
        --hub-model balarajr/triage-agent-27b
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def check_hf_token() -> str:
    """Verify HF token is available."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    if not token:
        logger.error(
            "No HF_TOKEN found. Set it via:\n"
            "  export HF_TOKEN='hf_...'\n"
            "  or: huggingface-cli login"
        )
        sys.exit(1)
    logger.info("HF token verified (ends: ...%s)", token[-4:])
    return token


def verify_dataset(repo_id: str, token: str):
    """Check that the dataset exists on HF Hub."""
    from huggingface_hub import dataset_info
    try:
        info = dataset_info(repo_id, token=token)
        logger.info("Dataset verified: %s (last modified: %s)", info.id, info.last_modified)
    except Exception as exc:
        logger.error("Dataset %s not found: %s", repo_id, exc)
        logger.error("Run: python scripts/push_dataset_to_hub.py --input data/grpo/combined_train.jsonl --repo %s", repo_id)
        sys.exit(1)


def create_training_space(
    hub_model: str,
    dataset_repo: str,
    base_model: str,
    token: str,
    hardware: str = "a100-large",
):
    """Create a HuggingFace Space to run training."""
    from huggingface_hub import HfApi, upload_file, create_repo

    api = HfApi(token=token)
    space_id = f"{hub_model}-trainer"

    # Create the Space
    logger.info("Creating Space: %s", space_id)
    try:
        create_repo(
            space_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware=hardware,
            token=token,
            private=True,
        )
    except Exception as exc:
        if "already exists" in str(exc).lower():
            logger.warning("Space already exists, updating...")
        else:
            raise

    # Upload training script
    train_script = SCRIPT_DIR / "train_grpo_hf.py"
    logger.info("Uploading training script...")
    api.upload_file(
        path_or_fileobj=str(train_script),
        path_in_repo="train_grpo_hf.py",
        repo_id=space_id,
        repo_type="space",
        token=token,
    )

    # Upload requirements
    req_file = SCRIPT_DIR / "hf_requirements.txt"
    api.upload_file(
        path_or_fileobj=str(req_file),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
        token=token,
    )

    # Upload Dockerfile
    dockerfile_content = f"""FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y git gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training script
COPY train_grpo_hf.py /app/train_grpo_hf.py
WORKDIR /app

# Run training
CMD ["python", "train_grpo_hf.py", \\
     "--model", "{base_model}", \\
     "--dataset", "{dataset_repo}", \\
     "--hub-model", "{hub_model}", \\
     "--push", \\
     "--epochs", "3", \\
     "--num-gen", "8", \\
     "--batch-size", "2", \\
     "--grad-accum", "4"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".Dockerfile", delete=False) as f:
        f.write(dockerfile_content)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="Dockerfile",
            repo_id=space_id,
            repo_type="space",
            token=token,
        )
        os.unlink(f.name)

    logger.info("Space created: https://huggingface.co/spaces/%s", space_id)
    return space_id


def print_manual_instructions(
    dataset_repo: str,
    base_model: str,
    hub_model: str,
):
    """Print manual instructions for running on HF."""
    print(f"""
{'═' * 70}
  MANUAL HF TRAINING INSTRUCTIONS
{'═' * 70}

  Option A: HuggingFace Notebook (Recommended for $35 budget)
  ─────────────────────────────────────────────────────────────

  1. Go to: https://huggingface.co/new-space
  2. Select: JupyterLab
  3. Hardware: A100 Large ($2.70/hr) or L40S ($1.80/hr)
  4. In the notebook, run:

     !pip install unsloth trl datasets transformers bitsandbytes peft accelerate wandb

     # Upload the training script
     !wget -O train_grpo_hf.py "https://huggingface.co/spaces/{hub_model}-trainer/resolve/main/train_grpo_hf.py"

     !python train_grpo_hf.py \\
         --model {base_model} \\
         --dataset {dataset_repo} \\
         --hub-model {hub_model} \\
         --push \\
         --epochs 3 \\
         --num-gen 8 \\
         --batch-size 2 \\
         --grad-accum 4

  Option B: HuggingFace Training Job (CLI)
  ─────────────────────────────────────────

     huggingface-cli jobs run \\
         --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \\
         --command "pip install unsloth trl datasets && python /app/train_grpo_hf.py --model {base_model} --dataset {dataset_repo} --hub-model {hub_model} --push" \\
         --hardware a100-large

  Option C: Google Colab (Free alternative)
  ──────────────────────────────────────────

     Upload train_grpo_hf.py to Colab with T4/A100 runtime.
     Use --model unsloth/Qwen3-14B-bnb-4bit for T4 compatibility.

  Budget estimate (A100 @ $2.70/hr):
    ~1600 steps × 8 gen/step ≈ 8-10 hours ≈ $22-$27

{'═' * 70}
""")


def main():
    parser = argparse.ArgumentParser(description="Launch HF training job")
    parser.add_argument("--dataset", default="balarajr/triage-grpo")
    parser.add_argument("--model", default="unsloth/Qwen3.6-27B-bnb-4bit")
    parser.add_argument("--hub-model", default="balarajr/triage-agent-27b")
    parser.add_argument("--hardware", default="a100-large",
                        choices=["a100-large", "a100-small", "l40s", "t4-medium"])
    parser.add_argument("--manual-only", action="store_true",
                        help="Only print manual instructions")
    args = parser.parse_args()

    if args.manual_only:
        print_manual_instructions(args.dataset, args.model, args.hub_model)
        return

    token = check_hf_token()
    verify_dataset(args.dataset, token)

    space_id = create_training_space(
        hub_model=args.hub_model,
        dataset_repo=args.dataset,
        base_model=args.model,
        token=token,
        hardware=args.hardware,
    )

    # Always print manual backup
    print_manual_instructions(args.dataset, args.model, args.hub_model)

    print(f"\n  Automated Space: https://huggingface.co/spaces/{space_id}")
    print(f"  Monitor logs at the Space URL above.\n")


if __name__ == "__main__":
    main()
