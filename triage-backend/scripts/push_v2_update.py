#!/usr/bin/env python3
"""Push updated V2 training script to HuggingFace Space."""
import sys, os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("✗ HF_TOKEN environment variable not set. Export it first:")
    print("  export HF_TOKEN=hf_your_token_here")
    sys.exit(1)
SPACE_ID = "Harshavardhan7975/triage-grpo-training-v2"
SCRIPT = Path(__file__).parent / "train_grpo_hf_v2.py"

api = HfApi(token=TOKEN)

# 1. Auth check
try:
    who = api.whoami()
    print(f"✓ Authenticated as: {who['name']}")
except Exception as e:
    print(f"✗ Auth failed: {e}")
    sys.exit(1)

# 2. Check space exists
try:
    info = api.repo_info(repo_id=SPACE_ID, repo_type="space")
    print(f"✓ Space exists: {SPACE_ID}")
except Exception as e:
    print(f"✗ Space not found: {e}")
    print("  Creating space...")
    api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker",
                    space_hardware="l40s", private=False)
    print(f"✓ Space created: {SPACE_ID}")

# 3. Upload updated V2 training script (renamed to train_grpo_hf.py for Dockerfile)
if not SCRIPT.exists():
    print(f"✗ Script not found: {SCRIPT}")
    sys.exit(1)

print(f"  Uploading {SCRIPT.name} -> train_grpo_hf.py ...")
api.upload_file(
    path_or_fileobj=str(SCRIPT),
    path_in_repo="train_grpo_hf.py",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="V2: expanded to 24 datasets (medical+ethics+crisis+reasoning)",
)
print("✓ Updated training script pushed!")

# 4. Also upload requirements
req = Path(__file__).parent / "hf_requirements.txt"
if req.exists():
    api.upload_file(
        path_or_fileobj=str(req),
        path_in_repo="requirements.txt",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="V2: requirements",
    )
    print("✓ Requirements pushed!")

# 5. Upload Dockerfile
dockerfile_content = '''FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential && \\
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# CRITICAL: Use cu128 index + torch>=2.10.0 to match V1's working setup
# cu124 gives older torch missing torch.utils._pytree.register_constant
RUN pip install --no-cache-dir "torch>=2.10.0" --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pin torchao to version compatible with torch 2.10
RUN pip install --no-cache-dir "torchao>=0.8.0"

RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

COPY train_grpo_hf.py .

CMD ["python3", "train_grpo_hf.py", "--model", "unsloth/Qwen3.5-27B", "--dataset", "balarajr/triage-grpo", "--hub-model", "Harshavardhan7975/triage-agent-27b-v2", "--push", "--epochs", "3", "--batch-size", "2", "--grad-accum", "2", "--num-gen", "2", "--lr", "5e-6", "--augment-hf", "--augment-max", "300"]
'''
api.upload_file(
    path_or_fileobj=dockerfile_content.encode(),
    path_in_repo="Dockerfile",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="V2: Dockerfile with 24-dataset augmentation",
)
print("✓ Dockerfile pushed!")

# 6. Upload README
readme = """---
title: TRIAGE GRPO Training V2
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# TRIAGE Agent GRPO Training — V2 (24 Expanded Datasets)

Training Qwen3.5-27B with **24 diverse HF datasets** for maximum coverage.
Seed=456 (different from V1 seed=123) for complementary LoRA adapter merge.
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="V2: README update",
)
print("✓ README pushed!")

# 7. Set HF_TOKEN secret
try:
    api.add_space_secret(repo_id=SPACE_ID, key="HF_TOKEN", value=TOKEN)
    print("✓ HF_TOKEN secret set!")
except Exception as e:
    print(f"⚠ Secret set warning: {e}")

# 8. Request hardware
try:
    api.request_space_hardware(repo_id=SPACE_ID, hardware="a100-large")
    print("✓ L40S hardware requested!")
except Exception as e:
    print(f"⚠ Hardware request: {e}")

print("\n🚀 V2 Space updated with 24 datasets! Monitor at:")
print(f"   https://huggingface.co/spaces/{SPACE_ID}")
