#!/usr/bin/env python3
"""Push the fully-merged GRPO model to HuggingFace Hub.

Usage:
    # First login (one-time):
    huggingface-cli login

    # Then push:
    python3 scripts/push_to_hf.py
"""
import os
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
REPO_ID = "balarajr/triage-qwen3.5-4b-grpo"
MODEL_DIR = Path(__file__).parent.parent / "models" / "merged_grpo_combined"
COMMIT_MSG = "Upload combined GRPO model (both training runs merged)"

# ── Validate ──────────────────────────────────────────────────
if not MODEL_DIR.exists():
    print(f"❌ Model directory not found: {MODEL_DIR}")
    sys.exit(1)

safetensor_files = list(MODEL_DIR.glob("*.safetensors"))
if not safetensor_files:
    print(f"❌ No safetensors files in {MODEL_DIR}")
    sys.exit(1)

print(f"📦 Model directory: {MODEL_DIR}")
print(f"   Files: {len(list(MODEL_DIR.iterdir()))}")
print(f"   Safetensors shards: {len(safetensor_files)}")

# ── Check login ───────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, whoami
    user = whoami()
    print(f"✅ Logged in as: {user.get('name', user.get('fullname', 'unknown'))}")
except Exception:
    print("❌ Not logged in. Run: huggingface-cli login")
    sys.exit(1)

# ── Upload ────────────────────────────────────────────────────
api = HfApi()

print(f"\n🚀 Uploading to: https://huggingface.co/{REPO_ID}")
print("   This may take 10-20 minutes depending on your connection...\n")

try:
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=COMMIT_MSG,
        ignore_patterns=["*.py", "*.sh", "__pycache__"],
    )
    print(f"\n✅ Upload complete!")
    print(f"   → https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Run: huggingface-cli login")
    print("  2. Make sure you have write access to the repo")
    print(f"  3. Or create it: huggingface-cli repo create {REPO_ID.split('/')[-1]} --type model")
    sys.exit(1)
