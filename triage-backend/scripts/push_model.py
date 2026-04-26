#!/usr/bin/env python3
"""Create model repo and upload merged model to HuggingFace Hub."""
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
REPO_ID = "balarajr/triage-qwen2.5-7b-grpo"
MODEL_DIR = "/home/balaraj/META final/triage-backend/models/merged_grpo_final"

# Create repo
print(f"Creating repo: {REPO_ID}")
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    print("  ✅ Repo created/exists")
except Exception as e:
    print(f"  Repo create: {e}")

# Upload entire folder
print(f"\nUploading model from {MODEL_DIR}...")
print(f"  Files:")
total = 0
for f in sorted(os.listdir(MODEL_DIR)):
    sz = os.path.getsize(os.path.join(MODEL_DIR, f))
    total += sz
    print(f"    {f}: {sz/1e6:.1f} MB")
print(f"  Total: {total/1e9:.2f} GB")
print(f"\n  Starting upload (this will take a while for ~8 GB)...")

api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload TRIAGE Qwen2.5-7B GRPO model — hackathon submission",
)

print(f"\n✅ Model uploaded successfully!")
print(f"🔗 https://huggingface.co/{REPO_ID}")
