#!/usr/bin/env python3
"""Push Space files to HuggingFace Space."""
import os
from huggingface_hub import HfApi

api = HfApi()
SPACE_ID = "balarajr/triage-multi-agent-system"
SPACE_DIR = "/home/balaraj/META final/triage-backend/spaces"

print(f"Pushing to Space: {SPACE_ID}")

# Upload all space files
files = ['README.md', 'app.py', 'requirements.txt', 'Dockerfile']
for f in files:
    fp = os.path.join(SPACE_DIR, f)
    if os.path.exists(fp):
        print(f"  Uploading {f} ({os.path.getsize(fp):,} bytes)...")
        api.upload_file(
            path_or_fileobj=fp,
            path_in_repo=f,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"Update {f} for hackathon submission"
        )
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} not found")

print("\n✅ Space updated successfully!")
print(f"🔗 https://huggingface.co/spaces/{SPACE_ID}")
