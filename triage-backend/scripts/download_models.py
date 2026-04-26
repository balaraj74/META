#!/usr/bin/env python3
"""Download TRIAGE runtime models to local disk.

Usage:
  python scripts/download_models.py
  python scripts/download_models.py --clinical-only
  python scripts/download_models.py --operations-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "clinical": {
        "repo_id": os.getenv("CLINICAL_MODEL", "Intelligent-Internet/II-Medical-8B"),
        "local_dir": os.getenv("CLINICAL_MODEL_PATH", str(ROOT / "models" / "ii-medical-8b")),
    },
    "operations": {
        "repo_id": os.getenv("OPERATIONS_MODEL", "Qwen/Qwen3-4B-Instruct"),
        "local_dir": os.getenv("OPERATIONS_MODEL_PATH", str(ROOT / "models" / "qwen3-4b")),
    },
}


def download_model(name: str, config: dict[str, str]) -> None:
    local_dir = Path(config["local_dir"])
    print(f"\nDownloading {name} model: {config['repo_id']}")
    print(f"  Destination: {local_dir}")

    if local_dir.is_dir() and any(local_dir.iterdir()):
        print("  Already exists; delete the folder to re-download.")
        return

    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=str(local_dir),
        token=os.getenv("HF_TOKEN") or None,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )
    print("  Downloaded successfully")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-only", action="store_true")
    parser.add_argument("--operations-only", action="store_true")
    args = parser.parse_args()

    if args.clinical_only and args.operations_only:
        parser.error("Choose only one of --clinical-only or --operations-only")

    if args.clinical_only:
        targets = ["clinical"]
    elif args.operations_only:
        targets = ["operations"]
    else:
        targets = list(MODELS)

    for target in targets:
        download_model(target, MODELS[target])

    print("\nAll requested models are available locally.")


if __name__ == "__main__":
    main()
