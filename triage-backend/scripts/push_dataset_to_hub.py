#!/usr/bin/env python3
"""
push_dataset_to_hub.py — Push local GRPO dataset to HuggingFace Hub.

Usage:
    python scripts/push_dataset_to_hub.py \\
        --input data/grpo/combined_train.jsonl \\
        --repo balarajr/triage-grpo

Requires: HF_TOKEN env variable or `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Push GRPO dataset to HF Hub")
    parser.add_argument("--input", required=True, help="Path to combined JSONL")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g. balarajr/triage-grpo)")
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load JSONL
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), input_path)

    # Extract only the 'prompt' field (GRPO only needs prompts)
    prompts = []
    for r in records:
        if "prompt" in r:
            prompts.append(r["prompt"])
        else:
            logger.warning("Record missing 'prompt' field, skipping")

    logger.info("Extracted %d prompts", len(prompts))

    # Create HF dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    logger.info("Dataset created: %s", dataset)
    logger.info("Pushing to %s ...", args.repo)

    dataset.push_to_hub(
        args.repo,
        split=args.split,
        private=args.private,
    )

    print(f"\n{'═' * 50}")
    print(f"  Dataset pushed to HF Hub")
    print(f"  Repo:    https://huggingface.co/datasets/{args.repo}")
    print(f"  Split:   {args.split}")
    print(f"  Prompts: {len(prompts)}")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
