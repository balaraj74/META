#!/usr/bin/env python3
"""
Memory-efficient GRPO LoRA merger.

RAM budget on this machine: ~6 GB available.
  - Qwen3.5-4B in bfloat16  ≈ 8 GB on disk, but loaded lazily → ~4 GB peak
  - LoRA adapter (r=16)     ≈ 41 MB
  - Saving in 2 GB shards   → never holds full tensor set in RAM simultaneously

Strategy:
  1. Load base model with low_cpu_mem_usage=True (maps weights, not copies)
  2. Load and merge LoRA adapter (in-place, frees adapter after merge)
  3. Save in 2 GB safetensor shards so torch only serialises one shard at a time
"""

import gc
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER_DIR  = "/home/balaraj/META final/triage-backend/models/triage_grpo_output"
MERGED_DIR   = "/home/balaraj/META final/triage-backend/models/merged_grpo_final"
BASE_MODEL   = "Qwen/Qwen3.5-4B"

def ram_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0

def main() -> None:
    print("=" * 52)
    print("  TRIAGE GRPO LoRA Merger  (memory-efficient)")
    print("=" * 52)
    print(f"\n  Available RAM before load : {ram_gb():.1f} GB")

    # ── 1. Tokenizer (tiny — always safe) ───────────────────────────────────
    print("\n[1/4] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    # ── 2. Base model ────────────────────────────────────────────────────────
    # bfloat16 → ~8 GB on disk but mmap'd lazily; peak RAM ≈ 4–5 GB
    print(f"[2/4] Loading base model ({BASE_MODEL}) in bfloat16 on CPU …")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,          # smaller & numerically stable
        low_cpu_mem_usage=True,        # mmap — never copies full tensor set
        device_map="cpu",
    )
    print(f"      RAM after base load    : {ram_gb():.1f} GB")

    # ── 3. Apply + merge LoRA ────────────────────────────────────────────────
    print("[3/4] Applying LoRA adapter and merging …")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()  # injects weights, frees adapter tensors
    del base_model
    gc.collect()
    print(f"      RAM after merge        : {ram_gb():.1f} GB")

    # ── 4. Save in shards ────────────────────────────────────────────────────
    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"[4/4] Saving merged model to {MERGED_DIR}")
    print("      (2 GB shards — keeps peak save RAM low) …")

    model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="2GB",          # serialise 2 GB at a time, not all at once
    )
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"\n  RAM after save             : {ram_gb():.1f} GB")
    print("\n✅  Merge complete!")
    print(f"    → {MERGED_DIR}")
    files = list(Path(MERGED_DIR).iterdir())
    for f in sorted(files):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"       {f.name:<40} {size_mb:>6.1f} MB")

if __name__ == "__main__":
    main()
