#!/usr/bin/env python3
"""
merge_lora_adapters.py — Merge two LoRA adapters trained on the same base model.

This script combines the LoRA weights from Account 1 and Account 2 training runs
using TIES-style merging or simple linear interpolation.

Requirements:
    pip install peft transformers torch accelerate huggingface_hub

Usage:
    # From local directories:
    python merge_lora_adapters.py \
        --base-model unsloth/Qwen3.5-27B \
        --adapter1 ./models/grpo_hf_output_lora \
        --adapter2 ./models/grpo_hf_output_v2_lora \
        --output ./models/merged_triage_27b \
        --strategy linear --weight1 0.5 --weight2 0.5

    # From HuggingFace Hub:
    python merge_lora_adapters.py \
        --base-model unsloth/Qwen3.5-27B \
        --adapter1 balarajr/triage-agent-27b-lora \
        --adapter2 <account2>/triage-agent-27b-v2-lora \
        --output ./models/merged_triage_27b \
        --push --hub-model balarajr/triage-agent-27b-merged

    # TIES merge (better for diverse training data):
    python merge_lora_adapters.py \
        --base-model unsloth/Qwen3.5-27B \
        --adapter1 balarajr/triage-agent-27b-lora \
        --adapter2 <account2>/triage-agent-27b-v2-lora \
        --output ./models/merged_triage_27b \
        --strategy ties --density 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_lora")


def load_lora_state_dict(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from local dir or HF Hub."""
    from peft import PeftModel
    from huggingface_hub import snapshot_download

    # If it's a HF repo, download it first
    if "/" in adapter_path and not Path(adapter_path).exists():
        logger.info("Downloading adapter from Hub: %s", adapter_path)
        local_path = snapshot_download(adapter_path)
    else:
        local_path = adapter_path

    # Load the adapter_model.safetensors or adapter_model.bin
    adapter_dir = Path(local_path)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
        logger.info("Loaded %d tensors from safetensors", len(state_dict))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        logger.info("Loaded %d tensors from bin", len(state_dict))
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    return state_dict


def linear_merge(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
    w1: float = 0.5,
    w2: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Simple weighted average of two LoRA state dicts."""
    merged = {}

    # Find common keys
    common = set(sd1.keys()) & set(sd2.keys())
    only_1 = set(sd1.keys()) - set(sd2.keys())
    only_2 = set(sd2.keys()) - set(sd1.keys())

    if only_1:
        logger.warning("Keys only in adapter1 (%d): %s", len(only_1), list(only_1)[:5])
    if only_2:
        logger.warning("Keys only in adapter2 (%d): %s", len(only_2), list(only_2)[:5])

    for key in common:
        if sd1[key].shape != sd2[key].shape:
            logger.error("Shape mismatch for %s: %s vs %s", key, sd1[key].shape, sd2[key].shape)
            raise ValueError(f"Shape mismatch for {key} — adapters incompatible")
        merged[key] = w1 * sd1[key].float() + w2 * sd2[key].float()

    # Include unique keys from each (scaled)
    for key in only_1:
        merged[key] = w1 * sd1[key].float()
    for key in only_2:
        merged[key] = w2 * sd2[key].float()

    logger.info("Linear merge: %d common, %d unique-1, %d unique-2 → %d total",
                len(common), len(only_1), len(only_2), len(merged))
    return merged


def ties_merge(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
    density: float = 0.5,
    w1: float = 0.5,
    w2: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    TIES-style merge: Trim + Elect Sign + Disjoint Merge.
    Better at preserving complementary knowledge from each adapter.

    1. Trim: Keep only top-k% parameters by magnitude
    2. Elect sign: For each parameter, pick the dominant sign
    3. Disjoint merge: Average only same-sign parameters
    """
    merged = {}
    common = set(sd1.keys()) & set(sd2.keys())

    for key in common:
        t1 = sd1[key].float()
        t2 = sd2[key].float()

        if t1.shape != t2.shape:
            raise ValueError(f"Shape mismatch for {key}")

        # Step 1: Trim — zero out bottom (1-density)% by magnitude
        for t in [t1, t2]:
            flat = t.abs().flatten()
            if flat.numel() > 0:
                threshold = torch.quantile(flat, 1 - density)
                mask = t.abs() >= threshold
                t.mul_(mask)

        # Step 2: Elect sign — majority vote
        signs = torch.sign(t1) + torch.sign(t2)
        elected_sign = torch.sign(signs)  # +1 if both positive, -1 if both negative, 0 if conflict

        # Step 3: Disjoint merge — only average if sign agrees with elected
        mask1 = (torch.sign(t1) == elected_sign) | (elected_sign == 0)
        mask2 = (torch.sign(t2) == elected_sign) | (elected_sign == 0)

        weighted = torch.zeros_like(t1)
        count = torch.zeros_like(t1)

        weighted += w1 * t1 * mask1.float()
        count += mask1.float() * w1

        weighted += w2 * t2 * mask2.float()
        count += mask2.float() * w2

        # Avoid division by zero
        count = count.clamp(min=1e-8)
        merged[key] = weighted / count

    # Handle unique keys
    for key in set(sd1.keys()) - common:
        merged[key] = sd1[key].float()
    for key in set(sd2.keys()) - common:
        merged[key] = sd2[key].float()

    logger.info("TIES merge (density=%.2f): %d keys merged", density, len(merged))
    return merged


def slerp_merge(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
    t: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Spherical linear interpolation (SLERP) merge.
    Preserves magnitude relationships better than linear for LoRA weights.
    """
    merged = {}
    common = set(sd1.keys()) & set(sd2.keys())

    for key in common:
        v1 = sd1[key].float().flatten()
        v2 = sd2[key].float().flatten()

        if v1.shape != v2.shape:
            raise ValueError(f"Shape mismatch for {key}")

        # Normalize
        v1_norm = v1.norm()
        v2_norm = v2.norm()

        if v1_norm < 1e-8 or v2_norm < 1e-8:
            merged[key] = ((1 - t) * sd1[key].float() + t * sd2[key].float())
            continue

        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm

        # Compute angle
        cos_angle = torch.clamp(torch.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle = torch.acos(cos_angle)

        if angle < 1e-6:
            # Vectors nearly parallel — use linear
            result = (1 - t) * v1 + t * v2
        else:
            sin_angle = torch.sin(angle)
            coeff1 = torch.sin((1 - t) * angle) / sin_angle
            coeff2 = torch.sin(t * angle) / sin_angle
            result = coeff1 * v1 + coeff2 * v2

        merged[key] = result.reshape(sd1[key].shape)

    # Unique keys
    for key in set(sd1.keys()) - common:
        merged[key] = sd1[key].float()
    for key in set(sd2.keys()) - common:
        merged[key] = sd2[key].float()

    logger.info("SLERP merge (t=%.2f): %d keys merged", t, len(merged))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge two LoRA adapters")
    parser.add_argument("--base-model", required=True,
                        help="Base model ID (must match both adapters)")
    parser.add_argument("--adapter1", required=True,
                        help="Path or HF repo for adapter 1 (Account 1)")
    parser.add_argument("--adapter2", required=True,
                        help="Path or HF repo for adapter 2 (Account 2)")
    parser.add_argument("--output", default="./models/merged_triage_27b",
                        help="Output directory for merged model")
    parser.add_argument("--strategy", choices=["linear", "ties", "slerp"],
                        default="ties", help="Merge strategy (default: ties)")
    parser.add_argument("--weight1", type=float, default=0.5,
                        help="Weight for adapter 1")
    parser.add_argument("--weight2", type=float, default=0.5,
                        help="Weight for adapter 2")
    parser.add_argument("--density", type=float, default=0.5,
                        help="TIES density (fraction of weights to keep)")
    parser.add_argument("--push", action="store_true",
                        help="Push merged model to HF Hub")
    parser.add_argument("--hub-model", default=None,
                        help="HF Hub model ID for push")
    parser.add_argument("--lora-only", action="store_true",
                        help="Save merged LoRA adapter only (no base model merge)")

    args = parser.parse_args()

    # ── Load adapters ─────────────────────────────────────────────────────────
    logger.info("Loading adapter 1: %s", args.adapter1)
    sd1 = load_lora_state_dict(args.adapter1)

    logger.info("Loading adapter 2: %s", args.adapter2)
    sd2 = load_lora_state_dict(args.adapter2)

    # ── Validate compatibility ────────────────────────────────────────────────
    common_keys = set(sd1.keys()) & set(sd2.keys())
    logger.info("Common keys: %d | A1 total: %d | A2 total: %d",
                len(common_keys), len(sd1), len(sd2))

    if len(common_keys) == 0:
        logger.error("No common keys — adapters are incompatible!")
        sys.exit(1)

    # Check shapes match
    mismatched = []
    for key in common_keys:
        if sd1[key].shape != sd2[key].shape:
            mismatched.append((key, sd1[key].shape, sd2[key].shape))

    if mismatched:
        logger.error("Shape mismatches found — adapters not compatible:")
        for key, s1, s2 in mismatched[:10]:
            logger.error("  %s: %s vs %s", key, s1, s2)
        sys.exit(1)

    logger.info("✓ Adapters are compatible (same architecture)")

    # ── Merge ─────────────────────────────────────────────────────────────────
    logger.info("Merging with strategy: %s", args.strategy)

    if args.strategy == "linear":
        merged_sd = linear_merge(sd1, sd2, args.weight1, args.weight2)
    elif args.strategy == "ties":
        merged_sd = ties_merge(sd1, sd2, args.density, args.weight1, args.weight2)
    elif args.strategy == "slerp":
        merged_sd = slerp_merge(sd1, sd2, args.weight1)  # SLERP uses weight1 as t
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Cast back to bfloat16 for storage efficiency
    for key in merged_sd:
        merged_sd[key] = merged_sd[key].bfloat16()

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lora_only:
        # Save as a LoRA adapter (copy config from adapter1, replace weights)
        from huggingface_hub import snapshot_download
        import shutil

        if "/" in args.adapter1 and not Path(args.adapter1).exists():
            adapter1_dir = Path(snapshot_download(args.adapter1))
        else:
            adapter1_dir = Path(args.adapter1)

        # Copy config files
        for cfg_file in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "tokenizer.model"]:
            src = adapter1_dir / cfg_file
            if src.exists():
                shutil.copy2(src, output_dir / cfg_file)

        # Save merged weights
        try:
            from safetensors.torch import save_file
            save_file(merged_sd, str(output_dir / "adapter_model.safetensors"))
            logger.info("Saved merged adapter (safetensors) to %s", output_dir)
        except ImportError:
            torch.save(merged_sd, str(output_dir / "adapter_model.bin"))
            logger.info("Saved merged adapter (bin) to %s", output_dir)

    else:
        # Load base model and apply merged LoRA
        logger.info("Loading base model for full merge: %s", args.base_model)

        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # First save merged adapter temporarily
        temp_adapter_dir = output_dir / "_temp_merged_adapter"
        temp_adapter_dir.mkdir(exist_ok=True)

        # Copy adapter config from adapter1
        from huggingface_hub import snapshot_download
        import shutil

        if "/" in args.adapter1 and not Path(args.adapter1).exists():
            adapter1_dir = Path(snapshot_download(args.adapter1))
        else:
            adapter1_dir = Path(args.adapter1)

        for cfg_file in ["adapter_config.json"]:
            src = adapter1_dir / cfg_file
            if src.exists():
                shutil.copy2(src, temp_adapter_dir / cfg_file)

        try:
            from safetensors.torch import save_file
            save_file(merged_sd, str(temp_adapter_dir / "adapter_model.safetensors"))
        except ImportError:
            torch.save(merged_sd, str(temp_adapter_dir / "adapter_model.bin"))

        # Load base + merged adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        model = PeftModel.from_pretrained(base_model, str(temp_adapter_dir))
        model = model.merge_and_unload()

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Saved fully merged model to %s", output_dir)

        # Cleanup temp
        shutil.rmtree(temp_adapter_dir, ignore_errors=True)

    # ── Push to Hub ──────────────────────────────────────────────────────────
    if args.push and args.hub_model:
        logger.info("Pushing merged model to %s", args.hub_model)
        from huggingface_hub import HfApi
        api = HfApi()
        token = os.environ.get("HF_TOKEN")
        api.create_repo(args.hub_model, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.hub_model,
            token=token,
        )
        logger.info("✓ Merged model pushed to %s", args.hub_model)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"""
{'═' * 60}
  LORA MERGE COMPLETE
  ────────────────────────────────────────────────────
  Strategy:  {args.strategy.upper()}
  Adapter 1: {args.adapter1} (w={args.weight1})
  Adapter 2: {args.adapter2} (w={args.weight2})
  Output:    {args.output}
  Keys:      {len(merged_sd)} tensors merged
  {'Hub:       ' + args.hub_model if args.hub_model else ''}
{'═' * 60}
""")


if __name__ == "__main__":
    main()
