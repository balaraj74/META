#!/usr/bin/env python3
"""
merge_adapters.py — Merge new LoRA adapter onto the already-merged GRPO model.

Strategy:
  1. Load the already-merged model (merged_grpo_final/) which has the OLD
     GRPO adapter baked into the base weights.
  2. Load the NEW LoRA adapter (triage_grpo_model/) on top of it.
  3. Merge and unload → produces a single weight file with BOTH training runs.
  4. Save to models/merged_grpo_combined/
"""

import sys
import os
import json
import shutil
from pathlib import Path

def main():
    # ─── Paths ───────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"

    PREV_MERGED = MODELS_DIR / "merged_grpo_final"
    NEW_ADAPTER = PROJECT_ROOT.parent / "triage_grpo_model"
    OUTPUT_DIR  = MODELS_DIR / "merged_grpo_combined"

    print("=" * 60)
    print("  TRIAGE GRPO — ADAPTER MERGE")
    print("=" * 60)
    print(f"  Previous merged model : {PREV_MERGED}")
    print(f"  New LoRA adapter      : {NEW_ADAPTER}")
    print(f"  Output                : {OUTPUT_DIR}")
    print("=" * 60)

    # ─── Validate paths ─────────────────────────────────────────────
    if not PREV_MERGED.exists():
        print(f"ERROR: Previous merged model not found: {PREV_MERGED}")
        sys.exit(1)

    if not NEW_ADAPTER.exists():
        print(f"ERROR: New adapter not found: {NEW_ADAPTER}")
        sys.exit(1)

    adapter_config = NEW_ADAPTER / "adapter_config.json"
    adapter_weights = NEW_ADAPTER / "adapter_model.safetensors"

    if not adapter_config.exists() or not adapter_weights.exists():
        print("ERROR: New adapter is missing adapter_config.json or adapter_model.safetensors")
        sys.exit(1)

    # ─── Read adapter config ────────────────────────────────────────
    with open(adapter_config) as f:
        acfg = json.load(f)

    print(f"\n  New adapter details:")
    print(f"    PEFT type      : {acfg.get('peft_type', 'N/A')}")
    print(f"    LoRA rank (r)  : {acfg.get('r', 'N/A')}")
    print(f"    LoRA alpha     : {acfg.get('lora_alpha', 'N/A')}")
    print(f"    Target modules : {acfg.get('target_modules', 'N/A')}")
    print(f"    Base model     : {acfg.get('base_model_name_or_path', 'N/A')}")

    # ─── Install deps if needed ──────────────────────────────────────
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install torch transformers peft accelerate")
        sys.exit(1)

    # ─── Step 1: Load previous merged model ─────────────────────────
    print(f"\n[1/4] Loading previously merged model from {PREV_MERGED}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(PREV_MERGED),
        torch_dtype=torch.bfloat16,
        device_map="cpu",           # Keep on CPU for merge — saves GPU RAM
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"  ✓ Loaded model: {model.config.architectures}")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ─── Step 2: Patch adapter config ────────────────────────────────
    # The new adapter's base_model_name_or_path points to HF hub.
    # We need to override it so PEFT loads onto our local merged model.
    print(f"\n[2/4] Patching adapter config to point to local merged model...")

    patched_adapter_dir = Path("/tmp/patched_adapter")
    patched_adapter_dir.mkdir(exist_ok=True)

    # Copy adapter weights
    shutil.copy2(adapter_weights, patched_adapter_dir / "adapter_model.safetensors")

    # Write patched config (base_model points to our merged model)
    patched_cfg = acfg.copy()
    patched_cfg["base_model_name_or_path"] = str(PREV_MERGED)
    patched_cfg["inference_mode"] = False  # Allow merging

    with open(patched_adapter_dir / "adapter_config.json", "w") as f:
        json.dump(patched_cfg, f, indent=2)

    print(f"  ✓ Adapter patched at {patched_adapter_dir}")

    # ─── Step 3: Apply new adapter and merge ─────────────────────────
    print(f"\n[3/4] Applying new LoRA adapter and merging weights...")
    model = PeftModel.from_pretrained(
        model,
        str(patched_adapter_dir),
        torch_dtype=torch.bfloat16,
    )

    # Count trainable params from adapter
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Adapter applied: {trainable:,} trainable / {total:,} total")

    # Merge LoRA weights into base model
    model = model.merge_and_unload()
    print(f"  ✓ Merge complete — all LoRA weights baked into base model")

    # ─── Step 4: Save the combined model ─────────────────────────────
    print(f"\n[4/4] Saving combined model to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        str(OUTPUT_DIR),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    print(f"  ✓ Model weights saved")

    # Save tokenizer from the new adapter (has latest chat template)
    tokenizer_src = NEW_ADAPTER
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_src),
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"  ✓ Tokenizer saved")

    # Copy chat template if present
    chat_template = NEW_ADAPTER / "chat_template.jinja"
    if chat_template.exists():
        shutil.copy2(chat_template, OUTPUT_DIR / "chat_template.jinja")
        print(f"  ✓ Chat template copied")

    # ─── Summary ─────────────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*") if f.is_file())
    num_shards = len(list(OUTPUT_DIR.glob("model-*.safetensors")))

    print(f"\n{'=' * 60}")
    print(f"  MERGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output path : {OUTPUT_DIR}")
    print(f"  Total size  : {total_size / 1e9:.2f} GB")
    print(f"  Shards      : {num_shards}")
    print(f"  Training runs merged:")
    print(f"    1. Previous GRPO (rank 16, baked into merged_grpo_final)")
    print(f"    2. New GRPO      (rank {acfg.get('r', '?')}, from triage_grpo_model)")
    print(f"{'=' * 60}")

    # Cleanup
    shutil.rmtree(patched_adapter_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
