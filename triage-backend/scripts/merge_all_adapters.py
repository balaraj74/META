#!/usr/bin/env python3
"""
merge_all_adapters.py v4 — Memory-efficient merge.
Uses gc.collect() aggressively and smaller shard sizes.
"""
import sys, os, json, shutil, gc
from pathlib import Path

PREV_MERGED = Path("/home/balaraj/META final/triage-backend/models/merged_grpo_final")
NEW_ADAPTER = Path("/home/balaraj/META final/triage_grpo_model")
OUTPUT_DIR  = Path("/home/balaraj/META final/triage-backend/models/merged_grpo_combined")

print("=" * 60)
print("  TRIAGE GRPO — COMBINE BOTH TRAINING RUNS")
print("=" * 60, flush=True)

assert PREV_MERGED.exists()
assert NEW_ADAPTER.exists()

with open(NEW_ADAPTER / "adapter_config.json") as f:
    acfg = json.load(f)
print(f"New adapter: LoRA r={acfg['r']}, alpha={acfg['lora_alpha']}", flush=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Step 1: Load model
print("\n[1/4] Loading previously merged model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    str(PREV_MERGED),
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
total = sum(p.numel() for p in model.parameters())
print(f"  ✓ Loaded: {total:,} parameters", flush=True)
gc.collect()

# Step 2: Prepare adapter
print("\n[2/4] Preparing adapter config...", flush=True)
patched = Path("/tmp/patched_adapter")
if patched.exists():
    shutil.rmtree(patched)
patched.mkdir(exist_ok=True)
shutil.copy2(NEW_ADAPTER / "adapter_model.safetensors",
             patched / "adapter_model.safetensors")

clean_cfg = {
    "base_model_name_or_path": str(PREV_MERGED),
    "bias": acfg.get("bias", "none"),
    "fan_in_fan_out": False,
    "inference_mode": False,
    "init_lora_weights": True,
    "lora_alpha": acfg.get("lora_alpha", 16),
    "lora_dropout": acfg.get("lora_dropout", 0.05),
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": acfg.get("r", 8),
    "target_modules": acfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]),
    "task_type": "CAUSAL_LM",
}
with open(patched / "adapter_config.json", "w") as f:
    json.dump(clean_cfg, f, indent=2)
print("  ✓ Done", flush=True)

# Step 3: Apply + merge
print("\n[3/4] Applying LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(model, str(patched), torch_dtype=torch.bfloat16)
print("  ✓ Adapter loaded", flush=True)

print("  Merging weights...", flush=True)
model = model.merge_and_unload()
print("  ✓ Merge complete", flush=True)
gc.collect()

# Step 4: Save
print(f"\n[4/4] Saving to {OUTPUT_DIR}...", flush=True)
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save with smaller shards to reduce peak memory
model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True, max_shard_size="1GB")
print("  ✓ Model weights saved", flush=True)
gc.collect()

# Save tokenizer
tok = AutoTokenizer.from_pretrained(str(NEW_ADAPTER), trust_remote_code=True)
tok.save_pretrained(str(OUTPUT_DIR))
print("  ✓ Tokenizer saved", flush=True)

ct = NEW_ADAPTER / "chat_template.jinja"
if ct.exists():
    shutil.copy2(ct, OUTPUT_DIR / "chat_template.jinja")

total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*") if f.is_file())
shards = len(list(OUTPUT_DIR.glob("model-*.safetensors")))

print(f"\n{'='*60}")
print(f"  ✅ MERGE COMPLETE")
print(f"  Output : {OUTPUT_DIR}")
print(f"  Size   : {total_size/1e9:.2f} GB  |  Shards: {shards}")
print(f"  Merged : OLD (r=16) + NEW (r={acfg['r']})")
print(f"{'='*60}", flush=True)

shutil.rmtree(patched, ignore_errors=True)
