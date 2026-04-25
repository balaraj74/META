#!/usr/bin/env python3
"""
train_grpo_hf_v2.py — GRPO training variant for SECOND HuggingFace account.

Differences from v1 (Account 1):
  - Different random seed (456 vs 123) → different prompt ordering
  - Expanded medical/clinical dataset sources (heavier reasoning)
  - Different augmentation emphasis (ethics + clinical reasoning)
  - Pushes to a DIFFERENT model repo for later merge

Both variants must share:
  - Same base model: unsloth/Qwen3.5-27B
  - Same LoRA architecture: rank=16, alpha=16, same target modules
  - Same reward functions (critical for GRPO consistency)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_grpo_hf_v2")

_STATUS = "initializing"


def _start_health_server(port: int = 7860):
    """Minimal HTTP health-check — required by HF Spaces."""
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {"status": "ok", "training": _STATUS, "variant": "v2"}, indent=2
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Health-check server listening on :%d", port)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — MUST MATCH V1 ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    # Model — MUST be identical to Account 1
    "model": "unsloth/Qwen3.5-27B",
    "max_seq_length": 512,

    # LoRA — MUST match Account 1 architecture for merge compatibility
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "lora_targets": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # GRPO — same constraints for L40S 48GB
    "num_generations": 2,
    "max_completion_length": 128,
    "temperature": 0.9,

    # Training
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 1,
    "lr": 5e-6,
    "logging_steps": 1,
    "save_steps": 200,

    # Paths — different output repo for Account 2
    "output_dir": "./models/grpo_hf_output_v2",
    "dataset": "./data/grpo/combined_train.jsonl",
}


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS — MUST BE IDENTICAL TO V1
# ═══════════════════════════════════════════════════════════════════════════════

def reward_json_valid(completions: list[list[dict]], **_kw) -> list[float]:
    scores = []
    for gen_list in completions:
        text = gen_list[0]["content"] if gen_list else ""
        try:
            json.loads(text)
            scores.append(2.0)
        except (json.JSONDecodeError, TypeError):
            if re.search(r'\{[^}]+\}', text):
                scores.append(0.5)
            else:
                scores.append(0.0)
    return scores


def reward_has_reasoning(completions: list[list[dict]], **_kw) -> list[float]:
    scores = []
    for gen_list in completions:
        text = gen_list[0]["content"] if gen_list else ""
        if '"reasoning"' in text and len(text) > 30:
            scores.append(2.0)
        elif "reason" in text.lower():
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


def reward_action_type(completions: list[list[dict]], **_kw) -> list[float]:
    valid_actions = {
        "TRIAGE_PATIENT", "ASSIGN_TREATMENT", "UPDATE_EHR",
        "TRANSFER_TO_ICU", "TRANSFER_TO_WARD", "ACTIVATE_OVERFLOW",
        "ORDER_MEDICATION", "FLAG_POLICY_VIOLATION",
        "OVERRIDE_DECISION", "REQUEST_STAFF", "VERIFY_INSURANCE",
    }
    scores = []
    for gen_list in completions:
        text = gen_list[0]["content"] if gen_list else ""
        found = any(act in text for act in valid_actions)
        scores.append(2.0 if found else 0.0)
    return scores


def reward_priority_range(completions: list[list[dict]], **_kw) -> list[float]:
    scores = []
    for gen_list in completions:
        text = gen_list[0]["content"] if gen_list else ""
        m = re.search(r'"priority"\s*:\s*(\d+)', text)
        if m:
            p = int(m.group(1))
            scores.append(1.5 if 1 <= p <= 10 else 0.0)
        else:
            scores.append(0.0)
    return scores


def reward_length_penalty(completions: list[list[dict]], **_kw) -> list[float]:
    scores = []
    for gen_list in completions:
        text = gen_list[0]["content"] if gen_list else ""
        length = len(text)
        if 50 < length < 400:
            scores.append(1.0)
        elif length >= 400:
            scores.append(0.3)
        else:
            scores.append(0.0)
    return scores


def reward_patient_id(completions: list[list[dict]], **_kw) -> list[float]:
    scores = []
    for gen_list in completions:
        text = gen_list[0]["content"] if gen_list else ""
        m = re.search(r'"target_id"\s*:\s*(\d+)', text)
        scores.append(1.5 if m else 0.0)
    return scores


REWARD_FUNCS = [
    reward_json_valid,
    reward_has_reasoning,
    reward_action_type,
    reward_priority_range,
    reward_length_penalty,
    reward_patient_id,
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_prompts(path: str) -> list[str]:
    """Load prompts from JSONL or HF dataset."""
    if path.startswith("balarajr/") or "/" in path and not Path(path).exists():
        logger.info("Loading dataset from HF Hub: %s", path)
        from datasets import load_dataset
        ds = load_dataset(path, split="train")
        return ds["prompt"]

    logger.info("Loading dataset from local: %s", path)
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                prompts.append(record["prompt"])
    return prompts


# ═══════════════════════════════════════════════════════════════════════════════
# V2 AUGMENTATION — EXPANDED CLINICAL + ETHICS FOCUS
# ═══════════════════════════════════════════════════════════════════════════════

# V2 uses DIFFERENT datasets than V1 for complementary coverage
_HF_SOURCES_V2 = [
    # Heavy medical reasoning (shared with V1 but different sampling seed)
    ("TachyHealth/medical_grpo", None, ["question", "input", "prompt"]),
    ("Intelligent-Internet/II-Medical-RL", None, ["question", "prompt"]),
    ("BAAI/AquilaMed-RL", None, ["instruction", "input"]),
    # More clinical data — V2 exclusive or higher max_per_source
    ("openlifescienceai/medmcqa", None, ["question"]),
    ("sdiazlor/medical-reasoning-dataset", None, ["question", "input"]),
    ("lavita/ChatDoctor-HealthCareMagic-100k", None, ["input", "instruction"]),
    # Ethics — V2 gets ALL splits (not just 2)
    ("hendrycks/ethics", "utilitarianism", ["input", "text"]),
    ("hendrycks/ethics", "deontology", ["input", "scenario"]),
    ("hendrycks/ethics", "justice", ["input", "scenario"]),
    ("hendrycks/ethics", "virtue", ["input", "scenario"]),
    # Clinical NLP & reasoning
    ("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", ["question", "context"]),
    ("GBaker/MedQA-USMLE-4-options", None, ["question"]),
]

_AUG_AGENTS = ["er_triage", "icu_management", "pharmacy", "cmo_oversight", "hr_rostering", "it_systems"]
_AUG_CRISES = ["mass_casualty", "outbreak", "equipment_failure", "staff_shortage",
               "chemical_spill", "power_outage", "water_contamination", "active_shooter"]
_AUG_ROLE = {
    "er_triage": "ER Triage — patient assessment",
    "icu_management": "ICU Management — bed allocation",
    "pharmacy": "Pharmacy — medication orders",
    "cmo_oversight": "CMO Oversight — policy enforcement",
    "hr_rostering": "HR Rostering — staff allocation",
    "it_systems": "IT Systems — EHR integrity",
}
_AUG_ACTIONS = {
    "er_triage": "TRIAGE_PATIENT, ASSIGN_TREATMENT, UPDATE_EHR",
    "icu_management": "TRANSFER_TO_ICU, TRANSFER_TO_WARD, ACTIVATE_OVERFLOW",
    "pharmacy": "ORDER_MEDICATION, FLAG_POLICY_VIOLATION",
    "cmo_oversight": "OVERRIDE_DECISION, ACTIVATE_OVERFLOW",
    "hr_rostering": "REQUEST_STAFF, FLAG_POLICY_VIOLATION",
    "it_systems": "UPDATE_EHR, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE",
}


def _augment_from_hf(max_per_source: int) -> list[str]:
    """V2 augmentation — different seed + expanded sources."""
    import random as _rng
    rng = _rng.Random(456)  # ← DIFFERENT SEED from V1 (123)
    prompts = []
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not available for augmentation")
        return prompts

    for repo, config, fields in _HF_SOURCES_V2:
        try:
            kwargs = {"split": "train", "streaming": True}
            if config:
                kwargs["name"] = config
            ds = load_dataset(repo, **kwargs)
            count = 0
            for row in ds:
                if count >= max_per_source:
                    break
                row_d = dict(row)
                text = ""
                for f in fields:
                    v = row_d.get(f)
                    if v and isinstance(v, str) and len(v.strip()) > 15:
                        text = v.strip()[:500]
                        break
                if not text:
                    continue
                agent = rng.choice(_AUG_AGENTS)
                crisis = rng.choice(_AUG_CRISES)
                icu = rng.randint(30, 98)
                crit = rng.randint(1, 10)
                pid_block = "\n".join(
                    f"  P-{rng.randint(1,99):03d}: CRITICAL — BP {rng.randint(55,95)}/{rng.randint(25,65)}, HR {rng.randint(95,160)}"
                    for _ in range(min(crit, 4))
                )
                prompt = (
                    f"You are the {agent.upper()} agent in a hospital crisis simulation.\n\n"
                    f"CRISIS: {crisis.upper()}\nSTEP: {rng.randint(0,19)}/20\n"
                    f"ICU OCCUPANCY: {icu}% ({icu*20//100}/20 beds)\n"
                    f"CRITICAL PATIENTS ({crit} total — top 5):\n{pid_block}\n"
                    f"VIOLATIONS INJECTED: {rng.randint(0,5)} | CAUGHT: {rng.randint(0,3)}\n"
                    f"SURVIVAL RATE: {rng.uniform(82,100):.1f}%\n\n"
                    f"MEDICAL CONTEXT:\n{text}\n\n"
                    f"Your role: {_AUG_ROLE[agent]}\n\n"
                    f"Respond with ONLY valid JSON:\n"
                    f'{{\n  "action_type": "<one of: {_AUG_ACTIONS[agent]}>",\n'
                    f'  "target_id": <patient ID integer or 0>,\n'
                    f'  "priority": <integer 1-10>,\n'
                    f'  "reasoning": "<1-2 sentences citing specific data>"\n}}'
                )
                prompts.append(prompt)
                count += 1
            logger.info("Augmented %d prompts from %s", count, repo)
        except Exception as exc:
            logger.warning("Augmentation failed for %s: %s", repo, exc)

    rng.shuffle(prompts)
    return prompts


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GRPO training V2 — Account 2")
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--output", default=DEFAULTS["output_dir"])
    parser.add_argument("--hub-model", default=None,
                        help="HF Hub model ID to push (e.g. <user2>/triage-agent-27b-v2)")
    parser.add_argument("--push", action="store_true", help="Push to HF Hub after training")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 steps)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--lora-r", type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--num-gen", type=int, default=DEFAULTS["num_generations"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--augment-hf", action="store_true",
                        help="Augment training with HF medical datasets (V2 expanded)")
    parser.add_argument("--augment-max", type=int, default=300,
                        help="Max samples per HF source (default 300 for V2)")

    args = parser.parse_args()
    global _STATUS

    _start_health_server()
    _STATUS = "loading_model"

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=DEFAULTS["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    # ── Apply LoRA — SAME architecture as V1 ─────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=DEFAULTS["lora_alpha"],
        lora_dropout=DEFAULTS["lora_dropout"],
        target_modules=DEFAULTS["lora_targets"],
        use_gradient_checkpointing="unsloth",
        random_state=456,  # Different random state from V1 (42)
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    _STATUS = "loading_data"
    try:
        prompts = load_dataset_prompts(args.dataset)
        logger.info("Loaded %d prompts from %s", len(prompts), args.dataset)
    except Exception as exc:
        logger.warning("Dataset load failed (%s), generating inline", exc)
        prompts = []

    # V2 augmentation — more data, different sources
    if args.augment_hf:
        _STATUS = "augmenting"
        logger.info("V2 augmentation: max %d per source from %d sources",
                     args.augment_max, len(_HF_SOURCES_V2))
        aug = _augment_from_hf(args.augment_max)
        prompts.extend(aug)
        logger.info("After V2 augmentation: %d total prompts", len(prompts))

    if not prompts:
        logger.error("No prompts available — generating fallback set")
        import random
        rng = random.Random(456)
        for i in range(500):
            agent = rng.choice(_AUG_AGENTS)
            crisis = rng.choice(_AUG_CRISES)
            prompts.append(
                f"You are the {agent.upper()} agent.\n"
                f"CRISIS: {crisis.upper()}\nSTEP: {rng.randint(0,19)}/20\n"
                f"ICU OCCUPANCY: {rng.randint(30,98)}%\n"
                f"Respond with ONLY valid JSON:\n"
                f'{{"action_type": "<one of: {_AUG_ACTIONS[agent]}>", '
                f'"target_id": <int>, "priority": <1-10>, "reasoning": "<text>"}}'
            )

    # Build HF dataset
    from datasets import Dataset

    ds = Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": p}] for p in prompts
        ]
    })

    logger.info("V2 training dataset: %d examples", len(ds))

    # ── Configure GRPO Trainer ────────────────────────────────────────────────
    _STATUS = "training"
    from trl import GRPOConfig, GRPOTrainer

    max_steps = 10 if args.quick else -1

    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs if max_steps == -1 else 1,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=DEFAULTS["logging_steps"],
        save_steps=DEFAULTS["save_steps"],
        save_total_limit=3,
        report_to="none",
        num_generations=args.num_gen,
        max_completion_length=DEFAULTS["max_completion_length"],
        temperature=DEFAULTS["temperature"],
        use_vllm=True,
        vllm_gpu_memory_utilization=0.75,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=ds,
    )

    logger.info("═══ V2 TRAINING STARTED ═══")
    logger.info("Steps: %s | LR: %s | LoRA r=%d",
                max_steps if max_steps > 0 else "auto",
                args.lr, args.lora_r)

    trainer.train(resume_from_checkpoint=args.resume)

    logger.info("═══ V2 TRAINING COMPLETE ═══")

    # ── Save & push ──────────────────────────────────────────────────────────
    _STATUS = "saving"

    if not args.no_merge:
        logger.info("Saving merged model (16-bit) to %s", args.output)
        model.save_pretrained_merged(args.output, tokenizer, save_method="merged_16bit")
    else:
        logger.info("Saving LoRA adapter to %s", args.output)
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

    # Also save raw LoRA adapter separately for merge script
    lora_output = args.output + "_lora"
    logger.info("Saving raw LoRA adapter to %s (for merge)", lora_output)
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)

    if args.push and args.hub_model:
        _STATUS = "pushing"
        logger.info("Pushing to HF Hub: %s", args.hub_model)
        try:
            model.push_to_hub_merged(
                args.hub_model, tokenizer,
                save_method="merged_16bit",
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info("✓ V2 model pushed to %s", args.hub_model)

            # Also push LoRA adapter for merge
            lora_hub = args.hub_model + "-lora"
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(lora_hub, exist_ok=True, token=os.environ.get("HF_TOKEN"))
            api.upload_folder(
                folder_path=lora_output,
                repo_id=lora_hub,
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info("✓ V2 LoRA adapter pushed to %s", lora_hub)
        except Exception as exc:
            logger.error("Push failed: %s", exc)

    _STATUS = "done"
    logger.info("V2 training pipeline complete — model ready for merge")

    # Hold for 5 minutes so logs are visible, then exit
    time.sleep(300)


if __name__ == "__main__":
    main()
