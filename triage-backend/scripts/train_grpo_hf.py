#!/usr/bin/env python3
"""
train_grpo_hf.py — Self-contained GRPO training for HuggingFace compute.

Trains Qwen3.6-27B (or configurable) on hospital triage agent tasks using
GRPO with 8 independent reward verifiers. Designed to run on HF A100 80GB.

This script is FULLY SELF-CONTAINED — all reward verifiers are inlined
so it runs in HF Jobs without needing the triage package.

Budget: $35 @ $2.50/hr = 14 hours on 1×A100

Usage (local dry-run):
    python scripts/train_grpo_hf.py --quick --model unsloth/Qwen3-0.6B-bnb-4bit

Usage (HF A100):
    python scripts/train_grpo_hf.py \\
        --model unsloth/Qwen3.6-27B-bnb-4bit \\
        --dataset balarajr/triage-grpo \\
        --hub-model balarajr/triage-agent-27b \\
        --push
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_grpo_hf")


# ═══════════════════════════════════════════════════════════════════════════════
# HF SPACES HEALTH-CHECK SERVER (port 7860)
# HF Docker Spaces require an HTTP endpoint to transition from APP_STARTING
# ═══════════════════════════════════════════════════════════════════════════════

_STATUS = {"phase": "initializing", "step": 0, "loss": None}


def _start_health_server(port: int = 7860) -> None:
    """Start a minimal HTTP server for HF Spaces health checks."""
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {"status": "ok", "training": _STATUS}, indent=2
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # suppress access logs
            pass

    server = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Health-check server listening on :%d", port)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    # Model — Qwen3.5-27B with 4-bit quantization via load_in_4bit=True
    # Requires transformers>=5.2.0 (upgraded in Dockerfile)
    "model": "unsloth/Qwen3.5-27B",
    "max_seq_length": 512,

    # LoRA — reduced rank for 27B on 48GB
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "lora_targets": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # GRPO — minimal generation for 27B on L40S 48GB
    "num_generations": 2,
    "max_completion_length": 128,
    "temperature": 0.9,

    # Training — conservative to avoid OOM
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 1,
    "lr": 5e-6,
    "logging_steps": 1,
    "save_steps": 200,

    # Paths
    "output_dir": "./models/grpo_hf_output",
    "dataset": "./data/grpo/combined_train.jsonl",
}


# ═══════════════════════════════════════════════════════════════════════════════
# INLINED REWARD VERIFIERS (self-contained for HF Jobs)
# ═══════════════════════════════════════════════════════════════════════════════

# Valid action types
_VALID_ACTIONS = frozenset({
    "TRIAGE_PATIENT", "ASSIGN_TREATMENT", "TRANSFER_TO_ICU",
    "TRANSFER_TO_WARD", "ACTIVATE_OVERFLOW", "ORDER_MEDICATION",
    "FLAG_POLICY_VIOLATION", "OVERRIDE_DECISION", "UPDATE_EHR",
    "REQUEST_STAFF", "VERIFY_INSURANCE",
})
_REQUIRED_KEYS = {"action_type", "target_id", "priority", "reasoning"}

# Evidence patterns for reasoning quality
_EVIDENCE_PATTERNS = [
    r"P-\d{2,3}", r"patient\s+\d+", r"\d+%", r"\d+/\d+",
    r"BP\s*\d+", r"HR\s*\d+", r"ICU\s+at\s+\d+", r"beds?\s+\d+",
    r"age\s+\d+", r"critical|immediate|urgent|stable",
]

# Filler phrases that indicate low-quality reasoning
_FILLER_PHRASES = [
    "i need more information", "i'm not sure", "let me think",
    "i cannot determine", "i don't know", "more data needed",
]

# Forbidden patterns for sandbox safety
_FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bexec\s*\(", r"\beval\s*\(", r"\b__import__\b",
    r"\breward\s*[:=]\s*1\.0\b", r"\bscore\s*[:=]\s*1\.0\b",
]


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from completion text."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _extract_state_from_prompt(prompt: str) -> dict:
    """Parse state variables from the structured prompt text."""
    state = {
        "alive_count": 20, "deceased_count": 0, "critical_count": 0,
        "icu_occupancy": 0.5, "violations_injected": 0,
        "violations_caught": 0, "survival_rate": 1.0,
        "crisis_type": "mass_casualty", "patients_summary": [],
    }
    m = re.search(r"ICU OCCUPANCY:\s*(\d+)%", prompt)
    if m:
        state["icu_occupancy"] = int(m.group(1)) / 100.0
    m = re.search(r"CRITICAL PATIENTS\s*\((\d+)", prompt)
    if m:
        state["critical_count"] = int(m.group(1))
    m = re.search(r"VIOLATIONS INJECTED:\s*(\d+)\s*\|\s*CAUGHT:\s*(\d+)", prompt)
    if m:
        state["violations_injected"] = int(m.group(1))
        state["violations_caught"] = int(m.group(2))
    m = re.search(r"SURVIVAL RATE:\s*(\d+\.?\d*)%", prompt)
    if m:
        state["survival_rate"] = float(m.group(1)) / 100.0
    m = re.search(r"CRISIS:\s*(\w+)", prompt)
    if m:
        state["crisis_type"] = m.group(1).lower()
    pids = []
    for m in re.finditer(r"P-(\d{2,3})", prompt):
        pids.append({"id": int(m.group(1)), "status": "CRITICAL"})
    state["patients_summary"] = pids
    total = max(20, len(pids))
    state["alive_count"] = int(state["survival_rate"] * total)
    state["deceased_count"] = total - state["alive_count"]
    return state


def _sandbox_check(completion: str) -> bool:
    """Return False if completion contains forbidden patterns."""
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, completion, re.IGNORECASE):
            return False
    if len(completion) > 3000:
        return False
    # Check for excessive repetition (reward hacking)
    words = completion.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return False
    return True


# --- Individual reward functions (TRL GRPOTrainer compatible) ---
# Signature: fn(completions: list[str], **kwargs) -> list[float]

def reward_format_compliance(completions: list[str], **kwargs) -> list[float]:
    """Hard gate: valid JSON with required fields."""
    rewards = []
    for c in completions:
        parsed = _extract_json(c)
        if parsed is None:
            rewards.append(0.0); continue
        if not _REQUIRED_KEYS.issubset(parsed.keys()):
            rewards.append(0.0); continue
        action = str(parsed.get("action_type", "")).upper()
        if action not in _VALID_ACTIONS:
            rewards.append(0.0); continue
        try:
            int(parsed["target_id"])
        except (ValueError, TypeError):
            rewards.append(0.0); continue
        try:
            p = int(parsed["priority"])
            if not 1 <= p <= 10:
                rewards.append(0.0); continue
        except (ValueError, TypeError):
            rewards.append(0.0); continue
        reasoning = str(parsed.get("reasoning", ""))
        if len(reasoning.strip()) < 10:
            rewards.append(0.0); continue
        rewards.append(1.0)
    return rewards


def reward_patient_survival(completions: list[str], **kwargs) -> list[float]:
    """0-1: fraction of patients alive from prompt state."""
    prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
    rewards = []
    for i, c in enumerate(completions):
        prompt = prompts[i] if i < len(prompts) else ""
        state = _extract_state_from_prompt(prompt)
        alive = state.get("alive_count", 0)
        deceased = state.get("deceased_count", 0)
        total = alive + deceased
        rewards.append(alive / total if total > 0 else 1.0)
    return rewards


def reward_icu_efficiency(completions: list[str], **kwargs) -> list[float]:
    """0-1: penalise ICU over-capacity."""
    prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
    rewards = []
    for i, c in enumerate(completions):
        prompt = prompts[i] if i < len(prompts) else ""
        state = _extract_state_from_prompt(prompt)
        occ = state.get("icu_occupancy", 0.5)
        if occ <= 0.85:
            rewards.append(1.0)
        elif occ <= 0.95:
            rewards.append(1.0 - (occ - 0.85) * 5.0)
        else:
            rewards.append(max(0.0, 0.5 - (occ - 0.95) * 10.0))
    return rewards


def reward_violation_detection(completions: list[str], **kwargs) -> list[float]:
    """0-1: fraction of violations caught."""
    prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
    rewards = []
    for i, c in enumerate(completions):
        prompt = prompts[i] if i < len(prompts) else ""
        state = _extract_state_from_prompt(prompt)
        inj = state.get("violations_injected", 0)
        caught = state.get("violations_caught", 0)
        rewards.append(min(1.0, caught / max(inj, 1)) if inj > 0 else 1.0)
    return rewards


def reward_reasoning_quality(completions: list[str], **kwargs) -> list[float]:
    """0-1: reasoning cites specific data."""
    rewards = []
    for c in completions:
        parsed = _extract_json(c)
        if parsed is None:
            rewards.append(0.0); continue
        reasoning = str(parsed.get("reasoning", ""))
        if len(reasoning) < 20:
            rewards.append(0.1); continue
        ev_count = sum(1 for p in _EVIDENCE_PATTERNS if re.search(p, reasoning, re.IGNORECASE))
        if any(f in reasoning.lower() for f in _FILLER_PHRASES):
            rewards.append(0.1); continue
        rewards.append(min(1.0, 0.3 + min(0.7, ev_count * 0.15)))
    return rewards


def reward_response_speed(completions: list[str], **kwargs) -> list[float]:
    """0-1: penalise overly long completions."""
    rewards = []
    for c in completions:
        length = len(c)
        if length <= 400:
            rewards.append(1.0)
        elif length <= 800:
            rewards.append(1.0 - (length - 400) * 0.001)
        else:
            rewards.append(max(0.2, 0.6 - (length - 800) * 0.0005))
    return rewards


def reward_no_hallucination(completions: list[str], **kwargs) -> list[float]:
    """Hard gate: no invented patient IDs."""
    prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
    rewards = []
    for i, c in enumerate(completions):
        parsed = _extract_json(c)
        if parsed is None:
            rewards.append(0.5); continue
        reasoning = str(parsed.get("reasoning", ""))
        mentioned = {int(m.group(1)) for m in re.finditer(r"P-(\d{2,3})", reasoning, re.IGNORECASE)}
        if not mentioned:
            rewards.append(1.0); continue
        prompt = prompts[i] if i < len(prompts) else ""
        valid_ids = {int(m.group(1)) for m in re.finditer(r"P-(\d{2,3})", prompt)}
        rewards.append(0.0 if mentioned - valid_ids else 1.0)
    return rewards


def reward_action_alignment(completions: list[str], **kwargs) -> list[float]:
    """0-1: action makes sense for current state."""
    prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
    rewards = []
    for i, c in enumerate(completions):
        parsed = _extract_json(c)
        if parsed is None:
            rewards.append(0.0); continue
        prompt = prompts[i] if i < len(prompts) else ""
        state = _extract_state_from_prompt(prompt)
        action = str(parsed.get("action_type", "")).upper()
        occ = state.get("icu_occupancy", 0.5)
        crit = state.get("critical_count", 0)
        viol = state.get("violations_injected", 0) - state.get("violations_caught", 0)

        score_map = {
            "TRIAGE_PATIENT": 1.0 if crit > 0 else 0.5,
            "TRANSFER_TO_ICU": 1.0 if occ < 0.9 and crit > 0 else 0.3,
            "ACTIVATE_OVERFLOW": 1.0 if occ >= 0.85 else 0.2,
            "TRANSFER_TO_WARD": 1.0 if occ >= 0.7 else 0.4,
            "FLAG_POLICY_VIOLATION": 1.0 if viol > 0 else 0.4,
            "OVERRIDE_DECISION": 1.0 if occ >= 0.9 or crit >= 8 else 0.3,
            "ORDER_MEDICATION": 0.8 if crit > 0 else 0.5,
            "REQUEST_STAFF": 1.0 if state.get("crisis_type") == "staff_shortage" or crit >= 5 else 0.4,
            "UPDATE_EHR": 0.5,
            "VERIFY_INSURANCE": 0.5,
            "ASSIGN_TREATMENT": 0.9 if crit > 0 else 0.5,
        }
        rewards.append(score_map.get(action, 0.5))
    return rewards


def reward_sandbox_safety(completions: list[str], **kwargs) -> list[float]:
    """Hard gate: reject unsafe completions."""
    return [1.0 if _sandbox_check(c) else 0.0 for c in completions]


# The full suite — passed as list to GRPOTrainer's reward_funcs
REWARD_FUNCS = [
    reward_format_compliance,
    reward_patient_survival,
    reward_icu_efficiency,
    reward_violation_detection,
    reward_reasoning_quality,
    reward_response_speed,
    reward_no_hallucination,
    reward_action_alignment,
    reward_sandbox_safety,
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_prompts(path: str) -> list[str]:
    """Load prompts from JSONL or HF dataset."""
    if path.startswith("balarajr/") or "/" in path and not Path(path).exists():
        # HF Hub dataset
        logger.info("Loading dataset from HF Hub: %s", path)
        from datasets import load_dataset
        ds = load_dataset(path, split="train")
        return ds["prompt"]

    # Local JSONL
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
# CLOUD-SIDE HF AUGMENTATION (runs on A100 with fast network)
# ═══════════════════════════════════════════════════════════════════════════════

_HF_SOURCES = [
    ("TachyHealth/medical_grpo", None, ["question", "input", "prompt"]),
    ("Intelligent-Internet/II-Medical-RL", None, ["question", "prompt"]),
    ("BAAI/AquilaMed-RL", None, ["instruction", "input"]),
    ("openlifescienceai/medmcqa", None, ["question"]),
    ("sdiazlor/medical-reasoning-dataset", None, ["question", "input"]),
    ("lavita/ChatDoctor-HealthCareMagic-100k", None, ["input", "instruction"]),
    ("hendrycks/ethics", "utilitarianism", ["input", "text"]),
    ("hendrycks/ethics", "deontology", ["input", "scenario"]),
    ("Anthropic/hh-rlhf", None, ["chosen"]),
    ("allenai/prosocial-dialog", None, ["context", "response"]),
]

_AUG_AGENTS = ["er_triage", "icu_management", "pharmacy", "cmo_oversight", "hr_rostering", "it_systems"]
_AUG_CRISES = ["mass_casualty", "outbreak", "equipment_failure", "staff_shortage"]
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
    """Download HF datasets and convert to triage prompts."""
    import random as _rng
    rng = _rng.Random(123)
    prompts = []
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not available for augmentation")
        return prompts

    for repo, config, fields in _HF_SOURCES:
        try:
            kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}
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
    parser = argparse.ArgumentParser(description="GRPO training on HuggingFace")
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--output", default=DEFAULTS["output_dir"])
    parser.add_argument("--hub-model", default=None,
                        help="HF Hub model ID to push (e.g. balarajr/triage-agent-27b)")
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
                        help="Augment dataset with HF medical datasets (cloud only)")
    parser.add_argument("--augment-max", type=int, default=200,
                        help="Max samples per HF source for augmentation")
    args = parser.parse_args()

    start = time.time()

    # Start health-check server FIRST so HF Spaces marks us as RUNNING
    _start_health_server()
    _STATUS["phase"] = "loading_model"

    # ── Step 1: Load model ────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Run: pip install unsloth")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.resume or args.model,
        max_seq_length=DEFAULTS["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    # ── Step 2: Apply LoRA ────────────────────────────────────────────────────
    if args.resume is None:
        logger.info("Applying LoRA (r=%d, alpha=%d)", args.lora_r, DEFAULTS["lora_alpha"])
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=DEFAULTS["lora_targets"],
            lora_alpha=DEFAULTS["lora_alpha"],
            lora_dropout=DEFAULTS["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # ── Step 3: Load dataset ──────────────────────────────────────────────────
    prompts = list(load_dataset_prompts(args.dataset))  # Convert Column → list
    logger.info("Loaded %d base prompts", len(prompts))

    # Cloud-side augmentation from HF medical datasets
    if args.augment_hf:
        logger.info("Augmenting with HF medical datasets (max %d/source)...", args.augment_max)
        augmented = _augment_from_hf(args.augment_max)
        prompts.extend(augmented)
        logger.info("Total after augmentation: %d prompts", len(prompts))

    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    if args.quick:
        dataset = dataset.select(range(min(40, len(dataset))))
        logger.info("Quick mode: %d prompts", len(dataset))

    # ── Step 4: Configure GRPOTrainer ─────────────────────────────────────────
    from trl import GRPOTrainer, GRPOConfig

    max_steps = 10 if args.quick else -1

    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1 if args.quick else args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=DEFAULTS["max_completion_length"],
        max_prompt_length=DEFAULTS["max_seq_length"] - DEFAULTS["max_completion_length"],
        num_generations=args.num_gen,
        temperature=DEFAULTS["temperature"],
        logging_steps=DEFAULTS["logging_steps"],
        save_steps=DEFAULTS["save_steps"] if not args.quick else 999999,
        save_total_limit=3,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        bf16=True,       # A100 / Ampere supports bf16
        fp16=False,
        seed=42,
        log_level="info",
        hub_model_id=args.hub_model if args.push else None,
        push_to_hub=args.push,
    )

    logger.info(
        "GRPOTrainer config: G=%d, batch=%d×%d=%d, lr=%.1e, epochs=%d",
        args.num_gen, args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum, args.lr, args.epochs,
    )

    # Pass individual reward functions — NOT a single aggregate
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    _STATUS["phase"] = "training"
    logger.info("Starting GRPO training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume)
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # ── Step 6: Save adapters ─────────────────────────────────────────────────
    logger.info("Saving LoRA adapters to %s", args.output)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # ── Step 7: Merge + push ──────────────────────────────────────────────────
    if not args.no_merge:
        merged_dir = args.output + "_merged"
        logger.info("Merging LoRA → %s", merged_dir)
        try:
            model.save_pretrained_merged(
                merged_dir, tokenizer, save_method="merged_16bit",
            )
            logger.info("Merged model saved.")

            if args.push and args.hub_model:
                logger.info("Pushing merged model to %s", args.hub_model)
                model.push_to_hub_merged(
                    args.hub_model, tokenizer, save_method="merged_16bit",
                )
                logger.info("Pushed to HF Hub.")

                # Also push raw LoRA adapter for multi-account merge
                lora_hub = args.hub_model + "-lora"
                try:
                    from huggingface_hub import HfApi
                    hf_api = HfApi()
                    hf_api.create_repo(lora_hub, exist_ok=True,
                                       token=os.environ.get("HF_TOKEN"))
                    hf_api.upload_folder(
                        folder_path=args.output,
                        repo_id=lora_hub,
                        token=os.environ.get("HF_TOKEN"),
                    )
                    logger.info("LoRA adapter pushed to %s", lora_hub)
                except Exception as lora_exc:
                    logger.warning("LoRA push failed: %s", lora_exc)
        except Exception as exc:
            logger.warning("Merge/push failed: %s. LoRA adapters saved.", exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"  GRPO Training Complete")
    print(f"  Model:      {args.model}")
    print(f"  Time:       {elapsed / 60:.1f} min")
    print(f"  Steps:      {train_result.global_step}")
    print(f"  Loss:       {train_result.training_loss:.4f}")
    print(f"  Adapters:   {args.output}")
    if args.push and args.hub_model:
        print(f"  Hub:        https://huggingface.co/{args.hub_model}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
