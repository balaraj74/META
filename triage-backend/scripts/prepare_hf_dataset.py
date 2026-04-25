#!/usr/bin/env python3
"""
prepare_hf_dataset.py — Build combined GRPO dataset from multiple sources.

Merges:
  1. Environment rollout prompts (local, from build_grpo_dataset.py)
  2. II-Medical-RL (medical reasoning with verifiable answers)
  3. hendrycks/ethics commonsense (ethical decision-making)
  4. medical-o1-reasoning-SFT (chain-of-thought medical reasoning)
  5. BAAI/AquilaMed-RL (medical RL pairs)
  6. drug-combo-extraction (drug interaction knowledge)

All external datasets are transformed into hospital triage agent prompts
that match the GRPO training format expected by our reward verifiers.

Usage:
    python scripts/prepare_hf_dataset.py --output data/grpo/combined_train.jsonl
    python scripts/prepare_hf_dataset.py --output data/grpo/combined_train.jsonl --max-per-source 500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Prompt template for converting medical QA → triage agent prompts
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_TYPES = [
    "er_triage", "icu_management", "pharmacy",
    "cmo_oversight", "hr_rostering", "it_systems",
]

CRISIS_TYPES = ["mass_casualty", "outbreak", "equipment_failure", "staff_shortage"]

ACTION_MAP = {
    "er_triage": ["TRIAGE_PATIENT", "ASSIGN_TREATMENT", "UPDATE_EHR"],
    "icu_management": ["TRANSFER_TO_ICU", "TRANSFER_TO_WARD", "ACTIVATE_OVERFLOW"],
    "pharmacy": ["ORDER_MEDICATION", "FLAG_POLICY_VIOLATION"],
    "cmo_oversight": ["OVERRIDE_DECISION", "ACTIVATE_OVERFLOW"],
    "hr_rostering": ["REQUEST_STAFF", "FLAG_POLICY_VIOLATION"],
    "it_systems": ["UPDATE_EHR", "FLAG_POLICY_VIOLATION", "VERIFY_INSURANCE"],
}

ROLE_DESC = {
    "er_triage": "ER Triage — patient assessment, severity classification, treatment assignment",
    "icu_management": "ICU Management — bed allocation, transfers, overflow protocols",
    "pharmacy": "Pharmacy — medication orders, drug interactions, formulary compliance",
    "cmo_oversight": "CMO Oversight — policy enforcement, escalation decisions, resource authorization",
    "hr_rostering": "HR Rostering — staff allocation, shift management, fatigue monitoring",
    "it_systems": "IT Systems — EHR integrity, backup protocols, policy compliance monitoring",
}


def build_triage_prompt(
    medical_context: str,
    agent_type: str,
    crisis_type: str,
    difficulty: float,
    step: int,
    rng: random.Random,
) -> str:
    """Convert medical knowledge into a triage agent prompt."""
    n_patients = rng.randint(3, 30)
    icu_occ = round(rng.uniform(0.2, 0.95), 2)
    critical = rng.randint(0, min(10, n_patients))
    violations_inj = rng.randint(0, 5)
    violations_caught = rng.randint(0, violations_inj)
    survival = round(1.0 - rng.uniform(0, 0.15) * difficulty, 3)

    # Build patient list
    patients = []
    for i in range(min(critical, 5)):
        pid = rng.randint(1, 99)
        patients.append(f"  P-{pid:03d}: CRITICAL — vitals declining, BP {rng.randint(60,90)}/{rng.randint(30,60)}, HR {rng.randint(100,150)}")

    patient_block = "\n".join(patients) if patients else "  (none)"
    actions = ACTION_MAP[agent_type]
    actions_str = ", ".join(actions)

    prompt = (
        f"You are the {agent_type.upper()} agent in a hospital crisis simulation.\n\n"
        f"CRISIS: {crisis_type.upper()}\n"
        f"STEP: {step}/20\n"
        f"ICU OCCUPANCY: {int(icu_occ * 100)}% ({int(icu_occ * 20)}/20 beds)\n"
        f"CRITICAL PATIENTS ({critical} total — top 5):\n{patient_block}\n"
        f"VIOLATIONS INJECTED: {violations_inj} | CAUGHT: {violations_caught}\n"
        f"SURVIVAL RATE: {survival * 100:.1f}%\n\n"
        f"MEDICAL CONTEXT:\n{medical_context[:500]}\n\n"
        f"Your role: {ROLE_DESC[agent_type]}\n\n"
        f"Decide the single most important action right now. Respond with ONLY valid JSON:\n"
        f'{{\n'
        f'  "action_type": "<one of: {actions_str}>",\n'
        f'  "target_id": <patient ID integer or 0 if not patient-specific>,\n'
        f'  "priority": <integer 1-10, where 1=highest>,\n'
        f'  "reasoning": "<1-2 sentences citing specific patient data or metrics>"\n'
        f'}}'
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders (each returns list of medical context strings)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ii_medical_rl(max_samples: int) -> list[dict]:
    """Intelligent-Internet/II-Medical-RL — medical QA with reasoning."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Intelligent-Internet/II-Medical-RL", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            context = f"Question: {row['question']}\nReasoning: {row.get('reasoning', '')[:300]}"
            records.append({"context": context, "source": "ii-medical-rl"})
        logger.info("Loaded %d from II-Medical-RL", len(records))
        return records
    except Exception as e:
        logger.warning("Failed to load II-Medical-RL: %s", e)
        return []


def load_ethics(max_samples: int) -> list[dict]:
    """hendrycks/ethics commonsense — ethical scenarios."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/ethics", "commonsense", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            label = "ethical" if row["label"] == 0 else "unethical"
            context = f"Ethical scenario ({label}): {row['input']}"
            records.append({"context": context, "source": "ethics-commonsense"})
        logger.info("Loaded %d from ethics/commonsense", len(records))
        return records
    except Exception as e:
        logger.warning("Failed to load ethics: %s", e)
        return []


def load_medical_o1(max_samples: int) -> list[dict]:
    """FreedomIntelligence/medical-o1-reasoning-SFT — chain-of-thought medical."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "FreedomIntelligence/medical-o1-reasoning-SFT", "en",
            split="train", streaming=True,
        )
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            # Extract the question from the messages
            messages = row.get("messages", [])
            question = ""
            for msg in messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")[:400]
                    break
            if not question:
                question = str(row.get("question", row.get("input", "")))[:400]
            context = f"Medical reasoning: {question}"
            records.append({"context": context, "source": "medical-o1"})
        logger.info("Loaded %d from medical-o1", len(records))
        return records
    except Exception as e:
        logger.warning("Failed to load medical-o1: %s", e)
        return []


def load_aquilamed_rl(max_samples: int) -> list[dict]:
    """BAAI/AquilaMed-RL — medical RL instruction pairs."""
    try:
        from datasets import load_dataset
        ds = load_dataset("BAAI/AquilaMed-RL", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            instruction = row.get("instruction", "")[:400]
            if not instruction:
                continue
            context = f"Medical instruction: {instruction}"
            records.append({"context": context, "source": "aquilamed-rl"})
        logger.info("Loaded %d from AquilaMed-RL", len(records))
        return records
    except Exception as e:
        logger.warning("Failed to load AquilaMed-RL: %s", e)
        return []


def load_local_env_prompts(path: str) -> list[dict]:
    """Load existing environment-generated prompts."""
    records = []
    p = Path(path)
    if not p.exists():
        logger.warning("Local prompts not found at %s", path)
        return records
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records.append(record)
    logger.info("Loaded %d local environment prompts from %s", len(records), path)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build combined GRPO dataset")
    parser.add_argument("--output", default="data/grpo/combined_train.jsonl")
    parser.add_argument("--max-per-source", type=int, default=400,
                        help="Max samples from each external dataset")
    parser.add_argument("--local-prompts", default="data/grpo/train_expanded.jsonl",
                        help="Path to local environment prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-external", action="store_true",
                        help="Only use local prompts (for testing)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    # 1. Local environment prompts (already in correct format)
    local = load_local_env_prompts(args.local_prompts)
    all_records.extend(local)

    if not args.skip_external:
        # 2-5. External medical datasets → convert to triage prompts
        external_loaders = [
            load_ii_medical_rl,
            load_ethics,
            load_medical_o1,
            load_aquilamed_rl,
        ]

        for loader in external_loaders:
            try:
                raw = loader(args.max_per_source)
                for item in raw:
                    agent = rng.choice(AGENT_TYPES)
                    crisis = rng.choice(CRISIS_TYPES)
                    difficulty = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
                    step = rng.randint(0, 19)

                    prompt = build_triage_prompt(
                        medical_context=item["context"],
                        agent_type=agent,
                        crisis_type=crisis,
                        difficulty=difficulty,
                        step=step,
                        rng=rng,
                    )

                    record = {
                        "prompt": prompt,
                        "crisis_type": crisis,
                        "difficulty": difficulty,
                        "step": step,
                        "episode": -1,
                        "agent_type": agent,
                        "source": item["source"],
                    }
                    all_records.append(record)
            except Exception as e:
                logger.error("Loader failed: %s", e)

    # Shuffle
    rng.shuffle(all_records)

    # Write
    with open(output, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # Summary
    from collections import Counter
    sources = Counter()
    for r in all_records:
        sources[r.get("source", "env-rollout")] += 1

    print(f"\n{'═' * 60}")
    print(f"  Combined GRPO Dataset Ready")
    print(f"  Total prompts: {len(all_records)}")
    print(f"  Output: {output}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")
    print(f"\n  Sources:")
    for src, count in sources.most_common():
        print(f"    {src:25s} {count:5d}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
