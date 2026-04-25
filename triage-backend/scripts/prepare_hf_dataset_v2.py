#!/usr/bin/env python3
"""
prepare_hf_dataset_v2.py — Build GRPO dataset from 34 HuggingFace sources.

Organized by priority tier and mapped to specific triage agent types.
Uses streaming to avoid OOM. Gracefully skips failed downloads.

Usage:
    python scripts/prepare_hf_dataset_v2.py --output data/grpo/combined_train.jsonl
    python scripts/prepare_hf_dataset_v2.py --output data/grpo/combined_train.jsonl --max-per-source 200
    python scripts/prepare_hf_dataset_v2.py --priorities 1,2,3 --max-per-source 500
"""
from __future__ import annotations
import argparse, json, logging, random, sys, traceback
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Agent + Crisis configuration
# ═════════════════════════════════════════════════════════════════════════════
AGENT_TYPES = ["er_triage","icu_management","pharmacy","cmo_oversight","hr_rostering","it_systems"]
CRISIS_TYPES = ["mass_casualty","outbreak","equipment_failure","staff_shortage"]
ACTION_MAP = {
    "er_triage": ["TRIAGE_PATIENT","ASSIGN_TREATMENT","UPDATE_EHR"],
    "icu_management": ["TRANSFER_TO_ICU","TRANSFER_TO_WARD","ACTIVATE_OVERFLOW"],
    "pharmacy": ["ORDER_MEDICATION","FLAG_POLICY_VIOLATION"],
    "cmo_oversight": ["OVERRIDE_DECISION","ACTIVATE_OVERFLOW"],
    "hr_rostering": ["REQUEST_STAFF","FLAG_POLICY_VIOLATION"],
    "it_systems": ["UPDATE_EHR","FLAG_POLICY_VIOLATION","VERIFY_INSURANCE"],
}
ROLE_DESC = {
    "er_triage": "ER Triage — patient assessment, severity classification, treatment assignment",
    "icu_management": "ICU Management — bed allocation, transfers, overflow protocols",
    "pharmacy": "Pharmacy — medication orders, drug interactions, formulary compliance",
    "cmo_oversight": "CMO Oversight — policy enforcement, escalation decisions, resource authorization",
    "hr_rostering": "HR Rostering — staff allocation, shift management, fatigue monitoring",
    "it_systems": "IT Systems — EHR integrity, backup protocols, policy compliance monitoring",
}

def build_triage_prompt(medical_context: str, agent_type: str, crisis_type: str,
                        difficulty: float, step: int, rng: random.Random) -> str:
    n_patients = rng.randint(3, 30)
    icu_occ = round(rng.uniform(0.2, 0.95), 2)
    critical = rng.randint(0, min(10, n_patients))
    viol_inj = rng.randint(0, 5)
    viol_caught = rng.randint(0, viol_inj)
    survival = round(1.0 - rng.uniform(0, 0.15) * difficulty, 3)
    patients = []
    for i in range(min(critical, 5)):
        pid = rng.randint(1, 99)
        patients.append(f"  P-{pid:03d}: CRITICAL — vitals declining, BP {rng.randint(60,90)}/{rng.randint(30,60)}, HR {rng.randint(100,150)}")
    patient_block = "\n".join(patients) if patients else "  (none)"
    actions_str = ", ".join(ACTION_MAP[agent_type])
    return (
        f"You are the {agent_type.upper()} agent in a hospital crisis simulation.\n\n"
        f"CRISIS: {crisis_type.upper()}\nSTEP: {step}/20\n"
        f"ICU OCCUPANCY: {int(icu_occ*100)}% ({int(icu_occ*20)}/20 beds)\n"
        f"CRITICAL PATIENTS ({critical} total — top 5):\n{patient_block}\n"
        f"VIOLATIONS INJECTED: {viol_inj} | CAUGHT: {viol_caught}\n"
        f"SURVIVAL RATE: {survival*100:.1f}%\n\n"
        f"MEDICAL CONTEXT:\n{medical_context[:500]}\n\n"
        f"Your role: {ROLE_DESC[agent_type]}\n\n"
        f"Decide the single most important action right now. Respond with ONLY valid JSON:\n"
        f'{{\n  "action_type": "<one of: {actions_str}>",\n'
        f'  "target_id": <patient ID integer or 0 if not patient-specific>,\n'
        f'  "priority": <integer 1-10, where 1=highest>,\n'
        f'  "reasoning": "<1-2 sentences citing specific patient data or metrics>"\n}}'
    )

# ═════════════════════════════════════════════════════════════════════════════
# Dataset Registry — all 34 sources
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class DatasetSpec:
    name: str           # HF repo ID
    priority: int       # 1-8
    agents: list[str]   # target agent types or ["all"]
    config: str = ""    # HF config/subset name
    split: str = "train"
    text_fields: list[str] = field(default_factory=lambda: ["question","input","instruction","text"])
    streaming: bool = True

REGISTRY: list[DatasetSpec] = [
    # P1 — GRPO-ready
    DatasetSpec("TachyHealth/medical_grpo", 1, ["all"], text_fields=["question","input","prompt"]),
    DatasetSpec("Intelligent-Internet/II-Medical-RL", 1, ["all"], text_fields=["question","prompt"]),
    # P2 — DPO-ready
    DatasetSpec("BAAI/AquilaMed-RL", 2, ["all"], text_fields=["instruction","input","question"]),
    DatasetSpec("empirischtech/med-qa-orpo-dpo", 2, ["all"], text_fields=["question","prompt","instruction"]),
    # P3 — ER Triage
    DatasetSpec("openlifescienceai/medmcqa", 3, ["er_triage"], text_fields=["question"]),
    DatasetSpec("lavita/ChatDoctor-HealthCareMagic-100k", 3, ["er_triage"], text_fields=["input","instruction"]),
    DatasetSpec("curaihealth/medical_questions_pairs", 3, ["er_triage"], text_fields=["question_1","question_2"]),
    DatasetSpec("CNTXTAI0/CNTXTAI_Medical_Case_Studies", 3, ["er_triage"], text_fields=["text","case","input"]),
    DatasetSpec("Ahmad0067/MedSynth", 3, ["er_triage"], text_fields=["input","text","dialogue"]),
    # P4 — CMO Oversight
    DatasetSpec("FreedomIntelligence/medical-o1-reasoning-SFT", 4, ["cmo_oversight"], config="en", text_fields=["question","input"]),
    DatasetSpec("sdiazlor/medical-reasoning-dataset", 4, ["cmo_oversight"], text_fields=["question","input"]),
    DatasetSpec("medalpaca/medical_meadow_medical_flashcards", 4, ["cmo_oversight"], text_fields=["input","instruction"]),
    DatasetSpec("lavita/medical-qa-shared-task-v1-toy", 4, ["cmo_oversight"], text_fields=["question","text"]),
    # P5 — ICU Management
    DatasetSpec("AGBonnet/augmented-clinical-notes", 5, ["icu_management"], text_fields=["text","note","input"]),
    DatasetSpec("medalpaca/medical_meadow_wikidoc_patient_information", 5, ["icu_management"], text_fields=["input","instruction"]),
    # P6 — Pharmacy + Safety
    DatasetSpec("allenai/drug-combo-extraction", 6, ["pharmacy"], text_fields=["text","sentence","input"]),
    DatasetSpec("medalpaca/medical_meadow_mmmlu", 6, ["pharmacy"], text_fields=["input","instruction"]),
    # P7 — Ethics
    DatasetSpec("hendrycks/ethics", 7, ["cmo_oversight","it_systems"], config="utilitarianism", text_fields=["input","text"]),
    DatasetSpec("hendrycks/ethics", 7, ["cmo_oversight","it_systems"], config="deontology", text_fields=["input","scenario"]),
    DatasetSpec("hendrycks/ethics", 7, ["cmo_oversight","it_systems"], config="virtue", text_fields=["input","scenario"]),
    DatasetSpec("Anthropic/hh-rlhf", 7, ["cmo_oversight"], text_fields=["chosen","rejected"]),
    DatasetSpec("PKU-Alignment/PKU-SafeRLHF", 7, ["cmo_oversight"], text_fields=["prompt","response_0"]),
    DatasetSpec("allenai/prosocial-dialog", 7, ["cmo_oversight","hr_rostering"], text_fields=["context","response"]),
    # P8 — General SFT
    DatasetSpec("Med-dataset/Med_Dataset", 8, ["all"], text_fields=["question","input","instruction"]),
    DatasetSpec("mlx-community/medfit-dataset", 8, ["all"], text_fields=["instruction","input","question"]),
    DatasetSpec("bigbio/med_qa", 8, ["all"], config="med_qa_en_bigbio_qa", text_fields=["question_id","question"]),
    DatasetSpec("lavita/medical-qa-instructions", 8, ["all"], text_fields=["instruction","input"]),
    DatasetSpec("gretelai/symptom_to_diagnosis", 8, ["er_triage","hr_rostering"], text_fields=["input_text","text"]),
    DatasetSpec("medalpaca/medical_meadow_wikidoc", 8, ["all"], text_fields=["input","instruction"]),
    DatasetSpec("lavita/ChatDoctor-iCliniq", 8, ["all"], text_fields=["input","instruction"]),
]

# ═════════════════════════════════════════════════════════════════════════════
# Extraction helpers
# ═════════════════════════════════════════════════════════════════════════════
def extract_text(row: dict, text_fields: list[str], max_len: int = 500) -> str:
    """Extract first non-empty text field from a row."""
    # Handle 'messages' format (chat datasets like medical-o1)
    if "messages" in row and isinstance(row["messages"], list):
        for msg in row["messages"]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = str(msg.get("content", ""))[:max_len]
                if len(content) > 10:
                    return content
    for field in text_fields:
        val = row.get(field)
        if val and isinstance(val, str) and len(val.strip()) > 10:
            return val.strip()[:max_len]
    # Fallback: concatenate all string values
    parts = []
    for v in row.values():
        if isinstance(v, str) and len(v.strip()) > 10:
            parts.append(v.strip()[:200])
        if len(" ".join(parts)) > max_len:
            break
    return " ".join(parts)[:max_len] if parts else ""

def load_hf_dataset(spec: DatasetSpec, max_samples: int) -> list[dict]:
    """Load samples from a single HF dataset spec."""
    from datasets import load_dataset
    kwargs = {"split": spec.split, "streaming": spec.streaming, "trust_remote_code": True}
    if spec.config:
        kwargs["name"] = spec.config
    ds = load_dataset(spec.name, **kwargs)
    records = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        text = extract_text(dict(row), spec.text_fields)
        if len(text) < 15:
            continue
        records.append({"context": text, "source": spec.name, "agents": spec.agents})
    return records

# ═════════════════════════════════════════════════════════════════════════════
# Local prompt loader
# ═════════════════════════════════════════════════════════════════════════════
def load_local_prompts(path: str) -> list[dict]:
    records = []
    p = Path(path)
    if not p.exists():
        logger.warning("Local prompts not found: %s", path)
        return records
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d local prompts from %s", len(records), path)
    return records

# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Build combined GRPO dataset v2 (34 sources)")
    parser.add_argument("--output", default="data/grpo/combined_train.jsonl")
    parser.add_argument("--max-per-source", type=int, default=300)
    parser.add_argument("--local-prompts", default="data/grpo/train_expanded.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--priorities", default="1,2,3,4,5,6,7,8", help="Comma-separated priority tiers to include")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    allowed_priorities = set(int(x) for x in args.priorities.split(","))

    all_records: list[dict] = []
    sources = Counter()

    # 1. Local environment prompts
    if not args.skip_local:
        local = load_local_prompts(args.local_prompts)
        all_records.extend(local)
        sources["env-rollout"] = len(local)

    # 2. External HF datasets
    if not args.skip_external:
        filtered = [s for s in REGISTRY if s.priority in allowed_priorities]
        logger.info("Loading %d datasets (priorities: %s)", len(filtered), args.priorities)

        for idx, spec in enumerate(filtered):
            tag = f"[{idx+1}/{len(filtered)}] P{spec.priority}"
            try:
                logger.info("%s Loading %s ...", tag, spec.name)
                raw = load_hf_dataset(spec, args.max_per_source)
                logger.info("%s Got %d samples from %s", tag, len(raw), spec.name)

                for item in raw:
                    agents = item["agents"]
                    if agents == ["all"]:
                        agent = rng.choice(AGENT_TYPES)
                    else:
                        agent = rng.choice(agents)
                    crisis = rng.choice(CRISIS_TYPES)
                    difficulty = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
                    step = rng.randint(0, 19)
                    prompt = build_triage_prompt(item["context"], agent, crisis, difficulty, step, rng)
                    all_records.append({
                        "prompt": prompt,
                        "crisis_type": crisis,
                        "difficulty": difficulty,
                        "step": step,
                        "episode": -1,
                        "agent_type": agent,
                        "source": item["source"],
                    })
                sources[spec.name] += len(raw)
            except Exception as exc:
                logger.warning("%s FAILED %s: %s", tag, spec.name, exc)
                traceback.print_exc()

    # Shuffle and write
    rng.shuffle(all_records)
    with open(output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Summary
    print(f"\n{'═'*65}")
    print(f"  Combined GRPO Dataset v2 Ready")
    print(f"  Total prompts: {len(all_records)}")
    print(f"  Output: {output}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")
    print(f"\n  Sources ({len(sources)} total):")
    for src, count in sources.most_common():
        print(f"    P{'?':1s} {src:50s} {count:5d}")
    print(f"{'═'*65}\n")

if __name__ == "__main__":
    main()
