#!/usr/bin/env python3
"""
TRIAGE GRPO Training — Kaggle Notebook (Self-Contained)
=======================================================
Run on Kaggle with GPU T4/P100. No external imports from triage package.

Setup cell (run first):
    !pip install -q unsloth trl>=0.12 datasets peft accelerate bitsandbytes

Usage:
    1. Upload this as a Kaggle notebook
    2. Enable GPU accelerator (T4 x2 recommended)
    3. Run all cells
    4. Download the merged model from /kaggle/working/grpo_merged/
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 1: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
# !pip install -q unsloth "trl>=0.12" datasets peft accelerate bitsandbytes

import json, re, random, logging, time, os
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("triage_grpo")

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 2: Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Model — Qwen3.5-2B for T4/P100, or 0.8B if low VRAM
MODEL_NAME = "Qwen/Qwen3-2B"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# GRPO
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 128
TEMPERATURE = 0.9
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 1
GRADIENT_ACCUM = 8
LEARNING_RATE = 5e-5

# Dataset
NUM_PROMPTS = 2000  # increase for better training
OUTPUT_DIR = "./grpo_output"
MERGED_DIR = "./grpo_merged"

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 3: Reward Verifiers (self-contained — no triage imports)
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_ACTIONS = frozenset({
    "TRIAGE_PATIENT","ASSIGN_TREATMENT","TRANSFER_TO_ICU","TRANSFER_TO_WARD",
    "ACTIVATE_OVERFLOW","ORDER_MEDICATION","FLAG_POLICY_VIOLATION",
    "OVERRIDE_DECISION","UPDATE_EHR","REQUEST_STAFF","VERIFY_INSURANCE",
})
_REQUIRED_KEYS = {"action_type","target_id","priority","reasoning"}

_EVIDENCE_PATTERNS = [
    r"P-\d{2,3}", r"patient\s+\d+", r"\d+%", r"\d+/\d+",
    r"BP\s*\d+", r"HR\s*\d+", r"ICU\s+at\s+\d+", r"beds?\s+\d+",
    r"age\s+\d+", r"critical|immediate|urgent|stable",
]

_FORBIDDEN_PATTERNS = [
    (re.compile(p, re.IGNORECASE), n) for p, n in [
        (r"import\s+(os|sys|subprocess)", "system_import"),
        (r"exec\s*\(", "exec_call"), (r"eval\s*\(", "eval_call"),
        (r"open\s*\(", "file_open"), (r"reward\s*=|set_reward", "reward_hack"),
    ]
]

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def validate_action(text: str) -> tuple[bool, str]:
    if len(text) > 2000: return False, "too_long"
    for p, n in _FORBIDDEN_PATTERNS:
        if p.search(text): return False, f"forbidden:{n}"
    return True, "ok"

def reward_patient_survival(state, completion, **kw):
    alive = state.get("alive_count", 0)
    total = alive + state.get("deceased_count", 0)
    return alive / total if total > 0 else 1.0

def reward_icu_efficiency(state, completion, **kw):
    occ = state.get("icu_occupancy", 0.5)
    if occ <= 0.85: return 1.0
    elif occ <= 0.95: return 1.0 - (occ - 0.85) * 5.0
    else: return max(0.0, 0.5 - (occ - 0.95) * 10.0)

def reward_violation_detection(state, completion, **kw):
    inj = state.get("violations_injected", 0)
    return min(1.0, state.get("violations_caught", 0) / max(inj, 1)) if inj > 0 else 1.0

def reward_format_compliance(state, completion, **kw):
    parsed = _extract_json(completion)
    if not parsed or not _REQUIRED_KEYS.issubset(parsed.keys()): return 0.0
    if str(parsed.get("action_type","")).upper() not in _VALID_ACTIONS: return 0.0
    try: int(parsed["target_id"])
    except: return 0.0
    try:
        p = int(parsed["priority"])
        if not 1 <= p <= 10: return 0.0
    except: return 0.0
    if len(str(parsed.get("reasoning",""))) < 10: return 0.0
    return 1.0

def reward_reasoning_quality(state, completion, **kw):
    parsed = _extract_json(completion)
    if not parsed: return 0.0
    reasoning = str(parsed.get("reasoning",""))
    if len(reasoning) < 20: return 0.1
    evidence = sum(1 for p in _EVIDENCE_PATTERNS if re.search(p, reasoning, re.IGNORECASE))
    for f in ["i need more information","i'm not sure","let me think","i cannot determine"]:
        if f in reasoning.lower(): return 0.1
    return min(1.0, 0.3 + min(0.7, evidence * 0.15))

def reward_response_speed(state, completion, **kw):
    l = len(completion)
    if l <= 400: return 1.0
    elif l <= 800: return 1.0 - (l - 400) * 0.001
    else: return max(0.2, 0.6 - (l - 800) * 0.0005)

def reward_no_hallucination(state, completion, **kw):
    parsed = _extract_json(completion)
    if not parsed: return 0.5
    mentioned = {int(m.group(1)) for m in re.finditer(r"P-(\d{2,3})", str(parsed.get("reasoning","")))}
    if not mentioned: return 1.0
    valid = {p.get("id",-1) for p in state.get("patients_summary",[])}
    return 0.0 if mentioned - valid else 1.0

def reward_action_alignment(state, completion, **kw):
    parsed = _extract_json(completion)
    if not parsed: return 0.0
    action = str(parsed.get("action_type","")).upper()
    occ = state.get("icu_occupancy", 0.5)
    crit = state.get("critical_count", 0)
    crisis = state.get("crisis_type","")
    lookup = {
        "TRIAGE_PATIENT": 1.0 if crit > 0 else 0.5,
        "TRANSFER_TO_ICU": 1.0 if occ < 0.9 and crit > 0 else 0.3,
        "ACTIVATE_OVERFLOW": 1.0 if occ >= 0.85 else 0.2,
        "TRANSFER_TO_WARD": 1.0 if occ >= 0.7 else 0.4,
        "FLAG_POLICY_VIOLATION": 1.0 if (state.get("violations_injected",0)-state.get("violations_caught",0)) > 0 else 0.4,
        "REQUEST_STAFF": 1.0 if crisis == "staff_shortage" or crit >= 5 else 0.4,
        "ORDER_MEDICATION": 0.8 if crit > 0 else 0.5,
        "ASSIGN_TREATMENT": 0.9 if crit > 0 else 0.5,
    }
    return lookup.get(action, 0.5)

VERIFIERS = [
    reward_patient_survival, reward_icu_efficiency, reward_violation_detection,
    reward_format_compliance, reward_reasoning_quality, reward_response_speed,
    reward_no_hallucination, reward_action_alignment,
]

def compute_all_rewards(state, completion):
    results = {}
    for v in VERIFIERS:
        name = v.__name__.replace("reward_","")
        try: results[name] = round(float(v(state, completion)), 4)
        except: results[name] = 0.0
    results["total"] = round(sum(results.values()) / len(results), 4)
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 4: Synthetic dataset generator (replaces openenv rollouts)
# ═══════════════════════════════════════════════════════════════════════════════

CRISIS_TYPES = ["mass_casualty","outbreak","equipment_failure","staff_shortage"]
AGENT_TYPES = ["er_triage","icu_management","pharmacy","cmo_oversight","hr_rostering","it_systems"]

AGENT_ROLES = {
    "cmo_oversight": "Chief Medical Officer — escalation authority, hospital-wide crisis governance",
    "er_triage": "Emergency Room Triage — patient severity classification, START protocol",
    "icu_management": "ICU Management — bed allocation, ventilator management, overflow protocols",
    "pharmacy": "Pharmacy — medication dispensing, drug interaction verification",
    "hr_rostering": "HR Rostering — emergency staff scheduling, fatigue monitoring",
    "it_systems": "IT Systems — EHR integrity, backup protocols, policy compliance",
}

AGENT_ACTIONS = {
    "er_triage": "TRIAGE_PATIENT, UPDATE_EHR, ASSIGN_TREATMENT",
    "icu_management": "TRANSFER_TO_ICU, TRANSFER_TO_WARD, ACTIVATE_OVERFLOW",
    "pharmacy": "ORDER_MEDICATION, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE",
    "hr_rostering": "REQUEST_STAFF, FLAG_POLICY_VIOLATION",
    "cmo_oversight": "OVERRIDE_DECISION, ACTIVATE_OVERFLOW, ASSIGN_TREATMENT",
    "it_systems": "UPDATE_EHR, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE",
}

NAMES = ["John Smith","Maria Garcia","Ahmed Hassan","Li Wei","Sarah Johnson",
         "Raj Patel","Emma Wilson","Carlos Ruiz","Yuki Tanaka","Fatima Ali",
         "James Brown","Ana Lopez","David Kim","Olga Ivanova","Moses Obi"]
CONDITIONS = ["Cardiac Arrest","Sepsis","Pneumothorax","Hemorrhage","Burns",
              "Fracture","Respiratory Failure","Stroke","Anaphylaxis","Trauma"]

def generate_synthetic_prompt(rng: random.Random) -> dict:
    """Generate one training prompt with realistic hospital state."""
    crisis = rng.choice(CRISIS_TYPES)
    agent = rng.choice(AGENT_TYPES)
    difficulty = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
    step = rng.randint(1, 50)
    max_steps = 50

    # Generate patient data
    n_patients = rng.randint(8, 30)
    icu_total = rng.randint(20, 60)
    icu_occ_pct = rng.randint(40, 100) if crisis == "mass_casualty" else rng.randint(30, 90)
    icu_occupied = min(icu_total, int(icu_total * icu_occ_pct / 100))

    patients = []
    critical_patients = []
    for i in range(n_patients):
        status = rng.choices(["CRITICAL","SERIOUS","STABLE","DISCHARGED"],
                             weights=[0.3,0.3,0.3,0.1])[0]
        p = {
            "id": rng.randint(1, 99),
            "name": rng.choice(NAMES),
            "age": rng.randint(18, 90),
            "status": status,
            "condition": rng.choice(CONDITIONS),
            "triage_score": rng.randint(1, 5),
            "deterioration_rate": round(rng.uniform(0.0, 0.5), 2),
        }
        patients.append(p)
        if status == "CRITICAL":
            critical_patients.append(p)

    alive = sum(1 for p in patients if p["status"] != "DISCHARGED")
    deceased = rng.randint(0, 3)
    total = alive + deceased
    survival_rate = alive / total if total > 0 else 1.0
    violations_inj = rng.randint(0, 5) if crisis in ["outbreak","equipment_failure"] else rng.randint(0,2)
    violations_caught = rng.randint(0, violations_inj)

    # Build patient lines
    top5 = critical_patients[:5]
    plines = []
    for p in top5:
        plines.append(
            f"  - P-{p['id']:02d}: {p['name']}, age {p['age']}, "
            f"status={p['status']}, condition={p['condition']}, "
            f"triage={p['triage_score']}, deterioration={p['deterioration_rate']:.2f}"
        )
    pstr = "\n".join(plines) if plines else "  (none)"

    prompt = f"""You are the {agent.upper()} agent in a hospital crisis simulation.

CRISIS: {crisis.upper()}
STEP: {step}/{max_steps}
ICU OCCUPANCY: {icu_occ_pct}% ({icu_occupied}/{icu_total} beds)
CRITICAL PATIENTS ({len(critical_patients)} total — top 5):
{pstr}
VIOLATIONS INJECTED: {violations_inj} | CAUGHT: {violations_caught}
SURVIVAL RATE: {survival_rate:.1%}

Your role: {AGENT_ROLES.get(agent, "Hospital staff")}

Decide the single most important action right now. Respond with ONLY valid JSON:
{{
  "action_type": "<one of: {AGENT_ACTIONS.get(agent, 'TRIAGE_PATIENT')}>",
  "target_id": <patient ID integer or 0 if not patient-specific>,
  "priority": <integer 1-10, where 1=highest>,
  "reasoning": "<1-2 sentences citing specific patient data or metrics>"
}}"""

    state = {
        "alive_count": alive,
        "deceased_count": deceased,
        "critical_count": len(critical_patients),
        "icu_occupancy": icu_occ_pct / 100.0,
        "violations_injected": violations_inj,
        "violations_caught": violations_caught,
        "survival_rate": survival_rate,
        "crisis_type": crisis,
        "patients_summary": [{"id":p["id"],"status":p["status"]} for p in patients[:20]],
    }

    return {"prompt": prompt, "crisis_type": crisis, "agent_type": agent,
            "difficulty": difficulty, "step": step, "state": state}


def build_dataset(n_prompts: int = 2000, seed: int = 42) -> list[dict]:
    """Generate synthetic prompt dataset."""
    rng = random.Random(seed)
    dataset = []
    for _ in range(n_prompts):
        dataset.append(generate_synthetic_prompt(rng))
    logger.info("Generated %d prompts across %s crisis types", len(dataset), CRISIS_TYPES)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 5: Reward function wrapper for GRPOTrainer
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_state_from_prompt(prompt: str) -> dict:
    state = {"alive_count":20,"deceased_count":0,"critical_count":0,
             "icu_occupancy":0.5,"violations_injected":0,"violations_caught":0,
             "survival_rate":1.0,"crisis_type":"mass_casualty","patients_summary":[]}
    m = re.search(r"ICU OCCUPANCY:\s*(\d+)%", prompt)
    if m: state["icu_occupancy"] = int(m.group(1)) / 100.0
    m = re.search(r"CRITICAL PATIENTS\s*\((\d+)", prompt)
    if m: state["critical_count"] = int(m.group(1))
    m = re.search(r"VIOLATIONS INJECTED:\s*(\d+)\s*\|\s*CAUGHT:\s*(\d+)", prompt)
    if m: state["violations_injected"],state["violations_caught"] = int(m.group(1)),int(m.group(2))
    m = re.search(r"SURVIVAL RATE:\s*(\d+\.?\d*)%", prompt)
    if m: state["survival_rate"] = float(m.group(1)) / 100.0
    m = re.search(r"CRISIS:\s*(\w+)", prompt)
    if m: state["crisis_type"] = m.group(1).lower()
    pids = [{"id":int(x.group(1)),"status":"CRITICAL"} for x in re.finditer(r"P-(\d{2,3})", prompt)]
    state["patients_summary"] = pids
    total = 20
    state["alive_count"] = int(state["survival_rate"] * total)
    state["deceased_count"] = total - state["alive_count"]
    return state


def build_reward_function():
    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        prompts = kwargs.get("prompts", kwargs.get("prompt", [""]))
        for i, completion in enumerate(completions):
            safe, _ = validate_action(completion)
            if not safe:
                rewards.append(0.0)
                continue
            prompt = prompts[i] if i < len(prompts) else ""
            state = _extract_state_from_prompt(prompt)
            scores = compute_all_rewards(state, completion)
            rewards.append(scores.get("total", 0.0))
        return rewards
    return reward_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 6: Training
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    start = time.time()

    # 1. Build dataset
    logger.info("Building synthetic dataset (%d prompts)...", NUM_PROMPTS)
    raw = build_dataset(NUM_PROMPTS)
    prompts = [r["prompt"] for r in raw]

    # Save dataset
    Path("./data").mkdir(exist_ok=True)
    with open("./data/train.jsonl","w") as f:
        for r in raw: f.write(json.dumps(r)+"\n")
    logger.info("Dataset saved to ./data/train.jsonl")

    # 2. Load model
    logger.info("Loading %s (4bit=%s)...", MODEL_NAME, LOAD_IN_4BIT)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT, dtype=None,
    )

    # 3. Apply LoRA
    logger.info("Applying LoRA (r=%d)...", LORA_R)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    # 4. Build HF dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    # 5. GRPO config
    from trl import GRPOTrainer, GRPOConfig
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        learning_rate=LEARNING_RATE,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS, temperature=TEMPERATURE,
        logging_steps=5, save_steps=100, save_total_limit=3,
        report_to="none", bf16=True, seed=42, log_level="info",
    )

    reward_fn = build_reward_function()
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=reward_fn, args=training_args, train_dataset=dataset,
    )

    # 6. Train
    logger.info("Starting GRPO training...")
    result = trainer.train()
    logger.info("Training done. Loss=%.4f Steps=%d", result.training_loss, result.global_step)

    # 7. Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 8. Merge
    try:
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        logger.info("Merged model saved to %s", MERGED_DIR)
    except Exception as e:
        logger.warning("Merge failed: %s — LoRA adapters saved separately.", e)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  GRPO Training Complete")
    print(f"  Time:    {elapsed/60:.1f} min")
    print(f"  Steps:   {result.global_step}")
    print(f"  Loss:    {result.training_loss:.4f}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Merged:  {MERGED_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
