#!/usr/bin/env python3
"""
TRIAGE GRPO Training — Kaggle (No Unsloth, P100/T4 compatible)
==============================================================
Uses: transformers + peft + trl + bitsandbytes
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 1: Install
# ═══════════════════════════════════════════════════════════════════════════════
# !pip install -q "transformers>=4.46" "trl>=0.12" "peft>=0.13" "bitsandbytes>=0.44" datasets accelerate

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 2: Config
# ═══════════════════════════════════════════════════════════════════════════════
import json, re, random, logging, time, os, torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('triage_grpo')

# Qwen3.5 public 4B repo is `Qwen/Qwen3.5-4B` (not `...-Instruct`).
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B")
MODEL_ALIASES = {
    "Qwen/Qwen3.5-4B-Instruct": "Qwen/Qwen3.5-4B",
}
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NUM_PROMPTS = 2000
NUM_EPOCHS = 3
BATCH = 2
GRAD_ACCUM = 4
LR = 5e-5
OUTPUT_DIR = "/kaggle/working/triage_grpo_output"

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 3: Reward Verifiers (all 8, self-contained)
# ═══════════════════════════════════════════════════════════════════════════════
VALID_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
VALID_ACTIONS = {
    "EVACUATE", "LOCKDOWN", "QUARANTINE", "DIVERT_AMBULANCE",
    "REQUEST_STAFF", "TRIAGE_OVERRIDE", "MONITOR", "SHELTER_IN_PLACE",
}
BLOCKED_PHRASES = [
    "as an ai", "i cannot", "i'm sorry", "i don't have",
    "hypothetically", "in theory", "i would suggest maybe",
]
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"system\s*prompt", r"you\s+are\s+now",
    r"pretend\s+to\s+be", r"jailbreak",
]

def verify_format(response: str, **_) -> float:
    txt = response.strip()
    has_severity = bool(re.search(r"SEVERITY:\s*(CRITICAL|HIGH|MEDIUM|LOW)", txt, re.I))
    has_action = bool(re.search(r"ACTION:\s*\w+", txt, re.I))
    has_reasoning = bool(re.search(r"REASONING:", txt, re.I))
    score = (0.4 * has_severity) + (0.4 * has_action) + (0.2 * has_reasoning)
    return round(score, 3)

def verify_reasoning(response: str, **_) -> float:
    m = re.search(r"REASONING:\s*(.+?)(?:ACTION:|$)", response, re.S | re.I)
    if not m:
        return 0.0
    reasoning = m.group(1).strip()
    words = reasoning.split()
    if len(words) < 10:
        return 0.1
    score = 0.3
    causal = ["because", "therefore", "since", "due to", "given that", "as a result"]
    if any(c in reasoning.lower() for c in causal):
        score += 0.3
    medical = ["patient", "vital", "triage", "icu", "bed", "staff", "oxygen",
               "critical", "mortality", "capacity", "ventilator", "blood"]
    med_count = sum(1 for t in medical if t in reasoning.lower())
    score += min(0.4, med_count * 0.1)
    return round(min(1.0, score), 3)

def verify_no_hallucination(response: str, **_) -> float:
    score = 1.0
    for phrase in BLOCKED_PHRASES:
        if phrase in response.lower():
            score -= 0.3
    for pat in INJECTION_PATTERNS:
        if re.search(pat, response, re.I):
            score -= 0.5
    return round(max(0.0, score), 3)

def verify_action_alignment(response: str, **kw) -> float:
    m = re.search(r"ACTION:\s*(\w+)", response, re.I)
    if not m:
        return 0.0
    action = m.group(1).upper()
    if action not in VALID_ACTIONS:
        return 0.0
    crisis = kw.get("crisis_type", "")
    good_map = {
        "mass_casualty": ["TRIAGE_OVERRIDE", "DIVERT_AMBULANCE", "REQUEST_STAFF"],
        "outbreak": ["QUARANTINE", "SHELTER_IN_PLACE", "MONITOR"],
        "equipment_failure": ["DIVERT_AMBULANCE", "REQUEST_STAFF", "MONITOR"],
        "staff_shortage": ["REQUEST_STAFF", "TRIAGE_OVERRIDE", "MONITOR"],
    }
    ideal = good_map.get(crisis, [])
    return 1.0 if action in ideal else 0.4

def verify_response_speed(response: str, **kw) -> float:
    words = len(response.split())
    if words > 300:
        return 0.2
    if words > 200:
        return 0.5
    return 1.0

def verify_patient_survival(response: str, **kw) -> float:
    state = kw.get("state", {})
    patients = state.get("patients", {})
    alive = patients.get("alive", 10)
    total = patients.get("total", 10)
    return round(alive / max(total, 1), 3) if total > 0 else 1.0

def verify_icu_efficiency(response: str, **kw) -> float:
    state = kw.get("state", {})
    icu = state.get("icu", {})
    used = icu.get("occupied", 5)
    cap = icu.get("capacity", 10)
    return round(1.0 - abs(used / max(cap, 1) - 0.8) / 0.8, 3) if cap > 0 else 0.5

def verify_violations(response: str, **kw) -> float:
    state = kw.get("state", {})
    v = state.get("violations", 0)
    if v == 0:
        return 1.0
    if v <= 2:
        return 0.5
    return 0.0

ALL_VERIFIERS = {
    "format_compliance": (verify_format, 0.20),
    "reasoning_quality": (verify_reasoning, 0.15),
    "no_hallucination": (verify_no_hallucination, 0.15),
    "action_alignment": (verify_action_alignment, 0.15),
    "response_speed": (verify_response_speed, 0.10),
    "patient_survival": (verify_patient_survival, 0.10),
    "icu_efficiency": (verify_icu_efficiency, 0.08),
    "violation_detection": (verify_violations, 0.07),
}

def compute_reward(response: str, **kw) -> float:
    total = 0.0
    for name, (fn, weight) in ALL_VERIFIERS.items():
        total += fn(response, **kw) * weight
    return round(total, 4)

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 4: Dataset Generator
# ═══════════════════════════════════════════════════════════════════════════════
CRISES = ['mass_casualty', 'outbreak', 'equipment_failure', 'staff_shortage']
ROLES = ['triage_officer', 'icu_manager', 'resource_allocator',
         'comms_officer', 'safety_monitor', 'logistics_coordinator']

def build_state(crisis):
    base = {
        "patients": {"alive": random.randint(5, 20), "total": random.randint(10, 25),
                      "critical": random.randint(1, 8), "waiting": random.randint(0, 15)},
        "icu": {"occupied": random.randint(3, 9), "capacity": 10,
                "ventilators_free": random.randint(0, 4)},
        "staff": {"doctors": random.randint(2, 8), "nurses": random.randint(4, 15),
                  "available": random.randint(1, 10)},
        "violations": random.randint(0, 3),
    }
    if crisis == 'mass_casualty':
        base["patients"]["critical"] = random.randint(5, 15)
        base["patients"]["waiting"] = random.randint(10, 30)
    elif crisis == 'outbreak':
        base["patients"]["total"] = random.randint(20, 50)
    elif crisis == 'equipment_failure':
        base["icu"]["ventilators_free"] = 0
    elif crisis == 'staff_shortage':
        base["staff"]["available"] = random.randint(0, 2)
    return base

SYSTEM_PROMPT = """You are a hospital triage AI. Respond EXACTLY in this format:
SEVERITY: <CRITICAL|HIGH|MEDIUM|LOW>
ACTION: <one of: EVACUATE, LOCKDOWN, QUARANTINE, DIVERT_AMBULANCE, REQUEST_STAFF, TRIAGE_OVERRIDE, MONITOR, SHELTER_IN_PLACE>
REASONING: <2-4 sentences explaining your decision with medical/operational justification>"""

def make_prompt(crisis, role, state):
    return (
        f"CRISIS: {crisis.upper().replace('_',' ')}\n"
        f"ROLE: {role}\n"
        f"PATIENTS: {state['patients']['critical']} critical, "
        f"{state['patients']['waiting']} waiting, "
        f"{state['patients']['alive']}/{state['patients']['total']} alive\n"
        f"ICU: {state['icu']['occupied']}/{state['icu']['capacity']} beds, "
        f"{state['icu']['ventilators_free']} ventilators free\n"
        f"STAFF: {state['staff']['available']} available "
        f"({state['staff']['doctors']}D/{state['staff']['nurses']}N)\n"
        f"VIOLATIONS: {state['violations']}\n\n"
        f"What is your triage decision?"
    )

def generate_dataset(n=NUM_PROMPTS):
    dataset_raw = []
    for i in range(n):
        crisis = random.choice(CRISES)
        role = random.choice(ROLES)
        state = build_state(crisis)
        user_msg = make_prompt(crisis, role, state)
        dataset_raw.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "crisis_type": crisis,
            "role": role,
            "state": state,
        })
    logger.info("Generated %d prompts", len(dataset_raw))
    return dataset_raw

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 5: Reward function for GRPOTrainer
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_crisis_from_prompt(prompt_text):
    for c in CRISES:
        if c.replace('_', ' ').upper() in prompt_text.upper():
            return c
    return "mass_casualty"

def reward_fn(completions, **kwargs):
    rewards = []
    prompts = kwargs.get("prompts", [None] * len(completions))
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        prompt_text = ""
        if prompts[i]:
            if isinstance(prompts[i], list):
                prompt_text = " ".join(m.get("content", "") for m in prompts[i])
            else:
                prompt_text = str(prompts[i])
        crisis = _extract_crisis_from_prompt(prompt_text)
        state = {
            "patients": {"alive": 15, "total": 20, "critical": 5, "waiting": 10},
            "icu": {"occupied": 7, "capacity": 10, "ventilators_free": 2},
            "staff": {"available": 3, "doctors": 4, "nurses": 8},
            "violations": 1,
        }
        r = compute_reward(text, crisis_type=crisis, state=state)
        rewards.append(r)
    return rewards

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 6: Load model with standard transformers + peft
# ═══════════════════════════════════════════════════════════════════════════════
def load_model():
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    def _resolve_model_name(name: str) -> str:
        model_name = name.strip()
        if model_name in MODEL_ALIASES:
            mapped = MODEL_ALIASES[model_name]
            logger.warning("Model '%s' not found as public repo. Using '%s'", model_name, mapped)
            return mapped
        return model_name

    def _resolve_token() -> str | None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        return token.strip() if token and token.strip() else None

    def _verify_access(model_name: str, token: str | None) -> str | bool:
        api = HfApi()
        if token:
            try:
                api.model_info(model_name, token=token)
                return token
            except Exception as exc:
                logger.warning(
                    "HF token auth failed (%s). Retrying anonymous access for public models.",
                    exc.__class__.__name__,
                )
        try:
            # Force anonymous access so stale/invalid local tokens do not cause 401 on public repos.
            api.model_info(model_name, token=False)
            return False
        except Exception as exc:
            raise RuntimeError(
                f"Cannot access model '{model_name}'. Use a valid public model id and set a valid "
                "Kaggle secret HF_TOKEN for gated/private repos."
            ) from exc

    model_name = _resolve_model_name(MODEL_NAME)
    hf_token: str | bool = _verify_access(model_name, _resolve_token())

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # P100 safe — no flash attention
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("Model loaded with LoRA.")
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 7: Train
# ═══════════════════════════════════════════════════════════════════════════════
def train():
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig

    model, tokenizer = load_model()
    dataset_raw = generate_dataset(NUM_PROMPTS)

    ds = Dataset.from_dict({"prompt": [r["prompt"] for r in dataset_raw]})

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_completion_length=MAX_SEQ_LENGTH,
        num_generations=4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=False,  # P100 doesn't support bf16
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=ds,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Training complete! Saved to %s", OUTPUT_DIR)
    return trainer

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 8: Quick eval
# ═══════════════════════════════════════════════════════════════════════════════
def quick_eval():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    test_prompt = make_prompt("mass_casualty", "triage_officer",
                              build_state("mass_casualty"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR, device_map="auto", torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("\n=== TRAINED MODEL OUTPUT ===")
    print(response)
    print(f"\nReward: {compute_reward(response, crisis_type='mass_casualty')}")

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 9: Package
# ═══════════════════════════════════════════════════════════════════════════════
def package():
    import shutil
    shutil.make_archive('/kaggle/working/triage_grpo_model', 'zip', OUTPUT_DIR)
    logger.info("Model packaged → /kaggle/working/triage_grpo_model.zip")

if __name__ == "__main__":
    train()
    quick_eval()
    package()
