
#!/usr/bin/env python3
"""
TRIAGE GRPO Training — Kaggle Full Dataset Edition
Qwen2.5-7B + 24 HF dataset sources + 9 reward verifiers
Compatible with Kaggle T4/P100 (no Unsloth needed)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════
# !pip install -q git+https://github.com/huggingface/transformers.git "trl>=0.12" "peft>=0.13" "bitsandbytes>=0.46.1" "datasets>=3.0" "accelerate>=1.0" wandb huggingface_hub

# ═══════════════════════════════════════════════════════════════
# Cell 2: Imports + Config
# ═══════════════════════════════════════════════════════════════
import json, re, random, logging, time, os, gc, torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("triage_grpo")

CFG = {
    "model": "Qwen/Qwen2.5-7B",
    "max_seq_length": 512,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0,
    "lora_targets": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "num_generations": 4, "max_completion_length": 200, "temperature": 0.9,
    "epochs": 3, "batch_size": 2, "grad_accum": 4, "lr": 5e-6,
    "logging_steps": 1, "save_steps": 200,
    "output_dir": "/kaggle/working/grpo_output",
    "base_dataset": "balarajr/triage-grpo",
    "augment_max_per_source": 150,
}

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["WANDB_DISABLED"] = "true"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# ═══════════════════════════════════════════════════════════════
# Cell 3: Reward Verifiers (9 functions — proven Grade A)
# ═══════════════════════════════════════════════════════════════
_VALID_ACTIONS = frozenset({
    "TRIAGE_PATIENT","ASSIGN_TREATMENT","TRANSFER_TO_ICU","TRANSFER_TO_WARD",
    "ACTIVATE_OVERFLOW","ORDER_MEDICATION","FLAG_POLICY_VIOLATION",
    "OVERRIDE_DECISION","UPDATE_EHR","REQUEST_STAFF","VERIFY_INSURANCE",
})
_REQUIRED_KEYS = {"action_type","target_id","priority","reasoning"}
_EVIDENCE = [r"P-\d{2,3}",r"patient\s+\d+",r"\d+%",r"\d+/\d+",r"BP\s*\d+",r"HR\s*\d+",r"ICU\s+at\s+\d+",r"beds?\s+\d+",r"age\s+\d+",r"critical|immediate|urgent|stable"]
_FILLER = ["i need more information","i'm not sure","let me think","i cannot determine","i don't know","more data needed"]
_FORBIDDEN = [r"\bimport\s+os\b",r"\bimport\s+sys\b",r"\bexec\s*\(",r"\beval\s*\(",r"\breward\s*[:=]\s*1\.0\b"]

def _extract_json(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def _state_from_prompt(prompt):
    s = {"alive_count":20,"deceased_count":0,"critical_count":0,"icu_occupancy":0.5,"violations_injected":0,"violations_caught":0,"survival_rate":1.0,"crisis_type":"mass_casualty"}
    m = re.search(r"ICU OCCUPANCY:\s*(\d+)%", prompt)
    if m: s["icu_occupancy"] = int(m.group(1))/100.0
    m = re.search(r"CRITICAL PATIENTS\s*\((\d+)", prompt)
    if m: s["critical_count"] = int(m.group(1))
    m = re.search(r"VIOLATIONS INJECTED:\s*(\d+)\s*\|\s*CAUGHT:\s*(\d+)", prompt)
    if m: s["violations_injected"],s["violations_caught"] = int(m.group(1)),int(m.group(2))
    m = re.search(r"SURVIVAL RATE:\s*(\d+\.?\d*)%", prompt)
    if m: s["survival_rate"] = float(m.group(1))/100.0
    m = re.search(r"CRISIS:\s*(\w+)", prompt)
    if m: s["crisis_type"] = m.group(1).lower()
    total = 20; s["alive_count"]=int(s["survival_rate"]*total); s["deceased_count"]=total-s["alive_count"]
    return s

def reward_format_compliance(completions, **kw):
    rewards = []
    for c in completions:
        p = _extract_json(c)
        if not p or not _REQUIRED_KEYS.issubset(p.keys()): rewards.append(0.0); continue
        a = str(p.get("action_type","")).upper()
        if a not in _VALID_ACTIONS: rewards.append(0.0); continue
        try: int(p["target_id"])
        except: rewards.append(0.0); continue
        try:
            pr = int(p["priority"])
            if not 1<=pr<=10: rewards.append(0.0); continue
        except: rewards.append(0.0); continue
        if len(str(p.get("reasoning","")).strip())<10: rewards.append(0.0); continue
        rewards.append(1.0)
    return rewards

def reward_patient_survival(completions, **kw):
    prompts = kw.get("prompts", kw.get("prompt",[""])); rewards = []
    for i,c in enumerate(completions):
        pr = prompts[i] if i<len(prompts) else ""
        s = _state_from_prompt(pr); t = s["alive_count"]+s["deceased_count"]
        rewards.append(s["alive_count"]/t if t>0 else 1.0)
    return rewards

def reward_icu_efficiency(completions, **kw):
    prompts = kw.get("prompts", kw.get("prompt",[""])); rewards = []
    for i,c in enumerate(completions):
        pr = prompts[i] if i<len(prompts) else ""
        occ = _state_from_prompt(pr)["icu_occupancy"]
        if occ<=0.85: rewards.append(1.0)
        elif occ<=0.95: rewards.append(1.0-(occ-0.85)*5.0)
        else: rewards.append(max(0.0,0.5-(occ-0.95)*10.0))
    return rewards

def reward_violation_detection(completions, **kw):
    prompts = kw.get("prompts", kw.get("prompt",[""])); rewards = []
    for i,c in enumerate(completions):
        pr = prompts[i] if i<len(prompts) else ""
        s = _state_from_prompt(pr)
        inj,caught = s["violations_injected"],s["violations_caught"]
        rewards.append(min(1.0,caught/max(inj,1)) if inj>0 else 1.0)
    return rewards

def reward_reasoning_quality(completions, **kw):
    rewards = []
    for c in completions:
        p = _extract_json(c)
        if not p: rewards.append(0.0); continue
        r = str(p.get("reasoning",""))
        if len(r)<20: rewards.append(0.1); continue
        ev = sum(1 for pat in _EVIDENCE if re.search(pat,r,re.I))
        if any(f in r.lower() for f in _FILLER): rewards.append(0.1); continue
        rewards.append(min(1.0,0.3+min(0.7,ev*0.15)))
    return rewards

def reward_response_speed(completions, **kw):
    rewards = []
    for c in completions:
        l = len(c)
        if l<=400: rewards.append(1.0)
        elif l<=800: rewards.append(1.0-(l-400)*0.001)
        else: rewards.append(max(0.2,0.6-(l-800)*0.0005))
    return rewards

def reward_no_hallucination(completions, **kw):
    prompts = kw.get("prompts", kw.get("prompt",[""])); rewards = []
    for i,c in enumerate(completions):
        p = _extract_json(c)
        if not p: rewards.append(0.5); continue
        r = str(p.get("reasoning",""))
        mentioned = {int(m.group(1)) for m in re.finditer(r"P-(\d{2,3})",r,re.I)}
        if not mentioned: rewards.append(1.0); continue
        pr = prompts[i] if i<len(prompts) else ""
        valid = {int(m.group(1)) for m in re.finditer(r"P-(\d{2,3})",pr)}
        rewards.append(0.0 if mentioned-valid else 1.0)
    return rewards

def reward_action_alignment(completions, **kw):
    prompts = kw.get("prompts", kw.get("prompt",[""])); rewards = []
    for i,c in enumerate(completions):
        p = _extract_json(c)
        if not p: rewards.append(0.0); continue
        pr = prompts[i] if i<len(prompts) else ""
        s = _state_from_prompt(pr); a = str(p.get("action_type","")).upper()
        occ,crit = s["icu_occupancy"],s["critical_count"]
        viol = s["violations_injected"]-s["violations_caught"]
        sm = {"TRIAGE_PATIENT":1.0 if crit>0 else 0.5,"TRANSFER_TO_ICU":1.0 if occ<0.9 and crit>0 else 0.3,
              "ACTIVATE_OVERFLOW":1.0 if occ>=0.85 else 0.2,"TRANSFER_TO_WARD":1.0 if occ>=0.7 else 0.4,
              "FLAG_POLICY_VIOLATION":1.0 if viol>0 else 0.4,"OVERRIDE_DECISION":1.0 if occ>=0.9 or crit>=8 else 0.3,
              "ORDER_MEDICATION":0.8 if crit>0 else 0.5,"REQUEST_STAFF":1.0 if s["crisis_type"]=="staff_shortage" or crit>=5 else 0.4,
              "UPDATE_EHR":0.5,"VERIFY_INSURANCE":0.5,"ASSIGN_TREATMENT":0.9 if crit>0 else 0.5}
        rewards.append(sm.get(a,0.5))
    return rewards

def reward_sandbox_safety(completions, **kw):
    rewards = []
    for c in completions:
        safe = True
        for pat in _FORBIDDEN:
            if re.search(pat,c,re.I): safe=False; break
        if len(c)>3000: safe=False
        w = c.split()
        if len(w)>20 and len(set(w))/len(w)<0.2: safe=False
        rewards.append(1.0 if safe else 0.0)
    return rewards

REWARD_FUNCS = [reward_format_compliance,reward_patient_survival,reward_icu_efficiency,
                reward_violation_detection,reward_reasoning_quality,reward_response_speed,
                reward_no_hallucination,reward_action_alignment,reward_sandbox_safety]

# ═══════════════════════════════════════════════════════════════
# Cell 4: Dataset Augmentation (24 HF sources)
# ═══════════════════════════════════════════════════════════════
_AGENTS = ["er_triage","icu_management","pharmacy","cmo_oversight","hr_rostering","it_systems"]
_CRISES = ["mass_casualty","outbreak","equipment_failure","staff_shortage","chemical_spill","power_outage","water_contamination","active_shooter"]
_ROLE = {"er_triage":"ER Triage — patient assessment","icu_management":"ICU Management — bed allocation",
         "pharmacy":"Pharmacy — medication orders","cmo_oversight":"CMO Oversight — policy enforcement",
         "hr_rostering":"HR Rostering — staff allocation","it_systems":"IT Systems — EHR integrity"}
_ACTS = {"er_triage":"TRIAGE_PATIENT, ASSIGN_TREATMENT, UPDATE_EHR","icu_management":"TRANSFER_TO_ICU, TRANSFER_TO_WARD, ACTIVATE_OVERFLOW",
         "pharmacy":"ORDER_MEDICATION, FLAG_POLICY_VIOLATION","cmo_oversight":"OVERRIDE_DECISION, ACTIVATE_OVERFLOW",
         "hr_rostering":"REQUEST_STAFF, FLAG_POLICY_VIOLATION","it_systems":"UPDATE_EHR, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE"}

HF_SOURCES = [
    # Medical Reasoning & QA
    ("TachyHealth/medical_grpo", None, ["question","input","prompt"]),
    ("Intelligent-Internet/II-Medical-RL", None, ["question","prompt"]),
    ("BAAI/AquilaMed-RL", None, ["instruction","input"]),
    ("openlifescienceai/medmcqa", None, ["question"]),
    ("sdiazlor/medical-reasoning-dataset", None, ["question","input"]),
    ("lavita/ChatDoctor-HealthCareMagic-100k", None, ["input","instruction"]),
    ("GBaker/MedQA-USMLE-4-options", None, ["question"]),
    ("TachyHealth/structured_medical", None, ["question","input","text","instruction"]),
    ("YuSun-AI/ReasonMed", None, ["question","input","prompt","text"]),
    ("mamachang/medical-reasoning", None, ["question","input","instruction","text"]),
    # Clinical Notes & Patient Data
    ("starmpcc/Asclepius-Synthetic-Clinical-Notes", None, ["note","text","input"]),
    ("zhengyun21/PMC-Patients", None, ["patient","text","case","input"]),
    ("BI55/MedText", None, ["text","input"]),
    # Pharmacology & Drug Safety
    ("roysc/medication_qa", None, ["Question","question","text"]),
    ("blaze999/Medical-NER", None, ["text","sentence","input"]),
    # Ethics, Safety & Alignment
    ("hendrycks/ethics", "utilitarianism", ["input","text"]),
    ("hendrycks/ethics", "deontology", ["input","scenario"]),
    ("Anthropic/hh-rlhf", None, ["chosen"]),
    ("allenai/prosocial-dialog", None, ["context","response"]),
    ("declare-lab/cicero", None, ["context","question","text","input"]),
    ("wtsheng/synthetic_reasoning_natural", None, ["question","prompt"]),
    ("open-thought/OpenThought-89K", None, ["question","prompt"]),
    ("Replete-AI/rStar-Math", None, ["question","prompt"]),
    ("ajibola16/reasoning-data", None, ["question","prompt"]),
    ("Jiayi-Pan/Tiny-GSM8k", None, ["question","prompt"]),
]

def augment_from_hf(max_per_source=150):
    rng = random.Random(42)
    prompts = []
    from datasets import load_dataset
    for repo, config, fields in HF_SOURCES:
        try:
            kw = {"split":"train","streaming":True}
            if config: kw["name"] = config
            ds = load_dataset(repo, **kw)
            count = 0
            for row in ds:
                if count >= max_per_source: break
                rd = dict(row); text = ""
                for f in fields:
                    v = rd.get(f)
                    if v and isinstance(v,str) and len(v.strip())>15:
                        text = v.strip()[:500]; break
                if not text: continue
                agent = rng.choice(_AGENTS); crisis = rng.choice(_CRISES)
                icu = rng.randint(30,98); crit = rng.randint(1,10)
                pid_block = "\n".join(f"  P-{rng.randint(1,99):03d}: CRITICAL — BP {rng.randint(55,95)}/{rng.randint(25,65)}, HR {rng.randint(95,160)}" for _ in range(min(crit,4)))
                prompt = (
                    f"You are the {agent.upper()} agent in a hospital crisis simulation.\n\n"
                    f"CRISIS: {crisis.upper()}\nSTEP: {rng.randint(0,19)}/20\n"
                    f"ICU OCCUPANCY: {icu}% ({icu*20//100}/20 beds)\n"
                    f"CRITICAL PATIENTS ({crit} total — top 5):\n{pid_block}\n"
                    f"VIOLATIONS INJECTED: {rng.randint(0,5)} | CAUGHT: {rng.randint(0,3)}\n"
                    f"SURVIVAL RATE: {rng.uniform(82,100):.1f}%\n\n"
                    f"MEDICAL CONTEXT:\n{text}\n\n"
                    f"Your role: {_ROLE[agent]}\n\n"
                    f"Respond with ONLY valid JSON:\n"
                    f'{{\n  "action_type": "<one of: {_ACTS[agent]}>",\n'
                    f'  "target_id": <patient ID integer or 0>,\n'
                    f'  "priority": <integer 1-10>,\n'
                    f'  "reasoning": "<1-2 sentences citing specific data>"\n}}'
                )
                prompts.append(prompt); count += 1
            logger.info("Augmented %d from %s", count, repo)
        except Exception as e:
            logger.warning("Skip %s: %s", repo, e)
    rng.shuffle(prompts)
    return prompts

# ═══════════════════════════════════════════════════════════════
# Cell 5: Load Model + LoRA
# ═══════════════════════════════════════════════════════════════
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True)
    logger.info("Loading %s ...", CFG["model"])
    model = AutoModelForCausalLM.from_pretrained(CFG["model"], quantization_config=bnb,
        device_map="auto", torch_dtype=COMPUTE_DTYPE, trust_remote_code=True, token=HF_TOKEN or None)
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True, token=HF_TOKEN or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
                      target_modules=CFG["lora_targets"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════
# Cell 6: Build Dataset
# ═══════════════════════════════════════════════════════════════
def build_dataset():
    from datasets import load_dataset as ld, Dataset
    logger.info("Loading base dataset: %s", CFG["base_dataset"])
    try:
        ds = ld(CFG["base_dataset"], split="train", token=HF_TOKEN or None)
        base = list(ds["prompt"])
    except:
        base = []
    logger.info("Base prompts: %d", len(base))
    logger.info("Augmenting from %d HF sources...", len(HF_SOURCES))
    aug = augment_from_hf(CFG["augment_max_per_source"])
    logger.info("Augmented prompts: %d", len(aug))
    all_prompts = base + aug
    random.shuffle(all_prompts)
    logger.info("Total training prompts: %d", len(all_prompts))
    return Dataset.from_dict({"prompt": all_prompts})

# ═══════════════════════════════════════════════════════════════
# Cell 7: Train
# ═══════════════════════════════════════════════════════════════
def train():
    model, tokenizer = load_model()
    dataset = build_dataset()
    import inspect
    from trl import GRPOTrainer, GRPOConfig
    grpo_kwargs = dict(
        output_dir=CFG["output_dir"], num_train_epochs=CFG["epochs"],
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"],
        max_completion_length=CFG["max_completion_length"],
        num_generations=CFG["num_generations"], temperature=CFG["temperature"],
        logging_steps=CFG["logging_steps"], save_steps=CFG["save_steps"],
        save_total_limit=2, report_to="none", bf16=USE_BF16, fp16=not USE_BF16, seed=42, log_level="info",
    )
    config_sig = inspect.signature(GRPOConfig)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in config_sig.parameters.values()):
        supported = set(config_sig.parameters)
        skipped = sorted(set(grpo_kwargs) - supported)
        if skipped:
            logger.info("Skipping unsupported GRPOConfig args for installed TRL: %s", skipped)
        grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in supported}
    args = GRPOConfig(**grpo_kwargs)

    trainer_kwargs = dict(model=model, reward_funcs=REWARD_FUNCS, args=args, train_dataset=dataset)
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    logger.info("Starting GRPO training (%d prompts, %d epochs)...", len(dataset), CFG["epochs"])
    result = trainer.train()
    logger.info("Done! Loss=%.4f Steps=%d", result.training_loss, result.global_step)
    trainer.save_model(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])
    return model, tokenizer, result

# ═══════════════════════════════════════════════════════════════
# Cell 8: Quick Eval
# ═══════════════════════════════════════════════════════════════
def quick_eval():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    base = AutoModelForCausalLM.from_pretrained(CFG["model"], quantization_config=bnb,
        device_map="auto", torch_dtype=COMPUTE_DTYPE, trust_remote_code=True, token=HF_TOKEN or None)
    model = PeftModel.from_pretrained(base, CFG["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(CFG["output_dir"], trust_remote_code=True)
    test_prompt = (
        "You are the ER_TRIAGE agent in a hospital crisis simulation.\n\n"
        "CRISIS: MASS_CASUALTY\nSTEP: 5/20\nICU OCCUPANCY: 85% (17/20 beds)\n"
        "CRITICAL PATIENTS (3 total):\n  P-042: CRITICAL — BP 72/40, HR 140\n"
        "  P-019: CRITICAL — BP 65/35, HR 155\n  P-067: CRITICAL — BP 80/50, HR 120\n"
        "VIOLATIONS INJECTED: 2 | CAUGHT: 1\nSURVIVAL RATE: 90.0%\n\n"
        'Respond with ONLY valid JSON:\n{\n  "action_type": "<one of: TRIAGE_PATIENT, ASSIGN_TREATMENT, UPDATE_EHR>",\n'
        '  "target_id": <patient ID integer or 0>,\n  "priority": <integer 1-10>,\n'
        '  "reasoning": "<1-2 sentences citing specific data>"\n}'
    )
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("=" * 60)
    print("EVAL RESPONSE:")
    print(response)
    print("=" * 60)
    parsed = _extract_json(response)
    if parsed:
        scores = {fn.__name__: fn([response], prompts=[test_prompt])[0] for fn in REWARD_FUNCS}
        total = sum(scores.values())/len(scores)*100
        print(f"\nReward Scores: {json.dumps(scores, indent=2)}")
        print(f"Overall: {total:.0f}/100")
    else:
        print("WARNING: Could not parse JSON from response")

# ═══════════════════════════════════════════════════════════════
# Cell 9: Run Everything
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = time.time()
    model, tokenizer, result = train()
    print(f"\nTraining time: {(time.time()-start)/60:.1f} min")
    del model; gc.collect(); torch.cuda.empty_cache()
    quick_eval()
    print("\n✅ TRIAGE GRPO training complete!")
