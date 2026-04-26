#!/usr/bin/env python3
"""
TRIAGE GRPO Benchmark — Kaggle GPU Edition
Run real LLM inference against 9 reward verifiers to get verified scores.
Compatible with Kaggle T4/P100 GPU (free tier).
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════
# !pip install -q git+https://github.com/huggingface/transformers.git peft accelerate huggingface_hub

# ═══════════════════════════════════════════════════════════════
# Cell 2: Imports + Config
# ═══════════════════════════════════════════════════════════════
import json, re, random, time, os, gc, torch
from pathlib import Path
from statistics import mean

CFG = {
    "model": "Qwen/Qwen3.5-4B",
    # Primary: load merged model from HF Hub (both GRPO training runs baked in)
    "merged_model_hf": "balarajr/triage-qwen3.5-4b-grpo",
    # Fallback: local Kaggle dataset paths
    "merged_model_local": "/kaggle/input/triage-grpo-combined",
    "adapter": "/kaggle/working/grpo_output",  # LoRA adapter path (last resort)
    "max_new_tokens": 250,
    "temperature": 0.7,
    "num_scenarios": 50,
    "batch_size": 5,
}

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Dtype: {COMPUTE_DTYPE}")

# ═══════════════════════════════════════════════════════════════
# Cell 3: HF Login
# ═══════════════════════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# ═══════════════════════════════════════════════════════════════
# Cell 4: Reward Verifiers (9 functions — same as training)
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
# Cell 5: Scenario Generator
# ═══════════════════════════════════════════════════════════════
_AGENTS = ["er_triage","icu_management","pharmacy","cmo_oversight","hr_rostering","it_systems"]
_CRISES = ["mass_casualty","outbreak","equipment_failure","staff_shortage","chemical_spill","power_outage"]
_ROLE = {"er_triage":"ER Triage — patient assessment","icu_management":"ICU Management — bed allocation",
         "pharmacy":"Pharmacy — medication orders","cmo_oversight":"CMO Oversight — policy enforcement",
         "hr_rostering":"HR Rostering — staff allocation","it_systems":"IT Systems — EHR integrity"}
_ACTS = {"er_triage":"TRIAGE_PATIENT, ASSIGN_TREATMENT, UPDATE_EHR","icu_management":"TRANSFER_TO_ICU, TRANSFER_TO_WARD, ACTIVATE_OVERFLOW",
         "pharmacy":"ORDER_MEDICATION, FLAG_POLICY_VIOLATION","cmo_oversight":"OVERRIDE_DECISION, ACTIVATE_OVERFLOW",
         "hr_rostering":"REQUEST_STAFF, FLAG_POLICY_VIOLATION","it_systems":"UPDATE_EHR, FLAG_POLICY_VIOLATION, VERIFY_INSURANCE"}

def generate_scenarios(n=50, seed=42):
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        agent = rng.choice(_AGENTS); crisis = rng.choice(_CRISES)
        icu = rng.randint(30,98); crit = rng.randint(1,10)
        vi = rng.randint(0,5); vc = rng.randint(0, min(vi, 3))
        sr = round(rng.uniform(82,100), 1)
        pid_block = "\n".join(
            f"  P-{rng.randint(1,99):03d}: CRITICAL — BP {rng.randint(55,95)}/{rng.randint(25,65)}, HR {rng.randint(95,160)}"
            for _ in range(min(crit,4))
        )
        prompt = (
            f"You are the {agent.upper()} agent in a hospital crisis simulation.\n\n"
            f"CRISIS: {crisis.upper()}\nSTEP: {rng.randint(0,19)}/20\n"
            f"ICU OCCUPANCY: {icu}% ({icu*20//100}/20 beds)\n"
            f"CRITICAL PATIENTS ({crit} total — top 5):\n{pid_block}\n"
            f"VIOLATIONS INJECTED: {vi} | CAUGHT: {vc}\n"
            f"SURVIVAL RATE: {sr}%\n\n"
            f"Your role: {_ROLE[agent]}\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{\n  "action_type": "<one of: {_ACTS[agent]}>",\n'
            f'  "target_id": <patient ID integer or 0>,\n'
            f'  "priority": <integer 1-10>,\n'
            f'  "reasoning": "<1-2 sentences citing specific data>"\n}}'
        )
        prompts.append(prompt)
    return prompts

# ═══════════════════════════════════════════════════════════════
# Cell 6: Load Model
# ═══════════════════════════════════════════════════════════════
def load_model(merged_path=None, adapter_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _is_hf_repo(path_str):
        """Check if string looks like a HuggingFace repo ID (e.g. 'user/model')."""
        return path_str and "/" in path_str and not path_str.startswith("/")

    def _is_local(path_str):
        """Check if string is a valid local directory with model files."""
        return path_str and Path(path_str).exists()

    # Priority: 1) Fully merged model (local or HF Hub)  2) Base + adapter  3) Base only
    if merged_path and (_is_local(merged_path) or _is_hf_repo(merged_path)):
        source = "LOCAL" if _is_local(merged_path) else "HuggingFace Hub"
        print(f"✅ Loading MERGED model ({source}): {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            device_map="auto",
            dtype=COMPUTE_DTYPE,
            trust_remote_code=True,
            attn_implementation="eager",
            token=HF_TOKEN or None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            merged_path, trust_remote_code=True, token=HF_TOKEN or None
        )
    else:
        print(f"Loading base model: {CFG['model']} in {COMPUTE_DTYPE}...")
        model = AutoModelForCausalLM.from_pretrained(
            CFG["model"],
            device_map="auto",
            dtype=COMPUTE_DTYPE,
            trust_remote_code=True,
            token=HF_TOKEN or None,
            attn_implementation="eager",
        )

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            print(f"Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        else:
            print("⚠ No adapter found — using base model (pre-training baseline)")
            tokenizer = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True, token=HF_TOKEN or None)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Fix padding side for decoder-only models
    tokenizer.padding_side = "left"

    model.eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    return model, tokenizer

# ═══════════════════════════════════════════════════════════════
# Cell 7: Inference Engine
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model, tokenizer, prompts, batch_size=5):
    responses = []
    latencies = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=512).to(model.device)
        t0 = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=CFG["max_new_tokens"],
                                 temperature=CFG["temperature"], do_sample=True,
                                 pad_token_id=tokenizer.pad_token_id)
        latency = (time.perf_counter() - t0) / len(batch)
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(response)
            latencies.append(latency)
        done = min(i + batch_size, len(prompts))
        print(f"  Inference: {done}/{len(prompts)} ({latency*1000:.0f}ms/sample)", end="\r")
    print()
    return responses, latencies

# ═══════════════════════════════════════════════════════════════
# Cell 8: Benchmark Runner
# ═══════════════════════════════════════════════════════════════
def run_benchmark(model, tokenizer, num_scenarios=50):
    print("="*60)
    print("  TRIAGE GRPO BENCHMARK — REAL MODEL INFERENCE")
    print("="*60)

    prompts = generate_scenarios(num_scenarios)
    print(f"\nGenerated {len(prompts)} test scenarios")
    print("Running inference...")

    responses, latencies = run_inference(model, tokenizer, prompts, CFG["batch_size"])

    # Score each response with all 9 reward functions
    print("\nScoring with 9 reward verifiers...")
    all_scores = {}
    for fn in REWARD_FUNCS:
        name = fn.__name__
        scores = fn(responses, prompts=prompts)
        all_scores[name] = scores
        avg = mean(scores)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {name:<30} {bar} {avg:.3f}")

    # Composite score
    per_sample = []
    for i in range(len(responses)):
        sample_avg = mean(all_scores[fn.__name__][i] for fn in REWARD_FUNCS)
        per_sample.append(sample_avg)

    composite = mean(per_sample) * 100
    grade = "A" if composite >= 75 else ("B" if composite >= 60 else ("C" if composite >= 45 else "D"))

    # Parse success rate
    parsed = sum(1 for r in responses if _extract_json(r) is not None)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Scenarios tested   : {num_scenarios}")
    print(f"  JSON parse rate    : {parsed}/{num_scenarios} ({parsed/num_scenarios*100:.1f}%)")
    print(f"  Avg latency        : {mean(latencies)*1000:.0f} ms/sample")
    print(f"  Avg reward (0-1)   : {mean(per_sample):.4f}")
    print(f"  Composite score    : {composite:.1f} / 100")
    print(f"  Grade              : {grade}")
    print(f"{'='*60}")

    # Detailed per-verifier breakdown
    results = {
        "model": CFG["model"],
        "adapter": CFG["adapter"],
        "num_scenarios": num_scenarios,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "dtype": str(COMPUTE_DTYPE),
        "json_parse_rate": parsed / num_scenarios,
        "avg_latency_ms": mean(latencies) * 1000,
        "composite_score": composite,
        "grade": grade,
        "per_verifier": {fn.__name__: round(mean(all_scores[fn.__name__]), 4) for fn in REWARD_FUNCS},
        "per_sample_scores": [round(s, 4) for s in per_sample],
        "sample_responses": [
            {"prompt_preview": p[:200], "response": r, "score": round(s, 4)}
            for p, r, s in list(zip(prompts, responses, per_sample))[:10]
        ],
    }

    # Save results
    out_path = Path("/kaggle/working/benchmark_results.json")
    if not out_path.parent.exists():
        out_path = Path("benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✔ Results saved → {out_path}")

    return results

# ═══════════════════════════════════════════════════════════════
# Cell 9: Run Everything
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = time.time()

    merged = None
    adapter = None

    # Strategy 1: Local Kaggle dataset (if user uploaded model as dataset)
    local_paths = [
        CFG["merged_model_local"],
        "/kaggle/input/triage-grpo-combined/merged_grpo_combined",
        "/kaggle/input/merged-grpo-combined",
    ]
    for p in local_paths:
        if Path(p).exists() and list(Path(p).glob("*.safetensors")):
            merged = p
            print(f"✅ Found local merged model: {p}")
            break

    # Strategy 2: HuggingFace Hub (download on-the-fly)
    if not merged:
        hf_repo = CFG["merged_model_hf"]
        print(f"📥 Loading merged model from HuggingFace Hub: {hf_repo}")
        merged = hf_repo  # transformers handles HF repo IDs natively

    # Strategy 3: LoRA adapter on base model (fallback)
    if not merged:
        adapter = CFG["adapter"]
        if not Path(adapter).exists():
            alt = Path("/kaggle/working/grpo_output")
            adapter = str(alt) if alt.exists() else None
        if not adapter:
            print("⚠ No merged model or adapter — benchmarking BASE model")

    model, tokenizer = load_model(merged_path=merged, adapter_path=adapter)
    results = run_benchmark(model, tokenizer, num_scenarios=CFG["num_scenarios"])

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print("✅ TRIAGE GRPO Benchmark complete!")
