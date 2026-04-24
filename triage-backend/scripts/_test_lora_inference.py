import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3.5-4B"
ADAPTER_DIR = "/home/balaraj/META final/triage-backend/models/triage_grpo_output"

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("2. Configuring 4-bit Quantization (to fit in 4GB VRAM)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"3. Loading Base Model ({MODEL_ID})...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("4. Applying LoRA Adapter from GRPO Training...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# System prompt exactly matching the training environment
SYSTEM_PROMPT = """You are a hospital triage AI. Respond EXACTLY in this format:
SEVERITY: <CRITICAL|HIGH|MEDIUM|LOW>
ACTION: <one of: EVACUATE, LOCKDOWN, QUARANTINE, DIVERT_AMBULANCE, REQUEST_STAFF, TRIAGE_OVERRIDE, MONITOR, SHELTER_IN_PLACE>
REASONING: <2-4 sentences explaining your decision with medical/operational justification>"""

# A difficult test scenario
user_prompt = """CRISIS: MASS CASUALTY
ROLE: triage_officer
PATIENTS: 12 critical, 25 waiting, 18/20 alive
ICU: 9/10 beds, 0 ventilators free
STAFF: 1 available (2D/3N)
VIOLATIONS: 0

What is your triage decision?"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt},
]

print("\n5. Formatting Chat Template (thinking mode OFF — structured output)...")
# enable_thinking=False keeps Qwen3.5 from spending all tokens on CoT
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

print("\n6. Generating Response...\n")
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,   # enough for full SEVERITY/ACTION/REASONING block
        temperature=0.3,
        do_sample=True,
    )

end_time = time.time()
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("=== TRAINED AI RESPONSE ===")
print(response)
print("===========================")
print(f"Generation took: {end_time - start_time:.2f} seconds")
