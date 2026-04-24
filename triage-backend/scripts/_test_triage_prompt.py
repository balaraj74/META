#!/usr/bin/env python3
"""Test qwen3.5 on a triage prompt to see raw output quality."""
import urllib.request
import json
import time

PROMPT = """You are the ER_TRIAGE agent in a hospital crisis simulation.

CRISIS: MASS_CASUALTY
STEP: 3/20
ICU OCCUPANCY: 92% (46/50 beds)
CRITICAL PATIENTS (12 total — top 5):
  - P-7: John, age 67, status=CRITICAL, condition=cardiac_arrest, triage=9, deterioration=0.85
  - P-12: Maria, age 45, status=CRITICAL, condition=trauma, triage=7, deterioration=0.60
  - P-23: Robert, age 72, status=CRITICAL, condition=respiratory_failure, triage=8, deterioration=0.90
  - P-31: Lisa, age 34, status=STABLE, condition=fracture, triage=3, deterioration=0.10
  - P-42: David, age 58, status=CRITICAL, condition=sepsis, triage=8, deterioration=0.75
VIOLATIONS INJECTED: 3 | CAUGHT: 1
SURVIVAL RATE: 93.3%

Your role: Triage incoming patients by severity, assign RED/YELLOW/GREEN tags, and prioritize ICU admission.

Decide the single most important action right now. Respond with ONLY valid JSON:
{
  "action_type": "<one of: TRIAGE_PATIENT, TRANSFER_TO_ICU, ASSIGN_TREATMENT, FLAG_POLICY_VIOLATION>",
  "target_id": <patient ID integer or 0 if not patient-specific>,
  "priority": <integer 1-10, where 1=highest>,
  "reasoning": "<1-2 sentences citing specific patient data or metrics>"
}"""


def generate(model, prompt, temperature=0.3, max_tokens=200):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
        }
    }).encode()
    
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    dt = time.time() - t0
    return data.get("response", ""), dt, data.get("eval_count", 0)


print("=" * 70)
print("Testing qwen3.5:0.8b on TRIAGE prompt")
print("=" * 70)

for i in range(3):
    response, dt, tokens = generate("qwen3.5:0.8b", PROMPT)
    print(f"\n--- Attempt {i+1} ({dt:.1f}s, {tokens} tokens) ---")
    print(repr(response[:500]))
    
    # Check if it's valid JSON
    import re
    cleaned = response.strip()
    # Remove <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        print(f"  -> VALID JSON: action={parsed.get('action_type')}")
    except:
        # Try to find JSON in the response
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                print(f"  -> EXTRACTED JSON: action={parsed.get('action_type')}")
            except:
                print(f"  -> INVALID JSON (even after extraction)")
        else:
            print(f"  -> NO JSON FOUND")

print("\n" + "=" * 70)
