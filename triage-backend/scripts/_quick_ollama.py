#!/usr/bin/env python3
"""Quick single-shot Ollama triage test — minimal, fast."""
import urllib.request, json, re, time, sys

PROMPT = """You are an ER_TRIAGE agent. CRISIS: MASS_CASUALTY. ICU at 92%. 12 critical patients.
Top patient: P-7, age 67, CRITICAL, cardiac_arrest, deterioration=0.85.
Respond with ONLY valid JSON:
{"action_type": "TRIAGE_PATIENT or TRANSFER_TO_ICU", "target_id": 7, "priority": 1, "reasoning": "1-2 sentences"}"""

payload = json.dumps({
    "model": "qwen3.5:0.8b",
    "prompt": PROMPT,
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 150, "num_ctx": 512}
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read())
dt = time.time() - t0
raw = data.get("response", "")
tokens = data.get("eval_count", 0)

# Strip <think>...</think>
cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

print(f"Time: {dt:.1f}s | Tokens: {tokens}")
print(f"Raw ({len(raw)} chars): {repr(raw[:300])}")
print(f"Cleaned: {repr(cleaned[:300])}")

# Try JSON parse
match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
if match:
    try:
        parsed = json.loads(match.group(0))
        print(f"PARSED: {json.dumps(parsed, indent=2)}")
    except:
        print("JSON extraction failed")
else:
    print("NO JSON found in output")
