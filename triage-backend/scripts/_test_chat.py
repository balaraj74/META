#!/usr/bin/env python3
"""Test Ollama chat API (not generate) — qwen3.5 needs chat mode for proper output."""
import urllib.request, json, re, time

SYSTEM = "You are an ER triage agent. Respond with ONLY valid JSON. No explanation, no markdown."

USER_MSG = """CRISIS: MASS_CASUALTY. ICU at 92% (46/50 beds). 12 critical patients.
Top: P-7 age 67 CRITICAL cardiac_arrest deterioration=0.85
     P-23 age 72 CRITICAL respiratory_failure deterioration=0.90
Violations: 3 injected, 1 caught. Survival: 93.3%.

Respond with ONLY this JSON format:
{"action_type": "TRIAGE_PATIENT", "target_id": 7, "priority": 1, "reasoning": "cite patient data"}"""

payload = json.dumps({
    "model": "qwen3.5:0.8b",
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_MSG}
    ],
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 200, "num_ctx": 1024}
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read())
dt = time.time() - t0

msg = data.get("message", {}).get("content", "")
tokens = data.get("eval_count", 0)

print(f"Time: {dt:.1f}s | Tokens: {tokens}")
print(f"Raw ({len(msg)} chars):")
print(msg[:500])
print()

# Strip thinking
cleaned = re.sub(r'<think>.*?</think>', '', msg, flags=re.DOTALL).strip()
print(f"After strip think ({len(cleaned)} chars):")
print(cleaned[:500])

# Try JSON
match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
if match:
    try:
        parsed = json.loads(match.group(0))
        print(f"\nPARSED JSON: {json.dumps(parsed, indent=2)}")
        print("STATUS: SUCCESS")
    except Exception as e:
        print(f"\nJSON parse error: {e}")
        print("STATUS: PARTIAL")
else:
    print("\nSTATUS: NO_JSON")
