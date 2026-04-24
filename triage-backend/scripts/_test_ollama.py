#!/usr/bin/env python3
"""Quick test: can we talk to Ollama?"""
import urllib.request
import json
import time
import sys

def test_ollama():
    # Test 1: tags endpoint
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        tags = json.loads(resp.read())
        models = [m["name"] for m in tags.get("models", [])]
        print(f"[OK] Ollama reachable. Models: {models}")
    except Exception as e:
        print(f"[FAIL] Cannot reach Ollama: {e}")
        sys.exit(1)

    # Test 2: generate
    try:
        payload = json.dumps({
            "model": "qwen3.5:0.8b",
            "prompt": "Respond with ONLY this JSON: {\"answer\": \"hello\"}",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 40}
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
        response_text = data.get("response", "")
        tokens = data.get("eval_count", "?")
        print(f"[OK] Generation took {dt:.1f}s, {tokens} tokens")
        print(f"     Response: {repr(response_text[:200])}")
    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        sys.exit(1)

    print("\nOllama is ready for live benchmark.")


if __name__ == "__main__":
    test_ollama()
