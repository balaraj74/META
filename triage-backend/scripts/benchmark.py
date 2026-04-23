#!/usr/bin/env python3
"""
TRIAGE Benchmark
================
Measures end-to-end latency and token throughput for:
  1. Ollama direct (think=False, num_ctx=512)  -- raw model speed
  2. Ollama direct (think=True,  num_ctx=512)  -- CoT overhead
  3. Backend /api/chat endpoint                -- full stack latency
  4. Backend /api/chat/stream endpoint         -- TTFF (time-to-first-token)

Usage:  python3 scripts/benchmark.py [--runs N]
"""

import argparse
import json
import statistics
import sys
import time
from typing import Any

import httpx

OLLAMA_URL = "http://localhost:11434"
BACKEND_URL = "http://localhost:8000"

PROMPTS = [
    "What is the current patient count in the emergency department?",
    "Administer 10mg morphine IV to patient in bay 3.",
    "We have a mass-casualty incident — activate surge protocol.",
    "Is OR-2 available for an emergency appendectomy?",
    "What is the trauma team status?",
]

AGENTS = ["CMO_OVERSIGHT", "TRIAGE_COORDINATOR", "PHARMACY", "OR_COORDINATOR", "RESOURCE_MANAGER"]


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(secs: float) -> str:
    return f"{secs*1000:.0f} ms"

def fmt_tps(tps: float) -> str:
    return f"{tps:.1f} tok/s"

def bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, (value - lo) / max(hi - lo, 1)))
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)

def percentile(data: list[float], p: float) -> float:
    data_sorted = sorted(data)
    idx = (len(data_sorted) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (idx - lo)


# ── bench functions ───────────────────────────────────────────────────────────

def bench_ollama(think: bool, runs: int) -> dict[str, Any]:
    label = f"Ollama direct (think={'on ' if think else 'off'})"
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    latencies: list[float] = []
    tps_list: list[float] = []
    token_counts: list[int] = []
    failures = 0

    for i, prompt in enumerate(PROMPTS[:runs]):
        try:
            t0 = time.perf_counter()
            r = httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": "qwen3.5:0.8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": think,
                    "options": {"num_ctx": 512, "num_predict": 150},
                },
                timeout=60.0,
            )
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            tokens = data.get("eval_count", 0)
            dur_s = data.get("eval_duration", 0) / 1e9
            tps = tokens / dur_s if dur_s > 0 else 0

            latencies.append(elapsed)
            tps_list.append(tps)
            token_counts.append(tokens)

            content = data.get("message", {}).get("content", "")[:60].replace("\n", " ")
            status = "✓" if content else "⚠ empty"
            print(f"  [{i+1}/{runs}] {fmt(elapsed):>8}  {fmt_tps(tps):>12}  {status}  {content!r}")
        except Exception as exc:
            failures += 1
            print(f"  [{i+1}/{runs}] FAILED: {exc}")

    if latencies:
        print(f"\n  {'Metric':<22} {'Value':>10}")
        print(f"  {'─'*34}")
        print(f"  {'Runs completed':<22} {len(latencies):>10}/{runs}")
        print(f"  {'Failures':<22} {failures:>10}")
        print(f"  {'Latency median':<22} {fmt(statistics.median(latencies)):>10}")
        print(f"  {'Latency mean':<22} {fmt(statistics.mean(latencies)):>10}")
        print(f"  {'Latency p95':<22} {fmt(percentile(latencies, 95)):>10}")
        print(f"  {'Latency min':<22} {fmt(min(latencies)):>10}")
        print(f"  {'Latency max':<22} {fmt(max(latencies)):>10}")
        print(f"  {'Throughput median':<22} {fmt_tps(statistics.median(tps_list)):>10}")
        print(f"  {'Throughput mean':<22} {fmt_tps(statistics.mean(tps_list)):>10}")
        print(f"  {'Avg tokens/reply':<22} {statistics.mean(token_counts):>10.1f}")

    return {
        "label": label,
        "latencies": latencies,
        "tps": tps_list,
        "failures": failures,
    }


def bench_backend_chat(runs: int) -> dict[str, Any]:
    label = "Backend /api/chat (full stack)"
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    latencies: list[float] = []
    failures = 0
    mock_count = 0

    pairs = list(zip(PROMPTS, AGENTS))[:runs]

    for i, (prompt, agent) in enumerate(pairs):
        try:
            t0 = time.perf_counter()
            r = httpx.post(
                f"{BACKEND_URL}/api/chat",
                json={"message": prompt, "agent_id": agent, "context": ""},
                timeout=60.0,
            )
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()

            model = data.get("data", {}).get("model", "?")
            response = data.get("data", {}).get("response", "")[:60].replace("\n", " ")
            is_mock = model == "mock"
            if is_mock:
                mock_count += 1

            latencies.append(elapsed)
            tag = "MOCK" if is_mock else model
            print(f"  [{i+1}/{runs}] {fmt(elapsed):>8}  [{tag:^20}]  {response!r}")
        except Exception as exc:
            failures += 1
            print(f"  [{i+1}/{runs}] FAILED: {exc}")

    if latencies:
        print(f"\n  {'Metric':<22} {'Value':>10}")
        print(f"  {'─'*34}")
        print(f"  {'Runs completed':<22} {len(latencies):>10}/{runs}")
        print(f"  {'Mock fallbacks':<22} {mock_count:>10}")
        print(f"  {'Failures':<22} {failures:>10}")
        print(f"  {'Latency median':<22} {fmt(statistics.median(latencies)):>10}")
        print(f"  {'Latency mean':<22} {fmt(statistics.mean(latencies)):>10}")
        print(f"  {'Latency p95':<22} {fmt(percentile(latencies, 95)):>10}")
        print(f"  {'Latency min':<22} {fmt(min(latencies)):>10}")
        print(f"  {'Latency max':<22} {fmt(max(latencies)):>10}")

    return {
        "label": label,
        "latencies": latencies,
        "mock_count": mock_count,
        "failures": failures,
    }


def bench_stream_ttff(runs: int) -> dict[str, Any]:
    label = "Backend /api/chat/stream (TTFF)"
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    ttff_list: list[float] = []
    total_list: list[float] = []
    failures = 0

    pairs = list(zip(PROMPTS, AGENTS))[:runs]

    for i, (prompt, agent) in enumerate(pairs):
        try:
            ttff = None
            t0 = time.perf_counter()
            with httpx.stream(
                "POST",
                f"{BACKEND_URL}/api/chat/stream",
                json={"message": prompt, "agent_id": agent, "context": ""},
                timeout=60.0,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        if ttff is None:
                            ttff = time.perf_counter() - t0
                        payload = json.loads(line[5:].strip())
                        if payload.get("done"):
                            break
            total = time.perf_counter() - t0

            if ttff is not None:
                ttff_list.append(ttff)
                total_list.append(total)
                print(f"  [{i+1}/{runs}] TTFF {fmt(ttff):>8}  total {fmt(total):>8}")
            else:
                failures += 1
                print(f"  [{i+1}/{runs}] No data received")
        except Exception as exc:
            failures += 1
            print(f"  [{i+1}/{runs}] FAILED: {exc}")

    if ttff_list:
        print(f"\n  {'Metric':<22} {'Value':>10}")
        print(f"  {'─'*34}")
        print(f"  {'Runs completed':<22} {len(ttff_list):>10}/{runs}")
        print(f"  {'TTFF median':<22} {fmt(statistics.median(ttff_list)):>10}")
        print(f"  {'TTFF mean':<22} {fmt(statistics.mean(ttff_list)):>10}")
        print(f"  {'TTFF p95':<22} {fmt(percentile(ttff_list, 95)):>10}")
        print(f"  {'Total time median':<22} {fmt(statistics.median(total_list)):>10}")
        print(f"  {'Total time mean':<22} {fmt(statistics.mean(total_list)):>10}")

    return {
        "label": label,
        "ttff": ttff_list,
        "total": total_list,
        "failures": failures,
    }


def print_summary(results: list[dict]) -> None:
    print(f"\n{'═'*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Suite':<34} {'Median':>8}  {'p95':>8}  {'Pass'}")
    print(f"  {'─'*58}")
    for r in results:
        latencies = r.get("latencies") or r.get("ttff") or []
        if not latencies:
            print(f"  {r['label']:<34} {'N/A':>8}  {'N/A':>8}  ✗")
            continue
        med = statistics.median(latencies)
        p95 = percentile(latencies, 95)
        ok = med < 5.0  # < 5 s median is our pass threshold
        print(f"  {r['label']:<34} {fmt(med):>8}  {fmt(p95):>8}  {'✓' if ok else '✗'}")
    print(f"{'═'*60}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TRIAGE benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Requests per suite (default 5)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  TRIAGE — LLM BENCHMARK")
    print(f"  Model: qwen3.5:0.8b   Runs per suite: {args.runs}")
    print(f"{'═'*60}")

    # Warm up the model (first request loads weights into GPU)
    print("\n  Warming up model…", end="", flush=True)
    try:
        httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "qwen3.5:0.8b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False, "think": False,
                "options": {"num_ctx": 512, "num_predict": 10},
            },
            timeout=60.0,
        )
        print(" done ✓")
    except Exception as e:
        print(f" FAILED ({e}) — results may include cold-start penalty")

    results = []
    results.append(bench_ollama(think=False, runs=args.runs))
    results.append(bench_ollama(think=True,  runs=args.runs))
    results.append(bench_backend_chat(runs=args.runs))
    results.append(bench_stream_ttff(runs=args.runs))

    print_summary(results)


if __name__ == "__main__":
    main()
