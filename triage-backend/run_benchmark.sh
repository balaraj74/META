#!/bin/bash
# Run the TRIAGE benchmark with 5 episodes × 30 steps
cd "$(dirname "$0")"
exec .venv/bin/python scripts/benchmark_agent.py --episodes 5 --steps 30 --output results/bench_full_30step.json 2>&1
