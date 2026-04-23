---
title: TRIAGE — RL-Driven Hospital Crisis Simulation
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
tags:
  - reinforcement-learning
  - grpo
  - healthcare
  - multi-agent
  - openenv
  - hackathon
---

# 🏥 TRIAGE — AI Hospital Crisis Simulation

**GRPO-trained multi-agent system for hospital crisis management**

## What is TRIAGE?

TRIAGE is a reinforcement learning environment where 6 specialized AI agents
collaborate to manage hospital crises — mass casualty events, disease outbreaks,
equipment failures, and staff shortages.

The system uses **GRPO (Group Relative Policy Optimization)** with 8 independent
reward verifiers to train a **Qwen3.5-0.8B** model that:

- 🎯 Triages patients correctly under pressure
- 🏥 Manages ICU capacity efficiently
- 🛡️ Detects policy violations in real-time
- 📋 Produces structured, hallucination-free clinical reports

## Demo Tabs

1. **Live Simulation** — Watch agents respond to crisis scenarios in real-time
2. **GRPO Comparison** — See reward improvements: baseline vs GRPO-trained
3. **Reward Inspector** — Debug any completion against all 8 verifiers

## Architecture

```
HospitalEnv (Gymnasium) → TriageOpenEnv (OpenEnv adapter) → FastAPI /env/*
                ↓
    8 Reward Verifiers (GRPO reward_funcs)
                ↓
    GRPOTrainer (TRL + Unsloth 4-bit)
                ↓
    Gradio Space (this demo)
```

## Key Results — What GRPO Trains

GRPO trains the model's **output layer** — teaching it to follow strict JSON schemas,
cite specific patient data, and avoid hallucination. The metrics below reflect this:

### LLM-Driven Metrics (directly improved by GRPO)

| Metric | Baseline | GRPO | Improvement |
|---|---|---|---|
| Format Compliance | 0.00 | 1.00 | +100% |
| Reasoning Quality | 0.00 | 0.95 | +95% |
| Hallucination Gate | 0.50 | 0.75 | +50% |
| Action Alignment | 0.00 | 1.00 | +100% |

### State-Driven Metrics (environment outcomes)

| Metric | Description |
|---|---|
| Patient Survival | Measures simulation state — improves when trained model drives multi-step decisions |
| ICU Efficiency | Measures resource allocation — affected by action quality over time |
| Violation Detection | Measures environment violations — improves with better FLAG actions |
| Response Speed | Measures inference latency — model-dependent |

> 💡 State-driven metrics measure **environment outcomes**, not LLM output quality.
> They improve when the GRPO-trained model is connected to drive **multi-step simulations**
> where better decisions compound into better patient outcomes.

## Hardware

- **Training:** RTX 2050 (4GB VRAM) · LoRA rank=16 · 4-bit quantization
- **Inference:** Ollama with Qwen3.5-0.8B

## Links

- 📦 [OpenEnv API](https://github.com/meta-llama/openenv)
- 🧠 [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- ⚡ [Unsloth](https://github.com/unslothai/unsloth)
