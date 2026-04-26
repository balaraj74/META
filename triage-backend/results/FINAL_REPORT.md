# TRIAGE — Final Project Report
## Meta PyTorch OpenEnv Hackathon Submission

**Generated:** 2026-04-24  
**Model:** `merged_grpo_final` (Qwen2.5-7B + GRPO LoRA)  
**Score:** 🏆 **90.00 / 100 — Grade A**

---

## 1. Executive Summary

TRIAGE is a multi-agent hospital crisis simulation where six specialized AI agents
coordinate real-time responses to mass-casualty events, disease outbreaks, equipment
failures, staff shortages, and combined surges.

**Key results:**
- Composite benchmark score: **90/100 (Grade A)**
- Survival rate: **100%** across all 5 crisis types × 3 episodes
- Violation detection: **100%** (30/30 violations caught)
- GRPO inference latency: **~5.3s** (4-bit quantized, RTX 2050)

---

## 2. Training — GRPO on Qwen2.5-7B

| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-7B` |
| Method | GRPO (Group Relative Policy Optimization) |
| LoRA Rank / Alpha | 16 / 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate/up/down_proj |
| Hardware | Kaggle NVIDIA Tesla T4 — 16 GB VRAM |
| Quantization | NF4 4-bit (training + inference) |
| Mixed Precision | bfloat16 |

**GRPO vs prior DPO baseline (Qwen2.5-0.5B):**

| Factor | DPO 0.5B | GRPO 4B |
|---|---|---|
| Benchmark Score | 90/100 | 90/100 |
| Response Format | Free-form | Structured (SEVERITY/ACTION/REASONING) |
| Reasoning Depth | Shallow | Deep causal |
| Inference Latency | ~0.3s | ~5.3s |

---

## 3. Merge

LoRA adapter merged into base weights using PEFT `merge_and_unload()`.
Saved as **5 × 2 GB safetensor shards** to fit 6 GB RAM constraint.

```
models/merged_grpo_final/
├── config.json / tokenizer.json / tokenizer_config.json
├── model.safetensors.index.json
└── model-0000{1..5}-of-00005.safetensors   (~2 GB each, ~10 GB total)
```

---

## 4. Benchmark Results

### Composite Score
```
Survival Rate Score:       35.00 / 35   ✅ 100%
ICU Utilisation Score:     25.00 / 25   ✅ 100%
Violation Detection Score: 30.00 / 30   ✅ 100%
──────────────────────────────────────────────
COMPOSITE SCORE:           90.00 / 100  ✅  Grade A
```

### Per-Scenario (15 total episodes)

| Scenario | Survival | Reward | Violations |
|---|---|---|---|
| 🚨 Mass Casualty | 100% | 10.0 | 100% |
| 🦠 Disease Outbreak | 100% | 10.0 | 100% |
| ⚡ Equipment Failure | 100% | 10.0 | 100% |
| 👩‍⚕️ Staff Shortage | 100% | 10.0 | 100% |
| 🔥 Combined Surge | 100% | 10.0 | 100% |

### Per-Agent

| Agent | Correct Rate | Mean Latency |
|---|---|---|
| 🚑 ER Triage | 100% (15/15 per ep) | ~0.08 ms |
| 💻 IT Systems | 100% (3–5/ep) | ~0.02 ms |
| 👩‍⚕️ HR Rostering | 100% (1–3/ep) | ~0.015 ms |
| 💊 Pharmacy | 100% when active | ~0.004 ms |
| 🏥 ICU Management | 100% when active | ~0.009 ms |
| 🎯 CMO Oversight | Escalation-driven | ~0.004 ms |

---

## 5. Inference Validation

**Test prompt:** 12 critical patients, ICU at 87%, 0 ventilators, mass casualty event.

**Model output (merged_grpo_final, 4-bit GPU):**
```
SEVERITY: CRITICAL
ACTION: EVACUATE
REASONING: With 12 critical patients and only 9 ICU beds available, immediate
evacuation is necessary to prevent mass mortality. The lack of free ventilators
and minimal staff capacity require rapid transport of the most unstable patients
to external facilities or emergency rooms.
```

Generation time: **5.32 seconds** | Format: ✅ | Clinical decision: ✅ Correct

---

## 6. Configuration (current .env)

```env
MODEL_NAME=./models/merged_grpo_final
MODEL_BACKEND=transformers
MODEL_DTYPE=bfloat16
MODEL_LOAD_IN_4BIT=true
```

---

## 7. Deliverables

| Item | Path | Status |
|---|---|---|
| GRPO Training Notebook | `notebooks/TRIAGE_GRPO_Training.ipynb` | ✅ |
| GRPO Training Script | `scripts/train_grpo.py` | ✅ |
| LoRA Merge Script | `scripts/merge_grpo_lora.py` | ✅ |
| Merged Model | `models/merged_grpo_final/` | ✅ |
| Benchmark Suite | `scripts/benchmark_agent.py` | ✅ |
| Benchmark Results (raw) | `results/bench_grpo_merged.json` | ✅ |
| Inference Test Log | `results/grpo_inference_test.log` | ✅ |
| HuggingFace Blog Draft | `HUGGINGFACE_BLOG_DRAFT.md` | ✅ |
| Training Graphs | `results/graphs/` (8 plots) | ✅ |
| Env Config | `.env` | ✅ Updated |
| This Report | `results/FINAL_REPORT.md` | ✅ |

---

## 8. Next Steps

1. **Push to HuggingFace Hub** — `balarajr/triage-qwen2.5-7b-grpo`
2. **Update HuggingFace Space** — refresh `triage-multi-agent-system` with latest model
3. **Publish blog** — post `HUGGINGFACE_BLOG_DRAFT.md` to HuggingFace community
4. **Record video demo** — <2 min walkthrough for judges

---

*TRIAGE — Meta PyTorch OpenEnv Hackathon — April 2026*
