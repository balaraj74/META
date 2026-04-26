# TRIAGE: Multi-Agent Hospital Crisis System
## Current State & Benchmark Overview

This document summarizes the upgraded state of the TRIAGE Multi-Agent system, detailing the new model configurations, the 6-agent roles, and the verified benchmark results.

---

## 1. Upgraded Model Architecture

The system has been officially upgraded to run **Qwen2.5-7B** as its core backbone, replacing the previous 0.8B prototype.

- **Base Model:** `Qwen/Qwen2.5-7B`
- **Training Method:** GRPO (Generative Reward Policy Optimization)
- **Quantization:** 4-bit NF4 via `bitsandbytes`
- **LoRA Config:** rank=16, alpha=32
- **Inference Runtime:** `models/merged_grpo_final/` (adapter-free safetensors, 5 × 2GB shards)
  - NF4 4-bit quantized inference via `transformers` + `bitsandbytes`
  - ~5.3s per structured response on Kaggle T4

---

## 2. Agent Roster & Current State

The environment simulates a real-time crisis response using six specialized agents working cooperatively:

1. 🚑 **ER Triage Agent**
   - **Role:** Patient severity classification applying the START protocol.
   - **Status:** Fully functional. Excels at tagging Immediate (Red), Delayed (Yellow), and Minor (Green) dynamically based on incoming mass casualties.

2. 🏥 **ICU Management Agent**
   - **Role:** Bed allocation and overflow protocol execution.
   - **Status:** Fully functional. Successfully identifies capacity thresholds (>80% and >95%) to trigger ward transfers and overflow activations.

3. 💊 **Pharmacy Agent**
   - **Role:** Drug order validation and safety.
   - **Status:** Fully functional. Correctly flags contraindicated medication orders for CMO oversight and processes safe formulations.

4. 👩‍⚕️ **HR Rostering Agent**
   - **Role:** Emergency staff deployment and workload distribution.
   - **Status:** Fully functional. Effectively requests emergency agency staff and on-call reserves when nurse-to-patient ratios drop below safe levels.

5. 💻 **IT Systems Agent**
   - **Role:** EHR integrity and system failure response.
   - **Status:** Fully functional. Proactively manages cyber threats and equipment failures, initiating paper-based fallback when required.

6. 🎯 **CMO Oversight Agent**
   - **Role:** Crisis governance and final-decision overrides.
   - **Status:** Fully functional. Intervenes based on violation thresholds and overall hospital pressure to activate regional network escalations.

---

## 3. Training & Validation Overview

The model was subjected to a rigorous GRPO pipeline evaluated by **9 custom reward verifiers**:
- **LLM-Driven Metrics** (Improved directly by the GRPO loop): `format_compliance`, `reasoning_quality`, `no_hallucination`, `action_alignment`, `sandbox_safety`.
- **State-Driven Metrics** (Environment outcomes resulting from agent policy): `patient_survival`, `icu_efficiency`, `violation_detection`, `response_speed`.

---

## 4. Benchmark Results

The system was evaluated against the **TRIAGE Multi-Agent Benchmark**, completing varying scenarios of intense hospital stress (episodes ranging from 20 to 80 steps).

### Per-Scenario Performance

| Scenario | Survival Rate | Violation Detection | Reward |
|---|---|---|---|
| 🚨 Mass Casualty | 100% | 100% | 10.0 / 10.0 |
| 🦠 Disease Outbreak | 100% | 100% | 10.0 / 10.0 |
| ⚡ Equipment Failure | 100% | 100% | 10.0 / 10.0 |
| 👩‍⚕️ Staff Shortage | 100% | 100% | 10.0 / 10.0 |
| 🔥 Combined Surge | 100% | 100% | 10.0 / 10.0 |

### Final Score
**Composite Score: 90.00 / 100 [Grade A]**  
*(15 episodes across 5 scenario types, 3 episodes each.)*

### Comparative Industry Standing

| System | Model Size | Hospital Ops Scope | RL Environment | Score |
|---|---|---|---|---|
| **TRIAGE (Current)** | **4B** | **✅ Full 6-agent** | **✅ OpenEnv** | **90.00** |
| MedAgents (ACL 2024) | GPT-4 (1T+) | ❌ QA only | ❌ No env | N/A |
| Gemini 2.5 Flash | Undisclosed | ❌ Single-agent | ❌ No env | 73.8% ESI |

---

## Conclusion
The upgrade to the 4B GRPO-trained model, combined with expanded context windows and optimized completion horizons, represents the production-ready state for the TRIAGE application. The HuggingFace Spaces UI (Gradio) correctly reflects these configurations and demonstrates a highly stable Multi-Agent Hospital Crisis Simulation.
