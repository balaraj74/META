# 🏥 TRIAGE — Multi-Agent Hospital Crisis Simulation

> **Meta PyTorch OpenEnv Hackathon Entry**
> A production-grade, OpenEnv-compatible multi-agent AI system that simulates real-time hospital crisis management with GRPO fine-tuning, 9 reward verifiers, safety middleware, and a live command center dashboard.

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 🤗 **HuggingFace Space (Live Demo)** | [balarajr/triage-multi-agent-system](https://huggingface.co/spaces/balarajr/triage-multi-agent-system) |
| 🤗 **Fine-Tuned Model** | [balarajr/triage-qwen2.5-7b-grpo](https://huggingface.co/balarajr/triage-qwen2.5-7b-grpo) |
| 📓 **Training Notebook** | [`notebooks/TRIAGE_GRPO_Training.ipynb`](./triage-backend/notebooks/TRIAGE_GRPO_Training.ipynb) |
| 🔥 **Kaggle Training (Live)** | [balarajr/notebook583d8fffed](https://www.kaggle.com/code/balarajr/notebook583d8fffed) |
| 📝 **Blog / Writeup** | [HUGGINGFACE_BLOG_DRAFT.md](./HUGGINGFACE_BLOG_DRAFT.md) |
| 📊 **Benchmark Report** | [results/FINAL_REPORT.md](./triage-backend/results/FINAL_REPORT.md) |
| 📈 **Training Graphs** | [results/graphs/](./triage-backend/results/graphs/) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Architecture](#architecture)
4. [Backend — triage-backend](#backend)
   - [Environment](#environment)
   - [Agents](#agents)
   - [Reward Model](#reward-model)
   - [Safety Constitution](#safety-constitution)
   - [Curriculum Learning](#curriculum-learning)
   - [Training Pipeline](#training-pipeline)
   - [API Reference](#api-reference)
5. [Frontend — triage-command-center](#frontend)
6. [AI Training System](#ai-training-system)
7. [Quick Start](#quick-start)
8. [Configuration](#configuration)
9. [Scripts Reference](#scripts-reference)
10. [Deployment](#deployment)

---

## Project Overview

TRIAGE is a **multi-agent reinforcement learning environment** that simulates a hospital under crisis — mass casualty events, disease outbreaks, equipment failures, staff shortages, and combined surges. Six specialized AI agents collaborate to manage patients, allocate resources, and maintain policy compliance, all orchestrated through a typed message bus.

The system is designed for the **Meta PyTorch OpenEnv Hackathon**, implementing the OpenEnv-compatible interface (`reset`, `step`, `state`) alongside a **GRPO training pipeline** with **9 reward verifiers** and a **14-source dataset** to teach the agents safe, clinically aligned decision-making.

### Key Highlights

| Feature | Detail |
|---|---|
| **Framework** | OpenEnv + PyTorch + HuggingFace Transformers |
| **Base Model** | `Qwen/Qwen2.5-7B` (NF4 4-bit quantized) |
| **Training Method** | GRPO (primary) via TRL + 9 reward verifiers |
| **Training Data** | 14 sources (7 HuggingFace + 6 Kaggle + 1 base) |
| **Agents** | 6 specialized agents (CMO, ER, ICU, Pharmacy, HR, IT) |
| **Crisis Types** | Mass Casualty, Disease Outbreak, Equipment Failure, Staff Shortage, Combined Surge |
| **Safety Layer** | SafetyConstitution — 10 hard-block rules, auto-fallback actions |
| **Reward System** | 9 independent GRPO verifiers + rule-based reward model |
| **Curriculum** | 5-tier auto-advancing difficulty scheduler |
| **API** | FastAPI + WebSocket for real-time streaming |
| **Dashboard** | React/TypeScript live command center |
| **Live Demo** | [HuggingFace Space](https://huggingface.co/spaces/balarajr/triage-multi-agent-system) |

---

## Key Results

### 🏆 Composite Benchmark Score: **90.00 / 100 — Grade A**

```
Survival Rate        (×40)  :  40.00 /  40.00   ✅ 100%
Reward Score         (×30)  :  30.00 /  30.00   ✅ 100%
Agent Correct-Action (×20)  :  10.00 /  20.00
Violation Detection  (×10)  :  10.00 /  10.00   ✅ 100%
──────────────────────────────────────────────────────
TOTAL SCORE                 :  90.00 / 100.00   [A]
```

### Per-Scenario Performance (15 episodes total)

| Scenario | Survival | Reward | Violations Caught |
|---|---|---|---|
| 🚨 Mass Casualty | **100%** | 10.0 / 10 | **100%** |
| 🦠 Disease Outbreak | **100%** | 10.0 / 10 | **100%** |
| ⚡ Equipment Failure | **100%** | 10.0 / 10 | **100%** |
| 👩‍⚕️ Staff Shortage | **100%** | 10.0 / 10 | **100%** |
| 🔥 Combined Surge | **100%** | 10.0 / 10 | **100%** |

### Training Visualizations

Training evidence and result graphs are located in [`results/graphs/`](./triage-backend/results/graphs/):

| Graph | Description |
|---|---|
| `01_reward_curve.png` | GRPO reward progression over training steps |
| `02_training_dashboard.png` | Multi-panel training metrics dashboard |
| `03_scenario_performance.png` | Per-scenario performance breakdown |
| `04_radar_chart.png` | Multi-dimensional agent capability radar |
| `05_episode_rewards.png` | Episode-level reward distribution |
| `06_agent_accuracy.png` | Per-agent action accuracy over time |
| `07_radar_chart.png` | Verifier-level performance radar |
| `08_loss_reward.png` | Loss vs reward correlation during training |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRIAGE System Architecture                   │
└─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐        ┌──────────────────────────────┐
  │  triage-frontend    │◄──────►│  triage-backend (FastAPI)    │
  │  React + TypeScript │  HTTP/ │  Port 8000                   │
  │  Port 8081          │  WS    │                              │
  └─────────────────────┘        │  ┌─────────────────────────┐ │
                                 │  │  AgentOrchestrator       │ │
                                 │  │  ┌──────────────────┐   │ │
                                 │  │  │ 🎯 CMO Oversight │   │ │
                                 │  │  │ 🚑 ER Triage     │   │ │
                                 │  │  │ 🏥 ICU Management│   │ │
                                 │  │  │ 💊 Pharmacy      │   │ │
                                 │  │  │ 👩‍⚕️ HR Rostering  │   │ │
                                 │  │  │ 💻 IT Systems    │   │ │
                                 │  │  └──────────────────┘   │ │
                                 │  │        │ MessageBus       │ │
                                 │  └────────┼────────────────┘ │
                                 │           ▼                   │
                                 │  ┌─────────────────────────┐ │
                                 │  │  SafetyConstitution     │ │
                                 │  │  hard-block + fallback  │ │
                                 │  └────────┼────────────────┘ │
                                 │           ▼                   │
                                 │  ┌─────────────────────────┐ │
                                 │  │  HospitalEnv (OpenEnv)  │ │
                                 │  │  reset / step / state   │ │
                                 │  └─────────────────────────┘ │
                                 │           │                   │
                                 │  ┌────────┼────────────────┐ │
                                 │  │  RewardModel            │ │
                                 │  │  9 GRPO Verifiers        │ │
                                 │  │  GRPO Pipeline           │ │
                                 │  └─────────────────────────┘ │
                                 └──────────────────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              │    SQLite DB  │  data/ files  │
                              │    triage.db  │  *.jsonl      │
                              └───────────────┴───────────────┘
```

---

## Backend

### Directory Structure

```
triage-backend/
├── triage/
│   ├── agents/
│   │   ├── base_agent.py        # Abstract base for all agents
│   │   ├── specialized.py       # 6 concrete agent implementations
│   │   ├── orchestrator.py      # Runs all agents per step
│   │   ├── message_bus.py       # Typed pub/sub message router
│   │   ├── routing_rules.py     # Priority queue + deadlock detection
│   │   ├── tools.py             # ToolSchema models for all 20 actions
│   │   ├── tool_validator.py    # Validates tool args against EnvironmentState
│   │   └── strategy_memory.py   # Agent lesson memory (BM25)
│   ├── env/
│   │   ├── hospital_env.py      # OpenEnv HospitalEnv class
│   │   ├── state.py             # All enums, dataclasses (world model)
│   │   ├── crisis_generator.py  # Procedural crisis + patient generation
│   │   ├── enterprise_apps/     # ICU manager, Pharmacy simulators
│   │   ├── curriculum.py        # CurriculumScheduler — 5 difficulty tiers
│   │   ├── grpo_env_adapter.py  # TRL environment_factory adapter
│   │   └── openenv_adapter.py   # OpenEnv SDK interface adapter
│   ├── safety/
│   │   └── constitution.py      # SafetyConstitution — 10 hard-block rules
│   ├── rewards/
│   │   ├── reward_model.py      # Multi-component reward model
│   │   ├── verifiers.py         # 9 independent GRPO verifiers
│   │   └── reward_logger.py     # Per-verifier logging
│   ├── training/
│   │   ├── dpo_trainer.py       # DPO training loop (TRL + PEFT)
│   │   ├── episode_collector.py # Collects episodes for training data
│   │   └── dataset_adapter.py   # Adapts pairs to HuggingFace Dataset
│   ├── api/
│   │   ├── main.py              # FastAPI app (CORS, lifespan, routes)
│   │   ├── service.py           # Backend service singleton
│   │   └── routers/             # REST + WebSocket endpoints
│   └── db/                      # SQLAlchemy models + session management
├── scripts/                     # Standalone utility scripts
├── notebooks/                   # Jupyter / Colab training notebooks
│   ├── TRIAGE_GRPO_Training.ipynb  # ⭐ Primary GRPO training notebook
│   └── generate_graphs.py         # Visualization suite
├── models/
│   ├── grpo_output/             # LoRA adapters from GRPO training
│   └── merged_grpo_final/       # Merged model (5 × 2GB shards)
├── results/
│   ├── FINAL_REPORT.md          # Benchmark report
│   ├── bench_grpo_merged.json   # Raw benchmark data
│   └── graphs/                  # Training visualizations (8 plots)
├── spaces/                      # HuggingFace Spaces Gradio app
├── data/                        # Datasets and training outputs
├── config/                      # YAML configuration files
├── tests/                       # pytest test suite
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

### Environment

**File:** `triage/env/hospital_env.py`

The `HospitalEnv` is the core simulation environment. It implements both the **OpenEnv async protocol** and a **Gymnasium-style shim** for numpy-based RL training.

#### Episode Lifecycle

```
reset(scenario?) ──► generates crisis, patients, resources ──► EnvironmentState
     │
     ▼
step(action) ──► applies AgentAction ──► advances world clock ──► (reward, done)
     │
     ▼  (repeat until terminated)
     │
state() ──► returns current EnvironmentState snapshot
```

#### Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `patients` | `(50, 12)` | Patient status vectors (vitals, acuity, location) |
| `resources` | `(8,)` | ICU beds, ventilators, blood supply, staff count |
| `agent_states` | `(6, 8)` | Each agent's action/message counters |
| `crisis_state` | `(10,)` | Crisis type, severity, time elapsed |
| `policy_state` | `(20,)` | Active policy flags |
| `expert_signals` | `(6,)` | Per-agent performance signals |

#### Action Space (Discrete, 20 types)

| ID | Action | Agent |
|----|--------|-------|
| 0 | `TRIAGE_PATIENT` | ER |
| 1 | `TRANSFER_TO_ICU` | ER / CMO |
| 2 | `TRANSFER_TO_WARD` | ER / ICU |
| 3 | `ASSIGN_TREATMENT` | Any |
| 4 | `ORDER_MEDICATION` | Pharmacy |
| 5 | `REQUEST_BLOOD` | ER / ICU |
| 6 | `ACTIVATE_PROTOCOL` | CMO |
| 7 | `REQUEST_STAFF` | HR |
| 8 | `ESCALATE_TO_CMO` | Any |
| 9 | `DISCHARGE_PATIENT` | ICU / Ward |
| 10 | `FLAG_POLICY_VIOLATION` | IT |
| 11 | `UPDATE_EHR` | IT |
| 12 | `VERIFY_INSURANCE` | IT |
| 13 | `ALLOCATE_EQUIPMENT` | ICU |
| 14 | `SEND_MESSAGE` | Any |
| 15 | `OVERRIDE_DECISION` | CMO |
| 16 | `REQUEST_SPECIALIST` | Any |
| 17 | `ACTIVATE_OVERFLOW` | CMO |
| 18 | `UPDATE_TREATMENT_PLAN` | Any |
| 19 | `CLOSE_CASE` | Any |

#### Crisis Types

| Type | Description |
|------|-------------|
| `MASS_CASUALTY` | 20–40 trauma patients with blast/crush injuries |
| `OUTBREAK` | Infectious disease spread through wards |
| `EQUIPMENT_FAILURE` | Critical machinery (ventilators, EHR) goes offline |
| `STAFF_SHORTAGE` | Emergency staff walkout with skeleton crew |
| `COMBINED_SURGE` | Multiple crises simultaneously |

---

### Agents

**File:** `triage/agents/specialized.py`

Six agents, each inheriting from `BaseAgent`, collaborate through the `MessageBus`.

#### Agent Hierarchy

```
🎯 CMO_OVERSIGHT  ──── supervisor, handles escalations and overrides
     │
     ├── 🚑 ER_TRIAGE          ── intake, initial patient classification
     ├── 🏥 ICU_MANAGEMENT     ── bed allocation, ventilator management
     ├── 💊 PHARMACY           ── medication dispensing, drug interaction checks
     ├── 👩‍⚕️ HR_ROSTERING       ── staff scheduling, fatigue monitoring
     └── 💻 IT_SYSTEMS         ── EHR integrity, policy compliance
```

#### BaseAgent Contract

Every agent implements:

```python
async def decide(
    state: EnvironmentState,
    inbox: list[AgentMessage],
) -> list[AgentAction]:
    ...
```

Agents support **two execution modes**:
- **LLM-backed mode** — calls the fine-tuned Qwen2.5-7B model with a structured system prompt
- **Rule-based mock mode** — deterministic fallback that runs without any GPU or internet connection

---

### Reward Model

**File:** `triage/rewards/reward_model.py`

The reward model produces a scalar reward from a multi-component evaluation:

| Component | Weight | Description |
|-----------|--------|-------------|
| Patient Outcomes | 40% | Survival rate, recovery velocity |
| Resource Efficiency | 20% | ICU/bed utilization vs. waste |
| Response Time | 15% | Critical patient time-to-treatment |
| Policy Compliance | 15% | Violations flagged and resolved |
| Agent Coordination | 10% | Message effectiveness, escalation quality |

---

## Safety Constitution

**File:** `triage/safety/constitution.py`

A middleware layer that wraps every agent's output. Hard-blocks 10 categories of unsafe decisions and replaces them with safe fallbacks automatically.

| Rule | Violation | Severity |
|---|---|---|
| Critical Patient Discharge | Discharging acuity >= 7 patient | 9 |
| Drug Interaction | Contraindicated medication order | 8 |
| Zero ICU Staff | Staff reduction leaving ICU unstaffed | 10 |
| Ventilator Over-Allocation | Allocating ventilator when none available | 9 |
| Blood Type Mismatch | Incompatible blood type transfusion | 10 |
| Unauthorized CMO Override | Non-CMO agent issuing override | 7 |
| Treatment Without Triage | Treatment before triage assessment | 6 |
| ICU Transfer No Bed | ICU transfer when beds = 0 | 8 |
| Medication Without Diagnosis | Prescribing with no diagnosis recorded | 7 |
| Duplicate Critical Action | Same action on same patient twice in one step | 5 |

---

## Curriculum Learning

**File:** `triage/env/curriculum.py`

Auto-advancing 5-tier difficulty system:

| Tier | Crisis Types | Patients | Compound Events |
|---|---|---|---|
| 1 | Mass Casualty only | 10-15 | No |
| 2 | Mass Casualty + Staff Shortage | 20-25 | No |
| 3 | Three crisis types | 30-35 | Yes |
| 4 | All four crisis types | 40-45 | Yes + Schema Drift |
| 5 | All types simultaneous | 48-50 | Maximum stress |

---

### Training Pipeline

The training system uses **GRPO (Group Relative Policy Optimization)** as the primary method.

#### GRPO Architecture

```
14-Source Dataset Pipeline
│
▼
HospitalEnv Header Injection (randomized crisis context)
│
▼
GRPOTrainer (TRL) + Qwen2.5-7B (NF4 4-bit)
├── num_generations=4
├── max_seq_length=512
├── LoRA r=16, alpha=32
└── 9 independent reward verifiers:
    ├── format_compliance     — valid JSON with required keys
    ├── patient_survival      — survival rate from crisis context
    ├── icu_efficiency        — ICU occupancy management
    ├── violation_detection   — violations caught vs. injected
    ├── reasoning_quality     — evidence citation depth
    ├── response_speed        — concise output preference
    ├── no_hallucination      — no fabricated patient IDs
    ├── action_alignment      — action matches crisis state
    └── sandbox_safety        — no code injection attempts
│
▼
LoRA Adapter → merge_and_unload() → models/merged_grpo_final/
```

#### GRPO Configuration

| Parameter | Value | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-7B` | 4-bit NF4 quantized |
| `load_in_4bit` | `True` | NF4 via bitsandbytes |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | 2× lora_r |
| `num_generations` | `4` | GRPO group size |
| `max_seq_length` | `512` | Per-sample context |
| `per_device_batch` | `1` | Micro-batch |
| `gradient_accum` | `4` | Effective batch = 4 |
| `learning_rate` | `5e-5` | Cosine scheduler |
| `num_epochs` | `1` | Single pass |
| `mixed_precision` | `bfloat16` | When supported |
| `hardware` | `Kaggle T4 / P100` | 16 GB VRAM (T4 free tier) |

---

### Dataset Sources (14 Total)

| # | Dataset | Source | Samples | Purpose |
|---|---------|--------|---------|---------|
| 0 | `balarajr/triage-grpo` | HuggingFace | Base | Core triage prompts |
| 1 | `FreedomIntelligence/medical-o1-reasoning-SFT` | HuggingFace | 500 | Clinical reasoning |
| 2 | `bigbio/med_qa` | HuggingFace | 500 | Medical Q&A |
| 3 | `sdiazlor/medical-reasoning-dataset` | HuggingFace | 500 | Medical reasoning |
| 4 | `Anthropic/hh-rlhf` | HuggingFace | 300 | Safety alignment |
| 5 | `PKU-Alignment/PKU-SafeRLHF` | HuggingFace | 300 | Safe RL alignment |
| 6 | `lavita/ChatDoctor-iCliniq` | HuggingFace | 500 | Clinical consultation |
| 7 | `medalpaca/medical_meadow_medical_flashcards` | HuggingFace | 500 | Medical knowledge |
| 8 | `thedevastator/medical-q-a-structured` | Kaggle | 500 | Structured medical Q&A |
| 9 | `nehaprabhavalkar/av-healthcare-analytics-ii` | Kaggle | 500 | Healthcare analytics |
| 10 | `jpmiller/layoutlm` | Kaggle | 300 | NLP medical records |
| 11 | `thedevastator/usmle-medical-licensing-examination` | Kaggle | 500 | USMLE questions |
| 12 | `kaushil268/disease-prediction-using-machine-learning` | Kaggle | 500 | Disease prediction |
| 13 | `maalona/hospital-triage-and-patient-history-data` | Kaggle | 500 | Hospital triage history |

Every prompt is injected with a **randomized hospital environment header** (crisis type, step count, ICU occupancy, critical patient count, violation state) to ensure context-dependent decision-making.

---

### API Reference

**Base URL:** `http://localhost:8000`

#### Episodes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/episodes/run` | Start a new simulation episode |
| `GET` | `/api/episodes/` | List all completed episodes |
| `GET` | `/api/episodes/{id}` | Get episode details |
| `GET` | `/api/episodes/{id}/replay` | Full step-by-step replay data |

#### Agents

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/agents/` | List all 6 agents and their states |
| `POST` | `/api/agents/{type}/override` | Manually override an agent decision |

#### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/ws/live` | Real-time simulation state stream |

---

## Frontend

**Location:** `triage-frontend/triage-command-center-main/`
**Tech Stack:** React 18 + TypeScript + Vite + TanStack Router + shadcn/ui

### Dashboard Features

- **Live patient board** — real-time patient status, acuity scores, ward locations
- **Agent activity feed** — live message bus stream showing inter-agent communications
- **Resource gauges** — ICU beds, ventilators, blood supply, staff hours
- **Reward curve chart** — total episode reward over training time
- **Crisis alert banner** — crisis type, severity, time elapsed
- **Training monitor** — GPU utilization, VRAM, loss curve, verifier rewards

---

## AI Training System

### Training Notebook (Colab / Kaggle)

The primary training artifact is **[`notebooks/TRIAGE_GRPO_Training.ipynb`](./triage-backend/notebooks/TRIAGE_GRPO_Training.ipynb)** — a self-contained notebook that:

1. Installs all dependencies (`transformers`, `trl`, `peft`, `bitsandbytes`, `kagglehub`)
2. Defines 9 reward verifier functions
3. Loads Qwen2.5-7B with NF4 4-bit quantization + LoRA
4. Builds a unified dataset from 14 sources (7 HF + 6 Kaggle + 1 base)
5. Runs GRPO training via TRL's `GRPOTrainer`
6. Saves LoRA adapters for merging

### Post-Training Merge

```bash
python scripts/merge_grpo_lora.py
# → models/merged_grpo_final/ (5 × 2GB safetensor shards, ~10GB total)
```

### Local Benchmark

```bash
cd triage-backend
source .venv/bin/activate
python scripts/benchmark_agent.py
# → Composite Score: 90.00 / 100 — Grade A
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- NVIDIA GPU (optional, auto-detected)
- 8 GB RAM minimum (16 GB recommended)

### 1. Clone and Start

```bash
git clone <repo-url>
cd "META final"
chmod +x start.sh
./start.sh
```

### 2. Backend Only

```bash
cd triage-backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn triage.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Only

```bash
cd triage-frontend/triage-command-center-main
npm install
npm run dev -- --port 8081
```

### 4. Run a Simulation

```bash
python scripts/run_episode.py
# or via API:
curl -X POST http://localhost:8000/api/episodes/run \
  -H "Content-Type: application/json" \
  -d '{"crisis_type": "mass_casualty", "max_steps": 20}'
```

---

## Configuration

### Environment Variables (`.env`)

```env
PORT=8000
DEBUG=true
MODEL_NAME=./models/merged_grpo_final
MODEL_BACKEND=transformers
MODEL_DTYPE=bfloat16
MODEL_LOAD_IN_4BIT=true
USE_MOCK_LLM=true
CONSTITUTION_ACTIVE=true
DATABASE_URL=sqlite+aiosqlite:///./triage.db
```

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/benchmark_agent.py` | Run full benchmark suite (5 scenarios × 3 episodes) |
| `scripts/merge_grpo_lora.py` | Merge LoRA adapters into base model |
| `scripts/run_episode.py` | Run a single simulation episode |
| `scripts/build_grpo_dataset.py` | Generate crisis scenario prompts |
| `scripts/train_grpo.py` | GRPO training script |
| `scripts/demo_before_after.py` | Baseline vs fine-tuned comparison |

---

## Deployment

### Docker

```bash
cd triage-backend
docker-compose up --build
```

### HuggingFace Spaces

The Gradio demo at [`spaces/`](./triage-backend/spaces/) is deployed to:
**https://huggingface.co/spaces/balarajr/triage-multi-agent-system**

---

## Tests

```bash
cd triage-backend
source .venv/bin/activate
pytest tests/ -v
```

---

## Reproducibility

| Component | Technology |
|---|---|
| Environment | OpenEnv-compatible `HospitalCrisisEnv` |
| Agents | Python 3.11 + PEFT + Transformers |
| GRPO Training | HuggingFace TRL `GRPOTrainer` + LoRA |
| Model | Qwen2.5-7B (NF4 4-bit) → merged safetensors |
| API | FastAPI + WebSocket |
| Frontend | React 18 + TypeScript + Vite |
| Demo | Gradio 4.x on HuggingFace Spaces |
| Database | SQLite + SQLAlchemy |

---

*TRIAGE — Meta PyTorch OpenEnv Hackathon — April 2026*
*Solo developer, one laptop, one GPU. [Balaraj R](https://huggingface.co/balarajr)*
