# 🏥 TRIAGE — Multi-Agent Hospital Crisis Simulation

> **Meta PyTorch OpenEnv Hackathon Entry**
> A production-grade, OpenEnv-compatible multi-agent AI system that simulates real-time hospital crisis management with Direct Preference Optimization (DPO) fine-tuning and a live command center dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend — triage-backend](#backend)
   - [Environment](#environment)
   - [Agents](#agents)
   - [Reward Model](#reward-model)
   - [Training Pipeline](#training-pipeline)
   - [API Reference](#api-reference)
4. [Frontend — triage-command-center](#frontend)
5. [AI Training System](#ai-training-system)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [Scripts Reference](#scripts-reference)
9. [Data Directory](#data-directory)
10. [Deployment](#deployment)

---

## Project Overview

TRIAGE is a **multi-agent reinforcement learning environment** that simulates a hospital under crisis — mass casualty events, disease outbreaks, equipment failures, and staff shortages. Six specialized AI agents collaborate to manage patients, allocate resources, and maintain policy compliance, all orchestrated through a typed message bus.

The system is designed for the **Meta PyTorch OpenEnv Hackathon**, implementing the OpenEnv-compatible interface (`reset`, `step`, `state`) alongside a complete DPO fine-tuning pipeline to teach the agents safe, clinically aligned decision-making.

### Key Highlights

| Feature | Detail |
|---|---|
| **Framework** | OpenEnv + PyTorch + HuggingFace Transformers |
| **Base Model** | Qwen/Qwen2.5-0.5B → upgradeable to 1.5B on GPU |
| **Training Method** | Direct Preference Optimization (DPO) with LoRA |
| **Agents** | 6 specialized agents (CMO, ER, ICU, Pharmacy, HR, IT) |
| **Crisis Types** | Mass Casualty, Disease Outbreak, Equipment Failure, Staff Shortage |
| **API** | FastAPI + WebSocket for real-time streaming |
| **Dashboard** | React/TypeScript live command center |
| **Database** | SQLite (dev) via SQLAlchemy + Alembic |

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
                                 │  │  │   CMO Oversight  │   │ │
                                 │  │  │   ER Triage      │   │ │
                                 │  │  │   ICU Management │   │ │
                                 │  │  │   Pharmacy       │   │ │
                                 │  │  │   HR Rostering   │   │ │
                                 │  │  │   IT Systems     │   │ │
                                 │  │  └──────────────────┘   │ │
                                 │  │        │ MessageBus       │ │
                                 │  └────────┼────────────────┘ │
                                 │           ▼                   │
                                 │  ┌─────────────────────────┐ │
                                 │  │  HospitalEnv (OpenEnv)  │ │
                                 │  │  reset / step / state   │ │
                                 │  └─────────────────────────┘ │
                                 │           │                   │
                                 │  ┌────────┼────────────────┐ │
                                 │  │  RewardModel            │ │
                                 │  │  EpisodeCollector        │ │
                                 │  │  PreferenceLabeler       │ │
                                 │  │  DPO Training Pipeline   │ │
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
│   │   └── strategy_memory.py  # Agent lesson memory (BM25)
│   ├── env/
│   │   ├── hospital_env.py      # OpenEnv HospitalEnv class
│   │   ├── state.py             # All enums, dataclasses (world model)
│   │   ├── crisis_generator.py  # Procedural crisis + patient generation
│   │   ├── enterprise_apps/     # ICU manager, Pharmacy simulators
│   │   ├── enterprise_registry.py
│   │   ├── openenv_adapter.py   # OpenEnv SDK interface adapter
│   │   └── schema_drift.py      # Simulated EHR schema drift (IT agent)
│   ├── reward/
│   │   └── evaluator.py         # Episode reward evaluation wrapper
│   ├── rewards/
│   │   └── reward_model.py      # Multi-component reward model
│   ├── training/
│   │   ├── dpo_trainer.py       # DPO training loop (TRL + PEFT)
│   │   ├── episode_collector.py # Collects episodes for training data
│   │   ├── preference_labeler.py# Converts episodes to chosen/rejected pairs
│   │   ├── dataset_adapter.py   # Adapts pairs to HuggingFace Dataset
│   │   ├── trajectory_collector.py
│   │   └── reporting.py         # Generates training reports
│   ├── api/
│   │   ├── main.py              # FastAPI app (CORS, lifespan, routes)
│   │   ├── service.py           # Backend service singleton (business logic)
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routers/
│   │       ├── agents.py        # /api/agents endpoints
│   │       ├── episodes.py      # /api/episodes endpoints
│   │       ├── metrics.py       # /api/metrics endpoints
│   │       ├── patients.py      # /api/patients endpoints
│   │       ├── training.py      # /api/training endpoints
│   │       └── websocket.py     # WebSocket /ws/live endpoint
│   ├── db/                      # SQLAlchemy models + session management
│   └── worker.py                # Celery background task worker
├── scripts/                     # Standalone utility scripts (see below)
├── notebooks/                   # Jupyter / Colab training notebooks
├── config/                      # YAML configuration files
├── data/                        # Dataset files and training outputs
├── models/                      # Saved model checkpoints
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
| `resources` | `(8,)` | ICU beds, ventilators, blood, staff count, etc. |
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

---

### Agents

**File:** `triage/agents/specialized.py`

Six agents, each inheriting from `BaseAgent`, collaborate through the `MessageBus`.

#### Agent Hierarchy

```
CMO_OVERSIGHT  ──── supervisor, handles all escalations
     │
     ├── ER_TRIAGE       ── intake, initial patient classification
     ├── ICU_MANAGEMENT  ── bed allocation, ventilator management
     ├── PHARMACY        ── medication dispensing, drug interaction checks
     ├── HR_ROSTERING    ── staff scheduling, fatigue monitoring
     └── IT_SYSTEMS      ── EHR integrity, schema drift detection, policy compliance
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
- **LLM-backed mode** — calls a local or remote LLM (Ollama, HuggingFace) with a structured system prompt
- **Rule-based mock mode** — deterministic fallback that runs without any GPU or internet connection

#### MessageBus

`triage/agents/message_bus.py` — A typed pub/sub broker that routes messages between agents:

```python
bus.subscribe(AgentType.CMO_OVERSIGHT, handler)
bus.publish(AgentMessage(
    from_agent=AgentType.ER_TRIAGE,
    to_agent=AgentType.CMO_OVERSIGHT,
    msg_type=MessageType.ALERT,
    priority=8,
    content="Patient deteriorating rapidly — requesting CMO override",
    patient_id="..."
))
```

#### Strategy Memory

`triage/agents/strategy_memory.py` — A BM25-indexed lesson store that retains successful strategies across episodes. Each agent can query past lessons before making decisions, enabling cross-episode learning without retraining.

---

### Reward Model

**File:** `triage/rewards/reward_model.py`

The reward model produces a scalar reward from a multi-component evaluation of the environment state:

| Component | Weight | Description |
|-----------|--------|-------------|
| Patient Outcomes | 40% | Survival rate, recovery velocity |
| Resource Efficiency | 20% | ICU/bed utilization vs. waste |
| Response Time | 15% | Critical patient time-to-treatment |
| Policy Compliance | 15% | Violations flagged and resolved |
| Agent Coordination | 10% | Message effectiveness, escalation quality |

**Episode Score** = weighted sum, clamped to `[-1.0, +1.0]`

---

### Training Pipeline

**Files:** `triage/training/`

The complete DPO fine-tuning pipeline converts simulation episodes into training data for the language model:

```
HospitalEnv Episodes
       │
       ▼
EpisodeCollector ─── runs N episodes, saves trajectory objects
       │
       ▼
PreferenceLabeler ─── compares episode pairs by total reward
                  ─── higher reward episode → "chosen"
                  ─── lower reward episode → "rejected"
       │
       ▼
DatasetAdapter ─── converts to HuggingFace Dataset format
                   {"prompt": ..., "chosen": ..., "rejected": ...}
       │
       ▼
TRIAGEDPOTrainer ─── LoRA + DPO on base model (Qwen2.5)
                  ─── saves adapter to models/dpo_output/
```

#### DPO Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Local (laptop); `1.5B` on Colab |
| `beta` | `0.1` | DPO temperature (lower = closer to reference) |
| `learning_rate` | `5e-5` | AdamW learning rate |
| `epochs` | `1` | Laptop default (3 on Colab) |
| `batch_size` | `1` | Laptop RTX 2050 (4 on Colab T4) |
| `lora_r` | `16` | LoRA rank |
| `max_length` | `512` | Max sequence length |
| `fp16` | `True` | Half-precision training |

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

#### Patients

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/patients/` | List all patients in the active episode |
| `GET` | `/api/patients/{id}` | Get a single patient's full record |

#### Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/metrics/reward-curve` | Episode reward history (all time) |
| `GET` | `/api/metrics/resources` | Resource utilization over last N steps |
| `GET` | `/api/metrics/comparison` | Compare current vs. previous episode |

#### Training

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/training/collect` | Run episodes and collect training data |
| `POST` | `/api/training/label` | Label preferences from collected data |
| `POST` | `/api/training/start-dpo` | Start DPO fine-tuning job |
| `GET` | `/api/training/status` | Live training metrics (loss, progress, GPU) |
| `GET` | `/api/training/memory` | Strategy memory contents |

#### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/ws/live` | Real-time simulation state stream |

**WebSocket Commands:**
```json
{ "command": "start_episode", "params": {"crisis_type": "mass_casualty"} }
{ "command": "step", "params": {} }
{ "command": "get_state", "params": {} }
{ "command": "override_agent", "params": {"agent": "cmo_oversight", "action": {...}} }
```

---

## Frontend

**Location:** `triage-frontend/triage-command-center-main/`
**Tech Stack:** React 18 + TypeScript + Vite + TanStack Router + shadcn/ui

### Pages

| Route | File | Description |
|-------|------|-------------|
| `/` | `index.tsx` | Landing / overview |
| `/dashboard` | `dashboard.tsx` | Live simulation command center |
| `/training` | `training.tsx` | DPO training monitor with live metrics |
| `/visualizer` | `visualizer.tsx` | Step-by-step episode visualizer |
| `/replay` | `replay.tsx` | Historical episode replay |
| `/pitch` | `pitch.tsx` | Hackathon pitch deck |
| `/sponsors` | `sponsors.tsx` | Sponsors page |
| `/mobile` | `mobile.tsx` | Mobile-optimized view |

### Key Hooks

| Hook | File | Description |
|------|------|-------------|
| `useSimulation` | `hooks/useSimulation.ts` | WebSocket connection + episode state |
| `useWebSocket` | `hooks/useWebSocket.ts` | Raw WebSocket wrapper with reconnect |
| `useTrainingStatus` | `hooks/useTrainingStatus.ts` | Polls `/api/training/status` every 2s |

### Dashboard Features

- **Live patient board** — real-time patient status, acuity scores, ward locations
- **Agent activity feed** — live message bus stream showing inter-agent communications
- **Resource gauges** — ICU beds, ventilators, blood supply, staff hours
- **Reward curve chart** — total episode reward over training time
- **Crisis alert banner** — crisis type, severity, time elapsed
- **Agent override controls** — manually inject CMO decisions
- **Training monitor** — GPU utilization, VRAM, loss curve, step progress

---

## AI Training System

### Local Training (Laptop — RTX 2050 / 4GB VRAM)

The system supports local training on modest hardware using a quantized 0.5B model:

```bash
cd triage-backend
source .venv/bin/activate
python scripts/train_dpo_gpu.py --epochs 1
```

The training script automatically:
1. Loads existing `.jsonl` dataset from `data/`
2. Applies 4-bit quantization (bitsandbytes) to fit in 4GB VRAM
3. Streams live progress to `data/training_live.json`
4. Saves the LoRA adapter to `models/dpo_output_gpu/`

### Cloud Training (Google Colab — T4 / 15GB VRAM)

For production-grade training on the `Qwen2.5-1.5B` model:

1. Open `notebooks/TRIAGE_Training_Colab.ipynb` in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run all 3 cells in order:
   - **Cell 1:** Install dependencies
   - **Cell 2:** Download Kaggle datasets and build 6,000+ DPO pairs
   - **Cell 3:** Train the 1.5B model with LoRA DPO

### Dataset Sources

| Dataset | Source | Records | Used for |
|---------|--------|---------|----------|
| Healthcare Disease Prediction | Kaggle (algozee) | ~5,000 | Clinical triage pairs |
| Pharma Drug Interactions | Kaggle (mdmahfuzsumon) | ~3,000 | Safe prescription pairs |
| Healthcare Fraud Detection | Kaggle (nudratabbas) | ~5,000 | Claim validation pairs |
| TRIAGE Episodes | Self-generated (simulation) | Variable | Core RL alignment |
| HuggingFace Medical DPO | HuggingFace Hub | ~2,000 | General medical reasoning |

**Total training pairs after combining:** ~6,000–10,000 chosen/rejected pairs

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- NVIDIA GPU (optional, auto-detected)
- 8 GB RAM minimum (16 GB recommended)

### 1. Clone and Start Everything

```bash
git clone <repo-url>
cd "META final"
chmod +x start.sh
./start.sh
```

This single script starts both the backend (port 8000) and frontend (port 8081).

### 2. Manual Start (Backend Only)

```bash
cd triage-backend
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Start the API server
uvicorn triage.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Manual Start (Frontend Only)

```bash
cd triage-frontend/triage-command-center-main
npm install
npm run dev -- --port 8081
```

### 4. Run a Simulation Episode

```bash
# Via CLI
cd triage-backend
source .venv/bin/activate
python scripts/run_episode.py

# Via API
curl -X POST http://localhost:8000/api/episodes/run \
  -H "Content-Type: application/json" \
  -d '{"crisis_type": "mass_casualty", "max_steps": 20}'
```

### 5. Start DPO Training

```bash
# Generate training data from episodes
python scripts/collect_episodes.py

# Start training
python scripts/train_dpo_gpu.py --epochs 1

# Monitor (in another terminal)
watch -n 2 "cat data/training_live.json | python3 -m json.tool"
```

---

## Configuration

### Environment Variables (`.env`)

```env
# API
PORT=8000
DEBUG=true

# Model
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
USE_MOCK_LLM=true          # Set false to use real LLM

# Database
DATABASE_URL=sqlite+aiosqlite:///./triage.db

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Training
TRAINING_OUTPUT_DIR=./models/dpo_output_gpu
DPO_EPOCHS=1
DPO_BATCH_SIZE=1
```

### Agent Configuration (`config/agents.yaml`)

Each agent's system prompt, tools, priority, and role are defined here. This is loaded at startup by `BaseAgent.__init__()`.

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/run_episode.py` | Run a single simulation episode |
| `scripts/run_simulation.py` | Run multiple episodes back-to-back |
| `scripts/collect_episodes.py` | Collect episodes → save as trajectories |
| `scripts/train_dpo.py` | Basic DPO training (CPU-safe) |
| `scripts/train_dpo_gpu.py` | Full GPU DPO training with live metrics |
| `scripts/generate_dpo_fast.py` | Fast synthetic DPO dataset generation |
| `scripts/generate_dpo_dataset_ollama.py` | Generate DPO data using Ollama (local LLM) |
| `scripts/download_hf_dataset.py` | Pull medical DPO data from HuggingFace |
| `scripts/merge_and_test.py` | Merge LoRA adapter and run inference test |
| `scripts/export_metrics.py` | Export episode metrics to CSV |
| `scripts/generate_training_report.py` | Generate HTML/markdown training report |
| `scripts/demo.py` | Interactive demo script |
| `scripts/colab_dpo_builder.py` | **For Colab:** Build massive Kaggle dataset |
| `scripts/colab_train_dpo.py` | **For Colab:** Train 1.5B model on T4 GPU |

---

## Data Directory

```
triage-backend/data/
├── training_live.json       # Live GPU training metrics (streamed by train_dpo_gpu.py)
├── preference_dataset.json  # Labeled preference pairs for DPO
├── full_training/
│   └── dpo_pairs.jsonl      # Combined DPO dataset (all sources)
├── episodes/                # Raw episode trajectories
└── reports/                 # Generated training reports
```

---

## Deployment

### Docker

```bash
cd triage-backend
docker-compose up --build
```

The `docker-compose.yml` starts:
- `triage-api` — FastAPI backend on port 8000
- `redis` — Message broker for Celery tasks

### Production Checklist

- [ ] Set `DEBUG=false` in `.env`
- [ ] Set `USE_MOCK_LLM=false` and point to your fine-tuned model
- [ ] Replace SQLite with PostgreSQL for production DB
- [ ] Set `CORS_ORIGINS` to your production frontend domain
- [ ] Enable HTTPS via nginx/caddy reverse proxy
- [ ] Configure Celery with a proper broker (not in-memory)

---

## Tests

```bash
cd triage-backend
source .venv/bin/activate
pytest tests/ -v
```

**Test coverage includes:**
- `test_api.py` — FastAPI route tests
- `test_core.py` — HospitalEnv and agent unit tests
- `test_env.py` — Environment reset/step/state integration tests
- `test_grand_finale.py` — Full end-to-end episode simulation

---
