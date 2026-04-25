# 🏥 TRIAGE — Multi-Agent Hospital Crisis Simulation

> **Meta PyTorch OpenEnv Hackathon Entry**
> A production-grade, OpenEnv-compatible multi-agent AI system that simulates real-time hospital crisis management with GRPO fine-tuning, DPO preference training, safety middleware, and a live command center dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend — triage-backend](#backend)
   - [Environment](#environment)
   - [Agents](#agents)
   - [Reward Model](#reward-model)
   - [Safety Constitution](#safety-constitution)
   - [Curriculum Learning](#curriculum-learning)
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

TRIAGE is a **multi-agent reinforcement learning environment** that simulates a hospital under crisis — mass casualty events, disease outbreaks, equipment failures, and staff shortages. Ten specialized AI agents collaborate to manage patients, allocate resources, and maintain policy compliance, all orchestrated through a typed message bus.

The system is designed for the **Meta PyTorch OpenEnv Hackathon**, implementing the OpenEnv-compatible interface (`reset`, `step`, `state`) alongside GRPO and DPO training pipelines to teach the agents safe, clinically aligned decision-making.

### Key Highlights

| Feature | Detail |
|---|---|
| **Framework** | OpenEnv + PyTorch + HuggingFace Transformers |
| **Base Model** | Qwen/Qwen3-27B (48GB VRAM HF Spaces) |
| **Training Method** | GRPO (primary) via TRL + Unsloth + DPO (secondary) |
| **Agents** | 10 specialized agents (CMO, ER, ICU, Pharmacy, HR, IT, BloodBank, EthicsCommittee, InfectionControl, AmbulanceDispatch) |
| **Crisis Types** | Mass Casualty, Disease Outbreak, Equipment Failure, Staff Shortage |
| **Safety Layer** | SafetyConstitution — 10 hard-block rules, auto-fallback actions |
| **Reward System** | 4 independent binary GRPO verifiers + rule-based reward model |
| **Curriculum** | 5-tier auto-advancing difficulty scheduler |
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
                                 │  │  │ CMO Oversight    │   │ │
                                 │  │  │ ER Triage        │   │ │
                                 │  │  │ ICU Management   │   │ │
                                 │  │  │ Pharmacy         │   │ │
                                 │  │  │ HR Rostering     │   │ │
                                 │  │  │ IT Systems       │   │ │
                                 │  │  │ Blood Bank       │   │ │
                                 │  │  │ Ethics Committee │   │ │
                                 │  │  │ Infection Control│   │ │
                                 │  │  │ Ambulance Dispatch│  │ │
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
                                 │  │  EpisodeCollector        │ │
                                 │  │  PreferenceLabeler       │ │
                                 │  │  GRPO + DPO Pipeline     │ │
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
│   │   ├── specialized.py       # 10 concrete agent implementations
│   │   │                         # Includes BloodBank, EthicsCommittee,
│   │   │                         # InfectionControl, AmbulanceDispatch
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
│   │   ├── enterprise_registry.py
│   │   ├── curriculum.py        # CurriculumScheduler — 5 difficulty tiers
│   │   ├── grpo_env_adapter.py  # TRL environment_factory adapter
│   │   ├── openenv_adapter.py   # OpenEnv SDK interface adapter
│   │   └── schema_drift.py      # Simulated EHR schema drift (IT agent)
│   ├── safety/
│   │   └── constitution.py      # SafetyConstitution — 10 hard-block rules
│   ├── reward/
│   │   └── evaluator.py         # Episode reward evaluation wrapper
│   ├── rewards/
│   │   ├── reward_model.py      # Multi-component reward model
│   │   ├── verifiers.py         # 4 independent binary GRPO verifiers
│   │   └── reward_logger.py     # Per-verifier W&B logging + variance monitoring
│   ├── eval/
│   │   └── benchmark.py         # 5 held-out crisis seeds, before/after eval
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
│   ├── train_grpo.py            # GRPO training with Unsloth + Qwen3-27B
│   ├── build_grpo_dataset.py    # Generates 500 crisis scenario prompts
│   └── demo_before_after.py     # Baseline vs fine-tuned comparison
├── notebooks/                   # Jupyter / Colab training notebooks
├── config/                      # YAML configuration files
├── data/                        # Dataset files and training outputs
│   ├── drug_interactions.json   # Drug contraindication database
│   ├── curriculum_state.json    # Persisted curriculum tier state
│   ├── eval_history.jsonl       # Benchmark run history
│   └── grpo_crisis_prompts/     # Arrow-format GRPO training dataset
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
| `agent_states` | `(10, 8)` | Each agent's action/message counters |
| `crisis_state` | `(10,)` | Crisis type, severity, time elapsed |
| `policy_state` | `(20,)` | Active policy flags |
| `expert_signals` | `(10,)` | Per-agent performance signals |

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

Ten agents, each inheriting from `BaseAgent`, collaborate through the `MessageBus`.

#### Agent Execution Order (per step)

1. `AMBULANCE_DISPATCH` — controls patient inflow first
2. `ER_TRIAGE` — intake and classify arriving patients
3. `INFECTION_CONTROL` — isolate before ICU transfers
4. `ICU_MANAGEMENT` — bed and ventilator allocation
5. `PHARMACY` — medication orders
6. `BLOOD_BANK` — blood inventory and requests
7. `HR_ROSTERING` — staff scheduling
8. `IT_SYSTEMS` — EHR and compliance
9. `CMO_OVERSIGHT` — escalations and overrides
10. `ETHICS_COMMITTEE` — audits all allocations LAST

#### Agent Hierarchy

```
CMO_OVERSIGHT  ──── supervisor, handles escalations and overrides
     │
     ├── AMBULANCE_DISPATCH ── controls inbound incidents and patient inflow
     ├── ER_TRIAGE          ── intake, initial patient classification
     ├── INFECTION_CONTROL  ── isolation, outbreak containment before transfers
     ├── ICU_MANAGEMENT     ── bed allocation, ventilator management
     ├── PHARMACY           ── medication dispensing, drug interaction checks
     ├── BLOOD_BANK         ── blood inventory, compatibility, transfusion requests
     ├── HR_ROSTERING       ── staff scheduling, fatigue monitoring
     ├── IT_SYSTEMS         ── EHR integrity, schema drift detection, policy compliance
     └── ETHICS_COMMITTEE   ── allocation audit and fairness review, runs last
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

## Safety Constitution

**File:** `triage/safety/constitution.py`

A middleware layer that wraps every agent's output before actions reach the environment. Hard-blocks 10 categories of unsafe decisions and replaces them with safe fallbacks automatically.

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

Ablation mode: set `CONSTITUTION_ACTIVE=false` in `.env` to disable for A/B testing (shows chaos without safety layer — powerful demo).

---

## Curriculum Learning

**File:** `triage/env/curriculum.py`

Auto-advancing 5-tier difficulty system. Tier advances when mean reward over last 5 episodes > 0.7. Regresses when < 0.2.

| Tier | Crisis Types | Patients | Compound Events |
|---|---|---|---|
| 1 | Mass Casualty only | 10-15 | No |
| 2 | Mass Casualty + Staff Shortage | 20-25 | No |
| 3 | Three crisis types | 30-35 | Yes |
| 4 | All four crisis types | 40-45 | Yes + Schema Drift |
| 5 | All types simultaneous | 48-50 | Maximum stress |

State persists to `data/curriculum_state.json` across training runs.

---

### Training Pipeline

**Files:** `triage/training/`

The training system is GRPO-first, with DPO retained as a secondary preference-alignment path:

```
HospitalEnv (OpenEnv reset/step/state)
│
▼
CurriculumScheduler ── selects difficulty tier per episode
│
▼
HospitalGRPOEnvironment ── environment_factory for TRL
│
▼
GRPOTrainer (TRL) + Unsloth Qwen3-27B
├── num_generations=16
├── max_seq_length=4096
└── 4 independent binary verifiers:
    ├── survival_verifier
    ├── safety_verifier
    ├── resource_verifier
    └── ethics_verifier
│
▼
TriageBenchmark ── eval on 5 held-out seeds every 100 steps
│
▼
W&B Dashboard ── per-verifier reward curves + curriculum tier
│
▼
LoRA Adapter → models/grpo_qwen3_27b/
```

#### GRPO Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3-27B` | 48GB VRAM HuggingFace Spaces target |
| `trainer` | `TRL GRPOTrainer` | Primary RLVR trainer |
| `adapter` | `Unsloth LoRA` | Memory-efficient fine-tuning |
| `dataset` | `data/grpo_crisis_prompts/` | Arrow-format crisis prompt dataset |
| `environment_factory` | `HospitalGRPOEnvironment` | Live HospitalEnv rollout adapter |
| `reward_funcs` | `triage.rewards.verifiers` | Independent verifier suite |
| `curriculum_state` | `data/curriculum_state.json` | Persistent difficulty tier |

#### GRPO Configuration Table

| Parameter | Value | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3-27B` | 48GB HF Spaces, bf16 |
| `load_in_4bit` | `False` | Full bf16 - no quantization at 48GB |
| `lora_r` | `64` | LoRA rank |
| `lora_alpha` | `128` | 2x lora_r |
| `num_generations` | `16` | GRPO group size |
| `max_seq_length` | `4096` | Full clinical context |
| `per_device_batch` | `2` | Effective batch = 16 |
| `gradient_accum` | `8` | |
| `learning_rate` | `2e-5` | Cosine scheduler |
| `num_epochs` | `3` | |
| `gradient_checkpointing` | `False` | Not needed at 48GB |
| `thinking_mode` | `True` | Qwen3 - CMO/Ethics agents |

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
| `GET` | `/api/agents/` | List all 10 agents and their states |
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


#### Safety
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents/safety/blocks` | All SafetyBlock records current episode |
| `GET` | `/api/agents/safety/stats` | Constitution report - blocks by type/agent |

#### Dispatch
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents/dispatch/status` | Ambulance fleet + diversion stats |

#### Infection
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents/infection/status` | Ward case counts + lockdown status |

#### Ethics
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents/ethics/decisions` | Full rationing decision log |

#### Blood Bank
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents/bloodbank/inventory` | Blood type inventory + pending requests |

#### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/ws/live` | Real-time simulation state stream |


| Event Type | Description |
|---|---|
| `safety_block` | SafetyConstitution blocked an action |
| `rationing_decision` | Ethics committee made a rationing call |
| `infection_spread` | New infection event between patients |
| `ambulance_dispatch` | Ambulance dispatched to incident |
| `patient_diverted` | Patient diverted to another hospital |

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
| `/training` | `training.tsx` | GRPO/DPO training monitor with live metrics |
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
- **Training monitor** — GPU utilization, VRAM, loss curve, verifier rewards, step progress

---

## AI Training System

### Local Training (Development / Smoke Test)

The system supports local DPO and dataset smoke tests on modest hardware. Full GRPO training targets larger GPU infrastructure.

```bash
cd triage-backend
source .venv/bin/activate
python scripts/build_grpo_dataset.py --num-prompts 500
python scripts/train_dpo_gpu.py --epochs 1  # secondary DPO path
```

The training script automatically:
1. Loads GRPO prompts or existing `.jsonl` preference data from `data/`
2. Applies quantized LoRA when running the local DPO fallback path
3. Streams live progress to `data/training_live.json`
4. Saves adapters under `models/`

### Cloud Training (HuggingFace Spaces — 48GB VRAM)

For production-grade GRPO training on `Qwen/Qwen3-27B`:

1. Build or refresh crisis prompts with `scripts/build_grpo_dataset.py`
2. Launch `scripts/train_grpo.py` with TRL + Unsloth
3. Use `HospitalGRPOEnvironment` for live rollouts
4. Track per-verifier rewards and curriculum tier progression
5. Save the LoRA adapter to `models/triage_grpo_output/`

### Dataset Sources

| Dataset | Source | Records | Used for |
|---------|--------|---------|----------|
| Healthcare Disease Prediction | Kaggle (algozee) | ~5,000 | Clinical triage pairs |
| Pharma Drug Interactions | Kaggle (mdmahfuzsumon) | ~3,000 | Safe prescription pairs |
| Healthcare Fraud Detection | Kaggle (nudratabbas) | ~5,000 | Claim validation pairs |
| TRIAGE Episodes | Self-generated (simulation) | Variable | Core RL alignment |
| GRPO Crisis Prompts | Self-generated (simulation) | 500+ | Live GRPO rollouts |
| HuggingFace Medical DPO | HuggingFace Hub | ~2,000 | General medical reasoning |

**Total DPO pairs after combining:** ~6,000–10,000 chosen/rejected pairs

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

### 5. Start GRPO Training

```bash
# Generate GRPO crisis prompts
python scripts/build_grpo_dataset.py --num-prompts 500

# Start GRPO training
python scripts/train_grpo.py

# Monitor (in another terminal)
watch -n 2 "cat data/training_live.json | python3 -m json.tool"
```

For the secondary DPO path, use `scripts/collect_episodes.py` followed by `scripts/train_dpo_gpu.py --epochs 1`.

---

## Configuration

### Environment Variables (`.env`)

```env
# API
PORT=8000
DEBUG=true

# Model
MODEL_NAME=Qwen/Qwen3-27B
USE_MOCK_LLM=true          # Set false to use real LLM

# Safety
CONSTITUTION_ACTIVE=true   # Set false for ablation demos

# Database
DATABASE_URL=sqlite+aiosqlite:///./triage.db

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# GRPO Training
GRPO_TRAINING_MODE=false
NUM_GENERATIONS=16
GRPO_OUTPUT_DIR=./models/grpo_qwen3_27b

# Safety
CONSTITUTION_ACTIVE=true

# Curriculum
CURRICULUM_STATE_PATH=./data/curriculum_state.json

# Monitoring
WANDB_PROJECT=triage-openenv-hackathon
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
| `scripts/build_grpo_dataset.py` | Generate 500 crisis scenario prompts |
| `scripts/train_grpo.py` | GRPO training with Unsloth + Qwen3-27B |
| `scripts/train_grpo_hf.py` | Self-contained GRPO training for HuggingFace compute |
| `scripts/train_dpo.py` | Basic DPO training (CPU-safe) |
| `scripts/train_dpo_gpu.py` | Full GPU DPO training with live metrics |
| `scripts/demo_before_after.py` | Baseline vs fine-tuned comparison |
| `scripts/generate_dpo_fast.py` | Fast synthetic DPO dataset generation |
| `scripts/generate_dpo_dataset_ollama.py` | Generate DPO data using Ollama (local LLM) |
| `scripts/download_hf_dataset.py` | Pull medical DPO data from HuggingFace |
| `scripts/merge_and_test.py` | Merge LoRA adapter and run inference test |
| `scripts/export_metrics.py` | Export episode metrics to CSV |
| `scripts/generate_training_report.py` | Generate HTML/markdown training report |
| `scripts/demo.py` | Interactive demo script |
| `scripts/colab_dpo_builder.py` | **For Colab:** Build massive Kaggle dataset |
| `scripts/colab_train_dpo.py` | **For Colab:** Train the secondary DPO baseline on T4 GPU |

---

## Data Directory

```
triage-backend/data/
├── training_live.json       # Live GPU training metrics
├── drug_interactions.json   # Drug contraindication database
├── curriculum_state.json    # Persisted curriculum tier state
├── eval_history.jsonl       # Benchmark run history
├── preference_dataset.json  # Labeled preference pairs for DPO
├── grpo_crisis_prompts/     # Arrow-format GRPO training dataset
├── grpo/                    # GRPO train/eval prompt files and comparisons
├── full_training/
│   └── dpo_pairs.jsonl      # Combined DPO dataset (all sources)
├── episodes/                # Raw episode trajectories
└── reports/                 # Generated training reports
```

---


## Model Merging

Two independently trained LoRA adapters on the same Qwen3-27B base 
can be merged using DARE-TIES merging via mergekit:

```bash
pip install mergekit

# merge_config.yml
merge_method: dare_ties
base_model: Qwen/Qwen3-27B
models:
  - model: your_username/triage-lora
    parameters: {density: 0.7, weight: 0.5}
  - model: friend_username/triage-lora
    parameters: {density: 0.7, weight: 0.5}
out_path: ./merged_triage_model
```

Run `scripts/demo_before_after.py` against all three models 
(model A, model B, merged) to show judges the improvement delta.

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
