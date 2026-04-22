# TRIAGE — Developer Guide & API Specification

> Internal reference for contributors. Last updated: April 2026.

---

## How to Add a New Agent

1. **Add the type** to `AgentType` enum in `triage/env/state.py`:
   ```python
   class AgentType(str, Enum):
       MY_NEW_AGENT = "my_new_agent"
   ```

2. **Add agent config** to `config/agents.yaml`:
   ```yaml
   my_new_agent:
     role: "Specialist"
     priority: 3
     system_prompt: "You are a specialist responsible for..."
     tools: []
   ```

3. **Create the class** in `triage/agents/specialized.py`:
   ```python
   class MyNewAgent(BaseAgent):
       def __init__(self, config, bus, mock_llm=True):
           super().__init__(AgentType.MY_NEW_AGENT, config, bus, mock_llm)

       async def decide(self, state, inbox):
           return self._rule_based_decision(state, inbox)

       def _rule_based_decision(self, state, inbox):
           actions = []
           # your logic here
           return actions
   ```

4. **Register in orchestrator** `triage/agents/orchestrator.py` — add to the agent init map.

---

## How to Add a New Crisis Type

1. Add to `CrisisType` enum in `triage/env/state.py`
2. Add a condition library list in `triage/env/crisis_generator.py`
3. Add a generation branch in `CrisisGenerator.generate()` method
4. Add weight to the random selector at the bottom of `generate()`

---

## How to Add a New Reward Component

1. Open `triage/rewards/reward_model.py`
2. Add a new `_compute_<name>(state) -> float` method
3. Add it to the `compute_step_reward()` weighted sum
4. Update the weight constants at the top to sum to 1.0

---

## WebSocket Message Protocol

### Server → Client (stream events)

```typescript
// State update (every step)
{
  type: "state_update",
  data: {
    episode_id: string,
    step: number,
    patients: Patient[],
    resources: ResourceState,
    agents: AgentState[],
    reward: number,
    done: boolean,
    crisis: Crisis
  }
}

// Agent action event
{
  type: "agent_action",
  data: {
    agent: AgentType,
    action_type: ActionType,
    target_id: string | null,
    reasoning: string,
    priority: number
  }
}

// Message bus event
{
  type: "agent_message",
  data: {
    from: AgentType,
    to: AgentType,
    msg_type: MessageType,
    content: string,
    priority: number
  }
}

// Episode complete
{
  type: "episode_complete",
  data: {
    episode_id: string,
    total_reward: number,
    steps: number,
    outcome: "success" | "failure" | "timeout"
  }
}

// Error
{ type: "error", data: { error: string } }
```

### Client → Server (commands)

```typescript
{ "command": "start_episode", "params": { "crisis_type": "mass_casualty" } }
{ "command": "step",          "params": {} }
{ "command": "get_state",     "params": {} }
{ "command": "pause",         "params": {} }
{ "command": "resume",        "params": {} }
{ "command": "override_agent", "params": {
    "agent": "cmo_oversight",
    "action": { "action_type": 6, "target_id": "...", "reasoning": "...", "priority": 9 }
  }
}
```

---

## Training Status JSON Schema

The file `data/training_live.json` (read by `/api/training/status`) follows this schema:

```json
{
  "phase": "training",
  "progress": 45,
  "step": 450,
  "total_steps": 1000,
  "loss": 0.342,
  "reward_margin": 1.21,
  "learning_rate": 0.00004,
  "train_samples": 900,
  "eval_loss": 0.391,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "vram_used_gb": 2.77,
  "gpu_pct": 75.3,
  "elapsed_s": 1240,
  "eta_s": 1510,
  "output_dir": "./models/dpo_output_gpu"
}
```

**Phase values:** `starting` | `loading_model` | `loading_data` | `training` | `saving` | `completed` | `failed`

---

## Database Schema

**Tables** (SQLite via SQLAlchemy):

| Table | Key Columns |
|-------|-------------|
| `episodes` | id, crisis_type, total_reward, steps, outcome, created_at |
| `patients` | id, episode_id, name, condition, status, acuity, ward |
| `agent_messages` | id, episode_id, from_agent, to_agent, msg_type, content, step |
| `rewards` | id, episode_id, step, total_reward, breakdown (JSON) |
| `strategy_lessons` | id, agent_type, lesson, embedding, created_at |

---

## Common Troubleshooting

### `OutOfMemoryError` during training
- Reduce `per_device_train_batch_size` to `1`
- Set `gradient_checkpointing=True`
- Enable 4-bit quant: `load_in_4bit=True`
- Use a smaller model: `Qwen/Qwen2.5-0.5B-Instruct`

### Training dashboard shows `not_started`
- Check that `data/training_live.json` exists and is fresh (< 2 min old)
- The `/api/training/status` route in `main.py` reads this file directly
- Run `python scripts/train_dpo_gpu.py` again to regenerate

### WebSocket disconnects immediately
- Check CORS settings in `api/main.py` allow your frontend origin
- Confirm the backend is running on port 8000
- Check browser console for the exact disconnect reason

### Agents always return rule-based decisions (no LLM reasoning)
- Set `USE_MOCK_LLM=false` in `.env`
- Ensure Ollama is running: `ollama serve` (for local LLM)
- Or set `MODEL_NAME` to a HuggingFace model ID

### `openenv-core` import error
- Install from project extras: `pip install openenv-core==0.2.3`
- The package is listed in `pyproject.toml` under `dependencies`
