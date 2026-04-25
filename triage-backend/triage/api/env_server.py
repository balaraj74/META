"""Small HTTP wrapper for HospitalGRPOEnvironment used by GRPO training."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from triage.env.grpo_env_adapter import HospitalGRPOEnvironment

app = FastAPI(title="TRIAGE GRPO Environment Server")
env: HospitalGRPOEnvironment | None = None


class ResetRequest(BaseModel):
    crisis_type: str = "mass_casualty"
    difficulty: float = 0.5


class ToolRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
async def startup() -> None:
    global env
    os.environ["GRPO_TRAINING_MODE"] = "false"
    env = HospitalGRPOEnvironment()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    observation = env.reset(crisis_type=request.crisis_type, difficulty=request.difficulty)
    return {"observation": observation, "state": env.current_state.to_json(), "step": env.step_count}


@app.post("/tool/{tool_name}")
async def tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    method = getattr(env, tool_name)
    observation = method(**payload)
    state = env.current_state.to_json() if hasattr(env.current_state, "to_json") else env.current_state
    return {
        "observation": observation,
        "state": state,
        "step": env.step_count,
        "reward": env._last_reward,
        "done": env._done,
    }


@app.get("/terminal_reward")
async def terminal_reward() -> dict[str, float]:
    return {"reward": env._get_terminal_reward()}
