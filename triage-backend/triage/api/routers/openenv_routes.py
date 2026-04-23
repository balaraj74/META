"""
openenv_routes.py — FastAPI routes exposing HospitalEnv as an OpenEnv-compliant API.

These routes let OpenEnv trainers discover and interact with the environment
remotely. They also power the Gradio Space's live simulation tab.

Endpoints:
    GET  /env/info   — environment metadata
    POST /env/reset  — reset and return initial observation
    POST /env/step   — apply action, return (obs, reward, done, info)
    GET  /env/state  — current state snapshot
    GET  /env/render — human-readable text view
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from triage.env.openenv_adapter import TriageOpenEnv

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/env", tags=["OpenEnv"])

# ── Module-level environment instance ─────────────────────────────────────────
# Lazily initialized on first reset. This keeps import cheap.
_env: TriageOpenEnv | None = None


def _get_env() -> TriageOpenEnv:
    global _env
    if _env is None:
        _env = TriageOpenEnv()
    return _env


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    crisis_type: str | None = Field(None, description="Crisis type: mass_casualty, outbreak, equipment_failure, staff_shortage")
    difficulty: float = Field(0.5, ge=0.0, le=1.0, description="Difficulty level")
    seed: int = Field(42, description="Random seed")
    max_steps: int = Field(50, ge=5, le=200, description="Max steps per episode")


class StepRequest(BaseModel):
    agent_type: str = Field("er_triage", description="Agent performing the action")
    action_type: str = Field(..., description="Action type e.g. TRIAGE_PATIENT")
    target_id: int = Field(0, description="Patient or resource target ID")
    priority: int = Field(5, ge=1, le=10, description="Action priority (1=highest)")
    reasoning: str = Field("", max_length=500, description="Agent reasoning text")


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class PromptRequest(BaseModel):
    agent_type: str = Field("er_triage", description="Agent type for prompt generation")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/info")
async def env_info() -> dict[str, Any]:
    """Return OpenEnv environment metadata. Used by trainers for discovery."""
    return TriageOpenEnv.metadata


@router.post("/reset")
async def env_reset(req: ResetRequest) -> dict[str, Any]:
    """Reset the environment and return the initial observation."""
    global _env
    _env = TriageOpenEnv(
        seed=req.seed,
        max_steps=req.max_steps,
        difficulty=req.difficulty,
        crisis_type=req.crisis_type,
    )
    try:
        obs = _env.reset()
        return {"observation": obs, "info": {"crisis_type": req.crisis_type or "random"}}
    except Exception as exc:
        logger.error("env_reset failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/step", response_model=StepResponse)
async def env_step(req: StepRequest) -> StepResponse:
    """Apply an action and advance the environment by one step."""
    env = _get_env()
    if env.is_done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /env/reset first.")

    action = {
        "agent_type": req.agent_type,
        "action_type": req.action_type,
        "target_id": req.target_id,
        "priority": req.priority,
        "reasoning": req.reasoning,
    }

    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except Exception as exc:
        logger.error("env_step failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/state")
async def env_state() -> dict[str, Any]:
    """Return the current environment state snapshot."""
    env = _get_env()
    return env.observation()


@router.get("/render")
async def env_render() -> dict[str, str]:
    """Return a human-readable text view of the current state."""
    env = _get_env()
    return {"render": env.render()}


@router.post("/prompt")
async def env_prompt(req: PromptRequest) -> dict[str, str]:
    """
    Generate an LLM prompt from the current environment state.

    Useful for testing reward verifiers manually:
        1. POST /env/reset
        2. POST /env/prompt → get prompt
        3. Send prompt to LLM → get completion
        4. POST /env/step with the completion → get reward
    """
    env = _get_env()
    try:
        prompt = env.state_to_prompt(req.agent_type)
        return {"prompt": prompt, "agent_type": req.agent_type}
    except Exception as exc:
        logger.error("env_prompt failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
