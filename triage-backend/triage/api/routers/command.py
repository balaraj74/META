"""Command Center demo endpoints — chat, crisis injection, violation injection."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from triage.api.schemas import ApiResponse, SimulationConfig
from triage.env.state import CrisisType


def _get_sim():
    """Lazy import of sim_manager to avoid circular/transitive deps at module load."""
    from triage.api.main import sim_manager  # noqa: PLC0415
    return sim_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Pydantic schemas ─────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    agent: str = "CMO_OVERSIGHT"
    context: str = ""


class ChatResponse(BaseModel):
    agent: str
    response: str
    tokens: int


class CrisisInjectRequest(BaseModel):
    crisis_type: str = "mass_casualty"
    severity: float = 0.7
    auto_step: bool = True


class ViolationInjectRequest(BaseModel):
    agent: str = "ER_TRIAGE"
    violation_type: str = "protocol_breach"
    description: str = ""


# ── Canned agent responses keyed by agent type ──────────────────────────────

_AGENT_CONTEXT: dict[str, dict[str, str]] = {
    "CMO_OVERSIGHT": {
        "persona": "Chief Medical Officer overseeing all hospital operations",
        "prefix": "CMO OVERSIGHT →",
        "color": "#a855f7",
    },
    "ER_TRIAGE": {
        "persona": "Emergency Room Triage specialist managing patient intake",
        "prefix": "ER TRIAGE →",
        "color": "#dc2626",
    },
    "ICU_MANAGEMENT": {
        "persona": "ICU Management specialist handling critical care capacity",
        "prefix": "ICU MGMT →",
        "color": "#0284c7",
    },
    "PHARMACY": {
        "persona": "Pharmacy director managing drug inventory and allocation",
        "prefix": "PHARMACY →",
        "color": "#d97706",
    },
    "HR_ROSTERING": {
        "persona": "HR Rostering coordinator handling staff scheduling",
        "prefix": "HR ROSTER →",
        "color": "#0d9488",
    },
    "IT_SYSTEMS": {
        "persona": "IT Systems engineer maintaining EHR and hospital infrastructure",
        "prefix": "IT SYSTEMS →",
        "color": "#6b7280",
    },
}

_SMART_RESPONSES: dict[str, list[str]] = {
    "CMO_OVERSIGHT": [
        "Monitoring all department actions. Current compliance rate: 94.2%. No critical protocol deviations detected.",
        "DPO-trained reasoning active. Evaluating agent decisions against clinical guidelines and ethical constraints.",
        "Oversight layer flagged 2 anomalies in the last episode. Both corrected within 3 steps — system self-healing.",
        "Running reward decomposition: survival bonus 0.42, compliance 0.31, resource efficiency 0.27. Total: 0.78.",
        "Agent coordination is functioning optimally. Message bus latency: 12ms. All 6 agents responsive.",
    ],
    "ER_TRIAGE": [
        "Incoming: 4 patients via ambulance. ETA 3 minutes. Pre-triage scores assigned based on symptom profiles.",
        "Current ER occupancy: 87%. Redirecting non-critical patients to WARD-B to free trauma bays.",
        "RED tag patient in Bay 3 requires immediate OR. Requesting ICU bed pre-clearance.",
        "Mass casualty protocol ALPHA-7 activated. Triage algorithm processing 12 simultaneous admissions.",
        "Vitals anomaly detected on PT-0047. Escalating to CRITICAL. Notifying CMO and ICU.",
    ],
    "ICU_MANAGEMENT": [
        "ICU at 76% capacity. 14/18 beds occupied. 2 ventilators available for immediate deployment.",
        "Predicting bed shortage in ~8 steps based on current intake rate. Recommending proactive transfer.",
        "Patient PT-0023 stable. Downgrading to WARD-A. Freeing ICU bed and ventilator slot.",
        "Implementing adaptive bed allocation — prioritizing polytrauma over stable cardiac cases.",
        "Resource optimization complete. Redistributed ventilators: ICU:3, ER:1, held:1.",
    ],
    "PHARMACY": [
        "Epinephrine stock at 18%. Emergency procurement initiated. ETA: 40 minutes.",
        "Drug allocation optimized for mass casualty scenario. Morphine, antibiotics, and blood products prioritized.",
        "Monitoring controlled substance usage. No anomalies detected. All dispensing within protocol.",
        "Automated reorder triggered for O-negative blood: 8 units requested from regional bank.",
        "Expiry audit complete. 3 medications flagged, removed from active inventory. Compliance maintained.",
    ],
    "HR_ROSTERING": [
        "Night shift coverage: 94%. Called in 3 additional nurses from on-call roster.",
        "Fatigue monitoring active. Dr. Reyes at 11-hour mark — recommending handoff to Dr. Kim.",
        "Staff ratio: 1:3.2 (nurse:patient). Within acceptable bounds for current crisis level.",
        "Scheduling optimization reduced overtime hours by 18% while maintaining full coverage.",
        "Emergency staffing protocol engaged. All department heads notified of extended shift requirements.",
    ],
    "IT_SYSTEMS": [
        "EHR sync complete. 31 patient records updated. Zero data integrity issues detected.",
        "Insurance verification running: 24/31 approved, 7 pending. Flagging 2 for manual review.",
        "Network latency nominal. Agent message bus throughput: 847 msg/s. No bottlenecks.",
        "Schema drift detected in patient intake form. Alerting CMO. Rollback initiated.",
        "Backup completed. All episode data persisted. Replay system ready.",
    ],
}


def _get_mock_response(agent: str, message: str) -> str:
    """Generate a context-aware mock response for demo purposes."""
    responses = _SMART_RESPONSES.get(agent, _SMART_RESPONSES["CMO_OVERSIGHT"])
    base = random.choice(responses)

    # Inject message keywords into response for demo realism
    lower = message.lower()
    if "status" in lower or "how" in lower:
        return base
    if "crisis" in lower or "emergency" in lower:
        return f"Crisis protocol engaged. {base}"
    if "patient" in lower:
        return f"Patient management active. {base}"
    if "drug" in lower or "medication" in lower or "pharma" in lower:
        return f"Pharmacy systems nominal. {base}"
    if "staff" in lower or "nurse" in lower or "doctor" in lower:
        return f"Staffing assessment complete. {base}"
    return base


def _try_llm_response(agent: str, message: str, context: str) -> str | None:
    """Attempt to call the local Ollama model. Returns None if unavailable.

    Qwen3.5 is a thinking model that by default puts all output into a
    ``thinking`` field and returns empty ``content``.  We disable thinking
    mode entirely with ``think: false`` for speed and real content.
    """
    try:
        import httpx
        import re

        ctx_info = _AGENT_CONTEXT.get(agent, _AGENT_CONTEXT["CMO_OVERSIGHT"])
        system_prompt = (
            f"You are the {ctx_info['persona']} in a hospital crisis simulation. "
            "Respond concisely in 1-2 sentences. Be specific and professional. "
            "Reference clinical details when possible."
        )
        chat = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            chat.append({"role": "user", "content": f"Context: {context}"})
        chat.append({"role": "user", "content": message})

        response = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": os.getenv("MODEL_NAME", "qwen3.5:4b"),
                "messages": chat,
                "stream": False,
                "think": False,          # disable CoT — returns real content instantly
                "options": {
                    "num_ctx": 2048,     # increased context for 4B model reasoning
                    "num_predict": 256,  # increased token cap for 4B model quality
                }
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        msg = data.get("message", {})
        content = msg.get("content", "").strip()

        # Safety fallback: if content is empty (CoT bled through), extract
        # from the thinking field by taking the last non-empty line.
        if not content:
            thinking = msg.get("thinking", "")
            # Strip <think>…</think> wrapper if present, grab last sentence
            thinking_clean = re.sub(r"</?think>", "", thinking).strip()
            lines = [l.strip() for l in thinking_clean.splitlines() if l.strip()]
            content = lines[-1] if lines else ""

        return content or None
    except Exception as exc:
        logger.warning("LLM inference failed, falling back to mock: %s", exc)
        return None


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/api/chat", response_model=ApiResponse)
async def chat(req: ChatRequest) -> ApiResponse:
    """Single-turn chat with a specific agent (tries local LLM, falls back to mock)."""
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        None, lambda: _try_llm_response(req.agent, req.message, req.context)
    )
    is_mock = reply is None
    if is_mock:
        reply = _get_mock_response(req.agent, req.message)

    ctx = _AGENT_CONTEXT.get(req.agent, _AGENT_CONTEXT["CMO_OVERSIGHT"])
    return ApiResponse(
        success=True,
        data={
            "agent": req.agent,
            "prefix": ctx["prefix"],
            "color": ctx["color"],
            "response": reply,
            "tokens": len(reply.split()),
            "model": "mock" if is_mock else os.getenv("MODEL_NAME", "qwen3.5:4b"),
        },
    )


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE streaming chat for typewriter effect. Falls back to mock chunked response."""

    async def _stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(
            None, lambda: _try_llm_response(req.agent, req.message, req.context)
        )
        if reply is None:
            reply = _get_mock_response(req.agent, req.message)

        ctx = _AGENT_CONTEXT.get(req.agent, _AGENT_CONTEXT["CMO_OVERSIGHT"])
        words = reply.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            payload = json.dumps(
                {
                    "agent": req.agent,
                    "prefix": ctx["prefix"],
                    "color": ctx["color"],
                    "chunk": chunk,
                    "done": i == len(words) - 1,
                }
            )
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.045)  # ~22 words/s — readable typewriter speed

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/api/inject/crisis", response_model=ApiResponse)
async def inject_crisis(req: CrisisInjectRequest) -> ApiResponse:
    """Inject a crisis event into the current simulation episode."""
    # Map UI string to backend CrisisType
    _CRISIS_MAP: dict[str, str] = {
        "mass_casualty": "mass_casualty",
        "outbreak": "outbreak",
        "equipment_failure": "equipment_failure",
        "staff_shortage": "staff_shortage",
    }
    crisis_str = _CRISIS_MAP.get(req.crisis_type, "mass_casualty")

    # Start a fresh episode with the injected crisis type
    result = await _get_sim().start(
        SimulationConfig(
            crisis_type=crisis_str,  # type: ignore[arg-type]
            difficulty=req.severity,
            max_steps=50,
            mock_llm=True,
            auto_step=req.auto_step,
            step_delay_ms=600,
        )
    )

    return ApiResponse(
        success=True,
        data={
            "injected": True,
            "crisis_type": crisis_str,
            "severity": req.severity,
            "episode_id": result.get("episode_id"),
            "message": f"Crisis '{crisis_str}' injected at severity {req.severity:.0%}. Agents responding.",
        },
    )


@router.post("/api/inject/violation", response_model=ApiResponse)
async def inject_violation(req: ViolationInjectRequest) -> ApiResponse:
    """Inject a deliberate protocol violation and let CMO catch it."""
    description = req.description or _DEFAULT_VIOLATIONS.get(req.violation_type, "Unknown violation")

    # Broadcast the violation via WebSocket for live UI
    await _get_sim()._broadcast(  # type: ignore[attr-defined]
        "oversight_alert",
        {
            "agent": req.agent,
            "violation_type": req.violation_type,
            "description": description,
            "cmo_response": _CMO_CATCHES.get(
                req.violation_type,
                "⚠️ CMO OVERSIGHT: Protocol deviation detected. Initiating corrective action.",
            ),
            "severity": "HIGH",
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        },
    )

    return ApiResponse(
        success=True,
        data={
            "injected": True,
            "agent": req.agent,
            "violation_type": req.violation_type,
            "description": description,
            "cmo_intercepted": True,
            "message": f"Violation injected for {req.agent}. CMO Oversight has flagged and corrected.",
        },
    )


_DEFAULT_VIOLATIONS: dict[str, str] = {
    "protocol_breach": "Agent dispensed medication exceeding protocol dosage by 15%.",
    "triage_mismatch": "Patient PT-0031 tagged GREEN despite critical vitals (BP 70/40, GCS 9).",
    "bed_overflow": "ICU admission attempted despite 100% capacity — bypass protocol ignored.",
    "drug_double_dose": "Morphine double-dose administered without authorization.",
    "unauthorized_discharge": "Patient discharged without attending physician sign-off.",
}

_CMO_CATCHES: dict[str, str] = {
    "protocol_breach": "⚠️ CMO OVERRIDE: Dosage violation detected on PHARMACY. Reverting to protocol baseline. Compliance rate updated.",
    "triage_mismatch": "⚠️ CMO OVERRIDE: Triage tag discrepancy on PT-0031. Escalating to RED. ICU notified. ER Triage log flagged.",
    "bed_overflow": "⚠️ CMO OVERRIDE: ICU capacity exceeded. Activating transfer protocol ALPHA-3. Regional hospital contacted.",
    "drug_double_dose": "⚠️ CMO OVERRIDE: Duplicate morphine order intercepted. Dispensing halted. Incident report generated.",
    "unauthorized_discharge": "⚠️ CMO OVERRIDE: Unauthorized discharge blocked. Patient retained. Attending physician paged.",
}
