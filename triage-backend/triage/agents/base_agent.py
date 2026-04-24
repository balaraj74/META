"""
BaseAgent — abstract base for all 6 specialized hospital agents.

Each agent:
  - Has a system prompt and tool definitions (from config/agents.yaml)
  - Can observe the environment state
  - Can execute tool calls against enterprise apps
  - Communicates via the typed MessageBus
  - Tracks its own token usage and action history
  - Supports both LLM-backed and rule-based (mock) modes
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
import httpx

from triage.agents.message_bus import MessageBus
from triage.env.state import (
    AgentAction,
    AgentMessage,
    AgentType,
    ActionType,
    EnvironmentState,
    MessageType,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all TRIAGE agents.

    Subclasses implement `decide()` to produce actions based on observations.
    The base class handles message routing, tool execution, and token tracking.
    """

    def __init__(
        self,
        agent_type: AgentType,
        config: dict[str, Any],
        message_bus: MessageBus,
        mock_llm: bool = True,
        model_name: Optional[str] = None,
    ) -> None:
        self.agent_type = agent_type
        self.config = config
        self.bus = message_bus
        self.mock_llm = mock_llm
        self.model_name = model_name or os.getenv("OLLAMA_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # From config/agents.yaml
        self.system_prompt: str = config.get("system_prompt", "")
        self.tools: list[dict[str, Any]] = config.get("tools", [])
        self.role: str = config.get("role", "agent")
        self.priority: int = config.get("priority", 5)

        # Tracking
        self.total_tokens: int = 0
        self.actions_taken: int = 0
        self.messages_sent: int = 0
        self._action_history: list[AgentAction] = []
        self._inbox: list[AgentMessage] = []

        # Register on message bus
        self.bus.subscribe(agent_type, self._on_message)

    # ── Abstract Methods ─────────────────────────────────

    @abstractmethod
    async def decide(
        self,
        state: EnvironmentState,
        inbox: list[AgentMessage],
    ) -> list[AgentAction]:
        """Decide what actions to take given current state and messages.

        Args:
            state: Current environment state.
            inbox: Messages received since last decision.

        Returns:
            List of actions to execute (may be empty).
        """
        ...

    @abstractmethod
    def _rule_based_decision(
        self,
        state: EnvironmentState,
        inbox: list[AgentMessage],
    ) -> list[AgentAction]:
        """Fallback rule-based decision when LLM is unavailable."""
        ...

    # ── Public API ───────────────────────────────────────

    async def act(
        self,
        state: EnvironmentState,
    ) -> list[AgentAction]:
        """Main entry point — observe, decide, act.

        Returns list of actions taken.
        """
        # Collect inbox
        inbox = list(self._inbox)
        self._inbox.clear()

        # Decide
        try:
            if self.mock_llm:
                actions = self._rule_based_decision(state, inbox)
            else:
                actions = await self.decide(state, inbox)
        except Exception:
            logger.exception("Agent %s decision failed, using fallback", self.agent_type.value)
            actions = self._rule_based_decision(state, inbox)

        # Record
        for action in actions:
            self.actions_taken += 1
            self._action_history.append(action)

        return actions

    async def send_message(
        self,
        to: AgentType | str,
        content: str,
        msg_type: MessageType = MessageType.ACTION,
        priority: int = 0,
        patient_id: str | None = None,
    ) -> bool:
        """Send a message through the bus."""
        token_count = len(content.split()) * 2  # rough estimate
        msg = AgentMessage(
            from_agent=self.agent_type,
            to_agent=to,
            content=content,
            msg_type=msg_type,
            priority=priority,
            patient_id=patient_id,
            token_count=token_count,
        )
        self.messages_sent += 1
        self.total_tokens += token_count
        return await self.bus.send(msg)

    async def broadcast(
        self,
        content: str,
        msg_type: MessageType = MessageType.BROADCAST,
        priority: int = 0,
    ) -> bool:
        """Broadcast a message to all agents."""
        return await self.send_message("ALL", content, msg_type, priority)

    async def escalate(
        self,
        content: str,
        patient_id: str | None = None,
        priority: int = 5,
    ) -> bool:
        """Escalate an issue to CMO Oversight."""
        return await self.send_message(
            AgentType.CMO_OVERSIGHT,
            content,
            MessageType.ALERT,
            priority,
            patient_id,
        )

    def get_recent_actions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent action history."""
        return [a.to_dict() for a in self._action_history[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Agent performance statistics."""
        return {
            "agent_type": self.agent_type.value,
            "role": self.role,
            "total_tokens": self.total_tokens,
            "actions_taken": self.actions_taken,
            "messages_sent": self.messages_sent,
            "inbox_size": len(self._inbox),
            "mock_llm": self.mock_llm,
        }

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.total_tokens = 0
        self.actions_taken = 0
        self.messages_sent = 0
        self._action_history.clear()
        self._inbox.clear()

    # ── LLM Integration ─────────────────────────────────

    async def _call_llm(
        self,
        prompt: str,
        state_context: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Call the LLM with the agent's system prompt + context.

        Returns parsed JSON response from the model.
        """
        full_prompt = f"""
{state_context}

---

{prompt}

Respond with a JSON object containing:
- "actions": list of actions to take, each with "action_type", "target_id", "priority", "reasoning"
- "messages": list of messages to send, each with "to_agent", "content", "msg_type", "priority"
"""
        t0 = time.perf_counter()
        
        try:
            # Logic for Gemini
            if "gemini" in self.model_name.lower():
                import google.generativeai as genai
                
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self.system_prompt,
                    generation_config={
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                    },
                )
                
                response = await asyncio.to_thread(
                    model.generate_content, full_prompt
                )
                text = response.text
                
            # Logic for Ollama / Local LLM
            else:
                url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": temperature,
                    }
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("response", "{}")

            elapsed = time.perf_counter() - t0
            tokens = len(text.split()) * 2
            self.total_tokens += tokens

            logger.debug(
                "Agent %s LLM call (%s): %d tokens, %.1fs",
                self.agent_type.value, self.model_name, tokens, elapsed,
            )

            return json.loads(text)
            
        except Exception as e:
            logger.warning("LLM call failed for %s (%s): %s", 
                        self.agent_type.value, self.model_name, e)
            return {"actions": [], "messages": []}

    # ── Internal ─────────────────────────────────────────

    async def _on_message(self, message: AgentMessage) -> None:
        """Handle incoming message from bus."""
        self._inbox.append(message)

    def _build_state_context(self, state: EnvironmentState) -> str:
        """Build a concise state summary for LLM context window."""
        stats = state.to_json()["stats"]
        patients_summary = []
        for p in state.patients[:20]:
            patients_summary.append(
                f"  - [{p.id}] {p.name} ({p.age}y) | {p.condition} | "
                f"Status: {p.status.value} | Triage: {p.triage_score} | "
                f"Ward: {p.ward.value} | ICU: {p.icu_required}"
            )

        resources = state.resources.to_dict()
        policies = [
            f"  - {p.name} v{p.version} ({'ACTIVE' if p.is_active else 'SUSPENDED'})"
            for p in state.active_policies.values()
        ]

        return f"""
## Current Hospital State — Step {state.step_count}

### Crisis: {state.crisis.name} ({state.crisis.type.value})
Severity: {state.crisis.severity} | Incoming rate: {state.crisis.incoming_rate}/step

### Statistics
- Alive: {stats['alive_count']} | Critical: {stats['critical_count']} | Deceased: {stats['deceased_count']}
- Survival Rate: {stats['survival_rate']:.1%}
- ICU Occupancy: {stats['icu_occupancy']:.1%}

### Resources
- ICU Beds: {resources['icu_beds_occupied']}/{resources['icu_beds_total']}
- Ventilators: {resources['ventilators_in_use']}/{resources['ventilators_total']}
- Staff Ratio: {resources['staff_ratio']:.1%}
- Equipment: {resources['equipment_status']:.1%}

### Patients (top 20)
{chr(10).join(patients_summary) if patients_summary else '  (none)'}

### Active Policies
{chr(10).join(policies) if policies else '  (none)'}

### Pending Patients: {len(state.pending_patients)}
""".strip()
