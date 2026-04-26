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
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
import httpx

from pydantic import BaseModel

from triage.agents.message_bus import MessageBus
from triage.agents.model_router import AGENT_MODEL_TIER, ModelRouter, ModelTier
try:
    from triage.agents.strategy_memory import StrategyMemory
except Exception:  # pragma: no cover - optional ChromaDB dependency
    class StrategyMemory:
        def get_strategy_prompt(self, *args, **kwargs) -> str:
            return ""
        def query_lessons(self, *args, **kwargs) -> list[dict[str, Any]]:
            return []
from triage.env.state import (
    AgentAction,
    AgentMessage,
    AgentType,
    ActionType,
    EnvironmentState,
    MessageType,
)
from triage.agents.tools import AGENT_TOOLS
from triage.agents.tool_validator import ToolValidationLayer, ValidatedAction, ValidationError

logger = logging.getLogger(__name__)


MAX_TOKENS_BY_AGENT: dict[str, int] = {
    "ambulance_dispatch": 400,
    "cmo_oversight": 800,
    "er_triage": 600,
    "infection_control": 600,
    "icu_management": 600,
    "pharmacy": 400,
    "hr_rostering": 400,
    "it_systems": 400,
    "blood_bank": 400,
    "ethics_committee": 800,
}

THINKING_ENABLED_BY_AGENT: dict[str, bool] = {
    "ambulance_dispatch": False,
    "cmo_oversight": True,
    "er_triage": False,
    "infection_control": False,
    "icu_management": False,
    "pharmacy": False,
    "hr_rostering": False,
    "it_systems": False,
    "blood_bank": False,
    "ethics_committee": True,
}


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
        self.model_name = model_name or os.getenv("OLLAMA_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.router = ModelRouter.get_instance()
        if not mock_llm and not self.router.initialized:
            try:
                ModelRouter.initialize_from_env()
            except Exception:
                logger.exception("ModelRouter initialization failed; %s will use rule-based mode", agent_type.value)
        self._loaded_model = self.router.get_model_for_agent(agent_type)
        self.mock_llm = mock_llm or self.router.mode == "mock"

        # From config/agents.yaml
        self.system_prompt: str = config.get("system_prompt", "")
        self.tools: list[dict[str, Any]] = config.get("tools", [])
        self.role: str = config.get("role", "agent")
        self.priority: int = config.get("priority", 5)
        self.max_tokens: int = int(
            config.get(
                "max_tokens",
                config.get("max_tokens_per_step", MAX_TOKENS_BY_AGENT.get(agent_type.value, 512)),
            )
        )
        self.think: bool = bool(config.get("think", THINKING_ENABLED_BY_AGENT.get(agent_type.value, False)))

        # Tools setup
        self.tool_stats = {}
        self.validator = ToolValidationLayer()

        # Tracking
        self.total_tokens: int = 0
        self.actions_taken: int = 0
        self.messages_sent: int = 0
        self._action_history: list[AgentAction] = []
        self._inbox: list[AgentMessage] = []
        self.memory = StrategyMemory()

        # Register on message bus
        self.bus.subscribe(agent_type, self._on_message)
        logger.info(
            "Initialized %s agent: mode=%s model=%s think=%s mock_llm=%s",
            agent_type.value,
            self.router.mode,
            self._loaded_model.model_id if self._loaded_model else "mock/ollama",
            self.think,
            self.mock_llm,
        )

    # ── Abstract Methods ─────────────────────────────────

    @abstractmethod
    async def decide(
        self,
        state: EnvironmentState,
        inbox: list[AgentMessage],
    ) -> list[BaseModel]:
        """Decide what actions to take given current state and messages.

        Args:
            state: Current environment state.
            inbox: Messages received since last decision.

        Returns:
            List of ToolSchema objects to execute (may be empty).
        """
        ...

    @abstractmethod
    def _rule_based_decision(
        self,
        state: EnvironmentState,
        inbox: list[AgentMessage],
    ) -> list[BaseModel]:
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
                tool_calls = self._rule_based_decision(state, inbox)
            else:
                tool_calls = await self.decide(state, inbox)
        except Exception:
            logger.exception("Agent %s decision failed, using fallback", self.agent_type.value)
            tool_calls = self._rule_based_decision(state, inbox)

        actions = []
        for tool in tool_calls:
            if isinstance(tool, AgentAction):
                self.actions_taken += 1
                self._action_history.append(tool)
                actions.append(tool)
                continue
            tool_name = tool.__class__.__name__
            val_res = self.validator.validate(tool_name, tool.model_dump(), state)
            if isinstance(val_res, ValidatedAction):
                action = AgentAction(
                    agent_type=self.agent_type,
                    action_type=val_res.action_type,
                    target_id=val_res.target_id,
                    priority=val_res.priority,
                    reasoning=val_res.reasoning,
                )
                
                # Special handling for SEND_MESSAGE action
                if val_res.action_type == ActionType.SEND_MESSAGE:
                    asyncio.ensure_future(self.send_message(
                        to=tool.to_agent, content=tool.content,
                        msg_type=MessageType.ACTION, priority=tool.urgency
                    ))
                elif val_res.action_type == ActionType.ESCALATE_TO_CMO:
                    asyncio.ensure_future(self.escalate(
                        content=tool.summary, patient_id=getattr(tool, "patient_id", None), priority=tool.urgency
                    ))
                else:
                    self.actions_taken += 1
                    self._action_history.append(action)
                    actions.append(action)

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
            "tool_usage": self.tool_stats,
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
        crisis_type: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Call the LLM with the agent's system prompt + context.

        Returns parsed JSON response from the model.
        """
        memory_block = self._build_memory_context(
            state_context=state_context,
            crisis_type=crisis_type,
        )
        full_prompt = f"""
{self.system_prompt}

{state_context}

{memory_block}

---

{prompt}

Respond with a JSON object containing:
- "actions": list of actions to take, each with "action_type", "target_id", "priority", "reasoning"
- "messages": list of messages to send, each with "to_agent", "content", "msg_type", "priority"
"""
        t0 = time.perf_counter()
        try:
            if self.router.mode == "hf" and self._loaded_model:
                text, tokens = await asyncio.to_thread(
                    self._call_hf,
                    self.system_prompt,
                    full_prompt,
                    temperature,
                )
            elif self.router.mode == "ollama":
                text, tokens = await self._call_ollama(self.system_prompt, full_prompt, temperature)
            else:
                return {"actions": [], "messages": []}

            elapsed = time.perf_counter() - t0
            self.total_tokens += tokens
            logger.debug(
                "Agent %s %s LLM call completed in %.2fs with %s tokens",
                self.agent_type.value,
                self.router.mode,
                elapsed,
                tokens,
            )

            return self._parse_llm_json(text)

        except Exception as e:
            logger.warning(
                "LLM call failed for %s: %s",
                self.agent_type.value, e,
            )
            return {"actions": [], "messages": []}

    def _build_memory_context(self, state_context: str, crisis_type: str | None = None) -> str:
        """Retrieve strategy lessons and format them for prompt injection."""
        try:
            lessons = self.memory.query_lessons(
                self.agent_type.value,
                current_context=state_context,
                top_k=3,
                crisis_type=crisis_type,
            )
        except Exception:
            logger.debug("Strategy memory unavailable for %s", self.agent_type.value, exc_info=True)
            lessons = []

        if not lessons:
            return "## Strategy Memory\n(no prior lessons found)"

        lines = ["## Strategy Memory (top lessons)"]
        for idx, lesson in enumerate(lessons, start=1):
            lines.append(
                (
                    f"{idx}. Action: {lesson.get('action_taken', 'n/a')} | "
                    f"Outcome: {lesson.get('outcome', 'n/a')} | "
                    f"Reward: {float(lesson.get('reward_delta', lesson.get('reward', 0.0))):.2f}"
                )
            )
        lines.append("Use these lessons only when they fit the current context.")
        return "\n".join(lines)

    def _call_hf(self, system_prompt: str, user_prompt: str, temperature: float) -> tuple[str, int]:
        """Run inference against the HuggingFace model assigned to this agent."""
        import torch

        loaded = self._loaded_model
        if loaded is None:
            return "{}", 0

        tokenizer = loaded.tokenizer
        model = loaded.model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.think,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            text = f"<|system|>{system_prompt}\n<|user|>{user_prompt}\n<|assistant|>"

        inputs = tokenizer(text, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._strip_thinking(response), int(len(new_tokens))

    async def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> tuple[str, int]:
        """Run inference against an Ollama fallback model."""
        tier = AGENT_MODEL_TIER.get(self.agent_type.value, ModelTier.CLINICAL)
        model_name = (
            os.getenv("OLLAMA_CLINICAL_MODEL", os.getenv("OLLAMA_BASE_MODEL", "qwen3:8b"))
            if tier == ModelTier.CLINICAL
            else os.getenv("OLLAMA_OPERATIONS_MODEL", "qwen3:4b")
        )
        base_url = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
        base_url = base_url.removesuffix("/api/generate").removesuffix("/api/chat").rstrip("/")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_ctx": 2048,
                "num_predict": self.max_tokens,
                "think": self.think,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        text = data.get("message", {}).get("content", "{}")
        return self._strip_thinking(text), int(data.get("eval_count", len(text.split()) * 2))

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
        """Parse model output into the existing action/message response shape."""
        cleaned = self._strip_thinking(text)
        cleaned = re.sub(r"```(?:json)?", "", cleaned).strip("` \n")
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                logger.warning("Agent %s produced non-JSON output: %s", self.agent_type.value, cleaned[:200])
                return {"actions": [], "messages": []}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Agent %s produced invalid JSON: %s", self.agent_type.value, cleaned[:200])
                return {"actions": [], "messages": []}

        if isinstance(parsed, list):
            parsed = {"actions": parsed, "messages": []}
        if not isinstance(parsed, dict):
            return {"actions": [], "messages": []}
        parsed.setdefault("actions", [])
        parsed.setdefault("messages", [])
        return parsed

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _coerce_target_id(self, value: Any, state: EnvironmentState) -> int:
        """Convert model target IDs into simulator integer target indices."""
        if value is None or value == "":
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        text = str(value)
        for index, patient in enumerate(getattr(state, "patients", [])):
            if patient.id == text:
                return index
        return 0

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
