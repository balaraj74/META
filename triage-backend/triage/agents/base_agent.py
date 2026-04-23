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
from typing import Any

from pydantic import BaseModel

from triage.agents.message_bus import MessageBus
from triage.agents.strategy_memory import StrategyMemory
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
    ) -> None:
        self.agent_type = agent_type
        self.config = config
        self.bus = message_bus
        self.mock_llm = mock_llm

        # From config/agents.yaml
        self.system_prompt: str = config.get("system_prompt", "")
        self.tools: list[dict[str, Any]] = config.get("tools", [])
        self.role: str = config.get("role", "agent")
        self.priority: int = config.get("priority", 5)

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
        self.tool_stats.clear()

    # ── LLM Integration ─────────────────────────────────

    def _get_anthropic_tools(self) -> list[dict[str, Any]]:
        tools = []
        for SchemaClass in AGENT_TOOLS.get(self.agent_type, []):
            schema = SchemaClass.model_json_schema()
            tools.append({
                "name": SchemaClass.__name__,
                "description": schema.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            })
        return tools

    async def _call_llm(
        self,
        prompt: str,
        state_context: str,
        state: EnvironmentState,
        temperature: float = 0.3,
    ) -> list[BaseModel]:
        """Call the LLM with the agent's system prompt + context using structured tools.

        Contains retry loop for tool validation errors.
        """
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))
            
            tools = self._get_anthropic_tools()
            tool_classes = {cls.__name__: cls for cls in AGENT_TOOLS.get(self.agent_type, [])}

            if not self.mock_llm:
                past_lessons = self.memory.query_lessons(self.agent_type.value, state_context, top_k=3)
                if past_lessons:
                    lessons_block = "## Past Lessons\n"
                    for i, lesson in enumerate(past_lessons, 1):
                        lessons_block += f"{i}. Action: {lesson['action_taken']}\n   Outcome: {lesson['outcome']}\n   Reward Delta: {lesson['reward_delta']:.2f}\n"
                    full_prompt = f"{state_context}\n\n{lessons_block}\n\n---\n\n{prompt}"
                else:
                    full_prompt = f"{state_context}\n\n---\n\n{prompt}"
            else:
                full_prompt = f"{state_context}\n\n---\n\n{prompt}"

            messages = [{"role": "user", "content": full_prompt}]
            
            valid_tool_schemas = []

            for attempt in range(3):
                t0 = time.perf_counter()
                response = await client.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                    system=self.system_prompt,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=self.config.get("max_tokens_per_step", 1000)
                )
                elapsed = time.perf_counter() - t0
                
                self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
                logger.debug("Agent %s LLM call: %.1fs", self.agent_type.value, elapsed)

                retry_messages = []
                all_tools_valid = True
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_kwargs = block.input
                        
                        if tool_name not in self.tool_stats:
                            self.tool_stats[tool_name] = {"called": 0, "validated": 0, "rejected": 0}
                        self.tool_stats[tool_name]["called"] += 1
                        
                        val_res = self.validator.validate(tool_name, tool_kwargs, state)
                        
                        if isinstance(val_res, ValidationError):
                            self.tool_stats[tool_name]["rejected"] += 1
                            all_tools_valid = False
                            retry_messages.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Validation Error: {val_res.reason}. Please correct your arguments and try again.",
                                "is_error": True
                            })
                        else:
                            self.tool_stats[tool_name]["validated"] += 1
                            if tool_name in tool_classes:
                                valid_tool_schemas.append(tool_classes[tool_name](**tool_kwargs))
                            retry_messages.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "Success."
                            })

                if response.stop_reason != "tool_use" or (all_tools_valid and not retry_messages):
                    break
                
                if all_tools_valid:
                    break
                
                valid_tool_schemas.clear()  # Clear to not mix attempts
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": retry_messages})
                
            return valid_tool_schemas

        except Exception as e:
            logger.warning("LLM call failed for %s: %s", self.agent_type.value, e)
            return []

    # ── Internal ─────────────────────────────────────────

    async def _on_message(self, message: AgentMessage) -> None:
        """Handle incoming message from bus."""
        self._inbox.append(message)
        if hasattr(self, "bus") and self.bus:
            # Need to create task if self.bus.ack is async
            if asyncio.iscoroutinefunction(self.bus.ack):
                await self.bus.ack(message.id)
            else:
                self.bus.ack(message.id)

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
