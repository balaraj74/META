import logging
from collections import defaultdict
from typing import Any

from triage.env.state import AgentMessage, AgentType, MessageType

logger = logging.getLogger(__name__)


class RoutingRuleSet:
    def __init__(self):
        # Track message frequencies for loop detection
        # shape: {(from_agent, msg_type): [step_received, step_received, ...]}
        self._history: dict[tuple[Any, Any], list[int]] = defaultdict(list)
        self.cmo_type = AgentType.CMO_OVERSIGHT

    def process_rules(
        self, msg: AgentMessage, current_step: int
    ) -> tuple[bool, list[AgentMessage]]:
        """
        Evaluate routing rules.
        Returns:
            suppress (bool): If true, drop the message.
            auto_forwards (list): New message instances to forward.
        """
        # Rule 1: Escalation Loop
        key = (msg.from_agent, msg.msg_type)
        history = self._history[key]
        # Keep only history from past 10 steps
        history = [ts for ts in history if current_step >= ts and current_step - ts <= 10]
        history.append(current_step)
        self._history[key] = history

        if len(history) >= 3:
            logger.warning("ESCALATION_LOOP detected for %s. Suppressing message.", key)
            return True, []

        # Rule 2: Auto-forward CRITICAL to CMO
        forwards = []
        if msg.priority >= 9 and msg.to_agent != self.cmo_type:
            # create a copy for CMO
            import copy
            fwd = copy.copy(msg)
            # Assign new ID so it can be acked separately
            import uuid
            fwd.id = str(uuid.uuid4())
            fwd.to_agent = self.cmo_type
            forwards.append(fwd)

        return False, forwards


class DeadlockDetector:
    def __init__(self):
        # We track who is waiting on whom: {agent: waiting_on_agent_id}
        self.waiting: dict[Any, Any] = {}

    def register_wait(self, waiter: Any, target: Any):
        self.waiting[waiter] = target

    def release_wait(self, waiter: Any, target: Any = None):
        if waiter in self.waiting:
            if target is None or self.waiting[waiter] == target:
                del self.waiting[waiter]

    def check_deadlock(self, agent_a: Any, agent_b: Any) -> bool:
        """Check if mutual waiting."""
        return self.waiting.get(agent_a) == agent_b and self.waiting.get(agent_b) == agent_a

    def resolve_deadlock(self, agent_a: Any, agent_b: Any) -> AgentMessage | None:
        """Create a CMO escalation message to resolve tie."""
        import uuid
        from datetime import datetime, timezone
        from triage.env.state import AgentMessage, AgentType, MessageType, MessagePriority

        logger.warning("Deadlock detected between %s and %s. Breaking tie via CMO.", agent_a, agent_b)
        msg = AgentMessage(
            id=str(uuid.uuid4()),
            from_agent=AgentType.CMO_OVERSIGHT,
            to_agent=agent_a,  # arbitrary tie break
            content=f"Deadlock detected with {agent_b}. CMO overrides: proceed immediately.",
            msg_type=MessageType.OVERSIGHT,
            priority=MessagePriority.CRITICAL.value,
        )
        return msg
