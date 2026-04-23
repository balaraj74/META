"""
MessageBus - priority-aware async hierarchical routing system.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Coroutine, NamedTuple

from triage.agents.routing_rules import DeadlockDetector, RoutingRuleSet
from triage.env.state import AgentMessage, AgentType, MessageType

logger = logging.getLogger(__name__)

MessageCallback = Callable[[AgentMessage], Coroutine[Any, Any, None]]

class QueueItem(NamedTuple):
    priority: int
    timestamp: float
    message: AgentMessage

    # Tuples are sorted item by item. priority is inserted as inverted (-priority).
    # so lower negative value (higher priority) comes first.

class MessageBus:
    """Async priority-based message bus with routing rules and acknowledgment."""

    def __init__(self, token_budget: int = 50_000) -> None:
        self._inboxes: dict[str, asyncio.PriorityQueue] = defaultdict(asyncio.PriorityQueue)
        self._subscribers: dict[str, list[MessageCallback]] = defaultdict(list)
        self._broadcast_subscribers: list[MessageCallback] = []
        
        self._history: list[AgentMessage] = []
        self._token_budget = token_budget
        self._tokens_used = 0
        self._message_count = 0
        
        self.routing_rules = RoutingRuleSet()
        self.deadlock_detector = DeadlockDetector()
        
        self._unacked: dict[str, AgentMessage] = {}
        self._message_step: dict[str, int] = {} # records the step when message was routed
        self._step = 0
        
        self._workers: list[asyncio.Task] = []
        self._lock = asyncio.Lock()

    # ── Subscription ─────────────────────────────────────

    def subscribe(self, agent_type: AgentType | str, callback: MessageCallback) -> None:
        """Subscribe an agent to receive direct messages via callback execution."""
        at = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
        if at not in self._subscribers or not self._subscribers[at]:
            self._subscribers[at].append(callback)
            # Start worker for this queue if there isn't one
            try:
                loop = asyncio.get_running_loop()
                worker = loop.create_task(self._queue_worker(at))
                self._workers.append(worker)
            except RuntimeError:
                pass # not in async context yet
        else:
            self._subscribers[at].append(callback)

    def subscribe_broadcast(self, callback: MessageCallback) -> None:
        self._broadcast_subscribers.append(callback)

    def unsubscribe_all(self, agent_type: AgentType | str) -> None:
        at = agent_type.value if hasattr(agent_type, "value") else str(agent_type)
        self._subscribers.pop(at, None)

    async def _queue_worker(self, agent: str) -> None:
        """Consumes priority queue for the agent and triggers callbacks."""
        queue = self._inboxes[agent]
        while True:
            try:
                item: QueueItem = await queue.get()
                msg = item.message
                callbacks = self._subscribers.get(agent, [])
                for cb in callbacks:
                    try:
                        await cb(msg)
                    except Exception:
                        logger.exception(f"Error handling callback for {agent}")
                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error for {agent}: {e}")

    # ── Sending ──────────────────────────────────────────

    async def send(self, message: AgentMessage) -> bool:
        return await self.publish(message)

    async def publish(self, message: AgentMessage) -> bool:
        """Publish a message (adapter shim mapping to new routing)."""
        async with self._lock:
            if self._tokens_used + message.token_count > self._token_budget:
                return False
            self._tokens_used += message.token_count
            self._message_count += 1
            self._history.append(message)
            
        suppress, auto_forwards = self.routing_rules.process_rules(message, self._step)
        if suppress:
            return True

        if message.to_agent == "ALL" or message.msg_type == MessageType.BROADCAST:
            # Enqueue everywhere
            for agent in [a.value for a in AgentType]:
                if hasattr(message.from_agent, "value") and agent == message.from_agent.value:
                    continue
                await self._enqueue(agent, message)
            # Also notify broadcast callbacks directly
            for cb in self._broadcast_subscribers:
                await cb(message)
        else:
            target = message.to_agent.value if hasattr(message.to_agent, "value") else str(message.to_agent)
            await self._enqueue(target, message)

        for fwd in auto_forwards:
            await self._enqueue(fwd.to_agent.value if hasattr(fwd.to_agent, "value") else str(fwd.to_agent), fwd)

        return True

    async def _enqueue(self, target: str, message: AgentMessage) -> None:
        """Place message on the priority queue for target agent."""
        item = QueueItem(priority=-message.priority, timestamp=time.perf_counter(), message=message)
        self._unacked[message.id] = message
        self._message_step[message.id] = self._step
        await self._inboxes[target].put(item)
        
        # Ensure worker is up just in case
        if target in self._subscribers and target not in [str(w.get_coro().cr_code) for w in self._workers if w]:
             # weak check, but okay if unstarted
             pass

    async def send_and_wait(self, message: AgentMessage, timeout: float = 30.0) -> AgentMessage | None:
        message.requires_response = True
        response_event = asyncio.Event()
        response_msg: list[AgentMessage] = []

        from_str = message.from_agent.value if hasattr(message.from_agent, "value") else str(message.from_agent)
        to_str = message.to_agent.value if hasattr(message.to_agent, "value") else str(message.to_agent)

        self.deadlock_detector.register_wait(from_str, to_str)

        async def _response_handler(msg: AgentMessage) -> None:
            if msg.msg_type == MessageType.RESPONSE and msg.action_id == message.id:
                response_msg.append(msg)
                response_event.set()

        self._subscribers[from_str].append(_response_handler)
        
        # Deadlock tiebreaker
        if self.deadlock_detector.check_deadlock(from_str, to_str):
            breaker_msg = self.deadlock_detector.resolve_deadlock(from_str, to_str)
            if breaker_msg:
                await self.send(breaker_msg)

        sent = await self.send(message)
        if not sent:
            self.deadlock_detector.release_wait(from_str, to_str)
            return None

        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            return response_msg[0] if response_msg else None
        except asyncio.TimeoutError:
            return None
        finally:
            self.deadlock_detector.release_wait(from_str, to_str)
            try:
                self._subscribers[from_str].remove(_response_handler)
            except ValueError:
                pass

    # ── Acknowledgment & Time ────────────────────────────

    async def ack(self, message_id: str) -> None:
        """Acknowledge a message removing it from unacked tracker."""
        if message_id in self._unacked:
            del self._unacked[message_id]
            self._message_step.pop(message_id, None)

    async def tick(self) -> None:
        """Advance the internal clock, requeueing unacked messages."""
        self._step += 1
        requeue = []
        for msg_id, msg in list(self._unacked.items()):
            step_sent = self._message_step.get(msg_id, self._step)
            if self._step - step_sent >= 3:
                # Re-queue
                msg.priority += 2
                self._message_step[msg_id] = self._step
                requeue.append(msg)
                
        for msg in requeue:
            target = msg.to_agent.value if hasattr(msg.to_agent, "value") else str(msg.to_agent)
            item = QueueItem(priority=-msg.priority, timestamp=time.perf_counter(), message=msg)
            await self._inboxes[target].put(item)
            logger.debug("Requeued unacked message %s with bumped priority %d", msg.id, msg.priority)

    # ── Query ────────────────────────────────────────────

    @property
    def history(self) -> list[AgentMessage]:
        return list(self._history)

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    def get_messages_for(self, agent_type: AgentType, limit: int = 50, msg_type: MessageType | None = None) -> list[AgentMessage]:
        msgs = [
            m for m in self._history
            if (
                (hasattr(m.to_agent, "value") and m.to_agent == agent_type)
                or m.to_agent == "ALL"
                or m.msg_type == MessageType.BROADCAST
            )
            and (msg_type is None or m.msg_type == msg_type)
        ]
        return msgs[-limit:]

    def get_conversation(self, agent_a: AgentType, agent_b: AgentType, limit: int = 20) -> list[AgentMessage]:
        msgs = [
            m for m in self._history
            if (
                (m.from_agent == agent_a and m.to_agent == agent_b)
                or (m.from_agent == agent_b and m.to_agent == agent_a)
            )
        ]
        return msgs[-limit:]

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = defaultdict(int)
        by_agent: dict[str, int] = defaultdict(int)
        route_pairs = []
        
        for m in self._history:
            by_type[m.msg_type.value] += 1
            sender = m.from_agent.value if hasattr(m.from_agent, "value") else str(m.from_agent)
            by_agent[sender] += 1
            target = m.to_agent.value if hasattr(m.to_agent, "value") else str(m.to_agent)
            route_pairs.append(f"{sender}->{target}")

        queue_depths = {k: v.qsize() for k, v in self._inboxes.items()}
        unacked_counts = len(self._unacked)
        top_routes = [r[0] for r in Counter(route_pairs).most_common(5)]

        return {
            "total_messages": self._message_count,
            "tokens_used": self._tokens_used,
            "queue_depths": queue_depths,
            "unacked_count": unacked_counts,
            "top_routes": top_routes,
            "by_type": dict(by_type),
            "by_agent": dict(by_agent),
        }

    # ── Reset ────────────────────────────────────────────

    def reset(self) -> None:
        self._history.clear()
        self._tokens_used = 0
        self._message_count = 0
        self._unacked.clear()
        self._message_step.clear()
        for q in self._inboxes.values():
            while not q.empty():
                q.get_nowait()
                q.task_done()
        self._step = 0
