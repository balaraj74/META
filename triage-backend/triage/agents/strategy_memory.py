"""
StrategyMemory — cross-episode self-improvement through strategy tracking using ChromaDB RAG.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Any
from collections import Counter
from pathlib import Path

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    chromadb = None
    SentenceTransformerEmbeddingFunction = None

logger = logging.getLogger(__name__)

class StrategyMemory:
    """Persistent strategy memory using ChromaDB for semantic RAG."""

    def __init__(self, storage_path: str = "./data/chroma_db/") -> None:
        self._storage_path = storage_path
        self._collections: dict[str, Any] = {}
        self._memory_file = self._resolve_memory_file(storage_path)
        self._lessons: list[dict[str, Any]] = []

        if chromadb is None:
            self._load_file_memory()
            self._client = None
            self._ef = None
            return

        os.makedirs(storage_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=storage_path)
        self._ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    @staticmethod
    def _resolve_memory_file(storage_path: str) -> Path:
        path = Path(storage_path)
        if path.suffix == ".json":
            return path
        return path / "strategy_memory.json"

    def _load_file_memory(self) -> None:
        if not self._memory_file.exists():
            self._lessons = []
            return
        try:
            self._lessons = json.loads(self._memory_file.read_text())
        except Exception:
            self._lessons = []

    def record(
        self,
        agent_type: str,
        crisis_type: str,
        description: str,
        episode: int,
        reward: float,
        success: bool,
    ) -> None:
        self.add_lesson(
            agent_type=agent_type,
            lesson={
                "crisis_type": crisis_type,
                "description": description,
                "episode": episode,
                "reward": reward,
                "success": success,
                "context": crisis_type,
                "action_taken": description,
                "outcome": "success" if success else "failure",
                "reward_delta": reward,
                "step": episode,
            },
        )

    def save(self) -> None:
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._memory_file.write_text(json.dumps(self._lessons, indent=2))

    def get_strategy_prompt(self, agent_type: str, crisis_type: str) -> str:
        lessons = self.query_lessons(agent_type, current_context=crisis_type, top_k=3, crisis_type=crisis_type)
        return "\n".join(str(item.get("action_taken", item.get("description", ""))) for item in lessons)

    def _get_collection(self, agent_type: str) -> Any:
        if chromadb is None:
            raise RuntimeError("ChromaDB is not installed")
        # agent_type could be an enum, so convert to string. Ensure no invalid characters.
        cleaned_type = str(getattr(agent_type, "value", agent_type)).lower().replace("_", "-")
        collection_name = f"triage-memory-{cleaned_type}"
        
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._ef
            )
        return self._collections[collection_name]

    def add_lesson(self, agent_type: str, lesson: dict[str, Any]) -> None:
        """Upsert a lesson to the agent's ChromaDB collection."""
        normalized_agent = str(getattr(agent_type, "value", agent_type))
        crisis_type = str(lesson.get("crisis_type", "")).strip().lower() or "unknown"
        context_str = str(lesson.get("context", ""))
        action_str = str(lesson.get("action_taken", ""))
        outcome_str = str(lesson.get("outcome", lesson.get("description", "")))
        step = int(lesson.get("step", 0))
        episode = int(lesson.get("episode", step))
        reward_delta = float(lesson.get("reward_delta", lesson.get("reward", 0.0)))
        success = bool(lesson.get("success", reward_delta >= 0))
        severity = str(lesson.get("severity", "unknown"))
        staff_ratio = float(lesson.get("staff_ratio", 0.0))
        icu_occupancy = float(lesson.get("icu_occupancy", 0.0))

        lesson_id = self._stable_lesson_id(
            normalized_agent,
            crisis_type,
            action_str,
            step=step,
            episode=episode,
        )

        document = (
            f"Agent: {normalized_agent}\n"
            f"Crisis: {crisis_type}\n"
            f"Severity: {severity}\n"
            f"Context: {context_str}\n"
            f"Resource pressure: ICU={icu_occupancy:.2f}, Staff={staff_ratio:.2f}\n"
            f"Action: {action_str}\n"
            f"Outcome: {outcome_str}\n"
            f"Reward delta: {reward_delta:.3f}"
        )

        metadata = {
            "id": lesson_id,
            "agent_type": normalized_agent,
            "description": str(lesson.get("description", action_str)),
            "context": context_str,
            "action_taken": action_str,
            "outcome": outcome_str,
            "reward_delta": reward_delta,
            "reward": reward_delta,
            "success": success,
            "success_rate": 1.0 if success else 0.0,
            "times_used": int(lesson.get("times_used", 1)),
            "crisis_type": crisis_type,
            "severity": severity,
            "staff_ratio": staff_ratio,
            "icu_occupancy": icu_occupancy,
            "episode": episode,
            "step": step,
        }

        if chromadb is None:
            existing_idx = next(
                (idx for idx, entry in enumerate(self._lessons) if entry.get("id") == lesson_id),
                None,
            )
            if existing_idx is not None:
                self._lessons[existing_idx] = metadata
            else:
                self._lessons.append(metadata)
            return

        col = self._get_collection(normalized_agent)
        col.upsert(ids=[lesson_id], documents=[document], metadatas=[metadata])
        logger.debug(f"Added memory to {agent_type}: {action_str} (reward: {metadata['reward_delta']})")

    def query_lessons(
        self,
        agent_type: str,
        current_context: str,
        top_k: int = 3,
        crisis_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k semantically similar past lessons."""
        normalized_agent = str(getattr(agent_type, "value", agent_type))
        normalized_crisis = str(crisis_type).strip().lower() if crisis_type else None

        if chromadb is None:
            rows = [
                lesson for lesson in self._lessons
                if lesson.get("agent_type") == normalized_agent
                and (normalized_crisis is None or lesson.get("crisis_type") == normalized_crisis)
            ]
            ranked = sorted(
                rows,
                key=lambda item: self._rerank_score(item, semantic_similarity=1.0),
                reverse=True,
            )
            return ranked[:top_k]

        col = self._get_collection(normalized_agent)

        # If collection is empty, return early
        if col.count() == 0:
            return []

        where = {"crisis_type": normalized_crisis} if normalized_crisis else None
        raw_k = min(max(top_k * 3, top_k), col.count())
        results = col.query(
            query_texts=[current_context],
            where=where,
            n_results=raw_k,
        )
        if not results or not results.get("metadatas"):
            return []

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, metadata in enumerate(metadatas):
            distance = distances[idx] if idx < len(distances) else 1.0
            semantic_similarity = max(0.0, 1.0 - float(distance))
            scored.append((self._rerank_score(metadata, semantic_similarity), metadata))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def forget_bad_lessons(self, agent_type: str, reward_threshold: float = -0.2) -> int:
        """Delete entries where reward_delta < threshold."""
        if chromadb is None:
            before = len(self._lessons)
            self._lessons = [
                lesson for lesson in self._lessons
                if not (
                    lesson.get("agent_type") == str(agent_type)
                    and float(lesson.get("reward_delta", lesson.get("reward", 0.0))) < reward_threshold
                )
            ]
            return before - len(self._lessons)
        col = self._get_collection(agent_type)
        if col.count() == 0:
            return 0
            
        initial_count = col.count()
        col.delete(where={"reward_delta": {"$lt": reward_threshold}})
        final_count = col.count()
        
        removed = initial_count - final_count
        if removed > 0:
            logger.info(f"Forgot {removed} bad lessons from {agent_type}")
        return removed

    def summarize(self, agent_type: str) -> dict[str, Any]:
        """Return count, avg reward_delta, and top crisis_type seen."""
        if chromadb is None:
            metadatas = [
                lesson for lesson in self._lessons
                if lesson.get("agent_type") == str(agent_type)
            ]
            count = len(metadatas)
            if count == 0:
                return {"count": 0, "avg_reward_delta": 0.0, "top_crisis_type": None}
            total_reward = sum(float(m.get("reward_delta", m.get("reward", 0.0))) for m in metadatas)
            crises = [m.get("crisis_type", "") for m in metadatas if m.get("crisis_type")]
            return {
                "count": count,
                "avg_reward_delta": total_reward / count,
                "top_crisis_type": Counter(crises).most_common(1)[0][0] if crises else None,
            }
        col = self._get_collection(agent_type)
        data = col.get()
        metadatas = data.get("metadatas", [])
        
        count = len(metadatas)
        if count == 0:
            return {"count": 0, "avg_reward_delta": 0.0, "top_crisis_type": None}
            
        total_reward = sum(m.get("reward_delta", 0.0) for m in metadatas)
        avg_reward = total_reward / count
        
        crises = [m.get("crisis_type", "") for m in metadatas if m.get("crisis_type")]
        top_crisis = Counter(crises).most_common(1)[0][0] if crises else None
        
        return {
            "count": count,
            "avg_reward_delta": avg_reward,
            "top_crisis_type": top_crisis
        }

    def get_best_lessons(self, agent_type: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return the top lessons by reward_delta."""
        if chromadb is None:
            lessons = [
                lesson for lesson in self._lessons
                if lesson.get("agent_type") == str(agent_type)
            ]
            return sorted(
                lessons,
                key=lambda item: float(item.get("reward_delta", item.get("reward", 0.0))),
                reverse=True,
            )[:limit]
        col = self._get_collection(agent_type)
        data = col.get()
        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []
            
        sorted_md = sorted(metadatas, key=lambda x: x.get("reward_delta", 0.0), reverse=True)
        return sorted_md[:limit]

    def get_all(self) -> dict[str, list[dict[str, Any]]]:
        """Return lessons grouped by `agent_type:crisis_type`."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        if chromadb is None:
            for lesson in self._lessons:
                agent = str(lesson.get("agent_type", "unknown"))
                crisis = str(lesson.get("crisis_type", "unknown"))
                grouped.setdefault(f"{agent}:{crisis}", []).append(lesson)
            return grouped

        for collection_obj in self._client.list_collections():
            collection_name = getattr(collection_obj, "name", "")
            if not collection_name.startswith("triage-memory-"):
                continue
            collection = self._client.get_collection(name=collection_name)
            data = collection.get()
            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])
            for idx, md in enumerate(metadatas):
                entry = dict(md)
                entry.setdefault("id", ids[idx] if idx < len(ids) else str(uuid.uuid4()))
                agent = str(entry.get("agent_type", collection_name.replace("triage-memory-", "").replace("-", "_")))
                crisis = str(entry.get("crisis_type", "unknown"))
                grouped.setdefault(f"{agent}:{crisis}", []).append(entry)
        return grouped

    def _stable_lesson_id(
        self,
        agent_type: str,
        crisis_type: str,
        action_taken: str,
        *,
        step: int,
        episode: int,
    ) -> str:
        payload = f"{agent_type}|{crisis_type}|{action_taken.strip().lower()}|{episode}|{step}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]

    def _rerank_score(self, lesson: dict[str, Any], semantic_similarity: float) -> float:
        reward = float(lesson.get("reward_delta", lesson.get("reward", 0.0)))
        success_rate = float(lesson.get("success_rate", 1.0 if reward >= 0 else 0.0))
        recency_marker = int(lesson.get("episode", lesson.get("step", 0)))
        recency_score = min(1.0, recency_marker / 1000.0) if recency_marker > 0 else 0.0
        reward_score = max(-1.0, min(1.0, reward / 5.0))
        return (
            0.55 * semantic_similarity
            + 0.30 * reward_score
            + 0.10 * success_rate
            + 0.05 * recency_score
        )
