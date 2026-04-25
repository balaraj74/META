"""
StrategyMemory — cross-episode self-improvement through strategy tracking using ChromaDB RAG.
"""

from __future__ import annotations

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
        self._lessons.append(
            {
                "agent_type": agent_type,
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
            }
        )

    def save(self) -> None:
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._memory_file.write_text(json.dumps(self._lessons, indent=2))

    def get_strategy_prompt(self, agent_type: str, crisis_type: str) -> str:
        if chromadb is not None:
            lessons = self.get_best_lessons(agent_type)
            return "\n".join(str(item.get("action_taken", "")) for item in lessons)
        matches = [
            lesson["description"]
            for lesson in self._lessons
            if lesson.get("agent_type") == agent_type and lesson.get("crisis_type") == crisis_type
        ]
        return "\n".join(matches)

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
        if chromadb is None:
            self._lessons.append({"agent_type": str(agent_type), **lesson})
            return
        col = self._get_collection(agent_type)
        
        context_str = lesson.get("context", "")
        action_str = lesson.get("action_taken", "")
        # Embed context + action_taken
        document = f"Context: {context_str}\nAction: {action_str}"
        
        # Store full dict as metadata
        metadata = {
            "context": context_str,
            "action_taken": action_str,
            "outcome": lesson.get("outcome", ""),
            "reward_delta": float(lesson.get("reward_delta", 0.0)),
            "crisis_type": lesson.get("crisis_type", ""),
            "step": int(lesson.get("step", 0))
        }
        
        lesson_id = str(uuid.uuid4())
        col.upsert(
            ids=[lesson_id],
            documents=[document],
            metadatas=[metadata]
        )
        logger.debug(f"Added memory to {agent_type}: {action_str} (reward: {metadata['reward_delta']})")

    def query_lessons(self, agent_type: str, current_context: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve top-k semantically similar past lessons."""
        if chromadb is None:
            return [
                lesson for lesson in self._lessons
                if lesson.get("agent_type") == str(agent_type)
            ][:top_k]
        col = self._get_collection(agent_type)
        
        # If collection is empty, return early
        if col.count() == 0:
            return []
            
        results = col.query(
            query_texts=[current_context],
            n_results=min(top_k, col.count())
        )
        
        lessons = []
        if results and results.get("metadatas") and len(results["metadatas"]) > 0:
            for metadata in results["metadatas"][0]:
                lessons.append(metadata)
        return lessons

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
