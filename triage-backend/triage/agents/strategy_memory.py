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

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

class StrategyMemory:
    """Persistent strategy memory using ChromaDB for semantic RAG."""

    def __init__(self, storage_path: str = "./data/chroma_db/") -> None:
        os.makedirs(storage_path, exist_ok=True)
        self._storage_path = storage_path
        self._client = chromadb.PersistentClient(path=storage_path)
        self._ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self._collections: dict[str, Any] = {}

    def _get_collection(self, agent_type: str) -> Any:
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
        col = self._get_collection(agent_type)
        data = col.get()
        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []
            
        sorted_md = sorted(metadatas, key=lambda x: x.get("reward_delta", 0.0), reverse=True)
        return sorted_md[:limit]
