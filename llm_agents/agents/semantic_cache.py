#!/usr/bin/env python3
"""
Semantic cache — embedding-based similarity matching for repeated queries.

Caches PI-GINOT prediction results so that semantically similar queries
(e.g., "analyze standard dogbone" vs "predict on a standard specimen")
can be served from cache without re-running inference.

FIX #6: Serializes custom_outputs to pickle-safe dicts (via model_dump)
before storing, since Pydantic objects and numpy arrays break raw pickle/json.
"""

import json
import os
import hashlib
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CacheEntry:
    """A single cache entry with embedding, query, and result."""
    query: str
    embedding: np.ndarray
    result: dict
    agent_name: str
    timestamp: float
    hit_count: int = 0
    geometry_hash: str = ""


class SemanticCache:
    """Embedding-based cache for PI-GINOT predictions.

    Uses cosine similarity between query embeddings to determine cache hits.
    Supports TTL expiration and geometry-aware hashing for exact matches.
    """

    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.92,
        max_entries: int = 200,
        ttl_seconds: float = 3600 * 24,  # 24 hours
        storage_dir: str = ".memory/cache",
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.storage_dir = storage_dir
        self._entries: List[CacheEntry] = []

        os.makedirs(storage_dir, exist_ok=True)

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a query text."""
        if self.embedding_model is None:
            return None
        try:
            # Supports both OpenAI and sentence-transformers style
            if hasattr(self.embedding_model, "embed_query"):
                vec = self.embedding_model.embed_query(text)
            else:
                vec = self.embedding_model.encode([text])[0]
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            print(f"[SemanticCache] Embedding failed: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _geometry_hash(self, geometry_params: Optional[dict]) -> str:
        """Deterministic hash for geometry parameters (exact match shortcut)."""
        if not geometry_params:
            return ""
        sorted_items = sorted(geometry_params.items())
        return hashlib.md5(json.dumps(sorted_items).encode()).hexdigest()[:12]

    def _evict_expired(self):
        """Remove expired entries."""
        now = time.time()
        self._entries = [
            e for e in self._entries
            if (now - e.timestamp) < self.ttl_seconds
        ]

    def _evict_lru(self):
        """Remove least-recently-used entries if over capacity."""
        if len(self._entries) > self.max_entries:
            # Sort by hit_count (ascending), then timestamp (ascending)
            self._entries.sort(key=lambda e: (e.hit_count, e.timestamp))
            self._entries = self._entries[-self.max_entries:]

    @staticmethod
    def _serialize_custom_outputs(outputs: list) -> list:
        """Convert Pydantic custom_outputs to serializable dicts.

        FIX #6: Pydantic objects and numpy arrays break when passed through
        pickle or json serialization. Convert to {__type__, data} format.
        """
        safe = []
        for out in outputs:
            if hasattr(out, "model_dump"):
                safe.append({
                    "__type__": type(out).__name__,
                    "data": out.model_dump(),
                })
            elif isinstance(out, dict):
                # Already serialized (e.g., from a previous cache round-trip)
                safe.append(out)
            else:
                safe.append({"__type__": "raw", "data": str(out)})
        return safe

    def lookup(
        self, query: str, geometry_params: Optional[dict] = None
    ) -> Tuple[bool, Optional[dict]]:
        """Check cache for a similar query.

        Returns:
            (hit, result) — hit is True if a cached result was found.
        """
        self._evict_expired()

        # Fast path: exact geometry hash match
        geo_hash = self._geometry_hash(geometry_params)
        if geo_hash:
            for entry in self._entries:
                if entry.geometry_hash == geo_hash:
                    entry.hit_count += 1
                    return True, entry.result

        # Semantic similarity path
        embedding = self._get_embedding(query)
        if embedding is None:
            return False, None

        best_sim = 0.0
        best_entry = None
        for entry in self._entries:
            if entry.embedding is not None:
                sim = self._cosine_similarity(embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        if best_sim >= self.similarity_threshold and best_entry is not None:
            best_entry.hit_count += 1
            return True, best_entry.result

        return False, None

    def store(
        self, query: str, result: dict, agent_name: str,
        geometry_params: Optional[dict] = None,
    ):
        """Store a new result in the cache.

        FIX #6: Serializes custom_outputs before storing to avoid
        numpy/Pydantic serialization issues on retrieval.
        """
        # Serialize custom outputs to safe dicts
        if "custom_outputs" in result and result["custom_outputs"]:
            result = {
                **result,
                "custom_outputs": self._serialize_custom_outputs(
                    result["custom_outputs"]
                ),
            }

        embedding = self._get_embedding(query)
        geo_hash = self._geometry_hash(geometry_params)

        entry = CacheEntry(
            query=query,
            embedding=embedding,
            result=result,
            agent_name=agent_name,
            timestamp=time.time(),
            geometry_hash=geo_hash,
        )
        self._entries.append(entry)
        self._evict_lru()

    def clear(self):
        """Clear all cache entries."""
        self._entries = []

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self._entries),
            "total_hits": sum(e.hit_count for e in self._entries),
            "max_entries": self.max_entries,
            "threshold": self.similarity_threshold,
        }
