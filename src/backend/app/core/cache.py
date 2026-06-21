"""
Semantic Cache for Career AI Platform.
Uses ChromaDB vector similarity to cache LLM responses.
When a user asks a similar question, return cached result directly.
"""
from __future__ import annotations

import json
import hashlib
import time
from typing import Optional

from app.core.config import config


class SemanticCache:
    """
    Semantic cache using ChromaDB for vector similarity matching.
    Caches LLM responses to avoid redundant API calls.
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._initialized = False
        self._cache_path = config.memory.chromadb_path + "/semantic_cache"
        self._ttl_seconds = 3600 * 24  # 24 hours

    async def initialize(self):
        """Initialize the cache collection."""
        if self._initialized:
            return
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self._cache_path)
            self._collection = self._client.get_or_create_collection(
                name="semantic_cache",
                metadata={"hnsw:space": "cosine"}
            )
            self._initialized = True
            print(f"[SemanticCache] Initialized with {self._collection.count()} cached entries")
        except Exception as e:
            print(f"[SemanticCache] Failed to initialize: {e}")

    async def get(self, query: str, threshold: float = 0.92) -> Optional[dict]:
        """
        Look up cached result for a query.
        Returns cached dict if similarity > threshold, else None.
        """
        if not self._initialized or self._collection is None:
            return None
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=1
            )
            if not results or not results.get("documents"):
                return None

            doc = results["documents"][0][0]
            distance = results["distances"][0][0] if results.get("distances") else 1.0

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            similarity = 1.0 - distance / 2.0

            if similarity >= threshold:
                cached = json.loads(doc)
                # Check TTL
                if time.time() - cached.get("timestamp", 0) < self._ttl_seconds:
                    print(f"[SemanticCache] Cache hit (similarity={similarity:.3f})")
                    return cached.get("result")
                else:
                    print(f"[SemanticCache] Cache expired")
            else:
                print(f"[SemanticCache] No match (similarity={similarity:.3f} < {threshold})")
        except Exception as e:
            print(f"[SemanticCache] Lookup failed: {e}")
        return None

    async def set(self, query: str, result: dict) -> bool:
        """Store a query-result pair in the cache."""
        if not self._initialized or self._collection is None:
            return False
        try:
            doc_id = hashlib.md5(query.encode()).hexdigest()
            doc = json.dumps({
                "query": query,
                "result": result,
                "timestamp": time.time()
            }, ensure_ascii=False)
            self._collection.upsert(
                documents=[doc],
                ids=[doc_id],
                metadatas=[{"query": query[:100]}]
            )
            return True
        except Exception as e:
            print(f"[SemanticCache] Store failed: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cached entries."""
        if not self._initialized or self._collection is None:
            return False
        try:
            self._client.delete_collection("semantic_cache")
            self._collection = self._client.get_or_create_collection(
                name="semantic_cache",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"[SemanticCache] Clear failed: {e}")
            return False


# Global singleton
semantic_cache = SemanticCache()