"""
Memory mechanism for Career AI Platform.
Implements short-term (LangGraph State) and long-term (ChromaDB vector) memory.
Supports MCP-attached external knowledge base pattern.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from app.core.config import config
from app.core.state import OrchestratorState


# ============================================================================
# In-Memory Session Store (Short-term)
# ============================================================================

class SessionMemory:
    """
    Session storage with JSON file persistence.
    Survives server restarts. For production, replace with Redis.
    """

    def __init__(self):
        self._sessions: dict[str, OrchestratorState] = {}
        self._persist_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "sessions.json"
        )
        self._load_from_disk()

    def _load_from_disk(self):
        """Load sessions from JSON file on startup."""
        try:
            if os.path.exists(self._persist_path):
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for sid, state_dict in data.items():
                    try:
                        self._sessions[sid] = OrchestratorState(**state_dict)
                    except Exception:
                        pass  # Skip corrupted sessions
                print(f"[Memory] Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            print(f"[Memory] Failed to load sessions: {e}")

    def _save_to_disk(self):
        """Persist sessions to JSON file."""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {}
            for sid, state in self._sessions.items():
                try:
                    data[sid] = state.model_dump()
                except Exception:
                    pass
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[Memory] Failed to save sessions: {e}")

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = OrchestratorState(session_id=session_id)
        self._save_to_disk()
        return session_id

    def get_session(self, session_id: str) -> Optional[OrchestratorState]:
        """Get session state by ID."""
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, state: OrchestratorState) -> None:
        """Update session state and persist to disk."""
        self._sessions[session_id] = state
        self._save_to_disk()

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)
        self._save_to_disk()

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def clear_expired(self, max_age_hours: int = 24) -> int:
        """Clear sessions older than max_age_hours. No-op for now."""
        return 0


# ============================================================================
# Vector Memory (Long-term via ChromaDB)
# ============================================================================

class VectorMemory:
    """
    Long-term memory using vector database.
    Stores user profiles, reports, and conversation embeddings for cross-session recall.
    """

    def __init__(self):
        self._db_type = config.memory.vector_db_type
        self._collection_name = config.memory.collection_name
        self._client = None
        self._collection = None
        self._initialized = False

    async def initialize(self):
        """Initialize the vector database connection."""
        if self._initialized:
            return

        if self._db_type == "chromadb":
            try:
                import chromadb
                self._client = chromadb.PersistentClient(
                    path=config.memory.chromadb_path
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name
                )
                self._initialized = True
            except ImportError:
                print("ChromaDB not installed. Vector memory will be unavailable.")
            except Exception as e:
                print(f"Failed to initialize ChromaDB: {e}")
        else:
            print(f"Vector DB type '{self._db_type}' not supported yet.")

    async def store(self, key: str, data: dict, metadata: Optional[dict] = None) -> bool:
        """Store a document in vector memory."""
        if not self._initialized or self._collection is None:
            return False
        try:
            doc_id = f"{key}_{datetime.now().timestamp()}"
            self._collection.add(
                documents=[json.dumps(data, ensure_ascii=False)],
                metadatas=[metadata or {"key": key, "timestamp": datetime.now().isoformat()}],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Vector memory store failed: {e}")
            return False

    async def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search vector memory by semantic similarity."""
        if not self._initialized or self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results
            )
            docs = []
            if results and results.get("documents"):
                for i, doc_list in enumerate(results["documents"]):
                    for doc in doc_list:
                        try:
                            docs.append(json.loads(doc))
                        except json.JSONDecodeError:
                            docs.append({"text": doc})
            return docs
        except Exception as e:
            print(f"Vector memory search failed: {e}")
            return []

    async def delete_by_key(self, key: str) -> bool:
        """Delete documents by key prefix."""
        if not self._initialized or self._collection is None:
            return False
        try:
            self._collection.delete(where={"key": key})
            return True
        except Exception as e:
            print(f"Vector memory delete failed: {e}")
            return False


# ============================================================================
# Memory Manager (Facade)
# ============================================================================

class MemoryManager:
    """
    Unified memory manager combining short-term and long-term memory.
    Implements the MCP-attached external knowledge base pattern.
    """

    def __init__(self):
        self.session_memory = SessionMemory()
        self.vector_memory = VectorMemory()
        self._knowledge_base: dict[str, str] = {}

    async def initialize(self):
        """Initialize all memory systems."""
        await self.vector_memory.initialize()

    # --- Session (Short-term) ---

    def create_session(self) -> str:
        return self.session_memory.create_session()

    def get_session(self, session_id: str) -> Optional[OrchestratorState]:
        return self.session_memory.get_session(session_id)

    def update_session(self, session_id: str, state: OrchestratorState) -> None:
        self.session_memory.update_session(session_id, state)

    # --- Vector (Long-term) ---

    async def remember(self, key: str, data: dict, metadata: Optional[dict] = None) -> bool:
        """Store a memory (long-term)."""
        return await self.vector_memory.store(key, data, metadata)

    async def recall(self, query: str, n_results: int = 5) -> list[dict]:
        """Recall memories by semantic query."""
        return await self.vector_memory.search(query, n_results)

    # --- Knowledge Base (MCP-attached) ---

    def register_knowledge(self, key: str, content: str):
        """Register external knowledge (from MCP servers)."""
        self._knowledge_base[key] = content

    def get_knowledge(self, key: str) -> Optional[str]:
        """Retrieve registered knowledge."""
        return self._knowledge_base.get(key)

    def list_knowledge_keys(self) -> list[str]:
        """List all registered knowledge keys."""
        return list(self._knowledge_base.keys())


# Global singleton
memory_manager = MemoryManager()