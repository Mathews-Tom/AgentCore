"""
SessionManager Memory Integration

Provides memory-backed session context for SessionManager service.

Features:
- Session context storage and retrieval
- Session state persistence in memory layer
- Context continuity across session lifecycle
- Memory compression for long-running sessions

Component ID: MEM-025
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord, StageType
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)

logger = structlog.get_logger(__name__)


class SessionMemoryContext(BaseModel):
    """Memory context data for a session."""

    session_id: str
    memory_ids: list[str] = Field(default_factory=list)
    last_query: str | None = None
    last_retrieval_count: int = 0
    context_size_bytes: int = 0
    compression_applied: bool = False
    strategic_insights: list[str] = Field(default_factory=list)


class SessionContextProvider:
    """
    Memory-backed session context provider.

    Integrates with SessionManager to provide:
    - Automatic context retrieval for session operations
    - Memory compression during long-running sessions
    - Session state persistence in memory layer
    - Context continuity across session lifecycle

    Usage:
        provider = SessionContextProvider(
            retrieval_service=retrieval_service,
            hybrid_search=hybrid_search
        )

        # Store session context in memory
        await provider.store_session_context(
            session_id="session-123",
            context_data={"user_id": "user-1", "auth": "admin"}
        )

        # Retrieve session context
        context = await provider.retrieve_session_context(
            session_id="session-123"
        )

        # Update session memory
        await provider.update_session_memory(
            session_id="session-123",
            updates={"last_action": "login"}
        )
    """

    def __init__(
        self,
        retrieval_service: EnhancedRetrievalService | None = None,
        hybrid_search: HybridSearchService | None = None,
        error_tracker: ErrorTracker | None = None,
    ):
        """
        Initialize SessionContextProvider.

        Args:
            retrieval_service: Enhanced retrieval service for scoring
            hybrid_search: Hybrid search service for memory retrieval
            error_tracker: Error tracker for error pattern detection
        """
        self.retrieval = retrieval_service or EnhancedRetrievalService()
        self.hybrid_search = hybrid_search
        self.error_tracker = error_tracker

        self._session_contexts: dict[str, SessionMemoryContext] = {}
        self._memory_store: dict[str, MemoryRecord] = {}

        self._logger = logger.bind(component="session_context_provider")
        self._logger.info("initialized_session_context_provider")

    async def store_session_context(
        self,
        session_id: str,
        context_data: dict[str, Any],
        agent_id: str = "system",
    ) -> str:
        """
        Store session context in memory.

        Args:
            session_id: Session identifier
            context_data: Context data to store
            agent_id: Agent ID (defaults to "system")

        Returns:
            Memory ID of stored context
        """
        # Create memory record for session context
        memory = MemoryRecord(
            memory_id=f"session-ctx-{session_id}-{datetime.now(UTC).isoformat()}",
            memory_layer=MemoryLayer.EPISODIC,
            content=json.dumps(context_data),
            summary=f"Session context for {session_id}",
            embedding=[],  # No embedding for context records
            agent_id=agent_id,
            session_id=session_id,
            task_id=None,
            keywords=["session", "context", session_id],
            is_critical=True,  # Session context is critical
        )

        # Store in memory
        self._memory_store[memory.memory_id] = memory

        # Update session context tracking
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = SessionMemoryContext(
                session_id=session_id
            )

        self._session_contexts[session_id].memory_ids.append(memory.memory_id)

        self._logger.info(
            "session_context_stored",
            session_id=session_id,
            memory_id=memory.memory_id,
            context_keys=list(context_data.keys()),
        )

        return memory.memory_id

    async def retrieve_session_context(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve session context from memory.

        Args:
            session_id: Session identifier

        Returns:
            Session context data or None if not found
        """
        # Get session context tracking
        context = self._session_contexts.get(session_id)
        if not context or not context.memory_ids:
            self._logger.debug(
                "session_context_not_found",
                session_id=session_id,
            )
            return None

        # Get most recent memory
        latest_memory_id = context.memory_ids[-1]
        memory = self._memory_store.get(latest_memory_id)

        if not memory:
            self._logger.warning(
                "session_memory_missing",
                session_id=session_id,
                memory_id=latest_memory_id,
            )
            return None

        # Parse and return context data
        try:
            context_data = json.loads(memory.content)
            self._logger.info(
                "session_context_retrieved",
                session_id=session_id,
                memory_id=memory.memory_id,
            )
            return context_data
        except json.JSONDecodeError as e:
            self._logger.error(
                "session_context_parse_error",
                session_id=session_id,
                error=str(e),
            )
            return None

    async def update_session_memory(
        self,
        session_id: str,
        updates: dict[str, Any],
        agent_id: str = "system",
    ) -> str:
        """
        Update session memory with new data.

        Args:
            session_id: Session identifier
            updates: Updates to apply to session context
            agent_id: Agent ID

        Returns:
            Memory ID of updated context
        """
        # Retrieve existing context
        existing_context = await self.retrieve_session_context(session_id)

        # Merge updates
        if existing_context:
            context_data = {**existing_context, **updates}
        else:
            context_data = updates

        # Store updated context
        memory_id = await self.store_session_context(
            session_id=session_id,
            context_data=context_data,
            agent_id=agent_id,
        )

        self._logger.info(
            "session_memory_updated",
            session_id=session_id,
            memory_id=memory_id,
            update_keys=list(updates.keys()),
        )

        return memory_id

    async def get_session_context(
        self,
        session_id: str,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        current_stage: StageType | None = None,
        max_memories: int = 10,
    ) -> list[MemoryRecord]:
        """
        Retrieve relevant memories for session context.

        Args:
            session_id: Session identifier
            query: Optional query string for semantic search
            query_embedding: Optional query embedding vector
            current_stage: Optional current reasoning stage
            max_memories: Maximum number of memories to retrieve

        Returns:
            List of relevant memory records
        """
        # Check if hybrid search is available
        if self.hybrid_search and query_embedding:
            # Use hybrid search for best results
            has_errors = False
            results = await self.hybrid_search.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                limit=max_memories,
                session_id=session_id,
                current_stage=current_stage,
                has_recent_errors=has_errors,
            )
            memories = [mem for mem, _, _ in results]
        else:
            # Fall back to retrieval service scoring on cached memories
            session_memories = [
                mem
                for mem in self._memory_store.values()
                if mem.session_id == session_id
            ]

            has_errors = False
            scored = await self.retrieval.retrieve_top_k(
                memories=session_memories,
                k=max_memories,
                query_embedding=query_embedding,
                current_stage=current_stage,
                has_recent_errors=has_errors,
            )
            memories = [mem for mem, _, _ in scored]

        # Update session context tracking
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = SessionMemoryContext(
                session_id=session_id
            )

        context = self._session_contexts[session_id]
        context.memory_ids = [mem.memory_id for mem in memories]
        context.last_query = query
        context.last_retrieval_count = len(memories)
        context.context_size_bytes = sum(len(mem.content.encode()) for mem in memories)

        self._logger.info(
            "session_context_retrieved",
            session_id=session_id,
            memory_count=len(memories),
            context_size_bytes=context.context_size_bytes,
        )

        return memories

    async def persist_session_state(
        self,
        session_id: str,
        state_data: dict[str, Any],
        agent_id: str = "system",
    ) -> str:
        """
        Persist session state as memory record.

        Args:
            session_id: Session identifier
            state_data: Session state data to persist
            agent_id: Agent ID (defaults to "system")

        Returns:
            Memory ID of persisted state
        """
        # Create memory record for session state
        memory = MemoryRecord(
            memory_id=f"session-state-{session_id}-{datetime.now(UTC).isoformat()}",
            memory_layer=MemoryLayer.EPISODIC,
            content=json.dumps(state_data),
            summary=f"Session state for {session_id}",
            embedding=[],
            agent_id=agent_id,
            session_id=session_id,
            task_id=None,
            keywords=["session", "state", session_id],
            is_critical=True,
        )

        # Store in memory
        self._memory_store[memory.memory_id] = memory

        # Ensure session context exists
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = SessionMemoryContext(
                session_id=session_id
            )

        self._logger.info(
            "session_state_persisted",
            session_id=session_id,
            memory_id=memory.memory_id,
            state_keys=list(state_data.keys()),
        )

        return memory.memory_id

    async def add_strategic_insight(
        self,
        session_id: str,
        insight: str,
    ) -> None:
        """
        Add strategic insight to session context.

        Args:
            session_id: Session identifier
            insight: Strategic insight text
        """
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = SessionMemoryContext(
                session_id=session_id
            )

        self._session_contexts[session_id].strategic_insights.append(insight)

        self._logger.debug(
            "strategic_insight_added",
            session_id=session_id,
            insight_count=len(self._session_contexts[session_id].strategic_insights),
        )

    def get_session_memory_stats(self, session_id: str) -> dict[str, Any]:
        """
        Get memory statistics for session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with memory statistics
        """
        context = self._session_contexts.get(session_id)

        if not context:
            return {
                "session_id": session_id,
                "exists": False,
                "memory_count": 0,
                "context_size_bytes": 0,
            }

        return {
            "session_id": session_id,
            "exists": True,
            "memory_count": len(context.memory_ids),
            "context_size_bytes": context.context_size_bytes,
            "last_query": context.last_query,
            "last_retrieval_count": context.last_retrieval_count,
            "compression_applied": context.compression_applied,
            "strategic_insights_count": len(context.strategic_insights),
        }
