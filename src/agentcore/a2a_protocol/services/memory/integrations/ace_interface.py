"""
ACE Strategic Context Interface

Provides strategic context interface for ACE (Agent Coordination Engine) framework.

Features:
- Strategic memory retrieval for high-level planning
- Tactical memory retrieval for execution
- Error and success pattern analysis
- Confidence scoring for decisions

Component ID: MEM-025
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)

logger = structlog.get_logger(__name__)


class ACEStrategicContext(BaseModel):
    """Strategic context for ACE framework integration."""

    context_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    session_id: str | None = None
    current_goal: str | None = None
    strategic_memories: list[str] = Field(default_factory=list)
    tactical_memories: list[str] = Field(default_factory=list)
    error_patterns: list[str] = Field(default_factory=list)
    success_patterns: list[str] = Field(default_factory=list)
    confidence_score: float = 0.5
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ACEStrategicContextInterface:
    """
    Strategic context interface for ACE (Agent Coordination Engine) framework.

    Provides:
    - Strategic memory retrieval for high-level planning
    - Tactical memory retrieval for execution
    - Error and success pattern analysis
    - Confidence scoring for decisions

    Usage:
        ace_interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service,
            error_tracker=error_tracker
        )

        # Get strategic context
        context = await ace_interface.get_strategic_context(
            agent_id="agent-1",
            goal="optimize performance"
        )

        # Store decision rationale
        await ace_interface.store_decision_rationale(
            decision_id="decision-123",
            rationale="Chose approach A because..."
        )

        # Retrieve similar decisions
        similar = await ace_interface.retrieve_similar_decisions(
            current_decision="Should I use caching?"
        )
    """

    def __init__(
        self,
        retrieval_service: EnhancedRetrievalService | None = None,
        hybrid_search: HybridSearchService | None = None,
        error_tracker: ErrorTracker | None = None,
    ):
        """
        Initialize ACEStrategicContextInterface.

        Args:
            retrieval_service: Enhanced retrieval service
            hybrid_search: Hybrid search service
            error_tracker: Error tracker service
        """
        self.retrieval = retrieval_service or EnhancedRetrievalService()
        self.hybrid_search = hybrid_search
        self.error_tracker = error_tracker

        self._strategic_contexts: dict[str, ACEStrategicContext] = {}
        self._memory_store: dict[str, MemoryRecord] = {}
        self._decision_memories: dict[str, MemoryRecord] = {}

        self._logger = logger.bind(component="ace_strategic_interface")
        self._logger.info("initialized_ace_strategic_interface")

    async def get_strategic_context(
        self,
        agent_id: str,
        goal: str,
        session_id: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> ACEStrategicContext:
        """
        Get strategic context for agent with specific goal.

        Args:
            agent_id: Agent identifier
            goal: Current goal
            session_id: Optional session identifier
            query_embedding: Optional query embedding for memory retrieval

        Returns:
            Strategic context object
        """
        # Build strategic context
        context = await self.build_strategic_context(
            agent_id=agent_id,
            session_id=session_id,
            goal=goal,
            query_embedding=query_embedding,
        )

        self._logger.info(
            "strategic_context_retrieved",
            agent_id=agent_id,
            goal=goal,
            strategic_memory_count=len(context.strategic_memories),
            confidence=context.confidence_score,
        )

        return context

    async def store_decision_rationale(
        self,
        decision_id: str,
        rationale: str,
        agent_id: str = "system",
        session_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """
        Store decision rationale in memory.

        Args:
            decision_id: Decision identifier
            rationale: Rationale text
            agent_id: Agent ID
            session_id: Optional session identifier
            embedding: Optional embedding vector

        Returns:
            Memory ID
        """
        # Create memory record for decision
        memory = MemoryRecord(
            memory_id=f"decision-{decision_id}-{datetime.now(UTC).isoformat()}",
            memory_layer=MemoryLayer.SEMANTIC,
            content=rationale,
            summary=f"Decision {decision_id}: {rationale[:100]}",
            embedding=embedding or [],
            agent_id=agent_id,
            session_id=session_id,
            task_id=None,
            keywords=["decision", "rationale", decision_id],
            is_critical=False,
        )

        # Store memory
        self._memory_store[memory.memory_id] = memory
        self._decision_memories[decision_id] = memory

        self._logger.info(
            "decision_rationale_stored",
            decision_id=decision_id,
            memory_id=memory.memory_id,
            agent_id=agent_id,
        )

        return memory.memory_id

    async def retrieve_similar_decisions(
        self,
        current_decision: str,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[tuple[str, MemoryRecord, float]]:
        """
        Retrieve similar past decisions.

        Args:
            current_decision: Current decision description
            query_embedding: Optional query embedding
            limit: Maximum number of results

        Returns:
            List of (decision_id, memory, score) tuples
        """
        if not query_embedding:
            self._logger.warning(
                "no_embedding_provided_for_similarity",
                returning_empty=True,
            )
            return []

        # Get decision memories
        decision_memories = list(self._decision_memories.values())

        if not decision_memories:
            self._logger.debug("no_decision_memories_available")
            return []

        # Score and rank
        scored = await self.retrieval.retrieve_top_k(
            memories=decision_memories,
            k=limit,
            query_embedding=query_embedding,
        )

        # Extract decision IDs
        results = []
        for mem, score, _ in scored:
            # Find decision ID from keywords
            decision_id = next(
                (kw for kw in mem.keywords if kw != "decision" and kw != "rationale"),
                mem.memory_id,
            )
            results.append((decision_id, mem, score))

        self._logger.info(
            "similar_decisions_retrieved",
            current_decision_preview=current_decision[:50],
            results_count=len(results),
        )

        return results

    async def build_strategic_context(
        self,
        agent_id: str,
        session_id: str | None = None,
        goal: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> ACEStrategicContext:
        """
        Build strategic context for agent.

        Args:
            agent_id: Agent identifier
            session_id: Optional session identifier
            goal: Optional current goal
            query_embedding: Optional query embedding for memory retrieval

        Returns:
            Strategic context object
        """
        # Get or create context
        context_key = f"{agent_id}:{session_id or 'global'}"
        if context_key not in self._strategic_contexts:
            self._strategic_contexts[context_key] = ACEStrategicContext(
                agent_id=agent_id,
                session_id=session_id,
            )

        context = self._strategic_contexts[context_key]
        context.current_goal = goal
        context.last_updated = datetime.now(UTC)

        # Retrieve strategic memories (long-term patterns)
        strategic_memories = await self._retrieve_strategic_memories(
            agent_id=agent_id,
            query_embedding=query_embedding,
        )
        context.strategic_memories = [mem.memory_id for mem in strategic_memories]

        # Retrieve tactical memories (recent, actionable)
        tactical_memories = await self._retrieve_tactical_memories(
            agent_id=agent_id,
            session_id=session_id,
            query_embedding=query_embedding,
        )
        context.tactical_memories = [mem.memory_id for mem in tactical_memories]

        # Analyze patterns
        error_patterns = await self.analyze_error_patterns(agent_id)
        context.error_patterns = error_patterns

        success_patterns = await self.analyze_success_patterns(agent_id)
        context.success_patterns = success_patterns

        self._logger.info(
            "strategic_context_built",
            agent_id=agent_id,
            session_id=session_id,
            goal=goal,
            strategic_memory_count=len(context.strategic_memories),
            tactical_memory_count=len(context.tactical_memories),
            confidence=context.confidence_score,
        )

        return context

    async def _retrieve_strategic_memories(
        self,
        agent_id: str,
        query_embedding: list[float] | None = None,
        max_memories: int = 5,
    ) -> list[MemoryRecord]:
        """
        Retrieve strategic (long-term pattern) memories.

        Args:
            agent_id: Agent identifier
            query_embedding: Optional query embedding
            max_memories: Maximum memories to retrieve

        Returns:
            List of strategic memory records
        """
        # Filter for semantic/procedural memories (strategic layer)
        strategic_mems = [
            mem
            for mem in self._memory_store.values()
            if mem.agent_id == agent_id
            and mem.memory_layer in (MemoryLayer.SEMANTIC, MemoryLayer.PROCEDURAL)
        ]

        if not strategic_mems:
            return []

        # Score and rank
        scored = await self.retrieval.retrieve_top_k(
            memories=strategic_mems,
            k=max_memories,
            query_embedding=query_embedding,
        )

        return [mem for mem, _, _ in scored]

    async def _retrieve_tactical_memories(
        self,
        agent_id: str,
        session_id: str | None = None,
        query_embedding: list[float] | None = None,
        max_memories: int = 10,
    ) -> list[MemoryRecord]:
        """
        Retrieve tactical (recent, actionable) memories.

        Args:
            agent_id: Agent identifier
            session_id: Optional session filter
            query_embedding: Optional query embedding
            max_memories: Maximum memories to retrieve

        Returns:
            List of tactical memory records
        """
        # Filter for episodic memories (tactical layer)
        tactical_mems = [
            mem
            for mem in self._memory_store.values()
            if mem.agent_id == agent_id
            and mem.memory_layer == MemoryLayer.EPISODIC
            and (session_id is None or mem.session_id == session_id)
        ]

        if not tactical_mems:
            return []

        # Score and rank with recency emphasis
        scored = await self.retrieval.retrieve_top_k(
            memories=tactical_mems,
            k=max_memories,
            query_embedding=query_embedding,
        )

        return [mem for mem, _, _ in scored]

    async def analyze_error_patterns(
        self,
        agent_id: str,
    ) -> list[str]:
        """
        Analyze error patterns from memories.

        Args:
            agent_id: Agent identifier

        Returns:
            List of error pattern descriptions
        """
        patterns = []

        # Analyze error-related memories
        error_mems = [
            mem
            for mem in self._memory_store.values()
            if mem.agent_id == agent_id
            and any(
                kw in ["error", "failure", "mistake", "fix"]
                for kw in mem.keywords
            )
        ]

        if error_mems:
            patterns.append(f"Historical error memories: {len(error_mems)}")

        return patterns

    async def analyze_success_patterns(
        self,
        agent_id: str,
    ) -> list[str]:
        """
        Analyze success patterns from memories.

        Args:
            agent_id: Agent identifier

        Returns:
            List of success pattern descriptions
        """
        patterns = []

        # Analyze success-related memories
        success_mems = [
            mem
            for mem in self._memory_store.values()
            if mem.agent_id == agent_id
            and any(
                kw in ["success", "completed", "achieved", "optimal"]
                for kw in mem.keywords
            )
        ]

        if success_mems:
            patterns.append(f"Historical success memories: {len(success_mems)}")

        return patterns

    async def update_confidence(
        self,
        agent_id: str,
        session_id: str | None = None,
        outcome_success: bool = True,
    ) -> float:
        """
        Update confidence score based on outcome.

        Args:
            agent_id: Agent identifier
            session_id: Optional session identifier
            outcome_success: Whether outcome was successful

        Returns:
            Updated confidence score
        """
        context_key = f"{agent_id}:{session_id or 'global'}"
        if context_key not in self._strategic_contexts:
            self._strategic_contexts[context_key] = ACEStrategicContext(
                agent_id=agent_id,
                session_id=session_id,
            )

        context = self._strategic_contexts[context_key]

        # Exponential moving average update
        if outcome_success:
            context.confidence_score = min(
                1.0, context.confidence_score * 0.9 + 0.1
            )
        else:
            context.confidence_score = max(
                0.0, context.confidence_score * 0.9 - 0.05
            )

        context.last_updated = datetime.now(UTC)

        self._logger.info(
            "confidence_updated",
            agent_id=agent_id,
            session_id=session_id,
            outcome_success=outcome_success,
            new_confidence=context.confidence_score,
        )

        return context.confidence_score

    async def store_strategic_memory(
        self,
        agent_id: str,
        content: str,
        memory_layer: str = "semantic",
        keywords: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """
        Store strategic memory for agent.

        Args:
            agent_id: Agent identifier
            content: Memory content
            memory_layer: Memory layer (semantic, procedural, episodic)
            keywords: Optional keywords
            embedding: Optional embedding vector

        Returns:
            Memory ID
        """
        # Map string to MemoryLayer enum
        layer_map = {
            "semantic": MemoryLayer.SEMANTIC,
            "procedural": MemoryLayer.PROCEDURAL,
            "episodic": MemoryLayer.EPISODIC,
            "working": MemoryLayer.WORKING,
        }
        layer_enum = layer_map.get(memory_layer, MemoryLayer.SEMANTIC)

        memory = MemoryRecord(
            memory_id=f"ace-{agent_id}-{datetime.now(UTC).isoformat()}",
            memory_layer=layer_enum,
            content=content,
            summary=content[:100] if len(content) > 100 else content,
            embedding=embedding or [],
            agent_id=agent_id,
            session_id=None,
            task_id=None,
            keywords=keywords or [],
        )

        self._memory_store[memory.memory_id] = memory

        self._logger.debug(
            "strategic_memory_stored",
            agent_id=agent_id,
            memory_id=memory.memory_id,
            memory_layer=memory_layer,
        )

        return memory.memory_id

    def get_context_stats(self) -> dict[str, Any]:
        """
        Get strategic context statistics.

        Returns:
            Dictionary with context statistics
        """
        return {
            "total_contexts": len(self._strategic_contexts),
            "total_memories": len(self._memory_store),
            "total_decisions": len(self._decision_memories),
            "contexts": {
                key: {
                    "agent_id": ctx.agent_id,
                    "session_id": ctx.session_id,
                    "confidence": ctx.confidence_score,
                    "strategic_memories": len(ctx.strategic_memories),
                    "tactical_memories": len(ctx.tactical_memories),
                    "error_patterns": len(ctx.error_patterns),
                    "last_updated": ctx.last_updated.isoformat(),
                }
                for key, ctx in self._strategic_contexts.items()
            },
        }
