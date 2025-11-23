"""
MessageRouter Memory Integration

Provides memory-aware routing for MessageRouter service.

Features:
- Memory-aware message routing based on context
- Historical performance tracking for agent selection
- Capability matching based on past interactions
- Error pattern awareness for routing decisions

Component ID: MEM-025
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)

logger = structlog.get_logger(__name__)


class RoutingMemoryInsight(BaseModel):
    """Memory-derived insights for routing decisions."""

    agent_id: str
    capability_match_score: float = 0.0
    historical_success_rate: float = 0.0
    memory_relevance_score: float = 0.0
    recommended: bool = False
    reasoning: str = ""


class MemoryAwareRouter:
    """
    Memory-informed routing service.

    Enhances MessageRouter with:
    - Historical performance memories for agent selection
    - Capability matching based on past interactions
    - Error pattern awareness for routing decisions
    - Success pattern learning

    Usage:
        router = MemoryAwareRouter(
            retrieval_service=retrieval_service,
            error_tracker=error_tracker
        )

        # Get relevant context for message
        context = await router.get_relevant_context(
            message="analyze Python code",
            conversation_id="conv-123"
        )

        # Enhance message with context
        enhanced = await router.enhance_message_with_context(
            message="analyze code",
            context=context
        )

        # Route with memory
        best_agent = await router.route_with_memory(
            message="analyze Python code",
            candidates=["agent-1", "agent-2"]
        )
    """

    def __init__(
        self,
        retrieval_service: EnhancedRetrievalService | None = None,
        error_tracker: ErrorTracker | None = None,
    ):
        """
        Initialize MemoryAwareRouter.

        Args:
            retrieval_service: Enhanced retrieval service for scoring
            error_tracker: Error tracker for pattern detection
        """
        self.retrieval = retrieval_service or EnhancedRetrievalService()
        self.error_tracker = error_tracker

        self._agent_memories: dict[str, list[MemoryRecord]] = {}
        self._agent_success_rates: dict[str, float] = {}
        self._routing_history: list[dict[str, Any]] = []
        self._conversation_memories: dict[str, list[MemoryRecord]] = {}

        self._logger = logger.bind(component="memory_aware_router")
        self._logger.info("initialized_memory_aware_router")

    async def get_relevant_context(
        self,
        message: str,
        conversation_id: str,
        query_embedding: list[float] | None = None,
        max_memories: int = 5,
    ) -> list[MemoryRecord]:
        """
        Get relevant context from conversation memories.

        Args:
            message: Message content
            conversation_id: Conversation identifier
            query_embedding: Optional query embedding
            max_memories: Maximum memories to retrieve

        Returns:
            List of relevant memory records
        """
        # Get conversation memories
        conversation_mems = self._conversation_memories.get(conversation_id, [])

        if not conversation_mems:
            self._logger.debug(
                "no_conversation_memories",
                conversation_id=conversation_id,
            )
            return []

        # Score and rank memories
        if query_embedding:
            scored = await self.retrieval.retrieve_top_k(
                memories=conversation_mems,
                k=max_memories,
                query_embedding=query_embedding,
            )
            memories = [mem for mem, _, _ in scored]
        else:
            # Return most recent memories if no embedding
            memories = sorted(
                conversation_mems,
                key=lambda m: m.created_at,
                reverse=True,
            )[:max_memories]

        self._logger.info(
            "relevant_context_retrieved",
            conversation_id=conversation_id,
            message_preview=message[:50],
            memory_count=len(memories),
        )

        return memories

    async def enhance_message_with_context(
        self,
        message: str,
        context: list[MemoryRecord],
    ) -> str:
        """
        Enhance message with contextual memories.

        Args:
            message: Original message
            context: Contextual memory records

        Returns:
            Enhanced message with context
        """
        if not context:
            return message

        # Build context string
        context_str = "\n\n".join([
            f"[Context {i+1}]: {mem.summary or mem.content[:100]}"
            for i, mem in enumerate(context)
        ])

        # Combine message with context
        enhanced = f"{message}\n\nRelevant Context:\n{context_str}"

        self._logger.debug(
            "message_enhanced_with_context",
            original_length=len(message),
            enhanced_length=len(enhanced),
            context_count=len(context),
        )

        return enhanced

    async def route_with_memory(
        self,
        message: str,
        candidates: list[str],
        query_embedding: list[float] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> str | None:
        """
        Route message using memory-aware selection.

        Args:
            message: Message to route
            candidates: Candidate agent IDs
            query_embedding: Optional query embedding
            required_capabilities: Optional required capabilities

        Returns:
            Selected agent ID or None if no suitable agent
        """
        if not candidates:
            return None

        # Get routing insights
        insights = await self.get_routing_insights(
            candidate_agents=candidates,
            query=message,
            query_embedding=query_embedding,
            required_capabilities=required_capabilities,
        )

        # Select top recommended agent
        if insights and insights[0].recommended:
            selected = insights[0].agent_id

            self._logger.info(
                "agent_selected_with_memory",
                selected_agent=selected,
                message_preview=message[:50],
                reasoning=insights[0].reasoning,
            )

            return selected

        # Fall back to first candidate if no strong recommendation
        fallback = candidates[0] if candidates else None

        self._logger.warning(
            "no_strong_recommendation_fallback",
            fallback_agent=fallback,
            candidates_count=len(candidates),
        )

        return fallback

    async def get_routing_insights(
        self,
        candidate_agents: list[str],
        query: str | None = None,
        query_embedding: list[float] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> list[RoutingMemoryInsight]:
        """
        Get memory-based routing insights for candidate agents.

        Args:
            candidate_agents: List of candidate agent IDs
            query: Optional query string
            query_embedding: Optional query embedding
            required_capabilities: Optional required capabilities

        Returns:
            List of routing insights for each candidate
        """
        insights = []

        for agent_id in candidate_agents:
            # Calculate capability match score
            capability_score = 0.0
            if required_capabilities:
                agent_memories = self._agent_memories.get(agent_id, [])
                matching_caps = 0
                for cap in required_capabilities:
                    if any(cap.lower() in mem.content.lower() for mem in agent_memories):
                        matching_caps += 1
                capability_score = (
                    matching_caps / len(required_capabilities)
                    if required_capabilities
                    else 0.0
                )

            # Get historical success rate
            success_rate = self._agent_success_rates.get(agent_id, 0.5)

            # Calculate memory relevance if query provided
            memory_relevance = 0.0
            if query_embedding and agent_id in self._agent_memories:
                agent_mems = self._agent_memories[agent_id]
                if agent_mems:
                    scored = await self.retrieval.retrieve_top_k(
                        memories=agent_mems,
                        k=1,
                        query_embedding=query_embedding,
                    )
                    if scored:
                        memory_relevance = scored[0][1]

            # Calculate overall score
            overall_score = (
                0.4 * capability_score
                + 0.3 * success_rate
                + 0.3 * memory_relevance
            )

            # Determine recommendation
            recommended = overall_score > 0.6
            reasoning = f"Capability: {capability_score:.2f}, Success: {success_rate:.2f}, Relevance: {memory_relevance:.2f}"

            insights.append(
                RoutingMemoryInsight(
                    agent_id=agent_id,
                    capability_match_score=capability_score,
                    historical_success_rate=success_rate,
                    memory_relevance_score=memory_relevance,
                    recommended=recommended,
                    reasoning=reasoning,
                )
            )

        # Sort by combined score descending
        insights.sort(
            key=lambda x: (
                0.4 * x.capability_match_score
                + 0.3 * x.historical_success_rate
                + 0.3 * x.memory_relevance_score
            ),
            reverse=True,
        )

        self._logger.info(
            "routing_insights_generated",
            candidate_count=len(candidate_agents),
            top_agent=insights[0].agent_id if insights else None,
        )

        return insights

    async def select_agent_with_memory(
        self,
        candidates: list[str],
        query_embedding: list[float] | None = None,
        task_type: str | None = None,
    ) -> str | None:
        """
        Select best agent based on memory insights.

        Args:
            candidates: List of candidate agent IDs
            query_embedding: Optional query embedding
            task_type: Optional task type for filtering

        Returns:
            Selected agent ID or None if no suitable agent
        """
        if not candidates:
            return None

        insights = await self.get_routing_insights(
            candidate_agents=candidates,
            query_embedding=query_embedding,
            required_capabilities=[task_type] if task_type else None,
        )

        if insights and insights[0].recommended:
            return insights[0].agent_id

        # Fall back to first candidate if no strong recommendation
        return candidates[0] if candidates else None

    async def record_routing_outcome(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        memory_content: str | None = None,
    ) -> None:
        """
        Record routing outcome for learning.

        Args:
            agent_id: Agent that was routed to
            task_type: Type of task
            success: Whether the routing was successful
            memory_content: Optional memory content from task
        """
        # Update success rate
        current_rate = self._agent_success_rates.get(agent_id, 0.5)
        # Exponential moving average
        new_rate = 0.9 * current_rate + 0.1 * (1.0 if success else 0.0)
        self._agent_success_rates[agent_id] = new_rate

        # Store memory if provided
        if memory_content:
            if agent_id not in self._agent_memories:
                self._agent_memories[agent_id] = []

            memory = MemoryRecord(
                memory_id=f"routing-{agent_id}-{datetime.now(UTC).isoformat()}",
                memory_layer=MemoryLayer.EPISODIC,
                content=memory_content,
                summary=f"Routing outcome for {task_type}: {'success' if success else 'failure'}",
                embedding=[],
                agent_id=agent_id,
                session_id=None,
                task_id=None,
                keywords=[task_type, "routing", "success" if success else "failure"],
            )
            self._agent_memories[agent_id].append(memory)

        # Record in history
        self._routing_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "agent_id": agent_id,
                "task_type": task_type,
                "success": success,
            }
        )

        self._logger.info(
            "routing_outcome_recorded",
            agent_id=agent_id,
            task_type=task_type,
            success=success,
            new_success_rate=new_rate,
        )

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        return {
            "total_agents_tracked": len(self._agent_success_rates),
            "total_memories": sum(
                len(mems) for mems in self._agent_memories.values()
            ),
            "total_routing_events": len(self._routing_history),
            "agent_success_rates": dict(self._agent_success_rates),
        }
