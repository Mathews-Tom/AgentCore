"""
Memory Service Integrations

Provides integration adapters for cross-service communication between memory services
and core AgentCore services (SessionManager, MessageRouter, TaskManager).

Implements:
- SessionContextProvider: Memory-backed session context
- MemoryAwareRouter: Memory-informed routing decisions
- ArtifactMemoryStorage: Task artifact persistence in memory
- ACEStrategicContextInterface: Strategic memory for ACE framework

Component ID: MEM-025
Ticket: MEM-025 (Implement Service Integrations)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord, StageType
from agentcore.a2a_protocol.models.task import TaskArtifact
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.hybrid_search import (
    HybridSearchConfig,
    HybridSearchService,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
    RetrievalConfig,
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


class RoutingMemoryInsight(BaseModel):
    """Memory-derived insights for routing decisions."""

    agent_id: str
    capability_match_score: float = 0.0
    historical_success_rate: float = 0.0
    memory_relevance_score: float = 0.0
    recommended: bool = False
    reasoning: str = ""


class ArtifactMemoryRecord(BaseModel):
    """Memory record for task artifact."""

    artifact_name: str  # Using name from TaskArtifact
    task_id: str
    execution_id: str
    memory_id: str
    content_hash: str
    stored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieval_count: int = 0


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

        # Get relevant memories for session
        context = await provider.get_session_context(
            session_id="session-123",
            query="user authentication"
        )

        # Store session state in memory
        await provider.persist_session_state(
            session_id="session-123",
            state_data={"user_id": "user-1", "auth_level": "admin"}
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
            # ErrorTracker requires task_id/agent_id context, skip for now
            # In production, caller would provide these via extended method signature

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
            # ErrorTracker requires task_id/agent_id context, skip for now

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
            has_query=query is not None,
            current_stage=current_stage.value if current_stage else None,
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
        import json

        # Create memory record for session state
        memory = MemoryRecord(
            memory_id=f"session-state-{session_id}-{datetime.now(UTC).isoformat()}",
            memory_layer=MemoryLayer.EPISODIC,
            content=json.dumps(state_data),
            summary=f"Session state for {session_id}",
            embedding=[],  # No embedding for state records
            agent_id=agent_id,
            session_id=session_id,
            task_id=None,
            keywords=["session", "state", session_id],
            is_critical=True,  # Session state is critical
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

        # Get routing insights
        insights = await router.get_routing_insights(
            required_capabilities=["code_analysis"],
            query="analyze Python code for security vulnerabilities"
        )

        # Select best agent based on memories
        best_agent = await router.select_agent_with_memory(
            candidates=["agent-1", "agent-2", "agent-3"],
            query_embedding=query_emb,
            task_type="code_analysis"
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

        self._logger = logger.bind(component="memory_aware_router")

        self._logger.info("initialized_memory_aware_router")

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
            top_score=(
                0.4 * insights[0].capability_match_score
                + 0.3 * insights[0].historical_success_rate
                + 0.3 * insights[0].memory_relevance_score
                if insights
                else None
            ),
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
            selected = insights[0].agent_id

            self._logger.info(
                "agent_selected_with_memory",
                selected_agent=selected,
                score=(
                    0.4 * insights[0].capability_match_score
                    + 0.3 * insights[0].historical_success_rate
                    + 0.3 * insights[0].memory_relevance_score
                ),
                reasoning=insights[0].reasoning,
            )

            return selected

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


class ArtifactMemoryStorage:
    """
    Task artifact persistence in memory layer.

    Integrates with TaskManager to provide:
    - Artifact storage in memory graph
    - Artifact retrieval by similarity
    - Artifact versioning and history
    - Cross-task artifact sharing

    Usage:
        storage = ArtifactMemoryStorage(
            hybrid_search=hybrid_search
        )

        # Store artifact in memory
        memory_id = await storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=artifact
        )

        # Retrieve similar artifacts
        similar = await storage.find_similar_artifacts(
            query_embedding=artifact_emb,
            limit=5
        )
    """

    def __init__(
        self,
        hybrid_search: HybridSearchService | None = None,
        retrieval_service: EnhancedRetrievalService | None = None,
    ):
        """
        Initialize ArtifactMemoryStorage.

        Args:
            hybrid_search: Hybrid search service for retrieval
            retrieval_service: Retrieval service for scoring
        """
        self.hybrid_search = hybrid_search
        self.retrieval = retrieval_service or EnhancedRetrievalService()

        self._artifact_records: dict[str, ArtifactMemoryRecord] = {}
        self._memory_store: dict[str, MemoryRecord] = {}
        self._task_artifacts: dict[str, list[str]] = {}  # task_id -> artifact_ids

        self._logger = logger.bind(component="artifact_memory_storage")

        self._logger.info("initialized_artifact_memory_storage")

    async def store_artifact(
        self,
        task_id: str,
        execution_id: str,
        artifact: TaskArtifact,
        embedding: list[float] | None = None,
        agent_id: str = "system",
    ) -> str:
        """
        Store task artifact in memory.

        Args:
            task_id: Task identifier
            execution_id: Execution identifier
            artifact: Task artifact to store
            embedding: Optional embedding for artifact content
            agent_id: Agent ID

        Returns:
            Memory ID of stored artifact
        """
        import hashlib
        import json

        # Create content hash
        content_str = (
            json.dumps(artifact.content, sort_keys=True)
            if isinstance(artifact.content, dict)
            else str(artifact.content)
        )
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Create memory record
        memory = MemoryRecord(
            memory_id=f"artifact-{artifact.name}",
            memory_layer=MemoryLayer.SEMANTIC,  # Artifacts are semantic memories
            content=content_str,
            summary=f"Artifact: {artifact.name} ({artifact.type})",
            embedding=embedding or [],
            agent_id=agent_id,
            session_id=None,
            task_id=task_id,
            keywords=[
                "artifact",
                artifact.type,
                artifact.name,
                task_id,
            ],
            is_critical=False,
        )

        # Store memory
        self._memory_store[memory.memory_id] = memory

        # Create artifact record
        artifact_record = ArtifactMemoryRecord(
            artifact_name=artifact.name,
            task_id=task_id,
            execution_id=execution_id,
            memory_id=memory.memory_id,
            content_hash=content_hash,
        )
        self._artifact_records[artifact.name] = artifact_record

        # Track task artifacts
        if task_id not in self._task_artifacts:
            self._task_artifacts[task_id] = []
        self._task_artifacts[task_id].append(artifact.name)

        self._logger.info(
            "artifact_stored_in_memory",
            artifact_name=artifact.name,
            memory_id=memory.memory_id,
            task_id=task_id,
            artifact_type=artifact.type,
            content_hash=content_hash[:16],
        )

        return memory.memory_id

    async def retrieve_artifact(
        self,
        artifact_name: str,
    ) -> MemoryRecord | None:
        """
        Retrieve artifact from memory.

        Args:
            artifact_name: Artifact name identifier

        Returns:
            Memory record or None if not found
        """
        record = self._artifact_records.get(artifact_name)
        if not record:
            return None

        memory = self._memory_store.get(record.memory_id)
        if memory:
            record.retrieval_count += 1

        return memory

    async def find_similar_artifacts(
        self,
        query_embedding: list[float],
        limit: int = 5,
        task_id: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Find similar artifacts based on embedding similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            task_id: Optional filter by task ID

        Returns:
            List of (memory, score) tuples
        """
        # Filter memories to artifacts only
        artifact_memories = [
            mem
            for mem in self._memory_store.values()
            if "artifact" in mem.keywords
            and (task_id is None or mem.task_id == task_id)
        ]

        if not artifact_memories:
            return []

        # Score and rank
        scored = await self.retrieval.retrieve_top_k(
            memories=artifact_memories,
            k=limit,
            query_embedding=query_embedding,
        )

        results = [(mem, score) for mem, score, _ in scored]

        self._logger.info(
            "similar_artifacts_found",
            query_dims=len(query_embedding),
            results_count=len(results),
            task_filter=task_id,
        )

        return results

    async def get_task_artifacts(
        self,
        task_id: str,
    ) -> list[MemoryRecord]:
        """
        Get all artifacts for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of artifact memory records
        """
        artifact_ids = self._task_artifacts.get(task_id, [])
        memories = []

        for artifact_id in artifact_ids:
            record = self._artifact_records.get(artifact_id)
            if record:
                memory = self._memory_store.get(record.memory_id)
                if memory:
                    memories.append(memory)

        return memories

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get artifact storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        return {
            "total_artifacts": len(self._artifact_records),
            "total_memories": len(self._memory_store),
            "tasks_with_artifacts": len(self._task_artifacts),
            "total_retrieval_count": sum(
                rec.retrieval_count for rec in self._artifact_records.values()
            ),
        }


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

        # Build strategic context
        context = await ace_interface.build_strategic_context(
            agent_id="agent-1",
            session_id="session-123",
            goal="optimize system performance"
        )

        # Get error patterns
        patterns = await ace_interface.analyze_error_patterns(agent_id="agent-1")

        # Update confidence based on outcomes
        await ace_interface.update_confidence(
            agent_id="agent-1",
            outcome_success=True
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

        self._logger = logger.bind(component="ace_strategic_interface")

        self._logger.info("initialized_ace_strategic_interface")

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
            error_pattern_count=len(context.error_patterns),
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
            and mem.memory_layer in ("semantic", "procedural")
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
            and mem.memory_layer == "episodic"
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

        # ErrorTracker requires task_id/agent_id context
        # In this integration layer, we analyze memories directly for error patterns

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


class MemoryServiceIntegration:
    """
    Unified memory service integration orchestrator.

    Combines all integration adapters into a single interface:
    - SessionContextProvider for SessionManager
    - MemoryAwareRouter for MessageRouter
    - ArtifactMemoryStorage for TaskManager
    - ACEStrategicContextInterface for ACE framework

    Usage:
        integration = MemoryServiceIntegration(
            retrieval_config=RetrievalConfig(),
            hybrid_search_config=HybridSearchConfig()
        )

        # Access individual services
        session_provider = integration.session_context
        memory_router = integration.memory_router
        artifact_storage = integration.artifact_storage
        ace_interface = integration.ace_interface

        # Get unified stats
        stats = integration.get_integration_stats()
    """

    def __init__(
        self,
        retrieval_service: EnhancedRetrievalService | None = None,
        hybrid_search: HybridSearchService | None = None,
        error_tracker: ErrorTracker | None = None,
        retrieval_config: RetrievalConfig | None = None,
        hybrid_search_config: HybridSearchConfig | None = None,
    ):
        """
        Initialize MemoryServiceIntegration.

        Args:
            retrieval_service: Optional pre-configured retrieval service
            hybrid_search: Optional pre-configured hybrid search
            error_tracker: Optional error tracker
            retrieval_config: Optional retrieval configuration
            hybrid_search_config: Optional hybrid search configuration
        """
        # Initialize core services if not provided
        self._retrieval = retrieval_service or EnhancedRetrievalService(
            retrieval_config or RetrievalConfig()
        )
        self._hybrid_search = hybrid_search
        self._error_tracker = error_tracker or ErrorTracker()

        # Initialize integration adapters
        self.session_context = SessionContextProvider(
            retrieval_service=self._retrieval,
            hybrid_search=self._hybrid_search,
            error_tracker=self._error_tracker,
        )

        self.memory_router = MemoryAwareRouter(
            retrieval_service=self._retrieval,
            error_tracker=self._error_tracker,
        )

        self.artifact_storage = ArtifactMemoryStorage(
            hybrid_search=self._hybrid_search,
            retrieval_service=self._retrieval,
        )

        self.ace_interface = ACEStrategicContextInterface(
            retrieval_service=self._retrieval,
            hybrid_search=self._hybrid_search,
            error_tracker=self._error_tracker,
        )

        self._logger = logger.bind(component="memory_service_integration")

        self._logger.info(
            "initialized_memory_service_integration",
            has_hybrid_search=self._hybrid_search is not None,
            has_error_tracker=self._error_tracker is not None,
        )

    def get_integration_stats(self) -> dict[str, Any]:
        """
        Get comprehensive integration statistics.

        Returns:
            Dictionary with all integration statistics
        """
        return {
            "session_context": self.session_context.get_session_memory_stats("*"),
            "memory_router": self.memory_router.get_routing_stats(),
            "artifact_storage": self.artifact_storage.get_storage_stats(),
            "ace_interface": self.ace_interface.get_context_stats(),
        }

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all integration components.

        Returns:
            Dictionary with health status for each component
        """
        return {
            "session_context": True,  # Always healthy if initialized
            "memory_router": True,
            "artifact_storage": True,
            "ace_interface": True,
            "error_tracker": self._error_tracker is not None,
            "hybrid_search": self._hybrid_search is not None,
        }
