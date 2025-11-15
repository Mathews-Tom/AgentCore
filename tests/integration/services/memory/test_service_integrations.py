"""
Integration tests for Memory Service Integrations (MEM-025)

Tests cross-service communication between memory services and core AgentCore services:
- SessionContextProvider with SessionManager
- MemoryAwareRouter with MessageRouter
- ArtifactMemoryStorage with TaskManager
- ACEStrategicContextInterface
"""

from datetime import UTC, datetime

import pytest

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord
from agentcore.a2a_protocol.models.task import TaskArtifact
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.integration import (
    ACEStrategicContext,
    ACEStrategicContextInterface,
    ArtifactMemoryStorage,
    MemoryAwareRouter,
    MemoryServiceIntegration,
    RoutingMemoryInsight,
    SessionContextProvider,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
    RetrievalConfig,
)


@pytest.fixture
def retrieval_service() -> EnhancedRetrievalService:
    """Create retrieval service for tests."""
    config = RetrievalConfig()
    return EnhancedRetrievalService(config)


@pytest.fixture
def error_tracker() -> ErrorTracker:
    """Create error tracker for tests."""
    return ErrorTracker()


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create sample embedding vector."""
    return [0.1] * 1536  # Standard embedding dimension


@pytest.fixture
def sample_memory(sample_embedding: list[float]) -> MemoryRecord:
    """Create sample memory record."""
    return MemoryRecord(
        memory_id="test-memory-1",
        memory_layer=MemoryLayer.EPISODIC,
        content="Test memory content for integration testing",
        summary="Test memory",
        embedding=sample_embedding,
        agent_id="agent-1",
        session_id="session-123",
        task_id="task-456",
        keywords=["test", "integration"],
    )


@pytest.fixture
def sample_artifact() -> TaskArtifact:
    """Create sample task artifact."""
    return TaskArtifact(
        name="test_artifact",
        type="data",
        content={"key": "value", "data": [1, 2, 3]},
        created_at=datetime.now(UTC),
    )


class TestSessionContextProvider:
    """Tests for SessionContextProvider integration."""

    @pytest.mark.asyncio
    async def test_initialization(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test provider initialization."""
        provider = SessionContextProvider(retrieval_service=retrieval_service)

        assert provider.retrieval is not None
        assert provider._session_contexts == {}
        assert provider._memory_store == {}

    @pytest.mark.asyncio
    async def test_get_session_context_empty(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test getting context for session with no memories."""
        provider = SessionContextProvider(retrieval_service=retrieval_service)

        memories = await provider.get_session_context(
            session_id="session-123",
            query="test query",
        )

        assert memories == []
        assert "session-123" in provider._session_contexts

    @pytest.mark.asyncio
    async def test_persist_session_state(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test persisting session state as memory."""
        provider = SessionContextProvider(retrieval_service=retrieval_service)

        state_data = {
            "user_id": "user-1",
            "auth_level": "admin",
            "preferences": {"theme": "dark"},
        }

        memory_id = await provider.persist_session_state(
            session_id="session-123",
            state_data=state_data,
            agent_id="system",
        )

        assert memory_id is not None
        assert memory_id.startswith("session-state-session-123")
        assert memory_id in provider._memory_store

        # Verify memory content
        memory = provider._memory_store[memory_id]
        assert memory.session_id == "session-123"
        assert memory.is_critical is True
        assert "session" in memory.keywords
        assert "state" in memory.keywords

    @pytest.mark.asyncio
    async def test_add_strategic_insight(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test adding strategic insight to session context."""
        provider = SessionContextProvider(retrieval_service=retrieval_service)

        await provider.add_strategic_insight(
            session_id="session-123",
            insight="User prefers concise responses",
        )

        context = provider._session_contexts["session-123"]
        assert len(context.strategic_insights) == 1
        assert "concise" in context.strategic_insights[0]

    @pytest.mark.asyncio
    async def test_get_session_memory_stats(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test getting memory statistics for session."""
        provider = SessionContextProvider(retrieval_service=retrieval_service)

        # Test non-existent session
        stats = provider.get_session_memory_stats("nonexistent")
        assert stats["exists"] is False
        assert stats["memory_count"] == 0

        # Create context
        await provider.get_session_context(session_id="session-123")

        stats = provider.get_session_memory_stats("session-123")
        assert stats["exists"] is True
        assert stats["session_id"] == "session-123"

    @pytest.mark.asyncio
    async def test_context_with_error_tracker(
        self,
        retrieval_service: EnhancedRetrievalService,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test context retrieval with error tracker integration."""
        provider = SessionContextProvider(
            retrieval_service=retrieval_service,
            error_tracker=error_tracker,
        )

        # ErrorTracker requires task/agent context for full functionality
        # Just test that provider works with error tracker configured
        memories = await provider.get_session_context(
            session_id="session-123",
            query="error handling",
        )

        # Should use error state in retrieval
        assert isinstance(memories, list)


class TestMemoryAwareRouter:
    """Tests for MemoryAwareRouter integration."""

    @pytest.mark.asyncio
    async def test_initialization(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test router initialization."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        assert router.retrieval is not None
        assert router._agent_memories == {}
        assert router._agent_success_rates == {}
        assert router._routing_history == []

    @pytest.mark.asyncio
    async def test_get_routing_insights_empty(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test getting routing insights for agents with no history."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        insights = await router.get_routing_insights(
            candidate_agents=["agent-1", "agent-2", "agent-3"],
            required_capabilities=["code_analysis"],
        )

        assert len(insights) == 3
        assert all(isinstance(i, RoutingMemoryInsight) for i in insights)
        # All should have default success rate
        for insight in insights:
            assert insight.historical_success_rate == 0.5

    @pytest.mark.asyncio
    async def test_record_routing_outcome_success(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test recording successful routing outcome."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        await router.record_routing_outcome(
            agent_id="agent-1",
            task_type="code_analysis",
            success=True,
            memory_content="Successfully analyzed Python code",
        )

        # Success rate should increase
        assert router._agent_success_rates["agent-1"] > 0.5
        assert len(router._agent_memories["agent-1"]) == 1
        assert len(router._routing_history) == 1

    @pytest.mark.asyncio
    async def test_record_routing_outcome_failure(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test recording failed routing outcome."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        await router.record_routing_outcome(
            agent_id="agent-2",
            task_type="data_processing",
            success=False,
        )

        # Success rate should decrease
        assert router._agent_success_rates["agent-2"] < 0.5

    @pytest.mark.asyncio
    async def test_select_agent_with_memory(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test selecting agent based on memory insights."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        # Set up different success rates
        router._agent_success_rates["agent-1"] = 0.9
        router._agent_success_rates["agent-2"] = 0.3
        router._agent_success_rates["agent-3"] = 0.5

        selected = await router.select_agent_with_memory(
            candidates=["agent-1", "agent-2", "agent-3"],
        )

        # Should select agent with highest success rate
        assert selected == "agent-1"

    @pytest.mark.asyncio
    async def test_get_routing_stats(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test getting routing statistics."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        # Record some outcomes
        await router.record_routing_outcome("agent-1", "task-a", True)
        await router.record_routing_outcome("agent-2", "task-b", False)

        stats = router.get_routing_stats()

        assert stats["total_agents_tracked"] == 2
        assert stats["total_routing_events"] == 2
        assert "agent-1" in stats["agent_success_rates"]
        assert "agent-2" in stats["agent_success_rates"]

    @pytest.mark.asyncio
    async def test_insights_with_embedding(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_embedding: list[float],
    ) -> None:
        """Test routing insights with query embedding."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        # Add some memories to agent
        memory = MemoryRecord(
            memory_id="test-mem",
            memory_layer="episodic",
            content="Code analysis successful",
            summary="Success",
            embedding=sample_embedding,
            agent_id="agent-1",
            keywords=["code", "analysis"],
        )
        router._agent_memories["agent-1"] = [memory]

        insights = await router.get_routing_insights(
            candidate_agents=["agent-1"],
            query_embedding=sample_embedding,
        )

        assert len(insights) == 1
        # Should have high memory relevance due to matching embedding
        assert insights[0].memory_relevance_score > 0


class TestArtifactMemoryStorage:
    """Tests for ArtifactMemoryStorage integration."""

    @pytest.mark.asyncio
    async def test_initialization(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test storage initialization."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        assert storage._artifact_records == {}
        assert storage._memory_store == {}
        assert storage._task_artifacts == {}

    @pytest.mark.asyncio
    async def test_store_artifact(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
    ) -> None:
        """Test storing task artifact in memory."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        memory_id = await storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
            agent_id="agent-1",
        )

        assert memory_id is not None
        assert memory_id == f"artifact-{sample_artifact.name}"
        assert sample_artifact.name in storage._artifact_records
        assert memory_id in storage._memory_store
        assert "task-123" in storage._task_artifacts

    @pytest.mark.asyncio
    async def test_retrieve_artifact(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
    ) -> None:
        """Test retrieving stored artifact."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        await storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
        )

        memory = await storage.retrieve_artifact(sample_artifact.name)

        assert memory is not None
        assert memory.memory_layer == MemoryLayer.SEMANTIC
        assert "artifact" in memory.keywords
        assert sample_artifact.type in memory.keywords

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_artifact(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test retrieving non-existent artifact."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        memory = await storage.retrieve_artifact("nonexistent")

        assert memory is None

    @pytest.mark.asyncio
    async def test_find_similar_artifacts(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
        sample_embedding: list[float],
    ) -> None:
        """Test finding similar artifacts by embedding."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        # Store artifact with embedding
        await storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
            embedding=sample_embedding,
        )

        results = await storage.find_similar_artifacts(
            query_embedding=sample_embedding,
            limit=5,
        )

        assert len(results) == 1
        assert results[0][0].memory_id == f"artifact-{sample_artifact.name}"
        # High similarity score expected
        assert results[0][1] > 0.5

    @pytest.mark.asyncio
    async def test_get_task_artifacts(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
    ) -> None:
        """Test getting all artifacts for a task."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        await storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
        )

        memories = await storage.get_task_artifacts("task-123")

        assert len(memories) == 1
        assert memories[0].task_id == "task-123"

    @pytest.mark.asyncio
    async def test_get_storage_stats(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
    ) -> None:
        """Test getting storage statistics."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        await storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
        )

        stats = storage.get_storage_stats()

        assert stats["total_artifacts"] == 1
        assert stats["total_memories"] == 1
        assert stats["tasks_with_artifacts"] == 1


class TestACEStrategicContextInterface:
    """Tests for ACEStrategicContextInterface integration."""

    @pytest.mark.asyncio
    async def test_initialization(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test interface initialization."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        assert interface.retrieval is not None
        assert interface._strategic_contexts == {}
        assert interface._memory_store == {}

    @pytest.mark.asyncio
    async def test_build_strategic_context(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test building strategic context for agent."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        context = await interface.build_strategic_context(
            agent_id="agent-1",
            session_id="session-123",
            goal="optimize performance",
        )

        assert isinstance(context, ACEStrategicContext)
        assert context.agent_id == "agent-1"
        assert context.session_id == "session-123"
        assert context.current_goal == "optimize performance"
        assert context.confidence_score == 0.5  # Default

    @pytest.mark.asyncio
    async def test_update_confidence_success(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test updating confidence after success."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        # Build initial context
        await interface.build_strategic_context(agent_id="agent-1")

        # Update with success
        new_confidence = await interface.update_confidence(
            agent_id="agent-1",
            outcome_success=True,
        )

        # Confidence should increase
        assert new_confidence > 0.5
        assert new_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_update_confidence_failure(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test updating confidence after failure."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        # Build initial context
        await interface.build_strategic_context(agent_id="agent-1")

        # Update with failure
        new_confidence = await interface.update_confidence(
            agent_id="agent-1",
            outcome_success=False,
        )

        # Confidence should decrease
        assert new_confidence < 0.5
        assert new_confidence >= 0.0

    @pytest.mark.asyncio
    async def test_store_strategic_memory(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_embedding: list[float],
    ) -> None:
        """Test storing strategic memory."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        memory_id = await interface.store_strategic_memory(
            agent_id="agent-1",
            content="Learned that parallel processing improves performance",
            memory_layer="semantic",
            keywords=["optimization", "performance", "parallel"],
            embedding=sample_embedding,
        )

        assert memory_id is not None
        assert memory_id in interface._memory_store
        memory = interface._memory_store[memory_id]
        assert memory.agent_id == "agent-1"
        assert memory.memory_layer == "semantic"

    @pytest.mark.asyncio
    async def test_analyze_error_patterns(
        self,
        retrieval_service: EnhancedRetrievalService,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test error pattern analysis."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service,
            error_tracker=error_tracker,
        )

        # Add error-related memory instead of using ErrorTracker.record_error
        # (which requires task_id/agent_id context)
        await interface.store_strategic_memory(
            agent_id="agent-1",
            content="Error occurred during processing",
            keywords=["error", "failure"],
        )

        patterns = await interface.analyze_error_patterns(agent_id="agent-1")

        assert isinstance(patterns, list)
        # Should detect error memory
        assert any("error" in p.lower() for p in patterns)

    @pytest.mark.asyncio
    async def test_analyze_success_patterns(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test success pattern analysis."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        # Add success memory
        await interface.store_strategic_memory(
            agent_id="agent-1",
            content="Task completed successfully",
            keywords=["success", "completed"],
        )

        patterns = await interface.analyze_success_patterns(agent_id="agent-1")

        assert isinstance(patterns, list)
        assert len(patterns) > 0

    @pytest.mark.asyncio
    async def test_get_context_stats(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test getting context statistics."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        # Build context
        await interface.build_strategic_context(
            agent_id="agent-1",
            session_id="session-123",
        )

        stats = interface.get_context_stats()

        assert stats["total_contexts"] == 1
        assert "agent-1:session-123" in stats["contexts"]


class TestMemoryServiceIntegration:
    """Tests for unified MemoryServiceIntegration orchestrator."""

    @pytest.mark.asyncio
    async def test_initialization(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test unified integration initialization."""
        integration = MemoryServiceIntegration(
            retrieval_service=retrieval_service
        )

        assert integration.session_context is not None
        assert integration.memory_router is not None
        assert integration.artifact_storage is not None
        assert integration.ace_interface is not None

    @pytest.mark.asyncio
    async def test_initialization_with_config(self) -> None:
        """Test initialization with configuration objects."""
        integration = MemoryServiceIntegration(
            retrieval_config=RetrievalConfig(
                embedding_similarity_weight=0.4,
                recency_decay_weight=0.2,
                frequency_weight=0.1,
                stage_relevance_weight=0.15,
                criticality_weight=0.1,
                error_correction_weight=0.05,
            )
        )

        assert integration._retrieval.config.embedding_similarity_weight == 0.4

    @pytest.mark.asyncio
    async def test_health_check(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test health check for all components."""
        integration = MemoryServiceIntegration(
            retrieval_service=retrieval_service
        )

        health = await integration.health_check()

        assert health["session_context"] is True
        assert health["memory_router"] is True
        assert health["artifact_storage"] is True
        assert health["ace_interface"] is True
        assert health["error_tracker"] is True

    @pytest.mark.asyncio
    async def test_get_integration_stats(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test getting unified statistics."""
        integration = MemoryServiceIntegration(
            retrieval_service=retrieval_service
        )

        stats = integration.get_integration_stats()

        assert "session_context" in stats
        assert "memory_router" in stats
        assert "artifact_storage" in stats
        assert "ace_interface" in stats

    @pytest.mark.asyncio
    async def test_cross_service_workflow(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
        sample_embedding: list[float],
    ) -> None:
        """Test complete workflow across all integration services."""
        integration = MemoryServiceIntegration(
            retrieval_service=retrieval_service
        )

        # 1. Create session context
        session_memories = await integration.session_context.get_session_context(
            session_id="session-123",
            query="code analysis task",
        )
        assert isinstance(session_memories, list)

        # 2. Get routing insights
        insights = await integration.memory_router.get_routing_insights(
            candidate_agents=["agent-1", "agent-2"],
        )
        assert len(insights) == 2

        # 3. Record routing outcome
        await integration.memory_router.record_routing_outcome(
            agent_id="agent-1",
            task_type="code_analysis",
            success=True,
        )

        # 4. Store task artifact
        memory_id = await integration.artifact_storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
        )
        assert memory_id is not None

        # 5. Build ACE context
        ace_context = await integration.ace_interface.build_strategic_context(
            agent_id="agent-1",
            session_id="session-123",
            goal="complete code analysis",
        )
        assert ace_context.agent_id == "agent-1"

        # 6. Update confidence based on success
        confidence = await integration.ace_interface.update_confidence(
            agent_id="agent-1",
            session_id="session-123",
            outcome_success=True,
        )
        assert confidence > 0.5

        # 7. Verify unified stats
        stats = integration.get_integration_stats()
        assert stats["memory_router"]["total_routing_events"] == 1
        assert stats["artifact_storage"]["total_artifacts"] == 1


class TestCrossServiceContracts:
    """Tests for cross-service contract satisfaction."""

    @pytest.mark.asyncio
    async def test_session_manager_contract(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test SessionManager integration contract."""
        provider = SessionContextProvider(retrieval_service=retrieval_service)

        # Contract: Must persist session state
        memory_id = await provider.persist_session_state(
            session_id="session-1",
            state_data={"key": "value"},
        )
        assert memory_id is not None

        # Contract: Must retrieve session context
        memories = await provider.get_session_context(session_id="session-1")
        assert isinstance(memories, list)

        # Contract: Must provide memory statistics
        stats = provider.get_session_memory_stats("session-1")
        assert "memory_count" in stats

    @pytest.mark.asyncio
    async def test_message_router_contract(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test MessageRouter integration contract."""
        router = MemoryAwareRouter(retrieval_service=retrieval_service)

        # Contract: Must provide routing insights
        insights = await router.get_routing_insights(
            candidate_agents=["agent-1"]
        )
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert hasattr(insights[0], "agent_id")
        assert hasattr(insights[0], "historical_success_rate")

        # Contract: Must select agent
        selected = await router.select_agent_with_memory(
            candidates=["agent-1", "agent-2"]
        )
        assert selected in ["agent-1", "agent-2"]

        # Contract: Must record outcomes
        await router.record_routing_outcome("agent-1", "task", True)
        stats = router.get_routing_stats()
        assert stats["total_routing_events"] == 1

    @pytest.mark.asyncio
    async def test_task_manager_contract(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_artifact: TaskArtifact,
    ) -> None:
        """Test TaskManager integration contract."""
        storage = ArtifactMemoryStorage(retrieval_service=retrieval_service)

        # Contract: Must store artifacts
        memory_id = await storage.store_artifact(
            task_id="task-1",
            execution_id="exec-1",
            artifact=sample_artifact,
        )
        assert memory_id is not None

        # Contract: Must retrieve artifacts
        memory = await storage.retrieve_artifact(sample_artifact.name)
        assert memory is not None

        # Contract: Must find similar artifacts
        results = await storage.find_similar_artifacts(
            query_embedding=[0.1] * 1536,
            limit=5,
        )
        assert isinstance(results, list)

        # Contract: Must get task artifacts
        task_mems = await storage.get_task_artifacts("task-1")
        assert len(task_mems) == 1

    @pytest.mark.asyncio
    async def test_ace_strategic_contract(
        self, retrieval_service: EnhancedRetrievalService
    ) -> None:
        """Test ACE strategic context interface contract."""
        interface = ACEStrategicContextInterface(
            retrieval_service=retrieval_service
        )

        # Contract: Must build strategic context
        context = await interface.build_strategic_context(
            agent_id="agent-1",
            goal="test goal",
        )
        assert isinstance(context, ACEStrategicContext)
        assert context.agent_id == "agent-1"

        # Contract: Must analyze patterns
        error_patterns = await interface.analyze_error_patterns("agent-1")
        assert isinstance(error_patterns, list)

        success_patterns = await interface.analyze_success_patterns("agent-1")
        assert isinstance(success_patterns, list)

        # Contract: Must update confidence
        confidence = await interface.update_confidence("agent-1", outcome_success=True)
        assert 0.0 <= confidence <= 1.0

        # Contract: Must store strategic memories
        mem_id = await interface.store_strategic_memory(
            agent_id="agent-1",
            content="Strategic insight",
        )
        assert mem_id is not None
