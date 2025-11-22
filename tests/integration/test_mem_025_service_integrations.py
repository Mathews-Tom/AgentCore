"""
Integration tests for MEM-025: Service Integrations

Tests integration between memory services and core AgentCore services:
- SessionManager memory integration (session context)
- MessageRouter memory-aware routing
- TaskManager artifact storage
- ACE strategic context interface

Component ID: MEM-025
Ticket: MEM-025 (Implement Service Integrations)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
import structlog

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord, StageType
from agentcore.a2a_protocol.models.task import TaskArtifact, TaskPriority
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.integration import (
    ACEStrategicContext,
    ACEStrategicContextInterface,
    ArtifactMemoryStorage,
    MemoryAwareRouter,
    MemoryServiceIntegration,
    RoutingMemoryInsight,
    SessionContextProvider,
    SessionMemoryContext,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)

logger = structlog.get_logger(__name__)


class TestSessionContextProvider:
    """Test SessionContextProvider integration with SessionManager."""

    @pytest.fixture
    def session_provider(self) -> SessionContextProvider:
        """Create SessionContextProvider instance."""
        return SessionContextProvider()

    @pytest.fixture
    def sample_memories(self) -> list[MemoryRecord]:
        """Create sample memories for testing."""
        return [
            MemoryRecord(
                memory_id=f"mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Session memory {i}",
                summary=f"Summary {i}",
                embedding=[0.1 * i] * 768,
                session_id="session-123",
                agent_id="agent-1",
                task_id=None,
                keywords=[f"keyword{i}"],
            )
            for i in range(10)
        ]

    @pytest.mark.asyncio
    async def test_get_session_context_basic(
        self,
        session_provider: SessionContextProvider,
        sample_memories: list[MemoryRecord],
    ):
        """Test basic session context retrieval."""
        # Add memories to provider's internal store
        for mem in sample_memories:
            session_provider._memory_store[mem.memory_id] = mem

        # Retrieve context
        query_embedding = [0.5] * 768
        results = await session_provider.get_session_context(
            session_id="session-123",
            query="test query",
            query_embedding=query_embedding,
            max_memories=5,
        )

        # Assert results
        assert len(results) <= 5
        assert all(mem.session_id == "session-123" for mem in results)
        assert "session-123" in session_provider._session_contexts

    @pytest.mark.asyncio
    async def test_get_session_context_with_stage(
        self,
        session_provider: SessionContextProvider,
        sample_memories: list[MemoryRecord],
    ):
        """Test session context retrieval with stage filtering."""
        # Add stage info to some memories
        for i, mem in enumerate(sample_memories[:5]):
            mem.stage_id = f"stage-{i % 3}"
            session_provider._memory_store[mem.memory_id] = mem

        query_embedding = [0.5] * 768
        results = await session_provider.get_session_context(
            session_id="session-123",
            query_embedding=query_embedding,
            current_stage=StageType.EXECUTION,
            max_memories=5,
        )

        # Assert results
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_persist_session_state(
        self,
        session_provider: SessionContextProvider,
    ):
        """Test session state persistence in memory."""
        state_data = {
            "user_id": "user-1",
            "auth_level": "admin",
            "session_vars": {"theme": "dark"},
        }

        memory_id = await session_provider.persist_session_state(
            session_id="session-123",
            state_data=state_data,
            agent_id="agent-1",
        )

        # Assert memory created
        assert memory_id.startswith("session-state-session-123")
        assert memory_id in session_provider._memory_store

        stored_memory = session_provider._memory_store[memory_id]
        assert stored_memory.is_critical is True
        assert stored_memory.session_id == "session-123"
        assert "session" in stored_memory.keywords

    @pytest.mark.asyncio
    async def test_add_strategic_insight(
        self,
        session_provider: SessionContextProvider,
    ):
        """Test adding strategic insights to session context."""
        await session_provider.add_strategic_insight(
            session_id="session-123",
            insight="User prefers detailed explanations",
        )

        context = session_provider._session_contexts["session-123"]
        assert len(context.strategic_insights) == 1
        assert context.strategic_insights[0] == "User prefers detailed explanations"

    def test_get_session_memory_stats(
        self,
        session_provider: SessionContextProvider,
    ):
        """Test session memory statistics retrieval."""
        # Create session context
        session_provider._session_contexts["session-123"] = SessionMemoryContext(
            session_id="session-123",
            memory_ids=["mem-1", "mem-2", "mem-3"],
            context_size_bytes=1024,
        )

        stats = session_provider.get_session_memory_stats("session-123")

        assert stats["exists"] is True
        assert stats["memory_count"] == 3
        assert stats["context_size_bytes"] == 1024
        assert stats["session_id"] == "session-123"

    def test_get_session_memory_stats_nonexistent(
        self,
        session_provider: SessionContextProvider,
    ):
        """Test stats for nonexistent session."""
        stats = session_provider.get_session_memory_stats("session-999")

        assert stats["exists"] is False
        assert stats["memory_count"] == 0


class TestMemoryAwareRouter:
    """Test MemoryAwareRouter integration with MessageRouter."""

    @pytest.fixture
    def memory_router(self) -> MemoryAwareRouter:
        """Create MemoryAwareRouter instance."""
        return MemoryAwareRouter()

    @pytest.fixture
    def agent_memories(self) -> dict[str, list[MemoryRecord]]:
        """Create sample agent memories."""
        return {
            "agent-1": [
                MemoryRecord(
                    memory_id="mem-a1-1",
                    memory_layer=MemoryLayer.EPISODIC,
                    content="code analysis python security vulnerability",
                    summary="code analysis",
                    embedding=[0.9] * 768,
                    agent_id="agent-1",
                    session_id=None,
                    task_id=None,
                    keywords=["code_analysis", "python", "security"],
                ),
            ],
            "agent-2": [
                MemoryRecord(
                    memory_id="mem-a2-1",
                    memory_layer=MemoryLayer.EPISODIC,
                    content="data processing analytics",
                    summary="data processing",
                    embedding=[0.3] * 768,
                    agent_id="agent-2",
                    session_id=None,
                    task_id=None,
                    keywords=["data_processing", "analytics"],
                ),
            ],
        }

    @pytest.mark.asyncio
    async def test_get_routing_insights(
        self,
        memory_router: MemoryAwareRouter,
        agent_memories: dict[str, list[MemoryRecord]],
    ):
        """Test routing insights generation."""
        # Set up agent memories
        memory_router._agent_memories = agent_memories
        memory_router._agent_success_rates = {
            "agent-1": 0.9,
            "agent-2": 0.7,
        }

        insights = await memory_router.get_routing_insights(
            candidate_agents=["agent-1", "agent-2"],
            query="code analysis for security",
            required_capabilities=["code_analysis"],
        )

        # Assert insights generated
        assert len(insights) == 2
        assert all(isinstance(insight, RoutingMemoryInsight) for insight in insights)
        assert insights[0].agent_id in ["agent-1", "agent-2"]
        # At least one insight should be generated (success rates differ)
        assert insights[0].historical_success_rate >= insights[1].historical_success_rate

    @pytest.mark.asyncio
    async def test_select_agent_with_memory(
        self,
        memory_router: MemoryAwareRouter,
        agent_memories: dict[str, list[MemoryRecord]],
    ):
        """Test agent selection based on memory insights."""
        memory_router._agent_memories = agent_memories
        memory_router._agent_success_rates = {
            "agent-1": 0.9,
            "agent-2": 0.7,
        }

        query_embedding = [0.9] * 768  # Similar to agent-1 memory
        selected = await memory_router.select_agent_with_memory(
            candidates=["agent-1", "agent-2"],
            query_embedding=query_embedding,
            task_type="code_analysis",
        )

        # Assert selection
        assert selected in ["agent-1", "agent-2"]

    @pytest.mark.asyncio
    async def test_record_routing_outcome_success(
        self,
        memory_router: MemoryAwareRouter,
    ):
        """Test recording successful routing outcome."""
        await memory_router.record_routing_outcome(
            agent_id="agent-1",
            task_type="code_analysis",
            success=True,
            memory_content="Successfully analyzed code for vulnerabilities",
        )

        # Assert success rate updated
        assert "agent-1" in memory_router._agent_success_rates
        success_rate = memory_router._agent_success_rates["agent-1"]
        assert success_rate > 0.5

        # Assert memory stored
        assert "agent-1" in memory_router._agent_memories
        assert len(memory_router._agent_memories["agent-1"]) == 1

    @pytest.mark.asyncio
    async def test_record_routing_outcome_failure(
        self,
        memory_router: MemoryAwareRouter,
    ):
        """Test recording failed routing outcome."""
        # Set initial success rate
        memory_router._agent_success_rates["agent-1"] = 0.8

        await memory_router.record_routing_outcome(
            agent_id="agent-1",
            task_type="code_analysis",
            success=False,
        )

        # Assert success rate decreased
        new_rate = memory_router._agent_success_rates["agent-1"]
        assert new_rate < 0.8

    def test_get_routing_stats(
        self,
        memory_router: MemoryAwareRouter,
    ):
        """Test routing statistics retrieval."""
        memory_router._agent_success_rates = {
            "agent-1": 0.9,
            "agent-2": 0.7,
        }
        memory_router._agent_memories = {
            "agent-1": [MemoryRecord(
                memory_id="mem-1",
                memory_layer=MemoryLayer.EPISODIC,
                content="test",
                summary="test",
                embedding=[],
                agent_id="agent-1",
                session_id=None,
                task_id=None,
            )],
        }

        stats = memory_router.get_routing_stats()

        assert stats["total_agents_tracked"] == 2
        assert stats["total_memories"] == 1
        assert "agent_success_rates" in stats


class TestArtifactMemoryStorage:
    """Test ArtifactMemoryStorage integration with TaskManager."""

    @pytest.fixture
    def artifact_storage(self) -> ArtifactMemoryStorage:
        """Create ArtifactMemoryStorage instance."""
        return ArtifactMemoryStorage()

    @pytest.fixture
    def sample_artifact(self) -> TaskArtifact:
        """Create sample task artifact."""
        return TaskArtifact(
            name="analysis_report",
            type="json",
            content={"findings": ["issue1", "issue2"], "score": 85},
            metadata={"generated_by": "agent-1"},
        )

    @pytest.mark.asyncio
    async def test_store_artifact(
        self,
        artifact_storage: ArtifactMemoryStorage,
        sample_artifact: TaskArtifact,
    ):
        """Test artifact storage in memory."""
        memory_id = await artifact_storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
            embedding=[0.5] * 768,
            agent_id="agent-1",
        )

        # Assert artifact stored
        assert memory_id.startswith("artifact-")
        assert memory_id in artifact_storage._memory_store
        assert sample_artifact.name in artifact_storage._artifact_records

        # Assert memory properties
        stored_memory = artifact_storage._memory_store[memory_id]
        assert stored_memory.task_id == "task-123"
        assert stored_memory.memory_layer == MemoryLayer.SEMANTIC
        assert "artifact" in stored_memory.keywords

    @pytest.mark.asyncio
    async def test_retrieve_artifact(
        self,
        artifact_storage: ArtifactMemoryStorage,
        sample_artifact: TaskArtifact,
    ):
        """Test artifact retrieval from memory."""
        # Store artifact first
        await artifact_storage.store_artifact(
            task_id="task-123",
            execution_id="exec-456",
            artifact=sample_artifact,
            agent_id="agent-1",
        )

        # Retrieve artifact
        memory = await artifact_storage.retrieve_artifact(sample_artifact.name)

        # Assert retrieved
        assert memory is not None
        assert memory.task_id == "task-123"
        assert "artifact" in memory.keywords

        # Assert retrieval count incremented
        record = artifact_storage._artifact_records[sample_artifact.name]
        assert record.retrieval_count == 1

    @pytest.mark.asyncio
    async def test_find_similar_artifacts(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ):
        """Test finding similar artifacts by embedding."""
        # Store multiple artifacts
        for i in range(5):
            artifact = TaskArtifact(
                name=f"artifact-{i}",
                type="json",
                content={"data": i},
            )
            await artifact_storage.store_artifact(
                task_id=f"task-{i}",
                execution_id=f"exec-{i}",
                artifact=artifact,
                embedding=[0.1 * i] * 768,
            )

        # Search for similar artifacts
        query_embedding = [0.25] * 768
        results = await artifact_storage.find_similar_artifacts(
            query_embedding=query_embedding,
            limit=3,
        )

        # Assert results
        assert len(results) <= 3
        assert all(isinstance(mem, MemoryRecord) for mem, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    @pytest.mark.asyncio
    async def test_get_task_artifacts(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ):
        """Test retrieving all artifacts for a task."""
        task_id = "task-123"

        # Store multiple artifacts for task
        for i in range(3):
            artifact = TaskArtifact(
                name=f"artifact-{i}",
                type="json",
                content={"data": i},
            )
            await artifact_storage.store_artifact(
                task_id=task_id,
                execution_id=f"exec-{i}",
                artifact=artifact,
            )

        # Get task artifacts
        artifacts = await artifact_storage.get_task_artifacts(task_id)

        # Assert all artifacts retrieved
        assert len(artifacts) == 3
        assert all(mem.task_id == task_id for mem in artifacts)

    def test_get_storage_stats(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ):
        """Test artifact storage statistics."""
        stats = artifact_storage.get_storage_stats()

        assert "total_artifacts" in stats
        assert "total_memories" in stats
        assert "tasks_with_artifacts" in stats


class TestACEStrategicContextInterface:
    """Test ACEStrategicContextInterface for ACE framework integration."""

    @pytest.fixture
    def ace_interface(self) -> ACEStrategicContextInterface:
        """Create ACEStrategicContextInterface instance."""
        return ACEStrategicContextInterface()

    @pytest.fixture
    def sample_memories(self) -> list[MemoryRecord]:
        """Create sample memories for ACE."""
        return [
            # Strategic memories (semantic/procedural)
            MemoryRecord(
                memory_id="mem-s1",
                memory_layer=MemoryLayer.SEMANTIC,
                content="High-level planning strategy",
                summary="planning strategy",
                embedding=[0.9] * 768,
                agent_id="agent-1",
                session_id=None,
                task_id=None,
                keywords=["planning", "strategy"],
            ),
            MemoryRecord(
                memory_id="mem-s2",
                memory_layer=MemoryLayer.PROCEDURAL,
                content="Execution workflow pattern",
                summary="workflow pattern",
                embedding=[0.8] * 768,
                agent_id="agent-1",
                session_id=None,
                task_id=None,
                keywords=["workflow", "execution"],
            ),
            # Tactical memories (episodic)
            MemoryRecord(
                memory_id="mem-t1",
                memory_layer=MemoryLayer.EPISODIC,
                content="Recent task execution",
                summary="recent execution",
                embedding=[0.7] * 768,
                agent_id="agent-1",
                session_id="session-123",
                task_id=None,
                keywords=["execution", "recent"],
            ),
        ]

    @pytest.mark.asyncio
    async def test_build_strategic_context(
        self,
        ace_interface: ACEStrategicContextInterface,
        sample_memories: list[MemoryRecord],
    ):
        """Test building strategic context for ACE."""
        # Add memories to interface
        for mem in sample_memories:
            ace_interface._memory_store[mem.memory_id] = mem

        query_embedding = [0.5] * 768
        context = await ace_interface.build_strategic_context(
            agent_id="agent-1",
            session_id="session-123",
            goal="optimize system performance",
            query_embedding=query_embedding,
        )

        # Assert context built
        assert isinstance(context, ACEStrategicContext)
        assert context.agent_id == "agent-1"
        assert context.session_id == "session-123"
        assert context.current_goal == "optimize system performance"
        assert len(context.strategic_memories) > 0
        assert len(context.tactical_memories) > 0

    @pytest.mark.asyncio
    async def test_update_confidence_success(
        self,
        ace_interface: ACEStrategicContextInterface,
    ):
        """Test confidence update on successful outcome."""
        initial_confidence = 0.5

        # Create initial context
        ace_interface._strategic_contexts["agent-1:global"] = ACEStrategicContext(
            agent_id="agent-1",
            confidence_score=initial_confidence,
        )

        # Update with successful outcome
        new_confidence = await ace_interface.update_confidence(
            agent_id="agent-1",
            outcome_success=True,
        )

        # Assert confidence increased
        assert new_confidence > initial_confidence
        assert new_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_update_confidence_failure(
        self,
        ace_interface: ACEStrategicContextInterface,
    ):
        """Test confidence update on failed outcome."""
        initial_confidence = 0.7

        # Create initial context
        ace_interface._strategic_contexts["agent-1:global"] = ACEStrategicContext(
            agent_id="agent-1",
            confidence_score=initial_confidence,
        )

        # Update with failed outcome
        new_confidence = await ace_interface.update_confidence(
            agent_id="agent-1",
            outcome_success=False,
        )

        # Assert confidence decreased
        assert new_confidence < initial_confidence
        assert new_confidence >= 0.0

    @pytest.mark.asyncio
    async def test_store_strategic_memory(
        self,
        ace_interface: ACEStrategicContextInterface,
    ):
        """Test storing strategic memory for agent."""
        memory_id = await ace_interface.store_strategic_memory(
            agent_id="agent-1",
            content="Long-term optimization strategy",
            memory_layer="semantic",
            keywords=["optimization", "strategy"],
            embedding=[0.5] * 768,
        )

        # Assert memory stored
        assert memory_id.startswith("ace-agent-1")
        assert memory_id in ace_interface._memory_store

        stored_memory = ace_interface._memory_store[memory_id]
        assert stored_memory.memory_layer == MemoryLayer.SEMANTIC
        assert stored_memory.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_analyze_error_patterns(
        self,
        ace_interface: ACEStrategicContextInterface,
    ):
        """Test error pattern analysis from memories."""
        # Add error-related memories
        for i in range(3):
            memory = MemoryRecord(
                memory_id=f"mem-err-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Error occurred: {i}",
                summary=f"error {i}",
                embedding=[],
                agent_id="agent-1",
                session_id=None,
                task_id=None,
                keywords=["error", "failure"],
            )
            ace_interface._memory_store[memory.memory_id] = memory

        patterns = await ace_interface.analyze_error_patterns(agent_id="agent-1")

        # Assert patterns detected
        assert len(patterns) > 0
        assert any("error" in pattern.lower() for pattern in patterns)

    def test_get_context_stats(
        self,
        ace_interface: ACEStrategicContextInterface,
    ):
        """Test strategic context statistics."""
        # Create sample contexts
        ace_interface._strategic_contexts["agent-1:global"] = ACEStrategicContext(
            agent_id="agent-1",
            strategic_memories=["mem-1", "mem-2"],
            tactical_memories=["mem-3"],
        )

        stats = ace_interface.get_context_stats()

        assert stats["total_contexts"] == 1
        assert "contexts" in stats
        assert "agent-1:global" in stats["contexts"]


class TestMemoryServiceIntegration:
    """Test unified MemoryServiceIntegration orchestrator."""

    @pytest.fixture
    def integration(self) -> MemoryServiceIntegration:
        """Create MemoryServiceIntegration instance."""
        return MemoryServiceIntegration()

    def test_initialization(
        self,
        integration: MemoryServiceIntegration,
    ):
        """Test integration initialization."""
        assert integration.session_context is not None
        assert integration.memory_router is not None
        assert integration.artifact_storage is not None
        assert integration.ace_interface is not None

    def test_get_integration_stats(
        self,
        integration: MemoryServiceIntegration,
    ):
        """Test comprehensive integration statistics."""
        stats = integration.get_integration_stats()

        assert "session_context" in stats
        assert "memory_router" in stats
        assert "artifact_storage" in stats
        assert "ace_interface" in stats

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        integration: MemoryServiceIntegration,
    ):
        """Test health check for all components."""
        health = await integration.health_check()

        assert health["session_context"] is True
        assert health["memory_router"] is True
        assert health["artifact_storage"] is True
        assert health["ace_interface"] is True


class TestCrossComponentIntegration:
    """Test cross-component integration scenarios."""

    @pytest.fixture
    def integration(self) -> MemoryServiceIntegration:
        """Create fully integrated memory service."""
        return MemoryServiceIntegration()

    @pytest.mark.asyncio
    async def test_session_to_routing_flow(
        self,
        integration: MemoryServiceIntegration,
    ):
        """Test flow from session context to routing decision."""
        # 1. Create session context
        session_id = "session-flow-123"
        memory = MemoryRecord(
            memory_id="mem-flow-1",
            memory_layer=MemoryLayer.EPISODIC,
            content="User prefers agent-1 for code tasks",
            summary="user preference",
            embedding=[0.9] * 768,
            session_id=session_id,
            agent_id="agent-1",
            task_id=None,
            keywords=["preference", "code"],
        )
        integration.session_context._memory_store[memory.memory_id] = memory

        # 2. Get session context
        session_memories = await integration.session_context.get_session_context(
            session_id=session_id,
            query="code analysis task",
            query_embedding=[0.9] * 768,
        )

        # 3. Use context for routing
        integration.memory_router._agent_memories["agent-1"] = session_memories
        integration.memory_router._agent_success_rates["agent-1"] = 0.9

        selected_agent = await integration.memory_router.select_agent_with_memory(
            candidates=["agent-1", "agent-2"],
            query_embedding=[0.9] * 768,
            task_type="code_analysis",
        )

        # Assert flow completed
        assert selected_agent == "agent-1"

    @pytest.mark.asyncio
    async def test_task_artifact_to_ace_context_flow(
        self,
        integration: MemoryServiceIntegration,
    ):
        """Test flow from task artifact storage to ACE strategic context."""
        # 1. Store task artifact
        artifact = TaskArtifact(
            name="performance_analysis",
            type="json",
            content={"cpu_usage": 85, "memory_usage": 70},
        )
        memory_id = await integration.artifact_storage.store_artifact(
            task_id="task-perf-123",
            execution_id="exec-456",
            artifact=artifact,
            embedding=[0.8] * 768,
            agent_id="agent-1",
        )

        # 2. Store strategic memory in ACE interface
        await integration.ace_interface.store_strategic_memory(
            agent_id="agent-1",
            content="Performance optimization strategy",
            memory_layer="semantic",
            keywords=["performance", "optimization"],
            embedding=[0.8] * 768,
        )

        # 3. Build strategic context with artifact context
        context = await integration.ace_interface.build_strategic_context(
            agent_id="agent-1",
            goal="optimize system performance",
            query_embedding=[0.8] * 768,
        )

        # Assert flow completed
        assert context.agent_id == "agent-1"
        assert context.current_goal == "optimize system performance"

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(
        self,
        integration: MemoryServiceIntegration,
    ):
        """Test complete memory lifecycle across all integrations."""
        agent_id = "agent-lifecycle"
        session_id = "session-lifecycle"
        task_id = "task-lifecycle"

        # 1. Session starts: persist state
        await integration.session_context.persist_session_state(
            session_id=session_id,
            state_data={"started": True},
            agent_id=agent_id,
        )

        # 2. Task artifact created
        artifact = TaskArtifact(
            name="lifecycle_artifact",
            type="text",
            content="Task output data",
        )
        await integration.artifact_storage.store_artifact(
            task_id=task_id,
            execution_id="exec-lifecycle",
            artifact=artifact,
            agent_id=agent_id,
        )

        # 3. Routing decision recorded
        await integration.memory_router.record_routing_outcome(
            agent_id=agent_id,
            task_type="lifecycle_test",
            success=True,
            memory_content="Lifecycle test completed successfully",
        )

        # 4. ACE strategic context updated
        await integration.ace_interface.update_confidence(
            agent_id=agent_id,
            session_id=session_id,
            outcome_success=True,
        )

        # 5. Verify all integrations have data
        session_stats = integration.session_context.get_session_memory_stats(session_id)
        routing_stats = integration.memory_router.get_routing_stats()
        storage_stats = integration.artifact_storage.get_storage_stats()
        ace_stats = integration.ace_interface.get_context_stats()

        # Assert lifecycle completed
        assert session_stats["exists"] is True
        assert routing_stats["total_agents_tracked"] >= 1
        assert storage_stats["total_artifacts"] >= 1
        assert ace_stats["total_contexts"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
