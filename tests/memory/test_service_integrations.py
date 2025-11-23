"""
Integration Tests for Memory Service Integrations

Tests for MEM-025: Service Integrations
- SessionManager memory integration
- MessageRouter memory-aware routing
- TaskManager artifact storage
- ACE strategic context interface

All integration tests with each service.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentcore.a2a_protocol.models.memory import MemoryLayer, StageType
from agentcore.a2a_protocol.models.task import TaskArtifact
from agentcore.a2a_protocol.services.memory.integrations.ace_interface import (
    ACEStrategicContextInterface,
)
from agentcore.a2a_protocol.services.memory.integrations.message_router import (
    MemoryAwareRouter,
)
from agentcore.a2a_protocol.services.memory.integrations.session_manager import (
    SessionContextProvider,
)
from agentcore.a2a_protocol.services.memory.integrations.task_manager import (
    ArtifactMemoryStorage,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)


class TestSessionManagerIntegration:
    """Tests for SessionManager memory integration."""

    @pytest.fixture
    def session_provider(self) -> SessionContextProvider:
        """Create SessionContextProvider instance."""
        return SessionContextProvider()

    @pytest.mark.asyncio
    async def test_store_session_context(
        self,
        session_provider: SessionContextProvider,
    ) -> None:
        """Test storing session context in memory."""
        # Arrange
        session_id = "session-123"
        context_data = {
            "user_id": "user-1",
            "auth_level": "admin",
            "preferences": {"theme": "dark"},
        }

        # Act
        memory_id = await session_provider.store_session_context(
            session_id=session_id,
            context_data=context_data,
        )

        # Assert
        assert memory_id is not None
        assert memory_id.startswith("session-ctx-")

        # Verify storage
        stats = session_provider.get_session_memory_stats(session_id)
        assert stats["exists"] is True
        assert stats["memory_count"] > 0

    @pytest.mark.asyncio
    async def test_retrieve_session_context(
        self,
        session_provider: SessionContextProvider,
    ) -> None:
        """Test retrieving session context from memory."""
        # Arrange
        session_id = "session-456"
        original_data = {"key1": "value1", "key2": "value2"}
        await session_provider.store_session_context(
            session_id=session_id,
            context_data=original_data,
        )

        # Act
        retrieved_data = await session_provider.retrieve_session_context(
            session_id=session_id
        )

        # Assert
        assert retrieved_data is not None
        assert retrieved_data["key1"] == "value1"
        assert retrieved_data["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_update_session_memory(
        self,
        session_provider: SessionContextProvider,
    ) -> None:
        """Test updating session memory."""
        # Arrange
        session_id = "session-789"
        initial_data = {"counter": 0, "status": "active"}
        await session_provider.store_session_context(
            session_id=session_id,
            context_data=initial_data,
        )

        # Act
        updates = {"counter": 5, "last_action": "login"}
        memory_id = await session_provider.update_session_memory(
            session_id=session_id,
            updates=updates,
        )

        # Assert
        assert memory_id is not None

        # Verify update
        retrieved = await session_provider.retrieve_session_context(session_id)
        assert retrieved is not None
        assert retrieved["counter"] == 5
        assert retrieved["status"] == "active"
        assert retrieved["last_action"] == "login"

    @pytest.mark.asyncio
    async def test_get_session_context_with_query(
        self,
        session_provider: SessionContextProvider,
    ) -> None:
        """Test getting session context with semantic query."""
        # Arrange
        session_id = "session-context-test"
        query_embedding = [0.1] * 1536  # Mock embedding

        # Act
        memories = await session_provider.get_session_context(
            session_id=session_id,
            query="user authentication",
            query_embedding=query_embedding,
            max_memories=10,
        )

        # Assert
        assert isinstance(memories, list)
        # Even with empty store, should return empty list without error

    @pytest.mark.asyncio
    async def test_persist_session_state(
        self,
        session_provider: SessionContextProvider,
    ) -> None:
        """Test persisting session state."""
        # Arrange
        session_id = "session-state-test"
        state_data = {"authenticated": True, "user_id": "user-123"}

        # Act
        memory_id = await session_provider.persist_session_state(
            session_id=session_id,
            state_data=state_data,
        )

        # Assert
        assert memory_id is not None
        assert memory_id.startswith("session-state-")

    @pytest.mark.asyncio
    async def test_add_strategic_insight(
        self,
        session_provider: SessionContextProvider,
    ) -> None:
        """Test adding strategic insight to session."""
        # Arrange
        session_id = "session-insight-test"
        insight = "User prefers concise responses"

        # Act
        await session_provider.add_strategic_insight(
            session_id=session_id,
            insight=insight,
        )

        # Assert
        stats = session_provider.get_session_memory_stats(session_id)
        assert stats["strategic_insights_count"] == 1


class TestMessageRouterIntegration:
    """Tests for MessageRouter memory-aware routing."""

    @pytest.fixture
    def memory_router(self) -> MemoryAwareRouter:
        """Create MemoryAwareRouter instance."""
        return MemoryAwareRouter()

    @pytest.mark.asyncio
    async def test_get_relevant_context(
        self,
        memory_router: MemoryAwareRouter,
    ) -> None:
        """Test getting relevant context for message routing."""
        # Arrange
        conversation_id = "conv-123"
        message = "analyze Python code for vulnerabilities"

        # Act
        context = await memory_router.get_relevant_context(
            message=message,
            conversation_id=conversation_id,
        )

        # Assert
        assert isinstance(context, list)
        # Empty conversation should return empty list

    @pytest.mark.asyncio
    async def test_enhance_message_with_context(
        self,
        memory_router: MemoryAwareRouter,
    ) -> None:
        """Test enhancing message with contextual memories."""
        # Arrange
        message = "run tests"
        context = []  # Empty context for now

        # Act
        enhanced = await memory_router.enhance_message_with_context(
            message=message,
            context=context,
        )

        # Assert
        assert enhanced == message  # No context, should return original

    @pytest.mark.asyncio
    async def test_route_with_memory(
        self,
        memory_router: MemoryAwareRouter,
    ) -> None:
        """Test routing message with memory-aware selection."""
        # Arrange
        message = "deploy application"
        candidates = ["agent-1", "agent-2", "agent-3"]

        # Act
        selected = await memory_router.route_with_memory(
            message=message,
            candidates=candidates,
        )

        # Assert
        assert selected is not None
        assert selected in candidates

    @pytest.mark.asyncio
    async def test_get_routing_insights(
        self,
        memory_router: MemoryAwareRouter,
    ) -> None:
        """Test getting routing insights for candidates."""
        # Arrange
        candidates = ["agent-a", "agent-b"]
        query_embedding = [0.2] * 1536

        # Act
        insights = await memory_router.get_routing_insights(
            candidate_agents=candidates,
            query="code review",
            query_embedding=query_embedding,
            required_capabilities=["code_analysis"],
        )

        # Assert
        assert len(insights) == 2
        assert all(insight.agent_id in candidates for insight in insights)
        assert all(hasattr(insight, "capability_match_score") for insight in insights)
        assert all(hasattr(insight, "historical_success_rate") for insight in insights)

    @pytest.mark.asyncio
    async def test_record_routing_outcome(
        self,
        memory_router: MemoryAwareRouter,
    ) -> None:
        """Test recording routing outcome for learning."""
        # Arrange
        agent_id = "agent-x"
        task_type = "data_analysis"

        # Act
        await memory_router.record_routing_outcome(
            agent_id=agent_id,
            task_type=task_type,
            success=True,
            memory_content="Successfully analyzed dataset",
        )

        # Assert
        stats = memory_router.get_routing_stats()
        assert stats["total_agents_tracked"] == 1
        assert stats["total_routing_events"] == 1
        assert agent_id in stats["agent_success_rates"]

    @pytest.mark.asyncio
    async def test_select_agent_with_memory(
        self,
        memory_router: MemoryAwareRouter,
    ) -> None:
        """Test selecting agent based on memory insights."""
        # Arrange
        candidates = ["agent-1", "agent-2"]
        query_embedding = [0.3] * 1536

        # Act
        selected = await memory_router.select_agent_with_memory(
            candidates=candidates,
            query_embedding=query_embedding,
            task_type="testing",
        )

        # Assert
        assert selected is not None
        assert selected in candidates


class TestTaskManagerIntegration:
    """Tests for TaskManager artifact storage integration."""

    @pytest.fixture
    def artifact_storage(self) -> ArtifactMemoryStorage:
        """Create ArtifactMemoryStorage instance."""
        return ArtifactMemoryStorage()

    @pytest.mark.asyncio
    async def test_store_task_artifact(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ) -> None:
        """Test storing task artifact in memory."""
        # Arrange
        task_id = "task-123"
        artifact_data = {
            "name": "test-result",
            "type": "result",
            "content": {"status": "passed", "coverage": 95},
        }

        # Act
        memory_id = await artifact_storage.store_task_artifact(
            task_id=task_id,
            artifact_data=artifact_data,
        )

        # Assert
        assert memory_id is not None
        assert memory_id.startswith("artifact-")

        # Verify storage
        stats = artifact_storage.get_storage_stats()
        assert stats["total_artifacts"] == 1
        assert stats["tasks_with_artifacts"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_task_artifacts(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ) -> None:
        """Test retrieving task artifacts."""
        # Arrange
        task_id = "task-456"
        artifact1 = {"name": "artifact1", "type": "data", "content": {"value": 1}}
        artifact2 = {"name": "artifact2", "type": "data", "content": {"value": 2}}

        await artifact_storage.store_task_artifact(task_id, artifact1)
        await artifact_storage.store_task_artifact(task_id, artifact2)

        # Act
        artifacts = await artifact_storage.retrieve_task_artifacts(task_id)

        # Assert
        assert len(artifacts) == 2
        assert any(a["name"] == "artifact1" for a in artifacts)
        assert any(a["name"] == "artifact2" for a in artifacts)

    @pytest.mark.asyncio
    async def test_link_artifacts(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ) -> None:
        """Test linking artifacts between tasks."""
        # Arrange
        task_id_1 = "task-a"
        task_id_2 = "task-b"
        relationship = "depends_on"

        # Act
        await artifact_storage.link_artifacts(
            task_id_1=task_id_1,
            task_id_2=task_id_2,
            relationship=relationship,
        )

        # Assert
        stats = artifact_storage.get_storage_stats()
        assert stats["total_links"] == 1

    @pytest.mark.asyncio
    async def test_store_artifact_legacy(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ) -> None:
        """Test storing artifact using legacy TaskArtifact model."""
        # Arrange
        task_id = "task-legacy"
        execution_id = "exec-1"
        artifact = TaskArtifact(
            name="legacy-artifact",
            type="data",
            content={"result": "success"},
            metadata={},
        )

        # Act
        memory_id = await artifact_storage.store_artifact(
            task_id=task_id,
            execution_id=execution_id,
            artifact=artifact,
        )

        # Assert
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_find_similar_artifacts(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ) -> None:
        """Test finding similar artifacts by embedding."""
        # Arrange
        query_embedding = [0.4] * 1536

        # Act
        results = await artifact_storage.find_similar_artifacts(
            query_embedding=query_embedding,
            limit=5,
        )

        # Assert
        assert isinstance(results, list)
        # Empty storage should return empty list

    @pytest.mark.asyncio
    async def test_get_task_artifacts(
        self,
        artifact_storage: ArtifactMemoryStorage,
    ) -> None:
        """Test getting all artifacts for a task."""
        # Arrange
        task_id = "task-get-all"
        artifact_data = {"name": "test", "type": "result", "content": {}}
        await artifact_storage.store_task_artifact(task_id, artifact_data)

        # Act
        memories = await artifact_storage.get_task_artifacts(task_id)

        # Assert
        assert len(memories) == 1
        assert "artifact" in memories[0].keywords


class TestACEInterfaceIntegration:
    """Tests for ACE strategic context interface."""

    @pytest.fixture
    def ace_interface(self) -> ACEStrategicContextInterface:
        """Create ACEStrategicContextInterface instance."""
        return ACEStrategicContextInterface()

    @pytest.mark.asyncio
    async def test_get_strategic_context(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test getting strategic context for agent."""
        # Arrange
        agent_id = "agent-123"
        goal = "optimize performance"

        # Act
        context = await ace_interface.get_strategic_context(
            agent_id=agent_id,
            goal=goal,
        )

        # Assert
        assert context is not None
        assert context.agent_id == agent_id
        assert context.current_goal == goal
        assert context.confidence_score == 0.5  # Initial confidence

    @pytest.mark.asyncio
    async def test_store_decision_rationale(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test storing decision rationale."""
        # Arrange
        decision_id = "decision-123"
        rationale = "Chose approach A because it has lower latency"

        # Act
        memory_id = await ace_interface.store_decision_rationale(
            decision_id=decision_id,
            rationale=rationale,
        )

        # Assert
        assert memory_id is not None
        assert memory_id.startswith("decision-")

        # Verify storage
        stats = ace_interface.get_context_stats()
        assert stats["total_decisions"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_similar_decisions(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test retrieving similar past decisions."""
        # Arrange
        current_decision = "Should I use caching?"
        query_embedding = [0.5] * 1536

        # Store some decisions first
        await ace_interface.store_decision_rationale(
            decision_id="dec-1",
            rationale="Use caching for frequently accessed data",
            embedding=query_embedding,
        )

        # Act
        similar = await ace_interface.retrieve_similar_decisions(
            current_decision=current_decision,
            query_embedding=query_embedding,
            limit=5,
        )

        # Assert
        assert isinstance(similar, list)
        # Should find the stored decision

    @pytest.mark.asyncio
    async def test_build_strategic_context(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test building strategic context."""
        # Arrange
        agent_id = "agent-456"
        session_id = "session-789"
        goal = "improve accuracy"

        # Act
        context = await ace_interface.build_strategic_context(
            agent_id=agent_id,
            session_id=session_id,
            goal=goal,
        )

        # Assert
        assert context.agent_id == agent_id
        assert context.session_id == session_id
        assert context.current_goal == goal
        assert isinstance(context.strategic_memories, list)
        assert isinstance(context.tactical_memories, list)

    @pytest.mark.asyncio
    async def test_update_confidence(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test updating confidence score."""
        # Arrange
        agent_id = "agent-confidence"

        # Act - Success outcome
        new_confidence = await ace_interface.update_confidence(
            agent_id=agent_id,
            outcome_success=True,
        )

        # Assert
        assert new_confidence > 0.5  # Should increase from initial 0.5

        # Act - Failure outcome
        new_confidence2 = await ace_interface.update_confidence(
            agent_id=agent_id,
            outcome_success=False,
        )

        # Assert
        assert new_confidence2 < new_confidence  # Should decrease

    @pytest.mark.asyncio
    async def test_store_strategic_memory(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test storing strategic memory."""
        # Arrange
        agent_id = "agent-mem"
        content = "Always validate inputs before processing"
        keywords = ["validation", "best_practice"]

        # Act
        memory_id = await ace_interface.store_strategic_memory(
            agent_id=agent_id,
            content=content,
            memory_layer="semantic",
            keywords=keywords,
        )

        # Assert
        assert memory_id is not None
        assert memory_id.startswith("ace-")

    @pytest.mark.asyncio
    async def test_analyze_error_patterns(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test analyzing error patterns."""
        # Arrange
        agent_id = "agent-errors"

        # Act
        patterns = await ace_interface.analyze_error_patterns(agent_id)

        # Assert
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_analyze_success_patterns(
        self,
        ace_interface: ACEStrategicContextInterface,
    ) -> None:
        """Test analyzing success patterns."""
        # Arrange
        agent_id = "agent-success"

        # Act
        patterns = await ace_interface.analyze_success_patterns(agent_id)

        # Assert
        assert isinstance(patterns, list)


class TestCrossServiceIntegration:
    """Tests for cross-service integration workflows."""

    @pytest.mark.asyncio
    async def test_session_to_routing_workflow(self) -> None:
        """Test workflow: Session context -> Message routing."""
        # Arrange
        session_provider = SessionContextProvider()
        memory_router = MemoryAwareRouter()

        session_id = "session-workflow-1"
        await session_provider.store_session_context(
            session_id=session_id,
            context_data={"user_preferences": {"language": "Python"}},
        )

        # Act - Route message with session context
        candidates = ["python-agent", "java-agent"]
        selected = await memory_router.route_with_memory(
            message="analyze code",
            candidates=candidates,
        )

        # Assert
        assert selected in candidates

    @pytest.mark.asyncio
    async def test_routing_to_task_workflow(self) -> None:
        """Test workflow: Message routing -> Task artifact storage."""
        # Arrange
        memory_router = MemoryAwareRouter()
        artifact_storage = ArtifactMemoryStorage()

        # Route to agent
        agent_id = "agent-workflow"
        await memory_router.record_routing_outcome(
            agent_id=agent_id,
            task_type="code_analysis",
            success=True,
        )

        # Store artifact from task
        task_id = "task-workflow"
        artifact_data = {
            "name": "analysis-result",
            "type": "result",
            "content": {"issues": []},
        }

        # Act
        memory_id = await artifact_storage.store_task_artifact(
            task_id=task_id,
            artifact_data=artifact_data,
        )

        # Assert
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_ace_to_routing_workflow(self) -> None:
        """Test workflow: ACE strategic context -> Message routing."""
        # Arrange
        ace_interface = ACEStrategicContextInterface()
        memory_router = MemoryAwareRouter()

        agent_id = "agent-ace-routing"

        # Build strategic context
        context = await ace_interface.get_strategic_context(
            agent_id=agent_id,
            goal="optimize routing",
        )

        # Use context confidence for routing decision
        candidates = ["agent-1", "agent-2"]

        # Act
        selected = await memory_router.route_with_memory(
            message="process request",
            candidates=candidates,
        )

        # Assert
        assert selected in candidates
        assert context.confidence_score >= 0.0

    @pytest.mark.asyncio
    async def test_full_integration_workflow(self) -> None:
        """Test full integration: Session -> Routing -> Task -> ACE."""
        # Arrange - Initialize all integrations
        session_provider = SessionContextProvider()
        memory_router = MemoryAwareRouter()
        artifact_storage = ArtifactMemoryStorage()
        ace_interface = ACEStrategicContextInterface()

        # Step 1: Create session context
        session_id = "full-workflow-session"
        await session_provider.store_session_context(
            session_id=session_id,
            context_data={"task_type": "analysis"},
        )

        # Step 2: Route message with memory
        candidates = ["analyst-agent", "executor-agent"]
        selected_agent = await memory_router.route_with_memory(
            message="analyze data",
            candidates=candidates,
        )
        assert selected_agent is not None

        # Step 3: Store task artifact
        task_id = "full-workflow-task"
        artifact_data = {
            "name": "analysis-output",
            "type": "result",
            "content": {"findings": ["issue1", "issue2"]},
        }
        artifact_id = await artifact_storage.store_task_artifact(
            task_id=task_id,
            artifact_data=artifact_data,
        )
        assert artifact_id is not None

        # Step 4: Update ACE strategic context
        await ace_interface.update_confidence(
            agent_id=selected_agent,
            outcome_success=True,
        )

        # Step 5: Record routing outcome
        await memory_router.record_routing_outcome(
            agent_id=selected_agent,
            task_type="analysis",
            success=True,
            memory_content="Successfully analyzed data",
        )

        # Assert - Verify all components worked together
        session_stats = session_provider.get_session_memory_stats(session_id)
        assert session_stats["exists"]

        routing_stats = memory_router.get_routing_stats()
        assert routing_stats["total_routing_events"] > 0

        artifact_stats = artifact_storage.get_storage_stats()
        assert artifact_stats["total_artifacts"] > 0

        ace_stats = ace_interface.get_context_stats()
        assert ace_stats["total_contexts"] > 0
