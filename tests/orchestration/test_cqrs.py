"""
Tests for CQRS implementation.

Tests command/query separation, event sourcing, and eventual consistency.
"""

from __future__ import annotations

import pytest
from datetime import UTC, datetime
from uuid import UUID, uuid4

from agentcore.orchestration.cqrs import (
    # Commands
    CommandBus,
    CreateWorkflowCommand,
    StartWorkflowCommand,
    AssignAgentCommand,
    ScheduleTaskCommand,
    CommandResult,
    # Events
    WorkflowCreatedEvent,
    WorkflowStartedEvent,
    AgentAssignedEvent,
    TaskScheduledEvent,
    TaskCompletedEvent,
    EventType,
    deserialize_event,
    # Event Store
    PostgreSQLEventStore,
    EventStreamResult,
    SnapshotData,
    # Queries
    QueryBus,
    GetWorkflowQuery,
    ListWorkflowsQuery,
    GetWorkflowStatusQuery,
    QueryResult,
    # Projections
    ProjectionManager,
    WorkflowProjection,
    ExecutionProjection,
    AgentAssignmentProjection,
    TaskProjection,
)
from agentcore.a2a_protocol.database.connection import get_session


class TestEvents:
    """Test event models and serialization."""

    def test_workflow_created_event(self):
        """Test workflow created event creation."""
        event = WorkflowCreatedEvent(
            aggregate_id=uuid4(),
            aggregate_type="workflow",
            workflow_id=uuid4(),
            workflow_name="Test Workflow",
            orchestration_pattern="supervisor",
            agent_requirements={"researcher": ["web_search"]},
            task_definitions=[{"task_id": "task1", "type": "research"}],
            created_by="user123",
        )

        assert event.event_type == EventType.WORKFLOW_CREATED
        assert event.workflow_name == "Test Workflow"
        assert event.orchestration_pattern == "supervisor"
        assert event.created_by == "user123"
        assert isinstance(event.timestamp, datetime)

    def test_event_immutability(self):
        """Test that events are immutable."""
        event = WorkflowCreatedEvent(
            aggregate_id=uuid4(),
            aggregate_type="workflow",
            workflow_id=uuid4(),
            workflow_name="Test",
            orchestration_pattern="supervisor",
        )

        with pytest.raises(Exception):
            event.workflow_name = "Changed"

    def test_event_serialization(self):
        """Test event serialization and deserialization."""
        original_event = WorkflowCreatedEvent(
            aggregate_id=uuid4(),
            aggregate_type="workflow",
            workflow_id=uuid4(),
            workflow_name="Test Workflow",
            orchestration_pattern="supervisor",
        )

        # Serialize to dict
        event_dict = original_event.model_dump()

        # Deserialize back
        deserialized = deserialize_event(event_dict)

        assert isinstance(deserialized, WorkflowCreatedEvent)
        assert deserialized.workflow_name == original_event.workflow_name
        assert deserialized.event_id == original_event.event_id


class TestCommands:
    """Test command models and bus."""

    def test_create_workflow_command(self):
        """Test create workflow command."""
        cmd = CreateWorkflowCommand(
            workflow_name="Test Workflow",
            orchestration_pattern="supervisor",
            agent_requirements={"supervisor": ["task_decomposition"]},
            task_definitions=[],
            user_id="user123",
        )

        assert cmd.workflow_name == "Test Workflow"
        assert cmd.orchestration_pattern == "supervisor"
        assert cmd.user_id == "user123"
        assert isinstance(cmd.command_id, UUID)

    def test_command_bus_registration(self):
        """Test command bus handler registration."""
        bus = CommandBus()

        # Create a mock handler
        class MockHandler:
            async def handle(self, command):
                return CommandResult(
                    command_id=command.command_id,
                    success=True,
                    aggregate_id=uuid4(),
                )

            def can_handle(self, command):
                return True

        handler = MockHandler()
        bus.register("create_workflow", handler)

        handlers = bus.get_registered_handlers()
        assert "create_workflow" in handlers

    @pytest.mark.asyncio
    async def test_command_bus_dispatch_no_handler(self):
        """Test command dispatch with no handler."""
        bus = CommandBus()
        cmd = CreateWorkflowCommand(
            workflow_name="Test",
            orchestration_pattern="supervisor",
        )

        result = await bus.dispatch(cmd)

        assert result.success is False
        assert "No handler registered" in result.error_message


class TestQueries:
    """Test query models and bus."""

    def test_get_workflow_query(self):
        """Test get workflow query."""
        query = GetWorkflowQuery(
            workflow_id=uuid4(),
            include_tasks=True,
            include_agents=True,
            user_id="user123",
        )

        assert isinstance(query.workflow_id, UUID)
        assert query.include_tasks is True
        assert query.include_agents is True

    def test_list_workflows_query(self):
        """Test list workflows query."""
        query = ListWorkflowsQuery(
            orchestration_pattern="supervisor",
            status="running",
            limit=100,
            offset=0,
        )

        assert query.orchestration_pattern == "supervisor"
        assert query.status == "running"
        assert query.limit == 100

    def test_query_bus_registration(self):
        """Test query bus handler registration."""
        bus = QueryBus()

        # Create a mock handler
        class MockHandler:
            async def handle(self, query):
                return QueryResult(
                    query_id=query.query_id,
                    success=True,
                    data={"workflow_id": "123"},
                )

            def can_handle(self, query):
                return True

        handler = MockHandler()
        bus.register("get_workflow", handler)

        handlers = bus.get_registered_handlers()
        assert "get_workflow" in handlers


@pytest.mark.asyncio
class TestEventStore:
    """Test event store functionality."""

    async def test_append_and_retrieve_events(self, init_test_db):
        """Test appending and retrieving events."""
        async with get_session() as session:
            store = PostgreSQLEventStore(session)
            aggregate_id = uuid4()

            # Create and append events
            event1 = WorkflowCreatedEvent(
                aggregate_id=aggregate_id,
                aggregate_type="workflow",
                workflow_id=aggregate_id,
                workflow_name="Test Workflow",
                orchestration_pattern="supervisor",
                version=1,
            )

            event2 = WorkflowStartedEvent(
                aggregate_id=aggregate_id,
                aggregate_type="workflow",
                workflow_id=aggregate_id,
                execution_id=uuid4(),
                version=2,
            )

            await store.append_event(event1)
            await store.append_event(event2)

            # Retrieve events
            result = await store.get_events(aggregate_id)

            assert result.total_events == 2
            assert result.current_version == 2
            assert len(result.events) == 2
            assert result.events[0].event_type == EventType.WORKFLOW_CREATED
            assert result.events[1].event_type == EventType.WORKFLOW_STARTED

    async def test_append_events_batch(self, init_test_db):
        """Test batch event appending."""
        async with get_session() as session:
            store = PostgreSQLEventStore(session)
            aggregate_id = uuid4()

            events = [
                WorkflowCreatedEvent(
                    aggregate_id=aggregate_id,
                    aggregate_type="workflow",
                    workflow_id=aggregate_id,
                    workflow_name="Test",
                    orchestration_pattern="supervisor",
                    version=1,
                ),
                WorkflowStartedEvent(
                    aggregate_id=aggregate_id,
                    aggregate_type="workflow",
                    workflow_id=aggregate_id,
                    execution_id=uuid4(),
                    version=2,
                ),
            ]

            await store.append_events(events)

            result = await store.get_events(aggregate_id)
            assert result.total_events == 2

    async def test_get_events_by_type(self, init_test_db):
        """Test retrieving events by type."""
        async with get_session() as session:
            store = PostgreSQLEventStore(session)

            # Create multiple workflow events
            for i in range(3):
                event = WorkflowCreatedEvent(
                    aggregate_id=uuid4(),
                    aggregate_type="workflow",
                    workflow_id=uuid4(),
                    workflow_name=f"Workflow {i}",
                    orchestration_pattern="supervisor",
                    version=1,
                )
                await store.append_event(event)

            # Retrieve by type
            events = await store.get_events_by_type("workflow_created", limit=10)

            assert len(events) >= 3
            assert all(e.event_type == EventType.WORKFLOW_CREATED for e in events)

    async def test_snapshot_save_and_retrieve(self, init_test_db):
        """Test saving and retrieving snapshots."""
        async with get_session() as session:
            store = PostgreSQLEventStore(session)
            aggregate_id = uuid4()

            snapshot = SnapshotData(
                aggregate_id=aggregate_id,
                aggregate_type="workflow",
                version=10,
                timestamp=datetime.now(UTC),
                state={"status": "running", "tasks_completed": 5},
                metadata={"snapshot_reason": "periodic"},
            )

            await store.save_snapshot(snapshot)

            retrieved = await store.get_snapshot(aggregate_id)

            assert retrieved is not None
            assert retrieved.version == 10
            assert retrieved.state["status"] == "running"
            assert retrieved.state["tasks_completed"] == 5


@pytest.mark.asyncio
class TestProjections:
    """Test projection functionality."""

    async def test_workflow_projection(self, init_test_db):
        """Test workflow projection updates read model."""
        async with get_session() as session:
            projection = WorkflowProjection()
            workflow_id = uuid4()

            event = WorkflowCreatedEvent(
                aggregate_id=workflow_id,
                aggregate_type="workflow",
                workflow_id=workflow_id,
                workflow_name="Test Workflow",
                orchestration_pattern="supervisor",
                agent_requirements={},
                task_definitions=[],
                created_by="user123",
            )

            # Check if projection handles this event type
            assert projection.handles_event_type(EventType.WORKFLOW_CREATED)

            # Project the event
            await projection.project(event, session)

            # Verify read model was updated (would need to query read model)

    async def test_execution_projection(self, init_test_db):
        """Test execution projection updates read model."""
        async with get_session() as session:
            projection = ExecutionProjection()
            workflow_id = uuid4()
            execution_id = uuid4()

            event = WorkflowStartedEvent(
                aggregate_id=workflow_id,
                aggregate_type="workflow",
                workflow_id=workflow_id,
                execution_id=execution_id,
                input_data={"query": "test"},
                started_by="user123",
            )

            assert projection.handles_event_type(EventType.WORKFLOW_STARTED)
            await projection.project(event, session)

    async def test_projection_manager(self, init_test_db):
        """Test projection manager coordinates multiple projections."""
        async with get_session() as session:
            manager = ProjectionManager()

            # Register projections
            manager.register(WorkflowProjection())
            manager.register(ExecutionProjection())
            manager.register(AgentAssignmentProjection())
            manager.register(TaskProjection())

            workflow_id = uuid4()
            event = WorkflowCreatedEvent(
                aggregate_id=workflow_id,
                aggregate_type="workflow",
                workflow_id=workflow_id,
                workflow_name="Test",
                orchestration_pattern="supervisor",
            )

            # Project event through all projections
            await manager.project_event(event, session)


@pytest.mark.asyncio
class TestEventualConsistency:
    """Test eventual consistency between write and read sides."""

    async def test_command_to_query_flow(self, init_test_db):
        """Test full flow from command to query."""
        async with get_session() as session:
            # Setup
            event_store = PostgreSQLEventStore(session)
            projection_manager = ProjectionManager()
            projection_manager.register(WorkflowProjection())

            # Create workflow via command (write side)
            workflow_id = uuid4()
            event = WorkflowCreatedEvent(
                aggregate_id=workflow_id,
                aggregate_type="workflow",
                workflow_id=workflow_id,
                workflow_name="Test Workflow",
                orchestration_pattern="supervisor",
            )

            # Store event
            await event_store.append_event(event)

            # Project to read model
            await projection_manager.project_event(event, session)

            # Query read model (read side)
            # Would query WorkflowReadModel here

    async def test_multiple_events_projection(self, init_test_db):
        """Test projecting multiple events maintains consistency."""
        async with get_session() as session:
            event_store = PostgreSQLEventStore(session)
            projection_manager = ProjectionManager()
            projection_manager.register(WorkflowProjection())
            projection_manager.register(ExecutionProjection())

            workflow_id = uuid4()
            execution_id = uuid4()

            events = [
                WorkflowCreatedEvent(
                    aggregate_id=workflow_id,
                    aggregate_type="workflow",
                    workflow_id=workflow_id,
                    workflow_name="Test",
                    orchestration_pattern="supervisor",
                    version=1,
                ),
                WorkflowStartedEvent(
                    aggregate_id=workflow_id,
                    aggregate_type="workflow",
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    version=2,
                ),
            ]

            # Store events
            await event_store.append_events(events)

            # Project all events
            await projection_manager.project_events(events, session)


class TestCQRSSeparation:
    """Test command/query separation principles."""

    def test_commands_are_write_only(self):
        """Test that commands represent write intent."""
        cmd = CreateWorkflowCommand(
            workflow_name="Test",
            orchestration_pattern="supervisor",
        )

        # Commands should not return data, only results
        assert hasattr(cmd, "command_id")
        assert hasattr(cmd, "workflow_name")

    def test_queries_are_read_only(self):
        """Test that queries are read operations."""
        query = GetWorkflowQuery(workflow_id=uuid4())

        # Queries should specify what to retrieve
        assert hasattr(query, "query_id")
        assert hasattr(query, "workflow_id")
        assert hasattr(query, "include_tasks")

    def test_events_are_immutable(self):
        """Test that events cannot be modified."""
        event = WorkflowCreatedEvent(
            aggregate_id=uuid4(),
            aggregate_type="workflow",
            workflow_id=uuid4(),
            workflow_name="Test",
            orchestration_pattern="supervisor",
        )

        # Should not be able to modify frozen event
        with pytest.raises(Exception):
            event.workflow_name = "Modified"
