"""Unit tests for stream event models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from agentcore.orchestration.streams.models import (
    AgentStartedEvent,
    AgentStoppedEvent,
    EventType,
    OrchestrationEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
    WorkflowCreatedEvent,
)


class TestOrchestrationEvent:
    """Tests for base OrchestrationEvent model."""

    def test_create_with_defaults(self) -> None:
        """Test creating event with default values."""
        event = OrchestrationEvent(event_type=EventType.TASK_CREATED)

        assert isinstance(event.event_id, UUID)
        assert event.event_type == EventType.TASK_CREATED
        assert isinstance(event.timestamp, datetime)
        assert event.trace_id is None
        assert event.source_agent_id is None
        assert event.workflow_id is None
        assert event.metadata == {}

    def test_create_with_custom_values(self) -> None:
        """Test creating event with custom values."""
        event_id = uuid4()
        trace_id = uuid4()
        workflow_id = uuid4()
        timestamp = datetime.now(UTC)

        event = OrchestrationEvent(
            event_id=event_id,
            event_type=EventType.AGENT_STARTED,
            timestamp=timestamp,
            trace_id=trace_id,
            source_agent_id="agent-123",
            workflow_id=workflow_id,
            metadata={"key": "value"},
        )

        assert event.event_id == event_id
        assert event.event_type == EventType.AGENT_STARTED
        assert event.timestamp == timestamp
        assert event.trace_id == trace_id
        assert event.source_agent_id == "agent-123"
        assert event.workflow_id == workflow_id
        assert event.metadata == {"key": "value"}

    def test_serialization(self) -> None:
        """Test JSON serialization."""
        event = OrchestrationEvent(event_type=EventType.TASK_COMPLETED)
        event_dict = event.model_dump(mode="json")

        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict


class TestTaskCreatedEvent:
    """Tests for TaskCreatedEvent."""

    def test_create_task_event(self) -> None:
        """Test creating task created event."""
        task_id = uuid4()

        event = TaskCreatedEvent(
            task_id=task_id,
            task_type="data_analysis",
            agent_id="agent-1",
            input_data={"query": "test"},
            timeout_seconds=600,
        )

        assert event.event_type == EventType.TASK_CREATED
        assert event.task_id == task_id
        assert event.task_type == "data_analysis"
        assert event.agent_id == "agent-1"
        assert event.input_data == {"query": "test"}
        assert event.timeout_seconds == 600

    def test_task_event_with_defaults(self) -> None:
        """Test task event with default values."""
        event = TaskCreatedEvent(task_id=uuid4(), task_type="test")

        assert event.agent_id is None
        assert event.input_data == {}
        assert event.timeout_seconds == 300


class TestTaskCompletedEvent:
    """Tests for TaskCompletedEvent."""

    def test_create_completed_event(self) -> None:
        """Test creating task completed event."""
        task_id = uuid4()

        event = TaskCompletedEvent(
            task_id=task_id,
            agent_id="agent-1",
            result_data={"status": "success"},
            execution_time_ms=1500,
        )

        assert event.event_type == EventType.TASK_COMPLETED
        assert event.task_id == task_id
        assert event.agent_id == "agent-1"
        assert event.result_data == {"status": "success"}
        assert event.execution_time_ms == 1500


class TestTaskFailedEvent:
    """Tests for TaskFailedEvent."""

    def test_create_failed_event(self) -> None:
        """Test creating task failed event."""
        task_id = uuid4()

        event = TaskFailedEvent(
            task_id=task_id,
            agent_id="agent-1",
            error_message="Task execution failed",
            error_type="RuntimeError",
            retry_count=2,
        )

        assert event.event_type == EventType.TASK_FAILED
        assert event.task_id == task_id
        assert event.agent_id == "agent-1"
        assert event.error_message == "Task execution failed"
        assert event.error_type == "RuntimeError"
        assert event.retry_count == 2


class TestAgentEvents:
    """Tests for agent lifecycle events."""

    def test_agent_started_event(self) -> None:
        """Test agent started event."""
        event = AgentStartedEvent(
            agent_id="agent-123",
            agent_type="research",
            capabilities=["web_search", "data_analysis"],
        )

        assert event.event_type == EventType.AGENT_STARTED
        assert event.agent_id == "agent-123"
        assert event.agent_type == "research"
        assert event.capabilities == ["web_search", "data_analysis"]

    def test_agent_stopped_event(self) -> None:
        """Test agent stopped event."""
        event = AgentStoppedEvent(
            agent_id="agent-123",
            reason="task_completed",
            uptime_seconds=3600,
        )

        assert event.event_type == EventType.AGENT_STOPPED
        assert event.agent_id == "agent-123"
        assert event.reason == "task_completed"
        assert event.uptime_seconds == 3600


class TestWorkflowEvents:
    """Tests for workflow lifecycle events."""

    def test_workflow_created_event(self) -> None:
        """Test workflow created event."""
        workflow_id = uuid4()

        event = WorkflowCreatedEvent(
            workflow_id=workflow_id,
            workflow_name="research_pipeline",
            workflow_version="1.0",
            orchestration_pattern="supervisor",
        )

        assert event.event_type == EventType.WORKFLOW_CREATED
        assert event.workflow_id == workflow_id
        assert event.workflow_name == "research_pipeline"
        assert event.workflow_version == "1.0"
        assert event.orchestration_pattern == "supervisor"
