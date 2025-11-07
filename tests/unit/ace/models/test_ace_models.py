"""
Unit tests for ACE Pydantic models.

Tests model validation, field constraints, and business logic.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentcore.ace.models import (
    ApplyDeltaRequest,
    ApplyDeltaResponse,
    CaptureTraceRequest,
    CaptureTraceResponse,
    ContextDelta,
    ContextPlaybook,
    CreatePlaybookRequest,
    CreatePlaybookResponse,
    EvolutionStatus,
    EvolutionStatusType,
    ExecutionTrace,
    TriggerEvolutionRequest,
    TriggerEvolutionResponse,
)


class TestContextPlaybook:
    """Tests for ContextPlaybook model."""

    def test_create_playbook_valid(self) -> None:
        """Test creating a valid playbook."""
        playbook = ContextPlaybook(
            agent_id="agent-001",
            context={"goal": "test"},
            version=1,
        )
        assert playbook.agent_id == "agent-001"
        assert playbook.context == {"goal": "test"}
        assert playbook.version == 1
        assert isinstance(playbook.playbook_id, UUID)
        assert isinstance(playbook.created_at, datetime)

    def test_playbook_empty_context_fails(self) -> None:
        """Test that empty context is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ContextPlaybook(
                agent_id="agent-001",
                context={},
            )
        assert "Context cannot be empty" in str(exc_info.value)

    def test_playbook_default_values(self) -> None:
        """Test default values are set correctly."""
        playbook = ContextPlaybook(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        assert playbook.version == 1
        assert playbook.metadata == {}
        assert playbook.created_at is not None
        assert playbook.updated_at is not None

    def test_playbook_agent_id_validation(self) -> None:
        """Test agent_id validation."""
        with pytest.raises(ValidationError):
            ContextPlaybook(
                agent_id="",  # Empty string should fail
                context={"goal": "test"},
            )

    def test_playbook_version_validation(self) -> None:
        """Test version must be >= 1."""
        with pytest.raises(ValidationError):
            ContextPlaybook(
                agent_id="agent-001",
                context={"goal": "test"},
                version=0,  # Should fail
            )


class TestContextDelta:
    """Tests for ContextDelta model."""

    def test_create_delta_valid(self) -> None:
        """Test creating a valid delta."""
        playbook_id = uuid4()
        delta = ContextDelta(
            playbook_id=playbook_id,
            changes={"temperature": 0.8},
            confidence=0.85,
            reasoning="Based on traces, higher temperature improves output quality.",
        )
        assert delta.playbook_id == playbook_id
        assert delta.changes == {"temperature": 0.8}
        assert delta.confidence == 0.85
        assert delta.applied is False
        assert delta.applied_at is None

    def test_delta_empty_changes_fails(self) -> None:
        """Test that empty changes are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ContextDelta(
                playbook_id=uuid4(),
                changes={},  # Empty
                confidence=0.85,
                reasoning="Valid reasoning here",
            )
        assert "Changes cannot be empty" in str(exc_info.value)

    def test_delta_confidence_validation(self) -> None:
        """Test confidence must be between 0 and 1."""
        playbook_id = uuid4()

        # Test confidence too low
        with pytest.raises(ValidationError):
            ContextDelta(
                playbook_id=playbook_id,
                changes={"temp": 0.8},
                confidence=-0.1,
                reasoning="Valid reasoning",
            )

        # Test confidence too high
        with pytest.raises(ValidationError):
            ContextDelta(
                playbook_id=playbook_id,
                changes={"temp": 0.8},
                confidence=1.5,
                reasoning="Valid reasoning",
            )

        # Test valid boundaries
        delta_min = ContextDelta(
            playbook_id=playbook_id,
            changes={"temp": 0.8},
            confidence=0.0,
            reasoning="Valid reasoning",
        )
        assert delta_min.confidence == 0.0

        delta_max = ContextDelta(
            playbook_id=playbook_id,
            changes={"temp": 0.8},
            confidence=1.0,
            reasoning="Valid reasoning",
        )
        assert delta_max.confidence == 1.0

    def test_delta_reasoning_validation(self) -> None:
        """Test reasoning validation."""
        playbook_id = uuid4()

        # Too short reasoning
        with pytest.raises(ValidationError) as exc_info:
            ContextDelta(
                playbook_id=playbook_id,
                changes={"temp": 0.8},
                confidence=0.85,
                reasoning="Short",
            )
        assert "at least 10 characters" in str(exc_info.value)

        # Valid reasoning with whitespace
        delta = ContextDelta(
            playbook_id=playbook_id,
            changes={"temp": 0.8},
            confidence=0.85,
            reasoning="  Valid reasoning with whitespace  ",
        )
        assert delta.reasoning == "Valid reasoning with whitespace"


class TestExecutionTrace:
    """Tests for ExecutionTrace model."""

    def test_create_trace_success(self) -> None:
        """Test creating a successful execution trace."""
        trace = ExecutionTrace(
            agent_id="agent-001",
            task_id="task-123",
            execution_time=2.5,
            success=True,
            output_quality=0.92,
        )
        assert trace.agent_id == "agent-001"
        assert trace.task_id == "task-123"
        assert trace.execution_time == 2.5
        assert trace.success is True
        assert trace.output_quality == 0.92
        assert trace.error_message is None

    def test_create_trace_failure(self) -> None:
        """Test creating a failed execution trace."""
        trace = ExecutionTrace(
            agent_id="agent-001",
            execution_time=1.0,
            success=False,
            error_message="Task failed due to timeout",
        )
        assert trace.success is False
        assert trace.error_message == "Task failed due to timeout"

    def test_trace_failed_requires_error_message(self) -> None:
        """Test that failed traces require error message."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionTrace(
                agent_id="agent-001",
                execution_time=1.0,
                success=False,
                # Missing error_message
            )
        assert "Error message required for failed executions" in str(exc_info.value)

    def test_trace_execution_time_validation(self) -> None:
        """Test execution_time must be >= 0."""
        with pytest.raises(ValidationError):
            ExecutionTrace(
                agent_id="agent-001",
                execution_time=-1.0,  # Negative
                success=True,
            )

    def test_trace_output_quality_validation(self) -> None:
        """Test output_quality must be between 0 and 1."""
        # Too low
        with pytest.raises(ValidationError):
            ExecutionTrace(
                agent_id="agent-001",
                execution_time=1.0,
                success=True,
                output_quality=-0.1,
            )

        # Too high
        with pytest.raises(ValidationError):
            ExecutionTrace(
                agent_id="agent-001",
                execution_time=1.0,
                success=True,
                output_quality=1.5,
            )

        # Valid boundaries
        trace_min = ExecutionTrace(
            agent_id="agent-001",
            execution_time=1.0,
            success=True,
            output_quality=0.0,
        )
        assert trace_min.output_quality == 0.0

        trace_max = ExecutionTrace(
            agent_id="agent-001",
            execution_time=1.0,
            success=True,
            output_quality=1.0,
        )
        assert trace_max.output_quality == 1.0

    def test_trace_default_metadata(self) -> None:
        """Test default metadata is empty dict."""
        trace = ExecutionTrace(
            agent_id="agent-001",
            execution_time=1.0,
            success=True,
        )
        assert trace.metadata == {}


class TestEvolutionStatus:
    """Tests for EvolutionStatus model."""

    def test_create_evolution_status(self) -> None:
        """Test creating evolution status."""
        status = EvolutionStatus(
            agent_id="agent-001",
            pending_traces=15,
            deltas_generated=10,
            deltas_applied=7,
            total_cost=0.25,
        )
        assert status.agent_id == "agent-001"
        assert status.pending_traces == 15
        assert status.deltas_generated == 10
        assert status.deltas_applied == 7
        assert status.total_cost == 0.25
        assert status.status == EvolutionStatusType.IDLE

    def test_evolution_status_defaults(self) -> None:
        """Test default values."""
        status = EvolutionStatus(agent_id="agent-001")
        assert status.last_evolution is None
        assert status.pending_traces == 0
        assert status.deltas_generated == 0
        assert status.deltas_applied == 0
        assert status.total_cost == 0.0
        assert status.status == EvolutionStatusType.IDLE

    def test_evolution_status_deltas_validation(self) -> None:
        """Test applied deltas cannot exceed generated."""
        with pytest.raises(ValidationError) as exc_info:
            EvolutionStatus(
                agent_id="agent-001",
                deltas_generated=5,
                deltas_applied=10,  # More than generated
            )
        assert "Applied deltas cannot exceed generated deltas" in str(exc_info.value)

    def test_evolution_status_negative_values_fail(self) -> None:
        """Test negative values are rejected."""
        with pytest.raises(ValidationError):
            EvolutionStatus(
                agent_id="agent-001",
                pending_traces=-1,
            )

        with pytest.raises(ValidationError):
            EvolutionStatus(
                agent_id="agent-001",
                total_cost=-0.5,
            )

    def test_evolution_status_type_enum(self) -> None:
        """Test status type enum values."""
        status = EvolutionStatus(
            agent_id="agent-001",
            status=EvolutionStatusType.PROCESSING,
        )
        assert status.status == EvolutionStatusType.PROCESSING
        assert status.status.value == "processing"


class TestRequestResponseModels:
    """Tests for API request/response models."""

    def test_create_playbook_request(self) -> None:
        """Test CreatePlaybookRequest validation."""
        request = CreatePlaybookRequest(
            agent_id="agent-001",
            initial_context={"goal": "test"},
            metadata={"source": "api"},
        )
        assert request.agent_id == "agent-001"
        assert request.initial_context == {"goal": "test"}
        assert request.metadata == {"source": "api"}

    def test_create_playbook_response(self) -> None:
        """Test CreatePlaybookResponse structure."""
        playbook = ContextPlaybook(agent_id="agent-001", context={"goal": "test"})
        response = CreatePlaybookResponse(playbook=playbook)
        assert response.playbook == playbook
        assert response.message == "Playbook created successfully"

    def test_apply_delta_request(self) -> None:
        """Test ApplyDeltaRequest validation."""
        delta_id = uuid4()
        request = ApplyDeltaRequest(delta_id=delta_id, force=True)
        assert request.delta_id == delta_id
        assert request.force is True

    def test_capture_trace_request(self) -> None:
        """Test CaptureTraceRequest validation."""
        request = CaptureTraceRequest(
            agent_id="agent-001",
            task_id="task-123",
            execution_time=2.5,
            success=True,
            output_quality=0.9,
        )
        assert request.agent_id == "agent-001"
        assert request.execution_time == 2.5
        assert request.success is True

    def test_trigger_evolution_request(self) -> None:
        """Test TriggerEvolutionRequest validation."""
        request = TriggerEvolutionRequest(agent_id="agent-001", force=False)
        assert request.agent_id == "agent-001"
        assert request.force is False

    def test_trigger_evolution_response(self) -> None:
        """Test TriggerEvolutionResponse structure."""
        status = EvolutionStatus(agent_id="agent-001")
        response = TriggerEvolutionResponse(
            status=status,
            deltas_generated=3,
            message="Evolution triggered",
        )
        assert response.status == status
        assert response.deltas_generated == 3
        assert response.message == "Evolution triggered"
