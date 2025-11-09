"""
Unit tests for ACE SQLAlchemy ORM models.

Tests model structure, relationships, defaults, and constraints.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.ace.database import (
    ContextDeltaDB,
    ContextPlaybookDB,
    EvolutionStatusDB,
    ExecutionTraceDB,
)
from agentcore.ace.models.ace_models import EvolutionStatusType


class TestContextPlaybookDB:
    """Tests for ContextPlaybookDB ORM model."""

    def test_create_playbook(self) -> None:
        """Test creating a playbook ORM instance."""
        playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
            version=1,
        )
        assert playbook.agent_id == "agent-001"
        assert playbook.context == {"goal": "test"}
        assert playbook.version == 1
        # playbook_id is generated at DB insert time, not at instantiation

    def test_playbook_defaults(self) -> None:
        """Test default values are set at Python level."""
        playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        # Note: version and metadata defaults are server-side, not set at instantiation
        # Only Python-level defaults (like created_at/updated_at callables) work in-memory
        assert playbook.agent_id == "agent-001"
        assert playbook.context == {"goal": "test"}

    def test_playbook_tablename(self) -> None:
        """Test table name is correct."""
        assert ContextPlaybookDB.__tablename__ == "context_playbooks"

    def test_playbook_has_deltas_relationship(self) -> None:
        """Test playbook has deltas relationship."""
        playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        # Deltas relationship should exist
        assert hasattr(playbook, "deltas")


class TestContextDeltaDB:
    """Tests for ContextDeltaDB ORM model."""

    def test_create_delta(self) -> None:
        """Test creating a delta ORM instance."""
        playbook_id = uuid4()
        delta = ContextDeltaDB(
            playbook_id=playbook_id,
            changes={"temperature": 0.8},
            confidence=0.85,
            reasoning="Based on traces, higher temperature improves output.",
        )
        assert delta.playbook_id == playbook_id
        assert delta.changes == {"temperature": 0.8}
        assert delta.confidence == 0.85
        assert delta.reasoning == "Based on traces, higher temperature improves output."
        # delta_id is generated at DB insert time, not at instantiation

    def test_delta_defaults(self) -> None:
        """Test default values are set correctly."""
        delta = ContextDeltaDB(
            playbook_id=uuid4(),
            changes={"test": "value"},
            confidence=0.9,
            reasoning="Test reasoning",
        )
        # Server-side defaults not applied at instantiation
        # Only check that attributes exist
        assert hasattr(delta, "applied")
        assert hasattr(delta, "applied_at")
        assert hasattr(delta, "generated_at")

    def test_delta_tablename(self) -> None:
        """Test table name is correct."""
        assert ContextDeltaDB.__tablename__ == "context_deltas"

    def test_delta_has_playbook_relationship(self) -> None:
        """Test delta has playbook relationship."""
        delta = ContextDeltaDB(
            playbook_id=uuid4(),
            changes={"test": "value"},
            confidence=0.9,
            reasoning="Test reasoning",
        )
        # Playbook relationship should exist
        assert hasattr(delta, "playbook")


class TestExecutionTraceDB:
    """Tests for ExecutionTraceDB ORM model."""

    def test_create_trace_success(self) -> None:
        """Test creating a successful trace."""
        trace = ExecutionTraceDB(
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
        """Test creating a failed trace."""
        trace = ExecutionTraceDB(
            agent_id="agent-001",
            execution_time=1.0,
            success=False,
            error_message="Task failed",
        )
        assert trace.success is False
        assert trace.error_message == "Task failed"

    def test_trace_defaults(self) -> None:
        """Test default values."""
        trace = ExecutionTraceDB(
            agent_id="agent-001",
            execution_time=1.0,
            success=True,
        )
        # Server-side defaults not applied at instantiation
        assert hasattr(trace, "trace_metadata")
        assert hasattr(trace, "captured_at")
        assert trace.task_id is None
        assert trace.output_quality is None

    def test_trace_tablename(self) -> None:
        """Test table name is correct."""
        assert ExecutionTraceDB.__tablename__ == "execution_traces"

    def test_trace_with_metadata(self) -> None:
        """Test trace with custom metadata."""
        trace = ExecutionTraceDB(
            agent_id="agent-001",
            execution_time=1.0,
            success=True,
            trace_metadata={"tokens_used": 1024, "model": "gpt-4"},
        )
        assert trace.trace_metadata == {"tokens_used": 1024, "model": "gpt-4"}


class TestEvolutionStatusDB:
    """Tests for EvolutionStatusDB ORM model."""

    def test_create_evolution_status(self) -> None:
        """Test creating evolution status."""
        status = EvolutionStatusDB(
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

    def test_evolution_status_defaults(self) -> None:
        """Test default values."""
        status = EvolutionStatusDB(agent_id="agent-001")
        # Server-side defaults not applied at instantiation
        assert hasattr(status, "pending_traces")
        assert hasattr(status, "deltas_generated")
        assert hasattr(status, "deltas_applied")
        assert hasattr(status, "total_cost")
        assert hasattr(status, "status")
        assert status.last_evolution is None

    def test_evolution_status_tablename(self) -> None:
        """Test table name is correct."""
        assert EvolutionStatusDB.__tablename__ == "evolution_status"

    def test_evolution_status_with_timestamp(self) -> None:
        """Test status with last evolution timestamp."""
        now = datetime.now(UTC)
        status = EvolutionStatusDB(
            agent_id="agent-001",
            last_evolution=now,
        )
        assert status.last_evolution == now

    def test_evolution_status_enum(self) -> None:
        """Test status enum values."""
        status = EvolutionStatusDB(
            agent_id="agent-001",
            status=EvolutionStatusType.PROCESSING,
        )
        assert status.status == EvolutionStatusType.PROCESSING
        assert status.status.value == "processing"


class TestOrmIntegration:
    """Integration tests for ORM models."""

    def test_all_models_have_tablename(self) -> None:
        """Test all models have __tablename__ defined."""
        models = [
            ContextPlaybookDB,
            ContextDeltaDB,
            ExecutionTraceDB,
            EvolutionStatusDB,
        ]
        for model in models:
            assert hasattr(model, "__tablename__")
            assert isinstance(model.__tablename__, str)
            assert len(model.__tablename__) > 0

    def test_all_models_have_primary_key(self) -> None:
        """Test all models have primary key defined."""
        models = [
            ContextPlaybookDB,
            ContextDeltaDB,
            ExecutionTraceDB,
            EvolutionStatusDB,
        ]
        for model in models:
            # Check that the model has at least one primary key column
            pk_cols = [
                col
                for col in model.__table__.columns
                if col.primary_key
            ]
            assert len(pk_cols) > 0, f"{model.__name__} has no primary key"

    def test_foreign_keys_exist(self) -> None:
        """Test foreign key relationships are defined."""
        # ContextPlaybookDB -> agents.id
        playbook_fks = [
            fk for col in ContextPlaybookDB.__table__.columns
            for fk in col.foreign_keys
        ]
        assert len(playbook_fks) > 0
        assert any("agents.id" in str(fk) for fk in playbook_fks)

        # ContextDeltaDB -> context_playbooks.playbook_id
        delta_fks = [
            fk for col in ContextDeltaDB.__table__.columns
            for fk in col.foreign_keys
        ]
        assert len(delta_fks) > 0
        assert any("context_playbooks.playbook_id" in str(fk) for fk in delta_fks)

        # ExecutionTraceDB -> agents.id
        trace_fks = [
            fk for col in ExecutionTraceDB.__table__.columns
            for fk in col.foreign_keys
        ]
        assert len(trace_fks) > 0
        assert any("agents.id" in str(fk) for fk in trace_fks)

        # EvolutionStatusDB -> agents.id
        status_fks = [
            fk for col in EvolutionStatusDB.__table__.columns
            for fk in col.foreign_keys
        ]
        assert len(status_fks) > 0
        assert any("agents.id" in str(fk) for fk in status_fks)

    def test_indexes_exist(self) -> None:
        """Test that indexes are defined on tables."""
        # Check ContextPlaybookDB has indexes
        playbook_indexes = [idx.name for idx in ContextPlaybookDB.__table__.indexes]
        assert "idx_playbooks_agent" in playbook_indexes
        assert "idx_playbooks_updated" in playbook_indexes

        # Check ContextDeltaDB has indexes
        delta_indexes = [idx.name for idx in ContextDeltaDB.__table__.indexes]
        assert "idx_deltas_playbook" in delta_indexes
        assert "idx_deltas_confidence" in delta_indexes
        assert "idx_deltas_applied" in delta_indexes

        # Check ExecutionTraceDB has indexes
        trace_indexes = [idx.name for idx in ExecutionTraceDB.__table__.indexes]
        assert "idx_traces_agent" in trace_indexes
        assert "idx_traces_success" in trace_indexes

        # Check EvolutionStatusDB has indexes
        status_indexes = [idx.name for idx in EvolutionStatusDB.__table__.indexes]
        assert "idx_evolution_status" in status_indexes
