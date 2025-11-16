"""
Unit tests for Memory System ORM Models

Tests SQLAlchemy model structure and Pydantic conversion.

Note: These tests focus on model attributes and Pydantic conversion.
PostgreSQL-specific features (ARRAY, UUID, JSONB, persistence) will be tested in integration tests.

Component ID: MEM-006
Ticket: MEM-006 (Implement SQLAlchemy ORM Models)
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.database.memory_models import (
    CompressionMetricsModel,
    ErrorModel,
    MemoryModel,
    StageMemoryModel,
    TaskContextModel,
)
from agentcore.a2a_protocol.models.memory import (
    ErrorType,
    MemoryLayer,
    StageType,
)


class TestMemoryModel:
    """Test MemoryModel structure and Pydantic conversion."""

    def test_create_memory_model_instance(self) -> None:
        """Test creating a MemoryModel instance (structure validation)."""
        memory_id = uuid4()
        agent_id = uuid4()
        task_id = uuid4()

        memory = MemoryModel(
            memory_id=memory_id,
            memory_layer=MemoryLayer.EPISODIC.value,
            content="User asked about authentication",
            summary="Question about auth",
            agent_id=agent_id,
            task_id=task_id,
            timestamp=datetime.now(UTC),
            entities=["user", "authentication"],
            facts=["question asked"],
            keywords=["auth"],
            is_critical=True,
            criticality_reason="Important security question",
        )

        # Verify attributes
        assert memory.memory_id == memory_id
        assert memory.memory_layer == MemoryLayer.EPISODIC.value
        assert memory.content == "User asked about authentication"
        assert memory.agent_id == agent_id
        assert memory.is_critical is True
        assert "authentication" in memory.entities

    def test_memory_to_pydantic_conversion(self) -> None:
        """Test converting MemoryModel to Pydantic MemoryRecord."""
        memory_id = uuid4()
        agent_id = uuid4()

        memory = MemoryModel(
            memory_id=memory_id,
            memory_layer=MemoryLayer.SEMANTIC.value,
            content="Test content",
            summary="Test summary",
            agent_id=agent_id,
            timestamp=datetime.now(UTC),
            relevance_score=0.95,
            access_count=5,
            entities=[],
            facts=[],
            keywords=[],
            related_memory_ids=[],
            actions=[],
        )

        # Convert to Pydantic
        pydantic_memory = memory.to_pydantic()

        assert pydantic_memory.memory_id == str(memory_id)
        assert pydantic_memory.memory_layer == MemoryLayer.SEMANTIC
        assert pydantic_memory.content == "Test content"
        assert pydantic_memory.summary == "Test summary"
        assert pydantic_memory.agent_id == str(agent_id)
        assert pydantic_memory.relevance_score == 0.95
        assert pydantic_memory.access_count == 5

    def test_memory_with_procedural_fields(self) -> None:
        """Test MemoryModel with procedural memory fields."""
        memory_id = uuid4()

        memory = MemoryModel(
            memory_id=memory_id,
            memory_layer=MemoryLayer.PROCEDURAL.value,
            content="Executed workflow",
            summary="Workflow execution",
            agent_id=uuid4(),
            timestamp=datetime.now(UTC),
            actions=["step1", "step2", "step3"],
            outcome="Successfully completed",
            success=True,
            entities=[],
            facts=[],
            keywords=[],
            related_memory_ids=[],
        )

        assert memory.actions == ["step1", "step2", "step3"]
        assert memory.outcome == "Successfully completed"
        assert memory.success is True

    def test_memory_repr(self) -> None:
        """Test __repr__ method."""
        memory_id = uuid4()
        agent_id = uuid4()

        memory = MemoryModel(
            memory_id=memory_id,
            memory_layer=MemoryLayer.EPISODIC.value,
            content="Test",
            summary="Test",
            agent_id=agent_id,
            timestamp=datetime.now(UTC),
            is_critical=True,
        )

        repr_str = repr(memory)
        assert "MemoryModel" in repr_str
        assert str(memory_id) in repr_str
        assert str(agent_id) in repr_str
        assert "is_critical=True" in repr_str


class TestStageMemoryModel:
    """Test StageMemoryModel structure and Pydantic conversion."""

    def test_create_stage_memory(self) -> None:
        """Test creating a StageMemoryModel instance."""
        stage_id = uuid4()
        task_id = uuid4()
        agent_id = uuid4()
        mem1_id = uuid4()
        mem2_id = uuid4()

        stage_memory = StageMemoryModel(
            stage_id=stage_id,
            task_id=task_id,
            agent_id=agent_id,
            stage_type=StageType.PLANNING.value,
            stage_summary="Analyzed requirements and created plan",
            stage_insights=["Use JWT", "Store in Redis"],
            raw_memory_refs=[mem1_id, mem2_id],
            compression_ratio=10.5,
            compression_model="gpt-4.1-mini",
            quality_metrics={"quality_score": 0.95, "fact_retention": 0.97},
        )

        # Verify attributes
        assert stage_memory.stage_id == stage_id
        assert stage_memory.stage_type == StageType.PLANNING.value
        assert stage_memory.compression_ratio == 10.5
        assert len(stage_memory.raw_memory_refs) == 2
        assert stage_memory.quality_metrics["quality_score"] == 0.95

    def test_stage_memory_to_pydantic(self) -> None:
        """Test converting StageMemoryModel to Pydantic StageMemory."""
        stage_id = uuid4()
        task_id = uuid4()
        agent_id = uuid4()

        stage_memory = StageMemoryModel(
            stage_id=stage_id,
            task_id=task_id,
            agent_id=agent_id,
            stage_type=StageType.EXECUTION.value,
            stage_summary="Executed implementation",
            stage_insights=["Completed API integration"],
            raw_memory_refs=[uuid4()],
            compression_ratio=8.5,
            compression_model="gpt-4.1-mini",
            quality_metrics={"quality_score": 0.92},
        )

        # Convert to Pydantic
        pydantic_stage = stage_memory.to_pydantic()

        assert pydantic_stage.stage_id == str(stage_id)
        assert pydantic_stage.task_id == str(task_id)
        assert pydantic_stage.stage_type == StageType.EXECUTION
        assert pydantic_stage.compression_ratio == 8.5
        assert pydantic_stage.quality_score == 0.92

    def test_stage_memory_repr(self) -> None:
        """Test __repr__ method."""
        stage_id = uuid4()
        task_id = uuid4()

        stage_memory = StageMemoryModel(
            stage_id=stage_id,
            task_id=task_id,
            agent_id=uuid4(),
            stage_type=StageType.REFLECTION.value,
            stage_summary="Reflected on errors",
            compression_ratio=9.2,
        )

        repr_str = repr(stage_memory)
        assert "StageMemoryModel" in repr_str
        assert str(stage_id) in repr_str
        assert str(task_id) in repr_str
        assert "compression=9.2" in repr_str


class TestTaskContextModel:
    """Test TaskContextModel structure and Pydantic conversion."""

    def test_create_task_context(self) -> None:
        """Test creating a TaskContextModel instance."""
        task_id = uuid4()
        agent_id = uuid4()
        stage_id = uuid4()

        task_context = TaskContextModel(
            task_id=task_id,
            agent_id=agent_id,
            task_goal="Implement authentication system",
            current_stage_id=stage_id,
            task_progress_summary="Completed planning and execution",
            critical_constraints=["Use JWT", "Redis storage"],
            performance_metrics={
                "error_rate": 0.05,
                "progress_rate": 0.75,
                "context_efficiency": 0.85,
            },
        )

        # Verify attributes
        assert task_context.task_id == task_id
        assert task_context.task_goal == "Implement authentication system"
        assert task_context.current_stage_id == stage_id
        assert len(task_context.critical_constraints) == 2
        assert task_context.performance_metrics["error_rate"] == 0.05

    def test_task_context_to_pydantic(self) -> None:
        """Test converting TaskContextModel to Pydantic TaskContext."""
        task_id = uuid4()
        agent_id = uuid4()

        task_context = TaskContextModel(
            task_id=task_id,
            agent_id=agent_id,
            task_goal="Build API",
            task_progress_summary="50% complete",
            critical_constraints=["REST API", "JSON responses"],
            performance_metrics={"progress": 0.5},
        )

        # Convert to Pydantic
        pydantic_context = task_context.to_pydantic()

        assert pydantic_context.task_id == str(task_id)
        assert pydantic_context.agent_id == str(agent_id)
        assert pydantic_context.task_goal == "Build API"
        assert pydantic_context.task_progress_summary == "50% complete"
        assert len(pydantic_context.critical_constraints) == 2

    def test_task_context_repr(self) -> None:
        """Test __repr__ method."""
        task_id = uuid4()
        agent_id = uuid4()
        stage_id = uuid4()

        task_context = TaskContextModel(
            task_id=task_id,
            agent_id=agent_id,
            task_goal="Test task",
            current_stage_id=stage_id,
        )

        repr_str = repr(task_context)
        assert "TaskContextModel" in repr_str
        assert str(task_id) in repr_str
        assert str(agent_id) in repr_str
        assert str(stage_id) in repr_str


class TestErrorModel:
    """Test ErrorModel structure and Pydantic conversion."""

    def test_create_error_record(self) -> None:
        """Test creating an ErrorModel instance."""
        error_id = uuid4()
        task_id = uuid4()
        stage_id = uuid4()
        agent_id = uuid4()

        error = ErrorModel(
            error_id=error_id,
            task_id=task_id,
            stage_id=stage_id,
            agent_id=agent_id,
            error_type=ErrorType.INCORRECT_ACTION.value,
            error_description="Used wrong API endpoint",
            context_when_occurred="During token refresh",
            recovery_action="Corrected to /auth/refresh",
            error_severity=0.6,
        )

        # Verify attributes
        assert error.error_id == error_id
        assert error.error_type == ErrorType.INCORRECT_ACTION.value
        assert error.error_severity == 0.6
        assert error.recovery_action == "Corrected to /auth/refresh"

    def test_error_to_pydantic(self) -> None:
        """Test converting ErrorModel to Pydantic ErrorRecord."""
        error_id = uuid4()
        task_id = uuid4()
        agent_id = uuid4()

        error = ErrorModel(
            error_id=error_id,
            task_id=task_id,
            agent_id=agent_id,
            error_type=ErrorType.HALLUCINATION.value,
            error_description="Generated false information",
            error_severity=0.8,
        )

        # Convert to Pydantic
        pydantic_error = error.to_pydantic()

        assert pydantic_error.error_id == str(error_id)
        assert pydantic_error.task_id == str(task_id)
        assert pydantic_error.error_type == ErrorType.HALLUCINATION
        assert pydantic_error.error_severity == 0.8

    def test_error_repr(self) -> None:
        """Test __repr__ method."""
        error_id = uuid4()
        task_id = uuid4()

        error = ErrorModel(
            error_id=error_id,
            task_id=task_id,
            agent_id=uuid4(),
            error_type=ErrorType.MISSING_INFO.value,
            error_description="Context missing",
            error_severity=0.4,
        )

        repr_str = repr(error)
        assert "ErrorModel" in repr_str
        assert str(error_id) in repr_str
        assert str(task_id) in repr_str
        assert "severity=0.4" in repr_str


class TestCompressionMetricsModel:
    """Test CompressionMetricsModel structure."""

    def test_create_compression_metrics(self) -> None:
        """Test creating a CompressionMetricsModel instance."""
        metric_id = uuid4()
        stage_id = uuid4()
        task_id = uuid4()

        metrics = CompressionMetricsModel(
            metric_id=metric_id,
            stage_id=stage_id,
            task_id=task_id,
            compression_type="stage",
            input_tokens=10000,
            output_tokens=1000,
            compression_ratio=10.0,
            critical_fact_retention_rate=0.95,
            coherence_score=0.92,
            cost_usd=0.05,
            model_used="gpt-4.1-mini",
        )

        # Verify attributes
        assert metrics.metric_id == metric_id
        assert metrics.compression_type == "stage"
        assert metrics.input_tokens == 10000
        assert metrics.output_tokens == 1000
        assert metrics.compression_ratio == 10.0
        assert float(metrics.cost_usd) == 0.05

    def test_compression_metrics_repr(self) -> None:
        """Test __repr__ method."""
        metric_id = uuid4()

        metrics = CompressionMetricsModel(
            metric_id=metric_id,
            compression_type="task",
            input_tokens=5000,
            output_tokens=1000,
            compression_ratio=5.0,
            cost_usd=0.03,
            model_used="gpt-4.1-mini",
        )

        repr_str = repr(metrics)
        assert "CompressionMetricsModel" in repr_str
        assert str(metric_id) in repr_str
        assert "type=task" in repr_str
        assert "ratio=5.0" in repr_str


class TestModelAttributes:
    """Test model table names and column configurations."""

    def test_memory_model_tablename(self) -> None:
        """Test MemoryModel has correct table name."""
        assert MemoryModel.__tablename__ == "memories"

    def test_stage_memory_model_tablename(self) -> None:
        """Test StageMemoryModel has correct table name."""
        assert StageMemoryModel.__tablename__ == "stage_memories"

    def test_task_context_model_tablename(self) -> None:
        """Test TaskContextModel has correct table name."""
        assert TaskContextModel.__tablename__ == "task_contexts"

    def test_error_model_tablename(self) -> None:
        """Test ErrorModel has correct table name."""
        assert ErrorModel.__tablename__ == "error_records"

    def test_compression_metrics_model_tablename(self) -> None:
        """Test CompressionMetricsModel has correct table name."""
        assert CompressionMetricsModel.__tablename__ == "compression_metrics"

    def test_models_have_primary_keys(self) -> None:
        """Test all models have proper primary key definitions."""
        # Check that primary key columns exist
        assert hasattr(MemoryModel, "memory_id")
        assert hasattr(StageMemoryModel, "stage_id")
        assert hasattr(TaskContextModel, "task_id")
        assert hasattr(ErrorModel, "error_id")
        assert hasattr(CompressionMetricsModel, "metric_id")
