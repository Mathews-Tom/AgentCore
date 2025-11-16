"""
Unit tests for Memory Models (MEM-005)

Tests all Pydantic models for the hybrid memory architecture:
- Field validation
- JSON serialization/deserialization
- Enum values
- Timestamp handling
- Default values
"""

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    ErrorRecord,
    ErrorType,
    MemoryLayer,
    MemoryRecord,
    RelationshipEdge,
    RelationshipType,
    StageMemory,
    StageType,
    TaskContext,
)


class TestMemoryLayer:
    """Test MemoryLayer enum."""

    def test_all_layers_defined(self):
        """Test all four memory layers are defined."""
        assert MemoryLayer.WORKING == "working"
        assert MemoryLayer.EPISODIC == "episodic"
        assert MemoryLayer.SEMANTIC == "semantic"
        assert MemoryLayer.PROCEDURAL == "procedural"

    def test_enum_values(self):
        """Test enum can be accessed by value."""
        assert MemoryLayer("working") == MemoryLayer.WORKING
        assert MemoryLayer("episodic") == MemoryLayer.EPISODIC


class TestStageType:
    """Test StageType enum."""

    def test_all_stages_defined(self):
        """Test all COMPASS stages are defined."""
        assert StageType.PLANNING == "planning"
        assert StageType.EXECUTION == "execution"
        assert StageType.REFLECTION == "reflection"
        assert StageType.VERIFICATION == "verification"


class TestErrorType:
    """Test ErrorType enum."""

    def test_all_error_types_defined(self):
        """Test all error types are defined."""
        assert ErrorType.HALLUCINATION == "hallucination"
        assert ErrorType.MISSING_INFO == "missing_info"
        assert ErrorType.INCORRECT_ACTION == "incorrect_action"
        assert ErrorType.CONTEXT_DEGRADATION == "context_degradation"


class TestEntityType:
    """Test EntityType enum."""

    def test_all_entity_types_defined(self):
        """Test all entity types are defined."""
        assert EntityType.PERSON == "person"
        assert EntityType.CONCEPT == "concept"
        assert EntityType.TOOL == "tool"
        assert EntityType.CONSTRAINT == "constraint"
        assert EntityType.OTHER == "other"


class TestRelationshipType:
    """Test RelationshipType enum."""

    def test_all_relationship_types_defined(self):
        """Test all relationship types are defined."""
        assert RelationshipType.MENTIONS == "mentions"
        assert RelationshipType.RELATES_TO == "relates_to"
        assert RelationshipType.PART_OF == "part_of"
        assert RelationshipType.FOLLOWS == "follows"
        assert RelationshipType.PRECEDES == "precedes"
        assert RelationshipType.CONTRADICTS == "contradicts"


class TestMemoryRecord:
    """Test MemoryRecord model."""

    def test_minimal_creation(self):
        """Test creating memory with minimal required fields."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Test memory content",
            summary="Test summary",
            agent_id="agent-123",
        )
        assert memory.memory_id.startswith("mem-")
        assert memory.memory_layer == MemoryLayer.SEMANTIC
        assert memory.content == "Test memory content"
        assert memory.summary == "Test summary"
        assert memory.agent_id == "agent-123"

    def test_default_values(self):
        """Test default field values."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.EPISODIC,
            content="Content",
            summary="Summary",
            agent_id="agent-123",
        )
        assert memory.embedding == []
        assert memory.entities == []
        assert memory.facts == []
        assert memory.keywords == []
        assert memory.related_memory_ids == []
        assert memory.relevance_score == 1.0
        assert memory.access_count == 0
        assert memory.is_critical is False
        assert memory.actions == []

    def test_timestamp_defaults_to_utc_now(self):
        """Test timestamp defaults to current UTC time."""
        before = datetime.now(UTC)
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Content",
            summary="Summary",
            agent_id="agent-123",
        )
        after = datetime.now(UTC)
        assert before <= memory.timestamp <= after
        assert memory.timestamp.tzinfo is not None

    def test_embedding_validation_768_dims(self):
        """Test embedding validation accepts 768 dimensions."""
        embedding = [0.1] * 768
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Content",
            summary="Summary",
            agent_id="agent-123",
            embedding=embedding,
        )
        assert len(memory.embedding) == 768

    def test_embedding_validation_1536_dims(self):
        """Test embedding validation accepts 1536 dimensions."""
        embedding = [0.1] * 1536
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Content",
            summary="Summary",
            agent_id="agent-123",
            embedding=embedding,
        )
        assert len(memory.embedding) == 1536

    def test_embedding_validation_rejects_invalid_dims(self):
        """Test embedding validation rejects invalid dimensions."""
        with pytest.raises(ValueError, match="768 or 1536 dimensions"):
            MemoryRecord(
                memory_layer=MemoryLayer.SEMANTIC,
                content="Content",
                summary="Summary",
                agent_id="agent-123",
                embedding=[0.1] * 512,  # Invalid dimension
            )

    def test_json_serialization(self):
        """Test model can be serialized to JSON."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Content",
            summary="Summary",
            agent_id="agent-123",
            entities=["entity1", "entity2"],
        )
        json_data = memory.model_dump()
        assert json_data["memory_layer"] == "semantic"
        assert json_data["content"] == "Content"
        assert json_data["entities"] == ["entity1", "entity2"]

    def test_json_deserialization(self):
        """Test model can be deserialized from JSON."""
        data = {
            "memory_layer": "semantic",
            "content": "Content",
            "summary": "Summary",
            "agent_id": "agent-123",
        }
        memory = MemoryRecord(**data)
        assert memory.memory_layer == MemoryLayer.SEMANTIC
        assert memory.content == "Content"

    def test_compass_fields(self):
        """Test COMPASS-specific fields."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.SEMANTIC,
            content="Content",
            summary="Summary",
            agent_id="agent-123",
            stage_id="stage-456",
            is_critical=True,
            criticality_reason="Contains user requirement",
        )
        assert memory.stage_id == "stage-456"
        assert memory.is_critical is True
        assert memory.criticality_reason == "Contains user requirement"

    def test_procedural_fields(self):
        """Test procedural memory specific fields."""
        memory = MemoryRecord(
            memory_layer=MemoryLayer.PROCEDURAL,
            content="Action sequence",
            summary="Summary",
            agent_id="agent-123",
            actions=["action1", "action2"],
            outcome="Success",
            success=True,
        )
        assert memory.actions == ["action1", "action2"]
        assert memory.outcome == "Success"
        assert memory.success is True


class TestStageMemory:
    """Test StageMemory model."""

    def test_minimal_creation(self):
        """Test creating stage memory with minimal fields."""
        stage = StageMemory(
            task_id="task-123",
            agent_id="agent-456",
            stage_type=StageType.PLANNING,
            stage_summary="Planning summary",
            compression_ratio=10.0,
            compression_model="gpt-4.1-mini",
        )
        assert stage.stage_id.startswith("stage-")
        assert stage.task_id == "task-123"
        assert stage.stage_type == StageType.PLANNING

    def test_compression_metrics(self):
        """Test compression metric fields."""
        stage = StageMemory(
            task_id="task-123",
            agent_id="agent-456",
            stage_type=StageType.EXECUTION,
            stage_summary="Execution summary",
            compression_ratio=10.2,
            compression_model="gpt-4.1-mini",
            quality_score=0.97,
        )
        assert stage.compression_ratio == 10.2
        assert 1.0 <= stage.compression_ratio <= 20.0
        assert stage.quality_score == 0.97
        assert 0.0 <= stage.quality_score <= 1.0

    def test_compression_ratio_validation(self):
        """Test compression ratio must be between 1 and 20."""
        with pytest.raises(ValueError):
            StageMemory(
                task_id="task-123",
                agent_id="agent-456",
                stage_type=StageType.PLANNING,
                stage_summary="Summary",
                compression_ratio=0.5,  # Too low
                compression_model="gpt-4.1-mini",
            )

        with pytest.raises(ValueError):
            StageMemory(
                task_id="task-123",
                agent_id="agent-456",
                stage_type=StageType.PLANNING,
                stage_summary="Summary",
                compression_ratio=25.0,  # Too high
                compression_model="gpt-4.1-mini",
            )

    def test_timestamps(self):
        """Test timestamp fields are set correctly."""
        before = datetime.now(UTC)
        stage = StageMemory(
            task_id="task-123",
            agent_id="agent-456",
            stage_type=StageType.REFLECTION,
            stage_summary="Reflection summary",
            compression_ratio=10.0,
            compression_model="gpt-4.1-mini",
        )
        after = datetime.now(UTC)
        assert before <= stage.created_at <= after
        assert before <= stage.updated_at <= after
        assert stage.completed_at is None

    def test_raw_memory_refs(self):
        """Test raw memory references tracking."""
        stage = StageMemory(
            task_id="task-123",
            agent_id="agent-456",
            stage_type=StageType.PLANNING,
            stage_summary="Summary",
            compression_ratio=10.0,
            compression_model="gpt-4.1-mini",
            raw_memory_refs=["mem-001", "mem-002", "mem-003"],
        )
        assert len(stage.raw_memory_refs) == 3
        assert "mem-001" in stage.raw_memory_refs


class TestTaskContext:
    """Test TaskContext model."""

    def test_minimal_creation(self):
        """Test creating task context with minimal fields."""
        task = TaskContext(
            agent_id="agent-123", task_goal="Implement authentication"
        )
        assert task.task_id.startswith("task-")
        assert task.agent_id == "agent-123"
        assert task.task_goal == "Implement authentication"

    def test_default_values(self):
        """Test default field values."""
        task = TaskContext(agent_id="agent-123", task_goal="Goal")
        assert task.task_progress_summary == ""
        assert task.critical_constraints == []
        assert task.performance_metrics == {}
        assert task.current_stage_id is None

    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        task = TaskContext(
            agent_id="agent-123",
            task_goal="Goal",
            performance_metrics={
                "error_rate": 0.05,
                "progress_rate": 0.75,
                "context_efficiency": 0.85,
            },
        )
        assert task.performance_metrics["error_rate"] == 0.05
        assert task.performance_metrics["progress_rate"] == 0.75
        assert task.performance_metrics["context_efficiency"] == 0.85

    def test_critical_constraints(self):
        """Test critical constraints tracking."""
        task = TaskContext(
            agent_id="agent-123",
            task_goal="Goal",
            critical_constraints=["Use JWT", "Redis storage", "TLS encryption"],
        )
        assert len(task.critical_constraints) == 3
        assert "Use JWT" in task.critical_constraints


class TestErrorRecord:
    """Test ErrorRecord model."""

    def test_minimal_creation(self):
        """Test creating error record with minimal fields."""
        error = ErrorRecord(
            task_id="task-123",
            agent_id="agent-456",
            error_type=ErrorType.INCORRECT_ACTION,
            error_description="Used wrong endpoint",
            error_severity=0.6,
        )
        assert error.error_id.startswith("err-")
        assert error.task_id == "task-123"
        assert error.error_type == ErrorType.INCORRECT_ACTION
        assert error.error_severity == 0.6

    def test_severity_validation_valid(self):
        """Test severity accepts values 0.0 to 1.0."""
        error = ErrorRecord(
            task_id="task-123",
            agent_id="agent-456",
            error_type=ErrorType.HALLUCINATION,
            error_description="False info",
            error_severity=0.0,
        )
        assert error.error_severity == 0.0

        error = ErrorRecord(
            task_id="task-123",
            agent_id="agent-456",
            error_type=ErrorType.HALLUCINATION,
            error_description="False info",
            error_severity=1.0,
        )
        assert error.error_severity == 1.0

    def test_severity_validation_rejects_below_zero(self):
        """Test severity rejects values below 0."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            ErrorRecord(
                task_id="task-123",
                agent_id="agent-456",
                error_type=ErrorType.MISSING_INFO,
                error_description="Missing",
                error_severity=-0.1,
            )

    def test_severity_validation_rejects_above_one(self):
        """Test severity rejects values above 1."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            ErrorRecord(
                task_id="task-123",
                agent_id="agent-456",
                error_type=ErrorType.CONTEXT_DEGRADATION,
                error_description="Degraded",
                error_severity=1.5,
            )

    def test_recovery_action(self):
        """Test recovery action tracking."""
        error = ErrorRecord(
            task_id="task-123",
            agent_id="agent-456",
            error_type=ErrorType.INCORRECT_ACTION,
            error_description="Wrong endpoint",
            error_severity=0.6,
            recovery_action="Corrected to /auth/refresh",
        )
        assert error.recovery_action == "Corrected to /auth/refresh"

    def test_timestamp_default(self):
        """Test recorded_at defaults to current UTC time."""
        before = datetime.now(UTC)
        error = ErrorRecord(
            task_id="task-123",
            agent_id="agent-456",
            error_type=ErrorType.HALLUCINATION,
            error_description="Error",
            error_severity=0.5,
        )
        after = datetime.now(UTC)
        assert before <= error.recorded_at <= after
        assert error.recorded_at.tzinfo is not None


class TestEntityNode:
    """Test EntityNode model."""

    def test_minimal_creation(self):
        """Test creating entity node with minimal fields."""
        entity = EntityNode(
            entity_name="JWT Authentication", entity_type=EntityType.CONCEPT
        )
        assert entity.entity_id.startswith("ent-")
        assert entity.entity_name == "JWT Authentication"
        assert entity.entity_type == EntityType.CONCEPT

    def test_default_values(self):
        """Test default field values."""
        entity = EntityNode(
            entity_name="User", entity_type=EntityType.PERSON
        )
        assert entity.properties == {}
        assert entity.embedding == []
        assert entity.memory_refs == []

    def test_properties_dict(self):
        """Test entity properties storage."""
        entity = EntityNode(
            entity_name="API",
            entity_type=EntityType.TOOL,
            properties={
                "domain": "security",
                "confidence": 0.95,
                "source": "documentation",
            },
        )
        assert entity.properties["domain"] == "security"
        assert entity.properties["confidence"] == 0.95

    def test_embedding_validation(self):
        """Test entity embedding validation."""
        embedding = [0.1] * 768
        entity = EntityNode(
            entity_name="Concept",
            entity_type=EntityType.CONCEPT,
            embedding=embedding,
        )
        assert len(entity.embedding) == 768

    def test_memory_refs_tracking(self):
        """Test memory references tracking."""
        entity = EntityNode(
            entity_name="Requirement",
            entity_type=EntityType.CONSTRAINT,
            memory_refs=["mem-001", "mem-002"],
        )
        assert len(entity.memory_refs) == 2
        assert "mem-001" in entity.memory_refs


class TestRelationshipEdge:
    """Test RelationshipEdge model."""

    def test_minimal_creation(self):
        """Test creating relationship edge with minimal fields."""
        rel = RelationshipEdge(
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relationship_type=RelationshipType.RELATES_TO,
        )
        assert rel.relationship_id.startswith("rel-")
        assert rel.source_entity_id == "ent-001"
        assert rel.target_entity_id == "ent-002"
        assert rel.relationship_type == RelationshipType.RELATES_TO

    def test_default_values(self):
        """Test default field values."""
        rel = RelationshipEdge(
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relationship_type=RelationshipType.MENTIONS,
        )
        assert rel.properties == {}
        assert rel.memory_refs == []
        assert rel.access_count == 0

    def test_relationship_properties(self):
        """Test relationship properties storage."""
        rel = RelationshipEdge(
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relationship_type=RelationshipType.RELATES_TO,
            properties={
                "strength": 0.85,
                "context": "authentication system",
            },
        )
        assert rel.properties["strength"] == 0.85
        assert rel.properties["context"] == "authentication system"

    def test_access_tracking(self):
        """Test relationship access tracking."""
        rel = RelationshipEdge(
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relationship_type=RelationshipType.FOLLOWS,
            access_count=5,
        )
        assert rel.access_count == 5

    def test_temporal_relationships(self):
        """Test temporal relationship types."""
        rel_follows = RelationshipEdge(
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relationship_type=RelationshipType.FOLLOWS,
        )
        assert rel_follows.relationship_type == RelationshipType.FOLLOWS

        rel_precedes = RelationshipEdge(
            source_entity_id="ent-002",
            target_entity_id="ent-003",
            relationship_type=RelationshipType.PRECEDES,
        )
        assert rel_precedes.relationship_type == RelationshipType.PRECEDES
