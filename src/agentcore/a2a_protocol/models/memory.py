"""
Memory Models for Hybrid Architecture (Mem0 + COMPASS + Graph)

Pydantic models for the memory service implementing:
- Four-layer memory architecture (Working, Episodic, Semantic, Procedural)
- COMPASS stage-aware context management
- Error tracking and pattern detection
- Knowledge graph entities and relationships (Neo4j integration)

Component ID: MEM-005
Ticket: MEM-005 (Implement Hybrid Pydantic Models)
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryLayer(str, Enum):
    """Four-layer memory architecture."""

    WORKING = "working"  # Immediate context (Redis, 1-hour TTL)
    EPISODIC = "episodic"  # Recent episodes (Qdrant, 50 episodes)
    SEMANTIC = "semantic"  # Long-term facts (Qdrant, 1000+ entries)
    PROCEDURAL = "procedural"  # Action patterns (Qdrant, workflows)


class StageType(str, Enum):
    """COMPASS reasoning stage types."""

    PLANNING = "planning"  # Initial task analysis and strategy
    EXECUTION = "execution"  # Action execution and observation
    REFLECTION = "reflection"  # Error analysis and learning
    VERIFICATION = "verification"  # Quality checks and validation


class ErrorType(str, Enum):
    """Error classification types."""

    HALLUCINATION = "hallucination"  # LLM generated false information
    MISSING_INFO = "missing_info"  # Required context not available
    INCORRECT_ACTION = "incorrect_action"  # Wrong tool or action chosen
    CONTEXT_DEGRADATION = "context_degradation"  # Context quality deteriorated


class EntityType(str, Enum):
    """Entity classification for knowledge graph."""

    PERSON = "person"  # People, users, agents
    CONCEPT = "concept"  # Abstract ideas, principles
    TOOL = "tool"  # Functions, APIs, services
    CONSTRAINT = "constraint"  # Requirements, rules, limits
    OTHER = "other"  # Unclassified entities


class RelationshipType(str, Enum):
    """Relationship types for knowledge graph edges."""

    MENTIONS = "mentions"  # Entity mentioned in memory
    RELATES_TO = "relates_to"  # Semantic relationship
    PART_OF = "part_of"  # Hierarchical relationship
    FOLLOWS = "follows"  # Temporal sequence
    PRECEDES = "precedes"  # Temporal precedence
    CONTRADICTS = "contradicts"  # Conflicting information


class MemoryRecord(BaseModel):
    """
    Base memory record for all memory layers.

    Stores interaction content, metadata, embeddings, and relationships.
    Used across episodic, semantic, and procedural memory layers.
    Working memory stored in Redis cache separately.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "memory_id": "mem-123e4567-e89b-12d3-a456-426614174000",
                "memory_layer": "semantic",
                "content": "User prefers detailed technical explanations",
                "summary": "User preference for technical detail",
                "embedding": [0.1, 0.2, 0.3],
                "agent_id": "agent-123",
                "session_id": "session-456",
                "task_id": "task-789",
                "timestamp": "2025-11-14T10:30:00Z",
                "entities": ["user", "preference"],
                "facts": ["prefers detailed explanations"],
                "keywords": ["technical", "detail"],
                "stage_id": "stage-101",
                "is_critical": True,
            }
        }
    )

    memory_id: str = Field(
        default_factory=lambda: f"mem-{uuid4()}",
        description="Unique memory identifier",
    )
    memory_layer: MemoryLayer = Field(
        ..., description="Memory layer classification"
    )
    content: str = Field(..., description="Raw memory content")
    summary: str = Field(..., description="Compressed summary of content")
    embedding: list[float] = Field(
        default_factory=list,
        description="Vector embedding (768 dims for local, 1536 for OpenAI)",
    )

    # Scope (at least one required)
    agent_id: str | None = Field(None, description="Associated agent ID")
    session_id: str | None = Field(None, description="Associated session ID")
    user_id: str | None = Field(None, description="Associated user ID")
    task_id: str | None = Field(None, description="Associated task ID")

    # Metadata
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When memory was created",
    )
    entities: list[str] = Field(
        default_factory=list, description="Extracted entities"
    )
    facts: list[str] = Field(
        default_factory=list, description="Extracted facts"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Relevant keywords"
    )

    # Relationships
    related_memory_ids: list[str] = Field(
        default_factory=list, description="Related memory IDs"
    )
    parent_memory_id: str | None = Field(
        None, description="Parent memory for hierarchical structure"
    )

    # Tracking
    relevance_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Base relevance score"
    )
    access_count: int = Field(0, ge=0, description="Number of times accessed")
    last_accessed: datetime | None = Field(
        None, description="Last access timestamp"
    )

    # COMPASS enhancements
    stage_id: str | None = Field(None, description="Associated reasoning stage ID")
    is_critical: bool = Field(
        False, description="Whether memory contains critical information"
    )
    criticality_reason: str | None = Field(
        None, description="Why memory is marked critical"
    )

    # Procedural memory specific
    actions: list[str] = Field(
        default_factory=list, description="Actions taken (procedural only)"
    )
    outcome: str | None = Field(None, description="Outcome of actions")
    success: bool | None = Field(None, description="Whether actions succeeded")

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dims(cls, v: list[float]) -> list[float]:
        """Validate embedding dimensions (768 or 1536)."""
        if v and len(v) not in (768, 1536):
            raise ValueError(
                f"Embedding must be 768 or 1536 dimensions, got {len(v)}"
            )
        return v

    @field_validator("timestamp", "last_accessed")
    @classmethod
    def ensure_timezone(cls, v: datetime | None) -> datetime | None:
        """Ensure datetime has timezone."""
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class StageMemory(BaseModel):
    """
    COMPASS stage memory with compression tracking.

    Stores compressed summaries of raw memories per reasoning stage.
    Implements 10:1 compression ratio target.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stage_id": "stage-123e4567-e89b-12d3-a456-426614174000",
                "task_id": "task-789",
                "agent_id": "agent-123",
                "stage_type": "planning",
                "stage_summary": "Analyzed authentication requirements...",
                "stage_insights": ["Use JWT tokens", "Store in Redis"],
                "raw_memory_refs": ["mem-001", "mem-002"],
                "compression_ratio": 10.2,
                "compression_model": "gpt-4.1-mini",
                "quality_score": 0.97,
            }
        }
    )

    stage_id: str = Field(
        default_factory=lambda: f"stage-{uuid4()}",
        description="Unique stage identifier",
    )
    task_id: str = Field(..., description="Parent task ID")
    agent_id: str = Field(..., description="Agent ID")
    stage_type: StageType = Field(..., description="Reasoning stage type")
    stage_summary: str = Field(..., description="Compressed stage summary")
    stage_insights: list[str] = Field(
        default_factory=list, description="Key insights from stage"
    )
    raw_memory_refs: list[str] = Field(
        default_factory=list, description="References to raw memory IDs"
    )

    # Compression metrics
    relevance_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Stage relevance score"
    )
    compression_ratio: float = Field(
        ..., ge=1.0, le=20.0, description="Achieved compression ratio"
    )
    compression_model: str = Field(
        ..., description="Model used for compression (e.g., gpt-4.1-mini)"
    )
    quality_score: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Compression quality score (fact retention)",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When stage was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )
    completed_at: datetime | None = Field(
        None, description="When stage was completed"
    )

    @field_validator("created_at", "updated_at", "completed_at")
    @classmethod
    def ensure_timezone(cls, v: datetime | None) -> datetime | None:
        """Ensure datetime has timezone."""
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class TaskContext(BaseModel):
    """
    COMPASS task context with progressive summary.

    Stores progressive task summary from stage summaries.
    Implements 5:1 compression ratio target.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "task-123e4567-e89b-12d3-a456-426614174000",
                "agent_id": "agent-123",
                "task_goal": "Implement authentication system",
                "current_stage_id": "stage-456",
                "task_progress_summary": "Completed planning and execution...",
                "critical_constraints": ["Use JWT", "Redis storage"],
                "performance_metrics": {
                    "error_rate": 0.05,
                    "progress_rate": 0.75,
                    "context_efficiency": 0.85,
                },
            }
        }
    )

    task_id: str = Field(
        default_factory=lambda: f"task-{uuid4()}",
        description="Unique task identifier",
    )
    agent_id: str = Field(..., description="Agent ID")
    task_goal: str = Field(..., description="High-level task objective")
    current_stage_id: str | None = Field(
        None, description="Current active stage ID"
    )
    task_progress_summary: str = Field(
        "", description="Progressive summary across stages"
    )
    critical_constraints: list[str] = Field(
        default_factory=list, description="Must-remember constraints"
    )
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Task performance tracking (error_rate, progress_rate, etc.)",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When task context was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    @field_validator("created_at", "updated_at")
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class ErrorRecord(BaseModel):
    """
    Error tracking record with severity scoring.

    Captures errors explicitly for pattern detection and learning.
    Severity scored on 0-1 scale (0=minor, 1=critical).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_id": "err-123e4567-e89b-12d3-a456-426614174000",
                "task_id": "task-789",
                "stage_id": "stage-456",
                "agent_id": "agent-123",
                "error_type": "incorrect_action",
                "error_description": "Used wrong API endpoint",
                "context_when_occurred": "During token refresh attempt",
                "recovery_action": "Corrected to /auth/refresh",
                "error_severity": 0.6,
            }
        }
    )

    error_id: str = Field(
        default_factory=lambda: f"err-{uuid4()}",
        description="Unique error identifier",
    )
    task_id: str = Field(..., description="Task ID where error occurred")
    stage_id: str | None = Field(
        None, description="Stage ID where error occurred"
    )
    agent_id: str = Field(..., description="Agent ID")
    error_type: ErrorType = Field(..., description="Error classification")
    error_description: str = Field(..., description="Detailed error description")
    context_when_occurred: str = Field(
        "", description="Context when error happened"
    )
    recovery_action: str | None = Field(
        None, description="Action taken to recover"
    )
    error_severity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Error severity score (0=minor, 1=critical)",
    )
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When error was recorded",
    )

    @field_validator("error_severity")
    @classmethod
    def validate_severity(cls, v: float) -> float:
        """Validate severity is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Severity must be 0.0-1.0, got {v}")
        return v

    @field_validator("recorded_at")
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class EntityNode(BaseModel):
    """
    Knowledge graph entity node (Neo4j).

    Represents entities extracted from memories for graph storage.
    Used in hybrid search combining vector + graph traversal.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_id": "ent-123e4567-e89b-12d3-a456-426614174000",
                "entity_name": "JWT Authentication",
                "entity_type": "concept",
                "properties": {
                    "domain": "security",
                    "confidence": 0.95,
                },
                "memory_refs": ["mem-001", "mem-002"],
            }
        }
    )

    entity_id: str = Field(
        default_factory=lambda: f"ent-{uuid4()}",
        description="Unique entity identifier",
    )
    entity_name: str = Field(..., description="Entity display name")
    entity_type: EntityType = Field(..., description="Entity classification")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity properties (domain, confidence, etc.)",
    )
    embedding: list[float] = Field(
        default_factory=list,
        description="Entity embedding for similarity search",
    )
    memory_refs: list[str] = Field(
        default_factory=list,
        description="Memory IDs where entity appears",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When entity was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dims(cls, v: list[float]) -> list[float]:
        """Validate embedding dimensions (768 or 1536)."""
        if v and len(v) not in (768, 1536):
            raise ValueError(
                f"Embedding must be 768 or 1536 dimensions, got {len(v)}"
            )
        return v

    @field_validator("created_at", "updated_at")
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class RelationshipEdge(BaseModel):
    """
    Knowledge graph relationship edge (Neo4j).

    Represents connections between entities in the knowledge graph.
    Used for graph traversal and contextual memory retrieval.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "relationship_id": "rel-123e4567-e89b-12d3-a456-426614174000",
                "source_entity_id": "ent-001",
                "target_entity_id": "ent-002",
                "relationship_type": "relates_to",
                "properties": {
                    "strength": 0.85,
                    "context": "authentication system",
                },
            }
        }
    )

    relationship_id: str = Field(
        default_factory=lambda: f"rel-{uuid4()}",
        description="Unique relationship identifier",
    )
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: RelationshipType = Field(
        ..., description="Relationship classification"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties (strength, context, etc.)",
    )
    memory_refs: list[str] = Field(
        default_factory=list,
        description="Memory IDs where relationship appears",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When relationship was created",
    )
    access_count: int = Field(
        0, ge=0, description="Number of times relationship traversed"
    )

    @field_validator("created_at")
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


# Export all models
__all__ = [
    "MemoryLayer",
    "StageType",
    "ErrorType",
    "EntityType",
    "RelationshipType",
    "MemoryRecord",
    "StageMemory",
    "TaskContext",
    "ErrorRecord",
    "EntityNode",
    "RelationshipEdge",
]
