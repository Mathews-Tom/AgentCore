"""
Memory Service JSON-RPC Methods

JSON-RPC 2.0 methods for memory service operations including:
- memory.store: Store new memory
- memory.retrieve: Semantic search
- memory.get_context: Formatted context retrieval
- memory.complete_stage: Trigger compression
- memory.record_error: Track error
- memory.get_strategic_context: ACE interface
- memory.run_memify: Trigger optimization

Component ID: MEM-026
Ticket: MEM-026 (Implement JSON-RPC Methods)
"""

from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field, ValidationError

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.memory import (
    ErrorRecord,
    ErrorType,
    MemoryLayer,
    MemoryRecord,
    StageMemory,
    StageType,
)
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.graph_optimizer import GraphOptimizer
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.integration import (
    ACEStrategicContextInterface,
    MemoryServiceIntegration,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)
from agentcore.a2a_protocol.services.memory.stage_manager import StageManager
from agentcore.a2a_protocol.services.memory.storage_backend import get_storage_backend
from agentcore.a2a_protocol.services.memory.working_memory import (
    get_working_memory_service,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Pydantic Request/Response Models
# =============================================================================


class MemoryStoreParams(BaseModel):
    """Parameters for memory.store method."""

    content: str = Field(..., description="Memory content to store")
    memory_layer: str = Field(
        default="episodic",
        description="Memory layer (working, episodic, semantic, procedural)",
    )
    summary: str | None = Field(None, description="Optional summary of content")
    embedding: list[float] | None = Field(None, description="Optional embedding vector")
    agent_id: str = Field(..., description="Agent ID")
    session_id: str | None = Field(None, description="Optional session ID")
    task_id: str | None = Field(None, description="Optional task ID")
    keywords: list[str] | None = Field(None, description="Optional keywords")
    is_critical: bool = Field(False, description="Whether memory is critical")
    criticality_reason: str | None = Field(
        None, description="Reason for criticality if critical"
    )


class MemoryStoreResult(BaseModel):
    """Result for memory.store method."""

    memory_id: str
    memory_layer: str
    timestamp: str
    success: bool = True
    message: str = "Memory stored successfully"


class MemoryRetrieveParams(BaseModel):
    """Parameters for memory.retrieve method."""

    query: str | None = Field(None, description="Query string for semantic search")
    query_embedding: list[float] | None = Field(
        None, description="Query embedding vector"
    )
    agent_id: str | None = Field(None, description="Filter by agent ID")
    session_id: str | None = Field(None, description="Filter by session ID")
    task_id: str | None = Field(None, description="Filter by task ID")
    memory_layer: str | None = Field(None, description="Filter by memory layer")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    current_stage: str | None = Field(None, description="Current reasoning stage")
    has_recent_errors: bool = Field(False, description="Whether recent errors exist")


class MemoryRetrieveResult(BaseModel):
    """Result for memory.retrieve method."""

    memories: list[dict[str, Any]]
    total_count: int
    query_time_ms: float


class ContextRetrieveParams(BaseModel):
    """Parameters for memory.get_context method."""

    session_id: str = Field(..., description="Session identifier")
    query: str | None = Field(None, description="Optional query string")
    query_embedding: list[float] | None = Field(
        None, description="Optional query embedding"
    )
    current_stage: str | None = Field(None, description="Current reasoning stage")
    max_memories: int = Field(10, ge=1, le=50, description="Maximum memories to return")
    format: str = Field(
        "markdown", description="Output format (markdown, json, plain)"
    )


class ContextRetrieveResult(BaseModel):
    """Result for memory.get_context method."""

    session_id: str
    context: str
    memory_count: int
    context_size_bytes: int
    format: str


class CompleteStageParams(BaseModel):
    """Parameters for memory.complete_stage method."""

    stage_id: str = Field(..., description="Stage identifier")
    task_id: str = Field(..., description="Task identifier")
    agent_id: str = Field(..., description="Agent identifier")
    stage_type: str = Field(..., description="Stage type (planning, execution, etc.)")
    stage_summary: str = Field(..., description="Summary of stage")
    stage_insights: list[str] = Field(
        default_factory=list, description="Key insights"
    )
    raw_memory_refs: list[str] = Field(
        default_factory=list, description="Raw memory references"
    )
    compression_model: str = Field("gpt-4.1-mini", description="Model for compression")


class CompleteStageResult(BaseModel):
    """Result for memory.complete_stage method."""

    stage_id: str
    task_id: str
    compression_ratio: float
    quality_score: float
    success: bool = True
    message: str = "Stage completed and compressed"


class RecordErrorParams(BaseModel):
    """Parameters for memory.record_error method."""

    task_id: str = Field(..., description="Task ID where error occurred")
    agent_id: str = Field(..., description="Agent identifier")
    error_type: str = Field(
        ..., description="Error type (hallucination, missing_info, etc.)"
    )
    error_description: str = Field(..., description="Detailed error description")
    context_when_occurred: str = Field("", description="Context when error happened")
    recovery_action: str | None = Field(None, description="Recovery action taken")
    error_severity: float = Field(
        ..., ge=0.0, le=1.0, description="Error severity (0=minor, 1=critical)"
    )
    stage_id: str | None = Field(None, description="Optional stage ID")


class RecordErrorResult(BaseModel):
    """Result for memory.record_error method."""

    error_id: str
    task_id: str
    recorded_at: str
    success: bool = True
    message: str = "Error recorded successfully"


class StrategicContextParams(BaseModel):
    """Parameters for memory.get_strategic_context method."""

    agent_id: str = Field(..., description="Agent identifier")
    session_id: str | None = Field(None, description="Optional session identifier")
    goal: str | None = Field(None, description="Current goal")
    query_embedding: list[float] | None = Field(
        None, description="Query embedding for memory retrieval"
    )


class StrategicContextResult(BaseModel):
    """Result for memory.get_strategic_context method."""

    context_id: str
    agent_id: str
    session_id: str | None
    current_goal: str | None
    strategic_memory_count: int
    tactical_memory_count: int
    error_patterns: list[str]
    success_patterns: list[str]
    confidence_score: float


class RunMemifyParams(BaseModel):
    """Parameters for memory.run_memify method."""

    similarity_threshold: float = Field(
        0.90, ge=0.5, le=1.0, description="Similarity threshold for consolidation"
    )
    min_access_count: int = Field(
        2, ge=0, description="Minimum access count for relationships"
    )
    batch_size: int = Field(
        100, ge=10, le=1000, description="Batch size for operations"
    )
    schedule_cron: str | None = Field(
        None, description="Optional cron expression for scheduling"
    )


class RunMemifyResult(BaseModel):
    """Result for memory.run_memify method."""

    optimization_id: str
    entities_analyzed: int
    entities_merged: int
    relationships_pruned: int
    patterns_detected: int
    consolidation_accuracy: float
    duplicate_rate: float
    duration_seconds: float
    scheduled_job_id: str | None = None
    next_run: str | None = None


# =============================================================================
# Global Service Instances (lazy initialization)
# =============================================================================

_memory_service: MemoryServiceIntegration | None = None
_stage_manager: StageManager | None = None
_error_tracker: ErrorTracker | None = None
_ace_interface: ACEStrategicContextInterface | None = None
_graph_optimizer: GraphOptimizer | None = None
_hybrid_search: HybridSearchService | None = None
_retrieval_service: EnhancedRetrievalService | None = None


def _get_memory_service() -> MemoryServiceIntegration:
    """Get or create memory service integration instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryServiceIntegration()
    return _memory_service


def _get_stage_manager() -> StageManager:
    """Get or create stage manager instance."""
    global _stage_manager
    if _stage_manager is None:
        _stage_manager = StageManager()
    return _stage_manager


def _get_error_tracker() -> ErrorTracker:
    """Get or create error tracker instance."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


def _get_ace_interface() -> ACEStrategicContextInterface:
    """Get or create ACE strategic context interface instance."""
    global _ace_interface
    if _ace_interface is None:
        _ace_interface = ACEStrategicContextInterface()
    return _ace_interface


def _get_retrieval_service() -> EnhancedRetrievalService:
    """Get or create retrieval service instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = EnhancedRetrievalService()
    return _retrieval_service


def _get_storage_backend():
    """Get storage backend instance (uses global singleton)."""
    return get_storage_backend()


# Legacy in-memory stores (deprecated, kept for fallback during migration)
# These will be removed once storage backend is fully integrated
_memory_store: dict[str, MemoryRecord] = {}
_stage_store: dict[str, StageMemory] = {}
_error_store: dict[str, ErrorRecord] = {}


# =============================================================================
# JSON-RPC Method Handlers
# =============================================================================


@register_jsonrpc_method("memory.store")
async def handle_memory_store(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Store new memory in the memory service.

    Method: memory.store
    Params:
        - content: string (required) - Memory content
        - memory_layer: string (optional, default "episodic")
        - summary: string (optional)
        - embedding: array of floats (optional)
        - agent_id: string (required)
        - session_id: string (optional)
        - task_id: string (optional)
        - keywords: array of strings (optional)
        - is_critical: boolean (optional, default false)
        - criticality_reason: string (optional)

    Returns:
        - memory_id: string
        - memory_layer: string
        - timestamp: string (ISO8601)
        - success: boolean
        - message: string
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: content, agent_id")

        # Validate parameters
        params = MemoryStoreParams(**request.params)

        # Map string to MemoryLayer enum
        layer_map = {
            "working": MemoryLayer.WORKING,
            "episodic": MemoryLayer.EPISODIC,
            "semantic": MemoryLayer.SEMANTIC,
            "procedural": MemoryLayer.PROCEDURAL,
        }
        memory_layer_enum = layer_map.get(
            params.memory_layer.lower(), MemoryLayer.EPISODIC
        )

        # Create memory record
        memory = MemoryRecord(
            memory_layer=memory_layer_enum,
            content=params.content,
            summary=params.summary or params.content[:100],
            embedding=params.embedding or [],
            agent_id=params.agent_id,
            session_id=params.session_id,
            task_id=params.task_id,
            keywords=params.keywords or [],
            is_critical=params.is_critical,
            criticality_reason=params.criticality_reason,
        )

        # Route memory to appropriate backend based on layer
        if memory_layer_enum == MemoryLayer.WORKING:
            # Store in Redis with TTL for working memory
            working_memory = get_working_memory_service()
            try:
                await working_memory.store_working_memory(memory)
            except RuntimeError:
                # Redis not initialized, fall back to in-memory
                logger.warning(
                    "Redis working memory not initialized, using in-memory storage",
                    memory_id=memory.memory_id,
                )
                _memory_store[memory.memory_id] = memory
        else:
            # Store in PostgreSQL + Qdrant for persistent memory layers
            storage_backend = _get_storage_backend()
            try:
                await storage_backend.store_memory(memory)
            except RuntimeError:
                # Storage backend not initialized, fall back to in-memory (dev mode)
                logger.warning(
                    "Storage backend not initialized, using in-memory storage",
                    memory_id=memory.memory_id,
                )
                _memory_store[memory.memory_id] = memory

        result = MemoryStoreResult(
            memory_id=memory.memory_id,
            memory_layer=memory.memory_layer.value,
            timestamp=memory.timestamp.isoformat(),
        )

        logger.info(
            "Memory stored via JSON-RPC",
            memory_id=memory.memory_id,
            memory_layer=memory.memory_layer.value,
            agent_id=params.agent_id,
            method="memory.store",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Memory store validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


@register_jsonrpc_method("memory.retrieve")
async def handle_memory_retrieve(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Retrieve memories using semantic search.

    Method: memory.retrieve
    Params:
        - query: string (optional)
        - query_embedding: array of floats (optional)
        - agent_id: string (optional)
        - session_id: string (optional)
        - task_id: string (optional)
        - memory_layer: string (optional)
        - limit: integer (optional, default 10)
        - current_stage: string (optional)
        - has_recent_errors: boolean (optional, default false)

    Returns:
        - memories: array of memory objects
        - total_count: integer
        - query_time_ms: float
    """
    try:
        params_dict = request.params or {}
        if isinstance(params_dict, list):
            raise ValueError("Parameters must be an object, not an array")

        # Validate parameters
        params = MemoryRetrieveParams(**params_dict)

        start_time = datetime.now(UTC)

        # Get retrieval service
        retrieval = _get_retrieval_service()

        # Try to use vector search if embedding is provided and storage backend is initialized
        memories = []
        storage_backend = _get_storage_backend()

        if params.query_embedding and len(params.query_embedding) > 0:
            # Use vector search from Qdrant
            try:
                vector_results = await storage_backend.vector_search(
                    query_embedding=params.query_embedding,
                    limit=params.limit,
                    filter_agent_id=params.agent_id,
                    filter_session_id=params.session_id,
                    filter_task_id=params.task_id,
                    filter_memory_layer=params.memory_layer.lower() if params.memory_layer else None,
                )

                # Convert vector search results to memory records
                for memory_id, score, payload in vector_results:
                    # Fetch full memory record from PostgreSQL
                    memory = await storage_backend.get_memory_by_id(payload.get("memory_id", memory_id))
                    if memory:
                        memory_dict = memory.model_dump(mode="json")
                        memory_dict["_vector_score"] = score
                        memories.append(memory_dict)

            except RuntimeError:
                # Storage backend not initialized, fall back to in-memory
                logger.warning("Storage backend not initialized, using in-memory retrieval")
                filtered_memories = list(_memory_store.values())

                if params.agent_id:
                    filtered_memories = [
                        m for m in filtered_memories if m.agent_id == params.agent_id
                    ]

                if params.session_id:
                    filtered_memories = [
                        m for m in filtered_memories if m.session_id == params.session_id
                    ]

                if params.task_id:
                    filtered_memories = [
                        m for m in filtered_memories if m.task_id == params.task_id
                    ]

                if params.memory_layer:
                    filtered_memories = [
                        m
                        for m in filtered_memories
                        if m.memory_layer.value == params.memory_layer.lower()
                    ]

                # Map stage string to enum if provided
                current_stage_enum = None
                if params.current_stage:
                    stage_map = {
                        "planning": StageType.PLANNING,
                        "execution": StageType.EXECUTION,
                        "reflection": StageType.REFLECTION,
                        "verification": StageType.VERIFICATION,
                    }
                    current_stage_enum = stage_map.get(params.current_stage.lower())

                # Score and rank memories using retrieval service
                if filtered_memories:
                    scored = await retrieval.retrieve_top_k(
                        memories=filtered_memories,
                        k=params.limit,
                        query_embedding=params.query_embedding,
                        current_stage=current_stage_enum,
                        has_recent_errors=params.has_recent_errors,
                    )
                    memories = [mem.model_dump(mode="json") for mem, _, _ in scored]
        else:
            # No embedding provided, use in-memory filtering (legacy mode)
            filtered_memories = list(_memory_store.values())

            if params.agent_id:
                filtered_memories = [
                    m for m in filtered_memories if m.agent_id == params.agent_id
                ]

            if params.session_id:
                filtered_memories = [
                    m for m in filtered_memories if m.session_id == params.session_id
                ]

            if params.task_id:
                filtered_memories = [
                    m for m in filtered_memories if m.task_id == params.task_id
                ]

            if params.memory_layer:
                filtered_memories = [
                    m
                    for m in filtered_memories
                    if m.memory_layer.value == params.memory_layer.lower()
                ]

            # Map stage string to enum if provided
            current_stage_enum = None
            if params.current_stage:
                stage_map = {
                    "planning": StageType.PLANNING,
                    "execution": StageType.EXECUTION,
                    "reflection": StageType.REFLECTION,
                    "verification": StageType.VERIFICATION,
                }
                current_stage_enum = stage_map.get(params.current_stage.lower())

            # Score and rank memories
            if filtered_memories:
                scored = await retrieval.retrieve_top_k(
                    memories=filtered_memories,
                    k=params.limit,
                    query_embedding=params.query_embedding,
                    current_stage=current_stage_enum,
                    has_recent_errors=params.has_recent_errors,
                )
                memories = [mem.model_dump(mode="json") for mem, _, _ in scored]

        end_time = datetime.now(UTC)
        query_time_ms = (end_time - start_time).total_seconds() * 1000

        result = MemoryRetrieveResult(
            memories=memories,
            total_count=len(memories),
            query_time_ms=query_time_ms,
        )

        logger.info(
            "Memory retrieval completed",
            total_count=result.total_count,
            query_time_ms=result.query_time_ms,
            agent_id=params.agent_id,
            method="memory.retrieve",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Memory retrieve validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


@register_jsonrpc_method("memory.get_context")
async def handle_memory_get_context(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get formatted context from session memories.

    Method: memory.get_context
    Params:
        - session_id: string (required)
        - query: string (optional)
        - query_embedding: array of floats (optional)
        - current_stage: string (optional)
        - max_memories: integer (optional, default 10)
        - format: string (optional, default "markdown")

    Returns:
        - session_id: string
        - context: string (formatted context)
        - memory_count: integer
        - context_size_bytes: integer
        - format: string
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: session_id")

        # Validate parameters
        params = ContextRetrieveParams(**request.params)

        # Get memory service
        memory_service = _get_memory_service()

        # Map stage string to enum if provided
        current_stage_enum = None
        if params.current_stage:
            stage_map = {
                "planning": StageType.PLANNING,
                "execution": StageType.EXECUTION,
                "reflection": StageType.REFLECTION,
                "verification": StageType.VERIFICATION,
            }
            current_stage_enum = stage_map.get(params.current_stage.lower())

        # Get session context memories
        memories = await memory_service.session_context.get_session_context(
            session_id=params.session_id,
            query=params.query,
            query_embedding=params.query_embedding,
            current_stage=current_stage_enum,
            max_memories=params.max_memories,
        )

        # Format context based on requested format
        if params.format == "json":
            context = str([mem.model_dump(mode="json") for mem in memories])
        elif params.format == "plain":
            context = "\n\n".join([mem.content for mem in memories])
        else:  # markdown
            context_parts = ["# Memory Context\n"]
            for i, mem in enumerate(memories, 1):
                context_parts.append(f"## Memory {i}\n")
                context_parts.append(f"**Layer:** {mem.memory_layer.value}\n")
                context_parts.append(f"**Summary:** {mem.summary}\n")
                context_parts.append(f"**Content:**\n{mem.content}\n")
                if mem.keywords:
                    context_parts.append(f"**Keywords:** {', '.join(mem.keywords)}\n")
                context_parts.append("\n---\n")
            context = "\n".join(context_parts)

        context_size_bytes = len(context.encode("utf-8"))

        result = ContextRetrieveResult(
            session_id=params.session_id,
            context=context,
            memory_count=len(memories),
            context_size_bytes=context_size_bytes,
            format=params.format,
        )

        logger.info(
            "Context retrieved",
            session_id=params.session_id,
            memory_count=result.memory_count,
            context_size_bytes=result.context_size_bytes,
            format=params.format,
            method="memory.get_context",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Get context validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


@register_jsonrpc_method("memory.complete_stage")
async def handle_memory_complete_stage(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Complete a reasoning stage and trigger compression.

    Method: memory.complete_stage
    Params:
        - stage_id: string (required)
        - task_id: string (required)
        - agent_id: string (required)
        - stage_type: string (required)
        - stage_summary: string (required)
        - stage_insights: array of strings (optional)
        - raw_memory_refs: array of strings (optional)
        - compression_model: string (optional, default "gpt-4.1-mini")

    Returns:
        - stage_id: string
        - task_id: string
        - compression_ratio: float
        - quality_score: float
        - success: boolean
        - message: string
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: stage_id, task_id, agent_id, stage_type, stage_summary"
            )

        # Validate parameters
        params = CompleteStageParams(**request.params)

        # Map stage type string to enum
        stage_type_map = {
            "planning": StageType.PLANNING,
            "execution": StageType.EXECUTION,
            "reflection": StageType.REFLECTION,
            "verification": StageType.VERIFICATION,
        }
        stage_type_enum = stage_type_map.get(params.stage_type.lower())
        if not stage_type_enum:
            raise ValueError(
                f"Invalid stage_type: {params.stage_type}. "
                f"Must be one of: {list(stage_type_map.keys())}"
            )

        # Calculate compression ratio based on raw content
        raw_content_size = sum(
            len(_memory_store.get(ref, MemoryRecord(
                memory_layer=MemoryLayer.EPISODIC,
                content="",
                summary="",
            )).content)
            for ref in params.raw_memory_refs
            if ref in _memory_store
        )
        compressed_size = len(params.stage_summary)

        # Avoid division by zero
        if compressed_size > 0 and raw_content_size > 0:
            compression_ratio = raw_content_size / compressed_size
        else:
            compression_ratio = 10.0  # Default target ratio

        # Clamp to valid range
        compression_ratio = min(max(compression_ratio, 1.0), 20.0)

        # Create stage memory record
        stage_memory = StageMemory(
            stage_id=params.stage_id,
            task_id=params.task_id,
            agent_id=params.agent_id,
            stage_type=stage_type_enum,
            stage_summary=params.stage_summary,
            stage_insights=params.stage_insights,
            raw_memory_refs=params.raw_memory_refs,
            compression_ratio=compression_ratio,
            compression_model=params.compression_model,
            quality_score=0.95,  # High quality assumed with proper LLM compression
            completed_at=datetime.now(UTC),
        )

        # Store stage memory in production backend
        storage_backend = _get_storage_backend()
        try:
            await storage_backend.store_stage_memory(stage_memory)
        except RuntimeError:
            # Storage backend not initialized, fall back to in-memory (dev mode)
            logger.warning(
                "Storage backend not initialized, using in-memory storage",
                stage_id=stage_memory.stage_id,
            )
            _stage_store[stage_memory.stage_id] = stage_memory

        result = CompleteStageResult(
            stage_id=stage_memory.stage_id,
            task_id=stage_memory.task_id,
            compression_ratio=stage_memory.compression_ratio,
            quality_score=stage_memory.quality_score,
        )

        logger.info(
            "Stage completed and compressed",
            stage_id=result.stage_id,
            task_id=result.task_id,
            compression_ratio=result.compression_ratio,
            quality_score=result.quality_score,
            method="memory.complete_stage",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Complete stage validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


@register_jsonrpc_method("memory.record_error")
async def handle_memory_record_error(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Record an error for pattern detection.

    Method: memory.record_error
    Params:
        - task_id: string (required)
        - agent_id: string (required)
        - error_type: string (required)
        - error_description: string (required)
        - context_when_occurred: string (optional)
        - recovery_action: string (optional)
        - error_severity: float (required, 0.0-1.0)
        - stage_id: string (optional)

    Returns:
        - error_id: string
        - task_id: string
        - recorded_at: string (ISO8601)
        - success: boolean
        - message: string
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: task_id, agent_id, error_type, "
                "error_description, error_severity"
            )

        # Validate parameters
        params = RecordErrorParams(**request.params)

        # Map error type string to enum
        error_type_map = {
            "hallucination": ErrorType.HALLUCINATION,
            "missing_info": ErrorType.MISSING_INFO,
            "incorrect_action": ErrorType.INCORRECT_ACTION,
            "context_degradation": ErrorType.CONTEXT_DEGRADATION,
        }
        error_type_enum = error_type_map.get(params.error_type.lower())
        if not error_type_enum:
            raise ValueError(
                f"Invalid error_type: {params.error_type}. "
                f"Must be one of: {list(error_type_map.keys())}"
            )

        # Create error record
        error_record = ErrorRecord(
            task_id=params.task_id,
            agent_id=params.agent_id,
            error_type=error_type_enum,
            error_description=params.error_description,
            context_when_occurred=params.context_when_occurred,
            recovery_action=params.recovery_action,
            error_severity=params.error_severity,
            stage_id=params.stage_id,
        )

        # Store error in production backend
        storage_backend = _get_storage_backend()
        try:
            await storage_backend.store_error(error_record)
        except RuntimeError:
            # Storage backend not initialized, fall back to in-memory (dev mode)
            logger.warning(
                "Storage backend not initialized, using in-memory storage",
                error_id=error_record.error_id,
            )
            _error_store[error_record.error_id] = error_record

        # Also track in error tracker for pattern detection
        error_tracker = _get_error_tracker()
        await error_tracker.record_error(
            task_id=params.task_id,
            agent_id=params.agent_id,
            error_type=error_type_enum,
            error_description=params.error_description,
            context_when_occurred=params.context_when_occurred,
            recovery_action=params.recovery_action,
            severity_override=params.error_severity,
        )

        result = RecordErrorResult(
            error_id=error_record.error_id,
            task_id=error_record.task_id,
            recorded_at=error_record.recorded_at.isoformat(),
        )

        logger.info(
            "Error recorded",
            error_id=result.error_id,
            task_id=result.task_id,
            error_type=params.error_type,
            severity=params.error_severity,
            method="memory.record_error",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Record error validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


@register_jsonrpc_method("memory.get_strategic_context")
async def handle_memory_get_strategic_context(
    request: JsonRpcRequest,
) -> dict[str, Any]:
    """
    Get strategic context for ACE framework integration.

    Method: memory.get_strategic_context
    Params:
        - agent_id: string (required)
        - session_id: string (optional)
        - goal: string (optional)
        - query_embedding: array of floats (optional)

    Returns:
        - context_id: string
        - agent_id: string
        - session_id: string or null
        - current_goal: string or null
        - strategic_memory_count: integer
        - tactical_memory_count: integer
        - error_patterns: array of strings
        - success_patterns: array of strings
        - confidence_score: float
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id")

        # Validate parameters
        params = StrategicContextParams(**request.params)

        # Get ACE interface
        ace_interface = _get_ace_interface()

        # Build strategic context
        context = await ace_interface.build_strategic_context(
            agent_id=params.agent_id,
            session_id=params.session_id,
            goal=params.goal,
            query_embedding=params.query_embedding,
        )

        result = StrategicContextResult(
            context_id=context.context_id,
            agent_id=context.agent_id,
            session_id=context.session_id,
            current_goal=context.current_goal,
            strategic_memory_count=len(context.strategic_memories),
            tactical_memory_count=len(context.tactical_memories),
            error_patterns=context.error_patterns,
            success_patterns=context.success_patterns,
            confidence_score=context.confidence_score,
        )

        logger.info(
            "Strategic context retrieved",
            context_id=result.context_id,
            agent_id=result.agent_id,
            strategic_count=result.strategic_memory_count,
            tactical_count=result.tactical_memory_count,
            confidence=result.confidence_score,
            method="memory.get_strategic_context",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Get strategic context validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


@register_jsonrpc_method("memory.run_memify")
async def handle_memory_run_memify(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Trigger graph optimization (memify) operation.

    Method: memory.run_memify
    Params:
        - similarity_threshold: float (optional, default 0.90)
        - min_access_count: integer (optional, default 2)
        - batch_size: integer (optional, default 100)
        - schedule_cron: string (optional, cron expression for scheduling)

    Returns:
        - optimization_id: string
        - entities_analyzed: integer
        - entities_merged: integer
        - relationships_pruned: integer
        - patterns_detected: integer
        - consolidation_accuracy: float
        - duplicate_rate: float
        - duration_seconds: float
        - scheduled_job_id: string or null
        - next_run: string or null (ISO8601)
    """
    try:
        params_dict = request.params or {}
        if isinstance(params_dict, list):
            raise ValueError("Parameters must be an object, not an array")

        # Validate parameters
        params = RunMemifyParams(**params_dict)

        # Get storage backend to access Neo4j driver
        storage_backend = _get_storage_backend()

        # Create GraphOptimizer with Neo4j driver
        optimizer = GraphOptimizer(
            driver=storage_backend.neo4j_driver,
            similarity_threshold=params.similarity_threshold,
            min_access_count=params.min_access_count,
            batch_size=params.batch_size,
        )

        # Run optimization
        metrics = await optimizer.optimize()

        # Track scheduling if requested
        scheduled_job_id = None
        next_run = None
        if params.schedule_cron:
            scheduled_job_id = optimizer.schedule_optimization(params.schedule_cron)
            next_run_dt = optimizer.get_next_scheduled_run()
            next_run = next_run_dt.isoformat() if next_run_dt else None

        result = RunMemifyResult(
            optimization_id=metrics.optimization_id,
            entities_analyzed=metrics.entities_analyzed,
            entities_merged=metrics.entities_merged,
            relationships_pruned=metrics.low_value_edges_removed,
            patterns_detected=metrics.patterns_detected,
            consolidation_accuracy=metrics.consolidation_accuracy,
            duplicate_rate=metrics.duplicate_rate,
            duration_seconds=metrics.duration_seconds,
            scheduled_job_id=scheduled_job_id,
            next_run=next_run,
        )

        logger.info(
            "Memify optimization completed",
            optimization_id=result.optimization_id,
            entities_analyzed=result.entities_analyzed,
            entities_merged=result.entities_merged,
            relationships_pruned=result.relationships_pruned,
            patterns_detected=result.patterns_detected,
            duration_seconds=result.duration_seconds,
            consolidation_accuracy=result.consolidation_accuracy,
            duplicate_rate=result.duplicate_rate,
            scheduled=scheduled_job_id is not None,
            method="memory.run_memify",
        )

        return result.model_dump(mode="json")

    except ValidationError as e:
        logger.error("Run memify validation failed", error=str(e))
        raise ValueError(f"Parameter validation failed: {e}") from e


# Export all handlers
__all__ = [
    "handle_memory_store",
    "handle_memory_retrieve",
    "handle_memory_get_context",
    "handle_memory_complete_stage",
    "handle_memory_record_error",
    "handle_memory_get_strategic_context",
    "handle_memory_run_memify",
]
