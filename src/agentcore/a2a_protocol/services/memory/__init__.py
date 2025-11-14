"""
Memory Service Components

COMPASS-based memory system with hierarchical stage management,
hybrid storage (Qdrant + Neo4j + PostgreSQL), and progressive compression.
"""

from agentcore.a2a_protocol.services.memory.ecl_pipeline import (
    ECLTask,
    Pipeline,
    TaskRegistry,
    TaskResult,
    TaskStatus,
    task_registry,
)
from agentcore.a2a_protocol.services.memory.entity_extractor import EntityExtractor
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.hybrid_search import (
    HybridSearchConfig,
    HybridSearchMetadata,
    HybridSearchService,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
    RetrievalConfig,
    ScoringBreakdown,
)
from agentcore.a2a_protocol.services.memory.stage_detector import (
    StageDetector,
    StageTransitionHandler,
)
from agentcore.a2a_protocol.services.memory.stage_manager import (
    CompressionTrigger,
    StageManager,
)

# Register entity extractor with global task registry
task_registry.register(EntityExtractor)

__all__ = [
    "StageManager",
    "CompressionTrigger",
    "StageDetector",
    "StageTransitionHandler",
    "ECLTask",
    "Pipeline",
    "TaskRegistry",
    "TaskResult",
    "TaskStatus",
    "task_registry",
    "EntityExtractor",
    "GraphMemoryService",
    "HybridSearchService",
    "HybridSearchConfig",
    "HybridSearchMetadata",
    "EnhancedRetrievalService",
    "RetrievalConfig",
    "ScoringBreakdown",
]
