"""Test helper for ECL Pipeline integration tests.

Provides a simplified ECLPipeline wrapper for integration testing.
"""

from __future__ import annotations

from typing import Any

from agentcore.a2a_protocol.models.memory import MemoryRecord
from agentcore.a2a_protocol.services.memory.entity_extractor import EntityExtractor
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.relationship_detector import (
    RelationshipDetector,
)
from agentcore.a2a_protocol.services.memory.storage_backend import VectorStorageBackend


class ECLPipeline:
    """Simplified ECL Pipeline for integration testing.

    Coordinates Extract, Contextualize, and Load phases for memory processing.
    """

    def __init__(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
        entity_extractor: EntityExtractor,
        relationship_detector: RelationshipDetector,
    ):
        """Initialize ECL Pipeline.

        Args:
            vector_backend: Vector storage backend (Qdrant)
            graph_service: Graph service (Neo4j)
            entity_extractor: Entity extraction service
            relationship_detector: Relationship detection service
        """
        self.vector_backend = vector_backend
        self.graph_service = graph_service
        self.entity_extractor = entity_extractor
        self.relationship_detector = relationship_detector

    async def process(
        self,
        conversation_data: list[dict[str, str]],
        agent_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Process conversation through ECL pipeline.

        Args:
            conversation_data: List of conversation messages
            agent_id: Agent ID
            session_id: Session ID

        Returns:
            dict: Pipeline execution results
        """
        # Extract phase: Create memories from conversation
        memories = await self.entity_extractor.extract_memories(
            conversation=conversation_data,
            agent_id=agent_id,
            session_id=session_id,
        )

        # Contextualize phase: Extract entities and relationships
        entities = []
        relationships = []

        for memory in memories:
            # Extract entities from each memory
            memory_entities = await self.entity_extractor.extract_entities(
                content=memory.content
            )
            entities.extend(memory_entities)

            # Detect relationships
            memory_relationships = (
                await self.relationship_detector.detect_relationships(
                    content=memory.content,
                    entities=memory_entities,
                )
            )
            relationships.extend(memory_relationships)

        # Link phase: Create connections in graph
        relationships_created = 0
        for entity in entities:
            await self.graph_service.store_entity(entity)

        for relationship in relationships:
            await self.graph_service.store_relationship(relationship)
            relationships_created += 1

        # Load phase: Store in both backends
        for memory in memories:
            await self.vector_backend.store_memory(memory)
            await self.graph_service.store_memory_node(memory)

        return {
            "memories_created": len(memories),
            "memory_records": memories,
            "extracted_entities": entities,
            "extracted_relationships": relationships,
            "relationships_created": relationships_created,
        }
