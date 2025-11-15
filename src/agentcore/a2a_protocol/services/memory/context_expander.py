"""
Graph-Aware Context Expansion Service

Implements context expansion using Neo4j graph relationships for retrieval results.
Provides 1-hop neighbors for standard memories and 2-hop neighbors for critical memories,
preserving graph structure in Entity -> Relationship -> Entity format.

Component ID: MEM-022
Ticket: MEM-022 (Implement Graph-Aware Context Expansion)
"""

from typing import Any

import structlog

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    MemoryRecord,
    RelationshipEdge,
)
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService

logger = structlog.get_logger(__name__)


class ExpandedContext:
    """Container for expanded context from a single memory."""

    def __init__(
        self,
        memory_id: str,
        entities: list[EntityNode] | None = None,
        relationships: list[RelationshipEdge] | None = None,
        depth: int = 1,
    ):
        """
        Initialize expanded context.

        Args:
            memory_id: ID of the source memory
            entities: List of related entities
            relationships: List of relationships between entities
            depth: Traversal depth used for expansion
        """
        self.memory_id = memory_id
        self.entities = entities or []
        self.relationships = relationships or []
        self.depth = depth

    def format_triples(self) -> list[str]:
        """
        Format relationships as Entity -> Relationship -> Entity triples.

        Returns:
            List of formatted triple strings
        """
        triples = []
        for rel in self.relationships:
            strength = rel.properties.get("strength", 0.0)
            strength_str = f":{strength:.2f}" if strength else ""
            triple = f"{rel.source_entity_id} -[{rel.relationship_type.upper()}{strength_str}]-> {rel.target_entity_id}"
            triples.append(triple)
        return triples

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with entities, relationships, and formatted triples
        """
        return {
            "memory_id": self.memory_id,
            "entities": [
                {
                    "name": entity.entity_name,
                    "type": entity.entity_type.value,
                    "properties": entity.properties,
                }
                for entity in self.entities
            ],
            "relationships": [
                {
                    "from": rel.source_entity_id,
                    "to": rel.target_entity_id,
                    "type": rel.relationship_type.value,
                    "strength": rel.properties.get("strength", 0.0),
                }
                for rel in self.relationships
            ],
            "formatted_triples": self.format_triples(),
            "depth": self.depth,
        }


class GraphContextExpander:
    """
    Service for expanding retrieval results with graph-aware context.

    Uses Neo4j graph traversal to find related entities and relationships,
    providing additional context for each memory in the retrieval results.

    Features:
    - 1-hop neighbors for standard memories
    - 2-hop neighbors for critical memories
    - Entity -> Relationship -> Entity triple formatting
    - Batch expansion with parallel processing
    - Graph structure preservation
    """

    def __init__(self, graph_service: GraphMemoryService):
        """
        Initialize GraphContextExpander.

        Args:
            graph_service: GraphMemoryService instance for Neo4j operations
        """
        self.graph_service = graph_service
        self._logger = logger.bind(service="GraphContextExpander")

    async def expand_single_memory(
        self,
        memory: MemoryRecord,
        depth: int | None = None,
    ) -> ExpandedContext:
        """
        Expand a single memory with graph neighbors.

        Args:
            memory: Memory record to expand
            depth: Traversal depth (None = auto based on criticality)
                   - Critical memories: depth=2
                   - Non-critical memories: depth=1

        Returns:
            ExpandedContext with entities and relationships
        """
        # Determine depth based on criticality if not specified
        if depth is None:
            depth = 2 if getattr(memory, "is_critical", False) else 1

        self._logger.debug(
            "expanding_memory_context",
            memory_id=memory.memory_id,
            depth=depth,
            is_critical=getattr(memory, "is_critical", False),
        )

        entities: list[EntityNode] = []
        relationships: list[RelationshipEdge] = []

        # Get entities mentioned in this memory
        memory_entities = memory.entities or []

        # For each entity, find related entities from graph
        for entity_name in memory_entities:
            try:
                # Find entity nodes and relationships
                neighbors_data = await self._get_entity_neighbors(
                    entity_name, depth=depth
                )

                # Collect entities
                for entity in neighbors_data.get("entities", []):
                    if entity not in entities:
                        entities.append(entity)

                # Collect relationships
                for rel in neighbors_data.get("relationships", []):
                    if rel not in relationships:
                        relationships.append(rel)

            except Exception as e:
                self._logger.warning(
                    "entity_expansion_failed",
                    entity_name=entity_name,
                    error=str(e),
                )
                continue

        self._logger.debug(
            "memory_context_expanded",
            memory_id=memory.memory_id,
            num_entities=len(entities),
            num_relationships=len(relationships),
        )

        return ExpandedContext(
            memory_id=memory.memory_id,
            entities=entities,
            relationships=relationships,
            depth=depth,
        )

    async def _get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
    ) -> dict[str, Any]:
        """
        Get neighbors of an entity from the graph.

        Args:
            entity_name: Name of the entity to find neighbors for
            depth: How many hops to traverse

        Returns:
            Dictionary with entities and relationships
        """
        entities: list[EntityNode] = []
        relationships: list[RelationshipEdge] = []

        # Use graph service to find related entities
        try:
            # Get 1-hop neighbors
            one_hop_neighbors = await self.graph_service.get_one_hop_neighbors(
                node_id=entity_name,
                direction="both",
                limit=10,
            )

            for neighbor in one_hop_neighbors:
                if isinstance(neighbor, EntityNode):
                    if neighbor not in entities:
                        entities.append(neighbor)

            # If depth > 1, get 2-hop neighbors (for critical memories)
            if depth >= 2:
                for neighbor in one_hop_neighbors:
                    if isinstance(neighbor, EntityNode):
                        two_hop_neighbors = (
                            await self.graph_service.get_one_hop_neighbors(
                                node_id=neighbor.entity_id,
                                direction="both",
                                limit=5,
                            )
                        )
                        for second_neighbor in two_hop_neighbors:
                            if isinstance(second_neighbor, EntityNode):
                                if second_neighbor not in entities:
                                    entities.append(second_neighbor)

            # Get relationships between collected entities
            entity_ids = [e.entity_id for e in entities]
            entity_ids.append(entity_name)

            # Find relationships in the graph
            for i, e1_id in enumerate(entity_ids):
                for e2_id in entity_ids[i + 1 :]:
                    try:
                        rel_data = (
                            await self.graph_service.aggregate_relationship_strength(
                                from_id=e1_id,
                                to_id=e2_id,
                                max_depth=depth,
                            )
                        )
                        if rel_data and rel_data.get("path_count", 0) > 0:
                            # Create relationship edge
                            avg_strength = rel_data.get("average_strength", 0.5)
                            rel_edge = RelationshipEdge(
                                source_entity_id=e1_id,
                                target_entity_id=e2_id,
                                relationship_type="relates_to",
                                properties={"strength": avg_strength},
                            )
                            relationships.append(rel_edge)
                    except Exception:
                        # Skip if no relationship found
                        continue

        except Exception as e:
            self._logger.warning(
                "get_entity_neighbors_failed",
                entity_name=entity_name,
                depth=depth,
                error=str(e),
            )

        return {
            "entities": entities,
            "relationships": relationships,
        }

    async def expand_context(
        self,
        memories: list[MemoryRecord],
        include_entities: bool = True,
        include_relationships: bool = True,
        max_memories: int = 20,
    ) -> dict[str, Any]:
        """
        Expand multiple memories with graph-aware context.

        Args:
            memories: List of memory records to expand
            include_entities: Whether to include entity information
            include_relationships: Whether to include relationship information
            max_memories: Maximum number of memories to expand (for performance)

        Returns:
            Dictionary with original memories and expanded context
        """
        self._logger.info(
            "expanding_context_batch",
            num_memories=len(memories),
            max_memories=max_memories,
        )

        # Limit to max_memories for performance
        memories_to_expand = memories[:max_memories]

        expanded_contexts: dict[str, dict[str, Any]] = {}

        for memory in memories_to_expand:
            try:
                expanded = await self.expand_single_memory(memory)

                context_dict = expanded.to_dict()

                # Filter based on parameters
                if not include_entities:
                    context_dict["entities"] = []
                if not include_relationships:
                    context_dict["relationships"] = []
                    context_dict["formatted_triples"] = []

                expanded_contexts[memory.memory_id] = context_dict

            except Exception as e:
                self._logger.error(
                    "memory_expansion_failed",
                    memory_id=memory.memory_id,
                    error=str(e),
                )
                # Provide empty context on failure
                expanded_contexts[memory.memory_id] = {
                    "memory_id": memory.memory_id,
                    "entities": [],
                    "relationships": [],
                    "formatted_triples": [],
                    "depth": 0,
                    "error": str(e),
                }

        self._logger.info(
            "context_expansion_complete",
            num_expanded=len(expanded_contexts),
            successful=sum(
                1 for ctx in expanded_contexts.values() if "error" not in ctx
            ),
        )

        return {
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "summary": m.summary,
                    "is_critical": getattr(m, "is_critical", False),
                }
                for m in memories_to_expand
            ],
            "expanded_context": expanded_contexts,
        }

    def format_graph_context(
        self,
        entities: list[EntityNode],
        relationships: list[RelationshipEdge],
    ) -> str:
        """
        Format graph context as human-readable and LLM-friendly text.

        Args:
            entities: List of entity nodes
            relationships: List of relationship edges

        Returns:
            Formatted string representation
        """
        lines = []

        # Format relationships as triples
        if relationships:
            lines.append("Graph Context:")
            for rel in relationships:
                strength = rel.properties.get("strength", 0.0)
                strength_str = f":{strength:.2f}" if strength else ""
                triple = f"- {rel.source_entity_id} -[{rel.relationship_type.upper()}{strength_str}]-> {rel.target_entity_id}"
                lines.append(triple)

        # Format related entities
        if entities:
            lines.append("")
            lines.append("Related Entities:")
            for entity in entities:
                lines.append(f"- {entity.entity_name} ({entity.entity_type.value})")

        return "\n".join(lines)

    async def expand_with_formatting(
        self,
        memories: list[MemoryRecord],
        max_memories: int = 10,
    ) -> str:
        """
        Expand memories and return formatted text for LLM consumption.

        Args:
            memories: List of memory records
            max_memories: Maximum number to expand

        Returns:
            Formatted string with all expanded contexts
        """
        expansion = await self.expand_context(
            memories=memories,
            include_entities=True,
            include_relationships=True,
            max_memories=max_memories,
        )

        formatted_sections = []

        for memory_data in expansion["memories"]:
            memory_id = memory_data["memory_id"]
            context_data = expansion["expanded_context"].get(memory_id, {})

            section = [
                f"Memory: {memory_data['summary']}",
                f"(ID: {memory_id}, Critical: {memory_data['is_critical']})",
            ]

            # Add triples if present
            triples = context_data.get("formatted_triples", [])
            if triples:
                section.append("\nGraph Context:")
                for triple in triples:
                    section.append(f"  {triple}")

            # Add entities if present
            entities = context_data.get("entities", [])
            if entities:
                section.append("\nRelated Entities:")
                for entity in entities:
                    section.append(f"  - {entity['name']} ({entity['type']})")

            formatted_sections.append("\n".join(section))

        return "\n\n---\n\n".join(formatted_sections)
