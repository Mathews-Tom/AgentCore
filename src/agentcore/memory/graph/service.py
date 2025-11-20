"""GraphMemoryService - Neo4j Integration for Memory System.

Provides graph database operations for storing and querying:
- Memory nodes (episodic, semantic, procedural memories)
- Entity nodes (people, concepts, tools, constraints)
- Concept nodes (high-level semantic concepts)
- Relationships (MENTIONS, RELATES_TO, PART_OF, FOLLOWS, PRECEDES, TRIGGERS)

Implements MEM-017 acceptance criteria:
- Store Memory, Entity, Concept nodes
- Create MENTIONS, RELATES_TO, PART_OF relationships
- Support temporal relationships (FOLLOWS, PRECEDES)
- Index entities by type and properties
- Traverse graph with depth 1-3 multi-hop queries
- <200ms graph traversal (p95, 2-hop)
- Async Neo4j driver integration

References:
    - MEM-017: GraphMemoryService (Neo4j Integration)
    - schema.cypher: Neo4j graph schema
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from neo4j import AsyncSession
from neo4j.exceptions import Neo4jError

from agentcore.memory.graph.connection import get_session

logger = structlog.get_logger(__name__)


class MemoryLayer(str, Enum):
    """Memory layer classification."""

    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures


class MemoryStage(str, Enum):
    """Memory stage classification (COMPASS stages)."""

    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    VERIFICATION = "verification"


class EntityType(str, Enum):
    """Entity type classification."""

    PERSON = "person"
    CONCEPT = "concept"
    TOOL = "tool"
    CONSTRAINT = "constraint"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Entity relationship types."""

    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    CAUSES = "causes"
    ENABLES = "enables"


class GraphMemoryService:
    """Neo4j graph database service for memory system.

    Provides async operations for storing and querying memory graph:
    - Node creation (Memory, Entity, Concept)
    - Relationship creation (MENTIONS, RELATES_TO, etc.)
    - Graph traversal queries (1-hop, 2-hop, 3-hop)
    - Temporal queries (FOLLOWS, PRECEDES chains)

    Example:
        ```python
        service = GraphMemoryService()

        # Store memory node
        memory_id = await service.create_memory_node(
            agent_id="agent-123",
            session_id="session-456",
            layer=MemoryLayer.EPISODIC,
            stage=MemoryStage.EXECUTION,
            content="User requested feature implementation",
            criticality=0.8
        )

        # Store entity and link to memory
        entity_id = await service.create_entity_node(
            name="feature implementation",
            entity_type=EntityType.CONCEPT,
            confidence=0.9
        )

        await service.create_mentions_relationship(
            memory_id=memory_id,
            entity_id=entity_id,
            position=0,
            context="User requested feature implementation"
        )

        # Query related entities
        entities = await service.get_related_entities(memory_id, max_depth=2)
        ```

    Attributes:
        None (uses global Neo4j driver via connection module)
    """

    async def create_memory_node(
        self,
        agent_id: str,
        session_id: str,
        layer: MemoryLayer,
        stage: MemoryStage,
        content: str,
        criticality: float,
        memory_id: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Create Memory node in graph.

        Args:
            agent_id: UUID of agent that created the memory
            session_id: UUID of session context
            layer: Memory layer (episodic, semantic, procedural)
            stage: Memory stage (planning, execution, reflection, verification)
            content: Memory content text
            criticality: Importance score 0.0-1.0
            memory_id: Optional UUID (generated if not provided)
            created_at: Optional creation timestamp (defaults to now)

        Returns:
            str: Memory node UUID

        Raises:
            ValueError: If criticality out of range or content empty
            Neo4jError: If database operation fails
        """
        if not 0.0 <= criticality <= 1.0:
            raise ValueError(f"Criticality must be in range [0.0, 1.0], got {criticality}")

        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")

        memory_id = memory_id or str(uuid4())
        created_at = created_at or datetime.now(UTC)

        query = """
        CREATE (m:Memory {
            memory_id: $memory_id,
            agent_id: $agent_id,
            session_id: $session_id,
            layer: $layer,
            stage: $stage,
            content: $content,
            criticality: $criticality,
            created_at: datetime($created_at),
            accessed_count: 0
        })
        RETURN m.memory_id AS memory_id
        """

        try:
            async with get_session() as session:
                result = await session.run(
                    query,
                    memory_id=memory_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    layer=layer.value,
                    stage=stage.value,
                    content=content,
                    criticality=criticality,
                    created_at=created_at.isoformat(),
                )
                record = await result.single()

                logger.info(
                    "Created Memory node",
                    memory_id=memory_id,
                    agent_id=agent_id,
                    layer=layer.value,
                    stage=stage.value,
                )

                return record["memory_id"]

        except Neo4jError as e:
            logger.error("Failed to create Memory node", error=str(e), memory_id=memory_id)
            raise

    async def create_entity_node(
        self,
        name: str,
        entity_type: EntityType,
        confidence: float,
        entity_id: str | None = None,
        properties: dict[str, Any] | None = None,
        first_seen: datetime | None = None,
        last_seen: datetime | None = None,
    ) -> str:
        """Create or update Entity node in graph.

        Uses MERGE to deduplicate entities with same name and type.
        Updates mention_count, last_seen, and confidence on match.

        Args:
            name: Entity name (normalized)
            entity_type: Entity classification
            confidence: Extraction confidence 0.0-1.0
            entity_id: Optional UUID (generated if not provided)
            properties: Optional additional metadata
            first_seen: Optional first seen timestamp (defaults to now)
            last_seen: Optional last seen timestamp (defaults to now)

        Returns:
            str: Entity node UUID

        Raises:
            ValueError: If confidence out of range or name empty
            Neo4jError: If database operation fails
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {confidence}")

        if not name or not name.strip():
            raise ValueError("Entity name cannot be empty")

        entity_id = entity_id or str(uuid4())
        properties = properties or {}
        now = datetime.now(UTC)
        first_seen = first_seen or now
        last_seen = last_seen or now

        query = """
        MERGE (e:Entity {name: $name, entity_type: $entity_type})
        ON CREATE SET
            e.entity_id = $entity_id,
            e.confidence = $confidence,
            e.properties = $properties,
            e.first_seen = datetime($first_seen),
            e.last_seen = datetime($last_seen),
            e.mention_count = 1
        ON MATCH SET
            e.last_seen = datetime($last_seen),
            e.mention_count = e.mention_count + 1,
            e.confidence = CASE
                WHEN $confidence > e.confidence THEN $confidence
                ELSE e.confidence
            END
        RETURN e.entity_id AS entity_id
        """

        try:
            async with get_session() as session:
                result = await session.run(
                    query,
                    entity_id=entity_id,
                    name=name,
                    entity_type=entity_type.value,
                    confidence=confidence,
                    properties=properties,
                    first_seen=first_seen.isoformat(),
                    last_seen=last_seen.isoformat(),
                )
                record = await result.single()

                logger.info(
                    "Created/updated Entity node",
                    entity_id=record["entity_id"],
                    name=name,
                    entity_type=entity_type.value,
                )

                return record["entity_id"]

        except Neo4jError as e:
            logger.error("Failed to create Entity node", error=str(e), name=name)
            raise

    async def create_concept_node(
        self,
        name: str,
        description: str,
        category: str,
        concept_id: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Create Concept node in graph.

        Args:
            name: Concept name
            description: Concept description
            category: Concept category
            concept_id: Optional UUID (generated if not provided)
            created_at: Optional creation timestamp (defaults to now)

        Returns:
            str: Concept node UUID

        Raises:
            ValueError: If name or description empty
            Neo4jError: If database operation fails
        """
        if not name or not name.strip():
            raise ValueError("Concept name cannot be empty")

        if not description or not description.strip():
            raise ValueError("Concept description cannot be empty")

        concept_id = concept_id or str(uuid4())
        created_at = created_at or datetime.now(UTC)

        query = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            name: $name,
            description: $description,
            category: $category,
            created_at: datetime($created_at),
            usage_count: 0
        })
        RETURN c.concept_id AS concept_id
        """

        try:
            async with get_session() as session:
                result = await session.run(
                    query,
                    concept_id=concept_id,
                    name=name,
                    description=description,
                    category=category,
                    created_at=created_at.isoformat(),
                )
                record = await result.single()

                logger.info(
                    "Created Concept node",
                    concept_id=concept_id,
                    name=name,
                    category=category,
                )

                return record["concept_id"]

        except Neo4jError as e:
            logger.error("Failed to create Concept node", error=str(e), name=name)
            raise

    async def create_mentions_relationship(
        self,
        memory_id: str,
        entity_id: str,
        position: int,
        context: str,
        sentiment: float | None = None,
    ) -> None:
        """Create MENTIONS relationship between Memory and Entity.

        Args:
            memory_id: Memory node UUID
            entity_id: Entity node UUID
            position: Position in text (character offset)
            context: Surrounding context snippet
            sentiment: Optional sentiment score -1.0 to 1.0

        Raises:
            ValueError: If sentiment out of range
            Neo4jError: If database operation fails
        """
        if sentiment is not None and not -1.0 <= sentiment <= 1.0:
            raise ValueError(f"Sentiment must be in range [-1.0, 1.0], got {sentiment}")

        query = """
        MATCH (m:Memory {memory_id: $memory_id})
        MATCH (e:Entity {entity_id: $entity_id})
        CREATE (m)-[r:MENTIONS {
            position: $position,
            context: $context,
            sentiment: $sentiment,
            created_at: datetime()
        }]->(e)
        """

        try:
            async with get_session() as session:
                await session.run(
                    query,
                    memory_id=memory_id,
                    entity_id=entity_id,
                    position=position,
                    context=context,
                    sentiment=sentiment,
                )

                logger.debug(
                    "Created MENTIONS relationship",
                    memory_id=memory_id,
                    entity_id=entity_id,
                )

        except Neo4jError as e:
            logger.error(
                "Failed to create MENTIONS relationship",
                error=str(e),
                memory_id=memory_id,
                entity_id=entity_id,
            )
            raise

    async def create_relates_to_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: RelationshipType,
        strength: float,
        confidence: float,
    ) -> None:
        """Create RELATES_TO relationship between two Entities.

        Args:
            source_entity_id: Source Entity UUID
            target_entity_id: Target Entity UUID
            relationship_type: Relationship classification
            strength: Relationship strength 0.0-1.0
            confidence: Detection confidence 0.0-1.0

        Raises:
            ValueError: If strength or confidence out of range
            Neo4jError: If database operation fails
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be in range [0.0, 1.0], got {strength}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in range [0.0, 1.0], got {confidence}")

        query = """
        MATCH (e1:Entity {entity_id: $source_entity_id})
        MATCH (e2:Entity {entity_id: $target_entity_id})
        MERGE (e1)-[r:RELATES_TO {relationship_type: $relationship_type}]->(e2)
        ON CREATE SET
            r.strength = $strength,
            r.confidence = $confidence,
            r.created_at = datetime(),
            r.last_reinforced = datetime(),
            r.reinforcement_count = 1
        ON MATCH SET
            r.last_reinforced = datetime(),
            r.reinforcement_count = r.reinforcement_count + 1,
            r.strength = (r.strength + $strength) / 2,
            r.confidence = CASE
                WHEN $confidence > r.confidence THEN $confidence
                ELSE r.confidence
            END
        """

        try:
            async with get_session() as session:
                await session.run(
                    query,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    relationship_type=relationship_type.value,
                    strength=strength,
                    confidence=confidence,
                )

                logger.debug(
                    "Created/updated RELATES_TO relationship",
                    source=source_entity_id,
                    target=target_entity_id,
                    type=relationship_type.value,
                )

        except Neo4jError as e:
            logger.error(
                "Failed to create RELATES_TO relationship",
                error=str(e),
                source=source_entity_id,
                target=target_entity_id,
            )
            raise

    async def create_part_of_relationship(
        self, entity_id: str, concept_id: str, relevance: float
    ) -> None:
        """Create PART_OF relationship between Entity and Concept.

        Args:
            entity_id: Entity node UUID
            concept_id: Concept node UUID
            relevance: Relevance score 0.0-1.0

        Raises:
            ValueError: If relevance out of range
            Neo4jError: If database operation fails
        """
        if not 0.0 <= relevance <= 1.0:
            raise ValueError(f"Relevance must be in range [0.0, 1.0], got {relevance}")

        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        MATCH (c:Concept {concept_id: $concept_id})
        CREATE (e)-[r:PART_OF {
            relevance: $relevance,
            created_at: datetime()
        }]->(c)
        """

        try:
            async with get_session() as session:
                await session.run(
                    query,
                    entity_id=entity_id,
                    concept_id=concept_id,
                    relevance=relevance,
                )

                logger.debug(
                    "Created PART_OF relationship",
                    entity_id=entity_id,
                    concept_id=concept_id,
                )

        except Neo4jError as e:
            logger.error(
                "Failed to create PART_OF relationship",
                error=str(e),
                entity_id=entity_id,
                concept_id=concept_id,
            )
            raise

    async def create_follows_relationship(
        self,
        source_memory_id: str,
        target_memory_id: str,
        time_delta: int,
        stage_transition: bool = False,
    ) -> None:
        """Create FOLLOWS temporal relationship between Memories.

        Args:
            source_memory_id: Source Memory UUID (earlier)
            target_memory_id: Target Memory UUID (later)
            time_delta: Seconds between memories
            stage_transition: Whether transition crosses stage boundary

        Raises:
            ValueError: If time_delta negative
            Neo4jError: If database operation fails
        """
        if time_delta < 0:
            raise ValueError(f"Time delta must be non-negative, got {time_delta}")

        query = """
        MATCH (m1:Memory {memory_id: $source_memory_id})
        MATCH (m2:Memory {memory_id: $target_memory_id})
        CREATE (m1)-[r:FOLLOWS {
            time_delta: $time_delta,
            stage_transition: $stage_transition,
            created_at: datetime()
        }]->(m2)
        """

        try:
            async with get_session() as session:
                await session.run(
                    query,
                    source_memory_id=source_memory_id,
                    target_memory_id=target_memory_id,
                    time_delta=time_delta,
                    stage_transition=stage_transition,
                )

                logger.debug(
                    "Created FOLLOWS relationship",
                    source=source_memory_id,
                    target=target_memory_id,
                    time_delta=time_delta,
                )

        except Neo4jError as e:
            logger.error(
                "Failed to create FOLLOWS relationship",
                error=str(e),
                source=source_memory_id,
                target=target_memory_id,
            )
            raise

    async def create_precedes_relationship(
        self,
        source_memory_id: str,
        target_memory_id: str,
        time_delta: int,
        stage_transition: bool = False,
    ) -> None:
        """Create PRECEDES temporal relationship between Memories.

        Args:
            source_memory_id: Source Memory UUID (later)
            target_memory_id: Target Memory UUID (earlier)
            time_delta: Seconds until next memory
            stage_transition: Whether transition crosses stage boundary

        Raises:
            ValueError: If time_delta negative
            Neo4jError: If database operation fails
        """
        if time_delta < 0:
            raise ValueError(f"Time delta must be non-negative, got {time_delta}")

        query = """
        MATCH (m1:Memory {memory_id: $source_memory_id})
        MATCH (m2:Memory {memory_id: $target_memory_id})
        CREATE (m1)-[r:PRECEDES {
            time_delta: $time_delta,
            stage_transition: $stage_transition,
            created_at: datetime()
        }]->(m2)
        """

        try:
            async with get_session() as session:
                await session.run(
                    query,
                    source_memory_id=source_memory_id,
                    target_memory_id=target_memory_id,
                    time_delta=time_delta,
                    stage_transition=stage_transition,
                )

                logger.debug(
                    "Created PRECEDES relationship",
                    source=source_memory_id,
                    target=target_memory_id,
                    time_delta=time_delta,
                )

        except Neo4jError as e:
            logger.error(
                "Failed to create PRECEDES relationship",
                error=str(e),
                source=source_memory_id,
                target=target_memory_id,
            )
            raise

    async def get_related_entities(
        self, memory_id: str, max_depth: int = 2, min_confidence: float = 0.5
    ) -> list[dict[str, Any]]:
        """Get entities related to a memory via multi-hop traversal.

        Traverses MENTIONS and RELATES_TO relationships up to max_depth hops.

        Args:
            memory_id: Memory node UUID
            max_depth: Maximum traversal depth (1-3)
            min_confidence: Minimum entity confidence threshold

        Returns:
            List of entity dictionaries with metadata

        Raises:
            ValueError: If max_depth out of range
            Neo4jError: If database operation fails
        """
        if not 1 <= max_depth <= 3:
            raise ValueError(f"Max depth must be in range [1, 3], got {max_depth}")

        query = f"""
        MATCH (m:Memory {{memory_id: $memory_id}})
        MATCH path = (m)-[:MENTIONS|RELATES_TO*1..{max_depth}]-(e:Entity)
        WHERE e.confidence >= $min_confidence
        RETURN DISTINCT
            e.entity_id AS entity_id,
            e.name AS name,
            e.entity_type AS entity_type,
            e.confidence AS confidence,
            length(path) AS distance
        ORDER BY distance ASC, e.confidence DESC
        """

        try:
            async with get_session() as session:
                result = await session.run(
                    query,
                    memory_id=memory_id,
                    min_confidence=min_confidence,
                )

                entities = []
                async for record in result:
                    entities.append(
                        {
                            "entity_id": record["entity_id"],
                            "name": record["name"],
                            "entity_type": record["entity_type"],
                            "confidence": record["confidence"],
                            "distance": record["distance"],
                        }
                    )

                logger.info(
                    "Retrieved related entities",
                    memory_id=memory_id,
                    count=len(entities),
                    max_depth=max_depth,
                )

                return entities

        except Neo4jError as e:
            logger.error(
                "Failed to retrieve related entities",
                error=str(e),
                memory_id=memory_id,
            )
            raise

    async def get_temporal_chain(
        self, memory_id: str, direction: str = "forward", max_length: int = 10
    ) -> list[dict[str, Any]]:
        """Get temporal chain of memories via FOLLOWS/PRECEDES relationships.

        Args:
            memory_id: Starting Memory node UUID
            direction: Chain direction ("forward" for FOLLOWS, "backward" for PRECEDES)
            max_length: Maximum chain length

        Returns:
            List of memory dictionaries in temporal order

        Raises:
            ValueError: If direction invalid or max_length out of range
            Neo4jError: If database operation fails
        """
        if direction not in ("forward", "backward"):
            raise ValueError(f"Direction must be 'forward' or 'backward', got {direction}")

        if not 1 <= max_length <= 100:
            raise ValueError(f"Max length must be in range [1, 100], got {max_length}")

        relationship = "FOLLOWS" if direction == "forward" else "PRECEDES"

        query = f"""
        MATCH path = (m:Memory {{memory_id: $memory_id}})-[:{relationship}*1..{max_length}]->(next:Memory)
        WITH nodes(path) AS memories
        UNWIND memories AS mem
        RETURN DISTINCT
            mem.memory_id AS memory_id,
            mem.content AS content,
            mem.stage AS stage,
            mem.layer AS layer,
            mem.created_at AS created_at
        ORDER BY mem.created_at {'ASC' if direction == 'forward' else 'DESC'}
        """

        try:
            async with get_session() as session:
                result = await session.run(
                    query,
                    memory_id=memory_id,
                )

                memories = []
                async for record in result:
                    memories.append(
                        {
                            "memory_id": record["memory_id"],
                            "content": record["content"],
                            "stage": record["stage"],
                            "layer": record["layer"],
                            "created_at": record["created_at"],
                        }
                    )

                logger.info(
                    "Retrieved temporal chain",
                    memory_id=memory_id,
                    direction=direction,
                    count=len(memories),
                )

                return memories

        except Neo4jError as e:
            logger.error(
                "Failed to retrieve temporal chain",
                error=str(e),
                memory_id=memory_id,
                direction=direction,
            )
            raise

    async def get_entity_by_name(
        self, name: str, entity_type: EntityType | None = None
    ) -> dict[str, Any] | None:
        """Get entity by name and optional type.

        Args:
            name: Entity name (exact match)
            entity_type: Optional entity type filter

        Returns:
            Entity dictionary or None if not found

        Raises:
            Neo4jError: If database operation fails
        """
        type_filter = f"AND e.entity_type = $entity_type" if entity_type else ""

        query = f"""
        MATCH (e:Entity {{name: $name}})
        WHERE true {type_filter}
        RETURN
            e.entity_id AS entity_id,
            e.name AS name,
            e.entity_type AS entity_type,
            e.confidence AS confidence,
            e.properties AS properties,
            e.mention_count AS mention_count
        LIMIT 1
        """

        try:
            async with get_session() as session:
                result = await session.run(
                    query,
                    name=name,
                    entity_type=entity_type.value if entity_type else None,
                )

                record = await result.single()
                if not record:
                    return None

                return {
                    "entity_id": record["entity_id"],
                    "name": record["name"],
                    "entity_type": record["entity_type"],
                    "confidence": record["confidence"],
                    "properties": record["properties"],
                    "mention_count": record["mention_count"],
                }

        except Neo4jError as e:
            logger.error("Failed to retrieve entity", error=str(e), name=name)
            raise
