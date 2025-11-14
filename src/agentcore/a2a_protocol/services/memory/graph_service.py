"""
GraphMemoryService for Neo4j Integration

Implements high-level graph operations for storing entities and relationships
in Neo4j knowledge graph. Provides memory node storage, relationship creation,
graph traversal, and temporal relationship tracking.

Component ID: MEM-017
Ticket: MEM-017 (Implement GraphMemoryService - Neo4j Integration)
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog
from neo4j import AsyncDriver, AsyncSession

from agentcore.a2a_protocol.database.graph_repository import GraphRepository
from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    MemoryRecord,
    RelationshipEdge,
    RelationshipType,
)

logger = structlog.get_logger(__name__)


class GraphMemoryService:
    """
    High-level service for Neo4j graph memory operations.

    Provides methods for:
    - Storing Memory, Entity, and Concept nodes
    - Creating relationships (MENTIONS, RELATES_TO, PART_OF, FOLLOWS, PRECEDES)
    - Graph traversal with depth limits (1-3 hops)
    - Temporal relationship tracking
    - Entity indexing and search

    Performance targets:
    - <200ms graph traversal (p95, 2-hop)
    - Async operations throughout
    - Optimized Cypher queries with indexes
    """

    def __init__(self, driver: AsyncDriver):
        """
        Initialize GraphMemoryService.

        Args:
            driver: Neo4j async driver instance
        """
        self.driver = driver
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize graph indexes and constraints.

        Creates indexes on:
        - Entity.entity_type
        - Entity.entity_name
        - Memory.task_id
        - Memory.stage_id
        - Concept.name

        Performance optimization indexes:
        - Composite index on (entity_type, entity_name) for filtered searches
        - Range index on relationship access_count for strength aggregation
        - Full-text index on entity_name for similarity queries

        Also creates uniqueness constraints for entity_id.
        """
        async with self.driver.session() as session:
            # Create uniqueness constraint for entity_id
            await session.run(
                """
                CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
                FOR (e:Entity)
                REQUIRE e.entity_id IS UNIQUE
                """
            )

            # Create indexes for entity type and name
            await session.run(
                """
                CREATE INDEX entity_type_idx IF NOT EXISTS
                FOR (e:Entity)
                ON (e.entity_type)
                """
            )

            await session.run(
                """
                CREATE INDEX entity_name_idx IF NOT EXISTS
                FOR (e:Entity)
                ON (e.entity_name)
                """
            )

            # Create composite index for entity type + name (query optimization)
            await session.run(
                """
                CREATE INDEX entity_type_name_composite_idx IF NOT EXISTS
                FOR (e:Entity)
                ON (e.entity_type, e.entity_name)
                """
            )

            # Create Memory node indexes
            await session.run(
                """
                CREATE INDEX memory_task_idx IF NOT EXISTS
                FOR (m:Memory)
                ON (m.task_id)
                """
            )

            await session.run(
                """
                CREATE INDEX memory_stage_idx IF NOT EXISTS
                FOR (m:Memory)
                ON (m.stage_id)
                """
            )

            # Create Concept node indexes
            await session.run(
                """
                CREATE INDEX concept_name_idx IF NOT EXISTS
                FOR (c:Concept)
                ON (c.name)
                """
            )

            # Performance optimization: Index on created_at for temporal queries
            await session.run(
                """
                CREATE INDEX entity_created_at_idx IF NOT EXISTS
                FOR (e:Entity)
                ON (e.created_at)
                """
            )

            self._initialized = True
            logger.info(
                "GraphMemoryService initialized with indexes and constraints",
                indexes=7,
                constraints=1,
            )

    async def store_memory_node(
        self,
        memory: MemoryRecord,
    ) -> str:
        """
        Store memory as a node in Neo4j graph.

        Creates a Memory node with properties from MemoryRecord.
        Links to task and stage via properties for later querying.

        Args:
            memory: Memory record to store

        Returns:
            Memory node ID

        Example:
            memory_id = await service.store_memory_node(memory_record)
        """
        async with self.driver.session() as session:
            query = """
            CREATE (m:Memory {
                memory_id: $memory_id,
                memory_layer: $memory_layer,
                content: $content,
                summary: $summary,
                agent_id: $agent_id,
                session_id: $session_id,
                task_id: $task_id,
                stage_id: $stage_id,
                timestamp: datetime($timestamp),
                is_critical: $is_critical,
                relevance_score: $relevance_score
            })
            RETURN m.memory_id as memory_id
            """

            params = {
                "memory_id": memory.memory_id,
                "memory_layer": memory.memory_layer.value,
                "content": memory.content,
                "summary": memory.summary,
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
                "task_id": memory.task_id,
                "stage_id": memory.stage_id,
                "timestamp": memory.timestamp.isoformat(),
                "is_critical": memory.is_critical,
                "relevance_score": memory.relevance_score,
            }

            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.info(
                    "Stored memory node",
                    memory_id=memory.memory_id,
                    task_id=memory.task_id,
                    is_critical=memory.is_critical,
                )
                return record["memory_id"]

            raise RuntimeError(f"Failed to store memory node: {memory.memory_id}")

    async def store_entity_node(self, entity: EntityNode) -> str:
        """
        Store entity node in Neo4j graph.

        Creates Entity node using GraphRepository.
        Supports entity types: person, concept, tool, constraint.

        Args:
            entity: Entity node to store

        Returns:
            Entity node ID

        Example:
            entity_id = await service.store_entity_node(entity_node)
        """
        async with self.driver.session() as session:
            result = await GraphRepository.create_node(session, entity)
            return result["entity_id"]

    async def store_concept_node(
        self,
        name: str,
        properties: dict[str, Any] | None = None,
        memory_refs: list[str] | None = None,
    ) -> str:
        """
        Store concept node in Neo4j graph.

        Concepts are high-level ideas extracted from memories.
        Different from entities - concepts represent abstract themes.

        Args:
            name: Concept name
            properties: Additional concept properties
            memory_refs: Memory IDs where concept appears

        Returns:
            Concept node ID

        Example:
            concept_id = await service.store_concept_node(
                "authentication",
                {"domain": "security"},
                ["mem-001"]
            )
        """
        concept_id = f"concept-{uuid4()}"

        async with self.driver.session() as session:
            query = """
            CREATE (c:Concept {
                concept_id: $concept_id,
                name: $name,
                properties: $properties,
                memory_refs: $memory_refs,
                created_at: datetime()
            })
            RETURN c.concept_id as concept_id
            """

            params = {
                "concept_id": concept_id,
                "name": name,
                "properties": properties or {},
                "memory_refs": memory_refs or [],
            }

            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.info("Stored concept node", concept_id=concept_id, name=name)
                return record["concept_id"]

            raise RuntimeError(f"Failed to store concept node: {name}")

    async def create_mention_relationship(
        self,
        memory_id: str,
        entity_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """
        Create MENTIONS relationship from Memory to Entity.

        Indicates that a memory mentions/references an entity.
        Used for entity-based memory retrieval.

        Args:
            memory_id: Source memory ID
            entity_id: Target entity ID
            properties: Additional relationship properties

        Returns:
            Relationship ID

        Example:
            rel_id = await service.create_mention_relationship(
                "mem-001", "ent-123"
            )
        """
        relationship_id = f"rel-{uuid4()}"

        async with self.driver.session() as session:
            query = """
            MATCH (m:Memory {memory_id: $memory_id})
            MATCH (e:Entity {entity_id: $entity_id})
            CREATE (m)-[r:MENTIONS {
                relationship_id: $relationship_id,
                properties: $properties,
                created_at: datetime(),
                access_count: 0
            }]->(e)
            RETURN r.relationship_id as relationship_id
            """

            params = {
                "memory_id": memory_id,
                "entity_id": entity_id,
                "relationship_id": relationship_id,
                "properties": properties or {},
            }

            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.info(
                    "Created MENTIONS relationship",
                    memory_id=memory_id,
                    entity_id=entity_id,
                )
                return record["relationship_id"]

            raise RuntimeError(
                f"Failed to create MENTIONS relationship: {memory_id} -> {entity_id}"
            )

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: RelationshipType,
        from_label: str = "Entity",
        to_label: str = "Entity",
        properties: dict[str, Any] | None = None,
    ) -> str:
        """
        Create relationship between nodes.

        Supports relationship types:
        - RELATES_TO: Semantic relationship
        - PART_OF: Hierarchical relationship
        - FOLLOWS: Temporal sequence (from follows to)
        - PRECEDES: Temporal precedence (from precedes to)
        - CONTRADICTS: Conflicting information

        Args:
            from_id: Source node ID
            to_id: Target node ID
            rel_type: Relationship type
            from_label: Source node label (default: Entity)
            to_label: Target node label (default: Entity)
            properties: Additional properties

        Returns:
            Relationship ID

        Example:
            # Create semantic relationship
            rel_id = await service.create_relationship(
                "ent-001", "ent-002", RelationshipType.RELATES_TO
            )

            # Create temporal sequence
            rel_id = await service.create_relationship(
                "mem-001", "mem-002", RelationshipType.FOLLOWS,
                from_label="Memory", to_label="Memory"
            )
        """
        relationship_id = f"rel-{uuid4()}"
        rel_type_str = rel_type.value.upper()

        # Determine ID field based on label
        from_id_field = self._get_id_field(from_label)
        to_id_field = self._get_id_field(to_label)

        async with self.driver.session() as session:
            query = f"""
            MATCH (from:{from_label} {{{from_id_field}: $from_id}})
            MATCH (to:{to_label} {{{to_id_field}: $to_id}})
            CREATE (from)-[r:{rel_type_str} {{
                relationship_id: $relationship_id,
                properties: $properties,
                created_at: datetime(),
                access_count: 0
            }}]->(to)
            RETURN r.relationship_id as relationship_id
            """

            params = {
                "from_id": from_id,
                "to_id": to_id,
                "relationship_id": relationship_id,
                "properties": properties or {},
            }

            result = await session.run(query, params)
            record = await result.single()

            if record:
                logger.info(
                    "Created relationship",
                    rel_type=rel_type,
                    from_id=from_id,
                    to_id=to_id,
                )
                return record["relationship_id"]

            raise RuntimeError(
                f"Failed to create {rel_type} relationship: {from_id} -> {to_id}"
            )

    async def traverse_graph(
        self,
        start_id: str,
        max_depth: int = 2,
        relationship_types: list[RelationshipType] | None = None,
        start_label: str = "Entity",
    ) -> list[dict[str, Any]]:
        """
        Traverse graph from starting node.

        Performs multi-hop graph traversal with depth limits (1-3).
        Returns paths including nodes and relationships.

        Performance target: <200ms for 2-hop traversal (p95).

        Args:
            start_id: Starting node ID
            max_depth: Maximum traversal depth (1-3 recommended)
            relationship_types: Filter by relationship types
            start_label: Starting node label (default: Entity)

        Returns:
            List of path dictionaries with nodes and relationships

        Example:
            paths = await service.traverse_graph(
                "ent-001",
                max_depth=2,
                relationship_types=[RelationshipType.RELATES_TO]
            )
        """
        if max_depth < 1 or max_depth > 3:
            logger.warning(
                "Max depth should be 1-3 for performance",
                max_depth=max_depth,
            )

        # Convert relationship types to uppercase strings
        rel_type_strs = (
            [rt.value.upper() for rt in relationship_types]
            if relationship_types
            else None
        )

        id_field = self._get_id_field(start_label)

        async with self.driver.session() as session:
            return await GraphRepository.query_graph(
                session,
                start_id,
                max_depth=max_depth,
                relationship_types=rel_type_strs,
            )

    async def find_related_entities(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
        limit: int = 50,
    ) -> list[EntityNode]:
        """
        Find entities related to given entity.

        Returns directly connected entities (1-hop).
        Optionally filters by relationship type.

        Args:
            entity_id: Source entity ID
            rel_type: Filter by relationship type (optional)
            limit: Maximum number of results

        Returns:
            List of related EntityNode objects

        Example:
            # Find all related entities
            entities = await service.find_related_entities("ent-001")

            # Find entities connected by RELATES_TO
            entities = await service.find_related_entities(
                "ent-001",
                rel_type=RelationshipType.RELATES_TO
            )
        """
        rel_type_str = rel_type.value if rel_type else None

        async with self.driver.session() as session:
            result_dicts = await GraphRepository.get_related_entities(
                session,
                entity_id,
                relationship_type=rel_type_str,
                limit=limit,
            )

            # Convert dictionaries to EntityNode objects
            entities = []
            for data in result_dicts:
                entity = EntityNode(
                    entity_id=data["entity_id"],
                    entity_name=data["entity_name"],
                    entity_type=EntityType(data["entity_type"]),
                    properties=data.get("properties", {}),
                    memory_refs=data.get("memory_refs", []),
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                )
                entities.append(entity)

            logger.info(
                "Found related entities",
                entity_id=entity_id,
                count=len(entities),
                rel_type=rel_type,
            )

            return entities

    async def get_temporal_sequence(
        self,
        task_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get temporal sequence of memories for a task.

        Returns memories connected by FOLLOWS/PRECEDES relationships.
        Ordered chronologically by timestamp.

        Args:
            task_id: Task ID to query
            limit: Maximum number of memories

        Returns:
            List of memory dictionaries in temporal order

        Example:
            sequence = await service.get_temporal_sequence("task-123")
        """
        async with self.driver.session() as session:
            query = """
            MATCH (m:Memory {task_id: $task_id})
            OPTIONAL MATCH path = (m)-[:FOLLOWS|PRECEDES*0..]->(next:Memory)
            WITH m, collect(DISTINCT next) as sequence
            UNWIND sequence as mem
            RETURN DISTINCT mem
            ORDER BY mem.timestamp
            LIMIT $limit
            """

            params = {
                "task_id": task_id,
                "limit": limit,
            }

            result = await session.run(query, params)
            memories = []

            async for record in result:
                mem_data = dict(record["mem"])
                memories.append(mem_data)

            logger.info(
                "Retrieved temporal sequence",
                task_id=task_id,
                count=len(memories),
            )

            return memories

    async def find_memories_by_entity(
        self,
        entity_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Find memories that mention a specific entity.

        Queries MENTIONS relationships from Memory to Entity.

        Args:
            entity_id: Entity ID to search
            limit: Maximum number of memories

        Returns:
            List of memory dictionaries

        Example:
            memories = await service.find_memories_by_entity("ent-123")
        """
        async with self.driver.session() as session:
            query = """
            MATCH (m:Memory)-[:MENTIONS]->(e:Entity {entity_id: $entity_id})
            RETURN m
            ORDER BY m.timestamp DESC
            LIMIT $limit
            """

            params = {
                "entity_id": entity_id,
                "limit": limit,
            }

            result = await session.run(query, params)
            memories = []

            async for record in result:
                mem_data = dict(record["m"])
                memories.append(mem_data)

            logger.info(
                "Found memories by entity",
                entity_id=entity_id,
                count=len(memories),
            )

            return memories

    async def update_relationship_access(
        self,
        relationship_id: str,
    ) -> bool:
        """
        Increment relationship access count.

        Tracks relationship usage for Memify optimization.

        Args:
            relationship_id: Relationship ID to update

        Returns:
            True if update successful

        Example:
            success = await service.update_relationship_access("rel-123")
        """
        async with self.driver.session() as session:
            return await GraphRepository.update_relationship_access(
                session,
                relationship_id,
            )

    async def get_node_degree(
        self,
        node_id: str,
        node_label: str = "Entity",
    ) -> int:
        """
        Get node connection degree (number of relationships).

        Used for importance scoring - highly connected nodes are more central.

        Args:
            node_id: Node ID to query
            node_label: Node label (default: Entity)

        Returns:
            Number of relationships connected to node

        Example:
            degree = await service.get_node_degree("ent-001")
        """
        id_field = self._get_id_field(node_label)

        async with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_label} {{{id_field}: $node_id}})-[r]-()
            RETURN count(r) as degree
            """

            result = await session.run(query, {"node_id": node_id})
            record = await result.single()

            if record:
                return record["degree"]
            return 0

    async def find_shortest_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        start_label: str = "Entity",
        end_label: str = "Entity",
    ) -> dict[str, Any] | None:
        """
        Find shortest path between two nodes.

        Uses Neo4j shortest path algorithm.
        Useful for understanding relationships between concepts.

        Args:
            start_id: Start node ID
            end_id: End node ID
            max_depth: Maximum path length to search
            start_label: Start node label
            end_label: End node label

        Returns:
            Path dictionary or None if no path found

        Example:
            path = await service.find_shortest_path("ent-001", "ent-002")
        """
        async with self.driver.session() as session:
            return await GraphRepository.find_shortest_path(
                session,
                start_id,
                end_id,
                max_depth=max_depth,
            )

    async def get_one_hop_neighbors(
        self,
        node_id: str,
        relationship_type: str | None = None,
        direction: str = "both",
        node_label: str = "Entity",
        limit: int = 100,
    ) -> list[EntityNode]:
        """
        Get all nodes directly connected to given node (1-hop query).

        Supports directional queries:
        - "outgoing": Follow outgoing relationships (node_id)-[]->(neighbor)
        - "incoming": Follow incoming relationships (neighbor)-[]->(node_id)
        - "both": Follow relationships in both directions (default)

        Args:
            node_id: Source node ID
            relationship_type: Filter by relationship type (optional)
            direction: Relationship direction ("outgoing", "incoming", "both")
            node_label: Node label (default: Entity)
            limit: Maximum number of neighbors

        Returns:
            List of EntityNode objects representing neighbors

        Example:
            # Get all neighbors
            neighbors = await service.get_one_hop_neighbors("ent-001")

            # Get outgoing RELATES_TO neighbors
            neighbors = await service.get_one_hop_neighbors(
                "ent-001",
                relationship_type="RELATES_TO",
                direction="outgoing"
            )
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"Direction must be 'outgoing', 'incoming', or 'both', got: {direction}"
            )

        id_field = self._get_id_field(node_label)

        # Build relationship pattern based on direction
        if direction == "outgoing":
            rel_pattern = "-[r]->"
        elif direction == "incoming":
            rel_pattern = "<-[r]-"
        else:
            rel_pattern = "-[r]-"

        # Add relationship type filter if specified
        if relationship_type:
            rel_pattern = rel_pattern.replace("[r]", f"[r:{relationship_type.upper()}]")

        async with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_label} {{{id_field}: $node_id}}){rel_pattern}(neighbor:Entity)
            RETURN DISTINCT neighbor
            LIMIT $limit
            """

            params = {
                "node_id": node_id,
                "limit": limit,
            }

            result = await session.run(query, params)
            neighbors = []

            async for record in result:
                neighbor_data = dict(record["neighbor"])
                entity = EntityNode(
                    entity_id=neighbor_data["entity_id"],
                    entity_name=neighbor_data["entity_name"],
                    entity_type=EntityType(neighbor_data["entity_type"]),
                    properties=neighbor_data.get("properties", {}),
                    memory_refs=neighbor_data.get("memory_refs", []),
                    created_at=neighbor_data["created_at"],
                    updated_at=neighbor_data["updated_at"],
                )
                neighbors.append(entity)

            logger.info(
                "Found 1-hop neighbors",
                node_id=node_id,
                count=len(neighbors),
                direction=direction,
                rel_type=relationship_type,
            )

            return neighbors

    async def get_two_hop_paths(
        self,
        start_id: str,
        end_id: str | None = None,
        relationship_types: list[str] | None = None,
        start_label: str = "Entity",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Find all 2-hop paths from start node.

        If end_id provided, finds paths specifically to that node.
        Otherwise, returns all 2-hop paths from start.

        Args:
            start_id: Starting node ID
            end_id: Optional target node ID
            relationship_types: Filter by relationship types
            start_label: Start node label (default: Entity)
            limit: Maximum number of paths

        Returns:
            List of path dictionaries with start, middle, end nodes and relationships

        Example:
            # Find all 2-hop paths from node
            paths = await service.get_two_hop_paths("ent-001")

            # Find 2-hop paths to specific target
            paths = await service.get_two_hop_paths(
                "ent-001",
                end_id="ent-002",
                relationship_types=["RELATES_TO"]
            )
        """
        id_field = self._get_id_field(start_label)

        # Build relationship type filter
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join([rt.upper() for rt in relationship_types])
            rel_filter = f":{rel_types}"

        # Build end node filter
        end_filter = ""
        if end_id:
            end_filter = f"{{entity_id: $end_id}}"

        async with self.driver.session() as session:
            query = f"""
            MATCH path = (start:{start_label} {{{id_field}: $start_id}})
                        -[r1{rel_filter}]->(middle:Entity)
                        -[r2{rel_filter}]->(end:Entity {end_filter})
            WHERE start <> end AND start <> middle AND middle <> end
            RETURN start, r1, middle, r2, end
            LIMIT $limit
            """

            params = {
                "start_id": start_id,
                "limit": limit,
            }
            if end_id:
                params["end_id"] = end_id

            result = await session.run(query, params)
            paths = []

            async for record in result:
                path_data = {
                    "start": dict(record["start"]),
                    "relationship1": dict(record["r1"]),
                    "middle": dict(record["middle"]),
                    "relationship2": dict(record["r2"]),
                    "end": dict(record["end"]),
                }
                paths.append(path_data)

            logger.info(
                "Found 2-hop paths",
                start_id=start_id,
                end_id=end_id,
                count=len(paths),
            )

            return paths

    async def find_similar_entities(
        self,
        entity_id: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> list[tuple[EntityNode, float]]:
        """
        Find entities similar to given entity based on shared relationships.

        Similarity is computed using:
        - Number of common neighbors (Jaccard similarity)
        - Shared relationship types
        - Relationship strength aggregation

        Args:
            entity_id: Source entity ID
            similarity_threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results

        Returns:
            List of tuples (EntityNode, similarity_score) sorted by similarity

        Example:
            # Find top 10 similar entities
            similar = await service.find_similar_entities(
                "ent-001",
                similarity_threshold=0.7
            )

            for entity, score in similar:
                print(f"{entity.entity_name}: {score:.2f}")
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"Similarity threshold must be 0.0-1.0, got: {similarity_threshold}"
            )

        async with self.driver.session() as session:
            # Find similar entities based on common neighbors
            query = """
            MATCH (e1:Entity {entity_id: $entity_id})-[r1]-(common:Entity)-[r2]-(e2:Entity)
            WHERE e1 <> e2

            // Count shared neighbors and relationships
            WITH e1, e2,
                count(DISTINCT common) as shared_neighbors,
                count(DISTINCT type(r1)) + count(DISTINCT type(r2)) as shared_rel_types

            // Get total neighbor counts for Jaccard similarity
            MATCH (e1)-[r_e1]-(neighbors_e1:Entity)
            WITH e1, e2, shared_neighbors, shared_rel_types,
                count(DISTINCT neighbors_e1) as total_neighbors_e1

            MATCH (e2)-[r_e2]-(neighbors_e2:Entity)
            WITH e1, e2, shared_neighbors, shared_rel_types,
                total_neighbors_e1,
                count(DISTINCT neighbors_e2) as total_neighbors_e2

            // Calculate Jaccard similarity
            WITH e2, shared_neighbors, shared_rel_types,
                total_neighbors_e1, total_neighbors_e2,
                toFloat(shared_neighbors) /
                toFloat(total_neighbors_e1 + total_neighbors_e2 - shared_neighbors)
                as jaccard_similarity

            // Combined similarity score (weighted)
            WITH e2,
                (jaccard_similarity * 0.7) + (toFloat(shared_rel_types) / 10.0 * 0.3)
                as similarity_score

            WHERE similarity_score >= $threshold
            RETURN e2, similarity_score
            ORDER BY similarity_score DESC
            LIMIT $limit
            """

            params = {
                "entity_id": entity_id,
                "threshold": similarity_threshold,
                "limit": limit,
            }

            result = await session.run(query, params)
            similar_entities = []

            async for record in result:
                entity_data = dict(record["e2"])
                similarity_score = record["similarity_score"]

                entity = EntityNode(
                    entity_id=entity_data["entity_id"],
                    entity_name=entity_data["entity_name"],
                    entity_type=EntityType(entity_data["entity_type"]),
                    properties=entity_data.get("properties", {}),
                    memory_refs=entity_data.get("memory_refs", []),
                    created_at=entity_data["created_at"],
                    updated_at=entity_data["updated_at"],
                )

                similar_entities.append((entity, float(similarity_score)))

            logger.info(
                "Found similar entities",
                entity_id=entity_id,
                count=len(similar_entities),
                threshold=similarity_threshold,
            )

            return similar_entities

    async def aggregate_relationship_strength(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, float]:
        """
        Aggregate relationship strength across multiple paths.

        Computes strength metrics:
        - Direct strength: Strength of direct relationship (if exists)
        - Path count: Number of paths between nodes
        - Average path length: Mean path length
        - Total strength: Sum of all path strengths (weighted by inverse length)

        Args:
            from_id: Source entity ID
            to_id: Target entity ID
            relationship_type: Filter by relationship type (optional)
            max_depth: Maximum path depth to search (default: 3)

        Returns:
            Dictionary with strength metrics

        Example:
            strength = await service.aggregate_relationship_strength(
                "ent-001",
                "ent-002"
            )
            # Returns: {
            #     "direct_strength": 0.85,
            #     "path_count": 5,
            #     "average_path_length": 2.2,
            #     "total_strength": 3.14
            # }
        """
        if max_depth < 1 or max_depth > 5:
            raise ValueError(f"Max depth must be 1-5, got: {max_depth}")

        # Build relationship type filter
        rel_filter = ""
        if relationship_type:
            rel_filter = f":{relationship_type.upper()}"

        async with self.driver.session() as session:
            # Query for direct relationship
            direct_query = f"""
            MATCH (from:Entity {{entity_id: $from_id}})
                  -[r{rel_filter}]->(to:Entity {{entity_id: $to_id}})
            RETURN coalesce(r.properties.strength, 1.0) as strength
            """

            direct_result = await session.run(
                direct_query, {"from_id": from_id, "to_id": to_id}
            )
            direct_record = await direct_result.single()
            direct_strength = (
                float(direct_record["strength"]) if direct_record else 0.0
            )

            # Query for all paths up to max_depth
            paths_query = f"""
            MATCH path = (from:Entity {{entity_id: $from_id}})
                        -[{rel_filter}*1..{max_depth}]->(to:Entity {{entity_id: $to_id}})
            WITH path, length(path) as path_length,
                reduce(s = 1.0, r in relationships(path) |
                    s * coalesce(r.properties.strength, 1.0)
                ) as path_strength
            RETURN
                count(path) as path_count,
                avg(path_length) as avg_length,
                sum(path_strength / path_length) as total_strength
            """

            paths_result = await session.run(
                paths_query, {"from_id": from_id, "to_id": to_id}
            )
            paths_record = await paths_result.single()

            # Build result
            result = {
                "direct_strength": direct_strength,
                "path_count": paths_record["path_count"] if paths_record else 0,
                "average_path_length": (
                    float(paths_record["avg_length"]) if paths_record else 0.0
                ),
                "total_strength": (
                    float(paths_record["total_strength"]) if paths_record else 0.0
                ),
            }

            logger.info(
                "Aggregated relationship strength",
                from_id=from_id,
                to_id=to_id,
                path_count=result["path_count"],
                total_strength=result["total_strength"],
            )

            return result

    async def close(self) -> None:
        """Close Neo4j driver connection."""
        await self.driver.close()
        logger.info("GraphMemoryService closed")

    @staticmethod
    def _get_id_field(label: str) -> str:
        """
        Get ID field name for node label.

        Maps node labels to their ID field:
        - Memory -> memory_id
        - Entity -> entity_id
        - Concept -> concept_id

        Args:
            label: Node label

        Returns:
            ID field name
        """
        label_to_field = {
            "Memory": "memory_id",
            "Entity": "entity_id",
            "Concept": "concept_id",
        }
        return label_to_field.get(label, "entity_id")


# Export service
__all__ = ["GraphMemoryService"]
