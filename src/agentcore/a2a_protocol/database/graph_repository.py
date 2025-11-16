"""
Graph Repository for Neo4j Operations

Provides async Neo4j operations for knowledge graph entities and relationships.
Implements graph-based memory retrieval and traversal for the memory system.

Component ID: MEM-007
Ticket: MEM-007 (Implement Repository Layer with Graph Support)
"""

from typing import Any
from uuid import UUID

import structlog
from neo4j import AsyncSession

from agentcore.a2a_protocol.models.memory import EntityNode, RelationshipEdge

logger = structlog.get_logger(__name__)


class GraphRepository:
    """Repository for Neo4j graph operations on entities and relationships."""

    @staticmethod
    async def create_node(
        session: AsyncSession, entity: EntityNode
    ) -> dict[str, Any]:
        """
        Create entity node in Neo4j graph.

        Args:
            session: Neo4j async session
            entity: Entity node to create

        Returns:
            Dictionary with created node properties

        Example:
            node = await GraphRepository.create_node(session, entity_node)
        """
        import json

        query = """
        CREATE (e:Entity {
            entity_id: $entity_id,
            entity_name: $entity_name,
            entity_type: $entity_type,
            properties_json: $properties_json,
            memory_refs: $memory_refs,
            created_at: datetime($created_at),
            updated_at: datetime($updated_at)
        })
        RETURN e
        """

        params = {
            "entity_id": entity.entity_id,
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type.value,
            "properties_json": json.dumps(entity.properties),
            "memory_refs": entity.memory_refs,
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
        }

        result = await session.run(query, params)
        record = await result.single()

        if record:
            node = record["e"]
            logger.info(
                "Created entity node",
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
            )
            return dict(node)

        raise RuntimeError(f"Failed to create entity node: {entity.entity_id}")

    @staticmethod
    async def create_relationship(
        session: AsyncSession, relationship: RelationshipEdge
    ) -> dict[str, Any]:
        """
        Create relationship edge in Neo4j graph.

        Args:
            session: Neo4j async session
            relationship: Relationship edge to create

        Returns:
            Dictionary with created relationship properties

        Example:
            rel = await GraphRepository.create_relationship(session, rel_edge)
        """
        import json

        # Dynamically create relationship type from enum value
        rel_type = relationship.relationship_type.value.upper()

        query = f"""
        MATCH (source:Entity {{entity_id: $source_id}})
        MATCH (target:Entity {{entity_id: $target_id}})
        CREATE (source)-[r:{rel_type} {{
            relationship_id: $relationship_id,
            properties_json: $properties_json,
            memory_refs: $memory_refs,
            created_at: datetime($created_at),
            access_count: $access_count
        }}]->(target)
        RETURN r
        """

        params = {
            "source_id": relationship.source_entity_id,
            "target_id": relationship.target_entity_id,
            "relationship_id": relationship.relationship_id,
            "properties_json": json.dumps(relationship.properties),
            "memory_refs": relationship.memory_refs,
            "created_at": relationship.created_at.isoformat(),
            "access_count": relationship.access_count,
        }

        result = await session.run(query, params)
        record = await result.single()

        if record:
            rel = record["r"]
            logger.info(
                "Created relationship",
                relationship_id=relationship.relationship_id,
                relationship_type=relationship.relationship_type,
            )
            return dict(rel)

        raise RuntimeError(
            f"Failed to create relationship: {relationship.relationship_id}"
        )

    @staticmethod
    async def get_node_by_id(
        session: AsyncSession, entity_id: str
    ) -> dict[str, Any] | None:
        """
        Get entity node by ID.

        Args:
            session: Neo4j async session
            entity_id: Entity ID to retrieve

        Returns:
            Dictionary with node properties or None if not found
        """
        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        RETURN e
        """

        result = await session.run(query, {"entity_id": entity_id})
        record = await result.single()

        if record:
            return dict(record["e"])
        return None

    @staticmethod
    async def query_graph(
        session: AsyncSession,
        start_entity_id: str,
        max_depth: int = 2,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query graph using traversal from starting entity.

        Args:
            session: Neo4j async session
            start_entity_id: Starting entity ID for traversal
            max_depth: Maximum traversal depth (default: 2 hops)
            relationship_types: Filter by relationship types (optional)

        Returns:
            List of dictionaries with path information (nodes, relationships)

        Example:
            paths = await GraphRepository.query_graph(
                session, "ent-123", max_depth=2, relationship_types=["RELATES_TO"]
            )
        """
        if relationship_types:
            # Build relationship type filter
            rel_filter = "|".join([f":{rt.upper()}" for rt in relationship_types])
            rel_pattern = f"[{rel_filter}*1..{max_depth}]"
        else:
            rel_pattern = f"[*1..{max_depth}]"

        query = f"""
        MATCH path = (start:Entity {{entity_id: $start_id}})-{rel_pattern}-(connected:Entity)
        RETURN path
        LIMIT 100
        """

        result = await session.run(query, {"start_id": start_entity_id})
        paths = []

        async for record in result:
            path = record["path"]
            path_data = {
                "nodes": [dict(node) for node in path.nodes],
                "relationships": [dict(rel) for rel in path.relationships],
            }
            paths.append(path_data)

        logger.info(
            "Queried graph",
            start_entity=start_entity_id,
            max_depth=max_depth,
            paths_found=len(paths),
        )

        return paths

    @staticmethod
    async def get_related_entities(
        session: AsyncSession,
        entity_id: str,
        relationship_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get entities directly related to given entity.

        Args:
            session: Neo4j async session
            entity_id: Source entity ID
            relationship_type: Filter by relationship type (optional)
            limit: Maximum number of results

        Returns:
            List of related entity dictionaries
        """
        if relationship_type:
            rel_pattern = f"[:{relationship_type.upper()}]"
        else:
            rel_pattern = ""

        query = f"""
        MATCH (source:Entity {{entity_id: $entity_id}})-{rel_pattern}-(related:Entity)
        RETURN related
        LIMIT $limit
        """

        result = await session.run(query, {"entity_id": entity_id, "limit": limit})
        entities = []

        async for record in result:
            entities.append(dict(record["related"]))

        return entities

    @staticmethod
    async def update_node(
        session: AsyncSession, entity_id: str, updates: dict[str, Any]
    ) -> bool:
        """
        Update entity node properties.

        Args:
            session: Neo4j async session
            entity_id: Entity ID to update
            updates: Dictionary of properties to update

        Returns:
            True if update successful
        """
        # Build SET clause dynamically
        set_clauses = [f"e.{key} = ${key}" for key in updates.keys()]
        set_clause = ", ".join(set_clauses)

        query = f"""
        MATCH (e:Entity {{entity_id: $entity_id}})
        SET {set_clause}
        RETURN e
        """

        params = {"entity_id": entity_id, **updates}
        result = await session.run(query, params)
        record = await result.single()

        return record is not None

    @staticmethod
    async def update_relationship_access(
        session: AsyncSession, relationship_id: str
    ) -> bool:
        """
        Increment relationship access count.

        Args:
            session: Neo4j async session
            relationship_id: Relationship ID to update

        Returns:
            True if update successful
        """
        query = """
        MATCH ()-[r {relationship_id: $relationship_id}]-()
        WITH r LIMIT 1
        SET r.access_count = r.access_count + 1
        RETURN r
        """

        result = await session.run(query, {"relationship_id": relationship_id})
        record = await result.single()

        return record is not None

    @staticmethod
    async def delete_node(session: AsyncSession, entity_id: str) -> bool:
        """
        Delete entity node and all connected relationships.

        Args:
            session: Neo4j async session
            entity_id: Entity ID to delete

        Returns:
            True if deletion successful
        """
        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """

        result = await session.run(query, {"entity_id": entity_id})
        record = await result.single()

        if record and record["deleted"] > 0:
            logger.info("Deleted entity node", entity_id=entity_id)
            return True
        return False

    @staticmethod
    async def delete_relationship(
        session: AsyncSession, relationship_id: str
    ) -> bool:
        """
        Delete relationship by ID.

        Args:
            session: Neo4j async session
            relationship_id: Relationship ID to delete

        Returns:
            True if deletion successful
        """
        query = """
        MATCH ()-[r {relationship_id: $relationship_id}]-()
        DELETE r
        RETURN count(r) as deleted
        """

        result = await session.run(query, {"relationship_id": relationship_id})
        record = await result.single()

        if record and record["deleted"] > 0:
            logger.info("Deleted relationship", relationship_id=relationship_id)
            return True
        return False

    @staticmethod
    async def find_shortest_path(
        session: AsyncSession,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 5,
    ) -> dict[str, Any] | None:
        """
        Find shortest path between two entities.

        Args:
            session: Neo4j async session
            start_entity_id: Start entity ID
            end_entity_id: End entity ID
            max_depth: Maximum path length to search

        Returns:
            Dictionary with path information or None if no path found
        """
        query = f"""
        MATCH path = shortestPath(
            (start:Entity {{entity_id: $start_id}})-[*1..{max_depth}]-(end:Entity {{entity_id: $end_id}})
        )
        RETURN path
        """

        result = await session.run(
            query, {"start_id": start_entity_id, "end_id": end_entity_id}
        )
        record = await result.single()

        if record:
            path = record["path"]
            return {
                "nodes": [dict(node) for node in path.nodes],
                "relationships": [dict(rel) for rel in path.relationships],
                "length": len(path.relationships),
            }
        return None

    @staticmethod
    async def search_entities_by_name(
        session: AsyncSession, name_pattern: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Search entities by name pattern (case-insensitive).

        Args:
            session: Neo4j async session
            name_pattern: Search pattern (supports wildcards with *)
            limit: Maximum number of results

        Returns:
            List of matching entity dictionaries
        """
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.entity_name) CONTAINS toLower($pattern)
        RETURN e
        ORDER BY e.entity_name
        LIMIT $limit
        """

        result = await session.run(query, {"pattern": name_pattern, "limit": limit})
        entities = []

        async for record in result:
            entities.append(dict(record["e"]))

        return entities


# Export repository
__all__ = ["GraphRepository"]
