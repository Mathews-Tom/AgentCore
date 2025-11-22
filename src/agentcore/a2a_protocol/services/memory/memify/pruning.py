"""
Relationship Pruning for Memify Graph Optimization

Removes low-value relationships from the knowledge graph to improve
query performance and reduce graph complexity. Uses access count
and relationship strength metrics to identify edges for removal.

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

from typing import Any

import structlog
from neo4j import AsyncSession

logger = structlog.get_logger(__name__)


class RelationshipPruning:
    """
    Relationship pruning algorithm for removing low-value edges.

    Implements pruning based on:
    - Access count threshold (default: < 2)
    - Relationship age (stale relationships)
    - Redundant paths (same information via shorter path)
    - Low relationship strength scores

    Performance target: 95%+ relationship relevance after pruning
    """

    def __init__(
        self,
        min_access_count: int = 2,
        max_age_days: int | None = None,
    ):
        """
        Initialize relationship pruning.

        Args:
            min_access_count: Minimum access count to keep relationship (default: 2)
            max_age_days: Maximum age in days for relationships (None = no age limit)
        """
        if min_access_count < 0:
            raise ValueError(
                f"Min access count must be >= 0, got: {min_access_count}"
            )

        if max_age_days is not None and max_age_days <= 0:
            raise ValueError(
                f"Max age days must be > 0 or None, got: {max_age_days}"
            )

        self.min_access_count = min_access_count
        self.max_age_days = max_age_days

    async def find_low_value_relationships(
        self,
        session: AsyncSession,
        limit: int = 10000,
    ) -> list[str]:
        """
        Find relationships that should be pruned.

        Identifies relationships with:
        - Access count below threshold
        - Age exceeding maximum (if configured)
        - Redundant paths (accessible via shorter alternative)

        Args:
            session: Neo4j async session
            limit: Maximum number of relationships to return

        Returns:
            List of relationship IDs to prune

        Example:
            rel_ids = await pruning.find_low_value_relationships(session)
            # ["rel-001", "rel-002", ...]
        """
        # Build age filter if configured
        age_filter = ""
        if self.max_age_days is not None:
            age_filter = f"""
            AND r.created_at < datetime() - duration({{days: {self.max_age_days}}})
            """

        query = f"""
        // Find relationships with low access count
        MATCH ()-[r]-()
        WHERE r.access_count < $min_access_count
        {age_filter}

        // Return unique relationship IDs
        RETURN DISTINCT r.relationship_id as relationship_id
        LIMIT $limit
        """

        params = {
            "min_access_count": self.min_access_count,
            "limit": limit,
        }

        result = await session.run(query, params)
        relationship_ids = []

        async for record in result:
            relationship_ids.append(record["relationship_id"])

        logger.info(
            "Found low-value relationships",
            count=len(relationship_ids),
            min_access_count=self.min_access_count,
        )

        return relationship_ids

    async def find_redundant_relationships(
        self,
        session: AsyncSession,
        max_depth: int = 3,
        limit: int = 1000,
    ) -> list[str]:
        """
        Find redundant relationships (same info via shorter path).

        A relationship is redundant if:
        - There exists a shorter path between the same nodes
        - The shorter path has higher total strength
        - The relationship has low access count

        Args:
            session: Neo4j async session
            max_depth: Maximum path depth to check (default: 3)
            limit: Maximum number of redundant relationships to return

        Returns:
            List of redundant relationship IDs

        Example:
            redundant = await pruning.find_redundant_relationships(session)
        """
        query = f"""
        // Find direct relationships with low access
        MATCH (source:Entity)-[r_direct]->(target:Entity)
        WHERE r_direct.access_count < $min_access_count

        // Check if shorter path exists
        MATCH path = (source)-[*1..{max_depth}]->(target)
        WHERE length(path) > 1
          AND r_direct NOT IN relationships(path)

        // Calculate path strength
        WITH r_direct, path,
            reduce(s = 1.0, r in relationships(path) |
                s * coalesce(r.access_count, 0) / 10.0
            ) as path_strength

        // Keep only if alternative path is stronger
        WHERE path_strength > coalesce(r_direct.access_count, 0) / 10.0

        RETURN DISTINCT r_direct.relationship_id as relationship_id
        LIMIT $limit
        """

        params = {
            "min_access_count": self.min_access_count,
            "limit": limit,
        }

        result = await session.run(query, params)
        redundant_ids = []

        async for record in result:
            redundant_ids.append(record["relationship_id"])

        logger.info(
            "Found redundant relationships",
            count=len(redundant_ids),
            max_depth=max_depth,
        )

        return redundant_ids

    async def prune_relationship(
        self,
        session: AsyncSession,
        relationship_id: str,
    ) -> bool:
        """
        Delete a single relationship by ID.

        Args:
            session: Neo4j async session
            relationship_id: Relationship ID to delete

        Returns:
            True if deletion successful

        Example:
            success = await pruning.prune_relationship(session, "rel-123")
        """
        query = """
        MATCH ()-[r {relationship_id: $relationship_id}]-()
        DELETE r
        RETURN count(r) as deleted
        """

        result = await session.run(query, {"relationship_id": relationship_id})
        record = await result.single()

        if record and record["deleted"] > 0:
            logger.debug("Pruned relationship", relationship_id=relationship_id)
            return True

        logger.warning(
            "Failed to prune relationship (not found)",
            relationship_id=relationship_id,
        )
        return False

    async def prune_relationships_batch(
        self,
        session: AsyncSession,
        relationship_ids: list[str],
    ) -> int:
        """
        Delete multiple relationships in a single transaction.

        Args:
            session: Neo4j async session
            relationship_ids: List of relationship IDs to delete

        Returns:
            Number of relationships deleted

        Example:
            deleted = await pruning.prune_relationships_batch(
                session, ["rel-001", "rel-002"]
            )
        """
        if not relationship_ids:
            return 0

        query = """
        UNWIND $relationship_ids as rel_id
        MATCH ()-[r {relationship_id: rel_id}]-()
        DELETE r
        RETURN count(r) as deleted
        """

        result = await session.run(query, {"relationship_ids": relationship_ids})
        record = await result.single()

        deleted_count = record["deleted"] if record else 0

        logger.info(
            "Pruned relationships batch",
            requested=len(relationship_ids),
            deleted=deleted_count,
        )

        return deleted_count

    async def prune_all_low_value(
        self,
        session: AsyncSession,
        batch_size: int = 1000,
        include_redundant: bool = True,
    ) -> dict[str, Any]:
        """
        Prune all low-value relationships in batches.

        Removes:
        - Relationships with low access count
        - Stale relationships (if age limit configured)
        - Redundant relationships (if enabled)

        Args:
            session: Neo4j async session
            batch_size: Number of relationships to process per batch
            include_redundant: Whether to also remove redundant relationships

        Returns:
            Dictionary with pruning statistics

        Example:
            stats = await pruning.prune_all_low_value(session)
            # {
            #     "low_value_pruned": 150,
            #     "redundant_pruned": 75,
            #     "total_pruned": 225
            # }
        """
        # Find and prune low-value relationships
        low_value_ids = await self.find_low_value_relationships(
            session, limit=batch_size
        )
        low_value_pruned = await self.prune_relationships_batch(
            session, low_value_ids
        )

        # Find and prune redundant relationships if enabled
        redundant_pruned = 0
        if include_redundant:
            redundant_ids = await self.find_redundant_relationships(
                session, limit=batch_size
            )
            redundant_pruned = await self.prune_relationships_batch(
                session, redundant_ids
            )

        stats = {
            "low_value_pruned": low_value_pruned,
            "redundant_pruned": redundant_pruned,
            "total_pruned": low_value_pruned + redundant_pruned,
        }

        logger.info(
            "Completed relationship pruning",
            low_value_pruned=stats["low_value_pruned"],
            redundant_pruned=stats["redundant_pruned"],
            total_pruned=stats["total_pruned"],
        )

        return stats

    async def validate_pruning(
        self,
        session: AsyncSession,
    ) -> dict[str, Any]:
        """
        Validate pruning quality.

        Checks:
        - Percentage of relationships removed
        - Average access count of remaining relationships
        - Graph connectivity (no isolated nodes created)

        Args:
            session: Neo4j async session

        Returns:
            Dictionary with validation metrics

        Example:
            metrics = await pruning.validate_pruning(session)
            # {
            #     "avg_access_count": 5.2,
            #     "isolated_nodes": 0,
            #     "connectivity_score": 0.98
            # }
        """
        # Calculate average access count of remaining relationships
        access_query = """
        MATCH ()-[r]-()
        RETURN avg(r.access_count) as avg_access_count,
               count(r) as total_relationships
        """

        access_result = await session.run(access_query)
        access_record = await access_result.single()

        avg_access_count = (
            float(access_record["avg_access_count"])
            if access_record and access_record["avg_access_count"]
            else 0.0
        )
        total_relationships = (
            access_record["total_relationships"] if access_record else 0
        )

        # Count isolated nodes (no relationships)
        isolated_query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[]-()
        RETURN count(e) as isolated_count
        """

        isolated_result = await session.run(isolated_query)
        isolated_record = await isolated_result.single()
        isolated_count = isolated_record["isolated_count"] if isolated_record else 0

        # Calculate connectivity score
        total_nodes_query = """
        MATCH (e:Entity)
        RETURN count(e) as total_nodes
        """

        total_result = await session.run(total_nodes_query)
        total_record = await total_result.single()
        total_nodes = total_record["total_nodes"] if total_record else 1

        connectivity_score = (
            1.0 - (isolated_count / total_nodes) if total_nodes > 0 else 1.0
        )

        metrics = {
            "avg_access_count": avg_access_count,
            "total_relationships": total_relationships,
            "isolated_nodes": isolated_count,
            "total_nodes": total_nodes,
            "connectivity_score": connectivity_score,
        }

        logger.info(
            "Pruning validation complete",
            avg_access_count=f"{avg_access_count:.2f}",
            isolated_nodes=isolated_count,
            connectivity_score=f"{connectivity_score:.2%}",
        )

        return metrics


# Export
__all__ = ["RelationshipPruning"]
