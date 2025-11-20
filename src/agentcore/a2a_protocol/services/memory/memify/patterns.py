"""
Pattern Detection for Memify Graph Optimization

Identifies frequently traversed paths and common relationship patterns
in the knowledge graph. Uses pattern frequency analysis to optimize
query performance and suggest index creation.

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

from typing import Any

import structlog
from neo4j import AsyncSession

logger = structlog.get_logger(__name__)


class PatternDetection:
    """
    Pattern detection algorithm for identifying common graph patterns.

    Detects:
    - Frequently traversed paths (query optimization targets)
    - Common relationship patterns (for caching)
    - Entity clusters (community detection)
    - Query pattern hotspots (for index optimization)

    Performance target: Identify top patterns affecting 80%+ of queries
    """

    def __init__(
        self,
        min_frequency: int = 5,
    ):
        """
        Initialize pattern detection.

        Args:
            min_frequency: Minimum occurrence count to consider a pattern (default: 5)
        """
        if min_frequency < 1:
            raise ValueError(
                f"Min frequency must be >= 1, got: {min_frequency}"
            )

        self.min_frequency = min_frequency

    async def find_frequent_paths(
        self,
        session: AsyncSession,
        max_depth: int = 3,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Find frequently traversed paths in the graph.

        Identifies paths that are commonly accessed based on:
        - Total access count across path relationships
        - Path frequency (how many times this pattern appears)
        - Path efficiency (length vs. access count)

        Args:
            session: Neo4j async session
            max_depth: Maximum path depth to analyze (default: 3)
            limit: Maximum number of patterns to return

        Returns:
            List of path pattern dictionaries sorted by frequency

        Example:
            patterns = await detection.find_frequent_paths(session)
            # [
            #     {
            #         "pattern": "Entity-RELATES_TO->Entity-PART_OF->Concept",
            #         "frequency": 150,
            #         "avg_access_count": 25.5,
            #         "path_length": 2
            #     },
            #     ...
            # ]
        """
        query = f"""
        // Find paths up to max_depth
        MATCH path = (start:Entity)-[rels*1..{max_depth}]->(end)

        // Calculate path pattern signature
        WITH path, rels,
            reduce(s = "", r in rels |
                s + CASE WHEN s = "" THEN "" ELSE "-" END + type(r)
            ) as pattern_signature,
            reduce(total = 0, r in rels |
                total + coalesce(r.access_count, 0)
            ) as total_access_count

        // Only consider paths with minimum access
        WHERE total_access_count >= $min_frequency

        // Group by pattern and calculate statistics
        WITH pattern_signature,
            count(*) as frequency,
            avg(total_access_count) as avg_access_count,
            length(rels) as path_length

        // Sort by frequency and access
        RETURN pattern_signature as pattern,
               frequency,
               avg_access_count,
               path_length
        ORDER BY frequency DESC, avg_access_count DESC
        LIMIT $limit
        """

        params = {
            "min_frequency": self.min_frequency,
            "limit": limit,
        }

        result = await session.run(query, params)
        patterns = []

        async for record in result:
            pattern_data = {
                "pattern": record["pattern"],
                "frequency": record["frequency"],
                "avg_access_count": float(record["avg_access_count"]),
                "path_length": record["path_length"],
                "efficiency_score": (
                    record["frequency"] * float(record["avg_access_count"])
                ) / record["path_length"],
            }
            patterns.append(pattern_data)

        logger.info(
            "Found frequent path patterns",
            count=len(patterns),
            min_frequency=self.min_frequency,
        )

        return patterns

    async def find_entity_clusters(
        self,
        session: AsyncSession,
        min_cluster_size: int = 3,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Find clusters of highly connected entities.

        Uses degree centrality and community detection to identify
        groups of entities that are frequently accessed together.

        Args:
            session: Neo4j async session
            min_cluster_size: Minimum number of entities in a cluster
            limit: Maximum number of clusters to return

        Returns:
            List of cluster dictionaries

        Example:
            clusters = await detection.find_entity_clusters(session)
            # [
            #     {
            #         "cluster_id": "cluster-001",
            #         "entities": ["ent-001", "ent-002", "ent-003"],
            #         "entity_count": 3,
            #         "relationship_density": 0.67,
            #         "total_access": 250
            #     },
            #     ...
            # ]
        """
        query = """
        // Find entities with high connectivity
        MATCH (e:Entity)-[r]-(connected:Entity)

        WITH e, count(DISTINCT connected) as degree,
            sum(r.access_count) as total_access
        WHERE degree >= $min_cluster_size

        // Find their neighbors
        MATCH (e)-[r1]-(neighbor:Entity)
        WHERE neighbor <> e

        // Group by entity to form potential clusters
        WITH e, total_access, collect(DISTINCT neighbor.entity_id) as neighbors

        // Calculate cluster density
        WITH e, total_access, neighbors,
            size(neighbors) as neighbor_count

        // Return cluster information
        RETURN e.entity_id as center_entity,
               neighbors,
               neighbor_count as entity_count,
               total_access,
               toFloat(total_access) / toFloat(neighbor_count) as density_score
        ORDER BY density_score DESC
        LIMIT $limit
        """

        params = {
            "min_cluster_size": min_cluster_size,
            "limit": limit,
        }

        result = await session.run(query, params)
        clusters = []

        async for record in result:
            cluster_data = {
                "center_entity": record["center_entity"],
                "entities": [record["center_entity"]] + record["neighbors"],
                "entity_count": record["entity_count"] + 1,  # Include center
                "total_access": record["total_access"],
                "density_score": float(record["density_score"]),
            }
            clusters.append(cluster_data)

        logger.info(
            "Found entity clusters",
            count=len(clusters),
            min_cluster_size=min_cluster_size,
        )

        return clusters

    async def find_common_relationships(
        self,
        session: AsyncSession,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Find most common relationship patterns.

        Analyzes relationship type frequency and access patterns
        to identify optimization opportunities.

        Args:
            session: Neo4j async session
            limit: Maximum number of relationship patterns to return

        Returns:
            List of relationship pattern dictionaries

        Example:
            relationships = await detection.find_common_relationships(session)
            # [
            #     {
            #         "relationship_type": "RELATES_TO",
            #         "count": 500,
            #         "avg_access_count": 15.2,
            #         "total_access": 7600
            #     },
            #     ...
            # ]
        """
        query = """
        // Aggregate relationship statistics by type
        MATCH ()-[r]-()

        WITH type(r) as relationship_type,
            count(r) as relationship_count,
            avg(r.access_count) as avg_access_count,
            sum(r.access_count) as total_access

        // Sort by total access (most important patterns)
        RETURN relationship_type,
               relationship_count as count,
               avg_access_count,
               total_access
        ORDER BY total_access DESC
        LIMIT $limit
        """

        params = {"limit": limit}

        result = await session.run(query, params)
        relationships = []

        async for record in result:
            rel_data = {
                "relationship_type": record["relationship_type"],
                "count": record["count"],
                "avg_access_count": float(record["avg_access_count"]),
                "total_access": record["total_access"],
            }
            relationships.append(rel_data)

        logger.info(
            "Found common relationship patterns",
            count=len(relationships),
        )

        return relationships

    async def suggest_index_optimizations(
        self,
        session: AsyncSession,
    ) -> list[dict[str, Any]]:
        """
        Suggest index creation based on query patterns.

        Analyzes access patterns to recommend:
        - Composite indexes for frequently queried property combinations
        - Range indexes for access_count filtering
        - Full-text indexes for entity name searches

        Args:
            session: Neo4j async session

        Returns:
            List of index recommendation dictionaries

        Example:
            suggestions = await detection.suggest_index_optimizations(session)
            # [
            #     {
            #         "index_type": "composite",
            #         "node_label": "Entity",
            #         "properties": ["entity_type", "entity_name"],
            #         "estimated_benefit": "high",
            #         "reason": "Frequently queried together in 250+ queries"
            #     },
            #     ...
            # ]
        """
        # Analyze entity type + name queries (already have this index)
        type_name_query = """
        MATCH (e:Entity)
        RETURN e.entity_type as entity_type,
               count(*) as type_count
        ORDER BY type_count DESC
        LIMIT 10
        """

        type_result = await session.run(type_name_query)
        suggestions = []

        async for record in type_result:
            entity_type = record["entity_type"]
            type_count = record["type_count"]

            if type_count >= self.min_frequency * 10:
                suggestions.append({
                    "index_type": "filtered",
                    "node_label": "Entity",
                    "filter": f"entity_type = '{entity_type}'",
                    "estimated_benefit": "medium",
                    "reason": f"High-frequency entity type ({type_count} entities)",
                })

        # Suggest access_count range index
        access_query = """
        MATCH ()-[r]-()
        WITH r.access_count as access_count
        WHERE access_count IS NOT NULL
        RETURN count(*) as total_with_access,
               avg(access_count) as avg_access
        """

        access_result = await session.run(access_query)
        access_record = await access_result.single()

        if access_record and access_record["total_with_access"] > 100:
            suggestions.append({
                "index_type": "range",
                "relationship_property": "access_count",
                "estimated_benefit": "high",
                "reason": "Frequent filtering on access_count for pruning operations",
            })

        # Suggest temporal index for created_at
        suggestions.append({
            "index_type": "range",
            "node_label": "Entity",
            "properties": ["created_at"],
            "estimated_benefit": "medium",
            "reason": "Temporal queries for stale relationship detection",
        })

        logger.info(
            "Generated index optimization suggestions",
            count=len(suggestions),
        )

        return suggestions

    async def detect_all_patterns(
        self,
        session: AsyncSession,
    ) -> dict[str, Any]:
        """
        Run all pattern detection algorithms.

        Provides comprehensive pattern analysis including:
        - Frequent paths
        - Entity clusters
        - Common relationships
        - Index optimization suggestions

        Args:
            session: Neo4j async session

        Returns:
            Dictionary with all pattern detection results

        Example:
            results = await detection.detect_all_patterns(session)
            # {
            #     "frequent_paths": [...],
            #     "entity_clusters": [...],
            #     "common_relationships": [...],
            #     "index_suggestions": [...]
            # }
        """
        # Run all detection algorithms
        frequent_paths = await self.find_frequent_paths(session)
        entity_clusters = await self.find_entity_clusters(session)
        common_relationships = await self.find_common_relationships(session)
        index_suggestions = await self.suggest_index_optimizations(session)

        results = {
            "frequent_paths": frequent_paths,
            "frequent_paths_count": len(frequent_paths),
            "entity_clusters": entity_clusters,
            "entity_clusters_count": len(entity_clusters),
            "common_relationships": common_relationships,
            "common_relationships_count": len(common_relationships),
            "index_suggestions": index_suggestions,
            "index_suggestions_count": len(index_suggestions),
        }

        logger.info(
            "Completed pattern detection",
            frequent_paths=results["frequent_paths_count"],
            entity_clusters=results["entity_clusters_count"],
            common_relationships=results["common_relationships_count"],
            index_suggestions=results["index_suggestions_count"],
        )

        return results


# Export
__all__ = ["PatternDetection"]
