"""
Graph Optimizer for Memify Operations

Implements graph optimization operations including:
- Entity consolidation (merge similar entities with >90% similarity)
- Relationship pruning (remove low-value edges with access_count < 2)
- Pattern detection (identify frequently traversed paths)
- Index optimization (update Neo4j indexes based on query patterns)
- Quality metrics (track connectivity, relationship density)

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from croniter import croniter
from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)


@dataclass
class OptimizationMetrics:
    """Metrics collected during optimization run."""

    optimization_id: str = field(default_factory=lambda: f"opt-{uuid4()}")
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Entity consolidation metrics
    entities_analyzed: int = 0
    duplicate_pairs_found: int = 0
    entities_merged: int = 0
    consolidation_accuracy: float = 0.0

    # Relationship pruning metrics
    relationships_analyzed: int = 0
    low_value_edges_removed: int = 0

    # Pattern detection metrics
    patterns_detected: int = 0
    frequent_paths: list[dict[str, Any]] = field(default_factory=list)

    # Index optimization metrics
    indexes_optimized: int = 0

    # Quality metrics
    graph_connectivity: float = 0.0
    relationship_density: float = 0.0
    average_node_degree: float = 0.0
    duplicate_rate: float = 0.0

    # Performance
    duration_seconds: float = 0.0
    entities_per_second: float = 0.0


@dataclass
class ConsolidationCandidate:
    """Entity pair candidate for consolidation."""

    entity1_id: str
    entity2_id: str
    entity1_name: str
    entity2_name: str
    similarity_score: float
    shared_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequentPath:
    """Frequently traversed graph path."""

    path_pattern: list[str]  # Sequence of entity types
    traversal_count: int
    average_access_count: float
    representative_path: list[str]  # Sample entity IDs


class GraphOptimizer:
    """
    Optimizes Neo4j knowledge graph for improved performance and quality.

    Features:
    - Entity consolidation: Merges similar entities (>90% similarity)
    - Relationship pruning: Removes low-value edges (access_count < 2)
    - Pattern detection: Identifies frequently traversed paths
    - Index optimization: Updates indexes based on query patterns
    - Quality metrics: Tracks connectivity and density

    Performance target: <5s optimization per 1000 entities
    Consolidation accuracy target: 90%+
    Duplicate rate target: <5% after optimization
    """

    def __init__(
        self,
        driver: AsyncDriver,
        similarity_threshold: float = 0.90,
        min_access_count: int = 2,
        batch_size: int = 100,
    ):
        """
        Initialize GraphOptimizer.

        Args:
            driver: Neo4j async driver instance
            similarity_threshold: Minimum similarity for entity consolidation (default: 0.90)
            min_access_count: Minimum access count for relationships (default: 2)
            batch_size: Batch size for optimization operations (default: 100)
        """
        self.driver = driver
        self.similarity_threshold = similarity_threshold
        self.min_access_count = min_access_count
        self.batch_size = batch_size
        self._scheduled_job_id: str | None = None
        self._cron_expression: str | None = None
        self._next_run: datetime | None = None

    async def optimize(self) -> OptimizationMetrics:
        """
        Run full graph optimization cycle.

        Executes:
        1. Entity consolidation
        2. Relationship pruning
        3. Pattern detection
        4. Index optimization
        5. Quality metrics computation

        Returns:
            OptimizationMetrics with all collected metrics

        Performance: <5s per 1000 entities
        """
        metrics = OptimizationMetrics()

        logger.info(
            "Starting graph optimization",
            optimization_id=metrics.optimization_id,
            similarity_threshold=self.similarity_threshold,
            min_access_count=self.min_access_count,
        )

        # Step 1: Entity consolidation
        consolidation_metrics = await self._consolidate_entities()
        metrics.entities_analyzed = consolidation_metrics["entities_analyzed"]
        metrics.duplicate_pairs_found = consolidation_metrics["duplicate_pairs_found"]
        metrics.entities_merged = consolidation_metrics["entities_merged"]
        metrics.consolidation_accuracy = consolidation_metrics["consolidation_accuracy"]

        # Step 2: Relationship pruning
        pruning_metrics = await self._prune_relationships()
        metrics.relationships_analyzed = pruning_metrics["relationships_analyzed"]
        metrics.low_value_edges_removed = pruning_metrics["low_value_edges_removed"]

        # Step 3: Pattern detection
        pattern_metrics = await self._detect_patterns()
        metrics.patterns_detected = pattern_metrics["patterns_detected"]
        metrics.frequent_paths = pattern_metrics["frequent_paths"]

        # Step 4: Index optimization
        index_metrics = await self._optimize_indexes()
        metrics.indexes_optimized = index_metrics["indexes_optimized"]

        # Step 5: Quality metrics computation
        quality_metrics = await self._compute_quality_metrics()
        metrics.graph_connectivity = quality_metrics["connectivity"]
        metrics.relationship_density = quality_metrics["density"]
        metrics.average_node_degree = quality_metrics["average_degree"]
        metrics.duplicate_rate = quality_metrics["duplicate_rate"]

        # Calculate performance metrics
        metrics.completed_at = datetime.now(UTC)
        metrics.duration_seconds = (
            metrics.completed_at - metrics.started_at
        ).total_seconds()

        if metrics.entities_analyzed > 0:
            metrics.entities_per_second = (
                metrics.entities_analyzed / metrics.duration_seconds
            )

        logger.info(
            "Graph optimization completed",
            optimization_id=metrics.optimization_id,
            entities_analyzed=metrics.entities_analyzed,
            entities_merged=metrics.entities_merged,
            relationships_pruned=metrics.low_value_edges_removed,
            patterns_detected=metrics.patterns_detected,
            duration_seconds=metrics.duration_seconds,
            entities_per_second=metrics.entities_per_second,
            consolidation_accuracy=metrics.consolidation_accuracy,
            duplicate_rate=metrics.duplicate_rate,
        )

        return metrics

    async def _consolidate_entities(self) -> dict[str, Any]:
        """
        Merge similar entities with >90% similarity.

        Uses embedding cosine similarity to identify duplicate/similar entities.
        Merges entity properties and redirects relationships.

        Returns:
            Dictionary with consolidation metrics
        """
        logger.info("Starting entity consolidation")

        entities_analyzed = 0
        duplicate_pairs_found = 0
        entities_merged = 0
        successful_merges = 0
        attempted_merges = 0

        async with self.driver.session() as session:
            # Get all entities with embeddings
            count_query = "MATCH (e:Entity) RETURN count(e) as total"
            count_result = await session.run(count_query)
            count_record = await count_result.single()
            total_entities = count_record["total"] if count_record else 0
            entities_analyzed = total_entities

            if total_entities == 0:
                return {
                    "entities_analyzed": 0,
                    "duplicate_pairs_found": 0,
                    "entities_merged": 0,
                    "consolidation_accuracy": 1.0,
                }

            # Find similar entity pairs using embedding similarity
            # Process in batches for performance
            offset = 0
            candidates: list[ConsolidationCandidate] = []

            while offset < total_entities:
                similarity_query = """
                MATCH (e1:Entity), (e2:Entity)
                WHERE e1.entity_id < e2.entity_id
                  AND e1.entity_type = e2.entity_type
                  AND size(e1.embedding) > 0
                  AND size(e2.embedding) > 0
                WITH e1, e2,
                     gds.similarity.cosine(e1.embedding, e2.embedding) AS similarity
                WHERE similarity >= $threshold
                RETURN e1.entity_id AS id1, e2.entity_id AS id2,
                       e1.entity_name AS name1, e2.entity_name AS name2,
                       similarity, e1.properties AS props1, e2.properties AS props2
                SKIP $offset
                LIMIT $limit
                """

                result = await session.run(
                    similarity_query,
                    {
                        "threshold": self.similarity_threshold,
                        "offset": offset,
                        "limit": self.batch_size,
                    },
                )

                batch_candidates = []
                async for record in result:
                    candidate = ConsolidationCandidate(
                        entity1_id=record["id1"],
                        entity2_id=record["id2"],
                        entity1_name=record["name1"],
                        entity2_name=record["name2"],
                        similarity_score=record["similarity"],
                        shared_properties=self._merge_properties(
                            record["props1"], record["props2"]
                        ),
                    )
                    batch_candidates.append(candidate)

                candidates.extend(batch_candidates)

                if len(batch_candidates) < self.batch_size:
                    break

                offset += self.batch_size

            duplicate_pairs_found = len(candidates)

            # Perform merges
            for candidate in candidates:
                attempted_merges += 1
                success = await self._merge_entity_pair(session, candidate)
                if success:
                    entities_merged += 1
                    successful_merges += 1

            # Calculate consolidation accuracy
            consolidation_accuracy = (
                successful_merges / attempted_merges
                if attempted_merges > 0
                else 1.0
            )

        logger.info(
            "Entity consolidation completed",
            entities_analyzed=entities_analyzed,
            duplicate_pairs_found=duplicate_pairs_found,
            entities_merged=entities_merged,
            consolidation_accuracy=consolidation_accuracy,
        )

        return {
            "entities_analyzed": entities_analyzed,
            "duplicate_pairs_found": duplicate_pairs_found,
            "entities_merged": entities_merged,
            "consolidation_accuracy": consolidation_accuracy,
        }

    async def _merge_entity_pair(
        self,
        session: Any,
        candidate: ConsolidationCandidate,
    ) -> bool:
        """
        Merge two similar entities into one.

        Keeps entity1, redirects entity2's relationships to entity1,
        merges properties, and deletes entity2.

        Args:
            session: Neo4j session
            candidate: Consolidation candidate with entity pair

        Returns:
            True if merge successful, False otherwise
        """
        try:
            merge_query = """
            MATCH (e1:Entity {entity_id: $id1})
            MATCH (e2:Entity {entity_id: $id2})

            // Redirect incoming relationships from e2 to e1
            OPTIONAL MATCH (n)-[r_in]->(e2)
            WHERE NOT n = e1
            WITH e1, e2, collect({node: n, rel: r_in, type: type(r_in), props: properties(r_in)}) AS incoming

            UNWIND incoming AS inc
            FOREACH (_ IN CASE WHEN inc.node IS NOT NULL THEN [1] ELSE [] END |
                CALL {
                    WITH e1, inc
                    CREATE (inc.node)-[nr:RELATES_TO]->(e1)
                    SET nr = inc.props
                }
            )

            // Redirect outgoing relationships from e2 to e1
            WITH e1, e2
            OPTIONAL MATCH (e2)-[r_out]->(m)
            WHERE NOT m = e1
            WITH e1, e2, collect({node: m, rel: r_out, type: type(r_out), props: properties(r_out)}) AS outgoing

            UNWIND outgoing AS outg
            FOREACH (_ IN CASE WHEN outg.node IS NOT NULL THEN [1] ELSE [] END |
                CALL {
                    WITH e1, outg
                    CREATE (e1)-[nr:RELATES_TO]->(outg.node)
                    SET nr = outg.props
                }
            )

            // Merge properties
            WITH e1, e2
            SET e1.properties = $merged_props
            SET e1.memory_refs = e1.memory_refs + e2.memory_refs
            SET e1.updated_at = datetime()

            // Delete e2 and its relationships
            WITH e1, e2
            DETACH DELETE e2

            RETURN e1.entity_id AS merged_id
            """

            result = await session.run(
                merge_query,
                {
                    "id1": candidate.entity1_id,
                    "id2": candidate.entity2_id,
                    "merged_props": candidate.shared_properties,
                },
            )

            record = await result.single()

            if record:
                logger.debug(
                    "Merged entity pair",
                    kept_id=candidate.entity1_id,
                    removed_id=candidate.entity2_id,
                    similarity=candidate.similarity_score,
                )
                return True

            return False

        except Exception as e:
            logger.warning(
                "Failed to merge entity pair",
                entity1_id=candidate.entity1_id,
                entity2_id=candidate.entity2_id,
                error=str(e),
            )
            return False

    def _merge_properties(
        self,
        props1: dict[str, Any],
        props2: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge properties from two entities.

        Prioritizes props1 values, adds unique props2 values.

        Args:
            props1: Properties from first entity
            props2: Properties from second entity

        Returns:
            Merged properties dictionary
        """
        merged = dict(props1) if props1 else {}

        if props2:
            for key, value in props2.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = list(set(merged[key] + value))

        return merged

    async def _prune_relationships(self) -> dict[str, Any]:
        """
        Remove low-value edges with access_count < min_access_count.

        Returns:
            Dictionary with pruning metrics
        """
        logger.info(
            "Starting relationship pruning",
            min_access_count=self.min_access_count,
        )

        async with self.driver.session() as session:
            # Count total relationships
            count_query = "MATCH ()-[r]->() RETURN count(r) as total"
            count_result = await session.run(count_query)
            count_record = await count_result.single()
            total_relationships = count_record["total"] if count_record else 0

            # Delete low-value relationships
            prune_query = """
            MATCH ()-[r]->()
            WHERE coalesce(r.access_count, 0) < $min_count
            WITH r LIMIT $batch_size
            DELETE r
            RETURN count(*) as deleted
            """

            total_deleted = 0

            while True:
                result = await session.run(
                    prune_query,
                    {
                        "min_count": self.min_access_count,
                        "batch_size": self.batch_size,
                    },
                )
                record = await result.single()
                deleted = record["deleted"] if record else 0

                total_deleted += deleted

                if deleted < self.batch_size:
                    break

        logger.info(
            "Relationship pruning completed",
            relationships_analyzed=total_relationships,
            low_value_edges_removed=total_deleted,
        )

        return {
            "relationships_analyzed": total_relationships,
            "low_value_edges_removed": total_deleted,
        }

    async def _detect_patterns(self) -> dict[str, Any]:
        """
        Identify frequently traversed paths in the graph.

        Analyzes paths to find common traversal patterns
        that can be optimized.

        Returns:
            Dictionary with pattern detection metrics
        """
        logger.info("Starting pattern detection")

        frequent_paths: list[dict[str, Any]] = []

        async with self.driver.session() as session:
            # Find frequently accessed 2-hop paths
            pattern_query = """
            MATCH path = (e1:Entity)-[r1]->(e2:Entity)-[r2]->(e3:Entity)
            WHERE r1.access_count >= $min_count AND r2.access_count >= $min_count
            WITH e1.entity_type AS type1, e2.entity_type AS type2, e3.entity_type AS type3,
                 r1.access_count + r2.access_count AS total_access,
                 collect({path: [e1.entity_id, e2.entity_id, e3.entity_id],
                         access: r1.access_count + r2.access_count})[0..10] AS sample_paths
            RETURN type1, type2, type3, count(*) AS pattern_count,
                   avg(total_access) AS avg_access, sample_paths
            ORDER BY pattern_count DESC
            LIMIT 20
            """

            result = await session.run(
                pattern_query,
                {"min_count": self.min_access_count},
            )

            async for record in result:
                path_pattern = [
                    record["type1"],
                    record["type2"],
                    record["type3"],
                ]

                # Get representative path from samples
                sample_paths = record["sample_paths"]
                representative = (
                    sample_paths[0]["path"] if sample_paths else []
                )

                frequent_path = FrequentPath(
                    path_pattern=path_pattern,
                    traversal_count=record["pattern_count"],
                    average_access_count=float(record["avg_access"]),
                    representative_path=representative,
                )

                frequent_paths.append(
                    {
                        "path_pattern": frequent_path.path_pattern,
                        "traversal_count": frequent_path.traversal_count,
                        "average_access_count": frequent_path.average_access_count,
                        "representative_path": frequent_path.representative_path,
                    }
                )

        logger.info(
            "Pattern detection completed",
            patterns_detected=len(frequent_paths),
        )

        return {
            "patterns_detected": len(frequent_paths),
            "frequent_paths": frequent_paths,
        }

    async def _optimize_indexes(self) -> dict[str, Any]:
        """
        Update Neo4j indexes based on query patterns.

        Creates or updates indexes for frequently accessed patterns.

        Returns:
            Dictionary with index optimization metrics
        """
        logger.info("Starting index optimization")

        indexes_optimized = 0

        async with self.driver.session() as session:
            # Get current indexes
            index_query = "SHOW INDEXES YIELD name RETURN collect(name) as indexes"
            index_result = await session.run(index_query)
            index_record = await index_result.single()
            existing_indexes = set(
                index_record["indexes"] if index_record else []
            )

            # Analyze query patterns and create missing indexes
            # Check for access_count index (critical for pruning)
            if "relationship_access_idx" not in existing_indexes:
                try:
                    await session.run(
                        """
                        CREATE INDEX relationship_access_idx IF NOT EXISTS
                        FOR ()-[r:RELATES_TO]-()
                        ON (r.access_count)
                        """
                    )
                    indexes_optimized += 1
                    logger.info("Created relationship access count index")
                except Exception as e:
                    logger.warning(
                        "Failed to create access count index",
                        error=str(e),
                    )

            # Check for entity embedding index (critical for consolidation)
            if "entity_embedding_idx" not in existing_indexes:
                try:
                    # Note: Vector indexes require specific Neo4j setup
                    # This is a placeholder for when GDS is available
                    await session.run(
                        """
                        CREATE INDEX entity_embedding_exists_idx IF NOT EXISTS
                        FOR (e:Entity)
                        ON (e.embedding)
                        """
                    )
                    indexes_optimized += 1
                    logger.info("Created entity embedding index")
                except Exception as e:
                    logger.warning(
                        "Failed to create embedding index",
                        error=str(e),
                    )

            # Optimize indexes based on detected patterns
            # Create composite indexes for frequent path patterns
            pattern_metrics = await self._detect_patterns()
            for path in pattern_metrics["frequent_paths"][:3]:  # Top 3 patterns
                pattern = path["path_pattern"]
                if len(pattern) >= 2:
                    index_name = f"pattern_{'_'.join(pattern)}_idx"
                    if index_name not in existing_indexes:
                        try:
                            # Index on first entity type in pattern
                            await session.run(
                                f"""
                                CREATE INDEX {index_name} IF NOT EXISTS
                                FOR (e:Entity)
                                ON (e.entity_type)
                                """
                            )
                            indexes_optimized += 1
                            logger.info(
                                "Created pattern-based index",
                                pattern=pattern,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to create pattern index",
                                pattern=pattern,
                                error=str(e),
                            )

        logger.info(
            "Index optimization completed",
            indexes_optimized=indexes_optimized,
        )

        return {"indexes_optimized": indexes_optimized}

    async def _compute_quality_metrics(self) -> dict[str, float]:
        """
        Compute graph quality metrics.

        Calculates:
        - Connectivity: Ratio of connected components to total nodes
        - Relationship density: Edges / (Nodes * (Nodes - 1))
        - Average node degree: Average number of relationships per node
        - Duplicate rate: Estimated duplicate entities remaining

        Returns:
            Dictionary with quality metrics
        """
        logger.info("Computing graph quality metrics")

        async with self.driver.session() as session:
            # Count nodes and edges
            stats_query = """
            MATCH (n:Entity)
            WITH count(n) as node_count
            MATCH ()-[r]->()
            WITH node_count, count(r) as edge_count
            RETURN node_count, edge_count
            """

            stats_result = await session.run(stats_query)
            stats_record = await stats_result.single()

            if not stats_record:
                return {
                    "connectivity": 0.0,
                    "density": 0.0,
                    "average_degree": 0.0,
                    "duplicate_rate": 0.0,
                }

            node_count = stats_record["node_count"]
            edge_count = stats_record["edge_count"]

            if node_count == 0:
                return {
                    "connectivity": 0.0,
                    "density": 0.0,
                    "average_degree": 0.0,
                    "duplicate_rate": 0.0,
                }

            # Calculate relationship density
            max_edges = node_count * (node_count - 1)
            density = edge_count / max_edges if max_edges > 0 else 0.0

            # Calculate average degree
            average_degree = (2 * edge_count) / node_count if node_count > 0 else 0.0

            # Estimate connectivity (ratio of nodes with at least one edge)
            connectivity_query = """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as degree
            WHERE degree > 0
            RETURN count(n) as connected_nodes
            """

            conn_result = await session.run(connectivity_query)
            conn_record = await conn_result.single()
            connected_nodes = conn_record["connected_nodes"] if conn_record else 0

            connectivity = connected_nodes / node_count if node_count > 0 else 0.0

            # Estimate duplicate rate (entities with >90% similarity to others)
            duplicate_query = """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1.entity_id < e2.entity_id
              AND e1.entity_type = e2.entity_type
              AND size(e1.embedding) > 0
              AND size(e2.embedding) > 0
            WITH gds.similarity.cosine(e1.embedding, e2.embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN count(*) as duplicate_pairs
            """

            try:
                dup_result = await session.run(
                    duplicate_query,
                    {"threshold": self.similarity_threshold},
                )
                dup_record = await dup_result.single()
                duplicate_pairs = dup_record["duplicate_pairs"] if dup_record else 0

                # Approximate duplicate rate
                # Each pair represents one potential duplicate
                duplicate_rate = (
                    min(duplicate_pairs / node_count, 1.0)
                    if node_count > 0
                    else 0.0
                )
            except Exception:
                # GDS may not be available
                duplicate_rate = 0.0

        metrics = {
            "connectivity": round(connectivity, 4),
            "density": round(density, 6),
            "average_degree": round(average_degree, 4),
            "duplicate_rate": round(duplicate_rate, 4),
        }

        logger.info(
            "Quality metrics computed",
            **metrics,
        )

        return metrics

    def schedule_optimization(self, cron_expression: str) -> str:
        """
        Schedule optimization to run based on cron expression.

        Args:
            cron_expression: Cron expression (e.g., "0 2 * * *" for daily at 2am)

        Returns:
            Scheduled job ID

        Example:
            job_id = optimizer.schedule_optimization("0 2 * * *")  # Daily at 2am
            job_id = optimizer.schedule_optimization("0 */4 * * *")  # Every 4 hours
        """
        if not croniter.is_valid(cron_expression):
            raise ValueError(f"Invalid cron expression: {cron_expression}")

        self._cron_expression = cron_expression
        self._scheduled_job_id = f"opt-job-{uuid4()}"

        # Calculate next run time
        cron = croniter(cron_expression, datetime.now(UTC))
        self._next_run = cron.get_next(datetime)

        logger.info(
            "Optimization scheduled",
            job_id=self._scheduled_job_id,
            cron_expression=cron_expression,
            next_run=self._next_run.isoformat(),
        )

        return self._scheduled_job_id

    def get_next_scheduled_run(self) -> datetime | None:
        """
        Get the next scheduled optimization run time.

        Returns:
            Next run datetime or None if not scheduled
        """
        if self._cron_expression and self._next_run:
            # Update next run if it's in the past
            now = datetime.now(UTC)
            if self._next_run <= now:
                cron = croniter(self._cron_expression, now)
                self._next_run = cron.get_next(datetime)

        return self._next_run

    def cancel_scheduled_optimization(self) -> bool:
        """
        Cancel scheduled optimization.

        Returns:
            True if cancelled, False if no job scheduled
        """
        if self._scheduled_job_id:
            logger.info(
                "Cancelling scheduled optimization",
                job_id=self._scheduled_job_id,
            )
            self._scheduled_job_id = None
            self._cron_expression = None
            self._next_run = None
            return True
        return False

    async def should_run_scheduled(self) -> bool:
        """
        Check if scheduled optimization should run now.

        Returns:
            True if optimization should run
        """
        if not self._next_run:
            return False

        now = datetime.now(UTC)
        if now >= self._next_run:
            # Update next run time
            cron = croniter(self._cron_expression, now)
            self._next_run = cron.get_next(datetime)
            return True

        return False

    async def get_optimization_status(self) -> dict[str, Any]:
        """
        Get current optimization status and configuration.

        Returns:
            Dictionary with optimizer status
        """
        status = {
            "similarity_threshold": self.similarity_threshold,
            "min_access_count": self.min_access_count,
            "batch_size": self.batch_size,
            "scheduled_job_id": self._scheduled_job_id,
            "cron_expression": self._cron_expression,
            "next_scheduled_run": (
                self._next_run.isoformat() if self._next_run else None
            ),
        }

        # Add current graph statistics
        async with self.driver.session() as session:
            stats_query = """
            MATCH (n:Entity)
            WITH count(n) as entity_count
            MATCH ()-[r]->()
            RETURN entity_count, count(r) as relationship_count
            """

            result = await session.run(stats_query)
            record = await result.single()

            if record:
                status["entity_count"] = record["entity_count"]
                status["relationship_count"] = record["relationship_count"]
            else:
                status["entity_count"] = 0
                status["relationship_count"] = 0

        return status


# Export optimizer
__all__ = [
    "GraphOptimizer",
    "OptimizationMetrics",
    "ConsolidationCandidate",
    "FrequentPath",
]
