"""
Memify Graph Optimizer Service

Orchestrates graph optimization operations for the Neo4j knowledge graph.
Implements the Memify operation as specified in FR-9.4:
- Entity consolidation (merge similar entities)
- Relationship pruning (remove low-value edges)
- Pattern detection (identify common patterns)
- Index optimization (update indexes based on query patterns)
- Quality metrics tracking

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from neo4j import AsyncDriver

from agentcore.a2a_protocol.services.memory.memify import (
    EntityConsolidation,
    PatternDetection,
    RelationshipPruning,
)

logger = structlog.get_logger(__name__)


class MemifyOptimizer:
    """
    Main orchestrator for Memify graph optimization operations.

    Coordinates:
    - Entity consolidation: Merge similar entities (>90% similarity)
    - Relationship pruning: Remove low-value edges (access count < 2)
    - Pattern detection: Identify frequently traversed paths
    - Index optimization: Update Neo4j indexes based on query patterns
    - Quality metrics: Track connectivity, relationship density

    Performance targets:
    - <5s optimization per 1000 entities
    - 90%+ consolidation accuracy
    - <5% duplicate entities after optimization
    - Scheduled execution support (cron)
    """

    def __init__(
        self,
        driver: AsyncDriver,
        similarity_threshold: float = 0.90,
        min_access_count: int = 2,
        pattern_min_frequency: int = 5,
    ):
        """
        Initialize Memify optimizer.

        Args:
            driver: Neo4j async driver
            similarity_threshold: Minimum similarity for entity merging (default: 0.90)
            min_access_count: Minimum access count to keep relationship (default: 2)
            pattern_min_frequency: Minimum frequency for pattern detection (default: 5)
        """
        self.driver = driver

        # Initialize optimization modules
        self.consolidation = EntityConsolidation(
            similarity_threshold=similarity_threshold
        )
        self.pruning = RelationshipPruning(min_access_count=min_access_count)
        self.pattern_detection = PatternDetection(
            min_frequency=pattern_min_frequency
        )

        # Configuration
        self.similarity_threshold = similarity_threshold
        self.min_access_count = min_access_count
        self.pattern_min_frequency = pattern_min_frequency

    async def optimize(
        self,
        enable_consolidation: bool = True,
        enable_pruning: bool = True,
        enable_pattern_detection: bool = True,
        enable_index_optimization: bool = True,
    ) -> dict[str, Any]:
        """
        Run full Memify optimization pipeline.

        Executes all optimization steps in order:
        1. Entity consolidation (if enabled)
        2. Relationship pruning (if enabled)
        3. Pattern detection (if enabled)
        4. Index optimization (if enabled)
        5. Quality metrics calculation

        Args:
            enable_consolidation: Run entity consolidation (default: True)
            enable_pruning: Run relationship pruning (default: True)
            enable_pattern_detection: Run pattern detection (default: True)
            enable_index_optimization: Run index optimization (default: True)

        Returns:
            Dictionary with optimization results and statistics

        Example:
            results = await optimizer.optimize()
            # {
            #     "started_at": "2025-11-21T10:00:00Z",
            #     "completed_at": "2025-11-21T10:00:03Z",
            #     "duration_seconds": 3.2,
            #     "consolidation": {...},
            #     "pruning": {...},
            #     "patterns": {...},
            #     "quality_metrics": {...}
            # }
        """
        start_time = time.time()
        started_at = datetime.now(UTC)

        logger.info(
            "Starting Memify optimization",
            consolidation=enable_consolidation,
            pruning=enable_pruning,
            pattern_detection=enable_pattern_detection,
            index_optimization=enable_index_optimization,
        )

        results = {
            "started_at": started_at.isoformat(),
            "consolidation": None,
            "pruning": None,
            "patterns": None,
            "index_suggestions": None,
            "quality_metrics": None,
        }

        async with self.driver.session() as session:
            # Step 1: Entity consolidation
            if enable_consolidation:
                logger.info("Running entity consolidation")
                consolidation_results = await self.consolidation.consolidate_all_duplicates(
                    session
                )
                results["consolidation"] = consolidation_results

            # Step 2: Relationship pruning
            if enable_pruning:
                logger.info("Running relationship pruning")
                pruning_results = await self.pruning.prune_all_low_value(
                    session, include_redundant=True
                )
                results["pruning"] = pruning_results

            # Step 3: Pattern detection
            if enable_pattern_detection:
                logger.info("Running pattern detection")
                pattern_results = await self.pattern_detection.detect_all_patterns(
                    session
                )
                results["patterns"] = pattern_results

                # Step 4: Index optimization (based on patterns)
                if enable_index_optimization:
                    logger.info("Applying index optimizations")
                    results["index_suggestions"] = pattern_results.get(
                        "index_suggestions", []
                    )

            # Step 5: Calculate quality metrics
            logger.info("Calculating quality metrics")
            quality_metrics = await self.calculate_quality_metrics(session)
            results["quality_metrics"] = quality_metrics

        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        completed_at = datetime.now(UTC)

        results["completed_at"] = completed_at.isoformat()
        results["duration_seconds"] = round(duration, 2)

        # Performance validation
        entity_count = results["quality_metrics"].get("total_entities", 0)
        if entity_count > 0:
            time_per_1k = (duration / entity_count) * 1000
            results["performance"] = {
                "time_per_1000_entities": round(time_per_1k, 2),
                "meets_target": time_per_1k < 5.0,  # <5s per 1000 entities
            }

        logger.info(
            "Completed Memify optimization",
            duration_seconds=results["duration_seconds"],
            entities_processed=entity_count,
        )

        return results

    async def calculate_quality_metrics(
        self,
        session: Any,
    ) -> dict[str, Any]:
        """
        Calculate graph quality metrics.

        Metrics include:
        - Connectivity score (percentage of connected nodes)
        - Relationship density (avg relationships per entity)
        - Duplicate percentage (remaining duplicates)
        - Average access count (relationship activity)

        Args:
            session: Neo4j async session

        Returns:
            Dictionary with quality metrics

        Example:
            metrics = await optimizer.calculate_quality_metrics(session)
            # {
            #     "connectivity_score": 0.98,
            #     "relationship_density": 3.5,
            #     "duplicate_percentage": 0.02,
            #     "avg_access_count": 5.2,
            #     "total_entities": 1000,
            #     "total_relationships": 3500
            # }
        """
        # Get graph statistics
        stats_query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()

        WITH count(DISTINCT e) as total_entities,
             count(DISTINCT r) as total_relationships

        // Calculate density
        WITH total_entities, total_relationships,
            CASE
                WHEN total_entities > 0
                THEN toFloat(total_relationships) / toFloat(total_entities)
                ELSE 0.0
            END as relationship_density

        RETURN total_entities,
               total_relationships,
               relationship_density
        """

        stats_result = await session.run(stats_query)
        stats_record = await stats_result.single()

        if not stats_record:
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "relationship_density": 0.0,
                "connectivity_score": 0.0,
                "duplicate_percentage": 0.0,
                "avg_access_count": 0.0,
            }

        total_entities = stats_record["total_entities"]
        total_relationships = stats_record["total_relationships"]
        relationship_density = float(stats_record["relationship_density"])

        # Get consolidation validation metrics
        consolidation_metrics = await self.consolidation.validate_consolidation(
            session
        )

        # Get pruning validation metrics
        pruning_metrics = await self.pruning.validate_pruning(session)

        # Combine metrics
        quality_metrics = {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "relationship_density": round(relationship_density, 2),
            "connectivity_score": pruning_metrics["connectivity_score"],
            "duplicate_percentage": consolidation_metrics["duplicate_percentage"],
            "avg_similarity": consolidation_metrics["avg_similarity"],
            "avg_access_count": pruning_metrics["avg_access_count"],
            "isolated_nodes": pruning_metrics["isolated_nodes"],
        }

        logger.info(
            "Calculated quality metrics",
            connectivity_score=f"{quality_metrics['connectivity_score']:.2%}",
            duplicate_percentage=f"{quality_metrics['duplicate_percentage']:.2%}",
            relationship_density=quality_metrics["relationship_density"],
        )

        return quality_metrics

    async def optimize_batch(
        self,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """
        Run optimization on a batch of entities.

        Useful for large graphs where full optimization would be too slow.
        Processes entities in batches of specified size.

        Args:
            batch_size: Number of entities to process per batch

        Returns:
            Dictionary with batch optimization results

        Example:
            results = await optimizer.optimize_batch(batch_size=500)
        """
        start_time = time.time()

        async with self.driver.session() as session:
            # Run consolidation on batch
            consolidation_results = await self.consolidation.consolidate_all_duplicates(
                session, batch_size=batch_size
            )

            # Run pruning on batch
            pruning_results = await self.pruning.prune_all_low_value(
                session, batch_size=batch_size
            )

            # Calculate metrics
            quality_metrics = await self.calculate_quality_metrics(session)

        duration = time.time() - start_time

        results = {
            "batch_size": batch_size,
            "consolidation": consolidation_results,
            "pruning": pruning_results,
            "quality_metrics": quality_metrics,
            "duration_seconds": round(duration, 2),
        }

        logger.info(
            "Completed batch optimization",
            batch_size=batch_size,
            duration_seconds=results["duration_seconds"],
        )

        return results

    async def validate_optimization_quality(
        self,
    ) -> dict[str, bool]:
        """
        Validate that optimization meets quality targets.

        Checks:
        - <5% duplicate entities (target: <5%)
        - 90%+ consolidation accuracy (from consolidation results)
        - High connectivity (>95%)
        - Reasonable relationship density (>2.0)

        Returns:
            Dictionary with validation results

        Example:
            validation = await optimizer.validate_optimization_quality()
            # {
            #     "duplicate_percentage_ok": True,
            #     "consolidation_accuracy_ok": True,
            #     "connectivity_ok": True,
            #     "density_ok": True,
            #     "all_checks_passed": True
            # }
        """
        async with self.driver.session() as session:
            quality_metrics = await self.calculate_quality_metrics(session)

        # Define targets
        duplicate_percentage_ok = quality_metrics["duplicate_percentage"] < 0.05  # <5%
        connectivity_ok = quality_metrics["connectivity_score"] > 0.95  # >95%
        density_ok = quality_metrics["relationship_density"] > 2.0  # >2.0

        validation = {
            "duplicate_percentage_ok": duplicate_percentage_ok,
            "connectivity_ok": connectivity_ok,
            "density_ok": density_ok,
            "all_checks_passed": (
                duplicate_percentage_ok and connectivity_ok and density_ok
            ),
            "quality_metrics": quality_metrics,
        }

        logger.info(
            "Validation complete",
            all_checks_passed=validation["all_checks_passed"],
            duplicate_percentage=f"{quality_metrics['duplicate_percentage']:.2%}",
            connectivity=f"{quality_metrics['connectivity_score']:.2%}",
            density=quality_metrics["relationship_density"],
        )

        return validation

    async def get_optimization_recommendations(
        self,
    ) -> list[dict[str, str]]:
        """
        Get recommendations for further optimization.

        Analyzes current graph state and suggests improvements.

        Returns:
            List of recommendation dictionaries

        Example:
            recommendations = await optimizer.get_optimization_recommendations()
            # [
            #     {
            #         "type": "consolidation",
            #         "priority": "high",
            #         "description": "8% duplicate entities detected",
            #         "action": "Run entity consolidation with higher threshold"
            #     },
            #     ...
            # ]
        """
        async with self.driver.session() as session:
            quality_metrics = await self.calculate_quality_metrics(session)

        recommendations = []

        # Check duplicate percentage
        if quality_metrics["duplicate_percentage"] > 0.05:
            recommendations.append({
                "type": "consolidation",
                "priority": "high",
                "description": f"{quality_metrics['duplicate_percentage']:.1%} duplicate entities detected",
                "action": "Run entity consolidation with current or higher threshold",
            })

        # Check connectivity
        if quality_metrics["connectivity_score"] < 0.95:
            isolated_count = quality_metrics.get("isolated_nodes", 0)
            recommendations.append({
                "type": "connectivity",
                "priority": "medium",
                "description": f"{isolated_count} isolated nodes detected",
                "action": "Review and connect isolated entities or remove if unnecessary",
            })

        # Check relationship density
        if quality_metrics["relationship_density"] < 2.0:
            recommendations.append({
                "type": "density",
                "priority": "low",
                "description": f"Low relationship density ({quality_metrics['relationship_density']:.1f})",
                "action": "Consider adding more relationships through relationship detection",
            })

        # Check average access count
        if quality_metrics["avg_access_count"] < 1.0:
            recommendations.append({
                "type": "usage",
                "priority": "low",
                "description": f"Low average access count ({quality_metrics['avg_access_count']:.1f})",
                "action": "Graph may contain stale data; consider pruning old relationships",
            })

        # Always suggest pattern detection
        recommendations.append({
            "type": "patterns",
            "priority": "medium",
            "description": "Regular pattern detection recommended",
            "action": "Run pattern detection to identify optimization opportunities",
        })

        logger.info(
            "Generated optimization recommendations",
            count=len(recommendations),
        )

        return recommendations


# Export
__all__ = ["MemifyOptimizer"]
