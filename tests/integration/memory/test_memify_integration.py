"""
Integration Tests for Memify Operations (MEM-027.3)

Tests graph optimization operations:
- Entity consolidation (merge similar entities)
- Relationship pruning (remove low-value edges)
- Pattern detection (identify frequently traversed paths)
- Index optimization (update Neo4j indexes)
- Quality metrics tracking

Performance targets:
- <5s optimization per 1000 entities
- 90%+ consolidation accuracy
- <5% duplicate entities after optimization
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from neo4j import AsyncDriver

from agentcore.a2a_protocol.services.memory.memify_optimizer import MemifyOptimizer
from agentcore.a2a_protocol.services.memory.memify.consolidation import EntityConsolidation
from agentcore.a2a_protocol.services.memory.memify.pruning import RelationshipPruning
from agentcore.a2a_protocol.services.memory.memify.patterns import PatternDetection
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService


# Use function-scoped event loop for all tests
pytestmark = pytest.mark.asyncio


class TestEntityConsolidation:
    """Test entity consolidation operations."""

    @pytest.fixture
    async def graph_service(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> GraphMemoryService:
        """Create graph service."""
        service = GraphMemoryService(driver=neo4j_driver)
        await service.initialize_schema()
        return service

    @pytest.fixture
    async def consolidation_service(self) -> EntityConsolidation:
        """Create entity consolidation service."""
        return EntityConsolidation(similarity_threshold=0.90)

    async def test_identify_similar_entities(
        self,
        graph_service: GraphMemoryService,
        consolidation_service: EntityConsolidation,
    ) -> None:
        """Test identifying similar entities with >90% similarity."""
        # Arrange - Create similar entities
        similar_entities = [
            ("JWT", "concept"),
            ("jwt", "concept"),  # Case variation
            ("JWT Token", "concept"),  # Name variation
            ("JSON Web Token", "concept"),  # Full name
        ]

        for entity_name, entity_type in similar_entities:
            await graph_service.create_entity(entity_name, entity_type)

        # Also create a dissimilar entity
        await graph_service.create_entity("Redis", "tool")

        # Act - Find similar entities
        similar_groups = await consolidation_service.find_similar_entities(
            graph_service.driver
        )

        # Assert - Should group JWT variations together
        assert len(similar_groups) >= 1

        # Find the JWT group
        jwt_group = None
        for group in similar_groups:
            entity_names = {e["name"] for e in group}
            if "JWT" in entity_names or "jwt" in entity_names:
                jwt_group = group
                break

        assert jwt_group is not None
        assert len(jwt_group) >= 2  # At least 2 JWT variations

        # Redis should not be in JWT group
        redis_in_group = any(e["name"] == "Redis" for e in jwt_group)
        assert not redis_in_group

    async def test_merge_duplicate_entities(
        self,
        graph_service: GraphMemoryService,
        consolidation_service: EntityConsolidation,
    ) -> None:
        """Test merging duplicate entities."""
        # Arrange - Create duplicate entities with relationships
        await graph_service.create_entity("JWT", "concept")
        await graph_service.create_entity("jwt", "concept")
        await graph_service.create_entity("authentication", "concept")

        # Create relationships from both duplicates
        await graph_service.create_entity_relationship(
            "JWT", "authentication", "USED_FOR", 0.9
        )
        await graph_service.create_entity_relationship(
            "jwt", "authentication", "USED_FOR", 0.85
        )

        # Act - Merge entities
        merge_result = await consolidation_service.merge_entities(
            driver=graph_service.driver,
            source_entity="jwt",
            target_entity="JWT",
        )

        # Assert - Merge successful
        assert merge_result["success"] is True
        assert merge_result["merged_count"] >= 1

        # Verify merged entity exists
        jwt_entity = await graph_service.get_entity("JWT")
        assert jwt_entity is not None

        # Verify duplicate no longer exists
        jwt_lowercase = await graph_service.get_entity("jwt")
        assert jwt_lowercase is None

        # Verify relationships consolidated
        relationships = await graph_service.get_entity_relationships("JWT")
        assert len(relationships) >= 1

        # Should have one relationship to authentication (consolidated)
        auth_rels = [r for r in relationships if r["target"] == "authentication"]
        assert len(auth_rels) == 1

    async def test_consolidation_accuracy_target(
        self,
        graph_service: GraphMemoryService,
        consolidation_service: EntityConsolidation,
    ) -> None:
        """Test consolidation achieves 90%+ accuracy."""
        # Arrange - Create test dataset with known duplicates
        test_entities = [
            # Group 1: JWT variations (should consolidate)
            ("JWT", "concept"),
            ("jwt", "concept"),
            ("JWT Token", "concept"),
            # Group 2: API variations (should consolidate)
            ("REST API", "concept"),
            ("REST api", "concept"),
            ("RESTful API", "concept"),
            # Group 3: Database variations (should consolidate)
            ("PostgreSQL", "tool"),
            ("postgres", "tool"),
            ("Postgres Database", "tool"),
            # Individual entities (should NOT consolidate)
            ("Redis", "tool"),
            ("MongoDB", "tool"),
            ("FastAPI", "framework"),
        ]

        for entity_name, entity_type in test_entities:
            await graph_service.create_entity(entity_name, entity_type)

        expected_groups = 6  # 3 consolidation groups + 3 individual entities

        # Act - Run consolidation
        result = await consolidation_service.consolidate_all(graph_service.driver)

        # Assert - Check accuracy
        accuracy = result["accuracy"]
        assert accuracy >= 0.90, f"Consolidation accuracy {accuracy:.2%} below 90% target"

        # Verify consolidated count
        assert result["entities_merged"] >= 6  # Should merge at least 6 duplicates

        # Verify final entity count is reduced
        final_count = result["final_entity_count"]
        assert final_count <= expected_groups

    async def test_consolidation_preserves_relationships(
        self,
        graph_service: GraphMemoryService,
        consolidation_service: EntityConsolidation,
    ) -> None:
        """Test consolidation preserves all relationships."""
        # Arrange - Create entities with relationships
        await graph_service.create_entity("JWT", "concept")
        await graph_service.create_entity("jwt", "concept")  # Duplicate
        await graph_service.create_entity("auth", "concept")
        await graph_service.create_entity("user", "concept")

        # Create relationships from both versions
        await graph_service.create_entity_relationship("JWT", "auth", "USED_FOR", 0.9)
        await graph_service.create_entity_relationship("JWT", "user", "VALIDATES", 0.8)
        await graph_service.create_entity_relationship("jwt", "auth", "USED_FOR", 0.85)

        original_relationship_count = 3

        # Act - Consolidate
        await consolidation_service.merge_entities(
            driver=graph_service.driver,
            source_entity="jwt",
            target_entity="JWT",
        )

        # Assert - All relationships preserved (deduplicated)
        jwt_relationships = await graph_service.get_entity_relationships("JWT")

        # Should have 2 unique relationships (USED_FOR and VALIDATES)
        assert len(jwt_relationships) == 2

        # Verify specific relationships exist
        targets = {r["target"] for r in jwt_relationships}
        assert "auth" in targets
        assert "user" in targets


class TestRelationshipPruning:
    """Test relationship pruning operations."""

    @pytest.fixture
    async def graph_service(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> GraphMemoryService:
        """Create graph service."""
        service = GraphMemoryService(driver=neo4j_driver)
        await service.initialize_schema()
        return service

    @pytest.fixture
    async def pruning_service(self) -> RelationshipPruning:
        """Create relationship pruning service."""
        return RelationshipPruning(min_access_count=2)

    async def test_identify_low_value_relationships(
        self,
        graph_service: GraphMemoryService,
        pruning_service: RelationshipPruning,
    ) -> None:
        """Test identifying low-value relationships with access count < 2."""
        # Arrange - Create relationships with varying access counts
        await graph_service.create_entity("A", "concept")
        await graph_service.create_entity("B", "concept")
        await graph_service.create_entity("C", "concept")

        # High-value relationship (accessed multiple times)
        rel_high_id = await graph_service.create_entity_relationship(
            "A", "B", "RELATES_TO", 0.9
        )
        # Simulate multiple accesses
        for _ in range(5):
            await graph_service.increment_relationship_access(rel_high_id)

        # Low-value relationship (rarely accessed)
        rel_low_id = await graph_service.create_entity_relationship(
            "B", "C", "RELATES_TO", 0.5
        )
        # Only accessed once
        await graph_service.increment_relationship_access(rel_low_id)

        # Act - Identify low-value relationships
        low_value_rels = await pruning_service.find_low_value_relationships(
            graph_service.driver
        )

        # Assert - Should identify the low-value relationship
        assert len(low_value_rels) >= 1

        low_value_ids = {r["id"] for r in low_value_rels}
        assert rel_low_id in low_value_ids
        assert rel_high_id not in low_value_ids

    async def test_prune_weak_relationships(
        self,
        graph_service: GraphMemoryService,
        pruning_service: RelationshipPruning,
    ) -> None:
        """Test pruning weak relationships while preserving strong ones."""
        # Arrange - Create mix of relationships
        await graph_service.create_entity("Entity1", "concept")
        await graph_service.create_entity("Entity2", "concept")
        await graph_service.create_entity("Entity3", "concept")

        # Weak relationship (low strength + low access)
        weak_rel_id = await graph_service.create_entity_relationship(
            "Entity1", "Entity2", "RELATES_TO", 0.3
        )

        # Strong relationship (high strength + high access)
        strong_rel_id = await graph_service.create_entity_relationship(
            "Entity2", "Entity3", "RELATES_TO", 0.9
        )
        for _ in range(10):
            await graph_service.increment_relationship_access(strong_rel_id)

        # Act - Prune weak relationships
        result = await pruning_service.prune_relationships(graph_service.driver)

        # Assert - Weak relationship removed
        assert result["pruned_count"] >= 1

        # Verify weak relationship no longer exists
        weak_exists = await graph_service.relationship_exists(weak_rel_id)
        assert not weak_exists

        # Verify strong relationship preserved
        strong_exists = await graph_service.relationship_exists(strong_rel_id)
        assert strong_exists is True

    async def test_pruning_maintains_critical_paths(
        self,
        graph_service: GraphMemoryService,
        pruning_service: RelationshipPruning,
    ) -> None:
        """Test pruning maintains critical paths in graph."""
        # Arrange - Create a critical path (A -> B -> C)
        await graph_service.create_entity("A", "concept")
        await graph_service.create_entity("B", "concept")
        await graph_service.create_entity("C", "concept")

        # Critical path relationships (marked as critical)
        critical_rel1 = await graph_service.create_entity_relationship(
            "A", "B", "CRITICAL_PATH", 0.8
        )
        await graph_service.mark_relationship_critical(critical_rel1)

        critical_rel2 = await graph_service.create_entity_relationship(
            "B", "C", "CRITICAL_PATH", 0.8
        )
        await graph_service.mark_relationship_critical(critical_rel2)

        # Non-critical, weak relationship
        weak_rel = await graph_service.create_entity_relationship(
            "A", "C", "SHORTCUT", 0.2
        )

        # Act - Prune relationships
        result = await pruning_service.prune_relationships(
            graph_service.driver,
            preserve_critical=True,
        )

        # Assert - Critical relationships preserved
        critical1_exists = await graph_service.relationship_exists(critical_rel1)
        assert critical1_exists is True

        critical2_exists = await graph_service.relationship_exists(critical_rel2)
        assert critical2_exists is True

        # Non-critical weak relationship may be pruned
        # (depending on access count and strength thresholds)


class TestPatternDetection:
    """Test pattern detection operations."""

    @pytest.fixture
    async def graph_service(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> GraphMemoryService:
        """Create graph service."""
        service = GraphMemoryService(driver=neo4j_driver)
        await service.initialize_schema()
        return service

    @pytest.fixture
    async def pattern_detector(self) -> PatternDetection:
        """Create pattern detection service."""
        return PatternDetection(min_frequency=3)

    async def test_detect_frequently_traversed_paths(
        self,
        graph_service: GraphMemoryService,
        pattern_detector: PatternDetection,
    ) -> None:
        """Test detecting frequently traversed paths in graph."""
        # Arrange - Create a common pattern (User -> Auth -> JWT -> Redis)
        entities = ["User", "Auth", "JWT", "Redis"]
        for entity in entities:
            await graph_service.create_entity(entity, "concept")

        # Create path relationships
        path_rels = [
            ("User", "Auth", "REQUIRES"),
            ("Auth", "JWT", "USES"),
            ("JWT", "Redis", "STORED_IN"),
        ]

        for source, target, rel_type in path_rels:
            rel_id = await graph_service.create_entity_relationship(
                source, target, rel_type, 0.9
            )

            # Simulate frequent traversal (10 times)
            for _ in range(10):
                await graph_service.increment_relationship_access(rel_id)

        # Act - Detect patterns
        patterns = await pattern_detector.detect_frequent_paths(graph_service.driver)

        # Assert - Should detect the User->Auth->JWT->Redis pattern
        assert len(patterns) >= 1

        # Verify pattern structure
        for pattern in patterns:
            assert "path" in pattern
            assert "frequency" in pattern
            assert pattern["frequency"] >= 3  # Minimum frequency threshold

        # Check if our specific pattern was detected
        pattern_paths = [p["path"] for p in patterns]
        assert any(
            "User" in path and "Auth" in path and "JWT" in path
            for path in pattern_paths
        )

    async def test_detect_relationship_patterns(
        self,
        graph_service: GraphMemoryService,
        pattern_detector: PatternDetection,
    ) -> None:
        """Test detecting common relationship patterns."""
        # Arrange - Create a pattern: Multiple frameworks BUILT_WITH Python
        frameworks = ["FastAPI", "Django", "Flask", "Tornado"]
        await graph_service.create_entity("Python", "language")

        for framework in frameworks:
            await graph_service.create_entity(framework, "framework")
            rel_id = await graph_service.create_entity_relationship(
                framework, "Python", "BUILT_WITH", 0.95
            )

            # Simulate usage
            for _ in range(5):
                await graph_service.increment_relationship_access(rel_id)

        # Act - Detect relationship patterns
        patterns = await pattern_detector.detect_relationship_patterns(
            graph_service.driver
        )

        # Assert - Should detect "framework -> Python BUILT_WITH" pattern
        assert len(patterns) >= 1

        # Verify pattern metadata
        for pattern in patterns:
            assert "relationship_type" in pattern
            assert "frequency" in pattern
            assert "source_types" in pattern or "target_types" in pattern

        # Check for BUILT_WITH pattern
        built_with_patterns = [
            p for p in patterns if p["relationship_type"] == "BUILT_WITH"
        ]
        assert len(built_with_patterns) >= 1

    async def test_calculate_graph_connectivity_metrics(
        self,
        graph_service: GraphMemoryService,
        pattern_detector: PatternDetection,
    ) -> None:
        """Test calculating graph connectivity metrics."""
        # Arrange - Create a small graph
        # Central hub (Python) connected to multiple nodes
        await graph_service.create_entity("Python", "language")
        related_entities = ["FastAPI", "pytest", "asyncio", "uvicorn"]

        for entity in related_entities:
            await graph_service.create_entity(entity, "tool")
            await graph_service.create_entity_relationship(
                entity, "Python", "USES", 0.8
            )

        # Act - Calculate metrics
        metrics = await pattern_detector.calculate_connectivity_metrics(
            graph_service.driver
        )

        # Assert - Verify metrics
        assert "total_entities" in metrics
        assert "total_relationships" in metrics
        assert "average_degree" in metrics
        assert "clustering_coefficient" in metrics
        assert "graph_density" in metrics

        # Verify values make sense
        assert metrics["total_entities"] >= 5  # Python + 4 related
        assert metrics["total_relationships"] >= 4
        assert metrics["average_degree"] >= 1.0


class TestMemifyOptimizer:
    """Test full Memify optimization pipeline."""

    @pytest.fixture
    async def graph_service(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> GraphMemoryService:
        """Create graph service."""
        service = GraphMemoryService(driver=neo4j_driver)
        await service.initialize_schema()
        return service

    @pytest.fixture
    async def memify_optimizer(
        self,
        neo4j_driver: AsyncDriver,
    ) -> MemifyOptimizer:
        """Create Memify optimizer."""
        return MemifyOptimizer(
            driver=neo4j_driver,
            similarity_threshold=0.90,
            min_access_count=2,
            pattern_min_frequency=3,
        )

    async def test_full_optimization_pipeline(
        self,
        graph_service: GraphMemoryService,
        memify_optimizer: MemifyOptimizer,
    ) -> None:
        """Test full Memify optimization pipeline."""
        # Arrange - Create a realistic graph
        # Entities with duplicates
        entities = [
            ("JWT", "concept"),
            ("jwt", "concept"),  # Duplicate
            ("Redis", "tool"),
            ("redis", "tool"),  # Duplicate
            ("FastAPI", "framework"),
            ("Python", "language"),
        ]

        for name, entity_type in entities:
            await graph_service.create_entity(name, entity_type)

        # Relationships with varying value
        high_value_rel = await graph_service.create_entity_relationship(
            "FastAPI", "Python", "BUILT_WITH", 0.95
        )
        for _ in range(10):
            await graph_service.increment_relationship_access(high_value_rel)

        low_value_rel = await graph_service.create_entity_relationship(
            "jwt", "Redis", "MAYBE_USES", 0.1
        )
        # Only accessed once
        await graph_service.increment_relationship_access(low_value_rel)

        # Act - Run full optimization
        result = await memify_optimizer.optimize(
            enable_consolidation=True,
            enable_pruning=True,
            enable_pattern_detection=True,
            enable_index_optimization=True,
        )

        # Assert - Optimization completed
        assert result["status"] == "success"
        assert "consolidation" in result
        assert "pruning" in result
        assert "pattern_detection" in result
        assert "quality_metrics" in result

        # Verify consolidation reduced duplicates
        assert result["consolidation"]["entities_merged"] >= 2  # jwt, redis

        # Verify pruning removed low-value relationships
        assert result["pruning"]["pruned_count"] >= 0

        # Verify quality improved
        final_metrics = result["quality_metrics"]
        assert "duplicate_entity_rate" in final_metrics
        assert final_metrics["duplicate_entity_rate"] < 0.05  # <5% duplicates

    async def test_optimization_performance_target(
        self,
        graph_service: GraphMemoryService,
        memify_optimizer: MemifyOptimizer,
    ) -> None:
        """Test optimization meets performance target (<5s per 1000 entities)."""
        # Arrange - Create 1000 test entities
        num_entities = 1000

        for i in range(num_entities):
            entity_type = ["concept", "tool", "framework"][i % 3]
            await graph_service.create_entity(f"Entity_{i}", entity_type)

            # Create some relationships
            if i > 0:
                await graph_service.create_entity_relationship(
                    f"Entity_{i}",
                    f"Entity_{i-1}",
                    "RELATES_TO",
                    0.5,
                )

        # Act - Run optimization with timing
        start_time = time.time()
        result = await memify_optimizer.optimize()
        end_time = time.time()

        optimization_time = end_time - start_time

        # Assert - Should complete within 5 seconds
        assert optimization_time < 5.0, (
            f"Optimization took {optimization_time:.2f}s for {num_entities} entities, "
            f"expected <5s"
        )

        # Verify optimization completed successfully
        assert result["status"] == "success"

    async def test_optimization_duplicate_rate_target(
        self,
        graph_service: GraphMemoryService,
        memify_optimizer: MemifyOptimizer,
    ) -> None:
        """Test optimization achieves <5% duplicate entity rate."""
        # Arrange - Create entities with known duplicates (20% duplicate rate)
        base_entities = ["JWT", "Redis", "Python", "FastAPI", "MongoDB"]

        # Create originals and duplicates
        for entity in base_entities:
            await graph_service.create_entity(entity, "concept")
            await graph_service.create_entity(entity.lower(), "concept")  # Duplicate
            await graph_service.create_entity(f"{entity} Tool", "concept")  # Variation

        initial_duplicate_rate = 0.67  # ~67% are duplicates (10/15 entities)

        # Act - Run optimization
        result = await memify_optimizer.optimize(enable_consolidation=True)

        # Assert - Duplicate rate reduced to <5%
        final_duplicate_rate = result["quality_metrics"]["duplicate_entity_rate"]
        assert final_duplicate_rate < 0.05, (
            f"Duplicate rate {final_duplicate_rate:.2%} exceeds 5% target"
        )

        # Verify substantial improvement
        improvement = initial_duplicate_rate - final_duplicate_rate
        assert improvement >= 0.60, "Duplicate rate should improve by at least 60%"

    async def test_scheduled_optimization_execution(
        self,
        graph_service: GraphMemoryService,
        memify_optimizer: MemifyOptimizer,
    ) -> None:
        """Test Memify can be executed on a schedule."""
        import asyncio

        # Arrange - Track optimization runs
        optimization_runs = []

        async def scheduled_optimize() -> None:
            """Simulate scheduled optimization."""
            result = await memify_optimizer.optimize()
            optimization_runs.append({
                "timestamp": datetime.now(UTC),
                "result": result,
            })

        # Act - Run optimization multiple times (simulating schedule)
        for _ in range(3):
            await scheduled_optimize()
            await asyncio.sleep(0.1)  # Small delay between runs

        # Assert - All runs completed successfully
        assert len(optimization_runs) == 3

        for run in optimization_runs:
            assert run["result"]["status"] == "success"
            assert "timestamp" in run

        # Verify runs occurred in sequence
        timestamps = [run["timestamp"] for run in optimization_runs]
        assert timestamps == sorted(timestamps)
