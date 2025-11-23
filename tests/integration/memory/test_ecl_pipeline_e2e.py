"""
Integration Tests for ECL Pipeline End-to-End (MEM-027.2)

Tests the Extract, Cognify, Load pipeline processes memories correctly.
Validates:
- Extract phase: Data ingestion from multiple sources
- Cognify phase: Knowledge extraction (entities, relationships, patterns)
- Load phase: Multi-backend storage (Qdrant, Neo4j, PostgreSQL, Redis)
- Pipeline composition and execution
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from agentcore.a2a_protocol.models.memory import MemoryLayer, StageType
from agentcore.a2a_protocol.services.memory.pipeline.pipeline import Pipeline
from agentcore.a2a_protocol.services.memory.pipeline.extract import ExtractTask
from agentcore.a2a_protocol.services.memory.pipeline.cognify import CognifyTask
from agentcore.a2a_protocol.services.memory.pipeline.load import LoadTask
from agentcore.a2a_protocol.services.memory.ecl_pipeline import Pipeline as ECLPipeline
from agentcore.a2a_protocol.services.memory.entity_extractor import EntityExtractor
from agentcore.a2a_protocol.services.memory.relationship_detector import RelationshipDetectorTask
from agentcore.a2a_protocol.services.memory.storage_backend import StorageBackendService
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService


# Use function-scoped event loop for all tests and mark as integration tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestExtractPhase:
    """Test Extract phase data ingestion."""

    @pytest.fixture
    async def extract_task(self) -> ExtractTask:
        """Create extract task instance."""
        return ExtractTask()

    async def test_extract_agent_interactions(
        self,
        extract_task: ExtractTask,
    ) -> None:
        """Test extracting agent interaction data."""
        # Arrange
        agent_interaction_data = {
            "agent_id": "agent-123",
            "timestamp": datetime.now(UTC).isoformat(),
            "action": "code_review",
            "input": "Review Python security practices",
            "output": "Identified SQL injection vulnerabilities",
            "metadata": {"language": "Python", "severity": "high"},
        }

        # Act
        extracted = await extract_task.extract_agent_interaction(agent_interaction_data)

        # Assert
        assert extracted is not None
        assert extracted["agent_id"] == "agent-123"
        assert extracted["content"] is not None
        assert "code_review" in extracted["content"]
        assert extracted["memory_layer"] == MemoryLayer.EPISODIC.value

    async def test_extract_session_context(
        self,
        extract_task: ExtractTask,
    ) -> None:
        """Test extracting session context data."""
        # Arrange
        session_data = {
            "session_id": "session-456",
            "user_id": "user-1",
            "preferences": {"theme": "dark", "language": "en"},
            "authentication_level": "admin",
            "start_time": datetime.now(UTC).isoformat(),
        }

        # Act
        extracted = await extract_task.extract_session_context(session_data)

        # Assert
        assert extracted is not None
        assert extracted["session_id"] == "session-456"
        assert extracted["content"] is not None
        assert extracted["memory_layer"] == MemoryLayer.SEMANTIC.value

    async def test_extract_task_artifacts(
        self,
        extract_task: ExtractTask,
    ) -> None:
        """Test extracting task artifact data."""
        # Arrange
        artifact_data = {
            "task_id": "task-789",
            "execution_id": "exec-1",
            "artifact_name": "test_results",
            "artifact_type": "result",
            "content": {
                "tests_passed": 95,
                "tests_failed": 5,
                "coverage": 92.5,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Act
        extracted = await extract_task.extract_task_artifact(artifact_data)

        # Assert
        assert extracted is not None
        assert extracted["task_id"] == "task-789"
        assert extracted["content"] is not None
        assert "test_results" in extracted["content"]
        assert extracted["memory_layer"] == MemoryLayer.PROCEDURAL.value

    async def test_extract_error_records(
        self,
        extract_task: ExtractTask,
    ) -> None:
        """Test extracting error record data."""
        # Arrange
        error_data = {
            "error_id": "error-001",
            "error_type": "hallucination",
            "severity": 0.8,
            "context": "Generated incorrect code snippet",
            "correction": "Applied factual validation",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Act
        extracted = await extract_task.extract_error_record(error_data)

        # Assert
        assert extracted is not None
        assert extracted["error_type"] == "hallucination"
        assert extracted["severity"] == 0.8
        assert extracted["content"] is not None
        assert "hallucination" in extracted["content"]


class TestCognifyPhase:
    """Test Cognify phase knowledge extraction."""

    @pytest.fixture
    async def entity_extractor(self) -> EntityExtractor:
        """Create entity extractor."""
        return EntityExtractor(api_key="test-key")

    @pytest.fixture
    async def relationship_detector(self) -> RelationshipDetectorTask:
        """Create relationship detector."""
        return RelationshipDetectorTask(api_key="test-key")

    @pytest.fixture
    async def cognify_task(
        self,
        entity_extractor: EntityExtractor,
        relationship_detector: RelationshipDetectorTask,
    ) -> CognifyTask:
        """Create cognify task instance."""
        return CognifyTask(
            entity_extractor=entity_extractor,
            relationship_detector=relationship_detector,
        )

    async def test_entity_extraction_accuracy(
        self,
        entity_extractor: EntityExtractor,
    ) -> None:
        """Test entity extraction achieves target accuracy."""
        # Arrange - Sample memory content with known entities
        memory_content = """
        Implemented JWT authentication using Redis for session storage.
        The FastAPI backend validates tokens and maintains user state.
        """

        expected_entities = {
            "JWT": "concept",
            "Redis": "tool",
            "FastAPI": "framework",
            "authentication": "concept",
            "session": "concept",
        }

        # Act
        extracted_entities = await entity_extractor.extract_entities(memory_content)

        # Assert - Check extraction accuracy
        extracted_names = {e["name"] for e in extracted_entities}

        # Calculate accuracy (allowing for variations in entity naming)
        matches = 0
        for expected_name, expected_type in expected_entities.items():
            if any(expected_name.lower() in name.lower() for name in extracted_names):
                matches += 1

        accuracy = matches / len(expected_entities)
        assert accuracy >= 0.8, f"Entity extraction accuracy {accuracy:.2%} below 80% target"

        # Assert - Verify entity types are classified
        for entity in extracted_entities:
            assert "name" in entity
            assert "type" in entity
            assert entity["type"] in ["concept", "tool", "framework", "person", "constraint"]

    async def test_relationship_detection_accuracy(
        self,
        relationship_detector: RelationshipDetectorTask,
        entity_extractor: EntityExtractor,
    ) -> None:
        """Test relationship detection achieves target accuracy."""
        # Arrange - Extract entities first
        memory_content = "FastAPI uses asyncio for asynchronous request handling"
        entities = await entity_extractor.extract_entities(memory_content)

        # Act - Detect relationships
        relationships = await relationship_detector.detect_relationships(
            content=memory_content,
            entities=entities,
        )

        # Assert - Should detect FastAPI -> asyncio relationship
        assert len(relationships) >= 1

        # Verify relationship structure
        for rel in relationships:
            assert "source" in rel
            assert "target" in rel
            assert "type" in rel
            assert "strength" in rel
            assert 0.0 <= rel["strength"] <= 1.0

        # Check for expected relationship
        has_fastapi_asyncio = any(
            ("FastAPI".lower() in rel["source"].lower() or "fastapi" in rel["source"].lower())
            and ("asyncio".lower() in rel["target"].lower())
            for rel in relationships
        )
        assert has_fastapi_asyncio, "Failed to detect FastAPI -> asyncio relationship"

    async def test_critical_fact_extraction(
        self,
        cognify_task: CognifyTask,
    ) -> None:
        """Test critical fact extraction from memory content."""
        # Arrange
        memory_data = {
            "content": "CRITICAL: Production database password changed to prevent security breach",
            "memory_layer": "episodic",
            "keywords": ["security", "critical"],
        }

        # Act
        result = await cognify_task.extract_critical_facts(memory_data)

        # Assert
        assert result["is_critical"] is True
        assert "critical_facts" in result
        assert len(result["critical_facts"]) >= 1
        assert any("password" in fact.lower() for fact in result["critical_facts"])

    async def test_pattern_recognition(
        self,
        cognify_task: CognifyTask,
    ) -> None:
        """Test pattern recognition across multiple memories."""
        # Arrange - Multiple memories with patterns
        memories = [
            {"content": "User requested Python code review", "memory_layer": "episodic"},
            {"content": "User requested JavaScript code review", "memory_layer": "episodic"},
            {"content": "User requested Go code review", "memory_layer": "episodic"},
        ]

        # Act
        patterns = await cognify_task.detect_patterns(memories)

        # Assert - Should detect "code review" pattern
        assert len(patterns) >= 1
        assert any("code review" in pattern["description"].lower() for pattern in patterns)

        # Verify pattern metadata
        for pattern in patterns:
            assert "description" in pattern
            assert "frequency" in pattern
            assert "confidence" in pattern
            assert pattern["frequency"] >= 1


class TestLoadPhase:
    """Test Load phase multi-backend storage."""

    @pytest.fixture
    async def storage_backend(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
    ) -> StorageBackendService:
        """Create storage backend."""
        return StorageBackendService(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )

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
    async def load_task(
        self,
        storage_backend: StorageBackendService,
        graph_service: GraphMemoryService,
    ) -> LoadTask:
        """Create load task instance."""
        return LoadTask(
            vector_backend=storage_backend,
            graph_service=graph_service,
        )

    async def test_load_to_qdrant_vector_storage(
        self,
        load_task: LoadTask,
        storage_backend: StorageBackendService,
    ) -> None:
        """Test loading memory to Qdrant vector storage."""
        # Arrange
        memory_data = {
            "memory_id": "mem-load-001",
            "content": "Test memory for vector storage",
            "embedding": [0.1] * 1536,
            "metadata": {"memory_layer": "episodic"},
        }

        # Act
        await load_task.load_to_vector_storage(memory_data)

        # Assert - Verify storage
        results = await storage_backend.search_similar(
            query_embedding=memory_data["embedding"],
            limit=1,
        )
        assert len(results) >= 1
        assert results[0]["id"] == memory_data["memory_id"]

    async def test_load_to_neo4j_graph_storage(
        self,
        load_task: LoadTask,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test loading entities and relationships to Neo4j."""
        # Arrange
        graph_data = {
            "memory_id": "mem-graph-001",
            "entities": [
                {"name": "Python", "type": "language"},
                {"name": "FastAPI", "type": "framework"},
            ],
            "relationships": [
                {
                    "source": "FastAPI",
                    "target": "Python",
                    "type": "BUILT_WITH",
                    "strength": 0.95,
                }
            ],
        }

        # Act
        await load_task.load_to_graph_storage(graph_data)

        # Assert - Verify entities created
        python_entity = await graph_service.get_entity("Python")
        assert python_entity is not None
        assert python_entity["type"] == "language"

        fastapi_entity = await graph_service.get_entity("FastAPI")
        assert fastapi_entity is not None

        # Assert - Verify relationship created
        relationships = await graph_service.get_entity_relationships("FastAPI")
        assert len(relationships) >= 1
        assert any(
            r["target"] == "Python" and r["type"] == "BUILT_WITH"
            for r in relationships
        )

    async def test_load_consistency_across_backends(
        self,
        load_task: LoadTask,
        storage_backend: StorageBackendService,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test loading maintains consistency across all backends."""
        # Arrange
        unified_data = {
            "memory_id": "mem-unified-001",
            "content": "JWT authentication using Redis",
            "embedding": [0.2] * 1536,
            "metadata": {"memory_layer": "episodic"},
            "entities": [
                {"name": "JWT", "type": "concept"},
                {"name": "Redis", "type": "tool"},
            ],
            "relationships": [
                {"source": "JWT", "target": "Redis", "type": "STORED_IN", "strength": 0.8}
            ],
        }

        # Act - Load to all backends
        await load_task.load_unified(unified_data)

        # Assert - Verify vector storage
        vector_results = await storage_backend.search_similar(
            query_embedding=unified_data["embedding"],
            limit=1,
        )
        assert len(vector_results) >= 1
        assert vector_results[0]["id"] == unified_data["memory_id"]

        # Assert - Verify graph storage
        memory_entities = await graph_service.get_memory_entities(unified_data["memory_id"])
        entity_names = {e["name"] for e in memory_entities}
        assert "JWT" in entity_names
        assert "Redis" in entity_names

        # Assert - Verify relationships
        jwt_relationships = await graph_service.get_entity_relationships("JWT")
        assert any(r["target"] == "Redis" for r in jwt_relationships)


class TestPipelineComposition:
    """Test pipeline composition and execution."""

    @pytest.fixture
    async def full_pipeline(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> ECLPipeline:
        """Create full ECL pipeline."""
        storage_backend = StorageBackendService(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )

        graph_service = GraphMemoryService(driver=neo4j_driver)
        await graph_service.initialize_schema()

        pipeline = ECLPipeline(
            vector_backend=storage_backend,
            graph_service=graph_service,
            api_key="test-key",
        )

        return pipeline

    async def test_pipeline_end_to_end_execution(
        self,
        full_pipeline: ECLPipeline,
    ) -> None:
        """Test end-to-end pipeline execution."""
        # Arrange - Raw agent interaction data
        raw_data = {
            "agent_id": "agent-pipeline-test",
            "timestamp": datetime.now(UTC).isoformat(),
            "action": "implement_feature",
            "input": "Add JWT authentication to FastAPI backend",
            "output": "Successfully implemented JWT auth with Redis session storage",
            "metadata": {
                "language": "Python",
                "framework": "FastAPI",
                "success": True,
            },
        }

        # Act - Process through full pipeline
        result = await full_pipeline.process(raw_data)

        # Assert - Verify pipeline completed
        assert result["status"] == "success"
        assert "memory_id" in result
        assert "entities_extracted" in result
        assert "relationships_created" in result

        # Verify data stored in all backends
        memory_id = result["memory_id"]

        # Check vector storage
        assert result["vector_stored"] is True

        # Check graph storage
        assert result["entities_extracted"] >= 2  # At least JWT, FastAPI, Redis
        assert result["relationships_created"] >= 1

    async def test_pipeline_parallel_execution(
        self,
        full_pipeline: ECLPipeline,
    ) -> None:
        """Test pipeline executes independent tasks in parallel."""
        import time
        import asyncio

        # Arrange - Multiple independent data items
        data_items = [
            {
                "agent_id": f"agent-parallel-{i}",
                "action": "test_action",
                "input": f"Input {i}",
                "output": f"Output {i}",
            }
            for i in range(5)
        ]

        # Act - Process items in parallel
        start_time = time.time()
        results = await asyncio.gather(
            *[full_pipeline.process(item) for item in data_items]
        )
        end_time = time.time()

        parallel_time = end_time - start_time

        # Act - Process items sequentially for comparison
        start_time = time.time()
        for item in data_items:
            await full_pipeline.process(item)
        end_time = time.time()

        sequential_time = end_time - start_time

        # Assert - Parallel should be faster
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)

        # Parallel execution should be at least 30% faster
        # (Being conservative due to test environment variability)
        speedup = sequential_time / parallel_time
        assert speedup >= 1.3, f"Parallel speedup {speedup:.2f}x below 1.3x target"

    async def test_pipeline_error_handling_and_retry(
        self,
        full_pipeline: ECLPipeline,
    ) -> None:
        """Test pipeline handles errors and retries failed tasks."""
        # Arrange - Data that may cause extraction failures
        problematic_data = {
            "agent_id": "agent-error-test",
            "action": "malformed_action",
            "input": "",  # Empty input
            "output": "",  # Empty output
        }

        # Act - Process with error handling
        result = await full_pipeline.process(
            problematic_data,
            max_retries=3,
            continue_on_error=True,
        )

        # Assert - Pipeline should handle errors gracefully
        assert result["status"] in ["success", "partial_success"]

        # Check which phases completed
        assert "extract_phase" in result
        assert "cognify_phase" in result
        assert "load_phase" in result

        # Even with errors, should attempt all phases
        if result["status"] == "partial_success":
            assert "errors" in result
            assert len(result["errors"]) >= 1

    async def test_pipeline_performance_target(
        self,
        full_pipeline: ECLPipeline,
    ) -> None:
        """Test pipeline meets performance target (<5s for 100 memories)."""
        import time

        # Arrange - 100 memory items
        memories = [
            {
                "agent_id": f"agent-perf-{i}",
                "action": "test_action",
                "input": f"Performance test input {i}",
                "output": f"Performance test output {i}",
            }
            for i in range(100)
        ]

        # Act - Process all memories
        start_time = time.time()

        # Process in batches to avoid overwhelming test containers
        batch_size = 10
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            await asyncio.gather(*[full_pipeline.process(item) for item in batch])

        end_time = time.time()
        total_time = end_time - start_time

        # Assert - Should complete within 5 seconds
        assert total_time < 5.0, f"Pipeline took {total_time:.2f}s, expected <5s for 100 memories"


class TestPipelineRegistry:
    """Test task registry and dynamic composition."""

    async def test_task_registration(self) -> None:
        """Test tasks can be registered in registry."""
        from agentcore.a2a_protocol.services.memory.pipeline.task_base import TaskRegistry

        # Arrange
        registry = TaskRegistry()

        # Act - Register custom task
        @registry.register("custom_extract")
        class CustomExtractTask:
            async def execute(self, data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "custom"}

        # Assert - Task registered
        assert "custom_extract" in registry.list_tasks()

        # Can retrieve task
        task_class = registry.get_task("custom_extract")
        assert task_class is not None

    async def test_dynamic_pipeline_composition(self) -> None:
        """Test composing pipelines dynamically from registry."""
        from agentcore.a2a_protocol.services.memory.pipeline.task_base import (
            TaskRegistry,
            TaskBase,
        )
        from agentcore.a2a_protocol.services.memory.pipeline.pipeline import Pipeline

        # Arrange
        registry = TaskRegistry()

        # Register test tasks
        @registry.register("step1")
        class Step1Task(TaskBase):
            def __init__(self):
                super().__init__(name="step1", description="Test step 1")

            async def execute(self, data: dict[str, Any]) -> dict[str, Any]:
                return {**data, "step1": True}

        @registry.register("step2")
        class Step2Task(TaskBase):
            def __init__(self):
                super().__init__(name="step2", description="Test step 2")

            async def execute(self, data: dict[str, Any]) -> dict[str, Any]:
                return {**data, "step2": True}

        # Act - Compose pipeline
        pipeline = Pipeline()
        pipeline.add_task(Step1Task())
        pipeline.add_task(Step2Task())

        # Execute pipeline
        result = await pipeline.execute({"input": "test"})

        # Assert - All steps executed
        assert result.is_success()
        assert result.task_results["step1"].output["step1"] is True
        assert result.task_results["step2"].output["step2"] is True
        assert result.task_results["step1"].output["input"]["input"] == "test"
