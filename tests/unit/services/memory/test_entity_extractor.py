"""
Unit Tests for Entity Extraction Task

Tests entity extraction, classification, normalization, and deduplication.
Target: 90%+ code coverage.

Component ID: MEM-015
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.models.memory import EntityNode, EntityType
from agentcore.a2a_protocol.services.memory.entity_extractor import EntityExtractor


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI-compatible LLM client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.fixture
def entity_extractor(mock_llm_client):
    """Entity extractor with mocked LLM client."""
    return EntityExtractor(
        llm_client=mock_llm_client,
        model="gpt-4.1-mini",
        max_entities_per_memory=20,
        confidence_threshold=0.5,
    )


@pytest.fixture
def sample_memory_content():
    """Sample memory content for testing."""
    return """
    User wants to implement authentication using JWT tokens stored in Redis.
    They prefer PostgreSQL for the main database and Neo4j for the knowledge graph.
    The system should use FastAPI framework with Pydantic for validation.
    Requirements: Must support 1000+ concurrent users, <100ms response time.
    """


@pytest.fixture
def sample_llm_response():
    """Sample LLM extraction response."""
    return {
        "entities": [
            {"name": "jwt", "type": "tool", "confidence": 0.95},
            {"name": "redis", "type": "tool", "confidence": 0.95},
            {"name": "postgresql", "type": "tool", "confidence": 0.95},
            {"name": "neo4j", "type": "tool", "confidence": 0.95},
            {"name": "fastapi", "type": "tool", "confidence": 0.95},
            {"name": "pydantic", "type": "tool", "confidence": 0.95},
            {"name": "authentication", "type": "concept", "confidence": 0.90},
            {"name": "validation", "type": "concept", "confidence": 0.85},
            {"name": "1000+ concurrent users", "type": "constraint", "confidence": 0.80},
            {"name": "<100ms response time", "type": "constraint", "confidence": 0.80},
        ]
    }


class TestEntityExtractorInitialization:
    """Test entity extractor initialization."""

    def test_default_initialization(self):
        """Test extractor initializes with default values."""
        extractor = EntityExtractor()
        assert extractor.name == "entity_extractor"
        assert extractor.model == "gpt-4.1-mini"
        assert extractor.max_entities_per_memory == 20
        assert extractor.confidence_threshold == 0.5
        assert extractor.max_retries == 3

    def test_custom_initialization(self, mock_llm_client):
        """Test extractor initializes with custom values."""
        extractor = EntityExtractor(
            llm_client=mock_llm_client,
            model="gpt-5",
            max_entities_per_memory=10,
            confidence_threshold=0.7,
        )
        assert extractor.llm_client == mock_llm_client
        assert extractor.model == "gpt-5"
        assert extractor.max_entities_per_memory == 10
        assert extractor.confidence_threshold == 0.7


class TestEntityExtractionLLM:
    """Test LLM-based entity extraction."""

    @pytest.mark.asyncio
    async def test_extract_entities_success(
        self, entity_extractor, mock_llm_client, sample_memory_content, sample_llm_response
    ):
        """Test successful entity extraction with LLM."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(sample_llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Extract entities
        result = await entity_extractor.execute({
            "content": sample_memory_content,
            "memory_id": "mem-123",
        })

        # Assertions
        assert "entities" in result
        assert "normalized_entities" in result
        assert "extraction_metadata" in result
        assert len(result["entities"]) > 0
        assert result["extraction_metadata"]["model_used"] == "gpt-4.1-mini"

        # Verify LLM was called with correct parameters
        mock_llm_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_llm_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1-mini"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_extract_entities_with_confidence_filter(
        self, entity_extractor, mock_llm_client, sample_memory_content
    ):
        """Test entities filtered by confidence threshold."""
        # Mock LLM response with varying confidence
        llm_response = {
            "entities": [
                {"name": "high_conf", "type": "tool", "confidence": 0.95},
                {"name": "medium_conf", "type": "concept", "confidence": 0.60},
                {"name": "low_conf", "type": "other", "confidence": 0.30},  # Below threshold
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Set threshold to 0.5
        entity_extractor.confidence_threshold = 0.5

        result = await entity_extractor.execute({
            "content": sample_memory_content,
            "memory_id": "mem-123",
        })

        # Only entities with confidence >= 0.5 should remain
        assert len(result["entities"]) == 2
        assert all(e["confidence"] >= 0.5 for e in result["entities"])
        assert result["extraction_metadata"]["total_extracted"] == 3
        assert result["extraction_metadata"]["after_confidence_filter"] == 2

    @pytest.mark.asyncio
    async def test_extract_entities_llm_failure(
        self, entity_extractor, mock_llm_client, sample_memory_content
    ):
        """Test handling of LLM extraction failure."""
        # Mock LLM failure
        mock_llm_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM API error")
        )

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="LLM entity extraction failed"):
            await entity_extractor.execute({
                "content": sample_memory_content,
                "memory_id": "mem-123",
            })

    @pytest.mark.asyncio
    async def test_extract_entities_max_limit(
        self, entity_extractor, mock_llm_client, sample_memory_content
    ):
        """Test max entities per memory limit."""
        # Mock LLM response with many entities
        llm_response = {
            "entities": [
                {"name": f"entity_{i}", "type": "tool", "confidence": 0.9}
                for i in range(50)
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Set max to 20
        entity_extractor.max_entities_per_memory = 20

        result = await entity_extractor.execute({
            "content": sample_memory_content,
            "memory_id": "mem-123",
        })

        # Should be limited to 20
        assert len(result["entities"]) <= 20


class TestFallbackExtraction:
    """Test fallback keyword-based extraction."""

    @pytest.mark.asyncio
    async def test_fallback_extraction_no_llm(self, sample_memory_content):
        """Test fallback extraction when no LLM client provided."""
        # Create extractor without LLM client
        extractor = EntityExtractor(llm_client=None)

        result = await extractor.execute({
            "content": sample_memory_content,
            "memory_id": "mem-123",
        })

        # Should extract some entities using pattern matching
        assert len(result["entities"]) > 0

        # Check for expected tools
        entity_names = [e["name"] for e in result["entities"]]
        assert any("redis" in name for name in entity_names)
        assert any("postgresql" in name for name in entity_names)
        assert any("fastapi" in name for name in entity_names)

    @pytest.mark.asyncio
    async def test_fallback_extraction_tools(self):
        """Test fallback extraction of tool entities."""
        extractor = EntityExtractor(llm_client=None)

        content = "Using Redis, PostgreSQL, and Docker for deployment"
        result = await extractor.execute({"content": content})

        entities = result["entities"]
        assert len(entities) >= 3
        assert all(e["type"] == "tool" for e in entities)

    @pytest.mark.asyncio
    async def test_fallback_extraction_concepts(self):
        """Test fallback extraction of concept entities."""
        extractor = EntityExtractor(llm_client=None)

        content = "Implementing authentication and authorization with caching optimization"
        result = await extractor.execute({"content": content})

        entities = result["entities"]
        assert len(entities) > 0
        assert any(e["type"] == "concept" for e in entities)


class TestNormalizationAndDeduplication:
    """Test entity normalization and deduplication."""

    def test_normalize_entity_name_lowercase(self, entity_extractor):
        """Test name normalization to lowercase."""
        assert entity_extractor._normalize_entity_name("Redis") == "redis"
        assert entity_extractor._normalize_entity_name("PostgreSQL") == "postgresql"

    def test_normalize_entity_name_whitespace(self, entity_extractor):
        """Test whitespace trimming and normalization."""
        assert entity_extractor._normalize_entity_name("  redis  ") == "redis"
        assert entity_extractor._normalize_entity_name("JWT  Token") == "jwt token"

    def test_normalize_entity_name_special_chars(self, entity_extractor):
        """Test special character removal."""
        assert entity_extractor._normalize_entity_name("Redis!") == "redis"
        assert entity_extractor._normalize_entity_name("JWT@Token") == "jwttoken"
        assert entity_extractor._normalize_entity_name("gpt-4") == "gpt-4"  # Hyphens preserved

    def test_normalize_entity_name_synonyms(self, entity_extractor):
        """Test synonym mapping."""
        assert entity_extractor._normalize_entity_name("postgres") == "postgresql"
        assert entity_extractor._normalize_entity_name("pg") == "postgresql"
        assert entity_extractor._normalize_entity_name("k8s") == "kubernetes"

    @pytest.mark.asyncio
    async def test_deduplication_exact_match(self, entity_extractor, mock_llm_client):
        """Test deduplication with exact name match."""
        # Create existing entity
        existing_entity = EntityNode(
            entity_name="redis",
            entity_type=EntityType.TOOL,
            memory_refs=["mem-001"],
        )

        # Mock LLM to extract same entity
        llm_response = {
            "entities": [
                {"name": "Redis", "type": "tool", "confidence": 0.95},  # Different case
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Using Redis for caching",
            "memory_id": "mem-002",
            "existing_entities": [existing_entity],
        })

        # Should deduplicate to single entity
        normalized = result["normalized_entities"]
        assert len(normalized) == 1
        assert normalized[0].entity_name == "redis"
        # Should add new memory reference
        assert "mem-002" in normalized[0].memory_refs
        assert "mem-001" in normalized[0].memory_refs

    @pytest.mark.asyncio
    async def test_deduplication_within_batch(self, entity_extractor, mock_llm_client):
        """Test deduplication within same extraction batch."""
        # Mock LLM to extract duplicate entities
        llm_response = {
            "entities": [
                {"name": "Redis", "type": "tool", "confidence": 0.95},
                {"name": "redis", "type": "tool", "confidence": 0.90},  # Duplicate
                {"name": "REDIS", "type": "tool", "confidence": 0.85},  # Duplicate
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Using Redis Redis REDIS",
            "memory_id": "mem-001",
        })

        # Should deduplicate to single entity
        normalized = result["normalized_entities"]
        assert len(normalized) == 1
        assert normalized[0].entity_name == "redis"

    @pytest.mark.asyncio
    async def test_no_deduplication_different_entities(
        self, entity_extractor, mock_llm_client
    ):
        """Test no deduplication for different entities."""
        # Mock LLM to extract different entities
        llm_response = {
            "entities": [
                {"name": "redis", "type": "tool", "confidence": 0.95},
                {"name": "postgresql", "type": "tool", "confidence": 0.95},
                {"name": "neo4j", "type": "tool", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Using Redis, PostgreSQL, and Neo4j",
            "memory_id": "mem-001",
        })

        # Should keep all three entities
        normalized = result["normalized_entities"]
        assert len(normalized) == 3
        names = {e.entity_name for e in normalized}
        assert names == {"redis", "postgresql", "neo4j"}


class TestEntityClassification:
    """Test entity type classification."""

    def test_classify_tool_entities(self, entity_extractor):
        """Test classification of tool entities."""
        assert entity_extractor.classify_entity_type("redis") == EntityType.TOOL
        assert entity_extractor.classify_entity_type("postgresql") == EntityType.TOOL
        assert entity_extractor.classify_entity_type("fastapi") == EntityType.TOOL
        assert entity_extractor.classify_entity_type("docker") == EntityType.TOOL

    def test_classify_concept_entities(self, entity_extractor):
        """Test classification of concept entities."""
        assert entity_extractor.classify_entity_type("authentication") == EntityType.CONCEPT
        assert entity_extractor.classify_entity_type("caching") == EntityType.CONCEPT
        assert entity_extractor.classify_entity_type("optimization") == EntityType.CONCEPT
        assert entity_extractor.classify_entity_type("architecture") == EntityType.CONCEPT

    def test_classify_constraint_entities(self, entity_extractor):
        """Test classification of constraint entities."""
        assert entity_extractor.classify_entity_type("requirement") == EntityType.CONSTRAINT
        assert entity_extractor.classify_entity_type("policy") == EntityType.CONSTRAINT
        assert entity_extractor.classify_entity_type("maximum") == EntityType.CONSTRAINT

    def test_classify_person_entities(self, entity_extractor):
        """Test classification of person entities."""
        # Proper nouns with spaces likely persons
        assert entity_extractor.classify_entity_type("John Doe") == EntityType.PERSON
        assert entity_extractor.classify_entity_type("Alice Smith") == EntityType.PERSON

    def test_classify_other_entities(self, entity_extractor):
        """Test classification of unrecognized entities."""
        assert entity_extractor.classify_entity_type("unknown") == EntityType.OTHER
        assert entity_extractor.classify_entity_type("xyz") == EntityType.OTHER


class TestExtractionValidation:
    """Test extraction input validation."""

    @pytest.mark.asyncio
    async def test_missing_content(self, entity_extractor):
        """Test error when content is missing."""
        with pytest.raises(ValueError, match="Content is required"):
            await entity_extractor.execute({})

    @pytest.mark.asyncio
    async def test_empty_content(self, entity_extractor):
        """Test error when content is empty."""
        with pytest.raises(ValueError, match="Content is required"):
            await entity_extractor.execute({"content": ""})

    @pytest.mark.asyncio
    async def test_valid_minimal_input(self, entity_extractor):
        """Test execution with minimal valid input."""
        # Should work with just content (no LLM client uses fallback)
        entity_extractor.llm_client = None

        result = await entity_extractor.execute({
            "content": "Using Redis for caching",
        })

        assert "entities" in result
        assert "normalized_entities" in result
        assert "extraction_metadata" in result


class TestExtractionMetadata:
    """Test extraction metadata tracking."""

    @pytest.mark.asyncio
    async def test_metadata_structure(self, entity_extractor, mock_llm_client):
        """Test metadata contains required fields."""
        llm_response = {
            "entities": [
                {"name": "redis", "type": "tool", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Using Redis",
            "memory_id": "mem-001",
        })

        metadata = result["extraction_metadata"]
        assert "total_extracted" in metadata
        assert "after_confidence_filter" in metadata
        assert "after_deduplication" in metadata
        assert "model_used" in metadata
        assert "confidence_threshold" in metadata

    @pytest.mark.asyncio
    async def test_metadata_accuracy(self, entity_extractor, mock_llm_client):
        """Test metadata values are accurate."""
        llm_response = {
            "entities": [
                {"name": "redis", "type": "tool", "confidence": 0.95},
                {"name": "postgres", "type": "tool", "confidence": 0.60},
                {"name": "low", "type": "other", "confidence": 0.30},  # Below threshold
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        entity_extractor.confidence_threshold = 0.5

        result = await entity_extractor.execute({
            "content": "Test content",
            "memory_id": "mem-001",
        })

        metadata = result["extraction_metadata"]
        assert metadata["total_extracted"] == 3
        assert metadata["after_confidence_filter"] == 2
        assert metadata["after_deduplication"] == 2
        assert metadata["model_used"] == "gpt-4.1-mini"
        assert metadata["confidence_threshold"] == 0.5


class TestEntityNodeCreation:
    """Test EntityNode instance creation."""

    @pytest.mark.asyncio
    async def test_entity_node_properties(self, entity_extractor, mock_llm_client):
        """Test EntityNode created with correct properties."""
        llm_response = {
            "entities": [
                {"name": "Redis", "type": "tool", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Using Redis",
            "memory_id": "mem-123",
        })

        entities = result["normalized_entities"]
        assert len(entities) == 1

        entity = entities[0]
        assert entity.entity_name == "redis"  # Normalized
        assert entity.entity_type == EntityType.TOOL
        assert "mem-123" in entity.memory_refs
        assert entity.properties["confidence"] == 0.95
        assert entity.properties["original_name"] == "Redis"

    @pytest.mark.asyncio
    async def test_invalid_entity_type_defaults_to_other(
        self, entity_extractor, mock_llm_client
    ):
        """Test invalid entity type defaults to OTHER."""
        llm_response = {
            "entities": [
                {"name": "test", "type": "invalid_type", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Test",
            "memory_id": "mem-001",
        })

        entities = result["normalized_entities"]
        assert entities[0].entity_type == EntityType.OTHER


class TestPromptGeneration:
    """Test extraction prompt generation."""

    def test_prompt_contains_instructions(self, entity_extractor):
        """Test prompt contains extraction instructions."""
        content = "Test content"
        prompt = entity_extractor._build_extraction_prompt(content)

        assert "Extract entities" in prompt
        assert "person" in prompt
        assert "concept" in prompt
        assert "tool" in prompt
        assert "constraint" in prompt
        assert content in prompt

    def test_prompt_includes_entity_limit(self, entity_extractor):
        """Test prompt includes max entities limit."""
        entity_extractor.max_entities_per_memory = 15
        prompt = entity_extractor._build_extraction_prompt("Test")

        assert "15" in prompt

    def test_prompt_json_format(self, entity_extractor):
        """Test prompt specifies JSON format."""
        prompt = entity_extractor._build_extraction_prompt("Test")

        assert "JSON" in prompt or "json" in prompt
        assert "entities" in prompt


class TestCustomEntityTypes:
    """Test custom entity types functionality."""

    def test_custom_types_initialization(self):
        """Test extractor initializes with custom types."""
        custom_types = {
            "location": "Geographic locations, cities, countries",
            "organization": "Companies, teams, institutions",
        }

        extractor = EntityExtractor(custom_types=custom_types)
        assert extractor.custom_types == custom_types

    def test_custom_types_in_prompt(self):
        """Test custom types appear in extraction prompt."""
        custom_types = {
            "location": "Geographic locations",
            "organization": "Companies and teams",
        }

        extractor = EntityExtractor(custom_types=custom_types)
        prompt = extractor._build_extraction_prompt("Test content")

        # Check custom types are in prompt
        assert "location" in prompt
        assert "Geographic locations" in prompt
        assert "organization" in prompt
        assert "Companies and teams" in prompt

        # Check standard types still present
        assert "person" in prompt
        assert "concept" in prompt
        assert "tool" in prompt

    @pytest.mark.asyncio
    async def test_extract_custom_entity_types(self, mock_llm_client):
        """Test extraction of custom entity types."""
        custom_types = {
            "location": "Geographic locations",
            "organization": "Companies and teams",
        }

        extractor = EntityExtractor(
            llm_client=mock_llm_client,
            custom_types=custom_types,
        )

        # Mock LLM to return custom types
        llm_response = {
            "entities": [
                {"name": "San Francisco", "type": "location", "confidence": 0.95},
                {"name": "Anthropic", "type": "organization", "confidence": 0.95},
                {"name": "redis", "type": "tool", "confidence": 0.90},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.execute({
            "content": "Using Redis in San Francisco at Anthropic",
            "memory_id": "mem-001",
        })

        entities = result["normalized_entities"]
        assert len(entities) == 3

        # Check custom type entities
        location_entity = next(e for e in entities if e.entity_name == "san francisco")
        assert location_entity.entity_type == EntityType.OTHER
        assert location_entity.properties["custom_type"] == "location"

        org_entity = next(e for e in entities if e.entity_name == "anthropic")
        assert org_entity.entity_type == EntityType.OTHER
        assert org_entity.properties["custom_type"] == "organization"

        # Check standard type entity
        tool_entity = next(e for e in entities if e.entity_name == "redis")
        assert tool_entity.entity_type == EntityType.TOOL
        assert "custom_type" not in tool_entity.properties

    @pytest.mark.asyncio
    async def test_invalid_custom_type(self, mock_llm_client):
        """Test handling of invalid type that's not in custom types."""
        custom_types = {"location": "Geographic locations"}

        extractor = EntityExtractor(
            llm_client=mock_llm_client,
            custom_types=custom_types,
        )

        # Mock LLM to return unknown type
        llm_response = {
            "entities": [
                {"name": "test", "type": "unknown_type", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.execute({
            "content": "Test",
            "memory_id": "mem-001",
        })

        entities = result["normalized_entities"]
        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.OTHER
        assert "custom_type" not in entities[0].properties

    @pytest.mark.asyncio
    async def test_multiple_custom_types(self, mock_llm_client):
        """Test extraction with multiple custom entity types."""
        custom_types = {
            "location": "Geographic locations",
            "organization": "Companies and teams",
            "product": "Software products and services",
        }

        extractor = EntityExtractor(
            llm_client=mock_llm_client,
            custom_types=custom_types,
        )

        llm_response = {
            "entities": [
                {"name": "New York", "type": "location", "confidence": 0.95},
                {"name": "Google", "type": "organization", "confidence": 0.95},
                {"name": "Gmail", "type": "product", "confidence": 0.90},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.execute({
            "content": "Google launched Gmail in New York",
            "memory_id": "mem-001",
        })

        entities = result["normalized_entities"]
        assert len(entities) == 3

        # Verify all custom types are handled
        custom_types_found = {e.properties.get("custom_type") for e in entities}
        assert custom_types_found == {"location", "organization", "product"}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_llm_response(self, entity_extractor, mock_llm_client):
        """Test handling of empty LLM response."""
        llm_response = {"entities": []}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Test content",
            "memory_id": "mem-001",
        })

        assert result["entities"] == []
        assert result["normalized_entities"] == []
        assert result["extraction_metadata"]["total_extracted"] == 0

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, entity_extractor, mock_llm_client):
        """Test handling of malformed JSON from LLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "invalid json"

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="LLM entity extraction failed"):
            await entity_extractor.execute({
                "content": "Test",
                "memory_id": "mem-001",
            })

    @pytest.mark.asyncio
    async def test_entity_without_name(self, entity_extractor, mock_llm_client):
        """Test handling of entities without names."""
        llm_response = {
            "entities": [
                {"name": "", "type": "tool", "confidence": 0.95},  # Empty name
                {"type": "tool", "confidence": 0.95},  # No name field
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await entity_extractor.execute({
            "content": "Test",
            "memory_id": "mem-001",
        })

        # Should skip entities without valid names
        assert result["normalized_entities"] == []

    @pytest.mark.asyncio
    async def test_very_long_content(self, entity_extractor, mock_llm_client):
        """Test handling of very long content."""
        long_content = "Redis " * 10000  # Very long content

        llm_response = {
            "entities": [
                {"name": "redis", "type": "tool", "confidence": 0.95},
            ]
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(llm_response).replace("'", '"')

        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Should handle without error
        result = await entity_extractor.execute({
            "content": long_content,
            "memory_id": "mem-001",
        })

        assert len(result["normalized_entities"]) > 0
