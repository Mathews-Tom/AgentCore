"""Unit tests for EntityExtractor.

Tests cover:
- Entity extraction with various content types
- Entity classification accuracy
- Confidence scoring
- Normalization and deduplication
- Error handling
- Edge cases (empty content, malformed responses)

References:
    - MEM-015: Entity Extraction Task
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage, ProviderError
from agentcore.memory.ecl.tasks.entity_extractor import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
)


# Sample test data
SAMPLE_CONTENT = """Alice used Python and FastAPI to build a web API with 90% test coverage.
The project follows Agile methodology and uses Neo4j for graph storage."""

SAMPLE_LLM_RESPONSE = """[
  {
    "name": "Alice",
    "type": "person",
    "confidence": 0.95,
    "context": "Alice used Python and FastAPI"
  },
  {
    "name": "python",
    "type": "tool",
    "confidence": 0.98,
    "context": "used Python and FastAPI to build"
  },
  {
    "name": "fastapi",
    "type": "tool",
    "confidence": 0.97,
    "context": "Python and FastAPI to build a web API"
  },
  {
    "name": "90% test coverage",
    "type": "constraint",
    "confidence": 0.85,
    "context": "web API with 90% test coverage"
  },
  {
    "name": "agile methodology",
    "type": "concept",
    "confidence": 0.88,
    "context": "follows Agile methodology"
  },
  {
    "name": "neo4j",
    "type": "tool",
    "confidence": 0.96,
    "context": "uses Neo4j for graph storage"
  }
]"""

SAMPLE_MARKDOWN_RESPONSE = f"""```json
{SAMPLE_LLM_RESPONSE}
```"""


class TestExtractedEntity:
    """Test ExtractedEntity dataclass."""

    def test_to_dict(self):
        """Test entity serialization to dictionary."""
        entity = ExtractedEntity(
            entity_id=str(uuid4()),
            name="Python",
            entity_type=EntityType.TOOL,
            confidence=0.95,
            context="Used Python for backend",
            properties={"version": "3.12"},
        )

        entity_dict = entity.to_dict()

        assert entity_dict["name"] == "Python"
        assert entity_dict["entity_type"] == "tool"
        assert entity_dict["confidence"] == 0.95
        assert entity_dict["context"] == "Used Python for backend"
        assert entity_dict["properties"]["version"] == "3.12"

    def test_from_dict(self):
        """Test entity deserialization from dictionary."""
        data = {
            "name": "FastAPI",
            "type": "tool",
            "confidence": 0.97,
            "context": "Built with FastAPI framework",
        }

        entity = ExtractedEntity.from_dict(data)

        assert entity.name == "FastAPI"
        assert entity.entity_type == EntityType.TOOL
        assert entity.confidence == 0.97
        assert entity.context == "Built with FastAPI framework"

    def test_from_dict_with_entity_id(self):
        """Test deserialization preserves entity_id if provided."""
        entity_id = str(uuid4())
        data = {
            "entity_id": entity_id,
            "name": "Alice",
            "type": "person",
            "confidence": 0.95,
            "context": "Alice is the developer",
        }

        entity = ExtractedEntity.from_dict(data)

        assert entity.entity_id == entity_id

    def test_from_dict_generates_id_if_missing(self):
        """Test deserialization generates entity_id if not provided."""
        data = {
            "name": "Neo4j",
            "type": "tool",
            "confidence": 0.96,
            "context": "Uses Neo4j database",
        }

        entity = ExtractedEntity.from_dict(data)

        assert entity.entity_id is not None
        assert len(entity.entity_id) > 0


class TestEntityExtractor:
    """Test EntityExtractor class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_registry(self, mock_llm_client):
        """Create mock provider registry."""
        with patch(
            "agentcore.memory.ecl.tasks.entity_extractor.ProviderRegistry"
        ) as MockRegistry:
            mock_instance = MockRegistry.return_value
            mock_instance.get_provider_for_model.return_value = mock_llm_client
            yield mock_instance

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self, mock_llm_client, mock_registry):
        """Test basic entity extraction."""
        # Setup mock response
        mock_llm_client.complete.return_value = LLMResponse(
            content=SAMPLE_LLM_RESPONSE,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            latency_ms=500.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Create extractor
        extractor = EntityExtractor(
            model="gpt-4.1-mini",
            max_entities=20,
            confidence_threshold=0.5,
            enable_refinement=False,  # Disable for basic test
        )

        # Extract entities
        entities = await extractor.extract_entities(SAMPLE_CONTENT)

        # Verify results
        assert len(entities) == 6
        assert any(e.name == "Alice" and e.entity_type == EntityType.PERSON for e in entities)
        assert any(e.name == "python" and e.entity_type == EntityType.TOOL for e in entities)
        assert any(e.name == "fastapi" and e.entity_type == EntityType.TOOL for e in entities)
        assert any(
            e.name == "90% test coverage" and e.entity_type == EntityType.CONSTRAINT
            for e in entities
        )

    @pytest.mark.asyncio
    async def test_extract_entities_with_markdown_response(
        self, mock_llm_client, mock_registry
    ):
        """Test extraction handles markdown code blocks."""
        # Setup mock response with markdown
        mock_llm_client.complete.return_value = LLMResponse(
            content=SAMPLE_MARKDOWN_RESPONSE,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            latency_ms=500.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        extractor = EntityExtractor(enable_refinement=False)
        entities = await extractor.extract_entities(SAMPLE_CONTENT)

        assert len(entities) == 6

    @pytest.mark.asyncio
    async def test_extract_entities_confidence_filtering(
        self, mock_llm_client, mock_registry
    ):
        """Test confidence threshold filtering."""
        # Setup mock response
        mock_llm_client.complete.return_value = LLMResponse(
            content=SAMPLE_LLM_RESPONSE,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            latency_ms=500.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Create extractor with high confidence threshold
        extractor = EntityExtractor(
            confidence_threshold=0.90,  # Filter out entities < 0.90
            enable_refinement=False,
        )

        entities = await extractor.extract_entities(SAMPLE_CONTENT)

        # Should only get high-confidence entities (Alice, Python, FastAPI, Neo4j)
        assert len(entities) == 4
        assert all(e.confidence >= 0.90 for e in entities)

    @pytest.mark.asyncio
    async def test_extract_entities_empty_content(self, mock_registry):
        """Test error handling for empty content."""
        extractor = EntityExtractor()

        with pytest.raises(ValueError, match="Content cannot be empty"):
            await extractor.extract_entities("")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            await extractor.extract_entities("   ")

    @pytest.mark.asyncio
    async def test_extract_entities_no_entities_found(
        self, mock_llm_client, mock_registry
    ):
        """Test handling when no entities are extracted."""
        # Setup mock response with empty array
        mock_llm_client.complete.return_value = LLMResponse(
            content="[]",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
            latency_ms=300.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        extractor = EntityExtractor(enable_refinement=False)
        entities = await extractor.extract_entities("No meaningful content here.")

        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_malformed_json(
        self, mock_llm_client, mock_registry
    ):
        """Test handling of malformed JSON response."""
        # Setup mock response with invalid JSON
        mock_llm_client.complete.return_value = LLMResponse(
            content="This is not JSON {invalid}",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
            latency_ms=300.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        extractor = EntityExtractor(enable_refinement=False)
        entities = await extractor.extract_entities(SAMPLE_CONTENT)

        # Should return empty list on parse error
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_provider_error(self, mock_llm_client, mock_registry):
        """Test handling of LLM provider errors."""
        # Setup mock to raise provider error
        mock_llm_client.complete.side_effect = ProviderError(
            provider="openai", original_error=Exception("API error")
        )

        extractor = EntityExtractor()

        with pytest.raises(ProviderError):
            await extractor.extract_entities(SAMPLE_CONTENT)

    @pytest.mark.asyncio
    async def test_extract_entities_with_metadata(self, mock_llm_client, mock_registry):
        """Test entity extraction with custom metadata."""
        # Setup mock response
        mock_llm_client.complete.return_value = LLMResponse(
            content=SAMPLE_LLM_RESPONSE,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            latency_ms=500.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        extractor = EntityExtractor(enable_refinement=False)

        # Extract with metadata
        metadata = {"session_id": "test-session", "agent_id": "test-agent"}
        entities = await extractor.extract_entities(SAMPLE_CONTENT, metadata=metadata)

        # Verify metadata attached
        assert all(e.properties is not None for e in entities)
        assert all(e.properties.get("session_id") == "test-session" for e in entities)
        assert all(e.properties.get("agent_id") == "test-agent" for e in entities)

    @pytest.mark.asyncio
    async def test_extract_entities_with_refinement(
        self, mock_llm_client, mock_registry
    ):
        """Test entity extraction with refinement pass."""
        # Setup mock responses for both passes
        # Pass 1: Initial extraction
        initial_response = LLMResponse(
            content=SAMPLE_LLM_RESPONSE,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            latency_ms=500.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        # Pass 2: Refinement (merge Python/FastAPI if duplicates existed)
        refined_response = LLMResponse(
            content=SAMPLE_LLM_RESPONSE,  # For simplicity, same as initial
            usage=LLMUsage(prompt_tokens=150, completion_tokens=180, total_tokens=330),
            latency_ms=450.0,
            provider="openai",
            model="gpt-4.1-mini",
        )

        mock_llm_client.complete.side_effect = [initial_response, refined_response]

        # Create extractor with refinement enabled
        extractor = EntityExtractor(enable_refinement=True)

        entities = await extractor.extract_entities(SAMPLE_CONTENT)

        # Should call LLM twice (extraction + refinement)
        assert mock_llm_client.complete.call_count == 2

        # Verify results
        assert len(entities) >= 1

    def test_parse_llm_response_pure_json(self):
        """Test parsing pure JSON array response."""
        extractor = EntityExtractor()
        result = extractor._parse_llm_response(SAMPLE_LLM_RESPONSE)

        assert isinstance(result, list)
        assert len(result) == 6

    def test_parse_llm_response_markdown(self):
        """Test parsing JSON in markdown code blocks."""
        extractor = EntityExtractor()
        result = extractor._parse_llm_response(SAMPLE_MARKDOWN_RESPONSE)

        assert isinstance(result, list)
        assert len(result) == 6

    def test_parse_llm_response_embedded_json(self):
        """Test parsing JSON embedded in text."""
        embedded = f"Here are the entities: {SAMPLE_LLM_RESPONSE} Thanks!"
        extractor = EntityExtractor()
        result = extractor._parse_llm_response(embedded)

        assert isinstance(result, list)
        assert len(result) == 6

    def test_parse_entities(self):
        """Test parsing raw entities to ExtractedEntity objects."""
        import json

        extractor = EntityExtractor()
        raw_entities = json.loads(SAMPLE_LLM_RESPONSE)

        entities = extractor._parse_entities(raw_entities, None)

        assert len(entities) == 6
        assert all(isinstance(e, ExtractedEntity) for e in entities)

    def test_parse_entities_with_invalid_data(self):
        """Test parsing skips invalid entities."""
        extractor = EntityExtractor()
        raw_entities = [
            {
                "name": "Valid",
                "type": "tool",
                "confidence": 0.9,
                "context": "context",
            },
            {
                "name": "Invalid",  # Missing type
                "confidence": 0.8,
            },
            {
                # Missing name
                "type": "person",
                "confidence": 0.7,
            },
        ]

        entities = extractor._parse_entities(raw_entities, None)

        # Should only get the valid entity
        assert len(entities) == 1
        assert entities[0].name == "Valid"


class TestEntityType:
    """Test EntityType enum."""

    def test_entity_types(self):
        """Test all entity type values."""
        assert EntityType.PERSON.value == "person"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.TOOL.value == "tool"
        assert EntityType.CONSTRAINT.value == "constraint"
        assert EntityType.OTHER.value == "other"

    def test_entity_type_from_string(self):
        """Test creating EntityType from string."""
        assert EntityType("person") == EntityType.PERSON
        assert EntityType("concept") == EntityType.CONCEPT
        assert EntityType("tool") == EntityType.TOOL
        assert EntityType("constraint") == EntityType.CONSTRAINT


@pytest.mark.integration
class TestEntityExtractorIntegration:
    """Integration tests with real LLM API (optional, requires API key)."""

    @pytest.mark.skip(reason="Requires real LLM API key and costs money")
    @pytest.mark.asyncio
    async def test_real_extraction(self):
        """Test entity extraction with real LLM API."""
        extractor = EntityExtractor(
            model="gpt-4.1-mini",
            max_entities=10,
            confidence_threshold=0.7,
        )

        content = """The AgentCore project uses FastAPI framework with Python 3.12.
        It integrates with Neo4j for graph storage and requires 90% test coverage.
        Alice leads the development following Agile methodology."""

        entities = await extractor.extract_entities(content)

        # Verify extraction quality
        assert len(entities) > 0
        assert any(e.entity_type == EntityType.PERSON for e in entities)
        assert any(e.entity_type == EntityType.TOOL for e in entities)
        assert any(e.entity_type == EntityType.CONCEPT for e in entities)
        assert any(e.entity_type == EntityType.CONSTRAINT for e in entities)
        assert all(e.confidence >= 0.7 for e in entities)
