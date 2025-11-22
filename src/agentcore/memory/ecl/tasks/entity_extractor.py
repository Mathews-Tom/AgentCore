"""Entity Extraction Task for ECL Pipeline.

Extracts entities (people, concepts, tools, constraints) from memory content using
LLM-based extraction with gpt-4.1-mini. Provides entity classification, normalization,
and deduplication.

This module implements MEM-015 acceptance criteria:
- Extract entities using gpt-4.1-mini
- Entity classification (person, concept, tool, constraint)
- Entity normalization and deduplication
- 80%+ extraction accuracy target
- Integration with ECL pipeline
- Comprehensive unit tests

References:
    - FR-9.2: Cognify Phase (Knowledge Extraction)
    - MEM-015: Entity Extraction Task
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from agentcore.a2a_protocol.models.llm import LLMRequest, ProviderError
from agentcore.a2a_protocol.services.llm_service import ProviderRegistry
from agentcore.memory.prompts.entity_extraction import (
    get_entity_extraction_messages,
    get_entity_refinement_prompt,
)

logger = structlog.get_logger(__name__)


class EntityType(str, Enum):
    """Types of entities that can be extracted."""

    PERSON = "person"  # Individuals, roles, personas
    CONCEPT = "concept"  # Abstract ideas, methodologies, principles
    TOOL = "tool"  # Software, frameworks, technologies, APIs
    CONSTRAINT = "constraint"  # Limitations, requirements, rules
    OTHER = "other"  # Uncategorized entities


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata.

    Attributes:
        entity_id: Unique identifier for the entity
        name: Normalized entity name
        entity_type: Classification (person, concept, tool, constraint)
        confidence: Confidence score 0.0-1.0
        context: Brief context where entity appears
        properties: Additional metadata (optional)
    """

    entity_id: str
    name: str
    entity_type: EntityType
    confidence: float
    context: str
    properties: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all entity fields
        """
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "properties": self.properties or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedEntity:
        """Create entity from dictionary.

        Args:
            data: Dictionary containing entity fields

        Returns:
            ExtractedEntity instance

        Raises:
            ValueError: If required fields missing or invalid
        """
        return cls(
            entity_id=data.get("entity_id", str(uuid4())),
            name=data["name"],
            entity_type=EntityType(data["type"]),
            confidence=float(data["confidence"]),
            context=data.get("context", ""),
            properties=data.get("properties"),
        )


class EntityExtractor:
    """Extracts entities from memory content using LLM-based extraction.

    This class implements the ECL Cognify task for entity extraction.
    It uses gpt-4.1-mini to identify and classify entities in text content,
    with support for normalization and deduplication.

    The extractor follows a two-pass approach:
    1. Initial extraction: LLM identifies entities with classifications
    2. Refinement pass: Deduplicate and normalize entity names

    Example:
        ```python
        extractor = EntityExtractor(
            model="gpt-4.1-mini",
            max_entities=20,
            confidence_threshold=0.5
        )

        result = await extractor.extract_entities(
            content="Alice used Python to build FastAPI service with 90% test coverage"
        )

        # Result contains:
        # [
        #   {name: "Alice", type: "person", confidence: 0.95, ...},
        #   {name: "python", type: "tool", confidence: 0.98, ...},
        #   {name: "fastapi", type: "tool", confidence: 0.97, ...},
        #   {name: "90% test coverage", type: "constraint", confidence: 0.85, ...}
        # ]
        ```

    Attributes:
        model: LLM model for extraction (default: gpt-4.1-mini)
        max_entities: Maximum entities to extract per content
        confidence_threshold: Minimum confidence score (0.0-1.0)
        enable_refinement: Whether to run deduplication pass
        llm_registry: LLM provider registry for API calls
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        max_entities: int = 20,
        confidence_threshold: float = 0.5,
        enable_refinement: bool = True,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize entity extractor.

        Args:
            model: LLM model name for extraction
            max_entities: Maximum entities to extract per content
            confidence_threshold: Minimum confidence score (0.0-1.0)
            enable_refinement: Enable entity normalization pass
            timeout: LLM request timeout in seconds
            max_retries: Maximum retry attempts for LLM calls
        """
        self.model = model
        self.max_entities = max_entities
        self.confidence_threshold = confidence_threshold
        self.enable_refinement = enable_refinement
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize LLM registry
        self.llm_registry = ProviderRegistry(
            timeout=timeout,
            max_retries=max_retries,
        )

    async def extract_entities(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> list[ExtractedEntity]:
        """Extract entities from content.

        This is the main entry point for entity extraction. It performs:
        1. LLM-based entity extraction using structured prompts
        2. Response parsing and validation
        3. Optional refinement pass for deduplication
        4. Confidence filtering

        Args:
            content: Text content to extract entities from
            metadata: Optional metadata to attach to entities

        Returns:
            List of ExtractedEntity objects

        Raises:
            ValueError: If content is empty or invalid
            ProviderError: If LLM API call fails
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        logger.info(
            "Extracting entities",
            model=self.model,
            content_length=len(content),
            max_entities=self.max_entities,
            confidence_threshold=self.confidence_threshold,
        )

        # Pass 1: Initial extraction
        raw_entities = await self._extract_with_llm(content)

        if not raw_entities:
            logger.info("No entities extracted", content_preview=content[:100])
            return []

        # Convert to ExtractedEntity objects
        entities = self._parse_entities(raw_entities, metadata)

        # Pass 2: Refinement (optional)
        if self.enable_refinement and len(entities) > 0:
            entities = await self._refine_entities(entities, content)

        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]

        logger.info(
            "Entity extraction complete",
            total_extracted=len(entities),
            by_type={
                entity_type.value: sum(
                    1 for e in entities if e.entity_type == entity_type
                )
                for entity_type in EntityType
            },
        )

        return entities

    async def _extract_with_llm(self, content: str) -> list[dict[str, Any]]:
        """Perform LLM-based entity extraction.

        Args:
            content: Text content to extract from

        Returns:
            List of raw entity dictionaries from LLM response

        Raises:
            ProviderError: If LLM API call fails
        """
        # Build extraction prompt
        messages = get_entity_extraction_messages(
            content=content,
            max_entities=self.max_entities,
            confidence_threshold=self.confidence_threshold,
        )

        # Create LLM request
        request = LLMRequest(
            model=self.model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=2000,  # Sufficient for entity list
        )

        try:
            # Get LLM client for model
            client = self.llm_registry.get_provider_for_model(self.model)

            # Execute completion
            response = await client.complete(request)

            # Parse JSON response
            entities = self._parse_llm_response(response.content)

            logger.debug(
                "LLM extraction complete",
                entity_count=len(entities),
                tokens_used=response.usage.total_tokens,
            )

            return entities

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON", error=str(e))
            return []
        except ProviderError as e:
            logger.error("LLM provider error during extraction", error=str(e))
            raise

    def _parse_llm_response(self, content: str) -> list[dict[str, Any]]:
        """Parse LLM response content to extract JSON array.

        Handles various response formats:
        - Pure JSON array: [...]
        - Markdown code blocks: ```json\n[...]\n```
        - Text with embedded JSON: "Here are the entities: [...]"

        Args:
            content: LLM response content

        Returns:
            Parsed list of entity dictionaries

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        # Strip whitespace
        content = content.strip()

        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        # Find JSON array in text
        if content.startswith("["):
            # Pure JSON array
            return json.loads(content)
        else:
            # Try to find JSON array within text
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)

        # Fallback: try to parse entire content
        return json.loads(content)

    def _parse_entities(
        self, raw_entities: list[dict[str, Any]], metadata: dict[str, Any] | None
    ) -> list[ExtractedEntity]:
        """Parse raw entity dictionaries to ExtractedEntity objects.

        Args:
            raw_entities: List of entity dictionaries from LLM
            metadata: Optional metadata to attach

        Returns:
            List of ExtractedEntity objects
        """
        entities: list[ExtractedEntity] = []

        for raw in raw_entities:
            try:
                entity = ExtractedEntity.from_dict(raw)

                # Attach metadata if provided
                if metadata:
                    if entity.properties is None:
                        entity.properties = {}
                    entity.properties.update(metadata)

                entities.append(entity)

            except (KeyError, ValueError) as e:
                logger.warning(
                    "Failed to parse entity",
                    raw_entity=raw,
                    error=str(e),
                )
                continue

        return entities

    async def _refine_entities(
        self, entities: list[ExtractedEntity], content: str
    ) -> list[ExtractedEntity]:
        """Refine entities through deduplication and normalization.

        Args:
            entities: List of extracted entities
            content: Original content for context

        Returns:
            Refined list of entities
        """
        logger.debug("Running entity refinement pass", count=len(entities))

        # Convert to dict format for LLM
        entity_dicts = [e.to_dict() for e in entities]

        # Build refinement prompt
        messages = get_entity_refinement_prompt(entity_dicts, content)

        # Create LLM request
        request = LLMRequest(
            model=self.model,
            messages=messages,
            temperature=0.0,  # Deterministic refinement
            max_tokens=2000,
        )

        try:
            # Get LLM client
            client = self.llm_registry.get_provider_for_model(self.model)

            # Execute refinement
            response = await client.complete(request)

            # Parse refined entities
            refined_raw = self._parse_llm_response(response.content)
            refined_entities = self._parse_entities(refined_raw, None)

            logger.debug(
                "Entity refinement complete",
                original_count=len(entities),
                refined_count=len(refined_entities),
            )

            return refined_entities

        except (json.JSONDecodeError, ProviderError) as e:
            logger.warning(
                "Entity refinement failed, using original entities", error=str(e)
            )
            return entities

    def extract_entities_sync(self, content: str) -> list[ExtractedEntity]:
        """Synchronous wrapper for extract_entities (for testing).

        Args:
            content: Text content to extract from

        Returns:
            List of extracted entities

        Note:
            This method is provided for testing convenience.
            Production code should use the async extract_entities method.
        """
        import asyncio

        return asyncio.run(self.extract_entities(content))
