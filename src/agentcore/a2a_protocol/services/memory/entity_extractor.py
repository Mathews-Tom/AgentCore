"""
Entity Extraction Task for ECL Pipeline

Implements ECL Cognify task for extracting entities (people, concepts, tools, constraints)
from memory content using LLM-based extraction.

Component ID: MEM-015
Ticket: MEM-015 (Implement Entity Extraction Task)
References: FR-9.2 (Cognify Phase - Entity Extraction Task)
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from agentcore.a2a_protocol.models.memory import EntityNode, EntityType
from agentcore.a2a_protocol.services.memory.ecl_pipeline import ECLTask, RetryStrategy

logger = structlog.get_logger(__name__)


class EntityExtractor(ECLTask):
    """
    Extract entities from memory content using LLM-based extraction.

    Supports entity classification, normalization, and deduplication.
    Target: 80%+ extraction accuracy.

    Entity types: person, concept, tool, constraint, other
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str = "gpt-4.1-mini",
        max_entities_per_memory: int = 20,
        confidence_threshold: float = 0.5,
        custom_types: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize entity extractor.

        Args:
            llm_client: LLM client for extraction (OpenAI compatible)
            model: Model to use for extraction (default: gpt-4.1-mini per CLAUDE.md)
            max_entities_per_memory: Maximum entities to extract per memory
            confidence_threshold: Minimum confidence score (0.0-1.0)
            custom_types: Custom entity types beyond standard 5
                         Format: {"type_name": "description"}
                         Example: {"location": "Geographic locations, cities, countries"}
            **kwargs: Additional ECLTask arguments (retry_strategy, max_retries, etc.)
        """
        # Set defaults for ECLTask if not provided in kwargs
        task_kwargs = {
            "name": "entity_extractor",
            "description": "Extract entities (people, concepts, tools, constraints) from memory content",
            "retry_strategy": RetryStrategy.EXPONENTIAL,
            "max_retries": 3,
            "retry_delay_ms": 1000,
        }
        # Override with any provided kwargs
        task_kwargs.update(kwargs)

        super().__init__(**task_kwargs)

        self.llm_client = llm_client
        self.model = model
        self.max_entities_per_memory = max_entities_per_memory
        self.confidence_threshold = confidence_threshold
        self.custom_types = custom_types or {}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute entity extraction on memory content.

        Args:
            input_data: Dictionary containing:
                - content: str - Memory content to extract entities from
                - memory_id: str (optional) - Memory ID for reference
                - existing_entities: list[EntityNode] (optional) - Known entities for deduplication
                OR
                - input: dict - Nested input dict from pipeline (will be unwrapped)

        Returns:
            Dictionary containing:
                - entities: list[dict] - Extracted entities with type and confidence
                - normalized_entities: list[EntityNode] - Deduplicated EntityNode instances
                - extraction_metadata: dict - Stats about extraction

        Raises:
            ValueError: If content not provided or invalid
            RuntimeError: If LLM extraction fails
        """
        # Handle pipeline wrapping of inputs
        # Pipeline passes {"input": original_data, "dep_task": dep_output, ...}
        if "content" not in input_data and "input" in input_data:
            input_data = input_data["input"]

        # Validate input
        content = input_data.get("content")
        if not content:
            raise ValueError("Content is required for entity extraction")

        memory_id = input_data.get("memory_id")
        existing_entities = input_data.get("existing_entities", [])

        logger.info(
            "Starting entity extraction",
            memory_id=memory_id,
            content_length=len(content),
        )

        # Extract entities using LLM
        raw_entities = await self._extract_entities_llm(content)

        # Filter by confidence threshold
        filtered_entities = [
            e for e in raw_entities if e.get("confidence", 0.0) >= self.confidence_threshold
        ]

        logger.debug(
            "Filtered entities by confidence",
            total=len(raw_entities),
            filtered=len(filtered_entities),
            threshold=self.confidence_threshold,
        )

        # Normalize and deduplicate
        normalized_entities = self._normalize_and_deduplicate(
            filtered_entities, existing_entities, memory_id
        )

        # Calculate extraction metadata
        extraction_metadata = {
            "total_extracted": len(raw_entities),
            "after_confidence_filter": len(filtered_entities),
            "after_deduplication": len(normalized_entities),
            "model_used": self.model,
            "confidence_threshold": self.confidence_threshold,
        }

        logger.info(
            "Entity extraction completed",
            memory_id=memory_id,
            entities_extracted=len(normalized_entities),
            metadata=extraction_metadata,
        )

        return {
            "entities": filtered_entities,
            "normalized_entities": normalized_entities,
            "extraction_metadata": extraction_metadata,
        }

    async def _extract_entities_llm(self, content: str) -> list[dict[str, Any]]:
        """
        Extract entities from content using LLM.

        Args:
            content: Text content to extract entities from

        Returns:
            List of entity dictionaries with name, type, confidence

        Raises:
            RuntimeError: If LLM call fails
        """
        if not self.llm_client:
            # Fallback: Simple keyword-based extraction if no LLM client
            logger.warning("No LLM client provided, using fallback extraction")
            return self._fallback_extraction(content)

        # Build extraction prompt
        prompt = self._build_extraction_prompt(content)

        try:
            # Call LLM for extraction
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert entity extraction system. Extract entities from text and classify them accurately.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            entities = result.get("entities", [])

            logger.debug(
                "LLM extraction successful",
                entities_count=len(entities),
                model=self.model,
            )

            return entities[:self.max_entities_per_memory]

        except Exception as e:
            logger.error(
                "LLM extraction failed",
                error=str(e),
                model=self.model,
            )
            raise RuntimeError(f"LLM entity extraction failed: {e}") from e

    def _build_extraction_prompt(self, content: str) -> str:
        """
        Build extraction prompt for LLM.

        Args:
            content: Content to extract entities from

        Returns:
            Formatted prompt string
        """
        # Build entity types list
        entity_types = [
            "- person: People, users, agents, individuals",
            "- concept: Abstract ideas, principles, methodologies, patterns",
            "- tool: Functions, APIs, services, software, libraries, frameworks",
            "- constraint: Requirements, rules, limits, restrictions, policies",
        ]

        # Add custom types if provided
        if self.custom_types:
            for type_name, description in self.custom_types.items():
                entity_types.append(f"- {type_name}: {description}")

        entity_types.append("- other: Unclassified entities")

        entity_types_str = "\n".join(entity_types)

        # Build type names for JSON example
        type_names = "|".join(["person", "concept", "tool", "constraint"] +
                              list(self.custom_types.keys()) + ["other"])

        return f"""Extract entities from the following memory content and classify them by type.

Entity Types:
{entity_types_str}

For each entity, provide:
1. name: The entity name (normalized, lowercase)
2. type: One of the entity types above
3. confidence: Confidence score 0.0-1.0

Return JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "{type_names}", "confidence": 0.95}},
        ...
    ]
}}

Memory Content:
{content}

Extract up to {self.max_entities_per_memory} most important entities. Focus on accuracy over quantity."""

    def _fallback_extraction(self, content: str) -> list[dict[str, Any]]:
        """
        Fallback keyword-based extraction when LLM not available.

        Simple pattern matching for common entity patterns.

        Args:
            content: Content to extract entities from

        Returns:
            List of entity dictionaries
        """
        entities = []

        # Tool patterns (libraries, frameworks, APIs)
        tool_patterns = [
            r'\b(Redis|PostgreSQL|Neo4j|Qdrant|Docker|Kubernetes|FastAPI|Pydantic|SQLAlchemy)\b',
            r'\b(OpenAI|GPT-\d+|gpt-\d+\.?\d*(?:-mini)?)\b',
            r'\b(JWT|OAuth|API|REST|GraphQL|WebSocket)\b',
        ]

        for pattern in tool_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(0).lower()
                if name not in [e["name"] for e in entities]:
                    entities.append({
                        "name": name,
                        "type": "tool",
                        "confidence": 0.7,
                    })

        # Concept patterns
        concept_patterns = [
            r'\b(authentication|authorization|caching|compression|optimization|validation)\b',
            r'\b(memory|storage|retrieval|extraction|detection|classification)\b',
        ]

        for pattern in concept_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(0).lower()
                if name not in [e["name"] for e in entities]:
                    entities.append({
                        "name": name,
                        "type": "concept",
                        "confidence": 0.6,
                    })

        logger.debug(
            "Fallback extraction completed",
            entities_count=len(entities),
        )

        return entities[:self.max_entities_per_memory]

    def _normalize_and_deduplicate(
        self,
        extracted_entities: list[dict[str, Any]],
        existing_entities: list[EntityNode],
        memory_id: str | None = None,
    ) -> list[EntityNode]:
        """
        Normalize entity names and deduplicate against existing entities.

        Normalization:
        - Convert to lowercase
        - Trim whitespace
        - Map common synonyms
        - Remove special characters (except hyphens)

        Deduplication:
        - Exact name match (after normalization)
        - If match found, update memory_refs instead of creating new entity

        Args:
            extracted_entities: Raw extracted entities
            existing_entities: Known entities for deduplication
            memory_id: Memory ID to add to entity references

        Returns:
            List of deduplicated EntityNode instances
        """
        # Build map of existing entities by normalized name
        existing_map: dict[str, EntityNode] = {}
        for entity in existing_entities:
            normalized_name = self._normalize_entity_name(entity.entity_name)
            existing_map[normalized_name] = entity

        # Process extracted entities
        unique_entities: dict[str, EntityNode] = {}

        for raw_entity in extracted_entities:
            name = raw_entity.get("name", "").strip()
            if not name:
                continue

            # Normalize name
            normalized_name = self._normalize_entity_name(name)

            # Check if entity already exists
            if normalized_name in existing_map:
                # Update existing entity's memory references
                existing_entity = existing_map[normalized_name]
                if memory_id and memory_id not in existing_entity.memory_refs:
                    existing_entity.memory_refs.append(memory_id)
                unique_entities[normalized_name] = existing_entity

            elif normalized_name in unique_entities:
                # Already processed in this batch
                if memory_id and memory_id not in unique_entities[normalized_name].memory_refs:
                    unique_entities[normalized_name].memory_refs.append(memory_id)

            else:
                # Create new entity
                entity_type_str = raw_entity.get("type", "other")
                custom_type = None

                # Try to map to standard EntityType
                try:
                    entity_type = EntityType(entity_type_str.lower())
                except ValueError:
                    # Check if it's a custom type
                    if entity_type_str.lower() in self.custom_types:
                        entity_type = EntityType.OTHER
                        custom_type = entity_type_str.lower()
                        logger.debug(
                            "Using custom entity type",
                            custom_type=custom_type,
                            entity=normalized_name,
                        )
                    else:
                        entity_type = EntityType.OTHER
                        logger.warning(
                            "Invalid entity type, using OTHER",
                            type=entity_type_str,
                            entity=normalized_name,
                        )

                confidence = raw_entity.get("confidence", 0.5)
                memory_refs = [memory_id] if memory_id else []

                # Build properties
                properties = {
                    "confidence": confidence,
                    "original_name": name,
                }
                if custom_type:
                    properties["custom_type"] = custom_type

                new_entity = EntityNode(
                    entity_name=normalized_name,
                    entity_type=entity_type,
                    properties=properties,
                    memory_refs=memory_refs,
                )

                unique_entities[normalized_name] = new_entity

        result = list(unique_entities.values())

        logger.debug(
            "Normalization and deduplication completed",
            input_count=len(extracted_entities),
            output_count=len(result),
            deduplication_rate=1.0 - (len(result) / max(len(extracted_entities), 1)),
        )

        return result

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for consistent matching.

        Rules:
        - Convert to lowercase
        - Trim leading/trailing whitespace
        - Replace multiple spaces with single space
        - Remove special characters except hyphens and underscores
        - Apply synonym mapping

        Args:
            name: Raw entity name

        Returns:
            Normalized name
        """
        # Lowercase and trim
        normalized = name.lower().strip()

        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove special characters except hyphens, underscores, dots
        normalized = re.sub(r'[^\w\s\-\.]', '', normalized)

        # Apply synonym mapping
        synonym_map = {
            'postgres': 'postgresql',
            'pg': 'postgresql',
            'gpt4': 'gpt-4',
            'gpt-4o': 'gpt-4',
            'openai': 'openai',
            'k8s': 'kubernetes',
            'js': 'javascript',
            'py': 'python',
        }

        for synonym, canonical in synonym_map.items():
            if normalized == synonym:
                normalized = canonical
                break

        return normalized

    def classify_entity_type(self, entity_name: str, context: str = "") -> EntityType:
        """
        Classify entity type based on name and context.

        Uses heuristic rules for classification without LLM.

        Args:
            entity_name: Entity name to classify
            context: Optional context for better classification

        Returns:
            EntityType classification
        """
        name_lower = entity_name.lower()

        # Tool patterns
        tool_keywords = [
            'api', 'library', 'framework', 'service', 'database', 'db',
            'redis', 'postgres', 'neo4j', 'docker', 'kubernetes',
            'fastapi', 'pydantic', 'sqlalchemy', 'openai', 'gpt',
        ]
        if any(keyword in name_lower for keyword in tool_keywords):
            return EntityType.TOOL

        # Concept patterns
        concept_keywords = [
            'pattern', 'principle', 'methodology', 'approach', 'strategy',
            'architecture', 'design', 'algorithm', 'optimization',
            'authentication', 'authorization', 'compression', 'caching',
        ]
        if any(keyword in name_lower for keyword in concept_keywords):
            return EntityType.CONCEPT

        # Constraint patterns
        constraint_keywords = [
            'requirement', 'rule', 'limit', 'constraint', 'restriction',
            'policy', 'must', 'should', 'cannot', 'maximum', 'minimum',
        ]
        if any(keyword in name_lower for keyword in constraint_keywords):
            return EntityType.CONSTRAINT

        # Person patterns (proper nouns with spaces, likely names)
        # Check original entity_name (not lowercased) for proper noun pattern
        if entity_name and entity_name[0].isupper() and ' ' in entity_name:
            return EntityType.PERSON

        # Default to other
        return EntityType.OTHER


# Export
__all__ = ["EntityExtractor"]
