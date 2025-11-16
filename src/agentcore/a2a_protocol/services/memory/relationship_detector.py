"""
Relationship Detection Task for ECL Pipeline

Implements ECL Cognify task for detecting connections between entities using
LLM-based relationship detection and pattern matching fallback.

Component ID: MEM-018
Ticket: MEM-018 (Implement Relationship Detection Task)
References: FR-9.2 (Cognify Phase - Relationship Detection Task)
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    RelationshipEdge,
    RelationshipType,
)
from agentcore.a2a_protocol.services.memory.ecl_pipeline import ECLTask, RetryStrategy

logger = structlog.get_logger(__name__)


class RelationshipDetectorTask(ECLTask):
    """
    Detect relationships between entities using LLM analysis and pattern matching.

    Supports relationship types:
    - MENTIONS: Entity mentioned in memory
    - RELATES_TO: Semantic relationship
    - PART_OF: Hierarchical relationship
    - FOLLOWS: Temporal sequence
    - PRECEDES: Temporal precedence
    - CONTRADICTS: Conflicting information

    Detection methods:
    1. LLM-based detection using gpt-4.1-mini
    2. Pattern matching for common relationships (fallback)
    3. Relationship strength scoring (0.0-1.0)

    Target: 75%+ detection accuracy
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str = "gpt-4.1-mini",
        max_relationships_per_pair: int = 3,
        strength_threshold: float = 0.3,
        enable_pattern_matching: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize relationship detector.

        Args:
            llm_client: LLM client for detection (OpenAI compatible)
            model: Model to use for detection (default: gpt-4.1-mini per CLAUDE.md)
            max_relationships_per_pair: Maximum relationships to detect per entity pair
            strength_threshold: Minimum relationship strength (0.0-1.0)
            enable_pattern_matching: Enable pattern matching fallback
            **kwargs: Additional ECLTask arguments (retry_strategy, max_retries, etc.)
        """
        # Set defaults for ECLTask if not provided in kwargs
        task_kwargs = {
            "name": "relationship_detector",
            "description": "Detect relationships between entities using LLM and pattern matching",
            "retry_strategy": RetryStrategy.EXPONENTIAL,
            "max_retries": 3,
            "retry_delay_ms": 1000,
        }
        # Override with any provided kwargs
        task_kwargs.update(kwargs)

        super().__init__(**task_kwargs)

        self.llm_client = llm_client
        self.model = model
        self.max_relationships_per_pair = max_relationships_per_pair
        self.strength_threshold = strength_threshold
        self.enable_pattern_matching = enable_pattern_matching

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute relationship detection on entities.

        Args:
            input_data: Dictionary containing:
                - entities: list[EntityNode] - Entities to detect relationships between
                - content: str - Original content for context
                - memory_id: str (optional) - Memory ID for reference
                OR
                - input: dict - Nested input dict from pipeline (will be unwrapped)

        Returns:
            Dictionary containing:
                - relationships: list[RelationshipEdge] - Detected relationships
                - detection_metadata: dict - Stats about detection

        Raises:
            ValueError: If entities or content not provided
            RuntimeError: If LLM detection fails
        """
        # Handle pipeline wrapping of inputs
        if "entities" not in input_data and "input" in input_data:
            input_data = input_data["input"]

        # Validate input
        entities = input_data.get("entities", [])
        content = input_data.get("content", "")

        if not entities:
            raise ValueError("Entities list is required for relationship detection")

        if not content:
            raise ValueError("Content is required for relationship detection")

        memory_id = input_data.get("memory_id")

        logger.info(
            "Starting relationship detection",
            memory_id=memory_id,
            entities_count=len(entities),
            content_length=len(content),
        )

        # Detect relationships using LLM
        llm_relationships = []
        if self.llm_client:
            llm_relationships = await self._detect_relationships_llm(
                entities, content
            )
            logger.debug(
                "LLM detection completed",
                relationships_count=len(llm_relationships),
            )

        # Apply pattern matching if enabled
        pattern_relationships = []
        if self.enable_pattern_matching:
            pattern_relationships = self._detect_relationships_pattern(
                entities, content
            )
            logger.debug(
                "Pattern matching completed",
                relationships_count=len(pattern_relationships),
            )

        # Merge and deduplicate relationships
        all_relationships = self._merge_relationships(
            llm_relationships, pattern_relationships
        )

        # Filter by strength threshold
        filtered_relationships = [
            rel
            for rel in all_relationships
            if rel.properties.get("strength", 0.0) >= self.strength_threshold
        ]

        logger.debug(
            "Filtered relationships by strength",
            total=len(all_relationships),
            filtered=len(filtered_relationships),
            threshold=self.strength_threshold,
        )

        # Add memory reference if provided
        if memory_id:
            for rel in filtered_relationships:
                if memory_id not in rel.memory_refs:
                    rel.memory_refs.append(memory_id)

        # Calculate detection metadata
        detection_metadata = {
            "total_relationships": len(all_relationships),
            "llm_detected": len(llm_relationships),
            "pattern_detected": len(pattern_relationships),
            "after_strength_filter": len(filtered_relationships),
            "avg_strength": (
                sum(
                    rel.properties.get("strength", 0.0)
                    for rel in filtered_relationships
                )
                / len(filtered_relationships)
                if filtered_relationships
                else 0.0
            ),
            "model_used": self.model if self.llm_client else "pattern_only",
            "strength_threshold": self.strength_threshold,
        }

        logger.info(
            "Relationship detection completed",
            memory_id=memory_id,
            relationships_detected=len(filtered_relationships),
            metadata=detection_metadata,
        )

        return {
            "relationships": filtered_relationships,
            "detection_metadata": detection_metadata,
        }

    async def _detect_relationships_llm(
        self, entities: list[EntityNode], content: str
    ) -> list[RelationshipEdge]:
        """
        Detect relationships using LLM analysis.

        Args:
            entities: List of entities to detect relationships between
            content: Content for context

        Returns:
            List of RelationshipEdge instances

        Raises:
            RuntimeError: If LLM call fails
        """
        if not self.llm_client:
            logger.warning("No LLM client provided, skipping LLM detection")
            return []

        # Build detection prompt
        prompt = self._build_detection_prompt(entities, content)

        try:
            # Call LLM for relationship detection
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert relationship detection system. Analyze entity relationships in text and classify them accurately.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            relationships_data = result.get("relationships", [])

            logger.debug(
                "LLM detection successful",
                relationships_count=len(relationships_data),
                model=self.model,
            )

            # Convert to RelationshipEdge objects
            relationships = []
            for rel_data in relationships_data[:self.max_relationships_per_pair * len(entities)]:
                try:
                    # Find entity IDs by name
                    from_entity = self._find_entity_by_name(
                        entities, rel_data.get("from_entity", "")
                    )
                    to_entity = self._find_entity_by_name(
                        entities, rel_data.get("to_entity", "")
                    )

                    if not from_entity or not to_entity:
                        logger.warning(
                            "Entity not found for relationship",
                            from_entity=rel_data.get("from_entity"),
                            to_entity=rel_data.get("to_entity"),
                        )
                        continue

                    # Parse relationship type
                    rel_type_str = rel_data.get("type", "relates_to").lower()
                    try:
                        rel_type = RelationshipType(rel_type_str)
                    except ValueError:
                        logger.warning(
                            "Invalid relationship type, using RELATES_TO",
                            type=rel_type_str,
                        )
                        rel_type = RelationshipType.RELATES_TO

                    # Extract strength and evidence
                    strength = float(rel_data.get("strength", 0.5))
                    evidence = rel_data.get("evidence", "")

                    # Create RelationshipEdge
                    relationship = RelationshipEdge(
                        source_entity_id=from_entity.entity_id,
                        target_entity_id=to_entity.entity_id,
                        relationship_type=rel_type,
                        properties={
                            "strength": strength,
                            "evidence": evidence,
                            "detection_method": "llm",
                        },
                        access_count=0,
                    )

                    relationships.append(relationship)

                except Exception as e:
                    logger.warning(
                        "Failed to parse relationship",
                        error=str(e),
                        relationship=rel_data,
                    )
                    continue

            return relationships

        except Exception as e:
            logger.error(
                "LLM relationship detection failed",
                error=str(e),
                model=self.model,
            )
            raise RuntimeError(f"LLM relationship detection failed: {e}") from e

    def _build_detection_prompt(
        self, entities: list[EntityNode], content: str
    ) -> str:
        """
        Build detection prompt for LLM.

        Args:
            entities: Entities to detect relationships between
            content: Content for context

        Returns:
            Formatted prompt string
        """
        # Build entity list
        entity_list = []
        for entity in entities:
            entity_list.append(
                f"- {entity.entity_name} (type: {entity.entity_type.value})"
            )

        entity_list_str = "\n".join(entity_list)

        # Build relationship types list
        relationship_types = [
            "- mentions: Entity mentioned in text",
            "- relates_to: Semantic or conceptual relationship",
            "- part_of: Hierarchical relationship (entity is part of another)",
            "- follows: Temporal sequence (entity comes after another)",
            "- precedes: Temporal precedence (entity comes before another)",
            "- contradicts: Conflicting or opposing information",
        ]

        relationship_types_str = "\n".join(relationship_types)

        return f"""Detect relationships between entities in the following text content.

Entities:
{entity_list_str}

Relationship Types:
{relationship_types_str}

For each relationship, provide:
1. from_entity: Source entity name (must match entity list)
2. to_entity: Target entity name (must match entity list)
3. type: One of the relationship types above
4. strength: Relationship strength 0.0-1.0 (0.0=weak, 1.0=strong)
5. evidence: Brief explanation or quote from text

Relationship strength guidelines:
- 0.8-1.0: Strong, explicit relationship with clear evidence
- 0.5-0.7: Moderate relationship with implicit connection
- 0.3-0.4: Weak relationship, potential connection
- <0.3: Very weak, speculative connection

Return JSON format:
{{
    "relationships": [
        {{
            "from_entity": "entity name",
            "to_entity": "entity name",
            "type": "mentions|relates_to|part_of|follows|precedes|contradicts",
            "strength": 0.85,
            "evidence": "brief explanation or quote"
        }},
        ...
    ]
}}

Text Content:
{content}

Detect up to {self.max_relationships_per_pair} most important relationships per entity pair. Focus on accuracy and evidence."""

    def _detect_relationships_pattern(
        self, entities: list[EntityNode], content: str
    ) -> list[RelationshipEdge]:
        """
        Detect relationships using pattern matching.

        Pattern types:
        1. Co-occurrence: Entities mentioned close together → RELATES_TO
        2. Temporal: Sequential mentions → FOLLOWS/PRECEDES
        3. Hierarchical: "part of", "component of" → PART_OF
        4. Action: "uses", "implements", "requires" → RELATES_TO

        Args:
            entities: List of entities
            content: Content to analyze

        Returns:
            List of RelationshipEdge instances
        """
        relationships = []
        content_lower = content.lower()

        # Build entity name to entity map
        entity_map = {entity.entity_name.lower(): entity for entity in entities}

        # Pattern 1: Co-occurrence (entities mentioned in same sentence)
        sentences = re.split(r'[.!?]+', content_lower)
        for sentence in sentences:
            sentence_entities = [
                entity
                for name, entity in entity_map.items()
                if name in sentence
            ]

            # Create RELATES_TO for co-occurring entities
            for i, from_entity in enumerate(sentence_entities):
                for to_entity in sentence_entities[i + 1:]:
                    # Calculate distance-based strength
                    from_pos = sentence.find(from_entity.entity_name.lower())
                    to_pos = sentence.find(to_entity.entity_name.lower())
                    distance = abs(to_pos - from_pos)
                    strength = max(0.3, 1.0 - (distance / len(sentence)))

                    relationship = RelationshipEdge(
                        source_entity_id=from_entity.entity_id,
                        target_entity_id=to_entity.entity_id,
                        relationship_type=RelationshipType.RELATES_TO,
                        properties={
                            "strength": strength,
                            "evidence": "Co-occurrence in same sentence",
                            "detection_method": "pattern_cooccurrence",
                        },
                        access_count=0,
                    )
                    relationships.append(relationship)

        # Pattern 2: Hierarchical relationships
        hierarchical_patterns = [
            r'(\w+)\s+(?:is\s+)?part\s+of\s+(\w+)',
            r'(\w+)\s+(?:is\s+a\s+)?component\s+of\s+(\w+)',
            r'(\w+)\s+contains\s+(\w+)',
            r'(\w+)\s+includes\s+(\w+)',
        ]

        for pattern in hierarchical_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                from_name = match.group(1).strip()
                to_name = match.group(2).strip()

                from_entity = entity_map.get(from_name)
                to_entity = entity_map.get(to_name)

                if from_entity and to_entity:
                    relationship = RelationshipEdge(
                        source_entity_id=from_entity.entity_id,
                        target_entity_id=to_entity.entity_id,
                        relationship_type=RelationshipType.PART_OF,
                        properties={
                            "strength": 0.7,
                            "evidence": f"Pattern match: {match.group(0)}",
                            "detection_method": "pattern_hierarchical",
                        },
                        access_count=0,
                    )
                    relationships.append(relationship)

        # Pattern 3: Temporal relationships
        temporal_patterns = [
            (r'(\w+)\s+(?:then|after|following)\s+(\w+)', RelationshipType.FOLLOWS),
            (r'(\w+)\s+(?:before|prior\s+to)\s+(\w+)', RelationshipType.PRECEDES),
        ]

        for pattern, rel_type in temporal_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                from_name = match.group(1).strip()
                to_name = match.group(2).strip()

                from_entity = entity_map.get(from_name)
                to_entity = entity_map.get(to_name)

                if from_entity and to_entity:
                    relationship = RelationshipEdge(
                        source_entity_id=from_entity.entity_id,
                        target_entity_id=to_entity.entity_id,
                        relationship_type=rel_type,
                        properties={
                            "strength": 0.6,
                            "evidence": f"Pattern match: {match.group(0)}",
                            "detection_method": "pattern_temporal",
                        },
                        access_count=0,
                    )
                    relationships.append(relationship)

        # Pattern 4: Action relationships
        action_patterns = [
            r'(\w+)\s+uses\s+(\w+)',
            r'(\w+)\s+implements\s+(\w+)',
            r'(\w+)\s+requires\s+(\w+)',
            r'(\w+)\s+depends\s+on\s+(\w+)',
        ]

        for pattern in action_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                from_name = match.group(1).strip()
                to_name = match.group(2).strip()

                from_entity = entity_map.get(from_name)
                to_entity = entity_map.get(to_name)

                if from_entity and to_entity:
                    relationship = RelationshipEdge(
                        source_entity_id=from_entity.entity_id,
                        target_entity_id=to_entity.entity_id,
                        relationship_type=RelationshipType.RELATES_TO,
                        properties={
                            "strength": 0.75,
                            "evidence": f"Pattern match: {match.group(0)}",
                            "detection_method": "pattern_action",
                        },
                        access_count=0,
                    )
                    relationships.append(relationship)

        logger.debug(
            "Pattern matching completed",
            relationships_count=len(relationships),
        )

        return relationships

    def _merge_relationships(
        self,
        llm_relationships: list[RelationshipEdge],
        pattern_relationships: list[RelationshipEdge],
    ) -> list[RelationshipEdge]:
        """
        Merge and deduplicate relationships from LLM and pattern matching.

        Deduplication rules:
        - If same source, target, and type: keep higher strength
        - Boost strength if detected by both methods

        Args:
            llm_relationships: Relationships from LLM
            pattern_relationships: Relationships from pattern matching

        Returns:
            Merged list of relationships
        """
        # Build map for deduplication (source_id, target_id, type) -> relationship
        relationship_map: dict[tuple[str, str, str], RelationshipEdge] = {}

        # Add LLM relationships first (priority)
        for rel in llm_relationships:
            key = (
                rel.source_entity_id,
                rel.target_entity_id,
                rel.relationship_type.value,
            )
            relationship_map[key] = rel

        # Merge pattern relationships
        for rel in pattern_relationships:
            key = (
                rel.source_entity_id,
                rel.target_entity_id,
                rel.relationship_type.value,
            )

            if key in relationship_map:
                # Relationship already exists from LLM
                existing = relationship_map[key]
                existing_strength = existing.properties.get("strength", 0.5)

                # Boost strength if detected by both methods
                boosted_strength = min(1.0, existing_strength + 0.15)
                existing.properties["strength"] = boosted_strength
                existing.properties["detection_method"] = "llm_and_pattern"

                logger.debug(
                    "Boosted relationship strength",
                    from_id=rel.source_entity_id,
                    to_id=rel.target_entity_id,
                    type=rel.relationship_type.value,
                    original_strength=existing_strength,
                    boosted_strength=boosted_strength,
                )
            else:
                # New relationship from pattern matching
                relationship_map[key] = rel

        result = list(relationship_map.values())

        logger.debug(
            "Merged relationships",
            llm_count=len(llm_relationships),
            pattern_count=len(pattern_relationships),
            merged_count=len(result),
        )

        return result

    @staticmethod
    def _find_entity_by_name(
        entities: list[EntityNode], name: str
    ) -> EntityNode | None:
        """
        Find entity by name (case-insensitive).

        Args:
            entities: List of entities to search
            name: Entity name to find

        Returns:
            EntityNode or None if not found
        """
        name_lower = name.lower().strip()
        for entity in entities:
            if entity.entity_name.lower() == name_lower:
                return entity
        return None


# Export
__all__ = ["RelationshipDetectorTask"]
