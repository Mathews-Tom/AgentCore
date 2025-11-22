"""Entity Extraction Prompt Templates.

This module provides prompt templates for LLM-based entity extraction
from memory content. Supports extraction of:
- People: Individuals, roles, personas
- Concepts: Abstract ideas, methodologies, principles
- Tools: Software, frameworks, technologies, APIs
- Constraints: Limitations, requirements, rules

References:
    - FR-9.2: Cognify Phase (Knowledge Extraction)
    - MEM-015: Entity Extraction Task
"""

from __future__ import annotations


ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert entity extraction system for AI agent memory.

Your task is to extract and classify entities from conversation content into these types:
- **person**: Individuals, roles, personas (e.g., "Alice", "Backend Engineer", "Project Manager")
- **concept**: Abstract ideas, methodologies, principles (e.g., "Machine Learning", "Agile Development")
- **tool**: Software, frameworks, technologies, APIs (e.g., "Python", "FastAPI", "Neo4j", "gpt-4.1")
- **constraint**: Limitations, requirements, rules (e.g., "90% test coverage", "max 200ms latency")

For each entity, provide:
1. **name**: Normalized entity name (canonical form)
2. **type**: Entity classification (person, concept, tool, constraint)
3. **confidence**: Confidence score 0.0-1.0 (how certain you are)
4. **context**: Brief context where entity appears (10-20 words)
5. **properties**: Additional metadata (optional)

Guidelines:
- Normalize names: "GPT-4.1 Mini" → "gpt-4.1-mini"
- Avoid duplicates: If "Alice" appears multiple times, extract once with combined context
- Filter noise: Skip common words unless they're domain-specific ("the", "a", "is")
- Prefer quality over quantity: Focus on meaningful entities
- Use high confidence (>0.8) for explicit entities
- Use medium confidence (0.5-0.8) for inferred entities
- Use low confidence (<0.5) for ambiguous entities

Return JSON array of entities. Each entity must have:
```json
{
  "name": "normalized_entity_name",
  "type": "person|concept|tool|constraint",
  "confidence": 0.85,
  "context": "Brief context snippet",
  "properties": {}
}
```

If no entities found, return empty array: []"""


ENTITY_EXTRACTION_USER_TEMPLATE = """Extract entities from the following memory content:

---
{content}
---

Instructions:
- Extract maximum {max_entities} entities
- Minimum confidence threshold: {confidence_threshold}
- Focus on entities relevant to the conversation context
- Return JSON array only, no additional text

Entities:"""


def get_entity_extraction_messages(
    content: str,
    max_entities: int = 20,
    confidence_threshold: float = 0.5,
) -> list[dict[str, str]]:
    """Build entity extraction prompt messages.

    Args:
        content: Text content to extract entities from
        max_entities: Maximum number of entities to extract
        confidence_threshold: Minimum confidence score (0.0-1.0)

    Returns:
        List of message dicts for LLM API (system + user)

    Example:
        ```python
        messages = get_entity_extraction_messages(
            content="Alice used Python to build FastAPI service",
            max_entities=10,
            confidence_threshold=0.7
        )
        # Returns:
        # [
        #   {"role": "system", "content": "..."},
        #   {"role": "user", "content": "..."}
        # ]
        ```
    """
    return [
        {
            "role": "system",
            "content": ENTITY_EXTRACTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": ENTITY_EXTRACTION_USER_TEMPLATE.format(
                content=content,
                max_entities=max_entities,
                confidence_threshold=confidence_threshold,
            ),
        },
    ]


def get_entity_refinement_prompt(
    entities: list[dict[str, any]], content: str
) -> list[dict[str, str]]:
    """Build prompt for refining extracted entities.

    Used for deduplication and normalization pass.

    Args:
        entities: List of extracted entities
        content: Original content for context

    Returns:
        List of message dicts for LLM API

    Example:
        ```python
        refined_messages = get_entity_refinement_prompt(
            entities=[...],
            content="original text"
        )
        ```
    """
    return [
        {
            "role": "system",
            "content": """You are an entity normalization system.

Review the extracted entities and:
1. Merge duplicates (e.g., "Python", "python", "Python 3" → "python")
2. Normalize names to canonical form
3. Remove low-confidence noise
4. Verify entity types are correct
5. Update confidence scores based on review

Return the refined JSON array of entities.""",
        },
        {
            "role": "user",
            "content": f"""Original content:
---
{content}
---

Extracted entities:
```json
{entities}
```

Refine these entities by merging duplicates and normalizing names.
Return refined JSON array:""",
        },
    ]
