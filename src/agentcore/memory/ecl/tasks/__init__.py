"""ECL Pipeline Tasks for Memory System.

This module contains Cognify phase tasks for knowledge extraction:
- Entity extraction
- Relationship detection
- Semantic analysis

Tasks integrate with the ECL pipeline framework and support:
- LLM-based extraction using gpt-4.1-mini
- Entity normalization and deduplication
- Confidence scoring
- Neo4j graph storage integration

References:
    - FR-9.2: Cognify Phase (Knowledge Extraction)
    - MEM-015: Entity Extraction Task
"""

from agentcore.memory.ecl.tasks.entity_extractor import EntityExtractor

__all__ = ["EntityExtractor"]
