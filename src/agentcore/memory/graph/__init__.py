"""
Neo4j Graph Database Integration for Memory System

Provides knowledge graph capabilities for entity relationships,
temporal connections, and semantic concepts.
"""

from pathlib import Path

# Schema file location
SCHEMA_FILE = Path(__file__).parent / "schema.cypher"

__all__ = ["SCHEMA_FILE"]
