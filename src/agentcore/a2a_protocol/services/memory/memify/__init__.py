"""
Memify Graph Optimization Module

Provides algorithms for optimizing the Neo4j knowledge graph:
- Entity consolidation (merge duplicates)
- Relationship pruning (remove low-value edges)
- Pattern detection (identify common patterns)

Component ID: MEM-023
Ticket: MEM-023 (Implement Memify Graph Optimizer)
"""

from .consolidation import EntityConsolidation
from .patterns import PatternDetection
from .pruning import RelationshipPruning

__all__ = [
    "EntityConsolidation",
    "RelationshipPruning",
    "PatternDetection",
]
