"""
Optimization insights and recommendations

Provides pattern-based recommendations, knowledge base management,
and best practices extraction for DSPy optimization strategies.
"""

from agentcore.dspy_optimization.insights.best_practices import (
    BestPractice,
    BestPracticeCategory,
    BestPracticeExtractor,
)
from agentcore.dspy_optimization.insights.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from agentcore.dspy_optimization.insights.recommendation_engine import (
    Recommendation,
    RecommendationContext,
    RecommendationEngine,
    RecommendationPriority,
)

__all__ = [
    "BestPractice",
    "BestPracticeCategory",
    "BestPracticeExtractor",
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeEntryType",
    "Recommendation",
    "RecommendationContext",
    "RecommendationEngine",
    "RecommendationPriority",
]
