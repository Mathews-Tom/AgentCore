"""
Service layer for the Context Reasoning framework.

This module contains service implementations:
- LLMClient: Async LLM client with stop sequences, retry logic, circuit breaker
- CarryoverGenerator: Compressed summary generation for bounded context iterations
- ReasoningStrategyRegistry: Central registry for strategy registration and discovery
- MetricsCalculator: Compute savings and performance metrics calculation
- StrategySelector: Logic for selecting strategies based on precedence rules
"""

from .carryover_generator import CarryoverGenerator
from .llm_client import (
    CircuitState,
    GenerationResult,
    LLMClient,
    LLMClientConfig,
)

__all__ = [
    "LLMClient",
    "LLMClientConfig",
    "GenerationResult",
    "CircuitState",
    "CarryoverGenerator",
]
