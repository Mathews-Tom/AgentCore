"""
Service layer for the Context Reasoning framework.

This module contains service implementations:
- LLMClient: Async LLM client with stop sequences, retry logic, circuit breaker
- CarryoverGenerator: Compressed summary generation for bounded context iterations
- MetricsCalculator: Compute savings and performance metrics calculation
- ReasoningStrategyRegistry: Central registry for strategy registration and discovery
- StrategySelector: Logic for selecting strategies based on precedence rules
"""

from .carryover_generator import CarryoverGenerator
from .llm_client import CircuitState, GenerationResult, LLMClient, LLMClientConfig
from .metrics_calculator import MetricsCalculator
from .strategy_registry import (
    ReasoningStrategyRegistry,
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
    registry,
)
from .strategy_selector import StrategySelectionError, StrategySelector

__all__ = [
    "LLMClient",
    "LLMClientConfig",
    "GenerationResult",
    "CircuitState",
    "CarryoverGenerator",
    "MetricsCalculator",
    "ReasoningStrategyRegistry",
    "StrategyNotFoundError",
    "StrategyAlreadyRegisteredError",
    "StrategySelector",
    "StrategySelectionError",
    "registry",
]
