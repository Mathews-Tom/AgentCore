"""
Service layer for the Context Reasoning framework.

This module contains service implementations:
- LLMClient: Async LLM client with stop sequences, retry logic, circuit breaker
- CarryoverGenerator: Compressed summary generation for bounded context iterations
- MetricsCalculator: Compute savings and performance metrics calculation
- ReasoningStrategyRegistry: Central registry for strategy registration and discovery
- StrategySelector: Logic for selecting strategies based on precedence rules
- reasoning_execute_jsonrpc: Unified JSON-RPC handler for reasoning.execute method
- reasoning_jsonrpc: Legacy bounded context JSON-RPC handler
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

# JSON-RPC handlers are imported for side effects (method registration)
# but not exported in __all__
# NOTE: These imports are commented out to avoid circular import issues.
# Import these modules directly in main.py instead.
# from . import reasoning_execute_jsonrpc, reasoning_jsonrpc

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
