"""
ReasoningStrategy protocol definition.

Defines the interface that all reasoning strategies must implement for
polymorphic usage within the reasoning framework.
"""

from __future__ import annotations

from typing import Any, Protocol

from .models.reasoning_models import ReasoningResult


class ReasoningStrategy(Protocol):
    """
    Protocol defining the interface for all reasoning strategies.

    All concrete strategy implementations (BoundedContext, ChainOfThought, ReAct, etc.)
    must implement this protocol to be usable within the reasoning framework.

    This enables:
    - Polymorphic strategy usage
    - Strategy registration and discovery
    - Runtime strategy selection
    - Consistent interface across strategies
    """

    async def reason(self, query: str, **kwargs: Any) -> ReasoningResult:
        """
        Execute reasoning for the given query.

        Args:
            query: The problem or question to solve
            **kwargs: Strategy-specific configuration parameters

        Returns:
            ReasoningResult with answer, metrics, and optional trace

        Raises:
            ValueError: If query is invalid or configuration is incompatible
            RuntimeError: If strategy execution fails
        """
        ...

    def get_config_schema(self) -> dict[str, Any]:
        """
        Get the configuration schema for this strategy.

        Returns a JSON schema describing the strategy-specific configuration
        parameters that can be passed via kwargs to reason().

        Returns:
            dict: JSON schema for strategy configuration

        Example:
            {
                "type": "object",
                "properties": {
                    "chunk_size": {"type": "integer", "minimum": 1024, "maximum": 32768},
                    "max_iterations": {"type": "integer", "minimum": 1, "maximum": 50}
                }
            }
        """
        ...

    def get_capabilities(self) -> list[str]:
        """
        Get the list of capabilities this strategy provides.

        Used for agent capability advertisement and discovery.

        Returns:
            list[str]: Capability identifiers (e.g., ["long_form_reasoning", "compute_efficient"])

        Example:
            ["reasoning.strategy.bounded_context", "long_form_reasoning"]
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the unique name of this strategy.

        Returns:
            str: Strategy name (e.g., "bounded_context", "chain_of_thought")
        """
        ...

    @property
    def version(self) -> str:
        """
        Get the version of this strategy implementation.

        Returns:
            str: Version string (e.g., "1.0.0")
        """
        ...
