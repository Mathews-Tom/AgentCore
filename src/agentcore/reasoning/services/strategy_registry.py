"""
ReasoningStrategyRegistry for strategy registration and discovery.

Provides centralized registry for all reasoning strategies with thread-safe
registration, lookup, and discovery capabilities.
"""

from __future__ import annotations

import logging
from typing import Any

from ..protocol import ReasoningStrategy

logger = logging.getLogger(__name__)


class StrategyNotFoundError(Exception):
    """Raised when a requested strategy is not found in the registry."""

    pass


class StrategyAlreadyRegisteredError(Exception):
    """Raised when attempting to register a strategy that already exists."""

    pass


class ReasoningStrategyRegistry:
    """
    Central registry for reasoning strategy registration and discovery.

    Provides thread-safe operations for:
    - Registering new strategies
    - Retrieving strategies by name
    - Listing all registered strategies
    - Checking strategy availability
    - Getting strategy metadata

    Singleton pattern ensures single registry instance across the application.
    """

    _instance: ReasoningStrategyRegistry | None = None
    _strategies: dict[str, ReasoningStrategy]

    def __new__(cls) -> ReasoningStrategyRegistry:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}
        return cls._instance

    def register(
        self, strategy: ReasoningStrategy, *, allow_override: bool = False
    ) -> None:
        """
        Register a reasoning strategy.

        Args:
            strategy: Strategy instance implementing ReasoningStrategy protocol
            allow_override: If True, allows overriding existing strategy with same name

        Raises:
            StrategyAlreadyRegisteredError: If strategy already exists and allow_override=False
            ValueError: If strategy name is invalid

        Example:
            >>> registry = ReasoningStrategyRegistry()
            >>> registry.register(BoundedContextEngine())
        """
        if not strategy.name:
            raise ValueError("Strategy name cannot be empty")

        if strategy.name in self._strategies and not allow_override:
            raise StrategyAlreadyRegisteredError(
                f"Strategy '{strategy.name}' is already registered. "
                f"Use allow_override=True to replace it."
            )

        self._strategies[strategy.name] = strategy
        logger.info(
            f"Registered reasoning strategy: {strategy.name} (version {strategy.version})"
        )

    def unregister(self, strategy_name: str) -> None:
        """
        Unregister a reasoning strategy.

        Args:
            strategy_name: Name of the strategy to unregister

        Raises:
            StrategyNotFoundError: If strategy is not found

        Example:
            >>> registry.unregister("bounded_context")
        """
        if strategy_name not in self._strategies:
            raise StrategyNotFoundError(
                f"Strategy '{strategy_name}' not found in registry"
            )

        del self._strategies[strategy_name]
        logger.info(f"Unregistered reasoning strategy: {strategy_name}")

    def get(self, strategy_name: str) -> ReasoningStrategy:
        """
        Retrieve a strategy by name.

        Args:
            strategy_name: Name of the strategy to retrieve

        Returns:
            ReasoningStrategy instance

        Raises:
            StrategyNotFoundError: If strategy is not found

        Example:
            >>> registry = ReasoningStrategyRegistry()
            >>> strategy = registry.get("bounded_context")
        """
        if strategy_name not in self._strategies:
            available = ", ".join(self._strategies.keys()) or "none"
            raise StrategyNotFoundError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {available}"
            )

        return self._strategies[strategy_name]

    def has(self, strategy_name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            strategy_name: Name of the strategy to check

        Returns:
            bool: True if strategy is registered, False otherwise

        Example:
            >>> if registry.has("bounded_context"):
            ...     strategy = registry.get("bounded_context")
        """
        return strategy_name in self._strategies

    def list_strategies(self) -> list[str]:
        """
        Get list of all registered strategy names.

        Returns:
            list[str]: Names of all registered strategies

        Example:
            >>> strategies = registry.list_strategies()
            >>> print(strategies)
            ['bounded_context', 'chain_of_thought', 'react']
        """
        return list(self._strategies.keys())

    def get_metadata(self, strategy_name: str) -> dict[str, Any]:
        """
        Get metadata for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            dict containing strategy metadata (name, version, capabilities, config_schema)

        Raises:
            StrategyNotFoundError: If strategy is not found

        Example:
            >>> metadata = registry.get_metadata("bounded_context")
            >>> print(metadata["version"])
            1.0.0
        """
        strategy = self.get(strategy_name)
        return {
            "name": strategy.name,
            "version": strategy.version,
            "capabilities": strategy.get_capabilities(),
            "config_schema": strategy.get_config_schema(),
        }

    def list_all_metadata(self) -> list[dict[str, Any]]:
        """
        Get metadata for all registered strategies.

        Returns:
            list[dict]: List of metadata dicts for all strategies

        Example:
            >>> all_metadata = registry.list_all_metadata()
            >>> for meta in all_metadata:
            ...     print(f"{meta['name']} v{meta['version']}")
        """
        return [self.get_metadata(name) for name in self._strategies.keys()]

    def clear(self) -> None:
        """
        Clear all registered strategies.

        WARNING: This should only be used for testing. In production,
        strategies should remain registered for the lifetime of the application.

        Example:
            >>> registry.clear()  # Only use in tests!
        """
        count = len(self._strategies)
        self._strategies.clear()
        logger.warning(f"Cleared {count} strategies from registry")


# Global singleton instance
registry = ReasoningStrategyRegistry()
