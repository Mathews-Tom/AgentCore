"""
StrategySelector for multi-level strategy selection logic.

Implements precedence-based strategy selection:
    Request-level > Agent-level > System-level

Validates strategy availability and provides clear error messages.
"""

from __future__ import annotations

import logging
from typing import Any

from .strategy_registry import ReasoningStrategyRegistry, StrategyNotFoundError

logger = logging.getLogger(__name__)


class StrategySelectionError(Exception):
    """Raised when strategy selection fails."""

    pass


class StrategySelector:
    """
    Multi-level strategy selector with precedence rules.

    Selects reasoning strategies based on configuration precedence:
    1. Request-level: Strategy specified in request parameters (highest priority)
    2. Agent-level: Strategy from agent capabilities/preferences
    3. System-level: Default strategy from system configuration (lowest priority)

    If no strategy is found at any level and no default is configured,
    raises StrategySelectionError.
    """

    def __init__(
        self,
        registry: ReasoningStrategyRegistry,
        default_strategy: str | None = None,
    ) -> None:
        """
        Initialize strategy selector.

        Args:
            registry: Strategy registry instance
            default_strategy: System-level default strategy name (optional)

        Example:
            >>> registry = ReasoningStrategyRegistry()
            >>> selector = StrategySelector(registry, default_strategy="chain_of_thought")
        """
        self._registry = registry
        self._default_strategy = default_strategy

    def select(
        self,
        *,
        request_strategy: str | None = None,
        agent_strategy: str | None = None,
        agent_capabilities: list[str] | None = None,
    ) -> str:
        """
        Select a strategy based on multi-level precedence.

        Precedence order:
        1. request_strategy (if provided and valid)
        2. agent_strategy (if provided and valid)
        3. Infer from agent_capabilities (if provided)
        4. default_strategy (system-level default)

        Args:
            request_strategy: Strategy specified in request params (highest priority)
            agent_strategy: Strategy from agent configuration
            agent_capabilities: Agent capabilities list for inference

        Returns:
            str: Selected strategy name

        Raises:
            StrategySelectionError: If no valid strategy can be selected
            StrategyNotFoundError: If specified strategy not registered

        Example:
            >>> strategy_name = selector.select(
            ...     request_strategy="bounded_context",
            ...     agent_strategy="chain_of_thought",
            ... )
            >>> print(strategy_name)
            'bounded_context'  # Request takes precedence
        """
        # Level 1: Request-level strategy (highest priority)
        if request_strategy:
            if not self._registry.has(request_strategy):
                available = ", ".join(self._registry.list_strategies()) or "none"
                raise StrategyNotFoundError(
                    f"Requested strategy '{request_strategy}' not found. "
                    f"Available strategies: {available}"
                )
            logger.debug(
                f"Selected strategy from request level: {request_strategy}"
            )
            return request_strategy

        # Level 2: Agent-level strategy
        if agent_strategy:
            if not self._registry.has(agent_strategy):
                logger.warning(
                    f"Agent strategy '{agent_strategy}' not found, "
                    f"falling back to lower precedence"
                )
            else:
                logger.debug(
                    f"Selected strategy from agent level: {agent_strategy}"
                )
                return agent_strategy

        # Level 2.5: Infer from agent capabilities
        if agent_capabilities:
            inferred = self._infer_from_capabilities(agent_capabilities)
            if inferred:
                logger.debug(
                    f"Inferred strategy from agent capabilities: {inferred}"
                )
                return inferred

        # Level 3: System-level default (lowest priority)
        if self._default_strategy:
            if not self._registry.has(self._default_strategy):
                raise StrategySelectionError(
                    f"System default strategy '{self._default_strategy}' "
                    f"not found in registry. Check configuration."
                )
            logger.debug(
                f"Selected strategy from system default: {self._default_strategy}"
            )
            return self._default_strategy

        # No strategy found at any level
        available = ", ".join(self._registry.list_strategies()) or "none"
        raise StrategySelectionError(
            "No reasoning strategy could be selected. "
            "Please specify a strategy in the request or configure a system default. "
            f"Available strategies: {available}"
        )

    def _infer_from_capabilities(
        self, agent_capabilities: list[str]
    ) -> str | None:
        """
        Infer strategy from agent capabilities.

        Looks for capability patterns like "reasoning.strategy.{name}"
        and returns the first matching registered strategy.

        Args:
            agent_capabilities: List of agent capability strings

        Returns:
            str | None: Inferred strategy name or None if no match

        Example:
            >>> caps = ["reasoning.strategy.bounded_context", "long_form_reasoning"]
            >>> inferred = selector._infer_from_capabilities(caps)
            >>> print(inferred)
            'bounded_context'
        """
        for capability in agent_capabilities:
            # Check for explicit strategy capability pattern
            if capability.startswith("reasoning.strategy."):
                strategy_name = capability.replace("reasoning.strategy.", "")
                if self._registry.has(strategy_name):
                    return strategy_name

        return None

    def validate_request(
        self,
        request_strategy: str | None,
        agent_capabilities: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Validate if a request strategy is compatible with agent capabilities.

        Args:
            request_strategy: Strategy requested in parameters
            agent_capabilities: Agent's advertised capabilities

        Returns:
            tuple[bool, str]: (is_valid, error_message)
                - (True, "") if valid
                - (False, "error message") if invalid

        Example:
            >>> is_valid, error = selector.validate_request(
            ...     "bounded_context",
            ...     ["reasoning.strategy.bounded_context"]
            ... )
            >>> assert is_valid
        """
        if not request_strategy:
            return True, ""  # No specific strategy requested, any agent works

        # Check if strategy exists
        if not self._registry.has(request_strategy):
            available = ", ".join(self._registry.list_strategies()) or "none"
            return False, (
                f"Strategy '{request_strategy}' not available. "
                f"Available: {available}"
            )

        # If agent capabilities provided, verify agent supports this strategy
        if agent_capabilities:
            expected_capability = f"reasoning.strategy.{request_strategy}"
            if expected_capability not in agent_capabilities:
                return False, (
                    f"Agent does not support strategy '{request_strategy}'. "
                    f"Required capability: {expected_capability}"
                )

        return True, ""

    def get_default_strategy(self) -> str | None:
        """
        Get the system-level default strategy.

        Returns:
            str | None: Default strategy name or None if not configured

        Example:
            >>> default = selector.get_default_strategy()
            >>> print(default)
            'chain_of_thought'
        """
        return self._default_strategy

    def set_default_strategy(self, strategy_name: str | None) -> None:
        """
        Set the system-level default strategy.

        Args:
            strategy_name: New default strategy name (or None to clear default)

        Raises:
            StrategyNotFoundError: If strategy_name is not None and not registered

        Example:
            >>> selector.set_default_strategy("bounded_context")
        """
        if strategy_name is not None and not self._registry.has(strategy_name):
            available = ", ".join(self._registry.list_strategies()) or "none"
            raise StrategyNotFoundError(
                f"Cannot set default to '{strategy_name}' - not registered. "
                f"Available: {available}"
            )

        old_default = self._default_strategy
        self._default_strategy = strategy_name
        logger.info(
            f"Default strategy changed: {old_default} -> {strategy_name}"
        )
