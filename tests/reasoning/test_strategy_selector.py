"""
Unit tests for StrategySelector.

Tests multi-level strategy selection logic with precedence rules.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.reasoning.models.reasoning_models import ReasoningResult
from agentcore.reasoning.services.strategy_registry import ReasoningStrategyRegistry
from agentcore.reasoning.services.strategy_selector import (
    StrategySelectionError,
    StrategySelector,
)


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, name: str = "mock_strategy", version: str = "1.0.0"):
        self._name = name
        self._version = version

    async def reason(self, query: str, **kwargs: Any) -> ReasoningResult:
        """Mock reason implementation."""
        return ReasoningResult(
            answer="mock answer",
            strategy_used=self.name,
            metrics={"total_tokens": 100, "execution_time_ms": 1000},
            trace=None,
        )

    def get_config_schema(self) -> dict[str, Any]:
        """Mock config schema."""
        return {"type": "object", "properties": {}}

    def get_capabilities(self) -> list[str]:
        """Mock capabilities."""
        return [f"reasoning.strategy.{self.name}"]

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    @property
    def version(self) -> str:
        """Strategy version."""
        return self._version


class TestStrategySelector:
    """Test suite for StrategySelector."""

    def setup_method(self):
        """Set up each test with clean registry and selector."""
        self.registry = ReasoningStrategyRegistry()
        self.registry.clear()

        # Register test strategies
        self.strategy1 = MockStrategy(name="strategy1")
        self.strategy2 = MockStrategy(name="strategy2")
        self.strategy3 = MockStrategy(name="strategy3")

        self.registry.register(self.strategy1)
        self.registry.register(self.strategy2)
        self.registry.register(self.strategy3)

        # Create selector with no default
        self.selector = StrategySelector(self.registry, default_strategy=None)

    def teardown_method(self):
        """Clean up after each test."""
        self.registry.clear()

    def test_select_from_request_level(self):
        """Test that request-level strategy has highest priority."""
        selected = self.selector.select(
            request_strategy="strategy1",
            agent_strategy="strategy2",
        )

        assert selected == "strategy1"

    def test_select_from_agent_level(self):
        """Test that agent-level strategy is used when no request strategy."""
        selected = self.selector.select(
            request_strategy=None,
            agent_strategy="strategy2",
        )

        assert selected == "strategy2"

    def test_select_from_system_default(self):
        """Test that system default is used when no higher level strategy."""
        selector = StrategySelector(self.registry, default_strategy="strategy3")

        selected = selector.select(
            request_strategy=None,
            agent_strategy=None,
        )

        assert selected == "strategy3"

    def test_select_infer_from_agent_capabilities(self):
        """Test inferring strategy from agent capabilities."""
        selected = self.selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=["reasoning.strategy.strategy2", "other_capability"],
        )

        assert selected == "strategy2"

    def test_select_precedence_order(self):
        """Test full precedence: request > agent > capabilities > default."""
        selector = StrategySelector(self.registry, default_strategy="strategy3")

        # Request overrides all
        selected = selector.select(
            request_strategy="strategy1",
            agent_strategy="strategy2",
            agent_capabilities=["reasoning.strategy.strategy3"],
        )
        assert selected == "strategy1"

        # Agent overrides capabilities and default
        selected = selector.select(
            request_strategy=None,
            agent_strategy="strategy2",
            agent_capabilities=["reasoning.strategy.strategy3"],
        )
        assert selected == "strategy2"

        # Capabilities override default
        selected = selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=["reasoning.strategy.strategy1"],
        )
        assert selected == "strategy1"

    def test_select_nonexistent_request_strategy_raises_error(self):
        """Test that requesting non-existent strategy raises error."""
        with pytest.raises(Exception) as exc_info:
            self.selector.select(request_strategy="nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert "strategy1" in str(exc_info.value)  # Shows available

    def test_select_invalid_agent_strategy_falls_back(self):
        """Test that invalid agent strategy falls back to lower precedence."""
        selector = StrategySelector(self.registry, default_strategy="strategy1")

        selected = selector.select(
            request_strategy=None,
            agent_strategy="nonexistent",  # Invalid, should fall back
        )

        assert selected == "strategy1"  # Falls back to system default

    def test_select_no_strategy_available_raises_error(self):
        """Test that no strategy at any level raises error."""
        with pytest.raises(StrategySelectionError) as exc_info:
            self.selector.select(
                request_strategy=None,
                agent_strategy=None,
                agent_capabilities=None,
            )

        error_msg = str(exc_info.value).lower()
        assert "no reasoning strategy" in error_msg
        assert "strategy1" in error_msg  # Shows available strategies

    def test_infer_from_capabilities_multiple_matches(self):
        """Test inference when multiple strategy capabilities present (uses first)."""
        selected = self.selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=[
                "reasoning.strategy.strategy1",
                "reasoning.strategy.strategy2",
            ],
        )

        # Should use first matching strategy
        assert selected in ["strategy1", "strategy2"]

    def test_infer_from_capabilities_no_match(self):
        """Test inference falls back when no capability matches."""
        selector = StrategySelector(self.registry, default_strategy="strategy3")

        selected = selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=["other_capability", "another_capability"],
        )

        assert selected == "strategy3"  # Falls back to default

    def test_validate_request_valid_strategy(self):
        """Test validating a valid request strategy."""
        is_valid, error = self.selector.validate_request(
            request_strategy="strategy1",
            agent_capabilities=["reasoning.strategy.strategy1"],
        )

        assert is_valid
        assert error == ""

    def test_validate_request_nonexistent_strategy(self):
        """Test validating non-existent strategy returns error."""
        is_valid, error = self.selector.validate_request(
            request_strategy="nonexistent",
        )

        assert not is_valid
        assert "not available" in error.lower()
        assert "strategy1" in error  # Shows available

    def test_validate_request_agent_not_capable(self):
        """Test validating strategy agent doesn't support."""
        is_valid, error = self.selector.validate_request(
            request_strategy="strategy1",
            agent_capabilities=["reasoning.strategy.strategy2"],  # Different strategy
        )

        assert not is_valid
        assert "does not support" in error.lower()
        assert "strategy1" in error

    def test_validate_request_none_strategy(self):
        """Test validating None strategy (always valid)."""
        is_valid, error = self.selector.validate_request(
            request_strategy=None,
        )

        assert is_valid
        assert error == ""

    def test_validate_request_no_capabilities(self):
        """Test validating when agent capabilities not provided."""
        is_valid, error = self.selector.validate_request(
            request_strategy="strategy1",
            agent_capabilities=None,
        )

        # Valid because we can't check agent capabilities
        assert is_valid
        assert error == ""

    def test_get_default_strategy(self):
        """Test getting the default strategy."""
        selector = StrategySelector(self.registry, default_strategy="strategy2")

        assert selector.get_default_strategy() == "strategy2"

    def test_get_default_strategy_none(self):
        """Test getting default when none configured."""
        assert self.selector.get_default_strategy() is None

    def test_set_default_strategy(self):
        """Test setting the default strategy."""
        assert self.selector.get_default_strategy() is None

        self.selector.set_default_strategy("strategy1")

        assert self.selector.get_default_strategy() == "strategy1"

        # Verify it's actually used in selection
        selected = self.selector.select(
            request_strategy=None,
            agent_strategy=None,
        )
        assert selected == "strategy1"

    def test_set_default_strategy_to_none(self):
        """Test clearing the default strategy."""
        selector = StrategySelector(self.registry, default_strategy="strategy1")

        selector.set_default_strategy(None)

        assert selector.get_default_strategy() is None

    def test_set_default_strategy_nonexistent_raises_error(self):
        """Test that setting default to non-existent strategy raises error."""
        with pytest.raises(Exception) as exc_info:
            self.selector.set_default_strategy("nonexistent")

        assert "not registered" in str(exc_info.value).lower()

    def test_invalid_system_default_raises_error_on_use(self):
        """Test that invalid system default raises error when used."""
        # Create selector with invalid default
        selector = StrategySelector(self.registry, default_strategy="invalid")
        # Unregister a strategy to make default invalid
        self.registry.unregister("strategy1")
        self.registry.unregister("strategy2")
        self.registry.unregister("strategy3")

        with pytest.raises(Exception):
            # Should fail when trying to use invalid default
            selector.select(
                request_strategy=None,
                agent_strategy=None,
            )
