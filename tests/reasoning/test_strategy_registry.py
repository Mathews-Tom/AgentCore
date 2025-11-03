"""
Unit tests for ReasoningStrategyRegistry.

Tests strategy registration, discovery, and management functionality.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.reasoning.models.reasoning_models import ReasoningResult
from agentcore.reasoning.protocol import ReasoningStrategy
from agentcore.reasoning.services.strategy_registry import (
    ReasoningStrategyRegistry,
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
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


class TestReasoningStrategyRegistry:
    """Test suite for ReasoningStrategyRegistry."""

    def setup_method(self):
        """Set up each test with a clean registry."""
        # Clear registry before each test
        self.registry = ReasoningStrategyRegistry()
        self.registry.clear()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear registry after each test
        self.registry.clear()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = ReasoningStrategyRegistry()
        registry2 = ReasoningStrategyRegistry()
        assert registry1 is registry2

    def test_register_strategy(self):
        """Test registering a new strategy."""
        strategy = MockStrategy(name="test_strategy")
        self.registry.register(strategy)

        assert self.registry.has("test_strategy")
        assert "test_strategy" in self.registry.list_strategies()

    def test_register_duplicate_strategy_raises_error(self):
        """Test that registering duplicate strategy raises error."""
        strategy = MockStrategy(name="test_strategy")
        self.registry.register(strategy)

        with pytest.raises(StrategyAlreadyRegisteredError) as exc_info:
            self.registry.register(strategy)

        assert "already registered" in str(exc_info.value).lower()

    def test_register_duplicate_with_allow_override(self):
        """Test that registering duplicate with allow_override works."""
        strategy1 = MockStrategy(name="test_strategy", version="1.0.0")
        strategy2 = MockStrategy(name="test_strategy", version="2.0.0")

        self.registry.register(strategy1)
        self.registry.register(strategy2, allow_override=True)

        retrieved = self.registry.get("test_strategy")
        assert retrieved.version == "2.0.0"

    def test_register_invalid_name_raises_error(self):
        """Test that registering strategy with empty name raises error."""
        strategy = MockStrategy(name="")

        with pytest.raises(ValueError) as exc_info:
            self.registry.register(strategy)

        assert "name cannot be empty" in str(exc_info.value).lower()

    def test_unregister_strategy(self):
        """Test unregistering a strategy."""
        strategy = MockStrategy(name="test_strategy")
        self.registry.register(strategy)

        assert self.registry.has("test_strategy")

        self.registry.unregister("test_strategy")

        assert not self.registry.has("test_strategy")
        assert "test_strategy" not in self.registry.list_strategies()

    def test_unregister_nonexistent_raises_error(self):
        """Test that unregistering non-existent strategy raises error."""
        with pytest.raises(StrategyNotFoundError) as exc_info:
            self.registry.unregister("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_get_strategy(self):
        """Test retrieving a registered strategy."""
        strategy = MockStrategy(name="test_strategy")
        self.registry.register(strategy)

        retrieved = self.registry.get("test_strategy")

        assert retrieved is strategy
        assert retrieved.name == "test_strategy"

    def test_get_nonexistent_strategy_raises_error(self):
        """Test that getting non-existent strategy raises error with available list."""
        strategy = MockStrategy(name="existing_strategy")
        self.registry.register(strategy)

        with pytest.raises(StrategyNotFoundError) as exc_info:
            self.registry.get("nonexistent")

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg
        assert "existing_strategy" in error_msg

    def test_has_strategy(self):
        """Test checking if strategy exists."""
        strategy = MockStrategy(name="test_strategy")

        assert not self.registry.has("test_strategy")

        self.registry.register(strategy)

        assert self.registry.has("test_strategy")

    def test_list_strategies(self):
        """Test listing all registered strategies."""
        assert self.registry.list_strategies() == []

        strategy1 = MockStrategy(name="strategy1")
        strategy2 = MockStrategy(name="strategy2")

        self.registry.register(strategy1)
        self.registry.register(strategy2)

        strategies = self.registry.list_strategies()
        assert len(strategies) == 2
        assert "strategy1" in strategies
        assert "strategy2" in strategies

    def test_get_metadata(self):
        """Test getting metadata for a strategy."""
        strategy = MockStrategy(name="test_strategy", version="1.2.3")
        self.registry.register(strategy)

        metadata = self.registry.get_metadata("test_strategy")

        assert metadata["name"] == "test_strategy"
        assert metadata["version"] == "1.2.3"
        assert "capabilities" in metadata
        assert "config_schema" in metadata
        assert "reasoning.strategy.test_strategy" in metadata["capabilities"]

    def test_get_metadata_nonexistent_raises_error(self):
        """Test that getting metadata for non-existent strategy raises error."""
        with pytest.raises(StrategyNotFoundError):
            self.registry.get_metadata("nonexistent")

    def test_list_all_metadata(self):
        """Test listing metadata for all strategies."""
        strategy1 = MockStrategy(name="strategy1", version="1.0.0")
        strategy2 = MockStrategy(name="strategy2", version="2.0.0")

        self.registry.register(strategy1)
        self.registry.register(strategy2)

        all_metadata = self.registry.list_all_metadata()

        assert len(all_metadata) == 2
        names = [m["name"] for m in all_metadata]
        assert "strategy1" in names
        assert "strategy2" in names

    def test_clear_registry(self):
        """Test clearing all strategies from registry."""
        strategy1 = MockStrategy(name="strategy1")
        strategy2 = MockStrategy(name="strategy2")

        self.registry.register(strategy1)
        self.registry.register(strategy2)

        assert len(self.registry.list_strategies()) == 2

        self.registry.clear()

        assert len(self.registry.list_strategies()) == 0
        assert not self.registry.has("strategy1")
        assert not self.registry.has("strategy2")

    def test_register_multiple_strategies(self):
        """Test registering multiple different strategies."""
        strategies = [
            MockStrategy(name="strategy1", version="1.0.0"),
            MockStrategy(name="strategy2", version="1.0.0"),
            MockStrategy(name="strategy3", version="1.0.0"),
        ]

        for strategy in strategies:
            self.registry.register(strategy)

        assert len(self.registry.list_strategies()) == 3

        for strategy in strategies:
            assert self.registry.has(strategy.name)
            retrieved = self.registry.get(strategy.name)
            assert retrieved is strategy
