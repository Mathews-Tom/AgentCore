"""
Integration tests for reasoning.execute JSON-RPC method.

Tests the unified reasoning API with strategy selection and routing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.reasoning.models.reasoning_models import (
    ReasoningMetrics,
    ReasoningResult,
)
from agentcore.reasoning.services.reasoning_execute_jsonrpc import (
    handle_reasoning_execute,
)
from agentcore.reasoning.services.strategy_registry import registry


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, name: str = "mock_strategy"):
        self._name = name
        self._version = "1.0.0"

    async def execute(self, query: str, **kwargs: Any) -> ReasoningResult:
        """Mock execute implementation."""
        return ReasoningResult(
            answer=f"Mock answer from {self.name}",
            strategy_used=self.name,
            metrics=ReasoningMetrics(
                total_tokens=100,
                execution_time_ms=1000,
                strategy_specific={"mock": True},
            ),
            trace=[{"iteration": 0, "content": "mock trace"}],
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


class TestReasoningExecute:
    """Test suite for reasoning.execute JSON-RPC method."""

    def setup_method(self):
        """Set up each test with clean registry."""
        registry.clear()

        # Register mock strategies
        self.mock_strategy1 = MockStrategy(name="test_strategy1")
        self.mock_strategy2 = MockStrategy(name="test_strategy2")

        registry.register(self.mock_strategy1)
        registry.register(self.mock_strategy2)

    def teardown_method(self):
        """Clean up after each test."""
        registry.clear()

    @pytest.mark.asyncio
    async def test_execute_with_explicit_strategy(self):
        """Test reasoning.execute with explicitly requested strategy."""
        # Create request
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "What is 2+2?",
                "strategy": "test_strategy1",
            },
            id="test-1",
        )

        # Execute
        result = await handle_reasoning_execute(request)

        # Assert
        assert "answer" in result
        assert result["answer"] == "Mock answer from test_strategy1"
        assert result["strategy_used"] == "test_strategy1"
        assert "metrics" in result
        assert result["metrics"]["total_tokens"] == 100

    @pytest.mark.asyncio
    async def test_execute_with_different_strategy(self):
        """Test routing to different strategy."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "What is the meaning of life?",
                "strategy": "test_strategy2",
            },
            id="test-2",
        )

        result = await handle_reasoning_execute(request)

        assert result["answer"] == "Mock answer from test_strategy2"
        assert result["strategy_used"] == "test_strategy2"

    @pytest.mark.asyncio
    async def test_execute_with_strategy_config(self):
        """Test passing strategy-specific configuration."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Complex question",
                "strategy": "test_strategy1",
                "strategy_config": {
                    "temperature": 0.9,
                    "max_tokens": 2000,
                },
            },
            id="test-3",
        )

        result = await handle_reasoning_execute(request)

        assert result["strategy_used"] == "test_strategy1"
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_execute_nonexistent_strategy_returns_error(self):
        """Test that requesting non-existent strategy returns error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "strategy": "nonexistent_strategy",
            },
            id="test-4",
        )

        result = await handle_reasoning_execute(request)

        # Should return JSON-RPC error response
        assert "error" in result
        assert result["error"]["code"] == -32001  # Strategy not found
        assert "not found" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_query_returns_error(self):
        """Test that missing query parameter returns validation error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "strategy": "test_strategy1",
                # Missing 'query' field
            },
            id="test-5",
        )

        result = await handle_reasoning_execute(request)

        assert "error" in result
        assert result["error"]["code"] == -32602  # Invalid params

    @pytest.mark.asyncio
    async def test_execute_empty_query_returns_error(self):
        """Test that empty query returns validation error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "",  # Empty string
                "strategy": "test_strategy1",
            },
            id="test-6",
        )

        result = await handle_reasoning_execute(request)

        assert "error" in result
        assert result["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_execute_invalid_params_type_returns_error(self):
        """Test that invalid params type returns error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params="invalid string params",  # Should be dict
            id="test-7",
        )

        result = await handle_reasoning_execute(request)

        assert "error" in result
        assert result["error"]["code"] == -32602

    @pytest.mark.asyncio
    @patch("agentcore.reasoning.config.reasoning_config")
    async def test_execute_uses_default_strategy(self, mock_config):
        """Test that default strategy is used when none specified."""
        # Configure mock with default strategy
        mock_config.default_strategy = "test_strategy1"
        mock_config.enabled_strategies = ["test_strategy1", "test_strategy2"]

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query without strategy",
                # No strategy specified
            },
            id="test-8",
        )

        result = await handle_reasoning_execute(request)

        assert "answer" in result
        assert result["strategy_used"] == "test_strategy1"

    @pytest.mark.asyncio
    async def test_execute_with_agent_capabilities_inference(self):
        """Test strategy inference from agent capabilities."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "agent_capabilities": [
                    "reasoning.strategy.test_strategy2",
                    "other_capability",
                ],
            },
            id="test-9",
        )

        result = await handle_reasoning_execute(request)

        assert result["strategy_used"] == "test_strategy2"

    @pytest.mark.asyncio
    async def test_execute_returns_trace(self):
        """Test that execution trace is included in response."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "strategy": "test_strategy1",
            },
            id="test-10",
        )

        result = await handle_reasoning_execute(request)

        assert "trace" in result
        assert isinstance(result["trace"], list)
        assert len(result["trace"]) > 0

    @pytest.mark.asyncio
    async def test_execute_returns_metrics(self):
        """Test that metrics are properly returned."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "strategy": "test_strategy1",
            },
            id="test-11",
        )

        result = await handle_reasoning_execute(request)

        assert "metrics" in result
        metrics = result["metrics"]
        assert "total_tokens" in metrics
        assert "execution_time_ms" in metrics
        assert "strategy_specific" in metrics
        assert isinstance(metrics["total_tokens"], int)
        assert isinstance(metrics["execution_time_ms"], int)

    @pytest.mark.asyncio
    async def test_execute_with_a2a_context(self):
        """Test that A2A context is properly handled."""
        from agentcore.a2a_protocol.models.jsonrpc import A2AContext

        a2a_context = A2AContext(
            trace_id="test-trace-123",
            source_agent="test-agent-1",
            target_agent="reasoning-agent",
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "strategy": "test_strategy1",
            },
            id="test-12",
            a2a_context=a2a_context,
        )

        result = await handle_reasoning_execute(request)

        # Should complete without error and include trace_id in logs
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_execute_strategy_failure_returns_error(self):
        """Test that strategy execution failure returns proper error."""

        # Create a strategy that raises an exception
        class FailingStrategy(MockStrategy):
            async def execute(self, query: str, **kwargs: Any) -> ReasoningResult:
                raise RuntimeError("Strategy execution failed")

        failing_strategy = FailingStrategy(name="failing_strategy")
        registry.register(failing_strategy)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "strategy": "failing_strategy",
            },
            id="test-13",
        )

        result = await handle_reasoning_execute(request)

        assert "error" in result
        assert result["error"]["code"] == -32603  # Internal error
        assert "execution failed" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_requests_sequentially(self):
        """Test multiple requests work correctly."""
        queries = ["Query 1", "Query 2", "Query 3"]

        for i, query in enumerate(queries):
            request = JsonRpcRequest(
                jsonrpc="2.0",
                method="reasoning.execute",
                params={
                    "query": query,
                    "strategy": "test_strategy1",
                },
                id=f"test-multi-{i}",
            )

            result = await handle_reasoning_execute(request)

            assert "answer" in result
            assert result["strategy_used"] == "test_strategy1"
