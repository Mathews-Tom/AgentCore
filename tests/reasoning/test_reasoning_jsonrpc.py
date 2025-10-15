"""
Unit tests for Reasoning JSON-RPC handler.

Tests request validation, parameter handling, error cases, and A2A context propagation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agentcore.reasoning.models.reasoning_models import (
    BoundedContextIterationResult,
    BoundedContextResult,
    CarryoverContent,
    IterationMetrics,
)
from src.agentcore.reasoning.services.reasoning_jsonrpc import (
    BoundedReasoningParams,
    BoundedReasoningResult,
    handle_bounded_reasoning,
)
from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest


@pytest.fixture
def mock_reasoning_result() -> BoundedContextResult:
    """Create mock reasoning result for testing."""
    iteration = BoundedContextIterationResult(
        content="Reasoning content <answer>42</answer>",
        has_answer=True,
        answer="42",
        carryover=None,
        metrics=IterationMetrics(
            iteration=0,
            tokens=500,
            has_answer=True,
            carryover_generated=False,
            execution_time_ms=1250,
        ),
    )

    return BoundedContextResult(
        answer="42",
        iterations=[iteration],
        total_tokens=500,
        total_iterations=1,
        compute_savings_pct=15.5,
        carryover_compressions=0,
        execution_time_ms=1250,
    )


@pytest.fixture
def valid_request_params() -> dict:
    """Valid request parameters for testing."""
    return {
        "query": "What is 2+2?",
        "temperature": 0.7,
        "chunk_size": 8192,
        "carryover_size": 4096,
        "max_iterations": 5,
    }


def test_bounded_reasoning_params_validation():
    """Test BoundedReasoningParams validation."""
    # Valid params
    params = BoundedReasoningParams(
        query="Test query",
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=5,
    )
    assert params.query == "Test query"
    assert params.chunk_size == 8192
    assert params.carryover_size == 4096

    # Invalid: carryover >= chunk_size
    with pytest.raises(ValueError, match="carryover_size.*must be less than chunk_size"):
        BoundedReasoningParams(
            query="Test",
            chunk_size=4096,
            carryover_size=4096,  # Equal to chunk_size
        )

    # Invalid: empty query
    with pytest.raises(ValueError):
        BoundedReasoningParams(query="")

    # Invalid: temperature out of range
    with pytest.raises(ValueError):
        BoundedReasoningParams(query="Test", temperature=3.0)

    # Invalid: max_iterations too high
    with pytest.raises(ValueError):
        BoundedReasoningParams(query="Test", max_iterations=100)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_success(
    valid_request_params: dict,
    mock_reasoning_result: BoundedContextResult,
):
    """Test successful bounded reasoning execution."""
    # Create request
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params=valid_request_params,
        id="test-123",
    )

    # Mock engine
    with (
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient") as mock_llm_class,
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class,
    ):
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=mock_reasoning_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await handle_bounded_reasoning(request)

        # Assertions
        assert result["success"] is True
        assert result["answer"] == "42"
        assert result["total_iterations"] == 1
        assert result["total_tokens"] == 500
        assert result["compute_savings_pct"] == 15.5
        assert result["execution_time_ms"] == 1250
        assert len(result["iterations"]) == 1
        assert result["iterations"][0]["iteration"] == 0
        assert result["iterations"][0]["tokens"] == 500
        assert result["iterations"][0]["has_answer"] is True

        # Verify engine was called correctly
        mock_engine.reason.assert_called_once()
        call_args = mock_engine.reason.call_args
        assert call_args.kwargs["query"] == "What is 2+2?"
        assert call_args.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_missing_params():
    """Test error handling for missing parameters."""
    # No params
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        id="test-123",
    )

    with pytest.raises(ValueError, match="Parameters required"):
        await handle_bounded_reasoning(request)

    # Empty params
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={},
        id="test-123",
    )

    with pytest.raises(ValueError, match="Invalid parameters|Parameters required|query"):
        await handle_bounded_reasoning(request)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_invalid_params():
    """Test error handling for invalid parameters."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={
            "query": "Test",
            "chunk_size": 4096,
            "carryover_size": 4096,  # Invalid: equal to chunk_size
        },
        id="test-123",
    )

    with pytest.raises(ValueError, match="Invalid parameters"):
        await handle_bounded_reasoning(request)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_a2a_context_propagation(
    valid_request_params: dict,
    mock_reasoning_result: BoundedContextResult,
):
    """Test A2A context is properly logged and propagated."""
    # Create request with A2A context
    a2a_context = A2AContext(
        source_agent="agent-123",
        target_agent="agent-456",
        trace_id="trace-abc",
        timestamp="2025-01-01T00:00:00Z",
        session_id="session-xyz",
    )

    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params=valid_request_params,
        id="test-123",
        a2a_context=a2a_context,
    )

    # Mock engine
    with (
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class,
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.logger") as mock_logger,
    ):
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=mock_reasoning_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        await handle_bounded_reasoning(request)

        # Verify A2A context was logged
        mock_logger.info.assert_any_call(
            "bounded_reasoning_request",
            query_length=len(valid_request_params["query"]),
            chunk_size=8192,
            max_iterations=5,
            trace_id="trace-abc",
            source_agent="agent-123",
            target_agent="agent-456",
        )


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_llm_failure(valid_request_params: dict):
    """Test error handling for LLM failures."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params=valid_request_params,
        id="test-123",
    )

    # Mock engine to raise RuntimeError
    with (
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class,
    ):
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(side_effect=RuntimeError("LLM service unavailable"))
        mock_engine_class.return_value = mock_engine

        with pytest.raises(RuntimeError, match="Reasoning execution failed"):
            await handle_bounded_reasoning(request)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_unexpected_error(valid_request_params: dict):
    """Test error handling for unexpected errors."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params=valid_request_params,
        id="test-123",
    )

    # Mock engine to raise unexpected error
    with (
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class,
    ):
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(side_effect=KeyError("unexpected"))
        mock_engine_class.return_value = mock_engine

        with pytest.raises(RuntimeError, match="Unexpected error during reasoning"):
            await handle_bounded_reasoning(request)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_max_iterations_reached(
    valid_request_params: dict,
):
    """Test handling when max iterations reached without finding answer."""
    # Create result with no answer - need 5 iterations to match total_iterations
    iterations = [
        BoundedContextIterationResult(
            content=f"Iteration {i} thinking...",
            has_answer=False,
            answer=None,
            carryover=None,
            metrics=IterationMetrics(
                iteration=i,
                tokens=600,
                has_answer=False,
                carryover_generated=(i < 4),
                execution_time_ms=1000,
            ),
        )
        for i in range(5)
    ]

    result_no_answer = BoundedContextResult(
        answer="Iteration 4 thinking...",
        iterations=iterations,
        total_tokens=3000,
        total_iterations=5,
        compute_savings_pct=20.0,
        carryover_compressions=4,
        execution_time_ms=5000,
    )

    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params=valid_request_params,
        id="test-123",
    )

    # Mock engine
    with (
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class,
    ):
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=result_no_answer)
        mock_engine_class.return_value = mock_engine

        # Execute - should still succeed but log warning
        result = await handle_bounded_reasoning(request)

        # Still returns success but with max iterations content
        assert result["success"] is True
        assert result["total_iterations"] == 5
        assert result["answer"] == "Iteration 4 thinking..."


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_with_system_prompt(
    mock_reasoning_result: BoundedContextResult,
):
    """Test bounded reasoning with custom system prompt."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={
            "query": "Test query",
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.5,
        },
        id="test-123",
    )

    # Mock engine
    with (
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine") as mock_engine_class,
    ):
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=mock_reasoning_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await handle_bounded_reasoning(request)

        # Verify system prompt was passed
        mock_engine.reason.assert_called_once()
        call_args = mock_engine.reason.call_args
        assert call_args.kwargs["system_prompt"] == "You are a helpful assistant."
        assert call_args.kwargs["temperature"] == 0.5
        assert result["success"] is True


def test_bounded_reasoning_result_model():
    """Test BoundedReasoningResult model."""
    result = BoundedReasoningResult(
        success=True,
        answer="42",
        total_iterations=1,
        total_tokens=500,
        compute_savings_pct=15.5,
        carryover_compressions=0,
        execution_time_ms=1250,
        iterations=[
            {
                "iteration": 0,
                "tokens": 500,
                "has_answer": True,
                "execution_time_ms": 1250,
                "carryover_generated": False,
                "content_preview": "Test content",
            }
        ],
    )

    assert result.success is True
    assert result.answer == "42"
    assert result.total_iterations == 1
    assert len(result.iterations) == 1

    # Test serialization
    data = result.model_dump()
    assert isinstance(data, dict)
    assert data["success"] is True
    assert data["answer"] == "42"
