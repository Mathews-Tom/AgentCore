"""
Integration Tests for modular.solve JSON-RPC Endpoint

Tests the complete orchestration flow through PEVG modules via the
modular.solve JSON-RPC method.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.jsonrpc import (
    A2AContext,
    JsonRpcRequest,
    JsonRpcVersion,
)
from agentcore.modular.jsonrpc import (
    ModularSolveConfig,
    ModularSolveRequest,
    ModularSolveResponse,
    handle_modular_solve,
)


@pytest.fixture
def a2a_context() -> A2AContext:
    """Create A2A context for testing."""
    return A2AContext(
        source_agent="test-client",
        target_agent="modular-agent",
        trace_id=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def base_request(a2a_context: A2AContext) -> JsonRpcRequest:
    """Create base JSON-RPC request."""
    return JsonRpcRequest(
        jsonrpc=JsonRpcVersion.V2_0,
        method="modular.solve",
        params={},
        id=str(uuid4()),
        a2a_context=a2a_context,
    )


# ============================================================================
# Successful Query Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_modular_solve_simple_query(
    base_request: JsonRpcRequest,
) -> None:
    """Test successful execution of simple query."""
    # Setup request with simple query that creates minimal plan
    base_request.params = {
        "query": "Hello",  # Simple query that creates single-step plan
        "config": {
            "max_iterations": 1,
            "output_format": "text",
            "include_reasoning": False,
        },
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Validate response structure
    assert isinstance(response, dict)
    assert "answer" in response
    assert "execution_trace" in response

    # Validate execution trace
    trace = response["execution_trace"]
    assert trace["iterations"] == 1
    assert len(trace["modules_invoked"]) == 4  # All four modules
    assert "planner" in trace["modules_invoked"]
    assert "executor" in trace["modules_invoked"]
    assert "verifier" in trace["modules_invoked"]
    assert "generator" in trace["modules_invoked"]
    assert trace["total_duration_ms"] > 0
    assert trace["step_count"] > 0
    assert isinstance(trace["verification_passed"], bool)
    assert 0.0 <= trace["confidence_score"] <= 1.0


@pytest.mark.asyncio
async def test_modular_solve_with_reasoning(
    base_request: JsonRpcRequest,
) -> None:
    """Test query execution with reasoning trace included."""
    # Setup request
    base_request.params = {
        "query": "Respond to this message",  # Simple query
        "config": {
            "max_iterations": 1,
            "output_format": "text",
            "include_reasoning": True,
        },
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Validate response
    assert "answer" in response
    assert "reasoning" in response
    assert response["reasoning"] is not None
    assert len(response["reasoning"]) > 0
    assert "Execution Reasoning Trace" in response["reasoning"]


@pytest.mark.asyncio
async def test_modular_solve_different_formats(
    base_request: JsonRpcRequest,
) -> None:
    """Test query execution with different output formats."""
    query = "Test"  # Simple query

    for output_format in ["text", "json", "markdown"]:
        # Setup request
        base_request.params = {
            "query": query,
            "config": {
                "max_iterations": 1,
                "output_format": output_format,
                "include_reasoning": False,
            },
        }

        # Execute
        response = await handle_modular_solve(base_request)

        # Validate response
        assert "answer" in response
        assert len(response["answer"]) > 0

        # Format-specific validation
        if output_format == "json":
            # JSON format should be valid JSON structure
            import json
            try:
                json.loads(response["answer"])
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in answer: {response['answer']}")


@pytest.mark.asyncio
async def test_modular_solve_with_sources(
    base_request: JsonRpcRequest,
) -> None:
    """Test that sources are tracked in response."""
    # Setup request
    base_request.params = {
        "query": "Info",  # Simple query
        "config": {
            "max_iterations": 1,
            "output_format": "text",
        },
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Validate sources
    assert "sources" in response
    assert isinstance(response["sources"], list)
    # Sources should include step IDs
    assert any("step:" in src for src in response["sources"])


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_modular_solve_default_config(
    base_request: JsonRpcRequest,
) -> None:
    """Test execution with default configuration."""
    # Setup request without config
    base_request.params = {
        "query": "Simple",  # Simple query
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Should succeed with defaults
    assert "answer" in response
    assert "execution_trace" in response


@pytest.mark.asyncio
async def test_modular_solve_custom_config(
    base_request: JsonRpcRequest,
) -> None:
    """Test execution with custom configuration."""
    # Setup request with custom config
    base_request.params = {
        "query": "Complex multi-step query",
        "config": {
            "max_iterations": 3,
            "timeout_seconds": 60,
            "confidence_threshold": 0.8,
            "output_format": "markdown",
            "include_reasoning": True,
        },
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Validate response
    assert "answer" in response
    assert "execution_trace" in response


# ============================================================================
# Request Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_modular_solve_missing_query(
    base_request: JsonRpcRequest,
) -> None:
    """Test error handling for missing query parameter."""
    # Setup request without query
    base_request.params = {
        "config": {
            "max_iterations": 1,
        },
    }

    # Execute and expect error
    with pytest.raises(RuntimeError) as exc_info:
        await handle_modular_solve(base_request)

    error_msg = str(exc_info.value)
    assert "query" in error_msg.lower() or "field required" in error_msg.lower()


@pytest.mark.asyncio
async def test_modular_solve_empty_query(
    base_request: JsonRpcRequest,
) -> None:
    """Test error handling for empty query."""
    # Setup request with empty query
    base_request.params = {
        "query": "",
    }

    # Execute and expect error
    with pytest.raises(RuntimeError) as exc_info:
        await handle_modular_solve(base_request)

    error_msg = str(exc_info.value)
    assert "query" in error_msg.lower() or "validation" in error_msg.lower()


@pytest.mark.asyncio
async def test_modular_solve_invalid_config(
    base_request: JsonRpcRequest,
) -> None:
    """Test error handling for invalid configuration."""
    # Setup request with invalid config
    base_request.params = {
        "query": "Test query",
        "config": {
            "max_iterations": 100,  # Exceeds maximum of 10
        },
    }

    # Execute and expect error
    with pytest.raises(RuntimeError) as exc_info:
        await handle_modular_solve(base_request)

    error_msg = str(exc_info.value)
    assert "max_iterations" in error_msg.lower() or "validation" in error_msg.lower()


@pytest.mark.asyncio
async def test_modular_solve_invalid_params_type(
    base_request: JsonRpcRequest,
) -> None:
    """Test error handling for non-dict params."""
    # Setup request with invalid params type
    base_request.params = ["invalid", "params"]

    # Execute and expect error
    with pytest.raises(RuntimeError) as exc_info:
        await handle_modular_solve(base_request)

    error_msg = str(exc_info.value)
    assert "dictionary" in error_msg.lower() or "params" in error_msg.lower()


# ============================================================================
# Execution Trace Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execution_trace_completeness(
    base_request: JsonRpcRequest,
) -> None:
    """Test that execution trace contains all required fields."""
    # Setup request
    base_request.params = {
        "query": "Test query for trace validation",
        "config": {
            "max_iterations": 1,
        },
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Validate trace completeness
    trace = response["execution_trace"]
    required_fields = [
        "plan_id",
        "iterations",
        "modules_invoked",
        "total_duration_ms",
        "verification_passed",
        "step_count",
        "successful_steps",
        "failed_steps",
        "confidence_score",
    ]

    for field in required_fields:
        assert field in trace, f"Missing required field: {field}"


@pytest.mark.asyncio
async def test_execution_trace_module_sequence(
    base_request: JsonRpcRequest,
) -> None:
    """Test that modules are invoked in correct sequence."""
    # Setup request
    base_request.params = {
        "query": "Test module sequence",
        "config": {
            "max_iterations": 1,
        },
    }

    # Execute
    response = await handle_modular_solve(base_request)

    # Validate module sequence
    trace = response["execution_trace"]
    modules = trace["modules_invoked"]

    # Expected sequence: planner → executor → verifier → generator
    assert modules[0] == "planner"
    assert modules[1] == "executor"
    assert modules[2] == "verifier"
    assert modules[3] == "generator"


# ============================================================================
# Pydantic Model Tests
# ============================================================================


def test_modular_solve_config_validation() -> None:
    """Test ModularSolveConfig validation."""
    # Valid config
    config = ModularSolveConfig(
        max_iterations=5,
        timeout_seconds=300,
        confidence_threshold=0.7,
    )
    assert config.max_iterations == 5
    assert config.timeout_seconds == 300
    assert config.confidence_threshold == 0.7

    # Test defaults
    default_config = ModularSolveConfig()
    assert default_config.max_iterations == 5
    assert default_config.timeout_seconds == 300
    assert default_config.confidence_threshold == 0.7
    assert default_config.output_format == "text"
    assert default_config.include_reasoning is False


def test_modular_solve_config_constraints() -> None:
    """Test ModularSolveConfig constraint validation."""
    # max_iterations must be between 1 and 10
    with pytest.raises(Exception):
        ModularSolveConfig(max_iterations=0)

    with pytest.raises(Exception):
        ModularSolveConfig(max_iterations=11)

    # confidence_threshold must be between 0.0 and 1.0
    with pytest.raises(Exception):
        ModularSolveConfig(confidence_threshold=-0.1)

    with pytest.raises(Exception):
        ModularSolveConfig(confidence_threshold=1.1)


def test_modular_solve_request_validation() -> None:
    """Test ModularSolveRequest validation."""
    # Valid request
    request = ModularSolveRequest(
        query="Test query",
        config=ModularSolveConfig(max_iterations=3),
    )
    assert request.query == "Test query"
    assert request.config.max_iterations == 3

    # Query is required
    with pytest.raises(Exception):
        ModularSolveRequest(query=None)

    # Empty query should fail
    with pytest.raises(Exception):
        ModularSolveRequest(query="")


def test_modular_solve_response_structure() -> None:
    """Test ModularSolveResponse structure."""
    from agentcore.modular.jsonrpc import ExecutionTrace

    trace = ExecutionTrace(
        plan_id="test-plan",
        iterations=1,
        modules_invoked=["planner", "executor", "verifier", "generator"],
        total_duration_ms=1500,
        verification_passed=True,
        step_count=3,
        successful_steps=3,
        failed_steps=0,
        confidence_score=0.95,
    )

    response = ModularSolveResponse(
        answer="Test answer",
        execution_trace=trace,
        reasoning="Test reasoning",
        sources=["source1", "source2"],
    )

    assert response.answer == "Test answer"
    assert response.execution_trace.plan_id == "test-plan"
    assert response.reasoning == "Test reasoning"
    assert len(response.sources) == 2
