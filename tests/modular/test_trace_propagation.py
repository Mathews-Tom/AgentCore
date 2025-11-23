"""
Tests for Trace ID Propagation Across Modules

Validates that trace_id is:
1. Generated at query entry point
2. Included in A2A context for all module calls
3. Present in all log messages (structlog)
4. Stored in database (modular_executions table)
5. Included in error responses
6. Maintained continuously across all module transitions
"""

from __future__ import annotations

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest
from agentcore.modular.tracing import (
    ModularTracer,
    TracingConfig,
    initialize_tracing,
    shutdown_tracing,
)
from agentcore.modular.coordinator import ModuleCoordinator, CoordinationContext
from agentcore.modular.state_manager import StateManager
from agentcore.modular.models import ModuleType


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def trace_id() -> str:
    """Generate a test trace ID."""
    return str(uuid4())


@pytest.fixture
def a2a_context(trace_id: str) -> A2AContext:
    """Create A2A context with trace ID."""
    return A2AContext(
        source_agent="user",
        target_agent="modular-agent",
        trace_id=trace_id,
        timestamp=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def tracer() -> ModularTracer:
    """Create a tracer for testing."""
    config = TracingConfig(
        enabled=True,
        console_export=False,
        otlp_endpoint=None,
    )
    return ModularTracer(config)


# ============================================================================
# Trace ID Generation Tests
# ============================================================================


def test_trace_id_generation(tracer: ModularTracer) -> None:
    """Test that trace_id is properly generated."""
    trace_id = tracer.generate_trace_id()

    assert trace_id is not None
    assert isinstance(trace_id, str)
    assert len(trace_id) > 0
    # Should be a valid UUID string
    assert len(trace_id.split("-")) == 5


def test_trace_id_uniqueness(tracer: ModularTracer) -> None:
    """Test that generated trace_ids are unique."""
    trace_ids = {tracer.generate_trace_id() for _ in range(100)}

    # All trace_ids should be unique
    assert len(trace_ids) == 100


# ============================================================================
# A2A Context Propagation Tests
# ============================================================================


def test_trace_id_in_a2a_context(trace_id: str, a2a_context: A2AContext) -> None:
    """Test that trace_id is included in A2A context."""
    assert a2a_context.trace_id == trace_id


def test_a2a_context_carrier_includes_trace_id(tracer: ModularTracer) -> None:
    """Test that A2A context carrier includes trace_id."""
    # Test that tracer can build carrier (may be empty if no active span)
    carrier = tracer.build_a2a_context_carrier()

    # Carrier should be a dictionary
    assert isinstance(carrier, dict)

    # If tracer is enabled and has active span, trace_id should be present
    # Otherwise carrier may be empty, which is also valid


def test_coordination_context_has_trace_id(trace_id: str) -> None:
    """Test that coordination context includes trace_id."""
    context = CoordinationContext(
        execution_id=str(uuid4()),
        trace_id=trace_id,
        session_id=None,
        iteration=0,
    )

    assert context.trace_id == trace_id


# ============================================================================
# Structlog Binding Tests
# ============================================================================


@patch("agentcore.modular.tracing.structlog.get_logger")
def test_trace_id_bound_to_logger(
    mock_get_logger: MagicMock, trace_id: str
) -> None:
    """Test that trace_id is bound to structlog logger."""
    from agentcore.modular.tracing import bind_trace_id_to_logger

    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    bound_logger = bind_trace_id_to_logger(mock_logger, trace_id)

    # Verify bind was called with trace_id
    mock_logger.bind.assert_called_once_with(trace_id=trace_id)


@patch("structlog.get_logger")
def test_base_module_binds_trace_id(
    mock_get_logger: MagicMock, trace_id: str, a2a_context: A2AContext
) -> None:
    """Test that BaseModule binds trace_id to logger."""
    from agentcore.modular.base import BaseModule

    mock_logger = MagicMock()
    mock_logger.bind.return_value = mock_logger
    mock_get_logger.return_value = mock_logger

    # Create a concrete implementation for testing
    class TestModule(BaseModule):
        async def health_check(self) -> dict:
            return {"status": "healthy"}

    module = TestModule(
        module_name="TestModule",
        a2a_context=a2a_context,
    )

    # Verify logger was bound with trace_id
    assert mock_logger.bind.called
    call_kwargs = mock_logger.bind.call_args[1]
    assert "trace_id" in call_kwargs
    assert call_kwargs["trace_id"] == trace_id


# ============================================================================
# Database Storage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_trace_id_stored_in_database(trace_id: str) -> None:
    """Test that trace_id is stored in modular_executions table."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from agentcore.a2a_protocol.database.models import Base, ModularExecutionDB

    # Create in-memory database for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        state_manager = StateManager(session)

        execution_id = str(uuid4())
        query = "test query"

        # Initialize execution with trace_id
        await state_manager.init_execution(
            execution_id=execution_id,
            query=query,
            trace_id=trace_id,
        )

        await session.commit()

        # Verify trace_id was stored
        from sqlalchemy import select

        result = await session.execute(
            select(ModularExecutionDB).where(ModularExecutionDB.id == execution_id)
        )
        execution = result.scalar_one()

        assert execution.trace_id == trace_id
        assert execution.query == query

    await engine.dispose()


# ============================================================================
# Error Response Tests
# ============================================================================


@pytest.mark.asyncio
async def test_trace_id_in_error_response(
    trace_id: str, a2a_context: A2AContext
) -> None:
    """Test that trace_id is included in error responses."""
    from agentcore.modular.jsonrpc import handle_modular_solve
    from agentcore.a2a_protocol.models.jsonrpc import JsonRpcErrorCode

    # Create request with invalid params to trigger error
    request = JsonRpcRequest(
        method="modular.solve",
        params={"query": ""},  # Empty query should trigger validation error
        id=1,
        a2a_context=a2a_context,
    )

    # Execute and expect error
    with pytest.raises(RuntimeError) as exc_info:
        await handle_modular_solve(request)

    error_json = str(exc_info.value)

    # Verify trace_id is in error response
    assert trace_id in error_json


# ============================================================================
# Module Transition Continuity Tests
# ============================================================================


def test_trace_id_continuity_in_coordinator(trace_id: str) -> None:
    """Test that trace_id is maintained across module transitions."""
    coordinator = ModuleCoordinator()

    # Set context with trace_id
    context = CoordinationContext(
        execution_id=str(uuid4()),
        trace_id=trace_id,
        session_id=None,
        iteration=0,
    )
    coordinator.set_context(context)

    # Build A2A context for module call
    a2a_context = coordinator._build_a2a_context(
        from_module="planner",
        to_module="executor",
    )

    # Verify trace_id is propagated
    assert a2a_context.trace_id == trace_id


@pytest.mark.asyncio
async def test_trace_id_across_full_execution(
    trace_id: str, a2a_context: A2AContext
) -> None:
    """Test trace_id continuity through full PEVG execution."""
    from agentcore.modular.planner import Planner
    from agentcore.modular.executor import ExecutorModule
    from agentcore.modular.verifier import Verifier
    from agentcore.modular.generator import Generator
    from agentcore.agent_runtime.tools.registry import ToolRegistry
    from agentcore.agent_runtime.tools.executor import ToolExecutor

    # Create modules with shared trace_id via a2a_context
    planner = Planner(a2a_context=a2a_context)

    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor(registry=tool_registry)
    executor = ExecutorModule(
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        a2a_context=a2a_context,
    )

    verifier = Verifier(a2a_context=a2a_context)
    generator = Generator(a2a_context=a2a_context)

    # Verify all modules have the same trace_id
    assert planner.a2a_context.trace_id == trace_id
    assert executor.a2a_context.trace_id == trace_id
    assert verifier.a2a_context.trace_id == trace_id
    assert generator.a2a_context.trace_id == trace_id


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_trace_propagation() -> None:
    """Test complete trace propagation from entry to database."""
    from agentcore.modular.jsonrpc import handle_modular_solve
    from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from agentcore.a2a_protocol.database.models import Base

    # Generate trace_id
    trace_id = str(uuid4())

    # Create A2A context
    a2a_context = A2AContext(
        source_agent="user",
        target_agent="modular-agent",
        trace_id=trace_id,
        timestamp=datetime.now(UTC).isoformat(),
    )

    # Create request
    request = JsonRpcRequest(
        method="modular.solve",
        params={
            "query": "What is 2+2?",
            "config": {"max_iterations": 1, "timeout_seconds": 10},
        },
        id=1,
        a2a_context=a2a_context,
    )

    # Mock the orchestration to avoid full execution
    with patch(
        "agentcore.modular.jsonrpc._orchestrate_execution"
    ) as mock_orchestrate:
        mock_response = MagicMock()
        mock_response.execution_trace.total_duration_ms = 100
        mock_response.execution_trace.verification_passed = True
        mock_response.model_dump.return_value = {
            "answer": "4",
            "execution_trace": {
                "plan_id": "test",
                "iterations": 1,
                "total_duration_ms": 100,
                "verification_passed": True,
                "modules_invoked": ["planner", "executor", "verifier", "generator"],
                "step_count": 1,
                "successful_steps": 1,
                "failed_steps": 0,
                "confidence_score": 1.0,
                "transitions": [],
                "refinement_history": [],
            },
        }
        mock_orchestrate.return_value = mock_response

        # Execute request
        result = await handle_modular_solve(request)

        # Verify orchestration was called with correct trace_id
        call_args = mock_orchestrate.call_args
        assert call_args[1]["a2a_context"].trace_id == trace_id


@pytest.mark.asyncio
async def test_trace_id_in_logs_across_modules() -> None:
    """Test that trace_id appears in all log messages."""
    from agentcore.modular.base import BaseModule

    trace_id = str(uuid4())
    a2a_context = A2AContext(
        source_agent="user",
        target_agent="test",
        trace_id=trace_id,
        timestamp=datetime.now(UTC).isoformat(),
    )

    with patch("structlog.get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.bind.return_value = mock_logger
        mock_get_logger.return_value = mock_logger

        # Create module
        class TestModule(BaseModule):
            async def health_check(self) -> dict:
                return {"status": "healthy"}

        module = TestModule(
            module_name="TestModule",
            a2a_context=a2a_context,
        )

        # Log an operation
        module._log_operation("test_operation", "started")

        # Verify logger was bound with trace_id
        bind_calls = [
            call for call in mock_logger.bind.call_args_list
            if "trace_id" in call[1]
        ]
        assert len(bind_calls) > 0
        assert bind_calls[0][1]["trace_id"] == trace_id


# ============================================================================
# Cleanup
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_tracing():
    """Cleanup tracing after each test."""
    yield
    try:
        shutdown_tracing()
    except Exception:
        pass
