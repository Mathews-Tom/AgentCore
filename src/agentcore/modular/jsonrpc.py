"""
JSON-RPC Methods for Modular Agent Core

Implements the primary modular.solve endpoint and supporting JSON-RPC methods
for the PEVG (Planner, Executor, Verifier, Generator) modular architecture.

Primary endpoint:
- modular.solve: Orchestrates query resolution through all four modules
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.jsonrpc import (
    A2AContext,
    JsonRpcErrorCode,
    JsonRpcRequest,
    create_error_response,
)
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.modular.executor import ExecutorModule
from agentcore.modular.generator import Generator
from agentcore.modular.interfaces import (
    ExecutionResult,
    GenerationRequest,
    PlannerQuery,
    VerificationRequest,
)
from agentcore.modular.planner import Planner
from agentcore.modular.verifier import Verifier
from agentcore.modular.coordinator import ModuleCoordinator, CoordinationContext

logger = structlog.get_logger()

# ============================================================================
# Request/Response Models
# ============================================================================


class ModularSolveConfig(BaseModel):
    """Configuration for modular.solve execution."""

    max_iterations: int = Field(
        default=5, ge=1, le=10, description="Maximum refinement iterations"
    )
    planner_version: str | None = Field(
        None, description="Specific planner version to use"
    )
    executor_version: str | None = Field(
        None, description="Specific executor version to use"
    )
    verifier_version: str | None = Field(
        None, description="Specific verifier version to use"
    )
    generator_version: str | None = Field(
        None, description="Specific generator version to use"
    )
    timeout_seconds: int = Field(
        default=300, ge=1, le=600, description="Overall execution timeout"
    )
    enable_llm_verification: bool = Field(
        default=False, description="Enable LLM-based verification"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for verification"
    )
    output_format: str = Field(
        default="text",
        description="Output format: text, json, markdown, html",
    )
    include_reasoning: bool = Field(
        default=False, description="Include execution trace in response"
    )


class ModularSolveRequest(BaseModel):
    """Request for modular.solve method."""

    query: str = Field(..., min_length=1, description="User query to solve")
    config: ModularSolveConfig | None = Field(
        None, description="Optional execution configuration"
    )


class ExecutionTrace(BaseModel):
    """Trace of module execution for observability."""

    plan_id: str = Field(..., description="Execution plan ID")
    iterations: int = Field(..., description="Number of iterations taken")
    modules_invoked: list[str] = Field(..., description="Module execution sequence")
    total_duration_ms: int = Field(..., description="Total execution time in ms")
    verification_passed: bool = Field(..., description="Final verification status")
    step_count: int = Field(..., description="Number of plan steps executed")
    successful_steps: int = Field(..., description="Number of successful steps")
    failed_steps: int = Field(..., description="Number of failed steps")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Final confidence score"
    )
    transitions: list[dict[str, Any]] = Field(
        default_factory=list, description="Module transition events"
    )
    refinement_history: list[dict[str, Any]] = Field(
        default_factory=list, description="History of plan refinements"
    )
    timeout: bool = Field(default=False, description="Whether execution timed out")


class ModularSolveResponse(BaseModel):
    """Response from modular.solve method."""

    answer: str = Field(..., description="Final generated response")
    execution_trace: ExecutionTrace = Field(
        ..., description="Trace of module execution"
    )
    reasoning: str | None = Field(
        None, description="Reasoning trace if requested"
    )
    sources: list[str] = Field(
        default_factory=list, description="Evidence sources from execution"
    )


# ============================================================================
# Module Instance Management
# ============================================================================

# Global module instances (initialized lazily)
_tool_registry: ToolRegistry | None = None
_tool_executor: ToolExecutor | None = None


def _get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        logger.info("tool_registry_initialized")
    return _tool_registry


def _get_tool_executor() -> ToolExecutor:
    """Get or create global tool executor."""
    global _tool_executor
    if _tool_executor is None:
        registry = _get_tool_registry()
        _tool_executor = ToolExecutor(registry=registry)
        logger.info("tool_executor_initialized")
    return _tool_executor


def _create_modules(
    a2a_context: A2AContext,
    config: ModularSolveConfig,
) -> tuple[Planner, ExecutorModule, Verifier, Generator]:
    """
    Create PEVG module instances with configuration.

    Args:
        a2a_context: A2A context for tracing
        config: Configuration for module behavior

    Returns:
        Tuple of (planner, executor, verifier, generator)
    """
    # Create Planner
    planner = Planner(
        a2a_context=a2a_context,
        max_steps=20,
        enable_parallel=False,
    )

    # Create Executor with tool integration
    executor = ExecutorModule(
        tool_registry=_get_tool_registry(),
        tool_executor=_get_tool_executor(),
        a2a_context=a2a_context,
        max_parallel_steps=5,
        enable_circuit_breaker=True,
    )

    # Create Verifier with configuration
    verifier = Verifier(
        a2a_context=a2a_context,
        enable_llm_verification=config.enable_llm_verification,
        confidence_threshold=config.confidence_threshold,
    )

    # Create Generator
    generator = Generator(
        a2a_context=a2a_context,
    )

    logger.info(
        "modules_created",
        planner_max_steps=20,
        executor_max_parallel=5,
        verifier_llm_enabled=config.enable_llm_verification,
        verifier_threshold=config.confidence_threshold,
    )

    return planner, executor, verifier, generator


# ============================================================================
# Orchestration Logic
# ============================================================================


async def _orchestrate_execution(
    query: str,
    config: ModularSolveConfig,
    a2a_context: A2AContext,
) -> ModularSolveResponse:
    """
    Orchestrate execution through PEVG modules with refinement loop.

    Full orchestration flow (MOD-018 - coordination loop with refinement):
    1. Planner: Create execution plan from query
    2. Executor: Execute plan steps with tool invocation
    3. Verifier: Validate execution results
    4. If verification fails AND iterations < max:
       - Planner: Refine plan with verifier feedback
       - Goto step 2 (execute refined plan)
    5. Generator: Synthesize final response

    Args:
        query: User query to solve
        config: Execution configuration
        a2a_context: A2A context for tracing

    Returns:
        Modular solve response with answer and trace

    Raises:
        RuntimeError: If execution fails
    """
    logger.info(
        "orchestration_started",
        query=query,
        trace_id=a2a_context.trace_id,
        max_iterations=config.max_iterations,
    )

    # Create module instances
    planner, executor, verifier, generator = _create_modules(a2a_context, config)

    # Create coordinator and set context
    coordinator = ModuleCoordinator()
    coordination_context = CoordinationContext(
        execution_id=str(uuid4()),
        trace_id=a2a_context.trace_id,
        session_id=a2a_context.session_id,
        iteration=0,
    )
    coordinator.set_context(coordination_context)

    # Execute coordination loop with refinement
    result = await coordinator.execute_with_refinement(
        query=query,
        planner=planner,
        executor=executor,
        verifier=verifier,
        generator=generator,
        max_iterations=config.max_iterations,
        timeout_seconds=config.timeout_seconds,
        confidence_threshold=config.confidence_threshold,
        output_format=config.output_format,
        include_reasoning=config.include_reasoning,
    )

    # Convert result dict to ModularSolveResponse
    execution_trace = ExecutionTrace(**result["execution_trace"])

    logger.info(
        "orchestration_complete",
        trace_id=a2a_context.trace_id,
        duration_ms=execution_trace.total_duration_ms,
        verification_passed=execution_trace.verification_passed,
        iterations=execution_trace.iterations,
    )

    return ModularSolveResponse(
        answer=result["answer"],
        execution_trace=execution_trace,
        reasoning=result.get("reasoning"),
        sources=result.get("sources", []),
    )


# ============================================================================
# JSON-RPC Handler
# ============================================================================


@register_jsonrpc_method("modular.solve")
async def handle_modular_solve(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Primary JSON-RPC endpoint for modular agent execution.

    Orchestrates query resolution through Planner → Executor → Verifier → Generator.

    Request params:
        query (str): User query to solve (required)
        config (ModularSolveConfig): Optional execution configuration

    Response:
        answer (str): Final generated response
        execution_trace (ExecutionTrace): Trace of module execution
        reasoning (str | None): Reasoning trace if requested
        sources (list[str]): Evidence sources

    Error codes:
        -32602: Invalid params (missing query, invalid config)
        -32000: Server error (module execution failure)
        -32001: Timeout error (execution exceeded timeout)

    Example request:
        {
            "jsonrpc": "2.0",
            "method": "modular.solve",
            "params": {
                "query": "What is the capital of France?",
                "config": {
                    "max_iterations": 3,
                    "output_format": "markdown",
                    "include_reasoning": true
                }
            },
            "id": 1
        }
    """
    try:
        # Extract and validate parameters
        params = request.params or {}

        if not isinstance(params, dict):
            logger.error(
                "invalid_params_type",
                request_id=request.id,
                params_type=type(params).__name__,
            )
            raise ValueError("Parameters must be a dictionary")

        # Parse request
        try:
            solve_request = ModularSolveRequest(**params)
        except Exception as e:
            logger.error(
                "request_validation_failed",
                request_id=request.id,
                error=str(e),
            )
            raise ValueError(f"Invalid request parameters: {str(e)}") from e

        # Use default config if not provided
        config = solve_request.config or ModularSolveConfig()

        # Build A2A context with trace_id generation at entry point
        # Generate trace_id at query entry if not provided
        from agentcore.modular.tracing import _get_global_tracer

        trace_id = None
        if request.a2a_context and request.a2a_context.trace_id:
            trace_id = request.a2a_context.trace_id
        else:
            # Generate new trace_id at query entry point
            tracer = _get_global_tracer()
            if tracer:
                trace_id = tracer.generate_trace_id()
            else:
                trace_id = str(uuid4())

        a2a_context = request.a2a_context or A2AContext(
            source_agent="user",
            target_agent="modular-agent",
            trace_id=trace_id,
            timestamp=datetime.now(UTC).isoformat(),
        )

        # Ensure trace_id is set even if a2a_context was provided
        if not a2a_context.trace_id:
            a2a_context.trace_id = trace_id

        # Bind trace_id to logger for all subsequent log messages
        bound_logger = logger.bind(trace_id=a2a_context.trace_id)

        bound_logger.info(
            "modular_solve_request",
            query=solve_request.query,
            request_id=request.id,
            max_iterations=config.max_iterations,
            timeout=config.timeout_seconds,
        )

        # Execute orchestration with timeout
        import asyncio

        try:
            response = await asyncio.wait_for(
                _orchestrate_execution(
                    query=solve_request.query,
                    config=config,
                    a2a_context=a2a_context,
                ),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            bound_logger.error(
                "execution_timeout",
                request_id=request.id,
                timeout=config.timeout_seconds,
            )
            raise TimeoutError(
                f"Execution exceeded timeout of {config.timeout_seconds} seconds"
            )

        bound_logger.info(
            "modular_solve_success",
            request_id=request.id,
            duration_ms=response.execution_trace.total_duration_ms,
            verification_passed=response.execution_trace.verification_passed,
        )

        # Return response as dict
        return response.model_dump(exclude_none=True)

    except ValueError as e:
        # Invalid parameters
        # Get trace_id from request if available
        trace_id = (
            request.a2a_context.trace_id
            if request.a2a_context
            else None
        )
        error_logger = logger.bind(trace_id=trace_id) if trace_id else logger
        error_logger.error(
            "invalid_params",
            request_id=request.id,
            error=str(e),
        )
        error_response = create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.INVALID_PARAMS,
            message=str(e),
            data={"details": str(e), "trace_id": trace_id},
        )
        raise RuntimeError(error_response.model_dump_json()) from e

    except TimeoutError as e:
        # Timeout error
        # Use bound_logger if available, otherwise get trace_id from request
        trace_id = (
            a2a_context.trace_id
            if 'a2a_context' in locals() and a2a_context
            else (request.a2a_context.trace_id if request.a2a_context else None)
        )
        error_logger = logger.bind(trace_id=trace_id) if trace_id else logger
        error_logger.error(
            "timeout_error",
            request_id=request.id,
            error=str(e),
        )
        error_response = create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.TIMEOUT_ERROR,
            message=str(e),
            data={"details": str(e), "trace_id": trace_id},
        )
        raise RuntimeError(error_response.model_dump_json()) from e

    except Exception as e:
        # Server error
        # Use bound_logger if available, otherwise get trace_id from request
        trace_id = (
            a2a_context.trace_id
            if 'a2a_context' in locals() and a2a_context
            else (request.a2a_context.trace_id if request.a2a_context else None)
        )
        error_logger = logger.bind(trace_id=trace_id) if trace_id else logger
        error_logger.error(
            "server_error",
            request_id=request.id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        error_response = create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.INTERNAL_ERROR,
            message=f"Module execution failed: {str(e)}",
            data={
                "details": str(e),
                "error_type": type(e).__name__,
                "trace_id": trace_id,
            },
        )
        raise RuntimeError(error_response.model_dump_json()) from e
