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
    Orchestrate execution through PEVG modules.

    Basic orchestration flow (MOD-017 - simple implementation):
    1. Planner: Create execution plan from query
    2. Executor: Execute plan steps with tool invocation
    3. Verifier: Validate execution results
    4. Generator: Synthesize final response

    Note: Full refinement loop will be implemented in MOD-018.

    Args:
        query: User query to solve
        config: Execution configuration
        a2a_context: A2A context for tracing

    Returns:
        Modular solve response with answer and trace

    Raises:
        RuntimeError: If execution fails
    """
    start_time = time.time()
    modules_invoked: list[str] = []

    logger.info(
        "orchestration_started",
        query=query,
        trace_id=a2a_context.trace_id,
        max_iterations=config.max_iterations,
    )

    # Create module instances
    planner, executor, verifier, generator = _create_modules(a2a_context, config)

    # Step 1: Planning
    logger.info("step_1_planning", trace_id=a2a_context.trace_id)
    modules_invoked.append("planner")

    planner_query = PlannerQuery(
        query=query,
        context={},
        constraints={"max_iterations": config.max_iterations},
    )

    plan = await planner.analyze_query(planner_query)
    logger.info(
        "planning_complete",
        plan_id=plan.plan_id,
        step_count=len(plan.steps),
        trace_id=a2a_context.trace_id,
    )

    # Step 2: Execution
    logger.info("step_2_execution", trace_id=a2a_context.trace_id, plan_id=plan.plan_id)
    modules_invoked.append("executor")

    execution_results = await executor.execute_plan(plan)
    successful_steps = sum(1 for r in execution_results if r.success)
    failed_steps = sum(1 for r in execution_results if not r.success)

    logger.info(
        "execution_complete",
        plan_id=plan.plan_id,
        total_steps=len(execution_results),
        successful=successful_steps,
        failed=failed_steps,
        trace_id=a2a_context.trace_id,
    )

    # Step 3: Verification
    logger.info("step_3_verification", trace_id=a2a_context.trace_id)
    modules_invoked.append("verifier")

    verification_request = VerificationRequest(
        results=execution_results,
        expected_json_schema=None,
        consistency_rules=["no_null_results"],
    )

    verification_result = await verifier.validate_results(verification_request)
    logger.info(
        "verification_complete",
        valid=verification_result.valid,
        errors_count=len(verification_result.errors),
        warnings_count=len(verification_result.warnings),
        confidence=verification_result.confidence,
        trace_id=a2a_context.trace_id,
    )

    # Step 4: Generation
    logger.info("step_4_generation", trace_id=a2a_context.trace_id)
    modules_invoked.append("generator")

    generation_request = GenerationRequest(
        verified_results=execution_results,
        format=config.output_format,
        include_reasoning=config.include_reasoning,
        max_length=None,
    )

    generated_response = await generator.synthesize_response(generation_request)
    logger.info(
        "generation_complete",
        content_length=len(generated_response.content),
        has_reasoning=generated_response.reasoning is not None,
        sources_count=len(generated_response.sources),
        trace_id=a2a_context.trace_id,
    )

    # Build execution trace
    total_duration_ms = int((time.time() - start_time) * 1000)

    execution_trace = ExecutionTrace(
        plan_id=plan.plan_id,
        iterations=1,  # Simple implementation - no refinement yet
        modules_invoked=modules_invoked,
        total_duration_ms=total_duration_ms,
        verification_passed=verification_result.valid,
        step_count=len(execution_results),
        successful_steps=successful_steps,
        failed_steps=failed_steps,
        confidence_score=verification_result.confidence,
    )

    logger.info(
        "orchestration_complete",
        trace_id=a2a_context.trace_id,
        duration_ms=total_duration_ms,
        verification_passed=verification_result.valid,
    )

    return ModularSolveResponse(
        answer=generated_response.content,
        execution_trace=execution_trace,
        reasoning=generated_response.reasoning,
        sources=generated_response.sources,
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

        # Build A2A context
        a2a_context = request.a2a_context or A2AContext(
            source_agent="user",
            target_agent="modular-agent",
            trace_id=str(uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
        )

        logger.info(
            "modular_solve_request",
            query=solve_request.query,
            request_id=request.id,
            trace_id=a2a_context.trace_id,
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
            logger.error(
                "execution_timeout",
                request_id=request.id,
                trace_id=a2a_context.trace_id,
                timeout=config.timeout_seconds,
            )
            raise TimeoutError(
                f"Execution exceeded timeout of {config.timeout_seconds} seconds"
            )

        logger.info(
            "modular_solve_success",
            request_id=request.id,
            trace_id=a2a_context.trace_id,
            duration_ms=response.execution_trace.total_duration_ms,
            verification_passed=response.execution_trace.verification_passed,
        )

        # Return response as dict
        return response.model_dump(exclude_none=True)

    except ValueError as e:
        # Invalid parameters
        logger.error(
            "invalid_params",
            request_id=request.id,
            error=str(e),
        )
        error_response = create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.INVALID_PARAMS,
            message=str(e),
            data={"details": str(e)},
        )
        raise RuntimeError(error_response.model_dump_json()) from e

    except TimeoutError as e:
        # Timeout error
        logger.error(
            "timeout_error",
            request_id=request.id,
            error=str(e),
        )
        error_response = create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.TIMEOUT_ERROR,
            message=str(e),
            data={"details": str(e)},
        )
        raise RuntimeError(error_response.model_dump_json()) from e

    except Exception as e:
        # Server error
        logger.error(
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
            },
        )
        raise RuntimeError(error_response.model_dump_json()) from e
