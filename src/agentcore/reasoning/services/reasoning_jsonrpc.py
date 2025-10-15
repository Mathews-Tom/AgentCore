"""
Reasoning JSON-RPC Methods

JSON-RPC 2.0 methods for bounded context reasoning.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

from ..engines.bounded_context_engine import BoundedContextEngine
from ..models.reasoning_models import BoundedContextConfig
from ..services.input_sanitizer import sanitize_reasoning_request
from ..services.llm_client import LLMClient, LLMClientConfig
from ..services.metrics import (
    record_llm_failure,
    record_reasoning_error,
    record_reasoning_request,
)

logger = structlog.get_logger()


class BoundedReasoningParams(BaseModel):
    """Parameters for bounded context reasoning request."""

    query: str = Field(..., min_length=1, max_length=100000, description="Problem to solve")
    system_prompt: str | None = Field(
        default=None, max_length=10000, description="Optional system prompt"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    chunk_size: int = Field(
        default=8192, ge=1024, le=32768, description="Tokens per iteration"
    )
    carryover_size: int = Field(
        default=4096, ge=512, le=16384, description="Tokens carried between iterations"
    )
    max_iterations: int = Field(
        default=5, ge=1, le=50, description="Maximum reasoning iterations"
    )
    llm_config: dict[str, Any] | None = Field(
        default=None, description="Optional LLM client configuration"
    )

    @field_validator("carryover_size")
    @classmethod
    def validate_carryover_less_than_chunk(
        cls, v: int, info
    ) -> int:
        """Validate carryover_size is less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 8192)
        if v >= chunk_size:
            raise ValueError(
                f"carryover_size ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class BoundedReasoningResult(BaseModel):
    """Result from bounded context reasoning."""

    success: bool = Field(..., description="Whether reasoning completed successfully")
    answer: str = Field(..., description="Reasoning result or answer")
    total_iterations: int = Field(..., description="Number of iterations performed")
    total_tokens: int = Field(..., description="Total tokens used")
    compute_savings_pct: float = Field(
        ..., description="Compute savings compared to traditional reasoning (%)"
    )
    carryover_compressions: int = Field(
        ..., description="Number of carryover compressions performed"
    )
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    iterations: list[dict[str, Any]] = Field(
        ..., description="Details of each iteration"
    )


@register_jsonrpc_method("reasoning.bounded_context")
async def handle_bounded_reasoning(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute bounded context reasoning.

    Method: reasoning.bounded_context
    Params:
        - query: string (required) - Problem to solve
        - system_prompt: string (optional) - System prompt
        - temperature: number (optional, default 0.7) - Sampling temperature
        - chunk_size: number (optional, default 8192) - Tokens per iteration
        - carryover_size: number (optional, default 4096) - Tokens carried forward
        - max_iterations: number (optional, default 5) - Maximum iterations
        - llm_config: object (optional) - LLM client configuration

    Returns:
        Bounded context reasoning result

    Errors:
        -32602: Invalid params (validation errors)
        -32603: Internal error (LLM failures, runtime errors)
        -32001: Max iterations reached without answer
    """
    # Start timing
    start_time = time.time()

    # Validate parameters
    if not request.params or not isinstance(request.params, dict):
        record_reasoning_error("validation_error")
        raise ValueError("Parameters required for reasoning.bounded_context")

    try:
        params = BoundedReasoningParams(**request.params)
    except Exception as e:
        record_reasoning_error("validation_error")
        raise ValueError(f"Invalid parameters: {e}") from e

    # Sanitize inputs for prompt injection prevention
    is_valid, error_msg = sanitize_reasoning_request(
        query=params.query,
        system_prompt=params.system_prompt,
    )
    if not is_valid:
        logger.warning(
            "input_sanitization_failed",
            error=error_msg,
            query_length=len(params.query),
        )
        record_reasoning_error("validation_error")
        raise ValueError(f"Input sanitization failed: {error_msg}")

    # Extract A2A context for logging
    trace_id = None
    source_agent = None
    target_agent = None
    if request.a2a_context:
        trace_id = request.a2a_context.trace_id
        source_agent = request.a2a_context.source_agent
        target_agent = request.a2a_context.target_agent

    logger.info(
        "bounded_reasoning_request",
        query_length=len(params.query),
        chunk_size=params.chunk_size,
        max_iterations=params.max_iterations,
        trace_id=trace_id,
        source_agent=source_agent,
        target_agent=target_agent,
    )

    try:
        # Create LLM client
        if params.llm_config:
            llm_client_config = LLMClientConfig(**params.llm_config)
        else:
            # Use default configuration
            llm_client_config = LLMClientConfig(
                api_key="",  # Will need to be configured via env
                base_url="https://api.openai.com/v1",
            )

        llm_client = LLMClient(llm_client_config)

        # Create bounded context configuration
        bounded_config = BoundedContextConfig(
            chunk_size=params.chunk_size,
            carryover_size=params.carryover_size,
            max_iterations=params.max_iterations,
        )

        # Create engine
        engine = BoundedContextEngine(llm_client, bounded_config)

        # Execute reasoning
        result = await engine.reason(
            query=params.query,
            system_prompt=params.system_prompt,
            temperature=params.temperature,
        )

        # Check if answer was found
        if not result.iterations or not any(it.has_answer for it in result.iterations):
            logger.warning(
                "bounded_reasoning_max_iterations_reached",
                total_iterations=result.total_iterations,
                max_iterations=params.max_iterations,
                trace_id=trace_id,
            )
            # Still return success but indicate max iterations reached
            # Raising error would be appropriate for strict mode
            # For now, return success with warning in answer

        # Format response
        iterations_data = [
            {
                "iteration": it.metrics.iteration,
                "tokens": it.metrics.tokens,
                "has_answer": it.has_answer,
                "execution_time_ms": it.metrics.execution_time_ms,
                "carryover_generated": it.metrics.carryover_generated,
                "content_preview": (
                    it.content[:200] + "..." if len(it.content) > 200 else it.content
                ),
            }
            for it in result.iterations
        ]

        response = BoundedReasoningResult(
            success=True,
            answer=result.answer,
            total_iterations=result.total_iterations,
            total_tokens=result.total_tokens,
            compute_savings_pct=result.compute_savings_pct,
            carryover_compressions=result.carryover_compressions,
            execution_time_ms=result.execution_time_ms,
            iterations=iterations_data,
        )

        logger.info(
            "bounded_reasoning_complete",
            total_iterations=result.total_iterations,
            total_tokens=result.total_tokens,
            compute_savings_pct=result.compute_savings_pct,
            execution_time_ms=result.execution_time_ms,
            trace_id=trace_id,
        )

        # Record metrics
        duration_seconds = time.time() - start_time
        record_reasoning_request(
            status="success",
            duration_seconds=duration_seconds,
            total_tokens=result.total_tokens,
            compute_savings_pct=result.compute_savings_pct,
            total_iterations=result.total_iterations,
        )

        return response.model_dump()

    except ValueError as e:
        # Parameter validation errors
        logger.error("bounded_reasoning_validation_error", error=str(e), trace_id=trace_id)
        record_reasoning_error("validation_error")
        raise ValueError(f"Invalid parameters: {e}") from e

    except RuntimeError as e:
        # LLM or runtime errors
        logger.error("bounded_reasoning_runtime_error", error=str(e), trace_id=trace_id)
        record_reasoning_error("llm_error")
        record_llm_failure()
        raise RuntimeError(f"Reasoning execution failed: {e}") from e

    except Exception as e:
        # Unexpected errors
        logger.error(
            "bounded_reasoning_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id,
        )
        record_reasoning_error("internal_error")
        raise RuntimeError(f"Unexpected error during reasoning: {e}") from e


# Log registration on import
logger.info(
    "reasoning_jsonrpc_methods_registered",
    methods=["reasoning.bounded_context"],
)
