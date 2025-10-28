"""LLM JSON-RPC Methods

JSON-RPC 2.0 methods for multi-provider LLM operations.
Exposes LLMService via A2A protocol with comprehensive error handling and metrics.
"""

from typing import Any

import structlog
from prometheus_client import REGISTRY
from pydantic import ValidationError

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    ModelNotAllowedError,
    ProviderError,
    ProviderTimeoutError,
)
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.llm_service import llm_service

logger = structlog.get_logger()


@register_jsonrpc_method("llm.complete")  # type: ignore[misc]
async def handle_llm_complete(request: JsonRpcRequest) -> dict[str, Any]:
    """Execute non-streaming LLM completion.

    Method: llm.complete
    Params:
        - model: string (required) - Model identifier (must be in ALLOWED_MODELS)
        - messages: array (required) - Conversation messages
        - temperature: number (optional, default 0.7) - Sampling temperature (0.0-2.0)
        - max_tokens: integer (optional) - Maximum tokens to generate
        - stream: boolean (optional, default false) - Enable streaming

    Returns:
        - content: string - Generated text content
        - usage: object - Token usage statistics
        - latency_ms: integer - Request latency in milliseconds
        - provider: string - Provider that generated the response
        - model: string - Model that was used
        - trace_id: string (optional) - A2A trace ID
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: model, messages, and optional temperature, max_tokens"
            )

        # Extract A2A context from request
        trace_id = None
        source_agent = None
        session_id = None
        if request.a2a_context:
            trace_id = request.a2a_context.trace_id
            source_agent = request.a2a_context.source_agent
            session_id = request.a2a_context.session_id

        # Add A2A context to params for LLMRequest
        params_with_context = {
            **request.params,
            "trace_id": trace_id,
            "source_agent": source_agent,
            "session_id": session_id,
        }

        # Validate and create LLM request
        llm_request = LLMRequest(**params_with_context)

        # Execute completion
        response = await llm_service.complete(llm_request)

        logger.info(
            "LLM completion via JSON-RPC",
            method="llm.complete",
            model=llm_request.model,
            provider=response.provider,
            latency_ms=response.latency_ms,
            total_tokens=response.usage.total_tokens,
            trace_id=trace_id,
        )

        return response.model_dump(mode="json")

    except ModelNotAllowedError as e:
        logger.warning(
            "Model not allowed",
            method="llm.complete",
            model=e.model,
            allowed_models=e.allowed,
            trace_id=trace_id if "trace_id" in locals() else None,
        )
        # Re-raise as ValueError for JSON-RPC INVALID_PARAMS error code
        raise ValueError(
            f"Model '{e.model}' not allowed. Allowed models: {e.allowed}"
        ) from e

    except ProviderTimeoutError as e:
        logger.error(
            "LLM request timeout",
            method="llm.complete",
            provider=e.provider,
            timeout_seconds=e.timeout_seconds,
            trace_id=trace_id if "trace_id" in locals() else None,
        )
        # Re-raise as generic exception for JSON-RPC INTERNAL_ERROR
        raise RuntimeError(
            f"Request timed out after {e.timeout_seconds}s"
        ) from e

    except ProviderError as e:
        logger.error(
            "Provider error",
            method="llm.complete",
            provider=e.provider,
            error=str(e.original_error),
            trace_id=trace_id if "trace_id" in locals() else None,
        )
        # Re-raise as generic exception for JSON-RPC INTERNAL_ERROR
        raise RuntimeError(f"Provider error: {e.provider}") from e

    except ValidationError as e:
        logger.error(
            "LLM request validation failed",
            method="llm.complete",
            error=str(e),
            trace_id=trace_id if "trace_id" in locals() else None,
        )
        raise ValueError(f"Invalid request parameters: {e}") from e

    except Exception as e:
        logger.error(
            "LLM completion handler failed",
            method="llm.complete",
            error=str(e),
            trace_id=trace_id if "trace_id" in locals() else None,
        )
        raise


@register_jsonrpc_method("llm.stream")  # type: ignore[misc]
async def handle_llm_stream(request: JsonRpcRequest) -> dict[str, Any]:
    """Execute streaming LLM completion.

    Method: llm.stream
    Params:
        - model: string (required) - Model identifier (must be in ALLOWED_MODELS)
        - messages: array (required) - Conversation messages
        - temperature: number (optional, default 0.7) - Sampling temperature (0.0-2.0)
        - max_tokens: integer (optional) - Maximum tokens to generate

    Returns:
        Note: Streaming is not directly supported via JSON-RPC 2.0.
        This method returns a message directing clients to use WebSocket/SSE for streaming.

    Alternative: Use WebSocket or SSE endpoints for true streaming support.
    """
    # JSON-RPC 2.0 does not natively support streaming responses
    # Clients should use WebSocket or SSE endpoints for streaming
    # This method returns a helpful error message

    logger.warning(
        "Streaming requested via JSON-RPC (not supported)",
        method="llm.stream",
        trace_id=request.a2a_context.trace_id if request.a2a_context else None,
    )

    return {
        "error": "Streaming not supported via JSON-RPC",
        "message": "Use WebSocket or Server-Sent Events (SSE) endpoints for streaming completions",
        "alternatives": [
            "WebSocket: /ws/llm/stream",
            "SSE: /api/v1/llm/stream (HTTP streaming)"
        ],
        "note": "For non-streaming completions, use llm.complete method"
    }


@register_jsonrpc_method("llm.models")  # type: ignore[misc]
async def handle_llm_models(request: JsonRpcRequest) -> dict[str, Any]:
    """List allowed LLM models.

    Method: llm.models
    Params: none

    Returns:
        - allowed_models: array - List of allowed model identifiers
        - default_model: string - Default model from configuration
        - count: integer - Number of allowed models
    """
    from agentcore.a2a_protocol.config import settings

    models = llm_service.registry.list_available_models()

    logger.debug(
        "Listed allowed models",
        method="llm.models",
        count=len(models),
        trace_id=request.a2a_context.trace_id if request.a2a_context else None,
    )

    return {
        "allowed_models": models,
        "default_model": settings.LLM_DEFAULT_MODEL,
        "count": len(models),
    }


@register_jsonrpc_method("llm.metrics")  # type: ignore[misc]
async def handle_llm_metrics(request: JsonRpcRequest) -> dict[str, Any]:
    """Get current LLM metrics snapshot.

    Method: llm.metrics
    Params: none

    Returns:
        - total_requests: integer - Total requests across all providers
        - total_tokens: integer - Total tokens across all providers
        - by_provider: object - Metrics breakdown by provider
        - governance_violations: integer - Total governance violations
    """
    try:
        # Collect metrics from Prometheus registry
        metrics_data: dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "by_provider": {},
            "governance_violations": 0,
        }

        # Iterate through all metric families in the registry
        for metric_family in REGISTRY.collect():
            # LLM requests total
            if metric_family.name == "llm_requests_total":
                for sample in metric_family.samples:
                    if sample.name == "llm_requests_total":
                        provider = sample.labels.get("provider", "unknown")
                        status = sample.labels.get("status", "unknown")
                        value = int(sample.value)

                        if status == "success":
                            metrics_data["total_requests"] = int(metrics_data["total_requests"]) + value

                            by_provider = metrics_data["by_provider"]
                            if provider not in by_provider:
                                by_provider[provider] = {
                                    "requests": 0,
                                    "tokens": 0,
                                }
                            by_provider[provider]["requests"] = int(by_provider[provider]["requests"]) + value

            # LLM tokens total
            elif metric_family.name == "llm_tokens_total":
                for sample in metric_family.samples:
                    if sample.name == "llm_tokens_total":
                        provider = sample.labels.get("provider", "unknown")
                        value = int(sample.value)

                        metrics_data["total_tokens"] = int(metrics_data["total_tokens"]) + value

                        by_provider = metrics_data["by_provider"]
                        if provider not in by_provider:
                            by_provider[provider] = {
                                "requests": 0,
                                "tokens": 0,
                            }
                        by_provider[provider]["tokens"] = int(by_provider[provider]["tokens"]) + value

            # Governance violations
            elif metric_family.name == "llm_governance_violations_total":
                for sample in metric_family.samples:
                    if sample.name == "llm_governance_violations_total":
                        metrics_data["governance_violations"] = int(metrics_data["governance_violations"]) + int(sample.value)

        logger.debug(
            "LLM metrics snapshot",
            method="llm.metrics",
            total_requests=metrics_data["total_requests"],
            total_tokens=metrics_data["total_tokens"],
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )

        return metrics_data

    except Exception as e:
        logger.error(
            "Failed to collect LLM metrics",
            method="llm.metrics",
            error=str(e),
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )
        raise


# Log registration on import
logger.info(
    "LLM JSON-RPC methods registered",
    methods=[
        "llm.complete",
        "llm.stream",
        "llm.models",
        "llm.metrics",
    ],
)
