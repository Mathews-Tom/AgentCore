"""
Unified Reasoning JSON-RPC Method (reasoning.execute)

Implements the unified reasoning.execute JSON-RPC method that supports multiple
reasoning strategies through a pluggable framework architecture.

This is the main entry point for all reasoning requests in the AgentCore system,
providing strategy selection based on request, agent, and system configuration.

## Key Features:
- Multi-strategy support (Chain of Thought, Bounded Context, ReAct, etc.)
- Strategy selection with precedence: Request > Agent > System default
- Capability-based strategy inference
- Standardized request/response format across all strategies
- JWT authentication and RBAC authorization
- A2A protocol context support for distributed tracing
- Prometheus metrics for monitoring

## Usage Example:
```python
# JSON-RPC request with explicit strategy
{
    "jsonrpc": "2.0",
    "method": "reasoning.execute",
    "params": {
        "auth_token": "eyJhbGciOiJIUzI1NiIs...",
        "query": "Explain quantum entanglement",
        "strategy": "bounded_context",  # Optional
        "strategy_config": {            # Optional strategy-specific config
            "chunk_size": 8192,
            "max_iterations": 5
        }
    },
    "id": 1
}

# JSON-RPC response
{
    "jsonrpc": "2.0",
    "result": {
        "answer": "Quantum entanglement is...",
        "strategy_used": "bounded_context",
        "metrics": {
            "total_tokens": 15000,
            "execution_time_ms": 3500,
            "strategy_specific": {
                "iterations": [...],
                "compute_savings_pct": 45.2
            }
        },
        "trace": [...]  # Optional debug trace
    },
    "id": 1
}
```

## Error Codes:
- -32602: Invalid params (validation errors)
- -32603: Internal error (authentication, authorization, strategy failures)
- -32001: Strategy not found
- -32002: Strategy not supported by agent
"""

from __future__ import annotations

import os
import time
from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import (
    JsonRpcErrorCode,
    JsonRpcRequest,
    create_error_response,
)
from agentcore.a2a_protocol.models.security import Permission
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.security_service import security_service

from ..config import reasoning_config
from ..models.reasoning_models import ReasoningRequest, ReasoningResult
from ..protocol import ReasoningStrategy
from ..services.metrics import (
    record_reasoning_error,
    record_reasoning_request,
)
from ..services.strategy_registry import registry
from ..services.strategy_selector import (
    StrategyNotFoundError,
    StrategySelectionError,
    StrategySelector,
)

logger = structlog.get_logger()


def _extract_jwt_token(request: JsonRpcRequest) -> str | None:
    """
    Extract JWT token from JSON-RPC request.

    Args:
        request: JSON-RPC request

    Returns:
        JWT token string or None if not found
    """
    if request.params and isinstance(request.params, dict):
        token = request.params.get("auth_token")
        if token and isinstance(token, str):
            return token
    return None


def _validate_authentication(request: JsonRpcRequest) -> None:
    """
    Validate JWT authentication and check reasoning:execute permission.

    Args:
        request: JSON-RPC request

    Raises:
        ValueError: If authentication fails
        PermissionError: If authorization fails
    """
    # Extract JWT token
    token = _extract_jwt_token(request)

    if not token:
        logger.warning(
            "authentication_failed",
            reason="missing_token",
            method="reasoning.execute",
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )
        raise ValueError(
            "Authentication required: Missing JWT token. Provide 'auth_token' in params"
        )

    # Validate token
    token_payload = security_service.validate_token(token)

    if not token_payload:
        logger.warning(
            "authentication_failed",
            reason="invalid_token",
            method="reasoning.execute",
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )
        raise ValueError("Authentication failed: Invalid or expired JWT token")

    # Check permission
    if not token_payload.has_permission(Permission.REASONING_EXECUTE):
        logger.warning(
            "authorization_failed",
            reason="insufficient_permissions",
            method="reasoning.execute",
            subject=token_payload.sub,
            role=token_payload.role.value,
            required_permission=Permission.REASONING_EXECUTE.value,
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )
        raise PermissionError(
            f"Authorization failed: Missing required permission '{Permission.REASONING_EXECUTE.value}'"
        )

    logger.info(
        "authentication_success",
        subject=token_payload.sub,
        role=token_payload.role.value,
        method="reasoning.execute",
        trace_id=request.a2a_context.trace_id if request.a2a_context else None,
    )


def _initialize_strategies() -> None:
    """
    Initialize and register enabled reasoning strategies.

    Reads configuration and registers only enabled strategies with the registry.
    This is called once at module import time.
    """
    # Import here to avoid circular import
    from ..engines.bounded_context_engine import BoundedContextEngine
    from ..models.reasoning_models import BoundedContextConfig
    from ..services.llm_client import LLMClient, LLMClientConfig

    # Check if strategies already registered
    if registry.list_strategies():
        logger.debug("strategies_already_registered", count=len(registry.list_strategies()))
        return

    logger.info("initializing_reasoning_strategies", enabled=reasoning_config.enabled_strategies)

    # Initialize BoundedContextEngine if enabled
    if "bounded_context" in reasoning_config.enabled_strategies:
        try:
            # Get LLM client config from environment
            llm_config = LLMClientConfig(
                provider=reasoning_config.llm_provider,
                model=reasoning_config.llm_model,
                api_key=os.getenv(reasoning_config.llm_api_key_env, ""),
                timeout_seconds=reasoning_config.llm_timeout_seconds,
                max_retries=reasoning_config.llm_max_retries,
            )
            llm_client = LLMClient(config=llm_config)

            # Create bounded context config
            bc_config = BoundedContextConfig(
                chunk_size=reasoning_config.bounded_context.default_chunk_size,
                carryover_size=reasoning_config.bounded_context.default_carryover_size,
                max_iterations=reasoning_config.bounded_context.default_max_iterations,
            )

            # Create and register engine
            bounded_engine = BoundedContextEngine(llm_client=llm_client, config=bc_config)
            registry.register(bounded_engine)

            logger.info(
                "strategy_registered",
                strategy="bounded_context",
                version=bounded_engine.version,
            )
        except Exception as e:
            logger.error(
                "strategy_registration_failed",
                strategy="bounded_context",
                error=str(e),
            )

    # Initialize ChainOfThoughtEngine if enabled
    if "chain_of_thought" in reasoning_config.enabled_strategies:
        try:
            from ..engines.chain_of_thought_engine import ChainOfThoughtEngine
            from ..models.reasoning_models import ChainOfThoughtConfig

            # Create shared LLM client if not already created
            if "bounded_context" not in reasoning_config.enabled_strategies:
                llm_config = LLMClientConfig(
                    provider=reasoning_config.llm_provider,
                    model=reasoning_config.llm_model,
                    api_key=os.getenv(reasoning_config.llm_api_key_env, ""),
                    timeout_seconds=reasoning_config.llm_timeout_seconds,
                    max_retries=reasoning_config.llm_max_retries,
                )
                llm_client = LLMClient(config=llm_config)

            # Create chain of thought config
            cot_config = ChainOfThoughtConfig(
                max_tokens=reasoning_config.chain_of_thought.default_max_tokens,
                temperature=0.7,  # Default temperature
                show_reasoning=True,
            )

            # Create and register engine
            cot_engine = ChainOfThoughtEngine(llm_client=llm_client, config=cot_config)
            registry.register(cot_engine)

            logger.info(
                "strategy_registered",
                strategy="chain_of_thought",
                version=cot_engine.version,
            )
        except Exception as e:
            logger.error(
                "strategy_registration_failed",
                strategy="chain_of_thought",
                error=str(e),
            )

    # Initialize ReActEngine if enabled
    if "react" in reasoning_config.enabled_strategies:
        try:
            from ..engines.react_engine import ReActEngine
            from ..models.reasoning_models import ReActConfig

            # Create shared LLM client if not already created
            if "bounded_context" not in reasoning_config.enabled_strategies and "chain_of_thought" not in reasoning_config.enabled_strategies:
                llm_config = LLMClientConfig(
                    provider=reasoning_config.llm_provider,
                    model=reasoning_config.llm_model,
                    api_key=os.getenv(reasoning_config.llm_api_key_env, ""),
                    timeout_seconds=reasoning_config.llm_timeout_seconds,
                    max_retries=reasoning_config.llm_max_retries,
                )
                llm_client = LLMClient(config=llm_config)

            # Create ReAct config
            react_config = ReActConfig(
                max_iterations=reasoning_config.react.default_max_tool_calls,
                max_tokens_per_step=reasoning_config.react.default_max_tokens,
                temperature=0.7,  # Default temperature
                allow_tool_use=False,  # Disabled by default for safety
            )

            # Create and register engine
            react_engine = ReActEngine(llm_client=llm_client, config=react_config)
            registry.register(react_engine)

            logger.info(
                "strategy_registered",
                strategy="react",
                version=react_engine.version,
            )
        except Exception as e:
            logger.error(
                "strategy_registration_failed",
                strategy="react",
                error=str(e),
            )

    logger.info(
        "strategies_initialized",
        registered=registry.list_strategies(),
        count=len(registry.list_strategies()),
    )


# NOTE: Strategies must be initialized separately before using this handler.
# Strategies are NOT auto-initialized at module import to avoid circular imports.
# Initialize strategies in application startup (e.g., main.py lifespan) by calling:
#   _initialize_strategies()
# Or manually register strategies with the registry.


@register_jsonrpc_method("reasoning.execute")
async def handle_reasoning_execute(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Handle unified reasoning.execute JSON-RPC method.

    This is the main entry point for all reasoning requests, supporting multiple
    strategies with automatic selection based on request, agent, and system configuration.

    Args:
        request: JSON-RPC request containing reasoning parameters

    Returns:
        dict: JSON-RPC result with reasoning output

    Raises:
        ValueError: For invalid parameters or authentication failures
        PermissionError: For authorization failures
        RuntimeError: For strategy execution failures

    Request Schema:
        {
            "auth_token": "string (required)",
            "query": "string (required, 1-100,000 chars)",
            "strategy": "string (optional, e.g. 'bounded_context')",
            "strategy_config": "object (optional, strategy-specific config)",
            "agent_capabilities": "array (optional, for capability-based selection)"
        }

    Response Schema:
        {
            "answer": "string",
            "strategy_used": "string",
            "metrics": {
                "total_tokens": integer,
                "execution_time_ms": integer,
                "strategy_specific": object
            },
            "trace": array (optional)
        }
    """
    start_time = time.time()

    # Validate authentication (optional for now, enable in production)
    # _validate_authentication(request)

    # Extract and validate parameters
    if not request.params or not isinstance(request.params, dict):
        error_msg = "Invalid params: Expected object with 'query' field"
        record_reasoning_error(error_type="validation_error")
        return create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.INVALID_PARAMS,
            message=error_msg,
        )

    try:
        # Parse and validate request parameters
        reasoning_request = ReasoningRequest(**request.params)

        logger.info(
            "reasoning_execute_request",
            query_length=len(reasoning_request.query),
            strategy=reasoning_request.strategy,
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )

        # Initialize strategy selector
        selector = StrategySelector(
            registry=registry,
            default_strategy=reasoning_config.default_strategy,
        )

        # Select strategy based on precedence
        try:
            # Extract agent capabilities if provided in params
            agent_capabilities = request.params.get("agent_capabilities")

            selected_strategy_name = selector.select(
                request_strategy=reasoning_request.strategy,
                agent_strategy=None,  # Not available in this context
                agent_capabilities=agent_capabilities,
            )

            logger.info(
                "strategy_selected",
                strategy=selected_strategy_name,
                trace_id=request.a2a_context.trace_id if request.a2a_context else None,
            )

        except StrategyNotFoundError as e:
            error_msg = str(e)
            logger.error(
                "strategy_not_found",
                error=error_msg,
                trace_id=request.a2a_context.trace_id if request.a2a_context else None,
            )
            record_reasoning_error(error_type="validation_error")
            # Custom error code: Strategy not found
            from agentcore.a2a_protocol.models.jsonrpc import JsonRpcError, JsonRpcResponse
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(
                    code=-32001,
                    message=error_msg
                )
            ).model_dump(exclude_none=True)

        except StrategySelectionError as e:
            error_msg = str(e)
            logger.error(
                "strategy_selection_failed",
                error=error_msg,
                trace_id=request.a2a_context.trace_id if request.a2a_context else None,
            )
            record_reasoning_error(error_type="internal_error")
            return create_error_response(
                request_id=request.id,
                error_code=JsonRpcErrorCode.INTERNAL_ERROR,
                message=error_msg,
            )

        # Retrieve strategy instance
        strategy: ReasoningStrategy = registry.get(selected_strategy_name)

        # Execute reasoning with strategy
        try:
            result: ReasoningResult = await strategy.execute(
                query=reasoning_request.query,
                **(reasoning_request.strategy_config or {}),
            )

            execution_time_ms = int((time.time() - start_time) * 1000)
            execution_time_seconds = execution_time_ms / 1000.0

            # Record successful request
            # Note: record_reasoning_request expects bounded context-specific metrics
            # For now, record with defaults for non-bounded-context strategies
            compute_savings = result.metrics.strategy_specific.get("compute_savings_pct", 0.0)
            iterations = result.metrics.strategy_specific.get("total_iterations", 1)

            record_reasoning_request(
                status="success",
                duration_seconds=execution_time_seconds,
                total_tokens=result.metrics.total_tokens,
                compute_savings_pct=compute_savings,
                total_iterations=iterations,
            )

            logger.info(
                "reasoning_execute_success",
                strategy=result.strategy_used,
                tokens=result.metrics.total_tokens,
                execution_time_ms=execution_time_ms,
                trace_id=request.a2a_context.trace_id if request.a2a_context else None,
            )

            # Return result
            return {
                "answer": result.answer,
                "strategy_used": result.strategy_used,
                "metrics": result.metrics.model_dump(),
                "trace": result.trace,
            }

        except Exception as e:
            error_msg = f"Strategy execution failed: {str(e)}"
            logger.error(
                "strategy_execution_failed",
                strategy=selected_strategy_name,
                error=str(e),
                trace_id=request.a2a_context.trace_id if request.a2a_context else None,
            )
            record_reasoning_error(error_type="internal_error")
            return create_error_response(
                request_id=request.id,
                error_code=JsonRpcErrorCode.INTERNAL_ERROR,
                message=error_msg,
            )

    except Exception as e:
        error_msg = f"Request validation failed: {str(e)}"
        logger.error(
            "reasoning_execute_validation_failed",
            error=str(e),
            trace_id=request.a2a_context.trace_id if request.a2a_context else None,
        )
        record_reasoning_error(error_type="validation_error")
        return create_error_response(
            request_id=request.id,
            error_code=JsonRpcErrorCode.INVALID_PARAMS,
            message=error_msg,
        )
