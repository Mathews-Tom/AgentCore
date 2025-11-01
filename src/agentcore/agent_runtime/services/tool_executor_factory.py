"""Factory for creating configured ToolExecutor instances."""

import structlog

from agentcore.agent_runtime.config.settings import get_settings
from agentcore.agent_runtime.services.rate_limiter import RateLimiter
from agentcore.agent_runtime.services.retry_handler import BackoffStrategy, RetryHandler
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_registry import ToolRegistry, get_tool_registry

logger = structlog.get_logger()


def create_tool_executor(
    registry: ToolRegistry | None = None,
    settings_override: dict | None = None,
) -> ToolExecutor:
    """
    Create a configured ToolExecutor instance.

    Args:
        registry: Tool registry (uses global if not provided)
        settings_override: Optional settings to override configuration

    Returns:
        Configured ToolExecutor instance
    """
    settings = get_settings()

    # Apply overrides if provided
    if settings_override:
        for key, value in settings_override.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

    # Get or create registry
    if registry is None:
        registry = get_tool_registry()

    # Initialize rate limiter if enabled
    rate_limiter = None
    if settings.rate_limiter_enabled:
        try:
            rate_limiter = RateLimiter(
                redis_url=settings.rate_limiter_redis_url,
                key_prefix=settings.rate_limiter_key_prefix,
            )
            logger.info(
                "rate_limiter_initialized",
                redis_url=settings.rate_limiter_redis_url,
            )
        except Exception as e:
            logger.warning(
                "rate_limiter_initialization_failed",
                error=str(e),
                fallback_to_no_rate_limiting=True,
            )
            rate_limiter = None

    # Initialize retry handler with configured strategy
    strategy_map = {
        "exponential": BackoffStrategy.EXPONENTIAL,
        "linear": BackoffStrategy.LINEAR,
        "fixed": BackoffStrategy.FIXED,
    }

    retry_handler = RetryHandler(
        max_retries=settings.tool_max_retries,
        base_delay=settings.tool_retry_base_delay,
        max_delay=settings.tool_retry_max_delay,
        strategy=strategy_map.get(
            settings.tool_retry_strategy, BackoffStrategy.EXPONENTIAL
        ),
        jitter=settings.tool_retry_jitter,
    )

    # Create tool executor
    executor = ToolExecutor(
        registry=registry,
        enable_metrics=settings.enable_metrics,
        rate_limiter=rate_limiter,
        retry_handler=retry_handler,
    )

    logger.info(
        "tool_executor_created",
        rate_limiter_enabled=rate_limiter is not None,
        retry_strategy=settings.tool_retry_strategy,
        max_retries=settings.tool_max_retries,
    )

    return executor
