"""Portkey AI Gateway client wrapper.

Provides a high-level async client interface for interacting with Portkey
Gateway and managing LLM requests across multiple providers.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from portkey_ai import AsyncPortkey

from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.exceptions import (
    PortkeyAuthenticationError,
    PortkeyError,
    PortkeyProviderError,
    PortkeyRateLimitError,
    PortkeyTimeoutError,
    PortkeyValidationError,
)
from agentcore.integration.portkey.metrics_collector import get_metrics_collector
from agentcore.integration.portkey.models import LLMRequest, LLMResponse

logger = structlog.get_logger(__name__)


class PortkeyClient:
    """Async client for Portkey AI Gateway integration.

    Wraps the Portkey Python SDK to provide:
    - Configuration management
    - Error handling and mapping
    - Request/response logging
    - Cost tracking (placeholder)
    - Performance monitoring
    """

    def __init__(
        self,
        config: PortkeyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Portkey client.

        Args:
            config: Optional PortkeyConfig instance. If not provided,
                    configuration will be loaded from environment variables.
            **kwargs: Additional keyword arguments to override configuration

        Raises:
            PortkeyConfigurationError: If required configuration is missing
        """
        # Load or use provided configuration
        self.config = config or PortkeyConfig.from_env()

        # Merge any override parameters
        client_config = self.config.merge_with_defaults(kwargs)

        # Initialize the async Portkey client
        self._client = AsyncPortkey(
            api_key=client_config["api_key"],
            base_url=client_config.get("base_url"),
            timeout=client_config.get("timeout"),
            max_retries=client_config.get("max_retries"),
            virtual_key=self.config.virtual_key,
        )

        # Track client state
        self._closed = False

        # Initialize metrics collector
        self._metrics_collector = get_metrics_collector()

        logger.info(
            "portkey_client_initialized",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            caching_enabled=self.config.enable_caching,
            metrics_enabled=self.config.enable_metrics,
        )

    async def complete(
        self,
        request: LLMRequest,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute an LLM completion request through Portkey.

        Args:
            request: LLM request with messages and model requirements
            **kwargs: Additional parameters to override request settings

        Returns:
            LLM response with completion and metadata

        Raises:
            PortkeyAuthenticationError: Authentication failed
            PortkeyProviderError: Provider error occurred
            PortkeyRateLimitError: Rate limit exceeded
            PortkeyTimeoutError: Request timed out
            PortkeyValidationError: Request validation failed
            PortkeyError: Other errors
        """
        if self._closed:
            raise PortkeyError("Client has been closed")

        start_time = time.time()

        # Log request
        if self.config.enable_logging:
            logger.info(
                "portkey_request_start",
                model=request.model,
                stream=request.stream,
                context=request.context,
            )

        # Generate request ID
        request_id = str(uuid.uuid4())

        try:
            # Prepare request parameters
            params = self._prepare_request_params(request, kwargs)

            # Execute completion through Portkey
            response = await self._client.chat.completions.create(**params)

            # Calculate request latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Convert to our response model
            llm_response = self._convert_response(
                response=response,
                latency_ms=latency_ms,
            )

            # Collect metrics if enabled
            if self.config.enable_metrics:
                await self._collect_success_metrics(
                    request_id=request_id,
                    request=request,
                    response=llm_response,
                    start_time=start_time,
                    params=params,
                )

            # Log success
            if self.config.enable_logging:
                logger.info(
                    "portkey_request_success",
                    request_id=request_id,
                    model=llm_response.model,
                    latency_ms=latency_ms,
                    tokens=llm_response.usage,
                    cost=llm_response.cost,
                )

            return llm_response

        except Exception as exc:
            # Calculate error latency
            error_latency_ms = int((time.time() - start_time) * 1000)

            # Collect error metrics if enabled
            if self.config.enable_metrics:
                await self._collect_error_metrics(
                    request_id=request_id,
                    request=request,
                    error=exc,
                    start_time=start_time,
                )

            # Log error
            logger.error(
                "portkey_request_failed",
                request_id=request_id,
                model=request.model,
                latency_ms=error_latency_ms,
                error=str(exc),
                error_type=type(exc).__name__,
            )

            # Map exception to our error types
            raise self._map_exception(exc) from exc

    async def stream_complete(
        self,
        request: LLMRequest,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a streaming LLM completion request.

        Args:
            request: LLM request with messages and model requirements
            **kwargs: Additional parameters to override request settings

        Yields:
            Streaming response chunks

        Raises:
            PortkeyError: If request fails or client is closed
        """
        if self._closed:
            raise PortkeyError("Client has been closed")

        # Force streaming mode
        request.stream = True

        # Log request
        if self.config.enable_logging:
            logger.info(
                "portkey_stream_start",
                model=request.model,
                context=request.context,
            )

        try:
            # Prepare request parameters
            params = self._prepare_request_params(request, kwargs)

            # Execute streaming completion
            stream = await self._client.chat.completions.create(**params)

            # Yield chunks as they arrive
            async for chunk in stream:
                yield chunk.model_dump() if hasattr(chunk, "model_dump") else chunk

            logger.info("portkey_stream_complete", model=request.model)

        except Exception as exc:
            logger.error(
                "portkey_stream_failed",
                model=request.model,
                error=str(exc),
            )
            raise self._map_exception(exc) from exc

    def _prepare_request_params(
        self,
        request: LLMRequest,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare parameters for Portkey API request.

        Args:
            request: LLM request model
            overrides: Additional parameters to override

        Returns:
            Dictionary of parameters for Portkey API
        """
        params: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": request.stream,
        }

        # Add optional parameters if provided
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            params["temperature"] = request.temperature

        # Add trace ID from context if available
        if self.config.enable_tracing and "trace_id" in request.context:
            params["metadata"] = params.get("metadata", {})
            params["metadata"]["trace_id"] = request.context["trace_id"]

        # Apply overrides
        params.update(overrides)

        return params

    def _convert_response(
        self,
        response: Any,
        latency_ms: int,
    ) -> LLMResponse:
        """Convert Portkey API response to our response model.

        Args:
            response: Raw response from Portkey
            latency_ms: Request latency in milliseconds

        Returns:
            Converted LLM response model
        """
        # Extract basic response data
        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else response
        )

        # Calculate cost (placeholder - will be implemented in INT-003)
        cost = self._calculate_cost(response_dict)

        return LLMResponse(
            id=response_dict.get("id", ""),
            model=response_dict.get("model", ""),
            provider=None,  # Portkey doesn't expose provider in response
            choices=response_dict.get("choices", []),
            usage=response_dict.get("usage"),
            cost=cost,
            latency_ms=latency_ms,
            metadata={},
        )

    def _calculate_cost(self, response: dict[str, Any]) -> float | None:
        """Calculate request cost based on token usage.

        This is a placeholder implementation. Actual cost calculation
        will be implemented in INT-003 (Cost Optimization System).

        Args:
            response: Response dictionary with usage data

        Returns:
            Estimated cost in USD, or None if calculation fails
        """
        # Placeholder: return None for now
        # Will be implemented in INT-003 with provider pricing data
        return None

    def _map_exception(self, exc: Exception) -> PortkeyError:
        """Map exceptions to our custom exception types.

        Args:
            exc: Original exception

        Returns:
            Mapped custom exception
        """
        exc_str = str(exc).lower()

        # Authentication errors
        if "auth" in exc_str or "unauthorized" in exc_str or "api key" in exc_str:
            return PortkeyAuthenticationError(str(exc))

        # Rate limit errors
        if "rate limit" in exc_str or "429" in exc_str:
            return PortkeyRateLimitError(str(exc))

        # Timeout errors
        if "timeout" in exc_str or "timed out" in exc_str:
            return PortkeyTimeoutError(str(exc), timeout=self.config.timeout)

        # Provider errors
        if "provider" in exc_str or "model" in exc_str:
            return PortkeyProviderError(str(exc))

        # Validation errors
        if "validation" in exc_str or "invalid" in exc_str:
            return PortkeyValidationError(str(exc))

        # Generic error
        return PortkeyError(str(exc))

    async def close(self) -> None:
        """Close the client and release resources.

        Should be called when the client is no longer needed to properly
        clean up connections and resources.
        """
        if not self._closed:
            # Portkey client doesn't require explicit cleanup
            # but we mark as closed to prevent further use
            self._closed = True
            logger.info("portkey_client_closed")

    async def __aenter__(self) -> PortkeyClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()

    async def _collect_success_metrics(
        self,
        request_id: str,
        request: LLMRequest,
        response: LLMResponse,
        start_time: float,
        params: dict[str, Any],
    ) -> None:
        """Collect metrics for successful request.

        Args:
            request_id: Request identifier
            request: Original request
            response: Response received
            start_time: Request start time
            params: Request parameters
        """
        # Build request data
        request_data = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

        # Build response data
        response_data = {
            "id": response.id,
            "model": response.model,
            "choices": response.choices,
            "usage": response.usage,
        }

        # Build context
        context = {
            "request_id": request_id,
            "trace_id": request.context.get("trace_id"),
            "tenant_id": request.context.get("tenant_id"),
            "workflow_id": request.context.get("workflow_id"),
            "agent_id": request.context.get("agent_id"),
            "session_id": request.context.get("session_id"),
            "cost_data": {
                "total_cost": response.cost or 0.0,
                "input_cost": 0.0,  # Would be calculated by cost tracker
                "output_cost": 0.0,
            },
            "cache_hit": False,  # Would be set by cache service
        }

        # Collect metrics
        await self._metrics_collector.collect_from_response(
            request=request_data,
            response=response_data,
            start_time=start_time,
            provider_id=response.provider or "unknown",
            provider_name=response.provider or "Unknown",
            context=context,
        )

    async def _collect_error_metrics(
        self,
        request_id: str,
        request: LLMRequest,
        error: Exception,
        start_time: float,
    ) -> None:
        """Collect metrics for failed request.

        Args:
            request_id: Request identifier
            request: Original request
            error: Exception that occurred
            start_time: Request start time
        """
        # Build request data
        request_data = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

        # Build context
        context = {
            "request_id": request_id,
            "trace_id": request.context.get("trace_id"),
            "tenant_id": request.context.get("tenant_id"),
            "workflow_id": request.context.get("workflow_id"),
            "agent_id": request.context.get("agent_id"),
            "session_id": request.context.get("session_id"),
        }

        # Collect metrics
        await self._metrics_collector.collect_from_error(
            request=request_data,
            error=error,
            start_time=start_time,
            provider_id="unknown",
            provider_name="Unknown",
            context=context,
        )
