"""Metrics collection for LLM request performance tracking.

Collects 50+ metrics per request including latency, tokens, cost,
quality, and resource usage for comprehensive performance monitoring.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from agentcore.llm_gateway.cost_tracker import get_cost_tracker
from agentcore.llm_gateway.metrics_models import RequestMetrics

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Collects comprehensive metrics for LLM requests.

    Tracks 50+ metrics per request including:
    - Latency breakdown (routing, queue, provider, network)
    - Token usage (input, output, cached)
    - Cost metrics (total, input, output, savings)
    - Quality metrics (success, completion, errors)
    - Resource usage (memory, CPU, network)
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._metrics_history: list[RequestMetrics] = []
        self._cost_tracker = get_cost_tracker()

        logger.info("metrics_collector_initialized")

    async def collect_request_metrics(
        self,
        request_id: str,
        provider_id: str,
        provider_name: str,
        model: str,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None,
        timing_data: dict[str, int],
        error_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> RequestMetrics:
        """Collect comprehensive metrics for a request.

        Args:
            request_id: Unique request identifier
            provider_id: Provider identifier
            provider_name: Provider display name
            model: Model used for request
            request_data: Original request data
            response_data: Response data (None if error)
            timing_data: Timing breakdown (ms)
            error_data: Error information if request failed
            context: Additional context (tenant_id, workflow_id, etc.)

        Returns:
            Collected request metrics
        """
        context = context or {}
        timestamp = datetime.now(UTC)

        # Extract timing metrics
        total_latency = timing_data.get("total", 0)
        routing_latency = timing_data.get("routing", 0)
        queue_latency = timing_data.get("queue", 0)
        provider_latency = timing_data.get("provider", 0)
        network_latency = timing_data.get("network", 0)
        ttft = timing_data.get("ttft")

        # Extract token metrics
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cached_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        if response_data and "usage" in response_data:
            usage = response_data["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            cached_tokens = usage.get("cached_tokens", 0)
            prompt_tokens = input_tokens
            completion_tokens = output_tokens

        # Calculate cost metrics
        total_cost = 0.0
        input_cost = 0.0
        output_cost = 0.0
        cache_cost_saved = 0.0

        if response_data:
            # Get cost from cost tracker if available
            cost_data = context.get("cost_data")
            if cost_data:
                total_cost = cost_data.get("total_cost", 0.0)
                input_cost = cost_data.get("input_cost", 0.0)
                output_cost = cost_data.get("output_cost", 0.0)
                cache_cost_saved = cost_data.get("cache_cost_saved", 0.0)

        # Calculate throughput
        tokens_per_second = 0.0
        if total_latency > 0 and total_tokens > 0:
            tokens_per_second = (total_tokens / total_latency) * 1000

        # Extract quality metrics
        success = error_data is None
        response_complete = True
        finish_reason = None
        quality_score = None

        if response_data:
            choices = response_data.get("choices", [])
            if choices:
                finish_reason = choices[0].get("finish_reason")
                response_complete = finish_reason in ("stop", "end_turn", None)

        # Extract error metrics
        error_occurred = error_data is not None
        error_type = None
        error_message = None
        retry_count = 0
        fallback_used = False

        if error_data:
            error_type = error_data.get("type")
            error_message = error_data.get("message")
            retry_count = error_data.get("retry_count", 0)
            fallback_used = error_data.get("fallback_used", False)

        # Extract cache metrics
        cache_hit = context.get("cache_hit", False)
        cache_level = context.get("cache_level")
        cache_lookup_ms = context.get("cache_lookup_ms")

        # Extract resource usage (if available)
        resource_data = context.get("resource_usage", {})
        memory_used_mb = resource_data.get("memory_mb")
        cpu_percent = resource_data.get("cpu_percent")
        network_bytes_sent = resource_data.get("bytes_sent")
        network_bytes_received = resource_data.get("bytes_received")

        # Extract request configuration
        temperature = request_data.get("temperature")
        max_tokens = request_data.get("max_tokens")
        stream = request_data.get("stream", False)

        # Create metrics object
        metrics = RequestMetrics(
            request_id=request_id,
            trace_id=context.get("trace_id"),
            timestamp=timestamp,
            provider_id=provider_id,
            provider_name=provider_name,
            model=model,
            model_version=context.get("model_version"),
            total_latency_ms=total_latency,
            routing_latency_ms=routing_latency,
            queue_latency_ms=queue_latency,
            provider_latency_ms=provider_latency,
            network_latency_ms=network_latency,
            ttft_ms=ttft,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=total_cost,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_cost_saved=cache_cost_saved,
            tokens_per_second=tokens_per_second,
            success=success,
            response_complete=response_complete,
            finish_reason=finish_reason,
            quality_score=quality_score,
            error_occurred=error_occurred,
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count,
            fallback_used=fallback_used,
            memory_used_mb=memory_used_mb,
            cpu_percent=cpu_percent,
            network_bytes_sent=network_bytes_sent,
            network_bytes_received=network_bytes_received,
            cache_hit=cache_hit,
            cache_level=cache_level,
            cache_lookup_ms=cache_lookup_ms,
            tenant_id=context.get("tenant_id"),
            workflow_id=context.get("workflow_id"),
            agent_id=context.get("agent_id"),
            session_id=context.get("session_id"),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tags=context.get("tags", {}),
            metadata=context.get("metadata", {}),
        )

        # Store metrics
        self._metrics_history.append(metrics)

        # Log metrics
        logger.info(
            "metrics_collected",
            request_id=request_id,
            provider_id=provider_id,
            model=model,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            total_cost=total_cost,
            success=success,
            cache_hit=cache_hit,
        )

        return metrics

    async def collect_from_response(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        start_time: float,
        provider_id: str,
        provider_name: str,
        context: dict[str, Any] | None = None,
    ) -> RequestMetrics:
        """Collect metrics from a completed request/response.

        Convenience method that extracts timing and data from request/response.

        Args:
            request: Original request data
            response: Response data
            start_time: Request start time (from time.time())
            provider_id: Provider identifier
            provider_name: Provider display name
            context: Additional context

        Returns:
            Collected metrics
        """
        end_time = time.time()
        total_latency = int((end_time - start_time) * 1000)

        # Generate request ID if not provided
        request_id = (
            context.get("request_id", str(uuid.uuid4()))
            if context
            else str(uuid.uuid4())
        )

        # Build timing data
        timing_data = {
            "total": total_latency,
            "routing": context.get("routing_latency_ms", 0) if context else 0,
            "queue": context.get("queue_latency_ms", 0) if context else 0,
            "provider": total_latency,  # Simplified - actual provider time
            "network": context.get("network_latency_ms", 0) if context else 0,
        }

        return await self.collect_request_metrics(
            request_id=request_id,
            provider_id=provider_id,
            provider_name=provider_name,
            model=request.get("model", "unknown"),
            request_data=request,
            response_data=response,
            timing_data=timing_data,
            error_data=None,
            context=context,
        )

    async def collect_from_error(
        self,
        request: dict[str, Any],
        error: Exception,
        start_time: float,
        provider_id: str,
        provider_name: str,
        context: dict[str, Any] | None = None,
    ) -> RequestMetrics:
        """Collect metrics from a failed request.

        Args:
            request: Original request data
            error: Exception that occurred
            start_time: Request start time (from time.time())
            provider_id: Provider identifier
            provider_name: Provider display name
            context: Additional context

        Returns:
            Collected metrics
        """
        end_time = time.time()
        total_latency = int((end_time - start_time) * 1000)

        # Generate request ID if not provided
        request_id = (
            context.get("request_id", str(uuid.uuid4()))
            if context
            else str(uuid.uuid4())
        )

        # Build timing data
        timing_data = {
            "total": total_latency,
            "routing": context.get("routing_latency_ms", 0) if context else 0,
            "queue": context.get("queue_latency_ms", 0) if context else 0,
            "provider": total_latency,
            "network": context.get("network_latency_ms", 0) if context else 0,
        }

        # Build error data
        error_data = {
            "type": type(error).__name__,
            "message": str(error),
            "retry_count": context.get("retry_count", 0) if context else 0,
            "fallback_used": context.get("fallback_used", False) if context else False,
        }

        return await self.collect_request_metrics(
            request_id=request_id,
            provider_id=provider_id,
            provider_name=provider_name,
            model=request.get("model", "unknown"),
            request_data=request,
            response_data=None,
            timing_data=timing_data,
            error_data=error_data,
            context=context,
        )

    def get_metrics_history(
        self,
        limit: int | None = None,
        provider_id: str | None = None,
        tenant_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[RequestMetrics]:
        """Get metrics history with optional filters.

        Args:
            limit: Maximum number of metrics to return
            provider_id: Filter by provider
            tenant_id: Filter by tenant
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of metrics matching filters
        """
        filtered = self._metrics_history

        # Apply filters
        if provider_id is not None:
            filtered = [m for m in filtered if m.provider_id == provider_id]

        if tenant_id is not None:
            filtered = [m for m in filtered if m.tenant_id == tenant_id]

        if start_time is not None:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time is not None:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        # Sort by timestamp (most recent first)
        filtered = sorted(filtered, key=lambda m: m.timestamp, reverse=True)

        # Apply limit
        if limit is not None:
            filtered = filtered[:limit]

        return filtered

    def get_stats(self) -> dict[str, Any]:
        """Get collector statistics.

        Returns:
            Dictionary with collector statistics
        """
        total_metrics = len(self._metrics_history)
        successful = len([m for m in self._metrics_history if m.success])
        failed = len([m for m in self._metrics_history if not m.success])

        return {
            "total_metrics": total_metrics,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": (successful / total_metrics * 100)
            if total_metrics > 0
            else 0.0,
        }

    def clear_history(self) -> None:
        """Clear metrics history.

        WARNING: This permanently deletes all collected metrics.
        """
        count = len(self._metrics_history)
        self._metrics_history.clear()

        logger.warning("metrics_history_cleared", cleared_count=count)


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
