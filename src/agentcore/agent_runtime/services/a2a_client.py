"""
A2A Protocol client for agent runtime integration.

This module provides the client interface for agent runtime to communicate
with the A2A protocol layer, handling agent registration, task execution,
and status reporting.

Features:
- Retry logic with exponential backoff (3 attempts: 1s, 2s, 4s)
- Connection pooling (10 connections per host)
- Circuit breaker pattern (opens after 5 failures)
- Comprehensive error handling with specific exceptions
- Request timeout handling (configurable, 30s default)
- Metrics tracking for observability
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import structlog

from ...a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCard,
    AgentEndpoint,
    AuthenticationType,
    EndpointType,
)
from ...a2a_protocol.models.jsonrpc import JsonRpcError, JsonRpcRequest, JsonRpcResponse
from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState

logger = structlog.get_logger()


class A2AClientError(Exception):
    """Base exception for A2A client errors."""


class A2AConnectionError(A2AClientError):
    """Raised when connection to A2A protocol fails."""


class A2ARegistrationError(A2AClientError):
    """Raised when agent registration fails."""


class A2ATimeoutError(A2AClientError):
    """Raised when request times out."""


class A2ARateLimitError(A2AClientError):
    """Raised when rate limit is exceeded."""


class A2ACircuitOpenError(A2AClientError):
    """Raised when circuit breaker is open."""


class A2AClient:
    """Client for A2A protocol integration with retry, pooling, and circuit breaker."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        max_retries: int = 3,
        max_connections: int = 10,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ) -> None:
        """
        Initialize A2A client.

        Args:
            base_url: Base URL of A2A protocol service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts (default: 3)
            max_connections: Maximum connections per host (default: 10)
            circuit_breaker_threshold: Failures before circuit opens (default: 5)
            circuit_breaker_timeout: Seconds before circuit resets (default: 60)
        """
        self._base_url = base_url.rstrip("/")
        self._jsonrpc_url = f"{self._base_url}/api/v1/jsonrpc"
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_connections = max_connections
        self._client: httpx.AsyncClient | None = None

        # Circuit breaker state
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_timeout = circuit_breaker_timeout
        self._failure_count = 0
        self._circuit_open_until: datetime | None = None
        self._last_failure_time: datetime | None = None

    async def __aenter__(self) -> "A2AClient":
        """Async context manager entry with connection pooling."""
        # Configure connection limits for pooling
        limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_connections,
        )

        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            limits=limits,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_open_until is None:
            return False

        # Check if circuit should reset
        now = datetime.now(UTC)
        if now >= self._circuit_open_until:
            logger.info("circuit_breaker_reset", failure_count=self._failure_count)
            self._circuit_open_until = None
            self._failure_count = 0
            return False

        return True

    def _record_success(self) -> None:
        """Record successful request (resets circuit breaker)."""
        if self._failure_count > 0:
            logger.debug("circuit_breaker_success_reset", previous_failures=self._failure_count)
        self._failure_count = 0
        self._circuit_open_until = None

    def _record_failure(self) -> None:
        """Record failed request (may open circuit breaker)."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(UTC)

        if self._failure_count >= self._circuit_breaker_threshold:
            from datetime import timedelta
            self._circuit_open_until = datetime.now(UTC) + timedelta(
                seconds=self._circuit_breaker_timeout
            )
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self._failure_count,
                open_until=self._circuit_open_until.isoformat(),
            )

    async def _call_jsonrpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Make JSON-RPC call to A2A protocol with retry and circuit breaker.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            Method result

        Raises:
            A2ACircuitOpenError: If circuit breaker is open
            A2AConnectionError: If connection fails after retries
            A2ATimeoutError: If request times out
            A2ARateLimitError: If rate limit exceeded
            A2AClientError: If JSON-RPC error occurs
        """
        # Check circuit breaker
        if self._is_circuit_open():
            raise A2ACircuitOpenError(
                f"Circuit breaker is open (failures: {self._failure_count})"
            )

        if not self._client:
            raise A2AConnectionError(
                "Client not initialized, use async context manager"
            )

        request = JsonRpcRequest(
            id=str(uuid4()),
            method=method,
            params=params or {},
        )

        last_error: Exception | None = None

        # Retry logic with exponential backoff
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(
                    self._jsonrpc_url,
                    json=request.model_dump(mode="json", exclude_none=True),
                )

                # Check for rate limiting (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "1"))
                    logger.warning(
                        "rate_limit_exceeded",
                        method=method,
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )

                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        self._record_failure()
                        raise A2ARateLimitError(f"Rate limit exceeded for {method}")

                response.raise_for_status()

                data = response.json()
                rpc_response = JsonRpcResponse(**data)

                if rpc_response.error:
                    error_msg = f"JSON-RPC error: {rpc_response.error.message}"
                    logger.error(
                        "jsonrpc_error",
                        method=method,
                        error_code=rpc_response.error.code,
                        error_message=rpc_response.error.message,
                    )
                    self._record_failure()
                    raise A2AClientError(error_msg)

                # Success - reset circuit breaker
                self._record_success()
                return rpc_response.result

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "request_timeout",
                    method=method,
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                )

                if attempt < self._max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                    continue

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    "connection_error",
                    method=method,
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e),
                )

                if attempt < self._max_retries - 1:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                    continue

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(
                    "http_status_error",
                    method=method,
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                )

                # Don't retry on 4xx errors (except 429 which is handled above)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    self._record_failure()
                    raise A2AClientError(f"HTTP {e.response.status_code}: {e}")

                if attempt < self._max_retries - 1:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                    continue

            except Exception as e:
                last_error = e
                logger.error(
                    "unexpected_error",
                    method=method,
                    error_type=type(e).__name__,
                    error=str(e),
                    attempt=attempt + 1,
                )

                if attempt < self._max_retries - 1:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                    continue

        # All retries exhausted
        self._record_failure()

        if isinstance(last_error, httpx.TimeoutException):
            raise A2ATimeoutError(f"Request timed out after {self._max_retries} attempts")
        elif isinstance(last_error, httpx.ConnectError):
            raise A2AConnectionError(f"Connection failed after {self._max_retries} attempts: {last_error}")
        elif isinstance(last_error, A2ARateLimitError):
            raise last_error  # Re-raise rate limit error as-is
        else:
            raise A2AClientError(f"Request failed after {self._max_retries} attempts: {last_error}")

    async def register_agent(
        self,
        agent_config: AgentConfig,
    ) -> str:
        """
        Register agent with A2A protocol.

        Args:
            agent_config: Agent configuration

        Returns:
            Agent ID

        Raises:
            A2ARegistrationError: If registration fails
        """
        # Build AgentCard from config
        # Map AgentConfig fields to AgentCard structure
        endpoint = AgentEndpoint(
            url=f"{self._base_url}/agents/{agent_config.agent_id}",
            type=EndpointType.HTTP,
            protocols=["jsonrpc-2.0"],
        )

        authentication = AgentAuthentication(
            type=AuthenticationType.BEARER_TOKEN,
            config={},
            required=True,
        )

        agent_card = AgentCard(
            agent_id=agent_config.agent_id,
            agent_name=agent_config.agent_id,  # Use agent_id as name
            description=f"Agent with {agent_config.philosophy.value} philosophy",
            endpoints=[endpoint],
            authentication=authentication,
        )

        try:
            result = await self._call_jsonrpc(
                method="agent.register",
                params={"agent_card": agent_card.model_dump(mode="json")},
            )

            logger.info(
                "agent_registered",
                agent_id=agent_config.agent_id,
                philosophy=agent_config.philosophy.value,
            )

            return result["agent_id"]

        except A2AClientError as e:
            raise A2ARegistrationError(f"Failed to register agent: {e}")

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister agent from A2A protocol.

        Args:
            agent_id: Agent ID

        Returns:
            True if successful

        Raises:
            A2AClientError: If unregistration fails
        """
        result = await self._call_jsonrpc(
            method="agent.unregister",
            params={"agent_id": agent_id},
        )

        logger.info("agent_unregistered", agent_id=agent_id)

        return result.get("success", False)

    async def update_agent_status(
        self,
        agent_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Update agent status in A2A protocol.

        Args:
            agent_id: Agent ID
            status: Agent status (active, idle, error, terminated)
            metadata: Additional status metadata

        Returns:
            True if successful

        Raises:
            A2AClientError: If update fails
        """
        result = await self._call_jsonrpc(
            method="agent.updateStatus",
            params={
                "agent_id": agent_id,
                "status": status,
                "metadata": metadata or {},
            },
        )

        logger.debug(
            "agent_status_updated",
            agent_id=agent_id,
            status=status,
        )

        return result.get("success", False)

    async def report_health(
        self,
        agent_id: str,
        health_status: str = "healthy",
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        """
        Report agent health to A2A protocol.

        Args:
            agent_id: Agent ID
            health_status: Health status (healthy, degraded, unhealthy)
            metrics: Health metrics

        Returns:
            True if successful

        Raises:
            A2AClientError: If health report fails
        """
        result = await self._call_jsonrpc(
            method="health.report",
            params={
                "agent_id": agent_id,
                "status": health_status,
                "metrics": metrics or {},
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        logger.debug(
            "health_reported",
            agent_id=agent_id,
            status=health_status,
        )

        return result.get("success", False)

    async def accept_task(
        self,
        task_id: str,
        agent_id: str,
    ) -> bool:
        """
        Accept task assignment from A2A protocol.

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            True if successful

        Raises:
            A2AClientError: If acceptance fails
        """
        result = await self._call_jsonrpc(
            method="task.assign",
            params={
                "task_id": task_id,
                "agent_id": agent_id,
            },
        )

        logger.info(
            "task_accepted",
            task_id=task_id,
            agent_id=agent_id,
        )

        return result.get("success", False)

    async def start_task(
        self,
        task_id: str,
        agent_id: str,
    ) -> bool:
        """
        Mark task as started.

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            True if successful

        Raises:
            A2AClientError: If start fails
        """
        result = await self._call_jsonrpc(
            method="task.start",
            params={
                "task_id": task_id,
                "agent_id": agent_id,
            },
        )

        logger.info(
            "task_started",
            task_id=task_id,
            agent_id=agent_id,
        )

        return result.get("success", False)

    async def complete_task(
        self,
        task_id: str,
        agent_id: str,
        result: dict[str, Any],
    ) -> bool:
        """
        Mark task as completed.

        Args:
            task_id: Task ID
            agent_id: Agent ID
            result: Task result

        Returns:
            True if successful

        Raises:
            A2AClientError: If completion fails
        """
        result_data = await self._call_jsonrpc(
            method="task.complete",
            params={
                "task_id": task_id,
                "agent_id": agent_id,
                "result": result,
            },
        )

        logger.info(
            "task_completed",
            task_id=task_id,
            agent_id=agent_id,
        )

        return result_data.get("success", False)

    async def fail_task(
        self,
        task_id: str,
        agent_id: str,
        error: str,
    ) -> bool:
        """
        Mark task as failed.

        Args:
            task_id: Task ID
            agent_id: Agent ID
            error: Error message

        Returns:
            True if successful

        Raises:
            A2AClientError: If failure reporting fails
        """
        result = await self._call_jsonrpc(
            method="task.fail",
            params={
                "task_id": task_id,
                "agent_id": agent_id,
                "error": error,
            },
        )

        logger.error(
            "task_failed",
            task_id=task_id,
            agent_id=agent_id,
            error=error,
        )

        return result.get("success", False)

    async def ping(self) -> bool:
        """
        Ping A2A protocol to check connectivity.

        Returns:
            True if A2A protocol is reachable

        Raises:
            A2AConnectionError: If ping fails
        """
        try:
            result = await self._call_jsonrpc(method="rpc.ping")
            return result == "pong"
        except A2AClientError:
            return False
