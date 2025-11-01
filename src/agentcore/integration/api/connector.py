"""Abstract base class for API connectors.

Provides connection lifecycle management, health checks, and metrics integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel
from redis.asyncio import Redis

from agentcore.integration.api.client import APIClient
from agentcore.integration.api.config import load_api_config
from agentcore.integration.api.exceptions import APIError
from agentcore.integration.api.models import APIConfig

logger = structlog.get_logger(__name__)


class ConnectorStatus(str, Enum):
    """API connector status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    UNHEALTHY = "unhealthy"
    ERROR = "error"


class HealthCheckResult(BaseModel):
    """Health check result."""

    healthy: bool
    status: ConnectorStatus
    latency_ms: float | None = None
    last_check: datetime
    error_message: str | None = None
    metadata: dict[str, Any] = {}


class APIConnector(ABC):
    """Abstract base class for API connectors.

    Provides:
    - Connection lifecycle management
    - Health check support
    - Metrics collection hooks
    - Circuit breaker integration hooks
    - Standardized error handling
    """

    def __init__(
        self,
        config: APIConfig,
        redis_client: Redis[Any] | None = None,
    ) -> None:
        """Initialize API connector.

        Args:
            config: API configuration
            redis_client: Optional Redis client for distributed features
        """
        self.config = config
        self._redis = redis_client
        self._status = ConnectorStatus.DISCONNECTED
        self._client: APIClient | None = None
        self._last_health_check: HealthCheckResult | None = None

        logger.info(
            "connector_initialized",
            name=config.name,
            base_url=config.base_url,
        )

    @classmethod
    def from_env(
        cls,
        name: str,
        base_url: str | None = None,
        redis_client: Redis[Any] | None = None,
        **kwargs: Any,
    ) -> APIConnector:
        """Create connector from environment variables.

        Args:
            name: Connector name
            base_url: Optional base URL
            redis_client: Optional Redis client
            **kwargs: Additional configuration

        Returns:
            APIConnector instance
        """
        config = load_api_config(name, base_url, **kwargs)
        return cls(config, redis_client)

    async def connect(self) -> None:
        """Establish connection to the API.

        Creates the HTTP client and performs initial health check.

        Raises:
            APIError: If connection fails
        """
        if self._status == ConnectorStatus.CONNECTED:
            logger.warning("connector_already_connected", name=self.config.name)
            return

        self._status = ConnectorStatus.CONNECTING
        logger.info("connector_connecting", name=self.config.name)

        try:
            # Create API client
            self._client = APIClient(self.config, self._redis)

            # Perform initial health check
            health = await self.health_check()

            if health.healthy:
                self._status = ConnectorStatus.CONNECTED
                logger.info(
                    "connector_connected",
                    name=self.config.name,
                    latency_ms=health.latency_ms,
                )
            else:
                self._status = ConnectorStatus.UNHEALTHY
                logger.warning(
                    "connector_unhealthy",
                    name=self.config.name,
                    error=health.error_message,
                )

        except Exception as e:
            self._status = ConnectorStatus.ERROR
            logger.error(
                "connector_connection_failed",
                name=self.config.name,
                error=str(e),
            )
            raise APIError(f"Failed to connect to {self.config.name}: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to the API.

        Releases resources and closes HTTP client.
        """
        if self._status == ConnectorStatus.DISCONNECTED:
            return

        logger.info("connector_disconnecting", name=self.config.name)

        try:
            if self._client:
                await self._client.close()
                self._client = None

            self._status = ConnectorStatus.DISCONNECTED
            logger.info("connector_disconnected", name=self.config.name)

        except Exception as e:
            logger.error(
                "connector_disconnect_failed",
                name=self.config.name,
                error=str(e),
            )
            raise

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on the API.

        Returns:
            Health check result with status and latency
        """
        start_time = datetime.now(UTC)

        try:
            # Call connector-specific health check
            is_healthy = await self._perform_health_check()

            end_time = datetime.now(UTC)
            latency_ms = (end_time - start_time).total_seconds() * 1000

            result = HealthCheckResult(
                healthy=is_healthy,
                status=ConnectorStatus.CONNECTED
                if is_healthy
                else ConnectorStatus.UNHEALTHY,
                latency_ms=latency_ms,
                last_check=end_time,
            )

        except Exception as e:
            end_time = datetime.now(UTC)
            latency_ms = (end_time - start_time).total_seconds() * 1000

            result = HealthCheckResult(
                healthy=False,
                status=ConnectorStatus.ERROR,
                latency_ms=latency_ms,
                last_check=end_time,
                error_message=str(e),
            )

            logger.error(
                "health_check_failed",
                name=self.config.name,
                error=str(e),
            )

        self._last_health_check = result
        return result

    @abstractmethod
    async def _perform_health_check(self) -> bool:
        """Perform connector-specific health check.

        Must be implemented by subclasses.

        Returns:
            True if healthy, False otherwise
        """
        pass

    def get_client(self) -> APIClient:
        """Get the API client.

        Returns:
            APIClient instance

        Raises:
            APIError: If not connected
        """
        if not self._client:
            raise APIError(f"Connector {self.config.name} is not connected")
        return self._client

    def get_status(self) -> ConnectorStatus:
        """Get current connector status.

        Returns:
            Current status
        """
        return self._status

    def get_last_health_check(self) -> HealthCheckResult | None:
        """Get last health check result.

        Returns:
            Last health check result or None
        """
        return self._last_health_check

    async def __aenter__(self) -> APIConnector:
        """Enter async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.disconnect()


class RestAPIConnector(APIConnector):
    """Generic REST API connector with standard health check.

    Uses HEAD or GET request to health check endpoint.
    """

    def __init__(
        self,
        config: APIConfig,
        redis_client: Redis[Any] | None = None,
        health_check_endpoint: str = "/health",
    ) -> None:
        """Initialize REST API connector.

        Args:
            config: API configuration
            redis_client: Optional Redis client
            health_check_endpoint: Health check endpoint path
        """
        super().__init__(config, redis_client)
        self.health_check_endpoint = health_check_endpoint

    async def _perform_health_check(self) -> bool:
        """Perform health check using HEAD or GET request.

        Returns:
            True if API responds successfully
        """
        if not self._client:
            return False

        try:
            # Try HEAD request first (lighter)
            response = await self._client.get(self.health_check_endpoint)
            return response.status_code < 400

        except Exception as e:
            logger.debug(
                "health_check_failed",
                name=self.config.name,
                error=str(e),
            )
            return False


# Global connector registry
_connectors: dict[str, APIConnector] = {}


def register_connector(name: str, connector: APIConnector) -> None:
    """Register a connector instance.

    Args:
        name: Connector name
        connector: APIConnector instance
    """
    _connectors[name] = connector
    logger.info("connector_registered", name=name)


def get_connector(name: str) -> APIConnector | None:
    """Get a registered connector by name.

    Args:
        name: Connector name

    Returns:
        APIConnector instance or None
    """
    return _connectors.get(name)


def list_connectors() -> list[str]:
    """List all registered connector names.

    Returns:
        List of connector names
    """
    return list(_connectors.keys())
