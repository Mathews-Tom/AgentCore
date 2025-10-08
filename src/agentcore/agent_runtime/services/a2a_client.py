"""
A2A Protocol client for agent runtime integration.

This module provides the client interface for agent runtime to communicate
with the A2A protocol layer, handling agent registration, task execution,
and status reporting.
"""

import asyncio
from datetime import datetime
from typing import Any
from uuid import uuid4

import httpx
import structlog

from ...a2a_protocol.models.agent import AgentCard
from ...a2a_protocol.models.jsonrpc import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
)
from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState

logger = structlog.get_logger()


class A2AClientError(Exception):
    """Base exception for A2A client errors."""


class A2AConnectionError(A2AClientError):
    """Raised when connection to A2A protocol fails."""


class A2ARegistrationError(A2AClientError):
    """Raised when agent registration fails."""


class A2AClient:
    """Client for A2A protocol integration."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize A2A client.

        Args:
            base_url: Base URL of A2A protocol service
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._jsonrpc_url = f"{self._base_url}/api/v1/jsonrpc"
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "A2AClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _call_jsonrpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """
        Make JSON-RPC call to A2A protocol.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            Method result

        Raises:
            A2AConnectionError: If connection fails
            A2AClientError: If JSON-RPC error occurs
        """
        if not self._client:
            raise A2AConnectionError("Client not initialized, use async context manager")

        request = JsonRpcRequest(
            id=str(uuid4()),
            method=method,
            params=params or {},
        )

        try:
            response = await self._client.post(
                self._jsonrpc_url,
                json=request.model_dump(mode="json", exclude_none=True),
            )
            response.raise_for_status()

            data = response.json()
            rpc_response = JsonRpcResponse(**data)

            if rpc_response.error:
                raise A2AClientError(
                    f"JSON-RPC error: {rpc_response.error.message}"
                )

            return rpc_response.result

        except httpx.HTTPError as e:
            raise A2AConnectionError(f"HTTP error: {e}")
        except Exception as e:
            raise A2AClientError(f"Unexpected error: {e}")

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
        agent_card = AgentCard(
            agent_id=agent_config.agent_id,
            name=agent_config.name,
            description=getattr(agent_config, "description", "Agent runtime agent"),
            capabilities=agent_config.capabilities,
            version="1.0.0",
            protocol_version="0.2",
            endpoint=f"{self._base_url}/agents/{agent_config.agent_id}",
            authentication={"type": "bearer"},
        )

        try:
            result = await self._call_jsonrpc(
                method="agent.register",
                params={"agent_card": agent_card.model_dump(mode="json")},
            )

            logger.info(
                "agent_registered",
                agent_id=agent_config.agent_id,
                name=agent_config.name,
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
                "timestamp": datetime.now().isoformat(),
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
