"""
Connection Pool Management

Manages WebSocket and SSE connections with health monitoring and scaling.
Optimized for 10,000+ concurrent connections per instance.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ConnectionState(str, Enum):
    """Connection state enumeration."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ConnectionType(str, Enum):
    """Connection type enumeration."""

    WEBSOCKET = "websocket"
    SSE = "sse"


@dataclass
class ConnectionInfo:
    """Connection information."""

    connection_id: str
    connection_type: ConnectionType
    client_id: str
    user_id: str | None
    state: ConnectionState
    created_at: float
    last_activity: float
    remote_addr: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] | None = None

    # Health monitoring
    ping_count: int = 0
    pong_count: int = 0
    last_ping: float | None = None
    last_pong: float | None = None

    # Statistics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert connection info to dictionary."""
        return {
            "connection_id": self.connection_id,
            "connection_type": self.connection_type.value,
            "client_id": self.client_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "remote_addr": self.remote_addr,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
            "health": {
                "ping_count": self.ping_count,
                "pong_count": self.pong_count,
                "last_ping": self.last_ping,
                "last_pong": self.last_pong,
            },
            "stats": {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
            },
        }


class ConnectionPool:
    """
    Connection pool for WebSocket and SSE connections.

    Optimized for 10,000+ concurrent connections with efficient lookup,
    health monitoring, and resource management.
    """

    def __init__(
        self,
        max_connections: int = 10000,
        heartbeat_interval: int = 30,
        connection_timeout: int = 300,
    ) -> None:
        """
        Initialize connection pool.

        Args:
            max_connections: Maximum number of concurrent connections
            heartbeat_interval: Heartbeat interval in seconds
            connection_timeout: Connection timeout in seconds
        """
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout

        # Connections by connection ID
        self._connections: dict[str, ConnectionInfo] = {}

        # Connections by client ID (one client can have multiple connections)
        self._client_connections: dict[str, set[str]] = {}

        # Connections by user ID
        self._user_connections: dict[str, set[str]] = {}

        # Heartbeat monitoring task
        self._heartbeat_task: asyncio.Task[None] | None = None

        # Statistics
        self._stats = {
            "total_connections": 0,
            "websocket_connections": 0,
            "sse_connections": 0,
            "active_clients": 0,
            "active_users": 0,
        }

        logger.info(
            "Connection pool initialized",
            max_connections=max_connections,
            heartbeat_interval=heartbeat_interval,
            connection_timeout=connection_timeout,
        )

    async def start(self) -> None:
        """Start connection pool monitoring."""
        if self._heartbeat_task is not None:
            logger.warning("Connection pool already started")
            return

        self._heartbeat_task = asyncio.create_task(self._monitor_connections())
        logger.info("Connection pool monitoring started")

    async def stop(self) -> None:
        """Stop connection pool monitoring."""
        if self._heartbeat_task is None:
            return

        # Cancel monitoring task
        self._heartbeat_task.cancel()
        try:
            await self._heartbeat_task
        except asyncio.CancelledError:
            pass

        self._heartbeat_task = None
        logger.info("Connection pool monitoring stopped")

    def add_connection(
        self,
        connection_id: str,
        connection_type: ConnectionType,
        client_id: str,
        user_id: str | None = None,
        remote_addr: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConnectionInfo | None:
        """
        Add connection to pool.

        Args:
            connection_id: Unique connection identifier
            connection_type: Connection type (WebSocket or SSE)
            client_id: Client identifier
            user_id: User identifier (optional)
            remote_addr: Remote IP address (optional)
            user_agent: User agent string (optional)
            metadata: Additional metadata (optional)

        Returns:
            ConnectionInfo if added, None if pool is full
        """
        # Check pool capacity
        if len(self._connections) >= self.max_connections:
            logger.warning(
                "Connection pool full",
                max_connections=self.max_connections,
                current=len(self._connections),
            )
            return None

        # Create connection info
        now = time.time()
        connection = ConnectionInfo(
            connection_id=connection_id,
            connection_type=connection_type,
            client_id=client_id,
            user_id=user_id,
            state=ConnectionState.CONNECTED,
            created_at=now,
            last_activity=now,
            remote_addr=remote_addr,
            user_agent=user_agent,
            metadata=metadata,
        )

        # Store connection
        self._connections[connection_id] = connection

        # Add to client connections
        if client_id not in self._client_connections:
            self._client_connections[client_id] = set()
        self._client_connections[client_id].add(connection_id)

        # Add to user connections
        if user_id:
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(connection_id)

        # Update statistics
        self._update_stats()

        logger.info(
            "Connection added",
            connection_id=connection_id,
            connection_type=connection_type.value,
            client_id=client_id,
            user_id=user_id,
        )

        return connection

    def remove_connection(self, connection_id: str) -> bool:
        """
        Remove connection from pool.

        Args:
            connection_id: Connection identifier

        Returns:
            True if removed, False if not found
        """
        if connection_id not in self._connections:
            return False

        # Get connection
        connection = self._connections[connection_id]

        # Update state
        connection.state = ConnectionState.DISCONNECTED

        # Remove from client connections
        if connection.client_id in self._client_connections:
            self._client_connections[connection.client_id].discard(connection_id)
            if not self._client_connections[connection.client_id]:
                del self._client_connections[connection.client_id]

        # Remove from user connections
        if connection.user_id and connection.user_id in self._user_connections:
            self._user_connections[connection.user_id].discard(connection_id)
            if not self._user_connections[connection.user_id]:
                del self._user_connections[connection.user_id]

        # Remove connection
        del self._connections[connection_id]

        # Update statistics
        self._update_stats()

        logger.info(
            "Connection removed",
            connection_id=connection_id,
            client_id=connection.client_id,
        )

        return True

    def get_connection(self, connection_id: str) -> ConnectionInfo | None:
        """Get connection by ID."""
        return self._connections.get(connection_id)

    def get_client_connections(self, client_id: str) -> list[ConnectionInfo]:
        """Get all connections for a client."""
        if client_id not in self._client_connections:
            return []

        connection_ids = self._client_connections[client_id]
        return [
            self._connections[conn_id]
            for conn_id in connection_ids
            if conn_id in self._connections
        ]

    def get_user_connections(self, user_id: str) -> list[ConnectionInfo]:
        """Get all connections for a user."""
        if user_id not in self._user_connections:
            return []

        connection_ids = self._user_connections[user_id]
        return [
            self._connections[conn_id]
            for conn_id in connection_ids
            if conn_id in self._connections
        ]

    def update_activity(self, connection_id: str) -> bool:
        """
        Update connection last activity timestamp.

        Args:
            connection_id: Connection identifier

        Returns:
            True if updated, False if not found
        """
        if connection_id not in self._connections:
            return False

        self._connections[connection_id].last_activity = time.time()
        return True

    def record_ping(self, connection_id: str) -> bool:
        """
        Record ping sent to connection.

        Args:
            connection_id: Connection identifier

        Returns:
            True if recorded, False if not found
        """
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]
        connection.ping_count += 1
        connection.last_ping = time.time()
        return True

    def record_pong(self, connection_id: str) -> bool:
        """
        Record pong received from connection.

        Args:
            connection_id: Connection identifier

        Returns:
            True if recorded, False if not found
        """
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]
        connection.pong_count += 1
        connection.last_pong = time.time()
        return True

    def record_message_sent(self, connection_id: str, size: int) -> bool:
        """
        Record message sent to connection.

        Args:
            connection_id: Connection identifier
            size: Message size in bytes

        Returns:
            True if recorded, False if not found
        """
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]
        connection.messages_sent += 1
        connection.bytes_sent += size
        connection.last_activity = time.time()
        return True

    def record_message_received(self, connection_id: str, size: int) -> bool:
        """
        Record message received from connection.

        Args:
            connection_id: Connection identifier
            size: Message size in bytes

        Returns:
            True if recorded, False if not found
        """
        if connection_id not in self._connections:
            return False

        connection = self._connections[connection_id]
        connection.messages_received += 1
        connection.bytes_received += size
        connection.last_activity = time.time()
        return True

    async def _monitor_connections(self) -> None:
        """Monitor connection health and remove stale connections."""
        logger.info("Connection monitoring started")

        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Check for stale connections
                now = time.time()
                stale_connections = []

                for conn_id, conn in self._connections.items():
                    if now - conn.last_activity > self.connection_timeout:
                        stale_connections.append(conn_id)

                # Remove stale connections
                for conn_id in stale_connections:
                    logger.warning(
                        "Removing stale connection",
                        connection_id=conn_id,
                        last_activity=self._connections[conn_id].last_activity,
                    )
                    self.remove_connection(conn_id)

            except asyncio.CancelledError:
                logger.info("Connection monitoring cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error monitoring connections",
                    error=str(e),
                    exc_info=True,
                )

    def _update_stats(self) -> None:
        """Update pool statistics."""
        websocket_count = sum(
            1
            for conn in self._connections.values()
            if conn.connection_type == ConnectionType.WEBSOCKET
        )
        sse_count = sum(
            1
            for conn in self._connections.values()
            if conn.connection_type == ConnectionType.SSE
        )

        self._stats = {
            "total_connections": len(self._connections),
            "websocket_connections": websocket_count,
            "sse_connections": sse_count,
            "active_clients": len(self._client_connections),
            "active_users": len(self._user_connections),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self._stats,
            "max_connections": self.max_connections,
            "utilization": len(self._connections) / self.max_connections,
        }


# Global connection pool instance
connection_pool = ConnectionPool()
