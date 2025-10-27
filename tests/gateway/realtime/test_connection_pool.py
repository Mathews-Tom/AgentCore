"""
Tests for Connection Pool

Tests connection management, health monitoring, and statistics.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gateway.realtime.connection_pool import ConnectionPool, ConnectionState, ConnectionType


@pytest.fixture
async def connection_pool():
    """Create connection pool instance for testing."""
    pool = ConnectionPool(
        max_connections=100,
        heartbeat_interval=1,
        connection_timeout=5)
    await pool.start()
    yield pool
    await pool.stop()


@pytest.mark.asyncio
class TestConnectionPool:
    """Connection pool tests."""

    async def test_add_connection(self, connection_pool):
        """Test adding connection to pool."""
        conn_info = connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1",
            user_id="user-1",
            remote_addr="127.0.0.1:5000",
            user_agent="TestAgent/1.0")

        assert conn_info is not None
        assert conn_info.connection_id == "conn-1"
        assert conn_info.connection_type == ConnectionType.WEBSOCKET
        assert conn_info.client_id == "client-1"
        assert conn_info.user_id == "user-1"
        assert conn_info.state == ConnectionState.CONNECTED
        assert conn_info.remote_addr == "127.0.0.1:5000"
        assert conn_info.user_agent == "TestAgent/1.0"

        # Cleanup
        connection_pool.remove_connection("conn-1")

    async def test_remove_connection(self, connection_pool):
        """Test removing connection from pool."""
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1")

        result = connection_pool.remove_connection("conn-1")
        assert result is True

        # Verify connection removed
        conn_info = connection_pool.get_connection("conn-1")
        assert conn_info is None

    async def test_max_connections_limit(self, connection_pool):
        """Test connection pool capacity limit."""
        # Fill pool to capacity
        for i in range(100):
            conn_info = connection_pool.add_connection(
                connection_id=f"conn-{i}",
                connection_type=ConnectionType.WEBSOCKET,
                client_id=f"client-{i}")
            assert conn_info is not None

        # Try to add one more connection (should fail)
        conn_info = connection_pool.add_connection(
            connection_id="conn-overflow",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-overflow")
        assert conn_info is None

        # Cleanup
        for i in range(100):
            connection_pool.remove_connection(f"conn-{i}")

    async def test_get_client_connections(self, connection_pool):
        """Test getting all connections for a client."""
        # Add multiple connections for same client
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1")
        connection_pool.add_connection(
            connection_id="conn-2",
            connection_type=ConnectionType.SSE,
            client_id="client-1")
        connection_pool.add_connection(
            connection_id="conn-3",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-2")

        # Get connections for client-1
        connections = connection_pool.get_client_connections("client-1")
        assert len(connections) == 2

        connection_ids = {conn.connection_id for conn in connections}
        assert "conn-1" in connection_ids
        assert "conn-2" in connection_ids

        # Cleanup
        connection_pool.remove_connection("conn-1")
        connection_pool.remove_connection("conn-2")
        connection_pool.remove_connection("conn-3")

    async def test_get_user_connections(self, connection_pool):
        """Test getting all connections for a user."""
        # Add connections for different users
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1",
            user_id="user-1")
        connection_pool.add_connection(
            connection_id="conn-2",
            connection_type=ConnectionType.SSE,
            client_id="client-2",
            user_id="user-1")
        connection_pool.add_connection(
            connection_id="conn-3",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-3",
            user_id="user-2")

        # Get connections for user-1
        connections = connection_pool.get_user_connections("user-1")
        assert len(connections) == 2

        connection_ids = {conn.connection_id for conn in connections}
        assert "conn-1" in connection_ids
        assert "conn-2" in connection_ids

        # Cleanup
        connection_pool.remove_connection("conn-1")
        connection_pool.remove_connection("conn-2")
        connection_pool.remove_connection("conn-3")

    async def test_update_activity(self, connection_pool):
        """Test updating connection activity timestamp."""
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1")

        # Get initial last activity
        conn_info = connection_pool.get_connection("conn-1")
        initial_activity = conn_info.last_activity

        # Wait a bit
        await asyncio.sleep(0.1)

        # Update activity
        result = connection_pool.update_activity("conn-1")
        assert result is True

        # Verify activity updated
        conn_info = connection_pool.get_connection("conn-1")
        assert conn_info.last_activity > initial_activity

        # Cleanup
        connection_pool.remove_connection("conn-1")

    async def test_record_ping_pong(self, connection_pool):
        """Test recording ping/pong messages."""
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1")

        # Record ping
        result = connection_pool.record_ping("conn-1")
        assert result is True

        # Record pong
        result = connection_pool.record_pong("conn-1")
        assert result is True

        # Verify counts
        conn_info = connection_pool.get_connection("conn-1")
        assert conn_info.ping_count == 1
        assert conn_info.pong_count == 1
        assert conn_info.last_ping is not None
        assert conn_info.last_pong is not None

        # Cleanup
        connection_pool.remove_connection("conn-1")

    async def test_record_messages(self, connection_pool):
        """Test recording sent/received messages."""
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1")

        # Record sent message
        result = connection_pool.record_message_sent("conn-1", 100)
        assert result is True

        # Record received message
        result = connection_pool.record_message_received("conn-1", 50)
        assert result is True

        # Verify statistics
        conn_info = connection_pool.get_connection("conn-1")
        assert conn_info.messages_sent == 1
        assert conn_info.messages_received == 1
        assert conn_info.bytes_sent == 100
        assert conn_info.bytes_received == 50

        # Cleanup
        connection_pool.remove_connection("conn-1")

    async def test_stale_connection_removal(self, connection_pool):
        """Test automatic removal of stale connections."""
        # Add connection
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1")

        # Get connection and manually set old last_activity
        conn_info = connection_pool.get_connection("conn-1")
        conn_info.last_activity = time.time() - 10  # 10 seconds ago

        # Wait for monitoring to run
        await asyncio.sleep(2)

        # Verify connection removed
        conn_info = connection_pool.get_connection("conn-1")
        assert conn_info is None

    async def test_get_stats(self, connection_pool):
        """Test getting connection pool statistics."""
        # Add some connections
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1",
            user_id="user-1")
        connection_pool.add_connection(
            connection_id="conn-2",
            connection_type=ConnectionType.SSE,
            client_id="client-2",
            user_id="user-1")

        # Get stats
        stats = connection_pool.get_stats()

        assert stats["total_connections"] == 2
        assert stats["websocket_connections"] == 1
        assert stats["sse_connections"] == 1
        assert stats["active_clients"] == 2
        assert stats["active_users"] == 1
        assert stats["max_connections"] == 100
        assert 0 < stats["utilization"] <= 1

        # Cleanup
        connection_pool.remove_connection("conn-1")
        connection_pool.remove_connection("conn-2")

    async def test_connection_info_to_dict(self, connection_pool):
        """Test connection info serialization."""
        connection_pool.add_connection(
            connection_id="conn-1",
            connection_type=ConnectionType.WEBSOCKET,
            client_id="client-1",
            user_id="user-1",
            remote_addr="127.0.0.1:5000",
            user_agent="TestAgent/1.0",
            metadata={"key": "value"})

        conn_info = connection_pool.get_connection("conn-1")
        conn_dict = conn_info.to_dict()

        assert conn_dict["connection_id"] == "conn-1"
        assert conn_dict["connection_type"] == ConnectionType.WEBSOCKET.value
        assert conn_dict["client_id"] == "client-1"
        assert conn_dict["user_id"] == "user-1"
        assert conn_dict["state"] == ConnectionState.CONNECTED.value
        assert conn_dict["remote_addr"] == "127.0.0.1:5000"
        assert conn_dict["user_agent"] == "TestAgent/1.0"
        assert conn_dict["metadata"] == {"key": "value"}
        assert "health" in conn_dict
        assert "stats" in conn_dict

        # Cleanup
        connection_pool.remove_connection("conn-1")
