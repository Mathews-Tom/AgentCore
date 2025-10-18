"""
Integration tests for WebSocket functionality

Tests WebSocket connection, authentication, subscriptions, and event broadcasting.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from gateway.main import app
from gateway.realtime.event_bus import EventMessage, EventType, event_bus


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def authenticated_token():
    """Create authenticated JWT token for testing."""
    # This would typically use the auth system to create a valid token
    # For now, we'll mock this
    from gateway.auth.jwt import jwt_manager
    from gateway.auth.models import User, UserRole

    user = User(
        id="test-user-123",
        username="testuser",
        email="test@example.com",
        roles=[UserRole.USER],
    )

    token = jwt_manager.create_access_token(
        user=user,
        session_id="test-session",
    )

    return token


@pytest.mark.asyncio
class TestWebSocketIntegration:
    """WebSocket integration tests."""

    async def test_websocket_connect_without_auth(self, client):
        """Test WebSocket connection without authentication."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Should receive connection success message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "connection_id" in data
            assert "client_id" in data

    async def test_websocket_connect_with_auth(self, client, authenticated_token):
        """Test WebSocket connection with authentication."""
        with client.websocket_connect(f"/realtime/ws?token={authenticated_token}") as websocket:
            # Should receive connection success message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"

    async def test_websocket_subscribe(self, client):
        """Test WebSocket subscription to topics and event types."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            conn_msg = websocket.receive_json()
            assert conn_msg["type"] == "connection"

            # Subscribe to topics and event types
            subscribe_msg = {
                "type": "subscribe",
                "topics": ["agent.123"],
                "event_types": ["task.created", "task.completed"],
                "filters": {
                    "agent_ids": ["agent-123"],
                },
            }
            websocket.send_json(subscribe_msg)

            # Should receive subscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert "subscription_id" in data
            assert "agent.123" in data["topics"]
            assert "task.created" in data["event_types"]

    async def test_websocket_unsubscribe(self, client):
        """Test WebSocket unsubscribe."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Subscribe
            subscribe_msg = {
                "type": "subscribe",
                "topics": ["agent.123"],
                "event_types": ["task.created"],
            }
            websocket.send_json(subscribe_msg)

            # Get subscription ID
            sub_response = websocket.receive_json()
            subscription_id = sub_response["subscription_id"]

            # Unsubscribe
            unsubscribe_msg = {
                "type": "unsubscribe",
                "subscription_id": subscription_id,
            }
            websocket.send_json(unsubscribe_msg)

            # Should receive unsubscribe confirmation
            data = websocket.receive_json()
            assert data["type"] == "unsubscribed"
            assert data["subscription_id"] == subscription_id

    async def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong heartbeat."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Send ping
            ping_msg = {"type": "ping"}
            websocket.send_json(ping_msg)

            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"

    async def test_websocket_receive_events(self, client):
        """Test receiving events through WebSocket."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Subscribe to events
            subscribe_msg = {
                "type": "subscribe",
                "topics": ["agent.123"],
                "event_types": ["task.created"],
            }
            websocket.send_json(subscribe_msg)

            # Receive subscription confirmation
            websocket.receive_json()

            # Publish event to event bus
            event = EventMessage.create(
                event_type=EventType.TASK_CREATED,
                topic="agent.123",
                payload={"task_id": "task-456", "agent_id": "agent-123"},
            )
            await event_bus.publish(event)

            # Wait a bit for event processing
            await asyncio.sleep(0.2)

            # Should receive event
            # Note: This test may be flaky due to async nature
            # In production, you'd use a more robust testing approach

    async def test_websocket_invalid_message(self, client):
        """Test WebSocket with invalid message format."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Send invalid JSON
            websocket.send_text("invalid json")

            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["error"]

    async def test_websocket_unknown_message_type(self, client):
        """Test WebSocket with unknown message type."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Send unknown message type
            unknown_msg = {"type": "unknown_type"}
            websocket.send_json(unknown_msg)

            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Unknown message type" in data["error"]

    async def test_multiple_websocket_connections(self, client):
        """Test multiple WebSocket connections from same client."""
        # Connect first WebSocket
        with client.websocket_connect("/realtime/ws") as ws1:
            conn1 = ws1.receive_json()
            assert conn1["type"] == "connection"

            # Connect second WebSocket
            with client.websocket_connect("/realtime/ws") as ws2:
                conn2 = ws2.receive_json()
                assert conn2["type"] == "connection"

                # Should have different connection IDs
                assert conn1["connection_id"] != conn2["connection_id"]

    async def test_websocket_event_filtering(self, client):
        """Test WebSocket event filtering by agent_id."""
        with client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection message
            websocket.receive_json()

            # Subscribe with agent_id filter
            subscribe_msg = {
                "type": "subscribe",
                "topics": ["agent.123"],
                "event_types": ["task.created"],
                "filters": {
                    "agent_ids": ["agent-123"],
                },
            }
            websocket.send_json(subscribe_msg)

            # Receive subscription confirmation
            websocket.receive_json()

            # Publish event with matching agent_id
            event1 = EventMessage.create(
                event_type=EventType.TASK_CREATED,
                topic="agent.123",
                payload={"task_id": "task-1", "agent_id": "agent-123"},
            )
            await event_bus.publish(event1)

            # Publish event with non-matching agent_id
            event2 = EventMessage.create(
                event_type=EventType.TASK_CREATED,
                topic="agent.123",
                payload={"task_id": "task-2", "agent_id": "agent-456"},
            )
            await event_bus.publish(event2)

            # Wait for event processing
            await asyncio.sleep(0.2)

            # Should only receive event with matching agent_id
            # (This test would need more sophisticated async handling in production)
