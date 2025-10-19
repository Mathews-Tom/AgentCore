"""
Integration tests for WebSocket functionality

Tests WebSocket connection, authentication, subscriptions, and event broadcasting.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def test_client(realtime_redis_db):
    """Create test client for WebSocket testing with lifespan support."""
    import sys
    import importlib
    from prometheus_client import REGISTRY

    # Import app AFTER Redis database is configured via conftest
    # realtime_redis_db fixture ensures Redis is ready with unique database

    # CRITICAL: Clear Prometheus registry to avoid duplicate metrics errors
    # Clear both the main REGISTRY and the test registry if it exists
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    # Clear test registry used by gateway.monitoring.metrics
    if "__gateway_metrics_registry__" in sys.modules:
        test_registry = sys.modules["__gateway_metrics_registry__"].registry
        test_collectors = list(test_registry._collector_to_names.keys())
        for collector in test_collectors:
            try:
                test_registry.unregister(collector)
            except Exception:
                pass

    # Clear metrics cache BEFORE removing modules
    if "gateway.monitoring.metrics" in sys.modules:
        import gateway.monitoring.metrics as metrics_module
        metrics_module._metrics_cache.clear()

    # Also clear the global cache if it exists
    if "__gateway_metrics_cache__" in sys.modules:
        sys.modules["__gateway_metrics_cache__"].cache.clear()

    # CRITICAL: Remove gateway modules from sys.modules to force fresh import
    # This ensures all modules, especially config with settings object,
    # are reloaded with the correct environment variables set by the fixture
    modules_to_remove = [
        mod for mod in list(sys.modules.keys())
        if mod.startswith("gateway.")
    ]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Remove gateway root module as well
    if "gateway" in sys.modules:
        del sys.modules["gateway"]

    from gateway.main import app

    # Use TestClient which handles lifespan events properly
    # The app's lifespan manager will initialize connection_pool and event_bus
    with TestClient(app) as client:
        yield client


class TestWebSocketIntegration:
    """WebSocket integration tests using Starlette TestClient."""

    def test_websocket_connect_without_auth(self, test_client):
        """Test WebSocket connection without authentication."""
        with test_client.websocket_connect("/realtime/ws") as websocket:
            # Should receive connection confirmation
            data = websocket.receive_json()

            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "connection_id" in data

    def test_websocket_connect_with_auth(self, test_client, authenticated_token):
        """Test WebSocket connection with authentication."""
        with test_client.websocket_connect(f"/realtime/ws?token={authenticated_token}") as websocket:
            # Should receive connection confirmation
            data = websocket.receive_json()

            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "connection_id" in data
            # Note: Authenticated status is tracked internally, not in the connection message

    def test_websocket_subscribe(self, test_client):
        """Test WebSocket subscription to topics and event types."""
        with test_client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection confirmation
            websocket.receive_json()

            # Subscribe to topics
            subscribe_message = {
                "type": "subscribe",
                "topics": ["agent.test-123"],
                "event_types": ["task.created", "task.completed"],
            }
            websocket.send_json(subscribe_message)

            # Receive subscription confirmation
            data = websocket.receive_json()

            assert data["type"] == "subscribed"
            assert "subscription_id" in data
            assert data["topics"] == ["agent.test-123"]
            assert set(data["event_types"]) == {"task.created", "task.completed"}

    def test_websocket_unsubscribe(self, test_client):
        """Test WebSocket unsubscribe."""
        with test_client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection confirmation
            websocket.receive_json()

            # Subscribe first
            subscribe_message = {
                "type": "subscribe",
                "topics": ["agent.test-123"],
            }
            websocket.send_json(subscribe_message)

            # Get subscription ID
            sub_data = websocket.receive_json()
            subscription_id = sub_data["subscription_id"]

            # Unsubscribe
            unsubscribe_message = {
                "type": "unsubscribe",
                "subscription_id": subscription_id,
            }
            websocket.send_json(unsubscribe_message)

            # Receive unsubscribe confirmation
            data = websocket.receive_json()

            assert data["type"] == "unsubscribed"
            assert data["subscription_id"] == subscription_id

    def test_websocket_ping_pong(self, test_client):
        """Test WebSocket ping/pong heartbeat."""
        with test_client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection confirmation
            websocket.receive_json()

            # Send ping
            ping_message = {"type": "ping"}
            websocket.send_json(ping_message)

            # Receive pong
            data = websocket.receive_json()

            assert data["type"] == "pong"

    def test_websocket_invalid_message(self, test_client):
        """Test WebSocket with invalid message format."""
        with test_client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection confirmation
            websocket.receive_json()

            # Send invalid JSON
            websocket.send_text("not valid json")

            # Receive error message
            data = websocket.receive_json()

            assert data["type"] == "error"
            assert "error" in data

    def test_websocket_unknown_message_type(self, test_client):
        """Test WebSocket with unknown message type."""
        with test_client.websocket_connect("/realtime/ws") as websocket:
            # Receive connection confirmation
            websocket.receive_json()

            # Send unknown message type
            unknown_message = {"type": "unknown_type"}
            websocket.send_json(unknown_message)

            # Receive error message
            data = websocket.receive_json()

            assert data["type"] == "error"
            assert "unknown" in data["error"].lower()

    def test_multiple_websocket_connections(self, test_client):
        """Test multiple WebSocket connections from same client."""
        with test_client.websocket_connect("/realtime/ws") as ws1:
            with test_client.websocket_connect("/realtime/ws") as ws2:
                # Receive connection confirmations
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()

                # Each should have unique connection ID
                assert data1["connection_id"] != data2["connection_id"]
                assert data1["type"] == "connection"
                assert data2["type"] == "connection"
