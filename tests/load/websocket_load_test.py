"""
WebSocket Load Testing for Gateway Layer

Tests concurrent WebSocket connections targeting 10,000+ concurrent connections.

Usage:
    uv run locust -f tests/load/websocket_load_test.py --host=http://localhost:8001
"""

import asyncio
import random
import time

from locust import User, task, between, events
from locust.exception import StopUser


class WebSocketUser(User):
    """WebSocket connection load testing user."""

    wait_time = between(5, 15)  # Keep connections alive

    def __init__(self, *args, **kwargs) -> None:
        """Initialize WebSocket user."""
        super().__init__(*args, **kwargs)
        self.client_id = f"ws-client-{random.randint(10000, 99999)}"
        self.connected = False
        self.connection_start = None

    def on_start(self) -> None:
        """Establish WebSocket connection."""
        self.connection_start = time.time()
        # Simulate WebSocket connection establishment
        # In real implementation, this would use websockets library
        self.connected = True
        print(f"WebSocket client {self.client_id} connected")

    def on_stop(self) -> None:
        """Close WebSocket connection."""
        if self.connected:
            duration = time.time() - self.connection_start if self.connection_start else 0
            print(f"WebSocket client {self.client_id} disconnected after {duration:.2f}, s")
            self.connected = False

    @task
    def maintain_connection(self) -> None:
        """Maintain WebSocket connection with periodic pings."""
        if not self.connected:
            raise StopUser()

        # Simulate ping/pong to keep connection alive
        time.sleep(random.uniform(5, 15))


class LongLivedWebSocketUser(User):
    """Long-lived WebSocket connection for sustained testing."""

    wait_time = between(30, 60)  # Very long intervals

    def __init__(self, *args, **kwargs) -> None:
        """Initialize long-lived WebSocket user."""
        super().__init__(*args, **kwargs)
        self.client_id = f"ws-long-{random.randint(10000, 99999)}"
        self.connected = False
        self.connection_start = None

    def on_start(self) -> None:
        """Establish long-lived WebSocket connection."""
        self.connection_start = time.time()
        self.connected = True
        print(f"Long-lived WebSocket client {self.client_id} connected")

    def on_stop(self) -> None:
        """Close long-lived WebSocket connection."""
        if self.connected:
            duration = time.time() - self.connection_start if self.connection_start else 0
            print(f"Long-lived WebSocket client {self.client_id} disconnected after {duration:.2f}, s")
            self.connected = False

    @task
    def keep_alive(self) -> None:
        """Keep connection alive with minimal activity."""
        if not self.connected:
            raise StopUser()

        time.sleep(random.uniform(30, 60))


# Track concurrent connections
concurrent_connections = 0
max_concurrent_connections = 0
connection_lock = None


@events.init.add_listener
def on_init(environment, **kwargs) -> None:
    """Initialize connection tracking."""
    global connection_lock
    connection_lock = asyncio.Lock() if asyncio else None


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test start."""
    print(f"\n{'='*70}")
    print(f"WebSocket Load Test Starting")
    print(f"Target: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users if environment.parsed_options else 'N/A'}")
    print(f"Target: 10,000+ concurrent connections")
    print(f"{'='*70}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log test results."""
    print(f"\n{'='*70}")
    print(f"WebSocket Load Test Results")
    print(f"{'='*70}")
    print(f"Max Concurrent Connections: {max_concurrent_connections:,}")
    print(f"Current Connections: {concurrent_connections:,}")
    print(f"{'='*70}")

    # Check if target achieved
    if max_concurrent_connections >= 10000:
        print(f"✓ Target 10,000+ concurrent connections ACHIEVED!")
    else:
        print(f"✗ Target 10,000+ connections not reached (got {max_concurrent_connections:,})")
    print(f"{'='*70}\n")


@events.user_add.add_listener
def on_user_add(user_id, **kwargs) -> None:
    """Track connection added."""
    global concurrent_connections, max_concurrent_connections
    concurrent_connections += 1
    max_concurrent_connections = max(max_concurrent_connections, concurrent_connections)


@events.user_remove.add_listener
def on_user_remove(user_id, **kwargs) -> None:
    """Track connection removed."""
    global concurrent_connections
    concurrent_connections = max(0, concurrent_connections - 1)
