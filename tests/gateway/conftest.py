"""Gateway-specific pytest configuration."""

from __future__ import annotations

import os
import sys
import time

import pytest
from prometheus_client import REGISTRY
from testcontainers.redis import RedisContainer

# Disable input validation middleware for tests BEFORE any imports
# This must happen at module level to ensure settings load with correct values
# Note: env_prefix is "GATEWAY_" so we need GATEWAY_VALIDATION_ENABLED
os.environ["GATEWAY_VALIDATION_ENABLED"] = "false"


def pytest_configure(config):
    """Clear Prometheus registry and metrics cache before collection."""
    # Clear all collectors from the registry
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    # Clear metrics cache from gateway monitoring module if imported
    if "gateway.monitoring.metrics" in sys.modules:
        import gateway.monitoring.metrics as metrics_module

        metrics_module._metrics_cache.clear()
        # Force module reload
        del sys.modules["gateway.monitoring.metrics"]
        if "gateway.monitoring" in sys.modules:
            del sys.modules["gateway.monitoring"]


@pytest.fixture(scope="session")
def redis_container():
    """
    Session-scoped Redis container shared across all gateway tests.

    This ensures a single Redis instance is used for the entire test session,
    avoiding port conflicts and connection issues.
    """
    container = RedisContainer("redis:7-alpine")

    try:
        container.start()

        # Wait for Redis to be ready with retry logic
        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            try:
                port = container.get_exposed_port(6379)
                # Set Redis URL environment variables BEFORE any gateway imports
                # Use database 0 for session-level container
                redis_url = f"redis://localhost:{port}/0"
                os.environ["GATEWAY_RATE_LIMIT_REDIS_URL"] = redis_url
                os.environ["GATEWAY_SESSION_REDIS_URL"] = redis_url

                # Test connection
                import redis
                client = redis.from_url(redis_url)
                client.ping()
                client.close()

                break
            except Exception:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                time.sleep(0.5)

        yield container

    finally:
        try:
            container.stop()
        except Exception:
            pass  # Ignore errors during cleanup
