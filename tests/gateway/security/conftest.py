"""Security test-specific pytest configuration.

Creates test fixtures with security features enabled to test production-like behavior.
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY


def pytest_configure(config):
    """Configure security tests to run with validation enabled."""
    # Enable input validation for security tests BEFORE any imports
    # This must happen before gateway modules are imported
    os.environ["GATEWAY_VALIDATION_ENABLED"] = "true"

    # Clear any cached gateway modules to force reimport with new settings
    modules_to_clear = [
        key for key in sys.modules.keys()
        if key.startswith("gateway.")
    ]
    for module in modules_to_clear:
        del sys.modules[module]

    # Clear Prometheus registry to avoid metric duplication
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def secure_client(redis_container):
    """
    Create test client with security features enabled.

    This fixture creates a FastAPI app with production-like security configuration:
    - Input validation ENABLED
    - Security headers ENABLED
    - Rate limiting ENABLED (uses redis_container)

    Use this fixture for security integration tests that verify security
    features work correctly.
    """
    # Import here to ensure settings are loaded with VALIDATION_ENABLED=true
    from gateway.main import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def client(secure_client):
    """
    Override default client fixture to use secure_client for security tests.

    This ensures all security tests use the secure configuration by default.
    """
    return secure_client
