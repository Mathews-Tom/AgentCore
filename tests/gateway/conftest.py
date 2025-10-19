"""Gateway-specific pytest configuration."""

from __future__ import annotations

import os
import sys

import pytest
from prometheus_client import REGISTRY

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
