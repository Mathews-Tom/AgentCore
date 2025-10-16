"""Root-level pytest configuration for all tests."""

from __future__ import annotations

import sys

from prometheus_client import REGISTRY

# Configure pytest plugins at top level
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Pytest configuration hook - runs before any imports or collection."""
    # Clear Prometheus registry before any test modules are imported
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    # Clear metrics cache if already imported
    if "agentcore.reasoning.services.metrics" in sys.modules:
        from agentcore.reasoning.services import metrics

        metrics._metrics_cache.clear()
