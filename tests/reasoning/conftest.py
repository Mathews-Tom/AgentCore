"""Shared pytest fixtures for reasoning tests."""

from __future__ import annotations

import sys

import pytest
from prometheus_client import REGISTRY


def _clear_metrics_and_registry():
    """Clear metrics cache and Prometheus registry."""
    # Clear metrics module cache if it's been imported
    if "agentcore.reasoning.services.metrics" in sys.modules:
        from agentcore.reasoning.services import metrics

        metrics._metrics_cache.clear()

    # Clear all collectors from registry
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def pytest_collection_modifyitems(config, items):
    """Hook to clear metrics before test collection completes."""
    _clear_metrics_and_registry()


# Run at the very beginning of test session before any imports
@pytest.fixture(scope="session", autouse=True)
def clear_prometheus_registry_session():
    """Clear Prometheus registry at session start to avoid duplicate metric registration."""
    _clear_metrics_and_registry()
    yield
    _clear_metrics_and_registry()
