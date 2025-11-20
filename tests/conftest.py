"""Root-level pytest configuration for all tests."""

from __future__ import annotations

import sys

from prometheus_client import REGISTRY

# Configure pytest plugins at top level
pytest_plugins = ('pytest_asyncio')

# Global metrics cache that persists across module reloads
_GLOBAL_GATEWAY_METRICS_CACHE: dict = {}

# Inject global cache IMMEDIATELY at module level (before pytest_configure)
# This ensures it's available for ALL imports during collection
sys.modules["__gateway_metrics_cache__"] = type(sys)("__gateway_metrics_cache__")
sys.modules["__gateway_metrics_cache__"].cache = _GLOBAL_GATEWAY_METRICS_CACHE


def pytest_configure(config):
    """Pytest configuration hook - runs before any imports or collection."""
    # Only clear registry if gateway metrics haven't been loaded yet
    # If they have, leave registry alone so cached metrics remain valid
    if "gateway.monitoring.metrics" not in sys.modules:
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

    # Do NOT delete gateway modules - let them be imported once and cached
    # The global metrics cache will handle multiple imports gracefully


def pytest_runtest_setup(item):
    """Reset security service and tracing before each test to prevent state pollution."""
    import sys

    if "agentcore.a2a_protocol.services.security_service" in sys.modules:
        from agentcore.a2a_protocol.services.security_service import SecurityService

        # Get the module and reset the singleton
        security_module = sys.modules["agentcore.a2a_protocol.services.security_service"]
        # Create fresh instance
        security_module.security_service = SecurityService()

    # Reset OpenTelemetry tracer provider for tests marked with 'tracing'
    # This prevents ProxyTracerProvider issues when running tests in parallel
    if "tracing" in [mark.name for mark in item.iter_markers()]:
        from opentelemetry import trace

        # Get current provider and check if it's a ProxyTracerProvider
        provider = trace.get_tracer_provider()
        # ProxyTracerProvider is the default when no provider is set
        # We need to ensure tests have a real provider if they need tracing
        if type(provider).__name__ == "ProxyTracerProvider":
            # Let the test's own setup_tracing_for_module fixture handle it
            pass
