"""
Pytest configuration for DSPy performance tests
"""

import pytest


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test (>10s)"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip all performance tests if dspy is not installed"""
    try:
        import dspy
    except ImportError:
        skip_dspy = pytest.mark.skip(reason="dspy not installed - skipping performance tests")
        for item in items:
            item.add_marker(skip_dspy)
