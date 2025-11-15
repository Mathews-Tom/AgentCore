"""Pytest configuration for memory integration tests."""

# Import fixtures from the fixtures module
from tests.integration.fixtures.qdrant import (
    qdrant_client,
    qdrant_container,
    qdrant_sample_points,
    qdrant_test_collection,
    qdrant_url,
)

# Re-export fixtures for pytest discovery
__all__ = [
    "qdrant_container",
    "qdrant_url",
    "qdrant_client",
    "qdrant_test_collection",
    "qdrant_sample_points",
]
