"""
Performance Optimization

Connection pooling, caching, and performance tuning for high-throughput scenarios.
"""

from __future__ import annotations

from .connection_pool import ConnectionPool, HTTPConnectionPool
from .response_cache import ResponseCache, CachePolicy

__all__ = [
    "ConnectionPool",
    "HTTPConnectionPool",
    "ResponseCache",
    "CachePolicy",
]
