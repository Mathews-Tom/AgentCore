"""
Performance optimization and benchmarking for orchestration engine.

This module provides:
- Benchmarking suite for workflow planning and event processing
- Performance profiling utilities
- Graph planning optimizations
- Event processing throughput improvements
"""

from .benchmarks import OrchestrationBenchmarks
from .graph_optimizer import GraphOptimizer

__all__ = ["OrchestrationBenchmarks", "GraphOptimizer"]
