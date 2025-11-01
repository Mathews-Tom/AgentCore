"""
Performance monitoring for DSPy optimization engine

Provides baseline measurement, metrics collection, statistical significance
testing, and real-time monitoring capabilities.
"""

from agentcore.dspy_optimization.monitoring.baseline import (
    BaselineService,
    BaselineConfig,
)
from agentcore.dspy_optimization.monitoring.collector import (
    MetricsCollector,
    CollectorConfig,
    MetricSnapshot,
)
from agentcore.dspy_optimization.monitoring.statistics import (
    StatisticalTester,
    SignificanceTest,
    SignificanceResult,
    ConfidenceInterval,
)
from agentcore.dspy_optimization.monitoring.dashboard import (
    DashboardService,
    PerformanceTrend,
    OptimizationHistory,
)

__all__ = [
    "BaselineService",
    "BaselineConfig",
    "MetricsCollector",
    "CollectorConfig",
    "MetricSnapshot",
    "StatisticalTester",
    "SignificanceTest",
    "SignificanceResult",
    "ConfidenceInterval",
    "DashboardService",
    "PerformanceTrend",
    "OptimizationHistory",
]
