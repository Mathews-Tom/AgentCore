"""
Performance analytics for DSPy optimization

Provides comprehensive analytics including improvement validation,
trend analysis, pattern recognition, and ROI calculation.
"""

from agentcore.dspy_optimization.analytics.improvement import (
    ImprovementAnalyzer,
    ImprovementValidation,
    ImprovementValidationConfig,
)
from agentcore.dspy_optimization.analytics.patterns import (
    OptimizationPattern,
    PatternRecognizer,
    PatternType,
)
from agentcore.dspy_optimization.analytics.roi import (
    ROICalculator,
    ROIMetrics,
    ROIReport,
)
from agentcore.dspy_optimization.analytics.trends import (
    ForecastConfig,
    TrendAnalyzer,
    TrendForecast,
    TrendResult,
)

__all__ = [
    "ImprovementAnalyzer",
    "ImprovementValidation",
    "ImprovementValidationConfig",
    "OptimizationPattern",
    "PatternRecognizer",
    "PatternType",
    "ROICalculator",
    "ROIMetrics",
    "ROIReport",
    "ForecastConfig",
    "TrendAnalyzer",
    "TrendForecast",
    "TrendResult",
]
