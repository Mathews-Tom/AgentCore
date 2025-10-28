"""
DSPy Optimization Engine

Provides systematic AI optimization using MIPROv2 and GEPA algorithms
for agent performance improvement.
"""

from __future__ import annotations

from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer
from agentcore.dspy_optimization.algorithms.gepa import GEPAOptimizer
from agentcore.dspy_optimization.pipeline import OptimizationPipeline
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationTarget,
    OptimizationObjective,
    OptimizationConstraints,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.tracking import MLflowTracker, MLflowConfig
from agentcore.dspy_optimization.security import (
    ModelEncryption,
    EncryptionConfig,
    PrivacyManager,
    PrivacyConfig,
    AccessController,
    AccessConfig,
    SecurityRole,
    SecurityPermission,
    AuditLogger,
    AuditConfig,
    AuditEventType,
    ComplianceValidator,
    ComplianceConfig,
    ComplianceStandard,
)

__all__ = [
    "MIPROv2Optimizer",
    "GEPAOptimizer",
    "OptimizationPipeline",
    "OptimizationRequest",
    "OptimizationResult",
    "OptimizationTarget",
    "OptimizationObjective",
    "OptimizationConstraints",
    "PerformanceMetrics",
    "MLflowTracker",
    "MLflowConfig",
    "ModelEncryption",
    "EncryptionConfig",
    "PrivacyManager",
    "PrivacyConfig",
    "AccessController",
    "AccessConfig",
    "SecurityRole",
    "SecurityPermission",
    "AuditLogger",
    "AuditConfig",
    "AuditEventType",
    "ComplianceValidator",
    "ComplianceConfig",
    "ComplianceStandard",
]
