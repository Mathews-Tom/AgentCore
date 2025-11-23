"""
Modular Agent Core

A modular architecture for complex multi-step agent workflows using the
four-module pattern: Planner, Executor, Verifier, and Generator (PEVG).

This package provides Protocol-based interfaces for implementing modular
agent systems with clean separation of concerns.
"""

from agentcore.modular.interfaces import (
    # Planner Models
    PlannerQuery,
    PlanStep,
    ExecutionPlan,
    PlanRefinement,
    # Executor Models
    ExecutionContext,
    ExecutionResult,
    RetryPolicy,
    # Verifier Models
    VerificationRequest,
    VerificationResult,
    ConsistencyCheck,
    # Generator Models
    GenerationRequest,
    GeneratedResponse,
    OutputFormat,
    # Protocol Interfaces
    PlannerInterface,
    ExecutorInterface,
    VerifierInterface,
    GeneratorInterface,
)

from agentcore.modular.base import (
    # Base Classes
    BaseModule,
    BasePlanner,
    BaseExecutor,
    BaseVerifier,
    BaseGenerator,
    # State Management
    ModuleState,
)

from agentcore.modular.models import (
    # Enums
    PlanStatus,
    StepStatus,
    ModuleType,
    VerificationLevel,
    # Enhanced Models
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    EnhancedVerificationResult,
    ModuleTransition,
    # Supporting Models
    ToolRequirement,
    StepDependency,
    SuccessCriterion,
    SuccessCriteria,
)

from agentcore.modular.coordinator import (
    # Coordinator
    ModuleCoordinator,
    # Coordination Models
    ModuleCapability,
    ModuleMessage,
    CoordinationContext,
    RefinementIteration,
)

from agentcore.modular.state_manager import (
    # State Manager
    StateManager,
    # State Models
    ExecutionCheckpoint,
    RecoveryInfo,
)

from agentcore.modular.metrics import (
    # Metrics Collector
    ModularMetricsCollector,
    # Trackers
    ModuleExecutionTracker,
    CoordinationExecutionTracker,
    # Error Types
    ErrorType,
    # Global Instance
    get_metrics,
    set_metrics,
)

__all__ = [
    # Planner
    "PlannerQuery",
    "PlanStep",
    "ExecutionPlan",
    "PlanRefinement",
    "PlannerInterface",
    "BasePlanner",
    # Executor
    "ExecutionContext",
    "ExecutionResult",
    "RetryPolicy",
    "ExecutorInterface",
    "BaseExecutor",
    # Verifier
    "VerificationRequest",
    "VerificationResult",
    "ConsistencyCheck",
    "VerifierInterface",
    "BaseVerifier",
    # Generator
    "GenerationRequest",
    "GeneratedResponse",
    "OutputFormat",
    "GeneratorInterface",
    "BaseGenerator",
    # Common
    "BaseModule",
    "ModuleState",
    # Enums
    "PlanStatus",
    "StepStatus",
    "ModuleType",
    "VerificationLevel",
    # Enhanced Models
    "EnhancedExecutionPlan",
    "EnhancedPlanStep",
    "EnhancedVerificationResult",
    "ModuleTransition",
    "ToolRequirement",
    "StepDependency",
    "SuccessCriterion",
    "SuccessCriteria",
    # Coordinator
    "ModuleCoordinator",
    "ModuleCapability",
    "ModuleMessage",
    "CoordinationContext",
    "RefinementIteration",
    # State Manager
    "StateManager",
    "ExecutionCheckpoint",
    "RecoveryInfo",
    # Metrics
    "ModularMetricsCollector",
    "ModuleExecutionTracker",
    "CoordinationExecutionTracker",
    "ErrorType",
    "get_metrics",
    "set_metrics",
]
