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

__all__ = [
    # Planner
    "PlannerQuery",
    "PlanStep",
    "ExecutionPlan",
    "PlanRefinement",
    "PlannerInterface",
    # Executor
    "ExecutionContext",
    "ExecutionResult",
    "RetryPolicy",
    "ExecutorInterface",
    # Verifier
    "VerificationRequest",
    "VerificationResult",
    "ConsistencyCheck",
    "VerifierInterface",
    # Generator
    "GenerationRequest",
    "GeneratedResponse",
    "OutputFormat",
    "GeneratorInterface",
]
