"""
Orchestration Patterns

Built-in orchestration patterns for multi-agent coordination.
"""

from agentcore.orchestration.patterns.swarm import (
    AgentProposal,
    AgentRole,
    AgentState,
    ConsensusStrategy,
    ProposalStatus,
    SwarmConfig,
    SwarmCoordinator,
    SwarmTask,
    Vote,
)
from agentcore.orchestration.patterns.saga import (
    CompensationStrategy,
    SagaConfig,
    SagaDefinition,
    SagaExecution,
    SagaOrchestrator,
    SagaStatus,
    SagaStep,
    SagaStepStatus,
)
from agentcore.orchestration.patterns.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FaultToleranceCoordinator,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    RetryPolicy,
    RetryStrategy,
)
from agentcore.orchestration.patterns.custom import (
    AgentRequirement,
    CoordinationConfig,
    CoordinationModel,
    PatternDefinition,
    PatternMetadata,
    PatternRegistry,
    PatternStatus,
    PatternType,
    TaskNode,
    ValidationRule,
    pattern_registry,
)

__all__ = [
    # Swarm Pattern
    "SwarmCoordinator",
    "SwarmConfig",
    "AgentState",
    "Vote",
    "AgentProposal",
    "SwarmTask",
    "ConsensusStrategy",
    "AgentRole",
    "ProposalStatus",
    # Saga Pattern
    "SagaOrchestrator",
    "SagaConfig",
    "SagaDefinition",
    "SagaExecution",
    "SagaStep",
    "SagaStatus",
    "SagaStepStatus",
    "CompensationStrategy",
    # Circuit Breaker Pattern
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "FaultToleranceCoordinator",
    "HealthCheck",
    "HealthMonitor",
    "HealthStatus",
    "RetryPolicy",
    "RetryStrategy",
    # Custom Pattern Framework
    "PatternDefinition",
    "PatternMetadata",
    "PatternRegistry",
    "PatternType",
    "PatternStatus",
    "CoordinationModel",
    "AgentRequirement",
    "TaskNode",
    "CoordinationConfig",
    "ValidationRule",
    "pattern_registry",
]
