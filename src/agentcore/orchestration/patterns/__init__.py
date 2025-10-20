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
]
