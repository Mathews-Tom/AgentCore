"""
Orchestration Patterns

Built-in orchestration patterns for multi-agent coordination.
"""

from agentcore.orchestration.patterns.handoff import (
    CapabilityGate,
    HandoffConfig,
    HandoffContext,
    HandoffCoordinator,
    HandoffGate,
    HandoffRecord,
    HandoffStatus,
    InputValidationGate,
    OutputValidationGate,
    ValidationResult,
)
from agentcore.orchestration.patterns.hierarchical import (
    AuthorityLevel,
    DelegationPolicy,
    EscalationReason,
    HierarchicalConfig,
    HierarchicalCoordinator,
    HierarchyNode,
    TaskDelegation,
    TaskEscalation,
)
from agentcore.orchestration.patterns.supervisor import (
    SupervisorCoordinator,
    SupervisorConfig,
    WorkerState,
    WorkerStatus,
)
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

__all__ = [
    # Supervisor Pattern
    "SupervisorCoordinator",
    "SupervisorConfig",
    "WorkerState",
    "WorkerStatus",
    # Hierarchical Pattern
    "HierarchicalCoordinator",
    "HierarchicalConfig",
    "HierarchyNode",
    "TaskDelegation",
    "TaskEscalation",
    "AuthorityLevel",
    "EscalationReason",
    "DelegationPolicy",
    # Handoff Pattern
    "HandoffCoordinator",
    "HandoffConfig",
    "HandoffContext",
    "HandoffRecord",
    "HandoffStatus",
    "HandoffGate",
    "InputValidationGate",
    "OutputValidationGate",
    "CapabilityGate",
    "ValidationResult",
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
]
