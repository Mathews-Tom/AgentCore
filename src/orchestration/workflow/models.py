"""Workflow data models for graph-based orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Type of workflow node."""

    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    JOIN = "join"


class EdgeType(str, Enum):
    """Type of workflow edge."""

    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"


class NodeStatus(str, Enum):
    """Execution status of a workflow node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RetryPolicy(BaseModel):
    """Retry policy for task execution."""

    max_attempts: int = Field(default=3, ge=1)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    initial_delay_seconds: float = Field(default=1.0, ge=0.1)
    max_delay_seconds: float = Field(default=60.0, ge=1.0)


class TaskNode(BaseModel):
    """Definition of a task node in the workflow."""

    node_id: str = Field(description="Unique identifier for the node")
    node_type: NodeType = Field(default=NodeType.TASK)
    agent_role: str = Field(description="Role of the agent to execute this task")
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of nodes this task depends on"
    )
    timeout_seconds: int = Field(default=300, ge=1)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    compensation_action: str | None = Field(
        default=None, description="Action to execute if workflow fails after this task"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConditionalEdge(BaseModel):
    """Conditional edge between nodes."""

    from_node: str
    to_node: str
    condition: str = Field(description="Python expression to evaluate")
    edge_type: EdgeType = EdgeType.CONDITIONAL


class CoordinationConfig(BaseModel):
    """Configuration for workflow coordination."""

    coordination_type: str = Field(
        default="hybrid", description="Type of coordination: event_driven, graph_based, or hybrid"
    )
    event_driven_events: list[str] = Field(
        default_factory=list, description="Events to handle in event-driven mode"
    )
    max_parallel_tasks: int = Field(
        default=10, ge=1, description="Maximum number of parallel tasks"
    )


class WorkflowDefinition(BaseModel):
    """Definition of a workflow graph."""

    workflow_id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Human-readable workflow name")
    version: str = Field(default="1.0.0")
    description: str | None = None

    nodes: list[TaskNode] = Field(
        default_factory=list, description="Nodes in the workflow graph"
    )
    conditional_edges: list[ConditionalEdge] = Field(
        default_factory=list, description="Conditional edges between nodes"
    )
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"arbitrary_types_allowed": True}


class NodeExecutionState(BaseModel):
    """Execution state of a single node."""

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_count: int = 0
    error_message: str | None = None
    result: dict[str, Any] | None = None


class WorkflowExecution(BaseModel):
    """Runtime execution state of a workflow."""

    execution_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    status: WorkflowStatus = WorkflowStatus.PLANNING

    node_states: dict[str, NodeExecutionState] = Field(
        default_factory=dict, description="Execution state for each node"
    )
    allocated_agents: dict[str, str] = Field(
        default_factory=dict, description="Mapping of node_id to agent_id"
    )

    current_phase: str = Field(default="initialization")
    coordination_overhead_ms: float = Field(default=0.0)

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    paused_at: datetime | None = None

    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
