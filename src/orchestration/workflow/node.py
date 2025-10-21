"""Workflow node abstractions and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from orchestration.workflow.models import NodeStatus, NodeType


class WorkflowNode(ABC):
    """Abstract base class for workflow nodes."""

    def __init__(self, node_id: str, node_type: NodeType) -> None:
        """Initialize a workflow node.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of the node
        """
        self.node_id = node_id
        self.node_type = node_type
        self.status = NodeStatus.PENDING
        self.dependencies: list[str] = []
        self.metadata: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the node with given context.

        Args:
            context: Execution context containing input data and state

        Returns:
            Result data from node execution

        Raises:
            Exception: If execution fails
        """

    def add_dependency(self, node_id: str) -> None:
        """Add a dependency to this node.

        Args:
            node_id: ID of the node this depends on
        """
        if node_id not in self.dependencies:
            self.dependencies.append(node_id)

    def is_ready(self, completed_nodes: set[str]) -> bool:
        """Check if node is ready to execute.

        Args:
            completed_nodes: Set of node IDs that have completed

        Returns:
            True if all dependencies are met
        """
        return all(dep in completed_nodes for dep in self.dependencies)


class TaskNode(WorkflowNode):
    """Node representing a task to be executed by an agent."""

    def __init__(
        self,
        node_id: str,
        agent_role: str,
        task_definition: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a task node.

        Args:
            node_id: Unique identifier for the node
            agent_role: Role of the agent to execute this task
            task_definition: Definition of the task to execute
        """
        super().__init__(node_id, NodeType.TASK)
        self.agent_role = agent_role
        self.task_definition = task_definition or {}

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the task.

        Args:
            context: Execution context

        Returns:
            Task execution result
        """
        # Task execution will be delegated to agent runtime in production
        # For now, return a mock result
        return {
            "node_id": self.node_id,
            "agent_role": self.agent_role,
            "status": "completed",
            "result": context.get("input", {}),
        }


class DecisionNode(WorkflowNode):
    """Node representing a conditional branch in the workflow."""

    def __init__(
        self,
        node_id: str,
        condition: str,
        branches: dict[str, str] | None = None,
    ) -> None:
        """Initialize a decision node.

        Args:
            node_id: Unique identifier for the node
            condition: Python expression to evaluate
            branches: Mapping of condition results to next node IDs
        """
        super().__init__(node_id, NodeType.DECISION)
        self.condition = condition
        self.branches = branches or {}

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate condition and determine next path.

        Args:
            context: Execution context

        Returns:
            Decision result with next node to execute
        """
        # Evaluate condition safely
        # In production, use a safe expression evaluator
        try:
            result = eval(self.condition, {"__builtins__": {}}, context)
            next_node = self.branches.get(str(result))
            return {
                "node_id": self.node_id,
                "condition_result": result,
                "next_node": next_node,
            }
        except Exception as e:
            return {
                "node_id": self.node_id,
                "error": str(e),
                "next_node": None,
            }


class ParallelNode(WorkflowNode):
    """Node representing parallel execution of multiple branches."""

    def __init__(
        self,
        node_id: str,
        parallel_branches: list[str] | None = None,
    ) -> None:
        """Initialize a parallel node.

        Args:
            node_id: Unique identifier for the node
            parallel_branches: List of node IDs to execute in parallel
        """
        super().__init__(node_id, NodeType.PARALLEL)
        self.parallel_branches = parallel_branches or []

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Initialize parallel execution.

        Args:
            context: Execution context

        Returns:
            Parallel execution configuration
        """
        return {
            "node_id": self.node_id,
            "parallel_branches": self.parallel_branches,
            "status": "initialized",
        }


class JoinNode(WorkflowNode):
    """Node representing the join point of parallel branches."""

    def __init__(
        self,
        node_id: str,
        join_branches: list[str] | None = None,
    ) -> None:
        """Initialize a join node.

        Args:
            node_id: Unique identifier for the node
            join_branches: List of node IDs to wait for
        """
        super().__init__(node_id, NodeType.JOIN)
        self.join_branches = join_branches or []

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Join parallel execution results.

        Args:
            context: Execution context

        Returns:
            Aggregated results from parallel branches
        """
        # Collect results from all parallel branches
        branch_results = {}
        for branch_id in self.join_branches:
            if branch_id in context.get("results", {}):
                branch_results[branch_id] = context["results"][branch_id]

        return {
            "node_id": self.node_id,
            "branch_results": branch_results,
            "status": "joined",
        }
