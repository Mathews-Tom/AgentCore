"""Workflow builder DSL for programmatic workflow definition."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from orchestration.workflow.models import (
    ConditionalEdge,
    CoordinationConfig,
    EdgeType,
    RetryPolicy,
    TaskNode,
    WorkflowDefinition,
)


class WorkflowBuilder:
    """Builder for creating workflows programmatically."""

    def __init__(self, name: str, version: str = "1.0.0") -> None:
        """Initialize a workflow builder.

        Args:
            name: Name of the workflow
            version: Version of the workflow
        """
        self._workflow_id: UUID = uuid4()
        self._name = name
        self._version = version
        self._description: str | None = None
        self._nodes: list[TaskNode] = []
        self._conditional_edges: list[ConditionalEdge] = []
        self._coordination = CoordinationConfig()
        self._current_node: TaskNode | None = None

    def description(self, description: str) -> WorkflowBuilder:
        """Set workflow description.

        Args:
            description: Workflow description

        Returns:
            Self for chaining
        """
        self._description = description
        return self

    def workflow_id(self, workflow_id: UUID) -> WorkflowBuilder:
        """Set workflow ID.

        Args:
            workflow_id: Workflow ID

        Returns:
            Self for chaining
        """
        self._workflow_id = workflow_id
        return self

    def add_task(
        self,
        node_id: str,
        agent_role: str,
        timeout_seconds: int = 300,
        retry_policy: RetryPolicy | None = None,
    ) -> WorkflowBuilder:
        """Add a task node to the workflow.

        Args:
            node_id: Unique identifier for the node
            agent_role: Role of the agent to execute the task
            timeout_seconds: Timeout in seconds
            retry_policy: Retry policy for the task

        Returns:
            Self for chaining
        """
        node = TaskNode(
            node_id=node_id,
            agent_role=agent_role,
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy or RetryPolicy(),
        )
        self._nodes.append(node)
        self._current_node = node
        return self

    def depends_on(self, *node_ids: str) -> WorkflowBuilder:
        """Set dependencies for the current node.

        Args:
            *node_ids: IDs of nodes the current node depends on

        Returns:
            Self for chaining

        Raises:
            ValueError: If no current node is set
        """
        if not self._current_node:
            raise ValueError("No current node to add dependencies to")

        self._current_node.depends_on.extend(node_ids)
        return self

    def with_compensation(self, compensation_action: str) -> WorkflowBuilder:
        """Set compensation action for the current node.

        Args:
            compensation_action: Action to execute if workflow fails

        Returns:
            Self for chaining

        Raises:
            ValueError: If no current node is set
        """
        if not self._current_node:
            raise ValueError("No current node to add compensation to")

        self._current_node.compensation_action = compensation_action
        return self

    def with_metadata(self, **metadata: Any) -> WorkflowBuilder:
        """Add metadata to the current node.

        Args:
            **metadata: Metadata key-value pairs

        Returns:
            Self for chaining

        Raises:
            ValueError: If no current node is set
        """
        if not self._current_node:
            raise ValueError("No current node to add metadata to")

        self._current_node.metadata.update(metadata)
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        to_node: str,
        condition: str,
    ) -> WorkflowBuilder:
        """Add a conditional edge between nodes.

        Args:
            from_node: Source node ID
            to_node: Destination node ID
            condition: Python expression for the condition

        Returns:
            Self for chaining
        """
        edge = ConditionalEdge(
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            edge_type=EdgeType.CONDITIONAL,
        )
        self._conditional_edges.append(edge)
        return self

    def set_coordination(
        self,
        coordination_type: str = "hybrid",
        event_driven_events: list[str] | None = None,
        max_parallel_tasks: int = 10,
    ) -> WorkflowBuilder:
        """Set coordination configuration.

        Args:
            coordination_type: Type of coordination
            event_driven_events: Events for event-driven coordination
            max_parallel_tasks: Maximum parallel tasks

        Returns:
            Self for chaining
        """
        self._coordination = CoordinationConfig(
            coordination_type=coordination_type,
            event_driven_events=event_driven_events or [],
            max_parallel_tasks=max_parallel_tasks,
        )
        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition.

        Returns:
            Complete workflow definition

        Raises:
            ValueError: If workflow is invalid
        """
        if not self._nodes:
            raise ValueError("Workflow must have at least one node")

        # Validate node IDs are unique
        node_ids = [node.node_id for node in self._nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")

        # Validate dependencies reference existing nodes
        all_node_ids = set(node_ids)
        for node in self._nodes:
            for dep in node.depends_on:
                if dep not in all_node_ids:
                    raise ValueError(
                        f"Node {node.node_id} depends on non-existent node {dep}"
                    )

        # Validate conditional edges reference existing nodes
        for edge in self._conditional_edges:
            if edge.from_node not in all_node_ids:
                raise ValueError(
                    f"Conditional edge references non-existent source node {edge.from_node}"
                )
            if edge.to_node not in all_node_ids:
                raise ValueError(
                    f"Conditional edge references non-existent destination node {edge.to_node}"
                )

        return WorkflowDefinition(
            workflow_id=self._workflow_id,
            name=self._name,
            version=self._version,
            description=self._description,
            nodes=self._nodes,
            conditional_edges=self._conditional_edges,
            coordination=self._coordination,
        )

    @staticmethod
    def from_yaml(yaml_str: str) -> WorkflowDefinition:
        """Create workflow definition from YAML string.

        Args:
            yaml_str: YAML workflow definition

        Returns:
            WorkflowDefinition instance

        Raises:
            ValueError: If YAML is invalid
        """
        import yaml

        try:
            data = yaml.safe_load(yaml_str)
            return WorkflowBuilder._from_dict(data)
        except Exception as e:
            raise ValueError(f"Invalid YAML workflow definition: {e}") from e

    @staticmethod
    def from_json(json_str: str) -> WorkflowDefinition:
        """Create workflow definition from JSON string.

        Args:
            json_str: JSON workflow definition

        Returns:
            WorkflowDefinition instance

        Raises:
            ValueError: If JSON is invalid
        """
        import json

        try:
            data = json.loads(json_str)
            return WorkflowBuilder._from_dict(data)
        except Exception as e:
            raise ValueError(f"Invalid JSON workflow definition: {e}") from e

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> WorkflowDefinition:
        """Create workflow definition from dictionary.

        Args:
            data: Dictionary representation of workflow

        Returns:
            WorkflowDefinition instance
        """
        # Parse nodes
        nodes = []
        for node_data in data.get("nodes", []):
            retry_policy_data = node_data.get("retry_policy", {})
            retry_policy = (
                RetryPolicy(**retry_policy_data) if retry_policy_data else RetryPolicy()
            )

            node = TaskNode(
                node_id=node_data["node_id"],
                agent_role=node_data["agent_role"],
                depends_on=node_data.get("depends_on", []),
                timeout_seconds=node_data.get("timeout_seconds", 300),
                retry_policy=retry_policy,
                compensation_action=node_data.get("compensation_action"),
                metadata=node_data.get("metadata", {}),
            )
            nodes.append(node)

        # Parse conditional edges
        conditional_edges = []
        for edge_data in data.get("conditional_edges", []):
            edge = ConditionalEdge(
                from_node=edge_data["from_node"],
                to_node=edge_data["to_node"],
                condition=edge_data["condition"],
            )
            conditional_edges.append(edge)

        # Parse coordination config
        coord_data = data.get("coordination", {})
        coordination = CoordinationConfig(
            coordination_type=coord_data.get("coordination_type", "hybrid"),
            event_driven_events=coord_data.get("event_driven_events", []),
            max_parallel_tasks=coord_data.get("max_parallel_tasks", 10),
        )

        return WorkflowDefinition(
            workflow_id=UUID(data.get("workflow_id", str(uuid4()))),
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description"),
            nodes=nodes,
            conditional_edges=conditional_edges,
            coordination=coordination,
        )
