"""Workflow execution engine."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from orchestration.workflow.graph import WorkflowGraph
from orchestration.workflow.models import (
    NodeStatus,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
)
from orchestration.workflow.node import WorkflowNode
from orchestration.workflow.state import WorkflowStateManager


class WorkflowExecutionError(Exception):
    """Base exception for workflow execution errors."""

    pass


class WorkflowExecutor:
    """Executes workflows based on graph definitions."""

    def __init__(
        self,
        state_manager: WorkflowStateManager | None = None,
    ) -> None:
        """Initialize the workflow executor.

        Args:
            state_manager: State manager for tracking execution state
        """
        self.state_manager = state_manager or WorkflowStateManager()
        self._node_factories: dict[str, type[WorkflowNode]] = {}

    def register_node_factory(
        self,
        node_type: str,
        factory: type[WorkflowNode],
    ) -> None:
        """Register a node factory for creating node instances.

        Args:
            node_type: Type identifier for the node
            factory: Factory class for creating node instances
        """
        self._node_factories[node_type] = factory

    async def execute(
        self,
        definition: WorkflowDefinition,
        input_data: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Execute a workflow from definition.

        Args:
            definition: Workflow definition to execute
            input_data: Input data for the workflow

        Returns:
            Completed workflow execution state

        Raises:
            WorkflowExecutionError: If execution fails
        """
        # Create workflow graph
        graph = WorkflowGraph.from_definition(definition)

        # Create execution state
        execution = self.state_manager.create_execution(
            workflow_id=definition.workflow_id,
            input_data=input_data,
        )

        # Initialize node states
        node_ids = [node.node_id for node in definition.nodes]
        self.state_manager.initialize_node_states(
            execution.execution_id, node_ids
        )

        # Update status to executing
        self.state_manager.update_execution_status(
            execution.execution_id, WorkflowStatus.EXECUTING
        )
        execution.current_phase = "execution"

        try:
            # Execute workflow
            await self._execute_graph(execution.execution_id, graph, definition)

            # Mark as completed
            self.state_manager.update_execution_status(
                execution.execution_id, WorkflowStatus.COMPLETED
            )

        except Exception as e:
            # Mark as failed
            self.state_manager.update_execution_status(
                execution.execution_id, WorkflowStatus.FAILED
            )
            raise WorkflowExecutionError(
                f"Workflow execution failed: {e}"
            ) from e

        return self.state_manager.get_execution(execution.execution_id) or execution

    async def _execute_graph(
        self,
        execution_id: UUID,
        graph: WorkflowGraph,
        definition: WorkflowDefinition,
    ) -> None:
        """Execute workflow graph.

        Args:
            execution_id: ID of the execution
            graph: Workflow graph to execute
            definition: Workflow definition
        """
        completed_nodes: set[str] = set()
        context: dict[str, Any] = {"results": {}}

        # Get execution order
        if definition.coordination.coordination_type == "graph_based":
            # Execute in topological order
            execution_order = graph.topological_sort()
            for node_id in execution_order:
                await self._execute_node(
                    execution_id, node_id, graph, context
                )
                completed_nodes.add(node_id)

        elif definition.coordination.coordination_type == "hybrid":
            # Execute with parallel groups
            parallel_groups = graph.get_parallel_execution_groups()
            for group in parallel_groups:
                # Execute nodes in group in parallel
                tasks = [
                    self._execute_node(execution_id, node_id, graph, context)
                    for node_id in group
                ]
                await asyncio.gather(*tasks)
                completed_nodes.update(group)

        else:
            # Event-driven execution - not implemented yet
            # For now, fall back to sequential execution
            execution_order = graph.topological_sort()
            for node_id in execution_order:
                await self._execute_node(
                    execution_id, node_id, graph, context
                )
                completed_nodes.add(node_id)

    async def _execute_node(
        self,
        execution_id: UUID,
        node_id: str,
        graph: WorkflowGraph,
        context: dict[str, Any],
    ) -> None:
        """Execute a single node with retry logic.

        Args:
            execution_id: ID of the execution
            node_id: ID of the node to execute
            graph: Workflow graph
            context: Execution context
        """
        start_time = datetime.now(UTC)
        node_def = graph.get_node(node_id)

        # Update status to running
        self.state_manager.update_node_status(
            execution_id, node_id, NodeStatus.RUNNING
        )

        max_attempts = node_def.retry_policy.max_attempts
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            self.state_manager.increment_node_attempts(execution_id, node_id)

            try:
                # Execute node
                # In production, this would delegate to the agent runtime
                # For now, we'll simulate execution
                result = await self._simulate_node_execution(node_def, context)

                # Store result in context
                context["results"][node_id] = result

                # Update status to completed
                self.state_manager.update_node_status(
                    execution_id,
                    node_id,
                    NodeStatus.COMPLETED,
                    result=result,
                )

                # Track coordination overhead
                end_time = datetime.now(UTC)
                overhead_ms = (end_time - start_time).total_seconds() * 1000
                self.state_manager.update_coordination_overhead(
                    execution_id, overhead_ms
                )

                return

            except Exception as e:
                if attempt >= max_attempts:
                    # Max attempts reached, mark as failed
                    self.state_manager.update_node_status(
                        execution_id,
                        node_id,
                        NodeStatus.FAILED,
                        error_message=str(e),
                    )
                    raise

                # Calculate backoff delay
                delay = (
                    node_def.retry_policy.initial_delay_seconds
                    * (node_def.retry_policy.backoff_multiplier ** (attempt - 1))
                )
                delay = min(delay, node_def.retry_policy.max_delay_seconds)

                # Wait before retry
                await asyncio.sleep(delay)

    async def _simulate_node_execution(
        self,
        node_def: Any,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Simulate node execution.

        Args:
            node_def: Node definition
            context: Execution context

        Returns:
            Simulated execution result
        """
        # Simulate some work
        await asyncio.sleep(0.01)

        return {
            "node_id": node_def.node_id,
            "agent_role": node_def.agent_role,
            "status": "completed",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def pause(self, execution_id: UUID) -> None:
        """Pause a workflow execution.

        Args:
            execution_id: ID of the execution to pause
        """
        self.state_manager.update_execution_status(
            execution_id, WorkflowStatus.PAUSED
        )

    async def resume(self, execution_id: UUID) -> WorkflowExecution:
        """Resume a paused workflow execution.

        Args:
            execution_id: ID of the execution to resume

        Returns:
            Resumed workflow execution

        Raises:
            WorkflowExecutionError: If execution cannot be resumed
        """
        execution = self.state_manager.get_execution(execution_id)
        if not execution:
            raise WorkflowExecutionError(
                f"Execution {execution_id} not found"
            )

        if execution.status != WorkflowStatus.PAUSED:
            raise WorkflowExecutionError(
                f"Execution {execution_id} is not paused"
            )

        # Update status to executing
        self.state_manager.update_execution_status(
            execution_id, WorkflowStatus.EXECUTING
        )

        # Continue execution from where it left off
        # This is simplified - in production would need to reconstruct
        # the workflow graph and continue from incomplete nodes

        return execution

    def get_execution_status(
        self, execution_id: UUID
    ) -> WorkflowExecution | None:
        """Get the current status of a workflow execution.

        Args:
            execution_id: ID of the execution

        Returns:
            Current execution state, or None if not found
        """
        return self.state_manager.get_execution(execution_id)
