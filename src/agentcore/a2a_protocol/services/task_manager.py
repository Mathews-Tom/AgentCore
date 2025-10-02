"""
Task Manager Service

Core service for managing task lifecycle, assignment, and dependency resolution.
Implements A2A protocol task management with state transitions and agent coordination.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog
from collections import defaultdict, deque

from agentcore.a2a_protocol.models.task import (
    TaskDefinition,
    TaskExecution,
    TaskStatus,
    TaskPriority,
    TaskQuery,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskQueryResponse,
    DependencyType,
    TaskArtifact
)
from agentcore.a2a_protocol.services.agent_manager import agent_manager


class TaskManager:
    """
    Task lifecycle manager with dependency resolution and agent assignment.

    Manages task creation, scheduling, execution tracking, and completion.
    Handles dependency graphs and agent capability matching.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # Task storage
        self._task_executions: Dict[str, TaskExecution] = {}
        self._task_definitions: Dict[str, TaskDefinition] = {}

        # Indexing for efficient queries
        self._tasks_by_status: Dict[TaskStatus, Set[str]] = defaultdict(set)
        self._tasks_by_agent: Dict[str, Set[str]] = defaultdict(set)
        self._tasks_by_type: Dict[str, Set[str]] = defaultdict(set)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # task_id -> dependent_task_ids
        self._reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # task_id -> prerequisite_task_ids

        # Execution tracking
        self._execution_locks: Dict[str, asyncio.Lock] = {}

    async def create_task(self, request: TaskCreateRequest) -> TaskCreateResponse:
        """
        Create a new task execution from definition.

        Args:
            request: Task creation request with definition and options

        Returns:
            Task creation response with execution details

        Raises:
            ValueError: If task definition is invalid or dependencies cannot be resolved
        """
        task_def = request.task_definition

        # Validate dependencies exist
        await self._validate_dependencies(task_def)

        # Create task execution
        execution = TaskExecution(task_definition=task_def)

        # Store task
        self._task_definitions[task_def.task_id] = task_def
        self._task_executions[execution.execution_id] = execution

        # Update indices
        self._update_indices_on_create(execution)

        # Build dependency graph
        await self._update_dependency_graph(task_def)

        self.logger.info(
            "Task created",
            task_id=task_def.task_id,
            execution_id=execution.execution_id,
            task_type=task_def.task_type,
            priority=task_def.priority.value
        )

        # Auto-assign if requested and no dependencies
        assigned_agent = None
        if request.auto_assign and not task_def.dependencies:
            assigned_agent = await self._auto_assign_task(execution, request.preferred_agent)

        return TaskCreateResponse(
            execution_id=execution.execution_id,
            task_id=task_def.task_id,
            status=execution.status.value,
            assigned_agent=assigned_agent,
            message=f"Task created successfully{'and assigned' if assigned_agent else ''}"
        )

    async def get_task(self, execution_id: str) -> Optional[TaskExecution]:
        """Get task execution by ID."""
        return self._task_executions.get(execution_id)

    async def assign_task(self, execution_id: str, agent_id: str) -> bool:
        """
        Assign task to a specific agent.

        Args:
            execution_id: Task execution ID
            agent_id: Target agent ID

        Returns:
            True if assignment successful, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        # Verify agent exists and is capable
        agent = await agent_manager.get_agent(agent_id)
        if not agent or not agent.is_active():
            self.logger.warning("Cannot assign to inactive agent", agent_id=agent_id)
            return False

        # Check agent capabilities match task requirements
        if not await self._agent_can_handle_task(agent_id, execution.task_definition):
            self.logger.warning(
                "Agent lacks required capabilities",
                agent_id=agent_id,
                required_capabilities=execution.task_definition.requirements.required_capabilities
            )
            return False

        try:
            # Update indices
            if execution.assigned_agent:
                self._tasks_by_agent[execution.assigned_agent].discard(execution_id)
            self._tasks_by_agent[agent_id].add(execution_id)

            # Update status indices
            self._tasks_by_status[execution.status].discard(execution_id)

            # Assign task
            execution.assign_to_agent(agent_id)

            # Update status indices
            self._tasks_by_status[execution.status].add(execution_id)

            self.logger.info(
                "Task assigned",
                execution_id=execution_id,
                task_id=execution.task_id,
                agent_id=agent_id
            )

            return True

        except ValueError as e:
            self.logger.error("Task assignment failed", error=str(e))
            return False

    async def start_task(self, execution_id: str) -> bool:
        """
        Start task execution.

        Args:
            execution_id: Task execution ID

        Returns:
            True if started successfully, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        # Check dependencies are satisfied
        if not await self._dependencies_satisfied(execution.task_definition):
            self.logger.warning(
                "Cannot start task - dependencies not satisfied",
                execution_id=execution_id,
                task_id=execution.task_id
            )
            return False

        try:
            # Update status indices
            self._tasks_by_status[execution.status].discard(execution_id)

            # Start execution
            execution.start_execution()

            # Update status indices
            self._tasks_by_status[execution.status].add(execution_id)

            self.logger.info(
                "Task started",
                execution_id=execution_id,
                task_id=execution.task_id,
                agent_id=execution.assigned_agent
            )

            return True

        except ValueError as e:
            self.logger.error("Task start failed", error=str(e))
            return False

    async def complete_task(
        self,
        execution_id: str,
        result_data: Dict[str, Any],
        artifacts: Optional[List[TaskArtifact]] = None
    ) -> bool:
        """
        Complete task execution successfully.

        Args:
            execution_id: Task execution ID
            result_data: Task execution results
            artifacts: Optional execution artifacts

        Returns:
            True if completed successfully, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        try:
            # Update status indices
            self._tasks_by_status[execution.status].discard(execution_id)

            # Complete execution
            execution.complete_execution(result_data, artifacts)

            # Update status indices
            self._tasks_by_status[execution.status].add(execution_id)

            self.logger.info(
                "Task completed",
                execution_id=execution_id,
                task_id=execution.task_id,
                agent_id=execution.assigned_agent,
                duration=execution.execution_duration
            )

            # Check for dependent tasks that can now be started
            await self._process_dependent_tasks(execution.task_id)

            return True

        except ValueError as e:
            self.logger.error("Task completion failed", error=str(e))
            return False

    async def fail_task(self, execution_id: str, error_message: str, should_retry: bool = True) -> bool:
        """
        Mark task as failed with optional retry.

        Args:
            execution_id: Task execution ID
            error_message: Failure reason
            should_retry: Whether to attempt retry

        Returns:
            True if failure recorded successfully, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        try:
            # Update status indices
            self._tasks_by_status[execution.status].discard(execution_id)

            # Fail execution
            execution.fail_execution(error_message, should_retry)

            # Update status indices
            self._tasks_by_status[execution.status].add(execution_id)

            self.logger.warning(
                "Task failed",
                execution_id=execution_id,
                task_id=execution.task_id,
                agent_id=execution.assigned_agent,
                error=error_message,
                retry_count=execution.retry_count
            )

            return True

        except ValueError as e:
            self.logger.error("Task failure recording failed", error=str(e))
            return False

    async def cancel_task(self, execution_id: str) -> bool:
        """
        Cancel task execution.

        Args:
            execution_id: Task execution ID

        Returns:
            True if cancelled successfully, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        try:
            # Update status indices
            self._tasks_by_status[execution.status].discard(execution_id)

            # Cancel execution
            execution.cancel_execution()

            # Update status indices
            self._tasks_by_status[execution.status].add(execution_id)

            self.logger.info(
                "Task cancelled",
                execution_id=execution_id,
                task_id=execution.task_id,
                agent_id=execution.assigned_agent
            )

            return True

        except ValueError as e:
            self.logger.error("Task cancellation failed", error=str(e))
            return False

    async def update_task_progress(self, execution_id: str, percentage: float, current_step: Optional[str] = None) -> bool:
        """
        Update task execution progress.

        Args:
            execution_id: Task execution ID
            percentage: Progress percentage (0-100)
            current_step: Optional current step description

        Returns:
            True if updated successfully, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        try:
            execution.update_progress(percentage, current_step)

            self.logger.debug(
                "Task progress updated",
                execution_id=execution_id,
                task_id=execution.task_id,
                progress=percentage,
                step=current_step
            )

            return True

        except ValueError as e:
            self.logger.error("Task progress update failed", error=str(e))
            return False

    async def add_task_artifact(
        self,
        execution_id: str,
        name: str,
        artifact_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an artifact to a task execution.

        Args:
            execution_id: Task execution ID
            name: Artifact name
            artifact_type: Artifact type (file, data, url, etc.)
            content: Artifact content
            metadata: Optional artifact metadata

        Returns:
            True if added successfully, False otherwise
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return False

        try:
            execution.add_artifact(name, artifact_type, content, metadata)

            self.logger.debug(
                "Task artifact added",
                execution_id=execution_id,
                task_id=execution.task_id,
                artifact_name=name,
                artifact_type=artifact_type
            )

            return True

        except ValueError as e:
            self.logger.error("Task artifact addition failed", error=str(e))
            return False

    async def get_task_artifacts(self, execution_id: str) -> Optional[List[TaskArtifact]]:
        """
        Get all artifacts for a task execution.

        Args:
            execution_id: Task execution ID

        Returns:
            List of artifacts or None if task not found
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return None

        return execution.artifacts

    async def get_task_status_transitions(self, execution_id: str) -> Optional[List[TaskStatus]]:
        """
        Get valid status transitions for a task execution.

        Args:
            execution_id: Task execution ID

        Returns:
            List of valid next statuses or None if task not found
        """
        execution = self._task_executions.get(execution_id)
        if not execution:
            return None

        valid_next = []
        for status in TaskStatus:
            if execution.can_transition_to(status):
                valid_next.append(status)

        return valid_next

    async def query_tasks(self, query: TaskQuery) -> TaskQueryResponse:
        """
        Query tasks with filtering and pagination.

        Args:
            query: Task query parameters

        Returns:
            Query response with matching tasks
        """
        # Start with all execution IDs
        candidate_ids = set(self._task_executions.keys())

        # Apply filters
        if query.status is not None:
            candidate_ids &= self._tasks_by_status[query.status]

        if query.task_type is not None:
            candidate_ids &= self._tasks_by_type[query.task_type]

        if query.assigned_agent is not None:
            candidate_ids &= self._tasks_by_agent[query.assigned_agent]

        # Filter by additional criteria
        filtered_executions = []
        for execution_id in candidate_ids:
            execution = self._task_executions[execution_id]
            task_def = execution.task_definition

            # Creator filter
            if query.created_by and task_def.created_by != query.created_by:
                continue

            # Tags filter (all must match)
            if query.tags and not all(tag in task_def.tags for tag in query.tags):
                continue

            # Priority filter
            if query.priority and task_def.priority != query.priority:
                continue

            # Time filters
            if query.created_after and task_def.created_at < query.created_after:
                continue

            if query.created_before and task_def.created_at > query.created_before:
                continue

            filtered_executions.append(execution)

        # Sort by creation time (newest first)
        filtered_executions.sort(key=lambda e: e.task_definition.created_at, reverse=True)

        # Apply pagination
        total_count = len(filtered_executions)
        start_idx = query.offset
        end_idx = start_idx + query.limit

        paginated_executions = filtered_executions[start_idx:end_idx]

        # Convert to summaries
        task_summaries = [execution.to_summary() for execution in paginated_executions]

        return TaskQueryResponse(
            tasks=task_summaries,
            total_count=total_count,
            has_more=end_idx < total_count,
            query=query
        )

    async def get_task_dependencies(self, task_id: str) -> Dict[str, List[str]]:
        """
        Get task dependency information.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with prerequisites and dependents
        """
        return {
            "prerequisites": list(self._reverse_dependency_graph.get(task_id, set())),
            "dependents": list(self._dependency_graph.get(task_id, set()))
        }

    async def get_ready_tasks(self) -> List[str]:
        """
        Get tasks that are ready to be assigned (pending with satisfied dependencies).

        Returns:
            List of execution IDs for ready tasks
        """
        ready_tasks = []

        for execution_id in self._tasks_by_status[TaskStatus.PENDING]:
            execution = self._task_executions[execution_id]
            if await self._dependencies_satisfied(execution.task_definition):
                ready_tasks.append(execution_id)

        return ready_tasks

    async def cleanup_old_tasks(self, max_age_days: int = 30) -> int:
        """
        Cleanup old completed/failed tasks.

        Args:
            max_age_days: Maximum age in days for task retention

        Returns:
            Number of tasks cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleanup_count = 0

        # Find old terminal tasks
        terminal_statuses = [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        cleanup_ids = []

        for status in terminal_statuses:
            for execution_id in list(self._tasks_by_status[status]):
                execution = self._task_executions[execution_id]
                if execution.completed_at and execution.completed_at < cutoff_time:
                    cleanup_ids.append(execution_id)

        # Remove old tasks
        for execution_id in cleanup_ids:
            execution = self._task_executions.pop(execution_id)
            task_def = self._task_definitions.pop(execution.task_id, None)

            # Update indices
            self._update_indices_on_remove(execution)
            cleanup_count += 1

        if cleanup_count > 0:
            self.logger.info("Old tasks cleaned up", count=cleanup_count, max_age_days=max_age_days)

        return cleanup_count

    # Private helper methods

    async def _validate_dependencies(self, task_def: TaskDefinition) -> None:
        """Validate task dependencies exist and are valid."""
        for dep in task_def.dependencies:
            if dep.task_id not in self._task_definitions:
                raise ValueError(f"Dependency task {dep.task_id} not found")

    async def _update_dependency_graph(self, task_def: TaskDefinition) -> None:
        """Update dependency graphs for task."""
        for dep in task_def.dependencies:
            if dep.type == DependencyType.PREDECESSOR:
                self._dependency_graph[dep.task_id].add(task_def.task_id)
                self._reverse_dependency_graph[task_def.task_id].add(dep.task_id)

    async def _dependencies_satisfied(self, task_def: TaskDefinition) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep in task_def.dependencies:
            if dep.type == DependencyType.PREDECESSOR:
                dep_task_executions = [
                    e for e in self._task_executions.values()
                    if e.task_id == dep.task_id
                ]
                if not dep_task_executions or not any(e.is_completed for e in dep_task_executions):
                    return False
        return True

    async def _auto_assign_task(self, execution: TaskExecution, preferred_agent: Optional[str]) -> Optional[str]:
        """Automatically assign task to capable agent."""
        # Try preferred agent first
        if preferred_agent and await self._agent_can_handle_task(preferred_agent, execution.task_definition):
            if await self.assign_task(execution.execution_id, preferred_agent):
                return preferred_agent

        # Find capable agents
        capable_agents = await self._find_capable_agents(execution.task_definition)

        # Simple round-robin assignment for now
        for agent_id in capable_agents:
            if await self.assign_task(execution.execution_id, agent_id):
                return agent_id

        return None

    async def _agent_can_handle_task(self, agent_id: str, task_def: TaskDefinition) -> bool:
        """Check if agent can handle task based on capabilities."""
        agent = await agent_manager.get_agent(agent_id)
        if not agent or not agent.is_active():
            return False

        # Check required capabilities
        for req_cap in task_def.requirements.required_capabilities:
            if not agent.has_capability(req_cap):
                return False

        # Check excluded agents
        if agent_id in task_def.requirements.excluded_agents:
            return False

        return True

    async def _find_capable_agents(self, task_def: TaskDefinition) -> List[str]:
        """Find agents capable of handling task."""
        capable_agents = []

        # Get all active agents
        all_agents = await agent_manager.list_all_agents()

        for agent_summary in all_agents:
            agent_id = agent_summary["agent_id"]
            if await self._agent_can_handle_task(agent_id, task_def):
                capable_agents.append(agent_id)

        return capable_agents

    async def _process_dependent_tasks(self, completed_task_id: str) -> None:
        """Process tasks that depend on the completed task."""
        dependent_task_ids = self._dependency_graph.get(completed_task_id, set())

        for task_id in dependent_task_ids:
            # Find pending executions for this task
            pending_executions = [
                e for e in self._task_executions.values()
                if e.task_id == task_id and e.status == TaskStatus.PENDING
            ]

            for execution in pending_executions:
                if await self._dependencies_satisfied(execution.task_definition):
                    self.logger.info(
                        "Dependent task ready for assignment",
                        task_id=task_id,
                        execution_id=execution.execution_id,
                        completed_dependency=completed_task_id
                    )

    def _update_indices_on_create(self, execution: TaskExecution) -> None:
        """Update indices when task is created."""
        self._tasks_by_status[execution.status].add(execution.execution_id)
        self._tasks_by_type[execution.task_type].add(execution.execution_id)

    def _update_indices_on_remove(self, execution: TaskExecution) -> None:
        """Update indices when task is removed."""
        # Remove from all indices
        for status_set in self._tasks_by_status.values():
            status_set.discard(execution.execution_id)

        for type_set in self._tasks_by_type.values():
            type_set.discard(execution.execution_id)

        for agent_set in self._tasks_by_agent.values():
            agent_set.discard(execution.execution_id)

        # Remove from dependency graphs
        task_id = execution.task_id
        for dep_set in self._dependency_graph.values():
            dep_set.discard(task_id)

        for dep_set in self._reverse_dependency_graph.values():
            dep_set.discard(task_id)

        self._dependency_graph.pop(task_id, None)
        self._reverse_dependency_graph.pop(task_id, None)


# Global task manager instance
task_manager = TaskManager()