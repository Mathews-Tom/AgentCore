"""
Hierarchical Pattern Implementation

Multi-level agent hierarchies with delegation, escalation, and authority management.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.patterns.supervisor import (
    SupervisorConfig,
    SupervisorCoordinator,
    WorkerState,
    WorkerStatus,
)
from agentcore.orchestration.streams.models import OrchestrationEvent
from agentcore.orchestration.streams.producer import StreamProducer


class AuthorityLevel(int, Enum):
    """Authority levels in the hierarchy."""

    EXECUTIVE = 0  # Highest authority
    SENIOR = 1
    INTERMEDIATE = 2
    JUNIOR = 3
    WORKER = 4  # Lowest authority


class EscalationReason(str, Enum):
    """Reasons for task escalation."""

    INSUFFICIENT_AUTHORITY = "insufficient_authority"
    RESOURCE_CONSTRAINT = "resource_constraint"
    COMPLEXITY_THRESHOLD = "complexity_threshold"
    FAILURE_THRESHOLD = "failure_threshold"
    EXPLICIT_REQUEST = "explicit_request"


class DelegationPolicy(str, Enum):
    """Delegation policy for task assignment."""

    STRICT_HIERARCHY = "strict_hierarchy"  # Only delegate down one level
    SKIP_LEVEL = "skip_level"  # Can skip levels
    BEST_FIT = "best_fit"  # Find best agent regardless of level


class HierarchyNode(BaseModel):
    """Node in the agent hierarchy."""

    agent_id: str = Field(description="Agent identifier")
    authority_level: AuthorityLevel = Field(description="Authority level")
    parent_id: str | None = Field(default=None, description="Parent agent ID")
    children_ids: list[str] = Field(default_factory=list, description="Child agent IDs")
    capabilities: list[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=5)
    current_task_count: int = Field(default=0)
    permissions: set[str] = Field(
        default_factory=set, description="Granted permissions"
    )

    model_config = {"frozen": False}


class TaskDelegation(BaseModel):
    """Record of task delegation."""

    delegation_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    from_agent_id: str
    to_agent_id: str
    delegated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = None
    authority_level_required: AuthorityLevel | None = None


class TaskEscalation(BaseModel):
    """Record of task escalation."""

    escalation_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    from_agent_id: str
    to_agent_id: str
    escalated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: EscalationReason
    context: dict[str, Any] = Field(default_factory=dict)


class HierarchicalConfig(BaseModel):
    """Configuration for hierarchical pattern."""

    max_hierarchy_depth: int = Field(default=5, ge=1, description="Maximum depth")
    delegation_policy: DelegationPolicy = Field(
        default=DelegationPolicy.STRICT_HIERARCHY
    )
    enable_escalation: bool = Field(default=True)
    escalation_threshold_failures: int = Field(
        default=2, description="Failures before escalation"
    )
    communication_optimization: bool = Field(
        default=True, description="Optimize communication paths"
    )
    supervisor_config: SupervisorConfig = Field(default_factory=SupervisorConfig)


class HierarchicalCoordinator:
    """
    Hierarchical pattern coordinator.

    Implements multi-level agent hierarchies with:
    - Multi-level agent organization
    - Delegation and escalation mechanisms
    - Authority and permission management
    - Communication flow optimization
    """

    def __init__(
        self,
        coordinator_id: str,
        config: HierarchicalConfig | None = None,
        event_producer: StreamProducer | None = None,
    ) -> None:
        """
        Initialize hierarchical coordinator.

        Args:
            coordinator_id: Unique coordinator identifier
            config: Hierarchical configuration
            event_producer: Event stream producer
        """
        self.coordinator_id = coordinator_id
        self.config = config or HierarchicalConfig()
        self.event_producer = event_producer

        # Hierarchy structure
        self._hierarchy: dict[str, HierarchyNode] = {}
        self._root_agents: set[str] = set()

        # Task tracking
        self._delegations: dict[UUID, TaskDelegation] = {}
        self._escalations: dict[UUID, list[TaskEscalation]] = {}
        self._task_assignments: dict[UUID, str] = {}  # task_id -> agent_id
        self._task_failures: dict[UUID, int] = {}  # task_id -> failure_count

        # Supervisors per level (for managing workers at each level)
        self._level_supervisors: dict[AuthorityLevel, SupervisorCoordinator] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def add_agent(
        self,
        agent_id: str,
        authority_level: AuthorityLevel,
        parent_id: str | None = None,
        capabilities: list[str] | None = None,
        permissions: set[str] | None = None,
    ) -> None:
        """
        Add an agent to the hierarchy.

        Args:
            agent_id: Agent identifier
            authority_level: Authority level
            parent_id: Parent agent ID (None for root)
            capabilities: Agent capabilities
            permissions: Agent permissions
        """
        async with self._lock:
            # Create hierarchy node
            node = HierarchyNode(
                agent_id=agent_id,
                authority_level=authority_level,
                parent_id=parent_id,
                capabilities=capabilities or [],
                permissions=permissions or set(),
            )

            # Add to hierarchy
            self._hierarchy[agent_id] = node

            # Update parent's children
            if parent_id:
                if parent_id in self._hierarchy:
                    self._hierarchy[parent_id].children_ids.append(agent_id)
            else:
                # Root agent
                self._root_agents.add(agent_id)

            # Register with level supervisor
            await self._get_or_create_supervisor(authority_level)

    async def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the hierarchy.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            if agent_id not in self._hierarchy:
                return

            node = self._hierarchy[agent_id]

            # Remove from parent's children
            if node.parent_id and node.parent_id in self._hierarchy:
                parent = self._hierarchy[node.parent_id]
                parent.children_ids.remove(agent_id)

            # Reassign children to parent or remove
            for child_id in node.children_ids:
                if child_id in self._hierarchy:
                    self._hierarchy[child_id].parent_id = node.parent_id
                    if node.parent_id and node.parent_id in self._hierarchy:
                        self._hierarchy[node.parent_id].children_ids.append(child_id)

            # Remove from root if applicable
            self._root_agents.discard(agent_id)

            # Remove from hierarchy
            del self._hierarchy[agent_id]

    async def delegate_task(
        self,
        task_id: UUID,
        from_agent_id: str,
        task_data: dict[str, Any],
        required_capabilities: list[str] | None = None,
        required_authority: AuthorityLevel | None = None,
    ) -> str | None:
        """
        Delegate a task down the hierarchy.

        Args:
            task_id: Task identifier
            from_agent_id: Delegating agent
            task_data: Task data
            required_capabilities: Required capabilities
            required_authority: Required authority level

        Returns:
            Target agent ID or None if delegation failed
        """
        async with self._lock:
            if from_agent_id not in self._hierarchy:
                return None

            delegator = self._hierarchy[from_agent_id]

            # Find suitable delegate
            target_agent_id = await self._find_delegate(
                delegator, required_capabilities, required_authority
            )

            if target_agent_id:
                # Record delegation
                delegation = TaskDelegation(
                    task_id=task_id,
                    from_agent_id=from_agent_id,
                    to_agent_id=target_agent_id,
                    authority_level_required=required_authority,
                )
                self._delegations[task_id] = delegation
                self._task_assignments[task_id] = target_agent_id

                # Update task counts
                self._hierarchy[target_agent_id].current_task_count += 1

                return target_agent_id

            return None

    async def escalate_task(
        self,
        task_id: UUID,
        from_agent_id: str,
        reason: EscalationReason,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Escalate a task up the hierarchy.

        Args:
            task_id: Task identifier
            from_agent_id: Escalating agent
            reason: Escalation reason
            context: Additional context

        Returns:
            Target agent ID or None if escalation failed
        """
        async with self._lock:
            if from_agent_id not in self._hierarchy:
                return None

            agent = self._hierarchy[from_agent_id]

            # Find escalation target (parent or higher)
            target_agent_id = agent.parent_id

            if not target_agent_id:
                # Already at root, cannot escalate
                return None

            # Record escalation
            escalation = TaskEscalation(
                task_id=task_id,
                from_agent_id=from_agent_id,
                to_agent_id=target_agent_id,
                reason=reason,
                context=context or {},
            )

            if task_id not in self._escalations:
                self._escalations[task_id] = []
            self._escalations[task_id].append(escalation)

            # Update task assignment
            self._task_assignments[task_id] = target_agent_id

            # Update task counts
            agent.current_task_count = max(0, agent.current_task_count - 1)
            self._hierarchy[target_agent_id].current_task_count += 1

            return target_agent_id

    async def handle_task_failure(
        self,
        task_id: UUID,
        agent_id: str,
        error_message: str,
    ) -> bool:
        """
        Handle task failure with automatic escalation if threshold reached.

        Args:
            task_id: Task identifier
            agent_id: Agent that failed
            error_message: Error message

        Returns:
            True if escalated, False otherwise
        """
        # First, update failure count and check if escalation is needed
        should_escalate = False
        failure_count = 0

        async with self._lock:
            # Track failure count
            self._task_failures[task_id] = self._task_failures.get(task_id, 0) + 1
            failure_count = self._task_failures[task_id]

            # Check escalation threshold
            should_escalate = (
                self.config.enable_escalation
                and failure_count >= self.config.escalation_threshold_failures
            )

        # Release lock before calling escalate_task to avoid deadlock
        if should_escalate:
            target = await self.escalate_task(
                task_id=task_id,
                from_agent_id=agent_id,
                reason=EscalationReason.FAILURE_THRESHOLD,
                context={
                    "error_message": error_message,
                    "failure_count": failure_count,
                },
            )
            return target is not None

        return False

    async def check_authority(
        self,
        agent_id: str,
        required_authority: AuthorityLevel,
        permission: str | None = None,
    ) -> bool:
        """
        Check if agent has required authority and permissions.

        Args:
            agent_id: Agent identifier
            required_authority: Required authority level
            permission: Required permission (optional)

        Returns:
            True if authorized, False otherwise
        """
        async with self._lock:
            if agent_id not in self._hierarchy:
                return False

            agent = self._hierarchy[agent_id]

            # Check authority level (lower value = higher authority)
            if agent.authority_level.value > required_authority.value:
                return False

            # Check permission if specified
            if permission and permission not in agent.permissions:
                return False

            return True

    async def grant_permission(self, agent_id: str, permission: str) -> None:
        """
        Grant permission to an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission to grant
        """
        async with self._lock:
            if agent_id in self._hierarchy:
                self._hierarchy[agent_id].permissions.add(permission)

    async def revoke_permission(self, agent_id: str, permission: str) -> None:
        """
        Revoke permission from an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission to revoke
        """
        async with self._lock:
            if agent_id in self._hierarchy:
                self._hierarchy[agent_id].permissions.discard(permission)

    async def get_hierarchy_tree(self) -> dict[str, Any]:
        """
        Get the complete hierarchy tree structure.

        Returns:
            Hierarchy tree as nested dictionary
        """
        async with self._lock:

            def build_tree(agent_id: str) -> dict[str, Any]:
                node = self._hierarchy[agent_id]
                return {
                    "agent_id": agent_id,
                    "authority_level": node.authority_level.name,
                    "capabilities": node.capabilities,
                    "current_tasks": node.current_task_count,
                    "children": [
                        build_tree(child_id) for child_id in node.children_ids
                    ],
                }

            return {
                "roots": [build_tree(root_id) for root_id in self._root_agents],
                "total_agents": len(self._hierarchy),
                "max_depth": await self._get_max_depth(),
            }

    async def get_communication_path(
        self,
        from_agent_id: str,
        to_agent_id: str,
    ) -> list[str]:
        """
        Get optimized communication path between two agents.

        Args:
            from_agent_id: Source agent
            to_agent_id: Target agent

        Returns:
            List of agent IDs in communication path
        """
        async with self._lock:
            if (
                from_agent_id not in self._hierarchy
                or to_agent_id not in self._hierarchy
            ):
                return []

            # Find common ancestor
            from_ancestors = await self._get_ancestors(from_agent_id)
            to_ancestors = await self._get_ancestors(to_agent_id)

            # Find lowest common ancestor
            common_ancestor = None
            for ancestor in from_ancestors:
                if ancestor in to_ancestors:
                    common_ancestor = ancestor
                    break

            if not common_ancestor:
                return []

            # Build path: from -> ancestor -> to
            path_up = []
            current = from_agent_id
            while current != common_ancestor:
                path_up.append(current)
                current = self._hierarchy[current].parent_id or common_ancestor

            path_down = []
            current = to_agent_id
            while current != common_ancestor:
                path_down.append(current)
                current = self._hierarchy[current].parent_id or common_ancestor

            # Combine paths
            return path_up + [common_ancestor] + list(reversed(path_down))

    async def _find_delegate(
        self,
        delegator: HierarchyNode,
        required_capabilities: list[str] | None,
        required_authority: AuthorityLevel | None,
    ) -> str | None:
        """Find suitable agent to delegate task to."""
        candidates = []

        if self.config.delegation_policy == DelegationPolicy.STRICT_HIERARCHY:
            # Only direct children
            candidates = [
                self._hierarchy[child_id]
                for child_id in delegator.children_ids
                if child_id in self._hierarchy
            ]
        elif self.config.delegation_policy == DelegationPolicy.SKIP_LEVEL:
            # All descendants
            candidates = await self._get_all_descendants(delegator.agent_id)
        else:  # BEST_FIT
            # All agents below this level
            candidates = [
                node
                for node in self._hierarchy.values()
                if node.authority_level.value > delegator.authority_level.value
            ]

        # Filter by capabilities and authority
        for candidate in candidates:
            # Check authority
            if (
                required_authority
                and candidate.authority_level.value > required_authority.value
            ):
                continue

            # Check capabilities
            if required_capabilities and not all(
                cap in candidate.capabilities for cap in required_capabilities
            ):
                continue

            # Check availability
            if candidate.current_task_count >= candidate.max_concurrent_tasks:
                continue

            return candidate.agent_id

        return None

    async def _get_all_descendants(self, agent_id: str) -> list[HierarchyNode]:
        """Get all descendants of an agent."""
        descendants = []
        if agent_id in self._hierarchy:
            node = self._hierarchy[agent_id]
            for child_id in node.children_ids:
                if child_id in self._hierarchy:
                    descendants.append(self._hierarchy[child_id])
                    descendants.extend(await self._get_all_descendants(child_id))
        return descendants

    async def _get_ancestors(self, agent_id: str) -> list[str]:
        """Get all ancestors of an agent."""
        ancestors = []
        current = agent_id
        while current in self._hierarchy:
            node = self._hierarchy[current]
            if node.parent_id:
                ancestors.append(node.parent_id)
                current = node.parent_id
            else:
                break
        return ancestors

    async def _get_max_depth(self) -> int:
        """Get maximum depth of the hierarchy."""
        max_depth = 0
        for root_id in self._root_agents:
            depth = await self._get_subtree_depth(root_id)
            max_depth = max(max_depth, depth)
        return max_depth

    async def _get_subtree_depth(self, agent_id: str) -> int:
        """Get depth of subtree rooted at agent."""
        if agent_id not in self._hierarchy:
            return 0

        node = self._hierarchy[agent_id]
        if not node.children_ids:
            return 1

        max_child_depth = 0
        for child_id in node.children_ids:
            child_depth = await self._get_subtree_depth(child_id)
            max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    async def _get_or_create_supervisor(
        self, level: AuthorityLevel
    ) -> SupervisorCoordinator:
        """Get or create supervisor for a level."""
        if level not in self._level_supervisors:
            supervisor = SupervisorCoordinator(
                supervisor_id=f"{self.coordinator_id}-level-{level.name}",
                config=self.config.supervisor_config,
                event_producer=self.event_producer,
            )
            self._level_supervisors[level] = supervisor
        return self._level_supervisors[level]

    async def get_hierarchy_status(self) -> dict[str, Any]:
        """
        Get current hierarchy status.

        Returns:
            Status dictionary
        """
        async with self._lock:
            level_counts = {}
            for node in self._hierarchy.values():
                level_name = node.authority_level.name
                level_counts[level_name] = level_counts.get(level_name, 0) + 1

            return {
                "coordinator_id": self.coordinator_id,
                "total_agents": len(self._hierarchy),
                "root_agents": len(self._root_agents),
                "max_depth": await self._get_max_depth(),
                "agents_by_level": level_counts,
                "active_delegations": len(self._delegations),
                "escalations": sum(len(esc) for esc in self._escalations.values()),
                "config": self.config.model_dump(),
            }
