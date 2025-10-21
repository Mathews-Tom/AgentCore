"""
Custom Pattern Framework

Extensible framework for defining and registering custom orchestration patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PatternType(str, Enum):
    """Built-in pattern types."""

    SUPERVISOR = "supervisor"
    HIERARCHICAL = "hierarchical"
    HANDOFF = "handoff"
    SWARM = "swarm"
    NETWORK = "network"
    SAGA = "saga"
    CIRCUIT_BREAKER = "circuit_breaker"
    CUSTOM = "custom"


class PatternStatus(str, Enum):
    """Pattern registration status."""

    DRAFT = "draft"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class CoordinationModel(str, Enum):
    """Coordination model for pattern execution."""

    EVENT_DRIVEN = "event_driven"
    GRAPH_BASED = "graph_based"
    HYBRID = "hybrid"


class PatternMetadata(BaseModel):
    """Metadata for a pattern definition."""

    name: str = Field(description="Pattern name")
    description: str = Field(description="Pattern description")
    version: str = Field(description="Pattern version (semver)")
    author: str | None = Field(default=None, description="Pattern author")
    tags: list[str] = Field(default_factory=list, description="Search tags")
    documentation_url: str | None = Field(default=None, description="Documentation URL")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Update timestamp"
    )


class AgentRequirement(BaseModel):
    """Agent requirement specification for pattern."""

    role: str = Field(description="Agent role name")
    capabilities: list[str] = Field(description="Required capabilities")
    min_count: int = Field(default=1, description="Minimum agent count")
    max_count: int | None = Field(default=None, description="Maximum agent count")
    resource_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Resource requirements (CPU, memory, etc.)"
    )


class TaskNode(BaseModel):
    """Task node in pattern workflow graph."""

    task_id: str = Field(description="Unique task identifier")
    task_name: str = Field(
        default="", description="Task name (alias for task_id if not provided)"
    )
    agent_role: str = Field(description="Agent role for this task")
    depends_on: list[str] = Field(default_factory=list, description="Task dependencies")
    parallel: bool = Field(default=False, description="Can execute in parallel")
    timeout_seconds: int = Field(default=300, description="Task timeout")
    retry_enabled: bool = Field(default=True, description="Enable retries")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    compensation_action: str | None = Field(
        default=None, description="Compensation action for saga pattern"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Task metadata")

    def model_post_init(self, __context: Any) -> None:
        """Set task_name to task_id if not provided."""
        if not self.task_name:
            self.task_name = self.task_id


class CoordinationConfig(BaseModel):
    """Coordination configuration for pattern."""

    model: CoordinationModel = Field(description="Coordination model")
    event_driven_triggers: list[str] = Field(
        default_factory=list, description="Event types for event-driven coordination"
    )
    graph_based_tasks: list[str] = Field(
        default_factory=list, description="Task IDs for graph-based coordination"
    )
    timeout_seconds: int = Field(default=3600, description="Overall workflow timeout")
    max_concurrent_tasks: int = Field(
        default=10, description="Maximum concurrent tasks"
    )


class ValidationRule(BaseModel):
    """Validation rule for pattern."""

    rule_id: str = Field(description="Rule identifier")
    rule_type: str = Field(
        description="Rule type (agent_capability, resource_constraint, etc.)"
    )
    condition: dict[str, Any] = Field(description="Validation condition")
    error_message: str = Field(description="Error message if validation fails")


class PatternDefinition(BaseModel):
    """
    Custom pattern definition.

    Defines a reusable orchestration pattern with agents, tasks, and coordination.
    """

    pattern_id: UUID = Field(
        default_factory=uuid4, description="Unique pattern identifier"
    )
    metadata: PatternMetadata = Field(description="Pattern metadata")
    pattern_type: PatternType = Field(description="Pattern type")
    status: PatternStatus = Field(
        default=PatternStatus.DRAFT, description="Pattern status"
    )

    # Pattern components
    agents: dict[str, AgentRequirement] = Field(
        description="Agent requirements by role"
    )
    tasks: list[TaskNode] = Field(description="Task workflow graph")
    coordination: CoordinationConfig = Field(description="Coordination configuration")

    # Validation
    validation_rules: list[ValidationRule] = Field(
        default_factory=list, description="Pattern validation rules"
    )

    # Template parameters
    template_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Configurable template parameters"
    )

    # Statistics
    execution_count: int = Field(default=0, description="Number of executions")
    success_count: int = Field(default=0, description="Successful executions")
    failure_count: int = Field(default=0, description="Failed executions")
    avg_execution_time_seconds: float = Field(
        default=0.0, description="Average execution time"
    )

    model_config = {"frozen": False}

    def validate_pattern(self) -> tuple[bool, list[str]]:
        """
        Validate pattern definition.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: list[str] = []

        # Validate agent requirements
        if not self.agents:
            errors.append("Pattern must define at least one agent requirement")

        # Validate tasks
        if not self.tasks:
            errors.append("Pattern must define at least one task")

        # Validate task dependencies
        task_ids = {task.task_id for task in self.tasks}
        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    errors.append(
                        f"Task '{task.task_id}' depends on unknown task '{dep}'"
                    )

        # Validate agent roles
        for task in self.tasks:
            if task.agent_role not in self.agents:
                errors.append(
                    f"Task '{task.task_id}' requires unknown agent role '{task.agent_role}'"
                )

        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Pattern contains circular task dependencies")

        return len(errors) == 0, errors

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in task graph."""
        # Build adjacency list
        graph: dict[str, list[str]] = {
            task.task_id: task.depends_on for task in self.tasks
        }

        # DFS to detect cycles
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task_id in graph:
            if task_id not in visited:
                if has_cycle(task_id):
                    return True

        return False

    def to_template(self) -> dict[str, Any]:
        """Convert pattern to template format."""
        return {
            "pattern_id": str(self.pattern_id),
            "name": self.metadata.name,
            "version": self.metadata.version,
            "type": self.pattern_type.value,
            "agents": {role: req.model_dump() for role, req in self.agents.items()},
            "tasks": [task.model_dump() for task in self.tasks],
            "coordination": self.coordination.model_dump(),
            "parameters": self.template_parameters,
        }


class PatternRegistry:
    """
    Registry for custom orchestration patterns.

    Manages pattern registration, validation, and retrieval.
    """

    def __init__(self):
        self._patterns: dict[UUID, PatternDefinition] = {}
        self._patterns_by_name: dict[str, UUID] = {}
        self._patterns_by_type: dict[PatternType, set[UUID]] = {}

    def register(self, pattern: PatternDefinition) -> tuple[bool, list[str]]:
        """
        Register a new pattern (alias for register_pattern).

        Args:
            pattern: Pattern definition to register

        Returns:
            Tuple of (success, error_messages)
        """
        return self.register_pattern(pattern)

    def register_pattern(self, pattern: PatternDefinition) -> tuple[bool, list[str]]:
        """
        Register a new pattern.

        Args:
            pattern: Pattern definition to register

        Returns:
            Tuple of (success, error_messages)
        """
        # Validate pattern
        is_valid, errors = pattern.validate_pattern()
        if not is_valid:
            return False, errors

        # Check for name conflicts
        if pattern.metadata.name in self._patterns_by_name:
            existing_id = self._patterns_by_name[pattern.metadata.name]
            if existing_id != pattern.pattern_id:
                return False, [
                    f"Pattern name '{pattern.metadata.name}' already registered"
                ]

        # Update status to active
        pattern.status = PatternStatus.ACTIVE
        pattern.metadata.updated_at = datetime.now(UTC)

        # Store pattern
        self._patterns[pattern.pattern_id] = pattern
        self._patterns_by_name[pattern.metadata.name] = pattern.pattern_id

        if pattern.pattern_type not in self._patterns_by_type:
            self._patterns_by_type[pattern.pattern_type] = set()
        self._patterns_by_type[pattern.pattern_type].add(pattern.pattern_id)

        return True, []

    def get(self, pattern_id: UUID) -> PatternDefinition | None:
        """Get pattern by ID (alias for get_pattern)."""
        return self.get_pattern(pattern_id)

    def get_pattern(self, pattern_id: UUID) -> PatternDefinition | None:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_pattern_by_name(self, name: str) -> PatternDefinition | None:
        """Get pattern by name."""
        pattern_id = self._patterns_by_name.get(name)
        if pattern_id:
            return self._patterns.get(pattern_id)
        return None

    def list_patterns(
        self,
        pattern_type: PatternType | None = None,
        status: PatternStatus | None = None,
    ) -> list[PatternDefinition]:
        """
        List patterns with optional filters.

        Args:
            pattern_type: Filter by pattern type
            status: Filter by status

        Returns:
            List of matching patterns
        """
        patterns = list(self._patterns.values())

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        if status:
            patterns = [p for p in patterns if p.status == status]

        return patterns

    def unregister_pattern(self, pattern_id: UUID) -> bool:
        """
        Unregister (archive) a pattern.

        Args:
            pattern_id: Pattern ID to unregister

        Returns:
            True if unregistered, False if not found
        """
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return False

        # Mark as archived
        pattern.status = PatternStatus.ARCHIVED
        pattern.metadata.updated_at = datetime.now(UTC)

        return True

    def update_statistics(
        self, pattern_id: UUID, execution_time: float, success: bool
    ) -> None:
        """Update pattern execution statistics."""
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return

        pattern.execution_count += 1
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        # Update average execution time
        total_time = pattern.avg_execution_time_seconds * (pattern.execution_count - 1)
        pattern.avg_execution_time_seconds = (
            total_time + execution_time
        ) / pattern.execution_count


# Global pattern registry instance
pattern_registry = PatternRegistry()
