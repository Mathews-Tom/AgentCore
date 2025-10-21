"""
CQRS Queries Module

Query models and handlers for read operations.
Queries retrieve data from optimized read models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries."""

    # Workflow queries
    GET_WORKFLOW = "get_workflow"
    LIST_WORKFLOWS = "list_workflows"
    GET_WORKFLOW_STATUS = "get_workflow_status"
    GET_WORKFLOW_HISTORY = "get_workflow_history"

    # Agent queries
    GET_AGENT_ASSIGNMENTS = "get_agent_assignments"
    LIST_AGENTS_IN_WORKFLOW = "list_agents_in_workflow"

    # Task queries
    GET_TASK = "get_task"
    LIST_TASKS = "list_tasks"
    GET_TASK_STATUS = "get_task_status"

    # Execution queries
    GET_EXECUTION = "get_execution"
    LIST_EXECUTIONS = "list_executions"
    GET_EXECUTION_METRICS = "get_execution_metrics"


class Query(BaseModel):
    """
    Base query for CQRS read operations.

    Queries retrieve data without side effects.
    """

    query_id: UUID = Field(default_factory=uuid4, description="Unique query identifier")
    query_type: QueryType = Field(description="Type of query")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Query timestamp"
    )
    user_id: str | None = Field(default=None, description="User issuing query")


class QueryResult(BaseModel):
    """Result of query execution."""

    query_id: UUID = Field(description="Query identifier")
    success: bool = Field(description="Whether query succeeded")
    data: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Query result data"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    error_type: str | None = Field(default=None, description="Error type if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )


class GetWorkflowQuery(Query):
    """Query to get workflow details."""

    query_type: QueryType = Field(default=QueryType.GET_WORKFLOW, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    include_tasks: bool = Field(default=False, description="Include task definitions")
    include_agents: bool = Field(default=False, description="Include agent assignments")


class ListWorkflowsQuery(Query):
    """Query to list workflows."""

    query_type: QueryType = Field(default=QueryType.LIST_WORKFLOWS, frozen=True)
    orchestration_pattern: str | None = Field(
        default=None, description="Filter by orchestration pattern"
    )
    status: str | None = Field(default=None, description="Filter by workflow status")
    created_by: str | None = Field(default=None, description="Filter by creator")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    order_by: str = Field(default="created_at", description="Sort field")
    ascending: bool = Field(default=False, description="Sort order")


class GetWorkflowStatusQuery(Query):
    """Query to get workflow execution status."""

    query_type: QueryType = Field(default=QueryType.GET_WORKFLOW_STATUS, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID | None = Field(
        default=None, description="Specific execution, or latest if None"
    )
    include_task_status: bool = Field(
        default=True, description="Include task status details"
    )


class GetWorkflowHistoryQuery(Query):
    """Query to get workflow event history."""

    query_type: QueryType = Field(default=QueryType.GET_WORKFLOW_HISTORY, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    from_timestamp: datetime | None = Field(
        default=None, description="Start timestamp for history"
    )
    to_timestamp: datetime | None = Field(
        default=None, description="End timestamp for history"
    )
    event_types: list[str] | None = Field(
        default=None, description="Filter by event types"
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum events")


class GetAgentAssignmentsQuery(Query):
    """Query to get agent assignments."""

    query_type: QueryType = Field(default=QueryType.GET_AGENT_ASSIGNMENTS, frozen=True)
    agent_id: str = Field(description="Agent identifier")
    workflow_id: UUID | None = Field(default=None, description="Filter by workflow")
    active_only: bool = Field(default=True, description="Only active assignments")


class ListAgentsInWorkflowQuery(Query):
    """Query to list agents in a workflow."""

    query_type: QueryType = Field(
        default=QueryType.LIST_AGENTS_IN_WORKFLOW, frozen=True
    )
    workflow_id: UUID = Field(description="Workflow identifier")
    role: str | None = Field(default=None, description="Filter by agent role")


class GetTaskQuery(Query):
    """Query to get task details."""

    query_type: QueryType = Field(default=QueryType.GET_TASK, frozen=True)
    task_id: UUID = Field(description="Task identifier")
    include_input: bool = Field(default=False, description="Include input data")
    include_output: bool = Field(default=False, description="Include output data")


class ListTasksQuery(Query):
    """Query to list tasks."""

    query_type: QueryType = Field(default=QueryType.LIST_TASKS, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID | None = Field(default=None, description="Filter by execution")
    status: str | None = Field(default=None, description="Filter by task status")
    agent_id: str | None = Field(default=None, description="Filter by agent")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")


class GetTaskStatusQuery(Query):
    """Query to get task execution status."""

    query_type: QueryType = Field(default=QueryType.GET_TASK_STATUS, frozen=True)
    task_id: UUID = Field(description="Task identifier")


class GetExecutionQuery(Query):
    """Query to get execution details."""

    query_type: QueryType = Field(default=QueryType.GET_EXECUTION, frozen=True)
    execution_id: UUID = Field(description="Execution identifier")
    include_tasks: bool = Field(default=True, description="Include task details")


class ListExecutionsQuery(Query):
    """Query to list workflow executions."""

    query_type: QueryType = Field(default=QueryType.LIST_EXECUTIONS, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    status: str | None = Field(default=None, description="Filter by execution status")
    from_timestamp: datetime | None = Field(default=None, description="Start timestamp")
    to_timestamp: datetime | None = Field(default=None, description="End timestamp")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class GetExecutionMetricsQuery(Query):
    """Query to get execution performance metrics."""

    query_type: QueryType = Field(default=QueryType.GET_EXECUTION_METRICS, frozen=True)
    execution_id: UUID = Field(description="Execution identifier")


class QueryHandler(ABC):
    """
    Abstract base class for query handlers.

    Each query handler processes one type of query.
    """

    @abstractmethod
    async def handle(self, query: Query) -> QueryResult:
        """
        Handle a query and return result.

        Args:
            query: Query to handle

        Returns:
            Query execution result
        """
        pass

    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """
        Check if this handler can handle the given query.

        Args:
            query: Query to check

        Returns:
            True if handler can handle this query
        """
        pass


QueryHandlerType = Callable[[Query], QueryResult]


class QueryBus:
    """
    Query bus for routing queries to handlers.

    Implements query dispatching with registration and routing.
    """

    def __init__(self) -> None:
        """Initialize query bus."""
        self._handlers: dict[QueryType, QueryHandler] = {}

    def register(self, query_type: QueryType, handler: QueryHandler) -> None:
        """
        Register a query handler.

        Args:
            query_type: Type of query to handle
            handler: Handler instance

        Raises:
            ValueError: If handler already registered for query type
        """
        if query_type in self._handlers:
            raise ValueError(f"Handler already registered for {query_type}")

        self._handlers[query_type] = handler

    def unregister(self, query_type: QueryType) -> None:
        """
        Unregister a query handler.

        Args:
            query_type: Type of query to unregister
        """
        if query_type in self._handlers:
            del self._handlers[query_type]

    async def dispatch(self, query: Query) -> QueryResult:
        """
        Dispatch query to appropriate handler.

        Args:
            query: Query to dispatch

        Returns:
            Query execution result

        Raises:
            ValueError: If no handler registered for query type
        """
        handler = self._handlers.get(query.query_type)
        if not handler:
            return QueryResult(
                query_id=query.query_id,
                success=False,
                error_message=f"No handler registered for query type: {query.query_type}",
                error_type="UnhandledQueryError",
            )

        try:
            result = await handler.handle(query)
            return result
        except Exception as e:
            return QueryResult(
                query_id=query.query_id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
            )

    def get_registered_handlers(self) -> dict[QueryType, QueryHandler]:
        """
        Get all registered handlers.

        Returns:
            Dictionary mapping query types to handlers
        """
        return self._handlers.copy()
