"""Workflow service for executing and monitoring workflows.

This service provides high-level operations for workflow management without
any knowledge of JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Any

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import (
    ValidationError,
    WorkflowNotFoundError,
    OperationError,
)


class WorkflowService:
    """Service for workflow operations.

    Provides business operations for workflow execution and monitoring:
    - Workflow execution from YAML definition
    - Workflow listing and filtering
    - Workflow information retrieval
    - Workflow stopping

    This service abstracts JSON-RPC protocol details and focuses on
    business logic and domain validation.

    Args:
        client: JSON-RPC client for API communication

    Attributes:
        client: JSON-RPC client instance

    Example:
        >>> transport = HttpTransport("http://localhost:8001")
        >>> client = JsonRpcClient(transport)
        >>> service = WorkflowService(client)
        >>> workflow_id = service.run("workflow.yaml")
        >>> print(workflow_id)
        'workflow-001'
    """

    def __init__(self, client: JsonRpcClient) -> None:
        """Initialize workflow service.

        Args:
            client: JSON-RPC client for API communication
        """
        self.client = client

    def run(
        self,
        definition: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Run a workflow from definition.

        Args:
            definition: Workflow definition (YAML parsed to dict)
            parameters: Optional workflow parameters

        Returns:
            Workflow ID (string)

        Raises:
            ValidationError: If validation fails
            OperationError: If workflow execution fails

        Example:
            >>> definition = {
            ...     "name": "analysis-workflow",
            ...     "steps": [{"agent": "analyzer", "task": "analyze"}]
            ... }
            >>> workflow_id = service.run(definition, parameters={"repo": "foo/bar"})
            >>> print(workflow_id)
            'workflow-001'
        """
        # Business validation
        if not definition:
            raise ValidationError("Workflow definition cannot be empty")

        if not isinstance(definition, dict):
            raise ValidationError("Workflow definition must be a dictionary")

        if "name" not in definition:
            raise ValidationError("Workflow definition must have a 'name' field")

        if "steps" not in definition:
            raise ValidationError("Workflow definition must have a 'steps' field")

        if not isinstance(definition["steps"], list):
            raise ValidationError("Workflow 'steps' must be a list")

        if not definition["steps"]:
            raise ValidationError("Workflow must have at least one step")

        # Prepare parameters
        params: dict[str, Any] = {
            "definition": definition,
        }

        if parameters:
            params["parameters"] = parameters

        # Call JSON-RPC method
        try:
            result = self.client.call("workflow.run", params)
        except Exception as e:
            raise OperationError(f"Workflow execution failed: {str(e)}")

        # Validate result
        workflow_id = result.get("workflow_id")
        if not workflow_id:
            raise OperationError("API did not return workflow_id")

        return str(workflow_id)

    def list_workflows(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List workflows with optional filtering.

        Args:
            status: Optional status filter ("running", "completed", "failed", "cancelled")
            limit: Maximum number of workflows to return (default: 100)
            offset: Number of workflows to skip (default: 0)

        Returns:
            List of workflow dictionaries

        Raises:
            ValidationError: If parameters are invalid
            OperationError: If listing fails

        Example:
            >>> workflows = service.list(status="running", limit=10)
            >>> for workflow in workflows:
            ...     print(workflow["name"])
            'analysis-workflow'
            'test-workflow'
        """
        # Validation
        if limit <= 0:
            raise ValidationError("Limit must be positive")

        if offset < 0:
            raise ValidationError("Offset cannot be negative")

        valid_statuses = ["running", "completed", "failed", "cancelled"]
        if status and status not in valid_statuses:
            raise ValidationError(
                f"Invalid status: {status}. Must be one of {valid_statuses}"
            )

        # Prepare parameters
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if status:
            params["status"] = status

        # Call JSON-RPC method
        try:
            result = self.client.call("workflow.list", params)
        except Exception as e:
            raise OperationError(f"Workflow listing failed: {str(e)}")

        # Extract workflows
        workflows = result.get("workflows", [])
        if not isinstance(workflows, list):
            raise OperationError("API returned invalid workflows list")

        return workflows

    def get(self, workflow_id: str) -> dict[str, Any]:
        """Get workflow information by ID.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow information dictionary

        Raises:
            ValidationError: If workflow_id is empty
            WorkflowNotFoundError: If workflow does not exist
            OperationError: If retrieval fails

        Example:
            >>> info = service.get("workflow-001")
            >>> print(info["name"])
            'analysis-workflow'
        """
        # Validation
        if not workflow_id or not workflow_id.strip():
            raise ValidationError("Workflow ID cannot be empty")

        # Call JSON-RPC method
        try:
            result = self.client.call("workflow.get", {"workflow_id": workflow_id.strip()})
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found")
            raise OperationError(f"Workflow retrieval failed: {str(e)}")

        # Validate result
        workflow = result.get("workflow")
        if not workflow:
            raise OperationError("API did not return workflow information")

        return dict(workflow)

    def stop(self, workflow_id: str, force: bool = False) -> bool:
        """Stop a workflow.

        Args:
            workflow_id: Workflow identifier
            force: Force stop even if workflow is in critical state (default: False)

        Returns:
            True if successful

        Raises:
            ValidationError: If workflow_id is empty
            WorkflowNotFoundError: If workflow does not exist
            OperationError: If stopping fails

        Example:
            >>> success = service.stop("workflow-001", force=True)
            >>> print(success)
            True
        """
        # Validation
        if not workflow_id or not workflow_id.strip():
            raise ValidationError("Workflow ID cannot be empty")

        # Prepare parameters
        params: dict[str, Any] = {
            "workflow_id": workflow_id.strip(),
            "force": force,
        }

        # Call JSON-RPC method
        try:
            result = self.client.call("workflow.stop", params)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found")
            raise OperationError(f"Workflow stopping failed: {str(e)}")

        # Validate result
        success = result.get("success", False)
        return bool(success)
