"""Agent service for managing agent lifecycle and discovery.

This service provides high-level operations for agent management without
any knowledge of JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Any

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import (
    ValidationError,
    AgentNotFoundError,
    OperationError,
)


class AgentService:
    """Service for agent operations.

    Provides business operations for agent lifecycle management:
    - Registration
    - Discovery and search
    - Information retrieval
    - Removal

    This service abstracts JSON-RPC protocol details and focuses on
    business logic and domain validation.

    Args:
        client: JSON-RPC client for API communication

    Attributes:
        client: JSON-RPC client instance

    Example:
        >>> transport = HttpTransport("http://localhost:8001")
        >>> client = JsonRpcClient(transport)
        >>> service = AgentService(client)
        >>> agent_id = service.register("analyzer", ["python", "analysis"])
        >>> print(agent_id)
        'agent-001'
    """

    def __init__(self, client: JsonRpcClient) -> None:
        """Initialize agent service.

        Args:
            client: JSON-RPC client for API communication
        """
        self.client = client

    def register(
        self,
        name: str,
        capabilities: list[str],
        endpoint_url: str | None = None,
        cost_per_request: float = 0.01,
        requirements: dict[str, Any] | None = None,
    ) -> str:
        """Register a new agent.

        Args:
            name: Agent name (must be unique)
            capabilities: List of agent capabilities (at least one required)
            endpoint_url: Agent endpoint URL (if None, uses a placeholder)
            cost_per_request: Cost per request in dollars (default: 0.01)
            requirements: Optional agent requirements (e.g., hardware, dependencies)

        Returns:
            Agent ID (string)

        Raises:
            ValidationError: If validation fails (empty name, no capabilities, etc.)
            OperationError: If registration fails

        Example:
            >>> agent_id = service.register(
            ...     "analyzer",
            ...     ["python", "analysis"],
            ...     endpoint_url="http://localhost:5000",
            ...     cost_per_request=0.02,
            ...     requirements={"memory": "4GB"}
            ... )
            >>> print(agent_id)
            'agent-001'
        """
        # Business validation
        if not name or not name.strip():
            raise ValidationError("Agent name cannot be empty")

        if not capabilities:
            raise ValidationError("At least one capability required")

        if cost_per_request < 0:
            raise ValidationError("Cost per request cannot be negative")

        # Construct AgentCard according to A2A protocol v0.2
        # Build capability objects
        capability_objects = [
            {
                "name": cap,
                "version": "1.0.0",
                "cost_per_request": cost_per_request,
            }
            for cap in capabilities
        ]

        # Build endpoint (required by A2A protocol)
        # If no endpoint URL provided, use a placeholder that points to localhost
        endpoint_url_str = endpoint_url or "http://localhost:8000"
        endpoint = {
            "url": endpoint_url_str,
            "type": "https" if endpoint_url_str.startswith("https") else "http",
            "protocols": ["jsonrpc-2.0"],
            "health_check_path": "/health",
        }

        # Build authentication (required by A2A protocol)
        # For CLI registration, we'll use a simple "none" auth type
        authentication = {
            "type": "none",
            "required": False,
            "config": {},
        }

        # Construct full agent_card
        agent_card = {
            "agent_name": name.strip(),
            "agent_version": "1.0.0",
            "status": "active",
            "endpoints": [endpoint],
            "capabilities": capability_objects,
            "authentication": authentication,
        }

        # Add optional requirements if provided
        if requirements:
            agent_card["requirements"] = requirements

        # Prepare parameters with agent_card wrapper (as API expects)
        params: dict[str, Any] = {
            "agent_card": agent_card,
            "override_existing": False,
        }

        # Call JSON-RPC method
        try:
            result = self.client.call("agent.register", params)
        except Exception as e:
            raise OperationError(f"Agent registration failed: {str(e)}")

        # Validate result
        agent_id = result.get("agent_id")
        if not agent_id:
            raise OperationError("API did not return agent_id")

        return str(agent_id)

    def list_agents(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List agents with optional filtering.

        Args:
            status: Optional status filter ("active", "inactive", "error")
            limit: Maximum number of agents to return (default: 100)
            offset: Number of agents to skip (default: 0)

        Returns:
            List of agent dictionaries

        Raises:
            ValidationError: If parameters are invalid
            OperationError: If listing fails

        Example:
            >>> agents = service.list(status="active", limit=10)
            >>> for agent in agents:
            ...     print(agent["name"])
            'analyzer'
            'tester'
        """
        # Validation
        if limit <= 0:
            raise ValidationError("Limit must be positive")

        if offset < 0:
            raise ValidationError("Offset cannot be negative")

        if status and status not in ["active", "inactive", "error"]:
            raise ValidationError(
                f"Invalid status: {status}. Must be 'active', 'inactive', or 'error'"
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
            result = self.client.call("agent.list", params)
        except Exception as e:
            raise OperationError(f"Agent listing failed: {str(e)}")

        # Extract agents
        agents = result.get("agents", [])
        if not isinstance(agents, list):
            raise OperationError("API returned invalid agents list")

        return agents

    def get(self, agent_id: str) -> dict[str, Any]:
        """Get agent information by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent information dictionary

        Raises:
            ValidationError: If agent_id is empty
            AgentNotFoundError: If agent does not exist
            OperationError: If retrieval fails

        Example:
            >>> info = service.get("agent-001")
            >>> print(info["name"])
            'analyzer'
        """
        # Validation
        if not agent_id or not agent_id.strip():
            raise ValidationError("Agent ID cannot be empty")

        # Call JSON-RPC method
        try:
            result = self.client.call("agent.get", {"agent_id": agent_id.strip()})
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found")
            raise OperationError(f"Agent retrieval failed: {str(e)}")

        # Validate result
        agent = result.get("agent")
        if not agent:
            raise OperationError("API did not return agent information")

        return dict(agent)

    def remove(self, agent_id: str, force: bool = False) -> bool:
        """Remove an agent.

        Args:
            agent_id: Agent identifier
            force: Force removal even if agent is active (default: False)

        Returns:
            True if successful

        Raises:
            ValidationError: If agent_id is empty
            AgentNotFoundError: If agent does not exist
            OperationError: If removal fails

        Example:
            >>> success = service.remove("agent-001", force=True)
            >>> print(success)
            True
        """
        # Validation
        if not agent_id or not agent_id.strip():
            raise ValidationError("Agent ID cannot be empty")

        # Prepare parameters
        params: dict[str, Any] = {
            "agent_id": agent_id.strip(),
            "force": force,
        }

        # Call JSON-RPC method
        try:
            result = self.client.call("agent.remove", params)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found")
            raise OperationError(f"Agent removal failed: {str(e)}")

        # Validate result
        success = result.get("success", False)
        return bool(success)

    def search(
        self,
        capability: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search agents by capability.

        Args:
            capability: Capability to search for
            limit: Maximum number of agents to return (default: 100)

        Returns:
            List of matching agent dictionaries

        Raises:
            ValidationError: If parameters are invalid
            OperationError: If search fails

        Example:
            >>> agents = service.search("python", limit=10)
            >>> for agent in agents:
            ...     print(agent["name"])
            'analyzer'
            'executor'
        """
        # Validation
        if not capability or not capability.strip():
            raise ValidationError("Capability cannot be empty")

        if limit <= 0:
            raise ValidationError("Limit must be positive")

        # Prepare parameters
        params: dict[str, Any] = {
            "capability": capability.strip(),
            "limit": limit,
        }

        # Call JSON-RPC method
        try:
            result = self.client.call("agent.search", params)
        except Exception as e:
            raise OperationError(f"Agent search failed: {str(e)}")

        # Extract agents
        agents = result.get("agents", [])
        if not isinstance(agents, list):
            raise OperationError("API returned invalid agents list")

        return agents
