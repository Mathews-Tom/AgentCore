"""
Agent Management JSON-RPC Methods

JSON-RPC 2.0 methods for agent registration, discovery, and lifecycle management.
Integrates with the AgentManager service and JSON-RPC processor.
"""

from typing import Any, Dict, List, Optional

import structlog
from pydantic import ValidationError

from agentcore.a2a_protocol.models.agent import (
    AgentCard,
    AgentStatus,
    AgentDiscoveryQuery,
    AgentRegistrationRequest,
)
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.agent_manager import agent_manager
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

logger = structlog.get_logger()


@register_jsonrpc_method("agent.register")
async def handle_agent_register(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Register a new agent in the A2A ecosystem.

    Method: agent.register
    Params:
        - agent_card: AgentCard object with agent details
        - override_existing: bool (optional, default false)

    Returns:
        - agent_id: string
        - status: string
        - discovery_url: string
        - message: string (optional)
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_card and optional override_existing")

        # Extract and validate parameters
        agent_card_data = request.params.get("agent_card")
        if not agent_card_data:
            raise ValueError("Missing required parameter: agent_card")

        override_existing = request.params.get("override_existing", False)

        # Create and validate agent card
        agent_card = AgentCard(**agent_card_data)

        # Create registration request
        registration_request = AgentRegistrationRequest(
            agent_card=agent_card,
            override_existing=override_existing
        )

        # Register agent
        response = await agent_manager.register_agent(registration_request)

        logger.info(
            "Agent registered via JSON-RPC",
            agent_id=response.agent_id,
            agent_name=agent_card.agent_name,
            method="agent.register"
        )

        return response.model_dump()

    except ValidationError as e:
        logger.error("Agent registration validation failed", error=str(e))
        raise ValueError(f"Agent card validation failed: {e}")
    except Exception as e:
        logger.error("Agent registration failed", error=str(e))
        raise


@register_jsonrpc_method("agent.get")
async def handle_agent_get(request: JsonRpcRequest) -> Optional[Dict[str, Any]]:
    """
    Get agent details by ID.

    Method: agent.get
    Params:
        - agent_id: string

    Returns:
        - AgentCard object or null if not found
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id")

        agent_id = request.params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")

        agent = await agent_manager.get_agent(agent_id)
        if agent:
            return agent.model_dump()
        return None

    except Exception as e:
        logger.error("Agent retrieval failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("agent.discover")
async def handle_agent_discover(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Discover agents based on criteria.

    Method: agent.discover
    Params:
        - capabilities: array of strings (optional)
        - status: string (optional)
        - tags: array of strings (optional)
        - category: string (optional)
        - name_pattern: string (optional)
        - limit: integer (optional, default 50)
        - offset: integer (optional, default 0)

    Returns:
        - agents: array of agent summaries
        - total_count: integer
        - has_more: boolean
        - query: original query parameters
    """
    try:
        # Parse query parameters
        params = request.params or {}
        if isinstance(params, list):
            raise ValueError("Parameters must be an object, not an array")

        query = AgentDiscoveryQuery(**params)

        # Perform discovery
        response = await agent_manager.discover_agents(query)

        logger.info(
            "Agent discovery completed",
            found_count=response.total_count,
            returned_count=len(response.agents),
            method="agent.discover"
        )

        return response.model_dump()

    except ValidationError as e:
        logger.error("Agent discovery validation failed", error=str(e))
        raise ValueError(f"Discovery query validation failed: {e}")
    except Exception as e:
        logger.error("Agent discovery failed", error=str(e))
        raise


@register_jsonrpc_method("agent.unregister")
async def handle_agent_unregister(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Unregister an agent.

    Method: agent.unregister
    Params:
        - agent_id: string

    Returns:
        - success: boolean
        - message: string
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id")

        agent_id = request.params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")

        success = await agent_manager.unregister_agent(agent_id)

        message = "Agent unregistered successfully" if success else "Agent not found"

        logger.info(
            "Agent unregistration attempted",
            agent_id=agent_id,
            success=success,
            method="agent.unregister"
        )

        return {
            "success": success,
            "message": message
        }

    except Exception as e:
        logger.error("Agent unregistration failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("agent.ping")
async def handle_agent_ping(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Ping an agent (update last seen timestamp).

    Method: agent.ping
    Params:
        - agent_id: string

    Returns:
        - success: boolean
        - message: string
        - timestamp: string (ISO8601)
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id")

        agent_id = request.params.get("agent_id")
        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")

        success = await agent_manager.ping_agent(agent_id)

        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()

        message = "Agent pinged successfully" if success else "Agent not found"

        logger.debug(
            "Agent ping",
            agent_id=agent_id,
            success=success,
            method="agent.ping"
        )

        return {
            "success": success,
            "message": message,
            "timestamp": timestamp
        }

    except Exception as e:
        logger.error("Agent ping failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("agent.update_status")
async def handle_agent_update_status(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Update agent status.

    Method: agent.update_status
    Params:
        - agent_id: string
        - status: string (active, inactive, maintenance, error)

    Returns:
        - success: boolean
        - message: string
        - new_status: string
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id and status")

        agent_id = request.params.get("agent_id")
        status_str = request.params.get("status")

        if not agent_id:
            raise ValueError("Missing required parameter: agent_id")
        if not status_str:
            raise ValueError("Missing required parameter: status")

        # Validate status
        try:
            status = AgentStatus(status_str)
        except ValueError:
            valid_statuses = [s.value for s in AgentStatus]
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")

        success = await agent_manager.update_agent_status(agent_id, status)

        message = "Agent status updated successfully" if success else "Agent not found"

        logger.info(
            "Agent status update",
            agent_id=agent_id,
            new_status=status.value,
            success=success,
            method="agent.update_status"
        )

        return {
            "success": success,
            "message": message,
            "new_status": status.value
        }

    except Exception as e:
        logger.error("Agent status update failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("agent.list")
async def handle_agent_list(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    List all registered agents (summaries).

    Method: agent.list
    Params: none

    Returns:
        - agents: array of agent summaries
        - count: integer
    """
    try:
        agents = await agent_manager.list_all_agents()
        count = len(agents)

        logger.info(
            "Agent list requested",
            count=count,
            method="agent.list"
        )

        return {
            "agents": agents,
            "count": count
        }

    except Exception as e:
        logger.error("Agent list failed", error=str(e))
        raise


@register_jsonrpc_method("agent.capabilities")
async def handle_agent_capabilities(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get capabilities index with agent counts.

    Method: agent.capabilities
    Params: none

    Returns:
        - capabilities: object with capability names as keys and counts as values
        - total_agents: integer
    """
    try:
        capabilities = await agent_manager.get_capabilities_index()
        total_agents = await agent_manager.get_agent_count()

        logger.info(
            "Capabilities index requested",
            unique_capabilities=len(capabilities),
            total_agents=total_agents,
            method="agent.capabilities"
        )

        return {
            "capabilities": capabilities,
            "total_agents": total_agents
        }

    except Exception as e:
        logger.error("Capabilities index failed", error=str(e))
        raise


@register_jsonrpc_method("agent.cleanup")
async def handle_agent_cleanup(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Clean up inactive agents.

    Method: agent.cleanup
    Params:
        - max_inactive_hours: integer (optional, default 24)

    Returns:
        - removed_count: integer
        - max_inactive_hours: integer
    """
    try:
        params = request.params or {}
        if isinstance(params, list):
            raise ValueError("Parameters must be an object, not an array")

        max_inactive_hours = params.get("max_inactive_hours", 24)

        if not isinstance(max_inactive_hours, int) or max_inactive_hours <= 0:
            raise ValueError("max_inactive_hours must be a positive integer")

        removed_count = await agent_manager.cleanup_inactive_agents(max_inactive_hours)

        logger.info(
            "Agent cleanup completed",
            removed_count=removed_count,
            max_inactive_hours=max_inactive_hours,
            method="agent.cleanup"
        )

        return {
            "removed_count": removed_count,
            "max_inactive_hours": max_inactive_hours
        }

    except Exception as e:
        logger.error("Agent cleanup failed", error=str(e))
        raise