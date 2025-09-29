"""
Health Monitoring JSON-RPC Methods

JSON-RPC 2.0 methods for agent health monitoring and service discovery (A2A-008).
"""

from typing import Any, Dict, List, Optional

import structlog
from datetime import datetime

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.health_monitor import health_monitor
from agentcore.a2a_protocol.database import get_session
from agentcore.a2a_protocol.database.repositories import AgentRepository
from agentcore.a2a_protocol.models.agent import AgentStatus

logger = structlog.get_logger()


@register_jsonrpc_method("health.check_agent")
async def handle_check_agent(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Check health of a specific agent.

    Method: health.check_agent
    Params:
        - agent_id: string

    Returns:
        Agent health status
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    is_healthy = await health_monitor.check_agent_health(agent_id)

    logger.info("Agent health check via JSON-RPC", agent_id=agent_id, is_healthy=is_healthy)

    return {
        "success": True,
        "agent_id": agent_id,
        "is_healthy": is_healthy,
        "checked_at": datetime.utcnow().isoformat()
    }


@register_jsonrpc_method("health.check_all")
async def handle_check_all(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Check health of all agents.

    Method: health.check_all
    Params: none

    Returns:
        Health status for all agents
    """
    results = await health_monitor.check_all_agents()

    logger.info("All agents health check via JSON-RPC", total=len(results))

    return {
        "success": True,
        "results": results,
        "total_agents": len(results),
        "healthy_count": sum(1 for h in results.values() if h),
        "unhealthy_count": sum(1 for h in results.values() if not h),
        "checked_at": datetime.utcnow().isoformat()
    }


@register_jsonrpc_method("health.get_history")
async def handle_get_history(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get health check history for an agent.

    Method: health.get_history
    Params:
        - agent_id: string
        - limit: number (optional, default 10)

    Returns:
        Health check history
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    limit = request.params.get("limit", 10)

    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    history = await health_monitor.get_agent_health_history(agent_id, limit)

    logger.debug("Health history retrieved via JSON-RPC", agent_id=agent_id, records=len(history))

    return {
        "success": True,
        "agent_id": agent_id,
        "history": history,
        "count": len(history)
    }


@register_jsonrpc_method("health.get_unhealthy")
async def handle_get_unhealthy(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get list of unhealthy agents.

    Method: health.get_unhealthy
    Params: none

    Returns:
        List of unhealthy agent IDs
    """
    unhealthy_agents = await health_monitor.get_unhealthy_agents()

    logger.debug("Unhealthy agents retrieved via JSON-RPC", count=len(unhealthy_agents))

    return {
        "success": True,
        "unhealthy_agents": unhealthy_agents,
        "count": len(unhealthy_agents)
    }


@register_jsonrpc_method("health.get_stats")
async def handle_get_stats(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get health monitoring statistics.

    Method: health.get_stats
    Params: none

    Returns:
        Health monitoring statistics
    """
    stats = health_monitor.get_statistics()

    logger.debug("Health stats retrieved via JSON-RPC")

    return {
        "success": True,
        "stats": stats
    }


@register_jsonrpc_method("discovery.find_agents")
async def handle_find_agents(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Service discovery: Find agents by capabilities and status.

    Method: discovery.find_agents
    Params:
        - capabilities: array of strings (optional)
        - status: string (optional, AgentStatus enum)

    Returns:
        List of matching agents
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    capabilities = params.get("capabilities")
    status_str = params.get("status")

    status = AgentStatus(status_str) if status_str else AgentStatus.ACTIVE

    async with get_session() as session:
        if capabilities:
            agents = await AgentRepository.get_by_capabilities(session, capabilities, status)
        else:
            agents = await AgentRepository.get_all(session, status)

    agent_list = [
        {
            "agent_id": a.id,
            "name": a.name,
            "version": a.version,
            "status": a.status.value,
            "capabilities": a.capabilities,
            "endpoint": a.endpoint,
            "current_load": a.current_load,
            "max_load": a.max_load,
            "last_seen": a.last_seen.isoformat() if a.last_seen else None
        }
        for a in agents
    ]

    logger.info(
        "Agent discovery via JSON-RPC",
        capabilities=capabilities,
        status=status.value,
        results=len(agent_list)
    )

    return {
        "success": True,
        "agents": agent_list,
        "count": len(agent_list)
    }


@register_jsonrpc_method("discovery.get_agent")
async def handle_get_agent(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Service discovery: Get agent details by ID.

    Method: discovery.get_agent
    Params:
        - agent_id: string

    Returns:
        Agent details
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    async with get_session() as session:
        agent = await AgentRepository.get_by_id(session, agent_id)

    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    logger.debug("Agent retrieved via JSON-RPC", agent_id=agent_id)

    return {
        "success": True,
        "agent": {
            "agent_id": agent.id,
            "name": agent.name,
            "version": agent.version,
            "status": agent.status.value,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "requirements": agent.requirements,
            "endpoint": agent.endpoint,
            "current_load": agent.current_load,
            "max_load": agent.max_load,
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat(),
            "last_seen": agent.last_seen.isoformat() if agent.last_seen else None
        }
    }


@register_jsonrpc_method("discovery.list_capabilities")
async def handle_list_capabilities(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Service discovery: List all available capabilities.

    Method: discovery.list_capabilities
    Params: none

    Returns:
        List of all capabilities across all agents
    """
    async with get_session() as session:
        agents = await AgentRepository.get_all(session, AgentStatus.ACTIVE)

    # Collect all unique capabilities
    all_capabilities = set()
    for agent in agents:
        all_capabilities.update(agent.capabilities)

    capabilities_list = sorted(list(all_capabilities))

    logger.debug("Capabilities listed via JSON-RPC", count=len(capabilities_list))

    return {
        "success": True,
        "capabilities": capabilities_list,
        "count": len(capabilities_list)
    }


# Log registration on import
logger.info("Health monitoring JSON-RPC methods registered",
           methods=[
               "health.check_agent", "health.check_all", "health.get_history",
               "health.get_unhealthy", "health.get_stats",
               "discovery.find_agents", "discovery.get_agent", "discovery.list_capabilities"
           ])