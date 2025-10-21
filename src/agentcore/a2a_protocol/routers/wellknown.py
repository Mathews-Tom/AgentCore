"""
Well-Known Endpoints

Standard A2A protocol discovery endpoints as per specification.
Provides agent discovery, protocol information, and service metadata.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.services.agent_manager import agent_manager

router = APIRouter()
logger = structlog.get_logger()


@router.get("/.well-known/agent.json", summary="A2A Protocol Service Information")
async def get_service_info() -> dict[str, Any]:
    """
    A2A protocol service information endpoint.

    Returns service metadata, supported protocols, and capabilities
    according to the A2A protocol specification.
    """
    from agentcore import __version__

    agent_count = await agent_manager.get_agent_count()
    capabilities_index = await agent_manager.get_capabilities_index()

    service_info = {
        "schema_version": settings.A2A_PROTOCOL_VERSION,
        "service_name": "AgentCore A2A Protocol Layer",
        "service_version": __version__,
        "description": "Agent2Agent communication infrastructure implementing Google's A2A protocol v0.2",
        "protocol_version": "2.0",
        "supported_methods": [
            "agent.register",
            "agent.get",
            "agent.discover",
            "agent.unregister",
            "agent.ping",
            "agent.update_status",
            "agent.list",
            "agent.capabilities",
            "agent.cleanup",
            "rpc.ping",
            "rpc.version",
            "rpc.methods",
        ],
        "endpoints": [
            {
                "url": f"http://localhost:{settings.PORT}/api/v1/jsonrpc",
                "type": "http",
                "protocols": ["jsonrpc-2.0"],
                "description": "Main JSON-RPC endpoint",
            }
        ],
        "discovery": {
            "agent_count": agent_count,
            "available_capabilities": list(capabilities_index.keys()),
            "discovery_endpoint": f"/.well-known/agents",
        },
        "authentication": {
            "type": "bearer_token",
            "required": False,
            "description": "JWT tokens supported for authenticated requests",
        },
        "limits": {
            "max_concurrent_connections": settings.MAX_CONCURRENT_CONNECTIONS,
            "message_timeout_seconds": settings.MESSAGE_TIMEOUT_SECONDS,
            "discovery_ttl_seconds": settings.AGENT_DISCOVERY_TTL,
        },
    }

    logger.info("Service info requested", agent_count=agent_count)
    return service_info


@router.get("/.well-known/agents", summary="List All Agents")
async def get_all_agents() -> dict[str, Any]:
    """
    Get all registered agents (discovery summaries).

    Returns a list of all currently registered agents with their
    basic information for discovery purposes.
    """
    agents = await agent_manager.list_all_agents()

    response = {
        "agents": agents,
        "count": len(agents),
        "timestamp": "2024-01-01T00:00:00Z",  # TODO: Use actual timestamp
    }

    logger.info("All agents discovery requested", count=len(agents))
    return response


@router.get("/.well-known/agents/{agent_id}", summary="Get Specific Agent")
async def get_agent_by_id(agent_id: str) -> dict[str, Any]:
    """
    Get specific agent information by ID.

    Args:
        agent_id: Agent identifier

    Returns:
        Agent discovery summary

    Raises:
        404: If agent not found
    """
    agent_summary = await agent_manager.get_agent_summary(agent_id)

    if not agent_summary:
        logger.warning("Agent not found for discovery", agent_id=agent_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} not found"
        )

    logger.info("Agent discovery requested", agent_id=agent_id)
    return agent_summary


@router.get("/.well-known/capabilities", summary="List Available Capabilities")
async def get_capabilities() -> dict[str, Any]:
    """
    Get all available capabilities across registered agents.

    Returns capabilities index with agent counts for each capability.
    """
    capabilities = await agent_manager.get_capabilities_index()
    total_agents = await agent_manager.get_agent_count()

    response = {
        "capabilities": capabilities,
        "total_agents": total_agents,
        "unique_capabilities": len(capabilities),
    }

    logger.info(
        "Capabilities discovery requested",
        unique_capabilities=len(capabilities),
        total_agents=total_agents,
    )
    return response


@router.get("/.well-known/health", summary="A2A Protocol Health Check")
async def get_protocol_health() -> dict[str, Any]:
    """
    A2A protocol layer health check.

    Returns health status specific to the A2A protocol implementation.
    """
    agent_count = await agent_manager.get_agent_count()

    health_info = {
        "status": "healthy",
        "protocol_version": settings.A2A_PROTOCOL_VERSION,
        "agent_count": agent_count,
        "timestamp": "2024-01-01T00:00:00Z",  # TODO: Use actual timestamp
        "checks": {
            "agent_manager": "healthy",
            "jsonrpc_processor": "healthy",
            "discovery_service": "healthy",
        },
    }

    logger.debug("A2A protocol health check", agent_count=agent_count)
    return health_info
