"""API endpoints for agent lifecycle management."""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState
from ..services.agent_lifecycle import (
    AgentLifecycleError,
    AgentLifecycleManager,
    AgentNotFoundException,
    AgentStateError,
)
from ..services.container_manager import ContainerManager

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])
logger = structlog.get_logger()

# Global service instances (will be initialized on startup)
_container_manager: ContainerManager | None = None
_lifecycle_manager: AgentLifecycleManager | None = None


async def get_lifecycle_manager() -> AgentLifecycleManager:
    """Get lifecycle manager instance."""
    if _lifecycle_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent lifecycle manager not initialized",
        )
    return _lifecycle_manager


class AgentCreationRequest(BaseModel):
    """Request to create a new agent."""

    config: AgentConfig = Field(description="Agent configuration")


class AgentCreationResponse(BaseModel):
    """Response after agent creation."""

    agent_id: str = Field(description="Created agent ID")
    container_id: str | None = Field(description="Container ID")
    status: str = Field(description="Initial status")


class AgentTerminationRequest(BaseModel):
    """Request to terminate an agent."""

    cleanup: bool = Field(
        default=True,
        description="Whether to remove container and state",
    )


@router.post(
    "",
    response_model=AgentCreationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent(request: AgentCreationRequest) -> AgentCreationResponse:
    """
    Create and initialize a new agent instance.

    Creates a new agent with the specified configuration including philosophy,
    resource limits, and security profile. Initializes the container but does
    not start execution.
    """
    lifecycle = await get_lifecycle_manager()

    try:
        state = await lifecycle.create_agent(request.config)

        return AgentCreationResponse(
            agent_id=state.agent_id,
            container_id=state.container_id,
            status=state.status,
        )

    except AgentLifecycleError as e:
        logger.error(
            "agent_creation_request_failed",
            agent_id=request.config.agent_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e


@router.post("/{agent_id}/start", status_code=status.HTTP_200_OK)
async def start_agent(agent_id: str) -> dict[str, str]:
    """
    Start agent execution.

    Starts the agent container and begins execution according to the
    configured philosophy. Initiates monitoring and resource tracking.
    """
    lifecycle = await get_lifecycle_manager()

    try:
        await lifecycle.start_agent(agent_id)
        return {"status": "started", "agent_id": agent_id}

    except AgentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except AgentStateError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


@router.post("/{agent_id}/pause", status_code=status.HTTP_200_OK)
async def pause_agent(agent_id: str) -> dict[str, str]:
    """
    Pause agent execution.

    Pauses the running agent and saves its current state. The agent
    can be resumed later from this checkpoint.
    """
    lifecycle = await get_lifecycle_manager()

    try:
        await lifecycle.pause_agent(agent_id)
        return {"status": "paused", "agent_id": agent_id}

    except AgentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except AgentStateError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


@router.delete("/{agent_id}", status_code=status.HTTP_200_OK)
async def terminate_agent(
    agent_id: str,
    request: AgentTerminationRequest | None = None,
) -> dict[str, str]:
    """
    Terminate agent and cleanup resources.

    Stops the agent execution, removes the container, and cleans up
    associated resources. Optionally preserves state for later inspection.
    """
    lifecycle = await get_lifecycle_manager()
    cleanup = request.cleanup if request else True

    try:
        await lifecycle.terminate_agent(agent_id, cleanup=cleanup)
        return {"status": "terminated", "agent_id": agent_id}

    except AgentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.get("/{agent_id}/status", response_model=AgentExecutionState)
async def get_agent_status(agent_id: str) -> AgentExecutionState:
    """
    Get current agent execution status.

    Returns detailed information about the agent's current state including
    execution status, performance metrics, and resource usage.
    """
    lifecycle = await get_lifecycle_manager()

    try:
        return await lifecycle.get_agent_status(agent_id)

    except AgentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.get("", response_model=list[AgentExecutionState])
async def list_agents() -> list[AgentExecutionState]:
    """
    List all tracked agents.

    Returns a list of all agents currently managed by the runtime,
    including their current status and basic metrics.
    """
    lifecycle = await get_lifecycle_manager()
    return await lifecycle.list_agents()


@router.post("/{agent_id}/checkpoint", status_code=status.HTTP_200_OK)
async def save_checkpoint(
    agent_id: str,
    checkpoint_data: dict[str, Any],
) -> dict[str, str]:
    """
    Save agent state checkpoint.

    Creates a checkpoint of the current agent state for recovery
    or migration purposes.
    """
    lifecycle = await get_lifecycle_manager()

    try:
        # Serialize checkpoint data
        import json
        checkpoint_bytes = json.dumps(checkpoint_data).encode()

        await lifecycle.save_checkpoint(agent_id, checkpoint_bytes)
        return {
            "status": "checkpoint_saved",
            "agent_id": agent_id,
            "size_bytes": str(len(checkpoint_bytes)),
        }

    except AgentNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


async def initialize_services(
    container_manager: ContainerManager,
    lifecycle_manager: AgentLifecycleManager,
) -> None:
    """Initialize global service instances."""
    global _container_manager, _lifecycle_manager
    _container_manager = container_manager
    _lifecycle_manager = lifecycle_manager
