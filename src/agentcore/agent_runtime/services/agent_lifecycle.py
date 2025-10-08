"""
Agent lifecycle management service.

This module manages the complete lifecycle of agent execution including
creation, execution, pause, resume, and termination with state management.
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState
from .a2a_client import A2AClient, A2ARegistrationError
from .container_manager import ContainerManager

logger = structlog.get_logger()


class AgentLifecycleError(Exception):
    """Base exception for agent lifecycle errors."""


class AgentNotFoundException(AgentLifecycleError):
    """Raised when agent is not found."""


class AgentStateError(AgentLifecycleError):
    """Raised when agent is in invalid state for operation."""


class AgentLifecycleManager:
    """Manages agent lifecycle from creation to termination."""

    def __init__(
        self,
        container_manager: ContainerManager,
        a2a_client: A2AClient | None = None,
    ) -> None:
        """
        Initialize lifecycle manager.

        Args:
            container_manager: Container management service
            a2a_client: A2A protocol client for integration
        """
        self._container_manager = container_manager
        self._a2a_client = a2a_client
        self._agents: dict[str, AgentExecutionState] = {}
        self._agent_tasks: dict[str, asyncio.Task[None]] = {}

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentExecutionState:
        """
        Create a new agent and initialize container.

        Args:
            agent_config: Agent configuration

        Returns:
            Initial agent execution state

        Raises:
            AgentLifecycleError: If agent creation fails
        """
        agent_id = agent_config.agent_id

        # Check if agent already exists
        if agent_id in self._agents:
            raise AgentLifecycleError(f"Agent {agent_id} already exists")

        # Create initial execution state
        state = AgentExecutionState(
            agent_id=agent_id,
            status="initializing",
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
        self._agents[agent_id] = state

        try:
            # Create container
            container_id = await self._container_manager.create_container(agent_config)
            state.container_id = container_id
            state.last_updated = datetime.now()

            logger.info(
                "agent_created",
                agent_id=agent_id,
                container_id=container_id,
                philosophy=agent_config.philosophy.value,
            )

            # Register with A2A protocol if client is available
            if self._a2a_client:
                try:
                    await self._a2a_client.register_agent(agent_config)
                    logger.info(
                        "agent_registered_with_a2a",
                        agent_id=agent_id,
                    )
                except (A2ARegistrationError, Exception) as e:
                    logger.warning(
                        "a2a_registration_failed",
                        agent_id=agent_id,
                        error=str(e),
                    )
                    # Don't fail agent creation if A2A registration fails
                    # Agent can still run locally

            return state

        except Exception as e:
            # Only fail if container creation failed, not A2A registration
            if "Connection refused" not in str(e):
                # Cleanup on failure
                state.status = "failed"
                state.failure_reason = str(e)
                logger.error(
                    "agent_creation_failed",
                    agent_id=agent_id,
                    error=str(e),
                )
                raise AgentLifecycleError(f"Failed to create agent: {e}") from e
            # If only A2A failed, return state anyway
            return state

    async def start_agent(self, agent_id: str) -> None:
        """
        Start agent execution.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundException: If agent not found
            AgentStateError: If agent not in valid state
        """
        state = self._get_agent_state(agent_id)

        if state.status not in ("initializing", "paused"):
            raise AgentStateError(
                f"Cannot start agent in {state.status} state"
            )

        try:
            # Start container
            await self._container_manager.start_container(agent_id)

            # Update state
            state.status = "running"
            state.last_updated = datetime.now()

            # Start monitoring task
            task = asyncio.create_task(self._monitor_agent(agent_id))
            self._agent_tasks[agent_id] = task

            # Update A2A status to active
            if self._a2a_client:
                try:
                    await self._a2a_client.update_agent_status(
                        agent_id=agent_id,
                        status="active",
                    )
                except Exception as e:
                    logger.warning(
                        "a2a_status_update_failed",
                        agent_id=agent_id,
                        error=str(e),
                    )

            logger.info("agent_started", agent_id=agent_id)

        except Exception as e:
            state.status = "failed"
            state.failure_reason = str(e)
            logger.error(
                "agent_start_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise

    async def pause_agent(self, agent_id: str) -> None:
        """
        Pause agent execution.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundException: If agent not found
            AgentStateError: If agent not running
        """
        state = self._get_agent_state(agent_id)

        if state.status != "running":
            raise AgentStateError(
                f"Cannot pause agent in {state.status} state"
            )

        try:
            # Stop container
            await self._container_manager.stop_container(agent_id)

            # Cancel monitoring task
            if agent_id in self._agent_tasks:
                self._agent_tasks[agent_id].cancel()
                del self._agent_tasks[agent_id]

            # Update state
            state.status = "paused"
            state.last_updated = datetime.now()

            logger.info("agent_paused", agent_id=agent_id)

        except Exception as e:
            logger.error(
                "agent_pause_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise

    async def terminate_agent(
        self,
        agent_id: str,
        cleanup: bool = True,
    ) -> None:
        """
        Terminate agent execution and cleanup resources.

        Args:
            agent_id: Agent identifier
            cleanup: Whether to remove container and state

        Raises:
            AgentNotFoundException: If agent not found
        """
        state = self._get_agent_state(agent_id)

        try:
            # Cancel monitoring task if running
            if agent_id in self._agent_tasks:
                self._agent_tasks[agent_id].cancel()
                del self._agent_tasks[agent_id]

            # Stop and remove container
            if state.container_id:
                await self._container_manager.stop_container(agent_id, timeout=5)
                if cleanup:
                    await self._container_manager.remove_container(
                        agent_id,
                        force=True,
                    )

            # Update state
            state.status = "terminated"
            state.last_updated = datetime.now()

            # Unregister from A2A protocol
            if self._a2a_client and cleanup:
                try:
                    await self._a2a_client.unregister_agent(agent_id)
                    logger.info("agent_unregistered_from_a2a", agent_id=agent_id)
                except Exception as e:
                    logger.warning(
                        "a2a_unregistration_failed",
                        agent_id=agent_id,
                        error=str(e),
                    )

            if cleanup:
                # Remove from tracking
                del self._agents[agent_id]

            logger.info(
                "agent_terminated",
                agent_id=agent_id,
                cleanup=cleanup,
            )

        except Exception as e:
            logger.error(
                "agent_termination_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise

    async def get_agent_status(self, agent_id: str) -> AgentExecutionState:
        """
        Get current agent execution status.

        Args:
            agent_id: Agent identifier

        Returns:
            Current agent execution state

        Raises:
            AgentNotFoundException: If agent not found
        """
        return self._get_agent_state(agent_id)

    async def list_agents(self) -> list[AgentExecutionState]:
        """
        List all tracked agents.

        Returns:
            List of agent execution states
        """
        return list(self._agents.values())

    async def update_agent_metrics(
        self,
        agent_id: str,
        metrics: dict[str, float],
    ) -> None:
        """
        Update agent performance metrics.

        Args:
            agent_id: Agent identifier
            metrics: Performance metrics dictionary

        Raises:
            AgentNotFoundException: If agent not found
        """
        state = self._get_agent_state(agent_id)
        state.performance_metrics.update(metrics)
        state.last_updated = datetime.now()

    async def save_checkpoint(
        self,
        agent_id: str,
        checkpoint_data: bytes,
    ) -> None:
        """
        Save agent state checkpoint.

        Args:
            agent_id: Agent identifier
            checkpoint_data: Serialized checkpoint data

        Raises:
            AgentNotFoundException: If agent not found
        """
        state = self._get_agent_state(agent_id)
        state.checkpoint_data = checkpoint_data
        state.last_updated = datetime.now()

        logger.info(
            "checkpoint_saved",
            agent_id=agent_id,
            size_bytes=len(checkpoint_data),
        )

    async def restore_checkpoint(self, agent_id: str) -> bytes | None:
        """
        Restore agent from checkpoint.

        Args:
            agent_id: Agent identifier

        Returns:
            Checkpoint data if available, None otherwise

        Raises:
            AgentNotFoundException: If agent not found
        """
        state = self._get_agent_state(agent_id)
        return state.checkpoint_data

    def _get_agent_state(self, agent_id: str) -> AgentExecutionState:
        """
        Get agent state or raise exception.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent execution state

        Raises:
            AgentNotFoundException: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentNotFoundException(f"Agent {agent_id} not found")
        return self._agents[agent_id]

    async def _monitor_agent(self, agent_id: str) -> None:
        """
        Monitor agent execution and update metrics.

        Args:
            agent_id: Agent identifier
        """
        state = self._get_agent_state(agent_id)

        try:
            while state.status == "running":
                # Check if container is still running
                is_running = await self._container_manager.container_is_running(
                    agent_id
                )

                if not is_running:
                    state.status = "completed"
                    state.last_updated = datetime.now()
                    logger.info("agent_completed", agent_id=agent_id)
                    break

                # Update performance metrics
                try:
                    stats = await self._container_manager.get_container_stats(agent_id)
                    await self.update_agent_metrics(agent_id, stats)

                    # Report health to A2A protocol
                    if self._a2a_client:
                        try:
                            await self._a2a_client.report_health(
                                agent_id=agent_id,
                                health_status="healthy",
                                metrics=stats,
                            )
                        except Exception as health_error:
                            logger.debug(
                                "a2a_health_report_failed",
                                agent_id=agent_id,
                                error=str(health_error),
                            )

                except Exception as e:
                    logger.warning(
                        "metrics_update_failed",
                        agent_id=agent_id,
                        error=str(e),
                    )

                # Wait before next check
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("agent_monitoring_cancelled", agent_id=agent_id)
            raise
        except Exception as e:
            state.status = "failed"
            state.failure_reason = str(e)
            logger.error(
                "agent_monitoring_failed",
                agent_id=agent_id,
                error=str(e),
            )
