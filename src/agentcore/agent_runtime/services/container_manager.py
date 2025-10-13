"""
Container management service for agent runtime.

This module handles Docker container lifecycle management including creation,
execution, monitoring, and cleanup for agent containers.
"""

import asyncio
from typing import Any

import aiodocker
import structlog
from aiodocker.containers import DockerContainer
from aiodocker.exceptions import DockerError

from ..config import get_settings
from ..models.agent_config import AgentConfig
from ..models.sandbox import SandboxConfig
from .performance_optimizer import get_performance_optimizer

settings = get_settings()
logger = structlog.get_logger()


class ContainerManager:
    """Manages Docker container lifecycle for agent execution."""

    def __init__(self) -> None:
        """Initialize container manager with Docker client."""
        self._docker: aiodocker.Docker | None = None
        self._containers: dict[str, DockerContainer] = {}
        self._optimizer = get_performance_optimizer()

    async def initialize(self) -> None:
        """Initialize Docker client connection."""
        try:
            self._docker = aiodocker.Docker()
            logger.info("container_manager_initialized")
        except Exception as e:
            logger.error("container_manager_init_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close Docker client connection and cleanup resources."""
        if self._docker:
            await self._docker.close()
            logger.info("container_manager_closed")

    async def create_container(
        self,
        agent_config: AgentConfig,
        sandbox_config: SandboxConfig | None = None,
    ) -> str:
        """
        Create a new container for agent execution.

        Args:
            agent_config: Agent configuration with resource limits and security
            sandbox_config: Optional sandbox configuration for additional security

        Returns:
            Container ID

        Raises:
            DockerError: If container creation fails
        """
        if not self._docker:
            raise RuntimeError("Container manager not initialized")

        container_config = self._build_container_config(agent_config, sandbox_config)

        try:
            container = await self._docker.containers.create(
                config=container_config,
                name=f"agent-{agent_config.agent_id}",
            )
            container_id = container.id
            self._containers[agent_config.agent_id] = container

            logger.info(
                "container_created",
                agent_id=agent_config.agent_id,
                container_id=container_id,
                philosophy=agent_config.philosophy.value,
                sandbox_enabled=sandbox_config is not None,
            )

            return container_id

        except DockerError as e:
            logger.error(
                "container_creation_failed",
                agent_id=agent_config.agent_id,
                error=str(e),
            )
            raise

    async def start_container(self, agent_id: str) -> None:
        """
        Start a created container.

        Args:
            agent_id: Agent identifier

        Raises:
            KeyError: If container not found
            DockerError: If container start fails
        """
        container = self._containers.get(agent_id)
        if not container:
            raise KeyError(f"Container for agent {agent_id} not found")

        try:
            await container.start()
            logger.info("container_started", agent_id=agent_id)
        except DockerError as e:
            logger.error(
                "container_start_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise

    async def stop_container(
        self,
        agent_id: str,
        timeout: int = 10,
    ) -> None:
        """
        Stop a running container gracefully.

        Args:
            agent_id: Agent identifier
            timeout: Seconds to wait before forcing stop

        Raises:
            KeyError: If container not found
            DockerError: If container stop fails
        """
        container = self._containers.get(agent_id)
        if not container:
            raise KeyError(f"Container for agent {agent_id} not found")

        try:
            await container.stop(timeout=timeout)
            logger.info(
                "container_stopped",
                agent_id=agent_id,
                timeout=timeout,
            )
        except DockerError as e:
            logger.error(
                "container_stop_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise

    async def remove_container(self, agent_id: str, force: bool = False) -> None:
        """
        Remove a container and cleanup resources.

        Args:
            agent_id: Agent identifier
            force: Force removal even if running

        Raises:
            KeyError: If container not found
            DockerError: If container removal fails
        """
        container = self._containers.get(agent_id)
        if not container:
            raise KeyError(f"Container for agent {agent_id} not found")

        try:
            await container.delete(force=force)
            del self._containers[agent_id]
            logger.info(
                "container_removed",
                agent_id=agent_id,
                forced=force,
            )
        except DockerError as e:
            logger.error(
                "container_removal_failed",
                agent_id=agent_id,
                error=str(e),
            )
            raise

    async def get_container_stats(self, agent_id: str) -> dict[str, Any]:
        """
        Get real-time resource usage statistics for container.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with CPU, memory, network, and I/O stats

        Raises:
            KeyError: If container not found
        """
        # Check cache first
        cache_key = f"stats:{agent_id}"
        cached_stats = self._optimizer.get_cached_pattern(cache_key)
        if cached_stats:
            return cached_stats

        container = self._containers.get(agent_id)
        if not container:
            raise KeyError(f"Container for agent {agent_id} not found")

        stats = await container.stats(stream=False)
        parsed_stats = self._parse_container_stats(stats)

        # Cache stats for 2 seconds to reduce Docker API calls
        self._optimizer.cache_execution_pattern(cache_key, parsed_stats)

        return parsed_stats

    async def get_container_logs(
        self,
        agent_id: str,
        tail: int = 100,
    ) -> str:
        """
        Get container logs.

        Args:
            agent_id: Agent identifier
            tail: Number of lines to retrieve from end

        Returns:
            Container logs as string

        Raises:
            KeyError: If container not found
        """
        container = self._containers.get(agent_id)
        if not container:
            raise KeyError(f"Container for agent {agent_id} not found")

        logs = await container.log(
            stdout=True,
            stderr=True,
            tail=tail,
        )
        return "".join(logs)

    async def container_is_running(self, agent_id: str) -> bool:
        """
        Check if container is currently running.

        Args:
            agent_id: Agent identifier

        Returns:
            True if container is running, False otherwise
        """
        container = self._containers.get(agent_id)
        if not container:
            return False

        try:
            info = await container.show()
            return info["State"]["Running"]
        except DockerError:
            return False

    def _build_container_config(
        self,
        agent_config: AgentConfig,
        sandbox_config: SandboxConfig | None = None,
    ) -> dict[str, Any]:
        """
        Build Docker container configuration from agent config.

        Args:
            agent_config: Agent configuration
            sandbox_config: Optional sandbox configuration

        Returns:
            Docker container configuration dictionary
        """
        # Resource limits (use sandbox limits if provided, otherwise agent config)
        if sandbox_config:
            memory_limit = sandbox_config.execution_limits.max_memory_mb * 1024 * 1024
            cpu_quota = int(
                (sandbox_config.execution_limits.max_cpu_percent / 100.0) * 100000
            )
            pids_limit = sandbox_config.execution_limits.max_processes
        else:
            memory_limit = agent_config.resource_limits.max_memory_mb * 1024 * 1024
            cpu_quota = int(agent_config.resource_limits.max_cpu_cores * 100000)
            pids_limit = 100

        # Environment variables
        env_vars = [f"{k}={v}" for k, v in agent_config.environment_variables.items()]
        env_vars.extend([
            f"AGENT_ID={agent_config.agent_id}",
            f"PHILOSOPHY={agent_config.philosophy.value}",
        ])

        # Add sandbox environment variables if provided
        if sandbox_config:
            env_vars.extend([
                f"{k}={v}" for k, v in sandbox_config.environment_variables.items()
            ])
            env_vars.append(f"SANDBOX_ID={sandbox_config.sandbox_id}")

        # Build security options
        security_opt = []
        if agent_config.security_profile.user_namespace:
            security_opt.append("userns-remap=default")
        if not agent_config.security_profile.no_new_privileges:
            security_opt.append("no-new-privileges")

        # Seccomp profile
        seccomp_profile = settings.seccomp_profile_path
        if agent_config.security_profile.profile_name == "minimal":
            security_opt.append(f"seccomp={seccomp_profile}")

        # Build volume mounts for sandbox read-only and writable paths
        binds = []
        if sandbox_config:
            for readonly_path in sandbox_config.read_only_paths:
                binds.append(f"{readonly_path}:{readonly_path}:ro")
            for writable_path in sandbox_config.writable_paths:
                binds.append(f"{writable_path}:{writable_path}:rw")

        # Determine network mode
        if sandbox_config and not sandbox_config.allow_network:
            network_mode = "none"
        else:
            network_mode = self._get_network_mode(
                agent_config.resource_limits.network_access
            )

        config = {
            "Image": agent_config.image_tag,
            "Env": env_vars,
            "HostConfig": {
                # Resource limits
                "Memory": memory_limit,
                "MemorySwap": memory_limit,  # Disable swap
                "CpuQuota": cpu_quota,
                "CpuPeriod": 100000,
                "PidsLimit": pids_limit,
                # Security
                "ReadonlyRootfs": agent_config.security_profile.read_only_filesystem,
                "SecurityOpt": security_opt,
                "CapDrop": ["ALL"],  # Drop all capabilities
                "Privileged": False,
                "UsernsMode": "host" if agent_config.security_profile.user_namespace else "",
                # Network
                "NetworkMode": network_mode,
                # Storage
                "StorageOpt": {
                    "size": f"{agent_config.resource_limits.storage_quota_mb}M",
                },
                # Volume binds
                "Binds": binds if binds else None,
            },
            "NetworkingConfig": {
                "EndpointsConfig": {},
            },
        }

        return config

    def _get_network_mode(self, network_access: str) -> str:
        """
        Get Docker network mode based on access level.

        Args:
            network_access: Network access level (none, restricted, full)

        Returns:
            Docker network mode string
        """
        if network_access == "none":
            return "none"
        elif network_access == "restricted":
            return "bridge"  # Isolated network with controlled access
        else:  # full
            return "host"

    def _parse_container_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """
        Parse Docker container stats into simplified metrics.

        Args:
            stats: Raw Docker stats

        Returns:
            Simplified metrics dictionary
        """
        # CPU usage calculation
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                   stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                      stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0

        # Memory usage
        memory_usage = stats["memory_stats"]["usage"]
        memory_limit = stats["memory_stats"]["limit"]
        memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0

        # Network I/O
        network_rx = 0
        network_tx = 0
        if "networks" in stats:
            for interface in stats["networks"].values():
                network_rx += interface.get("rx_bytes", 0)
                network_tx += interface.get("tx_bytes", 0)

        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
            "memory_percent": round(memory_percent, 2),
            "network_rx_mb": round(network_rx / (1024 * 1024), 2),
            "network_tx_mb": round(network_tx / (1024 * 1024), 2),
        }
