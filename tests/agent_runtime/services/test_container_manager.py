"""Tests for container manager service."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from aiodocker.exceptions import DockerError

from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy, ResourceLimits
from agentcore.agent_runtime.models.sandbox import ExecutionLimits, SandboxConfig
from agentcore.agent_runtime.services.container_manager import ContainerManager


@pytest.fixture
def mock_docker_container() -> Mock:
    """Create mock aiodocker container."""
    container = Mock()
    container.id = "test-container-123"
    container.start = AsyncMock()
    container.stop = AsyncMock()
    container.delete = AsyncMock()
    container.show = AsyncMock(return_value={
        "State": {
            "Running": True,
            "Status": "running",
        }
    })
    container.stats = AsyncMock(return_value={
        "cpu_stats": {
            "cpu_usage": {"total_usage": 2000000000},
            "system_cpu_usage": 4000000000,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 1000000000},
            "system_cpu_usage": 2000000000,
        },
        "memory_stats": {
            "usage": 268435456,
            "limit": 536870912,
        },
        "networks": {
            "eth0": {
                "rx_bytes": 1048576,
                "tx_bytes": 524288,
            }
        },
    })
    container.log = AsyncMock(return_value=[
        "line 1\n",
        "line 2\n",
        "line 3\n",
    ])
    return container


@pytest.fixture
def mock_docker_client(mock_docker_container: Mock) -> Mock:
    """Create mock aiodocker client."""
    client = Mock()
    client.containers = Mock()
    client.containers.create = AsyncMock(return_value=mock_docker_container)
    client.close = AsyncMock()
    return client


@pytest.fixture
async def container_manager(mock_docker_client: Mock, monkeypatch: pytest.MonkeyPatch) -> ContainerManager:
    """Create container manager with mocked Docker client."""
    manager = ContainerManager()

    # Mock aiodocker.Docker() to return our mock client
    async def mock_docker_init() -> Mock:
        return mock_docker_client

    monkeypatch.setattr("aiodocker.Docker", lambda: mock_docker_client)

    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-agent-001",
        philosophy=AgentPhilosophy.REACT,
        image_tag="python:3.12-slim",
        resource_limits=ResourceLimits(
            max_cpu_percent=50.0,
            max_memory_mb=512,
            max_cpu_cores=2.0,
            network_access="none",
            storage_quota_mb=1024))


@pytest.fixture
def sandbox_config() -> SandboxConfig:
    """Create test sandbox configuration."""
    return SandboxConfig(
        agent_id="test-agent-001",
        sandbox_id="sandbox-001",
        allow_network=False,
        environment_variables={"SANDBOX": "enabled"},
        execution_limits=ExecutionLimits(
            max_memory_mb=256,
            max_cpu_percent=25.0,
            max_execution_time_seconds=60,
            max_processes=50),
        read_only_paths=["/usr/lib"],
        writable_paths=["/tmp"])


@pytest.mark.asyncio
class TestContainerManager:
    """Test container manager functionality."""

    async def test_initialize(self, mock_docker_client: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test container manager initialization."""
        monkeypatch.setattr("aiodocker.Docker", lambda: mock_docker_client)

        manager = ContainerManager()
        await manager.initialize()

        assert manager._docker is not None

        await manager.close()

    async def test_initialize_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization failure handling."""

        def mock_docker_error() -> None:
            raise DockerError(status=500, data={"message": "Docker daemon not available"})

        monkeypatch.setattr("aiodocker.Docker", mock_docker_error)

        manager = ContainerManager()

        with pytest.raises(DockerError):
            await manager.initialize()

    async def test_create_container(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test container creation."""
        container_id = await container_manager.create_container(agent_config)

        assert container_id == "test-container-123"
        assert agent_config.agent_id in container_manager._containers
        assert container_manager._containers[agent_config.agent_id] == mock_docker_container

    async def test_create_container_with_sandbox(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        sandbox_config: SandboxConfig,
        mock_docker_container: Mock) -> None:
        """Test container creation with sandbox configuration."""
        container_id = await container_manager.create_container(agent_config, sandbox_config)

        assert container_id == "test-container-123"
        assert agent_config.agent_id in container_manager._containers

    async def test_create_container_not_initialized(self, agent_config: AgentConfig) -> None:
        """Test container creation fails if manager not initialized."""
        manager = ContainerManager()

        with pytest.raises(RuntimeError, match="Container manager not initialized"):
            await manager.create_container(agent_config)

    async def test_create_container_docker_error(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_client: Mock) -> None:
        """Test container creation with Docker error."""
        mock_docker_client.containers.create = AsyncMock(
            side_effect=DockerError(status=500, data={"message": "Image not found"})
        )

        with pytest.raises(DockerError):
            await container_manager.create_container(agent_config)

    async def test_start_container(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test starting a container."""
        await container_manager.create_container(agent_config)
        await container_manager.start_container(agent_config.agent_id)

        mock_docker_container.start.assert_called_once()

    async def test_start_container_not_found(self, container_manager: ContainerManager) -> None:
        """Test starting non-existent container."""
        with pytest.raises(KeyError, match="Container for agent nonexistent not found"):
            await container_manager.start_container("nonexistent")

    async def test_start_container_docker_error(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test container start with Docker error."""
        await container_manager.create_container(agent_config)
        mock_docker_container.start = AsyncMock(
            side_effect=DockerError(status=500, data={"message": "Start failed"})
        )

        with pytest.raises(DockerError):
            await container_manager.start_container(agent_config.agent_id)

    async def test_stop_container(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test stopping a container."""
        await container_manager.create_container(agent_config)
        await container_manager.stop_container(agent_config.agent_id, timeout=5)

        mock_docker_container.stop.assert_called_once_with(timeout=5)

    async def test_stop_container_not_found(self, container_manager: ContainerManager) -> None:
        """Test stopping non-existent container."""
        with pytest.raises(KeyError, match="Container for agent nonexistent not found"):
            await container_manager.stop_container("nonexistent")

    async def test_stop_container_docker_error(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test container stop with Docker error."""
        await container_manager.create_container(agent_config)
        mock_docker_container.stop = AsyncMock(
            side_effect=DockerError(status=500, data={"message": "Stop failed"})
        )

        with pytest.raises(DockerError):
            await container_manager.stop_container(agent_config.agent_id)

    async def test_remove_container(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test removing a container."""
        await container_manager.create_container(agent_config)
        await container_manager.remove_container(agent_config.agent_id, force=True)

        mock_docker_container.delete.assert_called_once_with(force=True)
        assert agent_config.agent_id not in container_manager._containers

    async def test_remove_container_not_found(self, container_manager: ContainerManager) -> None:
        """Test removing non-existent container."""
        with pytest.raises(KeyError, match="Container for agent nonexistent not found"):
            await container_manager.remove_container("nonexistent")

    async def test_remove_container_docker_error(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test container removal with Docker error."""
        await container_manager.create_container(agent_config)
        mock_docker_container.delete = AsyncMock(
            side_effect=DockerError(status=500, data={"message": "Remove failed"})
        )

        with pytest.raises(DockerError):
            await container_manager.remove_container(agent_config.agent_id)

    async def test_get_container_stats(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig) -> None:
        """Test getting container stats."""
        await container_manager.create_container(agent_config)
        stats = await container_manager.get_container_stats(agent_config.agent_id)

        assert "cpu_percent" in stats
        assert "memory_usage_mb" in stats
        assert "memory_percent" in stats
        assert "network_rx_mb" in stats
        assert "network_tx_mb" in stats

        assert stats["cpu_percent"] == 50.0
        assert stats["memory_usage_mb"] == 256.0
        assert stats["memory_percent"] == 50.0

    async def test_get_container_stats_not_found(self, container_manager: ContainerManager) -> None:
        """Test getting stats for non-existent container."""
        with pytest.raises(KeyError, match="Container for agent nonexistent not found"):
            await container_manager.get_container_stats("nonexistent")

    async def test_get_container_logs(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig) -> None:
        """Test getting container logs."""
        await container_manager.create_container(agent_config)
        logs = await container_manager.get_container_logs(agent_config.agent_id, tail=50)

        assert isinstance(logs, str)
        assert "line 1" in logs
        assert "line 2" in logs
        assert "line 3" in logs

    async def test_get_container_logs_not_found(self, container_manager: ContainerManager) -> None:
        """Test getting logs for non-existent container."""
        with pytest.raises(KeyError, match="Container for agent nonexistent not found"):
            await container_manager.get_container_logs("nonexistent")

    async def test_container_is_running(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig) -> None:
        """Test checking if container is running."""
        await container_manager.create_container(agent_config)
        is_running = await container_manager.container_is_running(agent_config.agent_id)

        assert is_running is True

    async def test_container_is_running_not_found(self, container_manager: ContainerManager) -> None:
        """Test checking non-existent container."""
        is_running = await container_manager.container_is_running("nonexistent")
        assert is_running is False

    async def test_container_is_running_docker_error(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        mock_docker_container: Mock) -> None:
        """Test checking container with Docker error."""
        await container_manager.create_container(agent_config)
        mock_docker_container.show = AsyncMock(
            side_effect=DockerError(status=404, data={"message": "Not found"})
        )

        is_running = await container_manager.container_is_running(agent_config.agent_id)
        assert is_running is False

    async def test_build_container_config(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig) -> None:
        """Test building container configuration."""
        config = container_manager._build_container_config(agent_config)

        assert config["Image"] == "python:3.12-slim"
        assert "AGENT_ID=test-agent-001" in config["Env"]
        assert "PHILOSOPHY=react" in config["Env"]
        assert config["HostConfig"]["Memory"] == 512 * 1024 * 1024
        assert config["HostConfig"]["Privileged"] is False
        assert config["HostConfig"]["NetworkMode"] == "none"

    async def test_build_container_config_with_sandbox(
        self,
        container_manager: ContainerManager,
        agent_config: AgentConfig,
        sandbox_config: SandboxConfig) -> None:
        """Test building container configuration with sandbox."""
        config = container_manager._build_container_config(agent_config, sandbox_config)

        assert config["HostConfig"]["Memory"] == 256 * 1024 * 1024  # From sandbox
        assert config["HostConfig"]["PidsLimit"] == 50  # From sandbox
        assert "SANDBOX_ID=sandbox-001" in config["Env"]
        assert "SANDBOX=enabled" in config["Env"]
        assert config["HostConfig"]["NetworkMode"] == "none"

        # Check volume binds
        binds = config["HostConfig"]["Binds"]
        assert "/usr/lib:/usr/lib:ro" in binds
        assert "/tmp:/tmp:rw" in binds

    async def test_get_network_mode(self, container_manager: ContainerManager) -> None:
        """Test network mode determination."""
        assert container_manager._get_network_mode("none") == "none"
        assert container_manager._get_network_mode("restricted") == "bridge"
        assert container_manager._get_network_mode("full") == "host"

    async def test_parse_container_stats(self, container_manager: ContainerManager) -> None:
        """Test parsing container stats."""
        raw_stats = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2000000000},
                "system_cpu_usage": 4000000000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000000000},
                "system_cpu_usage": 2000000000,
            },
            "memory_stats": {
                "usage": 536870912,
                "limit": 1073741824,
            },
            "networks": {
                "eth0": {"rx_bytes": 2097152, "tx_bytes": 1048576}
            },
        }

        parsed = container_manager._parse_container_stats(raw_stats)

        assert parsed["cpu_percent"] == 50.0
        assert parsed["memory_usage_mb"] == 512.0
        assert parsed["memory_percent"] == 50.0
        assert parsed["network_rx_mb"] == 2.0
        assert parsed["network_tx_mb"] == 1.0

    async def test_parse_container_stats_no_network(self, container_manager: ContainerManager) -> None:
        """Test parsing container stats without network data."""
        raw_stats = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1000000000},
                "system_cpu_usage": 2000000000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 500000000},
                "system_cpu_usage": 1000000000,
            },
            "memory_stats": {
                "usage": 268435456,
                "limit": 536870912,
            },
        }

        parsed = container_manager._parse_container_stats(raw_stats)

        assert parsed["network_rx_mb"] == 0.0
        assert parsed["network_tx_mb"] == 0.0

    async def test_close(
        self,
        container_manager: ContainerManager,
        mock_docker_client: Mock) -> None:
        """Test closing container manager."""
        await container_manager.close()
        mock_docker_client.close.assert_called_once()
