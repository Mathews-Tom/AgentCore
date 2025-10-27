"""Shared fixtures for agent runtime integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import docker
import pytest

from agentcore.agent_runtime.models.agent_config import (
    AgentConfig,
    AgentPhilosophy,
    ResourceLimits,
   SecurityProfile)
from agentcore.agent_runtime.services.a2a_client import A2AClient
from agentcore.agent_runtime.services.agent_lifecycle import AgentLifecycleManager
from agentcore.agent_runtime.services.container_manager import ContainerManager
from agentcore.agent_runtime.services.multi_agent_coordinator import (
    MultiAgentCoordinator)


@pytest.fixture
def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def docker_available() -> bool:
    """Check if Docker is available for integration tests."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture
async def real_docker_client() -> AsyncGenerator[docker.DockerClient | None, None]:
    """Provide real Docker client if available."""
    try:
        client = docker.from_env()
        client.ping()
        yield client
    except Exception:
        yield None
    finally:
        if client:
            client.close()


@pytest.fixture
def mock_docker_container():
    """Create mock Docker container."""
    container = MagicMock()
    container.id = "test-container-123"
    container.name = "test-agent-001"
    container.status = "running"
    container.attrs = {
        "State": {
            "Status": "running",
            "Running": True,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
        }
    }
    container.start = MagicMock()
    container.stop = MagicMock()
    container.pause = MagicMock()
    container.unpause = MagicMock()
    container.remove = MagicMock()
    container.reload = MagicMock()
    container.stats = MagicMock(
        return_value=iter(
            [
                {
                    "cpu_stats": {
                        "cpu_usage": {"total_usage": 1000000000},
                        "system_cpu_usage": 2000000000,
                    },
                    "precpu_stats": {
                        "cpu_usage": {"total_usage": 500000000},
                        "system_cpu_usage": 1000000000,
                    },
                    "memory_stats": {"usage": 268435456, "limit": 536870912},
                    "networks": {"eth0": {"rx_bytes": 1048576, "tx_bytes": 524288}},
                }
            ]
        )
    )
    container.logs = MagicMock(return_value=b"test logs")
    return container


@pytest.fixture
def mock_docker_client(mock_docker_container):
    """Create mock Docker client."""
    client = MagicMock(spec=docker.DockerClient)
    client.containers = MagicMock()
    client.containers.create = MagicMock(return_value=mock_docker_container)
    client.containers.get = MagicMock(return_value=mock_docker_container)
    client.containers.list = MagicMock(return_value=[mock_docker_container])
    client.images = MagicMock()
    client.images.pull = MagicMock()
    client.ping = MagicMock(return_value=True)
    client.version = MagicMock(return_value={"Version": "24.0.0"})
    return client


@pytest.fixture
async def mock_container_manager(mock_docker_client) -> ContainerManager:
    """Create mock container manager."""
    manager = MagicMock(spec=ContainerManager)
    manager.docker_client = mock_docker_client
    manager.create_container = AsyncMock(return_value="test-container-123")
    manager.start_container = AsyncMock()
    manager.stop_container = AsyncMock()
    manager.pause_container = AsyncMock()
    manager.unpause_container = AsyncMock()
    manager.remove_container = AsyncMock()
    manager.container_is_running = AsyncMock(return_value=True)
    manager.get_container_stats = AsyncMock(
        return_value={
            "cpu_percent": 50.0,
            "memory_usage_mb": 256.0,
            "memory_percent": 50.0,
            "network_rx_mb": 1.0,
            "network_tx_mb": 0.5,
        }
    )
    manager.get_container_logs = AsyncMock(return_value="test logs")
    manager.cleanup_containers = AsyncMock()
    return manager


@pytest.fixture
def mock_a2a_client() -> A2AClient:
    """Create mock A2A client."""
    client = MagicMock(spec=A2AClient)
    client.register_agent = AsyncMock(return_value="test-agent-001")
    client.unregister_agent = AsyncMock(return_value=True)
    client.update_agent_status = AsyncMock(return_value=True)
    client.report_health = AsyncMock(return_value=True)
    client.accept_task = AsyncMock(return_value=True)
    client.start_task = AsyncMock(return_value=True)
    client.complete_task = AsyncMock(return_value=True)
    client.fail_task = AsyncMock(return_value=True)
    client.ping = AsyncMock(return_value=True)
    return client


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-agent-001",
        philosophy=AgentPhilosophy.REACT,
        resource_limits=ResourceLimits(
            max_cpu_cores=0.5,
            max_memory_mb=512,
            storage_quota_mb=1024),
        security_profile=SecurityProfile(
            profile_name="standard"))


@pytest.fixture
def agent_config_cot() -> AgentConfig:
    """Create Chain-of-Thought agent configuration."""
    return AgentConfig(
        agent_id="test-agent-cot-001",
        philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT,
        resource_limits=ResourceLimits(
            max_cpu_cores=0.5,
            max_memory_mb=512,
            storage_quota_mb=1024))


@pytest.fixture
def agent_config_multi() -> AgentConfig:
    """Create multi-agent configuration."""
    return AgentConfig(
        agent_id="test-agent-multi-001",
        philosophy=AgentPhilosophy.MULTI_AGENT,
        resource_limits=ResourceLimits(
            max_cpu_cores=0.5,
            max_memory_mb=512,
            storage_quota_mb=1024))


@pytest.fixture
async def lifecycle_manager(
    mock_container_manager: ContainerManager,
    mock_a2a_client: A2AClient) -> AgentLifecycleManager:
    """Create lifecycle manager with mocks."""
    return AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client)


@pytest.fixture
async def multi_agent_coordinator() -> MultiAgentCoordinator:
    """Create multi-agent coordinator."""
    return MultiAgentCoordinator()


@pytest.fixture
def test_task_data() -> dict[str, Any]:
    """Create test task data."""
    return {
        "task_id": "test-task-001",
        "input": "Test input for agent execution",
        "parameters": {
            "max_steps": 5,
            "timeout": 30,
        },
        "expected_output_format": "json",
    }
