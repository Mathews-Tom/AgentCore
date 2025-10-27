"""Shared fixtures for agent runtime tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import docker
import pytest
from docker.models.containers import Container

from agentcore.agent_runtime.models.agent_config import (
    AgentConfig,
    AgentPhilosophy,
    ResourceLimits,
    SecurityProfile)
from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.agent_runtime.models.plugin import (
    PluginConfig,
    PluginMetadata,
    PluginState,
    PluginStatus,
    PluginType)
from agentcore.agent_runtime.models.sandbox import (
    AuditEventType,
    AuditLogEntry,
    ExecutionLimits,
    SandboxConfig)
from agentcore.agent_runtime.models.tool_integration import ToolDefinition
from agentcore.agent_runtime.services.a2a_client import A2AClient
from agentcore.agent_runtime.services.container_manager import ContainerManager

# ============================================================================
# Agent Configuration Fixtures
# ============================================================================


@pytest.fixture
def agent_id() -> str:
    """Return test agent ID."""
    return "test-agent-001"


@pytest.fixture
def resource_limits() -> ResourceLimits:
    """Return default resource limits for testing."""
    return ResourceLimits(
        max_cpu_cores=0.5,
        max_memory_mb=512,
        storage_quota_mb=1024)


@pytest.fixture
def security_profile() -> SecurityProfile:
    """Return default security profile for testing."""
    return SecurityProfile(
        profile_name="standard")


@pytest.fixture
def agent_config(
    agent_id: str, resource_limits: ResourceLimits, security_profile: SecurityProfile
) -> AgentConfig:
    """Return test agent configuration."""
    return AgentConfig(
        agent_id=agent_id,
        philosophy=AgentPhilosophy.REACT,
        resource_limits=resource_limits,
        security_profile=security_profile)


@pytest.fixture
def agent_config_cot(agent_id: str) -> AgentConfig:
    """Return Chain-of-Thought agent configuration."""
    return AgentConfig(
        agent_id=f"{agent_id}-cot",
        philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT)


@pytest.fixture
def agent_config_multi_agent(agent_id: str) -> AgentConfig:
    """Return multi-agent configuration."""
    return AgentConfig(
        agent_id=f"{agent_id}-multi",
        philosophy=AgentPhilosophy.MULTI_AGENT)


@pytest.fixture
def agent_config_autonomous(agent_id: str) -> AgentConfig:
    """Return autonomous agent configuration."""
    return AgentConfig(
        agent_id=f"{agent_id}-autonomous",
        philosophy=AgentPhilosophy.AUTONOMOUS)


@pytest.fixture
def agent_state(agent_id: str) -> AgentExecutionState:
    """Return test agent execution state."""
    return AgentExecutionState(
        agent_id=agent_id,
        status="running",
        container_id="test-container-123")


# ============================================================================
# Tool & Plugin Fixtures
# ============================================================================


@pytest.fixture
def tool_definition() -> ToolDefinition:
    """Return test tool definition."""
    return ToolDefinition(
        name="test_tool",
        description="A test tool for testing",
        parameters={
            "input": {
                "type": "string",
                "description": "Input parameter",
            }
        },
        returns={
            "type": "string",
            "description": "Output result",
        })


@pytest.fixture
def plugin_metadata() -> PluginMetadata:
    """Return test plugin metadata."""
    return PluginMetadata(
        plugin_id="com.test.plugin",
        name="test_plugin",
        version="1.0.0",
        description="Test plugin for unit tests",
        author="Test Author",
        license="MIT",
        plugin_type=PluginType.TOOL,
        entry_point="test_plugin.main")


@pytest.fixture
def plugin_config() -> PluginConfig:
    """Return test plugin configuration."""
    return PluginConfig(
        plugin_id="com.test.plugin",
        enabled=True,
        auto_load=False,
        priority=100,
        config={})


@pytest.fixture
def plugin_state(
    plugin_metadata: PluginMetadata, plugin_config: PluginConfig
) -> PluginState:
    """Return test plugin state."""
    return PluginState(
        plugin_id="com.test.plugin",
        status=PluginStatus.LOADED,
        metadata=plugin_metadata,
        config=plugin_config,
        load_time=datetime.now(UTC))


# ============================================================================
# Sandbox & Security Fixtures
# ============================================================================


@pytest.fixture
def sandbox_config() -> SandboxConfig:
    """Return test sandbox configuration."""
    return SandboxConfig(
        sandbox_id="sandbox-001",
        agent_id="test-agent-001",
        allow_network=False,
        allowed_hosts=[],
        environment_variables={},
        execution_limits=ExecutionLimits(
            max_memory_mb=512,
            max_cpu_percent=50.0,
            max_execution_time_seconds=60,
            max_processes=100))


@pytest.fixture
def audit_log() -> AuditLogEntry:
    """Return test audit log entry."""
    return AuditLogEntry(
        event_type=AuditEventType.PERMISSION_DENIED,
        sandbox_id="sandbox-001",
        agent_id="test-agent-001",
        operation="network_request",
        resource="https://example.com",
        result=False,
        reason="Network access not allowed")


# ============================================================================
# Mock Docker Fixtures
# ============================================================================


@pytest.fixture
def mock_docker_container() -> Mock:
    """Return mock Docker container."""
    container = Mock(spec=Container)
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

    # Mock methods
    container.start = Mock()
    container.stop = Mock()
    container.pause = Mock()
    container.unpause = Mock()
    container.remove = Mock()
    container.reload = Mock()
    container.stats = Mock(
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
    container.logs = Mock(return_value=b"test logs")

    return container


@pytest.fixture
def mock_docker_client(mock_docker_container: Mock) -> Mock:
    """Return mock Docker client."""
    client = Mock(spec=docker.DockerClient)
    client.containers = Mock()
    client.containers.create = Mock(return_value=mock_docker_container)
    client.containers.get = Mock(return_value=mock_docker_container)
    client.containers.list = Mock(return_value=[mock_docker_container])
    client.images = Mock()
    client.images.pull = Mock()
    client.ping = Mock(return_value=True)
    client.version = Mock(return_value={"Version": "24.0.0"})

    return client


@pytest.fixture
def mock_container_manager(mock_docker_client: Mock) -> ContainerManager:
    """Return mock container manager."""
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


# ============================================================================
# Mock A2A Client Fixtures
# ============================================================================


@pytest.fixture
def mock_a2a_response() -> dict[str, Any]:
    """Return mock A2A JSON-RPC response."""
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
        "result": {
            "agent_id": "test-agent-001",
            "status": "success",
        },
    }


@pytest.fixture
def mock_a2a_client(mock_a2a_response: dict[str, Any]) -> A2AClient:
    """Return mock A2A client."""
    client = MagicMock(spec=A2AClient)
    client.register_agent = AsyncMock(return_value=mock_a2a_response)
    client.send_message = AsyncMock(return_value=mock_a2a_response)
    client.create_task = AsyncMock(return_value=mock_a2a_response)
    client.get_task_status = AsyncMock(return_value=mock_a2a_response)
    client.update_task_status = AsyncMock(return_value=mock_a2a_response)
    client.list_agents = AsyncMock(return_value=mock_a2a_response)
    client.get_agent_card = AsyncMock(return_value=mock_a2a_response)
    client.health_check = AsyncMock(return_value=True)

    return client


# ============================================================================
# Async Utilities
# ============================================================================


@pytest.fixture
def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def test_execution_input() -> dict[str, Any]:
    """Return test execution input data."""
    return {
        "goal": "Test goal for agent execution",
        "max_steps": 5,
        "context": {},
    }


@pytest.fixture
def test_execution_result() -> dict[str, Any]:
    """Return test execution result data."""
    return {
        "completed": True,
        "steps": 3,
        "result": "Test execution completed successfully",
        "performance": {
            "execution_time_ms": 150.0,
            "memory_used_mb": 128.0,
        },
    }
