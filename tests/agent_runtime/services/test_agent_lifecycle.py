"""Tests for agent lifecycle manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.services.agent_lifecycle import (
    AgentLifecycleError,
    AgentLifecycleManager,
    AgentNotFoundException,
    AgentStateError)
from agentcore.agent_runtime.services.container_manager import ContainerManager


@pytest.fixture
def mock_container_manager() -> ContainerManager:
    """Create mock container manager."""
    manager = MagicMock(spec=ContainerManager)
    manager.create_container = AsyncMock(return_value="test-container-id")
    manager.start_container = AsyncMock()
    manager.stop_container = AsyncMock()
    manager.remove_container = AsyncMock()
    manager.container_is_running = AsyncMock(return_value=True)
    manager.get_container_stats = AsyncMock(return_value={
        "cpu_percent": 50.0,
        "memory_usage_mb": 256.0,
        "memory_percent": 50.0,
        "network_rx_mb": 1.0,
        "network_tx_mb": 0.5,
    })
    return manager


@pytest.fixture
def lifecycle_manager(mock_container_manager: ContainerManager) -> AgentLifecycleManager:
    """Create lifecycle manager with mocked container manager."""
    return AgentLifecycleManager(mock_container_manager)


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-agent-001",
        philosophy=AgentPhilosophy.REACT)


@pytest.mark.asyncio
async def test_create_agent(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test agent creation."""
    state = await lifecycle_manager.create_agent(agent_config)

    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"
    assert state.container_id == "test-container-id"


@pytest.mark.asyncio
async def test_create_duplicate_agent(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test creating duplicate agent fails."""
    await lifecycle_manager.create_agent(agent_config)

    with pytest.raises(AgentLifecycleError, match="already exists"):
        await lifecycle_manager.create_agent(agent_config)


@pytest.mark.asyncio
async def test_start_agent(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test starting agent."""
    state = await lifecycle_manager.create_agent(agent_config)
    assert state.status == "initializing"

    await lifecycle_manager.start_agent(agent_config.agent_id)

    updated_state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert updated_state.status == "running"


@pytest.mark.asyncio
async def test_start_nonexistent_agent(
    lifecycle_manager: AgentLifecycleManager) -> None:
    """Test starting nonexistent agent fails."""
    with pytest.raises(AgentNotFoundException):
        await lifecycle_manager.start_agent("nonexistent-agent")


@pytest.mark.asyncio
async def test_pause_agent(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test pausing agent."""
    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    await lifecycle_manager.pause_agent(agent_config.agent_id)

    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.status == "paused"


@pytest.mark.asyncio
async def test_pause_agent_invalid_state(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test pausing agent in invalid state fails."""
    await lifecycle_manager.create_agent(agent_config)

    with pytest.raises(AgentStateError, match="Cannot pause agent"):
        await lifecycle_manager.pause_agent(agent_config.agent_id)


@pytest.mark.asyncio
async def test_terminate_agent(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test terminating agent."""
    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    await lifecycle_manager.terminate_agent(agent_config.agent_id, cleanup=True)

    # Agent should be removed from tracking
    with pytest.raises(AgentNotFoundException):
        await lifecycle_manager.get_agent_status(agent_config.agent_id)


@pytest.mark.asyncio
async def test_list_agents(
    lifecycle_manager: AgentLifecycleManager) -> None:
    """Test listing agents."""
    # Create multiple agents
    for i in range(3):
        config = AgentConfig(
            agent_id=f"test-agent-{i}",
            philosophy=AgentPhilosophy.REACT)
        await lifecycle_manager.create_agent(config)

    agents = await lifecycle_manager.list_agents()
    assert len(agents) == 3


@pytest.mark.asyncio
async def test_save_and_restore_checkpoint(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test checkpoint save and restore."""
    await lifecycle_manager.create_agent(agent_config)

    # Save checkpoint
    checkpoint_data = b"test checkpoint data"
    await lifecycle_manager.save_checkpoint(agent_config.agent_id, checkpoint_data)

    # Restore checkpoint
    restored = await lifecycle_manager.restore_checkpoint(agent_config.agent_id)
    assert restored == checkpoint_data


@pytest.mark.asyncio
async def test_update_agent_metrics(
    lifecycle_manager: AgentLifecycleManager,
    agent_config: AgentConfig) -> None:
    """Test updating agent metrics."""
    await lifecycle_manager.create_agent(agent_config)

    metrics = {
        "cpu_percent": 75.0,
        "memory_usage_mb": 384.0,
    }
    await lifecycle_manager.update_agent_metrics(agent_config.agent_id, metrics)

    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.performance_metrics["cpu_percent"] == 75.0
    assert state.performance_metrics["memory_usage_mb"] == 384.0
