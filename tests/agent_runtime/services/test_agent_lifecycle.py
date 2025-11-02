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


# Additional tests for A2A protocol integration


@pytest.mark.asyncio
async def test_create_agent_with_a2a_client(
    mock_container_manager: ContainerManager,
    agent_config: AgentConfig) -> None:
    """Test agent creation with A2A protocol client."""
    mock_a2a_client = MagicMock()
    mock_a2a_client.register_agent = AsyncMock()

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    state = await lifecycle_manager.create_agent(agent_config)

    assert state.agent_id == "test-agent-001"
    mock_a2a_client.register_agent.assert_called_once_with(agent_config)


@pytest.mark.asyncio
async def test_create_agent_a2a_registration_fails(
    mock_container_manager: ContainerManager,
    agent_config: AgentConfig) -> None:
    """Test agent creation when A2A registration fails."""
    from agentcore.agent_runtime.services.a2a_client import A2ARegistrationError

    mock_a2a_client = MagicMock()
    mock_a2a_client.register_agent = AsyncMock(
        side_effect=A2ARegistrationError("Registration failed"))

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    # Should still create agent even if A2A registration fails
    state = await lifecycle_manager.create_agent(agent_config)
    assert state.agent_id == "test-agent-001"
    assert state.container_id == "test-container-id"


@pytest.mark.asyncio
async def test_create_agent_container_creation_fails(
    agent_config: AgentConfig) -> None:
    """Test agent creation when container creation fails."""
    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(
        side_effect=Exception("Container creation failed"))

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    with pytest.raises(AgentLifecycleError, match="Failed to create agent"):
        await lifecycle_manager.create_agent(agent_config)


@pytest.mark.asyncio
async def test_start_agent_with_a2a_status_update(
    mock_container_manager: ContainerManager,
    agent_config: AgentConfig) -> None:
    """Test starting agent with A2A status update."""
    mock_a2a_client = MagicMock()
    mock_a2a_client.update_agent_status = AsyncMock()

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    mock_a2a_client.update_agent_status.assert_called_once_with(
        agent_id=agent_config.agent_id,
        status="active")


@pytest.mark.asyncio
async def test_start_agent_container_start_fails(
    agent_config: AgentConfig) -> None:
    """Test starting agent when container start fails."""
    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock(
        side_effect=Exception("Container start failed"))

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    await lifecycle_manager.create_agent(agent_config)

    with pytest.raises(Exception, match="Container start failed"):
        await lifecycle_manager.start_agent(agent_config.agent_id)

    # Verify agent status was set to failed
    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.status == "failed"
    assert "Container start failed" in state.failure_reason


@pytest.mark.asyncio
async def test_pause_agent_error_handling(
    agent_config: AgentConfig) -> None:
    """Test pause agent error handling."""
    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock()
    mock_container_manager.stop_container = AsyncMock(
        side_effect=Exception("Container stop failed"))
    mock_container_manager.container_is_running = AsyncMock(return_value=True)
    mock_container_manager.get_container_stats = AsyncMock(return_value={})

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    with pytest.raises(Exception, match="Container stop failed"):
        await lifecycle_manager.pause_agent(agent_config.agent_id)


@pytest.mark.asyncio
async def test_terminate_agent_with_a2a_unregister(
    mock_container_manager: ContainerManager,
    agent_config: AgentConfig) -> None:
    """Test terminating agent with A2A unregistration."""
    mock_a2a_client = MagicMock()
    mock_a2a_client.register_agent = AsyncMock()
    mock_a2a_client.unregister_agent = AsyncMock()

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.terminate_agent(agent_config.agent_id, cleanup=True)

    mock_a2a_client.unregister_agent.assert_called_once_with(agent_config.agent_id)


@pytest.mark.asyncio
async def test_terminate_agent_a2a_unregister_fails(
    mock_container_manager: ContainerManager,
    agent_config: AgentConfig) -> None:
    """Test terminating agent when A2A unregistration fails."""
    mock_a2a_client = MagicMock()
    mock_a2a_client.register_agent = AsyncMock()
    mock_a2a_client.unregister_agent = AsyncMock(
        side_effect=Exception("Unregistration failed"))

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    await lifecycle_manager.create_agent(agent_config)

    # Should still terminate even if A2A unregistration fails
    await lifecycle_manager.terminate_agent(agent_config.agent_id, cleanup=True)

    # Agent should be removed
    with pytest.raises(AgentNotFoundException):
        await lifecycle_manager.get_agent_status(agent_config.agent_id)


@pytest.mark.asyncio
async def test_terminate_agent_error_handling(
    agent_config: AgentConfig) -> None:
    """Test terminate agent error handling."""
    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.stop_container = AsyncMock(
        side_effect=Exception("Container stop failed"))

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    await lifecycle_manager.create_agent(agent_config)

    with pytest.raises(Exception, match="Container stop failed"):
        await lifecycle_manager.terminate_agent(agent_config.agent_id)


# Additional tests for agent monitoring


@pytest.mark.asyncio
async def test_agent_monitoring_completes_successfully(
    agent_config: AgentConfig) -> None:
    """Test agent monitoring when container completes."""
    import asyncio

    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock()

    # Simulate container running then completing
    call_count = [0]

    async def mock_is_running(agent_id):
        call_count[0] += 1
        return call_count[0] == 1  # First call: running, second call: stopped

    mock_container_manager.container_is_running = mock_is_running
    mock_container_manager.get_container_stats = AsyncMock(return_value={
        "cpu_percent": 50.0,
        "memory_usage_mb": 256.0,
    })

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    # Give monitoring task time to detect completion (monitoring checks every 5 seconds)
    # Need to wait longer than one check cycle
    await asyncio.sleep(6.0)

    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.status == "completed"


@pytest.mark.asyncio
async def test_agent_monitoring_with_a2a_health_report(
    agent_config: AgentConfig) -> None:
    """Test agent monitoring with A2A health reporting."""
    import asyncio

    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock()

    # Keep running for one cycle
    call_count = [0]

    async def mock_is_running(agent_id):
        call_count[0] += 1
        return call_count[0] == 1

    mock_container_manager.container_is_running = mock_is_running
    mock_container_manager.get_container_stats = AsyncMock(return_value={
        "cpu_percent": 75.0,
        "memory_usage_mb": 512.0,
    })

    mock_a2a_client = MagicMock()
    mock_a2a_client.register_agent = AsyncMock()
    mock_a2a_client.report_health = AsyncMock()

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    # Give monitoring task time to run
    await asyncio.sleep(0.2)

    # Verify health was reported
    assert mock_a2a_client.report_health.called


@pytest.mark.asyncio
async def test_agent_monitoring_health_report_fails(
    agent_config: AgentConfig) -> None:
    """Test agent monitoring when health reporting fails."""
    import asyncio

    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock()

    call_count = [0]

    async def mock_is_running(agent_id):
        call_count[0] += 1
        return call_count[0] == 1

    mock_container_manager.container_is_running = mock_is_running
    mock_container_manager.get_container_stats = AsyncMock(return_value={})

    mock_a2a_client = MagicMock()
    mock_a2a_client.register_agent = AsyncMock()
    mock_a2a_client.report_health = AsyncMock(
        side_effect=Exception("Health report failed"))

    lifecycle_manager = AgentLifecycleManager(
        mock_container_manager,
        a2a_client=mock_a2a_client)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    # Give monitoring task time to run - should not crash
    # Monitoring checks every 5 seconds
    await asyncio.sleep(6.0)

    # Agent should still complete successfully
    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.status == "completed"


@pytest.mark.asyncio
async def test_agent_monitoring_metrics_update_fails(
    agent_config: AgentConfig) -> None:
    """Test agent monitoring when metrics update fails."""
    import asyncio

    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock()

    call_count = [0]

    async def mock_is_running(agent_id):
        call_count[0] += 1
        return call_count[0] == 1

    mock_container_manager.container_is_running = mock_is_running
    mock_container_manager.get_container_stats = AsyncMock(
        side_effect=Exception("Stats fetch failed"))

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    # Give monitoring task time to run - should not crash
    # Monitoring checks every 5 seconds
    await asyncio.sleep(6.0)

    # Agent should still complete successfully despite metrics failure
    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.status == "completed"


@pytest.mark.asyncio
async def test_agent_monitoring_exception_handling(
    agent_config: AgentConfig) -> None:
    """Test agent monitoring when unexpected exception occurs."""
    import asyncio

    mock_container_manager = MagicMock(spec=ContainerManager)
    mock_container_manager.create_container = AsyncMock(return_value="test-container-id")
    mock_container_manager.start_container = AsyncMock()
    mock_container_manager.container_is_running = AsyncMock(
        side_effect=Exception("Monitoring error"))

    lifecycle_manager = AgentLifecycleManager(mock_container_manager)

    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent(agent_config.agent_id)

    # Give monitoring task time to fail
    await asyncio.sleep(0.2)

    # Agent should be marked as failed
    state = await lifecycle_manager.get_agent_status(agent_config.agent_id)
    assert state.status == "failed"
    assert "Monitoring error" in state.failure_reason
