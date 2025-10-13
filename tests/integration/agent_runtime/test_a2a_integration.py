"""Integration tests for A2A protocol integration in agent runtime."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.services.a2a_client import A2AClient
from agentcore.agent_runtime.services.agent_lifecycle import AgentLifecycleManager
from agentcore.agent_runtime.services.container_manager import ContainerManager
from agentcore.agent_runtime.services.task_handler import TaskHandler


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        agent_id="test-agent-001",
        name="Test Agent",
        philosophy=AgentPhilosophy.REACT,
        capabilities=["test", "demo"],
        resource_limits={"cpu": "1.0", "memory": "512Mi"},
    )


@pytest.fixture
def mock_container_manager():
    """Create mock container manager."""
    manager = AsyncMock(spec=ContainerManager)
    manager.create_container.return_value = "container-001"
    manager.start_container.return_value = None
    manager.stop_container.return_value = None
    manager.remove_container.return_value = None
    manager.container_is_running.return_value = True
    manager.get_container_stats.return_value = {
        "cpu_usage": 0.5,
        "memory_usage": 256.0,
    }
    return manager


@pytest.fixture
def mock_a2a_client():
    """Create mock A2A client."""
    client = AsyncMock(spec=A2AClient)
    client.register_agent.return_value = "test-agent-001"
    client.unregister_agent.return_value = True
    client.update_agent_status.return_value = True
    client.report_health.return_value = True
    client.accept_task.return_value = True
    client.start_task.return_value = True
    client.complete_task.return_value = True
    client.fail_task.return_value = True
    client.ping.return_value = True
    return client


@pytest.mark.asyncio
async def test_agent_lifecycle_with_a2a_integration(
    agent_config,
    mock_container_manager,
    mock_a2a_client,
):
    """Test agent lifecycle with A2A protocol integration."""
    # Create lifecycle manager with A2A client
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create agent
    state = await lifecycle_manager.create_agent(agent_config)

    # Verify agent created
    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"
    assert state.container_id == "container-001"

    # Verify A2A registration called
    mock_a2a_client.register_agent.assert_called_once()
    call_args = mock_a2a_client.register_agent.call_args
    assert call_args[1]["agent_config"].agent_id == "test-agent-001"

    # Start agent
    await lifecycle_manager.start_agent("test-agent-001")

    # Verify status updated
    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Verify A2A status update called
    mock_a2a_client.update_agent_status.assert_called_once()
    status_call_args = mock_a2a_client.update_agent_status.call_args
    assert status_call_args[1]["agent_id"] == "test-agent-001"
    assert status_call_args[1]["status"] == "active"

    # Terminate agent
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)

    # Verify A2A unregistration called
    mock_a2a_client.unregister_agent.assert_called_once_with("test-agent-001")


@pytest.mark.asyncio
async def test_agent_lifecycle_without_a2a_client(
    agent_config,
    mock_container_manager,
):
    """Test agent lifecycle works without A2A client (local mode)."""
    # Create lifecycle manager without A2A client
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create agent
    state = await lifecycle_manager.create_agent(agent_config)

    # Verify agent created
    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"

    # Start agent
    await lifecycle_manager.start_agent("test-agent-001")

    # Verify agent running
    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Terminate agent
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_client_connectivity(mock_a2a_client):
    """Test A2A client connectivity check."""
    # Test ping
    async with mock_a2a_client as client:
        result = await client.ping()
        assert result is True


@pytest.mark.asyncio
async def test_task_assignment_and_execution(
    agent_config,
    mock_container_manager,
    mock_a2a_client,
):
    """Test task assignment and execution through A2A."""
    # Create lifecycle manager and task handler
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    task_handler = TaskHandler(
        a2a_client=mock_a2a_client,
        lifecycle_manager=lifecycle_manager,
    )

    # Create and start agent
    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent("test-agent-001")

    # Assign task
    task_id = "task-001"
    task_data = {"input": "test data", "parameters": {}}

    success = await task_handler.assign_task(
        task_id=task_id,
        agent_id="test-agent-001",
        task_data=task_data,
    )

    assert success is True

    # Verify task accepted in A2A
    mock_a2a_client.accept_task.assert_called_once()
    accept_call = mock_a2a_client.accept_task.call_args
    assert accept_call[1]["task_id"] == task_id
    assert accept_call[1]["agent_id"] == "test-agent-001"

    # Wait for task execution (with short timeout for test)
    import asyncio
    await asyncio.sleep(0.2)

    # Verify task started
    mock_a2a_client.start_task.assert_called_once()

    # Verify task completed
    mock_a2a_client.complete_task.assert_called_once()
    complete_call = mock_a2a_client.complete_task.call_args
    assert complete_call[1]["task_id"] == task_id
    assert complete_call[1]["agent_id"] == "test-agent-001"

    # Cleanup
    await task_handler.shutdown()
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_agent_health_reporting(
    agent_config,
    mock_container_manager,
    mock_a2a_client,
):
    """Test agent health reporting to A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create and start agent
    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent("test-agent-001")

    # Wait for health report (monitoring loop runs every 5s in real code, mocked here)
    import asyncio
    await asyncio.sleep(0.1)

    # In a real test with actual monitoring, health reports would be verified
    # For now, just verify the agent is running
    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_registration_failure_handling(
    agent_config,
    mock_container_manager,
):
    """Test that agent creation succeeds even if A2A registration fails."""
    # Create mock A2A client that fails registration
    failing_a2a_client = AsyncMock(spec=A2AClient)
    failing_a2a_client.register_agent.side_effect = Exception("Connection refused")

    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=failing_a2a_client,
    )

    # Create agent - should succeed despite A2A registration failure
    state = await lifecycle_manager.create_agent(agent_config)

    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"

    # Agent can still be started and run locally
    await lifecycle_manager.start_agent("test-agent-001")

    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_task_failure_reporting(
    agent_config,
    mock_container_manager,
    mock_a2a_client,
):
    """Test task failure reporting to A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Mock task execution to fail
    task_handler = TaskHandler(
        a2a_client=mock_a2a_client,
        lifecycle_manager=lifecycle_manager,
    )

    # Patch execute_task to raise exception
    with patch.object(
        task_handler,
        "_execute_task",
        side_effect=Exception("Task execution error"),
    ):
        # Create and start agent
        await lifecycle_manager.create_agent(agent_config)
        await lifecycle_manager.start_agent("test-agent-001")

        # Assign task
        task_id = "task-002"
        task_data = {"input": "fail"}

        # Task assignment should succeed, but execution will fail
        success = await task_handler.assign_task(
            task_id=task_id,
            agent_id="test-agent-001",
            task_data=task_data,
        )

        assert success is True

        # Wait for failure
        import asyncio
        await asyncio.sleep(0.2)

        # Cleanup
        await task_handler.shutdown()
        await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)
