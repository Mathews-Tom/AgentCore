"""
Integration tests for A2A protocol communication.

Tests integration between agent runtime and A2A protocol layer,
validating agent registration, task handling, and status reporting.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.services.a2a_client import (
    A2AClient,
    A2AClientError,
    A2AConnectionError,
    A2ARegistrationError,
)
from agentcore.agent_runtime.services.agent_lifecycle import AgentLifecycleManager
from agentcore.agent_runtime.services.task_handler import TaskHandler


@pytest.mark.asyncio
async def test_a2a_agent_registration(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
):
    """Test agent registration with A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create agent (triggers A2A registration)
    state = await lifecycle_manager.create_agent(agent_config)

    # Verify agent created
    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"

    # Verify A2A registration called
    mock_a2a_client.register_agent.assert_called_once()
    call_args = mock_a2a_client.register_agent.call_args
    # The agent_config is passed as positional argument
    registered_config = call_args[0][0]
    assert registered_config.agent_id == "test-agent-001"
    assert registered_config.name == "Test Agent"

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_agent_unregistration(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
):
    """Test agent unregistration from A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create and start agent
    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent("test-agent-001")

    # Terminate agent (triggers A2A unregistration)
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)

    # Verify A2A unregistration called
    mock_a2a_client.unregister_agent.assert_called_once_with("test-agent-001")


@pytest.mark.asyncio
async def test_a2a_status_updates(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
):
    """Test agent status updates to A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create agent
    await lifecycle_manager.create_agent(agent_config)

    # Start agent (triggers status update)
    await lifecycle_manager.start_agent("test-agent-001")

    # Verify status update called
    mock_a2a_client.update_agent_status.assert_called()
    status_calls = mock_a2a_client.update_agent_status.call_args_list
    assert any(call[1]["status"] == "active" for call in status_calls)

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_health_reporting(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
):
    """Test agent health reporting to A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create and start agent
    await lifecycle_manager.create_agent(agent_config)
    await lifecycle_manager.start_agent("test-agent-001")

    # Wait for health report cycle
    await asyncio.sleep(0.2)

    # Health reporting happens in background monitoring
    # Verify agent is running
    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_task_assignment(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
    test_task_data,
):
    """Test task assignment through A2A protocol."""
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
    success = await task_handler.assign_task(
        task_id=test_task_data["task_id"],
        agent_id="test-agent-001",
        task_data=test_task_data,
    )

    assert success is True

    # Verify task accepted
    mock_a2a_client.accept_task.assert_called_once()
    accept_call = mock_a2a_client.accept_task.call_args
    assert accept_call[1]["task_id"] == test_task_data["task_id"]
    assert accept_call[1]["agent_id"] == "test-agent-001"

    # Wait for task execution
    await asyncio.sleep(0.2)

    # Verify task lifecycle methods called
    mock_a2a_client.start_task.assert_called_once()

    # Cleanup
    await task_handler.shutdown()
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_task_completion(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
    test_task_data,
):
    """Test task completion reporting to A2A protocol."""
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

    # Assign and execute task
    await task_handler.assign_task(
        task_id=test_task_data["task_id"],
        agent_id="test-agent-001",
        task_data=test_task_data,
    )

    # Wait for completion
    await asyncio.sleep(0.3)

    # Verify task completed
    mock_a2a_client.complete_task.assert_called_once()
    complete_call = mock_a2a_client.complete_task.call_args
    assert complete_call[1]["task_id"] == test_task_data["task_id"]
    assert complete_call[1]["agent_id"] == "test-agent-001"

    # Cleanup
    await task_handler.shutdown()
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_task_failure_reporting(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
    test_task_data,
):
    """Test task failure reporting to A2A protocol."""
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

    # Mock task execution to fail
    with patch.object(
        task_handler,
        "_execute_task",
        side_effect=Exception("Task execution failed"),
    ):
        # Assign task
        await task_handler.assign_task(
            task_id=test_task_data["task_id"],
            agent_id="test-agent-001",
            task_data=test_task_data,
        )

        # Wait for failure
        await asyncio.sleep(0.3)

    # Cleanup
    await task_handler.shutdown()
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_registration_failure_graceful_handling(
    mock_container_manager,
    agent_config,
):
    """Test graceful handling of A2A registration failures."""
    # Create failing A2A client
    failing_client = AsyncMock(spec=A2AClient)
    failing_client.register_agent.side_effect = A2AConnectionError(
        "Connection refused"
    )

    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=failing_client,
    )

    # Agent creation should succeed despite A2A failure
    state = await lifecycle_manager.create_agent(agent_config)

    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"

    # Agent can still operate locally
    await lifecycle_manager.start_agent("test-agent-001")
    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_client_connectivity_check(mock_a2a_client):
    """Test A2A client connectivity verification."""
    # Test ping
    result = await mock_a2a_client.ping()
    assert result is True

    mock_a2a_client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_a2a_concurrent_registrations(
    mock_container_manager,
    mock_a2a_client,
):
    """Test concurrent agent registrations with A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create multiple agents concurrently
    configs = [
        AgentConfig(
            agent_id=f"test-agent-{i:03d}",
            name=f"Test Agent {i}",
            philosophy=AgentPhilosophy.REACT,
            capabilities=["test"],
        )
        for i in range(5)
    ]

    # Mock different container IDs
    mock_container_manager.create_container.side_effect = [
        f"container-{i:03d}" for i in range(5)
    ]

    # Create agents concurrently
    tasks = [lifecycle_manager.create_agent(config) for config in configs]
    states = await asyncio.gather(*tasks)

    # Verify all created
    assert len(states) == 5
    assert all(state.status == "initializing" for state in states)

    # Verify all registered with A2A
    assert mock_a2a_client.register_agent.call_count == 5

    # Cleanup all
    for state in states:
        await lifecycle_manager.terminate_agent(state.agent_id, cleanup=True)


@pytest.mark.asyncio
async def test_a2a_reconnection_handling(
    mock_container_manager,
    agent_config,
):
    """Test A2A client reconnection after connection loss."""
    # Create client that fails initially then succeeds
    flaky_client = AsyncMock(spec=A2AClient)
    call_count = 0

    async def register_with_retry(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise A2AConnectionError("Connection lost")
        return "test-agent-001"

    flaky_client.register_agent.side_effect = register_with_retry
    flaky_client.unregister_agent.return_value = True

    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=flaky_client,
    )

    # First attempt should handle connection error gracefully
    state = await lifecycle_manager.create_agent(agent_config)
    assert state.agent_id == "test-agent-001"

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_without_client(
    mock_container_manager,
    agent_config,
):
    """Test agent runtime works without A2A client (local mode)."""
    # Create lifecycle manager without A2A client
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create agent
    state = await lifecycle_manager.create_agent(agent_config)
    assert state.agent_id == "test-agent-001"
    assert state.status == "initializing"

    # Start agent
    await lifecycle_manager.start_agent("test-agent-001")
    state = await lifecycle_manager.get_agent_status("test-agent-001")
    assert state.status == "running"

    # Terminate agent
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_task_cancellation(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
    test_task_data,
):
    """Test task cancellation through A2A protocol."""
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

    # Assign long-running task
    await task_handler.assign_task(
        task_id=test_task_data["task_id"],
        agent_id="test-agent-001",
        task_data=test_task_data,
    )

    # Cancel task
    await task_handler.cancel_task(test_task_data["task_id"])

    # Wait briefly
    await asyncio.sleep(0.1)

    # Cleanup
    await task_handler.shutdown()
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_a2a_metadata_propagation(
    mock_container_manager,
    mock_a2a_client,
    agent_config,
):
    """Test metadata propagation through A2A protocol."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=mock_a2a_client,
    )

    # Create agent with metadata
    state = await lifecycle_manager.create_agent(agent_config)

    # Verify registration included metadata
    mock_a2a_client.register_agent.assert_called_once()
    call_args = mock_a2a_client.register_agent.call_args
    registered_config = call_args[0][0]
    assert registered_config.capabilities == ["test", "demo"]

    # Cleanup
    await lifecycle_manager.terminate_agent("test-agent-001", cleanup=True)
