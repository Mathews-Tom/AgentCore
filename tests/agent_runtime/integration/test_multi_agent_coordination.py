"""
Integration tests for multi-agent coordination.

Tests coordination between multiple agents including
messaging, consensus, conflict resolution, and shared state.
"""

import asyncio

import pytest

from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.agent_runtime.services.agent_lifecycle import AgentLifecycleManager
from agentcore.agent_runtime.services.multi_agent_coordinator import (
    ConflictResolutionStrategy,
    MessagePriority,
    MultiAgentCoordinator,
)


@pytest.mark.asyncio
async def test_multi_agent_registration_and_discovery(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test agent registration and discovery in multi-agent system."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create multiple agents
    agents = []
    for i in range(3):
        config = AgentConfig(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=[f"skill-{i}"],
        )

        # Mock container IDs
        mock_container_manager.create_container.return_value = f"container-{i:03d}"

        state = await lifecycle_manager.create_agent(config)
        agents.append(state.agent_id)

        # Register with coordinator
        await multi_agent_coordinator.register_agent(
            agent_id=state.agent_id,
            capabilities=config.capabilities,
        )

    # Verify all registered
    assert len(agents) == 3

    # Test discovery
    discovered = await multi_agent_coordinator.discover_agents(
        capability="skill-1"
    )
    assert "agent-001" in discovered

    # Cleanup
    for agent_id in agents:
        await multi_agent_coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
async def test_inter_agent_messaging(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test direct messaging between agents."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create two agents
    agent1_config = AgentConfig(
        agent_id="agent-001",
        name="Agent 1",
        philosophy=AgentPhilosophy.MULTI_AGENT,
        capabilities=["sender"],
    )

    agent2_config = AgentConfig(
        agent_id="agent-002",
        name="Agent 2",
        philosophy=AgentPhilosophy.MULTI_AGENT,
        capabilities=["receiver"],
    )

    # Mock container creation
    mock_container_manager.create_container.side_effect = [
        "container-001",
        "container-002",
    ]

    await lifecycle_manager.create_agent(agent1_config)
    await lifecycle_manager.create_agent(agent2_config)

    # Register with coordinator
    await multi_agent_coordinator.register_agent("agent-001", ["sender"])
    await multi_agent_coordinator.register_agent("agent-002", ["receiver"])

    # Send message
    await multi_agent_coordinator.send_message(
        from_agent="agent-001",
        to_agent="agent-002",
        message_type="test",
        payload={"data": "Hello from agent 1"},
        priority=MessagePriority.NORMAL,
    )

    # Receive message
    messages = await multi_agent_coordinator.receive_messages("agent-002")

    assert len(messages) > 0
    assert messages[0]["from_agent"] == "agent-001"
    assert messages[0]["payload"]["data"] == "Hello from agent 1"

    # Cleanup
    await multi_agent_coordinator.unregister_agent("agent-001")
    await multi_agent_coordinator.unregister_agent("agent-002")
    await lifecycle_manager.terminate_agent("agent-001", cleanup=True)
    await lifecycle_manager.terminate_agent("agent-002", cleanup=True)


@pytest.mark.asyncio
async def test_broadcast_messaging(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test broadcast messaging to all agents."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create multiple agents
    agents = []
    for i in range(3):
        config = AgentConfig(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["receiver"],
        )

        mock_container_manager.create_container.return_value = f"container-{i:03d}"
        await lifecycle_manager.create_agent(config)
        await multi_agent_coordinator.register_agent(config.agent_id, ["receiver"])
        agents.append(config.agent_id)

    # Broadcast message
    await multi_agent_coordinator.broadcast_message(
        from_agent="coordinator",
        message_type="announcement",
        payload={"data": "Broadcast to all"},
        priority=MessagePriority.HIGH,
    )

    # Verify all agents received message
    for agent_id in agents:
        messages = await multi_agent_coordinator.receive_messages(agent_id)
        assert len(messages) > 0
        assert messages[0]["payload"]["data"] == "Broadcast to all"

    # Cleanup
    for agent_id in agents:
        await multi_agent_coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
async def test_consensus_voting(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test consensus voting mechanism."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create voting agents
    agents = []
    for i in range(5):
        config = AgentConfig(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["voter"],
        )

        mock_container_manager.create_container.return_value = f"container-{i:03d}"
        await lifecycle_manager.create_agent(config)
        await multi_agent_coordinator.register_agent(config.agent_id, ["voter"])
        agents.append(config.agent_id)

    # Start voting
    vote_id = await multi_agent_coordinator.start_voting(
        proposal="Adopt new protocol",
        voting_agents=agents,
        threshold=0.6,  # 60% consensus required
    )

    # Cast votes (4 yes, 1 no = 80% yes)
    for i, agent_id in enumerate(agents):
        vote = "yes" if i < 4 else "no"
        await multi_agent_coordinator.cast_vote(vote_id, agent_id, vote)

    # Check consensus
    result = await multi_agent_coordinator.check_consensus(vote_id)

    assert result["consensus_reached"] is True
    assert result["winning_vote"] == "yes"
    assert result["vote_percentage"] >= 60.0

    # Cleanup
    for agent_id in agents:
        await multi_agent_coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
async def test_conflict_resolution_majority_vote(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test conflict resolution using majority vote strategy."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create agents
    agents = []
    for i in range(3):
        config = AgentConfig(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["resolver"],
        )

        mock_container_manager.create_container.return_value = f"container-{i:03d}"
        await lifecycle_manager.create_agent(config)
        await multi_agent_coordinator.register_agent(config.agent_id, ["resolver"])
        agents.append(config.agent_id)

    # Create conflict
    conflict_id = await multi_agent_coordinator.create_conflict(
        resource="shared_data",
        competing_agents=agents,
    )

    # Submit proposals
    await multi_agent_coordinator.submit_proposal(
        conflict_id, "agent-000", {"action": "update", "value": "A"}
    )
    await multi_agent_coordinator.submit_proposal(
        conflict_id, "agent-001", {"action": "update", "value": "A"}
    )
    await multi_agent_coordinator.submit_proposal(
        conflict_id, "agent-002", {"action": "update", "value": "B"}
    )

    # Resolve conflict
    resolution = await multi_agent_coordinator.resolve_conflict(
        conflict_id,
        strategy=ConflictResolutionStrategy.MAJORITY_VOTE,
    )

    assert resolution["resolved"] is True
    assert resolution["winning_proposal"]["value"] == "A"  # 2 votes for A

    # Cleanup
    for agent_id in agents:
        await multi_agent_coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
async def test_shared_state_management(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test shared state management across agents."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create agents
    agents = []
    for i in range(2):
        config = AgentConfig(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["state-manager"],
        )

        mock_container_manager.create_container.return_value = f"container-{i:03d}"
        await lifecycle_manager.create_agent(config)
        await multi_agent_coordinator.register_agent(
            config.agent_id, ["state-manager"]
        )
        agents.append(config.agent_id)

    # Agent 0 writes to shared state
    await multi_agent_coordinator.write_shared_state(
        agent_id="agent-000",
        key="counter",
        value=10,
    )

    # Agent 1 reads from shared state
    value = await multi_agent_coordinator.read_shared_state(
        agent_id="agent-001",
        key="counter",
    )

    assert value == 10

    # Agent 1 updates shared state
    await multi_agent_coordinator.write_shared_state(
        agent_id="agent-001",
        key="counter",
        value=20,
    )

    # Agent 0 reads updated value
    value = await multi_agent_coordinator.read_shared_state(
        agent_id="agent-000",
        key="counter",
    )

    assert value == 20

    # Cleanup
    for agent_id in agents:
        await multi_agent_coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)


@pytest.mark.asyncio
async def test_state_locking_mechanism(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test exclusive access to shared state through locking."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create agents
    for i in range(2):
        config = AgentConfig(
            agent_id=f"agent-{i:03d}",
            name=f"Agent {i}",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["locker"],
        )

        mock_container_manager.create_container.return_value = f"container-{i:03d}"
        await lifecycle_manager.create_agent(config)
        await multi_agent_coordinator.register_agent(config.agent_id, ["locker"])

    # Agent 0 acquires lock
    lock_acquired = await multi_agent_coordinator.acquire_lock(
        agent_id="agent-000",
        resource="critical_resource",
    )
    assert lock_acquired is True

    # Agent 1 tries to acquire same lock (should fail)
    lock_acquired = await multi_agent_coordinator.acquire_lock(
        agent_id="agent-001",
        resource="critical_resource",
        timeout=0.1,
    )
    assert lock_acquired is False

    # Agent 0 releases lock
    await multi_agent_coordinator.release_lock(
        agent_id="agent-000",
        resource="critical_resource",
    )

    # Now agent 1 can acquire
    lock_acquired = await multi_agent_coordinator.acquire_lock(
        agent_id="agent-001",
        resource="critical_resource",
    )
    assert lock_acquired is True

    # Cleanup
    await multi_agent_coordinator.release_lock("agent-001", "critical_resource")
    await multi_agent_coordinator.unregister_agent("agent-000")
    await multi_agent_coordinator.unregister_agent("agent-001")
    await lifecycle_manager.terminate_agent("agent-000", cleanup=True)
    await lifecycle_manager.terminate_agent("agent-001", cleanup=True)


@pytest.mark.asyncio
async def test_priority_based_message_handling(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test that high priority messages are processed first."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create agents
    mock_container_manager.create_container.side_effect = [
        "container-001",
        "container-002",
    ]

    await lifecycle_manager.create_agent(
        AgentConfig(
            agent_id="agent-001",
            name="Sender",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["sender"],
        )
    )

    await lifecycle_manager.create_agent(
        AgentConfig(
            agent_id="agent-002",
            name="Receiver",
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=["receiver"],
        )
    )

    await multi_agent_coordinator.register_agent("agent-001", ["sender"])
    await multi_agent_coordinator.register_agent("agent-002", ["receiver"])

    # Send messages with different priorities
    await multi_agent_coordinator.send_message(
        "agent-001",
        "agent-002",
        "low_priority",
        {"order": 1},
        MessagePriority.LOW,
    )

    await multi_agent_coordinator.send_message(
        "agent-001",
        "agent-002",
        "critical",
        {"order": 2},
        MessagePriority.CRITICAL,
    )

    await multi_agent_coordinator.send_message(
        "agent-001",
        "agent-002",
        "normal",
        {"order": 3},
        MessagePriority.NORMAL,
    )

    # Receive messages (should be ordered by priority)
    messages = await multi_agent_coordinator.receive_messages("agent-002")

    assert len(messages) == 3
    # Critical should be first
    assert messages[0]["message_type"] == "critical"

    # Cleanup
    await multi_agent_coordinator.unregister_agent("agent-001")
    await multi_agent_coordinator.unregister_agent("agent-002")
    await lifecycle_manager.terminate_agent("agent-001", cleanup=True)
    await lifecycle_manager.terminate_agent("agent-002", cleanup=True)


@pytest.mark.asyncio
async def test_coordinated_task_execution(
    mock_container_manager,
    multi_agent_coordinator,
):
    """Test coordinated task execution across multiple agents."""
    lifecycle_manager = AgentLifecycleManager(
        container_manager=mock_container_manager,
        a2a_client=None,
    )

    # Create specialized agents
    agents_config = [
        ("agent-collector", ["data_collection"]),
        ("agent-processor", ["data_processing"]),
        ("agent-aggregator", ["aggregation"]),
    ]

    for agent_id, capabilities in agents_config:
        config = AgentConfig(
            agent_id=agent_id,
            name=agent_id.title(),
            philosophy=AgentPhilosophy.MULTI_AGENT,
            capabilities=capabilities,
        )

        mock_container_manager.create_container.return_value = f"container-{agent_id}"
        await lifecycle_manager.create_agent(config)
        await multi_agent_coordinator.register_agent(agent_id, capabilities)

    # Coordinate workflow
    # 1. Collector gathers data
    await multi_agent_coordinator.send_message(
        "coordinator",
        "agent-collector",
        "collect",
        {"source": "database"},
        MessagePriority.HIGH,
    )

    # 2. Send to processor
    await multi_agent_coordinator.send_message(
        "agent-collector",
        "agent-processor",
        "process",
        {"data": "[collected]"},
        MessagePriority.NORMAL,
    )

    # 3. Send to aggregator
    await multi_agent_coordinator.send_message(
        "agent-processor",
        "agent-aggregator",
        "aggregate",
        {"processed_data": "[processed]"},
        MessagePriority.NORMAL,
    )

    # Verify messages received
    for agent_id in ["agent-collector", "agent-processor", "agent-aggregator"]:
        messages = await multi_agent_coordinator.receive_messages(agent_id)
        assert len(messages) > 0

    # Cleanup
    for agent_id, _ in agents_config:
        await multi_agent_coordinator.unregister_agent(agent_id)
        await lifecycle_manager.terminate_agent(agent_id, cleanup=True)
