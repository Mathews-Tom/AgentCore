"""Tests for Multi-Agent Coordinator."""

import asyncio

import pytest

from agentcore.agent_runtime.services.multi_agent_coordinator import (
    AgentMessage,
    ConflictResolutionStrategy,
    ConsensusRequest,
    MessagePriority,
    MessageType,
    MultiAgentCoordinator,
    SharedState,
    VoteOption,
)


@pytest.fixture
async def coordinator() -> MultiAgentCoordinator:
    """Create coordinator instance."""
    return MultiAgentCoordinator()


@pytest.mark.asyncio
class TestMultiAgentCoordinator:
    """Test suite for Multi-Agent Coordinator."""

    async def test_agent_registration(self, coordinator: MultiAgentCoordinator) -> None:
        """Test agent registration."""
        agent_id = "test-agent-1"
        metadata = {"capability": "calculator", "priority": 5}

        await coordinator.register_agent(agent_id, metadata)

        assert agent_id in coordinator.get_active_agents()
        assert coordinator.get_agent_metadata(agent_id) == metadata

    async def test_agent_unregistration(self, coordinator: MultiAgentCoordinator) -> None:
        """Test agent unregistration."""
        agent_id = "test-agent-2"
        await coordinator.register_agent(agent_id, {})

        assert agent_id in coordinator.get_active_agents()

        await coordinator.unregister_agent(agent_id)

        assert agent_id not in coordinator.get_active_agents()

    async def test_direct_message(self, coordinator: MultiAgentCoordinator) -> None:
        """Test direct message sending."""
        sender_id = "agent-sender"
        recipient_id = "agent-recipient"

        await coordinator.register_agent(sender_id, {})
        await coordinator.register_agent(recipient_id, {})

        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            content={"data": "test message"},
        )

        await coordinator.send_message(message)

        # Recipient should receive the message
        received = await coordinator.receive_message(recipient_id, timeout=1.0)

        assert received is not None
        assert received.sender_id == sender_id
        assert received.recipient_id == recipient_id
        assert received.content["data"] == "test message"

    async def test_broadcast_message(self, coordinator: MultiAgentCoordinator) -> None:
        """Test broadcast message."""
        sender_id = "agent-broadcaster"
        recipients = ["agent-1", "agent-2", "agent-3"]

        await coordinator.register_agent(sender_id, {})
        for recipient in recipients:
            await coordinator.register_agent(recipient, {})

        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.BROADCAST,
            content={"announcement": "Hello everyone"},
        )

        await coordinator.send_message(message)

        # All recipients should receive the message
        for recipient in recipients:
            received = await coordinator.receive_message(recipient, timeout=1.0)
            assert received is not None
            assert received.sender_id == sender_id
            assert received.content["announcement"] == "Hello everyone"

    async def test_message_timeout(self, coordinator: MultiAgentCoordinator) -> None:
        """Test message receive timeout."""
        agent_id = "test-agent"
        await coordinator.register_agent(agent_id, {})

        # No messages sent, should timeout
        received = await coordinator.receive_message(agent_id, timeout=0.1)
        assert received is None

    async def test_consensus_voting(self, coordinator: MultiAgentCoordinator) -> None:
        """Test consensus voting mechanism."""
        initiator_id = "agent-initiator"
        participants = ["agent-1", "agent-2", "agent-3"]

        await coordinator.register_agent(initiator_id, {})
        for participant in participants:
            await coordinator.register_agent(participant, {})

        # Create consensus request
        options = [
            VoteOption(option_id="option_a", description="Option A"),
            VoteOption(option_id="option_b", description="Option B"),
        ]

        request = ConsensusRequest(
            initiator_id=initiator_id,
            topic="Choose best approach",
            options=options,
            participating_agents=[initiator_id] + participants,
            required_votes=3,
            timeout_seconds=5,
        )

        request_id = await coordinator.initiate_consensus(request)

        # Cast votes
        await coordinator.cast_vote(request_id, initiator_id, "option_a")
        await coordinator.cast_vote(request_id, "agent-1", "option_a")
        await coordinator.cast_vote(request_id, "agent-2", "option_b")
        await coordinator.cast_vote(request_id, "agent-3", "option_a")

        # Check consensus
        result = await coordinator.check_consensus(request_id)

        assert result.consensus_reached is True
        assert result.winning_option is not None
        assert result.winning_option.option_id == "option_a"
        assert result.total_votes == 4

    async def test_consensus_not_reached(self, coordinator: MultiAgentCoordinator) -> None:
        """Test consensus when votes insufficient."""
        initiator_id = "agent-initiator"
        participants = ["agent-1", "agent-2"]

        await coordinator.register_agent(initiator_id, {})
        for participant in participants:
            await coordinator.register_agent(participant, {})

        options = [
            VoteOption(option_id="option_a", description="Option A"),
            VoteOption(option_id="option_b", description="Option B"),
        ]

        request = ConsensusRequest(
            initiator_id=initiator_id,
            topic="Choose approach",
            options=options,
            participating_agents=[initiator_id] + participants,
            required_votes=5,  # More than available agents
        )

        request_id = await coordinator.initiate_consensus(request)

        # Cast only 2 votes
        await coordinator.cast_vote(request_id, initiator_id, "option_a")
        await coordinator.cast_vote(request_id, "agent-1", "option_a")

        result = await coordinator.check_consensus(request_id)

        assert result.consensus_reached is False
        assert result.winning_option is None

    async def test_conflict_resolution_majority_vote(
        self,
        coordinator: MultiAgentCoordinator,
    ) -> None:
        """Test conflict resolution with majority vote."""
        agents = ["agent-1", "agent-2", "agent-3"]
        for agent in agents:
            await coordinator.register_agent(agent, {})

        conflict_data = {
            "topic": "Resource allocation",
            "options": ["Option A", "Option B"],
        }

        # Initiate resolution (this will timeout waiting for votes)
        task = asyncio.create_task(
            coordinator.resolve_conflict(
                conflict_data=conflict_data,
                strategy=ConflictResolutionStrategy.MAJORITY_VOTE,
                involved_agents=agents,
            )
        )

        # Wait a moment then cancel (since we're not manually voting)
        await asyncio.sleep(0.1)
        task.cancel()

    async def test_conflict_resolution_priority_based(
        self,
        coordinator: MultiAgentCoordinator,
    ) -> None:
        """Test priority-based conflict resolution."""
        agents = ["agent-1", "agent-2", "agent-3"]
        priorities = [3, 7, 5]

        for agent, priority in zip(agents, priorities):
            await coordinator.register_agent(agent, {"priority": priority})

        conflict_data = {"topic": "Resource conflict"}

        result = await coordinator.resolve_conflict(
            conflict_data=conflict_data,
            strategy=ConflictResolutionStrategy.PRIORITY_BASED,
            involved_agents=agents,
        )

        assert result["strategy"] == ConflictResolutionStrategy.PRIORITY_BASED.value
        assert result["selected_agent"] == "agent-2"  # Highest priority (7)
        assert result["priority"] == 7

    async def test_shared_state_creation(self, coordinator: MultiAgentCoordinator) -> None:
        """Test shared state creation."""
        owner_id = "agent-owner"
        await coordinator.register_agent(owner_id, {})

        state = SharedState(
            owner_id=owner_id,
            data={"key": "value"},
            access_control={"read": ["agent-reader"], "write": ["agent-writer"]},
        )

        state_id = await coordinator.create_shared_state(state)

        assert state_id == state.state_id

    async def test_shared_state_read_access(
        self,
        coordinator: MultiAgentCoordinator,
    ) -> None:
        """Test shared state read access control."""
        owner_id = "agent-owner"
        reader_id = "agent-reader"

        await coordinator.register_agent(owner_id, {})
        await coordinator.register_agent(reader_id, {})

        state = SharedState(
            owner_id=owner_id,
            data={"secret": "data"},
            access_control={"read": [reader_id], "write": []},
        )

        state_id = await coordinator.create_shared_state(state)

        # Reader should be able to read
        read_state = await coordinator.read_shared_state(state_id, reader_id)
        assert read_state.data["secret"] == "data"

        # Unauthorized agent should be denied
        unauthorized_id = "agent-unauthorized"
        await coordinator.register_agent(unauthorized_id, {})

        with pytest.raises(PermissionError):
            await coordinator.read_shared_state(state_id, unauthorized_id)

    async def test_shared_state_write_access(
        self,
        coordinator: MultiAgentCoordinator,
    ) -> None:
        """Test shared state write access control."""
        owner_id = "agent-owner"
        writer_id = "agent-writer"

        await coordinator.register_agent(owner_id, {})
        await coordinator.register_agent(writer_id, {})

        state = SharedState(
            owner_id=owner_id,
            data={"counter": 0},
            access_control={"read": [writer_id], "write": [writer_id]},
        )

        state_id = await coordinator.create_shared_state(state)

        # Writer should be able to update
        updated = await coordinator.update_shared_state(
            state_id,
            writer_id,
            {"counter": 1},
        )
        assert updated.data["counter"] == 1
        assert updated.version == 2

        # Unauthorized agent should be denied
        unauthorized_id = "agent-unauthorized"
        await coordinator.register_agent(unauthorized_id, {})

        with pytest.raises(PermissionError):
            await coordinator.update_shared_state(
                state_id,
                unauthorized_id,
                {"counter": 999},
            )

    async def test_shared_state_locking(
        self,
        coordinator: MultiAgentCoordinator,
    ) -> None:
        """Test shared state locking mechanism."""
        owner_id = "agent-owner"
        agent1_id = "agent-1"
        agent2_id = "agent-2"

        await coordinator.register_agent(owner_id, {})
        await coordinator.register_agent(agent1_id, {})
        await coordinator.register_agent(agent2_id, {})

        state = SharedState(
            owner_id=owner_id,
            data={"value": 0},
            access_control={"read": [agent1_id, agent2_id], "write": [agent1_id, agent2_id]},
        )

        state_id = await coordinator.create_shared_state(state)

        # Agent 1 acquires lock
        locked = await coordinator.lock_shared_state(state_id, agent1_id)
        assert locked is True

        # Agent 2 cannot acquire lock
        locked2 = await coordinator.lock_shared_state(state_id, agent2_id)
        assert locked2 is False

        # Agent 1 releases lock
        await coordinator.unlock_shared_state(state_id, agent1_id)

        # Now agent 2 can acquire lock
        locked3 = await coordinator.lock_shared_state(state_id, agent2_id)
        assert locked3 is True

    async def test_shared_state_lock_enforcement(
        self,
        coordinator: MultiAgentCoordinator,
    ) -> None:
        """Test that locked state prevents updates from other agents."""
        owner_id = "agent-owner"
        agent1_id = "agent-1"
        agent2_id = "agent-2"

        await coordinator.register_agent(owner_id, {})
        await coordinator.register_agent(agent1_id, {})
        await coordinator.register_agent(agent2_id, {})

        state = SharedState(
            owner_id=owner_id,
            data={"value": 0},
            access_control={"read": [agent1_id, agent2_id], "write": [agent1_id, agent2_id]},
        )

        state_id = await coordinator.create_shared_state(state)

        # Agent 1 locks and updates
        await coordinator.lock_shared_state(state_id, agent1_id)
        await coordinator.update_shared_state(state_id, agent1_id, {"value": 1})

        # Agent 2 cannot update while locked
        with pytest.raises(RuntimeError, match="State locked"):
            await coordinator.update_shared_state(state_id, agent2_id, {"value": 2})
