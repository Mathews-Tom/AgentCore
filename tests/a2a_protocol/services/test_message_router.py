"""
Comprehensive test suite for MessageRouter service.

Tests routing strategies, capability matching, load balancing, message queuing,
circuit breaker, and session-aware routing.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.agent import AgentCard, AgentStatus
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest, MessageEnvelope
from agentcore.a2a_protocol.models.session import SessionSnapshot, SessionState
from agentcore.a2a_protocol.services.message_router import (
    MessagePriority,
    MessageRouter,
    QueuedMessage,
    RoutingStrategy,
    message_router,
)


def create_test_envelope(
    message_id: str | None = None,
    source: str = "source",
    destination: str | None = None,
) -> MessageEnvelope:
    """Helper to create valid test MessageEnvelope."""
    return MessageEnvelope(
        message_id=message_id or str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
        source=source,
        destination=destination,
        payload=JsonRpcRequest(jsonrpc="2.0", method="test.action", params={}, id=1),
    )


@pytest.fixture
def router():
    """Create fresh MessageRouter instance for each test."""
    return MessageRouter()


@pytest.fixture
def sample_envelope():
    """Create sample message envelope."""
    return create_test_envelope()


@pytest.fixture
def mock_agent():
    """Create mock agent card."""
    from agentcore.a2a_protocol.models.agent import (
        AgentAuthentication,
        AgentCapability,
        AgentEndpoint,
        AuthenticationType,
        EndpointType,
    )

    return AgentCard(
        agent_id="agent-1",
        agent_name="Test Agent",
        agent_version="1.0.0",
        status=AgentStatus.ACTIVE,
        capabilities=[
            AgentCapability(name="text-generation"),
            AgentCapability(name="code-review"),
        ],
        endpoints=[AgentEndpoint(url="http://agent-1.local", type=EndpointType.HTTP)],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE, required=False
        ),
    )


@pytest.fixture
def mock_session():
    """Create mock session."""
    return SessionSnapshot(
        session_id="session-1",
        name="Test Session",
        state=SessionState.ACTIVE,
        owner_agent="agent-1",
    )


# ==================== Routing Tests ====================


class TestRouting:
    """Test message routing functionality."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_route_to_explicit_destination(
        self, mock_agent_mgr, router, sample_envelope, mock_agent
    ):
        """Test routing to explicitly specified destination."""
        sample_envelope.destination = "agent-1"
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router.route_message(sample_envelope)

        assert result == "agent-1"
        mock_agent_mgr.get_agent.assert_called_once_with("agent-1")

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_route_to_offline_destination_queues_message(
        self, mock_agent_mgr, router, sample_envelope
    ):
        """Test routing to offline agent queues the message."""
        sample_envelope.destination = "offline-agent"

        # Mock agent that is offline
        offline_agent = MagicMock()
        offline_agent.is_active = MagicMock(return_value=False)
        offline_agent.status = AgentStatus.INACTIVE

        mock_agent_mgr.get_agent = AsyncMock(return_value=offline_agent)

        result = await router.route_message(sample_envelope)

        assert result is None
        assert len(router._message_queues["offline-agent"]) == 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_route_with_required_capabilities(
        self, mock_agent_mgr, router, sample_envelope, mock_agent
    ):
        """Test routing with required capabilities."""
        mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(
            return_value=[{"agent_id": "agent-1"}]
        )
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router.route_message(
            sample_envelope, required_capabilities=["text-generation"]
        )

        assert result == "agent-1"
        mock_agent_mgr.discover_agents_by_capabilities.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_no_agents_available_raises_error(
        self, router, sample_envelope
    ):
        """Test routing fails when no agents available."""
        with patch(
            "agentcore.a2a_protocol.services.message_router.agent_manager"
        ) as mock_mgr:
            mock_mgr.discover_agents_by_capabilities = AsyncMock(return_value=[])

            with pytest.raises(ValueError, match="No agents available"):
                await router.route_message(
                    sample_envelope, required_capabilities=["nonexistent-capability"]
                )

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_route_without_capabilities_uses_all_agents(
        self, mock_agent_mgr, router, sample_envelope, mock_agent
    ):
        """Test routing without capabilities considers all active agents."""
        mock_agent_mgr.list_all_agents = AsyncMock(
            return_value=[{"agent_id": "agent-1"}]
        )
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router.route_message(sample_envelope)

        assert result == "agent-1"
        mock_agent_mgr.list_all_agents.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_increments_stats(self, router, sample_envelope):
        """Test routing increments statistics."""
        with patch(
            "agentcore.a2a_protocol.services.message_router.agent_manager"
        ) as mock_mgr:
            mock_agent = MagicMock()
            mock_agent.is_active = MagicMock(return_value=True)

            mock_mgr.list_all_agents = AsyncMock(return_value=[{"agent_id": "agent-1"}])
            mock_mgr.get_agent = AsyncMock(return_value=mock_agent)

            initial_routed = router._routing_stats["total_routed"]

            await router.route_message(sample_envelope)

            assert router._routing_stats["total_routed"] == initial_routed + 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    async def test_route_with_session(
        self,
        mock_sess_mgr,
        mock_agent_mgr,
        router,
        sample_envelope,
        mock_agent,
        mock_session,
    ):
        """Test routing with session context."""
        mock_sess_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_sess_mgr.set_agent_state = AsyncMock(return_value=True)
        mock_sess_mgr.record_event = AsyncMock()
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        mock_session.participant_agents = ["agent-1"]

        result = await router.route_with_session(
            sample_envelope, session_id="session-1"
        )

        assert result == "agent-1"
        mock_sess_mgr.record_event.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    async def test_route_with_session_not_found_raises_error(
        self, mock_sess_mgr, router, sample_envelope
    ):
        """Test routing with nonexistent session raises error."""
        mock_sess_mgr.get_session = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Session not found"):
            await router.route_with_session(sample_envelope, session_id="nonexistent")

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    async def test_route_with_terminal_session_raises_error(
        self, mock_sess_mgr, router, sample_envelope, mock_session
    ):
        """Test routing with terminal session raises error."""
        # Set state to terminal state (is_terminal is a computed property)
        mock_session.state = SessionState.COMPLETED
        mock_sess_mgr.get_session = AsyncMock(return_value=mock_session)

        with pytest.raises(ValueError, match="terminal state"):
            await router.route_with_session(sample_envelope, session_id="session-1")


# ==================== Routing Strategy Tests ====================


class TestRoutingStrategies:
    """Test routing strategy implementations."""

    @pytest.mark.asyncio
    async def test_round_robin_strategy(self, router):
        """Test round-robin agent selection."""
        candidates = ["agent-1", "agent-2", "agent-3"]

        # Call multiple times to verify round-robin
        results = []
        for _ in range(6):
            result = await router._round_robin_select(candidates)
            results.append(result)

        # Should cycle through agents
        assert results[:3] == candidates
        assert results[3:6] == candidates

    @pytest.mark.asyncio
    async def test_least_loaded_strategy(self, router):
        """Test least-loaded agent selection."""
        candidates = ["agent-1", "agent-2", "agent-3"]

        # Set different loads
        router._agent_load["agent-1"] = 5
        router._agent_load["agent-2"] = 2
        router._agent_load["agent-3"] = 10

        result = await router._least_loaded_select(candidates)

        assert result == "agent-2"  # Least loaded

    @pytest.mark.asyncio
    async def test_random_strategy(self, router):
        """Test random agent selection."""
        candidates = ["agent-1", "agent-2", "agent-3"]

        result = await router._select_agent(candidates, RoutingStrategy.RANDOM)

        assert result in candidates

    @pytest.mark.asyncio
    async def test_cost_optimized_strategy(self, router):
        """Test cost-optimized agent selection."""
        candidates = ["agent-1", "agent-2"]

        # Set different loads (affects cost optimization)
        router._agent_load["agent-1"] = 8
        router._agent_load["agent-2"] = 2

        result = await router._cost_optimized_select(candidates)

        # Should prefer less loaded agent
        assert result == "agent-2"

    @pytest.mark.asyncio
    async def test_cost_optimized_empty_candidates_raises_error(self, router):
        """Test cost-optimized selection with empty candidates raises error."""
        with pytest.raises(ValueError, match="No candidates"):
            await router._cost_optimized_select([])

    @pytest.mark.asyncio
    async def test_semantic_match_strategy_fallback(self, router):
        """Test semantic match strategy falls back to first candidate."""
        candidates = ["agent-1", "agent-2"]

        result = await router._select_agent(candidates, RoutingStrategy.SEMANTIC_MATCH)

        assert result == "agent-1"

    @pytest.mark.asyncio
    async def test_capability_match_strategy_default(self, router):
        """Test capability match returns first candidate."""
        candidates = ["agent-1", "agent-2", "agent-3"]

        result = await router._select_agent(
            candidates, RoutingStrategy.CAPABILITY_MATCH
        )

        assert result == "agent-1"

    @pytest.mark.asyncio
    async def test_select_agent_empty_candidates(self, router):
        """Test selecting from empty candidates returns None."""
        result = await router._select_agent([], RoutingStrategy.ROUND_ROBIN)
        assert result is None

    @pytest.mark.asyncio
    async def test_round_robin_increments_load_balanced_stat(self, router):
        """Test round-robin increments load_balanced stat."""
        initial_count = router._routing_stats["load_balanced"]

        await router._round_robin_select(["agent-1", "agent-2"])

        assert router._routing_stats["load_balanced"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_least_loaded_increments_load_balanced_stat(self, router):
        """Test least-loaded increments load_balanced stat."""
        initial_count = router._routing_stats["load_balanced"]

        await router._least_loaded_select(["agent-1", "agent-2"])

        assert router._routing_stats["load_balanced"] == initial_count + 1


# ==================== Capability Matching Tests ====================


class TestCapabilityMatching:
    """Test capability-based agent matching."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_find_capable_agents_single_capability(
        self, mock_agent_mgr, router, mock_agent
    ):
        """Test finding agents with single capability."""
        mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(
            return_value=[{"agent_id": "agent-1"}]
        )
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router._find_capable_agents(["text-generation"])

        assert result == ["agent-1"]

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_find_capable_agents_multiple_capabilities(
        self, mock_agent_mgr, router, mock_agent
    ):
        """Test finding agents with multiple capabilities (intersection)."""
        # Mock responses for each capability
        mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(
            side_effect=[
                [{"agent_id": "agent-1"}, {"agent_id": "agent-2"}],  # Has capability 1
                [{"agent_id": "agent-1"}],  # Has capability 2
            ]
        )
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router._find_capable_agents(["cap-1", "cap-2"])

        # Only agent-1 has both capabilities
        assert result == ["agent-1"]

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_find_capable_agents_no_match(self, mock_agent_mgr, router):
        """Test finding capable agents returns empty when no match."""
        mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(return_value=[])

        result = await router._find_capable_agents(["nonexistent-cap"])

        assert result == []

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_find_capable_agents_filters_unavailable(
        self, mock_agent_mgr, router
    ):
        """Test finding capable agents filters out unavailable agents."""
        mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(
            return_value=[{"agent_id": "agent-1"}, {"agent_id": "agent-2"}]
        )

        # agent-1 is available, agent-2 is circuit-broken
        router._circuit_breaker_reset_time["agent-2"] = datetime.now(UTC) + timedelta(
            seconds=60
        )

        mock_available_agent = MagicMock()
        mock_available_agent.is_active = MagicMock(return_value=True)

        mock_agent_mgr.get_agent = AsyncMock(
            side_effect=[
                mock_available_agent,  # agent-1 available
                None,  # agent-2 not returned (circuit broken)
            ]
        )

        result = await router._find_capable_agents(["capability"])

        assert "agent-1" in result
        assert "agent-2" not in result

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_is_agent_available_active_agent(
        self, mock_agent_mgr, router, mock_agent
    ):
        """Test checking if active agent is available."""
        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router._is_agent_available("agent-1")

        assert result is True

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_is_agent_available_circuit_broken(self, mock_agent_mgr, router):
        """Test circuit-broken agent is not available."""
        router._circuit_breaker_reset_time["agent-1"] = datetime.now(UTC) + timedelta(
            seconds=60
        )

        result = await router._is_agent_available("agent-1")

        assert result is False

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_is_agent_available_circuit_reset_after_timeout(
        self, mock_agent_mgr, router, mock_agent
    ):
        """Test circuit breaker resets after timeout."""
        # Set circuit breaker to past time
        router._circuit_breaker_reset_time["agent-1"] = datetime.now(UTC) - timedelta(
            seconds=10
        )
        router._circuit_breaker_failures["agent-1"] = 5

        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)

        result = await router._is_agent_available("agent-1")

        assert result is True
        assert "agent-1" not in router._circuit_breaker_reset_time
        assert router._circuit_breaker_failures["agent-1"] == 0

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_is_agent_available_inactive_agent(self, mock_agent_mgr, router):
        """Test inactive agent is not available."""
        inactive_agent = MagicMock()
        inactive_agent.is_active = MagicMock(return_value=False)
        mock_agent_mgr.get_agent = AsyncMock(return_value=inactive_agent)

        result = await router._is_agent_available("agent-1")

        assert result is False


# ==================== Message Queuing Tests ====================


class TestMessageQueuing:
    """Test message queuing functionality."""

    @pytest.mark.asyncio
    async def test_queue_message_normal_priority(self, router, sample_envelope):
        """Test queuing message with normal priority."""
        await router._queue_message(sample_envelope, "agent-1", MessagePriority.NORMAL)

        assert len(router._message_queues["agent-1"]) == 1
        assert router._routing_stats["queued_messages"] == 1

    @pytest.mark.asyncio
    async def test_queue_message_critical_priority_prepends(self, router):
        """Test critical priority messages are added to front of queue."""
        # Add normal message first
        envelope1 = create_test_envelope(message_id="msg-1")
        await router._queue_message(envelope1, "agent-1", MessagePriority.NORMAL)

        # Add critical message
        envelope2 = create_test_envelope(message_id="msg-2")
        await router._queue_message(envelope2, "agent-1", MessagePriority.CRITICAL)

        # Critical message should be first
        queue = router._message_queues["agent-1"]
        assert queue[0].message_id == "msg-2"
        assert queue[1].message_id == "msg-1"

    @pytest.mark.asyncio
    async def test_queue_message_with_custom_ttl(self, router, sample_envelope):
        """Test queuing message with custom TTL."""
        ttl_seconds = 600

        await router._queue_message(
            sample_envelope, "agent-1", MessagePriority.NORMAL, ttl_seconds=ttl_seconds
        )

        queued_msg = router._message_queues["agent-1"][0]
        expected_expiry = queued_msg.queued_at + timedelta(seconds=ttl_seconds)

        assert abs((queued_msg.expires_at - expected_expiry).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_process_queued_messages_success(self, router):
        """Test processing queued messages when agent comes online."""
        # Queue messages
        envelope1 = create_test_envelope(message_id="msg-1")
        envelope2 = create_test_envelope(message_id="msg-2")

        await router._queue_message(envelope1, "agent-1", MessagePriority.NORMAL)
        await router._queue_message(envelope2, "agent-1", MessagePriority.NORMAL)

        # Process queue
        processed = await router.process_queued_messages("agent-1")

        assert processed == 2
        assert len(router._message_queues.get("agent-1", [])) == 0

    @pytest.mark.asyncio
    async def test_process_queued_messages_no_queue(self, router):
        """Test processing queued messages when no queue exists."""
        processed = await router.process_queued_messages("nonexistent-agent")
        assert processed == 0

    @pytest.mark.asyncio
    async def test_process_queued_messages_skips_expired(self, router):
        """Test processing skips expired messages."""
        # Create expired message
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1",
            envelope=envelope,
            target_agent_id="agent-1",
            ttl_seconds=1,
        )
        queued_msg.expires_at = datetime.now(UTC) - timedelta(
            seconds=10
        )  # Already expired

        router._message_queues["agent-1"].append(queued_msg)

        processed = await router.process_queued_messages("agent-1")

        assert processed == 0

    @pytest.mark.asyncio
    async def test_process_queued_messages_retries_failures(self, router):
        """Test processing retries failed deliveries."""
        envelope = create_test_envelope(message_id="msg-1")
        await router._queue_message(envelope, "agent-1", MessagePriority.NORMAL)

        # Mock delivery to fail
        with patch.object(
            router,
            "_deliver_message",
            AsyncMock(side_effect=Exception("Delivery failed")),
        ):
            processed = await router.process_queued_messages("agent-1")

        # Message should be requeued
        assert processed == 0
        assert len(router._message_queues["agent-1"]) == 1
        assert router._message_queues["agent-1"][0].retry_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_messages(self, router):
        """Test cleanup of expired messages."""
        # Add expired message
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1",
            envelope=envelope,
            target_agent_id="agent-1",
            ttl_seconds=1,
        )
        queued_msg.expires_at = datetime.now(UTC) - timedelta(seconds=10)
        router._message_queues["agent-1"].append(queued_msg)

        # Add non-expired message
        envelope2 = create_test_envelope(message_id="msg-2")
        await router._queue_message(envelope2, "agent-1", MessagePriority.NORMAL)

        removed = await router.cleanup_expired_messages()

        assert removed == 1
        assert len(router._message_queues["agent-1"]) == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_messages_removes_empty_queues(self, router):
        """Test cleanup removes empty queues."""
        # Add only expired message
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1",
            envelope=envelope,
            target_agent_id="agent-1",
            ttl_seconds=1,
        )
        queued_msg.expires_at = datetime.now(UTC) - timedelta(seconds=10)
        router._message_queues["agent-1"].append(queued_msg)

        await router.cleanup_expired_messages()

        assert "agent-1" not in router._message_queues

    def test_queued_message_is_expired(self):
        """Test QueuedMessage expiration check."""
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1",
            envelope=envelope,
            target_agent_id="agent-1",
            ttl_seconds=1,
        )
        queued_msg.expires_at = datetime.now(UTC) - timedelta(seconds=10)

        assert queued_msg.is_expired() is True

    def test_queued_message_can_retry(self):
        """Test QueuedMessage retry check."""
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1", envelope=envelope, target_agent_id="agent-1"
        )

        queued_msg.retry_count = 2
        queued_msg.max_retries = 3

        assert queued_msg.can_retry() is True

    def test_queued_message_cannot_retry_max_exceeded(self):
        """Test QueuedMessage retry fails when max exceeded."""
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1", envelope=envelope, target_agent_id="agent-1"
        )

        queued_msg.retry_count = 3
        queued_msg.max_retries = 3

        assert queued_msg.can_retry() is False


# ==================== Circuit Breaker Tests ====================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_record_agent_failure(self, router):
        """Test recording agent failure."""
        initial_failures = router._circuit_breaker_failures["agent-1"]

        router.record_agent_failure("agent-1")

        assert router._circuit_breaker_failures["agent-1"] == initial_failures + 1

    def test_record_agent_failure_opens_circuit_at_threshold(self, router):
        """Test circuit opens after threshold failures."""
        # Record failures up to threshold
        for _ in range(router._circuit_breaker_threshold):
            router.record_agent_failure("agent-1")

        assert "agent-1" in router._circuit_breaker_reset_time
        assert router._circuit_breaker_reset_time["agent-1"] > datetime.now(UTC)

    def test_record_agent_success_decrements_failures(self, router):
        """Test recording success decrements failure count."""
        router._circuit_breaker_failures["agent-1"] = 3

        router.record_agent_success("agent-1")

        assert router._circuit_breaker_failures["agent-1"] == 2

    def test_record_agent_success_does_not_go_negative(self, router):
        """Test success counter does not go below zero."""
        router._circuit_breaker_failures["agent-1"] = 0

        router.record_agent_success("agent-1")

        assert router._circuit_breaker_failures["agent-1"] == 0

    def test_record_agent_success_no_prior_failures(self, router):
        """Test recording success for agent with no failures."""
        # Should not raise error
        router.record_agent_success("new-agent")
        assert "new-agent" not in router._circuit_breaker_failures

    def test_circuit_breaker_timeout(self, router):
        """Test circuit breaker resets after timeout."""
        # Set reset time to past
        router._circuit_breaker_reset_time["agent-1"] = datetime.now(UTC) - timedelta(
            seconds=10
        )

        # Circuit should be reset (tested in _is_agent_available)
        assert datetime.now(UTC) > router._circuit_breaker_reset_time["agent-1"]


# ==================== Load Balancing Tests ====================


class TestLoadBalancing:
    """Test load balancing functionality."""

    @pytest.mark.asyncio
    async def test_deliver_message_increments_load(self, router, sample_envelope):
        """Test delivering message increments agent load."""
        initial_load = router._agent_load.get("agent-1", 0)

        await router._deliver_message(sample_envelope, "agent-1")

        assert router._agent_load["agent-1"] == initial_load + 1

    def test_decrease_agent_load(self, router):
        """Test decreasing agent load."""
        router._agent_load["agent-1"] = 5

        router.decrease_agent_load("agent-1")

        assert router._agent_load["agent-1"] == 4

    def test_decrease_agent_load_does_not_go_negative(self, router):
        """Test load counter does not go below zero."""
        router._agent_load["agent-1"] = 0

        router.decrease_agent_load("agent-1")

        assert router._agent_load["agent-1"] == 0

    def test_decrease_agent_load_new_agent(self, router):
        """Test decreasing load for agent not in load tracking."""
        router.decrease_agent_load("new-agent")

        # Should not raise error, may or may not create entry
        assert router._agent_load.get("new-agent", 0) >= 0


# ==================== Statistics Tests ====================


class TestStatistics:
    """Test routing statistics."""

    @pytest.mark.asyncio
    async def test_get_routing_stats(self, router, sample_envelope):
        """Test getting routing statistics."""
        # Queue a message
        await router._queue_message(sample_envelope, "agent-1", MessagePriority.NORMAL)

        stats = router.get_routing_stats()

        assert "total_routed" in stats
        assert "capability_matched" in stats
        assert "load_balanced" in stats
        assert "queued_messages" in stats
        assert "failed_routes" in stats
        assert "current_queued_messages" in stats
        assert stats["current_queued_messages"] == 1

    def test_get_queue_info(self, router):
        """Test getting queue info for specific agent."""
        # Create queue with message
        envelope = create_test_envelope(message_id="msg-1")
        queued_msg = QueuedMessage(
            message_id="msg-1", envelope=envelope, target_agent_id="agent-1"
        )
        router._message_queues["agent-1"].append(queued_msg)
        router._agent_load["agent-1"] = 3

        info = router.get_queue_info("agent-1")

        assert info["agent_id"] == "agent-1"
        assert info["queue_size"] == 1
        assert info["oldest_message"] is not None
        assert info["current_load"] == 3

    def test_get_queue_info_no_queue(self, router):
        """Test getting queue info for agent with no queue."""
        info = router.get_queue_info("nonexistent-agent")

        assert info["agent_id"] == "nonexistent-agent"
        assert info["queue_size"] == 0
        assert info["oldest_message"] is None
        assert info["current_load"] == 0


# ==================== Session Context Tests ====================


class TestSessionContext:
    """Test session-aware routing functionality."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_preserve_session_context(
        self, mock_agent_mgr, mock_sess_mgr, router
    ):
        """Test preserving session context after message processing."""
        mock_sess_mgr.update_context = AsyncMock(return_value=True)

        context_updates = {"key": "value", "counter": 5}

        result = await router.preserve_session_context(
            session_id="session-1", agent_id="agent-1", context_updates=context_updates
        )

        assert result is True
        mock_sess_mgr.update_context.assert_called_once_with(
            "session-1", context_updates
        )

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    async def test_route_with_session_prefers_participants(
        self, mock_sess_mgr, router, sample_envelope, mock_session
    ):
        """Test session routing prefers existing participants."""
        # Import models to create a proper mock agent
        from agentcore.a2a_protocol.models.agent import (
            AgentAuthentication,
            AgentCapability,
            AgentCard,
            AgentEndpoint,
            AuthenticationType,
            EndpointType,
        )

        # Create a real AgentCard instance (not mock)
        test_agent = AgentCard(
            agent_id="agent-1",
            agent_name="Test Agent",
            agent_version="1.0.0",
            capabilities=[AgentCapability(name="text-generation")],
            endpoints=[
                AgentEndpoint(url="http://agent-1.local", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        mock_session.participant_agents = ["agent-1"]

        with patch(
            "agentcore.a2a_protocol.services.message_router.agent_manager"
        ) as mock_agent_mgr:
            mock_agent_mgr.get_agent = AsyncMock(return_value=test_agent)
            mock_sess_mgr.get_session = AsyncMock(return_value=mock_session)
            mock_sess_mgr.record_event = AsyncMock()
            mock_sess_mgr.set_agent_state = AsyncMock()

            result = await router.route_with_session(
                sample_envelope,
                session_id="session-1",
                required_capabilities=["text-generation"],
            )

            assert result == "agent-1"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    @patch("agentcore.a2a_protocol.services.message_router.agent_manager")
    async def test_route_with_session_adds_new_participant(
        self,
        mock_agent_mgr,
        mock_sess_mgr,
        router,
        sample_envelope,
        mock_agent,
        mock_session,
    ):
        """Test session routing adds new participant agent."""
        mock_session.participant_agents = []

        mock_agent_mgr.get_agent = AsyncMock(return_value=mock_agent)
        mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(
            return_value=[{"agent_id": "agent-2"}]
        )
        mock_sess_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_sess_mgr.set_agent_state = AsyncMock()
        mock_sess_mgr.record_event = AsyncMock()

        result = await router.route_with_session(
            sample_envelope,
            session_id="session-1",
            required_capabilities=["new-capability"],
        )

        assert result == "agent-2"
        mock_sess_mgr.set_agent_state.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.message_router.session_manager")
    async def test_route_with_session_no_agents_available(
        self, mock_sess_mgr, router, sample_envelope, mock_session
    ):
        """Test session routing returns None when no agents available."""
        mock_session.participant_agents = []
        mock_sess_mgr.get_session = AsyncMock(return_value=mock_session)

        with patch(
            "agentcore.a2a_protocol.services.message_router.agent_manager"
        ) as mock_agent_mgr:
            mock_agent_mgr.discover_agents_by_capabilities = AsyncMock(return_value=[])

            result = await router.route_with_session(
                sample_envelope,
                session_id="session-1",
                required_capabilities=["impossible-capability"],
            )

            assert result is None


# ==================== Global Instance Test ====================


class TestGlobalInstance:
    """Test global message router instance."""

    def test_global_instance_exists(self):
        """Test global message_router instance exists."""
        assert message_router is not None
        assert isinstance(message_router, MessageRouter)

    def test_global_instance_is_singleton(self):
        """Test global instance behaves like singleton."""
        from agentcore.a2a_protocol.services.message_router import message_router as mr1
        from agentcore.a2a_protocol.services.message_router import message_router as mr2

        assert mr1 is mr2
        assert mr1 is mr2
        from agentcore.a2a_protocol.services.message_router import message_router as mr1
        from agentcore.a2a_protocol.services.message_router import message_router as mr2

        assert mr1 is mr2
