"""
Unit tests for Message Routing JSON-RPC Service.

Tests for routing JSON-RPC method handlers covering message routing,
queue management, statistics, and circuit breaker functionality.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest, MessageEnvelope


class TestRouteMessage:
    """Test route.message JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_route_message_success(self, mock_router):
        """Test successful message routing."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_route_message

        mock_router.route_message = AsyncMock(return_value="agent-123")

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.message",
            params={
                "envelope": {
                    "message_id": "msg-123",
                    "source": "sender-agent",
                    "destination": None,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "payload": {
                        "jsonrpc": "2.0",
                        "method": "test.method",
                        "params": {"text": "test message"},
                        "id": "1",
                    },
                }
            },
            id="1",
        )

        result = await handle_route_message(request)

        assert result["success"] is True
        assert result["message_id"] == "msg-123"
        assert result["selected_agent"] == "agent-123"
        assert result["queued"] is False

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_route_message_queued(self, mock_router):
        """Test message routing when no agent available (queued)."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_route_message

        mock_router.route_message = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.message",
            params={
                "envelope": {
                    "message_id": "msg-123",
                    "source": "sender-agent",
                    "destination": None,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "payload": {
                        "jsonrpc": "2.0",
                        "method": "test.method",
                        "params": {"text": "test message"},
                        "id": "1",
                    },
                },
                "required_capabilities": ["capability-1"],
            },
            id="1",
        )

        result = await handle_route_message(request)

        assert result["success"] is True
        assert result["selected_agent"] is None
        assert result["queued"] is True

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_route_message_with_strategy(self, mock_router):
        """Test message routing with custom strategy."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_route_message

        mock_router.route_message = AsyncMock(return_value="agent-123")

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.message",
            params={
                "envelope": {
                    "message_id": "msg-123",
                    "source": "sender-agent",
                    "destination": None,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "payload": {
                        "jsonrpc": "2.0",
                        "method": "test.method",
                        "params": {"text": "test message"},
                        "id": "1",
                    },
                },
                "strategy": "round_robin",
                "priority": "high",
            },
            id="1",
        )

        result = await handle_route_message(request)

        assert result["success"] is True
        assert result["strategy"] == "round_robin"

    @pytest.mark.asyncio
    async def test_route_message_missing_params(self):
        """Test message routing with missing parameters."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_route_message

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.message",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_route_message(request)

    @pytest.mark.asyncio
    async def test_route_message_missing_envelope(self):
        """Test message routing with missing envelope."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_route_message

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.message",
            params={"strategy": "capability_match"},
            id="1",
        )

        with pytest.raises(ValueError, match="Missing required parameter: envelope"):
            await handle_route_message(request)


class TestProcessQueue:
    """Test route.process_queue JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_process_queue_success(self, mock_router):
        """Test successful queue processing."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_process_queue

        mock_router.process_queued_messages = AsyncMock(return_value=5)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.process_queue",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_process_queue(request)

        assert result["success"] is True
        assert result["agent_id"] == "agent-123"
        assert result["processed"] == 5

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_process_queue_no_messages(self, mock_router):
        """Test queue processing when no messages to process."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_process_queue

        mock_router.process_queued_messages = AsyncMock(return_value=0)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.process_queue",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_process_queue(request)

        assert result["processed"] == 0

    @pytest.mark.asyncio
    async def test_process_queue_missing_params(self):
        """Test queue processing with missing parameters."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_process_queue

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.process_queue",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_process_queue(request)


class TestQueueInfo:
    """Test route.get_queue_info JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_get_queue_info_success(self, mock_router):
        """Test getting queue information."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_get_queue_info,
        )

        queue_info = {
            "agent_id": "agent-123",
            "queue_size": 10,
            "oldest_message_age": 120,
            "priority_counts": {"high": 2, "normal": 8},
        }
        mock_router.get_queue_info = Mock(return_value=queue_info)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.get_queue_info",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_get_queue_info(request)

        assert result["agent_id"] == "agent-123"
        assert result["queue_size"] == 10

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_get_queue_info_empty(self, mock_router):
        """Test getting queue info when queue is empty."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_get_queue_info,
        )

        queue_info = {
            "agent_id": "agent-123",
            "queue_size": 0,
        }
        mock_router.get_queue_info = Mock(return_value=queue_info)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.get_queue_info",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_get_queue_info(request)

        assert result["queue_size"] == 0

    @pytest.mark.asyncio
    async def test_get_queue_info_missing_params(self):
        """Test getting queue info with missing parameters."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_get_queue_info,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.get_queue_info",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_get_queue_info(request)


class TestRoutingStats:
    """Test route.get_stats JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_get_routing_stats(self, mock_router):
        """Test getting routing statistics."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_get_routing_stats,
        )

        stats = {
            "total_messages_routed": 1000,
            "queued_messages": 50,
            "average_routing_time_ms": 25,
        }
        mock_router.get_routing_stats = Mock(return_value=stats)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.get_stats",
            params={},
            id="1",
        )

        result = await handle_get_routing_stats(request)

        assert result["success"] is True
        assert result["stats"] == stats
        assert "timestamp" in result


class TestCleanupExpired:
    """Test route.cleanup_expired JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_cleanup_expired_messages(self, mock_router):
        """Test cleaning up expired messages."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_cleanup_expired,
        )

        mock_router.cleanup_expired_messages = AsyncMock(return_value=10)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.cleanup_expired",
            params={},
            id="1",
        )

        result = await handle_cleanup_expired(request)

        assert result["success"] is True
        assert result["removed"] == 10

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_cleanup_expired_no_messages(self, mock_router):
        """Test cleanup when no expired messages."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_cleanup_expired,
        )

        mock_router.cleanup_expired_messages = AsyncMock(return_value=0)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.cleanup_expired",
            params={},
            id="1",
        )

        result = await handle_cleanup_expired(request)

        assert result["removed"] == 0


class TestCircuitBreaker:
    """Test circuit breaker methods (record_failure, record_success, decrease_load)."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_record_failure(self, mock_router):
        """Test recording agent failure."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_record_failure,
        )

        mock_router.record_agent_failure = Mock()

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.record_failure",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_record_failure(request)

        assert result["success"] is True
        assert result["agent_id"] == "agent-123"
        mock_router.record_agent_failure.assert_called_once_with("agent-123")

    @pytest.mark.asyncio
    async def test_record_failure_missing_params(self):
        """Test recording failure with missing parameters."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_record_failure,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.record_failure",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_record_failure(request)

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_record_success(self, mock_router):
        """Test recording agent success."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_record_success,
        )

        mock_router.record_agent_success = Mock()

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.record_success",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_record_success(request)

        assert result["success"] is True
        assert result["agent_id"] == "agent-123"
        mock_router.record_agent_success.assert_called_once_with("agent-123")

    @pytest.mark.asyncio
    async def test_record_success_missing_params(self):
        """Test recording success with missing parameters."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import (
            handle_record_success,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.record_success",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_record_success(request)

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.routing_jsonrpc.message_router")
    async def test_decrease_load(self, mock_router):
        """Test decreasing agent load."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_decrease_load

        mock_router.decrease_agent_load = Mock()

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.decrease_load",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_decrease_load(request)

        assert result["success"] is True
        assert result["agent_id"] == "agent-123"
        mock_router.decrease_agent_load.assert_called_once_with("agent-123")

    @pytest.mark.asyncio
    async def test_decrease_load_missing_params(self):
        """Test decreasing load with missing parameters."""
        from agentcore.a2a_protocol.services.routing_jsonrpc import handle_decrease_load

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="route.decrease_load",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_decrease_load(request)
