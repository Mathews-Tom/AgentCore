"""
Comprehensive tests for WebSocket router.

Tests cover WebSocket connection lifecycle, message handling, and error scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocket, WebSocketDisconnect

from agentcore.a2a_protocol.routers.websocket import websocket_events_endpoint


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = MagicMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    return ws


@pytest.fixture
def mock_event_manager():
    """Create a mock event manager."""
    manager = MagicMock()
    manager.register_websocket = AsyncMock(return_value="conn-123")
    manager.close_connection = AsyncMock()
    return manager


# ==================== Connection Tests ====================


@pytest.mark.asyncio
async def test_websocket_connection_success(mock_websocket, mock_event_manager):
    """Test successful WebSocket connection establishment."""
    # Setup: websocket will disconnect after receiving connection confirmation
    mock_websocket.receive_json.side_effect = WebSocketDisconnect()

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Verify connection flow
    mock_websocket.accept.assert_called_once()
    mock_event_manager.register_websocket.assert_called_once_with(mock_websocket, "subscriber-1")

    # Verify connection confirmation was sent
    mock_websocket.send_json.assert_called_once()
    call_args = mock_websocket.send_json.call_args[0][0]
    assert call_args["message_type"] == "connected"
    assert call_args["payload"]["connection_id"] == "conn-123"
    assert call_args["payload"]["subscriber_id"] == "subscriber-1"


@pytest.mark.asyncio
async def test_websocket_connection_cleanup(mock_websocket, mock_event_manager):
    """Test WebSocket connection cleanup on disconnect."""
    mock_websocket.receive_json.side_effect = WebSocketDisconnect()

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Verify cleanup was called
    mock_event_manager.close_connection.assert_called_once_with("conn-123")


# ==================== Message Handling Tests ====================


@pytest.mark.asyncio
async def test_websocket_ping_pong(mock_websocket, mock_event_manager):
    """Test ping/pong message handling."""
    # Setup: send ping, then disconnect
    mock_websocket.receive_json.side_effect = [
        {
            "message_type": "ping",
            "payload": {"timestamp": "2024-01-01T00:00:00Z"}
        },
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Verify pong response
    assert mock_websocket.send_json.call_count == 2  # connection + pong
    pong_call = mock_websocket.send_json.call_args_list[1][0][0]
    assert pong_call["message_type"] == "pong"
    assert pong_call["payload"]["timestamp"] == "2024-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_websocket_subscribe_message(mock_websocket, mock_event_manager):
    """Test subscribe message handling."""
    # Setup: send subscribe, then disconnect
    mock_websocket.receive_json.side_effect = [
        {
            "message_type": "subscribe",
            "payload": {"event_types": ["task.created"]}
        },
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Verify message was received (subscriptions handled via JSON-RPC)
    assert mock_websocket.receive_json.call_count == 2


@pytest.mark.asyncio
async def test_websocket_unknown_message_type(mock_websocket, mock_event_manager):
    """Test handling of unknown message type."""
    # Setup: send unknown message, then disconnect
    mock_websocket.receive_json.side_effect = [
        {
            "message_type": "unknown_type",
            "payload": {}
        },
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Should continue without error
    assert mock_websocket.receive_json.call_count == 2


# ==================== Timeout Tests ====================


@pytest.mark.asyncio
async def test_websocket_timeout_sends_ping(mock_websocket, mock_event_manager):
    """Test that timeout sends ping to keep connection alive."""
    # Setup: timeout, then disconnect
    mock_websocket.receive_json.side_effect = [
        asyncio.TimeoutError(),
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Verify ping was sent after timeout
    assert mock_websocket.send_json.call_count >= 2  # connection + ping
    # Find the ping message
    ping_sent = False
    for call in mock_websocket.send_json.call_args_list:
        if call[0][0].get("message_type") == "ping":
            ping_sent = True
            break
    assert ping_sent


@pytest.mark.asyncio
async def test_websocket_timeout_ping_fails(mock_websocket, mock_event_manager):
    """Test handling when ping fails after timeout."""
    # Setup: timeout, ping fails
    mock_websocket.receive_json.side_effect = asyncio.TimeoutError()
    mock_websocket.send_json.side_effect = [
        None,  # connection confirmation succeeds
        Exception("Connection lost")  # ping fails
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Should close connection gracefully
    mock_event_manager.close_connection.assert_called_once()


# ==================== Error Handling Tests ====================


@pytest.mark.asyncio
async def test_websocket_disconnect(mock_websocket, mock_event_manager):
    """Test WebSocketDisconnect exception handling."""
    mock_websocket.receive_json.side_effect = WebSocketDisconnect()

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        # Should not raise exception
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Connection should be closed
    mock_event_manager.close_connection.assert_called_once_with("conn-123")


@pytest.mark.asyncio
async def test_websocket_generic_exception(mock_websocket, mock_event_manager):
    """Test generic exception handling."""
    mock_websocket.receive_json.side_effect = Exception("Unexpected error")

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        # Should not raise exception
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Connection should be closed
    mock_event_manager.close_connection.assert_called_once_with("conn-123")


@pytest.mark.asyncio
async def test_websocket_registration_failure(mock_websocket):
    """Test handling when registration fails."""
    mock_event_manager = MagicMock()
    mock_event_manager.register_websocket = AsyncMock(side_effect=Exception("Registration failed"))
    mock_event_manager.close_connection = AsyncMock()

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        # Should handle error gracefully
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # close_connection should not be called if registration failed (connection_id is None)
    mock_event_manager.close_connection.assert_not_called()


# ==================== Multiple Message Tests ====================


@pytest.mark.asyncio
async def test_websocket_multiple_messages(mock_websocket, mock_event_manager):
    """Test handling multiple messages in sequence."""
    # Setup: ping, subscribe, ping, disconnect
    mock_websocket.receive_json.side_effect = [
        {"message_type": "ping", "payload": {"timestamp": "1"}},
        {"message_type": "subscribe", "payload": {}},
        {"message_type": "ping", "payload": {"timestamp": "2"}},
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Verify all messages were processed
    assert mock_websocket.receive_json.call_count == 4
    # Verify pong responses (2 pongs + 1 connection = 3 sends)
    assert mock_websocket.send_json.call_count == 3


# ==================== Edge Cases ====================


@pytest.mark.asyncio
async def test_websocket_empty_payload(mock_websocket, mock_event_manager):
    """Test message with empty payload."""
    mock_websocket.receive_json.side_effect = [
        {"message_type": "ping", "payload": {}},
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Should handle empty payload
    pong_call = mock_websocket.send_json.call_args_list[1][0][0]
    assert pong_call["message_type"] == "pong"
    assert pong_call["payload"]["timestamp"] is None


@pytest.mark.asyncio
async def test_websocket_missing_message_type(mock_websocket, mock_event_manager):
    """Test message without message_type field."""
    mock_websocket.receive_json.side_effect = [
        {"payload": {}},  # Missing message_type
        WebSocketDisconnect()
    ]

    with patch('agentcore.a2a_protocol.routers.websocket.event_manager', mock_event_manager):
        await websocket_events_endpoint(mock_websocket, "subscriber-1")

    # Should handle gracefully (message_type will be None)
    assert mock_websocket.receive_json.call_count == 2
