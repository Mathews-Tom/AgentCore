"""
Real-time Communication Module

WebSocket and Server-Sent Events support for real-time agent communication.
"""

from __future__ import annotations

from gateway.realtime.connection_pool import ConnectionPool
from gateway.realtime.event_bus import EventBus, EventMessage
from gateway.realtime.sse import SSEManager
from gateway.realtime.subscriptions import SubscriptionManager
from gateway.realtime.websocket import WebSocketConnectionManager

__all__ = [
    "ConnectionPool",
    "EventBus",
    "EventMessage",
    "SSEManager",
    "SubscriptionManager",
    "WebSocketConnectionManager",
]
