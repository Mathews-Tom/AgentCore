"""
Event System JSON-RPC Methods

JSON-RPC 2.0 methods for event publishing, subscriptions, and management.
"""

from datetime import UTC, datetime
from typing import Any

import structlog

from agentcore.a2a_protocol.models.events import (
    EventPriority,
    EventPublishRequest,
    EventSubscribeRequest,
    EventType,
    EventUnsubscribeRequest,
)
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.event_manager import event_manager
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method

logger = structlog.get_logger()


@register_jsonrpc_method("event.publish")
async def handle_publish_event(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Publish an event.

    Method: event.publish
    Params:
        - event_type: string (EventType enum)
        - source: string
        - data: object
        - priority: string (optional, default "normal")
        - metadata: object (optional)
        - correlation_id: string (optional)

    Returns:
        Event publish response with event_id and subscribers_notified
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: event_type, source, data")

    try:
        event_type_str = request.params.get("event_type")
        source = request.params.get("source")
        data = request.params.get("data")

        if not event_type_str or not source or data is None:
            raise ValueError(
                "Missing required parameters: event_type, source, and/or data"
            )

        # Parse event type
        event_type = EventType(event_type_str)

        # Parse priority
        priority_str = request.params.get("priority", "normal")
        priority = EventPriority(priority_str)

        metadata = request.params.get("metadata")
        correlation_id = request.params.get("correlation_id")

        # Publish event
        response = await event_manager.publish_event(
            event_type=event_type,
            source=source,
            data=data,
            priority=priority,
            metadata=metadata,
            correlation_id=correlation_id,
        )

        logger.info(
            "Event published via JSON-RPC",
            event_type=event_type.value,
            source=source,
            event_id=response.event_id,
            method="event.publish",
        )

        return response.model_dump()

    except Exception as e:
        logger.error("Event publish failed", error=str(e))
        raise


@register_jsonrpc_method("event.subscribe")
async def handle_subscribe(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Subscribe to events.

    Method: event.subscribe
    Params:
        - subscriber_id: string
        - event_types: array of EventType enums
        - filters: object (optional)
        - ttl_seconds: number (optional)

    Returns:
        Subscription response with subscription_id
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: subscriber_id, event_types")

    try:
        subscriber_id = request.params.get("subscriber_id")
        event_types_str = request.params.get("event_types")

        if not subscriber_id or not event_types_str:
            raise ValueError(
                "Missing required parameters: subscriber_id and/or event_types"
            )

        # Parse event types
        event_types = [EventType(et) for et in event_types_str]

        filters = request.params.get("filters")
        ttl_seconds = request.params.get("ttl_seconds")

        # Create subscription
        response = await event_manager.subscribe(
            subscriber_id=subscriber_id,
            event_types=event_types,
            filters=filters,
            ttl_seconds=ttl_seconds,
        )

        logger.info(
            "Event subscription created via JSON-RPC",
            subscriber_id=subscriber_id,
            subscription_id=response.subscription_id,
            event_types=[et.value for et in event_types],
            method="event.subscribe",
        )

        return response.model_dump()

    except Exception as e:
        logger.error("Event subscription failed", error=str(e))
        raise


@register_jsonrpc_method("event.unsubscribe")
async def handle_unsubscribe(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Unsubscribe from events.

    Method: event.unsubscribe
    Params:
        - subscription_id: string

    Returns:
        Unsubscribe confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: subscription_id")

    subscription_id = request.params.get("subscription_id")
    if not subscription_id:
        raise ValueError("Missing required parameter: subscription_id")

    try:
        success = await event_manager.unsubscribe(subscription_id)

        if not success:
            raise ValueError(f"Subscription not found: {subscription_id}")

        logger.info(
            "Event subscription removed via JSON-RPC",
            subscription_id=subscription_id,
            method="event.unsubscribe",
        )

        return {
            "success": True,
            "subscription_id": subscription_id,
            "message": "Subscription removed successfully",
        }

    except Exception as e:
        logger.error("Event unsubscribe failed", error=str(e))
        raise


@register_jsonrpc_method("event.list_subscriptions")
async def handle_list_subscriptions(request: JsonRpcRequest) -> dict[str, Any]:
    """
    List event subscriptions.

    Method: event.list_subscriptions
    Params:
        - subscriber_id: string (optional, filter by subscriber)

    Returns:
        List of subscriptions
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    subscriber_id = params.get("subscriber_id")

    subscriptions = event_manager.get_subscriptions(subscriber_id)

    logger.debug(
        "Subscriptions listed via JSON-RPC",
        subscriber_id=subscriber_id,
        count=len(subscriptions),
        method="event.list_subscriptions",
    )

    return {
        "success": True,
        "subscriptions": [sub.model_dump(mode="json") for sub in subscriptions],
        "count": len(subscriptions),
    }


@register_jsonrpc_method("event.get_history")
async def handle_get_history(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get event history.

    Method: event.get_history
    Params:
        - event_type: string (optional, filter by event type)
        - limit: number (optional, default 100)

    Returns:
        List of historical events
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    event_type_str = params.get("event_type")
    limit = params.get("limit", 100)

    event_type = None
    if event_type_str:
        event_type = EventType(event_type_str)

    events = event_manager.get_event_history(event_type, limit)

    logger.debug(
        "Event history retrieved via JSON-RPC",
        event_type=event_type.value if event_type else None,
        count=len(events),
        method="event.get_history",
    )

    return {
        "success": True,
        "events": [event.to_notification() for event in events],
        "count": len(events),
    }


@register_jsonrpc_method("event.get_dead_letter_queue")
async def handle_get_dead_letter_queue(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get messages from dead letter queue.

    Method: event.get_dead_letter_queue
    Params:
        - limit: number (optional, default 100)

    Returns:
        List of dead letter messages
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    limit = params.get("limit", 100)

    messages = event_manager.get_dead_letter_messages(limit)

    logger.debug(
        "Dead letter queue retrieved via JSON-RPC",
        count=len(messages),
        method="event.get_dead_letter_queue",
    )

    return {
        "success": True,
        "messages": [msg.model_dump(mode="json") for msg in messages],
        "count": len(messages),
    }


@register_jsonrpc_method("event.retry_dead_letter")
async def handle_retry_dead_letter(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Retry delivery of dead letter message.

    Method: event.retry_dead_letter
    Params:
        - message_id: string

    Returns:
        Retry result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: message_id")

    message_id = request.params.get("message_id")
    if not message_id:
        raise ValueError("Missing required parameter: message_id")

    try:
        success = await event_manager.retry_dead_letter_message(message_id)

        if not success:
            raise ValueError(f"Retry failed for message: {message_id}")

        logger.info(
            "Dead letter message retry successful via JSON-RPC",
            message_id=message_id,
            method="event.retry_dead_letter",
        )

        return {
            "success": True,
            "message_id": message_id,
            "message": "Dead letter message retried successfully",
        }

    except Exception as e:
        logger.error("Dead letter retry failed", error=str(e), message_id=message_id)
        raise


@register_jsonrpc_method("event.get_stats")
async def handle_get_event_stats(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get event system statistics.

    Method: event.get_stats
    Params: none

    Returns:
        Event system statistics
    """
    stats = event_manager.get_statistics()

    logger.debug("Event stats retrieved via JSON-RPC", method="event.get_stats")

    return {"success": True, "stats": stats, "timestamp": datetime.now(UTC).isoformat()}


@register_jsonrpc_method("event.cleanup_expired")
async def handle_cleanup_expired(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Cleanup expired subscriptions and dead connections.

    Method: event.cleanup_expired
    Params: none

    Returns:
        Cleanup results
    """
    subscriptions_removed = await event_manager.cleanup_expired_subscriptions()
    connections_removed = await event_manager.cleanup_dead_connections()

    logger.info(
        "Event system cleanup completed via JSON-RPC",
        subscriptions_removed=subscriptions_removed,
        connections_removed=connections_removed,
        method="event.cleanup_expired",
    )

    return {
        "success": True,
        "subscriptions_removed": subscriptions_removed,
        "connections_removed": connections_removed,
    }


# Log registration on import
logger.info(
    "Event JSON-RPC methods registered",
    methods=[
        "event.publish",
        "event.subscribe",
        "event.unsubscribe",
        "event.list_subscriptions",
        "event.get_history",
        "event.get_dead_letter_queue",
        "event.retry_dead_letter",
        "event.get_stats",
        "event.cleanup_expired",
    ],
)
