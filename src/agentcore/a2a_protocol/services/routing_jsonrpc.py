"""
Message Routing JSON-RPC Methods

JSON-RPC 2.0 methods for message routing, queue management, and routing statistics.
"""

from datetime import UTC, datetime
from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest, MessageEnvelope
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.message_router import (
    MessagePriority,
    RoutingStrategy,
    message_router,
)

logger = structlog.get_logger()


@register_jsonrpc_method("route.message")
async def handle_route_message(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Route a message to appropriate agent.

    Method: route.message
    Params:
        - envelope: MessageEnvelope object
        - required_capabilities: array of strings (optional)
        - strategy: string (optional, default "capability_match")
        - priority: string (optional, default "normal")

    Returns:
        Routing result with selected agent ID
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: envelope")

    try:
        # Extract parameters
        envelope_data = request.params.get("envelope")
        if not envelope_data:
            raise ValueError("Missing required parameter: envelope")

        required_capabilities = request.params.get("required_capabilities")
        strategy_str = request.params.get("strategy", "capability_match")
        priority_str = request.params.get("priority", "normal")

        # Parse envelope
        envelope = MessageEnvelope.model_validate(envelope_data)

        # Parse strategy and priority
        strategy = RoutingStrategy(strategy_str)
        priority = MessagePriority(priority_str)

        # Route message
        selected_agent = await message_router.route_message(
            envelope=envelope,
            required_capabilities=required_capabilities,
            strategy=strategy,
            priority=priority,
        )

        logger.info(
            "Message routed via JSON-RPC",
            message_id=envelope.message_id,
            selected_agent=selected_agent,
            strategy=strategy.value,
            method="route.message",
        )

        return {
            "success": True,
            "message_id": envelope.message_id,
            "selected_agent": selected_agent,
            "strategy": strategy.value,
            "queued": selected_agent is None,
        }

    except Exception as e:
        logger.error("Message routing failed", error=str(e))
        raise


@register_jsonrpc_method("route.process_queue")
async def handle_process_queue(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Process queued messages for an agent.

    Method: route.process_queue
    Params:
        - agent_id: string

    Returns:
        Number of messages processed
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    try:
        processed = await message_router.process_queued_messages(agent_id)

        logger.info(
            "Processed queued messages via JSON-RPC",
            agent_id=agent_id,
            processed=processed,
            method="route.process_queue",
        )

        return {"success": True, "agent_id": agent_id, "processed": processed}

    except Exception as e:
        logger.error("Queue processing failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("route.get_queue_info")
async def handle_get_queue_info(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get queue information for an agent.

    Method: route.get_queue_info
    Params:
        - agent_id: string

    Returns:
        Queue information
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    queue_info = message_router.get_queue_info(agent_id)

    logger.debug(
        "Queue info retrieved via JSON-RPC",
        agent_id=agent_id,
        method="route.get_queue_info",
    )

    return queue_info


@register_jsonrpc_method("route.get_stats")
async def handle_get_routing_stats(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get routing statistics.

    Method: route.get_stats
    Params: none

    Returns:
        Routing statistics
    """
    stats = message_router.get_routing_stats()

    logger.debug("Routing stats retrieved via JSON-RPC", method="route.get_stats")

    return {"success": True, "stats": stats, "timestamp": datetime.now(UTC).isoformat()}


@register_jsonrpc_method("route.cleanup_expired")
async def handle_cleanup_expired(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Cleanup expired messages from queues.

    Method: route.cleanup_expired
    Params: none

    Returns:
        Number of messages removed
    """
    removed = await message_router.cleanup_expired_messages()

    logger.info(
        "Expired messages cleaned up via JSON-RPC",
        removed=removed,
        method="route.cleanup_expired",
    )

    return {"success": True, "removed": removed}


@register_jsonrpc_method("route.record_failure")
async def handle_record_failure(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Record agent failure for circuit breaker.

    Method: route.record_failure
    Params:
        - agent_id: string

    Returns:
        Success confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    message_router.record_agent_failure(agent_id)

    logger.info(
        "Agent failure recorded via JSON-RPC",
        agent_id=agent_id,
        method="route.record_failure",
    )

    return {"success": True, "agent_id": agent_id, "message": "Failure recorded"}


@register_jsonrpc_method("route.record_success")
async def handle_record_success(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Record agent success.

    Method: route.record_success
    Params:
        - agent_id: string

    Returns:
        Success confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    message_router.record_agent_success(agent_id)

    logger.debug(
        "Agent success recorded via JSON-RPC",
        agent_id=agent_id,
        method="route.record_success",
    )

    return {"success": True, "agent_id": agent_id}


@register_jsonrpc_method("route.decrease_load")
async def handle_decrease_load(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Decrease agent load counter.

    Method: route.decrease_load
    Params:
        - agent_id: string

    Returns:
        Success confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    message_router.decrease_agent_load(agent_id)

    logger.debug(
        "Agent load decreased via JSON-RPC",
        agent_id=agent_id,
        method="route.decrease_load",
    )

    return {"success": True, "agent_id": agent_id}


# Log registration on import
logger.info(
    "Routing JSON-RPC methods registered",
    methods=[
        "route.message",
        "route.process_queue",
        "route.get_queue_info",
        "route.get_stats",
        "route.cleanup_expired",
        "route.record_failure",
        "route.record_success",
        "route.decrease_load",
    ],
)
