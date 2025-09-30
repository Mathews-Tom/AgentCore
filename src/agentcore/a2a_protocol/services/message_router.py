"""
Message Router Service

Intelligent message routing with capability-based selection, load balancing,
and message queuing for offline agents.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog

from agentcore.a2a_protocol.models.agent import AgentStatus
from agentcore.a2a_protocol.models.jsonrpc import MessageEnvelope, JsonRpcRequest
from agentcore.a2a_protocol.services.agent_manager import agent_manager


logger = structlog.get_logger()


class RoutingStrategy(str, Enum):
    """Message routing strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    CAPABILITY_MATCH = "capability_match"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class QueuedMessage:
    """Message queued for offline agent."""

    def __init__(
        self,
        message_id: str,
        envelope: MessageEnvelope,
        target_agent_id: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: int = 3600
    ):
        self.message_id = message_id
        self.envelope = envelope
        self.target_agent_id = target_agent_id
        self.priority = priority
        self.queued_at = datetime.utcnow()
        self.expires_at = self.queued_at + timedelta(seconds=ttl_seconds)
        self.retry_count = 0
        self.max_retries = 3

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return datetime.utcnow() > self.expires_at

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1


class MessageRouter:
    """
    Intelligent message routing service.

    Handles capability-based agent selection, load balancing,
    and message queuing for offline agents.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # Message queues for offline agents
        self._message_queues: Dict[str, deque[QueuedMessage]] = defaultdict(deque)

        # Agent load tracking (message count per agent)
        self._agent_load: Dict[str, int] = defaultdict(int)

        # Round-robin counters per capability
        self._round_robin_counters: Dict[str, int] = defaultdict(int)

        # Circuit breaker state
        self._circuit_breaker_failures: Dict[str, int] = defaultdict(int)
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60  # seconds
        self._circuit_breaker_reset_time: Dict[str, datetime] = {}

        # Routing statistics
        self._routing_stats = {
            "total_routed": 0,
            "capability_matched": 0,
            "load_balanced": 0,
            "queued_messages": 0,
            "failed_routes": 0,
        }

    async def route_message(
        self,
        envelope: MessageEnvelope,
        required_capabilities: Optional[List[str]] = None,
        strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Optional[str]:
        """
        Route message to appropriate agent.

        Args:
            envelope: Message envelope to route
            required_capabilities: Required agent capabilities (optional)
            strategy: Routing strategy to use
            priority: Message priority for queuing

        Returns:
            Selected agent ID, or None if no agent available

        Raises:
            ValueError: If routing fails
        """
        try:
            # Check if destination is explicitly specified
            if envelope.destination:
                agent = await agent_manager.get_agent(envelope.destination)
                if agent and agent.is_active():
                    await self._deliver_message(envelope, envelope.destination)
                    return envelope.destination

                # Queue message if agent is offline
                await self._queue_message(envelope, envelope.destination, priority)
                self.logger.info(
                    "Message queued for offline agent",
                    target_agent=envelope.destination,
                    message_id=envelope.message_id
                )
                return None

            # Find suitable agents based on capabilities
            if required_capabilities:
                candidates = await self._find_capable_agents(required_capabilities)
            else:
                # Get all active agents
                all_agents = await agent_manager.list_all_agents()
                candidates = [
                    a["agent_id"] for a in all_agents
                    if await self._is_agent_available(a["agent_id"])
                ]

            if not candidates:
                self._routing_stats["failed_routes"] += 1
                raise ValueError(
                    f"No agents available with required capabilities: {required_capabilities}"
                )

            # Select agent based on strategy
            selected_agent = await self._select_agent(candidates, strategy)

            if selected_agent:
                await self._deliver_message(envelope, selected_agent)
                self._routing_stats["total_routed"] += 1
                if required_capabilities:
                    self._routing_stats["capability_matched"] += 1

                self.logger.info(
                    "Message routed",
                    message_id=envelope.message_id,
                    source=envelope.source,
                    target=selected_agent,
                    strategy=strategy.value
                )
                return selected_agent

            self._routing_stats["failed_routes"] += 1
            raise ValueError("Failed to select agent for routing")

        except Exception as e:
            self.logger.error(
                "Message routing failed",
                error=str(e),
                message_id=envelope.message_id,
                required_capabilities=required_capabilities
            )
            raise

    async def _find_capable_agents(self, required_capabilities: List[str]) -> List[str]:
        """
        Find agents matching required capabilities.

        Args:
            required_capabilities: List of required capability names

        Returns:
            List of agent IDs that match all required capabilities
        """
        # Query agents with capability filters
        matching_agents = []

        for capability in required_capabilities:
            query_result = await agent_manager.discover_agents_by_capabilities([capability])
            if query_result:
                agent_ids = [agent["agent_id"] for agent in query_result]
                matching_agents.append(set(agent_ids))

        if not matching_agents:
            return []

        # Find intersection (agents with ALL required capabilities)
        capable_agents = set.intersection(*matching_agents)

        # Filter out unavailable agents
        available_agents = []
        for agent_id in capable_agents:
            if await self._is_agent_available(agent_id):
                available_agents.append(agent_id)

        return available_agents

    async def _is_agent_available(self, agent_id: str) -> bool:
        """
        Check if agent is available for routing.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent is active and not circuit broken
        """
        # Check circuit breaker
        if agent_id in self._circuit_breaker_reset_time:
            if datetime.utcnow() < self._circuit_breaker_reset_time[agent_id]:
                return False
            else:
                # Reset circuit breaker
                del self._circuit_breaker_reset_time[agent_id]
                self._circuit_breaker_failures[agent_id] = 0

        # Check agent status
        agent = await agent_manager.get_agent(agent_id)
        return agent is not None and agent.is_active()

    async def _select_agent(
        self,
        candidates: List[str],
        strategy: RoutingStrategy
    ) -> Optional[str]:
        """
        Select agent from candidates based on strategy.

        Args:
            candidates: List of candidate agent IDs
            strategy: Selection strategy

        Returns:
            Selected agent ID, or None
        """
        if not candidates:
            return None

        if strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._round_robin_select(candidates)

        elif strategy == RoutingStrategy.LEAST_LOADED:
            return await self._least_loaded_select(candidates)

        elif strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(candidates)

        else:  # CAPABILITY_MATCH or default
            return candidates[0]

    async def _round_robin_select(self, candidates: List[str]) -> str:
        """Round-robin selection across candidates."""
        key = "|".join(sorted(candidates))
        counter = self._round_robin_counters[key]
        selected = candidates[counter % len(candidates)]
        self._round_robin_counters[key] = (counter + 1) % len(candidates)
        self._routing_stats["load_balanced"] += 1
        return selected

    async def _least_loaded_select(self, candidates: List[str]) -> str:
        """Select least loaded agent."""
        least_loaded = min(candidates, key=lambda a: self._agent_load.get(a, 0))
        self._routing_stats["load_balanced"] += 1
        return least_loaded

    async def _deliver_message(self, envelope: MessageEnvelope, agent_id: str) -> None:
        """
        Deliver message to agent.

        Args:
            envelope: Message envelope
            agent_id: Target agent ID
        """
        # Increment agent load
        self._agent_load[agent_id] = self._agent_load.get(agent_id, 0) + 1

        # TODO: Implement actual message delivery via WebSocket
        # For now, just log the delivery
        self.logger.debug(
            "Message delivered",
            message_id=envelope.message_id,
            agent_id=agent_id,
            current_load=self._agent_load[agent_id]
        )

    async def _queue_message(
        self,
        envelope: MessageEnvelope,
        agent_id: str,
        priority: MessagePriority,
        ttl_seconds: int = 3600
    ) -> None:
        """
        Queue message for offline agent.

        Args:
            envelope: Message envelope
            agent_id: Target agent ID
            priority: Message priority
            ttl_seconds: Time-to-live in seconds
        """
        message = QueuedMessage(
            message_id=envelope.message_id,
            envelope=envelope,
            target_agent_id=agent_id,
            priority=priority,
            ttl_seconds=ttl_seconds
        )

        # Insert based on priority
        queue = self._message_queues[agent_id]

        if priority == MessagePriority.CRITICAL:
            queue.appendleft(message)
        else:
            queue.append(message)

        self._routing_stats["queued_messages"] += 1

        self.logger.info(
            "Message queued",
            message_id=envelope.message_id,
            agent_id=agent_id,
            priority=priority.value,
            queue_size=len(queue)
        )

    async def process_queued_messages(self, agent_id: str) -> int:
        """
        Process queued messages for agent that came online.

        Args:
            agent_id: Agent identifier

        Returns:
            Number of messages processed
        """
        queue = self._message_queues.get(agent_id)
        if not queue:
            return 0

        processed = 0
        failed = []

        while queue:
            message = queue.popleft()

            # Check if expired
            if message.is_expired():
                self.logger.warning(
                    "Queued message expired",
                    message_id=message.message_id,
                    agent_id=agent_id
                )
                continue

            try:
                # Attempt delivery
                await self._deliver_message(message.envelope, agent_id)
                processed += 1
            except Exception as e:
                self.logger.error(
                    "Failed to deliver queued message",
                    message_id=message.message_id,
                    error=str(e)
                )

                message.increment_retry()
                if message.can_retry():
                    failed.append(message)

        # Re-queue failed messages
        for message in failed:
            queue.append(message)

        self.logger.info(
            "Processed queued messages",
            agent_id=agent_id,
            processed=processed,
            failed=len(failed),
            remaining=len(queue)
        )

        return processed

    async def cleanup_expired_messages(self) -> int:
        """
        Remove expired messages from all queues.

        Returns:
            Number of messages removed
        """
        removed = 0

        for agent_id, queue in list(self._message_queues.items()):
            expired = []

            for message in queue:
                if message.is_expired():
                    expired.append(message)

            for message in expired:
                queue.remove(message)
                removed += 1

            # Remove empty queues
            if not queue:
                del self._message_queues[agent_id]

        if removed > 0:
            self.logger.info("Cleaned up expired messages", removed=removed)

        return removed

    def record_agent_failure(self, agent_id: str) -> None:
        """
        Record agent failure for circuit breaker.

        Args:
            agent_id: Agent identifier
        """
        self._circuit_breaker_failures[agent_id] += 1

        if self._circuit_breaker_failures[agent_id] >= self._circuit_breaker_threshold:
            reset_time = datetime.utcnow() + timedelta(seconds=self._circuit_breaker_timeout)
            self._circuit_breaker_reset_time[agent_id] = reset_time

            self.logger.warning(
                "Circuit breaker opened for agent",
                agent_id=agent_id,
                failures=self._circuit_breaker_failures[agent_id],
                reset_at=reset_time.isoformat()
            )

    def record_agent_success(self, agent_id: str) -> None:
        """
        Record successful agent communication.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._circuit_breaker_failures:
            self._circuit_breaker_failures[agent_id] = max(
                0,
                self._circuit_breaker_failures[agent_id] - 1
            )

    def decrease_agent_load(self, agent_id: str) -> None:
        """
        Decrease agent load counter after message processing.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agent_load:
            self._agent_load[agent_id] = max(0, self._agent_load[agent_id] - 1)

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_queued = sum(len(q) for q in self._message_queues.values())

        return {
            **self._routing_stats,
            "current_queued_messages": total_queued,
            "agents_with_queues": len(self._message_queues),
            "circuit_broken_agents": len(self._circuit_breaker_reset_time),
            "total_agent_load": sum(self._agent_load.values()),
        }

    def get_queue_info(self, agent_id: str) -> Dict[str, Any]:
        """
        Get queue information for specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Queue information dictionary
        """
        queue = self._message_queues.get(agent_id, deque())

        return {
            "agent_id": agent_id,
            "queue_size": len(queue),
            "oldest_message": queue[0].queued_at.isoformat() if queue else None,
            "current_load": self._agent_load.get(agent_id, 0),
        }


# Global message router instance
message_router = MessageRouter()