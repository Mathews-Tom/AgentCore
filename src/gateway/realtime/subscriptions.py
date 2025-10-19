"""
Subscription Management

Manages client subscriptions to topics and event types with filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from gateway.realtime.event_bus import EventType

logger = structlog.get_logger()


@dataclass
class SubscriptionFilter:
    """Subscription filter for event filtering."""

    # Filter by specific agent IDs
    agent_ids: set[str] = field(default_factory=set)

    # Filter by specific task IDs
    task_ids: set[str] = field(default_factory=set)

    # Filter by specific workflow IDs
    workflow_ids: set[str] = field(default_factory=set)

    # Filter by user ID (for user-specific events)
    user_id: str | None = None

    # Custom metadata filters (key-value pairs)
    metadata_filters: dict[str, Any] = field(default_factory=dict)

    def matches(self, event_data: dict[str, Any]) -> bool:
        """
        Check if event matches filter criteria.

        Args:
            event_data: Event payload and metadata

        Returns:
            True if event matches filter, False otherwise
        """
        # Extract IDs from event data
        agent_id = event_data.get("agent_id")
        task_id = event_data.get("task_id")
        workflow_id = event_data.get("workflow_id")
        event_user_id = event_data.get("user_id")
        event_metadata = event_data.get("metadata", {})

        # Check agent ID filter
        if self.agent_ids and agent_id not in self.agent_ids:
            return False

        # Check task ID filter
        if self.task_ids and task_id not in self.task_ids:
            return False

        # Check workflow ID filter
        if self.workflow_ids and workflow_id not in self.workflow_ids:
            return False

        # Check user ID filter
        if self.user_id and event_user_id != self.user_id:
            return False

        # Check metadata filters
        for key, value in self.metadata_filters.items():
            if event_metadata.get(key) != value:
                return False

        return True


@dataclass
class Subscription:
    """Client subscription to topics and event types."""

    subscription_id: str
    client_id: str
    topics: set[str]
    event_types: set[EventType]
    filters: SubscriptionFilter
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert subscription to dictionary."""
        return {
            "subscription_id": self.subscription_id,
            "client_id": self.client_id,
            "topics": list(self.topics),
            "event_types": [et.value for et in self.event_types],
            "filters": {
                "agent_ids": list(self.filters.agent_ids),
                "task_ids": list(self.filters.task_ids),
                "workflow_ids": list(self.filters.workflow_ids),
                "user_id": self.filters.user_id,
                "metadata_filters": self.filters.metadata_filters,
            },
            "created_at": self.created_at,
        }


class SubscriptionManager:
    """
    Subscription manager for client subscriptions.

    Manages subscriptions for WebSocket and SSE clients.
    """

    def __init__(self) -> None:
        """Initialize subscription manager."""
        # Subscriptions by subscription ID
        self._subscriptions: dict[str, Subscription] = {}

        # Subscriptions by client ID
        self._client_subscriptions: dict[str, set[str]] = {}

        # Statistics
        self._stats = {
            "total_subscriptions": 0,
            "active_clients": 0,
        }

        logger.info("Subscription manager initialized")

    def add_subscription(
        self,
        subscription_id: str,
        client_id: str,
        topics: set[str] | None = None,
        event_types: set[EventType] | None = None,
        filters: SubscriptionFilter | None = None,
    ) -> Subscription:
        """
        Add client subscription.

        Args:
            subscription_id: Unique subscription ID
            client_id: Client identifier
            topics: Set of topics to subscribe to
            event_types: Set of event types to subscribe to
            filters: Optional subscription filters

        Returns:
            Created subscription
        """
        import time

        subscription = Subscription(
            subscription_id=subscription_id,
            client_id=client_id,
            topics=topics or set(),
            event_types=event_types or set(),
            filters=filters or SubscriptionFilter(),
            created_at=time.time(),
        )

        # Store subscription
        self._subscriptions[subscription_id] = subscription

        # Add to client subscriptions
        if client_id not in self._client_subscriptions:
            self._client_subscriptions[client_id] = set()
        self._client_subscriptions[client_id].add(subscription_id)

        # Update stats
        self._stats["total_subscriptions"] = len(self._subscriptions)
        self._stats["active_clients"] = len(self._client_subscriptions)

        logger.debug(
            "Subscription added",
            subscription_id=subscription_id,
            client_id=client_id,
            topics=list(topics or []),
            event_types=[et.value for et in (event_types or [])],
        )

        return subscription

    def remove_subscription(self, subscription_id: str) -> bool:
        """
        Remove subscription.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if removed, False if not found
        """
        if subscription_id not in self._subscriptions:
            return False

        # Get subscription
        subscription = self._subscriptions[subscription_id]

        # Remove from client subscriptions
        if subscription.client_id in self._client_subscriptions:
            self._client_subscriptions[subscription.client_id].discard(subscription_id)
            if not self._client_subscriptions[subscription.client_id]:
                del self._client_subscriptions[subscription.client_id]

        # Remove subscription
        del self._subscriptions[subscription_id]

        # Update stats
        self._stats["total_subscriptions"] = len(self._subscriptions)
        self._stats["active_clients"] = len(self._client_subscriptions)

        logger.debug(
            "Subscription removed",
            subscription_id=subscription_id,
            client_id=subscription.client_id,
        )

        return True

    def remove_client_subscriptions(self, client_id: str) -> int:
        """
        Remove all subscriptions for a client.

        Args:
            client_id: Client identifier

        Returns:
            Number of subscriptions removed
        """
        if client_id not in self._client_subscriptions:
            return 0

        # Get all subscription IDs for client
        subscription_ids = self._client_subscriptions[client_id].copy()

        # Remove each subscription
        count = 0
        for subscription_id in subscription_ids:
            if self.remove_subscription(subscription_id):
                count += 1

        logger.debug(
            "Client subscriptions removed",
            client_id=client_id,
            count=count,
        )

        return count

    def get_subscription(self, subscription_id: str) -> Subscription | None:
        """Get subscription by ID."""
        return self._subscriptions.get(subscription_id)

    def get_client_subscriptions(self, client_id: str) -> list[Subscription]:
        """Get all subscriptions for a client."""
        if client_id not in self._client_subscriptions:
            return []

        subscription_ids = self._client_subscriptions[client_id]
        return [
            self._subscriptions[sub_id]
            for sub_id in subscription_ids
            if sub_id in self._subscriptions
        ]

    def update_subscription_filters(
        self,
        subscription_id: str,
        filters: SubscriptionFilter,
    ) -> bool:
        """
        Update subscription filters.

        Args:
            subscription_id: Subscription ID
            filters: New filters

        Returns:
            True if updated, False if not found
        """
        if subscription_id not in self._subscriptions:
            return False

        self._subscriptions[subscription_id].filters = filters

        logger.debug(
            "Subscription filters updated",
            subscription_id=subscription_id,
        )

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get subscription statistics."""
        return self._stats.copy()


# Global subscription manager instance
subscription_manager = SubscriptionManager()
