"""Coordination Service - Ripple Effect Protocol Implementation

Implements intelligent agent coordination based on sensitivity signals (load, capacity, quality).
Provides multi-objective optimization for agent selection to improve task distribution and
load balancing across the system.

Key Features:
- Signal registration and validation
- Agent coordination state management
- Multi-objective agent selection
- Signal expiry and cleanup
- Overload prediction

Based on: Ripple Effect Protocol (REP) research paper
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.coordination import (
    AgentCoordinationState,
    CoordinationMetrics,
    SensitivitySignal,
    SignalType,
)

logger = logging.getLogger(__name__)


class CoordinationService:
    """Coordination Service for intelligent agent selection.

    Manages sensitivity signals from agents and provides multi-objective optimization
    for routing decisions. Maintains per-agent coordination state with signal history
    for trend analysis and overload prediction.

    Attributes:
        coordination_states: Mapping of agent_id to coordination state
        signal_history: Signal history per agent for trend analysis
        metrics: Prometheus metrics for observability
    """

    def __init__(self) -> None:
        """Initialize coordination service."""
        self.coordination_states: dict[str, AgentCoordinationState] = {}
        self.signal_history: dict[str, list[SensitivitySignal]] = defaultdict(list)
        self.metrics = CoordinationMetrics()

        logger.info(
            "CoordinationService initialized",
            extra={
                "enable_rep": settings.COORDINATION_ENABLE_REP,
                "signal_ttl": settings.COORDINATION_SIGNAL_TTL,
                "max_history_size": settings.COORDINATION_MAX_HISTORY_SIZE,
            },
        )

    def register_signal(self, signal: SensitivitySignal) -> None:
        """Register a sensitivity signal from an agent.

        Validates signal format, normalizes values, updates agent coordination state,
        and stores signal in history for trend analysis.

        Args:
            signal: Sensitivity signal to register

        Raises:
            ValueError: If signal validation fails

        Example:
            >>> service = CoordinationService()
            >>> signal = SensitivitySignal(
            ...     agent_id="agent-001",
            ...     signal_type=SignalType.LOAD,
            ...     value=0.75,
            ...     ttl_seconds=60
            ... )
            >>> service.register_signal(signal)
        """
        # Validation is handled by Pydantic model
        # Additional business logic validation here

        # Ensure signal is not expired on registration
        if signal.is_expired():
            raise ValueError(
                f"Cannot register expired signal: {signal.signal_id} "
                f"(timestamp: {signal.timestamp}, ttl: {signal.ttl_seconds}s)"
            )

        # Normalize signal value (ensure 0.0-1.0 range - already validated by Pydantic)
        # This is defensive programming
        if not (0.0 <= signal.value <= 1.0):
            raise ValueError(
                f"Signal value must be in range [0.0, 1.0], got {signal.value}"
            )

        # Get or create coordination state for agent
        if signal.agent_id not in self.coordination_states:
            self.coordination_states[signal.agent_id] = AgentCoordinationState(
                agent_id=signal.agent_id
            )
            logger.info(
                "Created coordination state for new agent",
                extra={"agent_id": signal.agent_id},
            )

        state = self.coordination_states[signal.agent_id]

        # Update state with new signal
        state.signals[signal.signal_type] = signal
        state.last_updated = datetime.now(timezone.utc)

        # Store signal in history
        self._store_signal_history(signal)

        # Update metrics
        self.metrics.total_signals += 1
        self.metrics.signals_by_type[signal.signal_type] = (
            self.metrics.signals_by_type.get(signal.signal_type, 0) + 1
        )
        self.metrics.agents_tracked = len(self.coordination_states)

        logger.debug(
            "Signal registered",
            extra={
                "signal_id": str(signal.signal_id),
                "agent_id": signal.agent_id,
                "signal_type": signal.signal_type.value,
                "value": signal.value,
                "confidence": signal.confidence,
            },
        )

    def _store_signal_history(self, signal: SensitivitySignal) -> None:
        """Store signal in history for trend analysis.

        Maintains fixed-size history per agent (FIFO when limit reached).

        Args:
            signal: Signal to store in history
        """
        history = self.signal_history[signal.agent_id]

        # Add signal to history
        history.append(signal)

        # Enforce max history size (FIFO eviction)
        max_size = settings.COORDINATION_MAX_HISTORY_SIZE
        if len(history) > max_size:
            # Remove oldest signals
            self.signal_history[signal.agent_id] = history[-max_size:]

    def get_coordination_state(self, agent_id: str) -> AgentCoordinationState | None:
        """Get coordination state for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Coordination state if exists, None otherwise

        Example:
            >>> service = CoordinationService()
            >>> state = service.get_coordination_state("agent-001")
            >>> if state:
            ...     print(f"Routing score: {state.routing_score}")
        """
        return self.coordination_states.get(agent_id)

    def get_signal_history(
        self, agent_id: str, signal_type: SignalType | None = None, limit: int | None = None
    ) -> list[SensitivitySignal]:
        """Get signal history for an agent.

        Args:
            agent_id: Agent identifier
            signal_type: Optional filter by signal type
            limit: Optional limit on number of signals returned (most recent first)

        Returns:
            List of signals (most recent first)

        Example:
            >>> service = CoordinationService()
            >>> recent_load = service.get_signal_history(
            ...     "agent-001",
            ...     signal_type=SignalType.LOAD,
            ...     limit=10
            ... )
        """
        history = self.signal_history.get(agent_id, [])

        # Filter by signal type if specified
        if signal_type:
            history = [s for s in history if s.signal_type == signal_type]

        # Sort by timestamp (most recent first)
        history = sorted(history, key=lambda s: s.timestamp, reverse=True)

        # Apply limit
        if limit:
            history = history[:limit]

        return history

    def remove_expired_signals(self) -> int:
        """Remove expired signals from all coordination states.

        Returns:
            Number of signals removed

        Example:
            >>> service = CoordinationService()
            >>> removed_count = service.remove_expired_signals()
            >>> print(f"Removed {removed_count} expired signals")
        """
        removed_count = 0
        current_time = datetime.now(timezone.utc)

        for agent_id, state in list(self.coordination_states.items()):
            # Remove expired signals from state
            expired_types = [
                signal_type
                for signal_type, signal in state.signals.items()
                if signal.is_expired(current_time)
            ]

            for signal_type in expired_types:
                del state.signals[signal_type]
                removed_count += 1

            # Remove agent state if no active signals remain
            if not state.signals:
                del self.coordination_states[agent_id]
                logger.debug(
                    "Removed coordination state (no active signals)",
                    extra={"agent_id": agent_id},
                )

        if removed_count > 0:
            self.metrics.expired_signals_cleaned += removed_count
            self.metrics.agents_tracked = len(self.coordination_states)

            logger.info(
                "Expired signals removed",
                extra={
                    "removed_count": removed_count,
                    "active_agents": len(self.coordination_states),
                },
            )

        return removed_count

    def get_metrics(self) -> CoordinationMetrics:
        """Get current coordination metrics.

        Returns:
            Coordination metrics for observability

        Example:
            >>> service = CoordinationService()
            >>> metrics = service.get_metrics()
            >>> print(f"Total signals: {metrics.total_signals}")
        """
        return self.metrics

    def clear_state(self) -> None:
        """Clear all coordination state (for testing).

        WARNING: This removes all signals and coordination state.
        Should only be used in tests or admin operations.
        """
        self.coordination_states.clear()
        self.signal_history.clear()
        self.metrics = CoordinationMetrics()

        logger.warning("Coordination state cleared (all signals removed)")


# Global coordination service instance
coordination_service = CoordinationService()
