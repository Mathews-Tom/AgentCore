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

    def get_active_signals(self, agent_id: str) -> dict[SignalType, SensitivitySignal]:
        """Get active (non-expired) signals for an agent.

        Filters out expired signals based on current time. Only returns signals
        that are still within their TTL.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary of active signals by type

        Example:
            >>> service = CoordinationService()
            >>> active = service.get_active_signals("agent-001")
            >>> for signal_type, signal in active.items():
            ...     print(f"{signal_type}: {signal.value}")
        """
        state = self.coordination_states.get(agent_id)

        if not state:
            return {}

        current_time = datetime.now(timezone.utc)

        # Filter expired signals
        active_signals = {
            signal_type: signal
            for signal_type, signal in state.signals.items()
            if not signal.is_expired(current_time)
        }

        return active_signals

    def compute_individual_scores(self, agent_id: str) -> None:
        """Compute individual scores for an agent from active signals.

        Updates the agent's coordination state with computed scores:
        - load_score: Inverted load (1.0 - load_value)
        - capacity_score: Direct capacity value
        - quality_score: Direct quality value
        - cost_score: Direct cost value (normalized, lower=better)
        - availability_score: Direct availability value

        Agents without signals receive default score of 0.5 for each metric.
        Expired signals are automatically filtered out.

        Args:
            agent_id: Agent identifier

        Example:
            >>> service = CoordinationService()
            >>> service.compute_individual_scores("agent-001")
            >>> state = service.get_coordination_state("agent-001")
            >>> print(f"Load score: {state.load_score}")
        """
        state = self.coordination_states.get(agent_id)

        if not state:
            # Agent not found - create default state
            self.coordination_states[agent_id] = AgentCoordinationState(agent_id=agent_id)
            logger.debug(
                "Created default coordination state for unknown agent",
                extra={"agent_id": agent_id},
            )
            return

        current_time = datetime.now(timezone.utc)

        # Compute load score (inverted: high load = low score)
        if SignalType.LOAD in state.signals:
            load_signal = state.signals[SignalType.LOAD]
            if not load_signal.is_expired(current_time):
                decay = load_signal.decay_factor(current_time)
                # Invert load: 1.0 - value (high load = low routing preference)
                state.load_score = 1.0 - (load_signal.value * load_signal.confidence * decay)
            else:
                state.load_score = 0.5  # Expired signal, use default
        else:
            state.load_score = 0.5  # No signal, use default

        # Compute capacity score (direct)
        if SignalType.CAPACITY in state.signals:
            capacity_signal = state.signals[SignalType.CAPACITY]
            if not capacity_signal.is_expired(current_time):
                decay = capacity_signal.decay_factor(current_time)
                state.capacity_score = capacity_signal.value * capacity_signal.confidence * decay
            else:
                state.capacity_score = 0.5
        else:
            state.capacity_score = 0.5

        # Compute quality score (direct)
        if SignalType.QUALITY in state.signals:
            quality_signal = state.signals[SignalType.QUALITY]
            if not quality_signal.is_expired(current_time):
                decay = quality_signal.decay_factor(current_time)
                state.quality_score = quality_signal.value * quality_signal.confidence * decay
            else:
                state.quality_score = 0.5
        else:
            state.quality_score = 0.5

        # Compute cost score (direct, normalized to 0-1 where higher=better)
        if SignalType.COST in state.signals:
            cost_signal = state.signals[SignalType.COST]
            if not cost_signal.is_expired(current_time):
                decay = cost_signal.decay_factor(current_time)
                state.cost_score = cost_signal.value * cost_signal.confidence * decay
            else:
                state.cost_score = 0.5
        else:
            state.cost_score = 0.5

        # Compute availability score (direct)
        if SignalType.AVAILABILITY in state.signals:
            availability_signal = state.signals[SignalType.AVAILABILITY]
            if not availability_signal.is_expired(current_time):
                decay = availability_signal.decay_factor(current_time)
                state.availability_score = (
                    availability_signal.value * availability_signal.confidence * decay
                )
            else:
                state.availability_score = 0.5
        else:
            state.availability_score = 0.5

        state.last_updated = current_time

        logger.debug(
            "Individual scores computed",
            extra={
                "agent_id": agent_id,
                "load_score": state.load_score,
                "capacity_score": state.capacity_score,
                "quality_score": state.quality_score,
                "cost_score": state.cost_score,
                "availability_score": state.availability_score,
            },
        )

    def compute_routing_score(self, agent_id: str) -> float:
        """Compute composite routing score for agent selection.

        Applies weighted averaging of individual scores using configured weights.
        Automatically computes individual scores if not already computed.

        Formula:
            routing_score = Σ(weight_i × score_i) for all score dimensions

        Args:
            agent_id: Agent identifier

        Returns:
            Composite routing score [0.0, 1.0], higher is better

        Example:
            >>> service = CoordinationService()
            >>> score = service.compute_routing_score("agent-001")
            >>> print(f"Routing score: {score:.3f}")
        """
        # Ensure individual scores are computed
        self.compute_individual_scores(agent_id)

        state = self.coordination_states.get(agent_id)

        if not state:
            # Should not happen after compute_individual_scores, but defensive
            logger.warning(
                "Agent state missing after score computation",
                extra={"agent_id": agent_id},
            )
            return 0.5

        # Apply weighted averaging
        routing_score = (
            settings.ROUTING_WEIGHT_LOAD * state.load_score
            + settings.ROUTING_WEIGHT_CAPACITY * state.capacity_score
            + settings.ROUTING_WEIGHT_QUALITY * state.quality_score
            + settings.ROUTING_WEIGHT_COST * state.cost_score
            + settings.ROUTING_WEIGHT_AVAILABILITY * state.availability_score
        )

        # Ensure score is in valid range [0.0, 1.0]
        routing_score = max(0.0, min(1.0, routing_score))

        # Update state
        state.routing_score = routing_score
        state.last_updated = datetime.now(timezone.utc)

        logger.debug(
            "Routing score computed",
            extra={
                "agent_id": agent_id,
                "routing_score": routing_score,
                "weights": {
                    "load": settings.ROUTING_WEIGHT_LOAD,
                    "capacity": settings.ROUTING_WEIGHT_CAPACITY,
                    "quality": settings.ROUTING_WEIGHT_QUALITY,
                    "cost": settings.ROUTING_WEIGHT_COST,
                    "availability": settings.ROUTING_WEIGHT_AVAILABILITY,
                },
            },
        )

        return routing_score

    def select_optimal_agent(
        self,
        candidate_agents: list[str],
        optimization_weights: dict[str, float] | None = None,
    ) -> str | None:
        """Select optimal agent from candidates using multi-objective optimization.

        Computes routing scores for all candidates and returns the agent with
        the highest composite score. Uses custom weights if provided, otherwise
        uses configured defaults.

        Args:
            candidate_agents: List of candidate agent IDs
            optimization_weights: Optional custom weights (load, capacity, quality, cost, availability)

        Returns:
            Selected agent ID with highest score, or None if no candidates

        Example:
            >>> service = CoordinationService()
            >>> candidates = ["agent-001", "agent-002", "agent-003"]
            >>> best_agent = service.select_optimal_agent(candidates)
            >>> print(f"Selected: {best_agent}")
        """
        if not candidate_agents:
            logger.warning("No candidate agents provided for selection")
            return None

        # TODO: Support custom optimization weights in future (COORD-007 extension)
        # For now, use default weights from configuration
        if optimization_weights:
            logger.info(
                "Custom optimization weights provided but not yet supported",
                extra={"weights": optimization_weights},
            )

        # Compute routing scores for all candidates
        agent_scores: list[tuple[str, float]] = []

        for agent_id in candidate_agents:
            score = self.compute_routing_score(agent_id)
            agent_scores.append((agent_id, score))

        # Sort by score (descending) - highest score first
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top agent
        selected_agent, selected_score = agent_scores[0]

        # Get state for logging
        selected_state = self.coordination_states.get(selected_agent)

        # Log selection rationale
        logger.info(
            "Optimal agent selected",
            extra={
                "selected_agent": selected_agent,
                "routing_score": selected_score,
                "total_candidates": len(candidate_agents),
                "score_breakdown": {
                    "load": selected_state.load_score if selected_state else 0.5,
                    "capacity": selected_state.capacity_score if selected_state else 0.5,
                    "quality": selected_state.quality_score if selected_state else 0.5,
                    "cost": selected_state.cost_score if selected_state else 0.5,
                    "availability": selected_state.availability_score if selected_state else 0.5,
                },
                "runner_up_scores": [
                    {"agent": agent_id, "score": score}
                    for agent_id, score in agent_scores[1:min(4, len(agent_scores))]
                ],
            },
        )

        # Update metrics
        self.metrics.total_selections += 1
        self.metrics.coordination_score_avg = sum(score for _, score in agent_scores) / len(
            agent_scores
        )

        return selected_agent

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
