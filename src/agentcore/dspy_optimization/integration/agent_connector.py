"""
Agent Runtime Connector

Connects DSPy optimization pipeline to AgentCore runtime for real-time
agent performance tracking and optimization.
"""

from __future__ import annotations

from typing import Any

import structlog

from agentcore.a2a_protocol.models.agent import AgentCard, AgentCapability
from agentcore.a2a_protocol.services.agent_manager import AgentManager
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)

logger = structlog.get_logger()


class AgentRuntimeConnector:
    """
    Connects DSPy optimization to AgentCore runtime.

    Provides integration between optimization pipeline and agent lifecycle,
    enabling real-time performance tracking and optimization target specification.
    """

    def __init__(self, agent_manager: AgentManager) -> None:
        """
        Initialize agent runtime connector.

        Args:
            agent_manager: AgentCore agent manager instance
        """
        self.agent_manager = agent_manager
        self._performance_cache: dict[str, PerformanceMetrics] = {}

        logger.info("Agent runtime connector initialized")

    async def get_agent_card(self, agent_id: str) -> AgentCard | None:
        """
        Get agent card from runtime.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent card or None if not found
        """
        agent_card = await self.agent_manager.get_agent(agent_id)

        if agent_card:
            logger.debug("Retrieved agent card", agent_id=agent_id)
        else:
            logger.warning("Agent not found", agent_id=agent_id)

        return agent_card

    async def get_agent_capabilities(self, agent_id: str) -> list[AgentCapability]:
        """
        Get agent capabilities from runtime.

        Args:
            agent_id: Agent identifier

        Returns:
            List of agent capabilities
        """
        agent_card = await self.get_agent_card(agent_id)
        if not agent_card:
            return []

        return agent_card.capabilities

    async def get_agent_performance_metrics(
        self, agent_id: str
    ) -> PerformanceMetrics | None:
        """
        Extract performance metrics from agent capabilities.

        Args:
            agent_id: Agent identifier

        Returns:
            Performance metrics or None if unavailable
        """
        # Check cache first
        if agent_id in self._performance_cache:
            logger.debug("Using cached performance metrics", agent_id=agent_id)
            return self._performance_cache[agent_id]

        agent_card = await self.get_agent_card(agent_id)
        if not agent_card or not agent_card.capabilities:
            return None

        # Aggregate metrics from capabilities
        total_cost = 0.0
        total_latency = 0.0
        total_quality = 0.0
        capability_count = 0

        for capability in agent_card.capabilities:
            if capability.cost_per_request is not None:
                total_cost += capability.cost_per_request
                capability_count += 1
            if capability.avg_latency_ms is not None:
                total_latency += capability.avg_latency_ms
            if capability.quality_score is not None:
                total_quality += capability.quality_score

        if capability_count == 0:
            return None

        # Calculate averages
        avg_cost = total_cost / capability_count
        avg_latency = int(total_latency / capability_count) if total_latency > 0 else 0
        avg_quality = total_quality / capability_count if total_quality > 0 else 0.8

        # Default success rate (would be tracked separately in production)
        success_rate = 0.9

        metrics = PerformanceMetrics(
            success_rate=success_rate,
            avg_cost_per_task=avg_cost,
            avg_latency_ms=avg_latency,
            quality_score=avg_quality,
        )

        # Cache metrics
        self._performance_cache[agent_id] = metrics

        logger.info(
            "Extracted agent performance metrics",
            agent_id=agent_id,
            success_rate=success_rate,
            avg_cost=avg_cost,
            avg_latency=avg_latency,
            quality_score=avg_quality,
        )

        return metrics

    async def update_agent_performance_metrics(
        self, agent_id: str, metrics: PerformanceMetrics
    ) -> bool:
        """
        Update agent capabilities with optimized performance metrics.

        Args:
            agent_id: Agent identifier
            metrics: Optimized performance metrics

        Returns:
            True if updated successfully, False otherwise
        """
        agent_card = await self.get_agent_card(agent_id)
        if not agent_card:
            logger.error("Cannot update metrics for non-existent agent", agent_id=agent_id)
            return False

        # Update capabilities with new metrics
        for capability in agent_card.capabilities:
            capability.cost_per_request = metrics.avg_cost_per_task
            capability.avg_latency_ms = float(metrics.avg_latency_ms)
            capability.quality_score = metrics.quality_score

        # Update cache
        self._performance_cache[agent_id] = metrics

        logger.info(
            "Updated agent performance metrics",
            agent_id=agent_id,
            new_cost=metrics.avg_cost_per_task,
            new_latency=metrics.avg_latency_ms,
            new_quality=metrics.quality_score,
        )

        return True

    async def create_optimization_target(
        self, agent_id: str
    ) -> OptimizationTarget | None:
        """
        Create optimization target from agent ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Optimization target or None if agent not found
        """
        agent_card = await self.get_agent_card(agent_id)
        if not agent_card:
            return None

        target = OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id=agent_id,
        )

        logger.info("Created optimization target for agent", agent_id=agent_id)

        return target

    async def is_agent_eligible_for_optimization(self, agent_id: str) -> bool:
        """
        Check if agent is eligible for optimization.

        Args:
            agent_id: Agent identifier

        Returns:
            True if eligible, False otherwise
        """
        agent_card = await self.get_agent_card(agent_id)
        if not agent_card:
            return False

        # Check if agent is active
        if not agent_card.is_active():
            logger.debug("Agent not active, ineligible for optimization", agent_id=agent_id)
            return False

        # Check if agent has capabilities with performance metrics
        has_metrics = any(
            cap.cost_per_request is not None
            or cap.avg_latency_ms is not None
            or cap.quality_score is not None
            for cap in agent_card.capabilities
        )

        if not has_metrics:
            logger.debug(
                "Agent lacks performance metrics, ineligible for optimization",
                agent_id=agent_id,
            )
            return False

        logger.debug("Agent eligible for optimization", agent_id=agent_id)
        return True

    async def get_all_optimizable_agents(self) -> list[str]:
        """
        Get list of all agents eligible for optimization.

        Returns:
            List of agent IDs
        """
        all_agents = await self.agent_manager.list_all_agents()
        optimizable = []

        for agent_summary in all_agents:
            agent_id = agent_summary["agent_id"]
            if await self.is_agent_eligible_for_optimization(agent_id):
                optimizable.append(agent_id)

        logger.info(
            "Found optimizable agents",
            total_agents=len(all_agents),
            optimizable_count=len(optimizable),
        )

        return optimizable

    def clear_cache(self, agent_id: str | None = None) -> None:
        """
        Clear performance metrics cache.

        Args:
            agent_id: Specific agent ID to clear, or None to clear all
        """
        if agent_id:
            self._performance_cache.pop(agent_id, None)
            logger.debug("Cleared cache for agent", agent_id=agent_id)
        else:
            self._performance_cache.clear()
            logger.debug("Cleared all performance cache")

    async def validate_optimization_request(
        self, request: OptimizationRequest
    ) -> tuple[bool, str | None]:
        """
        Validate optimization request against runtime state.

        Args:
            request: Optimization request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if request.target.type != OptimizationTargetType.AGENT:
            return True, None  # Non-agent targets not validated here

        agent_id = request.target.id

        # Check if agent exists
        agent_card = await self.get_agent_card(agent_id)
        if not agent_card:
            return False, f"Agent {agent_id} not found in runtime"

        # Check if agent is active
        if not agent_card.is_active():
            return False, f"Agent {agent_id} is not active"

        # Check if agent has capabilities
        if not agent_card.capabilities:
            return False, f"Agent {agent_id} has no capabilities defined"

        logger.info("Optimization request validated", agent_id=agent_id)
        return True, None
