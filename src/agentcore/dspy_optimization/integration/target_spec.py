"""
Agent Optimization Target Specification

Defines optimization targets and objectives for agent runtime integration.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    MetricType,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationScope,
    OptimizationTarget,
    OptimizationTargetType,
)

logger = structlog.get_logger()


class AgentOptimizationProfile(str):
    """Pre-defined optimization profiles for common agent scenarios."""

    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"


class AgentOptimizationTarget:
    """
    Helper class for creating agent-specific optimization targets.

    Simplifies the process of specifying optimization objectives and constraints
    for agent runtime integration.
    """

    @staticmethod
    def create_target(
        agent_id: str, scope: OptimizationScope = OptimizationScope.INDIVIDUAL
    ) -> OptimizationTarget:
        """
        Create optimization target for an agent.

        Args:
            agent_id: Agent identifier
            scope: Optimization scope

        Returns:
            Optimization target
        """
        target = OptimizationTarget(
            type=OptimizationTargetType.AGENT, id=agent_id, scope=scope
        )

        logger.debug("Created optimization target", agent_id=agent_id, scope=scope)

        return target

    @staticmethod
    def create_cost_optimized_objectives() -> list[OptimizationObjective]:
        """
        Create objectives for cost optimization.

        Returns:
            List of optimization objectives prioritizing cost efficiency
        """
        return [
            OptimizationObjective(
                metric=MetricType.COST_EFFICIENCY, target_value=0.9, weight=1.0
            ),
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.95, weight=0.6
            ),
            OptimizationObjective(
                metric=MetricType.QUALITY_SCORE, target_value=0.8, weight=0.4
            ),
        ]

    @staticmethod
    def create_latency_optimized_objectives() -> list[OptimizationObjective]:
        """
        Create objectives for latency optimization.

        Returns:
            List of optimization objectives prioritizing low latency
        """
        return [
            OptimizationObjective(
                metric=MetricType.LATENCY, target_value=0.9, weight=1.0
            ),
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.95, weight=0.7
            ),
            OptimizationObjective(
                metric=MetricType.QUALITY_SCORE, target_value=0.85, weight=0.5
            ),
        ]

    @staticmethod
    def create_quality_optimized_objectives() -> list[OptimizationObjective]:
        """
        Create objectives for quality optimization.

        Returns:
            List of optimization objectives prioritizing quality
        """
        return [
            OptimizationObjective(
                metric=MetricType.QUALITY_SCORE, target_value=0.95, weight=1.0
            ),
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.98, weight=0.8
            ),
            OptimizationObjective(
                metric=MetricType.COST_EFFICIENCY, target_value=0.7, weight=0.3
            ),
        ]

    @staticmethod
    def create_balanced_objectives() -> list[OptimizationObjective]:
        """
        Create balanced objectives optimizing all metrics equally.

        Returns:
            List of balanced optimization objectives
        """
        return [
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE, target_value=0.95, weight=0.8
            ),
            OptimizationObjective(
                metric=MetricType.COST_EFFICIENCY, target_value=0.8, weight=0.7
            ),
            OptimizationObjective(
                metric=MetricType.LATENCY, target_value=0.85, weight=0.7
            ),
            OptimizationObjective(
                metric=MetricType.QUALITY_SCORE, target_value=0.85, weight=0.8
            ),
        ]

    @staticmethod
    def create_objectives_from_profile(
        profile: str,
    ) -> list[OptimizationObjective]:
        """
        Create optimization objectives from a profile name.

        Args:
            profile: Optimization profile name

        Returns:
            List of optimization objectives

        Raises:
            ValueError: If profile is unknown
        """
        profile_map = {
            AgentOptimizationProfile.COST_OPTIMIZED: AgentOptimizationTarget.create_cost_optimized_objectives,
            AgentOptimizationProfile.LATENCY_OPTIMIZED: AgentOptimizationTarget.create_latency_optimized_objectives,
            AgentOptimizationProfile.QUALITY_OPTIMIZED: AgentOptimizationTarget.create_quality_optimized_objectives,
            AgentOptimizationProfile.BALANCED: AgentOptimizationTarget.create_balanced_objectives,
        }

        creator = profile_map.get(profile)
        if not creator:
            raise ValueError(
                f"Unknown optimization profile: {profile}. "
                f"Available: {list(profile_map.keys())}"
            )

        objectives = creator()
        logger.info("Created objectives from profile", profile=profile)

        return objectives

    @staticmethod
    def create_custom_objectives(
        objectives_config: list[dict[str, Any]],
    ) -> list[OptimizationObjective]:
        """
        Create custom optimization objectives from configuration.

        Args:
            objectives_config: List of objective configurations with
                              'metric', 'target_value', and 'weight' keys

        Returns:
            List of optimization objectives

        Raises:
            ValueError: If configuration is invalid
        """
        objectives = []

        for config in objectives_config:
            if "metric" not in config:
                raise ValueError("Objective configuration must include 'metric'")

            metric_name = config["metric"]
            try:
                metric = MetricType(metric_name)
            except ValueError:
                raise ValueError(f"Invalid metric type: {metric_name}")

            objective = OptimizationObjective(
                metric=metric,
                target_value=config.get("target_value", 0.9),
                weight=config.get("weight", 1.0),
            )
            objectives.append(objective)

        logger.info("Created custom objectives", count=len(objectives))

        return objectives

    @staticmethod
    def create_optimization_request(
        agent_id: str,
        profile: str = AgentOptimizationProfile.BALANCED,
        algorithms: list[str] | None = None,
        constraints: OptimizationConstraints | None = None,
        scope: OptimizationScope = OptimizationScope.INDIVIDUAL,
    ) -> OptimizationRequest:
        """
        Create complete optimization request for an agent.

        Args:
            agent_id: Agent identifier
            profile: Optimization profile (default: balanced)
            algorithms: List of algorithms to use (default: miprov2, gepa)
            constraints: Custom constraints (default: default constraints)
            scope: Optimization scope (default: individual)

        Returns:
            Complete optimization request
        """
        target = AgentOptimizationTarget.create_target(agent_id, scope)
        objectives = AgentOptimizationTarget.create_objectives_from_profile(profile)

        if algorithms is None:
            algorithms = ["miprov2", "gepa"]

        if constraints is None:
            constraints = OptimizationConstraints()

        request = OptimizationRequest(
            target=target,
            objectives=objectives,
            algorithms=algorithms,
            constraints=constraints,
        )

        logger.info(
            "Created optimization request",
            agent_id=agent_id,
            profile=profile,
            algorithms=algorithms,
            objectives_count=len(objectives),
        )

        return request

    @staticmethod
    def create_multi_agent_request(
        agent_ids: list[str],
        profile: str = AgentOptimizationProfile.BALANCED,
        algorithms: list[str] | None = None,
        constraints: OptimizationConstraints | None = None,
    ) -> list[OptimizationRequest]:
        """
        Create optimization requests for multiple agents.

        Args:
            agent_ids: List of agent identifiers
            profile: Optimization profile
            algorithms: List of algorithms to use
            constraints: Custom constraints

        Returns:
            List of optimization requests
        """
        requests = []

        for agent_id in agent_ids:
            request = AgentOptimizationTarget.create_optimization_request(
                agent_id=agent_id,
                profile=profile,
                algorithms=algorithms,
                constraints=constraints,
                scope=OptimizationScope.INDIVIDUAL,
            )
            requests.append(request)

        logger.info(
            "Created multi-agent optimization requests",
            agent_count=len(agent_ids),
            profile=profile,
        )

        return requests


class OptimizationTargetBuilder(BaseModel):
    """
    Fluent builder for creating optimization targets and requests.

    Provides a chainable interface for constructing complex optimization
    configurations.
    """

    agent_id: str = Field(..., description="Agent identifier")
    profile: str = Field(
        default=AgentOptimizationProfile.BALANCED, description="Optimization profile"
    )
    algorithms: list[str] = Field(
        default_factory=lambda: ["miprov2", "gepa"], description="Optimization algorithms"
    )
    scope: OptimizationScope = Field(
        default=OptimizationScope.INDIVIDUAL, description="Optimization scope"
    )
    custom_objectives: list[OptimizationObjective] = Field(
        default_factory=list, description="Custom objectives"
    )
    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints, description="Optimization constraints"
    )

    def with_profile(self, profile: str) -> OptimizationTargetBuilder:
        """Set optimization profile."""
        self.profile = profile
        return self

    def with_algorithms(self, algorithms: list[str]) -> OptimizationTargetBuilder:
        """Set optimization algorithms."""
        self.algorithms = algorithms
        return self

    def with_scope(self, scope: OptimizationScope) -> OptimizationTargetBuilder:
        """Set optimization scope."""
        self.scope = scope
        return self

    def add_objective(
        self, metric: MetricType, target_value: float, weight: float = 1.0
    ) -> OptimizationTargetBuilder:
        """Add custom objective."""
        objective = OptimizationObjective(
            metric=metric, target_value=target_value, weight=weight
        )
        self.custom_objectives.append(objective)
        return self

    def with_constraints(
        self, constraints: OptimizationConstraints
    ) -> OptimizationTargetBuilder:
        """Set optimization constraints."""
        self.constraints = constraints
        return self

    def build(self) -> OptimizationRequest:
        """
        Build optimization request.

        Returns:
            Complete optimization request
        """
        target = AgentOptimizationTarget.create_target(self.agent_id, self.scope)

        # Use custom objectives if provided, otherwise use profile
        if self.custom_objectives:
            objectives = self.custom_objectives
        else:
            objectives = AgentOptimizationTarget.create_objectives_from_profile(
                self.profile
            )

        request = OptimizationRequest(
            target=target,
            objectives=objectives,
            algorithms=self.algorithms,
            constraints=self.constraints,
        )

        logger.info(
            "Built optimization request",
            agent_id=self.agent_id,
            profile=self.profile,
            objectives_count=len(objectives),
        )

        return request
