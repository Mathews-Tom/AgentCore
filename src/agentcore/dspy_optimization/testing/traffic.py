"""
Traffic splitting and routing

Provides percentage-based traffic allocation and user/request routing
for A/B testing control and treatment groups.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.testing.experiment import (
    Experiment,
    ExperimentGroup,
)


class TrafficSplitConfig(BaseModel):
    """Configuration for traffic splitting"""

    hash_seed: str = Field(
        default="agentcore_ab_test",
        description="Seed for consistent hashing",
    )
    use_sticky_routing: bool = Field(
        default=True,
        description="Use consistent routing for same user/request",
    )


class RoutingDecision(BaseModel):
    """Decision from traffic routing"""

    experiment_id: str
    group: ExperimentGroup
    version: str
    confidence: float = Field(
        default=1.0,
        description="Confidence in routing decision",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrafficSplitter:
    """
    Traffic splitter for A/B testing

    Implements percentage-based traffic allocation with consistent
    hashing for sticky routing. Ensures users/requests are routed
    to the same group for duration of experiment.
    """

    def __init__(self, config: TrafficSplitConfig | None = None) -> None:
        """
        Initialize traffic splitter

        Args:
            config: Traffic split configuration
        """
        self.config = config or TrafficSplitConfig()

    async def route_request(
        self,
        experiment: Experiment,
        request_id: str,
        user_id: str | None = None,
    ) -> RoutingDecision:
        """
        Route request to experiment group

        Args:
            experiment: Experiment configuration
            request_id: Request identifier
            user_id: Optional user identifier for sticky routing

        Returns:
            Routing decision

        Raises:
            ValueError: If experiment not active
        """
        if not experiment.is_active():
            raise ValueError(f"Experiment not active: {experiment.id}")

        # Determine routing key
        routing_key = user_id if user_id and self.config.use_sticky_routing else request_id

        # Calculate hash-based allocation
        group = self._calculate_group(
            experiment_id=experiment.id,
            routing_key=routing_key,
            traffic_percentage=experiment.config.traffic_percentage,
        )

        # Determine version
        version = (
            experiment.treatment_version
            if group == ExperimentGroup.TREATMENT
            else experiment.control_version
        )

        return RoutingDecision(
            experiment_id=experiment.id,
            group=group,
            version=version,
            metadata={
                "request_id": request_id,
                "user_id": user_id,
                "routing_key": routing_key,
            },
        )

    async def route_batch(
        self,
        experiment: Experiment,
        request_ids: list[str],
        user_ids: list[str] | None = None,
    ) -> list[RoutingDecision]:
        """
        Route batch of requests

        Args:
            experiment: Experiment configuration
            request_ids: List of request identifiers
            user_ids: Optional list of user identifiers

        Returns:
            List of routing decisions
        """
        if user_ids and len(user_ids) != len(request_ids):
            raise ValueError("user_ids must match length of request_ids")

        decisions = []
        for i, request_id in enumerate(request_ids):
            user_id = user_ids[i] if user_ids else None
            decision = await self.route_request(experiment, request_id, user_id)
            decisions.append(decision)

        return decisions

    async def get_traffic_distribution(
        self,
        experiment: Experiment,
        sample_size: int = 10000,
    ) -> dict[str, float]:
        """
        Calculate actual traffic distribution

        Simulates routing to verify traffic split accuracy.

        Args:
            experiment: Experiment configuration
            sample_size: Number of samples to simulate

        Returns:
            Dictionary with actual distribution percentages
        """
        control_count = 0
        treatment_count = 0

        for i in range(sample_size):
            request_id = f"sample_{i}"
            group = self._calculate_group(
                experiment_id=experiment.id,
                routing_key=request_id,
                traffic_percentage=experiment.config.traffic_percentage,
            )

            if group == ExperimentGroup.TREATMENT:
                treatment_count += 1
            else:
                control_count += 1

        return {
            "control": control_count / sample_size,
            "treatment": treatment_count / sample_size,
            "control_count": control_count,
            "treatment_count": treatment_count,
        }

    def _calculate_group(
        self,
        experiment_id: str,
        routing_key: str,
        traffic_percentage: float,
    ) -> ExperimentGroup:
        """
        Calculate group assignment using consistent hashing

        Args:
            experiment_id: Experiment ID
            routing_key: Key for consistent routing
            traffic_percentage: Percentage to treatment group

        Returns:
            Assigned experiment group
        """
        # Create consistent hash
        hash_input = f"{self.config.hash_seed}:{experiment_id}:{routing_key}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()

        # Convert to float in [0, 1)
        hash_float = int(hash_value[:16], 16) / (2**64)

        # Assign to treatment if hash < traffic_percentage
        if hash_float < traffic_percentage:
            return ExperimentGroup.TREATMENT
        else:
            return ExperimentGroup.CONTROL

    async def validate_split_accuracy(
        self,
        experiment: Experiment,
        sample_size: int = 10000,
        tolerance: float = 0.02,
    ) -> bool:
        """
        Validate traffic split accuracy

        Args:
            experiment: Experiment configuration
            sample_size: Number of samples to test
            tolerance: Acceptable deviation from target

        Returns:
            True if split is within tolerance
        """
        distribution = await self.get_traffic_distribution(experiment, sample_size)

        target_treatment = experiment.config.traffic_percentage
        actual_treatment = distribution["treatment"]

        deviation = abs(actual_treatment - target_treatment)

        return deviation <= tolerance
