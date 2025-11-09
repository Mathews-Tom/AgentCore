"""
Capability Evaluator - ACE-025

Evaluates agent capability fitness for tasks and identifies gaps.
Implements COMPASS ACE-4 specification for dynamic capability evaluation.

Performance target: <100ms evaluation latency
"""

import structlog
from datetime import UTC, datetime
from uuid import UUID

from ..models.ace_models import (
    CapabilityFitness,
    CapabilityGap,
    CapabilityRecommendation,
    CapabilityType,
    FitnessMetrics,
    TaskRequirement,
)


logger = structlog.get_logger(__name__)


class CapabilityEvaluator:
    """
    Evaluates agent capability fitness for tasks.

    Provides:
    - Task-capability matching
    - Fitness score computation (0-1 scale)
    - Capability gap identification
    - Performance-based scoring

    Target latency: <100ms for evaluation
    """

    def __init__(self):
        """Initialize capability evaluator."""
        self.logger = logger.bind(component="capability_evaluator")
        self._fitness_cache: dict[str, CapabilityFitness] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

    async def evaluate_fitness(
        self,
        agent_id: str,
        capability_id: str,
        capability_name: str,
        task_requirements: list[TaskRequirement] | None = None,
        task_type: str | None = None,
        performance_history: dict[str, any] | None = None,
    ) -> CapabilityFitness:
        """
        Evaluate fitness score for a specific capability.

        Args:
            agent_id: Agent identifier
            capability_id: Capability identifier
            capability_name: Capability display name
            task_requirements: Optional task requirements for matching
            task_type: Optional task type for context
            performance_history: Optional performance metrics

        Returns:
            CapabilityFitness with computed scores

        Performance: <100ms (p95)
        """
        start_time = datetime.now(UTC)

        self.logger.info(
            "evaluating_capability_fitness",
            agent_id=agent_id,
            capability_id=capability_id,
            task_type=task_type,
        )

        # Check cache
        cache_key = f"{agent_id}:{capability_id}:{task_type or 'general'}"
        if cache_key in self._fitness_cache:
            cached = self._fitness_cache[cache_key]
            cache_age = (datetime.now(UTC) - cached.evaluated_at).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                self.logger.debug("cache_hit", cache_key=cache_key, age_seconds=cache_age)
                return cached

        # Compute fitness metrics
        metrics = await self._compute_fitness_metrics(
            agent_id, capability_id, performance_history
        )

        # Compute coverage score
        coverage_score = await self._compute_coverage_score(
            capability_id, capability_name, task_requirements
        )

        # Compute performance score
        performance_score = await self._compute_performance_score(metrics)

        # Compute overall fitness score (weighted combination)
        fitness_score = self._compute_overall_fitness(
            coverage_score, performance_score, metrics
        )

        # Create fitness result
        fitness = CapabilityFitness(
            capability_id=capability_id,
            capability_name=capability_name,
            agent_id=agent_id,
            task_type=task_type,
            fitness_score=fitness_score,
            coverage_score=coverage_score,
            performance_score=performance_score,
            metrics=metrics,
            sample_size=metrics.usage_frequency,
            evaluated_at=datetime.now(UTC),
            metadata={
                "evaluation_method": "composite_scoring",
                "cache_key": cache_key,
            },
        )

        # Update cache
        self._fitness_cache[cache_key] = fitness

        # Log performance
        duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        self.logger.info(
            "fitness_evaluated",
            agent_id=agent_id,
            capability_id=capability_id,
            fitness_score=fitness_score,
            duration_ms=duration_ms,
        )

        return fitness

    async def evaluate_all_capabilities(
        self,
        agent_id: str,
        current_capabilities: list[dict[str, str]],
        task_requirements: list[TaskRequirement],
        task_type: str | None = None,
    ) -> dict[str, CapabilityFitness]:
        """
        Evaluate fitness for all agent capabilities.

        Args:
            agent_id: Agent identifier
            current_capabilities: List of current capabilities with id and name
            task_requirements: Task requirements to evaluate against
            task_type: Optional task type

        Returns:
            Dictionary mapping capability_id to CapabilityFitness
        """
        self.logger.info(
            "evaluating_all_capabilities",
            agent_id=agent_id,
            capability_count=len(current_capabilities),
            requirement_count=len(task_requirements),
        )

        fitness_scores = {}

        for capability in current_capabilities:
            cap_id = capability.get("id", capability.get("capability_id", "unknown"))
            cap_name = capability.get("name", capability.get("capability_name", cap_id))

            fitness = await self.evaluate_fitness(
                agent_id=agent_id,
                capability_id=cap_id,
                capability_name=cap_name,
                task_requirements=task_requirements,
                task_type=task_type,
            )

            fitness_scores[cap_id] = fitness

        self.logger.info(
            "all_capabilities_evaluated",
            agent_id=agent_id,
            total_evaluated=len(fitness_scores),
            avg_fitness=sum(f.fitness_score for f in fitness_scores.values())
            / len(fitness_scores)
            if fitness_scores
            else 0.0,
        )

        return fitness_scores

    async def identify_capability_gaps(
        self,
        agent_id: str,
        current_capabilities: list[str],
        task_requirements: list[TaskRequirement],
        fitness_scores: dict[str, CapabilityFitness] | None = None,
    ) -> list[CapabilityGap]:
        """
        Identify capability gaps for task requirements.

        Args:
            agent_id: Agent identifier
            current_capabilities: List of current capability IDs
            task_requirements: Task requirements
            fitness_scores: Optional pre-computed fitness scores

        Returns:
            List of identified capability gaps
        """
        self.logger.info(
            "identifying_capability_gaps",
            agent_id=agent_id,
            current_count=len(current_capabilities),
            requirement_count=len(task_requirements),
        )

        gaps = []

        for requirement in task_requirements:
            req_cap = requirement.capability_name
            current_fitness = None

            # Check if capability exists and get fitness
            if fitness_scores and req_cap in fitness_scores:
                current_fitness = fitness_scores[req_cap].fitness_score
            elif req_cap in current_capabilities:
                # Capability exists but no fitness score
                current_fitness = 0.5  # Assume neutral fitness

            # Determine if there's a gap
            required_threshold = 0.5 if requirement.required else 0.3
            has_capability = req_cap in current_capabilities
            meets_threshold = (
                current_fitness is not None and current_fitness >= required_threshold
            )

            if not has_capability or not meets_threshold:
                # Calculate gap severity
                if requirement.required and not has_capability:
                    severity = "critical"
                elif requirement.required and current_fitness is not None and current_fitness < 0.3:
                    severity = "high"
                elif requirement.weight > 0.7:
                    severity = "high"
                elif requirement.weight > 0.4:
                    severity = "medium"
                else:
                    severity = "low"

                gap = CapabilityGap(
                    required_capability=req_cap,
                    capability_type=requirement.capability_type,
                    current_fitness=current_fitness,
                    required_fitness=required_threshold,
                    impact=requirement.weight,
                    gap_severity=severity,
                    mitigation_suggestion=self._generate_gap_mitigation(
                        requirement, has_capability, current_fitness
                    ),
                )

                gaps.append(gap)

        self.logger.info(
            "gaps_identified",
            agent_id=agent_id,
            gap_count=len(gaps),
            critical_gaps=sum(1 for g in gaps if g.gap_severity == "critical"),
        )

        return gaps

    async def _compute_fitness_metrics(
        self,
        agent_id: str,
        capability_id: str,
        performance_history: dict[str, any] | None,
    ) -> FitnessMetrics:
        """
        Compute detailed fitness metrics from performance history.

        Args:
            agent_id: Agent identifier
            capability_id: Capability identifier
            performance_history: Performance data

        Returns:
            FitnessMetrics with computed values
        """
        # Default metrics if no history
        if not performance_history:
            return FitnessMetrics(
                success_rate=0.5,  # Neutral success rate
                error_correlation=0.0,
                usage_frequency=0,
                avg_execution_time_ms=1000.0,
                resource_efficiency=0.5,
            )

        # Extract metrics from history
        total_executions = performance_history.get("total_executions", 0)
        successful_executions = performance_history.get("successful_executions", 0)
        total_errors = performance_history.get("total_errors", 0)
        execution_times = performance_history.get("execution_times", [])
        resource_usage = performance_history.get("resource_usage", {})

        # Compute success rate
        success_rate = (
            successful_executions / total_executions if total_executions > 0 else 0.5
        )

        # Compute error correlation (errors per execution)
        error_correlation = (
            min(1.0, total_errors / max(1, total_executions))
            if total_executions > 0
            else 0.0
        )

        # Compute average execution time
        avg_time = (
            sum(execution_times) / len(execution_times)
            if execution_times
            else 1000.0
        )

        # Compute resource efficiency (inverse of resource usage)
        cpu_usage = resource_usage.get("cpu_percent", 50.0)
        memory_usage = resource_usage.get("memory_percent", 50.0)
        resource_efficiency = max(
            0.0, 1.0 - ((cpu_usage + memory_usage) / 200.0)
        )

        return FitnessMetrics(
            success_rate=success_rate,
            error_correlation=error_correlation,
            usage_frequency=total_executions,
            avg_execution_time_ms=avg_time,
            resource_efficiency=resource_efficiency,
        )

    async def _compute_coverage_score(
        self,
        capability_id: str,
        capability_name: str,
        task_requirements: list[TaskRequirement] | None,
    ) -> float:
        """
        Compute how well capability covers task requirements.

        Args:
            capability_id: Capability identifier
            capability_name: Capability name
            task_requirements: Task requirements

        Returns:
            Coverage score (0-1)
        """
        if not task_requirements:
            return 1.0  # Full coverage if no requirements

        # Check if capability matches any requirement
        matches = []
        for req in task_requirements:
            # Simple name matching (can be enhanced with semantic matching)
            name_match = (
                req.capability_name.lower() in capability_name.lower()
                or capability_name.lower() in req.capability_name.lower()
            )
            id_match = req.capability_name == capability_id

            if name_match or id_match:
                matches.append(req.weight)

        # Compute weighted coverage
        if not matches:
            return 0.0

        total_weight = sum(req.weight for req in task_requirements)
        matched_weight = sum(matches)

        return min(1.0, matched_weight / total_weight if total_weight > 0 else 0.0)

    async def _compute_performance_score(self, metrics: FitnessMetrics) -> float:
        """
        Compute performance score from fitness metrics.

        Args:
            metrics: Fitness metrics

        Returns:
            Performance score (0-1)
        """
        # Weight factors
        success_weight = 0.5
        error_weight = 0.3
        efficiency_weight = 0.2

        # Compute weighted score
        performance_score = (
            metrics.success_rate * success_weight
            + (1.0 - metrics.error_correlation) * error_weight
            + metrics.resource_efficiency * efficiency_weight
        )

        return min(1.0, max(0.0, performance_score))

    def _compute_overall_fitness(
        self,
        coverage_score: float,
        performance_score: float,
        metrics: FitnessMetrics,
    ) -> float:
        """
        Compute overall fitness score from components.

        Args:
            coverage_score: Task coverage score
            performance_score: Performance score
            metrics: Detailed fitness metrics

        Returns:
            Overall fitness score (0-1)
        """
        # Base formula: coverage * (1 - error_correlation) + performance
        # This matches spec: fitness = success_rate * (1 - error_correlation)
        base_fitness = metrics.success_rate * (1.0 - metrics.error_correlation)

        # Weight coverage and performance
        coverage_weight = 0.4
        performance_weight = 0.3
        base_weight = 0.3

        overall = (
            coverage_score * coverage_weight
            + performance_score * performance_weight
            + base_fitness * base_weight
        )

        return min(1.0, max(0.0, overall))

    def _generate_gap_mitigation(
        self,
        requirement: TaskRequirement,
        has_capability: bool,
        current_fitness: float | None,
    ) -> str:
        """
        Generate mitigation suggestion for capability gap.

        Args:
            requirement: Task requirement with gap
            has_capability: Whether agent has the capability
            current_fitness: Current fitness score if exists

        Returns:
            Mitigation suggestion text
        """
        if not has_capability:
            return f"Add '{requirement.capability_name}' capability to agent"

        if current_fitness is not None and current_fitness < 0.3:
            return f"Replace or upgrade '{requirement.capability_name}' (current fitness: {current_fitness:.2f})"

        return f"Improve '{requirement.capability_name}' through training or configuration"

    def clear_cache(self):
        """Clear fitness score cache."""
        self._fitness_cache.clear()
        self.logger.info("fitness_cache_cleared")

    def get_cache_stats(self) -> dict[str, any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._fitness_cache),
            "cache_ttl_seconds": self._cache_ttl_seconds,
        }
