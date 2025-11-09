"""
MEM Integration Layer for COMPASS Meta-Thinker (COMPASS ACE-3 - ACE-021)

Provides interface for ACE to query MEM for strategic context during
intervention decision making. Currently uses mock implementation until
MEM Phase 5 is available (Week 8).
"""

from __future__ import annotations

import random
import time
from typing import Any
from uuid import UUID, uuid4

import structlog

from agentcore.ace.models.ace_models import (
    MemoryQuery,
    MemoryQueryResult,
    PerformanceMetrics,
    QueryType,
    StrategicContext,
    TriggerSignal,
)
from agentcore.ace.monitors.error_accumulator import ErrorRecord

logger = structlog.get_logger()


class ACEMemoryInterface:
    """
    MEM interface for strategic context queries (COMPASS ACE-3 - ACE-021).

    Enables COMPASS Meta-Thinker to query MEM for strategic context to inform
    intervention decisions. Tracks query latency and context health.

    NOTE: Currently uses mock data generator. Will integrate with real MEM
    service when MEM Phase 5 is implemented (Week 8).

    Features:
    - Strategic context queries (decision, error, capability, refresh)
    - Context health score computation (0-1)
    - Relevance scoring for results
    - Query latency tracking (<150ms target)
    - Deterministic mock data for testing

    Performance target: <150ms query latency (p95)
    """

    def __init__(self, mem_client: Any | None = None, seed: int | None = None) -> None:
        """
        Initialize ACEMemoryInterface.

        Args:
            mem_client: MEM client instance (for future integration, currently unused)
            seed: Random seed for deterministic mock data (optional, for testing)
        """
        self.mem_client = mem_client  # TODO: Replace with real MEM client when available
        self._random = random.Random(seed)
        logger.info("ACEMemoryInterface initialized", has_mem_client=mem_client is not None)

    async def get_strategic_context(
        self,
        query_type: QueryType,
        agent_id: str,
        task_id: UUID,
        context: dict[str, Any] | None = None,
    ) -> MemoryQueryResult:
        """
        Query MEM for strategic context.

        This is the main entry point for strategic context queries. Generates
        mock data based on query_type with realistic patterns and latency.

        Args:
            query_type: Type of strategic context query
            agent_id: Agent identifier
            task_id: Task identifier
            context: Additional context for query (optional)

        Returns:
            MemoryQueryResult with strategic context, relevance score, and latency

        Raises:
            ValueError: If agent_id is empty or query_type is invalid
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id cannot be empty")

        if context is None:
            context = {}

        start_time = time.perf_counter()
        query_id = uuid4()

        logger.info(
            "Querying MEM for strategic context",
            query_id=str(query_id),
            query_type=query_type.value,
            agent_id=agent_id,
            task_id=str(task_id),
        )

        # Generate mock strategic context based on query type
        strategic_context = self._generate_mock_context(query_type, agent_id, task_id, context)

        # Compute relevance score based on query type and context
        relevance_score = self._compute_relevance_score(query_type, strategic_context)

        # Calculate latency (ensure realistic timing)
        latency_ms = max(1, int((time.perf_counter() - start_time) * 1000))

        # Add simulated latency for realistic behavior (50-150ms)
        simulated_latency = self._random.randint(50, 150)
        latency_ms = max(latency_ms, simulated_latency)

        logger.info(
            "MEM query completed",
            query_id=str(query_id),
            query_type=query_type.value,
            latency_ms=latency_ms,
            relevance_score=relevance_score,
            context_health=strategic_context.context_health_score,
        )

        return MemoryQueryResult(
            query_id=query_id,
            strategic_context=strategic_context,
            relevance_score=relevance_score,
            query_latency_ms=latency_ms,
            metadata={
                "source": "mock_mem_generator",
                "query_type": query_type.value,
                "token_count": self._estimate_token_count(strategic_context),
            },
        )

    def _generate_mock_context(
        self,
        query_type: QueryType,
        agent_id: str,
        task_id: UUID,
        context: dict[str, Any],
    ) -> StrategicContext:
        """
        Generate mock strategic context based on query type.

        Each query type returns context tailored to its purpose with
        realistic data patterns.

        Args:
            query_type: Type of query
            agent_id: Agent identifier
            task_id: Task identifier
            context: Additional context

        Returns:
            StrategicContext with mock data appropriate for query type
        """
        if query_type == QueryType.STRATEGIC_DECISION:
            return self._generate_strategic_decision_context()
        elif query_type == QueryType.ERROR_ANALYSIS:
            return self._generate_error_analysis_context()
        elif query_type == QueryType.CAPABILITY_EVALUATION:
            return self._generate_capability_evaluation_context()
        elif query_type == QueryType.CONTEXT_REFRESH:
            return self._generate_context_refresh_context()
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def _generate_strategic_decision_context(self) -> StrategicContext:
        """Generate mock context for strategic decision making."""
        stage_summaries = [
            "Planning stage completed with 85% confidence, identified 5 key subtasks",
            "Execution stage in progress, 60% complete with moderate performance",
            "Reflection stage shows task breakdown effective for current approach",
        ]

        critical_facts = [
            "Task requires data transformation capabilities",
            "Agent has access to file processing and API tools",
            "Current approach uses sequential execution pattern",
            "Memory retrieval shows 75% relevance for recent queries",
            "Context window utilization at 60% capacity",
            "Recent actions show consistent API response times",
            "Agent confidence scores averaging 0.82 across stages",
        ]

        error_patterns = [
            "Occasional file parsing timeout in large datasets",
            "Memory retrieval returning stale results in 10% of queries",
        ]

        successful_patterns = [
            "API calls completing successfully with <200ms latency",
            "Context refresh improving accuracy by 15% average",
            "Task decomposition leading to 85% subtask completion rate",
        ]

        # Health score: typically healthy for decision making (0.7-0.9)
        health_score = self._compute_health_score(
            stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            freshness=0.85,
        )

        return StrategicContext(
            relevant_stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            context_health_score=health_score,
        )

    def _generate_error_analysis_context(self) -> StrategicContext:
        """Generate mock context for error analysis."""
        stage_summaries = [
            "Execution stage showing 2x error rate increase over baseline",
            "Planning stage completed but task breakdown may be insufficient",
        ]

        critical_facts = [
            "Error rate increased from 10% to 20% in last 5 actions",
            "File parsing failures concentrated in CSV processing",
            "Memory retrieval timeout occurring in 25% of requests",
        ]

        error_patterns = [
            "Repeated file parsing failures on large CSV files (>10MB)",
            "Memory retrieval returning stale results in 40% of queries",
            "API timeout pattern detected in data fetching operations",
            "Context overflow errors when processing complex nested data",
            "Capability mismatch for advanced data transformation tasks",
        ]

        successful_patterns = [
            "Small file processing completing without errors",
            "Simple API calls succeeding consistently",
        ]

        # Health score: degraded due to errors (0.4-0.7)
        health_score = self._compute_health_score(
            stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            freshness=0.70,
        )

        return StrategicContext(
            relevant_stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            context_health_score=health_score,
        )

    def _generate_capability_evaluation_context(self) -> StrategicContext:
        """Generate mock context for capability evaluation."""
        stage_summaries = [
            "Planning stage identified need for data transformation capabilities",
            "Execution stage showing capability limitations in 30% of actions",
            "Verification stage detected mismatch between task and agent capabilities",
        ]

        critical_facts = [
            "Task requires advanced data transformation (filtering, aggregation, joins)",
            "Agent has basic file processing but limited transformation tools",
            "Current capability set: file_read, file_write, api_call, basic_search",
            "Missing capabilities: data_transform, parallel_execution, advanced_search",
            "Capability coverage estimated at 60% for current task",
            "Action failure rate 40% higher when using mismatched capabilities",
        ]

        error_patterns = [
            "Data transformation attempts failing due to capability limits",
            "Complex query operations timing out with basic search",
        ]

        successful_patterns = [
            "File read/write operations completing successfully",
            "Simple API calls executing within expected time",
            "Basic search queries returning relevant results",
            "Sequential execution pattern working for simple tasks",
        ]

        # Health score: moderate (0.6-0.85) due to capability gaps
        health_score = self._compute_health_score(
            stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            freshness=0.80,
        )

        return StrategicContext(
            relevant_stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            context_health_score=health_score,
        )

    def _generate_context_refresh_context(self) -> StrategicContext:
        """Generate mock context for context refresh operations."""
        stage_summaries = [
            "Planning stage completed successfully with comprehensive task breakdown",
            "Execution stage 75% complete with strong performance metrics",
            "Reflection stage shows effective learning from recent actions",
            "Verification stage confirming outputs meet quality thresholds",
        ]

        critical_facts = [
            "Task execution progressing smoothly with 85% success rate",
            "Recent context refresh improved accuracy by 15%",
            "Memory utilization optimized at 65% capacity",
            "Agent performance trending positively over last 10 actions",
            "Context relevance scores averaging 0.88 for recent queries",
            "Error rate maintained at baseline levels (<10%)",
            "API response times consistent at <200ms average",
            "Tool usage patterns aligned with task requirements",
            "Capability coverage strong at 90% for current subtasks",
            "Learning signals positive across all reasoning stages",
        ]

        error_patterns: list[str] = []  # Minimal errors after refresh

        successful_patterns = [
            "Context compression maintaining 95% information retention",
            "Strategic fact extraction improving decision accuracy",
            "Memory retrieval latency reduced by 30% post-refresh",
            "Task decomposition quality improved after reflection",
            "API integration patterns optimized for efficiency",
        ]

        # Health score: high after refresh (0.8-0.95)
        health_score = self._compute_health_score(
            stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            freshness=0.95,
        )

        return StrategicContext(
            relevant_stage_summaries=stage_summaries,
            critical_facts=critical_facts,
            error_patterns=error_patterns,
            successful_patterns=successful_patterns,
            context_health_score=health_score,
        )

    def _compute_health_score(
        self,
        stage_summaries: list[str],
        critical_facts: list[str],
        error_patterns: list[str],
        successful_patterns: list[str],
        freshness: float,
    ) -> float:
        """
        Compute context health score (0-1).

        Health score is calculated based on:
        - Freshness: Simulated time since last update (0-1)
        - Completeness: Number of facts vs expected (0-1)
        - Error density: Ratio of error patterns to total patterns (inverse)

        Formula: health_score = (0.4 * freshness) + (0.3 * completeness) + (0.3 * (1 - error_density))

        Args:
            stage_summaries: Stage summary list
            critical_facts: Critical facts list
            error_patterns: Error patterns list
            successful_patterns: Successful patterns list
            freshness: Freshness score (0-1)

        Returns:
            Health score (0-1)
        """
        # Completeness: based on number of critical facts (expect 5-10)
        expected_facts = 7.5  # Midpoint of 5-10 range
        actual_facts = len(critical_facts)
        completeness = min(1.0, actual_facts / expected_facts)

        # Error density: ratio of errors to total patterns
        total_patterns = len(error_patterns) + len(successful_patterns)
        if total_patterns > 0:
            error_density = len(error_patterns) / total_patterns
        else:
            error_density = 0.0

        # Weighted formula
        health_score = (0.4 * freshness) + (0.3 * completeness) + (0.3 * (1.0 - error_density))

        # Clamp to [0, 1]
        return max(0.0, min(1.0, health_score))

    def _compute_relevance_score(
        self,
        query_type: QueryType,
        strategic_context: StrategicContext,
    ) -> float:
        """
        Compute relevance score for query result.

        Relevance is based on context health and query-specific factors.

        Args:
            query_type: Type of query
            strategic_context: Generated strategic context

        Returns:
            Relevance score (0-1)
        """
        # Base relevance from context health
        base_relevance = strategic_context.context_health_score

        # Adjust based on query type
        if query_type == QueryType.ERROR_ANALYSIS:
            # Higher relevance if error patterns present
            error_factor = min(1.0, len(strategic_context.error_patterns) / 5.0)
            return min(1.0, base_relevance + (error_factor * 0.15))

        elif query_type == QueryType.STRATEGIC_DECISION:
            # Higher relevance if critical facts abundant
            fact_factor = min(1.0, len(strategic_context.critical_facts) / 8.0)
            return min(1.0, base_relevance + (fact_factor * 0.10))

        elif query_type == QueryType.CAPABILITY_EVALUATION:
            # Higher relevance if patterns present
            pattern_count = len(strategic_context.error_patterns) + len(
                strategic_context.successful_patterns
            )
            pattern_factor = min(1.0, pattern_count / 8.0)
            return min(1.0, base_relevance + (pattern_factor * 0.12))

        elif query_type == QueryType.CONTEXT_REFRESH:
            # Higher relevance if comprehensive
            comprehensive_factor = min(
                1.0,
                (len(strategic_context.critical_facts) + len(strategic_context.successful_patterns))
                / 12.0,
            )
            return min(1.0, base_relevance + (comprehensive_factor * 0.08))

        return base_relevance

    def _estimate_token_count(self, strategic_context: StrategicContext) -> int:
        """
        Estimate token count for strategic context.

        Rough estimate: ~4 characters per token average.

        Args:
            strategic_context: Strategic context

        Returns:
            Estimated token count
        """
        total_chars = 0

        for summary in strategic_context.relevant_stage_summaries:
            total_chars += len(summary)

        for fact in strategic_context.critical_facts:
            total_chars += len(fact)

        for pattern in strategic_context.error_patterns:
            total_chars += len(pattern)

        for pattern in strategic_context.successful_patterns:
            total_chars += len(pattern)

        # Rough estimate: 4 chars per token
        return total_chars // 4

    # Specialized Strategic Query Methods (COMPASS ACE-3 - ACE-022)

    async def query_for_strategic_decision(
        self,
        trigger: TriggerSignal,
        agent_id: str,
        task_id: UUID,
    ) -> MemoryQueryResult:
        """
        Query MEM for strategic decision-making context.

        Returns context optimized for intervention decision-making based
        on the trigger signal type and metrics.

        Graceful degradation: Returns partial context on MEM failure.

        Args:
            trigger: TriggerSignal with trigger type, confidence, and metric values
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            MemoryQueryResult with strategic context for decision-making

        Raises:
            ValueError: If agent_id is empty
        """
        try:
            # Build context from trigger signal
            context = {
                "trigger_type": trigger.trigger_type.value,
                "confidence": trigger.confidence,
                "signals": trigger.signals,
                "metric_values": trigger.metric_values,
            }

            # Query MEM for strategic decision context
            result = await self.get_strategic_context(
                query_type=QueryType.STRATEGIC_DECISION,
                agent_id=agent_id,
                task_id=task_id,
                context=context,
            )

            logger.info(
                "Strategic decision query completed",
                agent_id=agent_id,
                task_id=str(task_id),
                trigger_type=trigger.trigger_type.value,
                relevance_score=result.relevance_score,
            )

            return result

        except Exception as e:
            # Log error
            logger.error(
                "MEM query failed for strategic decision, using fallback",
                agent_id=agent_id,
                task_id=str(task_id),
                trigger_type=trigger.trigger_type.value,
                error=str(e),
            )

            # Return fallback result
            fallback_context = StrategicContext(
                relevant_stage_summaries=[
                    f"Fallback: MEM unavailable for trigger {trigger.trigger_type.value}"
                ],
                critical_facts=[],
                error_patterns=[],
                successful_patterns=[],
                context_health_score=0.3,
            )

            return MemoryQueryResult(
                query_id=uuid4(),
                strategic_context=fallback_context,
                relevance_score=0.3,
                query_latency_ms=0,
                metadata={"error": str(e), "fallback": True, "trigger_type": trigger.trigger_type.value},
            )

    async def query_for_error_analysis(
        self,
        errors: list[ErrorRecord],
        agent_id: str,
        task_id: UUID,
    ) -> MemoryQueryResult:
        """
        Query MEM for error pattern analysis context.

        Returns context focused on error patterns, compounding errors,
        and historical error remediation strategies.

        Graceful degradation: Returns basic error summary on MEM failure.

        Args:
            errors: List of ErrorRecord instances to analyze
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            MemoryQueryResult with error analysis context

        Raises:
            ValueError: If agent_id is empty
        """
        try:
            # Build context from error records
            error_types = [e.error_type for e in errors]
            severities = [e.severity.value for e in errors]
            stages = [e.stage for e in errors]

            context = {
                "error_types": error_types,
                "severities": severities,
                "stages": stages,
                "error_count": len(errors),
            }

            # Query MEM for error analysis context
            result = await self.get_strategic_context(
                query_type=QueryType.ERROR_ANALYSIS,
                agent_id=agent_id,
                task_id=task_id,
                context=context,
            )

            logger.info(
                "Error analysis query completed",
                agent_id=agent_id,
                task_id=str(task_id),
                error_count=len(errors),
                relevance_score=result.relevance_score,
            )

            return result

        except Exception as e:
            # Log error
            logger.error(
                "MEM query failed for error analysis, using fallback",
                agent_id=agent_id,
                task_id=str(task_id),
                error_count=len(errors),
                error=str(e),
            )

            # Return fallback result with basic error summary
            error_summary = f"Fallback: MEM unavailable - {len(errors)} errors detected"
            if errors:
                error_summary += f" (types: {', '.join(set(e.error_type for e in errors[:3]))})"

            fallback_context = StrategicContext(
                relevant_stage_summaries=[error_summary],
                critical_facts=[],
                error_patterns=[],
                successful_patterns=[],
                context_health_score=0.3,
            )

            return MemoryQueryResult(
                query_id=uuid4(),
                strategic_context=fallback_context,
                relevance_score=0.3,
                query_latency_ms=0,
                metadata={"error": str(e), "fallback": True, "error_count": len(errors)},
            )

    async def query_for_capability_evaluation(
        self,
        task_requirements: list[str],
        agent_capabilities: list[str],
        performance_metrics: PerformanceMetrics,
        agent_id: str,
        task_id: UUID,
    ) -> MemoryQueryResult:
        """
        Query MEM for capability fitness evaluation context.

        Returns context about capability usage, success rates,
        and historical capability performance data.

        Graceful degradation: Returns basic fitness score on MEM failure.

        Args:
            task_requirements: List of required capabilities
            agent_capabilities: List of agent's available capabilities
            performance_metrics: Current performance metrics
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            MemoryQueryResult with capability evaluation context

        Raises:
            ValueError: If agent_id is empty
        """
        try:
            # Compute capability coverage
            matching_caps = set(task_requirements) & set(agent_capabilities)
            coverage_ratio = len(matching_caps) / len(task_requirements) if task_requirements else 0.0

            # Build context from requirements and capabilities
            context = {
                "requirements": task_requirements,
                "capabilities": agent_capabilities,
                "coverage_ratio": coverage_ratio,
                "metrics": {
                    "success_rate": performance_metrics.stage_success_rate,
                    "error_rate": performance_metrics.stage_error_rate,
                    "velocity": performance_metrics.overall_progress_velocity,
                },
            }

            # Query MEM for capability evaluation context
            result = await self.get_strategic_context(
                query_type=QueryType.CAPABILITY_EVALUATION,
                agent_id=agent_id,
                task_id=task_id,
                context=context,
            )

            logger.info(
                "Capability evaluation query completed",
                agent_id=agent_id,
                task_id=str(task_id),
                coverage_ratio=coverage_ratio,
                relevance_score=result.relevance_score,
            )

            return result

        except Exception as e:
            # Log error
            logger.error(
                "MEM query failed for capability evaluation, using fallback",
                agent_id=agent_id,
                task_id=str(task_id),
                error=str(e),
            )

            # Return fallback result with basic fitness score
            matching_caps = set(task_requirements) & set(agent_capabilities)
            coverage_ratio = len(matching_caps) / len(task_requirements) if task_requirements else 0.0

            fallback_summary = f"Fallback: MEM unavailable - capability coverage {coverage_ratio:.0%}"

            fallback_context = StrategicContext(
                relevant_stage_summaries=[fallback_summary],
                critical_facts=[],
                error_patterns=[],
                successful_patterns=[],
                context_health_score=0.3,
            )

            return MemoryQueryResult(
                query_id=uuid4(),
                strategic_context=fallback_context,
                relevance_score=0.3,
                query_latency_ms=0,
                metadata={
                    "error": str(e),
                    "fallback": True,
                    "coverage_ratio": coverage_ratio,
                },
            )

    async def query_for_context_refresh(
        self,
        agent_id: str,
        task_id: UUID,
        current_stage: str,
    ) -> MemoryQueryResult:
        """
        Query MEM for fresh compressed context.

        Returns latest context with cleared stale information,
        optimized for context refresh interventions.

        Graceful degradation: Returns basic stage summary on MEM failure.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            current_stage: Current reasoning stage

        Returns:
            MemoryQueryResult with refreshed context

        Raises:
            ValueError: If agent_id is empty or stage is invalid
        """
        try:
            # Validate stage
            valid_stages = {"planning", "execution", "reflection", "verification"}
            if current_stage not in valid_stages:
                raise ValueError(
                    f"Invalid stage '{current_stage}'. Must be one of: {valid_stages}"
                )

            # Build minimal context (just current stage)
            context = {
                "current_stage": current_stage,
                "refresh_reason": "context_refresh_intervention",
            }

            # Query MEM for context refresh
            result = await self.get_strategic_context(
                query_type=QueryType.CONTEXT_REFRESH,
                agent_id=agent_id,
                task_id=task_id,
                context=context,
            )

            logger.info(
                "Context refresh query completed",
                agent_id=agent_id,
                task_id=str(task_id),
                current_stage=current_stage,
                relevance_score=result.relevance_score,
            )

            return result

        except Exception as e:
            # Log error
            logger.error(
                "MEM query failed for context refresh, using fallback",
                agent_id=agent_id,
                task_id=str(task_id),
                current_stage=current_stage,
                error=str(e),
            )

            # Return fallback result with basic stage summary
            fallback_context = StrategicContext(
                relevant_stage_summaries=[
                    f"Fallback: MEM unavailable - currently in {current_stage} stage"
                ],
                critical_facts=[],
                error_patterns=[],
                successful_patterns=[],
                context_health_score=0.3,
            )

            return MemoryQueryResult(
                query_id=uuid4(),
                strategic_context=fallback_context,
                relevance_score=0.3,
                query_latency_ms=0,
                metadata={
                    "error": str(e),
                    "fallback": True,
                    "current_stage": current_stage,
                },
            )
