"""
Planner Module Implementation

Implements task decomposition and execution plan generation for the modular agent core.
Provides intelligent query analysis, step-by-step planning, and plan refinement capabilities.

This module follows the PEVG (Planner, Executor, Verifier, Generator) architecture pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from agentcore.a2a_protocol.models.jsonrpc import A2AContext
from agentcore.modular.base import BasePlanner
from agentcore.modular.interfaces import (
    ExecutionPlan,
    PlannerQuery,
    PlanRefinement,
)
from agentcore.modular.models import (
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    PlanStatus,
    StepDependency,
    StepStatus,
    SuccessCriteria,
    SuccessCriterion,
    ToolRequirement,
)

logger = structlog.get_logger()


class Planner(BasePlanner):
    """
    Concrete Planner module implementation.

    Provides task decomposition using ReAct-style reasoning:
    1. Analyze the query to understand intent and requirements
    2. Decompose into sequential steps with clear actions
    3. Identify tool requirements for each step
    4. Define success criteria and step dependencies
    5. Generate structured execution plan

    Supports plan refinement based on execution feedback.
    """

    def __init__(
        self,
        a2a_context: A2AContext | None = None,
        logger: Any | None = None,
        max_steps: int = 20,
        enable_parallel: bool = False,
    ) -> None:
        """
        Initialize Planner module.

        Args:
            a2a_context: A2A protocol context for tracing
            logger: Structured logger instance
            max_steps: Maximum number of steps allowed in a plan
            enable_parallel: Whether to allow parallel step execution
        """
        super().__init__(a2a_context, logger)
        self.max_steps = max_steps
        self.enable_parallel = enable_parallel

        self.logger.info(
            "planner_initialized",
            max_steps=max_steps,
            enable_parallel=enable_parallel,
        )

    async def _analyze_query_impl(self, query: PlannerQuery) -> ExecutionPlan:
        """
        Analyze query and generate execution plan.

        Implementation uses rule-based task decomposition:
        1. Parse query for key verbs and entities
        2. Determine required actions (search, process, analyze, etc.)
        3. Build dependency graph of steps
        4. Generate structured plan with tool requirements

        Args:
            query: Query to analyze with optional context

        Returns:
            Structured execution plan with ordered steps

        Raises:
            ValueError: If query is invalid or too complex
        """
        self.logger.info("analyzing_query", query=query.query)

        # Basic validation
        if not query.query or not query.query.strip():
            raise ValueError("Query cannot be empty")

        # Parse query to identify intent
        query_lower = query.query.lower()
        steps: list[EnhancedPlanStep] = []

        # Determine query type and decompose accordingly
        # Check computation first, then multi-step, then information retrieval
        if self._is_computation_task(query_lower):
            steps = await self._plan_computation(query)
        elif self._is_multi_step_task(query_lower):
            steps = await self._plan_multi_step(query)
        elif self._is_information_retrieval(query_lower):
            steps = await self._plan_information_retrieval(query)
        else:
            # Default: single-step plan
            steps = await self._plan_simple(query)

        # Validate step count
        if len(steps) > self.max_steps:
            raise ValueError(
                f"Plan exceeds maximum steps: {len(steps)} > {self.max_steps}"
            )

        # Create success criteria
        success_criteria = SuccessCriteria(
            criteria=[
                SuccessCriterion(
                    criterion_id="query_answered",
                    description="Query successfully answered",
                    metric_name="verification_status",
                    operator="eq",
                    threshold="approved",
                    weight=1.0,
                    required=True,
                ),
                SuccessCriterion(
                    criterion_id="all_steps_completed",
                    description="All plan steps completed without errors",
                    metric_name="completed_steps",
                    operator="eq",
                    threshold=len(steps),
                    weight=0.8,
                    required=True,
                ),
            ]
        )

        # Build enhanced execution plan
        plan = EnhancedExecutionPlan(
            plan_id=str(uuid4()),
            query=query.query,
            steps=steps,
            success_criteria=success_criteria,
            status=PlanStatus.PENDING,
            max_iterations=query.constraints.get("max_iterations", 5)
            if query.constraints
            else 5,
            created_at=datetime.now(UTC).isoformat(),
        )

        self.logger.info(
            "plan_generated",
            plan_id=plan.plan_id,
            step_count=len(steps),
            query=query.query,
        )

        return plan

    async def _refine_plan_impl(self, refinement: PlanRefinement) -> ExecutionPlan:
        """
        Refine existing plan based on verification feedback.

        Refinement strategies:
        1. Add error handling steps for failed operations
        2. Adjust parameters based on feedback
        3. Insert additional validation steps
        4. Modify step ordering for efficiency

        Args:
            refinement: Refinement request with feedback and constraints

        Returns:
            Refined execution plan

        Raises:
            ValueError: If refinement cannot be applied
        """
        self.logger.info(
            "refining_plan",
            plan_id=refinement.plan_id,
            feedback=refinement.feedback,
        )

        # For now, create a new plan with adjusted constraints
        # In a real implementation, this would load the existing plan
        # and modify it based on specific feedback patterns

        query_text = refinement.constraints.get("original_query", "")
        if not query_text:
            raise ValueError("Refinement requires original_query in constraints")

        # Create new query with refinement context
        refined_query = PlannerQuery(
            query=query_text,
            context={
                "refinement_feedback": refinement.feedback,
                "previous_plan_id": refinement.plan_id,
                **refinement.constraints,
            },
            constraints=refinement.constraints,
        )

        # Generate refined plan
        refined_plan = await self._analyze_query_impl(refined_query)

        self.logger.info(
            "plan_refined",
            original_plan_id=refinement.plan_id,
            new_plan_id=refined_plan.plan_id,
        )

        return refined_plan

    # ========================================================================
    # Query Classification Helpers
    # ========================================================================

    def _is_information_retrieval(self, query: str) -> bool:
        """Check if query is information retrieval."""
        keywords = ["what", "who", "when", "where", "find", "search", "lookup"]
        return any(kw in query for kw in keywords)

    def _is_computation_task(self, query: str) -> bool:
        """Check if query requires computation."""
        keywords = ["calculate", "compute", "sum", "average", "count", "total"]
        return any(kw in query for kw in keywords)

    def _is_multi_step_task(self, query: str) -> bool:
        """Check if query requires multiple steps."""
        indicators = [" and ", " then ", " after", "after ", "first", "second", "finally"]
        return any(ind in query for ind in indicators) or query.count(".") > 1

    # ========================================================================
    # Plan Generation Strategies
    # ========================================================================

    async def _plan_information_retrieval(
        self, query: PlannerQuery
    ) -> list[EnhancedPlanStep]:
        """Generate plan for information retrieval queries."""
        steps: list[EnhancedPlanStep] = []

        # Step 1: Search for information
        steps.append(
            EnhancedPlanStep(
                step_id="search_1",
                action="search_knowledge",
                parameters={"query": query.query},
                tool_requirements=[
                    ToolRequirement(
                        tool_name="knowledge_search",
                        version="1.0",
                        required=True,
                    )
                ],
                status=StepStatus.PENDING,
            )
        )

        # Step 2: Extract and structure results
        steps.append(
            EnhancedPlanStep(
                step_id="extract_1",
                action="extract_information",
                parameters={"format": "structured"},
                dependencies=[
                    StepDependency(
                        step_id="search_1",
                        dependency_type="sequential",
                        required=True,
                    )
                ],
                status=StepStatus.PENDING,
            )
        )

        return steps

    async def _plan_computation(
        self, query: PlannerQuery
    ) -> list[EnhancedPlanStep]:
        """Generate plan for computational queries."""
        steps: list[EnhancedPlanStep] = []

        # Step 1: Gather required data
        steps.append(
            EnhancedPlanStep(
                step_id="gather_1",
                action="gather_data",
                parameters={"query": query.query},
                status=StepStatus.PENDING,
            )
        )

        # Step 2: Perform computation
        steps.append(
            EnhancedPlanStep(
                step_id="compute_1",
                action="perform_computation",
                parameters={"operation": "calculate"},
                dependencies=[
                    StepDependency(
                        step_id="gather_1",
                        dependency_type="sequential",
                        required=True,
                    )
                ],
                tool_requirements=[
                    ToolRequirement(
                        tool_name="calculator",
                        version="1.0",
                        required=True,
                    )
                ],
                status=StepStatus.PENDING,
            )
        )

        return steps

    async def _plan_multi_step(
        self, query: PlannerQuery
    ) -> list[EnhancedPlanStep]:
        """Generate plan for complex multi-step queries."""
        # For complex queries, decompose into 3-5 logical steps
        steps: list[EnhancedPlanStep] = []

        # Generic multi-step pattern
        step_count = 3  # Default to 3 steps for multi-step tasks

        for i in range(step_count):
            step_number = i + 1
            step = EnhancedPlanStep(
                step_id=f"step_{step_number}",
                action=f"execute_subtask_{step_number}",
                parameters={"query_part": f"subtask_{step_number}"},
                status=StepStatus.PENDING,
            )

            # Add dependency on previous step (sequential execution)
            if i > 0:
                step.dependencies = [
                    StepDependency(
                        step_id=f"step_{i}",
                        dependency_type="sequential",
                        required=True,
                    )
                ]

            steps.append(step)

        return steps

    async def _plan_simple(self, query: PlannerQuery) -> list[EnhancedPlanStep]:
        """Generate simple single-step plan."""
        return [
            EnhancedPlanStep(
                step_id="execute_1",
                action="execute_query",
                parameters={"query": query.query},
                status=StepStatus.PENDING,
            )
        ]

    async def health_check(self) -> dict[str, Any]:
        """
        Check Planner module health.

        Returns:
            Health status with module information
        """
        return {
            "status": "healthy",
            "module": "Planner",
            "max_steps": self.max_steps,
            "enable_parallel": self.enable_parallel,
            "execution_id": self.state.execution_id,
        }
