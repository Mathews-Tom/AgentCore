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

        # Load existing plan from constraints
        existing_plan = refinement.constraints.get("existing_plan")
        query_text = refinement.constraints.get("original_query", "")

        if not query_text:
            raise ValueError("Refinement requires original_query in constraints")

        # If existing plan is provided, perform intelligent modifications
        if existing_plan:
            self.logger.info("refining_existing_plan", plan_id=refinement.plan_id)

            # Parse existing plan into EnhancedExecutionPlan
            if isinstance(existing_plan, dict):
                plan = EnhancedExecutionPlan(**existing_plan)
            elif isinstance(existing_plan, EnhancedExecutionPlan):
                plan = existing_plan
            else:
                raise ValueError("existing_plan must be dict or EnhancedExecutionPlan")

            # Parse feedback to determine refinement strategy
            feedback_analysis = self._analyze_feedback(refinement.feedback)

            # Apply refinement strategies based on feedback
            refined_plan = await self._apply_refinement_strategies(
                plan, feedback_analysis, refinement
            )

            # Update plan metadata
            refined_plan.parent_plan_id = refinement.plan_id
            refined_plan.current_iteration = plan.current_iteration + 1

            # Generate new plan ID
            refined_plan.plan_id = str(uuid4())

            self.logger.info(
                "plan_refined_intelligently",
                original_plan_id=refinement.plan_id,
                new_plan_id=refined_plan.plan_id,
                strategies_applied=list(feedback_analysis.keys()),
            )

            return refined_plan

        # Fallback: create new plan with refinement context
        self.logger.info("creating_new_plan_with_feedback", plan_id=refinement.plan_id)

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

    # ========================================================================
    # Plan Refinement Strategies
    # ========================================================================

    def _analyze_feedback(self, feedback: str) -> dict[str, Any]:
        """
        Analyze verification feedback to determine refinement strategies.

        Parses feedback text to identify specific issues and determine
        which refinement strategies should be applied.

        Args:
            feedback: Feedback text from Verifier module

        Returns:
            Dict mapping strategy names to extracted information
        """
        analysis: dict[str, Any] = {}
        feedback_lower = feedback.lower()

        # Strategy 1: Error handling
        # Detect: "failed", "error", "exception", "crashed", "timeout"
        if any(kw in feedback_lower for kw in ["failed", "error", "exception", "crashed", "timeout"]):
            analysis["add_error_handling"] = {
                "detected": True,
                "keywords": [kw for kw in ["failed", "error", "exception", "crashed", "timeout"] if kw in feedback_lower],
            }

        # Strategy 2: Parameter adjustment
        # Detect: "invalid parameter", "wrong value", "incorrect", "adjust", "change parameter"
        if any(kw in feedback_lower for kw in ["invalid parameter", "wrong value", "incorrect", "adjust", "change parameter"]):
            analysis["adjust_parameters"] = {
                "detected": True,
                "keywords": [kw for kw in ["invalid parameter", "wrong value", "incorrect", "adjust", "change parameter"] if kw in feedback_lower],
            }

        # Strategy 3: Validation steps
        # Detect: "validation", "verify", "check", "validate", "missing validation"
        if any(kw in feedback_lower for kw in ["validation", "verify", "check", "validate", "missing validation"]):
            analysis["add_validation"] = {
                "detected": True,
                "keywords": [kw for kw in ["validation", "verify", "check", "validate", "missing validation"] if kw in feedback_lower],
            }

        # Strategy 4: Step ordering
        # Detect: "order", "sequence", "dependency", "before", "after", "wrong order"
        if any(kw in feedback_lower for kw in ["order", "sequence", "dependency", "before", "after", "wrong order"]):
            analysis["reorder_steps"] = {
                "detected": True,
                "keywords": [kw for kw in ["order", "sequence", "dependency", "before", "after", "wrong order"] if kw in feedback_lower],
            }

        return analysis

    async def _apply_refinement_strategies(
        self,
        plan: EnhancedExecutionPlan,
        feedback_analysis: dict[str, Any],
        refinement: PlanRefinement,
    ) -> EnhancedExecutionPlan:
        """
        Apply refinement strategies to existing plan.

        Args:
            plan: Existing execution plan to refine
            feedback_analysis: Analysis results from _analyze_feedback
            refinement: Original refinement request

        Returns:
            Refined execution plan
        """
        # Create a deep copy of the plan to modify
        refined_plan = EnhancedExecutionPlan(**plan.model_dump())

        # Strategy 1: Add error handling steps
        if "add_error_handling" in feedback_analysis:
            refined_plan = await self._add_error_handling_steps(refined_plan, refinement)

        # Strategy 2: Adjust parameters
        if "adjust_parameters" in feedback_analysis:
            refined_plan = await self._adjust_step_parameters(refined_plan, refinement)

        # Strategy 3: Insert validation steps
        if "add_validation" in feedback_analysis:
            refined_plan = await self._insert_validation_steps(refined_plan, refinement)

        # Strategy 4: Reorder steps
        if "reorder_steps" in feedback_analysis:
            refined_plan = await self._reorder_steps(refined_plan, refinement)

        return refined_plan

    async def _add_error_handling_steps(
        self,
        plan: EnhancedExecutionPlan,
        refinement: PlanRefinement,
    ) -> EnhancedExecutionPlan:
        """
        Add error handling steps for failed operations.

        Identifies failed steps and inserts retry/error handling logic.
        """
        failed_step_ids = refinement.constraints.get("failed_step_ids", [])

        for step in plan.steps:
            # Check if this step failed based on feedback or constraints
            if step.step_id in failed_step_ids or step.status == StepStatus.FAILED:
                # Increase max_retries for failed steps
                step.max_retries = min(step.max_retries + 2, 10)
                step.status = StepStatus.PENDING  # Reset to retry
                step.retry_count = 0
                step.error = None

                self.logger.info(
                    "added_error_handling",
                    step_id=step.step_id,
                    new_max_retries=step.max_retries,
                )

        return plan

    async def _adjust_step_parameters(
        self,
        plan: EnhancedExecutionPlan,
        refinement: PlanRefinement,
    ) -> EnhancedExecutionPlan:
        """
        Adjust step parameters based on feedback.

        Modifies parameters for steps that had invalid values.
        """
        parameter_adjustments = refinement.constraints.get("parameter_adjustments", {})

        for step in plan.steps:
            if step.step_id in parameter_adjustments:
                adjustments = parameter_adjustments[step.step_id]
                step.parameters.update(adjustments)

                self.logger.info(
                    "adjusted_parameters",
                    step_id=step.step_id,
                    adjustments=adjustments,
                )

        return plan

    async def _insert_validation_steps(
        self,
        plan: EnhancedExecutionPlan,
        refinement: PlanRefinement,
    ) -> EnhancedExecutionPlan:
        """
        Insert additional validation steps into the plan.

        Adds validation steps after operations that need verification.
        """
        steps_needing_validation = refinement.constraints.get("steps_needing_validation", [])

        new_steps: list[EnhancedPlanStep] = []
        for step in plan.steps:
            new_steps.append(step)

            # Insert validation step after this step if needed
            if step.step_id in steps_needing_validation:
                validation_step = EnhancedPlanStep(
                    step_id=f"{step.step_id}_validation",
                    action="validate_result",
                    parameters={
                        "target_step": step.step_id,
                        "validation_rules": ["result_not_null", "format_valid"],
                    },
                    dependencies=[
                        StepDependency(
                            step_id=step.step_id,
                            dependency_type="sequential",
                            required=True,
                        )
                    ],
                    status=StepStatus.PENDING,
                )
                new_steps.append(validation_step)

                self.logger.info(
                    "inserted_validation_step",
                    after_step=step.step_id,
                    validation_step_id=validation_step.step_id,
                )

        plan.steps = new_steps
        return plan

    async def _reorder_steps(
        self,
        plan: EnhancedExecutionPlan,
        refinement: PlanRefinement,
    ) -> EnhancedExecutionPlan:
        """
        Modify step ordering for efficiency.

        Reorders steps based on dependency analysis and feedback.
        """
        new_order = refinement.constraints.get("step_order", [])

        if new_order:
            # Reorder steps according to provided order
            step_map = {step.step_id: step for step in plan.steps}
            reordered_steps: list[EnhancedPlanStep] = []

            for step_id in new_order:
                if step_id in step_map:
                    reordered_steps.append(step_map[step_id])

            # Add any steps not in the new order at the end
            for step in plan.steps:
                if step.step_id not in new_order:
                    reordered_steps.append(step)

            plan.steps = reordered_steps

            self.logger.info(
                "reordered_steps",
                new_order=new_order,
            )

        return plan

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
