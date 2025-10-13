"""Autonomous Agent philosophy engine."""

import uuid
from datetime import datetime
from typing import Any

import structlog

from ..config.settings import get_settings
from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState
from ..services.llm_service import LLMConfig, LLMResponse, initialize_llm_service
from .autonomous_models import (
    AutonomousExecutionContext,
    AutonomousPromptTemplate,
    Decision,
    Goal,
    GoalPriority,
    GoalStatus,
    LearningExperience,
    Memory,
    MemoryType,
    TaskExecutionPlan,
)
from .base import PhilosophyEngine

logger = structlog.get_logger()


class AutonomousEngine(PhilosophyEngine):
    """Autonomous Agent philosophy execution engine."""

    def __init__(
        self,
        config: AgentConfig,
        use_real_llm: bool = True,
    ) -> None:
        """
        Initialize Autonomous engine.

        Args:
            config: Agent configuration
            use_real_llm: Whether to use real LLM (True) or simulated (False for testing)
        """
        super().__init__(config)
        self.prompt_template = AutonomousPromptTemplate()
        self.context: AutonomousExecutionContext | None = None
        self.use_real_llm = use_real_llm

        # Initialize LLM service if using real LLM
        if self.use_real_llm:
            try:
                runtime_config = get_settings()
                llm_config = LLMConfig(
                    portkey_api_key=runtime_config.portkey_api_key,
                    portkey_base_url=runtime_config.portkey_base_url,
                    default_model=runtime_config.default_llm_model,
                    fallback_models=runtime_config.llm_fallback_models,
                    default_temperature=runtime_config.llm_temperature,
                    default_max_tokens=runtime_config.llm_max_tokens,
                    timeout_seconds=runtime_config.llm_timeout_seconds,
                    max_retries=runtime_config.llm_max_retries,
                    cache_enabled=runtime_config.llm_cache_enabled,
                )
                self.llm_service = initialize_llm_service(llm_config)
            except Exception as e:
                logger.warning(
                    "autonomous_llm_initialization_failed",
                    error=str(e),
                    agent_id=self.agent_id,
                )
                self.use_real_llm = False
                self.llm_service = None
        else:
            self.llm_service = None

    async def initialize(self) -> None:
        """Initialize engine resources."""
        logger.info(
            "autonomous_engine_initialized",
            agent_id=self.agent_id,
        )

    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        logger.info(
            "autonomous_engine_cleanup",
            agent_id=self.agent_id,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        state: AgentExecutionState,
    ) -> dict[str, Any]:
        """
        Execute agent using autonomous goal-oriented logic.

        Args:
            input_data: Input with 'goal', optional 'priority', 'success_criteria'
            state: Current agent execution state

        Returns:
            Execution result with goal completion status and decision lineage
        """
        # Initialize execution context
        goal_description = input_data.get("goal", "")
        priority = GoalPriority(input_data.get("priority", "medium"))
        success_criteria = input_data.get("success_criteria", {})

        primary_goal = Goal(
            goal_id=str(uuid.uuid4()),
            description=goal_description,
            priority=priority,
            success_criteria=success_criteria,
        )

        self.context = AutonomousExecutionContext(
            agent_id=self.agent_id,
            primary_goal=primary_goal,
            active_goals=[primary_goal],
        )

        logger.info(
            "autonomous_execution_start",
            agent_id=self.agent_id,
            goal=goal_description,
            priority=priority.value,
        )

        # Execute autonomous loop
        try:
            # 1. Plan goal execution
            await self._plan_goal_execution(primary_goal)

            # 2. Execute plan
            await self._execute_plan()

            # 3. Evaluate results
            final_status = await self._evaluate_goal_completion(primary_goal)

            # 4. Learn from experience
            await self._learn_from_execution(primary_goal, final_status)

            logger.info(
                "autonomous_execution_complete",
                agent_id=self.agent_id,
                goal_status=final_status.value,
                decisions_made=len(self.context.decision_lineage),
                memories_created=len(self.context.long_term_memory),
            )

            return {
                "goal_status": final_status.value,
                "goal_progress": primary_goal.progress,
                "decisions_made": [d.model_dump() for d in self.context.decision_lineage],
                "memories_created": len(self.context.long_term_memory),
                "learning_experiences": [
                    exp.model_dump() for exp in self.context.learning_experiences
                ],
                "completed": final_status == GoalStatus.COMPLETED,
            }

        except Exception as e:
            logger.error(
                "autonomous_execution_failed",
                agent_id=self.agent_id,
                error=str(e),
            )
            raise

    async def _plan_goal_execution(self, goal: Goal) -> TaskExecutionPlan:
        """
        Create execution plan for goal.

        Args:
            goal: Goal to plan for

        Returns:
            Execution plan
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        # Analyze goal complexity
        is_complex = await self._is_complex_goal(goal)

        if is_complex:
            # Break down into sub-goals
            sub_goals = await self._decompose_goal(goal)
            goal.sub_goals = [sg.goal_id for sg in sub_goals]
            self.context.active_goals.extend(sub_goals)

            logger.info(
                "goal_decomposed",
                goal_id=goal.goal_id,
                sub_goals=len(sub_goals),
            )

        # Create execution plan
        steps = await self._generate_execution_steps(goal)

        plan = TaskExecutionPlan(
            plan_id=str(uuid.uuid4()),
            goal_id=goal.goal_id,
            steps=steps,
            estimated_duration=len(steps) * 60,  # Rough estimate
        )

        self.context.current_plan = plan

        # Record decision
        decision = Decision(
            decision_id=str(uuid.uuid4()),
            goal_id=goal.goal_id,
            description=f"Created execution plan with {len(steps)} steps",
            rationale="Analyzed goal and determined optimal execution strategy",
            confidence=0.8,
        )
        self.context.decision_lineage.append(decision)

        logger.info(
            "execution_plan_created",
            goal_id=goal.goal_id,
            steps=len(steps),
        )

        return plan

    async def _execute_plan(self) -> None:
        """Execute the current plan."""
        if not self.context or not self.context.current_plan:
            raise RuntimeError("No execution plan available")

        plan = self.context.current_plan
        goal_id = plan.goal_id

        # Find corresponding goal
        goal = next((g for g in self.context.active_goals if g.goal_id == goal_id), None)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")

        # Mark goal as in progress
        goal.status = GoalStatus.IN_PROGRESS
        goal.started_at = datetime.now()

        # Execute each step
        for i, step in enumerate(plan.steps):
            # Make decision for this step
            decision = await self._make_decision(goal, step)
            self.context.decision_lineage.append(decision)

            # Execute step
            success = await self._execute_step(step)

            # Update progress
            goal.progress = (i + 1) / len(plan.steps)

            # Store memory of execution
            memory = Memory(
                memory_id=str(uuid.uuid4()),
                memory_type=MemoryType.EPISODIC,
                content={
                    "step": step,
                    "decision": decision.decision_id,
                    "success": success,
                },
                importance=0.7 if success else 0.9,  # Failures are more important
                related_goals=[goal_id],
                tags=["execution", "step"],
            )
            await self._store_memory(memory)

            logger.debug(
                "step_executed",
                goal_id=goal_id,
                step=i + 1,
                total=len(plan.steps),
                success=success,
            )

    async def _evaluate_goal_completion(self, goal: Goal) -> GoalStatus:
        """
        Evaluate if goal is complete.

        Args:
            goal: Goal to evaluate

        Returns:
            Goal status
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        # Check success criteria
        if goal.success_criteria:
            criteria_met = await self._check_success_criteria(goal.success_criteria)
            if criteria_met:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.now()
                goal.progress = 1.0
            else:
                goal.status = GoalStatus.FAILED
        else:
            # No explicit criteria - assume completed if all steps done
            if goal.progress >= 1.0:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.now()
            else:
                goal.status = GoalStatus.FAILED

        # Move to completed goals
        if goal in self.context.active_goals:
            self.context.active_goals.remove(goal)
        self.context.completed_goals.append(goal)

        logger.info(
            "goal_evaluated",
            goal_id=goal.goal_id,
            status=goal.status.value,
            progress=goal.progress,
        )

        return goal.status

    async def _learn_from_execution(self, goal: Goal, status: GoalStatus) -> None:
        """
        Learn from execution experience.

        Args:
            goal: Executed goal
            status: Final goal status
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        success = status == GoalStatus.COMPLETED

        # Create learning experience
        experience = LearningExperience(
            experience_id=str(uuid.uuid4()),
            goal_id=goal.goal_id,
            action_taken=f"Executed goal: {goal.description}",
            outcome=f"Goal {status.value}",
            success=success,
            lesson_learned=await self._extract_lesson(goal, status),
            impact_score=0.8 if success else 0.9,  # Learn more from failures
        )

        self.context.learning_experiences.append(experience)

        # Store as semantic memory
        memory = Memory(
            memory_id=str(uuid.uuid4()),
            memory_type=MemoryType.SEMANTIC,
            content={
                "goal": goal.description,
                "status": status.value,
                "lesson": experience.lesson_learned,
            },
            importance=0.9,
            related_goals=[goal.goal_id],
            tags=["learning", "experience"],
        )
        await self._store_memory(memory)

        logger.info(
            "learning_complete",
            goal_id=goal.goal_id,
            lesson=experience.lesson_learned,
        )

    async def _is_complex_goal(self, goal: Goal) -> bool:
        """
        Determine if goal is complex and needs decomposition.

        Args:
            goal: Goal to analyze

        Returns:
            True if complex
        """
        # Simple heuristic - complex if description is long or contains "and"
        return len(goal.description.split()) > 10 or " and " in goal.description.lower()

    async def _decompose_goal(self, goal: Goal) -> list[Goal]:
        """
        Decompose complex goal into sub-goals.

        Args:
            goal: Goal to decompose

        Returns:
            List of sub-goals
        """
        # Simulated decomposition
        # In production, this would use LLM to intelligently break down goals

        if " and " in goal.description.lower():
            parts = goal.description.split(" and ")
            sub_goals = [
                Goal(
                    goal_id=str(uuid.uuid4()),
                    description=part.strip(),
                    priority=goal.priority,
                    parent_goal_id=goal.goal_id,
                )
                for part in parts
            ]
        else:
            # Create sequential sub-goals
            sub_goals = [
                Goal(
                    goal_id=str(uuid.uuid4()),
                    description=f"Sub-task {i+1} of: {goal.description}",
                    priority=goal.priority,
                    parent_goal_id=goal.goal_id,
                )
                for i in range(2)  # Create 2 sub-goals
            ]

        return sub_goals

    async def _generate_execution_steps(self, goal: Goal) -> list[dict[str, Any]]:
        """
        Generate execution steps for goal.

        Args:
            goal: Goal to generate steps for

        Returns:
            List of execution steps
        """
        # Simulated step generation
        # In production, would use LLM to create detailed steps

        return [
            {"step": 1, "action": f"Analyze requirements for: {goal.description}"},
            {"step": 2, "action": f"Execute main task for: {goal.description}"},
            {"step": 3, "action": f"Verify completion of: {goal.description}"},
        ]

    async def _make_decision(self, goal: Goal, step: dict[str, Any]) -> Decision:
        """
        Make decision for execution step.

        Args:
            goal: Current goal
            step: Execution step

        Returns:
            Decision record
        """
        return Decision(
            decision_id=str(uuid.uuid4()),
            goal_id=goal.goal_id,
            description=f"Executing step {step['step']}: {step['action']}",
            rationale="Following execution plan",
            alternatives_considered=["Skip step", "Modify approach"],
            confidence=0.85,
        )

    async def _execute_step(self, step: dict[str, Any]) -> bool:
        """
        Execute a single step.

        Args:
            step: Step to execute

        Returns:
            True if successful
        """
        # Simulated execution
        # In production, would perform actual work
        logger.debug("executing_step", step=step)
        return True  # Assume success for simulation

    async def _check_success_criteria(self, criteria: dict[str, Any]) -> bool:
        """
        Check if success criteria are met.

        Args:
            criteria: Success criteria

        Returns:
            True if criteria met
        """
        # Simulated criteria checking
        # In production, would evaluate actual criteria
        return True

    async def _extract_lesson(self, goal: Goal, status: GoalStatus) -> str:
        """
        Extract lesson from execution.

        Args:
            goal: Executed goal
            status: Goal status

        Returns:
            Lesson learned
        """
        if status == GoalStatus.COMPLETED:
            return f"Successfully completed goal: {goal.description}. Execution plan was effective."
        else:
            return f"Failed to complete goal: {goal.description}. Need to refine approach."

    async def _store_memory(self, memory: Memory) -> None:
        """
        Store memory in appropriate memory store.

        Args:
            memory: Memory to store
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        if memory.memory_type == MemoryType.WORKING:
            self.context.working_memory.append(memory)
            # Limit working memory size
            if len(self.context.working_memory) > 10:
                self.context.working_memory = self.context.working_memory[-10:]
        else:
            self.context.long_term_memory.append(memory)

        memory.access_count += 1
        memory.last_accessed = datetime.now()
