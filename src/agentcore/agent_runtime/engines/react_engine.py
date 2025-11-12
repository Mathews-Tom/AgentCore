"""ReAct (Reasoning and Acting) philosophy engine."""

import re
import uuid
from typing import Any

import structlog

from ..models.agent_config import AgentConfig
from ..models.agent_state import AgentExecutionState

from ..tools.base import ExecutionContext
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from ..tools.registration import register_native_builtin_tools

from .base import PhilosophyEngine
from .react_models import (
    ReActExecutionContext,
    ReActPromptTemplate,
    ReActStep,
    ReActStepType,
    ToolCall,
)

logger = structlog.get_logger()


class ReActEngine(PhilosophyEngine):
    """ReAct philosophy execution engine."""

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """
        Initialize ReAct engine.

        Args:
            config: Agent configuration
            tool_registry: Tool registry (uses global if not provided)
        """
        super().__init__(config)
        if tool_registry is None:
            tool_registry = ToolRegistry()
            register_native_builtin_tools(tool_registry)
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        self.prompt_template = ReActPromptTemplate()
        self.context: ReActExecutionContext | None = None

    async def initialize(self) -> None:
        """Initialize engine resources."""
        logger.info(
            "react_engine_initialized",
            agent_id=self.agent_id,
        )

    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        logger.info(
            "react_engine_cleanup",
            agent_id=self.agent_id,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        state: AgentExecutionState,
    ) -> dict[str, Any]:
        """
        Execute agent using ReAct reasoning cycle.

        Args:
            input_data: Input with 'goal' and optional 'max_iterations'
            state: Current agent execution state

        Returns:
            Execution result with final answer and reasoning steps
        """
        # Initialize execution context
        goal = input_data.get("goal", "")
        max_iterations = input_data.get("max_iterations", 10)

        self.context = ReActExecutionContext(
            agent_id=self.agent_id,
            goal=goal,
            max_iterations=max_iterations,
            available_tools=[tool.metadata.tool_id for tool in self.tool_registry.list_all()],
        )

        logger.info(
            "react_execution_start",
            agent_id=self.agent_id,
            goal=goal,
            max_iterations=max_iterations,
        )

        # Execute reasoning loop
        try:
            while not self.context.completed and self.context.current_iteration < max_iterations:
                self.context.current_iteration += 1

                # Generate thought
                thought = await self._generate_thought()
                self._add_step(ReActStepType.THOUGHT, thought)

                # Check for final answer
                if self._is_final_answer(thought):
                    final_answer = self._extract_final_answer(thought)
                    self.context.final_answer = final_answer
                    self.context.completed = True
                    self._add_step(ReActStepType.FINAL_ANSWER, final_answer)
                    break

                # Parse and execute action
                action = self._parse_action(thought)
                if action:
                    self._add_step(
                        ReActStepType.ACTION,
                        f"{action.tool_name}({action.parameters})",
                    )

                    # Execute tool
                    context = ExecutionContext(
                        agent_id=self.agent_id,
                        user_id=self.agent_id,
                    )
                    tool_result = await self.tool_executor.execute_tool(
                        action.tool_name,
                        action.parameters,
                        context,
                    )
                    result = {
                        "success": tool_result.status.value == "success",
                        "result": tool_result.result,
                        "error": tool_result.error,
                    }

                    # Create observation
                    observation = self._create_observation(result)
                    self._add_step(ReActStepType.OBSERVATION, observation)

            # Handle incomplete execution
            if not self.context.completed:
                logger.warning(
                    "react_max_iterations_reached",
                    agent_id=self.agent_id,
                    iterations=self.context.current_iteration,
                )
                self.context.final_answer = (
                    "Maximum iterations reached without finding a complete answer."
                )

            logger.info(
                "react_execution_complete",
                agent_id=self.agent_id,
                iterations=self.context.current_iteration,
                steps=len(self.context.steps),
            )

            return {
                "final_answer": self.context.final_answer,
                "iterations": self.context.current_iteration,
                "steps": [step.model_dump() for step in self.context.steps],
                "completed": self.context.completed,
            }

        except Exception as e:
            logger.error(
                "react_execution_failed",
                agent_id=self.agent_id,
                error=str(e),
            )
            raise

    async def _generate_thought(self) -> str:
        """
        Generate next reasoning step.

        Returns:
            Thought content (simulated for now)
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        # Format history
        history = self._format_history()

        # Simulate LLM reasoning (in production, this would call an LLM)
        # For now, we'll use simple rule-based logic
        iteration = self.context.current_iteration

        if iteration == 1:
            # First iteration - analyze the goal
            if "calculate" in self.context.goal.lower() or "+" in self.context.goal or "-" in self.context.goal:
                return f"THOUGHT: I need to use the calculator tool to solve: {self.context.goal}\nACTION: calculator(operation='+', a=5, b=3)"
            elif "time" in self.context.goal.lower():
                return "THOUGHT: I need to get the current time\nACTION: get_current_time()"
            else:
                return f"THOUGHT: Let me echo the goal to confirm understanding\nACTION: echo(message='{self.context.goal}')"
        elif iteration == 2:
            # Second iteration - provide final answer based on observation
            last_observation = self._get_last_observation()
            return f"THOUGHT: Based on the result, I can provide the final answer\nFINAL_ANSWER: {last_observation}"
        else:
            # Default final answer
            return f"FINAL_ANSWER: Completed task: {self.context.goal}"

    def _format_history(self) -> str:
        """
        Format execution history for prompt.

        Returns:
            Formatted history string
        """
        if not self.context or not self.context.steps:
            return "No previous steps"

        history_lines = []
        for step in self.context.steps:
            history_lines.append(f"{step.step_type.value.upper()}: {step.content}")

        return "\n".join(history_lines)

    def _get_last_observation(self) -> str:
        """Get the last observation from steps."""
        if not self.context:
            return ""

        for step in reversed(self.context.steps):
            if step.step_type == ReActStepType.OBSERVATION:
                return step.content

        return ""

    def _add_step(self, step_type: ReActStepType, content: str) -> None:
        """
        Add a step to execution context.

        Args:
            step_type: Type of step
            content: Step content
        """
        if not self.context:
            raise RuntimeError("Execution context not initialized")

        step = ReActStep(
            step_number=len(self.context.steps) + 1,
            step_type=step_type,
            content=content,
        )
        self.context.steps.append(step)

    def _is_final_answer(self, thought: str) -> bool:
        """
        Check if thought contains final answer.

        Args:
            thought: Thought content

        Returns:
            True if contains final answer
        """
        return "FINAL_ANSWER:" in thought.upper()

    def _extract_final_answer(self, thought: str) -> str:
        """
        Extract final answer from thought.

        Args:
            thought: Thought content

        Returns:
            Extracted final answer
        """
        match = re.search(r"FINAL_ANSWER:\s*(.+)", thought, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return thought

    def _parse_action(self, thought: str) -> ToolCall | None:
        """
        Parse action from thought.

        Args:
            thought: Thought content

        Returns:
            ToolCall if action found, None otherwise
        """
        # Look for ACTION: tool_name(params) pattern
        action_match = re.search(
            r"ACTION:\s*(\w+)\((.*?)\)",
            thought,
            re.IGNORECASE,
        )

        if not action_match:
            return None

        tool_name = action_match.group(1)
        params_str = action_match.group(2)

        # Parse parameters (simple key=value parsing)
        parameters = self._parse_parameters(params_str)

        return ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            call_id=str(uuid.uuid4()),
        )

    def _parse_parameters(self, params_str: str) -> dict[str, Any]:
        """
        Parse parameter string into dictionary.

        Args:
            params_str: Parameter string (e.g., "a=5, b=3")

        Returns:
            Parameters dictionary
        """
        if not params_str.strip():
            return {}

        parameters: dict[str, Any] = {}

        # Split by comma
        parts = params_str.split(",")

        for part in parts:
            part = part.strip()
            if "=" not in part:
                continue

            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes
            value = value.strip("'\"")

            # Try to convert to appropriate type
            try:
                # Try int
                parameters[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    parameters[key] = float(value)
                except ValueError:
                    # Keep as string
                    parameters[key] = value

        return parameters

    def _create_observation(self, result: Any) -> str:
        """
        Create observation from tool result.

        Args:
            result: Tool execution result

        Returns:
            Observation string
        """
        if result.success:
            return f"Tool executed successfully. Result: {result.result}"
        else:
            return f"Tool execution failed. Error: {result.error}"
