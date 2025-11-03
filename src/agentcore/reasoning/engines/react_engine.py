"""
ReAct Reasoning Engine (Reasoning + Acting).

Implements iterative reasoning with thought-action-observation cycles.
Synergizes reasoning and acting through interleaved generation of reasoning traces
and task-specific actions.

Key features:
- Multi-iteration thought-action-observation loop
- Explicit reasoning traces before actions
- Action execution with observation feedback
- Answer extraction when reasoning complete
- Supports tool/API integration (optional)

Based on "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., 2022) - https://arxiv.org/abs/2210.03629
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from ..models.reasoning_models import (
    ReActConfig,
    ReasoningMetrics,
    ReasoningResult,
)
from ..services.llm_client import LLMClient

logger = structlog.get_logger()


class ReActEngine:
    """
    ReAct (Reasoning + Acting) Engine.

    Executes iterative reasoning with thought-action-observation cycles.
    """

    def __init__(self, llm_client: LLMClient, config: ReActConfig):
        """
        Initialize ReAct engine.

        Args:
            llm_client: LLM client for generation
            config: ReAct configuration
        """
        self.llm_client = llm_client
        self.config = config

        logger.info(
            "react_engine_initialized",
            max_iterations=config.max_iterations,
            max_tokens_per_step=config.max_tokens_per_step,
            allow_tool_use=config.allow_tool_use,
        )

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "react"

    @property
    def version(self) -> str:
        """Get strategy version."""
        return "1.0.0"

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum thought-action-observation cycles",
                },
                "max_tokens_per_step": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 8192,
                    "default": 2048,
                    "description": "Maximum tokens per step",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.7,
                    "description": "Sampling temperature",
                },
                "allow_tool_use": {
                    "type": "boolean",
                    "default": False,
                    "description": "Allow external tool/API calls",
                },
            },
        }

    def get_capabilities(self) -> list[str]:
        """Get list of capabilities this strategy provides."""
        capabilities = ["reasoning.strategy.react"]
        if self.config.allow_tool_use:
            capabilities.append("reasoning.action.tool_use")
        return capabilities

    def _build_initial_prompt(self, query: str, system_prompt: str | None = None) -> str:
        """
        Build initial ReAct prompt with examples.

        Args:
            query: User query
            system_prompt: Optional system prompt override

        Returns:
            Formatted ReAct prompt
        """
        base_system = system_prompt or """You are a helpful AI assistant using the ReAct (Reasoning + Acting) framework.

For each step, you will:
1. Think: Reason about what to do next
2. Act: Decide on an action (if needed)
3. Observe: Get feedback from the action

Format your response as:
Thought: [your reasoning about what to do next]
Action: [action to take, or "Answer" if you have the final answer]
Observation: [result of the action, or final answer if Action was "Answer"]

When you have the final answer, use:
Thought: I now have enough information to answer.
Action: Answer
Observation: <answer>final answer here</answer>

Continue the thought-action-observation cycle until you reach a conclusion."""

        full_prompt = f"{base_system}\n\nQuestion: {query}\n\nLet's solve this step by step using the ReAct framework:\n"

        return full_prompt

    def _parse_step(self, content: str) -> dict[str, str]:
        """
        Parse a ReAct step into thought, action, observation.

        Args:
            content: Generated content from LLM

        Returns:
            dict with 'thought', 'action', 'observation' keys
        """
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", content, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*(.*?)(?=Observation:|$)", content, re.DOTALL | re.IGNORECASE)
        obs_match = re.search(r"Observation:\s*(.*?)(?=$)", content, re.DOTALL | re.IGNORECASE)

        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "observation": obs_match.group(1).strip() if obs_match else "",
        }

    def _extract_answer(self, content: str) -> str | None:
        """
        Extract final answer if present.

        Args:
            content: Content to check for answer

        Returns:
            Extracted answer or None if not found
        """
        # Look for <answer>...</answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        return None

    def _execute_action(self, action: str) -> str:
        """
        Execute an action and return observation.

        Args:
            action: Action to execute

        Returns:
            Observation result

        Note:
            This is a simplified implementation. In production, this would
            integrate with a tool registry for actual action execution.
        """
        # For now, simulate observations for common actions
        action_lower = action.lower()

        if "answer" in action_lower:
            return "Answer ready"
        elif "search" in action_lower or "lookup" in action_lower:
            return "Simulated search result: [Information would be retrieved from search]"
        elif "calculate" in action_lower or "compute" in action_lower:
            return "Simulated calculation: [Calculation would be performed]"
        else:
            return f"Simulated action result for: {action[:50]}..."

    async def execute(
        self,
        query: str,
        max_iterations: int | None = None,
        temperature: float | None = None,
        max_tokens_per_step: int | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Execute ReAct reasoning.

        Args:
            query: The question or problem to solve
            max_iterations: Optional max iterations override
            temperature: Optional temperature override
            max_tokens_per_step: Optional max tokens per step override
            system_prompt: Optional system prompt override
            **kwargs: Additional arguments (ignored)

        Returns:
            ReasoningResult with answer, metrics, and trace
        """
        start_time = time.time()

        # Use config defaults if not provided
        max_iter = max_iterations if max_iterations is not None else self.config.max_iterations
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens_per_step if max_tokens_per_step is not None else self.config.max_tokens_per_step

        logger.info(
            "react_execute_start",
            query_length=len(query),
            max_iterations=max_iter,
            temperature=temp,
        )

        # Build initial prompt
        prompt = self._build_initial_prompt(query, system_prompt)

        # Track iteration history
        iterations = []
        total_tokens = 0
        answer = None

        for iteration in range(max_iter):
            logger.debug("react_iteration_start", iteration=iteration)

            try:
                # Generate thought-action-observation step
                result = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=max_tok,
                )

                total_tokens += result.tokens_used

                # Parse step
                step = self._parse_step(result.content)

                # Check for answer
                extracted = self._extract_answer(result.content)
                if extracted:
                    answer = extracted
                    logger.info("react_answer_found", iteration=iteration)

                    iterations.append({
                        "iteration": iteration,
                        "thought": step["thought"],
                        "action": step["action"],
                        "observation": step["observation"],
                        "answer_found": True,
                        "tokens": result.tokens_used,
                    })
                    break

                # Execute action if not answer
                if step["action"] and "answer" not in step["action"].lower():
                    observation = self._execute_action(step["action"])
                    step["observation"] = observation

                iterations.append({
                    "iteration": iteration,
                    "thought": step["thought"],
                    "action": step["action"],
                    "observation": step["observation"],
                    "answer_found": False,
                    "tokens": result.tokens_used,
                })

                # Append to prompt for next iteration
                prompt += f"\nThought: {step['thought']}\n"
                prompt += f"Action: {step['action']}\n"
                prompt += f"Observation: {step['observation']}\n"

            except Exception as e:
                logger.error("react_iteration_failed", iteration=iteration, error=str(e))
                raise RuntimeError(f"ReAct iteration {iteration} failed: {str(e)}") from e

        # If no answer found, use last observation
        if answer is None:
            if iterations and iterations[-1]["observation"]:
                answer = iterations[-1]["observation"]
            else:
                answer = "Unable to determine answer after maximum iterations"
            logger.warning("react_max_iterations_reached", iterations=len(iterations))

        # Calculate metrics
        execution_time_ms = int((time.time() - start_time) * 1000)

        metrics = ReasoningMetrics(
            total_tokens=total_tokens,
            execution_time_ms=execution_time_ms,
            strategy_specific={
                "total_iterations": len(iterations),
                "answer_found_at_iteration": next((i["iteration"] for i in iterations if i.get("answer_found")), None),
                "temperature": temp,
                "max_iterations": max_iter,
            },
        )

        logger.info(
            "react_execute_complete",
            answer_length=len(answer),
            total_tokens=total_tokens,
            iterations=len(iterations),
            execution_time_ms=execution_time_ms,
        )

        return ReasoningResult(
            answer=answer,
            strategy_used=self.name,
            metrics=metrics,
            trace=iterations,
        )
