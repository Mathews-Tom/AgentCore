"""
Chain of Thought Reasoning Engine.

Implements single-pass reasoning with explicit step-by-step thinking.
Prompts the LLM to show its reasoning process before providing an answer.

Key features:
- Single-pass reasoning (no iterations)
- Explicit step-by-step prompting
- Natural language reasoning chain
- Answer extraction from reasoning trace
- Simpler than bounded context but less compute-efficient for long queries

Based on "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
(Wei et al., 2022) - https://arxiv.org/abs/2201.11903
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from ..models.reasoning_models import (
    ChainOfThoughtConfig,
    ReasoningMetrics,
    ReasoningResult,
)
from ..services.llm_client import LLMClient

logger = structlog.get_logger()


class ChainOfThoughtEngine:
    """
    Chain of Thought Reasoning Engine.

    Executes single-pass reasoning with explicit step-by-step thinking prompts.
    """

    def __init__(self, llm_client: LLMClient, config: ChainOfThoughtConfig):
        """
        Initialize Chain of Thought engine.

        Args:
            llm_client: LLM client for generation
            config: Chain of Thought configuration
        """
        self.llm_client = llm_client
        self.config = config

        logger.info(
            "chain_of_thought_engine_initialized",
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    @property
    def name(self) -> str:
        """Get strategy name."""
        return "chain_of_thought"

    @property
    def version(self) -> str:
        """Get strategy version."""
        return "1.0.0"

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema for this strategy."""
        return {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 32768,
                    "default": 4096,
                    "description": "Maximum tokens for reasoning output",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.7,
                    "description": "Sampling temperature",
                },
                "show_reasoning": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include reasoning steps in trace",
                },
            },
        }

    def get_capabilities(self) -> list[str]:
        """Get list of capabilities this strategy provides."""
        return ["reasoning.strategy.chain_of_thought"]

    def _build_prompt(self, query: str, system_prompt: str | None = None) -> str:
        """
        Build Chain of Thought prompt.

        Args:
            query: User query
            system_prompt: Optional system prompt override

        Returns:
            Formatted prompt with CoT instructions
        """
        base_system = system_prompt or """You are a helpful AI assistant that thinks step-by-step.
When solving problems, show your reasoning process clearly before providing the final answer.

Instructions:
1. Break down the problem into steps
2. Explain your reasoning for each step
3. Show your work and thought process
4. Provide the final answer wrapped in <answer>...</answer> tags

Example format:
Let me think through this step by step:
Step 1: [explanation]
Step 2: [explanation]
...
<answer>Final answer here</answer>"""

        # Combine system prompt with query
        full_prompt = f"{base_system}\n\nQuestion: {query}\n\nLet me think step by step:"

        return full_prompt

    def _extract_answer(self, content: str) -> str:
        """
        Extract final answer from reasoning chain.

        Args:
            content: Full reasoning output

        Returns:
            Extracted answer or full content if no answer tags found
        """
        # Try to extract content within <answer>...</answer> tags
        answer_match = re.search(
            r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE
        )

        if answer_match:
            answer = answer_match.group(1).strip()
            logger.debug("answer_extracted_from_tags", answer_length=len(answer))
            return answer

        # Fallback: try to find "Answer:" prefix
        answer_match = re.search(
            r"(?:final\s+)?answer:\s*(.*?)(?:\n|$)", content, re.IGNORECASE
        )

        if answer_match:
            answer = answer_match.group(1).strip()
            logger.debug("answer_extracted_from_prefix", answer_length=len(answer))
            return answer

        # Last resort: return last non-empty line
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if lines:
            answer = lines[-1]
            logger.debug("answer_extracted_from_last_line", answer_length=len(answer))
            return answer

        logger.warning("no_answer_extracted", content_length=len(content))
        return content.strip()

    async def execute(
        self,
        query: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        show_reasoning: bool = True,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Execute Chain of Thought reasoning.

        Args:
            query: The question or problem to solve
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            system_prompt: Optional system prompt override
            show_reasoning: Whether to include reasoning steps in trace
            **kwargs: Additional arguments (ignored)

        Returns:
            ReasoningResult with answer, metrics, and optional trace
        """
        start_time = time.time()

        # Use config defaults if not provided
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        logger.info(
            "chain_of_thought_execute_start",
            query_length=len(query),
            temperature=temp,
            max_tokens=max_tok,
        )

        # Build prompt
        prompt = self._build_prompt(query, system_prompt)

        # Generate reasoning
        try:
            result = await self.llm_client.generate(
                prompt=prompt, temperature=temp, max_tokens=max_tok
            )

            # Extract answer
            answer = self._extract_answer(result.content)

            # Calculate metrics
            execution_time_ms = int((time.time() - start_time) * 1000)

            metrics = ReasoningMetrics(
                total_tokens=result.tokens_used,
                execution_time_ms=execution_time_ms,
                strategy_specific={
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "finish_reason": result.finish_reason,
                    "model": result.model,
                },
            )

            # Build trace if requested
            trace = None
            if show_reasoning:
                trace = [
                    {
                        "type": "reasoning",
                        "content": result.content,
                        "tokens": result.tokens_used,
                    },
                    {"type": "answer", "content": answer, "extracted": True},
                ]

            logger.info(
                "chain_of_thought_execute_complete",
                answer_length=len(answer),
                total_tokens=result.tokens_used,
                execution_time_ms=execution_time_ms,
            )

            return ReasoningResult(
                answer=answer,
                strategy_used=self.name,
                metrics=metrics,
                trace=trace,
            )

        except Exception as e:
            logger.error(
                "chain_of_thought_execute_failed",
                error=str(e),
                query_length=len(query),
            )
            raise RuntimeError(f"Chain of Thought reasoning failed: {str(e)}") from e
