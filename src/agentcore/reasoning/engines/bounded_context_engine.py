"""
Bounded Context Reasoning Engine.

Implements fixed-window iterative reasoning with linear complexity scaling.
Maintains constant memory footprint regardless of reasoning depth by using
compressed carryover mechanisms between iterations.

Key features:
- Fixed context window (configurable chunk_size)
- Multi-iteration reasoning with carryover compression
- Answer detection via stop sequences
- O(N) computational complexity where N = number of chunks
- Predictable resource consumption
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from ..models.reasoning_models import (
    BoundedContextConfig,
    BoundedContextIterationResult,
    BoundedContextResult,
    CarryoverContent,
    IterationMetrics,
)
from ..services.carryover_generator import CarryoverGenerator
from ..services.llm_client import GenerationResult, LLMClient

logger = structlog.get_logger()


class BoundedContextEngine:
    """
    Bounded Context Reasoning Engine.

    Executes multi-iteration reasoning with fixed context windows,
    maintaining linear computational complexity and constant memory usage.
    """

    def __init__(self, llm_client: LLMClient, config: BoundedContextConfig):
        """
        Initialize bounded context engine.

        Args:
            llm_client: LLM client for generation
            config: Bounded context configuration
        """
        self.llm_client = llm_client
        self.config = config

        # Initialize carryover generator
        self.carryover_generator = CarryoverGenerator(
            llm_client=llm_client,
            max_carryover_tokens=config.carryover_size,
        )

        logger.info(
            "bounded_context_engine_initialized",
            chunk_size=config.chunk_size,
            carryover_size=config.carryover_size,
            max_iterations=config.max_iterations,
        )

    def _build_iteration_prompt(
        self,
        query: str,
        iteration: int,
        carryover: CarryoverContent | None = None,
    ) -> str:
        """
        Build prompt for a single iteration.

        Args:
            query: Original user query
            iteration: Current iteration number (0-indexed)
            carryover: Carryover from previous iteration (None for first)

        Returns:
            Formatted prompt for this iteration
        """
        if iteration == 0:
            # First iteration: query only
            prompt = f"""Solve the following problem step by step.

**Problem:**
{query}

**Instructions:**
- Think through the problem carefully
- If you find the answer, wrap it in <answer> tags like: <answer>your answer</answer>
- If you need more reasoning steps, end with <continue> to signal continuation
- Show your reasoning process

**Your response:**"""
            return prompt

        # Subsequent iterations: query + carryover
        carryover_text = carryover.to_text() if carryover else "No carryover available"

        prompt = f"""Continue solving the following problem.

**Original Problem:**
{query}

**Progress from Previous Iteration:**
{carryover_text}

**Instructions:**
- Continue from where you left off
- If you find the answer, wrap it in <answer> tags like: <answer>your answer</answer>
- If you need more reasoning steps, end with <continue> to signal continuation
- Build on the progress made so far

**Your response:**"""
        return prompt

    def _calculate_max_new_tokens(self, iteration: int, prompt_tokens: int) -> int:
        """
        Calculate max_new_tokens based on token budget.

        Args:
            iteration: Current iteration number
            prompt_tokens: Tokens in the current prompt

        Returns:
            Maximum new tokens to generate
        """
        # Token budget = chunk_size - prompt_tokens
        # Leave some buffer for stop sequences and formatting
        buffer = 100
        max_new = self.config.chunk_size - prompt_tokens - buffer

        # Ensure we have reasonable generation capacity
        min_generation = 512
        return max(min_generation, max_new)

    def _extract_answer(self, content: str) -> str | None:
        """
        Extract answer from content if <answer> tag is present.

        Args:
            content: Generated content

        Returns:
            Extracted answer or None if no answer found
        """
        if "<answer>" not in content:
            return None

        # Extract content between <answer> and </answer>
        start = content.find("<answer>") + len("<answer>")
        end = content.find("</answer>", start)

        if end == -1:
            # No closing tag, take everything after <answer>
            return content[start:].strip()

        return content[start:end].strip()

    async def _generate_carryover(
        self,
        query: str,
        iteration_content: str,
        previous_carryover: CarryoverContent | None,
    ) -> CarryoverContent:
        """
        Generate compressed carryover for next iteration.

        Args:
            query: Original query
            iteration_content: Content from current iteration
            previous_carryover: Carryover from previous iteration

        Returns:
            Compressed carryover for next iteration
        """
        return await self.carryover_generator.generate_carryover(
            query=query,
            iteration_content=iteration_content,
            previous_carryover=previous_carryover,
            temperature=0.3,  # Lower temperature for focused compression
        )

    async def reason(
        self,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> BoundedContextResult:
        """
        Execute bounded context reasoning.

        Args:
            query: Problem to solve
            system_prompt: Optional system prompt
            temperature: Sampling temperature

        Returns:
            Bounded context reasoning result

        Raises:
            RuntimeError: If reasoning fails
        """
        start_time = time.time()
        iterations: list[BoundedContextIterationResult] = []
        total_tokens = 0
        carryover: CarryoverContent | None = None
        answer: str | None = None

        logger.info("bounded_context_reasoning_started", query_length=len(query))

        for iteration in range(self.config.max_iterations):
            iteration_start = time.time()

            # Build prompt for this iteration
            prompt = self._build_iteration_prompt(query, iteration, carryover)
            prompt_tokens = self.llm_client.count_tokens(prompt)

            # Calculate max new tokens
            max_new_tokens = self._calculate_max_new_tokens(iteration, prompt_tokens)

            logger.debug(
                "bounded_context_iteration_start",
                iteration=iteration,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
            )

            # Generate response
            try:
                generation: GenerationResult = await self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_sequences=["<answer>", "<continue>"],
                )

                iteration_time = int((time.time() - iteration_start) * 1000)
                total_tokens += generation.tokens_used

                # Check for answer
                extracted_answer = self._extract_answer(generation.content)
                has_answer = extracted_answer is not None

                if has_answer:
                    answer = extracted_answer

                # Generate carryover if continuing
                carryover_generated = False
                if not has_answer and iteration < self.config.max_iterations - 1:
                    carryover = await self._generate_carryover(
                        query=query,
                        iteration_content=generation.content,
                        previous_carryover=carryover,
                    )
                    carryover_generated = True

                # Record iteration result
                iteration_result = BoundedContextIterationResult(
                    content=generation.content,
                    has_answer=has_answer,
                    answer=extracted_answer,
                    carryover=carryover if carryover_generated else None,
                    metrics=IterationMetrics(
                        iteration=iteration,
                        tokens=generation.tokens_used,
                        has_answer=has_answer,
                        carryover_generated=carryover_generated,
                        execution_time_ms=iteration_time,
                    ),
                )

                iterations.append(iteration_result)

                logger.info(
                    "bounded_context_iteration_complete",
                    iteration=iteration,
                    tokens=generation.tokens_used,
                    has_answer=has_answer,
                    carryover_generated=carryover_generated,
                )

                # Terminate if answer found
                if has_answer:
                    break

            except Exception as e:
                logger.error(
                    "bounded_context_iteration_failed",
                    iteration=iteration,
                    error=str(e),
                )
                raise RuntimeError(f"Iteration {iteration} failed: {e}") from e

        # Check if we found an answer
        if answer is None:
            logger.warning(
                "bounded_context_no_answer_found",
                iterations=len(iterations),
                max_iterations=self.config.max_iterations,
            )
            # Use last iteration content as answer
            answer = iterations[-1].content if iterations else "No answer found"

        # Calculate metrics
        execution_time = int((time.time() - start_time) * 1000)
        carryover_compressions = sum(1 for it in iterations if it.carryover is not None)

        # Calculate compute savings (placeholder - real calculation in BCR-008)
        # Assume traditional reasoning would use chunk_size * iterations tokens
        traditional_tokens = self.config.chunk_size * len(iterations)
        compute_savings_pct = ((traditional_tokens - total_tokens) / traditional_tokens * 100) if traditional_tokens > 0 else 0.0

        result = BoundedContextResult(
            answer=answer,
            iterations=iterations,
            total_tokens=total_tokens,
            total_iterations=len(iterations),
            compute_savings_pct=compute_savings_pct,
            carryover_compressions=carryover_compressions,
            execution_time_ms=execution_time,
        )

        logger.info(
            "bounded_context_reasoning_complete",
            total_iterations=len(iterations),
            total_tokens=total_tokens,
            compute_savings_pct=compute_savings_pct,
            execution_time_ms=execution_time,
            answer_found=answer is not None,
        )

        return result
