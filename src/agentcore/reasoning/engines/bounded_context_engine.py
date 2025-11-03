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
    ReasoningMetrics,
    ReasoningResult,
)
from ..services.carryover_generator import CarryoverGenerator
from ..services.llm_client import GenerationResult, LLMClient
from ..services.metrics_calculator import MetricsCalculator

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

        # Create preliminary result for metrics calculation
        preliminary_result = BoundedContextResult(
            answer=answer,
            iterations=iterations,
            total_tokens=total_tokens,
            total_iterations=len(iterations),
            compute_savings_pct=0.0,  # Will be calculated below
            carryover_compressions=carryover_compressions,
            execution_time_ms=execution_time,
        )

        # Calculate compute savings using MetricsCalculator
        query_tokens = self.llm_client.count_tokens(query)
        compute_savings_pct = MetricsCalculator.calculate_compute_savings(
            bounded_result=preliminary_result,
            config=self.config,
            query_tokens=query_tokens,
        )

        # Update result with actual compute savings
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

    # ReasoningStrategy Protocol Implementation

    async def execute(self, query: str, **kwargs: Any) -> ReasoningResult:
        """
        Execute bounded context reasoning (ReasoningStrategy protocol method).

        This is the protocol-compliant method that wraps the native reason() method.

        Args:
            query: The problem or question to solve
            **kwargs: Strategy-specific configuration parameters:
                - chunk_size: Maximum tokens per iteration (default from config)
                - carryover_size: Tokens to carry forward (default from config)
                - max_iterations: Maximum iterations (default from config)
                - system_prompt: Optional system prompt
                - temperature: Sampling temperature (default 0.7)

        Returns:
            ReasoningResult with standardized format

        Raises:
            ValueError: If query is invalid or configuration is incompatible
            RuntimeError: If strategy execution fails
        """
        # Extract strategy-specific parameters
        system_prompt = kwargs.get("system_prompt")
        temperature = kwargs.get("temperature", 0.7)

        # Override config if custom parameters provided
        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)
        carryover_size = kwargs.get("carryover_size", self.config.carryover_size)
        max_iterations = kwargs.get("max_iterations", self.config.max_iterations)

        # Create temporary config if parameters differ
        if (
            chunk_size != self.config.chunk_size
            or carryover_size != self.config.carryover_size
            or max_iterations != self.config.max_iterations
        ):
            original_config = self.config
            self.config = BoundedContextConfig(
                chunk_size=chunk_size,
                carryover_size=carryover_size,
                max_iterations=max_iterations,
            )
        else:
            original_config = None

        try:
            # Execute native bounded context reasoning
            bc_result = await self.reason(
                query=query,
                system_prompt=system_prompt,
                temperature=temperature,
            )

            # Convert to ReasoningResult format
            return self._convert_to_reasoning_result(bc_result)
        finally:
            # Restore original config if modified
            if original_config is not None:
                self.config = original_config

    def _convert_to_reasoning_result(
        self, bc_result: BoundedContextResult
    ) -> ReasoningResult:
        """
        Convert BoundedContextResult to ReasoningResult.

        Args:
            bc_result: Native bounded context result

        Returns:
            ReasoningResult with standardized format
        """
        # Build strategy-specific metrics
        strategy_specific = {
            "iterations": [
                {
                    "iteration": it.metrics.iteration,
                    "tokens": it.metrics.tokens,
                    "has_answer": it.metrics.has_answer,
                    "carryover_generated": it.metrics.carryover_generated,
                    "execution_time_ms": it.metrics.execution_time_ms,
                }
                for it in bc_result.iterations
            ],
            "total_iterations": bc_result.total_iterations,
            "compute_savings_pct": bc_result.compute_savings_pct,
            "carryover_compressions": bc_result.carryover_compressions,
        }

        # Build trace for debugging (optional)
        trace = [
            {
                "iteration": it.metrics.iteration,
                "content_preview": it.content[:200] + "..."
                if len(it.content) > 200
                else it.content,
                "has_answer": it.has_answer,
                "answer": it.answer,
                "tokens": it.metrics.tokens,
            }
            for it in bc_result.iterations
        ]

        return ReasoningResult(
            answer=bc_result.answer,
            strategy_used=self.name,
            metrics=ReasoningMetrics(
                total_tokens=bc_result.total_tokens,
                execution_time_ms=bc_result.execution_time_ms,
                strategy_specific=strategy_specific,
            ),
            trace=trace,
        )

    def get_config_schema(self) -> dict[str, Any]:
        """
        Get the configuration schema for bounded context strategy.

        Returns:
            JSON schema for strategy configuration
        """
        return {
            "type": "object",
            "properties": {
                "chunk_size": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 32768,
                    "default": 8192,
                    "description": "Maximum tokens per reasoning iteration",
                },
                "carryover_size": {
                    "type": "integer",
                    "minimum": 512,
                    "maximum": 16384,
                    "default": 4096,
                    "description": "Maximum tokens to carry forward between iterations",
                },
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "Maximum number of reasoning iterations allowed",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt for LLM",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.7,
                    "description": "Sampling temperature for generation",
                },
            },
        }

    def get_capabilities(self) -> list[str]:
        """
        Get the list of capabilities this strategy provides.

        Returns:
            Capability identifiers for agent advertisement
        """
        return [
            "reasoning.strategy.bounded_context",
            "long_form_reasoning",
            "compute_efficient",
            "memory_bounded",
        ]

    @property
    def name(self) -> str:
        """Get the unique name of this strategy."""
        return "bounded_context"

    @property
    def version(self) -> str:
        """Get the version of this strategy implementation."""
        return "1.0.0"
