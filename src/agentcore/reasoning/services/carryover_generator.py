"""
Carryover Generator for Bounded Context Reasoning.

Generates compressed summaries of reasoning progress between iterations,
preserving key insights while discarding redundant information.

The carryover mechanism is critical for maintaining reasoning continuity
while keeping context size bounded, enabling linear scaling for extended
reasoning tasks.
"""

from __future__ import annotations

import json
import re

import structlog

from ..models.reasoning_models import CarryoverContent
from .llm_client import LLMClient

logger = structlog.get_logger()


class CarryoverGenerator:
    """
    Generates compressed carryover summaries for bounded context reasoning.

    The carryover captures essential reasoning progress while discarding
    redundant or completed steps, enabling continuous reasoning within
    fixed context windows.
    """

    def __init__(self, llm_client: LLMClient, max_carryover_tokens: int = 4096):
        """
        Initialize carryover generator.

        Args:
            llm_client: LLM client for generation
            max_carryover_tokens: Maximum tokens for carryover content
        """
        self.llm_client = llm_client
        self.max_carryover_tokens = max_carryover_tokens

        logger.info(
            "carryover_generator_initialized",
            max_carryover_tokens=max_carryover_tokens,
        )

    def _build_carryover_prompt(
        self,
        query: str,
        iteration_content: str,
        previous_carryover: CarryoverContent | None,
    ) -> str:
        """
        Build prompt for carryover generation.

        Args:
            query: Original user query
            iteration_content: Content from current iteration
            previous_carryover: Previous carryover (if any)

        Returns:
            Formatted prompt for carryover generation
        """
        previous_section = ""
        if previous_carryover:
            previous_section = f"""
**Previous Progress:**
{previous_carryover.to_text()}
"""

        prompt = f"""You are helping compress reasoning progress for a multi-iteration problem solver.

**Original Problem:**
{query}
{previous_section}
**Current Iteration Output:**
{iteration_content}

**Your Task:**
Generate a compressed summary of the reasoning progress so far. This summary will be used in the next iteration to continue reasoning.

**Output Format (JSON):**
Return ONLY a JSON object with this exact structure:
{{
  "current_strategy": "Brief description of the reasoning approach being used",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
  "progress": "One paragraph summarizing what has been accomplished",
  "next_steps": ["Step 1", "Step 2", "Step 3"],
  "unresolved": ["Question 1", "Challenge 1"]
}}

**Guidelines:**
1. Be concise - keep total output under {self.max_carryover_tokens // 4} words
2. Preserve critical insights and intermediate results
3. Discard redundant information and completed steps
4. Focus on what's needed to continue reasoning
5. Identify clear next steps
6. Return ONLY the JSON object, no additional text

**Your JSON output:**"""

        return prompt

    def _parse_carryover_json(self, content: str) -> CarryoverContent | None:
        """
        Parse carryover JSON from LLM response.

        Args:
            content: LLM generated content

        Returns:
            Parsed CarryoverContent or None if parsing fails
        """
        try:
            # Extract JSON from response (handle potential markdown formatting)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.warning("carryover_json_not_found", content_preview=content[:200])
                return None

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Validate required fields
            required_fields = ["current_strategy", "key_findings", "progress", "next_steps", "unresolved"]
            if not all(field in data for field in required_fields):
                logger.warning("carryover_missing_fields", fields=list(data.keys()))
                return None

            # Create CarryoverContent
            carryover = CarryoverContent(
                current_strategy=data["current_strategy"],
                key_findings=data["key_findings"],
                progress=data["progress"],
                next_steps=data["next_steps"],
                unresolved=data["unresolved"],
            )

            return carryover

        except json.JSONDecodeError as e:
            logger.warning("carryover_json_parse_error", error=str(e), content_preview=content[:200])
            return None
        except Exception as e:
            logger.error("carryover_parse_unexpected_error", error=str(e))
            return None

    def _create_fallback_carryover(
        self,
        iteration_content: str,
        previous_carryover: CarryoverContent | None,
    ) -> CarryoverContent:
        """
        Create fallback carryover when generation fails.

        Args:
            iteration_content: Content from current iteration
            previous_carryover: Previous carryover (if any)

        Returns:
            Fallback carryover content
        """
        # Extract first few sentences as progress
        sentences = iteration_content.split('. ')[:3]
        progress = '. '.join(sentences) + '.'

        # Preserve previous strategy if available
        strategy = previous_carryover.current_strategy if previous_carryover else "Continue reasoning"

        return CarryoverContent(
            current_strategy=strategy,
            key_findings=["Reasoning in progress (fallback carryover)"],
            progress=progress[:500],  # Limit to 500 chars
            next_steps=["Continue analysis"],
            unresolved=["Complete reasoning task"],
        )

    async def generate_carryover(
        self,
        query: str,
        iteration_content: str,
        previous_carryover: CarryoverContent | None = None,
        temperature: float = 0.3,
    ) -> CarryoverContent:
        """
        Generate compressed carryover for next iteration.

        Args:
            query: Original user query
            iteration_content: Content from current iteration
            previous_carryover: Previous carryover (optional)
            temperature: Sampling temperature (lower = more focused)

        Returns:
            Compressed carryover content

        Raises:
            RuntimeError: If generation fails and fallback is needed
        """
        logger.debug(
            "carryover_generation_started",
            iteration_content_length=len(iteration_content),
            has_previous=previous_carryover is not None,
        )

        # Build prompt
        prompt = self._build_carryover_prompt(query, iteration_content, previous_carryover)
        prompt_tokens = self.llm_client.count_tokens(prompt)

        # Calculate max tokens for carryover
        # Reserve some buffer for JSON formatting
        max_tokens = min(self.max_carryover_tokens, 2048)

        try:
            # Generate carryover
            generation = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Parse JSON response
            carryover = self._parse_carryover_json(generation.content)

            if carryover is None:
                logger.warning(
                    "carryover_parse_failed_using_fallback",
                    generation_content_preview=generation.content[:200],
                )
                return self._create_fallback_carryover(iteration_content, previous_carryover)

            # Validate carryover size
            carryover_text = carryover.to_text()
            carryover_tokens = self.llm_client.count_tokens(carryover_text)

            if carryover_tokens > self.max_carryover_tokens:
                logger.warning(
                    "carryover_exceeds_limit",
                    carryover_tokens=carryover_tokens,
                    max_tokens=self.max_carryover_tokens,
                )
                # Trim key_findings and next_steps to fit
                while carryover_tokens > self.max_carryover_tokens and (carryover.key_findings or carryover.next_steps):
                    if carryover.key_findings:
                        carryover.key_findings.pop()
                    if carryover_tokens > self.max_carryover_tokens and carryover.next_steps:
                        carryover.next_steps.pop()
                    carryover_text = carryover.to_text()
                    carryover_tokens = self.llm_client.count_tokens(carryover_text)

            logger.info(
                "carryover_generated",
                carryover_tokens=carryover_tokens,
                key_findings_count=len(carryover.key_findings),
                next_steps_count=len(carryover.next_steps),
                unresolved_count=len(carryover.unresolved),
            )

            return carryover

        except Exception as e:
            logger.error(
                "carryover_generation_failed",
                error=str(e),
            )
            # Return fallback carryover
            return self._create_fallback_carryover(iteration_content, previous_carryover)
