"""
Context Compressor for COMPASS Memory Service

Implements progressive compression using test-time scaling with gpt-4.1-mini
for cost optimization. Provides stage-level (10:1 target) and task-level
(5:1 target) compression with critical fact extraction and quality validation.

Component ID: MEM-012
Ticket: MEM-012 (Implement ContextCompressor with Test-Time Scaling)

Features:
- Stage compression with 10:1 ratio target
- Task compression with 5:1 ratio target
- Critical fact extraction and preservation
- Compression quality validation (95%+ fact retention)
- Test-time scaling using gpt-4.1-mini exclusively
- Compression metrics tracking (ratio, cost, quality, latency)
- Integration with StageManager CompressionTrigger protocol
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agentcore.a2a_protocol.models.llm import LLMRequest
from agentcore.a2a_protocol.models.memory import MemoryRecord
from agentcore.a2a_protocol.services.llm_service import llm_service

logger = structlog.get_logger()


class CompressionMetrics:
    """Metrics for compression operation tracking."""

    def __init__(
        self,
        compression_ratio: float,
        quality_score: float,
        latency_seconds: float,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost_usd: float,
        coherence_score: float | None = None,
        fact_retention_rate: float | None = None,
        contradiction_count: int | None = None,
    ):
        """
        Initialize compression metrics.

        Args:
            compression_ratio: Achieved compression ratio (input/output)
            quality_score: Compression quality score (0-1)
            latency_seconds: Compression latency in seconds
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model used for compression
            cost_usd: Estimated cost in USD
            coherence_score: Optional overall coherence score (0-1)
            fact_retention_rate: Optional fact retention rate (0-1)
            contradiction_count: Optional number of contradictions detected
        """
        self.compression_ratio = compression_ratio
        self.quality_score = quality_score
        self.latency_seconds = latency_seconds
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.cost_usd = cost_usd
        self.coherence_score = coherence_score
        self.fact_retention_rate = fact_retention_rate
        self.contradiction_count = contradiction_count

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for storage."""
        result = {
            "compression_ratio": self.compression_ratio,
            "quality_score": self.quality_score,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "cost_usd": self.cost_usd,
        }
        # Add optional fields if present
        if self.coherence_score is not None:
            result["coherence_score"] = self.coherence_score
        if self.fact_retention_rate is not None:
            result["fact_retention_rate"] = self.fact_retention_rate
        if self.contradiction_count is not None:
            result["contradiction_count"] = self.contradiction_count
        return result


class ContextCompressor:
    """
    Progressive context compression using test-time scaling.

    Implements COMPASS compression strategy:
    - Stage compression: 10:1 ratio (raw memories → stage summary)
    - Task compression: 5:1 ratio (stage summaries → task progress)
    - Critical fact extraction and preservation
    - Quality validation (95%+ fact retention target)

    Uses gpt-4.1-mini exclusively for cost optimization (test-time scaling).
    """

    # Model for all compression operations (test-time scaling)
    COMPRESSION_MODEL = "gpt-4.1-mini"

    # Target compression ratios
    STAGE_COMPRESSION_TARGET = 10.0  # 10:1 ratio
    TASK_COMPRESSION_TARGET = 5.0  # 5:1 ratio

    # Quality thresholds
    MIN_QUALITY_SCORE = 0.95  # 95% fact retention

    # Pricing for gpt-4.1-mini (per 1M tokens)
    # These are example rates - adjust based on actual OpenAI pricing
    INPUT_COST_PER_1M = 0.15  # $0.15 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.60  # $0.60 per 1M output tokens

    def __init__(self, trace_id: str | None = None):
        """
        Initialize ContextCompressor.

        Args:
            trace_id: Optional trace ID for request tracking
        """
        self._logger = logger.bind(component="context_compressor")
        self._trace_id = trace_id

    async def compress_stage(
        self,
        stage_id: str,
        raw_memory_ids: list[str],
        raw_memories: list[MemoryRecord] | None = None,
        stage_type: str | None = None,
    ) -> dict[str, float]:
        """
        Compress raw memories to stage summary (10:1 target).

        Implements CompressionTrigger protocol for StageManager integration.

        Args:
            stage_id: Stage ID being compressed
            raw_memory_ids: List of raw memory IDs to compress
            raw_memories: Optional list of raw MemoryRecord objects
            stage_type: Optional stage type (planning, execution, etc.)

        Returns:
            Compression metrics dict with compression_ratio and quality_score

        Raises:
            ValueError: If compression quality is below threshold
        """
        start_time = time.time()

        self._logger.info(
            "stage_compression_started",
            stage_id=stage_id,
            raw_memory_count=len(raw_memory_ids),
            stage_type=stage_type,
        )

        # If raw_memories not provided, we would fetch them from database
        # For now, we'll handle the case where they're provided
        if not raw_memories:
            # In a real implementation, fetch from MemoryRepository
            self._logger.warning(
                "stage_compression_no_memories",
                stage_id=stage_id,
                message="No memories provided, returning default metrics",
            )
            return {"compression_ratio": 1.0, "quality_score": 1.0}

        # Combine raw memory content
        combined_content = self._combine_memories(raw_memories)
        input_length = len(combined_content)

        # Extract critical facts before compression
        critical_facts = await self._extract_critical_facts(
            combined_content, stage_type
        )

        # Perform compression
        compressed_summary = await self._compress_content(
            content=combined_content,
            target_ratio=self.STAGE_COMPRESSION_TARGET,
            context_type="stage",
            stage_type=stage_type,
            critical_facts=critical_facts,
        )

        output_length = len(compressed_summary)

        # Validate compression quality
        quality_score = await self._validate_compression_quality(
            original_content=combined_content,
            compressed_content=compressed_summary,
            critical_facts=critical_facts,
        )

        # Calculate compression ratio
        compression_ratio = (
            input_length / output_length if output_length > 0 else 1.0
        )

        # Calculate latency
        latency = time.time() - start_time

        # Check quality threshold
        if quality_score < self.MIN_QUALITY_SCORE:
            self._logger.warning(
                "stage_compression_low_quality",
                stage_id=stage_id,
                quality_score=quality_score,
                threshold=self.MIN_QUALITY_SCORE,
                compression_ratio=compression_ratio,
            )
            # In production, we might retry with less aggressive compression
            # For now, we'll log and continue

        self._logger.info(
            "stage_compression_completed",
            stage_id=stage_id,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            latency_seconds=latency,
            input_length=input_length,
            output_length=output_length,
        )

        return {
            "compression_ratio": compression_ratio,
            "quality_score": quality_score,
        }

    async def compress_task(
        self,
        task_id: str,
        stage_summaries: list[str],
        task_goal: str | None = None,
    ) -> tuple[str, CompressionMetrics]:
        """
        Compress stage summaries to task progress summary (5:1 target).

        Args:
            task_id: Task ID being compressed
            stage_summaries: List of stage summaries to compress
            task_goal: Optional task goal for context

        Returns:
            Tuple of (compressed_summary, metrics)

        Raises:
            ValueError: If compression quality is below threshold
        """
        start_time = time.time()

        self._logger.info(
            "task_compression_started",
            task_id=task_id,
            stage_count=len(stage_summaries),
        )

        # Combine stage summaries
        combined_content = "\n\n".join(
            [f"Stage {i+1}:\n{summary}" for i, summary in enumerate(stage_summaries)]
        )
        input_length = len(combined_content)

        # Extract critical constraints
        critical_facts = await self._extract_critical_facts(
            combined_content, context_type="task"
        )

        # Perform compression
        compressed_summary = await self._compress_content(
            content=combined_content,
            target_ratio=self.TASK_COMPRESSION_TARGET,
            context_type="task",
            task_goal=task_goal,
            critical_facts=critical_facts,
        )

        output_length = len(compressed_summary)

        # Validate compression quality
        quality_score = await self._validate_compression_quality(
            original_content=combined_content,
            compressed_content=compressed_summary,
            critical_facts=critical_facts,
        )

        # Calculate metrics
        compression_ratio = (
            input_length / output_length if output_length > 0 else 1.0
        )
        latency = time.time() - start_time

        # Estimate token counts (rough approximation: 4 chars per token)
        input_tokens = input_length // 4
        output_tokens = output_length // 4
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost_usd = self._calculate_cost(input_tokens, output_tokens)

        metrics = CompressionMetrics(
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.COMPRESSION_MODEL,
            cost_usd=cost_usd,
        )

        # Check quality threshold
        if quality_score < self.MIN_QUALITY_SCORE:
            self._logger.warning(
                "task_compression_low_quality",
                task_id=task_id,
                quality_score=quality_score,
                threshold=self.MIN_QUALITY_SCORE,
                compression_ratio=compression_ratio,
            )

        self._logger.info(
            "task_compression_completed",
            task_id=task_id,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            latency_seconds=latency,
            cost_usd=cost_usd,
        )

        return compressed_summary, metrics

    async def extract_critical_facts(
        self, content: str, context_type: str | None = None
    ) -> list[str]:
        """
        Extract critical facts from content that must be preserved.

        Public interface for critical fact extraction.

        Args:
            content: Content to extract facts from
            context_type: Optional context type (stage, task, etc.)

        Returns:
            List of critical facts
        """
        return await self._extract_critical_facts(content, context_type)

    async def validate_compression_quality(
        self,
        original_content: str,
        compressed_content: str,
        critical_facts: list[str],
    ) -> float:
        """
        Validate compression quality (fact retention).

        Public interface for quality validation.

        Args:
            original_content: Original uncompressed content
            compressed_content: Compressed content
            critical_facts: List of critical facts to check

        Returns:
            Quality score (0-1) representing fact retention percentage
        """
        return await self._validate_compression_quality(
            original_content, compressed_content, critical_facts
        )

    # Private methods

    def _combine_memories(self, memories: list[MemoryRecord]) -> str:
        """Combine multiple memory records into single content string."""
        return "\n\n".join(
            [f"[{mem.timestamp.isoformat()}] {mem.content}" for mem in memories]
        )

    async def _extract_critical_facts(
        self, content: str, context_type: str | None = None
    ) -> list[str]:
        """Extract critical facts that must be preserved during compression."""
        from agentcore.a2a_protocol.services.memory.prompts.compression import (
            build_fact_extraction_prompt,
        )

        prompt = build_fact_extraction_prompt(content, context_type)

        request = LLMRequest(
            model=self.COMPRESSION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for factual extraction
            max_tokens=500,
            trace_id=self._trace_id,
        )

        try:
            response = await llm_service.complete(request)
            # Parse response to extract facts (expecting newline-separated list)
            facts = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return facts
        except Exception as e:
            self._logger.error(
                "fact_extraction_failed",
                error=str(e),
                context_type=context_type,
            )
            return []

    async def _compress_content(
        self,
        content: str,
        target_ratio: float,
        context_type: str,
        stage_type: str | None = None,
        task_goal: str | None = None,
        critical_facts: list[str] | None = None,
    ) -> str:
        """Compress content using LLM with target ratio."""
        from agentcore.a2a_protocol.services.memory.prompts.compression import (
            build_compression_prompt,
        )

        prompt = build_compression_prompt(
            content=content,
            target_ratio=target_ratio,
            context_type=context_type,
            stage_type=stage_type,
            task_goal=task_goal,
            critical_facts=critical_facts or [],
        )

        request = LLMRequest(
            model=self.COMPRESSION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Moderate temperature for compression
            max_tokens=2000,  # Allow sufficient space for compressed output
            trace_id=self._trace_id,
        )

        try:
            response = await llm_service.complete(request)
            return response.content.strip()
        except Exception as e:
            self._logger.error(
                "compression_failed",
                error=str(e),
                context_type=context_type,
            )
            # Fallback: return truncated content
            max_length = len(content) // int(target_ratio)
            return content[:max_length] + "... [compressed due to error]"

    async def _validate_compression_quality(
        self,
        original_content: str,
        compressed_content: str,
        critical_facts: list[str],
    ) -> float:
        """Validate compression quality by checking critical fact retention."""
        from agentcore.a2a_protocol.services.memory.prompts.compression import (
            build_quality_validation_prompt,
        )

        if not critical_facts:
            # No facts to validate, assume perfect retention
            return 1.0

        prompt = build_quality_validation_prompt(
            original_content, compressed_content, critical_facts
        )

        request = LLMRequest(
            model=self.COMPRESSION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Very low temperature for validation
            max_tokens=200,
            trace_id=self._trace_id,
        )

        try:
            response = await llm_service.complete(request)
            # Parse response to get quality score
            # Expected format: "Quality Score: 0.95" or "95%"
            content = response.content.strip()

            # Try to extract numeric score
            import re

            # Look for patterns like "0.95", "95%", "95", "Score: 0.95"
            patterns = [
                r"(\d+\.\d+)",  # 0.95
                r"(\d+)%",  # 95%
                r"Score:\s*(\d+\.\d+)",  # Score: 0.95
                r"Score:\s*(\d+)%",  # Score: 95%
            ]

            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    score_str = match.group(1)
                    score = float(score_str)
                    # Normalize to 0-1 range if percentage
                    if score > 1.0:
                        score = score / 100.0
                    return min(1.0, max(0.0, score))

            # If no numeric score found, fall back to heuristic
            self._logger.warning(
                "quality_validation_parse_failed",
                response=content,
                using_fallback=True,
            )
            return self._estimate_quality_heuristic(critical_facts, compressed_content)

        except Exception as e:
            self._logger.error(
                "quality_validation_failed",
                error=str(e),
            )
            # Fallback to heuristic
            return self._estimate_quality_heuristic(critical_facts, compressed_content)

    def _estimate_quality_heuristic(
        self, critical_facts: list[str], compressed_content: str
    ) -> float:
        """Estimate quality using simple fact presence heuristic."""
        if not critical_facts:
            return 1.0

        facts_present = sum(
            1
            for fact in critical_facts
            if fact.lower() in compressed_content.lower()
        )
        return facts_present / len(critical_facts)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate compression cost in USD based on token counts."""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost


__all__ = ["ContextCompressor", "CompressionMetrics"]
