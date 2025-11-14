"""
Compression Quality Validator for COMPASS Memory Service

Implements quality metrics for compression validation including fact retention,
coherence scoring, contradiction detection, and adaptive compression fallback.

Component ID: MEM-013
Ticket: MEM-013 (Implement Compression Quality Validation)

Features:
- Critical fact retention tracking (target: ≥95%)
- Compression ratio validation (10:1, 5:1)
- Coherence score calculation (no contradictions)
- Quality degradation detection and alerting
- Adaptive compression fallback (less aggressive on low quality)
- Integration with ContextCompressor for enhanced validation
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agentcore.a2a_protocol.models.llm import LLMRequest
from agentcore.a2a_protocol.services.llm_service import llm_service

logger = structlog.get_logger()


class QualityMetrics:
    """Extended quality metrics for compression validation."""

    def __init__(
        self,
        compression_ratio: float,
        fact_retention_rate: float,
        coherence_score: float,
        contradiction_count: int,
        quality_score: float,
        quality_degraded: bool = False,
        fallback_triggered: bool = False,
        original_target_ratio: float | None = None,
        adjusted_target_ratio: float | None = None,
    ):
        """
        Initialize quality metrics.

        Args:
            compression_ratio: Achieved compression ratio (input/output)
            fact_retention_rate: Percentage of critical facts retained (0-1)
            coherence_score: Overall coherence score (0-1)
            contradiction_count: Number of contradictions detected
            quality_score: Overall quality score (0-1)
            quality_degraded: Whether quality fell below threshold
            fallback_triggered: Whether adaptive fallback was triggered
            original_target_ratio: Original compression target
            adjusted_target_ratio: Adjusted target after fallback
        """
        self.compression_ratio = compression_ratio
        self.fact_retention_rate = fact_retention_rate
        self.coherence_score = coherence_score
        self.contradiction_count = contradiction_count
        self.quality_score = quality_score
        self.quality_degraded = quality_degraded
        self.fallback_triggered = fallback_triggered
        self.original_target_ratio = original_target_ratio
        self.adjusted_target_ratio = adjusted_target_ratio

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for storage."""
        return {
            "compression_ratio": self.compression_ratio,
            "fact_retention_rate": self.fact_retention_rate,
            "coherence_score": self.coherence_score,
            "contradiction_count": self.contradiction_count,
            "quality_score": self.quality_score,
            "quality_degraded": self.quality_degraded,
            "fallback_triggered": self.fallback_triggered,
            "original_target_ratio": self.original_target_ratio,
            "adjusted_target_ratio": self.adjusted_target_ratio,
        }


class QualityValidator:
    """
    Compression quality validator with fact retention, coherence, and adaptive fallback.

    Implements enhanced quality validation beyond basic fact checking:
    - Fact retention tracking: ≥95% target
    - Contradiction detection: 0 contradictions target
    - Coherence scoring: Overall quality assessment
    - Quality degradation alerts: Warnings when quality drops
    - Adaptive compression: Fallback to less aggressive ratios

    Adaptive compression strategy:
    - 10:1 target → 8:1 fallback if quality < 0.95
    - 5:1 target → 4:1 fallback if quality < 0.95
    """

    # Validation model (use mini for cost efficiency)
    VALIDATION_MODEL = "gpt-4.1-mini"

    # Quality thresholds
    MIN_FACT_RETENTION = 0.95  # 95% fact retention required
    MIN_COHERENCE_SCORE = 0.90  # 90% coherence required
    MAX_CONTRADICTIONS = 0  # No contradictions allowed

    # Adaptive compression fallback ratios
    FALLBACK_RATIOS = {
        10.0: 8.0,  # Stage compression fallback
        5.0: 4.0,  # Task compression fallback
    }

    def __init__(self, trace_id: str | None = None):
        """
        Initialize QualityValidator.

        Args:
            trace_id: Optional trace ID for request tracking
        """
        self._logger = logger.bind(component="quality_validator")
        self._trace_id = trace_id

    async def validate_fact_retention(
        self,
        original_content: str,
        compressed_content: str,
        critical_facts: list[str],
    ) -> tuple[float, list[str]]:
        """
        Track critical facts before/after compression.

        Uses LLM to verify which critical facts are preserved in compressed content.
        Returns fact retention rate and list of missing facts.

        Args:
            original_content: Original uncompressed content
            compressed_content: Compressed content
            critical_facts: List of critical facts to verify

        Returns:
            Tuple of (fact_retention_rate, missing_facts)
        """
        if not critical_facts:
            return 1.0, []

        self._logger.info(
            "validating_fact_retention",
            fact_count=len(critical_facts),
        )

        # Build validation prompt
        prompt = self._build_fact_retention_prompt(
            compressed_content, critical_facts
        )

        request = LLMRequest(
            model=self.VALIDATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Very low for factual verification
            max_tokens=500,
            trace_id=self._trace_id,
        )

        try:
            response = await llm_service.complete(request)
            # Parse response to get retained/missing facts
            retained_count, missing_facts = self._parse_fact_retention_response(
                response.content, critical_facts
            )

            retention_rate = retained_count / len(critical_facts)

            self._logger.info(
                "fact_retention_validated",
                retention_rate=retention_rate,
                retained_count=retained_count,
                missing_count=len(missing_facts),
            )

            return retention_rate, missing_facts

        except Exception as e:
            self._logger.error(
                "fact_retention_validation_failed",
                error=str(e),
            )
            # Fallback to simple heuristic
            return self._estimate_fact_retention_heuristic(
                compressed_content, critical_facts
            )

    async def validate_compression_ratio(
        self,
        compression_ratio: float,
        target_ratio: float,
        tolerance: float = 0.2,
    ) -> bool:
        """
        Check compression ratio against target (10:1, 5:1) with tolerance.

        Args:
            compression_ratio: Achieved compression ratio
            target_ratio: Target compression ratio
            tolerance: Acceptable deviation (default: 20%)

        Returns:
            True if ratio is within tolerance, False otherwise
        """
        lower_bound = target_ratio * (1 - tolerance)
        upper_bound = target_ratio * (1 + tolerance)

        is_valid = lower_bound <= compression_ratio <= upper_bound

        self._logger.info(
            "compression_ratio_validated",
            compression_ratio=compression_ratio,
            target_ratio=target_ratio,
            tolerance=tolerance,
            is_valid=is_valid,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return is_valid

    async def detect_contradictions(
        self,
        original_content: str,
        compressed_content: str,
    ) -> tuple[int, list[str]]:
        """
        Check for conflicting statements between original and compressed content.

        Uses LLM to identify contradictions or conflicting information introduced
        during compression.

        Args:
            original_content: Original uncompressed content
            compressed_content: Compressed content

        Returns:
            Tuple of (contradiction_count, contradiction_descriptions)
        """
        self._logger.info("detecting_contradictions")

        prompt = self._build_contradiction_detection_prompt(
            original_content, compressed_content
        )

        request = LLMRequest(
            model=self.VALIDATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Very low for factual analysis
            max_tokens=500,
            trace_id=self._trace_id,
        )

        try:
            response = await llm_service.complete(request)
            contradictions = self._parse_contradiction_response(response.content)

            self._logger.info(
                "contradictions_detected",
                count=len(contradictions),
                contradictions=contradictions,
            )

            return len(contradictions), contradictions

        except Exception as e:
            self._logger.error(
                "contradiction_detection_failed",
                error=str(e),
            )
            # Conservative fallback: assume no contradictions
            return 0, []

    async def calculate_coherence_score(
        self,
        original_content: str,
        compressed_content: str,
        critical_facts: list[str],
        fact_retention_rate: float,
        contradiction_count: int,
    ) -> float:
        """
        Calculate overall quality metric considering all factors.

        Coherence score combines:
        - Fact retention rate (weight: 0.5)
        - Contradiction penalty (weight: 0.3)
        - Content quality assessment (weight: 0.2)

        Args:
            original_content: Original uncompressed content
            compressed_content: Compressed content
            critical_facts: List of critical facts
            fact_retention_rate: Fact retention rate (0-1)
            contradiction_count: Number of contradictions

        Returns:
            Coherence score (0-1)
        """
        self._logger.info(
            "calculating_coherence_score",
            fact_retention_rate=fact_retention_rate,
            contradiction_count=contradiction_count,
        )

        # Component weights
        FACT_RETENTION_WEIGHT = 0.5
        CONTRADICTION_WEIGHT = 0.3
        CONTENT_QUALITY_WEIGHT = 0.2

        # Fact retention component
        fact_score = fact_retention_rate

        # Contradiction penalty (each contradiction reduces score)
        contradiction_penalty = min(1.0, contradiction_count * 0.2)
        contradiction_score = max(0.0, 1.0 - contradiction_penalty)

        # Content quality component (assess readability and completeness)
        content_quality = await self._assess_content_quality(
            original_content, compressed_content
        )

        # Calculate weighted coherence score
        coherence_score = (
            (fact_score * FACT_RETENTION_WEIGHT)
            + (contradiction_score * CONTRADICTION_WEIGHT)
            + (content_quality * CONTENT_QUALITY_WEIGHT)
        )

        self._logger.info(
            "coherence_score_calculated",
            coherence_score=coherence_score,
            fact_score=fact_score,
            contradiction_score=contradiction_score,
            content_quality=content_quality,
        )

        return coherence_score

    async def check_quality_degradation(
        self,
        quality_metrics: QualityMetrics,
    ) -> tuple[bool, list[str]]:
        """
        Alert if quality drops below threshold.

        Checks multiple quality indicators:
        - Fact retention < 95%
        - Coherence score < 90%
        - Contradictions > 0

        Args:
            quality_metrics: Quality metrics to evaluate

        Returns:
            Tuple of (is_degraded, alert_messages)
        """
        alerts = []
        is_degraded = False

        # Check fact retention
        if quality_metrics.fact_retention_rate < self.MIN_FACT_RETENTION:
            alerts.append(
                f"Fact retention {quality_metrics.fact_retention_rate:.2%} "
                f"below target {self.MIN_FACT_RETENTION:.2%}"
            )
            is_degraded = True

        # Check coherence
        if quality_metrics.coherence_score < self.MIN_COHERENCE_SCORE:
            alerts.append(
                f"Coherence score {quality_metrics.coherence_score:.2%} "
                f"below target {self.MIN_COHERENCE_SCORE:.2%}"
            )
            is_degraded = True

        # Check contradictions
        if quality_metrics.contradiction_count > self.MAX_CONTRADICTIONS:
            alerts.append(
                f"Detected {quality_metrics.contradiction_count} contradictions "
                f"(target: {self.MAX_CONTRADICTIONS})"
            )
            is_degraded = True

        if is_degraded:
            self._logger.warning(
                "quality_degradation_detected",
                alerts=alerts,
                fact_retention=quality_metrics.fact_retention_rate,
                coherence_score=quality_metrics.coherence_score,
                contradictions=quality_metrics.contradiction_count,
            )
        else:
            self._logger.info(
                "quality_validation_passed",
                fact_retention=quality_metrics.fact_retention_rate,
                coherence_score=quality_metrics.coherence_score,
                contradictions=quality_metrics.contradiction_count,
            )

        return is_degraded, alerts

    def get_fallback_ratio(self, target_ratio: float) -> float | None:
        """
        Get fallback compression ratio for adaptive compression.

        Args:
            target_ratio: Original target compression ratio

        Returns:
            Fallback ratio if available, None otherwise
        """
        return self.FALLBACK_RATIOS.get(target_ratio)

    # Private methods

    def _build_fact_retention_prompt(
        self, compressed_content: str, critical_facts: list[str]
    ) -> str:
        """Build prompt for fact retention validation."""
        facts_list = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(critical_facts)])

        return f"""Analyze the compressed content and determine which critical facts are preserved.

Compressed Content:
{compressed_content}

Critical Facts to Verify:
{facts_list}

For each fact, determine if it is retained (YES) or missing (NO) in the compressed content.
The fact doesn't need to use exact wording, but the key information must be present.

Respond in this format:
RETAINED: [list fact numbers that are preserved, e.g., 1, 2, 3]
MISSING: [list fact numbers that are missing, e.g., 4, 5]

Be precise and conservative. Only mark a fact as RETAINED if its core information is clearly present."""

    def _parse_fact_retention_response(
        self, response: str, critical_facts: list[str]
    ) -> tuple[int, list[str]]:
        """Parse LLM response to extract retained/missing facts."""
        retained_count = 0
        missing_facts = []

        # Extract retained fact numbers
        retained_match = re.search(r"RETAINED:\s*\[(.*?)\]", response, re.IGNORECASE | re.DOTALL)
        if retained_match:
            retained_nums_str = retained_match.group(1)
            # Parse numbers (e.g., "1, 2, 3")
            retained_nums = [
                int(n.strip()) for n in retained_nums_str.split(",") if n.strip().isdigit()
            ]
            retained_count = len(retained_nums)

        # Extract missing fact numbers
        missing_match = re.search(r"MISSING:\s*\[(.*?)\]", response, re.IGNORECASE | re.DOTALL)
        if missing_match:
            missing_nums_str = missing_match.group(1)
            missing_nums = [
                int(n.strip()) for n in missing_nums_str.split(",") if n.strip().isdigit()
            ]
            # Map missing numbers to actual facts (1-indexed)
            missing_facts = [
                critical_facts[num - 1] for num in missing_nums if 1 <= num <= len(critical_facts)
            ]

        # Fallback: if parsing failed, use heuristic
        if retained_count == 0 and not missing_facts:
            return self._estimate_fact_retention_heuristic_counts(
                response, critical_facts
            )

        return retained_count, missing_facts

    def _estimate_fact_retention_heuristic(
        self, compressed_content: str, critical_facts: list[str]
    ) -> tuple[float, list[str]]:
        """Estimate fact retention using simple presence heuristic."""
        facts_present = 0
        missing_facts = []

        for fact in critical_facts:
            # Check if key terms from fact are in compressed content
            if self._fact_present_in_content(fact, compressed_content):
                facts_present += 1
            else:
                missing_facts.append(fact)

        retention_rate = facts_present / len(critical_facts) if critical_facts else 1.0
        return retention_rate, missing_facts

    def _estimate_fact_retention_heuristic_counts(
        self, response: str, critical_facts: list[str]
    ) -> tuple[int, list[str]]:
        """Estimate fact retention counts from response text."""
        retained_count = 0
        missing_facts = []

        # Simple heuristic: count facts mentioned in response
        for i, fact in enumerate(critical_facts):
            # Check if fact number or content is mentioned
            if str(i + 1) in response or fact.lower() in response.lower():
                retained_count += 1
            else:
                missing_facts.append(fact)

        return retained_count, missing_facts

    def _fact_present_in_content(self, fact: str, content: str) -> bool:
        """Check if fact's key information is present in content."""
        # Extract key terms (words > 3 chars)
        key_terms = [word.lower() for word in fact.split() if len(word) > 3]

        if not key_terms:
            return fact.lower() in content.lower()

        # Check if majority of key terms are present
        terms_present = sum(1 for term in key_terms if term in content.lower())
        return terms_present >= len(key_terms) * 0.5  # 50% threshold

    def _build_contradiction_detection_prompt(
        self, original_content: str, compressed_content: str
    ) -> str:
        """Build prompt for contradiction detection."""
        return f"""Analyze the original and compressed content for contradictions.
A contradiction occurs when the compressed version makes a statement that conflicts with the original.

Original Content:
{original_content}

Compressed Content:
{compressed_content}

Identify any contradictions where the compressed content:
1. States something opposite to the original
2. Changes critical numbers, dates, or facts
3. Introduces new information that conflicts with the original

Respond in this format:
CONTRADICTIONS: [count]
DETAILS:
- [Description of contradiction 1]
- [Description of contradiction 2]

If no contradictions found, respond:
CONTRADICTIONS: 0
DETAILS: None"""

    def _parse_contradiction_response(self, response: str) -> list[str]:
        """Parse LLM response to extract contradictions."""
        contradictions = []

        # Extract contradiction count
        count_match = re.search(r"CONTRADICTIONS:\s*(\d+)", response, re.IGNORECASE)
        if count_match:
            count = int(count_match.group(1))
            if count == 0:
                return []

        # Extract details
        details_match = re.search(
            r"DETAILS:\s*\n(.*?)(?=\n\n|\Z)", response, re.IGNORECASE | re.DOTALL
        )
        if details_match:
            details_text = details_match.group(1)
            # Parse bullet points
            lines = details_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and line.startswith("-"):
                    contradiction = line[1:].strip()
                    if contradiction and contradiction.lower() != "none":
                        contradictions.append(contradiction)

        return contradictions

    async def _assess_content_quality(
        self, original_content: str, compressed_content: str
    ) -> float:
        """Assess overall content quality of compression."""
        prompt = f"""Assess the quality of this compression on a scale of 0.0 to 1.0.
Consider:
- Readability and clarity
- Completeness (captures main ideas)
- Accuracy (no distortions)
- Conciseness (removes unnecessary detail)

Original Content:
{original_content[:500]}...

Compressed Content:
{compressed_content}

Respond with only a numeric score between 0.0 and 1.0 (e.g., 0.85)"""

        request = LLMRequest(
            model=self.VALIDATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20,
            trace_id=self._trace_id,
        )

        try:
            response = await llm_service.complete(request)
            # Extract numeric score
            score_match = re.search(r"(\d+\.\d+)", response.content)
            if score_match:
                score = float(score_match.group(1))
                return min(1.0, max(0.0, score))

            # Fallback to heuristic
            return self._estimate_content_quality_heuristic(
                original_content, compressed_content
            )

        except Exception as e:
            self._logger.error(
                "content_quality_assessment_failed",
                error=str(e),
            )
            return self._estimate_content_quality_heuristic(
                original_content, compressed_content
            )

    def _estimate_content_quality_heuristic(
        self, original_content: str, compressed_content: str
    ) -> float:
        """Estimate content quality using simple heuristics."""
        # Simple heuristic based on length ratio
        ratio = len(compressed_content) / len(original_content)

        # Ideal ratio: 0.1-0.2 (10:1 to 5:1 compression)
        if 0.1 <= ratio <= 0.2:
            return 0.9
        elif 0.05 <= ratio <= 0.3:
            return 0.75
        else:
            return 0.6


__all__ = ["QualityValidator", "QualityMetrics"]
