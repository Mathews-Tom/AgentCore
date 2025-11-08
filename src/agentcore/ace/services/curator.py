"""
Simple Curator Service

Filters context deltas based on confidence threshold.
Phase 1 implementation using confidence-based filtering instead of full reflection loop.
"""

from __future__ import annotations

import structlog

from agentcore.ace.models.ace_models import ContextDelta

logger = structlog.get_logger()


class SimpleCurator:
    """
    Simple delta curator with confidence-threshold filtering.

    Phase 1 implementation: filters deltas based on confidence scores.
    Phase 2 may include advanced curation strategies (reflection, validation).
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """
        Initialize Simple Curator.

        Args:
            threshold: Minimum confidence score for delta approval (0.0-1.0)

        Raises:
            ValueError: If threshold is not in range [0.0, 1.0]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        self.threshold = threshold
        logger.info("SimpleCurator initialized", threshold=threshold)

    def filter_deltas(
        self, deltas: list[ContextDelta], threshold: float | None = None
    ) -> list[ContextDelta]:
        """
        Filter deltas based on confidence threshold.

        Args:
            deltas: List of ContextDelta objects to filter
            threshold: Optional override for confidence threshold (uses instance default if None)

        Returns:
            List of approved deltas with confidence >= threshold
        """
        if not deltas:
            logger.info("No deltas to filter")
            return []

        # Use provided threshold or fall back to instance threshold
        confidence_threshold = threshold if threshold is not None else self.threshold

        approved_deltas: list[ContextDelta] = []
        rejected_count = 0

        for delta in deltas:
            if delta.confidence >= confidence_threshold:
                approved_deltas.append(delta)
            else:
                rejected_count += 1
                logger.info(
                    "Delta rejected - confidence below threshold",
                    delta_id=str(delta.delta_id),
                    playbook_id=str(delta.playbook_id),
                    confidence_score=delta.confidence,
                    threshold=confidence_threshold,
                    rationale=f"Confidence {delta.confidence:.3f} < threshold {confidence_threshold:.3f}",
                    reasoning=delta.reasoning[:100] if len(delta.reasoning) > 100 else delta.reasoning,
                )

        logger.info(
            "Delta filtering completed",
            total_deltas=len(deltas),
            approved_count=len(approved_deltas),
            rejected_count=rejected_count,
            threshold=confidence_threshold,
        )

        return approved_deltas
