"""
Error Accumulator (COMPASS ACE-1 - ACE-011)

Error accumulation tracking and pattern detection.
Critical for achieving COMPASS target of 90%+ critical error recall.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger()

# Constants
VALID_STAGES = {"planning", "execution", "reflection", "verification"}
COMPOUNDING_ERROR_THRESHOLD = 3  # 3+ errors trigger compounding detection
RELATED_ERROR_WINDOW = 5  # Check for related errors within N steps
PERFORMANCE_TARGET_MS = 50  # <50ms computation target


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorRecord:
    """Individual error record with metadata."""

    __slots__ = (
        "error_id",
        "agent_id",
        "task_id",
        "stage",
        "error_type",
        "severity",
        "error_message",
        "timestamp",
        "metadata",
    )

    def __init__(
        self,
        error_id: int,
        agent_id: str,
        task_id: UUID,
        stage: str,
        error_type: str,
        severity: ErrorSeverity,
        error_message: str,
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize ErrorRecord.

        Args:
            error_id: Sequential error ID within accumulator
            agent_id: Agent identifier
            task_id: Task identifier
            stage: Reasoning stage
            error_type: Error type/category
            severity: Error severity level
            error_message: Error message
            timestamp: Error timestamp
            metadata: Optional metadata
        """
        self.error_id = error_id
        self.agent_id = agent_id
        self.task_id = task_id
        self.stage = stage
        self.error_type = error_type
        self.severity = severity
        self.error_message = error_message
        self.timestamp = timestamp
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"ErrorRecord(id={self.error_id}, type={self.error_type}, "
            f"severity={self.severity.value}, stage={self.stage})"
        )


class ErrorPattern:
    """Detected error pattern with metadata."""

    __slots__ = ("pattern_type", "error_records", "detected_at", "metadata")

    def __init__(
        self,
        pattern_type: str,
        error_records: list[ErrorRecord],
        detected_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize ErrorPattern.

        Args:
            pattern_type: Pattern type (sequential, cascading, compounding)
            error_records: Errors involved in pattern
            detected_at: Pattern detection timestamp
            metadata: Optional metadata
        """
        self.pattern_type = pattern_type
        self.error_records = error_records
        self.detected_at = detected_at
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"ErrorPattern(type={self.pattern_type}, "
            f"errors={len(self.error_records)})"
        )


class ErrorAccumulator:
    """
    Error accumulator with pattern detection (COMPASS ACE-1 - ACE-011).

    Features:
    - Track errors per stage with severity distribution
    - Error count per stage and overall
    - Compounding error detection (related errors in sequence)
    - Error pattern analysis (sequential, cascading)
    - <50ms computation target (performance critical)
    - Integration placeholder for MEM error pattern detection (Phase 4)

    Performance targets:
    - Error tracking: <10ms (p95)
    - Pattern detection: <50ms (p95) - CRITICAL
    - Error accumulation computation: <50ms (p95)
    """

    def __init__(self) -> None:
        """Initialize ErrorAccumulator."""
        # Error storage: (agent_id, task_id) -> list[ErrorRecord]
        self._errors: dict[tuple[str, UUID], list[ErrorRecord]] = defaultdict(list)

        # Error ID counter for sequential tracking
        self._error_counter: dict[tuple[str, UUID], int] = defaultdict(int)

        # Detected patterns: (agent_id, task_id) -> list[ErrorPattern]
        self._patterns: dict[tuple[str, UUID], list[ErrorPattern]] = defaultdict(list)

        logger.info("ErrorAccumulator initialized")

    def track_error(
        self,
        agent_id: str,
        task_id: UUID,
        stage: str,
        error_type: str,
        severity: ErrorSeverity,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> ErrorRecord:
        """
        Track error occurrence.

        Validates stage, records error, and triggers pattern detection.
        Meets <50ms performance target through optimized in-memory operations.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            stage: Reasoning stage (planning, execution, reflection, verification)
            error_type: Error type/category
            severity: Error severity level
            error_message: Error message
            metadata: Optional metadata

        Returns:
            ErrorRecord instance

        Raises:
            ValueError: If stage is invalid
        """
        # Validate stage (fail-fast)
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        # Create error record
        key = (agent_id, task_id)
        error_id = self._error_counter[key]
        self._error_counter[key] += 1

        error_record = ErrorRecord(
            error_id=error_id,
            agent_id=agent_id,
            task_id=task_id,
            stage=stage,
            error_type=error_type,
            severity=severity,
            error_message=error_message,
            timestamp=datetime.now(UTC),
            metadata=metadata,
        )

        # Store error
        self._errors[key].append(error_record)

        logger.debug(
            "Error tracked",
            agent_id=agent_id,
            task_id=str(task_id),
            stage=stage,
            error_type=error_type,
            severity=severity.value,
            error_id=error_id,
        )

        # Trigger pattern detection after adding error
        self._detect_patterns(agent_id, task_id)

        return error_record

    def get_error_count(
        self,
        agent_id: str,
        task_id: UUID,
        stage: str | None = None,
    ) -> int:
        """
        Get error count for task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            stage: Optional stage filter (if None, returns total count)

        Returns:
            Error count

        Raises:
            ValueError: If stage is provided and invalid
        """
        # Validate stage if provided
        if stage is not None and stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        key = (agent_id, task_id)
        errors = self._errors.get(key, [])

        if stage is None:
            return len(errors)

        # Filter by stage
        return sum(1 for e in errors if e.stage == stage)

    def get_severity_distribution(
        self,
        agent_id: str,
        task_id: UUID,
        stage: str | None = None,
    ) -> dict[ErrorSeverity, int]:
        """
        Get error severity distribution.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            stage: Optional stage filter

        Returns:
            Dict mapping severity to count

        Raises:
            ValueError: If stage is provided and invalid
        """
        # Validate stage if provided
        if stage is not None and stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        key = (agent_id, task_id)
        errors = self._errors.get(key, [])

        # Filter by stage if provided
        if stage is not None:
            errors = [e for e in errors if e.stage == stage]

        # Compute distribution
        distribution: dict[ErrorSeverity, int] = {
            ErrorSeverity.LOW: 0,
            ErrorSeverity.MEDIUM: 0,
            ErrorSeverity.HIGH: 0,
            ErrorSeverity.CRITICAL: 0,
        }

        for error in errors:
            distribution[error.severity] += 1

        return distribution

    def get_error_rate_per_stage(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> dict[str, int]:
        """
        Get error count per stage.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Dict mapping stage to error count
        """
        key = (agent_id, task_id)
        errors = self._errors.get(key, [])

        # Compute error count per stage
        stage_counts: dict[str, int] = {stage: 0 for stage in VALID_STAGES}

        for error in errors:
            stage_counts[error.stage] += 1

        return stage_counts

    def detect_compounding_errors(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> list[ErrorPattern]:
        """
        Detect compounding error patterns.

        Detects:
        1. Sequential errors of same type (within RELATED_ERROR_WINDOW)
        2. Error cascades (error in one stage triggers error in next)
        3. Compounding errors (3+ errors in single stage)

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            List of detected error patterns
        """
        key = (agent_id, task_id)
        return self._patterns.get(key, [])

    def _detect_patterns(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> None:
        """
        Detect error patterns after new error is added.

        Internal method that runs pattern detection algorithms:
        1. Sequential same-type errors
        2. Error cascades across stages
        3. Compounding errors in single stage

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
        """
        key = (agent_id, task_id)
        errors = self._errors.get(key, [])

        if len(errors) < 2:
            # Need at least 2 errors for pattern detection
            return

        # Pattern 1: Sequential same-type errors (within window)
        self._detect_sequential_errors(agent_id, task_id, errors)

        # Pattern 2: Error cascades (error in stage N -> error in stage N+1)
        self._detect_cascading_errors(agent_id, task_id, errors)

        # Pattern 3: Compounding errors (3+ errors in single stage)
        self._detect_stage_compounding(agent_id, task_id, errors)

    def _detect_sequential_errors(
        self,
        agent_id: str,
        task_id: UUID,
        errors: list[ErrorRecord],
    ) -> None:
        """
        Detect sequential errors of same type within window.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            errors: List of error records
        """
        key = (agent_id, task_id)

        # Check last N errors for same type
        if len(errors) < 2:
            return

        recent_errors = errors[-RELATED_ERROR_WINDOW:]
        error_types = [e.error_type for e in recent_errors]

        # Find repeated error types (2+ occurrences)
        type_counts: dict[str, list[ErrorRecord]] = defaultdict(list)
        for error in recent_errors:
            type_counts[error.error_type].append(error)

        for error_type, error_list in type_counts.items():
            if len(error_list) >= 2:
                # Sequential error pattern detected
                pattern = ErrorPattern(
                    pattern_type="sequential",
                    error_records=error_list,
                    detected_at=datetime.now(UTC),
                    metadata={
                        "error_type": error_type,
                        "occurrence_count": len(error_list),
                        "window_size": RELATED_ERROR_WINDOW,
                    },
                )

                # Check if pattern already exists
                if not self._pattern_exists(key, pattern):
                    self._patterns[key].append(pattern)

                    logger.warning(
                        "Sequential error pattern detected",
                        agent_id=agent_id,
                        task_id=str(task_id),
                        error_type=error_type,
                        occurrences=len(error_list),
                    )

    def _detect_cascading_errors(
        self,
        agent_id: str,
        task_id: UUID,
        errors: list[ErrorRecord],
    ) -> None:
        """
        Detect cascading errors (error in stage N -> error in stage N+1).

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            errors: List of error records
        """
        key = (agent_id, task_id)

        # Define stage order
        stage_order = ["planning", "execution", "reflection", "verification"]

        # Check for consecutive stage errors
        for i in range(len(errors) - 1):
            current_error = errors[i]
            next_error = errors[i + 1]

            # Check if stages are consecutive
            try:
                current_idx = stage_order.index(current_error.stage)
                next_idx = stage_order.index(next_error.stage)

                if next_idx == current_idx + 1:
                    # Cascading error detected
                    pattern = ErrorPattern(
                        pattern_type="cascading",
                        error_records=[current_error, next_error],
                        detected_at=datetime.now(UTC),
                        metadata={
                            "from_stage": current_error.stage,
                            "to_stage": next_error.stage,
                            "from_error_type": current_error.error_type,
                            "to_error_type": next_error.error_type,
                        },
                    )

                    # Check if pattern already exists
                    if not self._pattern_exists(key, pattern):
                        self._patterns[key].append(pattern)

                        logger.warning(
                            "Cascading error pattern detected",
                            agent_id=agent_id,
                            task_id=str(task_id),
                            from_stage=current_error.stage,
                            to_stage=next_error.stage,
                        )
            except ValueError:
                # Stage not in order list - skip
                continue

    def _detect_stage_compounding(
        self,
        agent_id: str,
        task_id: UUID,
        errors: list[ErrorRecord],
    ) -> None:
        """
        Detect compounding errors (3+ errors in single stage).

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            errors: List of error records
        """
        key = (agent_id, task_id)

        # Count errors per stage
        stage_errors: dict[str, list[ErrorRecord]] = defaultdict(list)
        for error in errors:
            stage_errors[error.stage].append(error)

        # Check for compounding (3+ errors)
        for stage, error_list in stage_errors.items():
            if len(error_list) >= COMPOUNDING_ERROR_THRESHOLD:
                # Compounding error pattern detected
                pattern = ErrorPattern(
                    pattern_type="compounding",
                    error_records=error_list,
                    detected_at=datetime.now(UTC),
                    metadata={
                        "stage": stage,
                        "error_count": len(error_list),
                        "threshold": COMPOUNDING_ERROR_THRESHOLD,
                    },
                )

                # Check if pattern already exists
                if not self._pattern_exists(key, pattern):
                    self._patterns[key].append(pattern)

                    logger.warning(
                        "Compounding error pattern detected",
                        agent_id=agent_id,
                        task_id=str(task_id),
                        stage=stage,
                        error_count=len(error_list),
                    )

    def _pattern_exists(
        self,
        key: tuple[str, UUID],
        pattern: ErrorPattern,
    ) -> bool:
        """
        Check if pattern already exists.

        Args:
            key: (agent_id, task_id) tuple
            pattern: ErrorPattern to check

        Returns:
            True if pattern exists, False otherwise
        """
        existing_patterns = self._patterns.get(key, [])

        for existing in existing_patterns:
            if existing.pattern_type != pattern.pattern_type:
                continue

            # Compare error IDs
            existing_ids = {e.error_id for e in existing.error_records}
            new_ids = {e.error_id for e in pattern.error_records}

            if existing_ids == new_ids:
                return True

        return False

    def get_error_trends(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> dict[str, Any]:
        """
        Get error trends and analytics.

        Provides data for monitoring dashboard visualization.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Dict containing error trends:
                - total_errors: Total error count
                - errors_per_stage: Error count per stage
                - severity_distribution: Error severity distribution
                - detected_patterns: Detected error patterns
                - critical_error_count: Count of critical errors
                - error_types: Error type distribution
        """
        key = (agent_id, task_id)
        errors = self._errors.get(key, [])

        # Total errors
        total_errors = len(errors)

        # Errors per stage
        errors_per_stage = self.get_error_rate_per_stage(agent_id, task_id)

        # Severity distribution
        severity_distribution = self.get_severity_distribution(agent_id, task_id)

        # Detected patterns
        patterns = self._patterns.get(key, [])

        # Critical error count
        critical_error_count = severity_distribution.get(ErrorSeverity.CRITICAL, 0)

        # Error type distribution
        error_type_counts: dict[str, int] = defaultdict(int)
        for error in errors:
            error_type_counts[error.error_type] += 1

        return {
            "total_errors": total_errors,
            "errors_per_stage": errors_per_stage,
            "severity_distribution": {
                k.value: v for k, v in severity_distribution.items()
            },
            "detected_patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "error_count": len(p.error_records),
                    "detected_at": p.detected_at.isoformat(),
                    "metadata": p.metadata,
                }
                for p in patterns
            ],
            "critical_error_count": critical_error_count,
            "error_types": dict(error_type_counts),
        }

    def query_mem_error_patterns(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> dict[str, Any]:
        """
        Query MEM for error pattern detection.

        PLACEHOLDER: Integration with MEM (Memory System) for Phase 4.
        Currently returns stub response.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Dict with MEM query results (stub)
        """
        logger.debug(
            "MEM error pattern query (stub - not implemented)",
            agent_id=agent_id,
            task_id=str(task_id),
        )

        # Stub: Will be implemented in Phase 4 with MEM integration
        return {
            "status": "not_implemented",
            "message": "MEM integration pending (Phase 4)",
            "agent_id": agent_id,
            "task_id": str(task_id),
        }

    def reset_errors(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> None:
        """
        Reset error accumulation for task.

        Clears all errors and patterns for the given agent/task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
        """
        key = (agent_id, task_id)

        # Clear errors
        if key in self._errors:
            del self._errors[key]

        # Clear patterns
        if key in self._patterns:
            del self._patterns[key]

        # Reset counter
        if key in self._error_counter:
            del self._error_counter[key]

        logger.info(
            "Error accumulation reset",
            agent_id=agent_id,
            task_id=str(task_id),
        )

    def get_all_errors(
        self,
        agent_id: str,
        task_id: UUID,
    ) -> list[ErrorRecord]:
        """
        Get all errors for task.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            List of all error records
        """
        key = (agent_id, task_id)
        return self._errors.get(key, []).copy()
