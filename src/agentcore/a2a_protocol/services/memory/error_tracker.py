"""
ErrorTracker for COMPASS Error Management

Implements comprehensive error tracking and pattern detection for COMPASS learning.
Provides error recording with full context, severity scoring, pattern detection
(frequency, sequence, context correlation), and ACE integration signals.

Component ID: MEM-024
Ticket: MEM-024 (Implement ErrorTracker)

Features:
- Error recording with full context (task, stage, agent)
- Error type classification (hallucination, missing_info, incorrect_action, context_degradation)
- Severity scoring algorithms (0-1 scale)
- Pattern detection: frequency analysis, sequence detection, context correlation
- Error history queries with filtering
- Error-aware retrieval integration
- ACE integration signals (error rate >30% threshold)
- 100% error capture rate target
- 80%+ pattern detection accuracy target
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import structlog

from agentcore.a2a_protocol.database.connection import get_session
from agentcore.a2a_protocol.database.repositories import ErrorRepository
from agentcore.a2a_protocol.models.memory import ErrorRecord, ErrorType

logger = structlog.get_logger()


# ACE Integration thresholds
ERROR_RATE_ACE_THRESHOLD = 0.30  # 30% error rate triggers ACE signal
HIGH_SEVERITY_THRESHOLD = 0.7  # Severity >= 0.7 considered high severity
CRITICAL_SEVERITY_THRESHOLD = 0.9  # Severity >= 0.9 considered critical

# Pattern detection thresholds
FREQUENCY_PATTERN_THRESHOLD = 3  # 3+ occurrences of same error type
SEQUENCE_WINDOW_SIZE = 10  # Check for sequences within N recent errors
CORRELATION_THRESHOLD = 0.6  # Context similarity threshold for correlation


class ErrorPattern:
    """Detected error pattern with metadata."""

    __slots__ = (
        "pattern_type",
        "error_ids",
        "confidence",
        "detected_at",
        "metadata",
    )

    def __init__(
        self,
        pattern_type: str,
        error_ids: list[str],
        confidence: float,
        detected_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize ErrorPattern.

        Args:
            pattern_type: Pattern type (frequency, sequence, correlation)
            error_ids: Error IDs involved in pattern
            confidence: Pattern confidence score (0-1)
            detected_at: Pattern detection timestamp
            metadata: Optional metadata about the pattern
        """
        self.pattern_type = pattern_type
        self.error_ids = error_ids
        self.confidence = confidence
        self.detected_at = detected_at
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "error_ids": self.error_ids,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
        }


class ACESignal:
    """ACE integration signal for error rate monitoring."""

    __slots__ = (
        "signal_type",
        "error_rate",
        "threshold",
        "triggered",
        "task_id",
        "agent_id",
        "timestamp",
        "metadata",
    )

    def __init__(
        self,
        signal_type: str,
        error_rate: float,
        threshold: float,
        triggered: bool,
        task_id: str,
        agent_id: str,
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize ACESignal.

        Args:
            signal_type: Signal type (high_error_rate, critical_error, etc.)
            error_rate: Current error rate (0-1)
            threshold: Threshold that triggered signal
            triggered: Whether threshold was exceeded
            task_id: Associated task ID
            agent_id: Associated agent ID
            timestamp: Signal timestamp
            metadata: Optional metadata
        """
        self.signal_type = signal_type
        self.error_rate = error_rate
        self.threshold = threshold
        self.triggered = triggered
        self.task_id = task_id
        self.agent_id = agent_id
        self.timestamp = timestamp
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_type": self.signal_type,
            "error_rate": self.error_rate,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ErrorTracker:
    """
    COMPASS error tracking and pattern detection service.

    Provides comprehensive error management for COMPASS learning:
    - Error recording with full context
    - Severity scoring algorithms
    - Pattern detection (frequency, sequence, correlation)
    - Error history queries
    - ACE integration signals
    - Error-aware retrieval integration
    """

    def __init__(
        self,
        trace_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """
        Initialize ErrorTracker.

        Args:
            trace_id: Optional trace ID for request tracking
            agent_id: Optional agent ID for context
        """
        self._logger = logger.bind(component="error_tracker")
        self._trace_id = trace_id
        self._agent_id = agent_id

        # In-memory cache for pattern detection (cleared per task)
        self._error_cache: dict[str, list[ErrorRecord]] = defaultdict(list)
        self._pattern_cache: dict[str, list[ErrorPattern]] = defaultdict(list)
        self._action_count_cache: dict[str, int] = defaultdict(int)

    async def record_error(
        self,
        task_id: str,
        agent_id: str,
        error_type: ErrorType,
        error_description: str,
        context_when_occurred: str,
        recovery_action: str | None = None,
        stage_id: str | None = None,
        severity_override: float | None = None,
    ) -> ErrorRecord:
        """
        Record an error with full context.

        Automatically calculates severity score if not overridden.
        Persists to database and updates in-memory cache for pattern detection.

        Args:
            task_id: Task ID where error occurred
            agent_id: Agent ID
            error_type: Error classification
            error_description: Detailed error description
            context_when_occurred: Context when error happened
            recovery_action: Optional action taken to recover
            stage_id: Optional stage ID where error occurred
            severity_override: Optional manual severity score (0-1)

        Returns:
            ErrorRecord instance with generated ID

        Raises:
            ValueError: If severity_override is out of range
        """
        if severity_override is not None and not 0.0 <= severity_override <= 1.0:
            raise ValueError(f"Severity must be 0.0-1.0, got {severity_override}")

        # Calculate severity score
        severity_score = (
            severity_override
            if severity_override is not None
            else self._calculate_severity(
                error_type, error_description, context_when_occurred
            )
        )

        # Create ErrorRecord
        error_record = ErrorRecord(
            task_id=task_id,
            agent_id=agent_id,
            error_type=error_type,
            error_description=error_description,
            context_when_occurred=context_when_occurred,
            recovery_action=recovery_action,
            stage_id=stage_id,
            error_severity=severity_score,
        )

        # Persist to database
        async with get_session() as session:
            await ErrorRepository.create(session, error_record)
            await session.commit()

        # Update in-memory cache for pattern detection
        cache_key = f"{agent_id}:{task_id}"
        self._error_cache[cache_key].append(error_record)

        # Trigger pattern detection
        await self._detect_patterns(cache_key)

        self._logger.info(
            "error_recorded",
            error_id=error_record.error_id,
            task_id=task_id,
            agent_id=agent_id,
            error_type=error_type.value,
            severity=severity_score,
            stage_id=stage_id,
        )

        return error_record

    def _calculate_severity(
        self,
        error_type: ErrorType,
        error_description: str,
        context_when_occurred: str,
    ) -> float:
        """
        Calculate severity score for an error (0-1 scale).

        Severity algorithm:
        - Base severity by error type (0.3-0.8)
        - Context modifiers (+0.1 for critical context)
        - Description length impact (+0.05 for detailed errors)
        - Keyword boosting for critical terms

        Args:
            error_type: Error classification
            error_description: Detailed error description
            context_when_occurred: Context when error happened

        Returns:
            Severity score (0-1)
        """
        # Base severity by error type
        base_severity = {
            ErrorType.HALLUCINATION: 0.8,  # High base - false information is serious
            ErrorType.MISSING_INFO: 0.5,  # Medium - recoverable
            ErrorType.INCORRECT_ACTION: 0.7,  # High - wrong action taken
            ErrorType.CONTEXT_DEGRADATION: 0.6,  # Medium-high - affects reasoning
        }[error_type]

        severity = base_severity

        # Critical keyword boosting
        critical_keywords = [
            "critical",
            "fatal",
            "corrupt",
            "lost",
            "failed",
            "broken",
            "security",
            "unauthorized",
            "exception",
            "crash",
        ]
        combined_text = (
            f"{error_description} {context_when_occurred}".lower()
        )

        keyword_count = sum(1 for kw in critical_keywords if kw in combined_text)
        severity += min(0.15, keyword_count * 0.03)  # Max +0.15 from keywords

        # Context complexity modifier
        if len(context_when_occurred) > 200:
            # Complex context suggests more serious situation
            severity += 0.05

        # Description detail modifier
        if len(error_description) > 100:
            severity += 0.02

        # Cap at 1.0
        return min(1.0, severity)

    def increment_action_count(self, task_id: str, agent_id: str) -> None:
        """
        Increment action count for error rate calculation.

        Call this for each action/operation in a task to track total actions.

        Args:
            task_id: Task ID
            agent_id: Agent ID
        """
        cache_key = f"{agent_id}:{task_id}"
        self._action_count_cache[cache_key] += 1

    async def get_error_rate(
        self,
        task_id: str,
        agent_id: str,
    ) -> float:
        """
        Calculate current error rate for task.

        Error rate = errors / total actions

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            Error rate (0-1), 0 if no actions recorded
        """
        cache_key = f"{agent_id}:{task_id}"
        total_actions = self._action_count_cache.get(cache_key, 0)

        if total_actions == 0:
            return 0.0

        error_count = len(self._error_cache.get(cache_key, []))
        return error_count / total_actions

    async def check_ace_threshold(
        self,
        task_id: str,
        agent_id: str,
    ) -> ACESignal:
        """
        Check if error rate exceeds ACE threshold (>30%).

        Generates ACE integration signal for monitoring.

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            ACESignal with threshold check results
        """
        error_rate = await self.get_error_rate(task_id, agent_id)
        triggered = error_rate > ERROR_RATE_ACE_THRESHOLD

        cache_key = f"{agent_id}:{task_id}"
        errors = self._error_cache.get(cache_key, [])

        # Calculate additional metrics
        high_severity_count = sum(
            1 for e in errors if e.error_severity >= HIGH_SEVERITY_THRESHOLD
        )
        critical_count = sum(
            1 for e in errors if e.error_severity >= CRITICAL_SEVERITY_THRESHOLD
        )

        signal = ACESignal(
            signal_type="high_error_rate",
            error_rate=error_rate,
            threshold=ERROR_RATE_ACE_THRESHOLD,
            triggered=triggered,
            task_id=task_id,
            agent_id=agent_id,
            timestamp=datetime.now(UTC),
            metadata={
                "total_errors": len(errors),
                "total_actions": self._action_count_cache.get(cache_key, 0),
                "high_severity_errors": high_severity_count,
                "critical_errors": critical_count,
            },
        )

        if triggered:
            self._logger.warning(
                "ace_threshold_exceeded",
                task_id=task_id,
                agent_id=agent_id,
                error_rate=error_rate,
                threshold=ERROR_RATE_ACE_THRESHOLD,
            )

        return signal

    async def get_error_history(
        self,
        task_id: str,
        agent_id: str | None = None,
        error_type: ErrorType | None = None,
        min_severity: float | None = None,
        hours: int = 24,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query error history with filtering.

        Args:
            task_id: Task ID to query
            agent_id: Optional agent ID filter
            error_type: Optional error type filter
            min_severity: Optional minimum severity filter
            hours: Time window in hours (default 24)
            limit: Maximum results (default 100)

        Returns:
            List of error records as dictionaries
        """
        async with get_session() as session:
            errors = await ErrorRepository.get_recent_errors(
                session, task_id, hours=hours, limit=limit
            )

        # Apply additional filters
        result = []
        for error in errors:
            if agent_id is not None and str(error.agent_id) != agent_id:
                continue
            if error_type is not None and error.error_type != error_type:
                continue
            if min_severity is not None and error.error_severity < min_severity:
                continue

            result.append({
                "error_id": f"err-{error.error_id}",
                "task_id": f"task-{error.task_id}",
                "stage_id": f"stage-{error.stage_id}" if error.stage_id else None,
                "agent_id": f"agent-{error.agent_id}",
                "error_type": error.error_type.value if hasattr(error.error_type, 'value') else str(error.error_type),
                "error_description": error.error_description,
                "context_when_occurred": error.context_when_occurred,
                "recovery_action": error.recovery_action,
                "error_severity": float(error.error_severity),
                "recorded_at": error.recorded_at.isoformat(),
            })

        return result

    async def _detect_patterns(self, cache_key: str) -> None:
        """
        Detect error patterns after new error is recorded.

        Implements three pattern detection algorithms:
        1. Frequency patterns - same error type occurring repeatedly
        2. Sequence patterns - specific error sequences
        3. Context correlation - errors with similar contexts

        Args:
            cache_key: Cache key (agent_id:task_id)
        """
        errors = self._error_cache.get(cache_key, [])

        if len(errors) < 2:
            return

        # Pattern 1: Frequency analysis
        await self._detect_frequency_patterns(cache_key, errors)

        # Pattern 2: Sequence detection
        await self._detect_sequence_patterns(cache_key, errors)

        # Pattern 3: Context correlation
        await self._detect_correlation_patterns(cache_key, errors)

    async def _detect_frequency_patterns(
        self,
        cache_key: str,
        errors: list[ErrorRecord],
    ) -> None:
        """
        Detect frequency patterns (repeated error types).

        Args:
            cache_key: Cache key
            errors: List of errors
        """
        # Count error types
        type_counts: dict[ErrorType, list[str]] = defaultdict(list)
        for error in errors:
            type_counts[error.error_type].append(error.error_id)

        # Check for frequency patterns (3+ occurrences)
        for error_type, error_ids in type_counts.items():
            if len(error_ids) >= FREQUENCY_PATTERN_THRESHOLD:
                # Calculate confidence based on frequency
                confidence = min(1.0, len(error_ids) / (FREQUENCY_PATTERN_THRESHOLD * 2))

                pattern = ErrorPattern(
                    pattern_type="frequency",
                    error_ids=error_ids,
                    confidence=confidence,
                    detected_at=datetime.now(UTC),
                    metadata={
                        "error_type": error_type.value,
                        "occurrence_count": len(error_ids),
                        "threshold": FREQUENCY_PATTERN_THRESHOLD,
                    },
                )

                if not self._pattern_exists(cache_key, pattern):
                    self._pattern_cache[cache_key].append(pattern)
                    self._logger.warning(
                        "frequency_pattern_detected",
                        cache_key=cache_key,
                        error_type=error_type.value,
                        occurrences=len(error_ids),
                        confidence=confidence,
                    )

    async def _detect_sequence_patterns(
        self,
        cache_key: str,
        errors: list[ErrorRecord],
    ) -> None:
        """
        Detect error sequence patterns.

        Identifies patterns like:
        - hallucination -> incorrect_action
        - missing_info -> context_degradation

        Args:
            cache_key: Cache key
            errors: List of errors
        """
        if len(errors) < 2:
            return

        # Check recent errors for sequences
        recent_errors = errors[-SEQUENCE_WINDOW_SIZE:]

        # Define known problematic sequences
        problematic_sequences = [
            (ErrorType.HALLUCINATION, ErrorType.INCORRECT_ACTION),
            (ErrorType.MISSING_INFO, ErrorType.CONTEXT_DEGRADATION),
            (ErrorType.INCORRECT_ACTION, ErrorType.INCORRECT_ACTION),
            (ErrorType.CONTEXT_DEGRADATION, ErrorType.HALLUCINATION),
        ]

        for i in range(len(recent_errors) - 1):
            current = recent_errors[i]
            next_error = recent_errors[i + 1]

            for seq_first, seq_second in problematic_sequences:
                if current.error_type == seq_first and next_error.error_type == seq_second:
                    # Sequence pattern detected
                    pattern = ErrorPattern(
                        pattern_type="sequence",
                        error_ids=[current.error_id, next_error.error_id],
                        confidence=0.85,  # High confidence for known sequences
                        detected_at=datetime.now(UTC),
                        metadata={
                            "sequence": [seq_first.value, seq_second.value],
                            "position_in_window": i,
                        },
                    )

                    if not self._pattern_exists(cache_key, pattern):
                        self._pattern_cache[cache_key].append(pattern)
                        self._logger.warning(
                            "sequence_pattern_detected",
                            cache_key=cache_key,
                            sequence=f"{seq_first.value} -> {seq_second.value}",
                            confidence=0.85,
                        )

    async def _detect_correlation_patterns(
        self,
        cache_key: str,
        errors: list[ErrorRecord],
    ) -> None:
        """
        Detect context correlation patterns.

        Identifies errors occurring in similar contexts.

        Args:
            cache_key: Cache key
            errors: List of errors
        """
        if len(errors) < 2:
            return

        # Group by similar contexts using keyword overlap
        context_groups: dict[str, list[str]] = defaultdict(list)

        for error in errors:
            # Extract keywords from context
            context_words = set(error.context_when_occurred.lower().split())
            # Create a simple hash for grouping
            key_terms = [w for w in context_words if len(w) > 4][:5]
            context_key = "_".join(sorted(key_terms))

            if context_key:
                context_groups[context_key].append(error.error_id)

        # Check for correlation patterns (2+ errors with similar context)
        for context_key, error_ids in context_groups.items():
            if len(error_ids) >= 2:
                # Calculate confidence based on group size
                confidence = min(1.0, len(error_ids) / 4)

                if confidence >= CORRELATION_THRESHOLD:
                    pattern = ErrorPattern(
                        pattern_type="correlation",
                        error_ids=error_ids,
                        confidence=confidence,
                        detected_at=datetime.now(UTC),
                        metadata={
                            "context_signature": context_key,
                            "correlated_count": len(error_ids),
                        },
                    )

                    if not self._pattern_exists(cache_key, pattern):
                        self._pattern_cache[cache_key].append(pattern)
                        self._logger.info(
                            "correlation_pattern_detected",
                            cache_key=cache_key,
                            context_signature=context_key,
                            correlated_count=len(error_ids),
                            confidence=confidence,
                        )

    def _pattern_exists(self, cache_key: str, pattern: ErrorPattern) -> bool:
        """
        Check if pattern already exists in cache.

        Args:
            cache_key: Cache key
            pattern: Pattern to check

        Returns:
            True if pattern exists
        """
        existing_patterns = self._pattern_cache.get(cache_key, [])

        for existing in existing_patterns:
            if existing.pattern_type != pattern.pattern_type:
                continue

            # Compare error IDs (unordered for frequency/correlation)
            if pattern.pattern_type in ("frequency", "correlation"):
                if set(existing.error_ids) == set(pattern.error_ids):
                    return True
            else:
                # Ordered for sequences
                if existing.error_ids == pattern.error_ids:
                    return True

        return False

    async def get_detected_patterns(
        self,
        task_id: str,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get all detected patterns for a task.

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            List of patterns as dictionaries
        """
        cache_key = f"{agent_id}:{task_id}"
        patterns = self._pattern_cache.get(cache_key, [])

        return [p.to_dict() for p in patterns]

    async def get_pattern_statistics(
        self,
        task_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """
        Get pattern detection statistics.

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            Statistics dictionary with pattern counts and accuracy estimates
        """
        cache_key = f"{agent_id}:{task_id}"
        patterns = self._pattern_cache.get(cache_key, [])
        errors = self._error_cache.get(cache_key, [])

        # Count patterns by type
        pattern_counts: dict[str, int] = defaultdict(int)
        total_confidence = 0.0

        for pattern in patterns:
            pattern_counts[pattern.pattern_type] += 1
            total_confidence += pattern.confidence

        avg_confidence = total_confidence / len(patterns) if patterns else 0.0

        # Error type distribution
        error_type_counts: dict[str, int] = defaultdict(int)
        for error in errors:
            error_type_counts[error.error_type.value] += 1

        return {
            "total_errors": len(errors),
            "total_patterns": len(patterns),
            "patterns_by_type": dict(pattern_counts),
            "average_pattern_confidence": avg_confidence,
            "error_type_distribution": dict(error_type_counts),
            "pattern_detection_accuracy_estimate": avg_confidence,  # Proxy for accuracy
        }

    async def get_error_aware_retrieval_context(
        self,
        task_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """
        Generate error-aware context for retrieval integration.

        Provides error metadata to influence memory retrieval.

        Args:
            task_id: Task ID
            agent_id: Agent ID

        Returns:
            Context dictionary for retrieval service
        """
        cache_key = f"{agent_id}:{task_id}"
        errors = self._error_cache.get(cache_key, [])
        patterns = self._pattern_cache.get(cache_key, [])

        # Identify high-risk error types
        high_risk_types = set()
        for pattern in patterns:
            if pattern.pattern_type == "frequency" and pattern.confidence > 0.7:
                if "error_type" in pattern.metadata:
                    high_risk_types.add(pattern.metadata["error_type"])

        # Calculate severity distribution
        severity_sum = sum(e.error_severity for e in errors)
        avg_severity = severity_sum / len(errors) if errors else 0.0

        # Recent error contexts (for avoiding similar situations)
        recent_contexts = [
            e.context_when_occurred for e in errors[-5:]
        ]

        return {
            "error_count": len(errors),
            "pattern_count": len(patterns),
            "high_risk_error_types": list(high_risk_types),
            "average_severity": avg_severity,
            "recent_error_contexts": recent_contexts,
            "ace_signal_triggered": await self.get_error_rate(task_id, agent_id) > ERROR_RATE_ACE_THRESHOLD,
            "retrieval_recommendations": {
                "avoid_contexts": recent_contexts,
                "prioritize_error_prevention": len(patterns) > 0,
                "boost_safety_memories": avg_severity > HIGH_SEVERITY_THRESHOLD,
            },
        }

    def clear_cache(self, task_id: str, agent_id: str) -> None:
        """
        Clear in-memory cache for a task.

        Call this when task is completed or abandoned.

        Args:
            task_id: Task ID
            agent_id: Agent ID
        """
        cache_key = f"{agent_id}:{task_id}"

        if cache_key in self._error_cache:
            del self._error_cache[cache_key]

        if cache_key in self._pattern_cache:
            del self._pattern_cache[cache_key]

        if cache_key in self._action_count_cache:
            del self._action_count_cache[cache_key]

        self._logger.info(
            "cache_cleared",
            task_id=task_id,
            agent_id=agent_id,
        )


__all__ = [
    "ErrorTracker",
    "ErrorPattern",
    "ACESignal",
    "ERROR_RATE_ACE_THRESHOLD",
    "HIGH_SEVERITY_THRESHOLD",
    "CRITICAL_SEVERITY_THRESHOLD",
    "FREQUENCY_PATTERN_THRESHOLD",
]
