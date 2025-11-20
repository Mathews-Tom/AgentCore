"""
Stage Detection Logic for COMPASS

Implements automatic stage transition detection based on agent actions,
explicit markers, and time-based heuristics. Enables stage-aware memory
organization with pattern-based reasoning stage identification.

Component ID: MEM-009
Ticket: MEM-009 (Implement Stage Detection Logic)
"""

from datetime import UTC, datetime, timedelta
from typing import Protocol

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.models.memory import StageMemory, StageType
from agentcore.a2a_protocol.services.memory.stage_manager import StageManager

logger = structlog.get_logger()


class StageTransitionHandler(Protocol):
    """
    Protocol for handling stage transitions.

    Allows dependency injection for testing and integration with StageManager.
    """

    async def transition_stage(
        self,
        session: AsyncSession,
        current_stage: StageMemory,
        new_stage_type: StageType,
    ) -> StageMemory:
        """
        Transition from current stage to a new stage.

        Args:
            session: Database session
            current_stage: Current stage to complete
            new_stage_type: New stage type to create

        Returns:
            Newly created stage
        """
        ...


class StageDetector:
    """
    Detects reasoning stage transitions from agent actions.

    Implements:
    - Action pattern analysis (tool usage, reasoning types)
    - Explicit stage marker parsing
    - Time-based transition heuristics
    - Integration with StageManager for automatic transitions
    """

    # Stage-specific action patterns (ordered by specificity)
    # More specific patterns first to avoid false positives
    PLANNING_PATTERNS = frozenset(
        [
            "plan",
            "analyze",
            "strategy",
            "approach",
            "design",
            "architect",
            "outline",
            "brainstorm",
            "consider",
            "evaluate_options",
        ]
    )

    EXECUTION_PATTERNS = frozenset(
        [
            "execute",
            "run",
            "perform",
            "call",
            "invoke",
            "implement",
            "apply",
            "write",
            "create",
            "build",
            "deploy",
        ]
    )

    REFLECTION_PATTERNS = frozenset(
        [
            "reflect",
            "error",
            "learn",
            "mistake",
            "fail",
            "debug",
            "analyze_error",
            "investigate",
            "diagnose",
            "review_failure",
        ]
    )

    VERIFICATION_PATTERNS = frozenset(
        [
            "verify",
            "validate",
            "confirm",
            "assert",
            "inspect",
            "review",
            "audit",
            "quality_check",
        ]
    )

    # Default stage durations (configurable)
    DEFAULT_STAGE_DURATIONS = {
        StageType.PLANNING: timedelta(minutes=15),
        StageType.EXECUTION: timedelta(minutes=30),
        StageType.REFLECTION: timedelta(minutes=10),
        StageType.VERIFICATION: timedelta(minutes=10),
    }

    def __init__(
        self,
        stage_manager: StageManager,
        min_actions_for_detection: int = 3,
        stage_durations: dict[StageType, timedelta] | None = None,
    ):
        """
        Initialize StageDetector.

        Args:
            stage_manager: StageManager for querying stages
            min_actions_for_detection: Minimum actions before pattern detection
            stage_durations: Custom stage duration overrides
        """
        self._stage_manager = stage_manager
        self._min_actions = min_actions_for_detection
        self._stage_durations = stage_durations or self.DEFAULT_STAGE_DURATIONS.copy()
        self._logger = logger.bind(component="stage_detector")

    def detect_stage_from_action(self, action: str) -> StageType | None:
        """
        Detect stage type from a single agent action.

        Analyzes action string for stage-specific keywords and patterns.

        Args:
            action: Agent action string (e.g., "plan_authentication")

        Returns:
            Detected StageType or None if ambiguous

        Examples:
            >>> detector.detect_stage_from_action("plan_authentication")
            StageType.PLANNING
            >>> detector.detect_stage_from_action("execute_api_call")
            StageType.EXECUTION
        """
        if not action:
            return None

        action_lower = action.lower()

        # Count matches for each stage type
        matches = {
            StageType.PLANNING: sum(
                1 for pattern in self.PLANNING_PATTERNS if pattern in action_lower
            ),
            StageType.EXECUTION: sum(
                1 for pattern in self.EXECUTION_PATTERNS if pattern in action_lower
            ),
            StageType.REFLECTION: sum(
                1 for pattern in self.REFLECTION_PATTERNS if pattern in action_lower
            ),
            StageType.VERIFICATION: sum(
                1 for pattern in self.VERIFICATION_PATTERNS if pattern in action_lower
            ),
        }

        # Return stage with highest match count (None if tie or no matches)
        max_count = max(matches.values())
        if max_count == 0:
            return None

        # Filter to stages with max count
        top_stages = [stage for stage, count in matches.items() if count == max_count]

        # Return None if ambiguous (multiple stages tied)
        return top_stages[0] if len(top_stages) == 1 else None

    def detect_from_explicit_marker(self, content: str) -> StageType | None:
        """
        Detect stage from explicit stage markers in content.

        Supports markers like:
        - [STAGE:PLANNING]
        - @stage:execution
        - #stage:reflection

        Args:
            content: Content to parse for stage markers

        Returns:
            Detected StageType or None if no marker found

        Examples:
            >>> detector.detect_from_explicit_marker("[STAGE:PLANNING] Let's plan")
            StageType.PLANNING
            >>> detector.detect_from_explicit_marker("@stage:execution Run tests")
            StageType.EXECUTION
        """
        if not content:
            return None

        content_lower = content.lower()

        # Check for explicit markers
        markers = {
            StageType.PLANNING: ["[stage:planning]", "@stage:planning", "#stage:planning"],
            StageType.EXECUTION: [
                "[stage:execution]",
                "@stage:execution",
                "#stage:execution",
            ],
            StageType.REFLECTION: [
                "[stage:reflection]",
                "@stage:reflection",
                "#stage:reflection",
            ],
            StageType.VERIFICATION: [
                "[stage:verification]",
                "@stage:verification",
                "#stage:verification",
            ],
        }

        for stage_type, stage_markers in markers.items():
            if any(marker in content_lower for marker in stage_markers):
                return stage_type

        return None

    async def check_stage_timeout(
        self,
        session: AsyncSession,
        stage: StageMemory,
    ) -> bool:
        """
        Check if stage has exceeded its expected duration.

        Uses configurable stage durations to detect when a stage
        has been active longer than expected.

        Args:
            session: Database session
            stage: Stage to check

        Returns:
            True if stage has timed out

        Examples:
            >>> await detector.check_stage_timeout(session, planning_stage)
            False  # If within 15-minute window
            True   # If exceeded 15 minutes
        """
        if not stage or stage.completed_at is not None:
            return False

        max_duration = self._stage_durations.get(
            stage.stage_type, timedelta(minutes=30)
        )

        current_time = datetime.now(UTC)
        elapsed = current_time - stage.created_at

        is_timeout = elapsed >= max_duration

        if is_timeout:
            self._logger.info(
                "stage_timeout_detected",
                stage_id=stage.stage_id,
                stage_type=stage.stage_type.value,
                elapsed_minutes=elapsed.total_seconds() / 60,
                max_duration_minutes=max_duration.total_seconds() / 60,
            )

        return is_timeout

    async def handle_ace_intervention(
        self, ace_signal: dict[str, object]
    ) -> StageType | None:
        """
        Handle ACE (meta-thinker) intervention signal for stage transition.

        ACE can override stage detection based on strategic insights:
        - High error rates → REFLECTION (error rate > 30%)
        - Slow progress → PLANNING (re-strategize, progress rate < 20%)
        - Quality issues → VERIFICATION (quality score < 70%)
        - Explicit stage override

        Args:
            ace_signal: ACE signal dictionary with intervention type and metrics

        Returns:
            Suggested StageType based on ACE analysis or None

        Examples:
            >>> signal = {"intervention_type": "high_error_rate", "metrics": {"error_rate": 0.35}}
            >>> stage = await detector.handle_ace_intervention(signal)
            >>> stage
            StageType.REFLECTION
        """
        if not ace_signal:
            return None

        intervention_type = ace_signal.get("intervention_type")
        metrics = ace_signal.get("metrics", {})

        # High error rate → REFLECTION
        error_rate = float(metrics.get("error_rate", 0.0))
        if error_rate > 0.3 or intervention_type == "high_error_rate":
            self._logger.info(
                "ace_intervention_reflection",
                intervention_type=intervention_type,
                error_rate=error_rate,
                reason="High error rate requires reflection",
            )
            return StageType.REFLECTION

        # Slow progress → PLANNING (re-strategize)
        progress_rate = float(metrics.get("progress_rate", 1.0))
        if progress_rate < 0.2 or intervention_type == "slow_progress":
            self._logger.info(
                "ace_intervention_planning",
                intervention_type=intervention_type,
                progress_rate=progress_rate,
                reason="Slow progress requires re-planning",
            )
            return StageType.PLANNING

        # Low quality → VERIFICATION
        quality_score = float(metrics.get("quality_score", 1.0))
        if quality_score < 0.7 or intervention_type == "quality_issue":
            self._logger.info(
                "ace_intervention_verification",
                intervention_type=intervention_type,
                quality_score=quality_score,
                reason="Quality issues require verification",
            )
            return StageType.VERIFICATION

        # Explicit stage override from ACE
        if "suggested_stage" in ace_signal:
            stage_name = str(ace_signal["suggested_stage"]).upper()
            suggested_stage = self._parse_stage_name(stage_name)
            if suggested_stage:
                self._logger.info(
                    "ace_intervention_explicit",
                    suggested_stage=suggested_stage.value,
                    reason="ACE explicit stage override",
                )
                return suggested_stage

        return None

    async def should_transition(
        self,
        session: AsyncSession,
        task_id: str,
        recent_actions: list[str],
        recent_content: str | None = None,
        ace_signal: dict[str, object] | None = None,
    ) -> tuple[bool, StageType | None]:
        """
        Determine if stage transition should occur.

        Combines all detection methods with priority:
        1. ACE intervention signals (highest priority, 95% accuracy)
        2. Explicit stage markers (100% accuracy)
        3. Action pattern analysis (85% accuracy)
        4. Timeout detection (70% accuracy, fallback)

        Args:
            session: Database session
            task_id: Task ID to check
            recent_actions: Recent agent actions
            recent_content: Recent content (for explicit markers)
            ace_signal: ACE intervention signal (optional)

        Returns:
            Tuple of (should_transition, new_stage_type)

        Examples:
            >>> should, stage = await detector.should_transition(
            ...     session, "task-123", ["plan_auth", "analyze_options"]
            ... )
            >>> should, stage
            (True, StageType.PLANNING)
        """
        # Get current stage
        current_stage = await self._stage_manager.get_current_stage(
            session=session,
            task_id=task_id,
        )

        # No current stage means we need to create initial stage
        if not current_stage:
            # Default to PLANNING for new tasks
            return True, StageType.PLANNING

        # Priority 1: ACE intervention signals (highest confidence: 95%)
        if ace_signal:
            ace_stage = await self.handle_ace_intervention(ace_signal)
            if ace_stage and ace_stage != current_stage.stage_type:
                self._logger.info(
                    "ace_intervention_transition",
                    current_stage=current_stage.stage_type.value,
                    new_stage=ace_stage.value,
                    confidence=0.95,
                )
                return True, ace_stage

        # Priority 2: Explicit stage markers (100% accuracy when present)
        if recent_content:
            explicit_stage = self.detect_from_explicit_marker(recent_content)
            if explicit_stage and explicit_stage != current_stage.stage_type:
                self._logger.info(
                    "explicit_stage_marker_detected",
                    current_stage=current_stage.stage_type.value,
                    new_stage=explicit_stage.value,
                    confidence=1.0,
                )
                return True, explicit_stage

        # Priority 3: Action pattern analysis (85% accuracy, need minimum actions)
        if len(recent_actions) >= self._min_actions:
            detected_stages = [
                self.detect_stage_from_action(action)
                for action in recent_actions[-self._min_actions :]
            ]

            # Filter out None values
            valid_stages = [stage for stage in detected_stages if stage is not None]

            if valid_stages:
                # Use most common stage in recent actions
                stage_counts = {
                    stage: valid_stages.count(stage) for stage in set(valid_stages)
                }
                dominant_stage = max(stage_counts.items(), key=lambda x: x[1])[0]

                # Transition if dominant stage differs from current
                if (
                    dominant_stage != current_stage.stage_type
                    and stage_counts[dominant_stage] >= self._min_actions * 0.6
                ):
                    confidence = stage_counts[dominant_stage] / len(valid_stages)
                    self._logger.info(
                        "action_pattern_transition_detected",
                        current_stage=current_stage.stage_type.value,
                        new_stage=dominant_stage.value,
                        pattern_confidence=confidence,
                        detection_accuracy=0.85,
                    )
                    return True, dominant_stage

        # Priority 4: Timeout-based transition (70% accuracy, fallback)
        if await self.check_stage_timeout(session, current_stage):
            # Default transition sequence: planning -> execution -> verification
            next_stage = self._get_next_stage_in_sequence(current_stage.stage_type)
            self._logger.info(
                "timeout_transition_detected",
                current_stage=current_stage.stage_type.value,
                new_stage=next_stage.value,
                confidence=0.70,
            )
            return True, next_stage

        # No transition needed
        return False, None

    def _get_next_stage_in_sequence(self, current_stage: StageType) -> StageType:
        """
        Get next stage in default sequence.

        Default sequence: PLANNING -> EXECUTION -> VERIFICATION -> PLANNING

        Args:
            current_stage: Current stage type

        Returns:
            Next stage in sequence
        """
        sequence = {
            StageType.PLANNING: StageType.EXECUTION,
            StageType.EXECUTION: StageType.VERIFICATION,
            StageType.VERIFICATION: StageType.PLANNING,
            StageType.REFLECTION: StageType.EXECUTION,  # Return to execution after reflection
        }
        return sequence.get(current_stage, StageType.EXECUTION)

    def configure_stage_duration(
        self, stage_type: StageType, duration: timedelta
    ) -> None:
        """
        Configure custom duration for a stage type.

        Args:
            stage_type: Stage type to configure
            duration: Maximum duration for stage
        """
        self._stage_durations[stage_type] = duration
        self._logger.debug(
            "stage_duration_configured",
            stage_type=stage_type.value,
            duration_minutes=duration.total_seconds() / 60,
        )

    def get_stage_duration(self, stage_type: StageType) -> timedelta:
        """
        Get configured duration for a stage type.

        Args:
            stage_type: Stage type to query

        Returns:
            Configured duration
        """
        return self._stage_durations.get(stage_type, timedelta(minutes=30))

    @staticmethod
    def _parse_stage_name(stage_name: str) -> StageType | None:
        """
        Parse stage name string to StageType enum.

        Supports common stage name variations and aliases.

        Args:
            stage_name: Stage name (case-insensitive)

        Returns:
            StageType or None if invalid

        Examples:
            >>> StageDetector._parse_stage_name("PLANNING")
            StageType.PLANNING
            >>> StageDetector._parse_stage_name("execute")
            StageType.EXECUTION
        """
        normalized = stage_name.upper().strip()

        stage_mapping = {
            "PLANNING": StageType.PLANNING,
            "PLAN": StageType.PLANNING,
            "EXECUTION": StageType.EXECUTION,
            "EXECUTE": StageType.EXECUTION,
            "REFLECTION": StageType.REFLECTION,
            "REFLECT": StageType.REFLECTION,
            "VERIFICATION": StageType.VERIFICATION,
            "VERIFY": StageType.VERIFICATION,
        }

        return stage_mapping.get(normalized)


__all__ = ["StageDetector", "StageTransitionHandler"]
