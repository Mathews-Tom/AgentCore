"""
Unit Tests for StageDetector

Tests automatic stage transition detection based on action patterns,
explicit markers, and timeout heuristics.

Component ID: MEM-009
Ticket: MEM-009 (Implement Stage Detection Logic)
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.models.memory import StageMemory, StageType
from agentcore.a2a_protocol.services.memory import StageDetector, StageManager


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_stage_manager() -> AsyncMock:
    """Create mock StageManager."""
    manager = AsyncMock(spec=StageManager)
    return manager


@pytest.fixture
def stage_detector(mock_stage_manager: AsyncMock) -> StageDetector:
    """Create StageDetector instance."""
    return StageDetector(stage_manager=mock_stage_manager, min_actions_for_detection=3)


@pytest.fixture
def sample_planning_stage() -> StageMemory:
    """Create sample planning stage."""
    return StageMemory(
        stage_id=f"stage-{uuid4()}",
        task_id=f"task-{uuid4()}",
        agent_id=f"agent-{uuid4()}",
        stage_type=StageType.PLANNING,
        stage_summary="Planning authentication system",
        stage_insights=[],
        raw_memory_refs=[],
        compression_ratio=1.0,
        compression_model="none",
        quality_score=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        completed_at=None,
    )


@pytest.fixture
def sample_execution_stage() -> StageMemory:
    """Create sample execution stage."""
    return StageMemory(
        stage_id=f"stage-{uuid4()}",
        task_id=f"task-{uuid4()}",
        agent_id=f"agent-{uuid4()}",
        stage_type=StageType.EXECUTION,
        stage_summary="Executing authentication implementation",
        stage_insights=[],
        raw_memory_refs=[],
        compression_ratio=1.0,
        compression_model="none",
        quality_score=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        completed_at=None,
    )


class TestActionPatternDetection:
    """Test action pattern-based stage detection."""

    def test_detect_planning_actions(self, stage_detector: StageDetector):
        """Test detection of planning actions."""
        planning_actions = [
            "plan_authentication",
            "analyze_requirements",
            "design_strategy",
            "evaluate_options",
            "brainstorm_approach",
        ]

        for action in planning_actions:
            detected = stage_detector.detect_stage_from_action(action)
            assert (
                detected == StageType.PLANNING
            ), f"Action '{action}' should detect PLANNING"

    def test_detect_execution_actions(self, stage_detector: StageDetector):
        """Test detection of execution actions."""
        execution_actions = [
            "execute_api_call",
            "run_operation",
            "perform_database_query",
            "implement_feature",
            "build_component",
        ]

        for action in execution_actions:
            detected = stage_detector.detect_stage_from_action(action)
            assert (
                detected == StageType.EXECUTION
            ), f"Action '{action}' should detect EXECUTION"

    def test_detect_reflection_actions(self, stage_detector: StageDetector):
        """Test detection of reflection actions."""
        reflection_actions = [
            "reflect_on_error",
            "analyze_error_pattern",
            "learn_from_mistake",
            "debug_failure",
            "investigate_issue",
        ]

        for action in reflection_actions:
            detected = stage_detector.detect_stage_from_action(action)
            assert (
                detected == StageType.REFLECTION
            ), f"Action '{action}' should detect REFLECTION"

    def test_detect_verification_actions(self, stage_detector: StageDetector):
        """Test detection of verification actions."""
        verification_actions = [
            "verify_results",
            "validate_output",
            "confirm_success",
            "inspect_quality",
            "review_quality",
        ]

        for action in verification_actions:
            detected = stage_detector.detect_stage_from_action(action)
            assert (
                detected == StageType.VERIFICATION
            ), f"Action '{action}' should detect VERIFICATION"

    def test_detect_ambiguous_action_returns_none(self, stage_detector: StageDetector):
        """Test ambiguous actions return None."""
        ambiguous_actions = [
            "process_data",
            "handle_request",
            "generic_operation",
        ]

        for action in ambiguous_actions:
            detected = stage_detector.detect_stage_from_action(action)
            assert (
                detected is None
            ), f"Ambiguous action '{action}' should return None"

    def test_detect_empty_action_returns_none(self, stage_detector: StageDetector):
        """Test empty action returns None."""
        assert stage_detector.detect_stage_from_action("") is None
        assert stage_detector.detect_stage_from_action(None) is None

    def test_detect_case_insensitive(self, stage_detector: StageDetector):
        """Test pattern detection is case-insensitive."""
        assert stage_detector.detect_stage_from_action("PLAN_AUTH") == StageType.PLANNING
        assert (
            stage_detector.detect_stage_from_action("Execute_API") == StageType.EXECUTION
        )
        assert (
            stage_detector.detect_stage_from_action("Reflect_Error")
            == StageType.REFLECTION
        )
        assert (
            stage_detector.detect_stage_from_action("Verify_Output")
            == StageType.VERIFICATION
        )


class TestExplicitMarkerDetection:
    """Test explicit stage marker detection."""

    def test_detect_bracket_markers(self, stage_detector: StageDetector):
        """Test detection of bracket-style markers [STAGE:X]."""
        assert (
            stage_detector.detect_from_explicit_marker(
                "[STAGE:PLANNING] Let's plan the approach"
            )
            == StageType.PLANNING
        )
        assert (
            stage_detector.detect_from_explicit_marker(
                "[STAGE:EXECUTION] Running tests now"
            )
            == StageType.EXECUTION
        )
        assert (
            stage_detector.detect_from_explicit_marker(
                "[STAGE:REFLECTION] Analyzing error"
            )
            == StageType.REFLECTION
        )
        assert (
            stage_detector.detect_from_explicit_marker(
                "[STAGE:VERIFICATION] Checking quality"
            )
            == StageType.VERIFICATION
        )

    def test_detect_at_markers(self, stage_detector: StageDetector):
        """Test detection of @-style markers @stage:X."""
        assert (
            stage_detector.detect_from_explicit_marker(
                "@stage:planning Starting new phase"
            )
            == StageType.PLANNING
        )
        assert (
            stage_detector.detect_from_explicit_marker(
                "@stage:execution Implementing feature"
            )
            == StageType.EXECUTION
        )
        assert (
            stage_detector.detect_from_explicit_marker(
                "@stage:reflection Learning from failure"
            )
            == StageType.REFLECTION
        )
        assert (
            stage_detector.detect_from_explicit_marker(
                "@stage:verification Testing completed"
            )
            == StageType.VERIFICATION
        )

    def test_detect_hash_markers(self, stage_detector: StageDetector):
        """Test detection of #-style markers #stage:X."""
        assert (
            stage_detector.detect_from_explicit_marker("#stage:planning Planning phase")
            == StageType.PLANNING
        )
        assert (
            stage_detector.detect_from_explicit_marker("#stage:execution Executing now")
            == StageType.EXECUTION
        )
        assert (
            stage_detector.detect_from_explicit_marker("#stage:reflection Reflecting")
            == StageType.REFLECTION
        )
        assert (
            stage_detector.detect_from_explicit_marker("#stage:verification Verifying")
            == StageType.VERIFICATION
        )

    def test_detect_marker_case_insensitive(self, stage_detector: StageDetector):
        """Test marker detection is case-insensitive."""
        assert (
            stage_detector.detect_from_explicit_marker("[stage:PLANNING]")
            == StageType.PLANNING
        )
        assert (
            stage_detector.detect_from_explicit_marker("@STAGE:execution")
            == StageType.EXECUTION
        )
        assert (
            stage_detector.detect_from_explicit_marker("#StAgE:rEfLeCtIoN")
            == StageType.REFLECTION
        )

    def test_detect_no_marker_returns_none(self, stage_detector: StageDetector):
        """Test content without markers returns None."""
        assert (
            stage_detector.detect_from_explicit_marker("Regular content no marker")
            is None
        )
        assert stage_detector.detect_from_explicit_marker("") is None
        assert stage_detector.detect_from_explicit_marker(None) is None

    def test_detect_marker_in_middle_of_content(self, stage_detector: StageDetector):
        """Test markers detected anywhere in content."""
        content = "Some text before [STAGE:PLANNING] and text after"
        assert stage_detector.detect_from_explicit_marker(content) == StageType.PLANNING


class TestTimeoutDetection:
    """Test time-based stage timeout detection."""

    @pytest.mark.asyncio
    async def test_timeout_not_exceeded(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test stage within timeout returns False."""
        # Stage just created, should not timeout
        is_timeout = await stage_detector.check_stage_timeout(
            session=mock_session,
            stage=sample_planning_stage,
        )
        assert is_timeout is False

    @pytest.mark.asyncio
    async def test_timeout_exceeded(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test stage exceeding timeout returns True."""
        # Set stage created 20 minutes ago (planning default is 15 min)
        sample_planning_stage.created_at = datetime.now(UTC) - timedelta(minutes=20)

        is_timeout = await stage_detector.check_stage_timeout(
            session=mock_session,
            stage=sample_planning_stage,
        )
        assert is_timeout is True

    @pytest.mark.asyncio
    async def test_timeout_completed_stage_returns_false(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test completed stage never times out."""
        sample_planning_stage.created_at = datetime.now(UTC) - timedelta(hours=1)
        sample_planning_stage.completed_at = datetime.now(UTC)

        is_timeout = await stage_detector.check_stage_timeout(
            session=mock_session,
            stage=sample_planning_stage,
        )
        assert is_timeout is False

    @pytest.mark.asyncio
    async def test_timeout_none_stage_returns_false(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
    ):
        """Test None stage returns False."""
        is_timeout = await stage_detector.check_stage_timeout(
            session=mock_session,
            stage=None,
        )
        assert is_timeout is False

    @pytest.mark.asyncio
    async def test_timeout_different_stage_types(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        sample_execution_stage: StageMemory,
    ):
        """Test timeout respects different stage durations."""
        # Execution default is 30 minutes
        sample_execution_stage.created_at = datetime.now(UTC) - timedelta(minutes=25)

        is_timeout = await stage_detector.check_stage_timeout(
            session=mock_session,
            stage=sample_execution_stage,
        )
        assert is_timeout is False

        # Now exceed 30 minutes
        sample_execution_stage.created_at = datetime.now(UTC) - timedelta(minutes=35)

        is_timeout = await stage_detector.check_stage_timeout(
            session=mock_session,
            stage=sample_execution_stage,
        )
        assert is_timeout is True


class TestShouldTransition:
    """Test combined transition detection logic."""

    @pytest.mark.asyncio
    async def test_no_current_stage_creates_planning(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
    ):
        """Test no current stage defaults to creating PLANNING."""
        mock_stage_manager.get_current_stage = AsyncMock(return_value=None)

        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id="task-123",
            recent_actions=["some_action"],
        )

        assert should is True
        assert stage_type == StageType.PLANNING

    @pytest.mark.asyncio
    async def test_explicit_marker_highest_priority(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test explicit marker overrides action patterns."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["plan_auth", "plan_db"],  # Planning actions
            recent_content="[STAGE:EXECUTION] Let's execute",  # But explicit marker
        )

        assert should is True
        assert stage_type == StageType.EXECUTION

    @pytest.mark.asyncio
    async def test_action_pattern_transition(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test action pattern triggers transition."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        # Execution actions while in planning stage
        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=[
                "execute_api_call",
                "run_tests",
                "perform_operation",
            ],
        )

        assert should is True
        assert stage_type == StageType.EXECUTION

    @pytest.mark.asyncio
    async def test_insufficient_actions_no_transition(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test insufficient actions prevent pattern detection."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        # Only 2 actions, need 3 minimum
        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["execute_api_call", "run_tests"],
        )

        assert should is False
        assert stage_type is None

    @pytest.mark.asyncio
    async def test_timeout_triggers_transition(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test timeout triggers automatic transition."""
        # Stage exceeds timeout
        sample_planning_stage.created_at = datetime.now(UTC) - timedelta(minutes=20)
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["some_action"],
        )

        assert should is True
        assert stage_type == StageType.EXECUTION  # Next in sequence

    @pytest.mark.asyncio
    async def test_same_stage_no_transition(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test same stage type doesn't trigger transition."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        # Planning actions in planning stage
        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["plan_auth", "plan_db", "plan_api"],
        )

        assert should is False
        assert stage_type is None

    @pytest.mark.asyncio
    async def test_dominant_stage_threshold(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test dominant stage must meet threshold."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        # Mix of actions, but execution not dominant enough (2/5 = 40% < 60%)
        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=[
                "execute_api",
                "execute_test",
                "generic_action",
                "other_action",
                "another_action",
            ],
        )

        assert should is False
        assert stage_type is None


class TestStageSequence:
    """Test default stage sequencing."""

    def test_next_stage_sequence(self, stage_detector: StageDetector):
        """Test default stage sequence transitions."""
        # Planning -> Execution
        next_stage = stage_detector._get_next_stage_in_sequence(StageType.PLANNING)
        assert next_stage == StageType.EXECUTION

        # Execution -> Verification
        next_stage = stage_detector._get_next_stage_in_sequence(StageType.EXECUTION)
        assert next_stage == StageType.VERIFICATION

        # Verification -> Planning
        next_stage = stage_detector._get_next_stage_in_sequence(StageType.VERIFICATION)
        assert next_stage == StageType.PLANNING

        # Reflection -> Execution
        next_stage = stage_detector._get_next_stage_in_sequence(StageType.REFLECTION)
        assert next_stage == StageType.EXECUTION


class TestConfiguration:
    """Test StageDetector configuration."""

    def test_configure_stage_duration(
        self, mock_stage_manager: AsyncMock
    ):
        """Test configuring custom stage durations."""
        detector = StageDetector(stage_manager=mock_stage_manager)

        custom_duration = timedelta(minutes=45)
        detector.configure_stage_duration(StageType.PLANNING, custom_duration)

        assert detector.get_stage_duration(StageType.PLANNING) == custom_duration

    def test_get_default_stage_duration(
        self, mock_stage_manager: AsyncMock
    ):
        """Test getting default stage durations."""
        detector = StageDetector(stage_manager=mock_stage_manager)

        assert detector.get_stage_duration(StageType.PLANNING) == timedelta(minutes=15)
        assert detector.get_stage_duration(StageType.EXECUTION) == timedelta(minutes=30)
        assert detector.get_stage_duration(StageType.REFLECTION) == timedelta(minutes=10)
        assert detector.get_stage_duration(StageType.VERIFICATION) == timedelta(
            minutes=10
        )

    def test_custom_min_actions(
        self, mock_stage_manager: AsyncMock
    ):
        """Test custom minimum actions configuration."""
        detector = StageDetector(
            stage_manager=mock_stage_manager,
            min_actions_for_detection=5,
        )

        assert detector._min_actions == 5

    def test_custom_stage_durations_initialization(
        self, mock_stage_manager: AsyncMock
    ):
        """Test initializing with custom stage durations."""
        custom_durations = {
            StageType.PLANNING: timedelta(minutes=20),
            StageType.EXECUTION: timedelta(minutes=45),
        }

        detector = StageDetector(
            stage_manager=mock_stage_manager,
            stage_durations=custom_durations,
        )

        assert detector.get_stage_duration(StageType.PLANNING) == timedelta(minutes=20)
        assert detector.get_stage_duration(StageType.EXECUTION) == timedelta(minutes=45)


class TestIntegrationWithStageManager:
    """Test integration patterns with StageManager."""

    @pytest.mark.asyncio
    async def test_detector_queries_stage_manager(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test detector properly queries StageManager."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["execute_test"],
        )

        # Verify get_current_stage was called
        mock_stage_manager.get_current_stage.assert_called_once_with(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
        )

    @pytest.mark.asyncio
    async def test_detector_handles_stage_manager_errors(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
    ):
        """Test detector handles StageManager errors gracefully."""
        mock_stage_manager.get_current_stage = AsyncMock(
            side_effect=Exception("Database error")
        )

        with pytest.raises(Exception, match="Database error"):
            await stage_detector.should_transition(
                session=mock_session,
                task_id="task-123",
                recent_actions=["execute_test"],
            )


class TestACEIntervention:
    """Test ACE intervention signal handling."""

    @pytest.mark.asyncio
    async def test_ace_high_error_rate_triggers_reflection(
        self, stage_detector: StageDetector
    ):
        """Test ACE high error rate triggers REFLECTION."""
        signal = {
            "intervention_type": "high_error_rate",
            "metrics": {"error_rate": 0.35},
        }

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.REFLECTION

    @pytest.mark.asyncio
    async def test_ace_slow_progress_triggers_planning(
        self, stage_detector: StageDetector
    ):
        """Test ACE slow progress triggers PLANNING."""
        signal = {
            "intervention_type": "slow_progress",
            "metrics": {"progress_rate": 0.15},
        }

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.PLANNING

    @pytest.mark.asyncio
    async def test_ace_quality_issue_triggers_verification(
        self, stage_detector: StageDetector
    ):
        """Test ACE quality issue triggers VERIFICATION."""
        signal = {
            "intervention_type": "quality_issue",
            "metrics": {"quality_score": 0.65},
        }

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.VERIFICATION

    @pytest.mark.asyncio
    async def test_ace_explicit_stage_override(self, stage_detector: StageDetector):
        """Test ACE explicit stage override."""
        signal = {"suggested_stage": "execution"}

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.EXECUTION

    @pytest.mark.asyncio
    async def test_ace_high_error_rate_by_metric(self, stage_detector: StageDetector):
        """Test ACE detects high error rate from metrics alone."""
        signal = {"metrics": {"error_rate": 0.4}}

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.REFLECTION

    @pytest.mark.asyncio
    async def test_ace_slow_progress_by_metric(self, stage_detector: StageDetector):
        """Test ACE detects slow progress from metrics alone."""
        signal = {"metrics": {"progress_rate": 0.1}}

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.PLANNING

    @pytest.mark.asyncio
    async def test_ace_quality_issue_by_metric(self, stage_detector: StageDetector):
        """Test ACE detects quality issue from metrics alone."""
        signal = {"metrics": {"quality_score": 0.5}}

        stage = await stage_detector.handle_ace_intervention(signal)
        assert stage == StageType.VERIFICATION

    @pytest.mark.asyncio
    async def test_ace_empty_signal_returns_none(self, stage_detector: StageDetector):
        """Test empty ACE signal returns None."""
        assert await stage_detector.handle_ace_intervention({}) is None
        assert await stage_detector.handle_ace_intervention(None) is None

    @pytest.mark.asyncio
    async def test_ace_intervention_highest_priority(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test ACE intervention has highest priority."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        # ACE says REFLECTION, but actions suggest EXECUTION
        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["execute_api", "run_test", "perform_action"],
            recent_content="[STAGE:VERIFICATION] Verifying",
            ace_signal={"intervention_type": "high_error_rate", "metrics": {"error_rate": 0.4}},
        )

        # ACE signal should win
        assert should is True
        assert stage_type == StageType.REFLECTION


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_all_actions_ambiguous(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test handling all ambiguous actions."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["generic1", "generic2", "generic3"],
        )

        assert should is False
        assert stage_type is None

    @pytest.mark.asyncio
    async def test_mixed_stage_actions_below_threshold(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test mixed actions below confidence threshold."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        # 1 execution, 1 verification, 1 reflection - no clear winner
        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=["execute_api", "verify_output", "reflect_error"],
        )

        assert should is False
        assert stage_type is None

    def test_action_with_multiple_patterns(self, stage_detector: StageDetector):
        """Test action matching multiple patterns."""
        # This action has both "plan" and "execute" - planning has 2 matches (plan, strategy)
        action = "plan_and_execute_strategy"
        detected = stage_detector.detect_stage_from_action(action)

        # Planning wins with 2 matches vs execution's 1
        assert detected == StageType.PLANNING

    @pytest.mark.asyncio
    async def test_empty_actions_list(
        self,
        stage_detector: StageDetector,
        mock_session: AsyncMock,
        mock_stage_manager: AsyncMock,
        sample_planning_stage: StageMemory,
    ):
        """Test empty actions list."""
        mock_stage_manager.get_current_stage = AsyncMock(
            return_value=sample_planning_stage
        )

        should, stage_type = await stage_detector.should_transition(
            session=mock_session,
            task_id=sample_planning_stage.task_id,
            recent_actions=[],
        )

        assert should is False
        assert stage_type is None
