"""
Integration Tests for Stage Detection with Real Workflows

Tests stage detection accuracy with realistic agent workflow scenarios.
Validates 90%+ detection accuracy requirement for MEM-009.

Component ID: MEM-009
Ticket: MEM-009 (Implement Stage Detection Logic)

Note: These tests focus on detection logic accuracy with realistic
workflows. Database integration is covered in other test files.
"""

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.a2a_protocol.models.memory import StageType
from agentcore.a2a_protocol.services.memory import StageDetector, StageManager


@pytest.fixture
def stage_manager() -> StageManager:
    """Create StageManager instance for testing."""
    return StageManager(compression_trigger=None)


@pytest.fixture
def stage_detector(stage_manager: StageManager) -> StageDetector:
    """Create StageDetector instance for testing."""
    return StageDetector(stage_manager=stage_manager, min_actions_for_detection=3)


@pytest.mark.asyncio
class TestRealisticWorkflowScenarios:
    """Test stage detection with realistic agent workflows."""

    async def test_authentication_implementation_workflow(
        self,
        stage_detector: StageDetector,
    ):
        """
        Test stage detection for authentication implementation workflow.

        Workflow:
        1. Planning: analyze requirements, design architecture
        2. Execution: implement JWT service, create endpoints
        3. Verification: run tests, validate security
        4. Reflection: review code quality, identify improvements

        Expected: 100% accuracy (4/4 stages detected correctly)
        """
        # Stage 1: Planning phase
        actions_planning = [
            "analyze_requirements",
            "design_jwt_architecture",
            "plan_authentication_flow",
            "evaluate_security_options",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_planning
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        assert len(valid_stages) > 0

        # Most common stage should be PLANNING
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.PLANNING

        # Stage 2: Execution phase
        actions_execution = [
            "implement_jwt_service",
            "create_auth_endpoints",
            "build_token_validation",
            "deploy_auth_service",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_execution
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.EXECUTION

        # Stage 3: Verification phase
        actions_verification = [
            "test_jwt_generation",
            "validate_token_expiry",
            "verify_security_headers",
            "check_rate_limiting",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_verification
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.VERIFICATION

        # Stage 4: Reflection phase (with explicit marker)
        detected_stage = stage_detector.detect_from_explicit_marker(
            "[STAGE:REFLECTION] Reviewing authentication implementation"
        )
        assert detected_stage == StageType.REFLECTION

        # Accuracy: 4/4 = 100%

    async def test_bug_fix_workflow(
        self,
        stage_detector: StageDetector,
    ):
        """
        Test stage detection for bug fix workflow.

        Workflow:
        1. Reflection: analyze error, debug issue
        2. Planning: design fix strategy
        3. Execution: implement fix
        4. Verification: run regression tests

        Expected: 100% accuracy (4/4 stages detected correctly)
        """
        # Stage 1: Reflection phase (bug analysis)
        actions_reflection = [
            "analyze_error_logs",
            "debug_authentication_failure",
            "investigate_token_expiry",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_reflection
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.REFLECTION

        # Stage 2: Planning phase (fix strategy)
        actions_planning = [
            "plan_token_refresh_fix",
            "design_error_handling",
            "evaluate_fix_approaches",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_planning
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.PLANNING

        # Stage 3: Execution phase (implement fix)
        actions_execution = [
            "implement_token_refresh",
            "apply_error_handling",
            "deploy_fix",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_execution
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.EXECUTION

        # Stage 4: Verification phase (regression tests)
        actions_verification = [
            "run_regression_tests",
            "validate_fix_effectiveness",
            "verify_no_side_effects",
        ]

        detected_stages = [
            stage_detector.detect_stage_from_action(action) for action in actions_verification
        ]
        valid_stages = [s for s in detected_stages if s is not None]
        stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
        detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
        assert detected_stage == StageType.VERIFICATION

        # Accuracy: 4/4 = 100%

    async def test_ace_intervention_scenarios(
        self,
        stage_detector: StageDetector,
    ):
        """
        Test ACE intervention in various scenarios.

        Expected: 100% accuracy for ACE signals (documented 95% accuracy)
        """
        # Scenario 1: High error rate triggers reflection
        signal = {
            "intervention_type": "high_error_rate",
            "metrics": {"error_rate": 0.4},
        }
        detected_stage = await stage_detector.handle_ace_intervention(signal)
        assert detected_stage == StageType.REFLECTION

        # Scenario 2: Slow progress triggers planning
        signal = {
            "intervention_type": "slow_progress",
            "metrics": {"progress_rate": 0.15},
        }
        detected_stage = await stage_detector.handle_ace_intervention(signal)
        assert detected_stage == StageType.PLANNING

        # Scenario 3: Quality issue triggers verification
        signal = {
            "intervention_type": "quality_issue",
            "metrics": {"quality_score": 0.65},
        }
        detected_stage = await stage_detector.handle_ace_intervention(signal)
        assert detected_stage == StageType.VERIFICATION

        # Scenario 4: Explicit stage override
        signal = {"suggested_stage": "execution"}
        detected_stage = await stage_detector.handle_ace_intervention(signal)
        assert detected_stage == StageType.EXECUTION

        # Accuracy: 4/4 = 100%

    async def test_explicit_marker_variations(
        self,
        stage_detector: StageDetector,
    ):
        """
        Test explicit stage markers in various formats.

        Expected: 100% accuracy (explicit markers are unambiguous)
        """
        test_cases = [
            ("[STAGE:PLANNING] Starting planning", StageType.PLANNING),
            ("@stage:execution Running execution", StageType.EXECUTION),
            ("#stage:verification Verifying results", StageType.VERIFICATION),
            ("[STAGE:REFLECTION] Reflecting on errors", StageType.REFLECTION),
            ("@stage:planning Plan iteration 2", StageType.PLANNING),
            ("#stage:execution Execute phase 3", StageType.EXECUTION),
        ]

        correct_detections = 0
        for marker, expected_stage in test_cases:
            detected_stage = stage_detector.detect_from_explicit_marker(marker)
            if detected_stage == expected_stage:
                correct_detections += 1

        accuracy = correct_detections / len(test_cases)
        assert accuracy == 1.0  # 100% accuracy for explicit markers


@pytest.mark.asyncio
class TestDetectionAccuracy:
    """Test overall detection accuracy meets 90%+ requirement."""

    async def test_action_pattern_accuracy_validation(
        self,
        stage_detector: StageDetector,
    ):
        """
        Test action pattern detection accuracy with diverse action sets.

        Expected: 85%+ accuracy for action patterns (documented accuracy)
        """
        test_cases = [
            # (actions, expected_stage)
            (["plan_auth", "analyze_security", "design_api"], StageType.PLANNING),
            (["execute_tests", "run_migration", "deploy_service"], StageType.EXECUTION),
            (["verify_output", "validate_results", "confirm_success"], StageType.VERIFICATION),
            (["reflect_on_error", "analyze_failure", "learn_from_bug"], StageType.REFLECTION),
            (["plan_strategy", "outline_approach", "brainstorm_ideas"], StageType.PLANNING),
            (["implement_feature", "build_component", "create_endpoint"], StageType.EXECUTION),
            (["review_code", "inspect_quality", "audit_security"], StageType.VERIFICATION),
            (["debug_issue", "investigate_error", "diagnose_problem"], StageType.REFLECTION),
            (["evaluate_options", "consider_alternatives", "analyze_tradeoffs"], StageType.PLANNING),
            (["write_code", "invoke_api", "perform_operation"], StageType.EXECUTION),
            (["test_functionality", "validate_behavior", "check_correctness"], StageType.VERIFICATION),
            (["reflect_learnings", "error_analysis", "failure_review"], StageType.REFLECTION),
        ]

        correct_detections = 0
        total_cases = len(test_cases)

        for actions, expected_stage in test_cases:
            # Detect stage from majority of actions
            detected_stages = [stage_detector.detect_stage_from_action(action) for action in actions]
            valid_stages = [s for s in detected_stages if s is not None]

            if valid_stages:
                # Use most common stage
                stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
                detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]

                if detected_stage == expected_stage:
                    correct_detections += 1

        accuracy = correct_detections / total_cases
        assert accuracy >= 0.85, f"Action pattern accuracy {accuracy:.2%} below 85% target"

    async def test_combined_detection_accuracy(
        self,
        stage_detector: StageDetector,
    ):
        """
        Test combined detection accuracy (all methods).

        Expected: 90%+ overall accuracy when combining all detection methods
        """
        test_scenarios = [
            # (detection_method, input, expected_stage, accuracy_weight)
            ("ace", {"intervention_type": "high_error_rate", "metrics": {"error_rate": 0.4}}, StageType.REFLECTION, 0.95),
            ("ace", {"intervention_type": "slow_progress", "metrics": {"progress_rate": 0.1}}, StageType.PLANNING, 0.95),
            ("ace", {"intervention_type": "quality_issue", "metrics": {"quality_score": 0.6}}, StageType.VERIFICATION, 0.95),
            ("ace", {"suggested_stage": "execution"}, StageType.EXECUTION, 0.95),
            ("marker", "[STAGE:PLANNING] Starting", StageType.PLANNING, 1.0),
            ("marker", "@stage:execution Running", StageType.EXECUTION, 1.0),
            ("marker", "#stage:reflection Analyzing", StageType.REFLECTION, 1.0),
            ("marker", "[STAGE:VERIFICATION] Testing", StageType.VERIFICATION, 1.0),
            ("action", ["plan_auth", "analyze_reqs", "design_api"], StageType.PLANNING, 0.85),
            ("action", ["execute_test", "run_job", "perform_task"], StageType.EXECUTION, 0.85),
            ("action", ["verify_output", "validate_data", "check_quality"], StageType.VERIFICATION, 0.85),
            ("action", ["debug_error", "analyze_failure", "investigate_bug"], StageType.REFLECTION, 0.85),
        ]

        total_weighted_score = 0.0
        total_weight = 0.0

        for method, input_data, expected_stage, weight in test_scenarios:
            detected_stage = None

            if method == "ace":
                detected_stage = await stage_detector.handle_ace_intervention(input_data)
            elif method == "marker":
                detected_stage = stage_detector.detect_from_explicit_marker(input_data)
            elif method == "action":
                # Detect from action list
                detected_stages = [stage_detector.detect_stage_from_action(action) for action in input_data]
                valid_stages = [s for s in detected_stages if s is not None]
                if valid_stages:
                    stage_counts = {stage: valid_stages.count(stage) for stage in set(valid_stages)}
                    detected_stage = max(stage_counts.items(), key=lambda x: x[1])[0]

            if detected_stage == expected_stage:
                total_weighted_score += weight
            total_weight += weight

        overall_accuracy = total_weighted_score / total_weight if total_weight > 0 else 0.0

        assert overall_accuracy >= 0.90, f"Overall detection accuracy {overall_accuracy:.2%} below 90% target"

    async def test_ninety_percent_accuracy_comprehensive(
        self,
        stage_detector: StageDetector,
    ):
        """
        Comprehensive test demonstrating 90%+ detection accuracy.

        Tests 50+ realistic scenarios across all detection methods.
        """
        correct = 0
        total = 0

        # ACE interventions (15 scenarios, 95% accuracy expected)
        ace_scenarios = [
            ({"intervention_type": "high_error_rate", "metrics": {"error_rate": 0.35}}, StageType.REFLECTION),
            ({"intervention_type": "high_error_rate", "metrics": {"error_rate": 0.5}}, StageType.REFLECTION),
            ({"intervention_type": "slow_progress", "metrics": {"progress_rate": 0.1}}, StageType.PLANNING),
            ({"intervention_type": "slow_progress", "metrics": {"progress_rate": 0.15}}, StageType.PLANNING),
            ({"intervention_type": "quality_issue", "metrics": {"quality_score": 0.5}}, StageType.VERIFICATION),
            ({"metrics": {"error_rate": 0.45}}, StageType.REFLECTION),
            ({"metrics": {"progress_rate": 0.05}}, StageType.PLANNING),
            ({"metrics": {"quality_score": 0.6}}, StageType.VERIFICATION),
            ({"suggested_stage": "planning"}, StageType.PLANNING),
            ({"suggested_stage": "execution"}, StageType.EXECUTION),
            ({"suggested_stage": "reflection"}, StageType.REFLECTION),
            ({"suggested_stage": "verification"}, StageType.VERIFICATION),
            ({"intervention_type": "high_error_rate", "metrics": {"error_rate": 0.4}}, StageType.REFLECTION),
            ({"intervention_type": "slow_progress", "metrics": {"progress_rate": 0.18}}, StageType.PLANNING),
            ({"intervention_type": "quality_issue", "metrics": {"quality_score": 0.65}}, StageType.VERIFICATION),
        ]

        for signal, expected in ace_scenarios:
            detected = await stage_detector.handle_ace_intervention(signal)
            if detected == expected:
                correct += 1
            total += 1

        # Explicit markers (15 scenarios, 100% accuracy expected)
        marker_scenarios = [
            ("[STAGE:PLANNING] Plan", StageType.PLANNING),
            ("@stage:execution Execute", StageType.EXECUTION),
            ("#stage:reflection Reflect", StageType.REFLECTION),
            ("[STAGE:VERIFICATION] Verify", StageType.VERIFICATION),
            ("@stage:planning New plan", StageType.PLANNING),
            ("#stage:execution Run now", StageType.EXECUTION),
            ("[STAGE:REFLECTION] Analyze error", StageType.REFLECTION),
            ("@stage:verification Check quality", StageType.VERIFICATION),
            ("#stage:planning Strategy", StageType.PLANNING),
            ("[STAGE:EXECUTION] Implement", StageType.EXECUTION),
            ("@stage:reflection Review", StageType.REFLECTION),
            ("#stage:verification Test", StageType.VERIFICATION),
            ("[STAGE:PLANNING] Approach", StageType.PLANNING),
            ("@stage:execution Build", StageType.EXECUTION),
            ("#stage:reflection Learn", StageType.REFLECTION),
        ]

        for marker, expected in marker_scenarios:
            detected = stage_detector.detect_from_explicit_marker(marker)
            if detected == expected:
                correct += 1
            total += 1

        # Action patterns (20 scenarios, 85% accuracy expected)
        action_scenarios = [
            ("plan_authentication", StageType.PLANNING),
            ("execute_migration", StageType.EXECUTION),
            ("verify_results", StageType.VERIFICATION),
            ("reflect_on_error", StageType.REFLECTION),
            ("analyze_requirements", StageType.PLANNING),
            ("run_tests", StageType.EXECUTION),
            ("validate_output", StageType.VERIFICATION),
            ("debug_issue", StageType.REFLECTION),
            ("design_architecture", StageType.PLANNING),
            ("implement_feature", StageType.EXECUTION),
            ("confirm_quality", StageType.VERIFICATION),
            ("investigate_failure", StageType.REFLECTION),
            ("evaluate_options", StageType.PLANNING),
            ("deploy_service", StageType.EXECUTION),
            ("audit_security", StageType.VERIFICATION),
            ("learn_from_mistake", StageType.REFLECTION),
            ("brainstorm_solutions", StageType.PLANNING),
            ("build_component", StageType.EXECUTION),
            ("inspect_results", StageType.VERIFICATION),
            ("error_analysis", StageType.REFLECTION),
        ]

        for action, expected in action_scenarios:
            detected = stage_detector.detect_stage_from_action(action)
            if detected == expected:
                correct += 1
            total += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"Comprehensive accuracy {accuracy:.2%} below 90% target (got {correct}/{total})"
