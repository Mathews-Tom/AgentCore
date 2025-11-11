"""
Tests for Verifier Module Implementation

Validates result verification, consistency checking, and feedback generation.
"""

from __future__ import annotations

import pytest

from agentcore.modular.interfaces import (
    ConsistencyCheck,
    ExecutionResult,
    VerificationRequest,
)
from agentcore.modular.verifier import Verifier


class TestVerifierInitialization:
    """Test Verifier module initialization."""

    def test_verifier_default_initialization(self) -> None:
        """Test Verifier with default configuration."""
        verifier = Verifier()

        assert verifier.module_name == "Verifier"
        assert verifier.enable_llm_verification is False
        assert verifier.state is not None

    def test_verifier_custom_configuration(self) -> None:
        """Test Verifier with custom configuration."""
        verifier = Verifier(enable_llm_verification=True)

        assert verifier.enable_llm_verification is True

    @pytest.mark.asyncio
    async def test_verifier_health_check(self) -> None:
        """Test Verifier health check."""
        verifier = Verifier()
        health = await verifier.health_check()

        assert health["status"] == "healthy"
        assert health["module"] == "Verifier"
        assert health["enable_llm_verification"] is False
        assert "execution_id" in health


class TestValidateResults:
    """Test result validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_empty_results_raises_error(self) -> None:
        """Test that empty results list raises ValueError."""
        verifier = Verifier()
        request = VerificationRequest(results=[])

        with pytest.raises(ValueError, match="must contain at least one result"):
            await verifier.validate_results(request)

    @pytest.mark.asyncio
    async def test_validate_successful_result(self) -> None:
        """Test validation of successful result."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.valid is True
        assert len(verification.errors) == 0
        assert verification.confidence > 0.0

    @pytest.mark.asyncio
    async def test_validate_successful_result_with_null_data_fails(self) -> None:
        """Test that successful result with null data fails validation."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=None,
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.valid is False
        assert len(verification.errors) == 1
        assert "null result" in verification.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_failed_result_without_error_message(self) -> None:
        """Test that failed result without error message generates warning."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=False,
            result=None,
            error=None,
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.valid is True  # Warnings don't invalidate
        assert len(verification.warnings) == 1
        assert "missing error message" in verification.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_validate_with_json_schema_valid(self) -> None:
        """Test validation with valid JSON schema."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=42,
            execution_time=0.5,
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={"type": "integer"},
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is True
        assert len(verification.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_with_json_schema_invalid_type(self) -> None:
        """Test validation with invalid JSON schema type."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result="string_value",
            execution_time=0.5,
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={"type": "number"},
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is False
        assert len(verification.errors) == 1
        assert "type mismatch" in verification.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_object_schema_missing_required(self) -> None:
        """Test validation with missing required object properties."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"name": "test"},
            execution_time=0.5,
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={
                "type": "object",
                "required": ["name", "age"],
            },
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is False
        assert len(verification.errors) == 1
        assert "missing required property: age" in verification.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_array_schema_min_items(self) -> None:
        """Test validation with array minItems constraint."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=[1, 2],
            execution_time=0.5,
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={
                "type": "array",
                "minItems": 5,
            },
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is False
        assert len(verification.errors) == 1
        assert "minimum is 5" in verification.errors[0]

    @pytest.mark.asyncio
    async def test_validate_consistency_rule_all_successful(self) -> None:
        """Test consistency rule: all_successful."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=False,
                result=None,
                error="Failed",
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 1},
                execution_time=0.5,
            ),
        ]
        request = VerificationRequest(
            results=results,
            consistency_rules=["all_successful"],
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is False
        assert any("all_successful" in err for err in verification.errors)

    @pytest.mark.asyncio
    async def test_validate_consistency_rule_no_null_results(self) -> None:
        """Test consistency rule: no_null_results."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result=None,
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 1},
                execution_time=0.5,
            ),
        ]
        request = VerificationRequest(
            results=results,
            consistency_rules=["no_null_results"],
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is False
        assert any("no_null_results" in err for err in verification.errors)


class TestResultFormatValidation:
    """Test result format validation warnings."""

    @pytest.mark.asyncio
    async def test_warn_suspiciously_fast_execution(self) -> None:
        """Test warning for extremely fast execution time."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.001,  # 0.001s is < 0.01s threshold
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.valid is True
        assert len(verification.warnings) == 1
        assert "suspiciously fast" in verification.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_warn_json_string_result(self) -> None:
        """Test warning for JSON string that should be parsed."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result='{"value": 42}',  # JSON string instead of dict
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.valid is True
        assert len(verification.warnings) == 1
        assert "json string" in verification.warnings[0].lower()


class TestConsistencyChecking:
    """Test consistency checking functionality."""

    @pytest.mark.asyncio
    async def test_check_consistency_empty_result_ids_raises_error(self) -> None:
        """Test that empty result_ids raises ValueError."""
        verifier = Verifier()
        check = ConsistencyCheck(result_ids=[], rule="values_match")

        with pytest.raises(ValueError, match="at least one result_id"):
            await verifier.check_consistency(check)

    @pytest.mark.asyncio
    async def test_check_consistency_empty_rule_raises_error(self) -> None:
        """Test that empty rule raises ValueError."""
        verifier = Verifier()
        check = ConsistencyCheck(result_ids=["r1", "r2"], rule="")

        with pytest.raises(ValueError, match="must specify a rule"):
            await verifier.check_consistency(check)

    @pytest.mark.asyncio
    async def test_check_consistency_unknown_rule(self) -> None:
        """Test handling of unknown consistency rule."""
        verifier = Verifier()
        check = ConsistencyCheck(
            result_ids=["r1", "r2"],
            rule="unknown_rule",
        )

        verification = await verifier.check_consistency(check)

        assert verification.valid is False
        assert len(verification.errors) == 1
        assert "unknown consistency rule" in verification.errors[0].lower()
        assert "available rules" in verification.errors[0].lower()

    @pytest.mark.asyncio
    async def test_check_consistency_values_match(self) -> None:
        """Test values_match consistency rule."""
        verifier = Verifier()
        check = ConsistencyCheck(
            result_ids=["r1", "r2"],
            rule="values_match",
        )

        verification = await verifier.check_consistency(check)

        # Placeholder implementation returns valid
        assert verification.valid is True
        assert verification.confidence == 1.0

    @pytest.mark.asyncio
    async def test_check_consistency_types_match(self) -> None:
        """Test types_match consistency rule."""
        verifier = Verifier()
        check = ConsistencyCheck(
            result_ids=["r1", "r2"],
            rule="types_match",
        )

        verification = await verifier.check_consistency(check)

        # Placeholder implementation returns valid
        assert verification.valid is True
        assert verification.confidence == 1.0

    @pytest.mark.asyncio
    async def test_check_consistency_ranges_valid(self) -> None:
        """Test ranges_valid consistency rule."""
        verifier = Verifier()
        check = ConsistencyCheck(
            result_ids=["r1", "r2"],
            rule="ranges_valid",
            parameters={"min": 0, "max": 100},
        )

        verification = await verifier.check_consistency(check)

        # Placeholder implementation returns valid
        assert verification.valid is True
        assert verification.confidence == 1.0

    @pytest.mark.asyncio
    async def test_check_consistency_no_contradictions(self) -> None:
        """Test no_contradictions consistency rule."""
        verifier = Verifier()
        check = ConsistencyCheck(
            result_ids=["r1", "r2"],
            rule="no_contradictions",
        )

        verification = await verifier.check_consistency(check)

        # Placeholder implementation returns valid
        assert verification.valid is True
        assert verification.confidence == 1.0


class TestFeedbackGeneration:
    """Test feedback generation functionality."""

    @pytest.mark.asyncio
    async def test_provide_feedback_empty_results_raises_error(self) -> None:
        """Test that empty results list raises ValueError."""
        verifier = Verifier()

        with pytest.raises(ValueError, match="cannot be empty"):
            await verifier.provide_feedback([])

    @pytest.mark.asyncio
    async def test_provide_feedback_all_successful(self) -> None:
        """Test feedback for all successful results."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.6,
            ),
        ]

        feedback = await verifier.provide_feedback(results)

        assert feedback == "All execution results are valid and complete."

    @pytest.mark.asyncio
    async def test_provide_feedback_with_failures(self) -> None:
        """Test feedback generation for failed results."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=False,
                result=None,
                error="Connection timeout",
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.6,
            ),
        ]

        feedback = await verifier.provide_feedback(results)

        assert "FAILED STEPS" in feedback
        assert "step_1" in feedback
        assert "Connection timeout" in feedback
        assert "RECOMMENDATION" in feedback

    @pytest.mark.asyncio
    async def test_provide_feedback_with_slow_results(self) -> None:
        """Test feedback generation for slow executions."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=15.0,  # > 10s threshold
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.5,
            ),
        ]

        feedback = await verifier.provide_feedback(results)

        assert "SLOW STEPS" in feedback
        assert "step_1" in feedback
        assert "15.00s" in feedback
        assert "RECOMMENDATION" in feedback

    @pytest.mark.asyncio
    async def test_provide_feedback_with_incomplete_results(self) -> None:
        """Test feedback generation for incomplete results."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result="",  # Empty string = incomplete
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result=[],  # Empty list = incomplete
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_3",
                success=True,
                result={},  # Empty dict = incomplete
                execution_time=0.5,
            ),
        ]

        feedback = await verifier.provide_feedback(results)

        assert "INCOMPLETE RESULTS" in feedback
        assert "step_1" in feedback
        assert "step_2" in feedback
        assert "step_3" in feedback

    @pytest.mark.asyncio
    async def test_provide_feedback_partial_data_structure(self) -> None:
        """Test feedback for partial data structures."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"status": "ok"},  # Has status but no data
                execution_time=0.5,
            ),
        ]

        feedback = await verifier.provide_feedback(results)

        assert "INCOMPLETE RESULTS" in feedback
        assert "step_1" in feedback


class TestConfidenceScoring:
    """Test confidence score calculation."""

    @pytest.mark.asyncio
    async def test_confidence_perfect_validation(self) -> None:
        """Test confidence score for perfect validation."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=42,
            execution_time=0.5,
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={"type": "integer"},
        )

        verification = await verifier.validate_results(request)

        # Perfect validation + schema check
        assert verification.confidence >= 1.0

    @pytest.mark.asyncio
    async def test_confidence_with_errors(self) -> None:
        """Test confidence score decreases with errors."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=None,  # Causes error
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        # Errors reduce confidence by 0.2 per error
        assert verification.confidence < 1.0
        assert verification.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_confidence_with_warnings(self) -> None:
        """Test confidence score decreases slightly with warnings."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.001,  # Causes warning
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        # Warnings reduce confidence by 0.05 per warning
        assert verification.confidence < 1.0
        assert verification.confidence > 0.9  # Should be high despite warning

    @pytest.mark.asyncio
    async def test_confidence_with_consistency_rules(self) -> None:
        """Test confidence increases with consistency rules."""
        verifier = Verifier()
        results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=0.5,
            ),
            ExecutionResult(
                step_id="step_2",
                success=True,
                result={"value": 43},
                execution_time=0.5,
            ),
        ]
        request = VerificationRequest(
            results=results,
            consistency_rules=["all_successful"],
        )

        verification = await verifier.validate_results(request)

        # Consistency rules increase confidence
        assert verification.confidence >= 1.0


class TestLLMVerification:
    """Test LLM-based verification (placeholder)."""

    @pytest.mark.asyncio
    async def test_llm_verification_disabled_by_default(self) -> None:
        """Test that LLM verification is disabled by default."""
        verifier = Verifier()
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        # Should succeed without LLM verification
        assert verification.valid is True

    @pytest.mark.asyncio
    async def test_llm_verification_enabled_placeholder(self) -> None:
        """Test that LLM verification placeholder doesn't break flow."""
        verifier = Verifier(enable_llm_verification=True)
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result={"value": 42},
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        # Placeholder returns no errors, should pass
        assert verification.valid is True


class TestJsonSchemaHelpers:
    """Test JSON schema validation helpers."""

    def test_get_json_type_null(self) -> None:
        """Test JSON type detection for null."""
        verifier = Verifier()
        assert verifier._get_json_type(None) == "null"

    def test_get_json_type_boolean(self) -> None:
        """Test JSON type detection for boolean."""
        verifier = Verifier()
        assert verifier._get_json_type(True) == "boolean"
        assert verifier._get_json_type(False) == "boolean"

    def test_get_json_type_integer(self) -> None:
        """Test JSON type detection for integer."""
        verifier = Verifier()
        assert verifier._get_json_type(42) == "integer"
        assert verifier._get_json_type(0) == "integer"

    def test_get_json_type_number(self) -> None:
        """Test JSON type detection for number (float)."""
        verifier = Verifier()
        assert verifier._get_json_type(3.14) == "number"

    def test_get_json_type_string(self) -> None:
        """Test JSON type detection for string."""
        verifier = Verifier()
        assert verifier._get_json_type("hello") == "string"

    def test_get_json_type_array(self) -> None:
        """Test JSON type detection for array."""
        verifier = Verifier()
        assert verifier._get_json_type([1, 2, 3]) == "array"

    def test_get_json_type_object(self) -> None:
        """Test JSON type detection for object."""
        verifier = Verifier()
        assert verifier._get_json_type({"key": "value"}) == "object"

    def test_get_json_type_unknown(self) -> None:
        """Test JSON type detection for unknown types."""
        verifier = Verifier()
        assert verifier._get_json_type(object()) == "unknown"


class TestFeedbackHelpers:
    """Test feedback generation helpers."""

    def test_generate_failure_feedback(self) -> None:
        """Test failure feedback generation."""
        verifier = Verifier()
        failed_results = [
            ExecutionResult(
                step_id="step_1",
                success=False,
                result=None,
                error="Timeout error",
                execution_time=0.5,
            ),
        ]

        feedback = verifier._generate_failure_feedback(failed_results)

        assert "FAILED STEPS" in feedback
        assert "step_1" in feedback
        assert "Timeout error" in feedback

    def test_generate_performance_feedback(self) -> None:
        """Test performance feedback generation."""
        verifier = Verifier()
        slow_results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result={"value": 42},
                execution_time=15.5,
            ),
        ]

        feedback = verifier._generate_performance_feedback(slow_results)

        assert "SLOW STEPS" in feedback
        assert "step_1" in feedback
        assert "15.50s" in feedback

    def test_generate_completeness_feedback(self) -> None:
        """Test completeness feedback generation."""
        verifier = Verifier()
        incomplete_results = [
            ExecutionResult(
                step_id="step_1",
                success=True,
                result="",
                execution_time=0.5,
            ),
        ]

        feedback = verifier._generate_completeness_feedback(incomplete_results)

        assert "INCOMPLETE RESULTS" in feedback
        assert "step_1" in feedback

    def test_is_incomplete_empty_structures(self) -> None:
        """Test incomplete detection for empty data structures."""
        verifier = Verifier()

        empty_string = ExecutionResult(
            step_id="s1", success=True, result="", execution_time=0.5
        )
        empty_list = ExecutionResult(
            step_id="s2", success=True, result=[], execution_time=0.5
        )
        empty_dict = ExecutionResult(
            step_id="s3", success=True, result={}, execution_time=0.5
        )

        assert verifier._is_incomplete(empty_string) is True
        assert verifier._is_incomplete(empty_list) is True
        assert verifier._is_incomplete(empty_dict) is True

    def test_is_incomplete_partial_dict(self) -> None:
        """Test incomplete detection for partial dict."""
        verifier = Verifier()

        partial_result = ExecutionResult(
            step_id="s1",
            success=True,
            result={"status": "ok"},
            execution_time=0.5,
        )
        complete_result = ExecutionResult(
            step_id="s2",
            success=True,
            result={"status": "ok", "data": [1, 2, 3]},
            execution_time=0.5,
        )

        assert verifier._is_incomplete(partial_result) is True
        assert verifier._is_incomplete(complete_result) is False


class TestConsolidatedFeedback:
    """Test consolidated feedback generation."""

    def test_consolidated_feedback_none_for_no_issues(self) -> None:
        """Test that no feedback is generated for clean results."""
        verifier = Verifier()

        feedback = verifier._generate_consolidated_feedback(
            [], [], [], refinement_needed=False, confidence=1.0
        )

        assert feedback is None

    def test_consolidated_feedback_with_errors(self) -> None:
        """Test consolidated feedback with errors."""
        verifier = Verifier()

        feedback = verifier._generate_consolidated_feedback(
            errors=["Error 1", "Error 2"],
            warnings=[],
            feedback_parts=[],
            refinement_needed=False,
            confidence=0.6,
        )

        assert feedback is not None
        assert "ERRORS (2)" in feedback
        assert "Error 1" in feedback
        assert "Error 2" in feedback

    def test_consolidated_feedback_with_warnings(self) -> None:
        """Test consolidated feedback with warnings."""
        verifier = Verifier()

        feedback = verifier._generate_consolidated_feedback(
            errors=[],
            warnings=["Warning 1"],
            feedback_parts=[],
            refinement_needed=False,
            confidence=0.95,
        )

        assert feedback is not None
        assert "WARNINGS (1)" in feedback
        assert "Warning 1" in feedback

    def test_consolidated_feedback_with_additional_parts(self) -> None:
        """Test consolidated feedback with additional feedback parts."""
        verifier = Verifier()

        feedback = verifier._generate_consolidated_feedback(
            errors=[],
            warnings=[],
            feedback_parts=["Additional feedback"],
            refinement_needed=False,
            confidence=1.0,
        )

        assert feedback is not None
        assert "ADDITIONAL FEEDBACK" in feedback
        assert "Additional feedback" in feedback

    def test_consolidated_feedback_with_all_parts(self) -> None:
        """Test consolidated feedback with all parts."""
        verifier = Verifier()

        feedback = verifier._generate_consolidated_feedback(
            errors=["Error 1"],
            warnings=["Warning 1"],
            feedback_parts=["Additional feedback"],
            refinement_needed=False,
            confidence=0.8,
        )

        assert feedback is not None
        assert "ERRORS" in feedback
        assert "WARNINGS" in feedback
        assert "ADDITIONAL FEEDBACK" in feedback


class TestConfidenceThreshold:
    """Test confidence threshold configuration and refinement logic (MOD-014)."""

    def test_default_confidence_threshold(self) -> None:
        """Test that default confidence threshold is 0.7."""
        verifier = Verifier()

        assert verifier.confidence_threshold == 0.7

    def test_custom_confidence_threshold(self) -> None:
        """Test setting custom confidence threshold."""
        verifier = Verifier(confidence_threshold=0.9)

        assert verifier.confidence_threshold == 0.9

    def test_confidence_threshold_validation_low(self) -> None:
        """Test that confidence threshold must be >= 0.0."""
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            Verifier(confidence_threshold=-0.1)

    def test_confidence_threshold_validation_high(self) -> None:
        """Test that confidence threshold must be <= 1.0."""
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            Verifier(confidence_threshold=1.1)

    def test_meets_confidence_threshold_above(self) -> None:
        """Test that confidence above threshold meets it."""
        verifier = Verifier(confidence_threshold=0.7)

        assert verifier.meets_confidence_threshold(0.8) is True
        assert verifier.meets_confidence_threshold(0.7) is True

    def test_meets_confidence_threshold_below(self) -> None:
        """Test that confidence below threshold does not meet it."""
        verifier = Verifier(confidence_threshold=0.7)

        assert verifier.meets_confidence_threshold(0.6) is False
        assert verifier.meets_confidence_threshold(0.0) is False

    def test_meets_confidence_threshold_validation(self) -> None:
        """Test that meets_confidence_threshold validates input."""
        verifier = Verifier()

        with pytest.raises(ValueError, match="Confidence must be between"):
            verifier.meets_confidence_threshold(-0.1)

        with pytest.raises(ValueError, match="Confidence must be between"):
            verifier.meets_confidence_threshold(1.5)

    async def test_refinement_needed_for_low_confidence(self) -> None:
        """Test that refinement_needed is set when confidence is low."""
        verifier = Verifier(confidence_threshold=0.85)
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=None,  # Causes error, reduces confidence by 0.2
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        # This should fail validation and have low confidence (0.8 < 0.85)
        assert verification.valid is False
        assert verification.confidence == 0.8  # 1.0 - 0.2 = 0.8
        assert verification.refinement_needed is True

    async def test_refinement_not_needed_for_high_confidence(self) -> None:
        """Test that refinement_needed is not set when confidence is high."""
        verifier = Verifier(confidence_threshold=0.7)
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=42,
            execution_time=0.5,
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={"type": "integer"},
        )

        verification = await verifier.validate_results(request)

        assert verification.valid is True
        assert verification.confidence >= 0.7
        assert verification.refinement_needed is False

    async def test_low_confidence_feedback_message(self) -> None:
        """Test that low confidence includes refinement feedback."""
        verifier = Verifier(confidence_threshold=0.85)
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result=None,  # Causes error, confidence = 0.8
            execution_time=0.5,
        )
        request = VerificationRequest(results=[result])

        verification = await verifier.validate_results(request)

        assert verification.refinement_needed is True
        assert verification.feedback is not None
        assert "LOW CONFIDENCE" in verification.feedback
        assert "Plan refinement needed" in verification.feedback
        assert "0.85" in verification.feedback

    async def test_high_threshold_requires_better_validation(self) -> None:
        """Test that higher threshold requires better validation."""
        # With low threshold (0.5), warnings might still pass
        verifier_low = Verifier(confidence_threshold=0.5)
        result = ExecutionResult(
            step_id="step_1",
            success=True,
            result="42",  # String instead of number (causes warning)
            execution_time=0.001,  # Very fast (causes warning)
        )
        request = VerificationRequest(
            results=[result],
            expected_json_schema={"type": "integer"},
        )

        verification_low = await verifier_low.validate_results(request)

        # Low threshold should not trigger refinement despite warnings
        # Note: This depends on exact confidence calculation
        assert verification_low.confidence >= 0.5

        # With high threshold (0.95), same validation might need refinement
        verifier_high = Verifier(confidence_threshold=0.95)
        verification_high = await verifier_high.validate_results(request)

        # High threshold might trigger refinement with schema mismatch
        if verification_high.confidence < 0.95:
            assert verification_high.refinement_needed is True

    def test_health_check_includes_threshold(self) -> None:
        """Test that health check includes confidence threshold."""
        verifier = Verifier(confidence_threshold=0.85)

        health = asyncio.run(verifier.health_check())

        assert health["confidence_threshold"] == 0.85

    def test_consolidated_feedback_with_low_confidence(self) -> None:
        """Test consolidated feedback generation with low confidence."""
        verifier = Verifier(confidence_threshold=0.7)

        feedback = verifier._generate_consolidated_feedback(
            errors=[],
            warnings=["Some warning"],
            feedback_parts=[],
            refinement_needed=True,
            confidence=0.5,
        )

        assert feedback is not None
        assert "LOW CONFIDENCE: 0.50" in feedback
        assert "threshold: 0.70" in feedback
        assert "Plan refinement needed" in feedback


# Import pytest at top if not already imported
import pytest
import asyncio
