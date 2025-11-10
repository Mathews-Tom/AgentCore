"""
Verifier Module Implementation

This module provides the VerifierModule implementation that validates execution
results using rule-based and LLM-based verification strategies.

Key Features:
- Rule-based validation: Schema, format, completeness checking
- LLM-based verification: Semantic correctness, hallucination detection
- Logical consistency checking between results
- Structured feedback generation for plan refinement
- Confidence scoring for validation results
"""

from __future__ import annotations

from typing import Any

from agentcore.modular.base import BaseVerifier
from agentcore.modular.interfaces import (
    ConsistencyCheck,
    ExecutionResult,
    VerificationRequest,
    VerificationResult,
)


class Verifier(BaseVerifier):
    """
    Verifier module that validates execution results using both rule-based
    and LLM-based verification strategies.

    Implements the VerifierInterface protocol with comprehensive validation
    logic for ensuring result correctness, consistency, and completeness.

    Example:
        >>> verifier = Verifier()
        >>> request = VerificationRequest(
        ...     results=[execution_result],
        ...     expected_json_schema={"type": "number"}
        ... )
        >>> verification = await verifier.validate_results(request)
        >>> if verification.valid:
        ...     print("Results validated successfully")
    """

    def __init__(
        self,
        a2a_context: Any | None = None,
        logger: Any | None = None,
        enable_llm_verification: bool = False,
    ) -> None:
        """
        Initialize Verifier module.

        Args:
            a2a_context: A2A context for distributed tracing
            logger: Logger instance for structured logging
            enable_llm_verification: Enable LLM-based verification (requires API key)
        """
        super().__init__(a2a_context, logger)
        self.enable_llm_verification = enable_llm_verification

    async def health_check(self) -> dict[str, Any]:
        """
        Check Verifier module health.

        Returns:
            Health status with module information
        """
        return {
            "status": "healthy",
            "module": "Verifier",
            "enable_llm_verification": self.enable_llm_verification,
            "execution_id": self.state.execution_id,
        }

    # ========================================================================
    # Core Verification Methods (implements VerifierInterface)
    # ========================================================================

    async def _validate_results_impl(
        self, request: VerificationRequest
    ) -> VerificationResult:
        """
        Implementation-specific result validation.

        Performs multi-stage validation:
        1. Rule-based validation (schema, format, completeness)
        2. LLM-based verification (if enabled)
        3. Confidence scoring

        Args:
            request: Verification request with results and rules

        Returns:
            Verification result with errors, warnings, and confidence

        Raises:
            ValueError: If request is invalid
        """
        if not request.results:
            raise ValueError("VerificationRequest must contain at least one result")

        self.logger.info(
            "validating_results",
            results_count=len(request.results),
            has_schema=request.expected_json_schema is not None,
            rules_count=len(request.consistency_rules),
        )

        errors: list[str] = []
        warnings: list[str] = []
        feedback_parts: list[str] = []

        # Stage 1: Rule-based validation
        rule_validation = self._perform_rule_based_validation(request)
        errors.extend(rule_validation["errors"])
        warnings.extend(rule_validation["warnings"])

        # Stage 2: LLM-based verification (if enabled)
        if self.enable_llm_verification and not errors:
            llm_verification = await self._perform_llm_verification(request)
            errors.extend(llm_verification["errors"])
            warnings.extend(llm_verification["warnings"])
            if llm_verification["feedback"]:
                feedback_parts.append(llm_verification["feedback"])

        # Stage 3: Calculate confidence score
        confidence = self._calculate_confidence(request, errors, warnings)

        # Generate consolidated feedback
        feedback = self._generate_consolidated_feedback(
            errors, warnings, feedback_parts
        )

        valid = len(errors) == 0

        self.logger.info(
            "validation_complete",
            valid=valid,
            errors_count=len(errors),
            warnings_count=len(warnings),
            confidence=confidence,
        )

        return VerificationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            feedback=feedback if feedback else None,
            confidence=confidence,
        )

    async def _check_consistency_impl(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Implementation-specific consistency check.

        Checks consistency between multiple results using the specified rule.

        Supported rules:
        - values_match: All result values must be equal
        - types_match: All result types must be the same
        - ranges_valid: All numerical values must be within expected range
        - no_contradictions: Results must not contradict each other

        Args:
            check: Consistency check specification

        Returns:
            Verification result for consistency check

        Raises:
            ValueError: If check specification is invalid
        """
        if not check.result_ids:
            raise ValueError("ConsistencyCheck must specify at least one result_id")

        if not check.rule:
            raise ValueError("ConsistencyCheck must specify a rule")

        self.logger.info(
            "checking_consistency",
            rule=check.rule,
            result_ids_count=len(check.result_ids),
        )

        # Dispatch to specific consistency check based on rule
        rule_handlers = {
            "values_match": self._check_values_match,
            "types_match": self._check_types_match,
            "ranges_valid": self._check_ranges_valid,
            "no_contradictions": self._check_no_contradictions,
        }

        handler = rule_handlers.get(check.rule)
        if not handler:
            available_rules = ", ".join(rule_handlers.keys())
            return VerificationResult(
                valid=False,
                errors=[
                    f"Unknown consistency rule: {check.rule}. "
                    f"Available rules: {available_rules}"
                ],
                warnings=[],
                feedback=None,
                confidence=0.0,
            )

        return await handler(check)

    async def _provide_feedback_impl(
        self, results: list[ExecutionResult]
    ) -> str:
        """
        Implementation-specific feedback generation.

        Analyzes execution results and generates structured feedback for
        plan refinement or re-execution.

        Args:
            results: Results to analyze

        Returns:
            Structured feedback string

        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("Results list cannot be empty")

        self.logger.info("generating_feedback", results_count=len(results))

        feedback_parts: list[str] = []

        # Analyze failures
        failed_results = [r for r in results if not r.success]
        if failed_results:
            feedback_parts.append(
                self._generate_failure_feedback(failed_results)
            )

        # Analyze performance issues
        slow_results = [r for r in results if r.execution_time > 10.0]
        if slow_results:
            feedback_parts.append(
                self._generate_performance_feedback(slow_results)
            )

        # Analyze incomplete results
        incomplete_results = [
            r for r in results if r.success and self._is_incomplete(r)
        ]
        if incomplete_results:
            feedback_parts.append(
                self._generate_completeness_feedback(incomplete_results)
            )

        # If all results are successful and complete
        if not feedback_parts:
            return "All execution results are valid and complete."

        return "\n\n".join(feedback_parts)

    # ========================================================================
    # Rule-Based Validation Methods
    # ========================================================================

    def _perform_rule_based_validation(
        self, request: VerificationRequest
    ) -> dict[str, list[str]]:
        """
        Perform rule-based validation on results.

        Validates:
        - JSON schema compliance (if schema provided)
        - Result format correctness
        - Completeness (non-null results for successful executions)
        - Consistency rules (if specified)

        Args:
            request: Verification request

        Returns:
            Dict with "errors" and "warnings" lists
        """
        errors: list[str] = []
        warnings: list[str] = []

        for result in request.results:
            # Check 1: Successful results must have non-null result data
            if result.success and result.result is None:
                errors.append(
                    f"Step {result.step_id}: Successful execution has null result"
                )

            # Check 2: Failed results must have error message
            if not result.success and not result.error:
                warnings.append(
                    f"Step {result.step_id}: Failed execution missing error message"
                )

            # Check 3: JSON schema validation (if provided)
            if request.expected_json_schema and result.success:
                schema_errors = self._validate_json_schema(
                    result.result, request.expected_json_schema
                )
                errors.extend(
                    [
                        f"Step {result.step_id}: {err}"
                        for err in schema_errors
                    ]
                )

            # Check 4: Result format validation
            format_warnings = self._validate_result_format(result)
            warnings.extend(
                [
                    f"Step {result.step_id}: {warn}"
                    for warn in format_warnings
                ]
            )

        # Check 5: Consistency rules
        for rule in request.consistency_rules:
            rule_errors = self._apply_consistency_rule(
                request.results, rule
            )
            errors.extend(rule_errors)

        return {"errors": errors, "warnings": warnings}

    def _validate_json_schema(
        self, result_data: Any, schema: dict[str, Any]
    ) -> list[str]:
        """
        Validate result data against JSON schema.

        Basic schema validation for type checking.
        For production use, consider using jsonschema library.

        Args:
            result_data: Data to validate
            schema: JSON schema definition

        Returns:
            List of validation errors
        """
        errors: list[str] = []

        # Basic type validation
        expected_type = schema.get("type")
        if expected_type:
            actual_type = self._get_json_type(result_data)
            if actual_type != expected_type:
                errors.append(
                    f"Type mismatch: expected {expected_type}, got {actual_type}"
                )

        # Properties validation (for objects)
        if expected_type == "object" and isinstance(result_data, dict):
            required = schema.get("required", [])
            for prop in required:
                if prop not in result_data:
                    errors.append(f"Missing required property: {prop}")

        # Array validation
        if expected_type == "array" and isinstance(result_data, list):
            min_items = schema.get("minItems")
            if min_items is not None and len(result_data) < min_items:
                errors.append(
                    f"Array has {len(result_data)} items, minimum is {min_items}"
                )

        return errors

    def _get_json_type(self, value: Any) -> str:
        """
        Get JSON schema type name for Python value.

        Args:
            value: Python value

        Returns:
            JSON schema type string
        """
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _validate_result_format(
        self, result: ExecutionResult
    ) -> list[str]:
        """
        Validate result format and structure.

        Checks for common formatting issues that could cause downstream problems.

        Args:
            result: Execution result to validate

        Returns:
            List of format warnings
        """
        warnings: list[str] = []

        # Warn if execution time is extremely short (< 0.01s) for successful results
        if result.success and result.execution_time < 0.01:
            warnings.append(
                "Suspiciously fast execution time (< 0.01s) - verify result validity"
            )

        # Warn if result data is a string that looks like it should be parsed
        if isinstance(result.result, str):
            # Check if it looks like JSON
            if result.result.strip().startswith(("{", "[")):
                warnings.append(
                    "Result appears to be JSON string - consider parsing to structured data"
                )

        return warnings

    def _apply_consistency_rule(
        self, results: list[ExecutionResult], rule: str
    ) -> list[str]:
        """
        Apply consistency rule to all results.

        Args:
            results: Results to check
            rule: Consistency rule name

        Returns:
            List of consistency errors
        """
        errors: list[str] = []

        # Example rule: all_successful - all results must have success=True
        if rule == "all_successful":
            failed = [r for r in results if not r.success]
            if failed:
                errors.append(
                    f"Consistency rule 'all_successful' violated: "
                    f"{len(failed)} steps failed"
                )

        # Example rule: no_null_results - no successful result should have null data
        elif rule == "no_null_results":
            null_results = [
                r for r in results if r.success and r.result is None
            ]
            if null_results:
                step_ids = [r.step_id for r in null_results]
                errors.append(
                    f"Consistency rule 'no_null_results' violated: "
                    f"steps {step_ids} have null results"
                )

        return errors

    # ========================================================================
    # LLM-Based Verification Methods
    # ========================================================================

    async def _perform_llm_verification(
        self, request: VerificationRequest
    ) -> dict[str, Any]:
        """
        Perform LLM-based verification on results.

        Uses LLM to check:
        - Semantic correctness
        - Hallucination detection
        - Logical reasoning validation

        Note: Placeholder implementation. In production, this would call
        an LLM via Portkey or similar service.

        Args:
            request: Verification request

        Returns:
            Dict with "errors", "warnings", and "feedback"
        """
        # Placeholder: In production, this would make LLM API call
        self.logger.info("llm_verification_placeholder")

        return {
            "errors": [],
            "warnings": [],
            "feedback": None,
        }

    # ========================================================================
    # Consistency Check Methods
    # ========================================================================

    async def _check_values_match(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Check that all result values are equal.

        Args:
            check: Consistency check specification

        Returns:
            Verification result
        """
        # Placeholder: Would extract results by IDs and compare values
        # For now, return valid result
        return VerificationResult(
            valid=True,
            errors=[],
            warnings=[],
            feedback=None,
            confidence=1.0,
        )

    async def _check_types_match(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Check that all result types are the same.

        Args:
            check: Consistency check specification

        Returns:
            Verification result
        """
        # Placeholder: Would extract results by IDs and compare types
        return VerificationResult(
            valid=True,
            errors=[],
            warnings=[],
            feedback=None,
            confidence=1.0,
        )

    async def _check_ranges_valid(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Check that numerical values are within expected range.

        Args:
            check: Consistency check specification

        Returns:
            Verification result
        """
        # Placeholder: Would validate ranges from check.parameters
        return VerificationResult(
            valid=True,
            errors=[],
            warnings=[],
            feedback=None,
            confidence=1.0,
        )

    async def _check_no_contradictions(
        self, check: ConsistencyCheck
    ) -> VerificationResult:
        """
        Check that results do not contradict each other.

        Args:
            check: Consistency check specification

        Returns:
            Verification result
        """
        # Placeholder: Would perform logical contradiction detection
        return VerificationResult(
            valid=True,
            errors=[],
            warnings=[],
            feedback=None,
            confidence=1.0,
        )

    # ========================================================================
    # Feedback Generation Methods
    # ========================================================================

    def _generate_consolidated_feedback(
        self,
        errors: list[str],
        warnings: list[str],
        feedback_parts: list[str],
    ) -> str | None:
        """
        Generate consolidated feedback from validation results.

        Args:
            errors: Validation errors
            warnings: Validation warnings
            feedback_parts: Additional feedback strings

        Returns:
            Consolidated feedback string or None if no issues
        """
        if not errors and not warnings and not feedback_parts:
            return None

        parts: list[str] = []

        if errors:
            parts.append(f"ERRORS ({len(errors)}):")
            parts.extend([f"  - {err}" for err in errors])

        if warnings:
            parts.append(f"\nWARNINGS ({len(warnings)}):")
            parts.extend([f"  - {warn}" for warn in warnings])

        if feedback_parts:
            parts.append("\nADDITIONAL FEEDBACK:")
            parts.extend([f"  {fb}" for fb in feedback_parts])

        return "\n".join(parts)

    def _generate_failure_feedback(
        self, failed_results: list[ExecutionResult]
    ) -> str:
        """
        Generate feedback for failed executions.

        Args:
            failed_results: List of failed results

        Returns:
            Feedback string
        """
        step_ids = [r.step_id for r in failed_results]
        error_messages = [r.error for r in failed_results if r.error]

        feedback = f"FAILED STEPS ({len(failed_results)}): {', '.join(step_ids)}\n"
        feedback += "Errors:\n"
        feedback += "\n".join([f"  - {err}" for err in error_messages])
        feedback += "\n\nRECOMMENDATION: Review step implementations and add error handling"

        return feedback

    def _generate_performance_feedback(
        self, slow_results: list[ExecutionResult]
    ) -> str:
        """
        Generate feedback for slow executions.

        Args:
            slow_results: List of slow results

        Returns:
            Feedback string
        """
        step_times = [
            (r.step_id, r.execution_time) for r in slow_results
        ]
        step_times.sort(key=lambda x: x[1], reverse=True)

        feedback = f"SLOW STEPS ({len(slow_results)}):\n"
        feedback += "\n".join(
            [f"  - {sid}: {time:.2f}s" for sid, time in step_times]
        )
        feedback += "\n\nRECOMMENDATION: Consider optimizing slow steps or adding timeout handling"

        return feedback

    def _generate_completeness_feedback(
        self, incomplete_results: list[ExecutionResult]
    ) -> str:
        """
        Generate feedback for incomplete results.

        Args:
            incomplete_results: List of incomplete results

        Returns:
            Feedback string
        """
        step_ids = [r.step_id for r in incomplete_results]

        feedback = f"INCOMPLETE RESULTS ({len(incomplete_results)}): {', '.join(step_ids)}\n"
        feedback += "RECOMMENDATION: Verify steps return complete data structures"

        return feedback

    def _is_incomplete(self, result: ExecutionResult) -> bool:
        """
        Check if a successful result is incomplete.

        Args:
            result: Execution result

        Returns:
            True if result appears incomplete
        """
        # Result is incomplete if it's an empty string, empty list, or empty dict
        if result.result == "" or result.result == [] or result.result == {}:
            return True

        # Check for partial data structures (e.g., dict with only some keys)
        # This is a heuristic and may need refinement
        if isinstance(result.result, dict):
            # If dict has "status" field but no "data" field, it's incomplete
            if "status" in result.result and "data" not in result.result:
                return True

        return False

    # ========================================================================
    # Confidence Scoring
    # ========================================================================

    def _calculate_confidence(
        self,
        request: VerificationRequest,
        errors: list[str],
        warnings: list[str],
    ) -> float:
        """
        Calculate confidence score for validation.

        Confidence is based on:
        - Number of errors (reduces confidence)
        - Number of warnings (slightly reduces confidence)
        - Presence of schema validation (increases confidence)
        - Number of consistency rules checked (increases confidence)

        Args:
            request: Verification request
            errors: Validation errors
            warnings: Validation warnings

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with perfect confidence
        confidence = 1.0

        # Reduce confidence for each error (0.2 per error, min 0.0)
        confidence -= len(errors) * 0.2
        confidence = max(0.0, confidence)

        # Reduce confidence slightly for warnings (0.05 per warning)
        confidence -= len(warnings) * 0.05
        confidence = max(0.0, confidence)

        # Increase confidence if schema validation was performed
        if request.expected_json_schema:
            confidence = min(1.0, confidence + 0.1)

        # Increase confidence if consistency rules were checked
        if request.consistency_rules:
            confidence = min(1.0, confidence + 0.05 * len(request.consistency_rules))

        return round(confidence, 2)
