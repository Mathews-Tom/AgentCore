"""Tests for comprehensive error categorization system.

Tests the error categorization, error codes, and recovery strategies
as specified in TOOL-021.
"""

import pytest

from agentcore.agent_runtime.tools.errors import (
    ERROR_RECOVERY_STRATEGIES,
    ErrorRecoveryStrategy,
    ToolErrorCategory,
    ToolErrorCode,
    categorize_error,
    get_error_metadata,
)


class TestErrorCategorization:
    """Test error categorization function."""

    def test_categorize_tool_not_found_error(self):
        """Test categorization of ToolNotFoundError."""
        category, code, strategy = categorize_error("ToolNotFoundError")

        assert category == ToolErrorCategory.NOT_FOUND_ERROR
        assert code == ToolErrorCode.TOOL_NOT_FOUND
        assert strategy == ErrorRecoveryStrategy.USER_INTERVENTION

    def test_categorize_authentication_error(self):
        """Test categorization of authentication errors."""
        category, code, strategy = categorize_error("AuthenticationError")

        assert category == ToolErrorCategory.AUTHENTICATION_ERROR
        assert code == ToolErrorCode.INVALID_CREDENTIALS
        assert strategy == ErrorRecoveryStrategy.USER_INTERVENTION

    def test_categorize_authorization_error(self):
        """Test categorization of authorization errors."""
        category, code, strategy = categorize_error("AuthorizationError", "access denied")

        assert category == ToolErrorCategory.AUTHORIZATION_ERROR
        assert code == ToolErrorCode.ACCESS_DENIED
        assert strategy == ErrorRecoveryStrategy.USER_INTERVENTION

    def test_categorize_validation_error(self):
        """Test categorization of validation errors."""
        category, code, strategy = categorize_error("ValidationError", "invalid parameter")

        assert category == ToolErrorCategory.VALIDATION_ERROR
        assert code == ToolErrorCode.INVALID_PARAMETERS
        assert strategy == ErrorRecoveryStrategy.USER_INTERVENTION

    def test_categorize_rate_limit_error(self):
        """Test categorization of rate limit errors."""
        category, code, strategy = categorize_error("RateLimitError", "rate limit exceeded")

        assert category == ToolErrorCategory.RATE_LIMIT_ERROR
        assert code == ToolErrorCode.RATE_LIMIT_EXCEEDED
        assert strategy == ErrorRecoveryStrategy.RETRY_WITH_BACKOFF

    def test_categorize_timeout_error(self):
        """Test categorization of timeout errors."""
        category, code, strategy = categorize_error("TimeoutError", "operation timed out")

        assert category == ToolErrorCategory.TIMEOUT_ERROR
        assert code == ToolErrorCode.EXECUTION_TIMEOUT
        assert strategy == ErrorRecoveryStrategy.RETRY

    def test_categorize_network_error(self):
        """Test categorization of network errors."""
        category, code, strategy = categorize_error("ConnectionError")

        assert category == ToolErrorCategory.NETWORK_ERROR
        assert code == ToolErrorCode.CONNECTION_FAILED
        assert strategy == ErrorRecoveryStrategy.RETRY_WITH_BACKOFF

    def test_categorize_resource_error(self):
        """Test categorization of resource errors."""
        category, code, strategy = categorize_error("MemoryError")

        assert category == ToolErrorCategory.RESOURCE_ERROR
        assert code == ToolErrorCode.OUT_OF_MEMORY
        assert strategy == ErrorRecoveryStrategy.RETRY_WITH_BACKOFF

    def test_categorize_sandbox_error(self):
        """Test categorization of sandbox errors."""
        category, code, strategy = categorize_error("SandboxError", "unsafe operation")

        assert category == ToolErrorCategory.SANDBOX_ERROR
        assert code == ToolErrorCode.SANDBOX_VIOLATION
        assert strategy == ErrorRecoveryStrategy.FAIL

    def test_categorize_unknown_error(self):
        """Test categorization of unknown error types."""
        category, code, strategy = categorize_error("UnknownError")

        assert category == ToolErrorCategory.INTERNAL_ERROR
        assert code == ToolErrorCode.UNEXPECTED_ERROR
        assert strategy == ErrorRecoveryStrategy.FAIL

    def test_categorize_error_with_none_type(self):
        """Test categorization when error type is None."""
        category, code, strategy = categorize_error(None, "some error message")

        # Should default to internal error
        assert category == ToolErrorCategory.INTERNAL_ERROR
        assert code == ToolErrorCode.UNEXPECTED_ERROR
        assert strategy == ErrorRecoveryStrategy.FAIL

    def test_categorize_error_case_insensitive(self):
        """Test that error categorization is case-insensitive."""
        category1, code1, strategy1 = categorize_error("ToolNotFoundError")
        category2, code2, strategy2 = categorize_error("toolnotfounderror")

        assert category1 == category2
        assert code1 == code2
        assert strategy1 == strategy2


class TestErrorMetadata:
    """Test error metadata generation."""

    def test_get_error_metadata_for_validation_error(self):
        """Test metadata generation for validation errors."""
        metadata = get_error_metadata(
            ToolErrorCategory.VALIDATION_ERROR,
            ToolErrorCode.INVALID_PARAMETERS,
            ErrorRecoveryStrategy.USER_INTERVENTION,
        )

        assert metadata["category"] == "validation_error"
        assert metadata["error_code"] == "TOOL_E1001"
        assert metadata["is_retryable"] is False
        assert metadata["recovery_strategy"] == "user_intervention"
        assert "user_message" in metadata
        assert "recovery_guidance" in metadata
        assert "invalid" in metadata["user_message"].lower()

    def test_get_error_metadata_for_rate_limit_error(self):
        """Test metadata generation for rate limit errors."""
        metadata = get_error_metadata(
            ToolErrorCategory.RATE_LIMIT_ERROR,
            ToolErrorCode.RATE_LIMIT_EXCEEDED,
            ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
        )

        assert metadata["category"] == "rate_limit_error"
        assert metadata["error_code"] == "TOOL_E1401"
        assert metadata["is_retryable"] is True
        assert metadata["recovery_strategy"] == "retry_with_backoff"
        assert "rate limit" in metadata["user_message"].lower()
        assert "backoff" in metadata["recovery_guidance"].lower()

    def test_get_error_metadata_for_timeout_error(self):
        """Test metadata generation for timeout errors."""
        metadata = get_error_metadata(
            ToolErrorCategory.TIMEOUT_ERROR,
            ToolErrorCode.EXECUTION_TIMEOUT,
            ErrorRecoveryStrategy.RETRY,
        )

        assert metadata["category"] == "timeout_error"
        assert metadata["error_code"] == "TOOL_E2001"
        assert metadata["is_retryable"] is True
        assert metadata["recovery_strategy"] == "retry"
        assert "timeout" in metadata["user_message"].lower() or "timed out" in metadata["user_message"].lower()

    def test_get_error_metadata_for_sandbox_error(self):
        """Test metadata generation for sandbox errors."""
        metadata = get_error_metadata(
            ToolErrorCategory.SANDBOX_ERROR,
            ToolErrorCode.SANDBOX_VIOLATION,
            ErrorRecoveryStrategy.FAIL,
        )

        assert metadata["category"] == "sandbox_error"
        assert metadata["error_code"] == "TOOL_E3101"
        assert metadata["is_retryable"] is False
        assert metadata["recovery_strategy"] == "fail"
        assert "security" in metadata["user_message"].lower() or "violates" in metadata["user_message"].lower()


class TestErrorRecoveryStrategies:
    """Test error recovery strategy mappings."""

    def test_error_recovery_strategies_mapping(self):
        """Test that all error categories have recovery strategies."""
        # Verify all categories have a recovery strategy
        for category in ToolErrorCategory:
            assert category in ERROR_RECOVERY_STRATEGIES
            strategy = ERROR_RECOVERY_STRATEGIES[category]
            assert isinstance(strategy, ErrorRecoveryStrategy)

    def test_user_intervention_for_client_errors(self):
        """Test that client errors require user intervention."""
        client_errors = [
            ToolErrorCategory.VALIDATION_ERROR,
            ToolErrorCategory.AUTHENTICATION_ERROR,
            ToolErrorCategory.AUTHORIZATION_ERROR,
            ToolErrorCategory.NOT_FOUND_ERROR,
        ]

        for category in client_errors:
            strategy = ERROR_RECOVERY_STRATEGIES[category]
            assert strategy == ErrorRecoveryStrategy.USER_INTERVENTION

    def test_retry_for_transient_errors(self):
        """Test that transient errors use retry strategies."""
        transient_errors = [
            ToolErrorCategory.TIMEOUT_ERROR,
            ToolErrorCategory.NETWORK_ERROR,
            ToolErrorCategory.RESOURCE_ERROR,
            ToolErrorCategory.DEPENDENCY_ERROR,
        ]

        for category in transient_errors:
            strategy = ERROR_RECOVERY_STRATEGIES[category]
            assert strategy in [ErrorRecoveryStrategy.RETRY, ErrorRecoveryStrategy.RETRY_WITH_BACKOFF]


class TestErrorCodes:
    """Test error code organization and uniqueness."""

    def test_error_codes_unique(self):
        """Test that all error codes are unique."""
        codes = [code.value for code in ToolErrorCode]
        assert len(codes) == len(set(codes)), "Error codes must be unique"

    def test_error_code_ranges(self):
        """Test that error codes follow the defined ranges."""
        for code in ToolErrorCode:
            code_num = int(code.value.split("_E")[1])

            # Validation errors: 1000-1099
            if 1000 <= code_num < 1100:
                assert "PARAMETER" in code.name or "REQUIRED" in code.name or code.name.startswith("INVALID")
            # Authentication errors: 1100-1199
            elif 1100 <= code_num < 1200:
                assert "CREDENTIAL" in code.name or "API_KEY" in code.name
            # Authorization errors: 1200-1299
            elif 1200 <= code_num < 1300:
                assert "PERMISSION" in code.name or "ACCESS" in code.name or "AUTHORIZED" in code.name
            # Not found errors: 1300-1399
            elif 1300 <= code_num < 1400:
                assert "NOT_FOUND" in code.name or code.name.startswith("TOOL_NOT") or code.name.startswith("RESOURCE_NOT") or code.name.startswith("FILE_NOT")
            # Rate limit errors: 1400-1499
            elif 1400 <= code_num < 1500:
                assert "LIMIT" in code.name
            # Quota errors: 1500-1599
            elif 1500 <= code_num < 1600:
                assert "QUOTA" in code.name
            # Timeout errors: 2000-2099
            elif 2000 <= code_num < 2100:
                assert "TIMEOUT" in code.name
            # Execution errors: 2100-2199
            elif 2100 <= code_num < 2200:
                assert "EXECUTION" in code.name or "RUNTIME" in code.name or "ASSERTION" in code.name or "COMMAND" in code.name
            # Network errors: 2200-2299
            elif 2200 <= code_num < 2300:
                assert "NETWORK" in code.name or "CONNECTION" in code.name or "DNS" in code.name or "SSL" in code.name
            # Resource errors: 2300-2399
            elif 2300 <= code_num < 2400:
                assert "MEMORY" in code.name or "DISK" in code.name or "CPU" in code.name
            # Dependency errors: 2400-2499
            elif 2400 <= code_num < 2500:
                assert "DEPENDENCY" in code.name or "API" in code.name or "DATABASE" in code.name or "CACHE" in code.name
            # Internal errors: 2500-2599
            elif 2500 <= code_num < 2600:
                assert "ERROR" in code.name or "LOGIC" in code.name or "STATE" in code.name
            # Configuration errors: 3000-3099
            elif 3000 <= code_num < 3100:
                assert "CONFIGURATION" in code.name or "CONFIG" in code.name
            # Sandbox errors: 3100-3199
            elif 3100 <= code_num < 3200:
                assert "SANDBOX" in code.name or "UNSAFE" in code.name or "FILE_SYSTEM" in code.name or "RESTRICTION" in code.name
