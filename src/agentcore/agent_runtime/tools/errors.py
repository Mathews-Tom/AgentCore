"""Comprehensive error categorization system for tool execution.

Implements FR-2.7: Error Handling from docs/specs/tool-integration/spec.md
Provides standardized error categories, codes, and recovery strategies.
"""

from enum import Enum
from typing import Any


class ToolErrorCategory(str, Enum):
    """High-level error categories for tool execution failures."""

    # User/Input Errors (4xx equivalent - client errors)
    VALIDATION_ERROR = "validation_error"  # Invalid parameters, schema mismatch
    AUTHENTICATION_ERROR = "authentication_error"  # Missing or invalid credentials
    AUTHORIZATION_ERROR = "authorization_error"  # Insufficient permissions
    NOT_FOUND_ERROR = "not_found_error"  # Tool or resource not found
    RATE_LIMIT_ERROR = "rate_limit_error"  # Rate limit exceeded
    QUOTA_ERROR = "quota_error"  # Usage quota exceeded

    # Execution Errors (5xx equivalent - server/runtime errors)
    TIMEOUT_ERROR = "timeout_error"  # Execution timeout
    EXECUTION_ERROR = "execution_error"  # Tool execution failed
    NETWORK_ERROR = "network_error"  # Network/connectivity issues
    RESOURCE_ERROR = "resource_error"  # Insufficient resources (memory, disk, etc.)
    DEPENDENCY_ERROR = "dependency_error"  # External dependency failed
    INTERNAL_ERROR = "internal_error"  # Unexpected internal error

    # Configuration Errors
    CONFIGURATION_ERROR = "configuration_error"  # Tool misconfiguration
    SANDBOX_ERROR = "sandbox_error"  # Sandbox/security constraint violation


class ToolErrorCode(str, Enum):
    """Detailed error codes for specific failure scenarios."""

    # Validation Errors (1000-1099)
    INVALID_PARAMETERS = "TOOL_E1001"  # Invalid parameter values
    MISSING_REQUIRED_PARAMETER = "TOOL_E1002"  # Required parameter missing
    PARAMETER_TYPE_MISMATCH = "TOOL_E1003"  # Parameter type doesn't match schema
    PARAMETER_OUT_OF_RANGE = "TOOL_E1004"  # Parameter value out of valid range
    INVALID_PARAMETER_FORMAT = "TOOL_E1005"  # Parameter format invalid (e.g., bad URL)

    # Authentication Errors (1100-1199)
    MISSING_CREDENTIALS = "TOOL_E1101"  # No credentials provided
    INVALID_CREDENTIALS = "TOOL_E1102"  # Credentials invalid or expired
    CREDENTIAL_EXPIRED = "TOOL_E1103"  # Credentials expired
    INVALID_API_KEY = "TOOL_E1104"  # API key invalid

    # Authorization Errors (1200-1299)
    INSUFFICIENT_PERMISSIONS = "TOOL_E1201"  # User lacks required permissions
    ACCESS_DENIED = "TOOL_E1202"  # Access to resource denied
    AGENT_NOT_AUTHORIZED = "TOOL_E1203"  # Agent not authorized for this tool

    # Not Found Errors (1300-1399)
    TOOL_NOT_FOUND = "TOOL_E1301"  # Tool ID not registered
    RESOURCE_NOT_FOUND = "TOOL_E1302"  # External resource not found
    FILE_NOT_FOUND = "TOOL_E1303"  # File not found

    # Rate Limit Errors (1400-1499)
    RATE_LIMIT_EXCEEDED = "TOOL_E1401"  # Rate limit exceeded
    CONCURRENT_LIMIT_EXCEEDED = "TOOL_E1402"  # Too many concurrent executions

    # Quota Errors (1500-1599)
    QUOTA_EXCEEDED = "TOOL_E1501"  # Usage quota exceeded
    STORAGE_QUOTA_EXCEEDED = "TOOL_E1502"  # Storage quota exceeded
    API_CALL_QUOTA_EXCEEDED = "TOOL_E1503"  # API call quota exceeded

    # Timeout Errors (2000-2099)
    EXECUTION_TIMEOUT = "TOOL_E2001"  # Tool execution timeout
    NETWORK_TIMEOUT = "TOOL_E2002"  # Network request timeout
    DEPENDENCY_TIMEOUT = "TOOL_E2003"  # External dependency timeout

    # Execution Errors (2100-2199)
    EXECUTION_FAILED = "TOOL_E2101"  # General execution failure
    RUNTIME_ERROR = "TOOL_E2102"  # Runtime error during execution
    ASSERTION_FAILED = "TOOL_E2103"  # Assertion or validation failed
    EXTERNAL_COMMAND_FAILED = "TOOL_E2104"  # External command failed

    # Network Errors (2200-2299)
    NETWORK_UNREACHABLE = "TOOL_E2201"  # Network unreachable
    CONNECTION_FAILED = "TOOL_E2202"  # Connection failed
    DNS_RESOLUTION_FAILED = "TOOL_E2203"  # DNS resolution failed
    SSL_ERROR = "TOOL_E2204"  # SSL/TLS error

    # Resource Errors (2300-2399)
    OUT_OF_MEMORY = "TOOL_E2301"  # Insufficient memory
    DISK_FULL = "TOOL_E2302"  # Disk space exhausted
    CPU_LIMIT_EXCEEDED = "TOOL_E2303"  # CPU limit exceeded

    # Dependency Errors (2400-2499)
    DEPENDENCY_UNAVAILABLE = "TOOL_E2401"  # External dependency unavailable
    API_ERROR = "TOOL_E2402"  # External API error
    DATABASE_ERROR = "TOOL_E2403"  # Database error
    CACHE_ERROR = "TOOL_E2404"  # Cache service error

    # Internal Errors (2500-2599)
    UNEXPECTED_ERROR = "TOOL_E2501"  # Unexpected internal error
    LOGIC_ERROR = "TOOL_E2502"  # Logic error in tool implementation
    STATE_ERROR = "TOOL_E2503"  # Invalid state transition

    # Configuration Errors (3000-3099)
    MISCONFIGURATION = "TOOL_E3001"  # Tool misconfigured
    MISSING_CONFIGURATION = "TOOL_E3002"  # Required configuration missing
    INVALID_CONFIGURATION = "TOOL_E3003"  # Configuration invalid

    # Sandbox Errors (3100-3199)
    SANDBOX_VIOLATION = "TOOL_E3101"  # Sandbox security constraint violated
    UNSAFE_OPERATION = "TOOL_E3102"  # Operation not allowed in sandbox
    FILE_SYSTEM_RESTRICTION = "TOOL_E3103"  # File system access restricted


class ErrorRecoveryStrategy(str, Enum):
    """Recovery strategies for different error categories."""

    RETRY = "retry"  # Retry the operation (for transient errors)
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    FALLBACK = "fallback"  # Use fallback tool or method
    USER_INTERVENTION = "user_intervention"  # Require user to fix and retry
    FAIL = "fail"  # Fail immediately, no recovery possible


# Error category to recovery strategy mapping
ERROR_RECOVERY_STRATEGIES: dict[ToolErrorCategory, ErrorRecoveryStrategy] = {
    # Client errors - typically need user intervention
    ToolErrorCategory.VALIDATION_ERROR: ErrorRecoveryStrategy.USER_INTERVENTION,
    ToolErrorCategory.AUTHENTICATION_ERROR: ErrorRecoveryStrategy.USER_INTERVENTION,
    ToolErrorCategory.AUTHORIZATION_ERROR: ErrorRecoveryStrategy.USER_INTERVENTION,
    ToolErrorCategory.NOT_FOUND_ERROR: ErrorRecoveryStrategy.USER_INTERVENTION,
    # Rate limits - retry with backoff
    ToolErrorCategory.RATE_LIMIT_ERROR: ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
    ToolErrorCategory.QUOTA_ERROR: ErrorRecoveryStrategy.FAIL,  # Can't recover from quota
    # Transient errors - can retry
    ToolErrorCategory.TIMEOUT_ERROR: ErrorRecoveryStrategy.RETRY,
    ToolErrorCategory.NETWORK_ERROR: ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
    ToolErrorCategory.RESOURCE_ERROR: ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
    ToolErrorCategory.DEPENDENCY_ERROR: ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
    # Fatal errors - fail or fallback
    ToolErrorCategory.EXECUTION_ERROR: ErrorRecoveryStrategy.FALLBACK,
    ToolErrorCategory.INTERNAL_ERROR: ErrorRecoveryStrategy.FAIL,
    ToolErrorCategory.CONFIGURATION_ERROR: ErrorRecoveryStrategy.USER_INTERVENTION,
    ToolErrorCategory.SANDBOX_ERROR: ErrorRecoveryStrategy.FAIL,
}


def categorize_error(
    error_type: str | None,
    error_message: str | None = None,
) -> tuple[ToolErrorCategory, ToolErrorCode, ErrorRecoveryStrategy]:
    """Categorize an error based on error type and message.

    Args:
        error_type: Error type string (e.g., "ToolNotFoundError", "TimeoutError")
        error_message: Optional error message for additional context

    Returns:
        Tuple of (category, error_code, recovery_strategy)

    Example:
        >>> category, code, strategy = categorize_error("ToolNotFoundError")
        >>> assert category == ToolErrorCategory.NOT_FOUND_ERROR
        >>> assert code == ToolErrorCode.TOOL_NOT_FOUND
        >>> assert strategy == ErrorRecoveryStrategy.USER_INTERVENTION
    """
    error_type = error_type or ""
    error_message = (error_message or "").lower()

    # Tool Not Found
    if "toolnotfound" in error_type.lower() or "tool" in error_message and "not found" in error_message:
        return (
            ToolErrorCategory.NOT_FOUND_ERROR,
            ToolErrorCode.TOOL_NOT_FOUND,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.NOT_FOUND_ERROR],
        )

    # Authentication Errors
    if "authentication" in error_type.lower() or "auth" in error_message:
        return (
            ToolErrorCategory.AUTHENTICATION_ERROR,
            ToolErrorCode.INVALID_CREDENTIALS,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.AUTHENTICATION_ERROR],
        )

    # Authorization Errors
    if "authorization" in error_type.lower() or "permission" in error_message or "access denied" in error_message:
        return (
            ToolErrorCategory.AUTHORIZATION_ERROR,
            ToolErrorCode.ACCESS_DENIED,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.AUTHORIZATION_ERROR],
        )

    # Validation Errors
    if "validation" in error_type.lower() or "invalid param" in error_message or "parameter" in error_message:
        return (
            ToolErrorCategory.VALIDATION_ERROR,
            ToolErrorCode.INVALID_PARAMETERS,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.VALIDATION_ERROR],
        )

    # Rate Limit Errors
    if "ratelimit" in error_type.lower() or "rate limit" in error_message:
        return (
            ToolErrorCategory.RATE_LIMIT_ERROR,
            ToolErrorCode.RATE_LIMIT_EXCEEDED,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.RATE_LIMIT_ERROR],
        )

    # Timeout Errors
    if "timeout" in error_type.lower() or "timed out" in error_message:
        return (
            ToolErrorCategory.TIMEOUT_ERROR,
            ToolErrorCode.EXECUTION_TIMEOUT,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.TIMEOUT_ERROR],
        )

    # Network Errors
    if any(keyword in error_type.lower() for keyword in ["connection", "network", "dns", "ssl"]):
        return (
            ToolErrorCategory.NETWORK_ERROR,
            ToolErrorCode.CONNECTION_FAILED,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.NETWORK_ERROR],
        )

    # Resource Errors
    if any(keyword in error_type.lower() for keyword in ["memory", "disk", "cpu", "resource"]):
        return (
            ToolErrorCategory.RESOURCE_ERROR,
            ToolErrorCode.OUT_OF_MEMORY,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.RESOURCE_ERROR],
        )

    # Sandbox Errors
    if "sandbox" in error_type.lower() or "unsafe" in error_message:
        return (
            ToolErrorCategory.SANDBOX_ERROR,
            ToolErrorCode.SANDBOX_VIOLATION,
            ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.SANDBOX_ERROR],
        )

    # Default to internal error for unknown errors
    return (
        ToolErrorCategory.INTERNAL_ERROR,
        ToolErrorCode.UNEXPECTED_ERROR,
        ERROR_RECOVERY_STRATEGIES[ToolErrorCategory.INTERNAL_ERROR],
    )


def get_error_metadata(
    category: ToolErrorCategory,
    error_code: ToolErrorCode,
    recovery_strategy: ErrorRecoveryStrategy,
) -> dict[str, Any]:
    """Get metadata for an error including user-friendly message and recovery guidance.

    Args:
        category: Error category
        error_code: Specific error code
        recovery_strategy: Recovery strategy

    Returns:
        Dictionary with error metadata including:
        - user_message: User-friendly error message
        - is_retryable: Whether the error is retryable
        - recovery_guidance: How to recover from this error
    """
    # Determine if error is retryable
    is_retryable = recovery_strategy in [
        ErrorRecoveryStrategy.RETRY,
        ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
    ]

    # Generate user-friendly messages
    user_messages = {
        ToolErrorCategory.VALIDATION_ERROR: "The provided parameters are invalid. Please check the tool requirements and try again.",
        ToolErrorCategory.AUTHENTICATION_ERROR: "Authentication failed. Please verify your credentials.",
        ToolErrorCategory.AUTHORIZATION_ERROR: "You don't have permission to use this tool.",
        ToolErrorCategory.NOT_FOUND_ERROR: "The requested tool or resource was not found.",
        ToolErrorCategory.RATE_LIMIT_ERROR: "Rate limit exceeded. Please wait before trying again.",
        ToolErrorCategory.QUOTA_ERROR: "Usage quota exceeded. Please upgrade or wait for quota reset.",
        ToolErrorCategory.TIMEOUT_ERROR: "The operation timed out. Please try again.",
        ToolErrorCategory.EXECUTION_ERROR: "Tool execution failed. Please try a different approach.",
        ToolErrorCategory.NETWORK_ERROR: "Network connection failed. Please check your connection and try again.",
        ToolErrorCategory.RESOURCE_ERROR: "Insufficient resources available. Please try again later.",
        ToolErrorCategory.DEPENDENCY_ERROR: "External service unavailable. Please try again later.",
        ToolErrorCategory.INTERNAL_ERROR: "An unexpected error occurred. Please contact support.",
        ToolErrorCategory.CONFIGURATION_ERROR: "Tool configuration is invalid. Please contact administrator.",
        ToolErrorCategory.SANDBOX_ERROR: "Operation violates security constraints.",
    }

    recovery_guidance = {
        ErrorRecoveryStrategy.RETRY: "Retry the operation immediately.",
        ErrorRecoveryStrategy.RETRY_WITH_BACKOFF: "Wait and retry with exponential backoff.",
        ErrorRecoveryStrategy.FALLBACK: "Try an alternative tool or approach.",
        ErrorRecoveryStrategy.USER_INTERVENTION: "Correct the issue and retry.",
        ErrorRecoveryStrategy.FAIL: "Operation cannot be recovered automatically.",
    }

    return {
        "category": category.value,
        "error_code": error_code.value,
        "user_message": user_messages.get(category, "An error occurred during tool execution."),
        "is_retryable": is_retryable,
        "recovery_strategy": recovery_strategy.value,
        "recovery_guidance": recovery_guidance.get(recovery_strategy, "No recovery guidance available."),
    }
