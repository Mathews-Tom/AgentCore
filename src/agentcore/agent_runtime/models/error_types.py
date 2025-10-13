"""
Error categorization and types for agent runtime.

This module defines comprehensive error categories, severity levels,
and error metadata for proper error handling and recovery.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """Categories of errors in agent runtime."""

    # Infrastructure errors - related to container/resource management
    INFRASTRUCTURE = "infrastructure"
    # Resource exhaustion - memory, CPU, disk, network limits exceeded
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    # Network errors - connectivity issues, timeouts
    NETWORK = "network"
    # Security errors - permission denied, sandbox violations
    SECURITY = "security"
    # Configuration errors - invalid config, missing parameters
    CONFIGURATION = "configuration"
    # Execution errors - agent logic failures, tool errors
    EXECUTION = "execution"
    # State errors - invalid state transitions, corruption
    STATE = "state"
    # External service errors - A2A protocol, LLM, tools
    EXTERNAL_SERVICE = "external_service"
    # Timeout errors - execution deadline exceeded
    TIMEOUT = "timeout"
    # Unknown errors - unexpected failures
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    # Critical - system integrity compromised, immediate action required
    CRITICAL = "critical"
    # High - agent cannot continue, recovery required
    HIGH = "high"
    # Medium - degraded operation possible
    MEDIUM = "medium"
    # Low - minor issue, can continue with warning
    LOW = "low"


class RecoveryStrategy(str, Enum):
    """Available recovery strategies for errors."""

    # Retry with exponential backoff
    RETRY_EXPONENTIAL = "retry_exponential"
    # Retry with constant delay
    RETRY_CONSTANT = "retry_constant"
    # Restart agent from checkpoint
    RESTART_CHECKPOINT = "restart_checkpoint"
    # Restart agent from beginning
    RESTART_CLEAN = "restart_clean"
    # Failover to backup resource
    FAILOVER = "failover"
    # Degrade to reduced functionality
    DEGRADE = "degrade"
    # Circuit breaker - stop attempts temporarily
    CIRCUIT_BREAK = "circuit_break"
    # Manual intervention required
    MANUAL = "manual"
    # No recovery possible
    NONE = "none"


class ErrorMetadata(BaseModel):
    """Metadata about an error occurrence."""

    error_id: str = Field(description="Unique error identifier")
    category: ErrorCategory = Field(description="Error category")
    severity: ErrorSeverity = Field(description="Error severity level")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error details",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When error occurred",
    )
    agent_id: str | None = Field(
        default=None,
        description="Associated agent identifier",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution context at time of error",
    )
    stack_trace: str | None = Field(
        default=None,
        description="Stack trace if available",
    )
    recovery_attempted: bool = Field(
        default=False,
        description="Whether recovery was attempted",
    )
    recovery_strategy: RecoveryStrategy | None = Field(
        default=None,
        description="Recovery strategy used",
    )
    recovery_successful: bool | None = Field(
        default=None,
        description="Whether recovery succeeded",
    )


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts",
    )
    initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay before first retry",
    )
    max_delay_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay between retries",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=10.0,
        description="Base for exponential backoff",
    )
    jitter: bool = Field(
        default=True,
        description="Add random jitter to delays",
    )


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before opening circuit",
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Successes needed to close circuit",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Time before attempting recovery",
    )
    half_open_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max attempts in half-open state",
    )


class DegradationLevel(str, Enum):
    """Levels of service degradation."""

    FULL = "full"  # All features available
    REDUCED = "reduced"  # Non-essential features disabled
    MINIMAL = "minimal"  # Only core functionality
    EMERGENCY = "emergency"  # Survival mode, minimal operations


class ErrorRecoveryResult(BaseModel):
    """Result of error recovery attempt."""

    success: bool = Field(description="Whether recovery succeeded")
    strategy_used: RecoveryStrategy = Field(description="Recovery strategy applied")
    attempts: int = Field(description="Number of attempts made")
    duration_seconds: float = Field(description="Time spent on recovery")
    new_error: ErrorMetadata | None = Field(
        default=None,
        description="New error if recovery failed",
    )
    degradation_level: DegradationLevel | None = Field(
        default=None,
        description="Service degradation level if applicable",
    )
    message: str = Field(description="Recovery result message")


# Error category to recovery strategy mapping
DEFAULT_RECOVERY_STRATEGIES: dict[ErrorCategory, list[RecoveryStrategy]] = {
    ErrorCategory.INFRASTRUCTURE: [
        RecoveryStrategy.RETRY_EXPONENTIAL,
        RecoveryStrategy.FAILOVER,
        RecoveryStrategy.RESTART_CLEAN,
    ],
    ErrorCategory.RESOURCE_EXHAUSTION: [
        RecoveryStrategy.DEGRADE,
        RecoveryStrategy.RESTART_CHECKPOINT,
        RecoveryStrategy.FAILOVER,
    ],
    ErrorCategory.NETWORK: [
        RecoveryStrategy.RETRY_EXPONENTIAL,
        RecoveryStrategy.CIRCUIT_BREAK,
        RecoveryStrategy.FAILOVER,
    ],
    ErrorCategory.SECURITY: [
        RecoveryStrategy.MANUAL,
        RecoveryStrategy.NONE,
    ],
    ErrorCategory.CONFIGURATION: [
        RecoveryStrategy.MANUAL,
        RecoveryStrategy.DEGRADE,
    ],
    ErrorCategory.EXECUTION: [
        RecoveryStrategy.RETRY_CONSTANT,
        RecoveryStrategy.RESTART_CHECKPOINT,
        RecoveryStrategy.DEGRADE,
    ],
    ErrorCategory.STATE: [
        RecoveryStrategy.RESTART_CHECKPOINT,
        RecoveryStrategy.RESTART_CLEAN,
    ],
    ErrorCategory.EXTERNAL_SERVICE: [
        RecoveryStrategy.RETRY_EXPONENTIAL,
        RecoveryStrategy.CIRCUIT_BREAK,
        RecoveryStrategy.DEGRADE,
    ],
    ErrorCategory.TIMEOUT: [
        RecoveryStrategy.RETRY_CONSTANT,
        RecoveryStrategy.DEGRADE,
    ],
    ErrorCategory.UNKNOWN: [
        RecoveryStrategy.RETRY_CONSTANT,
        RecoveryStrategy.MANUAL,
    ],
}

# Error severity to max retry attempts mapping
SEVERITY_MAX_RETRIES: dict[ErrorSeverity, int] = {
    ErrorSeverity.CRITICAL: 0,  # No automatic retry for critical errors
    ErrorSeverity.HIGH: 1,  # Single retry attempt
    ErrorSeverity.MEDIUM: 3,  # Standard retry attempts
    ErrorSeverity.LOW: 5,  # More lenient retry policy
}
