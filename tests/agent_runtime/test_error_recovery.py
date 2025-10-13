"""Tests for error recovery service."""

import asyncio

import pytest

from agentcore.agent_runtime.models.error_types import (
    DegradationLevel,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy,
    RetryConfig,
)
from agentcore.agent_runtime.services.circuit_breaker import CircuitBreakerError
from agentcore.agent_runtime.services.error_recovery import ErrorRecoveryService


@pytest.fixture
def error_recovery() -> ErrorRecoveryService:
    """Create error recovery service."""
    return ErrorRecoveryService()


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create test retry configuration."""
    return RetryConfig(
        max_attempts=3,
        initial_delay_seconds=0.1,
        max_delay_seconds=1.0,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.mark.asyncio
class TestErrorRecovery:
    """Test error recovery service."""

    async def test_successful_execution(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test successful execution without errors."""

        async def success_func() -> str:
            return "success"

        result, recovery_result = await error_recovery.execute_with_recovery(
            success_func,
            agent_id="test_agent",
        )

        assert result == "success"
        assert recovery_result is None

    async def test_exponential_retry_success(
        self,
        error_recovery: ErrorRecoveryService,
        retry_config: RetryConfig,
    ) -> None:
        """Test successful execution after retries with exponential backoff."""
        attempts = 0

        async def retry_func() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Temporary error")
            return "success"

        result, recovery_result = await error_recovery.execute_with_recovery(
            retry_func,
            agent_id="test_agent",
            error_category=ErrorCategory.NETWORK,
            retry_config=retry_config,
            use_circuit_breaker=False,
        )

        assert result == "success"
        assert recovery_result is None
        assert attempts == 2

    async def test_retry_exhaustion(
        self,
        error_recovery: ErrorRecoveryService,
        retry_config: RetryConfig,
    ) -> None:
        """Test failure after exhausting retries."""

        async def fail_func() -> None:
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            await error_recovery.execute_with_recovery(
                fail_func,
                agent_id="test_agent",
                error_category=ErrorCategory.NETWORK,
                retry_config=retry_config,
                use_circuit_breaker=False,
            )

    async def test_circuit_breaker_integration(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test integration with circuit breaker."""
        attempts = 0

        async def fail_func() -> None:
            nonlocal attempts
            attempts += 1
            raise ValueError("Circuit breaker test")

        # Execute until circuit opens
        for _ in range(6):  # More than failure threshold
            try:
                await error_recovery.execute_with_recovery(
                    fail_func,
                    agent_id="circuit_test",
                    error_category=ErrorCategory.NETWORK,
                    retry_config=RetryConfig(max_attempts=1),
                    use_circuit_breaker=True,
                )
            except (ValueError, CircuitBreakerError):
                pass

        # Verify circuit breaker opened (should raise CircuitBreakerError)
        with pytest.raises(CircuitBreakerError):
            await error_recovery.execute_with_recovery(
                fail_func,
                agent_id="circuit_test",
                error_category=ErrorCategory.NETWORK,
                retry_config=RetryConfig(max_attempts=1),
                use_circuit_breaker=True,
            )

    async def test_degraded_execution(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test execution with degraded functionality."""

        async def degraded_func(degraded: bool = False, **kwargs: object) -> str:
            if degraded:
                return "degraded_result"
            return "normal_result"

        result, recovery_result = await error_recovery.execute_with_recovery(
            degraded_func,
            agent_id="test_agent",
            error_category=ErrorCategory.RESOURCE_EXHAUSTION,
            use_circuit_breaker=False,
        )

        # Should attempt degraded execution
        assert recovery_result is not None
        if recovery_result.degradation_level:
            assert result == "degraded_result"

    async def test_error_history_recording(
        self,
        error_recovery: ErrorRecoveryService,
        retry_config: RetryConfig,
    ) -> None:
        """Test error history recording."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        try:
            await error_recovery.execute_with_recovery(
                fail_func,
                agent_id="test_agent",
                error_category=ErrorCategory.EXECUTION,
                retry_config=retry_config,
                use_circuit_breaker=False,
            )
        except ValueError:
            pass

        history = await error_recovery.get_error_history("test_agent")
        assert len(history) > 0
        assert history[0].category == ErrorCategory.EXECUTION

    async def test_degradation_level_progression(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test degradation level increases with more errors."""

        async def fail_func(degraded: bool = False, **kwargs: object) -> str:
            if not degraded:
                raise ValueError("Force degradation")
            return "degraded"

        # Generate multiple errors to increase degradation
        for _ in range(6):
            try:
                await error_recovery.execute_with_recovery(
                    fail_func,
                    agent_id="degradation_test",
                    error_category=ErrorCategory.RESOURCE_EXHAUSTION,
                    use_circuit_breaker=False,
                )
            except ValueError:
                pass

        degradation = await error_recovery.get_degradation_state("degradation_test")
        # Should have some degradation level
        assert degradation in (
            DegradationLevel.REDUCED,
            DegradationLevel.MINIMAL,
            DegradationLevel.EMERGENCY,
        )

    async def test_degradation_reset(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test resetting degradation state."""

        async def degraded_func(degraded: bool = False, **kwargs: object) -> str:
            return "result"

        # Set degradation
        await error_recovery.execute_with_recovery(
            degraded_func,
            agent_id="test_agent",
            error_category=ErrorCategory.RESOURCE_EXHAUSTION,
            use_circuit_breaker=False,
        )

        # Reset
        await error_recovery.reset_degradation("test_agent")

        degradation = await error_recovery.get_degradation_state("test_agent")
        assert degradation is None

    async def test_statistics_collection(
        self,
        error_recovery: ErrorRecoveryService,
        retry_config: RetryConfig,
    ) -> None:
        """Test statistics collection."""

        async def fail_func() -> None:
            raise ValueError("Test error")

        # Generate some errors
        for i in range(3):
            try:
                await error_recovery.execute_with_recovery(
                    fail_func,
                    agent_id=f"agent_{i}",
                    error_category=ErrorCategory.NETWORK,
                    retry_config=retry_config,
                    use_circuit_breaker=False,
                )
            except ValueError:
                pass

        stats = await error_recovery.get_statistics()
        assert stats["total_errors"] > 0
        assert stats["agents_with_errors"] == 3
        assert ErrorCategory.NETWORK.value in stats["errors_by_category"]

    async def test_error_severity_determination(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test error severity determination."""
        # Test with security error (should be critical)
        metadata = error_recovery._create_error_metadata(
            ValueError("Test"),
            ErrorCategory.SECURITY,
            "test_agent",
        )
        assert metadata.severity == ErrorSeverity.CRITICAL

        # Test with network error (should be medium)
        metadata = error_recovery._create_error_metadata(
            ConnectionError("Test"),
            ErrorCategory.NETWORK,
            "test_agent",
        )
        assert metadata.severity == ErrorSeverity.MEDIUM

    async def test_recovery_strategy_selection(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test recovery strategy selection based on error category."""
        # Network errors should use retry and circuit breaker
        strategies = error_recovery._get_recovery_strategies(ErrorCategory.NETWORK)
        assert RecoveryStrategy.RETRY_EXPONENTIAL in strategies
        assert RecoveryStrategy.CIRCUIT_BREAK in strategies

        # Security errors should require manual intervention
        strategies = error_recovery._get_recovery_strategies(ErrorCategory.SECURITY)
        assert RecoveryStrategy.MANUAL in strategies

    async def test_constant_delay_retry(
        self,
        error_recovery: ErrorRecoveryService,
        retry_config: RetryConfig,
    ) -> None:
        """Test constant delay retry strategy."""
        attempts = 0

        async def retry_func() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry test")
            return "success"

        result, _ = await error_recovery.execute_with_recovery(
            retry_func,
            agent_id="test_agent",
            error_category=ErrorCategory.TIMEOUT,  # Uses constant retry
            retry_config=retry_config,
            use_circuit_breaker=False,
        )

        assert result == "success"
        assert attempts == 2

    async def test_sync_function_execution(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test execution of synchronous functions."""

        def sync_func() -> str:
            return "sync_result"

        result, recovery_result = await error_recovery.execute_with_recovery(
            sync_func,
            agent_id="test_agent",
            use_circuit_breaker=False,
        )

        assert result == "sync_result"
        assert recovery_result is None

    async def test_error_metadata_creation(
        self,
        error_recovery: ErrorRecoveryService,
    ) -> None:
        """Test error metadata creation."""
        error = ValueError("Test error message")

        metadata = error_recovery._create_error_metadata(
            error,
            ErrorCategory.EXECUTION,
            "test_agent",
        )

        assert metadata.category == ErrorCategory.EXECUTION
        assert metadata.message == "Test error message"
        assert metadata.agent_id == "test_agent"
        assert metadata.stack_trace is not None
        assert "ValueError" in metadata.details["exception_type"]
