"""
Tests for Circuit Breaker Pattern Implementation
"""

import asyncio

import pytest

from agentcore.orchestration.patterns.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FaultToleranceCoordinator,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    RetryPolicy,
    RetryStrategy,
)


@pytest.fixture
def circuit_config() -> CircuitBreakerConfig:
    """Create test circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1,  # Short timeout for tests
        max_retries=2,
        initial_retry_delay_seconds=0.1,  # Fast retries for tests
    )


@pytest.fixture
def circuit_breaker(circuit_config: CircuitBreakerConfig) -> CircuitBreaker:
    """Create test circuit breaker."""
    return CircuitBreaker(
        service_name="test-service",
        config=circuit_config,
    )


@pytest.fixture
def fault_tolerance_coordinator() -> FaultToleranceCoordinator:
    """Create test fault tolerance coordinator."""
    return FaultToleranceCoordinator()


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_circuit_initial_state(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.total_requests == 0
        assert circuit_breaker.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_successful_execution(self, circuit_breaker: CircuitBreaker) -> None:
        """Test successful function execution."""

        async def success_func():
            return {"result": "success"}

        result = await circuit_breaker.call(success_func)

        assert result == {"result": "success"}
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.failed_requests == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit opens after failure threshold."""

        async def failing_func():
            raise ValueError("Service error")

        # Cause failures up to threshold
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        assert (
            circuit_breaker.metrics.consecutive_failures
            == circuit_breaker.config.failure_threshold
        )

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit rejects requests when open."""

        async def failing_func():
            raise ValueError("Service error")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Next request should be rejected
        async def success_func():
            return {"result": "success"}

        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            await circuit_breaker.call(success_func)

        assert circuit_breaker.metrics.rejected_requests == 1

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit transitions to HALF_OPEN after timeout."""

        async def failing_func():
            raise ValueError("Service error")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(circuit_breaker.config.timeout_seconds + 0.1)

        # Check can execute (transitions to HALF_OPEN)
        can_exec = await circuit_breaker.can_execute()
        assert can_exec
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_from_half_open(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit closes after success threshold in HALF_OPEN."""

        async def failing_func():
            raise ValueError("Service error")

        async def success_func():
            return {"result": "success"}

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Wait for timeout to reach HALF_OPEN
        await asyncio.sleep(circuit_breaker.config.timeout_seconds + 0.1)
        await circuit_breaker.can_execute()  # Transition to HALF_OPEN

        # Successful requests to close circuit
        for _ in range(circuit_breaker.config.success_threshold):
            await circuit_breaker.call(success_func)

        # Circuit should be closed
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_reopens_from_half_open_on_failure(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test circuit reopens on any failure in HALF_OPEN."""

        async def failing_func():
            raise ValueError("Service error")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Wait for timeout to reach HALF_OPEN
        await asyncio.sleep(circuit_breaker.config.timeout_seconds + 0.1)
        await circuit_breaker.can_execute()  # Transition to HALF_OPEN
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Any failure reopens circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_manual_reset(self, circuit_breaker: CircuitBreaker) -> None:
        """Test manual circuit breaker reset."""

        async def failing_func():
            raise ValueError("Service error")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Manual reset
        await circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_get_status(self, circuit_breaker: CircuitBreaker) -> None:
        """Test getting circuit breaker status."""

        async def success_func():
            return {"result": "success"}

        await circuit_breaker.call(success_func)

        status = circuit_breaker.get_status()

        assert status["service_name"] == "test-service"
        assert status["state"] == CircuitState.CLOSED
        assert status["metrics"]["total_requests"] == 1
        assert status["metrics"]["successful_requests"] == 1
        assert status["metrics"]["error_rate"] == 0.0


class TestRetryPolicy:
    """Test RetryPolicy class."""

    def test_exponential_backoff_calculation(self) -> None:
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay_seconds=1.0,
            multiplier=2.0,
            jitter=False,
        )

        # First attempt
        delay = policy.calculate_delay()
        assert delay == 1.0

        # Second attempt
        policy.record_attempt()
        delay = policy.calculate_delay()
        assert delay == 2.0

        # Third attempt
        policy.record_attempt()
        delay = policy.calculate_delay()
        assert delay == 4.0

    def test_linear_backoff_calculation(self) -> None:
        """Test linear backoff delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.LINEAR,
            initial_delay_seconds=1.0,
            jitter=False,
        )

        # First attempt
        delay = policy.calculate_delay()
        assert delay == 1.0

        # Second attempt
        policy.record_attempt()
        delay = policy.calculate_delay()
        assert delay == 2.0

        # Third attempt
        policy.record_attempt()
        delay = policy.calculate_delay()
        assert delay == 3.0

    def test_fixed_backoff_calculation(self) -> None:
        """Test fixed backoff delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.FIXED,
            initial_delay_seconds=1.0,
            jitter=False,
        )

        for _ in range(3):
            delay = policy.calculate_delay()
            assert delay == 1.0
            policy.record_attempt()

    def test_immediate_retry(self) -> None:
        """Test immediate retry strategy."""
        policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.IMMEDIATE,
        )

        delay = policy.calculate_delay()
        assert delay == 0.0

    def test_max_delay_cap(self) -> None:
        """Test maximum delay cap."""
        policy = RetryPolicy(
            max_retries=10,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay_seconds=1.0,
            max_delay_seconds=5.0,
            multiplier=2.0,
            jitter=False,
        )

        # Go through multiple attempts
        for _ in range(5):
            policy.record_attempt()

        # Delay should be capped at max
        delay = policy.calculate_delay()
        assert delay <= 5.0

    def test_should_retry(self) -> None:
        """Test retry decision logic."""
        policy = RetryPolicy(max_retries=2)

        assert policy.should_retry()  # Attempt 0

        policy.record_attempt()
        assert policy.should_retry()  # Attempt 1

        policy.record_attempt()
        assert not policy.should_retry()  # Attempt 2 (max reached)

    def test_reset(self) -> None:
        """Test policy reset."""
        policy = RetryPolicy(max_retries=3)

        policy.record_attempt()
        policy.record_attempt()
        assert policy.current_attempt == 2

        policy.reset()
        assert policy.current_attempt == 0
        assert policy.last_attempt_at is None


class TestHealthMonitor:
    """Test HealthMonitor class."""

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        monitor = HealthMonitor(service_name="test-service")

        async def healthy_check():
            return {"status": "ok"}

        check = await monitor.check_health(healthy_check)

        assert check.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert check.response_time_ms is not None
        assert check.error_message is None

    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        """Test failed health check."""
        monitor = HealthMonitor(service_name="test-service")

        async def failing_check():
            raise RuntimeError("Service down")

        check = await monitor.check_health(failing_check)

        assert check.status == HealthStatus.UNHEALTHY
        assert check.error_message == "Service down"

    @pytest.mark.asyncio
    async def test_health_check_timeout(self) -> None:
        """Test health check timeout."""
        monitor = HealthMonitor(service_name="test-service")

        async def slow_check():
            await asyncio.sleep(20)  # Exceeds 10s timeout
            return {"status": "ok"}

        check = await monitor.check_health(slow_check)

        assert check.status == HealthStatus.UNHEALTHY
        assert "timeout" in check.error_message.lower()

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self) -> None:
        """Test starting and stopping monitoring."""
        monitor = HealthMonitor(
            service_name="test-service",
            check_interval_seconds=1,
        )

        async def healthy_check():
            return {"status": "ok"}

        # Start monitoring
        await monitor.start_monitoring(healthy_check)
        assert monitor.is_monitoring

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor.is_monitoring

    @pytest.mark.asyncio
    async def test_health_summary(self) -> None:
        """Test health monitoring summary."""
        monitor = HealthMonitor(service_name="test-service")

        async def healthy_check():
            return {"status": "ok"}

        # Perform some checks
        await monitor.check_health(healthy_check)
        await monitor.check_health(healthy_check)

        summary = monitor.get_health_summary()

        assert summary["service_name"] == "test-service"
        assert summary["current_status"] in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        ]
        assert summary["total_checks"] == 2


class TestFaultToleranceCoordinator:
    """Test FaultToleranceCoordinator class."""

    @pytest.mark.asyncio
    async def test_register_circuit_breaker(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test registering circuit breaker."""
        breaker = fault_tolerance_coordinator.register_circuit_breaker("service-a")

        assert breaker.service_name == "service-a"
        assert fault_tolerance_coordinator.get_circuit_breaker("service-a") == breaker

    @pytest.mark.asyncio
    async def test_register_health_monitor(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test registering health monitor."""
        monitor = fault_tolerance_coordinator.register_health_monitor("service-a")

        assert monitor.service_name == "service-a"
        assert fault_tolerance_coordinator.get_health_monitor("service-a") == monitor

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test successful execution with retry."""
        policy = RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE,
        )

        async def success_func():
            return {"result": "success"}

        result = await fault_tolerance_coordinator.execute_with_retry(
            success_func, policy
        )

        assert result == {"result": "success"}
        assert policy.current_attempt == 0  # No retries needed

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test execution succeeds after retries."""
        policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.IMMEDIATE,
        )

        attempt_count = {"count": 0}

        async def flaky_func():
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise ValueError("Transient error")
            return {"result": "success"}

        result = await fault_tolerance_coordinator.execute_with_retry(
            flaky_func, policy
        )

        assert result == {"result": "success"}
        assert attempt_count["count"] == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test execution fails after retry exhaustion."""
        policy = RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE,
        )

        async def failing_func():
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            await fault_tolerance_coordinator.execute_with_retry(failing_func, policy)

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test execution with circuit breaker."""

        async def success_func():
            return {"result": "success"}

        result = await fault_tolerance_coordinator.execute_with_circuit_breaker(
            "service-a", success_func
        )

        assert result == {"result": "success"}

        # Verify breaker was registered
        breaker = fault_tolerance_coordinator.get_circuit_breaker("service-a")
        assert breaker is not None
        assert breaker.metrics.successful_requests == 1

    @pytest.mark.asyncio
    async def test_execute_with_full_fault_tolerance(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test execution with circuit breaker and retry."""
        policy = RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE,
        )

        attempt_count = {"count": 0}

        async def flaky_func():
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise ValueError("Transient error")
            return {"result": "success"}

        result = await fault_tolerance_coordinator.execute_with_fault_tolerance(
            "service-b", flaky_func, policy
        )

        assert result == {"result": "success"}
        assert attempt_count["count"] == 2

    @pytest.mark.asyncio
    async def test_get_coordinator_status(
        self, fault_tolerance_coordinator: FaultToleranceCoordinator
    ) -> None:
        """Test getting coordinator status."""
        # Register components
        fault_tolerance_coordinator.register_circuit_breaker("service-a")
        fault_tolerance_coordinator.register_health_monitor("service-b")

        status = await fault_tolerance_coordinator.get_coordinator_status()

        assert status["total_breakers"] == 1
        assert status["total_monitors"] == 1
        assert "service-a" in status["circuit_breakers"]
        assert "service-b" in status["health_monitors"]


class TestCircuitBreakerMetrics:
    """Test CircuitBreakerMetrics class."""

    @pytest.mark.asyncio
    async def test_metrics_record_success(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test recording successful requests."""
        circuit_breaker.metrics.record_success()
        circuit_breaker.metrics.record_success()

        assert circuit_breaker.metrics.successful_requests == 2
        assert circuit_breaker.metrics.consecutive_successes == 2
        assert circuit_breaker.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_metrics_record_failure(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test recording failed requests."""
        circuit_breaker.metrics.record_failure()
        circuit_breaker.metrics.record_failure()

        assert circuit_breaker.metrics.failed_requests == 2
        assert circuit_breaker.metrics.consecutive_failures == 2
        assert circuit_breaker.metrics.consecutive_successes == 0

    @pytest.mark.asyncio
    async def test_error_rate_calculation(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test error rate calculation."""
        # Add mix of successes and failures
        circuit_breaker.metrics.record_success()
        circuit_breaker.metrics.record_success()
        circuit_breaker.metrics.record_failure()

        error_rate = circuit_breaker.metrics.get_error_rate()
        success_rate = circuit_breaker.metrics.get_success_rate()

        assert error_rate == pytest.approx(1 / 3)
        assert success_rate == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_sliding_window_limit(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Test sliding window maintains size limit."""
        circuit_breaker.metrics.window_size = 10

        # Add more than window size
        for _ in range(15):
            circuit_breaker.metrics.record_success()

        assert len(circuit_breaker.metrics.recent_requests) == 10
