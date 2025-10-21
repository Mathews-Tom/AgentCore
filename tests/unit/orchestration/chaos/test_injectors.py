"""
Unit tests for fault injectors.
"""

import asyncio

import pytest

from agentcore.orchestration.chaos.injectors import (
    ExceptionInjector,
    NetworkFaultInjector,
    ServiceCrashInjector,
    TimeoutInjector,
)
from agentcore.orchestration.chaos.models import FaultConfig, FaultType


class TestNetworkFaultInjector:
    """Test network fault injector."""

    @pytest.fixture
    def injector(self) -> NetworkFaultInjector:
        """Create network injector."""
        return NetworkFaultInjector()

    @pytest.mark.asyncio
    async def test_inject_network_latency(
        self, injector: NetworkFaultInjector
    ) -> None:
        """Test network latency injection."""
        config = FaultConfig(
            fault_type=FaultType.NETWORK_LATENCY,
            target_service="test_service",
            latency_ms=100,
        )

        result = await injector.inject(config)

        assert result.success
        assert result.fault_type == FaultType.NETWORK_LATENCY
        assert result.target_service == "test_service"
        assert result.metadata["latency_ms"] == 100

        # Verify latency is applied
        status = injector.get_network_status()
        assert "test_service" in status["latency_overrides"]
        assert status["latency_overrides"]["test_service"] == 100

    @pytest.mark.asyncio
    async def test_inject_packet_loss(self, injector: NetworkFaultInjector) -> None:
        """Test packet loss injection."""
        config = FaultConfig(
            fault_type=FaultType.NETWORK_PACKET_LOSS,
            target_service="test_service",
            packet_loss_rate=0.5,
        )

        result = await injector.inject(config)

        assert result.success
        assert result.metadata["packet_loss_rate"] == 0.5

        status = injector.get_network_status()
        assert status["packet_loss_rates"]["test_service"] == 0.5

    @pytest.mark.asyncio
    async def test_inject_network_partition(
        self, injector: NetworkFaultInjector
    ) -> None:
        """Test network partition injection."""
        config = FaultConfig(
            fault_type=FaultType.NETWORK_PARTITION,
            target_service="test_service",
        )

        result = await injector.inject(config)

        assert result.success
        assert result.metadata["partition"]

        # Partition should create 100% packet loss
        status = injector.get_network_status()
        assert status["packet_loss_rates"]["test_service"] == 1.0

    @pytest.mark.asyncio
    async def test_remove_fault(self, injector: NetworkFaultInjector) -> None:
        """Test fault removal."""
        config = FaultConfig(
            fault_type=FaultType.NETWORK_LATENCY,
            target_service="test_service",
            latency_ms=100,
        )

        result = await injector.inject(config)
        injection_id = result.injection_id

        removed = await injector.remove(injection_id)

        assert removed
        assert result.removed_at is not None

        status = injector.get_network_status()
        assert "test_service" not in status["latency_overrides"]

    @pytest.mark.asyncio
    async def test_apply_network_effects_latency(
        self, injector: NetworkFaultInjector
    ) -> None:
        """Test applying network effects with latency."""
        config = FaultConfig(
            fault_type=FaultType.NETWORK_LATENCY,
            target_service="test_service",
            latency_ms=100,
        )

        await injector.inject(config)

        async def test_func() -> str:
            return "success"

        # Measure latency
        import time

        start = time.time()
        result = await injector.apply_network_effects(
            "test_service", test_func
        )
        elapsed = (time.time() - start) * 1000

        assert result == "success"
        assert elapsed >= 100  # Should have at least 100ms latency

    @pytest.mark.asyncio
    async def test_apply_network_effects_packet_loss(
        self, injector: NetworkFaultInjector
    ) -> None:
        """Test applying network effects with packet loss."""
        config = FaultConfig(
            fault_type=FaultType.NETWORK_PACKET_LOSS,
            target_service="test_service",
            packet_loss_rate=1.0,  # 100% loss
        )

        await injector.inject(config)

        async def test_func() -> str:
            return "success"

        # Should raise ConnectionError due to packet loss
        with pytest.raises(ConnectionError, match="Network packet loss"):
            await injector.apply_network_effects("test_service", test_func)


class TestServiceCrashInjector:
    """Test service crash injector."""

    @pytest.fixture
    def injector(self) -> ServiceCrashInjector:
        """Create service injector."""
        return ServiceCrashInjector()

    @pytest.mark.asyncio
    async def test_inject_service_crash(
        self, injector: ServiceCrashInjector
    ) -> None:
        """Test service crash injection."""
        config = FaultConfig(
            fault_type=FaultType.SERVICE_CRASH,
            target_service="test_service",
        )

        result = await injector.inject(config)

        assert result.success
        assert result.metadata["crashed"]

        # Verify service is crashed
        is_healthy = await injector.check_service_health("test_service")
        assert not is_healthy

    @pytest.mark.asyncio
    async def test_inject_service_hang(self, injector: ServiceCrashInjector) -> None:
        """Test service hang injection."""
        config = FaultConfig(
            fault_type=FaultType.SERVICE_HANG,
            target_service="test_service",
            hang_duration_seconds=0.5,
        )

        result = await injector.inject(config)

        assert result.success
        assert result.metadata["hang_duration"] == 0.5

    @pytest.mark.asyncio
    async def test_apply_service_effects_crash(
        self, injector: ServiceCrashInjector
    ) -> None:
        """Test applying service effects with crash."""
        config = FaultConfig(
            fault_type=FaultType.SERVICE_CRASH,
            target_service="test_service",
        )

        await injector.inject(config)

        async def test_func() -> str:
            return "success"

        # Should raise RuntimeError due to crash
        with pytest.raises(RuntimeError, match="has crashed"):
            await injector.apply_service_effects("test_service", test_func)

    @pytest.mark.asyncio
    async def test_apply_service_effects_hang(
        self, injector: ServiceCrashInjector
    ) -> None:
        """Test applying service effects with hang."""
        config = FaultConfig(
            fault_type=FaultType.SERVICE_HANG,
            target_service="test_service",
            hang_duration_seconds=0.2,
        )

        await injector.inject(config)

        async def test_func() -> str:
            return "success"

        # Measure hang duration
        import time

        start = time.time()
        result = await injector.apply_service_effects("test_service", test_func)
        elapsed = time.time() - start

        assert result == "success"
        assert elapsed >= 0.2  # Should have hung for at least 200ms


class TestTimeoutInjector:
    """Test timeout injector."""

    @pytest.fixture
    def injector(self) -> TimeoutInjector:
        """Create timeout injector."""
        return TimeoutInjector()

    @pytest.mark.asyncio
    async def test_inject_timeout(self, injector: TimeoutInjector) -> None:
        """Test timeout injection."""
        config = FaultConfig(
            fault_type=FaultType.TIMEOUT,
            target_service="test_service",
            duration_seconds=10.0,
            intensity=1.0,
        )

        result = await injector.inject(config)

        assert result.success
        assert "timeout_seconds" in result.metadata

        status = injector.get_timeout_status()
        assert "test_service" in status["timeout_overrides"]

    @pytest.mark.asyncio
    async def test_apply_timeout_triggers(self, injector: TimeoutInjector) -> None:
        """Test that timeout is triggered."""
        config = FaultConfig(
            fault_type=FaultType.TIMEOUT,
            target_service="test_service",
            duration_seconds=1.0,
            intensity=1.0,
        )

        await injector.inject(config)

        async def slow_func() -> str:
            await asyncio.sleep(2.0)  # Slower than timeout
            return "success"

        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await injector.apply_timeout("test_service", slow_func)

    @pytest.mark.asyncio
    async def test_apply_timeout_passes(self, injector: TimeoutInjector) -> None:
        """Test that fast operations pass through timeout."""
        async def fast_func() -> str:
            return "success"

        # Should succeed with default timeout
        result = await injector.apply_timeout("test_service", fast_func)
        assert result == "success"


class TestExceptionInjector:
    """Test exception injector."""

    @pytest.fixture
    def injector(self) -> ExceptionInjector:
        """Create exception injector."""
        return ExceptionInjector()

    @pytest.mark.asyncio
    async def test_inject_exception(self, injector: ExceptionInjector) -> None:
        """Test exception injection."""
        config = FaultConfig(
            fault_type=FaultType.EXCEPTION,
            target_service="test_service",
            exception_type="ValueError",
            exception_rate=1.0,
        )

        result = await injector.inject(config)

        assert result.success
        assert result.metadata["exception_type"] == "ValueError"
        assert result.metadata["exception_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_check_and_raise_valueerror(
        self, injector: ExceptionInjector
    ) -> None:
        """Test raising ValueError."""
        config = FaultConfig(
            fault_type=FaultType.EXCEPTION,
            target_service="test_service",
            exception_type="ValueError",
            exception_rate=1.0,
        )

        await injector.inject(config)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Injected ValueError"):
            await injector.check_and_raise("test_service")

    @pytest.mark.asyncio
    async def test_check_and_raise_runtimeerror(
        self, injector: ExceptionInjector
    ) -> None:
        """Test raising RuntimeError."""
        config = FaultConfig(
            fault_type=FaultType.EXCEPTION,
            target_service="test_service",
            exception_type="RuntimeError",
            exception_rate=1.0,
        )

        await injector.inject(config)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Injected RuntimeError"):
            await injector.check_and_raise("test_service")

    @pytest.mark.asyncio
    async def test_exception_rate_partial(self, injector: ExceptionInjector) -> None:
        """Test that exception rate works probabilistically."""
        config = FaultConfig(
            fault_type=FaultType.EXCEPTION,
            target_service="test_service",
            exception_type="ValueError",
            exception_rate=0.5,  # 50% rate
        )

        await injector.inject(config)

        # With 50% rate, should have some successes and some failures
        # Run multiple times and count exceptions
        exceptions_raised = 0
        for _ in range(20):
            try:
                await injector.check_and_raise("test_service")
            except ValueError:
                exceptions_raised += 1

        # Should have raised some exceptions but not all
        assert 0 < exceptions_raised < 20
