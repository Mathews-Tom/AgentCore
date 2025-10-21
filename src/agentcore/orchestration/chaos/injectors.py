"""
Fault Injectors

Implementations for various fault injection types including network failures,
service crashes, and timeout simulation.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.chaos.models import FaultConfig, FaultType


class InjectionResult(BaseModel):
    """Result of fault injection."""

    injection_id: UUID = Field(default_factory=uuid4)
    fault_type: FaultType
    target_service: str
    success: bool
    injected_at: datetime
    removed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class FaultInjector(ABC):
    """
    Base class for fault injectors.

    Provides interface for injecting and removing faults.
    """

    def __init__(self, injector_id: str) -> None:
        """Initialize fault injector."""
        self.injector_id = injector_id
        self._active_injections: dict[UUID, InjectionResult] = {}
        self._lock = asyncio.Lock()

    @abstractmethod
    async def inject(self, config: FaultConfig) -> InjectionResult:
        """
        Inject fault according to configuration.

        Args:
            config: Fault configuration

        Returns:
            Injection result
        """
        pass

    @abstractmethod
    async def remove(self, injection_id: UUID) -> bool:
        """
        Remove injected fault.

        Args:
            injection_id: Injection identifier

        Returns:
            True if removed successfully
        """
        pass

    async def get_active_injections(self) -> list[InjectionResult]:
        """Get all active injections."""
        async with self._lock:
            return list(self._active_injections.values())

    async def remove_all(self) -> int:
        """Remove all active injections."""
        async with self._lock:
            count = len(self._active_injections)
            injection_ids = list(self._active_injections.keys())

        for injection_id in injection_ids:
            await self.remove(injection_id)

        return count


class NetworkFaultInjector(FaultInjector):
    """
    Network fault injector.

    Injects network latency, packet loss, and bandwidth constraints.
    """

    def __init__(self, injector_id: str = "network_injector") -> None:
        """Initialize network fault injector."""
        super().__init__(injector_id)
        self._latency_overrides: dict[str, int] = {}
        self._packet_loss_rates: dict[str, float] = {}
        self._bandwidth_limits: dict[str, int] = {}

    async def inject(self, config: FaultConfig) -> InjectionResult:
        """Inject network fault."""
        result = InjectionResult(
            fault_type=config.fault_type,
            target_service=config.target_service,
            success=False,
            injected_at=datetime.now(UTC),
        )

        try:
            async with self._lock:
                if config.fault_type == FaultType.NETWORK_LATENCY:
                    if config.latency_ms is None:
                        raise ValueError(
                            "latency_ms required for network latency fault"
                        )
                    self._latency_overrides[config.target_service] = config.latency_ms
                    result.metadata["latency_ms"] = config.latency_ms

                elif config.fault_type == FaultType.NETWORK_PACKET_LOSS:
                    if config.packet_loss_rate is None:
                        raise ValueError(
                            "packet_loss_rate required for packet loss fault"
                        )
                    self._packet_loss_rates[config.target_service] = (
                        config.packet_loss_rate
                    )
                    result.metadata["packet_loss_rate"] = config.packet_loss_rate

                elif config.fault_type == FaultType.NETWORK_PARTITION:
                    # Simulate complete network partition (100% packet loss)
                    self._packet_loss_rates[config.target_service] = 1.0
                    result.metadata["partition"] = True

                else:
                    raise ValueError(
                        f"Unsupported network fault type: {config.fault_type}"
                    )

                result.success = True
                self._active_injections[result.injection_id] = result

        except Exception as e:
            result.error_message = str(e)

        return result

    async def remove(self, injection_id: UUID) -> bool:
        """Remove network fault."""
        async with self._lock:
            if injection_id not in self._active_injections:
                return False

            result = self._active_injections[injection_id]
            service = result.target_service

            # Remove fault based on type
            if result.fault_type == FaultType.NETWORK_LATENCY:
                self._latency_overrides.pop(service, None)
            elif result.fault_type in (
                FaultType.NETWORK_PACKET_LOSS,
                FaultType.NETWORK_PARTITION,
            ):
                self._packet_loss_rates.pop(service, None)

            result.removed_at = datetime.now(UTC)
            del self._active_injections[injection_id]

        return True

    async def apply_network_effects(
        self,
        service: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply network effects to a function call.

        Args:
            service: Target service name
            func: Function to wrap
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result with network effects applied

        Raises:
            Exception: If packet loss occurs or function fails
        """
        # Check for packet loss
        if service in self._packet_loss_rates:
            loss_rate = self._packet_loss_rates[service]
            if random.random() < loss_rate:
                raise ConnectionError(f"Network packet loss simulated for {service}")

        # Apply latency
        if service in self._latency_overrides:
            latency_ms = self._latency_overrides[service]
            await asyncio.sleep(latency_ms / 1000.0)

        # Execute function
        return await func(*args, **kwargs)

    def get_network_status(self) -> dict[str, Any]:
        """Get current network fault status."""
        return {
            "injector_id": self.injector_id,
            "active_injections": len(self._active_injections),
            "latency_overrides": self._latency_overrides.copy(),
            "packet_loss_rates": self._packet_loss_rates.copy(),
            "bandwidth_limits": self._bandwidth_limits.copy(),
        }


class ServiceCrashInjector(FaultInjector):
    """
    Service crash injector.

    Simulates service crashes and hangs.
    """

    def __init__(self, injector_id: str = "service_crash_injector") -> None:
        """Initialize service crash injector."""
        super().__init__(injector_id)
        self._crashed_services: set[str] = set()
        self._hanging_services: dict[str, float] = {}

    async def inject(self, config: FaultConfig) -> InjectionResult:
        """Inject service crash fault."""
        result = InjectionResult(
            fault_type=config.fault_type,
            target_service=config.target_service,
            success=False,
            injected_at=datetime.now(UTC),
        )

        try:
            async with self._lock:
                if config.fault_type == FaultType.SERVICE_CRASH:
                    self._crashed_services.add(config.target_service)
                    result.metadata["crashed"] = True

                elif config.fault_type == FaultType.SERVICE_HANG:
                    if config.hang_duration_seconds is None:
                        raise ValueError(
                            "hang_duration_seconds required for service hang fault"
                        )
                    self._hanging_services[config.target_service] = (
                        config.hang_duration_seconds
                    )
                    result.metadata["hang_duration"] = config.hang_duration_seconds

                else:
                    raise ValueError(
                        f"Unsupported service fault type: {config.fault_type}"
                    )

                result.success = True
                self._active_injections[result.injection_id] = result

        except Exception as e:
            result.error_message = str(e)

        return result

    async def remove(self, injection_id: UUID) -> bool:
        """Remove service crash fault."""
        async with self._lock:
            if injection_id not in self._active_injections:
                return False

            result = self._active_injections[injection_id]
            service = result.target_service

            # Remove fault based on type
            if result.fault_type == FaultType.SERVICE_CRASH:
                self._crashed_services.discard(service)
            elif result.fault_type == FaultType.SERVICE_HANG:
                self._hanging_services.pop(service, None)

            result.removed_at = datetime.now(UTC)
            del self._active_injections[injection_id]

        return True

    async def check_service_health(self, service: str) -> bool:
        """
        Check if service is healthy.

        Args:
            service: Service name

        Returns:
            True if healthy, False if crashed
        """
        return service not in self._crashed_services

    async def apply_service_effects(
        self,
        service: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply service effects to a function call.

        Args:
            service: Target service name
            func: Function to wrap
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result with service effects applied

        Raises:
            RuntimeError: If service is crashed
            asyncio.TimeoutError: If service is hanging
        """
        # Check for crash
        if service in self._crashed_services:
            raise RuntimeError(f"Service {service} has crashed")

        # Apply hang
        if service in self._hanging_services:
            hang_duration = self._hanging_services[service]
            await asyncio.sleep(hang_duration)

        # Execute function
        return await func(*args, **kwargs)

    def get_service_status(self) -> dict[str, Any]:
        """Get current service fault status."""
        return {
            "injector_id": self.injector_id,
            "active_injections": len(self._active_injections),
            "crashed_services": list(self._crashed_services),
            "hanging_services": self._hanging_services.copy(),
        }


class TimeoutInjector(FaultInjector):
    """
    Timeout injector.

    Forces timeouts on operations.
    """

    def __init__(self, injector_id: str = "timeout_injector") -> None:
        """Initialize timeout injector."""
        super().__init__(injector_id)
        self._timeout_overrides: dict[str, float] = {}

    async def inject(self, config: FaultConfig) -> InjectionResult:
        """Inject timeout fault."""
        result = InjectionResult(
            fault_type=config.fault_type,
            target_service=config.target_service,
            success=False,
            injected_at=datetime.now(UTC),
        )

        try:
            async with self._lock:
                if config.fault_type != FaultType.TIMEOUT:
                    raise ValueError(f"Unsupported fault type: {config.fault_type}")

                # Set timeout to very small value to force timeout
                timeout_seconds = config.duration_seconds * config.intensity * 0.01
                self._timeout_overrides[config.target_service] = timeout_seconds
                result.metadata["timeout_seconds"] = timeout_seconds

                result.success = True
                self._active_injections[result.injection_id] = result

        except Exception as e:
            result.error_message = str(e)

        return result

    async def remove(self, injection_id: UUID) -> bool:
        """Remove timeout fault."""
        async with self._lock:
            if injection_id not in self._active_injections:
                return False

            result = self._active_injections[injection_id]
            service = result.target_service

            self._timeout_overrides.pop(service, None)

            result.removed_at = datetime.now(UTC)
            del self._active_injections[injection_id]

        return True

    async def apply_timeout(
        self,
        service: str,
        func: Callable[..., Any],
        default_timeout: float = 30.0,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply timeout to a function call.

        Args:
            service: Target service name
            func: Function to wrap
            default_timeout: Default timeout if no override
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            asyncio.TimeoutError: If timeout occurs
        """
        timeout = self._timeout_overrides.get(service, default_timeout)

        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        except TimeoutError as e:
            raise TimeoutError(
                f"Operation timed out for {service} after {timeout}s"
            ) from e

    def get_timeout_status(self) -> dict[str, Any]:
        """Get current timeout fault status."""
        return {
            "injector_id": self.injector_id,
            "active_injections": len(self._active_injections),
            "timeout_overrides": self._timeout_overrides.copy(),
        }


class ExceptionInjector(FaultInjector):
    """
    Exception injector.

    Throws exceptions at configurable rate.
    """

    def __init__(self, injector_id: str = "exception_injector") -> None:
        """Initialize exception injector."""
        super().__init__(injector_id)
        self._exception_configs: dict[str, tuple[str, float]] = {}

    async def inject(self, config: FaultConfig) -> InjectionResult:
        """Inject exception fault."""
        result = InjectionResult(
            fault_type=config.fault_type,
            target_service=config.target_service,
            success=False,
            injected_at=datetime.now(UTC),
        )

        try:
            async with self._lock:
                if config.fault_type != FaultType.EXCEPTION:
                    raise ValueError(f"Unsupported fault type: {config.fault_type}")

                if config.exception_type is None:
                    raise ValueError("exception_type required for exception fault")

                exception_rate = config.exception_rate or 1.0
                self._exception_configs[config.target_service] = (
                    config.exception_type,
                    exception_rate,
                )
                result.metadata["exception_type"] = config.exception_type
                result.metadata["exception_rate"] = exception_rate

                result.success = True
                self._active_injections[result.injection_id] = result

        except Exception as e:
            result.error_message = str(e)

        return result

    async def remove(self, injection_id: UUID) -> bool:
        """Remove exception fault."""
        async with self._lock:
            if injection_id not in self._active_injections:
                return False

            result = self._active_injections[injection_id]
            service = result.target_service

            self._exception_configs.pop(service, None)

            result.removed_at = datetime.now(UTC)
            del self._active_injections[injection_id]

        return True

    async def check_and_raise(self, service: str) -> None:
        """
        Check if should raise exception.

        Args:
            service: Service name

        Raises:
            Exception: Based on configuration
        """
        if service not in self._exception_configs:
            return

        exception_type, exception_rate = self._exception_configs[service]

        if random.random() < exception_rate:
            # Raise exception based on type
            if exception_type == "ValueError":
                raise ValueError(f"Injected ValueError for {service}")
            elif exception_type == "RuntimeError":
                raise RuntimeError(f"Injected RuntimeError for {service}")
            elif exception_type == "ConnectionError":
                raise ConnectionError(f"Injected ConnectionError for {service}")
            else:
                raise Exception(f"Injected {exception_type} for {service}")

    def get_exception_status(self) -> dict[str, Any]:
        """Get current exception fault status."""
        return {
            "injector_id": self.injector_id,
            "active_injections": len(self._active_injections),
            "exception_configs": {
                svc: {"type": exc_type, "rate": rate}
                for svc, (exc_type, rate) in self._exception_configs.items()
            },
        }
