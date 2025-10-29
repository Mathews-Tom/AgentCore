"""
Load testing framework for scalability validation

Generates synthetic load, executes scalability benchmarks,
and identifies performance bottlenecks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationObjective,
    MetricType,
)

logger = logging.getLogger(__name__)


class LoadPattern(str, Enum):
    """Load test patterns"""

    CONSTANT = "constant"  # Constant load
    RAMP_UP = "ramp_up"  # Gradually increasing
    SPIKE = "spike"  # Sudden burst
    WAVE = "wave"  # Oscillating load


@dataclass
class LoadProfile:
    """Load test profile configuration"""

    pattern: LoadPattern = LoadPattern.CONSTANT
    duration_seconds: int = 300
    target_rps: int = 10  # Requests per second
    max_concurrent: int = 100
    ramp_up_seconds: int = 60
    cool_down_seconds: int = 30


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during load test"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    response_times: list[float] = field(default_factory=list)


@dataclass
class BottleneckInfo:
    """Information about performance bottleneck"""

    component: str
    severity: str  # "low", "medium", "high"
    description: str
    metric_value: float
    threshold_value: float


@dataclass
class LoadTestResults:
    """Complete load test results"""

    profile: LoadProfile
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    metrics: PerformanceMetrics
    bottlenecks: list[BottleneckInfo]
    resource_usage: dict[str, Any]
    passed: bool
    notes: list[str] = field(default_factory=list)


class LoadTestRunner:
    """
    Executes load tests for optimization pipelines

    Key features:
    - Multiple load patterns
    - Synthetic request generation
    - Real-time metrics collection
    - Bottleneck detection
    - Resource monitoring
    """

    def __init__(
        self,
        optimization_handler: Callable[[OptimizationRequest], Awaitable[Any]],
        enable_monitoring: bool = True,
    ) -> None:
        """
        Initialize load test runner

        Args:
            optimization_handler: Function to handle optimization requests
            enable_monitoring: Enable resource monitoring
        """
        self.optimization_handler = optimization_handler
        self.enable_monitoring = enable_monitoring
        self._active_requests: set[str] = set()

    async def run_load_test(
        self,
        profile: LoadProfile,
        resource_manager: Any | None = None,
    ) -> LoadTestResults:
        """
        Execute load test with specified profile

        Args:
            profile: Load test profile
            resource_manager: Optional resource manager for monitoring

        Returns:
            LoadTestResults with performance metrics
        """
        logger.info(
            f"Starting load test: {profile.pattern.value} "
            f"({profile.target_rps} RPS, {profile.duration_seconds}s)"
        )

        start_time = datetime.utcnow()
        metrics = PerformanceMetrics()
        resource_usage: dict[str, Any] = {}

        # Generate load according to pattern
        try:
            await self._execute_load_pattern(profile, metrics)

            # Calculate final metrics
            self._calculate_metrics(metrics, start_time)

            # Collect resource usage
            if resource_manager and self.enable_monitoring:
                resource_usage = resource_manager.get_all_stats()

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Detect bottlenecks
            bottlenecks = self._detect_bottlenecks(metrics, resource_usage)

            # Determine if test passed
            passed = self._evaluate_test_results(metrics, profile)

            results = LoadTestResults(
                profile=profile,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metrics=metrics,
                bottlenecks=bottlenecks,
                resource_usage=resource_usage,
                passed=passed,
            )

            logger.info(
                f"Load test completed: {metrics.successful_requests}/{metrics.total_requests} successful "
                f"(RPS: {metrics.throughput_rps:.1f}, Error rate: {metrics.error_rate * 100:.1f}%)"
            )

            return results

        except Exception as e:
            logger.error(f"Load test failed: {e}")
            raise

    async def _execute_load_pattern(
        self, profile: LoadProfile, metrics: PerformanceMetrics
    ) -> None:
        """
        Execute load according to pattern

        Args:
            profile: Load profile
            metrics: Metrics to update
        """
        if profile.pattern == LoadPattern.CONSTANT:
            await self._execute_constant_load(profile, metrics)
        elif profile.pattern == LoadPattern.RAMP_UP:
            await self._execute_ramp_up_load(profile, metrics)
        elif profile.pattern == LoadPattern.SPIKE:
            await self._execute_spike_load(profile, metrics)
        elif profile.pattern == LoadPattern.WAVE:
            await self._execute_wave_load(profile, metrics)

    async def _execute_constant_load(
        self, profile: LoadProfile, metrics: PerformanceMetrics
    ) -> None:
        """Execute constant load pattern"""
        interval = 1.0 / profile.target_rps
        start = time.perf_counter()
        tasks = []

        while time.perf_counter() - start < profile.duration_seconds:
            # Limit concurrent requests
            while len(self._active_requests) >= profile.max_concurrent:
                await asyncio.sleep(0.01)

            # Create request
            request = self._create_synthetic_request()
            task = asyncio.create_task(
                self._execute_request(request, metrics)
            )
            tasks.append(task)

            # Wait for next request
            await asyncio.sleep(interval)

        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_ramp_up_load(
        self, profile: LoadProfile, metrics: PerformanceMetrics
    ) -> None:
        """Execute ramp-up load pattern"""
        ramp_duration = profile.ramp_up_seconds
        steady_duration = profile.duration_seconds - ramp_duration - profile.cool_down_seconds
        cool_down = profile.cool_down_seconds

        start = time.perf_counter()
        tasks = []

        # Ramp up phase
        for i in range(int(profile.target_rps * ramp_duration)):
            if time.perf_counter() - start > ramp_duration:
                break

            while len(self._active_requests) >= profile.max_concurrent:
                await asyncio.sleep(0.01)

            request = self._create_synthetic_request()
            task = asyncio.create_task(self._execute_request(request, metrics))
            tasks.append(task)

            # Gradually decrease interval
            progress = i / (profile.target_rps * ramp_duration)
            interval = (1.0 / profile.target_rps) * (1.0 - progress * 0.5)
            await asyncio.sleep(interval)

        # Steady state
        steady_start = time.perf_counter()
        while time.perf_counter() - steady_start < steady_duration:
            while len(self._active_requests) >= profile.max_concurrent:
                await asyncio.sleep(0.01)

            request = self._create_synthetic_request()
            task = asyncio.create_task(self._execute_request(request, metrics))
            tasks.append(task)

            await asyncio.sleep(1.0 / profile.target_rps)

        # Cool down
        await asyncio.sleep(cool_down)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_spike_load(
        self, profile: LoadProfile, metrics: PerformanceMetrics
    ) -> None:
        """Execute spike load pattern"""
        normal_rps = profile.target_rps // 2
        spike_rps = profile.target_rps * 3

        start = time.perf_counter()
        tasks = []

        while time.perf_counter() - start < profile.duration_seconds:
            elapsed = time.perf_counter() - start

            # Spike in the middle
            if profile.duration_seconds * 0.4 < elapsed < profile.duration_seconds * 0.6:
                current_rps = spike_rps
            else:
                current_rps = normal_rps

            while len(self._active_requests) >= profile.max_concurrent:
                await asyncio.sleep(0.01)

            request = self._create_synthetic_request()
            task = asyncio.create_task(self._execute_request(request, metrics))
            tasks.append(task)

            await asyncio.sleep(1.0 / current_rps)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_wave_load(
        self, profile: LoadProfile, metrics: PerformanceMetrics
    ) -> None:
        """Execute wave load pattern"""
        import math

        start = time.perf_counter()
        tasks = []

        while time.perf_counter() - start < profile.duration_seconds:
            elapsed = time.perf_counter() - start

            # Sine wave pattern
            wave_factor = (math.sin(2 * math.pi * elapsed / 60) + 1) / 2
            current_rps = int(profile.target_rps * (0.5 + 0.5 * wave_factor))

            while len(self._active_requests) >= profile.max_concurrent:
                await asyncio.sleep(0.01)

            request = self._create_synthetic_request()
            task = asyncio.create_task(self._execute_request(request, metrics))
            tasks.append(task)

            await asyncio.sleep(1.0 / max(current_rps, 1))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_request(
        self, request: OptimizationRequest, metrics: PerformanceMetrics
    ) -> None:
        """
        Execute single optimization request

        Args:
            request: Optimization request
            metrics: Metrics to update
        """
        request_id = id(request)
        self._active_requests.add(str(request_id))

        start = time.perf_counter()
        try:
            await self.optimization_handler(request)
            metrics.successful_requests += 1
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            metrics.failed_requests += 1
        finally:
            duration = time.perf_counter() - start
            metrics.response_times.append(duration)
            metrics.total_requests += 1
            self._active_requests.discard(str(request_id))

    def _create_synthetic_request(self) -> OptimizationRequest:
        """
        Create synthetic optimization request

        Returns:
            OptimizationRequest
        """
        return OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT, id="test-agent"
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE, target_value=0.9
                )
            ],
            algorithms=["miprov2"],
        )

    def _calculate_metrics(
        self, metrics: PerformanceMetrics, start_time: datetime
    ) -> None:
        """
        Calculate final performance metrics

        Args:
            metrics: Metrics to calculate
            start_time: Test start time
        """
        if not metrics.response_times:
            return

        response_times = sorted(metrics.response_times)
        count = len(response_times)

        metrics.avg_response_time = sum(response_times) / count
        metrics.min_response_time = response_times[0]
        metrics.max_response_time = response_times[-1]

        # Percentiles
        metrics.p50_response_time = response_times[int(count * 0.5)]
        metrics.p95_response_time = response_times[int(count * 0.95)]
        metrics.p99_response_time = response_times[int(count * 0.99)]

        # Throughput
        duration = (datetime.utcnow() - start_time).total_seconds()
        metrics.throughput_rps = metrics.total_requests / duration if duration > 0 else 0.0

        # Error rate
        metrics.error_rate = (
            metrics.failed_requests / metrics.total_requests
            if metrics.total_requests > 0
            else 0.0
        )

    def _detect_bottlenecks(
        self, metrics: PerformanceMetrics, resource_usage: dict[str, Any]
    ) -> list[BottleneckInfo]:
        """
        Detect performance bottlenecks

        Args:
            metrics: Performance metrics
            resource_usage: Resource usage statistics

        Returns:
            List of detected bottlenecks
        """
        bottlenecks = []

        # High error rate
        if metrics.error_rate > 0.05:
            bottlenecks.append(
                BottleneckInfo(
                    component="error_handling",
                    severity="high",
                    description=f"High error rate: {metrics.error_rate * 100:.1f}%",
                    metric_value=metrics.error_rate,
                    threshold_value=0.05,
                )
            )

        # High response time
        if metrics.p95_response_time > 5.0:
            bottlenecks.append(
                BottleneckInfo(
                    component="response_time",
                    severity="medium",
                    description=f"High P95 response time: {metrics.p95_response_time:.2f}s",
                    metric_value=metrics.p95_response_time,
                    threshold_value=5.0,
                )
            )

        # Low throughput
        if metrics.throughput_rps < 1.0:
            bottlenecks.append(
                BottleneckInfo(
                    component="throughput",
                    severity="high",
                    description=f"Low throughput: {metrics.throughput_rps:.2f} RPS",
                    metric_value=metrics.throughput_rps,
                    threshold_value=1.0,
                )
            )

        # Resource utilization
        if resource_usage:
            for pool_name, pool_stats in resource_usage.items():
                if isinstance(pool_stats, dict) and "utilization" in pool_stats:
                    util = pool_stats["utilization"]
                    if util > 0.9:
                        bottlenecks.append(
                            BottleneckInfo(
                                component=pool_name,
                                severity="high",
                                description=f"{pool_name} pool at {util * 100:.1f}% utilization",
                                metric_value=util,
                                threshold_value=0.9,
                            )
                        )

        return bottlenecks

    def _evaluate_test_results(
        self, metrics: PerformanceMetrics, profile: LoadProfile
    ) -> bool:
        """
        Evaluate if load test passed

        Args:
            metrics: Performance metrics
            profile: Load profile

        Returns:
            True if test passed
        """
        # Check error rate
        if metrics.error_rate > 0.1:
            return False

        # Check throughput
        if metrics.throughput_rps < profile.target_rps * 0.8:
            return False

        # Check response time
        if metrics.p95_response_time > 10.0:
            return False

        return True
