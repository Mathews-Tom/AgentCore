"""
Chaos Engineering Models

Data models for chaos experiments, scenarios, and resilience benchmarks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FaultType(str, Enum):
    """Types of faults to inject."""

    NETWORK_LATENCY = "network_latency"  # Inject network delays
    NETWORK_PARTITION = "network_partition"  # Partition network segments
    NETWORK_PACKET_LOSS = "network_packet_loss"  # Drop packets
    SERVICE_CRASH = "service_crash"  # Crash service instances
    SERVICE_HANG = "service_hang"  # Hang service responses
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Exhaust CPU/memory
    TIMEOUT = "timeout"  # Force timeouts
    EXCEPTION = "exception"  # Throw exceptions
    DATA_CORRUPTION = "data_corruption"  # Corrupt data
    DEPENDENCY_FAILURE = "dependency_failure"  # Fail dependencies


class ExperimentStatus(str, Enum):
    """Status of chaos experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class RecoveryStatus(str, Enum):
    """Recovery status after fault injection."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    RECOVERED = "recovered"
    PARTIAL_RECOVERY = "partial_recovery"
    FAILED = "failed"


class FaultConfig(BaseModel):
    """Configuration for a specific fault injection."""

    fault_type: FaultType = Field(description="Type of fault to inject")
    target_service: str = Field(description="Service to inject fault into")

    # Fault parameters
    duration_seconds: float = Field(default=10.0, description="Fault duration")
    intensity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Fault intensity (0-1)"
    )

    # Network fault parameters
    latency_ms: int | None = Field(
        default=None, description="Network latency to inject (ms)"
    )
    packet_loss_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Packet loss rate (0-1)"
    )
    bandwidth_limit_kbps: int | None = Field(
        default=None, description="Bandwidth limit (kbps)"
    )

    # Service fault parameters
    crash_probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Crash probability (0-1)"
    )
    hang_duration_seconds: float | None = Field(
        default=None, description="Hang duration"
    )

    # Resource fault parameters
    cpu_percentage: int | None = Field(
        default=None, ge=0, le=100, description="CPU to consume (%)"
    )
    memory_mb: int | None = Field(default=None, description="Memory to consume (MB)")

    # Exception fault parameters
    exception_type: str | None = Field(default=None, description="Exception to raise")
    exception_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Exception rate (0-1)"
    )

    # General parameters
    delay_before_fault_seconds: float = Field(
        default=0.0, description="Delay before injecting fault"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecoveryMetrics(BaseModel):
    """Metrics for recovery from fault injection."""

    recovery_status: RecoveryStatus = Field(default=RecoveryStatus.NOT_STARTED)

    # Recovery timeline
    fault_injected_at: datetime | None = None
    fault_detected_at: datetime | None = None
    recovery_started_at: datetime | None = None
    recovery_completed_at: datetime | None = None

    # Recovery measurements
    time_to_detect_seconds: float | None = Field(
        default=None, description="Time to detect fault"
    )
    time_to_recover_seconds: float | None = Field(
        default=None, description="Time to full recovery"
    )

    # Impact measurements
    requests_failed: int = Field(default=0, description="Failed requests during fault")
    requests_succeeded: int = Field(
        default=0, description="Successful requests during recovery"
    )
    availability_percentage: float = Field(
        default=100.0, description="Availability during experiment"
    )

    # Recovery actions
    compensation_triggered: bool = Field(
        default=False, description="Saga compensation triggered"
    )
    circuit_breaker_opened: bool = Field(
        default=False, description="Circuit breaker opened"
    )
    retry_attempts: int = Field(default=0, description="Retry attempts")
    fallback_used: bool = Field(default=False, description="Fallback mechanism used")

    error_messages: list[str] = Field(
        default_factory=list, description="Errors encountered"
    )

    model_config = {"frozen": False}


class ChaosScenario(BaseModel):
    """
    Chaos engineering scenario.

    Defines a series of faults to inject and expected outcomes.
    """

    scenario_id: UUID = Field(default_factory=uuid4, description="Scenario identifier")
    name: str = Field(description="Scenario name")
    description: str = Field(description="Scenario description")

    # Fault configuration
    faults: list[FaultConfig] = Field(description="Faults to inject")

    # Expected behavior
    expected_recovery: bool = Field(
        default=True, description="Should system recover"
    )
    max_recovery_time_seconds: float = Field(
        default=60.0, description="Max acceptable recovery time"
    )
    min_availability_percentage: float = Field(
        default=99.0, description="Min availability during fault"
    )

    # Test configuration
    concurrent_requests: int = Field(
        default=10, description="Concurrent requests during test"
    )
    test_duration_seconds: float = Field(
        default=120.0, description="Total test duration"
    )

    # Validation
    validate_circuit_breaker: bool = Field(
        default=True, description="Validate circuit breaker activation"
    )
    validate_saga_compensation: bool = Field(
        default=True, description="Validate saga compensation"
    )
    validate_retry_mechanism: bool = Field(
        default=True, description="Validate retry mechanism"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Result of a chaos experiment."""

    result_id: UUID = Field(default_factory=uuid4, description="Result identifier")
    experiment_id: UUID = Field(description="Experiment identifier")
    scenario_id: UUID = Field(description="Scenario identifier")

    status: ExperimentStatus = Field(description="Experiment status")

    # Timeline
    started_at: datetime = Field(description="Experiment start time")
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Fault injection results
    faults_injected: int = Field(default=0, description="Number of faults injected")
    faults_failed: int = Field(default=0, description="Faults that failed to inject")

    # Recovery metrics
    recovery_metrics: RecoveryMetrics = Field(
        default_factory=RecoveryMetrics, description="Recovery measurements"
    )

    # Validation results
    circuit_breaker_validated: bool = Field(
        default=False, description="Circuit breaker validated"
    )
    saga_compensation_validated: bool = Field(
        default=False, description="Saga compensation validated"
    )
    retry_mechanism_validated: bool = Field(
        default=False, description="Retry mechanism validated"
    )

    # Performance impact
    throughput_degradation_percentage: float = Field(
        default=0.0, description="Throughput degradation"
    )
    latency_increase_percentage: float = Field(
        default=0.0, description="Latency increase"
    )

    # Overall assessment
    passed: bool = Field(description="Experiment passed acceptance criteria")
    failure_reason: str | None = Field(
        default=None, description="Reason for failure"
    )

    logs: list[str] = Field(default_factory=list, description="Experiment logs")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class ChaosExperiment(BaseModel):
    """
    Chaos engineering experiment.

    Executes a scenario and collects results.
    """

    experiment_id: UUID = Field(
        default_factory=uuid4, description="Experiment identifier"
    )
    scenario: ChaosScenario = Field(description="Scenario to execute")

    status: ExperimentStatus = Field(
        default=ExperimentStatus.PENDING, description="Experiment status"
    )

    # Execution tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_fault_index: int = Field(default=0, description="Current fault being injected")

    # Results
    result: ExperimentResult | None = None

    # Control
    abort_on_failure: bool = Field(
        default=False, description="Abort experiment on first failure"
    )
    dry_run: bool = Field(default=False, description="Dry run without actual faults")

    model_config = {"frozen": False}


class ResilienceBenchmark(BaseModel):
    """
    Resilience benchmark metrics.

    Aggregated metrics across multiple chaos experiments.
    """

    benchmark_id: UUID = Field(
        default_factory=uuid4, description="Benchmark identifier"
    )
    name: str = Field(description="Benchmark name")
    description: str | None = None

    # Experiment tracking
    total_experiments: int = Field(default=0, description="Total experiments run")
    passed_experiments: int = Field(default=0, description="Passed experiments")
    failed_experiments: int = Field(default=0, description="Failed experiments")

    # Recovery metrics (aggregated)
    avg_time_to_detect_seconds: float = Field(
        default=0.0, description="Average detection time"
    )
    avg_time_to_recover_seconds: float = Field(
        default=0.0, description="Average recovery time"
    )
    avg_availability_percentage: float = Field(
        default=100.0, description="Average availability"
    )

    # Recovery mechanism success rates
    circuit_breaker_success_rate: float = Field(
        default=0.0, description="Circuit breaker success rate"
    )
    saga_compensation_success_rate: float = Field(
        default=0.0, description="Saga compensation success rate"
    )
    retry_success_rate: float = Field(default=0.0, description="Retry success rate")

    # Performance impact
    avg_throughput_degradation: float = Field(
        default=0.0, description="Average throughput degradation"
    )
    avg_latency_increase: float = Field(default=0.0, description="Average latency increase")

    # SLO compliance
    recovery_time_slo_seconds: float = Field(
        default=60.0, description="Recovery time SLO"
    )
    availability_slo_percentage: float = Field(
        default=99.9, description="Availability SLO"
    )

    slo_compliance_rate: float = Field(
        default=0.0, description="SLO compliance rate (0-1)"
    )

    # Timeline
    first_experiment_at: datetime | None = None
    last_experiment_at: datetime | None = None

    # Detailed results
    experiment_results: list[ExperimentResult] = Field(
        default_factory=list, description="All experiment results"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": False}

    def add_result(self, result: ExperimentResult) -> None:
        """Add experiment result to benchmark."""
        self.experiment_results.append(result)
        self.total_experiments += 1

        if result.passed:
            self.passed_experiments += 1
        else:
            self.failed_experiments += 1

        # Update timestamps
        if not self.first_experiment_at:
            self.first_experiment_at = result.started_at
        self.last_experiment_at = result.completed_at or datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

        # Recalculate aggregated metrics
        self._recalculate_metrics()

    def _recalculate_metrics(self) -> None:
        """Recalculate aggregated metrics."""
        if not self.experiment_results:
            return

        # Recovery time metrics
        detection_times = [
            r.recovery_metrics.time_to_detect_seconds
            for r in self.experiment_results
            if r.recovery_metrics.time_to_detect_seconds is not None
        ]
        if detection_times:
            self.avg_time_to_detect_seconds = sum(detection_times) / len(detection_times)

        recovery_times = [
            r.recovery_metrics.time_to_recover_seconds
            for r in self.experiment_results
            if r.recovery_metrics.time_to_recover_seconds is not None
        ]
        if recovery_times:
            self.avg_time_to_recover_seconds = sum(recovery_times) / len(recovery_times)

        # Availability
        availabilities = [
            r.recovery_metrics.availability_percentage for r in self.experiment_results
        ]
        if availabilities:
            self.avg_availability_percentage = sum(availabilities) / len(availabilities)

        # Recovery mechanism success rates
        total = len(self.experiment_results)
        self.circuit_breaker_success_rate = (
            sum(1 for r in self.experiment_results if r.circuit_breaker_validated)
            / total
        )
        self.saga_compensation_success_rate = (
            sum(1 for r in self.experiment_results if r.saga_compensation_validated)
            / total
        )
        self.retry_success_rate = (
            sum(1 for r in self.experiment_results if r.retry_mechanism_validated)
            / total
        )

        # Performance impact
        throughput_degradations = [
            r.throughput_degradation_percentage for r in self.experiment_results
        ]
        if throughput_degradations:
            self.avg_throughput_degradation = sum(throughput_degradations) / len(
                throughput_degradations
            )

        latency_increases = [
            r.latency_increase_percentage for r in self.experiment_results
        ]
        if latency_increases:
            self.avg_latency_increase = sum(latency_increases) / len(latency_increases)

        # SLO compliance
        slo_compliant = sum(
            1
            for r in self.experiment_results
            if (
                (
                    r.recovery_metrics.time_to_recover_seconds is None
                    or r.recovery_metrics.time_to_recover_seconds
                    <= self.recovery_time_slo_seconds
                )
                and r.recovery_metrics.availability_percentage
                >= self.availability_slo_percentage
            )
        )
        self.slo_compliance_rate = slo_compliant / total

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark summary."""
        return {
            "benchmark_id": str(self.benchmark_id),
            "name": self.name,
            "total_experiments": self.total_experiments,
            "passed_experiments": self.passed_experiments,
            "failed_experiments": self.failed_experiments,
            "pass_rate": (
                self.passed_experiments / self.total_experiments
                if self.total_experiments > 0
                else 0.0
            ),
            "avg_recovery_time": self.avg_time_to_recover_seconds,
            "avg_availability": self.avg_availability_percentage,
            "slo_compliance_rate": self.slo_compliance_rate,
            "circuit_breaker_success_rate": self.circuit_breaker_success_rate,
            "saga_compensation_success_rate": self.saga_compensation_success_rate,
            "retry_success_rate": self.retry_success_rate,
        }
