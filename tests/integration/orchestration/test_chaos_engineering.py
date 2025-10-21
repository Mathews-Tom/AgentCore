"""
Integration tests for chaos engineering.

Tests end-to-end chaos scenarios with real fault injection and recovery validation.
"""

import asyncio

import pytest

from agentcore.orchestration.chaos.models import (
    ChaosScenario,
    ExperimentStatus,
    FaultConfig,
    FaultType,
    RecoveryStatus,
)
from agentcore.orchestration.chaos.orchestrator import ChaosOrchestrator
from agentcore.orchestration.patterns.circuit_breaker import (
    CircuitBreakerConfig,
    FaultToleranceCoordinator,
    RetryPolicy,
    RetryStrategy,
)
from agentcore.orchestration.patterns.saga import (
    SagaConfig,
    SagaDefinition,
    SagaOrchestrator,
    SagaStep,
)


@pytest.fixture
def fault_tolerance_coordinator() -> FaultToleranceCoordinator:
    """Create fault tolerance coordinator."""
    return FaultToleranceCoordinator()


@pytest.fixture
def saga_orchestrator() -> SagaOrchestrator:
    """Create saga orchestrator."""
    return SagaOrchestrator("chaos_test_saga", config=SagaConfig())


@pytest.fixture
def chaos_orchestrator(
    fault_tolerance_coordinator: FaultToleranceCoordinator,
    saga_orchestrator: SagaOrchestrator,
) -> ChaosOrchestrator:
    """Create chaos orchestrator with dependencies."""
    return ChaosOrchestrator(
        "chaos_test",
        fault_tolerance_coordinator=fault_tolerance_coordinator,
        saga_orchestrator=saga_orchestrator,
    )


class TestChaosEngineeringIntegration:
    """Integration tests for chaos engineering."""

    @pytest.mark.asyncio
    async def test_network_latency_with_circuit_breaker(
        self,
        chaos_orchestrator: ChaosOrchestrator,
        fault_tolerance_coordinator: FaultToleranceCoordinator,
    ) -> None:
        """Test network latency with circuit breaker protection."""
        # Register circuit breaker
        breaker = fault_tolerance_coordinator.register_circuit_breaker(
            "test_service",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=10,
            ),
        )

        # Create scenario
        scenario = ChaosScenario(
            name="Network Latency with Circuit Breaker",
            description="Test network latency triggers circuit breaker",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=500,
                    duration_seconds=2.0,
                )
            ],
            max_recovery_time_seconds=30.0,
            validate_circuit_breaker=True,
        )

        # Run experiment
        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        # Verify experiment completed
        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1
        assert result.recovery_metrics.recovery_status == RecoveryStatus.RECOVERED

        # Verify recovery time is reasonable
        assert result.recovery_metrics.time_to_recover_seconds is not None
        assert result.recovery_metrics.time_to_recover_seconds < 30.0

    @pytest.mark.asyncio
    async def test_service_crash_with_retry(
        self,
        chaos_orchestrator: ChaosOrchestrator,
        fault_tolerance_coordinator: FaultToleranceCoordinator,
    ) -> None:
        """Test service crash with retry mechanism."""
        # Create retry policy
        retry_policy = RetryPolicy(
            max_retries=3,
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay_seconds=0.1,
        )

        # Create scenario
        scenario = ChaosScenario(
            name="Service Crash with Retry",
            description="Test service crash triggers retry",
            faults=[
                FaultConfig(
                    fault_type=FaultType.SERVICE_CRASH,
                    target_service="test_service",
                    duration_seconds=1.0,
                )
            ],
            max_recovery_time_seconds=20.0,
            validate_retry_mechanism=True,
        )

        # Run experiment
        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        # Verify experiment completed
        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1
        assert result.retry_mechanism_validated

    @pytest.mark.asyncio
    async def test_timeout_with_saga_compensation(
        self,
        chaos_orchestrator: ChaosOrchestrator,
        saga_orchestrator: SagaOrchestrator,
    ) -> None:
        """Test timeout triggers saga compensation."""
        # Register saga definition
        saga = SagaDefinition(
            name="test_saga",
            description="Test saga for chaos testing",
            steps=[
                SagaStep(name="step1", order=1),
                SagaStep(name="step2", order=2),
            ],
        )
        saga_orchestrator.register_saga(saga)

        # Create scenario
        scenario = ChaosScenario(
            name="Timeout with Saga Compensation",
            description="Test timeout triggers saga compensation",
            faults=[
                FaultConfig(
                    fault_type=FaultType.TIMEOUT,
                    target_service="test_service",
                    duration_seconds=1.0,
                    intensity=1.0,
                )
            ],
            max_recovery_time_seconds=15.0,
            validate_saga_compensation=True,
        )

        # Run experiment
        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        # Verify experiment completed
        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1
        assert result.saga_compensation_validated

    @pytest.mark.asyncio
    async def test_multiple_faults_resilience(
        self,
        chaos_orchestrator: ChaosOrchestrator,
        fault_tolerance_coordinator: FaultToleranceCoordinator,
    ) -> None:
        """Test resilience against multiple concurrent faults."""
        # Register circuit breakers for multiple services
        for i in range(3):
            fault_tolerance_coordinator.register_circuit_breaker(
                f"service{i}",
                CircuitBreakerConfig(failure_threshold=5),
            )

        # Create complex scenario
        scenario = ChaosScenario(
            name="Multiple Faults Resilience Test",
            description="Test system resilience with multiple concurrent faults",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="service0",
                    latency_ms=200,
                    duration_seconds=1.5,
                ),
                FaultConfig(
                    fault_type=FaultType.SERVICE_CRASH,
                    target_service="service1",
                    duration_seconds=1.0,
                    delay_before_fault_seconds=0.5,
                ),
                FaultConfig(
                    fault_type=FaultType.TIMEOUT,
                    target_service="service2",
                    duration_seconds=1.0,
                    intensity=1.0,
                    delay_before_fault_seconds=1.0,
                ),
            ],
            max_recovery_time_seconds=30.0,
            min_availability_percentage=95.0,
            validate_circuit_breaker=True,
            validate_retry_mechanism=True,
        )

        # Run experiment
        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        # Verify experiment completed
        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 3
        assert result.faults_failed == 0

        # Verify recovery occurred
        assert result.recovery_metrics.recovery_status == RecoveryStatus.RECOVERED
        assert result.recovery_metrics.time_to_recover_seconds is not None

        # Verify availability maintained
        assert result.recovery_metrics.availability_percentage >= 95.0

    @pytest.mark.asyncio
    async def test_resilience_benchmark(
        self, chaos_orchestrator: ChaosOrchestrator
    ) -> None:
        """Test resilience benchmarking with multiple scenarios."""
        # Create scenarios with different fault types (no circuit breaker validation)
        scenarios = [
            ChaosScenario(
                name="Network Latency Benchmark",
                description="Test network latency resilience",
                faults=[
                    FaultConfig(
                        fault_type=FaultType.NETWORK_LATENCY,
                        target_service="benchmark_service",
                        latency_ms=300,
                        duration_seconds=1.0,
                    )
                ],
                max_recovery_time_seconds=20.0,
                validate_circuit_breaker=False,  # Don't validate circuit breaker
            ),
            ChaosScenario(
                name="Service Crash Benchmark",
                description="Test service crash resilience",
                faults=[
                    FaultConfig(
                        fault_type=FaultType.SERVICE_CRASH,
                        target_service="benchmark_service",
                        duration_seconds=1.0,
                    )
                ],
                max_recovery_time_seconds=20.0,
                validate_circuit_breaker=False,  # Don't validate circuit breaker
            ),
            ChaosScenario(
                name="Timeout Benchmark",
                description="Test timeout resilience",
                faults=[
                    FaultConfig(
                        fault_type=FaultType.TIMEOUT,
                        target_service="benchmark_service",
                        duration_seconds=1.0,
                        intensity=1.0,
                    )
                ],
                max_recovery_time_seconds=20.0,
                validate_circuit_breaker=False,  # Don't validate circuit breaker
            ),
        ]

        # Run scenario batch
        benchmark_id = await chaos_orchestrator.run_scenario_batch(
            scenarios, "Resilience Benchmark Suite"
        )

        # Get benchmark results
        benchmark = await chaos_orchestrator.get_benchmark(benchmark_id)

        # Verify benchmark metrics
        assert benchmark.total_experiments == 3
        assert benchmark.passed_experiments > 0

        # Verify aggregated metrics
        assert benchmark.avg_time_to_recover_seconds >= 0
        assert benchmark.avg_availability_percentage > 0

        # Verify SLO compliance calculated
        assert 0.0 <= benchmark.slo_compliance_rate <= 1.0

        # Get summary
        summary = benchmark.get_summary()
        assert summary["total_experiments"] == 3
        assert "pass_rate" in summary
        assert "avg_recovery_time" in summary
        assert "slo_compliance_rate" in summary

    @pytest.mark.asyncio
    async def test_packet_loss_recovery(
        self, chaos_orchestrator: ChaosOrchestrator
    ) -> None:
        """Test recovery from packet loss."""
        scenario = ChaosScenario(
            name="Packet Loss Recovery",
            description="Test recovery from packet loss",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_PACKET_LOSS,
                    target_service="test_service",
                    packet_loss_rate=0.5,  # 50% packet loss
                    duration_seconds=2.0,
                )
            ],
            max_recovery_time_seconds=25.0,
            validate_retry_mechanism=True,
        )

        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1
        assert result.recovery_metrics.recovery_status == RecoveryStatus.RECOVERED

    @pytest.mark.asyncio
    async def test_network_partition_recovery(
        self, chaos_orchestrator: ChaosOrchestrator
    ) -> None:
        """Test recovery from network partition."""
        scenario = ChaosScenario(
            name="Network Partition Recovery",
            description="Test recovery from complete network partition",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_PARTITION,
                    target_service="test_service",
                    duration_seconds=1.5,
                )
            ],
            max_recovery_time_seconds=30.0,
            validate_circuit_breaker=True,
            validate_retry_mechanism=True,
        )

        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1
        assert result.recovery_metrics.recovery_status == RecoveryStatus.RECOVERED

    @pytest.mark.asyncio
    async def test_cascading_failures(
        self,
        chaos_orchestrator: ChaosOrchestrator,
        fault_tolerance_coordinator: FaultToleranceCoordinator,
    ) -> None:
        """Test resilience against cascading failures."""
        # Register circuit breakers to prevent cascade
        for i in range(5):
            fault_tolerance_coordinator.register_circuit_breaker(
                f"cascade_service{i}",
                CircuitBreakerConfig(failure_threshold=2, timeout_seconds=5),
            )

        # Create cascading failure scenario
        scenario = ChaosScenario(
            name="Cascading Failures Test",
            description="Test circuit breakers prevent cascading failures",
            faults=[
                FaultConfig(
                    fault_type=FaultType.SERVICE_CRASH,
                    target_service=f"cascade_service{i}",
                    duration_seconds=0.5,
                    delay_before_fault_seconds=i * 0.2,
                )
                for i in range(5)
            ],
            max_recovery_time_seconds=40.0,
            validate_circuit_breaker=True,
        )

        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        # Verify all faults were injected
        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 5

        # Verify recovery occurred (circuit breakers should have helped)
        assert result.recovery_metrics.recovery_status == RecoveryStatus.RECOVERED

    @pytest.mark.asyncio
    async def test_performance_degradation_tracking(
        self, chaos_orchestrator: ChaosOrchestrator
    ) -> None:
        """Test tracking performance degradation during faults."""
        scenario = ChaosScenario(
            name="Performance Degradation Test",
            description="Test performance impact measurement",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="perf_service",
                    latency_ms=500,
                    duration_seconds=2.0,
                )
            ],
        )

        experiment_id = await chaos_orchestrator.create_experiment(scenario)
        result = await chaos_orchestrator.run_experiment(experiment_id)

        # Verify performance impact was tracked
        assert result.throughput_degradation_percentage >= 0
        assert result.latency_increase_percentage >= 0

        # With latency injection, should see performance impact
        assert result.latency_increase_percentage > 0

    @pytest.mark.asyncio
    async def test_slo_validation(
        self, chaos_orchestrator: ChaosOrchestrator
    ) -> None:
        """Test SLO validation in benchmark."""
        # Create benchmark with strict SLOs
        benchmark_id = await chaos_orchestrator.create_benchmark(
            name="SLO Validation Test",
            recovery_time_slo_seconds=5.0,  # Very strict
            availability_slo_percentage=99.9,  # Very strict
        )

        # Run multiple scenarios
        for i in range(3):
            scenario = ChaosScenario(
                name=f"SLO Test {i}",
                description=f"SLO validation scenario {i}",
                faults=[
                    FaultConfig(
                        fault_type=FaultType.NETWORK_LATENCY,
                        target_service="slo_service",
                        latency_ms=100,
                        duration_seconds=0.5,
                    )
                ],
            )

            experiment_id = await chaos_orchestrator.create_experiment(scenario)
            result = await chaos_orchestrator.run_experiment(experiment_id)
            await chaos_orchestrator.add_result_to_benchmark(benchmark_id, result)

        # Get benchmark and check SLO compliance
        benchmark = await chaos_orchestrator.get_benchmark(benchmark_id)
        assert benchmark.total_experiments == 3

        # SLO compliance rate should be calculated
        assert 0.0 <= benchmark.slo_compliance_rate <= 1.0
