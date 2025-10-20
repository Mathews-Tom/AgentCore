"""
Unit tests for chaos orchestrator.
"""

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
    CircuitBreaker,
    CircuitBreakerConfig,
    FaultToleranceCoordinator,
)
from agentcore.orchestration.patterns.saga import SagaOrchestrator


class TestChaosOrchestrator:
    """Test chaos orchestrator."""

    @pytest.fixture
    def fault_tolerance_coordinator(self) -> FaultToleranceCoordinator:
        """Create fault tolerance coordinator."""
        return FaultToleranceCoordinator()

    @pytest.fixture
    def saga_orchestrator(self) -> SagaOrchestrator:
        """Create saga orchestrator."""
        return SagaOrchestrator("test_saga_orchestrator")

    @pytest.fixture
    def orchestrator(
        self,
        fault_tolerance_coordinator: FaultToleranceCoordinator,
        saga_orchestrator: SagaOrchestrator,
    ) -> ChaosOrchestrator:
        """Create chaos orchestrator."""
        return ChaosOrchestrator(
            "test_orchestrator",
            fault_tolerance_coordinator=fault_tolerance_coordinator,
            saga_orchestrator=saga_orchestrator,
        )

    @pytest.mark.asyncio
    async def test_create_experiment(self, orchestrator: ChaosOrchestrator) -> None:
        """Test creating experiment."""
        scenario = ChaosScenario(
            name="Test Scenario",
            description="Test description",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=100,
                    duration_seconds=1.0,
                )
            ],
        )

        experiment_id = await orchestrator.create_experiment(scenario)

        assert experiment_id is not None

        # Verify experiment was created
        status = await orchestrator.get_orchestrator_status()
        assert status["total_experiments"] == 1

    @pytest.mark.asyncio
    async def test_run_experiment_network_latency(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test running experiment with network latency."""
        scenario = ChaosScenario(
            name="Network Latency Test",
            description="Test network latency injection",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=100,
                    duration_seconds=0.5,
                )
            ],
            max_recovery_time_seconds=10.0,
            min_availability_percentage=95.0,
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        result = await orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1
        assert result.faults_failed == 0
        assert result.recovery_metrics.recovery_status == RecoveryStatus.RECOVERED
        assert result.duration_seconds is not None
        assert len(result.logs) > 0

    @pytest.mark.asyncio
    async def test_run_experiment_service_crash(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test running experiment with service crash."""
        scenario = ChaosScenario(
            name="Service Crash Test",
            description="Test service crash injection",
            faults=[
                FaultConfig(
                    fault_type=FaultType.SERVICE_CRASH,
                    target_service="test_service",
                    duration_seconds=0.5,
                )
            ],
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        result = await orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1

    @pytest.mark.asyncio
    async def test_run_experiment_timeout(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test running experiment with timeout."""
        scenario = ChaosScenario(
            name="Timeout Test",
            description="Test timeout injection",
            faults=[
                FaultConfig(
                    fault_type=FaultType.TIMEOUT,
                    target_service="test_service",
                    duration_seconds=0.5,
                    intensity=1.0,
                )
            ],
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        result = await orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1

    @pytest.mark.asyncio
    async def test_run_experiment_multiple_faults(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test running experiment with multiple faults."""
        scenario = ChaosScenario(
            name="Multiple Faults Test",
            description="Test multiple fault injections",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="service1",
                    latency_ms=100,
                    duration_seconds=0.3,
                ),
                FaultConfig(
                    fault_type=FaultType.SERVICE_CRASH,
                    target_service="service2",
                    duration_seconds=0.3,
                ),
                FaultConfig(
                    fault_type=FaultType.TIMEOUT,
                    target_service="service3",
                    duration_seconds=0.3,
                    intensity=1.0,
                ),
            ],
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        result = await orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 3
        assert result.faults_failed == 0

    @pytest.mark.asyncio
    async def test_run_experiment_dry_run(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test running experiment in dry run mode."""
        scenario = ChaosScenario(
            name="Dry Run Test",
            description="Test dry run mode",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=100,
                    duration_seconds=0.3,
                )
            ],
        )

        experiment_id = await orchestrator.create_experiment(
            scenario, dry_run=True
        )
        result = await orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        assert result.faults_injected == 1

        # Verify no actual faults were injected
        status = await orchestrator.get_orchestrator_status()
        network_status = status["network_injector"]
        assert network_status["active_injections"] == 0

    @pytest.mark.asyncio
    async def test_create_benchmark(self, orchestrator: ChaosOrchestrator) -> None:
        """Test creating benchmark."""
        benchmark_id = await orchestrator.create_benchmark(
            name="Test Benchmark",
            description="Test benchmark description",
            recovery_time_slo_seconds=30.0,
            availability_slo_percentage=99.5,
        )

        assert benchmark_id is not None

        benchmark = await orchestrator.get_benchmark(benchmark_id)
        assert benchmark.name == "Test Benchmark"
        assert benchmark.recovery_time_slo_seconds == 30.0
        assert benchmark.availability_slo_percentage == 99.5

    @pytest.mark.asyncio
    async def test_add_result_to_benchmark(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test adding result to benchmark."""
        # Create benchmark
        benchmark_id = await orchestrator.create_benchmark("Test Benchmark")

        # Run experiment
        scenario = ChaosScenario(
            name="Test Scenario",
            description="Test scenario",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=100,
                    duration_seconds=0.3,
                )
            ],
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        result = await orchestrator.run_experiment(experiment_id)

        # Add to benchmark
        await orchestrator.add_result_to_benchmark(benchmark_id, result)

        # Verify benchmark updated
        benchmark = await orchestrator.get_benchmark(benchmark_id)
        assert benchmark.total_experiments == 1
        assert len(benchmark.experiment_results) == 1

    @pytest.mark.asyncio
    async def test_run_scenario_batch(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test running batch of scenarios."""
        scenarios = [
            ChaosScenario(
                name=f"Scenario {i}",
                description=f"Test scenario {i}",
                faults=[
                    FaultConfig(
                        fault_type=FaultType.NETWORK_LATENCY,
                        target_service=f"service{i}",
                        latency_ms=100,
                        duration_seconds=0.3,
                    )
                ],
            )
            for i in range(3)
        ]

        benchmark_id = await orchestrator.run_scenario_batch(
            scenarios, "Batch Benchmark"
        )

        benchmark = await orchestrator.get_benchmark(benchmark_id)
        assert benchmark.total_experiments == 3
        assert len(benchmark.experiment_results) == 3

    @pytest.mark.asyncio
    async def test_benchmark_metrics_calculation(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test benchmark metrics calculation."""
        benchmark_id = await orchestrator.create_benchmark("Metrics Test")

        # Run multiple experiments
        for i in range(5):
            scenario = ChaosScenario(
                name=f"Scenario {i}",
                description=f"Test scenario {i}",
                faults=[
                    FaultConfig(
                        fault_type=FaultType.NETWORK_LATENCY,
                        target_service=f"service{i}",
                        latency_ms=100,
                        duration_seconds=0.3,
                    )
                ],
            )

            experiment_id = await orchestrator.create_experiment(scenario)
            result = await orchestrator.run_experiment(experiment_id)
            await orchestrator.add_result_to_benchmark(benchmark_id, result)

        # Get benchmark and verify metrics
        benchmark = await orchestrator.get_benchmark(benchmark_id)
        assert benchmark.total_experiments == 5

        # Verify aggregated metrics are calculated
        assert benchmark.avg_time_to_recover_seconds >= 0
        assert benchmark.avg_availability_percentage > 0

        # Get summary
        summary = benchmark.get_summary()
        assert summary["total_experiments"] == 5
        assert "pass_rate" in summary
        assert "avg_recovery_time" in summary

    @pytest.mark.asyncio
    async def test_cleanup_after_experiment(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test cleanup after experiment."""
        scenario = ChaosScenario(
            name="Cleanup Test",
            description="Test cleanup",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=100,
                    duration_seconds=0.3,
                )
            ],
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        await orchestrator.run_experiment(experiment_id)

        # Verify all faults are cleaned up
        status = await orchestrator.get_orchestrator_status()
        assert status["network_injector"]["active_injections"] == 0
        assert status["service_injector"]["active_injections"] == 0
        assert status["timeout_injector"]["active_injections"] == 0

    @pytest.mark.asyncio
    async def test_experiment_validation(
        self,
        orchestrator: ChaosOrchestrator,
        fault_tolerance_coordinator: FaultToleranceCoordinator,
    ) -> None:
        """Test experiment validation with recovery mechanisms."""
        # Register circuit breaker
        fault_tolerance_coordinator.register_circuit_breaker(
            "test_service",
            CircuitBreakerConfig(failure_threshold=1),
        )

        scenario = ChaosScenario(
            name="Validation Test",
            description="Test validation",
            faults=[
                FaultConfig(
                    fault_type=FaultType.NETWORK_LATENCY,
                    target_service="test_service",
                    latency_ms=100,
                    duration_seconds=0.3,
                )
            ],
            validate_circuit_breaker=True,
            validate_saga_compensation=True,
            validate_retry_mechanism=True,
        )

        experiment_id = await orchestrator.create_experiment(scenario)
        result = await orchestrator.run_experiment(experiment_id)

        assert result.status == ExperimentStatus.COMPLETED
        # Validations should pass (even if mocked)
        assert result.retry_mechanism_validated

    @pytest.mark.asyncio
    async def test_get_orchestrator_status(
        self, orchestrator: ChaosOrchestrator
    ) -> None:
        """Test getting orchestrator status."""
        status = await orchestrator.get_orchestrator_status()

        assert status["orchestrator_id"] == "test_orchestrator"
        assert "total_experiments" in status
        assert "active_experiments" in status
        assert "benchmarks" in status
        assert "network_injector" in status
        assert "service_injector" in status
        assert "timeout_injector" in status
