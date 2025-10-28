"""
Tests for MLflow experiment tracking integration
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pytest

from agentcore.dspy_optimization.models import (
    MetricType,
    OptimizationConstraints,
    OptimizationDetails,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    OptimizationScope,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.tracking.mlflow_tracker import (
    MLflowConfig,
    MLflowTracker,
)


@pytest.fixture
def mlflow_config() -> MLflowConfig:
    """Create test MLflow configuration"""
    # Use temporary directory for artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file://{tmpdir}/mlruns"
        artifact_location = f"{tmpdir}/artifacts"

        return MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name="test-dspy-optimization",
            artifact_location=artifact_location,
        )


@pytest.fixture
def mlflow_tracker(mlflow_config: MLflowConfig) -> MLflowTracker:
    """Create MLflow tracker for testing"""
    return MLflowTracker(config=mlflow_config)


@pytest.fixture
def optimization_request() -> OptimizationRequest:
    """Create test optimization request"""
    return OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent-001",
            scope=OptimizationScope.INDIVIDUAL,
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.85,
                weight=0.5,
            ),
            OptimizationObjective(
                metric=MetricType.COST_EFFICIENCY,
                target_value=0.9,
                weight=0.3,
            ),
        ],
        algorithms=["miprov2", "gepa"],
        constraints=OptimizationConstraints(
            max_optimization_time=3600,
            min_improvement_threshold=0.1,
            max_resource_usage=0.3,
        ),
    )


@pytest.fixture
def baseline_metrics() -> PerformanceMetrics:
    """Create baseline performance metrics"""
    return PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.15,
        avg_latency_ms=2000,
        quality_score=0.7,
    )


@pytest.fixture
def optimized_metrics() -> PerformanceMetrics:
    """Create optimized performance metrics"""
    return PerformanceMetrics(
        success_rate=0.92,
        avg_cost_per_task=0.10,
        avg_latency_ms=1500,
        quality_score=0.85,
    )


@pytest.fixture
def optimization_result(
    baseline_metrics: PerformanceMetrics,
    optimized_metrics: PerformanceMetrics,
) -> OptimizationResult:
    """Create optimization result"""
    return OptimizationResult(
        status=OptimizationStatus.COMPLETED,
        baseline_performance=baseline_metrics,
        optimized_performance=optimized_metrics,
        improvement_percentage=22.7,
        statistical_significance=0.001,
        optimization_details=OptimizationDetails(
            algorithm_used="miprov2",
            iterations=45,
            key_improvements=[
                "Enhanced reasoning chain structure",
                "Improved error handling patterns",
                "Optimized tool selection logic",
            ],
            parameters={"learning_rate": 0.01, "batch_size": 32},
        ),
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
class TestMLflowConfig:
    """Tests for MLflow configuration"""

    def test_config_defaults(self) -> None:
        """Test default configuration values"""
        config = MLflowConfig()

        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "dspy-optimization"
        assert config.artifact_location is None
        assert config.registry_uri is None

    def test_config_custom(self) -> None:
        """Test custom configuration"""
        config = MLflowConfig(
            tracking_uri="http://mlflow.example.com:5000",
            experiment_name="custom-experiment",
            artifact_location="s3://bucket/artifacts",
            registry_uri="http://registry.example.com:5000",
        )

        assert config.tracking_uri == "http://mlflow.example.com:5000"
        assert config.experiment_name == "custom-experiment"
        assert config.artifact_location == "s3://bucket/artifacts"
        assert config.registry_uri == "http://registry.example.com:5000"


@pytest.mark.asyncio
class TestMLflowTracker:
    """Tests for MLflow tracker"""

    async def test_tracker_initialization(
        self, mlflow_config: MLflowConfig
    ) -> None:
        """Test tracker initialization"""
        tracker = MLflowTracker(config=mlflow_config)

        assert tracker.config == mlflow_config
        assert tracker.experiment is not None
        assert tracker.experiment.name == mlflow_config.experiment_name

    async def test_start_run(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test starting MLflow run"""
        run_id = await mlflow_tracker.start_run(
            request=optimization_request,
            run_name="test-run",
            tags={"environment": "test"},
        )

        assert run_id is not None

        # Verify run exists
        run = mlflow.get_run(run_id)
        assert run.info.run_name == "test-run"
        assert run.data.tags["target_type"] == "agent"
        assert run.data.tags["target_id"] == "test-agent-001"
        assert run.data.tags["environment"] == "test"

        # Clean up
        await mlflow_tracker.end_run()

    async def test_log_baseline_metrics(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test logging baseline metrics"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)
        await mlflow_tracker.log_baseline_metrics(baseline_metrics)

        # Verify metrics logged
        run = mlflow.get_run(run_id)
        assert run.data.metrics["baseline_success_rate"] == 0.75
        assert run.data.metrics["baseline_avg_cost"] == 0.15
        assert run.data.metrics["baseline_avg_latency_ms"] == 2000.0
        assert run.data.metrics["baseline_quality_score"] == 0.7

        await mlflow_tracker.end_run()

    async def test_log_optimized_metrics(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        optimized_metrics: PerformanceMetrics,
    ) -> None:
        """Test logging optimized metrics"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)
        await mlflow_tracker.log_optimized_metrics(optimized_metrics)

        # Verify metrics logged
        run = mlflow.get_run(run_id)
        assert run.data.metrics["optimized_success_rate"] == 0.92
        assert run.data.metrics["optimized_avg_cost"] == 0.10
        assert run.data.metrics["optimized_avg_latency_ms"] == 1500.0
        assert run.data.metrics["optimized_quality_score"] == 0.85

        await mlflow_tracker.end_run()

    async def test_log_improvement_metrics(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        optimization_result: OptimizationResult,
    ) -> None:
        """Test logging improvement metrics"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)
        await mlflow_tracker.log_improvement_metrics(optimization_result)

        # Verify improvement metrics
        run = mlflow.get_run(run_id)
        assert run.data.metrics["success_rate_improvement"] == pytest.approx(0.17, abs=0.01)
        assert run.data.metrics["cost_reduction"] == pytest.approx(0.05, abs=0.01)
        assert run.data.metrics["latency_reduction_ms"] == 500.0
        assert run.data.metrics["quality_improvement"] == pytest.approx(0.15, abs=0.01)
        assert run.data.metrics["improvement_percentage"] == 22.7
        assert run.data.metrics["statistical_significance"] == 0.001

        await mlflow_tracker.end_run()

    async def test_log_optimization_details(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test logging optimization details"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)

        await mlflow_tracker.log_optimization_details(
            algorithm="miprov2",
            iterations=45,
            key_improvements=[
                "Enhanced reasoning",
                "Improved error handling",
            ],
            parameters={"learning_rate": 0.01},
        )

        # Verify details logged
        run = mlflow.get_run(run_id)
        assert run.data.params["algorithm"] == "miprov2"
        assert run.data.metrics["iterations"] == 45.0

        await mlflow_tracker.end_run()

    async def test_log_model_artifact(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test logging model artifact"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)

        # Create test model
        test_model = {"type": "optimized_agent", "version": "1.0"}
        metadata = {"algorithm": "miprov2", "improvement": 0.25}

        await mlflow_tracker.log_model_artifact(
            model=test_model,
            artifact_name="test_model",
            metadata=metadata,
        )

        await mlflow_tracker.end_run()

        # Verify artifact exists (artifacts are under models/ directory)
        run = mlflow.get_run(run_id)
        artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "models")]
        assert any("test_model.pkl" in a for a in artifacts)

    async def test_load_model_artifact(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test loading model artifact"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)

        # Log model
        test_model = {"type": "optimized_agent", "version": "1.0"}
        await mlflow_tracker.log_model_artifact(
            model=test_model,
            artifact_name="test_model",
        )

        await mlflow_tracker.end_run()

        # Load model
        loaded_model = await mlflow_tracker.load_model_artifact(
            run_id=run_id,
            artifact_name="test_model",
        )

        assert loaded_model == test_model

    async def test_log_training_data(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test logging training data"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)

        training_data = [
            {"input": "test1", "output": "result1"},
            {"input": "test2", "output": "result2"},
        ]

        await mlflow_tracker.log_training_data(training_data)

        await mlflow_tracker.end_run()

        # Verify artifact exists
        run = mlflow.get_run(run_id)
        artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
        assert any("training_data.json" in a for a in artifacts)

    async def test_log_result(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        optimization_result: OptimizationResult,
    ) -> None:
        """Test logging complete optimization result"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)
        await mlflow_tracker.log_result(optimization_result)

        # Verify all metrics logged
        run = mlflow.get_run(run_id)

        assert run.data.tags["status"] == "completed"
        assert run.data.metrics["baseline_success_rate"] == 0.75
        assert run.data.metrics["optimized_success_rate"] == 0.92
        assert run.data.metrics["improvement_percentage"] == 22.7
        assert run.data.params["algorithm"] == "miprov2"
        assert run.data.metrics["iterations"] == 45.0

        await mlflow_tracker.end_run()

    async def test_search_runs(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        optimization_result: OptimizationResult,
    ) -> None:
        """Test searching for runs"""
        # Create multiple runs
        for i in range(3):
            run_id = await mlflow_tracker.start_run(
                request=optimization_request,
                run_name=f"test-run-{i}",
            )
            result = optimization_result.model_copy(deep=True)
            result.improvement_percentage = 20.0 + i * 5
            await mlflow_tracker.log_result(result)
            await mlflow_tracker.end_run()

        # Search for runs
        runs = await mlflow_tracker.search_runs(max_results=10)
        assert len(runs) >= 3

    async def test_get_best_run(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        optimization_result: OptimizationResult,
    ) -> None:
        """Test getting best run"""
        # Create runs with different improvements
        best_improvement = 30.0
        for i, improvement in enumerate([20.0, best_improvement, 15.0]):
            run_id = await mlflow_tracker.start_run(
                request=optimization_request,
                run_name=f"test-run-{i}",
            )
            result = optimization_result.model_copy(deep=True)
            result.improvement_percentage = improvement
            await mlflow_tracker.log_result(result)
            await mlflow_tracker.end_run()

        # Get best run - search_runs returns pandas DataFrame by default
        # We need to search manually
        runs = mlflow.search_runs(
            experiment_ids=[mlflow_tracker.experiment.experiment_id],
            max_results=1,
            order_by=["metrics.improvement_percentage DESC"],
        )

        assert len(runs) > 0
        assert runs.iloc[0]["metrics.improvement_percentage"] == best_improvement

    async def test_end_run_success(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test ending run with success status"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)
        await mlflow_tracker.end_run(status="FINISHED")

        run = mlflow.get_run(run_id)
        assert run.info.status == "FINISHED"

    async def test_end_run_failure(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test ending run with failure status"""
        run_id = await mlflow_tracker.start_run(request=optimization_request)
        await mlflow_tracker.end_run(status="FAILED")

        run = mlflow.get_run(run_id)
        assert run.info.status == "FAILED"


@pytest.mark.asyncio
class TestMLflowTrackerIntegration:
    """Integration tests for MLflow tracker"""

    async def test_complete_optimization_workflow(
        self,
        mlflow_tracker: MLflowTracker,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        optimized_metrics: PerformanceMetrics,
        optimization_result: OptimizationResult,
    ) -> None:
        """Test complete optimization tracking workflow"""
        # Start run
        run_id = await mlflow_tracker.start_run(
            request=optimization_request,
            run_name="complete-workflow-test",
        )

        # Log baseline
        await mlflow_tracker.log_baseline_metrics(baseline_metrics)

        # Log training data
        training_data = [{"input": "test", "output": "result"}]
        await mlflow_tracker.log_training_data(training_data)

        # Log optimization result
        await mlflow_tracker.log_result(optimization_result)

        # Log model
        test_model = {"optimized": True}
        await mlflow_tracker.log_model_artifact(
            model=test_model,
            artifact_name="final_model",
        )

        # End run
        await mlflow_tracker.end_run(status="FINISHED")

        # Verify complete workflow
        run = mlflow.get_run(run_id)

        assert run.info.status == "FINISHED"
        assert run.data.metrics["baseline_success_rate"] == 0.75
        assert run.data.metrics["optimized_success_rate"] == 0.92
        assert run.data.metrics["improvement_percentage"] == 22.7
        assert run.data.params["algorithm"] == "miprov2"

        # Verify artifacts
        all_artifacts = mlflow.MlflowClient().list_artifacts(run_id)
        artifact_paths = [a.path for a in all_artifacts]
        model_artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "models")]

        assert any("training_data.json" in a for a in artifact_paths)
        assert any("final_model.pkl" in a for a in model_artifacts)

    async def test_multiple_experiments(
        self,
        mlflow_config: MLflowConfig,
        optimization_request: OptimizationRequest,
    ) -> None:
        """Test handling multiple experiments"""
        # Create tracker for experiment 1
        config1 = mlflow_config.model_copy(deep=True)
        config1.experiment_name = "experiment-1"
        tracker1 = MLflowTracker(config=config1)

        # Create tracker for experiment 2
        config2 = mlflow_config.model_copy(deep=True)
        config2.experiment_name = "experiment-2"
        tracker2 = MLflowTracker(config=config2)

        # Create run in experiment 1
        run_id1 = await tracker1.start_run(request=optimization_request)
        await tracker1.end_run()

        # Create run in experiment 2 (after ending first run)
        run_id2 = await tracker2.start_run(request=optimization_request)
        await tracker2.end_run()

        # Verify runs in different experiments
        run1 = mlflow.get_run(run_id1)
        run2 = mlflow.get_run(run_id2)

        assert run1.info.experiment_id != run2.info.experiment_id
