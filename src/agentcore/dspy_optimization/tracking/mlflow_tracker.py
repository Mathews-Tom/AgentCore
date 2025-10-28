"""
MLflow experiment tracking integration

Provides experiment logging, model versioning, artifact management,
and performance metrics tracking for DSPy optimization.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
from mlflow.entities import Run
from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    PerformanceMetrics,
)


class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking"""

    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    experiment_name: str = Field(
        default="dspy-optimization",
        description="Name of MLflow experiment",
    )
    artifact_location: str | None = Field(
        default=None,
        description="Base location for artifacts (S3, local path, etc.)",
    )
    registry_uri: str | None = Field(
        default=None,
        description="Model registry URI (defaults to tracking_uri)",
    )


class MLflowTracker:
    """
    MLflow tracking integration for optimization experiments

    Provides comprehensive experiment logging, model artifact management,
    and performance metrics tracking for DSPy optimization workflows.
    """

    def __init__(self, config: MLflowConfig | None = None) -> None:
        """
        Initialize MLflow tracker

        Args:
            config: MLflow configuration (uses defaults if not provided)
        """
        self.config = config or MLflowConfig()

        # Configure MLflow
        mlflow.set_tracking_uri(self.config.tracking_uri)

        if self.config.registry_uri:
            mlflow.set_registry_uri(self.config.registry_uri)

        # Create or get experiment
        self.experiment = self._setup_experiment()

    def _setup_experiment(self) -> mlflow.entities.Experiment:
        """
        Set up MLflow experiment

        Returns:
            MLflow experiment instance
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                )
                experiment = mlflow.get_experiment(experiment_id)
            return experiment
        except Exception as e:
            raise RuntimeError(
                f"Failed to set up MLflow experiment: {e}"
            ) from e

    async def start_run(
        self,
        request: OptimizationRequest,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start MLflow tracking run

        Args:
            request: Optimization request
            run_name: Optional custom run name
            tags: Additional tags for the run

        Returns:
            MLflow run ID
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_name = f"opt_{request.target.type}_{request.target.id}_{timestamp}"

        # Prepare tags
        run_tags = {
            "target_type": request.target.type.value,
            "target_id": request.target.id,
            "target_scope": request.target.scope.value,
            "algorithms": ",".join(request.algorithms),
        }
        if tags:
            run_tags.update(tags)

        # Start run
        run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name,
            tags=run_tags,
        )

        # Log request parameters
        self._log_request_params(request)

        return run.info.run_id

    def _log_request_params(self, request: OptimizationRequest) -> None:
        """
        Log optimization request parameters

        Args:
            request: Optimization request
        """
        # Log constraints
        mlflow.log_param("max_optimization_time", request.constraints.max_optimization_time)
        mlflow.log_param("min_improvement_threshold", request.constraints.min_improvement_threshold)
        mlflow.log_param("max_resource_usage", request.constraints.max_resource_usage)

        # Log objectives
        for i, objective in enumerate(request.objectives):
            mlflow.log_param(f"objective_{i}_metric", objective.metric.value)
            mlflow.log_param(f"objective_{i}_target", objective.target_value)
            mlflow.log_param(f"objective_{i}_weight", objective.weight)

    async def log_baseline_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        Log baseline performance metrics

        Args:
            metrics: Baseline performance metrics
        """
        mlflow.log_metrics(
            {
                "baseline_success_rate": metrics.success_rate,
                "baseline_avg_cost": metrics.avg_cost_per_task,
                "baseline_avg_latency_ms": float(metrics.avg_latency_ms),
                "baseline_quality_score": metrics.quality_score,
            }
        )

    async def log_optimized_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        Log optimized performance metrics

        Args:
            metrics: Optimized performance metrics
        """
        mlflow.log_metrics(
            {
                "optimized_success_rate": metrics.success_rate,
                "optimized_avg_cost": metrics.avg_cost_per_task,
                "optimized_avg_latency_ms": float(metrics.avg_latency_ms),
                "optimized_quality_score": metrics.quality_score,
            }
        )

    async def log_improvement_metrics(self, result: OptimizationResult) -> None:
        """
        Log improvement metrics

        Args:
            result: Optimization result with improvements
        """
        if result.baseline_performance and result.optimized_performance:
            # Calculate improvements
            success_rate_improvement = (
                result.optimized_performance.success_rate
                - result.baseline_performance.success_rate
            )
            cost_reduction = (
                result.baseline_performance.avg_cost_per_task
                - result.optimized_performance.avg_cost_per_task
            )
            latency_reduction = (
                result.baseline_performance.avg_latency_ms
                - result.optimized_performance.avg_latency_ms
            )
            quality_improvement = (
                result.optimized_performance.quality_score
                - result.baseline_performance.quality_score
            )

            mlflow.log_metrics(
                {
                    "success_rate_improvement": success_rate_improvement,
                    "cost_reduction": cost_reduction,
                    "latency_reduction_ms": float(latency_reduction),
                    "quality_improvement": quality_improvement,
                    "improvement_percentage": result.improvement_percentage,
                    "statistical_significance": result.statistical_significance,
                }
            )

    async def log_optimization_details(
        self,
        algorithm: str,
        iterations: int,
        key_improvements: list[str],
        parameters: dict[str, Any],
    ) -> None:
        """
        Log optimization algorithm details

        Args:
            algorithm: Algorithm name used
            iterations: Number of optimization iterations
            key_improvements: List of key improvements made
            parameters: Algorithm-specific parameters
        """
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_metric("iterations", float(iterations))

        # Log key improvements as text artifact
        if key_improvements:
            improvements_text = "\n".join(f"- {imp}" for imp in key_improvements)
            mlflow.log_text(improvements_text, "key_improvements.txt")

        # Log parameters as JSON artifact
        if parameters:
            mlflow.log_dict(parameters, "algorithm_parameters.json")

    async def log_model_artifact(
        self,
        model: Any,
        artifact_name: str = "optimized_model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log model artifact using cloudpickle

        Args:
            model: Model object to save
            artifact_name: Name for the artifact
            metadata: Optional metadata to save with model
        """
        # Create temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save model using cloudpickle
            model_path = tmpdir_path / f"{artifact_name}.pkl"
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)

            # Save metadata if provided
            if metadata:
                metadata_path = tmpdir_path / f"{artifact_name}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            # Log artifacts
            mlflow.log_artifacts(tmpdir, artifact_path="models")

    async def load_model_artifact(
        self,
        run_id: str,
        artifact_name: str = "optimized_model",
    ) -> Any:
        """
        Load model artifact from MLflow

        Args:
            run_id: MLflow run ID
            artifact_name: Name of the artifact to load

        Returns:
            Loaded model object
        """
        # Download artifact
        artifact_path = f"models/{artifact_name}.pkl"
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )

        # Load model using cloudpickle
        with open(local_path, "rb") as f:
            model = cloudpickle.load(f)

        return model

    async def log_training_data(
        self,
        training_data: list[dict[str, Any]],
        artifact_name: str = "training_data",
    ) -> None:
        """
        Log training data as artifact

        Args:
            training_data: Training data to log
            artifact_name: Name for the artifact
        """
        mlflow.log_dict(training_data, f"{artifact_name}.json")

    async def end_run(
        self,
        status: str = "FINISHED",
    ) -> None:
        """
        End MLflow tracking run

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)

    async def log_result(self, result: OptimizationResult) -> None:
        """
        Log complete optimization result

        Args:
            result: Optimization result to log
        """
        # Log status
        mlflow.set_tag("status", result.status.value)

        # Log metrics if available
        if result.baseline_performance:
            await self.log_baseline_metrics(result.baseline_performance)

        if result.optimized_performance:
            await self.log_optimized_metrics(result.optimized_performance)
            await self.log_improvement_metrics(result)

        # Log optimization details
        if result.optimization_details:
            await self.log_optimization_details(
                algorithm=result.optimization_details.algorithm_used,
                iterations=result.optimization_details.iterations,
                key_improvements=result.optimization_details.key_improvements,
                parameters=result.optimization_details.parameters,
            )

        # Log timestamps
        mlflow.log_param("created_at", result.created_at.isoformat())
        if result.completed_at:
            mlflow.log_param("completed_at", result.completed_at.isoformat())
            duration = (result.completed_at - result.created_at).total_seconds()
            mlflow.log_metric("duration_seconds", duration)

        # Log error if failed
        if result.error_message:
            mlflow.set_tag("error", result.error_message)

    async def get_run(self, run_id: str) -> Run:
        """
        Get MLflow run by ID

        Args:
            run_id: Run ID to retrieve

        Returns:
            MLflow Run object
        """
        return mlflow.get_run(run_id)

    async def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
    ) -> list[Run]:
        """
        Search for runs in the experiment

        Args:
            filter_string: MLflow filter string (e.g., "metrics.improvement > 0.2")
            max_results: Maximum number of results to return

        Returns:
            List of matching runs
        """
        return mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            output_format="list",
        )

    async def get_best_run(
        self,
        metric: str = "improvement_percentage",
        order_by: str = "DESC",
    ) -> Run | None:
        """
        Get best run based on metric

        Args:
            metric: Metric to optimize
            order_by: Sort order (DESC or ASC)

        Returns:
            Best run or None if no runs exist
        """
        runs = await self.search_runs(
            filter_string="",
            max_results=1,
        )

        if not runs:
            return None

        # Sort by metric
        runs.sort(
            key=lambda r: r.data.metrics.get(metric, 0.0),
            reverse=(order_by == "DESC"),
        )

        return runs[0] if runs else None

    def cleanup_experiment(self) -> None:
        """
        Delete the experiment and all its runs

        Warning: This is irreversible!
        """
        mlflow.delete_experiment(self.experiment.experiment_id)
