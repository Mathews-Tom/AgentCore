"""
Continuous learning pipeline

Integrates online learning, drift detection, versioning, and retraining
into a complete continuous learning system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.learning.drift import (
    DriftConfig,
    DriftDetector,
    DriftResult,
    DriftStatus,
)
from agentcore.dspy_optimization.learning.online import (
    LearningUpdate,
    OnlineLearner,
    OnlineLearningConfig,
)
from agentcore.dspy_optimization.learning.retraining import (
    RetrainingConfig,
    RetrainingJob,
    RetrainingManager,
    RetrainingStatus,
    RetrainingTrigger,
)
from agentcore.dspy_optimization.learning.versioning import (
    DeploymentStrategy,
    ModelStatus,
    ModelVersion,
    ModelVersionManager,
)
from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    PerformanceMetrics,
)


class PipelineStatus(str, Enum):
    """Status of continuous learning pipeline"""

    IDLE = "idle"
    LEARNING = "learning"
    RETRAINING = "retraining"
    DEPLOYING = "deploying"
    ERROR = "error"


class PipelineConfig(BaseModel):
    """Configuration for continuous learning pipeline"""

    online_learning: OnlineLearningConfig = Field(
        default_factory=OnlineLearningConfig,
    )
    drift_detection: DriftConfig = Field(
        default_factory=DriftConfig,
    )
    retraining: RetrainingConfig = Field(
        default_factory=RetrainingConfig,
    )
    auto_deploy_on_improvement: bool = Field(
        default=True,
        description="Automatically deploy improvements",
    )
    deployment_strategy: DeploymentStrategy = Field(
        default=DeploymentStrategy.AB_TEST,
        description="Default deployment strategy",
    )


class PipelineMetrics(BaseModel):
    """Pipeline performance metrics"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    total_updates: int = 0
    total_retrainings: int = 0
    total_deployments: int = 0
    drift_detections: int = 0
    current_performance: PerformanceMetrics | None = None
    baseline_performance: PerformanceMetrics | None = None
    cumulative_improvement: float = 0.0
    last_update: datetime | None = None
    last_retraining: datetime | None = None
    last_deployment: datetime | None = None


class ContinuousLearningPipeline:
    """
    Continuous learning pipeline

    Integrates all continuous learning components into a unified
    pipeline for self-improving AI systems.

    Key features:
    - Automatic online learning
    - Drift detection and monitoring
    - Automatic retraining triggers
    - Safe model deployment
    - Performance tracking
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """
        Initialize continuous learning pipeline

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.online_learner = OnlineLearner(self.config.online_learning)
        self.drift_detector = DriftDetector(self.config.drift_detection)
        self.retraining_manager = RetrainingManager(self.config.retraining)
        self.version_manager = ModelVersionManager()

        # Pipeline state
        self._status: dict[str, PipelineStatus] = {}
        self._metrics: dict[str, PipelineMetrics] = {}

    async def process_sample(
        self,
        target: OptimizationTarget,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process new training sample through pipeline

        Args:
            target: Optimization target
            sample: Training sample with performance metrics

        Returns:
            Processing result with actions taken
        """
        target_key = self._get_target_key(target)
        result: dict[str, Any] = {
            "target": target,
            "actions": [],
            "status": self._get_status(target_key),
        }

        try:
            # Update status
            self._set_status(target_key, PipelineStatus.LEARNING)

            # Extract metrics from sample
            metrics = PerformanceMetrics(
                success_rate=sample.get("success_rate", 0.0),
                avg_cost_per_task=sample.get("avg_cost_per_task", 0.0),
                avg_latency_ms=sample.get("avg_latency_ms", 0),
                quality_score=sample.get("quality_score", 0.0),
            )

            # Record metrics for drift detection
            await self.drift_detector.record_metrics(target, metrics)

            # Add to online learner
            update = await self.online_learner.add_training_sample(target, sample)

            if update:
                result["actions"].append({
                    "type": "online_update",
                    "improvement": update.improvement,
                })
                await self._update_metrics(target, "update")

            # Check for drift
            drift = await self.drift_detector.check_drift(target)

            if drift:
                result["actions"].append({
                    "type": "drift_detected",
                    "status": drift.status.value,
                    "degradation": drift.degradation_percentage,
                })
                await self._update_metrics(target, "drift")

                # Check if retraining should be triggered
                trigger = await self._handle_drift(target, drift)

                if trigger:
                    result["actions"].append({
                        "type": "retraining_triggered",
                        "condition": trigger.condition.value,
                    })

            # Check for other retraining triggers
            sample_count = len(self.online_learner._training_data.get(target_key, []))
            trigger = await self.retraining_manager.check_triggers(
                target,
                drift_detected=drift is not None,
                performance_degradation=drift.degradation_percentage if drift else 0.0,
                sample_count=sample_count,
            )

            if trigger and "retraining_triggered" not in [a["type"] for a in result["actions"]]:
                result["actions"].append({
                    "type": "retraining_triggered",
                    "condition": trigger.condition.value,
                })

                # Start retraining job
                job = await self._start_retraining(target, trigger, sample_count)
                result["actions"].append({
                    "type": "retraining_started",
                    "job_id": job.id,
                })

            # Return to idle
            self._set_status(target_key, PipelineStatus.IDLE)

        except Exception as e:
            self._set_status(target_key, PipelineStatus.ERROR)
            result["error"] = str(e)

        return result

    async def deploy_model(
        self,
        target: OptimizationTarget,
        version_id: str,
        strategy: DeploymentStrategy | None = None,
    ) -> ModelVersion:
        """
        Deploy model version

        Args:
            target: Optimization target
            version_id: Version to deploy
            strategy: Deployment strategy

        Returns:
            Deployed model version
        """
        target_key = self._get_target_key(target)

        try:
            self._set_status(target_key, PipelineStatus.DEPLOYING)

            # Get deployment strategy
            deploy_strategy = strategy or self.config.deployment_strategy

            # Deploy based on strategy
            if deploy_strategy == DeploymentStrategy.IMMEDIATE:
                version = await self.version_manager.deploy_version(version_id, 1.0)

            elif deploy_strategy == DeploymentStrategy.GRADUAL:
                # Start with 10% traffic
                version = await self.version_manager.deploy_version(version_id, 0.1)

            elif deploy_strategy == DeploymentStrategy.AB_TEST:
                # Start with 50% traffic for A/B testing
                version = await self.version_manager.deploy_version(version_id, 0.5)

            elif deploy_strategy == DeploymentStrategy.BLUE_GREEN:
                # Deploy to staging first (0% production traffic)
                version = await self.version_manager.deploy_version(version_id, 0.0)

            else:
                version = await self.version_manager.deploy_version(version_id, 1.0)

            # Update metrics
            await self._update_metrics(target, "deployment")

            self._set_status(target_key, PipelineStatus.IDLE)

            return version

        except Exception as e:
            self._set_status(target_key, PipelineStatus.ERROR)
            raise

    async def rollback_model(
        self,
        target: OptimizationTarget,
        version_id: str,
    ) -> ModelVersion:
        """
        Rollback model version

        Args:
            target: Optimization target
            version_id: Version to rollback

        Returns:
            Rolled back version
        """
        version = await self.version_manager.rollback_version(version_id)

        # Reset baseline after rollback
        if version.performance_metrics:
            await self.drift_detector.reset_baseline(
                target,
                version.performance_metrics,
            )

        return version

    async def get_pipeline_metrics(
        self,
        target: OptimizationTarget,
    ) -> PipelineMetrics:
        """
        Get pipeline metrics for target

        Args:
            target: Optimization target

        Returns:
            Pipeline metrics
        """
        target_key = self._get_target_key(target)

        if target_key not in self._metrics:
            self._metrics[target_key] = PipelineMetrics(target=target)

        return self._metrics[target_key]

    async def get_learning_history(
        self,
        target: OptimizationTarget,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Get learning history for target

        Args:
            target: Optimization target
            limit: Result limit

        Returns:
            Learning history
        """
        return {
            "updates": await self.online_learner.get_update_history(target, limit),
            "retraining_jobs": await self.retraining_manager.list_jobs(target),
            "versions": await self.version_manager.list_versions(target),
            "triggers": await self.retraining_manager.get_trigger_history(target, limit),
        }

    async def _handle_drift(
        self,
        target: OptimizationTarget,
        drift: DriftResult,
    ) -> RetrainingTrigger | None:
        """
        Handle drift detection

        Args:
            target: Optimization target
            drift: Drift result

        Returns:
            Retraining trigger if created
        """
        # Trigger retraining for critical drift
        if drift.status == DriftStatus.CRITICAL_DRIFT:
            return await self.retraining_manager.create_manual_trigger(
                target,
                {
                    "reason": "critical_drift",
                    "degradation": drift.degradation_percentage,
                },
            )

        return None

    async def _start_retraining(
        self,
        target: OptimizationTarget,
        trigger: RetrainingTrigger,
        sample_count: int,
    ) -> RetrainingJob:
        """
        Start retraining job

        Args:
            target: Optimization target
            trigger: Retraining trigger
            sample_count: Sample count

        Returns:
            Started retraining job
        """
        target_key = self._get_target_key(target)

        self._set_status(target_key, PipelineStatus.RETRAINING)

        # Create job
        job = await self.retraining_manager.start_retraining(trigger, sample_count)

        # Update metrics
        await self._update_metrics(target, "retraining")

        return job

    async def _update_metrics(
        self,
        target: OptimizationTarget,
        event_type: str,
    ) -> None:
        """
        Update pipeline metrics

        Args:
            target: Optimization target
            event_type: Type of event
        """
        target_key = self._get_target_key(target)

        if target_key not in self._metrics:
            self._metrics[target_key] = PipelineMetrics(target=target)

        metrics = self._metrics[target_key]

        if event_type == "update":
            metrics.total_updates += 1
            metrics.last_update = datetime.utcnow()

        elif event_type == "drift":
            metrics.drift_detections += 1

        elif event_type == "retraining":
            metrics.total_retrainings += 1
            metrics.last_retraining = datetime.utcnow()

        elif event_type == "deployment":
            metrics.total_deployments += 1
            metrics.last_deployment = datetime.utcnow()

    def _get_status(self, target_key: str) -> PipelineStatus:
        """Get pipeline status"""
        return self._status.get(target_key, PipelineStatus.IDLE)

    def _set_status(self, target_key: str, status: PipelineStatus) -> None:
        """Set pipeline status"""
        self._status[target_key] = status

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """Get target storage key"""
        return f"{target.type.value}:{target.id}:{target.scope.value}"
