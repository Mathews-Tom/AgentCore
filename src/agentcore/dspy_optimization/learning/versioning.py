"""
Model version management

Manages model versions, deployment strategies, and rollback capabilities
for safe model updates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import OptimizationTarget, PerformanceMetrics


class DeploymentStrategy(str, Enum):
    """Deployment strategy for model updates"""

    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    AB_TEST = "ab_test"
    BLUE_GREEN = "blue_green"


class ModelStatus(str, Enum):
    """Status of model version"""

    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


class ModelVersion(BaseModel):
    """Model version metadata"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    version_number: int
    target: OptimizationTarget
    status: ModelStatus
    performance_metrics: PerformanceMetrics | None = None
    deployment_strategy: DeploymentStrategy
    traffic_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Traffic percentage for gradual deployment",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deployed_at: datetime | None = None
    deprecated_at: datetime | None = None
    parent_version_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_deployed(self) -> bool:
        """Check if version is deployed"""
        return self.status == ModelStatus.DEPLOYED

    def is_active(self) -> bool:
        """Check if version is actively serving traffic"""
        return self.status == ModelStatus.DEPLOYED and self.traffic_percentage > 0


class ModelVersionManager:
    """
    Manages model versions and deployments

    Provides version control, deployment strategies, and rollback
    capabilities for optimization models.

    Key features:
    - Version tracking and metadata
    - Multiple deployment strategies
    - Traffic management for gradual rollout
    - Rollback capabilities
    """

    def __init__(self) -> None:
        """Initialize version manager"""
        self._versions: dict[str, list[ModelVersion]] = {}
        self._active_versions: dict[str, str] = {}

    async def create_version(
        self,
        target: OptimizationTarget,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.AB_TEST,
        parent_version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """
        Create new model version

        Args:
            target: Optimization target
            deployment_strategy: Deployment strategy
            parent_version_id: Optional parent version ID
            metadata: Optional version metadata

        Returns:
            Created model version
        """
        target_key = self._get_target_key(target)

        # Get version number
        existing_versions = self._versions.get(target_key, [])
        version_number = len(existing_versions) + 1

        # Create version
        version = ModelVersion(
            version_number=version_number,
            target=target,
            status=ModelStatus.TRAINING,
            deployment_strategy=deployment_strategy,
            parent_version_id=parent_version_id,
            metadata=metadata or {},
        )

        # Store version
        if target_key not in self._versions:
            self._versions[target_key] = []

        self._versions[target_key].append(version)

        return version

    async def update_version_status(
        self,
        version_id: str,
        status: ModelStatus,
        performance_metrics: PerformanceMetrics | None = None,
    ) -> ModelVersion:
        """
        Update version status

        Args:
            version_id: Version ID
            status: New status
            performance_metrics: Optional performance metrics

        Returns:
            Updated version

        Raises:
            ValueError: If version not found
        """
        version = await self._find_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        version.status = status

        if performance_metrics:
            version.performance_metrics = performance_metrics

        if status == ModelStatus.DEPLOYED:
            version.deployed_at = datetime.now(UTC)
        elif status in (ModelStatus.DEPRECATED, ModelStatus.ROLLED_BACK):
            version.deprecated_at = datetime.now(UTC)

        return version

    async def deploy_version(
        self,
        version_id: str,
        traffic_percentage: float = 1.0,
    ) -> ModelVersion:
        """
        Deploy model version

        Args:
            version_id: Version ID
            traffic_percentage: Traffic percentage (0.0-1.0)

        Returns:
            Deployed version

        Raises:
            ValueError: If version not found or invalid traffic percentage
        """
        if not 0.0 <= traffic_percentage <= 1.0:
            raise ValueError(f"Invalid traffic percentage: {traffic_percentage}")

        version = await self._find_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        # Update version
        version.status = ModelStatus.DEPLOYED
        version.traffic_percentage = traffic_percentage
        version.deployed_at = datetime.now(UTC)

        # Update active version
        target_key = self._get_target_key(version.target)
        self._active_versions[target_key] = version_id

        return version

    async def gradual_rollout(
        self,
        version_id: str,
        target_percentage: float,
        step_size: float = 0.1,
    ) -> ModelVersion:
        """
        Gradually increase traffic to version

        Args:
            version_id: Version ID
            target_percentage: Target traffic percentage
            step_size: Increment step size

        Returns:
            Updated version

        Raises:
            ValueError: If version not found or invalid parameters
        """
        if not 0.0 <= target_percentage <= 1.0:
            raise ValueError(f"Invalid target percentage: {target_percentage}")

        if not 0.0 < step_size <= 1.0:
            raise ValueError(f"Invalid step size: {step_size}")

        version = await self._find_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        # Calculate new percentage
        new_percentage = min(version.traffic_percentage + step_size, target_percentage)

        # Update traffic
        version.traffic_percentage = new_percentage

        return version

    async def rollback_version(
        self,
        version_id: str,
        target_version_id: str | None = None,
    ) -> ModelVersion:
        """
        Rollback to previous version

        Args:
            version_id: Version to rollback
            target_version_id: Optional target version (defaults to parent)

        Returns:
            Rolled back version

        Raises:
            ValueError: If versions not found
        """
        version = await self._find_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        # Mark current version as rolled back
        version.status = ModelStatus.ROLLED_BACK
        version.traffic_percentage = 0.0
        version.deprecated_at = datetime.now(UTC)

        # Get target version
        rollback_to_id = target_version_id or version.parent_version_id
        if not rollback_to_id:
            raise ValueError("No target version specified for rollback")

        rollback_version = await self._find_version(rollback_to_id)
        if not rollback_version:
            raise ValueError(f"Rollback target version not found: {rollback_to_id}")

        # Deploy rollback version
        await self.deploy_version(rollback_to_id, traffic_percentage=1.0)

        return version

    async def get_version(
        self,
        version_id: str,
    ) -> ModelVersion | None:
        """
        Get version by ID

        Args:
            version_id: Version ID

        Returns:
            Model version or None
        """
        return await self._find_version(version_id)

    async def get_active_version(
        self,
        target: OptimizationTarget,
    ) -> ModelVersion | None:
        """
        Get active version for target

        Args:
            target: Optimization target

        Returns:
            Active model version or None
        """
        target_key = self._get_target_key(target)
        version_id = self._active_versions.get(target_key)

        if not version_id:
            return None

        return await self._find_version(version_id)

    async def list_versions(
        self,
        target: OptimizationTarget,
        status: ModelStatus | None = None,
    ) -> list[ModelVersion]:
        """
        List versions for target

        Args:
            target: Optimization target
            status: Optional status filter

        Returns:
            List of model versions
        """
        target_key = self._get_target_key(target)
        versions = self._versions.get(target_key, [])

        if status:
            versions = [v for v in versions if v.status == status]

        return versions

    async def get_version_history(
        self,
        target: OptimizationTarget,
    ) -> list[ModelVersion]:
        """
        Get version history for target

        Args:
            target: Optimization target

        Returns:
            Version history sorted by version number
        """
        versions = await self.list_versions(target)
        return sorted(versions, key=lambda v: v.version_number, reverse=True)

    async def cleanup_old_versions(
        self,
        target: OptimizationTarget,
        keep_count: int = 10,
    ) -> int:
        """
        Clean up old deprecated versions

        Args:
            target: Optimization target
            keep_count: Number of versions to keep

        Returns:
            Number of versions cleaned up
        """
        target_key = self._get_target_key(target)
        versions = self._versions.get(target_key, [])

        # Filter deprecated versions
        deprecated = [
            v
            for v in versions
            if v.status in (ModelStatus.DEPRECATED, ModelStatus.ROLLED_BACK)
        ]

        # Sort by version number
        deprecated.sort(key=lambda v: v.version_number, reverse=True)

        # Keep most recent
        to_remove = deprecated[keep_count:]

        # Remove from storage
        if to_remove:
            remaining = [v for v in versions if v not in to_remove]
            self._versions[target_key] = remaining

        return len(to_remove)

    async def _find_version(
        self,
        version_id: str,
    ) -> ModelVersion | None:
        """
        Find version by ID

        Args:
            version_id: Version ID

        Returns:
            Model version or None
        """
        for versions in self._versions.values():
            for version in versions:
                if version.id == version_id:
                    return version

        return None

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """
        Get target storage key

        Args:
            target: Optimization target

        Returns:
            Target key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
