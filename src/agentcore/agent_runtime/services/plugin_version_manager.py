"""Plugin version management and compatibility checking."""

from __future__ import annotations

import structlog

from ..models.plugin import PluginDependency, PluginMetadata, PluginVersionConflictError

logger = structlog.get_logger()


class PluginVersionManager:
    """Service for managing plugin versions and checking compatibility."""

    # Current runtime version (should be loaded from config)
    RUNTIME_VERSION = "0.1.0"

    def __init__(self, runtime_version: str | None = None) -> None:
        """
        Initialize version manager.

        Args:
            runtime_version: Runtime version (uses default if None)
        """
        self._runtime_version = runtime_version or self.RUNTIME_VERSION

        logger.info(
            "plugin_version_manager_initialized",
            runtime_version=self._runtime_version,
        )

    def check_runtime_compatibility(self, metadata: PluginMetadata) -> tuple[bool, str]:
        """
        Check if plugin is compatible with current runtime version.

        Args:
            metadata: Plugin metadata

        Returns:
            Tuple of (is_compatible, reason)
        """
        min_version = metadata.min_runtime_version
        max_version = metadata.max_runtime_version

        # Check minimum version
        if not self._satisfies_constraint(self._runtime_version, f">={min_version}"):
            return (
                False,
                f"Runtime version {self._runtime_version} is below minimum required {min_version}",
            )

        # Check maximum version (if not wildcard)
        if max_version != "*" and not self._satisfies_constraint(
            self._runtime_version, f"<={max_version}"
        ):
            return (
                False,
                f"Runtime version {self._runtime_version} exceeds maximum compatible {max_version}",
            )

        return (True, "Compatible")

    def check_dependency_compatibility(
        self,
        dependency: PluginDependency,
        available_version: str,
    ) -> tuple[bool, str]:
        """
        Check if available plugin version satisfies dependency constraint.

        Args:
            dependency: Plugin dependency specification
            available_version: Available plugin version

        Returns:
            Tuple of (is_compatible, reason)
        """
        constraint = dependency.version_constraint

        if self._satisfies_constraint(available_version, constraint):
            return (True, "Compatible")

        return (
            False,
            f"Version {available_version} does not satisfy constraint {constraint}",
        )

    def resolve_dependencies(
        self,
        metadata: PluginMetadata,
        available_plugins: dict[str, str],
    ) -> tuple[bool, list[str]]:
        """
        Resolve all plugin dependencies.

        Args:
            metadata: Plugin metadata
            available_plugins: Dict of plugin_id -> version for available plugins

        Returns:
            Tuple of (all_satisfied, missing_dependencies)
        """
        missing: list[str] = []

        for dependency in metadata.dependencies:
            # Check if dependency is available
            if dependency.plugin_id not in available_plugins:
                if not dependency.optional:
                    missing.append(f"{dependency.plugin_id} (required, not found)")
                continue

            # Check version compatibility
            available_version = available_plugins[dependency.plugin_id]
            is_compatible, reason = self.check_dependency_compatibility(
                dependency=dependency,
                available_version=available_version,
            )

            if not is_compatible and not dependency.optional:
                missing.append(f"{dependency.plugin_id} ({reason})")

        return (len(missing) == 0, missing)

    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two semantic versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            -1 if version1 < version2
            0 if version1 == version2
            1 if version1 > version2
        """
        v1_parts = self._parse_version(version1)
        v2_parts = self._parse_version(version2)

        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1

        # If all compared parts are equal, check length
        if len(v1_parts) < len(v2_parts):
            return -1
        elif len(v1_parts) > len(v2_parts):
            return 1

        return 0

    def get_latest_version(self, versions: list[str]) -> str:
        """
        Get latest version from list of versions.

        Args:
            versions: List of version strings

        Returns:
            Latest version

        Raises:
            ValueError: If versions list is empty
        """
        if not versions:
            raise ValueError("Versions list cannot be empty")

        return max(versions, key=lambda v: self._parse_version(v))

    def is_backward_compatible(self, old_version: str, new_version: str) -> bool:
        """
        Check if new version is backward compatible with old version (same major version).

        Args:
            old_version: Old version
            new_version: New version

        Returns:
            True if backward compatible
        """
        old_parts = self._parse_version(old_version)
        new_parts = self._parse_version(new_version)

        # Backward compatible if major version is same and minor/patch increased
        if old_parts[0] == new_parts[0]:
            return self.compare_versions(new_version, old_version) >= 0

        return False

    def _satisfies_constraint(self, version: str, constraint: str) -> bool:
        """
        Check if version satisfies constraint.

        Supports: *, >=, <=, >, <, ==, ^, ~

        Args:
            version: Version to check
            constraint: Version constraint

        Returns:
            True if constraint satisfied
        """
        # Wildcard always matches
        if constraint == "*" or constraint == "":
            return True

        # Parse constraint operator
        if constraint.startswith(">="):
            target = constraint[2:].strip()
            return self.compare_versions(version, target) >= 0
        elif constraint.startswith("<="):
            target = constraint[2:].strip()
            return self.compare_versions(version, target) <= 0
        elif constraint.startswith(">"):
            target = constraint[1:].strip()
            return self.compare_versions(version, target) > 0
        elif constraint.startswith("<"):
            target = constraint[1:].strip()
            return self.compare_versions(version, target) < 0
        elif constraint.startswith("=="):
            target = constraint[2:].strip()
            return self.compare_versions(version, target) == 0
        elif constraint.startswith("^"):
            # Caret: compatible with version (same major version)
            target = constraint[1:].strip()
            return self.is_backward_compatible(target, version)
        elif constraint.startswith("~"):
            # Tilde: approximately equivalent (same major and minor)
            target = constraint[1:].strip()
            v_parts = self._parse_version(version)
            t_parts = self._parse_version(target)
            return (
                v_parts[0] == t_parts[0]
                and v_parts[1] == t_parts[1]
                and v_parts[2] >= t_parts[2]
            )
        else:
            # No operator, assume exact match
            return self.compare_versions(version, constraint) == 0

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """
        Parse semantic version string into tuple of integers.

        Args:
            version: Version string (e.g., "1.2.3")

        Returns:
            Tuple of (major, minor, patch)

        Raises:
            ValueError: If version format is invalid
        """
        # Remove 'v' prefix if present
        version = version.lstrip("v")

        # Split and parse
        parts = version.split(".")

        if len(parts) < 2:
            raise ValueError(f"Invalid version format: {version}")

        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {version}") from e
