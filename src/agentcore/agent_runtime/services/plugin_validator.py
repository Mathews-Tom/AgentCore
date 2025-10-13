"""Plugin security validator for validating plugins before loading."""

from __future__ import annotations

import ast
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from ..models.plugin import (
    PluginMetadata,
    PluginValidationResult,
)

logger = structlog.get_logger()


class PluginValidator:
    """Service for validating plugin security and compatibility."""

    # Dangerous imports that are not allowed
    DANGEROUS_IMPORTS = {
        "os",
        "subprocess",
        "sys",
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "socket",
        "urllib",
        "requests",
        "http",
    }

    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "raw_input",
    }

    # Safe patterns (lower security risk)
    SAFE_PATTERNS = [
        r"import\s+(json|datetime|math|random|typing|enum|dataclasses|pydantic)",
        r"from\s+(typing|enum|dataclasses|pydantic)\s+import",
    ]

    def __init__(
        self,
        max_file_size_mb: int = 10,
        enable_code_scanning: bool = True,
        enable_checksum_validation: bool = True,
    ) -> None:
        """
        Initialize plugin validator.

        Args:
            max_file_size_mb: Maximum plugin file size in MB
            enable_code_scanning: Enable static code analysis
            enable_checksum_validation: Enable checksum validation
        """
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._enable_code_scanning = enable_code_scanning
        self._enable_checksum_validation = enable_checksum_validation

        logger.info(
            "plugin_validator_initialized",
            max_file_size_mb=max_file_size_mb,
            code_scanning=enable_code_scanning,
            checksum_validation=enable_checksum_validation,
        )

    async def validate_plugin(
        self,
        plugin_path: Path,
        metadata: PluginMetadata,
        expected_checksum: str | None = None,
    ) -> PluginValidationResult:
        """
        Validate plugin security and compatibility.

        Args:
            plugin_path: Path to plugin directory
            metadata: Plugin metadata
            expected_checksum: Expected SHA-256 checksum (if known)

        Returns:
            Validation result with errors/warnings
        """
        logger.info(
            "validating_plugin",
            plugin_id=metadata.plugin_id,
            version=metadata.version,
        )

        errors: list[str] = []
        warnings: list[str] = []
        security_score = 100.0

        # Validate plugin structure
        structure_errors = self._validate_structure(plugin_path)
        errors.extend(structure_errors)
        if structure_errors:
            security_score -= 20.0

        # Validate metadata
        metadata_errors = self._validate_metadata(metadata)
        errors.extend(metadata_errors)
        if metadata_errors:
            security_score -= 15.0

        # Validate file sizes
        size_errors, size_warnings = await self._validate_file_sizes(plugin_path)
        errors.extend(size_errors)
        warnings.extend(size_warnings)
        if size_errors:
            security_score -= 10.0

        # Validate checksums if enabled
        if self._enable_checksum_validation and expected_checksum:
            checksum_errors = await self._validate_checksum(
                plugin_path, expected_checksum
            )
            errors.extend(checksum_errors)
            if checksum_errors:
                security_score -= 30.0

        # Scan code if enabled
        if self._enable_code_scanning:
            code_errors, code_warnings = await self._scan_code(
                plugin_path, metadata
            )
            errors.extend(code_errors)
            warnings.extend(code_warnings)
            if code_errors:
                security_score -= 25.0
            if code_warnings:
                security_score -= len(code_warnings) * 2.0

        # Validate permissions
        permission_warnings = await self._validate_permissions(metadata)
        warnings.extend(permission_warnings)
        if permission_warnings:
            security_score -= len(permission_warnings) * 3.0

        # Ensure score is within bounds
        security_score = max(0.0, min(100.0, security_score))

        # Determine risk level
        if security_score >= 80.0:
            risk_level = "low"
        elif security_score >= 60.0:
            risk_level = "medium"
        elif security_score >= 40.0:
            risk_level = "high"
        else:
            risk_level = "critical"

        result = PluginValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            security_score=security_score,
            risk_level=risk_level,
            scanned_at=datetime.utcnow(),
        )

        logger.info(
            "plugin_validation_complete",
            plugin_id=metadata.plugin_id,
            valid=result.valid,
            security_score=security_score,
            risk_level=risk_level,
            errors_count=len(errors),
            warnings_count=len(warnings),
        )

        return result

    def _validate_structure(self, plugin_path: Path) -> list[str]:
        """Validate plugin directory structure."""
        errors: list[str] = []

        # Check plugin.json exists
        if not (plugin_path / "plugin.json").exists():
            errors.append("Missing required plugin.json manifest")

        # Check entry point exists (will be validated by loader)
        # This is just a basic check

        # Check for suspicious files
        suspicious_patterns = [
            "*.exe",
            "*.dll",
            "*.so",
            "*.dylib",
            "*.sh",
            "*.bat",
            "*.cmd",
        ]

        for pattern in suspicious_patterns:
            suspicious_files = list(plugin_path.glob(f"**/{pattern}"))
            if suspicious_files:
                errors.append(
                    f"Suspicious executable files found: {', '.join(str(f.name) for f in suspicious_files[:5])}"
                )

        return errors

    def _validate_metadata(self, metadata: PluginMetadata) -> list[str]:
        """Validate plugin metadata."""
        errors: list[str] = []

        # Validate required fields
        if not metadata.plugin_id:
            errors.append("Plugin ID is required")

        if not metadata.name:
            errors.append("Plugin name is required")

        if not metadata.version:
            errors.append("Plugin version is required")

        if not metadata.entry_point:
            errors.append("Entry point is required")

        # Validate entry point format
        if metadata.entry_point:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", metadata.entry_point):
                errors.append(
                    f"Invalid entry point format: {metadata.entry_point}"
                )

        return errors

    async def _validate_file_sizes(
        self, plugin_path: Path
    ) -> tuple[list[str], list[str]]:
        """Validate file sizes in plugin directory."""
        errors: list[str] = []
        warnings: list[str] = []

        total_size = 0
        large_files: list[str] = []

        for file_path in plugin_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size

                # Warn about large individual files (>1MB)
                if size > 1024 * 1024:
                    large_files.append(f"{file_path.name} ({size // 1024 // 1024}MB)")

        # Check total size
        if total_size > self._max_file_size_bytes:
            errors.append(
                f"Plugin size ({total_size // 1024 // 1024}MB) exceeds limit ({self._max_file_size_bytes // 1024 // 1024}MB)"
            )

        # Warn about large files
        if large_files:
            warnings.append(
                f"Large files detected: {', '.join(large_files[:5])}"
            )

        return errors, warnings

    async def _validate_checksum(
        self, plugin_path: Path, expected_checksum: str
    ) -> list[str]:
        """Validate plugin checksum."""
        errors: list[str] = []

        try:
            # Calculate SHA-256 checksum of all files
            sha256 = hashlib.sha256()

            for file_path in sorted(plugin_path.rglob("*")):
                if file_path.is_file():
                    with file_path.open("rb") as f:
                        while chunk := f.read(8192):
                            sha256.update(chunk)

            actual_checksum = sha256.hexdigest()

            if actual_checksum != expected_checksum:
                errors.append(
                    f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                )

        except Exception as e:
            errors.append(f"Checksum validation failed: {e}")

        return errors

    async def _scan_code(
        self, plugin_path: Path, metadata: PluginMetadata
    ) -> tuple[list[str], list[str]]:
        """Scan plugin code for security issues."""
        errors: list[str] = []
        warnings: list[str] = []

        # Scan all Python files
        for py_file in plugin_path.rglob("*.py"):
            try:
                code = py_file.read_text()

                # Parse AST for static analysis
                tree = ast.parse(code, filename=str(py_file))

                # Check for dangerous imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(
                                alias.name.startswith(danger)
                                for danger in self.DANGEROUS_IMPORTS
                            ):
                                errors.append(
                                    f"Dangerous import detected in {py_file.name}: {alias.name}"
                                )

                    elif isinstance(node, ast.ImportFrom):
                        if node.module and any(
                            node.module.startswith(danger)
                            for danger in self.DANGEROUS_IMPORTS
                        ):
                            errors.append(
                                f"Dangerous import detected in {py_file.name}: from {node.module}"
                            )

                    # Check for eval/exec usage
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in self.DANGEROUS_BUILTINS:
                                errors.append(
                                    f"Dangerous built-in detected in {py_file.name}: {node.func.id}()"
                                )

                # Check for network operations
                if any(
                    pattern in code.lower()
                    for pattern in ["socket", "http", "urllib", "requests"]
                ):
                    if not metadata.permissions.network_hosts:
                        warnings.append(
                            f"Network code detected in {py_file.name} but no network hosts specified in permissions"
                        )

                # Check for file operations
                if "open(" in code or "Path(" in code:
                    if (
                        not metadata.permissions.filesystem_read
                        and not metadata.permissions.filesystem_write
                    ):
                        warnings.append(
                            f"File operations detected in {py_file.name} but no filesystem permissions specified"
                        )

            except SyntaxError as e:
                errors.append(f"Syntax error in {py_file.name}: {e}")
            except Exception as e:
                warnings.append(f"Code scan error in {py_file.name}: {e}")

        return errors, warnings

    async def _validate_permissions(
        self, metadata: PluginMetadata
    ) -> list[str]:
        """Validate plugin permissions."""
        warnings: list[str] = []

        # Warn about broad permissions
        if metadata.permissions.filesystem_write:
            if any(
                path in ["/*", "/", "*"]
                for path in metadata.permissions.filesystem_write
            ):
                warnings.append(
                    "Plugin requests write access to root filesystem - use more specific paths"
                )

        if metadata.permissions.network_hosts:
            if "*" in metadata.permissions.network_hosts:
                warnings.append(
                    "Plugin requests access to all network hosts - use specific hosts"
                )

        if metadata.permissions.external_apis:
            if "*" in metadata.permissions.external_apis:
                warnings.append(
                    "Plugin requests access to all external APIs - use specific APIs"
                )

        return warnings
