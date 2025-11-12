"""Native Tool ABC implementations for file operations.

This module provides secure file operations tools with path validation,
size limits, and directory whitelisting to prevent security vulnerabilities.

Security Features:
- Path traversal prevention (no ../ escapes)
- File size limits (10MB default)
- Directory whitelisting
- Permission error handling
- Detailed security logging

Migration from: N/A (new implementation)
Status: TOOL-013 Implementation
"""

import os
import time
from pathlib import Path
from typing import Any

import structlog

from ...models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from ..base import ExecutionContext, Tool

logger = structlog.get_logger()

# Security constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_ALLOWED_DIRS = [
    "/tmp",
    "/var/tmp",
    "./data",
    "./uploads",
    "./workspace",
]


class FileOperationsTool(Tool):
    """Secure file operations tool for read, write, and list operations.

    Implements strict security controls:
    - Path validation to prevent directory traversal
    - File size limits (10MB default)
    - Directory whitelisting
    - Permission and existence checks
    """

    def __init__(self, allowed_directories: list[str] | None = None):
        """Initialize file operations tool with security settings.

        Args:
            allowed_directories: List of allowed base directories.
                                Defaults to safe temporary directories.
        """
        self.allowed_directories = allowed_directories or DEFAULT_ALLOWED_DIRS
        self.max_file_size = MAX_FILE_SIZE_BYTES

        metadata = ToolDefinition(
            tool_id="file_operations",
            name="File Operations",
            description="Secure file operations: read, write, list_directory with path validation and size limits",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "operation": ToolParameter(
                    name="operation",
                    type="string",
                    description="File operation to perform",
                    required=True,
                    enum=["read", "write", "list_directory"],
                ),
                "path": ToolParameter(
                    name="path",
                    type="string",
                    description="File or directory path",
                    required=True,
                    min_length=1,
                    max_length=1024,
                ),
                "content": ToolParameter(
                    name="content",
                    type="string",
                    description="File content (for write operation)",
                    required=False,
                    max_length=MAX_FILE_SIZE_BYTES,
                ),
            },
            auth_method=AuthMethod.NONE,
            is_retryable=False,  # File operations should not be auto-retried
            max_retries=0,
            timeout_seconds=30,
            tags=["file", "filesystem", "io", "utility"],
        )
        super().__init__(metadata)

    def _validate_path(self, path: str) -> tuple[bool, str | None]:
        """Validate file path for security.

        Checks:
        1. No directory traversal (../)
        2. Path is within allowed directories
        3. Path is absolute or safely relative

        Args:
            path: File path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Resolve to absolute path to catch traversal attempts
            abs_path = os.path.abspath(path)

            # Check for directory traversal attempts
            if ".." in path:
                return False, "Directory traversal (..) not allowed"

            # Check if path is within allowed directories
            is_allowed = False
            for allowed_dir in self.allowed_directories:
                allowed_abs = os.path.abspath(allowed_dir)
                if abs_path.startswith(allowed_abs):
                    is_allowed = True
                    break

            if not is_allowed:
                allowed_list = ", ".join(self.allowed_directories)
                return (
                    False,
                    f"Path must be within allowed directories: {allowed_list}",
                )

            return True, None

        except Exception as e:
            return False, f"Path validation error: {str(e)}"

    async def _read_file(
        self, path: str, context: ExecutionContext
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Read file contents with size limit check.

        Args:
            path: File path to read
            context: Execution context

        Returns:
            Tuple of (result_data, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(path):
                return None, f"File not found: {path}"

            # Check if it's a file (not directory)
            if not os.path.isfile(path):
                return None, f"Path is not a file: {path}"

            # Check file size
            file_size = os.path.getsize(path)
            if file_size > self.max_file_size:
                max_mb = self.max_file_size / (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                return (
                    None,
                    f"File too large: {actual_mb:.2f}MB (max: {max_mb}MB)",
                )

            # Read file contents
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(
                "file_read_success",
                path=path,
                size_bytes=file_size,
                agent_id=context.agent_id,
                trace_id=context.trace_id,
            )

            return {
                "content": content,
                "size_bytes": file_size,
                "path": path,
                "encoding": "utf-8",
            }, None

        except PermissionError:
            return None, f"Permission denied: {path}"
        except UnicodeDecodeError:
            return None, f"File is not valid UTF-8 text: {path}"
        except Exception as e:
            return None, f"Read error: {str(e)}"

    async def _write_file(
        self, path: str, content: str, context: ExecutionContext
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Write content to file with size limit check.

        Args:
            path: File path to write
            content: Content to write
            context: Execution context

        Returns:
            Tuple of (result_data, error_message)
        """
        try:
            # Check content size
            content_size = len(content.encode("utf-8"))
            if content_size > self.max_file_size:
                max_mb = self.max_file_size / (1024 * 1024)
                actual_mb = content_size / (1024 * 1024)
                return (
                    None,
                    f"Content too large: {actual_mb:.2f}MB (max: {max_mb}MB)",
                )

            # Create parent directory if needed
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                "file_write_success",
                path=path,
                size_bytes=content_size,
                agent_id=context.agent_id,
                trace_id=context.trace_id,
            )

            return {
                "path": path,
                "size_bytes": content_size,
                "encoding": "utf-8",
                "operation": "write",
            }, None

        except PermissionError:
            return None, f"Permission denied: {path}"
        except Exception as e:
            return None, f"Write error: {str(e)}"

    async def _list_directory(
        self, path: str, context: ExecutionContext
    ) -> tuple[dict[str, Any] | None, str | None]:
        """List directory contents with metadata.

        Args:
            path: Directory path to list
            context: Execution context

        Returns:
            Tuple of (result_data, error_message)
        """
        try:
            # Check directory exists
            if not os.path.exists(path):
                return None, f"Directory not found: {path}"

            # Check if it's a directory
            if not os.path.isdir(path):
                return None, f"Path is not a directory: {path}"

            # List directory contents
            entries = []
            for entry_name in os.listdir(path):
                entry_path = os.path.join(path, entry_name)
                try:
                    stat_info = os.stat(entry_path)
                    entries.append(
                        {
                            "name": entry_name,
                            "path": entry_path,
                            "type": "file" if os.path.isfile(entry_path) else "directory",
                            "size_bytes": stat_info.st_size if os.path.isfile(entry_path) else None,
                            "modified_time": stat_info.st_mtime,
                        }
                    )
                except (PermissionError, FileNotFoundError):
                    # Skip entries we can't access
                    entries.append(
                        {
                            "name": entry_name,
                            "path": entry_path,
                            "type": "unknown",
                            "error": "Permission denied or not found",
                        }
                    )

            logger.info(
                "directory_list_success",
                path=path,
                entry_count=len(entries),
                agent_id=context.agent_id,
                trace_id=context.trace_id,
            )

            return {
                "path": path,
                "entries": entries,
                "count": len(entries),
            }, None

        except PermissionError:
            return None, f"Permission denied: {path}"
        except Exception as e:
            return None, f"List error: {str(e)}"

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
    ) -> ToolResult:
        """Execute file operation with security validation.

        Args:
            parameters: Dictionary with keys:
                - operation: str - Operation type (read, write, list_directory)
                - path: str - File or directory path
                - content: str - Content for write operation (optional)
            context: Execution context

        Returns:
            ToolResult with operation result or error
        """
        start_time = time.time()

        try:
            # Validate parameters
            is_valid, error = await self.validate_parameters(parameters)
            if not is_valid:
                execution_time_ms = (time.time() - start_time) * 1000
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error=error,
                    error_type="ValidationError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            operation = parameters["operation"]
            path = parameters["path"]

            # Validate path security
            is_valid, error = self._validate_path(path)
            if not is_valid:
                execution_time_ms = (time.time() - start_time) * 1000
                logger.warning(
                    "file_operation_security_violation",
                    operation=operation,
                    path=path,
                    error=error,
                    agent_id=context.agent_id,
                    trace_id=context.trace_id,
                )
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error=f"Security violation: {error}",
                    error_type="SecurityError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            # Execute operation
            result_data = None
            error_msg = None

            if operation == "read":
                result_data, error_msg = await self._read_file(path, context)
            elif operation == "write":
                content = parameters.get("content", "")
                result_data, error_msg = await self._write_file(path, content, context)
            elif operation == "list_directory":
                result_data, error_msg = await self._list_directory(path, context)
            else:
                error_msg = f"Invalid operation: {operation}"

            execution_time_ms = (time.time() - start_time) * 1000

            if error_msg:
                return ToolResult(
                    request_id=context.request_id,
                    tool_id=self.metadata.tool_id,
                    status=ToolExecutionStatus.FAILED,
                    result={},
                    error=error_msg,
                    error_type="OperationError",
                    execution_time_ms=execution_time_ms,
                    metadata={"trace_id": context.trace_id},
                )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.SUCCESS,
                result=result_data or {},
                error=None,
                execution_time_ms=execution_time_ms,
                metadata={
                    "trace_id": context.trace_id,
                    "agent_id": context.agent_id,
                    "operation": operation,
                },
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "file_operation_error",
                error=str(e),
                parameters=parameters,
                agent_id=context.agent_id,
                trace_id=context.trace_id,
            )

            return ToolResult(
                request_id=context.request_id,
                tool_id=self.metadata.tool_id,
                status=ToolExecutionStatus.FAILED,
                result={},
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                metadata={"trace_id": context.trace_id},
            )
