"""Integration tests for FileOperationsTool (TOOL-015).

These tests validate the File Operations Tool implementation with real filesystem
operations. Tests cover:
- Read, write, and list_directory operations
- Path validation and security restrictions
- Size limits and permission handling
- Error scenarios (not found, permission denied, traversal attempts)

Note: Tests use temporary directories for safe filesystem operations.
"""

import os
import tempfile
from pathlib import Path

import pytest

from agentcore.agent_runtime.models.tool_integration import ToolExecutionStatus
from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.builtin.file_operations_tools import FileOperationsTool


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create execution context for testing."""
    return ExecutionContext(
        user_id="integration-test-user",
        agent_id="integration-test-agent",
        trace_id="integration-trace-file-ops-123",
        session_id="integration-session-file-ops",
    )


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        yield workspace


@pytest.fixture
def file_operations_tool(temp_workspace):
    """Create File Operations Tool with temporary workspace."""
    return FileOperationsTool(allowed_directories=[str(temp_workspace)])


# Read Operation Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_read_success(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test reading a file successfully."""
    # Create test file
    test_file = temp_workspace / "test.txt"
    test_content = "Hello, World!\nThis is a test file.\n"
    test_file.write_text(test_content)

    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(test_file),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result

    # Validate result structure
    assert result_data["content"] == test_content
    assert result_data["size_bytes"] == len(test_content.encode("utf-8"))
    assert result_data["path"] == str(test_file)
    assert result_data["encoding"] == "utf-8"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_read_not_found(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test reading a non-existent file."""
    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(temp_workspace / "nonexistent.txt"),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "not found" in result.error.lower()
    assert result.error_type == "OperationError"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_read_directory(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test attempting to read a directory (should fail)."""
    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(temp_workspace),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "not a file" in result.error.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_read_large_file(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test reading a large file (within size limit)."""
    # Create 1MB file (within 10MB limit)
    test_file = temp_workspace / "large.txt"
    content = "x" * (1024 * 1024)  # 1MB
    test_file.write_text(content)

    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(test_file),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result["size_bytes"] == len(content)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_read_too_large(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test reading a file exceeding size limit."""
    # Create 11MB file (exceeds 10MB limit)
    test_file = temp_workspace / "too_large.txt"
    # Write in chunks to avoid memory issues
    with open(test_file, "w") as f:
        for _ in range(11):
            f.write("x" * (1024 * 1024))  # 11MB total

    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(test_file),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "too large" in result.error.lower()


# Write Operation Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_write_success(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test writing a file successfully."""
    test_file = temp_workspace / "output.txt"
    test_content = "This is new content.\nWritten by the tool.\n"

    result = await file_operations_tool.execute(
        parameters={
            "operation": "write",
            "path": str(test_file),
            "content": test_content,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result

    # Validate result
    assert result_data["path"] == str(test_file)
    assert result_data["size_bytes"] == len(test_content.encode("utf-8"))
    assert result_data["operation"] == "write"

    # Verify file was actually written
    assert test_file.exists()
    assert test_file.read_text() == test_content


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_write_overwrite(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test overwriting an existing file."""
    test_file = temp_workspace / "overwrite.txt"
    test_file.write_text("Original content")

    new_content = "New content"
    result = await file_operations_tool.execute(
        parameters={
            "operation": "write",
            "path": str(test_file),
            "content": new_content,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify file was overwritten
    assert test_file.read_text() == new_content


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_write_create_subdirectory(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test writing to a file in a non-existent subdirectory (should create it)."""
    test_file = temp_workspace / "subdir" / "nested" / "file.txt"
    test_content = "Content in nested directory"

    result = await file_operations_tool.execute(
        parameters={
            "operation": "write",
            "path": str(test_file),
            "content": test_content,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify directory and file were created
    assert test_file.exists()
    assert test_file.read_text() == test_content


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_write_empty_content(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test writing an empty file."""
    test_file = temp_workspace / "empty.txt"

    result = await file_operations_tool.execute(
        parameters={
            "operation": "write",
            "path": str(test_file),
            "content": "",
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert test_file.exists()
    assert test_file.read_text() == ""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_write_unicode(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test writing unicode content."""
    test_file = temp_workspace / "unicode.txt"
    test_content = "Hello 世界! Café ☕ 🐍"

    result = await file_operations_tool.execute(
        parameters={
            "operation": "write",
            "path": str(test_file),
            "content": test_content,
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert test_file.read_text(encoding="utf-8") == test_content


# List Directory Operation Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_list_directory_success(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test listing directory contents successfully."""
    # Create test files and directories
    (temp_workspace / "file1.txt").write_text("content1")
    (temp_workspace / "file2.txt").write_text("content2")
    (temp_workspace / "subdir").mkdir()
    (temp_workspace / "subdir" / "nested.txt").write_text("nested")

    result = await file_operations_tool.execute(
        parameters={
            "operation": "list_directory",
            "path": str(temp_workspace),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.error is None
    result_data = result.result

    # Validate result structure
    assert result_data["path"] == str(temp_workspace)
    assert "entries" in result_data
    assert "count" in result_data
    assert result_data["count"] == 3  # file1.txt, file2.txt, subdir

    # Validate entry structure
    entries_by_name = {entry["name"]: entry for entry in result_data["entries"]}

    # Check file1.txt
    assert "file1.txt" in entries_by_name
    file1_entry = entries_by_name["file1.txt"]
    assert file1_entry["type"] == "file"
    assert file1_entry["size_bytes"] > 0
    assert "modified_time" in file1_entry

    # Check subdir
    assert "subdir" in entries_by_name
    subdir_entry = entries_by_name["subdir"]
    assert subdir_entry["type"] == "directory"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_list_directory_empty(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test listing an empty directory."""
    empty_dir = temp_workspace / "empty"
    empty_dir.mkdir()

    result = await file_operations_tool.execute(
        parameters={
            "operation": "list_directory",
            "path": str(empty_dir),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    result_data = result.result

    assert result_data["count"] == 0
    assert len(result_data["entries"]) == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_list_directory_not_found(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test listing a non-existent directory."""
    result = await file_operations_tool.execute(
        parameters={
            "operation": "list_directory",
            "path": str(temp_workspace / "nonexistent"),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_list_directory_file(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test attempting to list a file (should fail)."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    result = await file_operations_tool.execute(
        parameters={
            "operation": "list_directory",
            "path": str(test_file),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert "not a directory" in result.error.lower()


# Security Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_security_directory_traversal(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test that directory traversal attempts are blocked."""
    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(temp_workspace / ".." / ".." / "etc" / "passwd"),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error_type == "SecurityError"
    assert "traversal" in result.error.lower() or "allowed" in result.error.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_security_outside_allowed_directory(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
):
    """Test that access outside allowed directories is blocked."""
    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": "/tmp/test.txt",  # Not in allowed directories
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error_type == "SecurityError"
    assert "allowed" in result.error.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_security_path_validation(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test various path validation scenarios."""
    # Create a file
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    # Try to access it with .. in path
    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(temp_workspace / "subdir" / ".." / "test.txt"),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error_type == "SecurityError"


# Parameter Validation Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_invalid_operation(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test with invalid operation type."""
    result = await file_operations_tool.execute(
        parameters={
            "operation": "delete",  # Not supported
            "path": str(temp_workspace / "test.txt"),
        },
        context=execution_context,
    )

    assert result.status == ToolExecutionStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_missing_content_for_write(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test write operation without content parameter."""
    # Should use empty string as default
    result = await file_operations_tool.execute(
        parameters={
            "operation": "write",
            "path": str(temp_workspace / "test.txt"),
            # content parameter missing
        },
        context=execution_context,
    )

    # Should succeed with empty content
    assert result.status == ToolExecutionStatus.SUCCESS


# Metadata Tests


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_metadata_populated(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test that result metadata is correctly populated."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(test_file),
        },
        context=execution_context,
    )

    assert result.request_id == execution_context.request_id
    assert result.tool_id == "file_operations"
    assert result.timestamp is not None
    assert result.metadata["trace_id"] == execution_context.trace_id
    assert result.metadata["agent_id"] == execution_context.agent_id
    assert result.metadata["operation"] == "read"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_file_execution_time_tracking(
    file_operations_tool: FileOperationsTool,
    execution_context: ExecutionContext,
    temp_workspace: Path,
):
    """Test that execution time is tracked."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    result = await file_operations_tool.execute(
        parameters={
            "operation": "read",
            "path": str(test_file),
        },
        context=execution_context,
    )

    assert result.execution_time_ms > 0
    # File operations should be fast
    assert result.execution_time_ms < 1000  # Less than 1 second
