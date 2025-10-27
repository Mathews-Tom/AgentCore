"""
Comprehensive tests for Task JSON-RPC handlers.

Tests cover all task management JSON-RPC methods including creation, assignment,
lifecycle management, querying, and artifact handling.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.task import (
    TaskArtifact,
    TaskCreateResponse,
    TaskDefinition,
    TaskExecution,
    TaskPriority,
    TaskQuery,
    TaskQueryResponse,
    TaskRequirement,
    TaskStatus)
from agentcore.a2a_protocol.services import task_jsonrpc

# ==================== task.create Tests ====================


@pytest.mark.asyncio
async def test_task_create_success():
    """Test successful task creation."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        # Mock response
        mock_response = TaskCreateResponse(
            execution_id="exec-1",
            task_id="task-1",
            status=TaskStatus.PENDING,
            assigned_agent="agent-1",
            message="Task created")
        mock_manager.create_task = AsyncMock(return_value=mock_response)

        # Create request
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.create",
            id="1",
            params={
                "task_definition": {
                    "task_id": "task-1",
                    "task_type": "text.generation",
                    "title": "Test Task",
                    "description": "Test task",
                },
                "auto_assign": True,
            })

        result = await task_jsonrpc.handle_task_create(request)

        assert result["execution_id"] == "exec-1"
        assert result["task_id"] == "task-1"
        assert result["status"] == "pending"


@pytest.mark.asyncio
async def test_task_create_missing_params():
    """Test task creation with missing parameters."""
    request = JsonRpcRequest(jsonrpc="2.0", method="task.create", id="1", params=None)

    with pytest.raises(ValueError, match="Parameters required"):
        await task_jsonrpc.handle_task_create(request)


@pytest.mark.asyncio
async def test_task_create_missing_task_definition():
    """Test task creation without task_definition."""
    request = JsonRpcRequest(
        jsonrpc="2.0", method="task.create", id="1", params={"auto_assign": True}
    )

    with pytest.raises(ValueError, match="Missing required parameter: task_definition"):
        await task_jsonrpc.handle_task_create(request)


# ==================== task.get Tests ====================


@pytest.mark.asyncio
async def test_task_get_success():
    """Test successful task retrieval."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        # Mock execution
        task_def = TaskDefinition(
            task_id="task-1",
            task_type="text.generation",
            title="Test Task",
            description="Test")
        mock_execution = TaskExecution(
            execution_id="exec-1", task_definition=task_def, status=TaskStatus.RUNNING
        )
        mock_manager.get_task = AsyncMock(return_value=mock_execution)

        request = JsonRpcRequest(
            jsonrpc="2.0", method="task.get", id="1", params={"execution_id": "exec-1"}
        )

        result = await task_jsonrpc.handle_task_get(request)

        assert "task" in result
        assert result["task"]["execution_id"] == "exec-1"


@pytest.mark.asyncio
async def test_task_get_not_found():
    """Test retrieving non-existent task."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.get_task = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.get",
            id="1",
            params={"execution_id": "nonexistent"})

        with pytest.raises(ValueError, match="Task not found"):
            await task_jsonrpc.handle_task_get(request)


@pytest.mark.asyncio
async def test_task_get_missing_params():
    """Test task.get without execution_id."""
    request = JsonRpcRequest(jsonrpc="2.0", method="task.get", id="1", params={})

    with pytest.raises(ValueError, match="Parameter required: execution_id"):
        await task_jsonrpc.handle_task_get(request)


# ==================== task.assign Tests ====================


@pytest.mark.asyncio
async def test_task_assign_success():
    """Test successful task assignment."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.assign_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.assign",
            id="1",
            params={"execution_id": "exec-1", "agent_id": "agent-1"})

        result = await task_jsonrpc.handle_task_assign(request)

        assert result["success"] is True
        assert result["execution_id"] == "exec-1"
        assert result["agent_id"] == "agent-1"


@pytest.mark.asyncio
async def test_task_assign_failure():
    """Test failed task assignment."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.assign_task = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.assign",
            id="1",
            params={"execution_id": "exec-1", "agent_id": "agent-1"})

        with pytest.raises(ValueError, match="Task assignment failed"):
            await task_jsonrpc.handle_task_assign(request)


@pytest.mark.asyncio
async def test_task_assign_missing_params():
    """Test task.assign without required parameters."""
    request = JsonRpcRequest(
        jsonrpc="2.0", method="task.assign", id="1", params={"execution_id": "exec-1"}
    )

    with pytest.raises(ValueError, match="Missing required parameters"):
        await task_jsonrpc.handle_task_assign(request)


# ==================== task.start Tests ====================


@pytest.mark.asyncio
async def test_task_start_success():
    """Test successful task start."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.start_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.start",
            id="1",
            params={"execution_id": "exec-1"})

        result = await task_jsonrpc.handle_task_start(request)

        assert result["success"] is True
        assert result["execution_id"] == "exec-1"


@pytest.mark.asyncio
async def test_task_start_failure():
    """Test failed task start."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.start_task = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.start",
            id="1",
            params={"execution_id": "exec-1"})

        with pytest.raises(ValueError, match="Task start failed"):
            await task_jsonrpc.handle_task_start(request)


@pytest.mark.asyncio
async def test_task_start_missing_params():
    """Test task.start without execution_id."""
    request = JsonRpcRequest(jsonrpc="2.0", method="task.start", id="1", params={})

    with pytest.raises(ValueError, match="Parameter required: execution_id"):
        await task_jsonrpc.handle_task_start(request)


# ==================== task.complete Tests ====================


@pytest.mark.asyncio
async def test_task_complete_success():
    """Test successful task completion."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.complete_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.complete",
            id="1",
            params={"execution_id": "exec-1", "result_data": {"output": "result"}})

        result = await task_jsonrpc.handle_task_complete(request)

        assert result["success"] is True
        assert result["execution_id"] == "exec-1"


@pytest.mark.asyncio
async def test_task_complete_with_artifacts():
    """Test task completion with artifacts."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.complete_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.complete",
            id="1",
            params={
                "execution_id": "exec-1",
                "result_data": {"output": "result"},
                "artifacts": [
                    {"name": "output.txt", "type": "file", "content": "test content"}
                ],
            })

        result = await task_jsonrpc.handle_task_complete(request)

        assert result["success"] is True
        # Verify artifacts were parsed
        mock_manager.complete_task.assert_called_once()
        call_args = mock_manager.complete_task.call_args
        assert call_args[0][2] is not None  # artifacts parameter


@pytest.mark.asyncio
async def test_task_complete_failure():
    """Test failed task completion."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.complete_task = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.complete",
            id="1",
            params={"execution_id": "exec-1", "result_data": {"output": "result"}})

        with pytest.raises(ValueError, match="Task completion failed"):
            await task_jsonrpc.handle_task_complete(request)


@pytest.mark.asyncio
async def test_task_complete_missing_params():
    """Test task.complete without required parameters."""
    request = JsonRpcRequest(
        jsonrpc="2.0", method="task.complete", id="1", params={"execution_id": "exec-1"}
    )

    with pytest.raises(ValueError, match="Missing required parameters"):
        await task_jsonrpc.handle_task_complete(request)


# ==================== task.fail Tests ====================


@pytest.mark.asyncio
async def test_task_fail_success():
    """Test successful task failure recording."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.fail_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.fail",
            id="1",
            params={
                "execution_id": "exec-1",
                "error_message": "Task failed",
                "should_retry": True,
            })

        result = await task_jsonrpc.handle_task_fail(request)

        assert result["success"] is True
        assert result["error_message"] == "Task failed"
        assert result["should_retry"] is True


@pytest.mark.asyncio
async def test_task_fail_default_retry():
    """Test task fail with default should_retry."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.fail_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.fail",
            id="1",
            params={"execution_id": "exec-1", "error_message": "Task failed"})

        result = await task_jsonrpc.handle_task_fail(request)

        assert result["should_retry"] is True


@pytest.mark.asyncio
async def test_task_fail_missing_params():
    """Test task.fail without required parameters."""
    request = JsonRpcRequest(
        jsonrpc="2.0", method="task.fail", id="1", params={"execution_id": "exec-1"}
    )

    with pytest.raises(ValueError, match="Missing required parameters"):
        await task_jsonrpc.handle_task_fail(request)


# ==================== task.cancel Tests ====================


@pytest.mark.asyncio
async def test_task_cancel_success():
    """Test successful task cancellation."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.cancel_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.cancel",
            id="1",
            params={"execution_id": "exec-1"})

        result = await task_jsonrpc.handle_task_cancel(request)

        assert result["success"] is True
        assert result["execution_id"] == "exec-1"


@pytest.mark.asyncio
async def test_task_cancel_failure():
    """Test failed task cancellation."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.cancel_task = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.cancel",
            id="1",
            params={"execution_id": "exec-1"})

        with pytest.raises(ValueError, match="Task cancellation failed"):
            await task_jsonrpc.handle_task_cancel(request)


# ==================== task.update_progress Tests ====================


@pytest.mark.asyncio
async def test_task_update_progress_success():
    """Test successful progress update."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.update_task_progress = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.update_progress",
            id="1",
            params={
                "execution_id": "exec-1",
                "percentage": 50,
                "current_step": "Processing data",
            })

        result = await task_jsonrpc.handle_task_update_progress(request)

        assert result["success"] is True
        assert result["percentage"] == 50
        assert result["current_step"] == "Processing data"


@pytest.mark.asyncio
async def test_task_update_progress_without_step():
    """Test progress update without current_step."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.update_task_progress = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.update_progress",
            id="1",
            params={"execution_id": "exec-1", "percentage": 75})

        result = await task_jsonrpc.handle_task_update_progress(request)

        assert result["success"] is True
        assert result["percentage"] == 75


@pytest.mark.asyncio
async def test_task_update_progress_failure():
    """Test failed progress update."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.update_task_progress = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.update_progress",
            id="1",
            params={"execution_id": "exec-1", "percentage": 50})

        with pytest.raises(ValueError, match="Task progress update failed"):
            await task_jsonrpc.handle_task_update_progress(request)


# ==================== task.query Tests ====================


@pytest.mark.asyncio
async def test_task_query_all():
    """Test querying all tasks."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_response = TaskQueryResponse(
            tasks=[{"task_id": "task-1"}, {"task_id": "task-2"}],
            total_count=2,
            has_more=False,
            query=TaskQuery())
        mock_manager.query_tasks = AsyncMock(return_value=mock_response)

        request = JsonRpcRequest(jsonrpc="2.0", method="task.query", id="1", params={})

        result = await task_jsonrpc.handle_task_query(request)

        assert len(result["tasks"]) == 2
        assert result["total_count"] == 2
        assert result["count"] == 2  # Compatibility alias


@pytest.mark.asyncio
async def test_task_query_with_status_filter():
    """Test querying tasks by status."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_response = TaskQueryResponse(
            tasks=[{"task_id": "task-1"}],
            total_count=1,
            has_more=False,
            query=TaskQuery(status=TaskStatus.RUNNING))
        mock_manager.query_tasks = AsyncMock(return_value=mock_response)

        request = JsonRpcRequest(
            jsonrpc="2.0", method="task.query", id="1", params={"status": "running"}
        )

        result = await task_jsonrpc.handle_task_query(request)

        assert len(result["tasks"]) == 1


@pytest.mark.asyncio
async def test_task_query_with_filters():
    """Test querying tasks with multiple filters."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_response = TaskQueryResponse(
            tasks=[], total_count=0, has_more=False, query=TaskQuery()
        )
        mock_manager.query_tasks = AsyncMock(return_value=mock_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.query",
            id="1",
            params={
                "task_type": "text.generation",
                "assigned_agent": "agent-1",
                "priority": "high",
                "tags": ["ai", "nlp"],
                "limit": 10,
                "offset": 5,
            })

        result = await task_jsonrpc.handle_task_query(request)

        # Verify query was called
        mock_manager.query_tasks.assert_called_once()
        call_query = mock_manager.query_tasks.call_args[0][0]
        assert call_query.task_type == "text.generation"
        assert call_query.limit == 10
        assert call_query.offset == 5


@pytest.mark.asyncio
async def test_task_query_with_time_filters():
    """Test querying tasks with time filters."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_response = TaskQueryResponse(
            tasks=[], total_count=0, has_more=False, query=TaskQuery()
        )
        mock_manager.query_tasks = AsyncMock(return_value=mock_response)

        created_after = datetime.now(UTC) - timedelta(days=7)
        created_before = datetime.now(UTC)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.query",
            id="1",
            params={
                "created_after": created_after.isoformat(),
                "created_before": created_before.isoformat(),
            })

        result = await task_jsonrpc.handle_task_query(request)

        # Verify time filters were parsed
        mock_manager.query_tasks.assert_called_once()


# ==================== task.dependencies Tests ====================


@pytest.mark.asyncio
async def test_task_dependencies_success():
    """Test retrieving task dependencies."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_dependencies = {
            "prerequisites": ["task-0"],
            "dependents": ["task-2", "task-3"],
        }
        mock_manager.get_task_dependencies = AsyncMock(return_value=mock_dependencies)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.dependencies",
            id="1",
            params={"task_id": "task-1"})

        result = await task_jsonrpc.handle_task_dependencies(request)

        assert result["task_id"] == "task-1"
        assert result["dependencies"]["prerequisites"] == ["task-0"]


@pytest.mark.asyncio
async def test_task_dependencies_missing_params():
    """Test task.dependencies without task_id."""
    request = JsonRpcRequest(
        jsonrpc="2.0", method="task.dependencies", id="1", params={}
    )

    with pytest.raises(ValueError, match="Parameter required: task_id"):
        await task_jsonrpc.handle_task_dependencies(request)


# ==================== task.ready Tests ====================


@pytest.mark.asyncio
async def test_task_ready_success():
    """Test retrieving ready tasks."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.get_ready_tasks = AsyncMock(return_value=["exec-1", "exec-2"])

        request = JsonRpcRequest(jsonrpc="2.0", method="task.ready", id="1", params={})

        result = await task_jsonrpc.handle_task_ready(request)

        assert len(result["ready_tasks"]) == 2
        assert result["count"] == 2


# ==================== task.cleanup Tests ====================


@pytest.mark.asyncio
async def test_task_cleanup_success():
    """Test successful task cleanup."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.cleanup_old_tasks = AsyncMock(return_value=5)

        request = JsonRpcRequest(
            jsonrpc="2.0", method="task.cleanup", id="1", params={"max_age_days": 30}
        )

        result = await task_jsonrpc.handle_task_cleanup(request)

        assert result["success"] is True
        assert result["cleanup_count"] == 5
        assert result["max_age_days"] == 30


@pytest.mark.asyncio
async def test_task_cleanup_default_age():
    """Test task cleanup with default max_age_days."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.cleanup_old_tasks = AsyncMock(return_value=3)

        request = JsonRpcRequest(
            jsonrpc="2.0", method="task.cleanup", id="1", params={}
        )

        result = await task_jsonrpc.handle_task_cleanup(request)

        assert result["max_age_days"] == 30  # Default


# ==================== task.summary Tests ====================


@pytest.mark.asyncio
async def test_task_summary_success():
    """Test retrieving task summary."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_execution = MagicMock()
        mock_execution.to_summary.return_value = {
            "execution_id": "exec-1",
            "task_id": "task-1",
            "status": "RUNNING",
        }
        mock_manager.get_task = AsyncMock(return_value=mock_execution)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.summary",
            id="1",
            params={"execution_id": "exec-1"})

        result = await task_jsonrpc.handle_task_summary(request)

        assert result["execution_id"] == "exec-1"
        assert result["task_id"] == "task-1"


@pytest.mark.asyncio
async def test_task_summary_not_found():
    """Test task summary for non-existent task."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.get_task = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.summary",
            id="1",
            params={"execution_id": "nonexistent"})

        with pytest.raises(ValueError, match="Task not found"):
            await task_jsonrpc.handle_task_summary(request)


# ==================== task.add_artifact Tests ====================


@pytest.mark.asyncio
async def test_task_add_artifact_success():
    """Test successful artifact addition."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.add_task_artifact = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.add_artifact",
            id="1",
            params={
                "execution_id": "exec-1",
                "name": "output.txt",
                "type": "file",
                "content": "test content",
                "metadata": {"size": 100},
            })

        result = await task_jsonrpc.handle_task_add_artifact(request)

        assert result["success"] is True
        assert result["artifact_name"] == "output.txt"
        assert result["artifact_type"] == "file"


@pytest.mark.asyncio
async def test_task_add_artifact_without_metadata():
    """Test adding artifact without metadata."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.add_task_artifact = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.add_artifact",
            id="1",
            params={
                "execution_id": "exec-1",
                "name": "output.txt",
                "type": "file",
                "content": "test content",
            })

        result = await task_jsonrpc.handle_task_add_artifact(request)

        assert result["success"] is True


@pytest.mark.asyncio
async def test_task_add_artifact_missing_params():
    """Test adding artifact without required parameters."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="task.add_artifact",
        id="1",
        params={"execution_id": "exec-1", "name": "output.txt"})

    with pytest.raises(ValueError, match="Missing required parameters"):
        await task_jsonrpc.handle_task_add_artifact(request)


# ==================== task.get_artifacts Tests ====================


@pytest.mark.asyncio
async def test_task_get_artifacts_success():
    """Test retrieving task artifacts."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_artifacts = [
            TaskArtifact(name="output.txt", type="file", content="test"),
            TaskArtifact(name="data.json", type="json", content={"key": "value"}),
        ]
        mock_manager.get_task_artifacts = AsyncMock(return_value=mock_artifacts)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.get_artifacts",
            id="1",
            params={"execution_id": "exec-1"})

        result = await task_jsonrpc.handle_task_get_artifacts(request)

        assert len(result["artifacts"]) == 2
        assert result["count"] == 2


@pytest.mark.asyncio
async def test_task_get_artifacts_task_not_found():
    """Test getting artifacts for non-existent task."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.get_task_artifacts = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.get_artifacts",
            id="1",
            params={"execution_id": "nonexistent"})

        with pytest.raises(ValueError, match="Task not found"):
            await task_jsonrpc.handle_task_get_artifacts(request)


# ==================== task.status_transitions Tests ====================


@pytest.mark.asyncio
async def test_task_status_transitions_success():
    """Test retrieving valid status transitions."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_execution = MagicMock()
        mock_execution.status = MagicMock()
        mock_execution.status.value = "assigned"
        mock_execution.is_terminal = False
        mock_manager.get_task = AsyncMock(return_value=mock_execution)
        mock_manager.get_task_status_transitions = AsyncMock(
            return_value=[TaskStatus.RUNNING, TaskStatus.CANCELLED]
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.status_transitions",
            id="1",
            params={"execution_id": "exec-1"})

        result = await task_jsonrpc.handle_task_status_transitions(request)

        assert result["current_status"] == "assigned"
        assert "running" in result["valid_transitions"]
        assert "cancelled" in result["valid_transitions"]
        assert result["is_terminal"] is False


@pytest.mark.asyncio
async def test_task_status_transitions_not_found():
    """Test status transitions for non-existent task."""
    with patch(
        "agentcore.a2a_protocol.services.task_jsonrpc.task_manager"
    ) as mock_manager:
        mock_manager.get_task = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="task.status_transitions",
            id="1",
            params={"execution_id": "nonexistent"})

        with pytest.raises(ValueError, match="Task not found"):
            await task_jsonrpc.handle_task_status_transitions(request)
