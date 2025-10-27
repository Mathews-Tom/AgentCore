"""
Tests for Task Manager Service

Comprehensive test suite for TaskManager to achieve 85%+ coverage.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agentcore.a2a_protocol.models.agent import AgentCard, AgentStatus
from agentcore.a2a_protocol.models.task import (
    DependencyType,
    TaskArtifact,
    TaskCreateRequest,
    TaskDefinition,
    TaskDependency,
    TaskExecution,
    TaskPriority,
    TaskQuery,
    TaskRequirement,
    TaskStatus)
from agentcore.a2a_protocol.services.task_manager import TaskManager, task_manager


@pytest.fixture
def manager():
    """Create fresh TaskManager instance for each test."""
    return TaskManager()


@pytest.fixture
def sample_task_def():
    """Create sample task definition."""
    return TaskDefinition(
        task_id=str(uuid4()),
        task_type="text.generation",
        title="Test Task",
        description="Test Description",
        priority=TaskPriority.NORMAL,
        parameters={"key": "value"})


@pytest.fixture
def sample_agent():
    """Create sample agent for testing."""
    from agentcore.a2a_protocol.models.agent import (
        AgentAuthentication,
        AgentCapability,
        AgentEndpoint,
        AuthenticationType,
        EndpointType)

    return AgentCard(
        agent_id="test-agent",
        agent_name="Test Agent",
        agent_version="1.0.0",
        status=AgentStatus.ACTIVE,
        capabilities=[AgentCapability(name="text.generation", version="1.0")],
        endpoints=[AgentEndpoint(url="http://test.local", type=EndpointType.HTTP)],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE, required=False
        ))


# ==================== Task Creation Tests ====================


@pytest.mark.asyncio
async def test_create_task_basic(manager, sample_task_def):
    """Test basic task creation."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)

    response = await manager.create_task(request)

    assert response.task_id == sample_task_def.task_id
    assert response.status == TaskStatus.PENDING.value
    assert response.assigned_agent is None

    # Verify task stored
    execution = await manager.get_task(response.execution_id)
    assert execution is not None
    assert execution.task_id == sample_task_def.task_id
    assert execution.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_create_task_with_dependencies_fails_when_missing(manager):
    """Test task creation fails when dependency doesn't exist."""
    task_def = TaskDefinition(
        task_id=str(uuid4()),
        task_type="text.generation",
        title="Test Task",
        description="Test",
        dependencies=[
            TaskDependency(task_id="nonexistent", type=DependencyType.PREDECESSOR)
        ])

    request = TaskCreateRequest(task_definition=task_def, auto_assign=False)

    with pytest.raises(ValueError, match="Dependency task"):
        await manager.create_task(request)


@pytest.mark.asyncio
async def test_create_task_with_valid_dependencies(manager):
    """Test task creation with valid dependencies."""
    # Create prerequisite task first
    prereq_task = TaskDefinition(
        task_id="prereq-task",
        task_type="text.generation",
        title="Prerequisite",
        description="Test")
    prereq_request = TaskCreateRequest(task_definition=prereq_task, auto_assign=False)
    await manager.create_task(prereq_request)

    # Create dependent task
    dependent_task = TaskDefinition(
        task_id="dependent-task",
        task_type="text.generation",
        title="Dependent",
        description="Test",
        dependencies=[
            TaskDependency(task_id="prereq-task", type=DependencyType.PREDECESSOR)
        ])
    dependent_request = TaskCreateRequest(
        task_definition=dependent_task, auto_assign=False
    )

    response = await manager.create_task(dependent_request)

    assert response.task_id == "dependent-task"
    assert response.status == TaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_create_task_with_auto_assign(manager, sample_task_def, sample_agent):
    """Test task creation with auto-assignment."""
    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        # Mock agent manager methods
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)
        mock_agent_mgr.list_all_agents = AsyncMock(
            return_value=[{"agent_id": "test-agent"}]
        )

        # Create task with auto-assign
        request = TaskCreateRequest(
            task_definition=sample_task_def,
            auto_assign=True,
            preferred_agent="test-agent")

        response = await manager.create_task(request)

        assert response.assigned_agent == "test-agent"

        # Verify task assigned
        execution = await manager.get_task(response.execution_id)
        assert execution.assigned_agent == "test-agent"
        assert execution.status == TaskStatus.ASSIGNED


# ==================== Task Assignment Tests ====================


@pytest.mark.asyncio
async def test_assign_task_success(manager, sample_task_def, sample_agent):
    """Test successful task assignment."""
    # Create task
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        # Assign task
        success = await manager.assign_task(response.execution_id, "test-agent")

        assert success is True

        # Verify assignment
        execution = await manager.get_task(response.execution_id)
        assert execution.assigned_agent == "test-agent"
        assert execution.status == TaskStatus.ASSIGNED


@pytest.mark.asyncio
async def test_assign_task_nonexistent_task(manager):
    """Test assigning nonexistent task returns False."""
    success = await manager.assign_task("nonexistent", "test-agent")
    assert success is False


@pytest.mark.asyncio
async def test_assign_task_inactive_agent(manager, sample_task_def):
    """Test assignment fails for inactive agent."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    inactive_agent = MagicMock()
    inactive_agent.is_active.return_value = False

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=inactive_agent)

        success = await manager.assign_task(response.execution_id, "inactive-agent")

        assert success is False


@pytest.mark.asyncio
async def test_assign_task_lacking_capabilities(manager):
    """Test assignment fails when agent lacks capabilities."""
    # Create task with specific requirements
    task_def = TaskDefinition(
        task_id=str(uuid4()),
        task_type="text.generation",
        title="Test Task",
        description="Test",
        requirements=TaskRequirement(required_capabilities=["special-capability"]))
    request = TaskCreateRequest(task_definition=task_def, auto_assign=False)
    response = await manager.create_task(request)

    incapable_agent = MagicMock()
    incapable_agent.is_active.return_value = True
    incapable_agent.has_capability.return_value = False

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=incapable_agent)

        success = await manager.assign_task(response.execution_id, "incapable-agent")

        assert success is False


# ==================== Task Execution Tests ====================


@pytest.mark.asyncio
async def test_start_task_success(manager, sample_task_def, sample_agent):
    """Test starting task execution."""
    # Create and assign task
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")

        # Start task
        success = await manager.start_task(response.execution_id)

        assert success is True

        # Verify status
        execution = await manager.get_task(response.execution_id)
        assert execution.status == TaskStatus.RUNNING
        assert execution.started_at is not None


@pytest.mark.asyncio
async def test_start_task_nonexistent(manager):
    """Test starting nonexistent task returns False."""
    success = await manager.start_task("nonexistent")
    assert success is False


@pytest.mark.asyncio
async def test_start_task_with_unsatisfied_dependencies(manager):
    """Test starting task fails when dependencies not satisfied."""
    # Create prerequisite task
    prereq_task = TaskDefinition(
        task_id="prereq-task",
        task_type="text.generation",
        title="Prerequisite",
        description="Test")
    await manager.create_task(
        TaskCreateRequest(task_definition=prereq_task, auto_assign=False)
    )

    # Create dependent task
    dependent_task = TaskDefinition(
        task_id="dependent-task",
        task_type="text.generation",
        title="Dependent",
        description="Test",
        dependencies=[
            TaskDependency(task_id="prereq-task", type=DependencyType.PREDECESSOR)
        ])
    dependent_response = await manager.create_task(
        TaskCreateRequest(task_definition=dependent_task, auto_assign=False)
    )

    # Try to start dependent task before prerequisite completes
    success = await manager.start_task(dependent_response.execution_id)

    assert success is False


@pytest.mark.asyncio
async def test_complete_task_success(manager, sample_task_def, sample_agent):
    """Test completing task execution."""
    # Create, assign, and start task
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")
        await manager.start_task(response.execution_id)

        # Complete task
        result_data = {"output": "success"}
        success = await manager.complete_task(response.execution_id, result_data)

        assert success is True

        # Verify completion
        execution = await manager.get_task(response.execution_id)
        assert execution.status == TaskStatus.COMPLETED
        assert execution.result_data == result_data
        assert execution.completed_at is not None


@pytest.mark.asyncio
async def test_complete_task_nonexistent(manager):
    """Test completing nonexistent task returns False."""
    success = await manager.complete_task("nonexistent", {"output": "test"})
    assert success is False


@pytest.mark.asyncio
async def test_complete_task_with_artifacts(manager, sample_task_def, sample_agent):
    """Test completing task with artifacts."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")
        await manager.start_task(response.execution_id)

        # Complete with artifacts
        artifacts = [
            TaskArtifact(
                name="output.txt",
                type="file",
                content="test content")
        ]
        success = await manager.complete_task(
            response.execution_id, {"output": "success"}, artifacts
        )

        assert success is True

        # Verify artifacts
        execution = await manager.get_task(response.execution_id)
        assert len(execution.artifacts) == 1
        assert execution.artifacts[0].name == "output.txt"


@pytest.mark.asyncio
async def test_fail_task_success(manager, sample_task_def, sample_agent):
    """Test failing task execution."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")
        await manager.start_task(response.execution_id)

        # Fail task
        error_message = "Test error"
        success = await manager.fail_task(response.execution_id, error_message)

        assert success is True

        # Verify failure
        execution = await manager.get_task(response.execution_id)
        assert execution.status == TaskStatus.FAILED
        assert execution.error_message == error_message


@pytest.mark.asyncio
async def test_fail_task_nonexistent(manager):
    """Test failing nonexistent task returns False."""
    success = await manager.fail_task("nonexistent", "error")
    assert success is False


@pytest.mark.asyncio
async def test_cancel_task_success(manager, sample_task_def):
    """Test cancelling task."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    # Cancel task
    success = await manager.cancel_task(response.execution_id)

    assert success is True

    # Verify cancellation
    execution = await manager.get_task(response.execution_id)
    assert execution.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_task_nonexistent(manager):
    """Test cancelling nonexistent task returns False."""
    success = await manager.cancel_task("nonexistent")
    assert success is False


# ==================== Progress and Artifacts Tests ====================


@pytest.mark.asyncio
async def test_update_task_progress_success(manager, sample_task_def, sample_agent):
    """Test updating task progress."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")
        await manager.start_task(response.execution_id)

        # Update progress
        success = await manager.update_task_progress(
            response.execution_id, 50.0, "halfway done"
        )

        assert success is True

        # Verify progress
        execution = await manager.get_task(response.execution_id)
        assert execution.progress_percentage == 50.0
        assert execution.current_step == "halfway done"


@pytest.mark.asyncio
async def test_update_task_progress_nonexistent(manager):
    """Test updating progress of nonexistent task returns False."""
    success = await manager.update_task_progress("nonexistent", 50.0)
    assert success is False


@pytest.mark.asyncio
async def test_add_task_artifact_success(manager, sample_task_def, sample_agent):
    """Test adding artifact to task."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")
        await manager.start_task(response.execution_id)

        # Add artifact
        success = await manager.add_task_artifact(
            response.execution_id,
            "output.txt",
            "file",
            "test content",
            {"size": 100})

        assert success is True

        # Verify artifact
        artifacts = await manager.get_task_artifacts(response.execution_id)
        assert len(artifacts) == 1
        assert artifacts[0].name == "output.txt"
        assert artifacts[0].type == "file"


@pytest.mark.asyncio
async def test_add_task_artifact_nonexistent(manager):
    """Test adding artifact to nonexistent task returns False."""
    success = await manager.add_task_artifact(
        "nonexistent", "test.txt", "file", "content"
    )
    assert success is False


@pytest.mark.asyncio
async def test_get_task_artifacts_nonexistent(manager):
    """Test getting artifacts from nonexistent task returns None."""
    artifacts = await manager.get_task_artifacts("nonexistent")
    assert artifacts is None


# ==================== Query and Status Tests ====================


@pytest.mark.asyncio
async def test_query_tasks_all(manager, sample_task_def):
    """Test querying all tasks."""
    # Create multiple tasks
    for i in range(5):
        task_def = TaskDefinition(
            task_id=f"task-{i}",
            task_type="text.generation",
            title=f"Task {i}",
            description="Test")
        await manager.create_task(
            TaskCreateRequest(task_definition=task_def, auto_assign=False)
        )

    # Query all tasks
    query = TaskQuery()
    response = await manager.query_tasks(query)

    assert response.total_count == 5
    assert len(response.tasks) == 5


@pytest.mark.asyncio
async def test_query_tasks_by_status(manager):
    """Test querying tasks by status."""
    # Create tasks with different statuses
    task1 = TaskDefinition(
        task_id="task-1",
        task_type="text.generation",
        title="Task 1",
        description="Test")
    response1 = await manager.create_task(
        TaskCreateRequest(task_definition=task1, auto_assign=False)
    )

    task2 = TaskDefinition(
        task_id="task-2",
        task_type="text.generation",
        title="Task 2",
        description="Test")
    response2 = await manager.create_task(
        TaskCreateRequest(task_definition=task2, auto_assign=False)
    )

    # Cancel one task
    await manager.cancel_task(response2.execution_id)

    # Query pending tasks
    query = TaskQuery(status=TaskStatus.PENDING)
    response = await manager.query_tasks(query)

    assert response.total_count == 1
    assert response.tasks[0]["task_id"] == "task-1"


@pytest.mark.asyncio
async def test_query_tasks_with_pagination(manager):
    """Test task query pagination."""
    # Create 10 tasks
    for i in range(10):
        task_def = TaskDefinition(
            task_id=f"task-{i}",
            task_type="text.generation",
            title=f"Task {i}",
            description="Test")
        await manager.create_task(
            TaskCreateRequest(task_definition=task_def, auto_assign=False)
        )

    # Query with pagination
    query = TaskQuery(limit=5, offset=0)
    response = await manager.query_tasks(query)

    assert response.total_count == 10
    assert len(response.tasks) == 5
    assert response.has_more is True

    # Second page
    query2 = TaskQuery(limit=5, offset=5)
    response2 = await manager.query_tasks(query2)

    assert len(response2.tasks) == 5
    assert response2.has_more is False


@pytest.mark.asyncio
async def test_get_task_status_transitions(manager, sample_task_def):
    """Test getting valid status transitions."""
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    # Get valid transitions for pending task
    transitions = await manager.get_task_status_transitions(response.execution_id)

    assert transitions is not None
    assert TaskStatus.ASSIGNED in transitions
    assert TaskStatus.CANCELLED in transitions


@pytest.mark.asyncio
async def test_get_task_status_transitions_nonexistent(manager):
    """Test getting transitions for nonexistent task returns None."""
    transitions = await manager.get_task_status_transitions("nonexistent")
    assert transitions is None


@pytest.mark.asyncio
async def test_get_task_dependencies(manager):
    """Test getting task dependencies."""
    # Create prerequisite task
    prereq_task = TaskDefinition(
        task_id="prereq-task",
        task_type="text.generation",
        title="Prerequisite",
        description="Test")
    await manager.create_task(
        TaskCreateRequest(task_definition=prereq_task, auto_assign=False)
    )

    # Create dependent task
    dependent_task = TaskDefinition(
        task_id="dependent-task",
        task_type="text.generation",
        title="Dependent",
        description="Test",
        dependencies=[
            TaskDependency(task_id="prereq-task", type=DependencyType.PREDECESSOR)
        ])
    await manager.create_task(
        TaskCreateRequest(task_definition=dependent_task, auto_assign=False)
    )

    # Get dependencies
    deps = await manager.get_task_dependencies("dependent-task")

    assert "prereq-task" in deps["prerequisites"]
    assert deps["dependents"] == []

    # Get dependent relationships
    prereq_deps = await manager.get_task_dependencies("prereq-task")
    assert "dependent-task" in prereq_deps["dependents"]


@pytest.mark.asyncio
async def test_get_ready_tasks(manager):
    """Test getting ready tasks."""
    # Create independent task (ready immediately)
    task1 = TaskDefinition(
        task_id="task-1",
        task_type="text.generation",
        title="Task 1",
        description="Test")
    await manager.create_task(
        TaskCreateRequest(task_definition=task1, auto_assign=False)
    )

    # Get ready tasks
    ready = await manager.get_ready_tasks()

    assert len(ready) == 1


# ==================== Cleanup Tests ====================


@pytest.mark.asyncio
async def test_cleanup_old_tasks(manager, sample_task_def, sample_agent):
    """Test cleanup of old completed tasks."""
    # Create and complete a task
    request = TaskCreateRequest(task_definition=sample_task_def, auto_assign=False)
    response = await manager.create_task(request)

    with patch(
        "agentcore.a2a_protocol.services.task_manager.agent_manager"
    ) as mock_agent_mgr:
        mock_agent_mgr.get_agent = AsyncMock(return_value=sample_agent)

        await manager.assign_task(response.execution_id, "test-agent")
        await manager.start_task(response.execution_id)
        await manager.complete_task(response.execution_id, {"output": "success"})

        # Manually set completed_at to old date
        execution = await manager.get_task(response.execution_id)
        execution.completed_at = datetime.now(UTC) - timedelta(days=31)

        # Cleanup old tasks
        cleanup_count = await manager.cleanup_old_tasks(max_age_days=30)

        assert cleanup_count == 1

        # Verify task removed
        removed_execution = await manager.get_task(response.execution_id)
        assert removed_execution is None


@pytest.mark.asyncio
async def test_cleanup_old_tasks_no_old_tasks(manager):
    """Test cleanup when no old tasks exist."""
    cleanup_count = await manager.cleanup_old_tasks(max_age_days=30)
    assert cleanup_count == 0


# ==================== Global Instance Test ====================


def test_global_task_manager_instance():
    """Test that global task_manager instance exists."""
    assert task_manager is not None
    assert isinstance(task_manager, TaskManager)
