"""
Unit Tests for Workflow Hooks System

Tests hook configuration, execution, event matching, and integration.
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.orchestration.hooks.executor import HookExecutor
from agentcore.orchestration.hooks.manager import HookManager
from agentcore.orchestration.hooks.models import (
    HookConfig,
    HookEvent,
    HookExecution,
    HookExecutionMode,
    HookStatus,
    HookTrigger,
)


class TestHookConfig:
    """Test HookConfig model."""

    def test_hook_config_creation(self):
        """Test creating a hook configuration."""
        hook = HookConfig(
            name="test_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["hello"],
            priority=50,
        )

        assert hook.name == "test_hook"
        assert hook.trigger == HookTrigger.POST_TASK
        assert hook.command == "echo"
        assert hook.args == ["hello"]
        assert hook.priority == 50
        assert hook.enabled is True
        assert hook.timeout_ms == 30000

    def test_hook_event_matching_no_filters(self):
        """Test hook matches any event when no filters set."""
        hook = HookConfig(
            name="test_hook", trigger=HookTrigger.POST_TASK, command="echo"
        )

        event_data = {"task_id": "123", "status": "completed"}
        assert hook.matches_event(event_data) is True

    def test_hook_event_matching_with_filters(self):
        """Test hook matches events with filters."""
        hook = HookConfig(
            name="test_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            event_filters={"status": "completed"},
        )

        # Matching event
        assert hook.matches_event({"status": "completed"}) is True

        # Non-matching event
        assert hook.matches_event({"status": "failed"}) is False

    def test_hook_event_matching_nested_filters(self):
        """Test hook matches events with nested filters."""
        hook = HookConfig(
            name="test_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            event_filters={"task.status": "completed"},
        )

        # Matching event
        assert hook.matches_event({"task": {"status": "completed"}}) is True

        # Non-matching event
        assert hook.matches_event({"task": {"status": "failed"}}) is False


class TestHookExecution:
    """Test HookExecution model."""

    def test_hook_execution_creation(self):
        """Test creating a hook execution record."""
        hook_id = uuid4()
        execution = HookExecution(
            hook_id=hook_id,
            trigger=HookTrigger.POST_TASK,
            status=HookStatus.PENDING,
            started_at=datetime.now(UTC),
            input_data={"test": "data"},
        )

        assert execution.hook_id == hook_id
        assert execution.trigger == HookTrigger.POST_TASK
        assert execution.status == HookStatus.PENDING
        assert execution.retry_count == 0
        assert execution.is_retry is False

    def test_mark_completed(self):
        """Test marking execution as completed."""
        import time

        execution = HookExecution(
            hook_id=uuid4(),
            trigger=HookTrigger.POST_TASK,
            status=HookStatus.RUNNING,
            started_at=datetime.now(UTC),
            input_data={},
        )

        # Small delay to ensure measurable duration
        time.sleep(0.001)

        output = {"result": "success"}
        execution.mark_completed(output)

        assert execution.status == HookStatus.COMPLETED
        assert execution.output_data == output
        assert execution.completed_at is not None
        assert execution.duration_ms is not None
        assert execution.duration_ms >= 0

    def test_mark_failed(self):
        """Test marking execution as failed."""
        execution = HookExecution(
            hook_id=uuid4(),
            trigger=HookTrigger.POST_TASK,
            status=HookStatus.RUNNING,
            started_at=datetime.now(UTC),
            input_data={},
        )

        error_msg = "Test error"
        execution.mark_failed(error_msg)

        assert execution.status == HookStatus.FAILED
        assert execution.error_message == error_msg
        assert execution.completed_at is not None
        assert execution.duration_ms is not None

    def test_mark_timeout(self):
        """Test marking execution as timed out."""
        execution = HookExecution(
            hook_id=uuid4(),
            trigger=HookTrigger.POST_TASK,
            status=HookStatus.RUNNING,
            started_at=datetime.now(UTC),
            input_data={},
        )

        execution.mark_timeout()

        assert execution.status == HookStatus.TIMEOUT
        assert "timeout" in execution.error_message.lower()
        assert execution.completed_at is not None


class TestHookExecutor:
    """Test HookExecutor."""

    @pytest.mark.asyncio
    async def test_execute_shell_command_success(self):
        """Test executing a successful shell command."""
        executor = HookExecutor()

        hook = HookConfig(
            name="echo_test",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["test"],
        )

        event = HookEvent(
            trigger=HookTrigger.POST_TASK, source="test", data={"key": "value"}
        )

        execution = await executor.execute_hook(hook, event)

        assert execution.status == HookStatus.COMPLETED
        assert execution.output_data is not None
        assert execution.output_data["exit_code"] == 0
        assert "test" in execution.output_data["stdout"]

    @pytest.mark.asyncio
    async def test_execute_shell_command_failure(self):
        """Test executing a failing shell command."""
        executor = HookExecutor()

        hook = HookConfig(
            name="fail_test",
            trigger=HookTrigger.POST_TASK,
            command="false",  # Always exits with code 1
        )

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        execution = await executor.execute_hook(hook, event)

        assert execution.status == HookStatus.FAILED
        assert execution.error_message is not None

    @pytest.mark.asyncio
    async def test_execute_hook_timeout(self):
        """Test hook execution timeout."""
        executor = HookExecutor()

        hook = HookConfig(
            name="sleep_test",
            trigger=HookTrigger.POST_TASK,
            command="sleep",
            args=["10"],
            timeout_ms=100,  # 100ms timeout
        )

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        execution = await executor.execute_hook(hook, event)

        assert execution.status == HookStatus.TIMEOUT
        assert execution.error_message is not None

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """Test retry logic when first attempt succeeds."""
        executor = HookExecutor()

        hook = HookConfig(
            name="echo_test",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["success"],
            retry_enabled=True,
            max_retries=3,
        )

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        execution = await executor.execute_with_retry(hook, event)

        assert execution.status == HookStatus.COMPLETED
        assert execution.retry_count == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_disabled(self):
        """Test that retries are skipped when disabled."""
        executor = HookExecutor()

        hook = HookConfig(
            name="fail_test",
            trigger=HookTrigger.POST_TASK,
            command="false",
            retry_enabled=False,
            max_retries=3,
        )

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        execution = await executor.execute_with_retry(hook, event)

        assert execution.status == HookStatus.FAILED
        assert execution.retry_count == 0


class TestHookManager:
    """Test HookManager."""

    def test_register_hook_success(self):
        """Test successful hook registration."""
        manager = HookManager()

        hook = HookConfig(
            name="test_hook", trigger=HookTrigger.POST_TASK, command="echo"
        )

        success, message = manager.register_hook(hook)

        assert success is True
        assert "successfully" in message.lower()
        assert hook.hook_id in manager._hooks

    def test_register_hook_duplicate_name(self):
        """Test registering hook with duplicate name fails."""
        manager = HookManager()

        hook1 = HookConfig(
            name="duplicate_hook", trigger=HookTrigger.POST_TASK, command="echo"
        )
        hook2 = HookConfig(
            name="duplicate_hook", trigger=HookTrigger.POST_TASK, command="cat"
        )

        manager.register_hook(hook1)
        success, message = manager.register_hook(hook2)

        assert success is False
        assert "already exists" in message.lower()

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        manager = HookManager()

        hook = HookConfig(
            name="test_hook", trigger=HookTrigger.POST_TASK, command="echo"
        )
        manager.register_hook(hook)

        result = manager.unregister_hook(hook.hook_id)

        assert result is True
        assert hook.hook_id not in manager._hooks

    def test_list_hooks_no_filter(self):
        """Test listing all hooks."""
        manager = HookManager()

        hook1 = HookConfig(
            name="hook1", trigger=HookTrigger.POST_TASK, command="echo"
        )
        hook2 = HookConfig(
            name="hook2", trigger=HookTrigger.PRE_TASK, command="cat"
        )

        manager.register_hook(hook1)
        manager.register_hook(hook2)

        hooks = manager.list_hooks()
        assert len(hooks) == 2

    def test_list_hooks_with_trigger_filter(self):
        """Test listing hooks filtered by trigger."""
        manager = HookManager()

        hook1 = HookConfig(
            name="hook1", trigger=HookTrigger.POST_TASK, command="echo"
        )
        hook2 = HookConfig(
            name="hook2", trigger=HookTrigger.PRE_TASK, command="cat"
        )
        hook3 = HookConfig(
            name="hook3", trigger=HookTrigger.POST_TASK, command="ls"
        )

        manager.register_hook(hook1)
        manager.register_hook(hook2)
        manager.register_hook(hook3)

        hooks = manager.list_hooks(trigger=HookTrigger.POST_TASK)
        assert len(hooks) == 2
        assert all(h.trigger == HookTrigger.POST_TASK for h in hooks)

    def test_list_hooks_enabled_only(self):
        """Test listing only enabled hooks."""
        manager = HookManager()

        hook1 = HookConfig(
            name="hook1", trigger=HookTrigger.POST_TASK, command="echo", enabled=True
        )
        hook2 = HookConfig(
            name="hook2", trigger=HookTrigger.POST_TASK, command="cat", enabled=False
        )

        manager.register_hook(hook1)
        manager.register_hook(hook2)

        hooks = manager.list_hooks(enabled_only=True)
        assert len(hooks) == 1
        assert hooks[0].name == "hook1"

    @pytest.mark.asyncio
    async def test_trigger_hooks_no_matches(self):
        """Test triggering hooks with no matches."""
        manager = HookManager()

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        executions = await manager.trigger_hooks(event)
        assert len(executions) == 0

    @pytest.mark.asyncio
    async def test_trigger_hooks_sync_execution(self):
        """Test triggering hooks with synchronous execution."""
        manager = HookManager()

        hook = HookConfig(
            name="test_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["test"],
            execution_mode=HookExecutionMode.SYNC,
        )
        manager.register_hook(hook)

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 1
        assert executions[0].status == HookStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_trigger_hooks_priority_ordering(self):
        """Test hooks execute in priority order."""
        manager = HookManager()

        # Lower priority runs first
        hook1 = HookConfig(
            name="high_priority",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["first"],
            priority=10,
            execution_mode=HookExecutionMode.SYNC,
        )
        hook2 = HookConfig(
            name="low_priority",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["second"],
            priority=100,
            execution_mode=HookExecutionMode.SYNC,
        )

        manager.register_hook(hook2)  # Register in reverse order
        manager.register_hook(hook1)

        event = HookEvent(trigger=HookTrigger.POST_TASK, source="test", data={})

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 2
        # First execution should be from high priority hook
        assert executions[0].hook_id == hook1.hook_id
        assert executions[1].hook_id == hook2.hook_id

    @pytest.mark.asyncio
    async def test_trigger_hooks_with_filters(self):
        """Test hooks trigger only when event matches filters."""
        manager = HookManager()

        hook = HookConfig(
            name="filtered_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            event_filters={"status": "completed"},
            execution_mode=HookExecutionMode.SYNC,
        )
        manager.register_hook(hook)

        # Matching event
        event1 = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={"status": "completed"},
        )
        executions1 = await manager.trigger_hooks(event1)
        assert len(executions1) == 1

        # Non-matching event
        event2 = HookEvent(
            trigger=HookTrigger.POST_TASK, source="test", data={"status": "failed"}
        )
        executions2 = await manager.trigger_hooks(event2)
        assert len(executions2) == 0

    def test_get_statistics(self):
        """Test getting hook statistics."""
        manager = HookManager()

        hook1 = HookConfig(
            name="hook1", trigger=HookTrigger.POST_TASK, command="echo"
        )
        hook2 = HookConfig(
            name="hook2", trigger=HookTrigger.PRE_TASK, command="cat"
        )

        manager.register_hook(hook1)
        manager.register_hook(hook2)

        stats = manager.get_statistics()

        assert stats["total_hooks"] == 2
        assert stats["active_hooks"] == 2
        assert HookTrigger.POST_TASK.value in stats["hooks_by_trigger"]


class TestHookEvent:
    """Test HookEvent model."""

    def test_hook_event_creation(self):
        """Test creating a hook event."""
        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test_source",
            data={"key": "value"},
            workflow_id="wf-123",
            task_id="task-456",
        )

        assert event.trigger == HookTrigger.POST_TASK
        assert event.source == "test_source"
        assert event.data == {"key": "value"}
        assert event.workflow_id == "wf-123"
        assert event.task_id == "task-456"
        assert event.event_id is not None
        assert event.timestamp is not None
