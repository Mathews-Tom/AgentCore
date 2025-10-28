"""
Integration Tests for Workflow Hooks System

Tests real hook execution, event integration, and end-to-end flows.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from agentcore.orchestration.hooks.integration import (
    HookEventIntegration,
    hook_integration)
from agentcore.orchestration.hooks.manager import HookManager
from agentcore.orchestration.hooks.models import (
    HookConfig,
    HookEvent,
    HookExecutionMode,
    HookTrigger)
from agentcore.a2a_protocol.models.events import Event, EventType, EventPriority


class TestHookIntegration:
    """Test hook integration with A2A Event System."""

    @pytest.mark.asyncio
    async def test_event_to_hook_trigger_mapping(self):
        """Test A2A events are mapped to hook triggers."""
        integration = HookEventIntegration()

        # Verify mappings exist
        assert EventType.TASK_CREATED in integration._event_to_hook_mapping
        assert EventType.TASK_COMPLETED in integration._event_to_hook_mapping
        assert integration._event_to_hook_mapping[EventType.TASK_CREATED] == HookTrigger.PRE_TASK
        assert integration._event_to_hook_mapping[EventType.TASK_COMPLETED] == HookTrigger.POST_TASK

    @pytest.mark.asyncio
    async def test_convert_a2a_event_to_hook_event(self):
        """Test converting A2A Event to HookEvent."""
        integration = HookEventIntegration()

        a2a_event = Event(
            event_type=EventType.TASK_COMPLETED,
            source="test_agent",
            data={"task_id": "task-123", "result": "success"},
            metadata={"session_id": "session-456"})

        hook_event = integration._convert_to_hook_event(
            a2a_event, HookTrigger.POST_TASK
        )

        assert hook_event.trigger == HookTrigger.POST_TASK
        assert hook_event.source == "test_agent"
        assert hook_event.task_id == "task-123"
        assert hook_event.session_id == "session-456"
        assert hook_event.data == a2a_event.data

    @pytest.mark.asyncio
    async def test_trigger_custom_hook_directly(self):
        """Test triggering hooks directly without A2A event."""
        integration = HookEventIntegration()
        manager = HookManager()

        # Register a hook
        hook = HookConfig(
            name="test_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["test"],
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        # Trigger directly
        integration.trigger_custom_hook(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={"result": "success"},
            task_id="task-123")

        # Give async task time to execute
        await asyncio.sleep(0.1)


class TestRealHookExecution:
    """Test real hook execution with actual commands."""

    @pytest.mark.asyncio
    async def test_shell_command_hook_execution(self):
        """Test executing real shell command hook."""
        manager = HookManager()

        hook = HookConfig(
            name="echo_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["Hello from hook"],
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={"task": "completed"})

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 1
        assert executions[0].status.value == "completed"
        assert "Hello from hook" in executions[0].output_data["stdout"]

    @pytest.mark.asyncio
    async def test_file_creation_hook(self):
        """Test hook that creates a file."""
        manager = HookManager()

        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "hook_output.txt"

            hook = HookConfig(
                name="file_hook",
                trigger=HookTrigger.POST_TASK,
                command="sh",
                args=["-c", f"echo 'Hook executed' > {test_file}"],
                execution_mode=HookExecutionMode.SYNC)
            manager.register_hook(hook)

            event = HookEvent(
                trigger=HookTrigger.POST_TASK,
                source="test",
                data={})

            executions = await manager.trigger_hooks(event)

            assert len(executions) == 1
            assert executions[0].status.value == "completed"
            assert test_file.exists()
            assert test_file.read_text().strip() == "Hook executed"

    @pytest.mark.asyncio
    async def test_multiple_hooks_execution_order(self):
        """Test multiple hooks execute in priority order."""
        manager = HookManager()

        # Create hooks with different priorities
        hook1 = HookConfig(
            name="high_priority",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["first"],
            priority=10,
            execution_mode=HookExecutionMode.SYNC)

        hook2 = HookConfig(
            name="medium_priority",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["second"],
            priority=50,
            execution_mode=HookExecutionMode.SYNC)

        hook3 = HookConfig(
            name="low_priority",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["third"],
            priority=100,
            execution_mode=HookExecutionMode.SYNC)

        manager.register_hook(hook3)  # Register out of order
        manager.register_hook(hook1)
        manager.register_hook(hook2)

        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={})

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 3
        # Verify execution order matches priority
        assert executions[0].hook_id == hook1.hook_id
        assert executions[1].hook_id == hook2.hook_id
        assert executions[2].hook_id == hook3.hook_id

    @pytest.mark.asyncio
    async def test_hook_with_event_filter(self):
        """Test hook only executes when event matches filter."""
        manager = HookManager()

        hook = HookConfig(
            name="filtered_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["filtered"],
            event_filters={"status": "success", "priority": "high"},
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        # Event that matches filter
        matching_event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={"status": "success", "priority": "high"})
        executions1 = await manager.trigger_hooks(matching_event)
        assert len(executions1) == 1

        # Event that doesn't match filter
        non_matching_event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={"status": "failed", "priority": "low"})
        executions2 = await manager.trigger_hooks(non_matching_event)
        assert len(executions2) == 0

    @pytest.mark.asyncio
    async def test_hook_retry_on_failure(self):
        """Test hook retries on failure."""
        manager = HookManager()

        hook = HookConfig(
            name="retry_hook",
            trigger=HookTrigger.POST_TASK,
            command="false",  # Always fails
            retry_enabled=True,
            max_retries=2,
            retry_delay_ms=100,
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={})

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 1
        # Should have failed after retries
        assert executions[0].status.value == "failed"
        assert executions[0].retry_count == 2

    @pytest.mark.asyncio
    async def test_hook_timeout_handling(self):
        """Test hook timeout is enforced."""
        manager = HookManager()

        hook = HookConfig(
            name="timeout_hook",
            trigger=HookTrigger.POST_TASK,
            command="sleep",
            args=["10"],
            timeout_ms=200,  # 200ms timeout
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={})

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 1
        assert executions[0].status.value == "timeout"
        assert "timeout" in executions[0].error_message.lower()

    @pytest.mark.asyncio
    async def test_hook_execution_context_preservation(self):
        """Test hook execution preserves workflow context."""
        manager = HookManager()

        hook = HookConfig(
            name="context_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["context_test"],
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={},
            workflow_id="wf-123",
            task_id="task-456",
            session_id="session-789")

        executions = await manager.trigger_hooks(event)

        assert len(executions) == 1
        assert executions[0].workflow_id == "wf-123"
        assert executions[0].task_id == "task-456"
        assert executions[0].session_id == "session-789"

    @pytest.mark.asyncio
    async def test_disabled_hook_not_executed(self):
        """Test disabled hooks are not executed."""
        manager = HookManager()

        hook = HookConfig(
            name="disabled_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["should_not_run"],
            enabled=False,  # Disabled
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={})

        executions = await manager.trigger_hooks(event)

        # No executions because hook is disabled
        assert len(executions) == 0

    @pytest.mark.asyncio
    async def test_hook_statistics_tracking(self):
        """Test hook execution statistics are tracked."""
        manager = HookManager()

        hook = HookConfig(
            name="stats_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["stats"],
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        # Execute hook multiple times
        for _ in range(3):
            event = HookEvent(
                trigger=HookTrigger.POST_TASK,
                source="test",
                data={})
            await manager.trigger_hooks(event)

        stats = manager.get_statistics()

        assert stats["total_hooks"] == 1
        assert stats["active_hooks"] == 1
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 3
        assert stats["failed_executions"] == 0
        assert stats["avg_execution_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_hook_execution_history(self):
        """Test hook execution history is maintained."""
        manager = HookManager()

        hook = HookConfig(
            name="history_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["history"],
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        # Execute hook
        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={"test": "data"})
        await manager.trigger_hooks(event)

        # Get execution history
        history = manager.get_execution_history(hook_id=hook.hook_id, limit=10)

        assert len(history) == 1
        assert history[0].hook_id == hook.hook_id
        assert history[0].trigger == HookTrigger.POST_TASK
        assert history[0].input_data == {"test": "data"}

    @pytest.mark.asyncio
    async def test_cleanup_old_executions(self):
        """Test cleanup of old execution records."""
        manager = HookManager()

        hook = HookConfig(
            name="cleanup_hook",
            trigger=HookTrigger.POST_TASK,
            command="echo",
            args=["cleanup"],
            execution_mode=HookExecutionMode.SYNC)
        manager.register_hook(hook)

        # Execute hook
        event = HookEvent(
            trigger=HookTrigger.POST_TASK,
            source="test",
            data={})
        await manager.trigger_hooks(event)

        # Verify execution exists
        history_before = manager.get_execution_history()
        assert len(history_before) > 0

        # Cleanup (with 0 day retention to clean everything)
        cleaned = await manager.cleanup_old_executions(retention_days=0)

        # All executions should be cleaned
        history_after = manager.get_execution_history()
        assert len(history_after) == 0
