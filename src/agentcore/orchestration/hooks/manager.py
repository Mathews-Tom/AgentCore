"""
Hook Manager

Manages hook registration, event matching, and async execution via Redis Streams.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import structlog

from agentcore.orchestration.hooks.executor import HookExecutor
from agentcore.orchestration.hooks.models import (
    HookConfig,
    HookEvent,
    HookExecution,
    HookExecutionMode,
    HookStatus,
    HookTrigger,
)
from agentcore.orchestration.streams.producer import StreamProducer

logger = structlog.get_logger()


class HookManager:
    """
    Hook manager for workflow automation.

    Manages hook registration, event matching, and async execution.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # Hook storage (in-memory, backed by PostgreSQL)
        self._hooks: dict[UUID, HookConfig] = {}
        self._hooks_by_trigger: dict[HookTrigger, list[UUID]] = defaultdict(list)

        # Execution tracking
        self._executions: dict[UUID, HookExecution] = {}
        self._execution_history: list[HookExecution] = []

        # Executor
        self._executor = HookExecutor()

        # Redis Streams producer for async execution
        self._stream_producer: StreamProducer | None = None

        # Statistics
        self._stats = {
            "total_hooks": 0,
            "active_hooks": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0.0,
        }

    def set_stream_producer(self, producer: StreamProducer) -> None:
        """Set Redis Streams producer for async hook execution."""
        self._stream_producer = producer

    # ==================== Hook Registration ====================

    def register_hook(self, hook: HookConfig) -> tuple[bool, str]:
        """
        Register a new hook.

        Args:
            hook: Hook configuration

        Returns:
            Tuple of (success, message)
        """
        # Validate hook
        if not hook.name:
            return False, "Hook name is required"

        if not hook.command:
            return False, "Hook command is required"

        # Check for duplicate names (same name, different ID)
        for existing_hook in self._hooks.values():
            if (
                existing_hook.name == hook.name
                and existing_hook.hook_id != hook.hook_id
            ):
                return False, f"Hook with name '{hook.name}' already exists"

        # Store hook
        self._hooks[hook.hook_id] = hook
        self._hooks_by_trigger[hook.trigger].append(hook.hook_id)

        # Update stats
        self._stats["total_hooks"] = len(self._hooks)
        self._stats["active_hooks"] = sum(
            1 for h in self._hooks.values() if h.enabled
        )

        self.logger.info(
            "Hook registered",
            hook_id=str(hook.hook_id),
            hook_name=hook.name,
            trigger=hook.trigger.value,
            priority=hook.priority,
        )

        return True, f"Hook '{hook.name}' registered successfully"

    def unregister_hook(self, hook_id: UUID) -> bool:
        """
        Unregister a hook.

        Args:
            hook_id: Hook ID to unregister

        Returns:
            True if unregistered, False if not found
        """
        hook = self._hooks.get(hook_id)
        if not hook:
            return False

        # Remove from indexes
        self._hooks_by_trigger[hook.trigger].remove(hook_id)

        # Remove hook
        del self._hooks[hook_id]

        # Update stats
        self._stats["total_hooks"] = len(self._hooks)
        self._stats["active_hooks"] = sum(
            1 for h in self._hooks.values() if h.enabled
        )

        self.logger.info(
            "Hook unregistered", hook_id=str(hook_id), hook_name=hook.name
        )

        return True

    def get_hook(self, hook_id: UUID) -> HookConfig | None:
        """Get hook by ID."""
        return self._hooks.get(hook_id)

    def list_hooks(
        self, trigger: HookTrigger | None = None, enabled_only: bool = False
    ) -> list[HookConfig]:
        """
        List hooks with optional filters.

        Args:
            trigger: Filter by trigger type
            enabled_only: Only return enabled hooks

        Returns:
            List of matching hooks
        """
        hooks = list(self._hooks.values())

        if trigger:
            hook_ids = self._hooks_by_trigger.get(trigger, [])
            hooks = [self._hooks[hid] for hid in hook_ids if hid in self._hooks]

        if enabled_only:
            hooks = [h for h in hooks if h.enabled]

        return hooks

    # ==================== Event Handling ====================

    async def trigger_hooks(self, event: HookEvent) -> list[HookExecution]:
        """
        Trigger hooks matching the event.

        Args:
            event: Event to process

        Returns:
            List of hook executions
        """
        # Find matching hooks
        matching_hooks = self._find_matching_hooks(event)

        if not matching_hooks:
            self.logger.debug(
                "No matching hooks for event",
                trigger=event.trigger.value,
                source=event.source,
            )
            return []

        # Sort by priority (lower priority runs first)
        matching_hooks.sort(key=lambda h: h.priority)

        self.logger.info(
            "Triggering hooks",
            trigger=event.trigger.value,
            hook_count=len(matching_hooks),
            hooks=[h.name for h in matching_hooks],
        )

        # Execute hooks based on execution mode
        executions: list[HookExecution] = []

        for hook in matching_hooks:
            if hook.execution_mode == HookExecutionMode.SYNC:
                # Synchronous execution
                execution = await self._execute_hook_sync(hook, event)
                executions.append(execution)

                # Stop if hook failed and not always_run
                if execution.status != HookStatus.COMPLETED and not hook.always_run:
                    self.logger.warning(
                        "Hook failed, stopping execution chain",
                        hook_name=hook.name,
                        failed_count=len(
                            [e for e in executions if e.status != HookStatus.COMPLETED]
                        ),
                    )
                    break

            elif hook.execution_mode == HookExecutionMode.ASYNC:
                # Asynchronous execution via Redis Streams
                execution = await self._execute_hook_async(hook, event)
                executions.append(execution)

            else:  # FIRE_AND_FORGET
                # Fire and forget (no tracking)
                asyncio.create_task(self._execute_hook_fire_and_forget(hook, event))

        return executions

    def _find_matching_hooks(self, event: HookEvent) -> list[HookConfig]:
        """Find hooks that match the event."""
        matching: list[HookConfig] = []

        # Get hooks for this trigger type
        hook_ids = self._hooks_by_trigger.get(event.trigger, [])

        for hook_id in hook_ids:
            hook = self._hooks.get(hook_id)
            if not hook:
                continue

            # Skip disabled hooks
            if not hook.enabled:
                continue

            # Check event filters
            if hook.matches_event(event.data):
                matching.append(hook)

        return matching

    async def _execute_hook_sync(
        self, hook: HookConfig, event: HookEvent
    ) -> HookExecution:
        """Execute hook synchronously with retry logic."""
        execution = await self._executor.execute_with_retry(hook, event)

        # Store execution
        self._executions[execution.execution_id] = execution
        self._execution_history.append(execution)

        # Update stats
        self._update_statistics(execution)

        return execution

    async def _execute_hook_async(
        self, hook: HookConfig, event: HookEvent
    ) -> HookExecution:
        """
        Execute hook asynchronously via Redis Streams.

        Creates a pending execution record and publishes to stream.
        """
        # Create pending execution record
        execution = HookExecution(
            hook_id=hook.hook_id,
            trigger=event.trigger,
            status=HookStatus.PENDING,
            started_at=datetime.now(UTC),
            input_data=event.data,
            workflow_id=event.workflow_id,
            task_id=event.task_id,
            session_id=event.session_id,
        )

        # Store execution
        self._executions[execution.execution_id] = execution

        # Publish to Redis Streams if available
        if self._stream_producer:
            try:
                message_data = {
                    "execution_id": str(execution.execution_id),
                    "hook_id": str(hook.hook_id),
                    "event": event.model_dump(mode="json"),
                }

                await self._stream_producer.publish_event(
                    stream_name="hook_executions",
                    event_type="hook.execute",
                    data=message_data,
                )

                self.logger.info(
                    "Hook execution queued",
                    execution_id=str(execution.execution_id),
                    hook_name=hook.name,
                )

            except Exception as e:
                execution.mark_failed(f"Failed to queue execution: {str(e)}")
                self.logger.error(
                    "Failed to queue hook execution",
                    execution_id=str(execution.execution_id),
                    error=str(e),
                )
        else:
            # No stream producer, execute immediately
            self.logger.warning(
                "No stream producer configured, executing hook immediately"
            )
            execution = await self._execute_hook_sync(hook, event)

        return execution

    async def _execute_hook_fire_and_forget(
        self, hook: HookConfig, event: HookEvent
    ) -> None:
        """Execute hook without tracking result."""
        try:
            await self._executor.execute_hook(hook, event)
        except Exception as e:
            self.logger.error(
                "Fire-and-forget hook failed",
                hook_name=hook.name,
                error=str(e),
            )

    # ==================== Execution Management ====================

    def get_execution(self, execution_id: UUID) -> HookExecution | None:
        """Get execution by ID."""
        return self._executions.get(execution_id)

    def get_execution_history(
        self,
        hook_id: UUID | None = None,
        trigger: HookTrigger | None = None,
        status: HookStatus | None = None,
        limit: int = 100,
    ) -> list[HookExecution]:
        """
        Get execution history with filters.

        Args:
            hook_id: Filter by hook ID
            trigger: Filter by trigger type
            status: Filter by execution status
            limit: Maximum number of executions to return

        Returns:
            List of executions
        """
        executions = list(self._execution_history)

        if hook_id:
            executions = [e for e in executions if e.hook_id == hook_id]

        if trigger:
            executions = [e for e in executions if e.trigger == trigger]

        if status:
            executions = [e for e in executions if e.status == status]

        # Sort by started_at descending (most recent first)
        executions.sort(key=lambda e: e.started_at, reverse=True)

        return executions[:limit]

    def update_execution(self, execution: HookExecution) -> None:
        """Update execution status (for async executions)."""
        if execution.execution_id in self._executions:
            self._executions[execution.execution_id] = execution
            self._update_statistics(execution)

    def _update_statistics(self, execution: HookExecution) -> None:
        """Update execution statistics."""
        if execution.status not in (HookStatus.COMPLETED, HookStatus.FAILED):
            return

        self._stats["total_executions"] += 1

        if execution.status == HookStatus.COMPLETED:
            self._stats["successful_executions"] += 1
        else:
            self._stats["failed_executions"] += 1

        # Update average execution time
        if execution.duration_ms:
            total_time = (
                self._stats["avg_execution_time_ms"]
                * (self._stats["total_executions"] - 1)
            )
            self._stats["avg_execution_time_ms"] = (
                total_time + execution.duration_ms
            ) / self._stats["total_executions"]

    # ==================== Cleanup & Statistics ====================

    async def cleanup_old_executions(self, retention_days: int = 7) -> int:
        """Remove old execution records."""
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)

        old_executions = [
            e for e in self._execution_history if e.started_at < cutoff_date
        ]

        for execution in old_executions:
            self._execution_history.remove(execution)
            if execution.execution_id in self._executions:
                del self._executions[execution.execution_id]

        if old_executions:
            self.logger.info(
                "Old hook executions cleaned up", count=len(old_executions)
            )

        return len(old_executions)

    def get_statistics(self) -> dict[str, Any]:
        """Get hook system statistics."""
        return {
            **self._stats,
            "hooks_by_trigger": {
                trigger.value: len(hooks)
                for trigger, hooks in self._hooks_by_trigger.items()
            },
            "execution_history_size": len(self._execution_history),
        }


# Global hook manager instance
hook_manager = HookManager()
