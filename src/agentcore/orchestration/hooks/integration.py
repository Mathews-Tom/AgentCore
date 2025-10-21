"""
Hook Integration with A2A Event System

Integrates workflow hooks with the A2A-007 Event System for automated triggering.
"""

from __future__ import annotations

import structlog

from agentcore.a2a_protocol.models.events import Event, EventType
from agentcore.a2a_protocol.services.event_manager import event_manager
from agentcore.orchestration.hooks.manager import hook_manager
from agentcore.orchestration.hooks.models import HookEvent, HookTrigger

logger = structlog.get_logger()


class HookEventIntegration:
    """
    Integrates hooks with A2A Event System.

    Maps A2A events to hook triggers and executes matching hooks.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # Mapping of A2A event types to hook triggers
        self._event_to_hook_mapping: dict[EventType, HookTrigger] = {
            # Task events → hook triggers
            EventType.TASK_CREATED: HookTrigger.PRE_TASK,
            EventType.TASK_STARTED: HookTrigger.PRE_TASK,
            EventType.TASK_COMPLETED: HookTrigger.POST_TASK,
            EventType.TASK_FAILED: HookTrigger.POST_TASK,
            # System events → session hooks
            EventType.SYSTEM_STARTUP: HookTrigger.SESSION_START,
            EventType.SYSTEM_SHUTDOWN: HookTrigger.SESSION_END,
        }

    def register_event_hooks(self) -> None:
        """Register hooks with the A2A event system."""
        for event_type in self._event_to_hook_mapping.keys():
            event_manager.register_hook(event_type, self._handle_event)

        self.logger.info(
            "Hook integration registered with event system",
            event_types=[et.value for et in self._event_to_hook_mapping.keys()],
        )

    async def _handle_event(self, event: Event) -> None:
        """
        Handle A2A event and trigger matching hooks.

        Args:
            event: A2A event to process
        """
        # Map A2A event type to hook trigger
        hook_trigger = self._event_to_hook_mapping.get(event.event_type)
        if not hook_trigger:
            return

        # Convert A2A event to HookEvent
        hook_event = self._convert_to_hook_event(event, hook_trigger)

        # Trigger hooks
        try:
            executions = await hook_manager.trigger_hooks(hook_event)

            self.logger.info(
                "A2A event triggered hooks",
                event_type=event.event_type.value,
                hook_trigger=hook_trigger.value,
                executions_count=len(executions),
            )

        except Exception as e:
            self.logger.error(
                "Failed to trigger hooks for A2A event",
                event_type=event.event_type.value,
                error=str(e),
            )

    def _convert_to_hook_event(self, event: Event, trigger: HookTrigger) -> HookEvent:
        """
        Convert A2A Event to HookEvent.

        Args:
            event: A2A event
            trigger: Hook trigger type

        Returns:
            HookEvent
        """
        # Extract context from event data
        workflow_id = event.data.get("workflow_id")
        task_id = event.data.get("task_id")
        session_id = event.metadata.get("session_id")

        return HookEvent(
            event_id=event.event_id,
            trigger=trigger,
            timestamp=event.timestamp,
            source=event.source,
            data=event.data,
            metadata=event.metadata,
            workflow_id=workflow_id,
            task_id=task_id,
            session_id=session_id,
        )

    def trigger_custom_hook(
        self, trigger: HookTrigger, source: str, data: dict, **context
    ) -> None:
        """
        Trigger hooks directly without A2A event.

        Args:
            trigger: Hook trigger type
            source: Event source
            data: Event data
            **context: Additional context (workflow_id, task_id, session_id)
        """
        import asyncio

        hook_event = HookEvent(
            trigger=trigger,
            source=source,
            data=data,
            workflow_id=context.get("workflow_id"),
            task_id=context.get("task_id"),
            session_id=context.get("session_id"),
        )

        # Run async in event loop
        asyncio.create_task(hook_manager.trigger_hooks(hook_event))


# Global integration instance
hook_integration = HookEventIntegration()


def initialize_hook_integration() -> None:
    """Initialize hook integration with event system."""
    hook_integration.register_event_hooks()
    logger.info("Hook integration initialized")
