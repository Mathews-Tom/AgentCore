"""
Workflow Hooks System

Event-driven hooks for automated workflow enhancement.
"""

from agentcore.orchestration.hooks.models import (
    HookConfig,
    HookEvent,
    HookExecution,
    HookExecutionMode,
    HookExecutionResult,
    HookRegistrationRequest,
    HookRegistrationResponse,
    HookStatus,
    HookTrigger,
)
from agentcore.orchestration.hooks.manager import hook_manager
from agentcore.orchestration.hooks.executor import HookExecutor

__all__ = [
    # Models
    "HookConfig",
    "HookEvent",
    "HookExecution",
    "HookExecutionMode",
    "HookExecutionResult",
    "HookRegistrationRequest",
    "HookRegistrationResponse",
    "HookStatus",
    "HookTrigger",
    # Manager
    "hook_manager",
    # Executor
    "HookExecutor",
]
