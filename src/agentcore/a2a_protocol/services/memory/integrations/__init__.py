"""
Memory Service Integrations

Provides integration modules for memory service with core AgentCore services.

Components:
- session_manager: SessionManager memory integration
- message_router: MessageRouter memory-aware routing
- task_manager: TaskManager artifact storage integration
- ace_interface: ACE strategic context interface
"""

from __future__ import annotations

from agentcore.a2a_protocol.services.memory.integrations.ace_interface import (
    ACEStrategicContextInterface,
    ACEStrategicContext,
)
from agentcore.a2a_protocol.services.memory.integrations.message_router import (
    MemoryAwareRouter,
    RoutingMemoryInsight,
)
from agentcore.a2a_protocol.services.memory.integrations.session_manager import (
    SessionContextProvider,
    SessionMemoryContext,
)
from agentcore.a2a_protocol.services.memory.integrations.task_manager import (
    ArtifactMemoryStorage,
    ArtifactMemoryRecord,
)

__all__ = [
    "SessionContextProvider",
    "SessionMemoryContext",
    "MemoryAwareRouter",
    "RoutingMemoryInsight",
    "ArtifactMemoryStorage",
    "ArtifactMemoryRecord",
    "ACEStrategicContextInterface",
    "ACEStrategicContext",
]
