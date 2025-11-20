"""ECL Pipeline package for Extract, Cognify, Load architecture.

This package implements the modular, task-based ECL pipeline inspired by Cognee's
architecture. It provides:

- Base classes for composable tasks (TaskBase, Pipeline)
- Phase-specific implementations (Extract, Cognify, Load)
- Task registry for dynamic task discovery
- Async execution with dependency resolution
- Error handling and retry logic

Components:
    - task_base: Abstract base class for all pipeline tasks
    - pipeline: Pipeline orchestration and task composition
    - extract: Extract phase for data source ingestion
    - cognify: Cognify phase for knowledge extraction
    - load: Load phase for multi-backend storage

References:
    - FR-9: ECL Pipeline Architecture (spec.md)
    - MEM-010: ECL Pipeline Base Classes ticket
"""

from __future__ import annotations

from agentcore.a2a_protocol.services.memory.pipeline.pipeline import (
    Pipeline,
    PipelineResult,
)
from agentcore.a2a_protocol.services.memory.pipeline.task_base import (
    RetryStrategy,
    TaskBase,
    TaskResult,
    TaskStatus,
)

# Phase implementations
from agentcore.a2a_protocol.services.memory.pipeline.cognify import CognifyTask
from agentcore.a2a_protocol.services.memory.pipeline.extract import ExtractTask
from agentcore.a2a_protocol.services.memory.pipeline.load import LoadTask

__all__ = [
    # Base classes
    "TaskBase",
    "Pipeline",
    "TaskResult",
    "TaskStatus",
    "RetryStrategy",
    "PipelineResult",
    # Phase tasks
    "ExtractTask",
    "CognifyTask",
    "LoadTask",
]
