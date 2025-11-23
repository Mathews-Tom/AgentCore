"""Extract phase tasks for ECL Pipeline.

This module provides Extract phase base classes for data source extraction:
- ExtractTask: Base class for all extraction tasks
- Support for conversation logs, tool outputs, task artifacts
- Async streaming support
- Multiple data source types

References:
    - FR-9.1: Extract Phase (Data Ingestion)
    - MEM-010: ECL Pipeline Base Classes
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from agentcore.a2a_protocol.models.memory import MemoryLayer
from agentcore.a2a_protocol.services.memory.pipeline.task_base import (
    RetryStrategy,
    TaskBase,
)

logger = logging.getLogger(__name__)


class ExtractTask(TaskBase):
    """Base class for Extract phase tasks.

    The Extract phase is responsible for ingesting data from various sources:
    - Agent interactions (messages, tool calls, responses)
    - Session context (conversation history)
    - Task artifacts (files, outputs, results)
    - Error records and recovery actions

    Subclasses should implement the execute() method to handle specific
    data source extraction logic.

    Example:
        ```python
        class ConversationExtractor(ExtractTask):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                session_id = input_data["session_id"]
                messages = await self.fetch_messages(session_id)
                return {
                    "messages": messages,
                    "message_count": len(messages),
                    "source_type": "conversation"
                }
        ```

    Attributes:
        source_type: Type of data source being extracted
        streaming_enabled: Enable async streaming for large datasets
        batch_size: Number of items to process per batch
    """

    def __init__(
        self,
        name: str = "extract_task",
        description: str = "Extract data from source",
        source_type: str = "generic",
        streaming_enabled: bool = False,
        batch_size: int = 100,
        **kwargs: Any,
    ):
        """Initialize extract task.

        Args:
            name: Task name
            description: Task description
            source_type: Type of data source (conversation, artifact, error, etc.)
            streaming_enabled: Enable async streaming for large datasets
            batch_size: Number of items to process per batch
            **kwargs: Additional TaskBase arguments
        """
        # Set defaults if not provided
        task_kwargs = {
            "dependencies": [],
            "retry_strategy": RetryStrategy.EXPONENTIAL,
            "max_retries": 3,
            "retry_delay_ms": 1000,
        }
        task_kwargs.update(kwargs)

        super().__init__(name=name, description=description, **task_kwargs)

        self.source_type = source_type
        self.streaming_enabled = streaming_enabled
        self.batch_size = batch_size

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute data extraction.

        This method should be overridden by subclasses to implement
        specific extraction logic.

        Args:
            input_data: Dictionary containing:
                - source: str - Source identifier (session_id, task_id, etc.)
                - filters: dict (optional) - Filters for data extraction
                - limit: int (optional) - Maximum items to extract
                - offset: int (optional) - Offset for pagination

        Returns:
            Dictionary containing:
                - data: list - Extracted data items
                - count: int - Number of items extracted
                - source_type: str - Type of source
                - metadata: dict - Additional extraction metadata

        Raises:
            ValueError: If required input parameters missing
            RuntimeError: If extraction fails
        """
        # Default implementation - subclasses should override
        return {
            "data": [],
            "count": 0,
            "source_type": self.source_type,
            "metadata": {"batch_size": self.batch_size},
        }

    async def stream(
        self, input_data: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream data extraction for large datasets.

        Yields data in batches to avoid memory issues.

        Args:
            input_data: Same as execute()

        Yields:
            Dictionary containing batch of data items

        Raises:
            NotImplementedError: If streaming not supported by subclass
        """
        if not self.streaming_enabled:
            raise NotImplementedError(
                f"Streaming not enabled for task {self.name}"
            )

        # Default implementation - subclasses should override
        # This is just a placeholder
        yield {
            "data": [],
            "batch_index": 0,
            "is_last_batch": True,
        }

    async def extract_agent_interaction(
        self, agent_interaction_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract agent interaction data into memory format.

        Args:
            agent_interaction_data: Dictionary containing agent interaction details

        Returns:
            Extracted memory with standardized format
        """
        agent_id = agent_interaction_data.get("agent_id", "unknown")
        action = agent_interaction_data.get("action", "unknown")
        input_text = agent_interaction_data.get("input", "")
        output_text = agent_interaction_data.get("output", "")
        timestamp = agent_interaction_data.get("timestamp", "")
        metadata = agent_interaction_data.get("metadata", {})

        # Format content from interaction
        content = f"Agent {agent_id} performed {action}. "
        if input_text:
            content += f"Input: {input_text}. "
        if output_text:
            content += f"Output: {output_text}"

        return {
            "agent_id": agent_id,
            "content": content,
            "memory_layer": MemoryLayer.EPISODIC.value,
            "timestamp": timestamp,
            "metadata": metadata,
        }

    async def extract_session_context(
        self, session_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract session context data into memory format.

        Args:
            session_data: Dictionary containing session context details

        Returns:
            Extracted memory with standardized format
        """
        session_id = session_data.get("session_id", "unknown")
        user_id = session_data.get("user_id", "")
        preferences = session_data.get("preferences", {})
        auth_level = session_data.get("authentication_level", "")
        start_time = session_data.get("start_time", "")

        # Format content from session
        content = f"Session {session_id}"
        if user_id:
            content += f" for user {user_id}"
        if auth_level:
            content += f" with {auth_level} access"
        if preferences:
            pref_str = ", ".join(f"{k}: {v}" for k, v in preferences.items())
            content += f". Preferences: {pref_str}"

        return {
            "session_id": session_id,
            "content": content,
            "memory_layer": MemoryLayer.SEMANTIC.value,
            "timestamp": start_time,
            "metadata": {
                "user_id": user_id,
                "preferences": preferences,
                "authentication_level": auth_level,
            },
        }

    async def extract_task_artifact(
        self, artifact_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract task artifact data into memory format.

        Args:
            artifact_data: Dictionary containing task artifact details

        Returns:
            Extracted memory with standardized format
        """
        task_id = artifact_data.get("task_id", "unknown")
        artifact_type = artifact_data.get("artifact_type", "file")
        artifact_name = artifact_data.get("artifact_name", "")
        artifact_content = artifact_data.get("artifact_content", "")
        timestamp = artifact_data.get("timestamp", "")
        metadata = artifact_data.get("metadata", {})

        # Format content from artifact
        content = f"Task {task_id} produced {artifact_type}"
        if artifact_name:
            content += f" '{artifact_name}'"
        if artifact_content:
            content += f". Content: {artifact_content}"

        return {
            "task_id": task_id,
            "content": content,
            "memory_layer": MemoryLayer.PROCEDURAL.value,
            "timestamp": timestamp,
            "metadata": {
                **metadata,
                "artifact_type": artifact_type,
                "artifact_name": artifact_name,
            },
        }

    async def extract_error_record(
        self, error_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract error record data into memory format.

        Args:
            error_data: Dictionary containing error record details

        Returns:
            Extracted memory with standardized format
        """
        error_type = error_data.get("error_type", "unknown")
        error_message = error_data.get("error_message", "")
        context = error_data.get("context", {})
        recovery_action = error_data.get("recovery_action", "")
        timestamp = error_data.get("timestamp", "")
        task_id = error_data.get("task_id", "")
        severity = error_data.get("severity", 0.0)

        # Format content from error
        content = f"Error occurred: {error_type}"
        if error_message:
            content += f". Message: {error_message}"
        if recovery_action:
            content += f". Recovery: {recovery_action}"
        if context:
            content += f". Context: {context}"
        if task_id:
            content += f" (Task: {task_id})"

        return {
            "content": content,
            "error_type": error_type,
            "severity": severity,
            "memory_layer": MemoryLayer.EPISODIC.value,
            "timestamp": timestamp,
            "metadata": {
                "error_type": error_type,
                "error_message": error_message,
                "context": context,
                "recovery_action": recovery_action,
                "task_id": task_id,
                "severity": severity,
            },
        }


class ConversationExtractor(ExtractTask):
    """Extract conversation logs from agent interactions.

    Extracts messages, tool calls, and responses from conversation history.

    Example:
        ```python
        extractor = ConversationExtractor()
        result = await extractor.execute({
            "session_id": "session-123",
            "limit": 100
        })
        messages = result["data"]
        ```
    """

    def __init__(self, **kwargs: Any):
        """Initialize conversation extractor."""
        super().__init__(
            name="conversation_extractor",
            description="Extract conversation logs from agent interactions",
            source_type="conversation",
            streaming_enabled=True,
            **kwargs,
        )

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Extract conversation messages.

        Args:
            input_data: Dictionary containing session_id and optional filters

        Returns:
            Extracted conversation data

        Raises:
            ValueError: If session_id not provided
        """
        # Handle pipeline input wrapping
        if "input" in input_data and isinstance(input_data["input"], dict):
            input_data = input_data["input"]

        session_id = input_data.get("session_id")
        if not session_id:
            raise ValueError("session_id required for conversation extraction")

        # TODO: Implement actual message fetching from SessionManager
        # For now, return placeholder structure
        logger.info(f"Extracting conversation data for session {session_id}")

        return {
            "data": [],  # Will be populated with actual messages
            "count": 0,
            "source_type": "conversation",
            "session_id": session_id,
            "metadata": {
                "batch_size": self.batch_size,
                "streaming_available": self.streaming_enabled,
            },
        }


class ArtifactExtractor(ExtractTask):
    """Extract task artifacts (files, outputs, results).

    Extracts artifacts associated with tasks for memory processing.

    Example:
        ```python
        extractor = ArtifactExtractor()
        result = await extractor.execute({
            "task_id": "task-123",
            "artifact_types": ["file", "output"]
        })
        artifacts = result["data"]
        ```
    """

    def __init__(self, **kwargs: Any):
        """Initialize artifact extractor."""
        super().__init__(
            name="artifact_extractor",
            description="Extract task artifacts (files, outputs, results)",
            source_type="artifact",
            **kwargs,
        )

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Extract task artifacts.

        Args:
            input_data: Dictionary containing task_id and optional artifact_types

        Returns:
            Extracted artifact data

        Raises:
            ValueError: If task_id not provided
        """
        # Handle pipeline input wrapping
        if "input" in input_data and isinstance(input_data["input"], dict):
            input_data = input_data["input"]

        task_id = input_data.get("task_id")
        if not task_id:
            raise ValueError("task_id required for artifact extraction")

        artifact_types = input_data.get("artifact_types", ["file", "output", "result"])

        # TODO: Implement actual artifact fetching from TaskManager
        logger.info(
            f"Extracting artifacts for task {task_id}, types: {artifact_types}"
        )

        return {
            "data": [],  # Will be populated with actual artifacts
            "count": 0,
            "source_type": "artifact",
            "task_id": task_id,
            "artifact_types": artifact_types,
            "metadata": {"batch_size": self.batch_size},
        }


class ErrorRecordExtractor(ExtractTask):
    """Extract error records and recovery actions.

    Extracts error information for pattern detection and learning.

    Example:
        ```python
        extractor = ErrorRecordExtractor()
        result = await extractor.execute({
            "task_id": "task-123",
            "min_severity": 0.5
        })
        errors = result["data"]
        ```
    """

    def __init__(self, **kwargs: Any):
        """Initialize error record extractor."""
        super().__init__(
            name="error_record_extractor",
            description="Extract error records and recovery actions",
            source_type="error",
            **kwargs,
        )

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Extract error records.

        Args:
            input_data: Dictionary containing task_id/stage_id and optional filters

        Returns:
            Extracted error data

        Raises:
            ValueError: If task_id not provided
        """
        # Handle pipeline input wrapping
        if "input" in input_data and isinstance(input_data["input"], dict):
            input_data = input_data["input"]

        task_id = input_data.get("task_id")
        if not task_id:
            raise ValueError("task_id required for error record extraction")

        min_severity = input_data.get("min_severity", 0.0)

        # TODO: Implement actual error fetching from ErrorTracker
        logger.info(
            f"Extracting error records for task {task_id}, "
            f"min_severity: {min_severity}"
        )

        return {
            "data": [],  # Will be populated with actual error records
            "count": 0,
            "source_type": "error",
            "task_id": task_id,
            "min_severity": min_severity,
            "metadata": {"batch_size": self.batch_size},
        }
