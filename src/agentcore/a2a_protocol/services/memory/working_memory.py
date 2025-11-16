"""
Redis Working Memory Service

Implements TTL-based caching for immediate context (working memory layer).
Working memory provides fast access to recent context with automatic expiration.

Features:
- Session-scoped working memory with 1-hour default TTL
- Agent-scoped working memory for cross-session context
- Automatic key expiration and cleanup
- JSON serialization of memory records
- Atomic operations for consistency

Component ID: MEM-030
Ticket: MEM-030 (Implement Redis Working Memory Service)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog
from redis.asyncio import Redis, from_url

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord

logger = structlog.get_logger(__name__)

# Default TTL values (in seconds)
DEFAULT_WORKING_MEMORY_TTL = 3600  # 1 hour
SESSION_CONTEXT_TTL = 7200  # 2 hours
AGENT_CONTEXT_TTL = 86400  # 24 hours


class WorkingMemoryService:
    """
    Redis-based working memory service for immediate context storage.

    Provides fast access to recent memories with automatic TTL expiration.
    Supports session-scoped and agent-scoped working memory.

    Usage:
        service = WorkingMemoryService()
        await service.initialize()

        # Store working memory
        await service.store_working_memory(memory_record, ttl=3600)

        # Get session context
        memories = await service.get_session_working_memory(session_id, limit=10)

        # Clear session
        await service.clear_session(session_id)

        await service.close()
    """

    def __init__(self) -> None:
        """Initialize working memory service (connection not yet established)."""
        self._redis: Redis | None = None
        self._initialized = False
        self._logger = logger.bind(component="working_memory")

    @property
    def redis(self) -> Redis:
        """Get Redis client (raises if not initialized)."""
        if self._redis is None:
            raise RuntimeError("WorkingMemoryService not initialized. Call initialize() first.")
        return self._redis

    async def initialize(self) -> None:
        """
        Initialize Redis connection.

        Raises:
            RuntimeError: If already initialized
            ConnectionError: If connection fails
        """
        if self._initialized:
            self._logger.warning("WorkingMemoryService already initialized")
            return

        self._logger.info(
            "initializing_working_memory_service",
            redis_url=settings.REDIS_URL.split("@")[-1],  # Hide credentials
        )

        # Create Redis connection
        self._redis = from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )

        # Verify connection
        await self._redis.ping()

        self._initialized = True
        self._logger.info("working_memory_service_initialized")

    async def close(self) -> None:
        """Close Redis connection."""
        if not self._initialized:
            return

        self._logger.info("closing_working_memory_service")

        if self._redis:
            await self._redis.close()
            self._redis = None

        self._initialized = False
        self._logger.info("working_memory_service_closed")

    def _memory_to_dict(self, memory: MemoryRecord) -> dict[str, Any]:
        """Convert MemoryRecord to JSON-serializable dict."""
        data = memory.model_dump(mode="json")
        # Convert enums to strings
        data["memory_layer"] = memory.memory_layer.value
        # Convert datetime to ISO string
        data["timestamp"] = memory.timestamp.isoformat()
        if memory.last_accessed:
            data["last_accessed"] = memory.last_accessed.isoformat()
        return data

    def _dict_to_memory(self, data: dict[str, Any]) -> MemoryRecord:
        """Convert dict back to MemoryRecord."""
        # Convert string back to enum
        if isinstance(data.get("memory_layer"), str):
            data["memory_layer"] = MemoryLayer(data["memory_layer"])
        # Parse datetime strings
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        if isinstance(data.get("last_accessed"), str) and data["last_accessed"]:
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"].replace("Z", "+00:00"))
        return MemoryRecord(**data)

    async def store_working_memory(
        self,
        memory: MemoryRecord,
        ttl: int = DEFAULT_WORKING_MEMORY_TTL,
    ) -> str:
        """
        Store memory in working memory cache.

        Args:
            memory: MemoryRecord to store
            ttl: Time-to-live in seconds (default: 1 hour)

        Returns:
            str: Memory ID stored

        Keys created:
        - memory:{memory_id} - Full memory record
        - session:{session_id}:memories - Sorted set of memory IDs by timestamp
        - agent:{agent_id}:memories - Sorted set of memory IDs by timestamp
        """
        # Store the full memory record
        memory_key = f"memory:{memory.memory_id}"
        memory_data = self._memory_to_dict(memory)
        await self.redis.setex(
            memory_key,
            ttl,
            json.dumps(memory_data),
        )

        # Add to session index if session_id exists
        if memory.session_id:
            session_key = f"session:{memory.session_id}:memories"
            score = memory.timestamp.timestamp()
            await self.redis.zadd(session_key, {memory.memory_id: score})
            await self.redis.expire(session_key, SESSION_CONTEXT_TTL)

        # Add to agent index
        if memory.agent_id:
            agent_key = f"agent:{memory.agent_id}:memories"
            score = memory.timestamp.timestamp()
            await self.redis.zadd(agent_key, {memory.memory_id: score})
            await self.redis.expire(agent_key, AGENT_CONTEXT_TTL)

        self._logger.info(
            "working_memory_stored",
            memory_id=memory.memory_id,
            session_id=memory.session_id,
            agent_id=memory.agent_id,
            ttl=ttl,
        )

        return memory.memory_id

    async def get_working_memory(self, memory_id: str) -> MemoryRecord | None:
        """
        Retrieve single memory from working memory.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            MemoryRecord or None if not found/expired
        """
        memory_key = f"memory:{memory_id}"
        data = await self.redis.get(memory_key)

        if data:
            memory_dict = json.loads(data)
            return self._dict_to_memory(memory_dict)

        return None

    async def get_session_working_memory(
        self,
        session_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """
        Get recent working memories for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of memories to return
            offset: Number of memories to skip

        Returns:
            List of MemoryRecords (most recent first)
        """
        session_key = f"session:{session_id}:memories"

        # Get memory IDs sorted by timestamp (descending)
        memory_ids = await self.redis.zrevrange(
            session_key,
            offset,
            offset + limit - 1,
        )

        if not memory_ids:
            return []

        # Fetch all memories in batch
        memories = []
        for memory_id in memory_ids:
            memory = await self.get_working_memory(memory_id)
            if memory:
                memories.append(memory)
            else:
                # Memory expired, remove from index
                await self.redis.zrem(session_key, memory_id)

        self._logger.info(
            "session_working_memory_retrieved",
            session_id=session_id,
            count=len(memories),
            limit=limit,
        )

        return memories

    async def get_agent_working_memory(
        self,
        agent_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """
        Get recent working memories for an agent (across sessions).

        Args:
            agent_id: Agent identifier
            limit: Maximum number of memories to return
            offset: Number of memories to skip

        Returns:
            List of MemoryRecords (most recent first)
        """
        agent_key = f"agent:{agent_id}:memories"

        # Get memory IDs sorted by timestamp (descending)
        memory_ids = await self.redis.zrevrange(
            agent_key,
            offset,
            offset + limit - 1,
        )

        if not memory_ids:
            return []

        # Fetch all memories in batch
        memories = []
        for memory_id in memory_ids:
            memory = await self.get_working_memory(memory_id)
            if memory:
                memories.append(memory)
            else:
                # Memory expired, remove from index
                await self.redis.zrem(agent_key, memory_id)

        self._logger.info(
            "agent_working_memory_retrieved",
            agent_id=agent_id,
            count=len(memories),
            limit=limit,
        )

        return memories

    async def extend_ttl(
        self,
        memory_id: str,
        ttl: int = DEFAULT_WORKING_MEMORY_TTL,
    ) -> bool:
        """
        Extend TTL for a memory (keep it in working memory longer).

        Args:
            memory_id: Memory ID
            ttl: New TTL in seconds

        Returns:
            True if extended, False if memory not found
        """
        memory_key = f"memory:{memory_id}"
        result = await self.redis.expire(memory_key, ttl)

        if result:
            self._logger.info(
                "working_memory_ttl_extended",
                memory_id=memory_id,
                new_ttl=ttl,
            )

        return bool(result)

    async def clear_session(self, session_id: str) -> int:
        """
        Clear all working memories for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of memories cleared
        """
        session_key = f"session:{session_id}:memories"

        # Get all memory IDs
        memory_ids = await self.redis.zrange(session_key, 0, -1)

        if not memory_ids:
            return 0

        # Delete all memory records
        memory_keys = [f"memory:{mid}" for mid in memory_ids]
        deleted = await self.redis.delete(*memory_keys, session_key)

        self._logger.info(
            "session_working_memory_cleared",
            session_id=session_id,
            memories_cleared=len(memory_ids),
        )

        return len(memory_ids)

    async def get_working_memory_stats(self) -> dict[str, Any]:
        """
        Get statistics about working memory usage.

        Returns:
            Dictionary with stats (total_memories, memory_by_layer, etc.)
        """
        info = await self.redis.info("memory")
        keys_info = await self.redis.info("keyspace")

        stats = {
            "redis_memory_used_bytes": info.get("used_memory", 0),
            "redis_memory_peak_bytes": info.get("used_memory_peak", 0),
            "redis_memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
            "total_keys": 0,
            "session_keys": 0,
            "agent_keys": 0,
            "memory_keys": 0,
        }

        # Count keys by pattern
        async for key in self.redis.scan_iter("memory:*", count=100):
            stats["memory_keys"] += 1
            stats["total_keys"] += 1

        async for key in self.redis.scan_iter("session:*:memories", count=100):
            stats["session_keys"] += 1
            stats["total_keys"] += 1

        async for key in self.redis.scan_iter("agent:*:memories", count=100):
            stats["agent_keys"] += 1
            stats["total_keys"] += 1

        self._logger.info(
            "working_memory_stats_retrieved",
            stats=stats,
        )

        return stats

    async def health_check(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized or not self._redis:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception as e:
            self._logger.error("redis_health_check_failed", error=str(e))
            return False


# Global singleton instance
_working_memory: WorkingMemoryService | None = None


def get_working_memory_service() -> WorkingMemoryService:
    """Get or create global working memory service instance."""
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemoryService()
    return _working_memory


async def initialize_working_memory() -> None:
    """Initialize the global working memory service (call during app startup)."""
    service = get_working_memory_service()
    await service.initialize()


async def close_working_memory() -> None:
    """Close the global working memory service (call during app shutdown)."""
    global _working_memory
    if _working_memory:
        await _working_memory.close()
        _working_memory = None


__all__ = [
    "WorkingMemoryService",
    "get_working_memory_service",
    "initialize_working_memory",
    "close_working_memory",
    "DEFAULT_WORKING_MEMORY_TTL",
    "SESSION_CONTEXT_TTL",
    "AGENT_CONTEXT_TTL",
]
