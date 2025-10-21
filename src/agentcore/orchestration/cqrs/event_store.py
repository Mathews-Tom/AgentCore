"""
Event Store Module

Event persistence and retrieval for event sourcing.
Provides append-only event storage with snapshot support.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    and_,
    desc,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

from agentcore.a2a_protocol.database.connection import Base
from agentcore.orchestration.cqrs.events import DomainEvent, deserialize_event


class EventRecord(Base):
    """
    PostgreSQL table for event storage.

    Events are append-only and immutable.
    """

    __tablename__ = "orchestration_events"

    # Primary key is event_id
    event_id = Column(String(36), primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)
    aggregate_id = Column(String(36), nullable=False, index=True)
    aggregate_type = Column(String(50), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    event_data = Column(Text, nullable=False)  # JSON serialized event
    event_metadata = Column(Text, nullable=True)  # JSON serialized metadata

    __table_args__ = (
        Index("idx_aggregate_version", "aggregate_id", "version", unique=True),
        Index("idx_event_timestamp", "timestamp"),
        Index("idx_event_type_timestamp", "event_type", "timestamp"),
    )


class SnapshotRecord(Base):
    """
    PostgreSQL table for aggregate snapshots.

    Snapshots optimize event replay for large event streams.
    """

    __tablename__ = "orchestration_snapshots"

    # Composite primary key: aggregate_id + version
    aggregate_id = Column(String(36), primary_key=True)
    version = Column(Integer, primary_key=True)
    aggregate_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    snapshot_data = Column(Text, nullable=False)  # JSON serialized state
    snapshot_metadata = Column(Text, nullable=True)

    __table_args__ = (Index("idx_snapshot_timestamp", "timestamp"),)


class EventStreamResult(BaseModel):
    """Result of event stream query."""

    aggregate_id: UUID
    aggregate_type: str
    events: list[DomainEvent]
    current_version: int
    total_events: int


class SnapshotData(BaseModel):
    """Snapshot data model."""

    aggregate_id: UUID
    aggregate_type: str
    version: int
    timestamp: datetime
    state: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventStore(ABC):
    """
    Abstract event store interface.

    Defines contract for event persistence and retrieval.
    """

    @abstractmethod
    async def append_event(self, event: DomainEvent) -> None:
        """
        Append event to store.

        Args:
            event: Event to append

        Raises:
            ConcurrencyError: If version conflict occurs
        """
        pass

    @abstractmethod
    async def append_events(self, events: list[DomainEvent]) -> None:
        """
        Append multiple events atomically.

        Args:
            events: Events to append

        Raises:
            ConcurrencyError: If version conflict occurs
        """
        pass

    @abstractmethod
    async def get_events(
        self,
        aggregate_id: UUID,
        from_version: int = 0,
        to_version: int | None = None,
    ) -> EventStreamResult:
        """
        Get events for an aggregate.

        Args:
            aggregate_id: Aggregate identifier
            from_version: Starting version (inclusive)
            to_version: Ending version (inclusive), None for all

        Returns:
            Event stream result
        """
        pass

    @abstractmethod
    async def get_events_by_type(
        self,
        event_type: str,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        limit: int = 100,
    ) -> list[DomainEvent]:
        """
        Get events by type within time range.

        Args:
            event_type: Event type to filter
            from_timestamp: Start timestamp (inclusive)
            to_timestamp: End timestamp (inclusive)
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        pass

    @abstractmethod
    async def save_snapshot(self, snapshot: SnapshotData) -> None:
        """
        Save aggregate snapshot.

        Args:
            snapshot: Snapshot data
        """
        pass

    @abstractmethod
    async def get_snapshot(self, aggregate_id: UUID) -> SnapshotData | None:
        """
        Get latest snapshot for aggregate.

        Args:
            aggregate_id: Aggregate identifier

        Returns:
            Snapshot data or None if no snapshot exists
        """
        pass


class PostgreSQLEventStore(EventStore):
    """
    PostgreSQL-based event store implementation.

    Uses PostgreSQL for durable event storage with ACID guarantees.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize event store.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def append_event(self, event: DomainEvent) -> None:
        """Append single event to store."""
        record = EventRecord(
            event_id=str(event.event_id),
            event_type=event.event_type.value,
            aggregate_id=str(event.aggregate_id),
            aggregate_type=event.aggregate_type,
            version=event.version,
            timestamp=event.timestamp,
            event_data=event.model_dump_json(),
            event_metadata=json.dumps(event.metadata) if event.metadata else None,
        )

        self.session.add(record)
        await self.session.flush()

    async def append_events(self, events: list[DomainEvent]) -> None:
        """Append multiple events atomically."""
        records = [
            EventRecord(
                event_id=str(event.event_id),
                event_type=event.event_type.value,
                aggregate_id=str(event.aggregate_id),
                aggregate_type=event.aggregate_type,
                version=event.version,
                timestamp=event.timestamp,
                event_data=event.model_dump_json(),
                event_metadata=json.dumps(event.metadata) if event.metadata else None,
            )
            for event in events
        ]

        self.session.add_all(records)
        await self.session.flush()

    async def get_events(
        self,
        aggregate_id: UUID,
        from_version: int = 0,
        to_version: int | None = None,
    ) -> EventStreamResult:
        """Get events for aggregate."""
        query = select(EventRecord).where(
            and_(
                EventRecord.aggregate_id == str(aggregate_id),
                EventRecord.version >= from_version,
            )
        )

        if to_version is not None:
            query = query.where(EventRecord.version <= to_version)

        query = query.order_by(EventRecord.version.asc())

        result = await self.session.execute(query)
        records = result.scalars().all()

        events = []
        aggregate_type = ""
        max_version = 0

        for record in records:
            event_data = json.loads(record.event_data)
            event = deserialize_event(event_data)
            events.append(event)
            aggregate_type = record.aggregate_type
            max_version = max(max_version, record.version)

        return EventStreamResult(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            events=events,
            current_version=max_version,
            total_events=len(events),
        )

    async def get_events_by_type(
        self,
        event_type: str,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        limit: int = 100,
    ) -> list[DomainEvent]:
        """Get events by type within time range."""
        query = select(EventRecord).where(EventRecord.event_type == event_type)

        if from_timestamp:
            query = query.where(EventRecord.timestamp >= from_timestamp)

        if to_timestamp:
            query = query.where(EventRecord.timestamp <= to_timestamp)

        query = query.order_by(EventRecord.timestamp.desc()).limit(limit)

        result = await self.session.execute(query)
        records = result.scalars().all()

        events = []
        for record in records:
            event_data = json.loads(record.event_data)
            event = deserialize_event(event_data)
            events.append(event)

        return events

    async def save_snapshot(self, snapshot: SnapshotData) -> None:
        """Save aggregate snapshot."""
        record = SnapshotRecord(
            aggregate_id=str(snapshot.aggregate_id),
            version=snapshot.version,
            aggregate_type=snapshot.aggregate_type,
            timestamp=snapshot.timestamp,
            snapshot_data=json.dumps(snapshot.state),
            snapshot_metadata=json.dumps(snapshot.metadata)
            if snapshot.metadata
            else None,
        )

        # Use merge to handle upsert
        await self.session.merge(record)
        await self.session.flush()

    async def get_snapshot(self, aggregate_id: UUID) -> SnapshotData | None:
        """Get latest snapshot for aggregate."""
        query = (
            select(SnapshotRecord)
            .where(SnapshotRecord.aggregate_id == str(aggregate_id))
            .order_by(desc(SnapshotRecord.version))
            .limit(1)
        )

        result = await self.session.execute(query)
        record = result.scalar_one_or_none()

        if not record:
            return None

        return SnapshotData(
            aggregate_id=UUID(record.aggregate_id),
            aggregate_type=record.aggregate_type,
            version=record.version,
            timestamp=record.timestamp,
            state=json.loads(record.snapshot_data),
            metadata=json.loads(record.snapshot_metadata)
            if record.snapshot_metadata
            else {},
        )


class ConcurrencyError(Exception):
    """Raised when event version conflict occurs."""

    pass
