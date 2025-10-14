"""Audit logging service for sandbox security events."""

import asyncio
import json
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from ..models.sandbox import AuditEventType, AuditLogEntry

logger = structlog.get_logger()


class AuditLogger:
    """Service for logging and querying security audit events."""

    def __init__(
        self,
        log_directory: Path,
        max_logs_in_memory: int = 10000,
        retention_days: int = 90,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            log_directory: Directory for storing audit log files
            max_logs_in_memory: Maximum audit logs kept in memory
            retention_days: Number of days to retain logs
        """
        self._log_directory = log_directory
        self._max_logs = max_logs_in_memory
        self._retention_days = retention_days
        self._log_buffer: deque[AuditLogEntry] = deque(maxlen=max_logs_in_memory)
        self._write_queue: asyncio.Queue[AuditLogEntry] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._running = False

        # Ensure log directory exists
        self._log_directory.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start audit logger background tasks."""
        if self._running:
            return

        self._running = True
        self._writer_task = asyncio.create_task(self._log_writer())
        logger.info(
            "audit_logger_started",
            log_directory=str(self._log_directory),
            retention_days=self._retention_days,
        )

    async def stop(self) -> None:
        """Stop audit logger and flush pending logs."""
        if not self._running:
            return

        self._running = False

        # Drain write queue
        while not self._write_queue.empty():
            await asyncio.sleep(0.1)

        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass

        logger.info("audit_logger_stopped")

    async def log_event(self, entry: AuditLogEntry) -> None:
        """
        Log a security audit event.

        Args:
            entry: Audit log entry to record
        """
        # Add to memory buffer
        self._log_buffer.append(entry)

        # Queue for file writing
        await self._write_queue.put(entry)

        # Log critical events immediately
        if entry.event_type in {
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.LIMIT_EXCEEDED,
        }:
            logger.warning(
                "security_audit_event",
                event_type=entry.event_type.value,
                sandbox_id=entry.sandbox_id,
                agent_id=entry.agent_id,
                operation=entry.operation,
                resource=entry.resource,
                result=entry.result,
                reason=entry.reason,
            )

    async def query_logs(
        self,
        sandbox_id: str | None = None,
        agent_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        result: bool | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """
        Query audit logs with filters.

        Args:
            sandbox_id: Filter by sandbox ID
            agent_id: Filter by agent ID
            event_type: Filter by event type
            start_time: Filter logs after this time
            end_time: Filter logs before this time
            result: Filter by result (True/False)
            limit: Maximum number of logs to return

        Returns:
            List of matching audit log entries
        """
        results: list[AuditLogEntry] = []

        # Search in-memory buffer (most recent logs)
        for entry in reversed(self._log_buffer):
            if len(results) >= limit:
                break

            if sandbox_id and entry.sandbox_id != sandbox_id:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            if event_type and entry.event_type != event_type:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if result is not None and entry.result != result:
                continue

            results.append(entry)

        return results

    async def get_stats(
        self,
        sandbox_id: str | None = None,
        agent_id: str | None = None,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get audit statistics for a time period.

        Args:
            sandbox_id: Filter by sandbox ID
            agent_id: Filter by agent ID
            hours: Number of hours to analyze

        Returns:
            Dictionary with audit statistics
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        logs = await self.query_logs(
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            start_time=cutoff_time,
            limit=100000,
        )

        stats: dict[str, Any] = {
            "total_events": len(logs),
            "by_event_type": {},
            "by_result": {"allowed": 0, "denied": 0},
            "violations_count": 0,
            "period_hours": hours,
        }

        for entry in logs:
            # Count by event type
            event_type_key = entry.event_type.value
            stats["by_event_type"][event_type_key] = (
                stats["by_event_type"].get(event_type_key, 0) + 1
            )

            # Count by result
            if entry.result:
                stats["by_result"]["allowed"] += 1
            else:
                stats["by_result"]["denied"] += 1

            # Count violations
            if entry.event_type in {
                AuditEventType.SECURITY_VIOLATION,
                AuditEventType.PERMISSION_DENIED,
            }:
                stats["violations_count"] += 1

        return stats

    async def cleanup_old_logs(self) -> None:
        """Remove audit logs older than retention period."""
        cutoff_date = datetime.now(UTC) - timedelta(days=self._retention_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        deleted_count = 0
        for log_file in self._log_directory.glob("audit_*.jsonl"):
            # Extract date from filename (audit_YYYY-MM-DD.jsonl)
            try:
                file_date_str = log_file.stem.split("_")[1]
                if file_date_str < cutoff_str:
                    log_file.unlink()
                    deleted_count += 1
            except (IndexError, ValueError):
                continue

        if deleted_count > 0:
            logger.info(
                "audit_logs_cleaned",
                deleted_count=deleted_count,
                retention_days=self._retention_days,
            )

    async def _log_writer(self) -> None:
        """Background task to write audit logs to disk."""
        while self._running:
            try:
                # Get entry with timeout to allow periodic cleanup
                try:
                    entry = await asyncio.wait_for(
                        self._write_queue.get(),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    # Perform periodic cleanup
                    await self.cleanup_old_logs()
                    continue

                # Write to daily log file
                await self._write_entry_to_file(entry)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "audit_log_write_failed",
                    error=str(e),
                )

    async def _write_entry_to_file(self, entry: AuditLogEntry) -> None:
        """
        Write audit log entry to daily log file.

        Args:
            entry: Audit log entry to write
        """
        # Daily log file: audit_YYYY-MM-DD.jsonl
        log_file = self._log_directory / (
            f"audit_{entry.timestamp.strftime('%Y-%m-%d')}.jsonl"
        )

        try:
            # Write as JSON line
            log_line = entry.model_dump_json() + "\n"

            # Append to file (create if doesn't exist)
            async with asyncio.Lock():
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(log_line)

        except Exception as e:
            logger.error(
                "audit_log_file_write_failed",
                log_file=str(log_file),
                error=str(e),
            )

    async def read_log_file(self, date: datetime) -> list[AuditLogEntry]:
        """
        Read all audit logs from a specific date.

        Args:
            date: Date to read logs for

        Returns:
            List of audit log entries from that date
        """
        log_file = self._log_directory / f"audit_{date.strftime('%Y-%m-%d')}.jsonl"

        if not log_file.exists():
            return []

        entries: list[AuditLogEntry] = []
        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = AuditLogEntry(**data)
                        entries.append(entry)
                    except Exception as e:
                        logger.error(
                            "audit_log_parse_failed",
                            line=line[:100],
                            error=str(e),
                        )
                        continue

        except Exception as e:
            logger.error(
                "audit_log_file_read_failed",
                log_file=str(log_file),
                error=str(e),
            )

        return entries
