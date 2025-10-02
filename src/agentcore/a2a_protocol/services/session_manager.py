"""
Session Manager Service

Core service for managing session lifecycle, state persistence, and context
preservation for long-running agent workflows.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog
from collections import defaultdict

from agentcore.a2a_protocol.models.session import (
    SessionSnapshot,
    SessionState,
    SessionPriority,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionQuery,
    SessionQueryResponse,
    SessionContext,
)
from agentcore.a2a_protocol.database.connection import get_session
from agentcore.a2a_protocol.database.repositories import SessionRepository


class SessionManager:
    """
    Session lifecycle manager with state persistence and timeout handling.

    Manages session creation, state transitions, context preservation,
    and automatic cleanup of expired sessions.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # Session storage
        self._sessions: Dict[str, SessionSnapshot] = {}

        # Indexing for efficient queries
        self._sessions_by_state: Dict[SessionState, Set[str]] = defaultdict(set)
        self._sessions_by_owner: Dict[str, Set[str]] = defaultdict(set)
        self._sessions_by_participant: Dict[str, Set[str]] = defaultdict(set)

        # Timeout tracking
        self._session_timers: Dict[str, asyncio.TimerHandle] = {}

    async def create_session(self, request: SessionCreateRequest) -> SessionCreateResponse:
        """
        Create a new session.

        Args:
            request: Session creation request

        Returns:
            Session creation response

        Raises:
            ValueError: If session creation fails
        """
        # Create session snapshot
        session = SessionSnapshot(
            name=request.name,
            description=request.description,
            owner_agent=request.owner_agent,
            priority=request.priority,
            timeout_seconds=request.timeout_seconds,
            max_idle_seconds=request.max_idle_seconds,
            tags=request.tags,
        )

        # Set expiration
        session.expires_at = datetime.utcnow() + timedelta(seconds=request.timeout_seconds)

        # Initialize context if provided
        if request.initial_context:
            session.context.variables = request.initial_context

        # Persist to database
        async with get_session() as db_session:
            await SessionRepository.create(db_session, session)
            await db_session.commit()

        # Store in memory cache
        self._sessions[session.session_id] = session

        # Update indices
        self._update_indices_on_create(session)

        # Schedule timeout check
        self._schedule_timeout_check(session.session_id, request.timeout_seconds)

        self.logger.info(
            "Session created",
            session_id=session.session_id,
            name=session.name,
            owner=request.owner_agent,
            priority=session.priority.value,
            timeout=request.timeout_seconds
        )

        return SessionCreateResponse(
            session_id=session.session_id,
            state=session.state.value,
            message="Session created successfully"
        )

    async def get_session(self, session_id: str) -> Optional[SessionSnapshot]:
        """
        Get session by ID, loading from database if not in cache.

        Args:
            session_id: Session ID

        Returns:
            SessionSnapshot or None if not found
        """
        # Check cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Load from database
        async with get_session() as db_session:
            session_db = await SessionRepository.get_by_id(db_session, session_id)
            if not session_db:
                return None

            # Convert to snapshot and cache
            snapshot = SessionRepository.to_snapshot(session_db)
            self._sessions[session_id] = snapshot
            self._update_indices_on_create(snapshot)

            return snapshot

    async def pause_session(self, session_id: str) -> bool:
        """
        Pause active session.

        Args:
            session_id: Session ID

        Returns:
            True if paused successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        try:
            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Pause session
            session.pause()

            # Update state indices
            self._sessions_by_state[session.state].add(session_id)

            # Persist to database
            await self._persist_session(session)

            # Cancel timeout timer
            self._cancel_timeout_check(session_id)

            self.logger.info("Session paused", session_id=session_id)
            return True

        except ValueError as e:
            self.logger.error("Session pause failed", session_id=session_id, error=str(e))
            return False

    async def resume_session(self, session_id: str) -> bool:
        """
        Resume paused or suspended session.

        Args:
            session_id: Session ID

        Returns:
            True if resumed successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        try:
            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Resume session
            session.resume()

            # Update state indices
            self._sessions_by_state[session.state].add(session_id)

            # Persist to database
            await self._persist_session(session)

            # Reschedule timeout
            remaining_time = max(0, (session.expires_at - datetime.utcnow()).total_seconds())
            if remaining_time > 0:
                self._schedule_timeout_check(session_id, int(remaining_time))

            self.logger.info("Session resumed", session_id=session_id)
            return True

        except ValueError as e:
            self.logger.error("Session resume failed", session_id=session_id, error=str(e))
            return False

    async def suspend_session(self, session_id: str) -> bool:
        """
        Suspend session for later resumption.

        Args:
            session_id: Session ID

        Returns:
            True if suspended successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        try:
            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Suspend session
            session.suspend()

            # Update state indices
            self._sessions_by_state[session.state].add(session_id)

            # Persist to database
            await self._persist_session(session)

            # Cancel timeout timer
            self._cancel_timeout_check(session_id)

            self.logger.info("Session suspended", session_id=session_id)
            return True

        except ValueError as e:
            self.logger.error("Session suspend failed", session_id=session_id, error=str(e))
            return False

    async def complete_session(self, session_id: str) -> bool:
        """
        Mark session as completed.

        Args:
            session_id: Session ID

        Returns:
            True if completed successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        try:
            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Complete session
            session.complete()

            # Update state indices
            self._sessions_by_state[session.state].add(session_id)

            # Persist to database
            await self._persist_session(session)

            # Cancel timeout timer
            self._cancel_timeout_check(session_id)

            self.logger.info(
                "Session completed",
                session_id=session_id,
                duration=session.duration,
                tasks=len(session.task_ids)
            )
            return True

        except ValueError as e:
            self.logger.error("Session completion failed", session_id=session_id, error=str(e))
            return False

    async def fail_session(self, session_id: str, reason: Optional[str] = None) -> bool:
        """
        Mark session as failed.

        Args:
            session_id: Session ID
            reason: Failure reason

        Returns:
            True if failed successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        try:
            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Fail session
            session.fail(reason)

            # Update state indices
            self._sessions_by_state[session.state].add(session_id)

            # Persist to database
            await self._persist_session(session)

            # Cancel timeout timer
            self._cancel_timeout_check(session_id)

            self.logger.warning(
                "Session failed",
                session_id=session_id,
                reason=reason,
                duration=session.duration
            )
            return True

        except ValueError as e:
            self.logger.error("Session failure recording failed", session_id=session_id, error=str(e))
            return False

    async def update_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session context variables.

        Args:
            session_id: Session ID
            updates: Context variable updates

        Returns:
            True if updated successfully
        """
        session = await self.get_session(session_id)
        if not session or session.is_terminal:
            return False

        session.update_context(updates)

        # Persist to database
        await self._persist_session(session)

        self.logger.debug(
            "Session context updated",
            session_id=session_id,
            update_count=len(updates)
        )
        return True

    async def set_agent_state(self, session_id: str, agent_id: str, state: Dict[str, Any]) -> bool:
        """
        Set state for a specific agent in session.

        Args:
            session_id: Session ID
            agent_id: Agent ID
            state: Agent state data

        Returns:
            True if set successfully
        """
        session = await self.get_session(session_id)
        if not session or session.is_terminal:
            return False

        session.set_agent_state(agent_id, state)

        # Add as participant if not already
        if agent_id not in session.participant_agents:
            session.add_participant(agent_id)
            self._sessions_by_participant[agent_id].add(session_id)

        # Persist to database
        await self._persist_session(session)

        self.logger.debug(
            "Agent state updated",
            session_id=session_id,
            agent_id=agent_id
        )
        return True

    async def get_agent_state(self, session_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get state for a specific agent in session.

        Args:
            session_id: Session ID
            agent_id: Agent ID

        Returns:
            Agent state data or None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        return session.get_agent_state(agent_id)

    async def add_task(self, session_id: str, task_id: str) -> bool:
        """
        Add task to session.

        Args:
            session_id: Session ID
            task_id: Task ID

        Returns:
            True if added successfully
        """
        session = await self.get_session(session_id)
        if not session or session.is_terminal:
            return False

        session.add_task(task_id)

        # Persist to database
        await self._persist_session(session)

        self.logger.debug("Task added to session", session_id=session_id, task_id=task_id)
        return True

    async def add_artifact(self, session_id: str, artifact_id: str) -> bool:
        """
        Add artifact to session.

        Args:
            session_id: Session ID
            artifact_id: Artifact ID

        Returns:
            True if added successfully
        """
        session = await self.get_session(session_id)
        if not session or session.is_terminal:
            return False

        session.add_artifact(artifact_id)

        # Persist to database
        await self._persist_session(session)

        self.logger.debug("Artifact added to session", session_id=session_id, artifact_id=artifact_id)
        return True

    async def record_event(self, session_id: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Record execution event in session history.

        Args:
            session_id: Session ID
            event_type: Event type
            event_data: Event data

        Returns:
            True if recorded successfully
        """
        session = await self.get_session(session_id)
        if not session or session.is_terminal:
            return False

        session.record_event(event_type, event_data)

        # Persist to database
        await self._persist_session(session)

        return True

    async def create_checkpoint(self, session_id: str) -> bool:
        """
        Create checkpoint for session.

        Args:
            session_id: Session ID

        Returns:
            True if checkpoint created successfully
        """
        session = await self.get_session(session_id)
        if not session or session.is_terminal:
            return False

        session.create_checkpoint()

        # Persist to database
        await self._persist_session(session)

        self.logger.info(
            "Session checkpoint created",
            session_id=session_id,
            checkpoint_count=session.checkpoint_count
        )
        return True

    async def query_sessions(self, query: SessionQuery) -> SessionQueryResponse:
        """
        Query sessions with filtering and pagination.

        Args:
            query: Session query parameters

        Returns:
            Query response with matching sessions
        """
        # Start with all session IDs
        candidate_ids = set(self._sessions.keys())

        # Apply filters
        if query.state is not None:
            candidate_ids &= self._sessions_by_state[query.state]

        if query.owner_agent is not None:
            candidate_ids &= self._sessions_by_owner[query.owner_agent]

        if query.participant_agent is not None:
            candidate_ids &= self._sessions_by_participant[query.participant_agent]

        # Filter by additional criteria
        filtered_sessions = []
        for session_id in candidate_ids:
            session = self._sessions[session_id]

            # Priority filter
            if query.priority and session.priority != query.priority:
                continue

            # Tags filter (all must match)
            if query.tags and not all(tag in session.tags for tag in query.tags):
                continue

            # Time filters
            if query.created_after and session.created_at < query.created_after:
                continue

            if query.created_before and session.created_at > query.created_before:
                continue

            # Expired filter
            if not query.include_expired and session.is_expired:
                continue

            filtered_sessions.append(session)

        # Sort by creation time (newest first)
        filtered_sessions.sort(key=lambda s: s.created_at, reverse=True)

        # Apply pagination
        total_count = len(filtered_sessions)
        start_idx = query.offset
        end_idx = start_idx + query.limit

        paginated_sessions = filtered_sessions[start_idx:end_idx]

        # Convert to summaries
        session_summaries = [session.to_summary() for session in paginated_sessions]

        return SessionQueryResponse(
            sessions=session_summaries,
            total_count=total_count,
            has_more=end_idx < total_count,
            query=query
        )

    async def cleanup_expired_sessions(self) -> int:
        """
        Cleanup expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleanup_count = 0
        expired_ids = []

        # Find expired sessions
        for session_id, session in self._sessions.items():
            if session.is_expired and not session.is_terminal:
                expired_ids.append(session_id)

        # Expire sessions
        for session_id in expired_ids:
            session = self._sessions[session_id]

            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Expire session
            try:
                session.expire()
                self._sessions_by_state[session.state].add(session_id)
                cleanup_count += 1

                self.logger.info(
                    "Session expired",
                    session_id=session_id,
                    duration=session.duration
                )
            except ValueError as e:
                self.logger.error("Session expiration failed", session_id=session_id, error=str(e))

        return cleanup_count

    async def cleanup_idle_sessions(self) -> int:
        """
        Cleanup idle sessions that exceeded max idle time.

        Returns:
            Number of sessions suspended
        """
        cleanup_count = 0

        # Find idle active sessions
        for session_id in list(self._sessions_by_state[SessionState.ACTIVE]):
            session = self._sessions[session_id]

            if session.is_idle:
                # Suspend idle session
                if await self.suspend_session(session_id):
                    cleanup_count += 1
                    self.logger.info(
                        "Idle session suspended",
                        session_id=session_id,
                        idle_time=session.time_since_update
                    )

        return cleanup_count

    async def delete_session(self, session_id: str, hard_delete: bool = False) -> bool:
        """
        Delete session (soft or hard delete).

        Args:
            session_id: Session ID to delete
            hard_delete: If True, permanently delete from database; if False, mark as expired

        Returns:
            True if deleted successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        if hard_delete:
            # Permanently delete from database
            async with get_session() as db_session:
                deleted = await SessionRepository.delete(db_session, session_id)
                await db_session.commit()

            if not deleted:
                return False

            # Remove from memory cache
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                self._update_indices_on_remove(session)
                self._cancel_timeout_check(session_id)

            self.logger.info("Session hard deleted", session_id=session_id)
            return True

        else:
            # Soft delete - mark as expired
            if session.is_terminal:
                # Already in terminal state
                return True

            # Update state indices
            self._sessions_by_state[session.state].discard(session_id)

            # Expire session
            session.expire()

            # Update state indices
            self._sessions_by_state[session.state].add(session_id)

            # Persist to database
            await self._persist_session(session)

            # Cancel timeout timer
            self._cancel_timeout_check(session_id)

            self.logger.info("Session soft deleted (expired)", session_id=session_id)
            return True

    async def delete_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Delete old terminal sessions.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of sessions deleted
        """
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleanup_count = 0

        terminal_states = [SessionState.COMPLETED, SessionState.FAILED, SessionState.EXPIRED]
        delete_ids = []

        for state in terminal_states:
            for session_id in list(self._sessions_by_state[state]):
                session = self._sessions[session_id]
                if session.completed_at and session.completed_at < cutoff_time:
                    delete_ids.append(session_id)

        # Remove old sessions
        for session_id in delete_ids:
            session = self._sessions.pop(session_id)
            self._update_indices_on_remove(session)
            self._cancel_timeout_check(session_id)
            cleanup_count += 1

        if cleanup_count > 0:
            self.logger.info("Old sessions deleted", count=cleanup_count, max_age_days=max_age_days)

        return cleanup_count

    async def export_session(self, session_id: str, include_history: bool = True) -> Optional[str]:
        """
        Export session to JSON format.

        Args:
            session_id: Session ID to export
            include_history: Whether to include execution history

        Returns:
            JSON string representation of session or None if not found
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        # Create export data structure
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "session": session.model_dump(mode="json"),
        }

        # Optionally exclude execution history to reduce size
        if not include_history and "context" in export_data["session"]:
            context = export_data["session"]["context"]
            if "execution_history" in context:
                context["execution_history"] = []

        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, default=str)

        self.logger.info(
            "Session exported",
            session_id=session_id,
            size_bytes=len(json_str),
            include_history=include_history
        )

        return json_str

    async def import_session(self, json_data: str, overwrite: bool = False) -> Optional[str]:
        """
        Import session from JSON format.

        Args:
            json_data: JSON string containing session data
            overwrite: Whether to overwrite existing session

        Returns:
            Imported session ID or None if import failed

        Raises:
            ValueError: If JSON is invalid or session already exists
        """
        try:
            # Parse JSON
            import_data = json.loads(json_data)

            # Validate format
            if "session" not in import_data:
                raise ValueError("Invalid export format: missing 'session' key")

            session_data = import_data["session"]

            # Check if session exists
            existing_session = await self.get_session(session_data["session_id"])
            if existing_session and not overwrite:
                raise ValueError(f"Session already exists: {session_data['session_id']}")

            # Reconstruct SessionSnapshot
            session = SessionSnapshot.model_validate(session_data)

            # If not overwriting, generate new ID
            if existing_session and not overwrite:
                session.session_id = str(uuid4())

            # Persist to database
            async with get_session() as db_session:
                if existing_session and overwrite:
                    await SessionRepository.update(db_session, session)
                else:
                    await SessionRepository.create(db_session, session)
                await db_session.commit()

            # Store in memory cache
            self._sessions[session.session_id] = session

            # Update indices
            if not existing_session:
                self._update_indices_on_create(session)

            # Reschedule timeout if active
            if session.state == SessionState.ACTIVE and session.expires_at:
                remaining_time = max(0, (session.expires_at - datetime.utcnow()).total_seconds())
                if remaining_time > 0:
                    self._schedule_timeout_check(session.session_id, int(remaining_time))

            self.logger.info(
                "Session imported",
                session_id=session.session_id,
                overwrite=overwrite,
                state=session.state.value
            )

            return session.session_id

        except json.JSONDecodeError as e:
            self.logger.error("Session import failed: invalid JSON", error=str(e))
            raise ValueError(f"Invalid JSON: {str(e)}")

        except Exception as e:
            self.logger.error("Session import failed", error=str(e))
            raise

    async def export_sessions_batch(
        self,
        session_ids: List[str],
        include_history: bool = True
    ) -> str:
        """
        Export multiple sessions to JSON format.

        Args:
            session_ids: List of session IDs to export
            include_history: Whether to include execution history

        Returns:
            JSON string containing all sessions
        """
        sessions_data = []

        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session:
                session_dict = session.model_dump(mode="json")

                # Optionally exclude history
                if not include_history and "context" in session_dict:
                    context = session_dict["context"]
                    if "execution_history" in context:
                        context["execution_history"] = []

                sessions_data.append(session_dict)

        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "count": len(sessions_data),
            "sessions": sessions_data,
        }

        json_str = json.dumps(export_data, indent=2, default=str)

        self.logger.info(
            "Sessions batch exported",
            count=len(sessions_data),
            size_bytes=len(json_str),
            include_history=include_history
        )

        return json_str

    async def import_sessions_batch(
        self,
        json_data: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Import multiple sessions from JSON format.

        Args:
            json_data: JSON string containing multiple sessions
            overwrite: Whether to overwrite existing sessions

        Returns:
            Import results with counts and errors
        """
        try:
            # Parse JSON
            import_data = json.loads(json_data)

            # Validate format
            if "sessions" not in import_data:
                raise ValueError("Invalid export format: missing 'sessions' key")

            sessions_data = import_data["sessions"]
            results = {
                "total": len(sessions_data),
                "imported": 0,
                "skipped": 0,
                "failed": 0,
                "errors": []
            }

            for session_data in sessions_data:
                try:
                    session = SessionSnapshot.model_validate(session_data)

                    # Check if exists
                    existing = await self.get_session(session.session_id)
                    if existing and not overwrite:
                        results["skipped"] += 1
                        continue

                    # Persist to database
                    async with get_session() as db_session:
                        if existing and overwrite:
                            await SessionRepository.update(db_session, session)
                        else:
                            await SessionRepository.create(db_session, session)
                        await db_session.commit()

                    # Cache in memory
                    self._sessions[session.session_id] = session

                    # Update indices
                    if not existing:
                        self._update_indices_on_create(session)

                    results["imported"] += 1

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "session_id": session_data.get("session_id", "unknown"),
                        "error": str(e)
                    })

            self.logger.info(
                "Sessions batch imported",
                total=results["total"],
                imported=results["imported"],
                skipped=results["skipped"],
                failed=results["failed"]
            )

            return results

        except json.JSONDecodeError as e:
            self.logger.error("Batch import failed: invalid JSON", error=str(e))
            raise ValueError(f"Invalid JSON: {str(e)}")

        except Exception as e:
            self.logger.error("Batch import failed", error=str(e))
            raise

    # Private helper methods

    async def _persist_session(self, session: SessionSnapshot) -> None:
        """Persist session updates to database."""
        async with get_session() as db_session:
            await SessionRepository.update(db_session, session)
            await db_session.commit()

    def _update_indices_on_create(self, session: SessionSnapshot) -> None:
        """Update indices when session is created."""
        self._sessions_by_state[session.state].add(session.session_id)
        self._sessions_by_owner[session.owner_agent].add(session.session_id)

    def _update_indices_on_remove(self, session: SessionSnapshot) -> None:
        """Update indices when session is removed."""
        # Remove from all indices
        for state_set in self._sessions_by_state.values():
            state_set.discard(session.session_id)

        for owner_set in self._sessions_by_owner.values():
            owner_set.discard(session.session_id)

        for participant_set in self._sessions_by_participant.values():
            participant_set.discard(session.session_id)

    def _schedule_timeout_check(self, session_id: str, timeout_seconds: int) -> None:
        """Schedule timeout check for session."""
        # Cancel existing timer if any
        self._cancel_timeout_check(session_id)

        # Schedule new timer
        loop = asyncio.get_event_loop()
        timer = loop.call_later(
            timeout_seconds,
            lambda: asyncio.create_task(self._handle_timeout(session_id))
        )
        self._session_timers[session_id] = timer

    def _cancel_timeout_check(self, session_id: str) -> None:
        """Cancel timeout check for session."""
        if session_id in self._session_timers:
            self._session_timers[session_id].cancel()
            del self._session_timers[session_id]

    async def _handle_timeout(self, session_id: str) -> None:
        """Handle session timeout."""
        session = self._sessions.get(session_id)
        if not session or session.is_terminal:
            return

        if session.is_expired:
            await self.fail_session(session_id, "Session timeout exceeded")


# Global session manager instance
session_manager = SessionManager()
