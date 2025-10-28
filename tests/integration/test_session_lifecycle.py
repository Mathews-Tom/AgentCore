"""
Integration Tests for Session Management

Tests the complete session lifecycle including creation, state transitions,
context management, and cleanup operations.
"""

from datetime import UTC, datetime

import pytest

from agentcore.a2a_protocol.database.connection import get_session
from agentcore.a2a_protocol.database.repositories import SessionRepository
from agentcore.a2a_protocol.models.session import (
    SessionCreateRequest,
    SessionPriority,
    SessionQuery,
    SessionSnapshot,
    SessionState)
from agentcore.a2a_protocol.services.session_manager import session_manager


@pytest.mark.asyncio
class TestSessionLifecycle:
    """Test session lifecycle operations."""

    async def test_session_creation(self, init_test_db):
        """Test creating a new session."""
        request = SessionCreateRequest(
            name="Test Workflow Session",
            description="Integration test session",
            owner_agent="test-agent-001",
            priority=SessionPriority.NORMAL,
            timeout_seconds=3600,
            max_idle_seconds=300,
            tags=["test", "integration"],
            initial_context={"workflow": "test", "step": 1})

        response = await session_manager.create_session(request)

        assert response.session_id is not None
        assert response.state == "active"
        assert "successfully" in response.message.lower()

        # Verify session was persisted
        async with get_session() as db_session:
            session_db = await SessionRepository.get_by_id(
                db_session, response.session_id
            )
            assert session_db is not None
            assert session_db.name == "Test Workflow Session"
            assert session_db.owner_agent == "test-agent-001"
            assert session_db.state == SessionState.ACTIVE

        # Cleanup
        await session_manager.delete_session(response.session_id, hard_delete=True)

    async def test_session_get(self, init_test_db):
        """Test retrieving session details."""
        # Create session
        request = SessionCreateRequest(
            name="Get Test Session",
            owner_agent="test-agent-002")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Get session
        session = await session_manager.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.name == "Get Test Session"
        assert session.owner_agent == "test-agent-002"
        assert session.state == SessionState.ACTIVE

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_pause_resume(self, init_test_db):
        """Test pausing and resuming a session."""
        # Create session
        request = SessionCreateRequest(
            name="Pause Resume Test",
            owner_agent="test-agent-003")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Pause session
        success = await session_manager.pause_session(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.PAUSED

        # Resume session
        success = await session_manager.resume_session(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.ACTIVE

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_suspend(self, init_test_db):
        """Test suspending a session."""
        # Create session
        request = SessionCreateRequest(
            name="Suspend Test",
            owner_agent="test-agent-004")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Suspend session
        success = await session_manager.suspend_session(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.SUSPENDED

        # Resume from suspended
        success = await session_manager.resume_session(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.ACTIVE

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_complete(self, init_test_db):
        """Test marking session as completed."""
        # Create session
        request = SessionCreateRequest(
            name="Complete Test",
            owner_agent="test-agent-005")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Complete session
        success = await session_manager.complete_session(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.COMPLETED
        assert session.completed_at is not None
        assert session.is_terminal is True

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_fail(self, init_test_db):
        """Test marking session as failed."""
        # Create session
        request = SessionCreateRequest(
            name="Fail Test",
            owner_agent="test-agent-006")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Fail session
        reason = "Test failure reason"
        success = await session_manager.fail_session(session_id, reason)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.FAILED
        assert session.completed_at is not None
        assert session.metadata.get("failure_reason") == reason
        assert session.is_terminal is True

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_context_update(self, init_test_db):
        """Test updating session context."""
        # Create session
        request = SessionCreateRequest(
            name="Context Test",
            owner_agent="test-agent-007",
            initial_context={"key1": "value1"})
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Update context
        updates = {"key2": "value2", "key3": "value3"}
        success = await session_manager.update_context(session_id, updates)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.context.variables["key1"] == "value1"
        assert session.context.variables["key2"] == "value2"
        assert session.context.variables["key3"] == "value3"

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_agent_state(self, init_test_db):
        """Test setting and getting agent state."""
        # Create session
        request = SessionCreateRequest(
            name="Agent State Test",
            owner_agent="test-agent-008")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Set agent state
        agent_id = "worker-agent-001"
        agent_state = {"status": "processing", "progress": 50}
        success = await session_manager.set_agent_state(
            session_id, agent_id, agent_state
        )
        assert success is True

        # Verify agent added as participant
        session = await session_manager.get_session(session_id)
        assert agent_id in session.participant_agents

        # Get agent state
        retrieved_state = await session_manager.get_agent_state(session_id, agent_id)
        assert retrieved_state == agent_state

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_add_task_and_artifact(self, init_test_db):
        """Test adding tasks and artifacts to session."""
        # Create session
        request = SessionCreateRequest(
            name="Resources Test",
            owner_agent="test-agent-009")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Add task
        task_id = "task-001"
        success = await session_manager.add_task(session_id, task_id)
        assert success is True

        # Add artifact
        artifact_id = "artifact-001"
        success = await session_manager.add_artifact(session_id, artifact_id)
        assert success is True

        # Verify
        session = await session_manager.get_session(session_id)
        assert task_id in session.task_ids
        assert artifact_id in session.artifact_ids

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_record_event(self, init_test_db):
        """Test recording events in session history."""
        # Create session
        request = SessionCreateRequest(
            name="Events Test",
            owner_agent="test-agent-010")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Record events
        event1 = {"action": "start", "timestamp": datetime.now(UTC).isoformat()}
        success = await session_manager.record_event(
            session_id, "workflow.start", event1
        )
        assert success is True

        event2 = {"action": "progress", "step": 1}
        success = await session_manager.record_event(
            session_id, "workflow.progress", event2
        )
        assert success is True

        # Verify
        session = await session_manager.get_session(session_id)
        assert len(session.context.execution_history) >= 2

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_checkpoint(self, init_test_db):
        """Test creating session checkpoints."""
        # Create session
        request = SessionCreateRequest(
            name="Checkpoint Test",
            owner_agent="test-agent-011")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        initial_checkpoint_count = 0

        # Create checkpoint
        success = await session_manager.create_checkpoint(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.checkpoint_count == initial_checkpoint_count + 1
        assert session.last_checkpoint_at is not None

        # Create another checkpoint
        success = await session_manager.create_checkpoint(session_id)
        assert success is True

        session = await session_manager.get_session(session_id)
        assert session.checkpoint_count == initial_checkpoint_count + 2

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_query(self, init_test_db):
        """Test querying sessions."""
        # Create multiple sessions
        sessions_created = []
        for i in range(5):
            request = SessionCreateRequest(
                name=f"Query Test Session {i}",
                owner_agent="test-agent-012",
                priority=SessionPriority.HIGH if i < 2 else SessionPriority.NORMAL,
                tags=["query-test", f"batch-{i // 2}"])
            response = await session_manager.create_session(request)
            sessions_created.append(response.session_id)

        # Query by owner
        query = SessionQuery(owner_agent="test-agent-012", limit=10)
        response = await session_manager.query_sessions(query)
        assert response.total_count >= 5
        assert len(response.sessions) >= 5

        # Query by priority
        query = SessionQuery(
            owner_agent="test-agent-012", priority=SessionPriority.HIGH, limit=10
        )
        response = await session_manager.query_sessions(query)
        assert response.total_count >= 2

        # Query by tags
        query = SessionQuery(
            owner_agent="test-agent-012", tags=["query-test"], limit=10
        )
        response = await session_manager.query_sessions(query)
        assert response.total_count >= 5

        # Cleanup
        for session_id in sessions_created:
            await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_export_import(self, init_test_db):
        """Test exporting and importing sessions."""
        # Create session
        request = SessionCreateRequest(
            name="Export Import Test",
            owner_agent="test-agent-013",
            initial_context={"export": "test"})
        response = await session_manager.create_session(request)
        original_session_id = response.session_id

        # Add some data
        await session_manager.add_task(original_session_id, "task-export-001")
        await session_manager.record_event(
            original_session_id, "test.event", {"data": "test"}
        )

        # Export session
        json_data = await session_manager.export_session(original_session_id)
        assert json_data is not None
        assert "session" in json_data

        # Delete original session
        await session_manager.delete_session(original_session_id, hard_delete=True)

        # Import session
        imported_session_id = await session_manager.import_session(
            json_data, overwrite=False
        )
        assert imported_session_id is not None

        # Verify imported session
        imported_session = await session_manager.get_session(imported_session_id)
        assert imported_session is not None
        assert imported_session.name == "Export Import Test"
        assert "task-export-001" in imported_session.task_ids
        assert imported_session.context.variables["export"] == "test"

        # Cleanup
        await session_manager.delete_session(imported_session_id, hard_delete=True)

    async def test_session_soft_delete(self, init_test_db):
        """Test soft deleting a session."""
        # Create session
        request = SessionCreateRequest(
            name="Soft Delete Test",
            owner_agent="test-agent-014")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Suspend session first (required before expiring)
        success = await session_manager.suspend_session(session_id)
        assert success is True

        # Soft delete (expire) - now session is in SUSPENDED state
        success = await session_manager.delete_session(session_id, hard_delete=False)
        assert success is True

        # Verify session is expired
        session = await session_manager.get_session(session_id)
        assert session.state == SessionState.EXPIRED
        assert session.is_terminal is True

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)

    async def test_session_hard_delete(self, init_test_db):
        """Test hard deleting a session."""
        # Create session
        request = SessionCreateRequest(
            name="Hard Delete Test",
            owner_agent="test-agent-015")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Hard delete
        success = await session_manager.delete_session(session_id, hard_delete=True)
        assert success is True

        # Verify session is gone
        session = await session_manager.get_session(session_id)
        assert session is None

    async def test_session_state_transitions(self, init_test_db):
        """Test valid and invalid state transitions."""
        # Create session
        request = SessionCreateRequest(
            name="State Transition Test",
            owner_agent="test-agent-016")
        response = await session_manager.create_session(request)
        session_id = response.session_id

        # Valid transitions: ACTIVE -> PAUSED -> ACTIVE
        success = await session_manager.pause_session(session_id)
        assert success is True

        success = await session_manager.resume_session(session_id)
        assert success is True

        # Valid transitions: ACTIVE -> COMPLETED
        success = await session_manager.complete_session(session_id)
        assert success is True

        # Invalid transition: COMPLETED -> ACTIVE
        success = await session_manager.resume_session(session_id)
        assert success is False

        # Cleanup
        await session_manager.delete_session(session_id, hard_delete=True)
