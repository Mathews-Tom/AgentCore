"""
Unit tests for Session JSON-RPC Service.

Tests for session management JSON-RPC method handlers covering all 21 methods.
Tests parameter validation, success paths, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.session import (
    SessionState,
    SessionPriority,
    SessionCreateResponse,
    SessionQueryResponse,
    SessionQuery)


class TestSessionCreate:
    """Test session.create JSON-RPC method."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_create_session_success(self, mock_manager):
        """Test successful session creation."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_create

        # Setup mock response
        create_response = SessionCreateResponse(
            session_id="session-123",
            state="active",
            message="Session created successfully")
        mock_manager.create_session = AsyncMock(return_value=create_response)

        # Create request
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.create",
            params={
                "name": "Test Session",
                "owner_agent": "test-agent",
                "priority": "normal",
            },
            id="1")

        # Execute
        result = await handle_session_create(request)

        # Verify
        assert result["session_id"] == "session-123"
        assert result["state"] == "active"
        assert result["message"] == "Session created successfully"
        mock_manager.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_missing_params(self):
        """Test session creation with missing parameters."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_create

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.create",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_session_create(request)

    @pytest.mark.asyncio
    async def test_create_session_missing_required_fields(self):
        """Test session creation with missing required fields."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_create

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.create",
            params={"name": "Test Session"},  # Missing owner_agent
            id="1")

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_session_create(request)


class TestSessionGet:
    """Test session.get JSON-RPC method."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_get_session_success(self, mock_manager):
        """Test successful session retrieval."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_get

        mock_session = Mock()
        mock_session.model_dump.return_value = {
            "session_id": "session-123",
            "name": "Test Session",
            "state": "active",
        }
        mock_manager.get_session = AsyncMock(return_value=mock_session)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.get",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_get(request)

        assert result["session_id"] == "session-123"
        assert result["name"] == "Test Session"
        mock_manager.get_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_get_session_missing_params(self):
        """Test get session with missing parameters."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_get

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.get",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameter required: session_id"):
            await handle_session_get(request)

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_get_session_not_found(self, mock_manager):
        """Test get session when session does not exist."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_get

        mock_manager.get_session = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.get",
            params={"session_id": "nonexistent"},
            id="1")

        with pytest.raises(ValueError, match="Session not found"):
            await handle_session_get(request)


class TestSessionDelete:
    """Test session.delete JSON-RPC method."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_delete_session_success(self, mock_manager):
        """Test successful session deletion."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_delete

        mock_manager.delete_session = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.delete",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_delete(request)

        assert result["success"] is True
        assert result["session_id"] == "session-123"
        mock_manager.delete_session.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_delete_session_failure(self, mock_manager):
        """Test delete session when deletion fails."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_delete

        mock_manager.delete_session = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.delete",
            params={"session_id": "session-123"},
            id="1")

        with pytest.raises(ValueError, match="Session deletion failed"):
            await handle_session_delete(request)


class TestSessionStateTransitions:
    """Test session state transition methods (pause, resume, suspend, complete, fail)."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_pause_session(self, mock_manager):
        """Test pausing a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_pause

        mock_manager.pause_session = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.pause",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_pause(request)

        assert result["success"] is True
        assert result["session_id"] == "session-123"
        mock_manager.pause_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_resume_session(self, mock_manager):
        """Test resuming a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_resume

        mock_manager.resume_session = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.resume",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_resume(request)

        assert result["success"] is True
        mock_manager.resume_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_suspend_session(self, mock_manager):
        """Test suspending a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_suspend

        mock_manager.suspend_session = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.suspend",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_suspend(request)

        assert result["success"] is True
        mock_manager.suspend_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_complete_session(self, mock_manager):
        """Test completing a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_complete

        mock_manager.complete_session = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.complete",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_complete(request)

        assert result["success"] is True
        mock_manager.complete_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_fail_session(self, mock_manager):
        """Test failing a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_fail

        mock_manager.fail_session = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.fail",
            params={
                "session_id": "session-123",
                "reason": "Test failure",
            },
            id="1")

        result = await handle_session_fail(request)

        assert result["success"] is True
        assert result["reason"] == "Test failure"
        mock_manager.fail_session.assert_called_once_with("session-123", "Test failure")


class TestSessionContext:
    """Test session context management methods."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_update_context(self, mock_manager):
        """Test updating session context."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_update_context

        mock_manager.update_context = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.update_context",
            params={
                "session_id": "session-123",
                "updates": {"key": "value", "counter": 42},
            },
            id="1")

        result = await handle_session_update_context(request)

        assert result["success"] is True
        assert result["session_id"] == "session-123"
        mock_manager.update_context.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_set_agent_state(self, mock_manager):
        """Test setting agent state in session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_set_agent_state

        mock_manager.set_agent_state = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.set_agent_state",
            params={
                "session_id": "session-123",
                "agent_id": "test-agent",
                "state": {"status": "processing", "progress": 0.5},
            },
            id="1")

        result = await handle_session_set_agent_state(request)

        assert result["success"] is True
        assert result["agent_id"] == "test-agent"
        mock_manager.set_agent_state.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_get_agent_state(self, mock_manager):
        """Test getting agent state from session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_get_agent_state

        agent_state = {"status": "processing", "progress": 0.75}
        mock_manager.get_agent_state = AsyncMock(return_value=agent_state)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.get_agent_state",
            params={
                "session_id": "session-123",
                "agent_id": "test-agent",
            },
            id="1")

        result = await handle_session_get_agent_state(request)

        assert result["state"] == agent_state
        mock_manager.get_agent_state.assert_called_once_with("session-123", "test-agent")

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_get_agent_state_not_found(self, mock_manager):
        """Test getting agent state when agent not in session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_get_agent_state

        mock_manager.get_agent_state = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.get_agent_state",
            params={
                "session_id": "session-123",
                "agent_id": "nonexistent-agent",
            },
            id="1")

        with pytest.raises(ValueError, match="Agent state not found"):
            await handle_session_get_agent_state(request)


class TestSessionTasksAndEvents:
    """Test session task and event tracking methods."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_add_task(self, mock_manager):
        """Test adding task to session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_add_task

        mock_manager.add_task = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.add_task",
            params={
                "session_id": "session-123",
                "task_id": "task-123",
            },
            id="1")

        result = await handle_session_add_task(request)

        assert result["success"] is True
        assert result["task_id"] == "task-123"
        mock_manager.add_task.assert_called_once_with("session-123", "task-123")

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_record_event(self, mock_manager):
        """Test recording event in session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_record_event

        mock_manager.record_event = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.record_event",
            params={
                "session_id": "session-123",
                "event_type": "state_change",
                "event_data": {"old": "active", "new": "paused"},
            },
            id="1")

        result = await handle_session_record_event(request)

        assert result["success"] is True
        assert result["event_type"] == "state_change"
        mock_manager.record_event.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_checkpoint(self, mock_manager):
        """Test creating session checkpoint."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_checkpoint

        mock_manager.create_checkpoint = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.checkpoint",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_checkpoint(request)

        assert result["success"] is True
        assert result["session_id"] == "session-123"
        mock_manager.create_checkpoint.assert_called_once_with("session-123")


class TestSessionQuery:
    """Test session query methods."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_query_sessions(self, mock_manager):
        """Test querying sessions with filters."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_query

        query_response = SessionQueryResponse(
            sessions=[{"session_id": "session-123", "name": "Test Session"}],
            total_count=1,
            has_more=False,
            query=SessionQuery(owner_agent="test-agent", state=SessionState.ACTIVE, limit=10, offset=0))
        mock_manager.query_sessions = AsyncMock(return_value=query_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.query",
            params={
                "owner_agent": "test-agent",
                "state": "active",
                "limit": 10,
            },
            id="1")

        result = await handle_session_query(request)

        assert result["total_count"] == 1
        assert len(result["sessions"]) == 1
        mock_manager.query_sessions.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_cleanup_expired(self, mock_manager):
        """Test cleaning up expired sessions."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_cleanup_expired

        mock_manager.cleanup_expired_sessions = AsyncMock(return_value=5)  # 5 sessions cleaned

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.cleanup_expired",
            params={},
            id="1")

        result = await handle_session_cleanup_expired(request)

        assert result["cleanup_count"] == 5
        mock_manager.cleanup_expired_sessions.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_cleanup_idle(self, mock_manager):
        """Test cleaning up idle sessions."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_cleanup_idle

        mock_manager.cleanup_idle_sessions = AsyncMock(return_value=3)  # 3 sessions suspended

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.cleanup_idle",
            params={},
            id="1")

        result = await handle_session_cleanup_idle(request)

        assert result["suspended_count"] == 3
        mock_manager.cleanup_idle_sessions.assert_called_once()


class TestSessionImportExport:
    """Test session import/export methods."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_export_session(self, mock_manager):
        """Test exporting a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_export

        json_data = '{"session_id": "session-123", "name": "Test Session"}'
        mock_manager.export_session = AsyncMock(return_value=json_data)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.export",
            params={"session_id": "session-123"},
            id="1")

        result = await handle_session_export(request)

        assert result["session_id"] == "session-123"
        assert result["json_data"] == json_data
        assert result["size_bytes"] == len(json_data)
        mock_manager.export_session.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_import_session(self, mock_manager):
        """Test importing a session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_import

        json_data = '{"session_id": "session-123", "name": "Test Session"}'
        mock_manager.import_session = AsyncMock(return_value="session-123")

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.import",
            params={"json_data": json_data},
            id="1")

        result = await handle_session_import(request)

        assert result["success"] is True
        assert result["session_id"] == "session-123"
        mock_manager.import_session.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_export_batch(self, mock_manager):
        """Test batch exporting sessions."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_export_batch

        json_data = '[{"session_id": "session-1"}, {"session_id": "session-2"}]'
        mock_manager.export_sessions_batch = AsyncMock(return_value=json_data)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.export_batch",
            params={"session_ids": ["session-1", "session-2"]},
            id="1")

        result = await handle_session_export_batch(request)

        assert result["count"] == 2
        assert result["json_data"] == json_data
        mock_manager.export_sessions_batch.assert_called_once()

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_import_batch(self, mock_manager):
        """Test batch importing sessions."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_import_batch

        results = {
            "total": 3,
            "imported": 3,
            "skipped": 0,
            "failed": 0,
        }
        mock_manager.import_sessions_batch = AsyncMock(return_value=results)

        json_data = '[{"session_id": "session-1"}]'

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.import_batch",
            params={"json_data": json_data},
            id="1")

        result = await handle_session_import_batch(request)

        assert result["success"] is True
        assert result["results"]["total"] == 3
        assert result["results"]["imported"] == 3
        mock_manager.import_sessions_batch.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_pause_session_failure(self, mock_manager):
        """Test pause session when operation fails."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_pause

        mock_manager.pause_session = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.pause",
            params={"session_id": "session-123"},
            id="1")

        with pytest.raises(ValueError, match="Session pause failed"):
            await handle_session_pause(request)

    @pytest.mark.asyncio
    async def test_update_context_missing_updates(self):
        """Test update context without updates parameter."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_update_context

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.update_context",
            params={"session_id": "session-123"},  # Missing updates
            id="1")

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_session_update_context(request)

    @pytest.mark.asyncio
    @patch('agentcore.a2a_protocol.services.session_jsonrpc.session_manager')
    async def test_export_session_not_found(self, mock_manager):
        """Test exporting non-existent session."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_export

        mock_manager.export_session = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.export",
            params={"session_id": "nonexistent"},
            id="1")

        with pytest.raises(ValueError, match="Session not found"):
            await handle_session_export(request)

    @pytest.mark.asyncio
    async def test_export_batch_invalid_param(self):
        """Test export batch with invalid session_ids parameter."""
        from agentcore.a2a_protocol.services.session_jsonrpc import handle_session_export_batch

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="session.export_batch",
            params={"session_ids": "not-a-list"},  # Should be array
            id="1")

        with pytest.raises(ValueError, match="must be array"):
            await handle_session_export_batch(request)
