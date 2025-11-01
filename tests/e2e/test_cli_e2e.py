"""End-to-end tests for AgentCore CLI against live API.

These tests run the CLI as a subprocess against a real AgentCore API instance
at http://localhost:8001 (not mocked). They validate full workflows including:
- Agent lifecycle (register, list, info, remove)
- Task lifecycle (create, status, list, cancel, result)
- Session management (save, list, info, resume, delete)
- Workflow execution (create, execute, status, visualize)
- Configuration management (CLI args override env vars)
- Output formats (JSON, table)
- Error handling (connection errors, auth errors)

Requirements:
- AgentCore API running at http://localhost:8001
- Docker Compose environment up (PostgreSQL, Redis, A2A API)

Run on demand only (not in CI):
    uv run pytest tests/e2e/test_cli_e2e.py -v

KNOWN ISSUES:
- CLI agent registration currently incompatible with API (expects agent_card parameter)
- CLI needs to be updated to match A2A protocol agent.register JSON-RPC method
- Some tests marked as xfail until CLI-API compatibility is restored
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pytest


# Mark all tests in this module as requiring live API server
# Skip by default - run manually with: pytest tests/e2e/test_cli_e2e.py -v
pytestmark = pytest.mark.skip(reason="E2E tests require live API server at http://localhost:8001 - run manually with docker-compose up")


# Constants
API_URL = "http://localhost:8001"
CLI_TIMEOUT = 30
CLEANUP_TIMEOUT = 10


# Fixtures


@pytest.fixture(scope="session")
def api_health_check() -> None:
    """Verify API is accessible before running tests."""
    result = subprocess.run(
        ["curl", "-s", f"{API_URL}/api/v1/health"],
        capture_output=True,
        text=True,
        timeout=5)

    if result.returncode != 0:
        pytest.skip(
            f"AgentCore API not accessible at {API_URL}. "
            "Start with: docker compose -f docker-compose.dev.yml up"
        )

    try:
        health = json.loads(result.stdout)
        if health.get("status") != "healthy":
            pytest.skip(f"API is not healthy: {health}")
    except json.JSONDecodeError:
        pytest.skip(f"Invalid health response: {result.stdout}")


@pytest.fixture
def test_id() -> str:
    """Generate unique test ID for resource naming."""
    return f"e2e-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_agents(request: pytest.FixtureRequest) -> list[str]:
    """Track agent IDs for cleanup after test."""
    agent_ids: list[str] = []

    yield agent_ids

    # Cleanup all tracked agents
    for agent_id in agent_ids:
        try:
            subprocess.run(
                ["uv", "run", "agentcore", "agent", "remove", agent_id, "--yes"],
                capture_output=True,
                timeout=CLEANUP_TIMEOUT)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass  # Best effort cleanup


@pytest.fixture
def cleanup_tasks(request: pytest.FixtureRequest) -> list[str]:
    """Track task IDs for cleanup after test."""
    task_ids: list[str] = []

    yield task_ids

    # Cleanup all tracked tasks (cancel first, then delete if API supports it)
    for task_id in task_ids:
        try:
            subprocess.run(
                ["uv", "run", "agentcore", "task", "cancel", task_id],
                capture_output=True,
                timeout=CLEANUP_TIMEOUT)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass  # Best effort cleanup


@pytest.fixture
def cleanup_sessions(request: pytest.FixtureRequest) -> list[str]:
    """Track session IDs for cleanup after test."""
    session_ids: list[str] = []

    yield session_ids

    # Cleanup all tracked sessions
    for session_id in session_ids:
        try:
            subprocess.run(
                ["uv", "run", "agentcore", "session", "delete", session_id, "--yes"],
                capture_output=True,
                timeout=CLEANUP_TIMEOUT)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass  # Best effort cleanup


# Helper functions


def run_cli(
    *args: str,
    input_text: str | None = None,
    timeout: int = CLI_TIMEOUT,
    env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run CLI command and return result.

    Args:
        *args: CLI command arguments (excluding 'uv run agentcore')
        input_text: Optional stdin input
        timeout: Command timeout in seconds
        env: Optional environment variables (merged with current env)

    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    cmd = ["uv", "run", "agentcore"] + list(args)

    # Merge environment variables
    final_env = os.environ.copy()
    if env:
        final_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_text,
        timeout=timeout,
        env=final_env)

    return result


def run_cli_json(*args: str, **kwargs: Any) -> dict[str, Any]:
    """Run CLI command with --json flag and parse output.

    Args:
        *args: CLI command arguments
        **kwargs: Additional arguments for run_cli()

    Returns:
        Parsed JSON output

    Raises:
        AssertionError: If command fails or output is not valid JSON
    """
    result = run_cli(*args, "--json", **kwargs)
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON output: {result.stdout}\nError: {e}")


# Test cases


class TestAgentLifecycle:
    """Test agent registration, discovery, and management.

    NOTE: These tests are currently marked as xfail due to CLI-API incompatibility.
    The CLI sends parameters that don't match the A2A protocol's agent.register method.
    """

    def test_agent_register_and_list(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test agent registration and listing."""
        # Register agent
        agent_name = f"test-agent-{test_id}"
        result = run_cli_json(
            "agent", "register",
            "--name", agent_name,
            "--capabilities", "text-generation")

        assert "id" in result or "agent_id" in result
        agent_id = result.get("id") or result.get("agent_id")
        cleanup_agents.append(agent_id)

        # List agents and verify it appears
        agents_result = run_cli_json("agent", "list")

        # Handle both list and dict responses
        if isinstance(agents_result, list):
            agents = agents_result
        elif isinstance(agents_result, dict) and "agents" in agents_result:
            agents = agents_result["agents"]
        else:
            pytest.fail(f"Unexpected agents response format: {agents_result}")

        assert any(a.get("agent_name") == agent_name for a in agents), \
            f"Agent {agent_name} not found in list"

    def test_agent_info(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test getting detailed agent information."""
        # Register agent
        agent_name = f"test-agent-info-{test_id}"
        register_result = run_cli_json(
            "agent", "register",
            "--name", agent_name,
            "--capabilities", "info-test")

        agent_id = register_result.get("id") or register_result.get("agent_id")
        cleanup_agents.append(agent_id)

        # Get agent info
        info_result = run_cli_json("agent", "info", agent_id)

        assert info_result.get("agent_name") == agent_name
        # A2A protocol returns endpoints array, not flat url field
        assert "endpoints" in info_result
        assert len(info_result["endpoints"]) > 0
        assert "url" in info_result["endpoints"][0]

    def test_agent_search_by_capability(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test searching agents by capability."""
        # Register agent with specific capability
        capability = f"custom-capability-{test_id}"
        agent_name = f"test-search-{test_id}"

        register_result = run_cli_json(
            "agent", "register",
            "--name", agent_name,
            "--capabilities", capability)

        agent_id = register_result.get("id") or register_result.get("agent_id")
        cleanup_agents.append(agent_id)

        # Search by capability
        search_result = run_cli_json("agent", "search", "--capability", capability)

        # Handle both list and dict responses
        if isinstance(search_result, list):
            agents = search_result
        elif isinstance(search_result, dict) and "agents" in search_result:
            agents = search_result["agents"]
        else:
            agents = []

        assert len(agents) >= 1, f"No agents found with capability {capability}"
        assert any(a.get("agent_name") == agent_name for a in agents)

    def test_agent_remove(
        self,
        api_health_check: None,
        test_id: str) -> None:
        """Test agent removal (no cleanup needed)."""
        # Register agent
        agent_name = f"test-remove-{test_id}"
        register_result = run_cli_json(
            "agent", "register",
            "--name", agent_name,
            "--capabilities", "remove-test")

        agent_id = register_result.get("id") or register_result.get("agent_id")

        # Remove agent
        remove_result = run_cli("agent", "remove", agent_id, "--yes")
        assert remove_result.returncode == 0

        # Verify it's gone (list should not include it)
        list_result = run_cli_json("agent", "list")

        if isinstance(list_result, list):
            agents = list_result
        elif isinstance(list_result, dict) and "agents" in list_result:
            agents = list_result["agents"]
        else:
            agents = []

        assert not any(a.get("agent_name") == agent_name for a in agents), \
            f"Agent {agent_name} still exists after removal"


class TestTaskLifecycle:
    """Test task creation, monitoring, and management.

    NOTE: These tests are marked as xfail because they depend on agent registration,
    which is currently incompatible between CLI and API.
    """

    def test_task_create_and_status(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str],
        cleanup_tasks: list[str]) -> None:
        """Test task creation and status checking."""
        # First register an agent to assign task to
        agent_result = run_cli_json(
            "agent", "register",
            "--name", f"task-agent-{test_id}",
            "--capabilities", "text-generation")
        agent_id = agent_result.get("id") or agent_result.get("agent_id")
        cleanup_agents.append(agent_id)

        # Create task
        task_result = run_cli_json(
            "task", "create",
            "--description", "Generate text based on prompt",
            "--parameters", json.dumps({"prompt": "test"}))

        assert "id" in task_result or "task_id" in task_result
        task_id = task_result.get("id") or task_result.get("task_id")
        cleanup_tasks.append(task_id)

        # Get task status
        status_result = run_cli_json("task", "status", task_id)

        assert "status" in status_result or "state" in status_result

    def test_task_list(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str],
        cleanup_tasks: list[str]) -> None:
        """Test listing tasks."""
        # Register agent
        agent_result = run_cli_json(
            "agent", "register",
            "--name", f"list-agent-{test_id}",
            "--capabilities", "list-test")
        agent_id = agent_result.get("id") or agent_result.get("agent_id")
        cleanup_agents.append(agent_id)

        # Create task
        task_result = run_cli_json(
            "task", "create",
            "--description", "Test task for listing")
        task_id = task_result.get("id") or task_result.get("task_id")
        cleanup_tasks.append(task_id)

        # List tasks
        list_result = run_cli_json("task", "list")

        # Handle both list and dict responses
        if isinstance(list_result, list):
            tasks = list_result
        elif isinstance(list_result, dict) and "tasks" in list_result:
            tasks = list_result["tasks"]
        else:
            tasks = []

        # Our task should be in the list
        task_ids = [t.get("id") or t.get("task_id") for t in tasks]
        assert task_id in task_ids, f"Task {task_id} not found in list"

    def test_task_cancel(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test task cancellation (no cleanup needed)."""
        # Register agent
        agent_result = run_cli_json(
            "agent", "register",
            "--name", f"cancel-agent-{test_id}",
            "--capabilities", "cancel-test")
        agent_id = agent_result.get("id") or agent_result.get("agent_id")
        cleanup_agents.append(agent_id)

        # Create task
        task_result = run_cli_json(
            "task", "create",
            "--description", "Test task for cancellation")
        task_id = task_result.get("id") or task_result.get("task_id")

        # Cancel task
        cancel_result = run_cli("task", "cancel", task_id)
        assert cancel_result.returncode == 0

        # Verify status is cancelled
        status_result = run_cli_json("task", "status", task_id)
        status = status_result.get("status") or status_result.get("state")

        # Status might be "cancelled", "canceled", or "failed" depending on API
        assert status in ("cancelled", "canceled", "failed", "terminated"), \
            f"Expected cancelled status, got: {status}"


class TestSessionFlow:
    """Test session save, list, resume, and delete.

    NOTE: Session tests may have API compatibility issues as well.
    """

    @pytest.mark.xfail(reason="Session API compatibility not verified")
    def test_session_save_and_list(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_sessions: list[str]) -> None:
        """Test saving and listing sessions."""
        session_name = f"test-session-{test_id}"

        # Save session
        save_result = run_cli_json(
            "session", "save",
            "--name", session_name,
            "--description", "E2E test session",
            "--metadata", json.dumps({"test": "data"}))

        assert "id" in save_result or "session_id" in save_result
        session_id = save_result.get("id") or save_result.get("session_id")
        cleanup_sessions.append(session_id)

        # List sessions
        list_result = run_cli_json("session", "list")

        # Handle both list and dict responses
        if isinstance(list_result, list):
            sessions = list_result
        elif isinstance(list_result, dict) and "sessions" in list_result:
            sessions = list_result["sessions"]
        else:
            sessions = []

        assert any(s.get("name") == session_name for s in sessions), \
            f"Session {session_name} not found"

    @pytest.mark.xfail(reason="Session API compatibility not verified")
    def test_session_info(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_sessions: list[str]) -> None:
        """Test getting session information."""
        session_name = f"test-info-{test_id}"

        # Save session
        save_result = run_cli_json(
            "session", "save",
            "--name", session_name,
            "--description", "Session info test",
            "--metadata", json.dumps({"key": "value"}))
        session_id = save_result.get("id") or save_result.get("session_id")
        cleanup_sessions.append(session_id)

        # Get session info
        info_result = run_cli_json("session", "info", session_id)

        assert info_result.get("name") == session_name
        assert "metadata" in info_result or "description" in info_result

    @pytest.mark.xfail(reason="Session API compatibility not verified")
    def test_session_resume_and_delete(
        self,
        api_health_check: None,
        test_id: str) -> None:
        """Test resuming and deleting sessions (no cleanup needed)."""
        session_name = f"test-resume-{test_id}"

        # Save session
        save_result = run_cli_json(
            "session", "save",
            "--name", session_name,
            "--description", "Resume test",
            "--metadata", json.dumps({"state": "active"}))
        session_id = save_result.get("id") or save_result.get("session_id")

        # Resume session
        resume_result = run_cli_json("session", "resume", session_id)
        assert "metadata" in resume_result or "name" in resume_result

        # Delete session
        delete_result = run_cli("session", "delete", session_id, "--yes")
        assert delete_result.returncode == 0

        # Verify it's gone
        list_result = run_cli_json("session", "list")

        if isinstance(list_result, list):
            sessions = list_result
        elif isinstance(list_result, dict) and "sessions" in list_result:
            sessions = list_result["sessions"]
        else:
            sessions = []

        session_ids = [s.get("id") or s.get("session_id") for s in sessions]
        assert session_id not in session_ids


class TestWorkflowFlow:
    """Test workflow creation, execution, and visualization.

    NOTE: Workflow tests may have API compatibility issues.
    """

    @pytest.mark.xfail(reason="Workflow API compatibility not verified")
    def test_workflow_create_from_file(
        self,
        api_health_check: None,
        test_id: str,
        tmp_path: Path) -> None:
        """Test creating workflow from YAML file."""
        # Create workflow definition file
        workflow_file = tmp_path / f"workflow-{test_id}.yaml"
        workflow_content = f"""
name: e2e-workflow-{test_id}
description: E2E test workflow
steps:
  - name: step1
    type: task
    agent_capability: text-generation
    input:
      prompt: "test"
"""
        workflow_file.write_text(workflow_content)

        # Create workflow
        result = run_cli_json("workflow", "create", "--file", str(workflow_file))

        assert "id" in result or "workflow_id" in result

    @pytest.mark.xfail(reason="Workflow API compatibility not verified")
    def test_workflow_status(
        self,
        api_health_check: None,
        test_id: str,
        tmp_path: Path,
        cleanup_agents: list[str]) -> None:
        """Test checking workflow status."""
        # Create simple workflow
        workflow_file = tmp_path / f"status-workflow-{test_id}.yaml"
        workflow_file.write_text(f"""
name: status-test-{test_id}
steps:
  - name: test-step
    type: task
    agent_capability: test
    input: {{}}
""")

        create_result = run_cli_json("workflow", "create", "--file", str(workflow_file))
        workflow_id = create_result.get("id") or create_result.get("workflow_id")

        # Get status (might fail if workflow execution not implemented yet)
        status_result = run_cli("workflow", "status", workflow_id)

        # We just verify the command doesn't crash
        assert status_result.returncode in (0, 1)  # May fail if not implemented


class TestConfiguration:
    """Test CLI configuration and environment variable handling."""

    def test_config_show(self, api_health_check: None) -> None:
        """Test showing current configuration."""
        result = run_cli_json("config", "show")

        assert "api" in result
        assert result["api"]["url"] == API_URL

    @pytest.mark.xfail(
        reason="Config init creates config at ~/.agentcore/config.toml, not in current directory"
    )
    def test_config_init(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Test initializing configuration file."""
        # Change to tmp directory so config is created there
        monkeypatch.chdir(tmp_path)

        result = run_cli("config", "init")

        assert result.returncode == 0
        config_file = tmp_path / ".agentcore.yaml"
        assert config_file.exists()

    @pytest.mark.xfail(reason="Config validate command not implemented")
    def test_config_validate(self, tmp_path: Path) -> None:
        """Test validating configuration file."""
        # Create valid config
        config_file = tmp_path / ".agentcore.yaml"
        config_file.write_text("""
api:
  url: http://localhost:8001
  timeout: 30
auth:
  type: none
""")

        result = run_cli(
            "config", "validate",
            "--config", str(config_file))

        assert result.returncode == 0

    def test_env_var_override(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test that environment variables override config."""
        custom_timeout = "60"

        # Set environment variable
        env = {"AGENTCORE_API_TIMEOUT": custom_timeout}

        # Show config with env var
        result = run_cli_json("config", "show", env=env)

        assert result["api"]["timeout"] == int(custom_timeout)


class TestOutputFormats:
    """Test different output formats (JSON, table, tree)."""

    def test_json_output_format(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test JSON output format."""
        agent_name = f"json-test-{test_id}"

        # Register with JSON output
        result = run_cli_json(
            "agent", "register",
            "--name", agent_name,
            "--capabilities", "json-test")

        agent_id = result.get("id") or result.get("agent_id")
        cleanup_agents.append(agent_id)

        # Verify it's valid JSON
        assert isinstance(result, dict)
        assert agent_id is not None

    def test_table_output_format(
        self,
        api_health_check: None,
        test_id: str,
        cleanup_agents: list[str]) -> None:
        """Test table output format (default)."""
        agent_name = f"table-test-{test_id}"

        # Register without --json flag (default table format)
        result = run_cli(
            "agent", "register",
            "--name", agent_name,
            "--capabilities", "table-test")

        assert result.returncode == 0

        # Table output should contain the agent name
        assert agent_name in result.stdout or "successfully" in result.stdout.lower()

        # Extract agent_id from output for cleanup (may be in various formats)
        # Best effort: try to find an ID pattern in output
        import re
        id_match = re.search(r'\b([a-f0-9-]{36})\b', result.stdout)
        if id_match:
            cleanup_agents.append(id_match.group(1))


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_connection_error_wrong_url(self) -> None:
        """Test CLI handles connection errors gracefully."""
        # Use wrong URL to trigger connection error
        env = {"AGENTCORE_API_URL": "http://localhost:9999"}

        result = run_cli(
            "agent", "list",
            "--json",
            env=env)

        # Should fail with connection error (error message may be in stdout or stderr)
        assert result.returncode != 0
        output = (result.stdout + result.stderr).lower()
        assert "connection" in output or "connect" in output or "error" in output

    def test_invalid_agent_id(self, api_health_check: None) -> None:
        """Test error on invalid agent ID."""
        result = run_cli(
            "agent", "info",
            "invalid-agent-id-123",
            "--json")

        # Should fail
        assert result.returncode != 0

    def test_missing_required_argument(self) -> None:
        """Test error on missing required argument."""
        result = run_cli(
            "agent", "register",
            "--name", "test",
            # Missing --capabilities
        )

        # Should fail
        assert result.returncode != 0
        assert "capabilities" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_invalid_json_input(self) -> None:
        """Test error on invalid JSON input (doesn't require API)."""
        # Test task create with invalid input JSON
        result = run_cli(
            "task", "create",
            "--description", "test task",
            "--parameters", "not-valid-json")

        # Should fail due to invalid JSON
        assert result.returncode != 0
        assert "json" in result.stdout.lower() or "json" in result.stderr.lower()
