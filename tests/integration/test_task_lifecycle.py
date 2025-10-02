"""
Integration Tests for Task Lifecycle

Tests for task creation, assignment, and execution.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestTaskLifecycle:
    """Test complete task lifecycle."""

    async def test_task_create(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_task_definition
    ):
        """Test task creation."""
        request = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["task_id"] == sample_task_definition["task_id"]
        assert data["result"]["status"] == "pending"
        assert "execution_id" in data["result"]

    async def test_task_get(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_task_definition
    ):
        """Test retrieving task details."""
        # Create task
        create_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        await async_client.post("/api/v1/jsonrpc", json=create_req)

        # Get task
        get_req = jsonrpc_request_template("task.get", {
            "task_id": sample_task_definition["task_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=get_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["task"]["task_id"] == sample_task_definition["task_id"]
        assert data["result"]["task"]["status"] == "pending"

    async def test_task_assign(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test task assignment to agent."""
        # Register agent
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        # Create task
        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        await async_client.post("/api/v1/jsonrpc", json=task_req)

        # Assign task
        assign_req = jsonrpc_request_template("task.assign", {
            "task_id": sample_task_definition["task_id"],
            "agent_id": sample_agent_card["agent_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=assign_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True

    async def test_task_start(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test starting task execution."""
        # Setup: register agent and create task
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        await async_client.post("/api/v1/jsonrpc", json=task_req)

        assign_req = jsonrpc_request_template("task.assign", {
            "task_id": sample_task_definition["task_id"],
            "agent_id": sample_agent_card["agent_id"]
        })
        await async_client.post("/api/v1/jsonrpc", json=assign_req)

        # Start task
        start_req = jsonrpc_request_template("task.start", {
            "task_id": sample_task_definition["task_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=start_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True

    async def test_task_complete(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_task_definition
    ):
        """Test task completion."""
        # Create task
        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        await async_client.post("/api/v1/jsonrpc", json=task_req)

        # Complete task
        complete_req = jsonrpc_request_template("task.complete", {
            "task_id": sample_task_definition["task_id"],
            "result": {"output": "Task completed successfully"}
        })
        response = await async_client.post("/api/v1/jsonrpc", json=complete_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True

    async def test_task_fail(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_task_definition
    ):
        """Test task failure."""
        # Create task
        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        await async_client.post("/api/v1/jsonrpc", json=task_req)

        # Fail task
        fail_req = jsonrpc_request_template("task.fail", {
            "task_id": sample_task_definition["task_id"],
            "error": "Test error message"
        })
        response = await async_client.post("/api/v1/jsonrpc", json=fail_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True

    async def test_task_query(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_task_definition
    ):
        """Test querying tasks by status."""
        # Create task
        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        await async_client.post("/api/v1/jsonrpc", json=task_req)

        # Query pending tasks
        query_req = jsonrpc_request_template("task.query", {
            "status": "pending"
        })
        response = await async_client.post("/api/v1/jsonrpc", json=query_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["count"] >= 1
        task_ids = [t["task_id"] for t in data["result"]["tasks"]]
        assert sample_task_definition["task_id"] in task_ids

    async def test_task_summary(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_task_definition
    ):
        """Test task summary statistics."""
        # Create task
        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        create_response = await async_client.post("/api/v1/jsonrpc", json=task_req)
        create_data = create_response.json()
        execution_id = create_data["result"]["execution_id"]

        # Get summary
        summary_req = jsonrpc_request_template("task.summary", {
            "execution_id": execution_id
        })
        response = await async_client.post("/api/v1/jsonrpc", json=summary_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["execution_id"] == execution_id
        assert data["result"]["status"] == "pending"

    async def test_task_progress_update(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test updating task progress."""
        # Setup: register agent and create running task
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        create_response = await async_client.post("/api/v1/jsonrpc", json=task_req)
        execution_id = create_response.json()["result"]["execution_id"]

        assign_req = jsonrpc_request_template("task.assign", {
            "execution_id": execution_id,
            "agent_id": sample_agent_card["agent_id"]
        })
        await async_client.post("/api/v1/jsonrpc", json=assign_req)

        start_req = jsonrpc_request_template("task.start", {
            "execution_id": execution_id
        })
        await async_client.post("/api/v1/jsonrpc", json=start_req)

        # Update progress
        progress_req = jsonrpc_request_template("task.update_progress", {
            "execution_id": execution_id,
            "percentage": 50.0,
            "current_step": "Processing data"
        })
        response = await async_client.post("/api/v1/jsonrpc", json=progress_req)

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["success"] is True
        assert data["result"]["percentage"] == 50.0
        assert data["result"]["current_step"] == "Processing data"

        # Verify progress was updated
        get_req = jsonrpc_request_template("task.get", {
            "execution_id": execution_id
        })
        get_response = await async_client.post("/api/v1/jsonrpc", json=get_req)
        task_data = get_response.json()
        assert task_data["result"]["progress_percentage"] == 50.0
        assert task_data["result"]["current_step"] == "Processing data"

    async def test_task_artifact_management(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test adding and retrieving task artifacts."""
        # Setup: create running task
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        create_response = await async_client.post("/api/v1/jsonrpc", json=task_req)
        execution_id = create_response.json()["result"]["execution_id"]

        assign_req = jsonrpc_request_template("task.assign", {
            "execution_id": execution_id,
            "agent_id": sample_agent_card["agent_id"]
        })
        await async_client.post("/api/v1/jsonrpc", json=assign_req)

        start_req = jsonrpc_request_template("task.start", {
            "execution_id": execution_id
        })
        await async_client.post("/api/v1/jsonrpc", json=start_req)

        # Add artifact
        artifact_req = jsonrpc_request_template("task.add_artifact", {
            "execution_id": execution_id,
            "name": "output_data.json",
            "type": "json",
            "content": {"key": "value", "count": 42},
            "metadata": {"source": "test"}
        })
        add_response = await async_client.post("/api/v1/jsonrpc", json=artifact_req)

        assert add_response.status_code == 200
        add_data = add_response.json()
        assert add_data["result"]["success"] is True
        assert add_data["result"]["artifact_name"] == "output_data.json"
        assert add_data["result"]["artifact_type"] == "json"

        # Get artifacts
        get_artifacts_req = jsonrpc_request_template("task.get_artifacts", {
            "execution_id": execution_id
        })
        get_response = await async_client.post("/api/v1/jsonrpc", json=get_artifacts_req)

        assert get_response.status_code == 200
        artifacts_data = get_response.json()
        assert artifacts_data["result"]["count"] == 1
        artifact = artifacts_data["result"]["artifacts"][0]
        assert artifact["name"] == "output_data.json"
        assert artifact["type"] == "json"
        assert artifact["content"] == {"key": "value", "count": 42}
        assert artifact["metadata"]["source"] == "test"

    async def test_task_status_transitions(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test task status transition validation."""
        # Create task (pending)
        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition,
            "auto_assign": False
        })
        create_response = await async_client.post("/api/v1/jsonrpc", json=task_req)
        execution_id = create_response.json()["result"]["execution_id"]

        # Check valid transitions from pending
        transitions_req = jsonrpc_request_template("task.status_transitions", {
            "execution_id": execution_id
        })
        response = await async_client.post("/api/v1/jsonrpc", json=transitions_req)

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["current_status"] == "pending"
        assert "assigned" in data["result"]["valid_transitions"]
        assert "cancelled" in data["result"]["valid_transitions"]
        assert data["result"]["is_terminal"] is False

        # Assign task
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        assign_req = jsonrpc_request_template("task.assign", {
            "execution_id": execution_id,
            "agent_id": sample_agent_card["agent_id"]
        })
        await async_client.post("/api/v1/jsonrpc", json=assign_req)

        # Check valid transitions from assigned
        transitions_req = jsonrpc_request_template("task.status_transitions", {
            "execution_id": execution_id
        })
        response = await async_client.post("/api/v1/jsonrpc", json=transitions_req)

        data = response.json()
        assert data["result"]["current_status"] == "assigned"
        assert "running" in data["result"]["valid_transitions"]
        assert "cancelled" in data["result"]["valid_transitions"]

    async def test_task_complete_with_artifacts(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test completing task with artifacts."""
        # Setup: create running task
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        create_response = await async_client.post("/api/v1/jsonrpc", json=task_req)
        execution_id = create_response.json()["result"]["execution_id"]

        assign_req = jsonrpc_request_template("task.assign", {
            "execution_id": execution_id,
            "agent_id": sample_agent_card["agent_id"]
        })
        await async_client.post("/api/v1/jsonrpc", json=assign_req)

        start_req = jsonrpc_request_template("task.start", {
            "execution_id": execution_id
        })
        await async_client.post("/api/v1/jsonrpc", json=start_req)

        # Complete with artifacts
        complete_req = jsonrpc_request_template("task.complete", {
            "execution_id": execution_id,
            "result_data": {"status": "success", "output": "Result data"},
            "artifacts": [
                {
                    "name": "report.txt",
                    "type": "text",
                    "content": "Final report content",
                    "metadata": {"format": "markdown"}
                }
            ]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=complete_req)

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["success"] is True

        # Verify task status and artifacts
        get_req = jsonrpc_request_template("task.get", {
            "execution_id": execution_id
        })
        get_response = await async_client.post("/api/v1/jsonrpc", json=get_req)
        task_data = get_response.json()
        assert task_data["result"]["status"] == "completed"
        assert task_data["result"]["progress_percentage"] == 100.0
        assert len(task_data["result"]["artifacts"]) == 1
        assert task_data["result"]["artifacts"][0]["name"] == "report.txt"

    async def test_invalid_artifact_type(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card,
        sample_task_definition
    ):
        """Test that invalid artifact types are rejected."""
        # Setup: create running task
        agent_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=agent_req)

        task_req = jsonrpc_request_template("task.create", {
            "task_definition": sample_task_definition
        })
        create_response = await async_client.post("/api/v1/jsonrpc", json=task_req)
        execution_id = create_response.json()["result"]["execution_id"]

        assign_req = jsonrpc_request_template("task.assign", {
            "execution_id": execution_id,
            "agent_id": sample_agent_card["agent_id"]
        })
        await async_client.post("/api/v1/jsonrpc", json=assign_req)

        start_req = jsonrpc_request_template("task.start", {
            "execution_id": execution_id
        })
        await async_client.post("/api/v1/jsonrpc", json=start_req)

        # Try to add artifact with invalid type
        artifact_req = jsonrpc_request_template("task.add_artifact", {
            "execution_id": execution_id,
            "name": "output.xyz",
            "type": "invalid_type",
            "content": "some content"
        })
        response = await async_client.post("/api/v1/jsonrpc", json=artifact_req)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Invalid artifact type" in data["error"]["message"]