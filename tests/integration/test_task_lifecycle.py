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
        assert data["result"]["success"] is True
        assert data["result"]["task_id"] == sample_task_definition["task_id"]

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
        await async_client.post("/api/v1/jsonrpc", json=task_req)

        # Get summary
        summary_req = jsonrpc_request_template("task.summary")
        response = await async_client.post("/api/v1/jsonrpc", json=summary_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "by_status" in data["result"]
        assert data["result"]["total"] >= 1