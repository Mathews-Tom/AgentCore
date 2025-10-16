"""Integration tests for training JSON-RPC API."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.training.training_jsonrpc import (
    handle_cancel,
    handle_get_status,
    handle_list_jobs,
    handle_start_grpo,
)


@pytest.fixture
def sample_training_data():
    """Create sample training data (minimum 100 queries)."""
    return [
        {
            "query": f"Test query {i}",
            "expected_outcome": {"success": True, "result": "test"},
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_config():
    """Create sample training configuration."""
    return {
        "n_iterations": 5,
        "batch_size": 16,
        "n_trajectories_per_query": 8,
        "learning_rate": 0.0001,
        "max_budget_usd": "5.00",
        "checkpoint_interval": 2,
        "max_steps_per_trajectory": 20,
        "gamma": 0.99,
    }


@pytest.mark.asyncio
async def test_start_grpo_success(sample_training_data, sample_config):
    """Test successful GRPO job start."""
    request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "test-agent",
            "training_queries": sample_training_data,
            "config": sample_config,
        },
    )

    response = await handle_start_grpo(request)

    assert response["success"] is True
    assert "job_id" in response
    assert response["agent_id"] == "test-agent"
    assert response["status"] in ["queued", "running", "completed"]
    assert response["total_iterations"] == 5
    assert response["budget_usd"] == 5.0


@pytest.mark.asyncio
async def test_start_grpo_with_defaults(sample_training_data):
    """Test GRPO job start with default configuration."""
    request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "test-agent",
            "training_queries": sample_training_data,
            # No config - should use defaults
        },
    )

    response = await handle_start_grpo(request)

    assert response["success"] is True
    assert response["total_iterations"] == 10  # Default
    assert response["budget_usd"] == 10.0  # Default


@pytest.mark.asyncio
async def test_start_grpo_missing_agent_id(sample_training_data):
    """Test error handling for missing agent_id."""
    request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "training_queries": sample_training_data,
            # Missing agent_id
        },
    )

    with pytest.raises(ValueError, match="Missing required parameter: agent_id"):
        await handle_start_grpo(request)


@pytest.mark.asyncio
async def test_start_grpo_missing_training_queries(sample_config):
    """Test error handling for missing training_queries."""
    request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "test-agent",
            "config": sample_config,
            # Missing training_queries
        },
    )

    with pytest.raises(ValueError, match="Missing required parameter: training_queries"):
        await handle_start_grpo(request)


@pytest.mark.asyncio
async def test_start_grpo_insufficient_queries():
    """Test error handling for insufficient training queries."""
    request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "test-agent",
            "training_queries": [
                {"query": "test", "expected_outcome": {"success": True}}
            ],  # Only 1 query
        },
    )

    with pytest.raises(ValueError, match="must contain at least 100 queries"):
        await handle_start_grpo(request)


@pytest.mark.asyncio
async def test_start_grpo_invalid_params_type():
    """Test error handling for invalid params type.

    NOTE: Pydantic validates params type before handler is called,
    so we test with valid dict but missing required fields.
    """
    request = JsonRpcRequest(
        method="training.start_grpo",
        params={"invalid": "data"},  # Valid dict but missing required fields
    )

    with pytest.raises(ValueError, match="Missing required parameter: agent_id"):
        await handle_start_grpo(request)


@pytest.mark.asyncio
async def test_get_status_success(sample_training_data, sample_config):
    """Test successful job status retrieval."""
    # First create a job
    start_request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "test-agent",
            "training_queries": sample_training_data,
            "config": sample_config,
        },
    )

    start_response = await handle_start_grpo(start_request)
    job_id = start_response["job_id"]

    # Get status
    status_request = JsonRpcRequest(
        method="training.get_status",
        params={"job_id": job_id},
    )

    response = await handle_get_status(status_request)

    assert response["success"] is True
    assert response["job_id"] == job_id
    assert response["agent_id"] == "test-agent"
    assert "status" in response
    assert "progress" in response
    assert "metrics" in response
    assert "cost" in response


@pytest.mark.asyncio
async def test_get_status_missing_job_id():
    """Test error handling for missing job_id."""
    request = JsonRpcRequest(
        method="training.get_status",
        params={},  # Missing job_id
    )

    with pytest.raises(ValueError, match="Parameter required: job_id"):
        await handle_get_status(request)


@pytest.mark.asyncio
async def test_get_status_invalid_job_id():
    """Test error handling for invalid job_id format."""
    request = JsonRpcRequest(
        method="training.get_status",
        params={"job_id": "not-a-uuid"},
    )

    with pytest.raises(ValueError, match="Invalid job_id format"):
        await handle_get_status(request)


@pytest.mark.asyncio
async def test_get_status_job_not_found():
    """Test error handling for non-existent job."""
    fake_uuid = "12345678-1234-1234-1234-123456789012"

    request = JsonRpcRequest(
        method="training.get_status",
        params={"job_id": fake_uuid},
    )

    with pytest.raises(ValueError, match="Job .* not found"):
        await handle_get_status(request)


@pytest.mark.asyncio
async def test_cancel_success(sample_training_data, sample_config):
    """Test successful job cancellation."""
    # Create and start a job with many iterations
    long_config = {**sample_config, "n_iterations": 1000}

    start_request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "test-agent",
            "training_queries": sample_training_data,
            "config": long_config,
        },
    )

    start_response = await handle_start_grpo(start_request)
    job_id = start_response["job_id"]

    # Cancel the job
    cancel_request = JsonRpcRequest(
        method="training.cancel",
        params={"job_id": job_id},
    )

    response = await handle_cancel(cancel_request)

    assert response["success"] is True
    assert response["job_id"] == job_id
    # Status should be cancelled or completed (if it finished before cancel)
    assert response["status"] in ["cancelled", "completed"]


@pytest.mark.asyncio
async def test_cancel_missing_job_id():
    """Test error handling for missing job_id in cancel."""
    request = JsonRpcRequest(
        method="training.cancel",
        params={},  # Missing job_id
    )

    with pytest.raises(ValueError, match="Parameter required: job_id"):
        await handle_cancel(request)


@pytest.mark.asyncio
async def test_cancel_invalid_job_id():
    """Test error handling for invalid job_id format in cancel."""
    request = JsonRpcRequest(
        method="training.cancel",
        params={"job_id": "not-a-uuid"},
    )

    with pytest.raises(ValueError, match="Invalid job_id format"):
        await handle_cancel(request)


@pytest.mark.asyncio
async def test_cancel_job_not_found():
    """Test error handling for canceling non-existent job."""
    fake_uuid = "12345678-1234-1234-1234-123456789012"

    request = JsonRpcRequest(
        method="training.cancel",
        params={"job_id": fake_uuid},
    )

    with pytest.raises(ValueError, match="Job .* not found"):
        await handle_cancel(request)


@pytest.mark.asyncio
async def test_list_jobs_all(sample_training_data, sample_config):
    """Test listing all jobs."""
    # Create jobs for different agents
    for i in range(3):
        request = JsonRpcRequest(
            method="training.start_grpo",
            params={
                "agent_id": f"agent-{i}",
                "training_queries": sample_training_data,
                "config": sample_config,
            },
        )
        await handle_start_grpo(request)

    # List all jobs
    list_request = JsonRpcRequest(
        method="training.list_jobs",
        params={},
    )

    response = await handle_list_jobs(list_request)

    assert response["success"] is True
    assert "jobs" in response
    assert response["count"] >= 3  # At least the 3 we created


@pytest.mark.asyncio
async def test_list_jobs_filtered_by_agent(sample_training_data, sample_config):
    """Test listing jobs filtered by agent_id."""
    # Create jobs for different agents
    for i in range(2):
        request = JsonRpcRequest(
            method="training.start_grpo",
            params={
                "agent_id": f"filter-agent-{i}",
                "training_queries": sample_training_data,
                "config": sample_config,
            },
        )
        await handle_start_grpo(request)

    # List jobs for specific agent
    list_request = JsonRpcRequest(
        method="training.list_jobs",
        params={"agent_id": "filter-agent-0"},
    )

    response = await handle_list_jobs(list_request)

    assert response["success"] is True
    assert "jobs" in response
    # Should only have jobs for filter-agent-0
    for job in response["jobs"]:
        assert job["agent_id"] == "filter-agent-0"


@pytest.mark.asyncio
async def test_list_jobs_empty():
    """Test listing jobs when none exist for agent."""
    request = JsonRpcRequest(
        method="training.list_jobs",
        params={"agent_id": "nonexistent-agent"},
    )

    response = await handle_list_jobs(request)

    assert response["success"] is True
    assert response["jobs"] == []
    assert response["count"] == 0


@pytest.mark.asyncio
async def test_list_jobs_no_params():
    """Test listing jobs with no params."""
    request = JsonRpcRequest(
        method="training.list_jobs",
        params=None,  # No params
    )

    response = await handle_list_jobs(request)

    assert response["success"] is True
    assert "jobs" in response
    assert "count" in response


@pytest.mark.asyncio
async def test_end_to_end_job_lifecycle(sample_training_data, sample_config):
    """Test complete job lifecycle: create, status, cancel."""
    # 1. Start job
    start_request = JsonRpcRequest(
        method="training.start_grpo",
        params={
            "agent_id": "lifecycle-agent",
            "training_queries": sample_training_data,
            "config": sample_config,
        },
    )

    start_response = await handle_start_grpo(start_request)
    assert start_response["success"] is True
    job_id = start_response["job_id"]

    # 2. Check status
    status_request = JsonRpcRequest(
        method="training.get_status",
        params={"job_id": job_id},
    )

    status_response = await handle_get_status(status_request)
    assert status_response["success"] is True
    assert status_response["job_id"] == job_id

    # 3. List jobs (should include this one)
    list_request = JsonRpcRequest(
        method="training.list_jobs",
        params={"agent_id": "lifecycle-agent"},
    )

    list_response = await handle_list_jobs(list_request)
    assert list_response["success"] is True
    assert any(job["job_id"] == job_id for job in list_response["jobs"])

    # 4. Cancel or wait for completion
    if status_response["status"] == "running":
        cancel_request = JsonRpcRequest(
            method="training.cancel",
            params={"job_id": job_id},
        )

        cancel_response = await handle_cancel(cancel_request)
        assert cancel_response["success"] is True
