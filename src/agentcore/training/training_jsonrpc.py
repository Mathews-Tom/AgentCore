"""
Training JSON-RPC Methods

JSON-RPC 2.0 methods for GRPO training operations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.training.job_manager import TrainingJobManager
from agentcore.training.models import (
    GRPOConfig as GRPOConfigModel,
)
from agentcore.training.models import (
    TrainingQuery,
)

logger = structlog.get_logger()

# Global job manager instance
_job_manager: TrainingJobManager | None = None


def get_job_manager() -> TrainingJobManager:
    """Get or create global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = TrainingJobManager()
    return _job_manager


@register_jsonrpc_method("training.start_grpo")
async def handle_start_grpo(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Start GRPO training job for an agent.

    Method: training.start_grpo
    Params:
        - agent_id: string (required)
        - training_queries: array of objects (required)
          - query: string
          - expected_outcome: object
        - config: object (optional)
          - n_iterations: int (default: 10)
          - batch_size: int (default: 16)
          - n_trajectories_per_query: int (default: 8)
          - learning_rate: float (default: 0.0001)
          - max_budget_usd: string/decimal (default: "10.00")
          - checkpoint_interval: int (default: 5)
          - max_steps_per_trajectory: int (default: 20)
          - gamma: float (default: 0.99)

    Returns:
        Training job details
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: agent_id, training_queries")

    # Extract required parameters
    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    training_queries_data = request.params.get("training_queries")
    if not training_queries_data:
        raise ValueError("Missing required parameter: training_queries")

    if not isinstance(training_queries_data, list):
        raise ValueError("training_queries must be an array")

    if len(training_queries_data) < 100:
        raise ValueError(
            f"training_queries must contain at least 100 queries, got {len(training_queries_data)}"
        )

    # Parse training queries
    training_queries = [
        TrainingQuery(
            query=q["query"],
            expected_outcome=q["expected_outcome"],
        )
        for q in training_queries_data
    ]

    # Parse configuration
    config_data = request.params.get("config", {})
    config = GRPOConfigModel(
        n_iterations=config_data.get("n_iterations", 10),
        batch_size=config_data.get("batch_size", 16),
        n_trajectories_per_query=config_data.get("n_trajectories_per_query", 8),
        learning_rate=config_data.get("learning_rate", 0.0001),
        max_budget_usd=Decimal(
            str(config_data.get("max_budget_usd", "10.00"))
        ),  # Convert to Decimal
        checkpoint_interval=config_data.get("checkpoint_interval", 5),
        max_steps_per_trajectory=config_data.get("max_steps_per_trajectory", 20),
        gamma=config_data.get("gamma", 0.99),
    )

    # Create job
    job_manager = get_job_manager()
    job = await job_manager.create_job(
        agent_id=agent_id,
        training_data=training_queries,
        config=config,
    )

    # Start job execution
    await job_manager.start_job(job.job_id)

    logger.info(
        "Training job started via JSON-RPC",
        job_id=str(job.job_id),
        agent_id=agent_id,
        n_queries=len(training_queries),
        n_iterations=config.n_iterations,
    )

    return {
        "success": True,
        "job_id": str(job.job_id),
        "agent_id": agent_id,
        "status": job.status.value,
        "total_iterations": job.total_iterations,
        "budget_usd": float(job.budget_usd),
        "created_at": datetime.now(UTC).isoformat(),
    }


@register_jsonrpc_method("training.get_status")
async def handle_get_status(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get training job status and metrics.

    Method: training.get_status
    Params:
        - job_id: string (required) - UUID of training job

    Returns:
        Job status and metrics
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: job_id")

    job_id_str = request.params.get("job_id")
    if not job_id_str:
        raise ValueError("Missing required parameter: job_id")

    try:
        job_id = UUID(job_id_str)
    except ValueError as e:
        raise ValueError(f"Invalid job_id format: {job_id_str}") from e

    # Get job status
    job_manager = get_job_manager()
    status = job_manager.get_job_status(job_id)

    logger.debug("Training job status retrieved via JSON-RPC", job_id=job_id_str)

    return {
        "success": True,
        **status,
    }


@register_jsonrpc_method("training.cancel")
async def handle_cancel(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Cancel a running training job.

    Method: training.cancel
    Params:
        - job_id: string (required) - UUID of training job

    Returns:
        Cancellation confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: job_id")

    job_id_str = request.params.get("job_id")
    if not job_id_str:
        raise ValueError("Missing required parameter: job_id")

    try:
        job_id = UUID(job_id_str)
    except ValueError as e:
        raise ValueError(f"Invalid job_id format: {job_id_str}") from e

    # Cancel job
    job_manager = get_job_manager()
    await job_manager.cancel_job(job_id)

    # Get updated status
    job = job_manager.get_job(job_id)

    logger.info("Training job cancelled via JSON-RPC", job_id=job_id_str)

    return {
        "success": True,
        "job_id": str(job.job_id),
        "status": job.status.value,
        "cancelled_at": datetime.now(UTC).isoformat(),
    }


@register_jsonrpc_method("training.list_jobs")
async def handle_list_jobs(request: JsonRpcRequest) -> dict[str, Any]:
    """
    List training jobs, optionally filtered by agent_id.

    Method: training.list_jobs
    Params:
        - agent_id: string (optional) - Filter by agent ID

    Returns:
        List of training jobs
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    agent_id = params.get("agent_id")

    # List jobs
    job_manager = get_job_manager()
    jobs = job_manager.list_jobs(agent_id=agent_id)

    logger.debug(
        "Training jobs listed via JSON-RPC",
        agent_id=agent_id,
        count=len(jobs),
    )

    return {
        "success": True,
        "jobs": jobs,
        "count": len(jobs),
    }


# Log registration on import
logger.info(
    "Training JSON-RPC methods registered",
    methods=[
        "training.start_grpo",
        "training.get_status",
        "training.cancel",
        "training.list_jobs",
    ],
)
