"""
DSPy Optimization Service - FastAPI Application

Production-ready REST API for DSPy optimization workflows with health checks,
metrics, and A2A protocol integration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
)
from agentcore.dspy_optimization.pipeline import OptimizationPipeline
from agentcore.dspy_optimization.tracking import MLflowConfig


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    version: str


class OptimizationJobResponse(BaseModel):
    """Response for optimization job submission"""

    job_id: str
    status: OptimizationStatus
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager"""
    # Initialize pipeline
    mlflow_config = MLflowConfig(
        tracking_uri="http://mlflow-service:5000", experiment_name="dspy-optimization"
    )
    app.state.pipeline = OptimizationPipeline(
        llm=None, mlflow_config=mlflow_config, enable_tracking=True
    )

    yield

    # Cleanup
    if hasattr(app.state, "pipeline"):
        del app.state.pipeline


app = FastAPI(
    title="DSPy Optimization Service",
    description="Production API for DSPy optimization workflows",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/api/v1/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe"""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/api/v1/health/ready", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Kubernetes readiness probe"""
    if not hasattr(app.state, "pipeline"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    return HealthResponse(status="ready", version="0.1.0")


@app.post("/api/v1/optimize", response_model=OptimizationJobResponse)
async def optimize(request: OptimizationRequest) -> OptimizationJobResponse:
    """
    Submit optimization job

    Args:
        request: Optimization request with target, objectives, and constraints

    Returns:
        Job submission response with job ID and status
    """
    try:
        pipeline: OptimizationPipeline = app.state.pipeline
        result: OptimizationResult = await pipeline.optimize(request)

        return OptimizationJobResponse(
            job_id=result.request_id,
            status=result.status,
            message="Optimization job submitted successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {e!s}",
        ) from e


@app.get("/api/v1/optimize/{job_id}", response_model=OptimizationResult)
async def get_optimization_result(job_id: str) -> OptimizationResult:
    """
    Retrieve optimization result

    Args:
        job_id: Optimization job ID

    Returns:
        Optimization result with metrics and artifacts
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Result retrieval not yet implemented",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentcore.dspy_optimization.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
