"""
Agent Runtime Layer - Main Application Entry Point.

This module provides the FastAPI application for the Agent Runtime Layer,
handling secure agent execution with Docker-based containerization.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings
from .routers import agents
from .services.agent_lifecycle import AgentLifecycleManager
from .services.container_manager import ContainerManager

settings = get_settings()
logger = structlog.get_logger()

# Global service instances
container_manager: ContainerManager | None = None
lifecycle_manager: AgentLifecycleManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    global container_manager, lifecycle_manager

    # Startup
    logger.info(
        "agent_runtime_startup",
        port=settings.agent_runtime_port,
        kubernetes=settings.is_kubernetes,
        max_concurrent_agents=settings.max_concurrent_agents,
    )

    try:
        # Initialize container manager
        container_manager = ContainerManager()
        await container_manager.initialize()

        # Initialize lifecycle manager
        lifecycle_manager = AgentLifecycleManager(container_manager)

        # Initialize router services
        await agents.initialize_services(container_manager, lifecycle_manager)

        logger.info("agent_runtime_services_initialized")

    except Exception as e:
        logger.error("agent_runtime_startup_failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("agent_runtime_shutdown")

    try:
        if container_manager:
            await container_manager.close()
        logger.info("agent_runtime_services_closed")
    except Exception as e:
        logger.error("agent_runtime_shutdown_error", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="AgentCore Agent Runtime Layer",
    description="Secure, isolated execution environments for multi-philosophy AI agents",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
if settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Include routers
app.include_router(agents.router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "agent-runtime"}


@app.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check for Kubernetes."""
    return {"status": "ready", "service": "agent-runtime"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "AgentCore Agent Runtime Layer",
        "version": "0.1.0",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.agent_runtime_port,
        log_level=settings.log_level.lower(),
    )
