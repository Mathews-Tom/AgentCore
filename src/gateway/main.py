"""
Gateway Layer - FastAPI Application

Main application entry point for the AgentCore API Gateway.
Provides unified entry point for all external interactions with AgentCore.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response

from gateway.config import settings
from gateway.middleware.cors import setup_cors
from gateway.middleware.logging import logging_middleware
from gateway.middleware.metrics import metrics_middleware
from gateway.routes import health


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger = structlog.get_logger()

    try:
        logger.info(
            "Starting API Gateway",
            version=settings.GATEWAY_VERSION,
            name=settings.GATEWAY_NAME
        )

        # Future: Initialize backend service connections
        # Future: Setup rate limiting
        # Future: Initialize authentication providers

        yield
    finally:
        logger.info("Shutting down API Gateway")

        # Future: Cleanup connections


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.GATEWAY_NAME,
        description="High-performance API gateway for AgentCore providing unified entry point for all external interactions",
        version=settings.GATEWAY_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Setup CORS middleware
    setup_cors(app)

    # Add logging middleware
    @app.middleware("http")
    async def add_logging_middleware(request, call_next):
        return await logging_middleware(request, call_next)

    # Add metrics middleware if enabled
    if settings.ENABLE_METRICS:
        @app.middleware("http")
        async def add_metrics_middleware(request, call_next):
            return await metrics_middleware(request, call_next)

    # Include routers
    app.include_router(health.router, tags=["health"])

    # Prometheus instrumentation
    if settings.ENABLE_METRICS:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
