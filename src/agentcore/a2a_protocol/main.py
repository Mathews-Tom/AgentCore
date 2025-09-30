"""
A2A Protocol Layer - FastAPI Application

Main application entry point for the Agent2Agent protocol implementation.
Provides JSON-RPC 2.0 compliant endpoints for agent communication.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.routers import health, jsonrpc, wellknown, websocket
from agentcore.a2a_protocol.middleware import setup_middleware
from agentcore.a2a_protocol.database import init_db, close_db
# Import JSON-RPC methods to register them
from agentcore.a2a_protocol.services import (
    agent_jsonrpc,
    task_jsonrpc,
    routing_jsonrpc,
    security_jsonrpc,
    event_jsonrpc,
    health_jsonrpc,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger = structlog.get_logger()

    try:
        logger.info("Starting A2A Protocol Layer", version="0.1.0")

        # Initialize database connection
        await init_db()
        logger.info("Database initialized")

        # TODO: Initialize Redis connection
        # TODO: Setup WebSocket manager
        yield
    finally:
        logger.info("Shutting down A2A Protocol Layer")

        # Cleanup database connections
        await close_db()
        logger.info("Database connections closed")

        # TODO: Cleanup Redis connections


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="AgentCore A2A Protocol Layer",
        description="Agent2Agent communication infrastructure implementing Google's A2A protocol v0.2",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Setup middleware
    setup_middleware(app)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(jsonrpc.router, prefix="/api/v1", tags=["jsonrpc"])
    app.include_router(wellknown.router, tags=["discovery"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])

    # Prometheus instrumentation
    if settings.ENABLE_METRICS:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentcore.a2a_protocol.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True,
    )