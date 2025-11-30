"""
A2A Protocol Layer - FastAPI Application

Main application entry point for the Agent2Agent protocol implementation.
Provides JSON-RPC 2.0 compliant endpoints for agent communication.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.database import close_db, init_db
from agentcore.a2a_protocol.middleware import setup_middleware
from agentcore.a2a_protocol.routers import health, jsonrpc, websocket, wellknown
from agentcore.a2a_protocol.services.coordination_service import coordination_service
from agentcore.a2a_protocol.services.memory.storage_backend import (
    close_storage_backend,
    initialize_storage_backend,
)
from agentcore.a2a_protocol.services.memory.working_memory import (
    close_working_memory,
    initialize_working_memory,
)

# Import JSON-RPC methods to register them
from agentcore.a2a_protocol.services import (  # noqa: F401
    agent_jsonrpc,
    coordination_jsonrpc,
    event_jsonrpc,
    health_jsonrpc,
    llm_jsonrpc,
    memory_jsonrpc,
    routing_jsonrpc,
    security_jsonrpc,
    session_jsonrpc,
    task_jsonrpc,
    workflow_jsonrpc,
)

# Import reasoning JSON-RPC methods
# Import directly to avoid circular imports
from agentcore.reasoning.services import (  # noqa: F401
    reasoning_execute_jsonrpc,
    reasoning_jsonrpc,
)

# Import training JSON-RPC methods
from agentcore.training.services import training_jsonrpc  # noqa: F401

# Import modular agent JSON-RPC methods
try:
    from agentcore.modular import jsonrpc as modular_jsonrpc  # noqa: F401
except ImportError:
    # Modular agent optional
    modular_jsonrpc = None

# Import agent runtime tool JSON-RPC methods
try:
    from agentcore.agent_runtime.jsonrpc import tools_jsonrpc
    from agentcore.agent_runtime.tools.startup import initialize_tool_system
except ImportError:
    # Agent runtime tools optional
    tools_jsonrpc = None
    initialize_tool_system = None

# Import ACE JSON-RPC methods
try:
    from agentcore.ace import jsonrpc as ace_jsonrpc

    # Import ACE integration module for runtime intervention support
    from agentcore.ace.integration import runtime_interface as ace_runtime_integration
except ImportError:
    # ACE optional
    ace_jsonrpc = None
    ace_runtime_integration = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger = structlog.get_logger()

    try:
        logger.info("Starting A2A Protocol Layer", version="0.1.0")

        # Initialize database connection
        await init_db()
        logger.info("Database initialized")

        # Initialize memory storage backends (Qdrant, Neo4j)
        try:
            await initialize_storage_backend()
            logger.info("Memory storage backends initialized (PostgreSQL, Qdrant, Neo4j)")
        except Exception as e:
            logger.warning(
                "Memory storage backends initialization failed, using fallback mode",
                error=str(e),
            )

        # Initialize Redis working memory
        try:
            await initialize_working_memory()
            logger.info("Redis working memory initialized")
        except Exception as e:
            logger.warning(
                "Redis working memory initialization failed, using fallback mode",
                error=str(e),
            )

        # Start coordination service cleanup task
        if settings.COORDINATION_ENABLE_REP:
            coordination_service.start_cleanup_task()
            logger.info("Coordination cleanup task started")

        # Initialize tool system and register built-in tools
        if initialize_tool_system is not None:
            try:
                tool_registry = await initialize_tool_system()
                app.state.tool_registry = tool_registry
                logger.info(
                    "Tool system initialized",
                    total_tools=len(tool_registry.list_all()),
                )
            except Exception as e:
                logger.warning(
                    "Tool system initialization failed, continuing without tools",
                    error=str(e),
                )
        else:
            logger.info("Tool system not available (agent_runtime not installed)")

        # Register PEVG modules as A2A agents
        try:
            from agentcore.modular.registration import register_module_agents
            discovery_urls = await register_module_agents()
            logger.info(
                "PEVG modules registered as A2A agents",
                modules=list(discovery_urls.keys()),
            )
        except Exception as e:
            logger.warning(
                "PEVG module registration failed, continuing without module agents",
                error=str(e),
            )

        # TODO: Setup WebSocket manager
        yield
    finally:
        logger.info("Shutting down A2A Protocol Layer")

        # Unregister PEVG modules
        try:
            from agentcore.modular.registration import unregister_module_agents
            await unregister_module_agents()
            logger.info("PEVG modules unregistered")
        except Exception as e:
            logger.warning("PEVG module unregistration failed", error=str(e))

        # Stop coordination service cleanup task
        if settings.COORDINATION_ENABLE_REP:
            await coordination_service.stop_cleanup_task()
            logger.info("Coordination cleanup task stopped")

        # Close memory storage backends
        try:
            await close_storage_backend()
            logger.info("Memory storage backends closed")
        except Exception as e:
            logger.warning("Memory storage backends cleanup failed", error=str(e))

        # Close Redis working memory
        try:
            await close_working_memory()
            logger.info("Redis working memory closed")
        except Exception as e:
            logger.warning("Redis working memory cleanup failed", error=str(e))

        # Cleanup database connections
        await close_db()
        logger.info("Database connections closed")


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
