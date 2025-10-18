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

from gateway.auth.jwt import jwt_manager
from gateway.auth.oauth.registry import initialize_oauth_providers
from gateway.auth.oauth.state import oauth_state_manager
from gateway.auth.session import session_manager
from gateway.config import settings
from gateway.middleware.cors import setup_cors
from gateway.middleware.ddos_protection import DDoSConfig, DDoSProtector
from gateway.middleware.logging import logging_middleware
from gateway.middleware.metrics import metrics_middleware
from gateway.middleware.rate_limit import RateLimitMiddleware
from gateway.middleware.rate_limiter import (
    RateLimitAlgorithmType,
    RateLimitPolicy,
    RateLimiter,
)
from gateway.routes import auth, health, oauth


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

        # Initialize JWT manager
        await jwt_manager.initialize()
        logger.info("JWT manager initialized")

        # Initialize session manager
        await session_manager.initialize()
        logger.info("Session manager initialized")

        # Initialize rate limiter
        if settings.RATE_LIMIT_ENABLED:
            rate_limiter = RateLimiter(
                redis_url=settings.RATE_LIMIT_REDIS_URL,
                default_algorithm=RateLimitAlgorithmType(settings.RATE_LIMIT_ALGORITHM),
            )
            await rate_limiter.initialize()
            app.state.rate_limiter = rate_limiter
            logger.info("Rate limiter initialized")

            # Initialize DDoS protector
            if settings.DDOS_PROTECTION_ENABLED:
                ddos_config = DDoSConfig(
                    global_requests_per_second=settings.DDOS_GLOBAL_REQUESTS_PER_SECOND,
                    global_requests_per_minute=settings.DDOS_GLOBAL_REQUESTS_PER_MINUTE,
                    ip_requests_per_second=settings.DDOS_IP_REQUESTS_PER_SECOND,
                    ip_requests_per_minute=settings.DDOS_IP_REQUESTS_PER_MINUTE,
                    burst_threshold_multiplier=settings.DDOS_BURST_THRESHOLD_MULTIPLIER,
                    burst_window_seconds=settings.DDOS_BURST_WINDOW_SECONDS,
                    enable_auto_blocking=settings.DDOS_AUTO_BLOCKING_ENABLED,
                    auto_block_duration_seconds=settings.DDOS_AUTO_BLOCK_DURATION_SECONDS,
                    auto_block_threshold=settings.DDOS_AUTO_BLOCK_THRESHOLD,
                )
                ddos_protector = DDoSProtector(rate_limiter, ddos_config)
                app.state.ddos_protector = ddos_protector
                logger.info("DDoS protector initialized")

        # Initialize OAuth state manager
        if settings.OAUTH_ENABLED:
            await oauth_state_manager.initialize()
            logger.info("OAuth state manager initialized")

            # Initialize OAuth providers
            initialize_oauth_providers()
            logger.info("OAuth providers initialized")

        # Future: Initialize backend service connections

        yield
    finally:
        logger.info("Shutting down API Gateway")

        # Cleanup rate limiter
        if settings.RATE_LIMIT_ENABLED and hasattr(app.state, "rate_limiter"):
            await app.state.rate_limiter.close()
            logger.info("Rate limiter closed")

        # Cleanup session manager
        await session_manager.close()
        logger.info("Session manager closed")

        # Cleanup OAuth state manager
        if settings.OAUTH_ENABLED:
            await oauth_state_manager.close()
            logger.info("OAuth state manager closed")


def _setup_rate_limiting(app: FastAPI) -> None:
    """Setup rate limiting middleware with deferred initialization."""
    if not settings.RATE_LIMIT_ENABLED:
        return

    # Build default policies from configuration
    default_policies = {
        "client_ip": RateLimitPolicy(
            limit=settings.RATE_LIMIT_CLIENT_IP_LIMIT,
            window_seconds=settings.RATE_LIMIT_CLIENT_IP_WINDOW,
            algorithm=RateLimitAlgorithmType(settings.RATE_LIMIT_ALGORITHM),
        ),
        "endpoint": RateLimitPolicy(
            limit=settings.RATE_LIMIT_ENDPOINT_LIMIT,
            window_seconds=settings.RATE_LIMIT_ENDPOINT_WINDOW,
            algorithm=RateLimitAlgorithmType(settings.RATE_LIMIT_ALGORITHM),
        ),
        "user": RateLimitPolicy(
            limit=settings.RATE_LIMIT_USER_LIMIT,
            window_seconds=settings.RATE_LIMIT_USER_WINDOW,
            algorithm=RateLimitAlgorithmType(settings.RATE_LIMIT_ALGORITHM),
        ),
    }

    # Middleware that uses app state (initialized during lifespan)
    @app.middleware("http")
    async def rate_limit_middleware_wrapper(request, call_next):
        if hasattr(app.state, "rate_limiter"):
            middleware = RateLimitMiddleware(
                app=app,
                rate_limiter=app.state.rate_limiter,
                ddos_protector=getattr(app.state, "ddos_protector", None),
                default_policies=default_policies,
                enable_ddos_protection=settings.DDOS_PROTECTION_ENABLED,
                exempt_paths=settings.RATE_LIMIT_EXEMPT_PATHS,
            )
            return await middleware.dispatch(request, call_next)
        else:
            # Rate limiter not yet initialized, skip rate limiting
            return await call_next(request)


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

    # Setup rate limiting middleware (deferred initialization)
    _setup_rate_limiting(app)

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
    app.include_router(auth.router, tags=["authentication"])

    # Include OAuth router if enabled
    if settings.OAUTH_ENABLED:
        app.include_router(oauth.router, tags=["oauth"])

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
