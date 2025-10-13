"""
CORS Middleware

Cross-Origin Resource Sharing configuration for web applications.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.config import settings


def setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the gateway.

    Allows cross-origin requests from configured origins with proper
    credentials and headers support.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-Request-ID"],
    )
