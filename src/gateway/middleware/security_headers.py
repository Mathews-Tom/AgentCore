"""
Security Headers Middleware

Implements comprehensive security headers following OWASP best practices.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from gateway.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add comprehensive security headers to all responses.

    Implements OWASP recommended security headers including:
    - Strict-Transport-Security (HSTS)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Content-Security-Policy
    - Referrer-Policy
    - Permissions-Policy
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # HSTS - Force HTTPS for 1 year with subdomain inclusion
        if settings.SECURITY_HSTS_ENABLED:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={settings.SECURITY_HSTS_MAX_AGE}; "
                "includeSubDomains; preload"
            )

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = settings.SECURITY_X_FRAME_OPTIONS

        # XSS Protection (for legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        if settings.SECURITY_CSP_ENABLED:
            response.headers["Content-Security-Policy"] = settings.SECURITY_CSP_POLICY

        # Referrer Policy
        response.headers["Referrer-Policy"] = settings.SECURITY_REFERRER_POLICY

        # Permissions Policy (formerly Feature-Policy)
        if settings.SECURITY_PERMISSIONS_POLICY:
            response.headers["Permissions-Policy"] = settings.SECURITY_PERMISSIONS_POLICY

        # Remove server identification
        # Note: MutableHeaders doesn't support .pop(), use del instead
        if "Server" in response.headers:
            del response.headers["Server"]

        # Add custom security headers
        for header, value in settings.SECURITY_CUSTOM_HEADERS.items():
            response.headers[header] = value

        return response
