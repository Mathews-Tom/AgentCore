"""
Rate Limiting Middleware

FastAPI middleware for automatic rate limiting with RFC 6585 compliant headers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from gateway.middleware.ddos_protection import DDoSProtector, ThreatLevel
from gateway.middleware.rate_limiter import (
    RateLimiter,
    RateLimitPolicy,
    RateLimitResult,
    RateLimitType,
)

logger = structlog.get_logger()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Enforces rate limits per client IP, endpoint, and authenticated user.
    Adds RFC 6585 compliant rate limit headers to responses.
    """

    def __init__(
        self,
        app: Any,
        rate_limiter: RateLimiter,
        ddos_protector: DDoSProtector | None = None,
        default_policies: dict[str, RateLimitPolicy] | None = None,
        endpoint_policies: dict[str, RateLimitPolicy] | None = None,
        enable_ddos_protection: bool = True,
        exempt_paths: list[str] | None = None,
    ) -> None:
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application instance
            rate_limiter: RateLimiter instance
            ddos_protector: DDoS protector (optional, created if not provided)
            default_policies: Default policies for different limit types
            endpoint_policies: Per-endpoint rate limit policies
            enable_ddos_protection: Enable DDoS protection checks
            exempt_paths: Paths exempt from rate limiting (e.g., health checks)
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.ddos_protector = ddos_protector
        self.enable_ddos_protection = enable_ddos_protection
        self.exempt_paths = exempt_paths or ["/health", "/metrics", "/.well-known/"]

        # Default policies
        self.default_policies = default_policies or {
            "client_ip": RateLimitPolicy(
                limit=1000,
                window_seconds=60,
            ),
            "endpoint": RateLimitPolicy(
                limit=100,
                window_seconds=60,
            ),
            "user": RateLimitPolicy(
                limit=5000,
                window_seconds=60,
            ),
        }

        # Per-endpoint policies
        self.endpoint_policies = endpoint_policies or {}

    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if path is exempt from rate limiting.

        Args:
            path: Request path

        Returns:
            True if path is exempt, False otherwise
        """
        return any(path.startswith(exempt) for exempt in self.exempt_paths)

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.

        Handles X-Forwarded-For and X-Real-IP headers for proxy scenarios.

        Args:
            request: FastAPI request object

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP (client IP)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _get_user_id(self, request: Request) -> str | None:
        """
        Extract authenticated user ID from request.

        Args:
            request: FastAPI request object

        Returns:
            User ID if authenticated, None otherwise
        """
        # Check if user is authenticated (from JWT middleware)
        if hasattr(request.state, "user") and request.state.user:
            return str(request.state.user.get("id") or request.state.user.get("sub"))

        return None

    def _get_endpoint_key(self, request: Request) -> str:
        """
        Generate endpoint key for rate limiting.

        Args:
            request: FastAPI request object

        Returns:
            Endpoint identifier
        """
        # Use path + method for endpoint identification
        return f"{request.method}:{request.url.path}"

    def _get_endpoint_policy(self, request: Request) -> RateLimitPolicy:
        """
        Get rate limit policy for endpoint.

        Args:
            request: FastAPI request object

        Returns:
            RateLimitPolicy for the endpoint
        """
        endpoint_key = self._get_endpoint_key(request)

        # Check for exact match
        if endpoint_key in self.endpoint_policies:
            return self.endpoint_policies[endpoint_key]

        # Check for path-only match
        path_key = request.url.path
        if path_key in self.endpoint_policies:
            return self.endpoint_policies[path_key]

        # Return default endpoint policy
        return self.default_policies["endpoint"]

    def _add_rate_limit_headers(
        self,
        response: Response,
        results: list[RateLimitResult],
    ) -> None:
        """
        Add RFC 6585 compliant rate limit headers to response.

        Headers:
        - X-RateLimit-Limit: Maximum requests allowed
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp when limit resets
        - Retry-After: Seconds to wait (if rate limited)

        Args:
            response: FastAPI response object
            results: List of rate limit check results
        """
        # Find most restrictive limit
        most_restrictive = min(
            results,
            key=lambda r: r.remaining / r.limit if r.limit > 0 else 0,
        )

        # Add standard rate limit headers
        response.headers["X-RateLimit-Limit"] = str(most_restrictive.limit)
        response.headers["X-RateLimit-Remaining"] = str(most_restrictive.remaining)
        response.headers["X-RateLimit-Reset"] = str(most_restrictive.reset_at)

        # Add Retry-After if rate limited
        if not most_restrictive.allowed and most_restrictive.retry_after > 0:
            response.headers["Retry-After"] = str(most_restrictive.retry_after)

        # Add custom headers for debugging (optional, can be disabled in production)
        response.headers["X-RateLimit-Type"] = most_restrictive.limit_type.value

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        """
        Process request through rate limiting middleware.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain

        Returns:
            Response object
        """
        start_time = time.time()

        # Skip exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Extract request metadata
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        endpoint_key = self._get_endpoint_key(request)
        user_agent = request.headers.get("User-Agent")

        # DDoS protection check
        if self.enable_ddos_protection and self.ddos_protector:
            try:
                threat_assessment = await self.ddos_protector.assess_threat(
                    client_ip=client_ip,
                    user_agent=user_agent,
                    endpoint=endpoint_key,
                )

                if threat_assessment.is_blocked:
                    logger.warning(
                        "Request blocked by DDoS protection",
                        client_ip=client_ip,
                        threat_level=threat_assessment.threat_level.value,
                        reasons=threat_assessment.reasons,
                    )

                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": {
                                "code": "DDOS_PROTECTION",
                                "message": "Request blocked by DDoS protection",
                                "threat_level": threat_assessment.threat_level.value,
                                "reasons": threat_assessment.reasons,
                            }
                        },
                        headers={
                            "Retry-After": str(
                                threat_assessment.metadata.get("ip_limit", {}).get(
                                    "retry_after", 60
                                )
                            ),
                        },
                    )

            except Exception as e:
                logger.error(
                    "DDoS protection check failed",
                    error=str(e),
                    client_ip=client_ip,
                )
                # Continue with rate limiting on DDoS check failure

        # Build rate limit checks
        checks: list[tuple[RateLimitType, str, RateLimitPolicy]] = []

        # 1. Per-client IP rate limit
        checks.append(
            (
                RateLimitType.CLIENT_IP,
                client_ip,
                self.default_policies["client_ip"],
            )
        )

        # 2. Per-endpoint rate limit
        endpoint_policy = self._get_endpoint_policy(request)
        checks.append((RateLimitType.ENDPOINT, endpoint_key, endpoint_policy))

        # 3. Per-user rate limit (if authenticated)
        if user_id:
            checks.append((RateLimitType.USER, user_id, self.default_policies["user"]))

        # Execute rate limit checks
        try:
            results = await self.rate_limiter.check_multiple_limits(checks)

            # Check if any limit was exceeded
            blocked_results = [r for r in results if not r.allowed]

            if blocked_results:
                # Find most restrictive limit
                most_restrictive = min(
                    blocked_results,
                    key=lambda r: r.retry_after,
                )

                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    user_id=user_id,
                    endpoint=endpoint_key,
                    limit_type=most_restrictive.limit_type.value,
                    retry_after=most_restrictive.retry_after,
                )

                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": f"Rate limit exceeded for {most_restrictive.limit_type.value}",
                            "limit": most_restrictive.limit,
                            "retry_after": most_restrictive.retry_after,
                        }
                    },
                )

                self._add_rate_limit_headers(response, results)
                return response

            # Request allowed, proceed with handler
            response = await call_next(request)

            # Add rate limit headers to successful response
            self._add_rate_limit_headers(response, results)

            # Log rate limiting overhead
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > 1.0:  # Log if overhead > 1ms
                logger.warning(
                    "Rate limiting overhead exceeded 1ms",
                    elapsed_ms=elapsed_ms,
                    client_ip=client_ip,
                )

            return response

        except Exception as e:
            logger.error(
                "Rate limiting middleware error",
                error=str(e),
                client_ip=client_ip,
                endpoint=endpoint_key,
            )
            # Fail open - allow request but log error
            return await call_next(request)
