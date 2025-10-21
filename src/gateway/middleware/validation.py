"""
Request Validation Middleware

Implements comprehensive input validation to prevent injection attacks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from gateway.config import settings


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate and sanitize request inputs to prevent injection attacks.

    Checks for:
    - SQL injection patterns
    - XSS (Cross-Site Scripting) patterns
    - Path traversal attempts
    - Command injection patterns
    - Malicious headers
    """

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
        r"(';|\")",  # Quote escaping
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers (onclick, onerror, etc.)
        r"<iframe",
        r"<object",
        r"<embed",
        r"eval\s*\(",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"%252e%252e",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",  # Shell metacharacters
        r"\$\(.*\)",  # Command substitution
        r"`.*`",  # Backticks
        r"\|\|",  # OR operator
        r"&&",  # AND operator
    ]

    def __init__(
        self,
        app,
        enable_sql_injection_check: bool = True,
        enable_xss_check: bool = True,
        enable_path_traversal_check: bool = True,
        enable_command_injection_check: bool = True,
        max_param_length: int = 10000,
        max_header_length: int = 8192,
    ):
        """
        Initialize validation middleware.

        Args:
            app: FastAPI application
            enable_sql_injection_check: Enable SQL injection detection
            enable_xss_check: Enable XSS detection
            enable_path_traversal_check: Enable path traversal detection
            enable_command_injection_check: Enable command injection detection
            max_param_length: Maximum length for query/body parameters
            max_header_length: Maximum length for header values
        """
        super().__init__(app)
        self.enable_sql_injection_check = enable_sql_injection_check
        self.enable_xss_check = enable_xss_check
        self.enable_path_traversal_check = enable_path_traversal_check
        self.enable_command_injection_check = enable_command_injection_check
        self.max_param_length = max_param_length
        self.max_header_length = max_header_length

        # Compile regex patterns
        if enable_sql_injection_check:
            self.sql_injection_regex = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.SQL_INJECTION_PATTERNS
            ]

        if enable_xss_check:
            self.xss_regex = [
                re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS
            ]

        if enable_path_traversal_check:
            self.path_traversal_regex = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.PATH_TRAVERSAL_PATTERNS
            ]

        if enable_command_injection_check:
            self.command_injection_regex = [
                re.compile(pattern) for pattern in self.COMMAND_INJECTION_PATTERNS
            ]

    def _check_patterns(
        self, value: str, patterns: list[re.Pattern], attack_type: str
    ) -> None:
        """
        Check value against attack patterns.

        Args:
            value: String value to check
            patterns: List of compiled regex patterns
            attack_type: Type of attack for error message

        Raises:
            HTTPException: If malicious pattern detected
        """
        for pattern in patterns:
            if pattern.search(value):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Potential {attack_type} attack detected",
                )

    def _validate_value(self, value: Any, name: str = "parameter") -> None:
        """
        Validate a single value.

        Args:
            value: Value to validate
            name: Parameter name for error messages

        Raises:
            HTTPException: If validation fails
        """
        if not isinstance(value, str):
            return

        # Check length
        if len(value) > self.max_param_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{name} exceeds maximum length ({self.max_param_length} characters)",
            )

        # Check SQL injection
        if self.enable_sql_injection_check:
            self._check_patterns(value, self.sql_injection_regex, "SQL injection")

        # Check XSS
        if self.enable_xss_check:
            self._check_patterns(value, self.xss_regex, "XSS")

        # Check path traversal
        if self.enable_path_traversal_check:
            self._check_patterns(value, self.path_traversal_regex, "path traversal")

        # Check command injection
        if self.enable_command_injection_check:
            self._check_patterns(
                value, self.command_injection_regex, "command injection"
            )

    def _validate_headers(self, headers: dict[str, str]) -> None:
        """
        Validate request headers.

        Args:
            headers: Request headers

        Raises:
            HTTPException: If validation fails
        """
        for name, value in headers.items():
            # Skip standard headers that may contain special chars
            if name.lower() in {"authorization", "cookie", "user-agent"}:
                continue

            # Check header length
            if len(value) > self.max_header_length:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Header {name} exceeds maximum length",
                )

            # Validate header value
            self._validate_value(value, f"Header {name}")

    def _validate_query_params(self, query_params: dict[str, Any]) -> None:
        """
        Validate query parameters.

        Args:
            query_params: Query parameters

        Raises:
            HTTPException: If validation fails
        """
        for name, value in query_params.items():
            if isinstance(value, list):
                for item in value:
                    self._validate_value(str(item), f"Query parameter {name}")
            else:
                self._validate_value(str(value), f"Query parameter {name}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request inputs."""
        try:
            # Validate headers
            self._validate_headers(dict(request.headers))

            # Validate query parameters
            if request.query_params:
                self._validate_query_params(dict(request.query_params))

            # Validate path parameters
            if request.path_params:
                for name, value in request.path_params.items():
                    self._validate_value(str(value), f"Path parameter {name}")

            # Process request
            response = await call_next(request)
            return response

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Validation error: {str(e)}",
            )
