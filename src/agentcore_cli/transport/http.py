"""HTTP transport layer implementation.

This module provides HTTP communication for the CLI with retry logic,
connection pooling, and proper error handling. It has NO knowledge of
JSON-RPC protocol or business logic.
"""

from __future__ import annotations

from typing import Any
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agentcore_cli.transport.exceptions import (
    HttpError,
    NetworkError,
    TimeoutError,
    TransportError,
)


class HttpTransport:
    """HTTP transport for network communication.

    This class handles all HTTP operations with connection pooling,
    retry logic, and proper error translation. It operates at the
    network layer only and has no knowledge of JSON-RPC or business logic.

    Features:
        - Connection pooling for performance
        - Exponential backoff retry logic
        - Configurable SSL/TLS verification
        - Timeout handling
        - Network error translation

    Args:
        base_url: Base URL for API server (e.g., "http://localhost:8001")
        timeout: Request timeout in seconds (default: 30)
        retries: Number of retry attempts (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)

    Attributes:
        base_url: Base URL for API server
        timeout: Request timeout in seconds
        verify_ssl: SSL verification flag
        session: Configured requests session with retry logic

    Example:
        >>> transport = HttpTransport("http://localhost:8001")
        >>> response = transport.post("/api/v1/jsonrpc", {"method": "ping"})
        >>> print(response)
        {"jsonrpc": "2.0", "result": "pong", "id": 1}
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retries: int = 3,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            base_url: Base URL for API server
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            verify_ssl: Whether to verify SSL certificates

        Raises:
            ValueError: If base_url is empty or invalid
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = self._create_session(retries)

    def _create_session(self, retries: int) -> requests.Session:
        """Create configured requests session with retry logic.

        Configures:
        - Connection pooling (pool_connections=10, pool_maxsize=10)
        - Exponential backoff (1s, 2s, 4s, 8s)
        - Retry on status codes: 429, 500, 502, 503, 504

        Args:
            retries: Number of retry attempts

        Returns:
            Configured requests session

        Example:
            >>> transport = HttpTransport("http://localhost:8001")
            >>> session = transport._create_session(3)
            >>> # Session configured with retry strategy
        """
        session = requests.Session()

        # Configure retry strategy
        # Exponential backoff: {backoff factor} * (2 ** ({number of total retries} - 1))
        # With backoff_factor=1: 1s, 2s, 4s, 8s
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,  # 1s, 2s, 4s, 8s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],  # Only retry POST requests
        )

        # Create HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )

        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def post(
        self,
        endpoint: str,
        data: dict[str, Any] | list[dict[str, Any]],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Send HTTP POST request.

        Args:
            endpoint: API endpoint path (e.g., "/api/v1/jsonrpc")
            data: Request body data (dict or list, will be JSON-encoded)
            headers: Optional HTTP headers

        Returns:
            Parsed JSON response body (dict or list)

        Raises:
            NetworkError: If connection fails
            TimeoutError: If request times out
            HttpError: If server returns error status code
            TransportError: For other transport-level errors

        Example:
            >>> transport = HttpTransport("http://localhost:8001")
            >>> response = transport.post(
            ...     "/api/v1/jsonrpc",
            ...     {"jsonrpc": "2.0", "method": "ping", "id": 1},
            ... )
            >>> print(response["result"])
            "pong"
        """
        url = f"{self.base_url}{endpoint}"

        # Prepare headers
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        try:
            # Send POST request
            response = self.session.post(
                url,
                json=data,
                headers=request_headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            # Check for HTTP errors
            if response.status_code >= 400:
                raise HttpError(
                    message=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

            # Parse JSON response
            try:
                result: dict[str, Any] = response.json()
                return result
            except ValueError as e:
                raise TransportError(
                    message="Invalid JSON response from server",
                    status_code=response.status_code,
                    cause=e,
                )

        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                message=f"Request timed out after {self.timeout}s",
                cause=e,
            )

        except requests.exceptions.ConnectionError as e:
            raise NetworkError(
                message=f"Connection failed: {str(e)}",
                cause=e,
            )

        except requests.exceptions.RequestException as e:
            # Catch-all for other requests exceptions
            raise TransportError(
                message=f"Transport error: {str(e)}",
                cause=e,
            )

    def close(self) -> None:
        """Close the HTTP session and release resources.

        This should be called when the transport is no longer needed
        to properly clean up connection pools.

        Example:
            >>> transport = HttpTransport("http://localhost:8001")
            >>> try:
            ...     transport.post("/api/v1/jsonrpc", {})
            ... finally:
            ...     transport.close()
        """
        self.session.close()

    def __enter__(self) -> HttpTransport:
        """Context manager entry.

        Returns:
            Self for context manager protocol

        Example:
            >>> with HttpTransport("http://localhost:8001") as transport:
            ...     transport.post("/api/v1/jsonrpc", {})
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        self.close()
