"""JSON-RPC 2.0 client for AgentCore API."""

from __future__ import annotations

from typing import Any

import requests  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry

from agentcore_cli.exceptions import (
    ApiError,
    ConnectionError as CliConnectionError,
    JsonRpcError,
    TimeoutError as CliTimeoutError,
)


class AgentCoreClient:
    """JSON-RPC 2.0 client for AgentCore API.

    This client handles:
    - JSON-RPC 2.0 request/response formatting
    - Connection pooling for efficient HTTP reuse
    - Retry logic with exponential backoff
    - Timeout configuration
    - SSL/TLS verification
    - Error translation to user-friendly messages
    """

    def __init__(
        self,
        api_url: str,
        timeout: int = 30,
        retries: int = 3,
        verify_ssl: bool = True,
        auth_token: str | None = None,
    ) -> None:
        """Initialize AgentCore JSON-RPC client.

        Args:
            api_url: Base URL of AgentCore API (e.g., http://localhost:8001)
            timeout: Request timeout in seconds (default: 30)
            retries: Number of retry attempts (default: 3)
            verify_ssl: Whether to verify SSL certificates (default: True)
            auth_token: Optional JWT authentication token
        """
        self.api_url = f"{api_url.rstrip('/')}/api/v1/jsonrpc"
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.auth_token = auth_token
        self.session = self._create_session(retries)
        self.request_id = 0

    def _create_session(self, retries: int) -> requests.Session:
        """Create HTTP session with connection pooling and retry strategy.

        Args:
            retries: Number of retry attempts

        Returns:
            Configured requests.Session
        """
        session = requests.Session()

        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,  # Wait 1s, 2s, 4s, 8s, etc.
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],  # Only retry POST requests
            raise_on_status=False,
        )

        # Mount adapter with retry strategy for both HTTP and HTTPS
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,
            pool_maxsize=1,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a JSON-RPC 2.0 method call.

        Args:
            method: JSON-RPC method name (e.g., "agent.register")
            params: Optional parameters dictionary

        Returns:
            Result data from the JSON-RPC response

        Raises:
            ConnectionError: Failed to connect to API
            TimeoutError: Request timed out
            AuthenticationError: Authentication failed (401)
            JsonRpcError: JSON-RPC protocol error
            ApiError: HTTP error response
        """
        self.request_id += 1

        # Build JSON-RPC 2.0 request
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id,
        }

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            # Make HTTP POST request
            response = self.session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            # Check for HTTP errors
            if response.status_code == 401:
                from agentcore_cli.exceptions import AuthenticationError

                raise AuthenticationError(
                    "Authentication failed. Check your token with: echo $AGENTCORE_TOKEN"
                )
            elif response.status_code >= 400:
                raise ApiError(
                    response.status_code,
                    self._get_error_message(response),
                )

            # Parse JSON response
            try:
                data: dict[str, Any] = response.json()
            except requests.exceptions.JSONDecodeError as e:
                raise ApiError(
                    response.status_code,
                    f"Invalid JSON response: {e}",
                ) from e

            # Check for JSON-RPC error
            if "error" in data:
                error_obj = data["error"]
                if isinstance(error_obj, dict):
                    raise JsonRpcError(error_obj)
                raise JsonRpcError({"code": -32603, "message": str(error_obj)})

            # Return result
            result = data.get("result", {})
            return result if isinstance(result, dict) else {}

        except requests.exceptions.Timeout as e:
            raise CliTimeoutError(
                f"Request timed out after {self.timeout}s. "
                f"Try increasing timeout with --api-timeout or AGENTCORE_API_TIMEOUT"
            ) from e

        except requests.exceptions.ConnectionError as e:
            raise CliConnectionError(
                f"Cannot connect to AgentCore API at {self.api_url}\n"
                f"Suggestions:\n"
                f"  • Check if AgentCore server is running\n"
                f"  • Verify API URL: agentcore config show\n"
                f"  • Test connectivity: agentcore health"
            ) from e

        except requests.exceptions.RequestException as e:
            raise CliConnectionError(
                f"Request failed: {e}"
            ) from e

    def _get_error_message(self, response: requests.Response) -> str:
        """Extract error message from HTTP response.

        Args:
            response: HTTP response object

        Returns:
            User-friendly error message
        """
        try:
            data = response.json()
            if isinstance(data, dict):
                # Try to extract error message from various formats
                if "error" in data:
                    if isinstance(data["error"], dict):
                        return str(data["error"].get("message", data["error"]))
                    return str(data["error"])
                if "message" in data:
                    return str(data["message"])
                if "detail" in data:
                    return str(data["detail"])
        except Exception:
            pass

        # Fallback to status text
        return response.reason or "Unknown error"

    def batch_call(
        self,
        requests_data: list[tuple[str, dict[str, Any] | None]],
    ) -> list[dict[str, Any]]:
        """Make multiple JSON-RPC calls in a single batch request.

        Args:
            requests_data: List of (method, params) tuples

        Returns:
            List of results in the same order as requests

        Raises:
            Same exceptions as call() method
        """
        # Build batch request
        batch_payload = []
        for i, (method, params) in enumerate(requests_data):
            self.request_id += 1
            batch_payload.append({
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": self.request_id - len(requests_data) + i + 1,
            })

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            # Make HTTP POST request
            response = self.session.post(
                self.api_url,
                json=batch_payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            # Check for HTTP errors
            if response.status_code >= 400:
                raise ApiError(
                    response.status_code,
                    self._get_error_message(response),
                )

            # Parse JSON response
            data = response.json()

            # Extract results
            results = []
            for item in data:
                if "error" in item:
                    raise JsonRpcError(item["error"])
                results.append(item.get("result", {}))

            return results

        except requests.exceptions.Timeout as e:
            raise CliTimeoutError(
                f"Batch request timed out after {self.timeout}s"
            ) from e

        except requests.exceptions.ConnectionError as e:
            raise CliConnectionError(
                f"Cannot connect to AgentCore API at {self.api_url}"
            ) from e

        except requests.exceptions.RequestException as e:
            raise CliConnectionError(
                f"Batch request failed: {e}"
            ) from e

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        self.session.close()

    def __enter__(self) -> AgentCoreClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
