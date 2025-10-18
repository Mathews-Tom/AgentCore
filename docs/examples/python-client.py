"""
AgentCore API Gateway - Python Client Examples

Demonstrates authentication, API calls, error handling, and real-time communication.
"""

from __future__ import annotations

import time
from typing import Any
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AgentCoreClient:
    """
    Python client for AgentCore API Gateway.

    Features:
    - Automatic token management and refresh
    - Retry logic with exponential backoff
    - Rate limit handling
    - Error handling
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        username: str | None = None,
        password: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password

        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.token_expires_at: datetime | None = None

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def authenticate(self, username: str | None = None, password: str | None = None) -> dict[str, Any]:
        """
        Authenticate with username/password and obtain JWT tokens.

        Args:
            username: Username (optional if provided in __init__)
            password: Password (optional if provided in __init__)

        Returns:
            Token response with access_token, refresh_token, and expiration

        Example:
            >>> client = AgentCoreClient()
            >>> tokens = client.authenticate("user", "user123")
            >>> print(tokens["access_token"])
        """
        username = username or self.username
        password = password or self.password

        if not username or not password:
            raise ValueError("Username and password required")

        response = self.session.post(
            f"{self.base_url}/auth/token",
            json={
                "grant_type": "password",
                "username": username,
                "password": password,
                "scope": "user:read user:write agent:read agent:execute",
            },
        )

        if response.status_code != 200:
            self._handle_error(response)

        token_data = response.json()

        # Store tokens
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token")

        # Calculate token expiration
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        return token_data

    def refresh_access_token(self) -> dict[str, Any]:
        """
        Refresh access token using refresh token.

        Returns:
            New token response

        Example:
            >>> tokens = client.refresh_access_token()
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available")

        response = self.session.post(
            f"{self.base_url}/auth/refresh",
            json={"refresh_token": self.refresh_token},
        )

        if response.status_code != 200:
            self._handle_error(response)

        token_data = response.json()

        # Update access token
        self.access_token = token_data["access_token"]

        # Update expiration
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        return token_data

    def _ensure_authenticated(self) -> None:
        """Ensure access token is valid, refresh if needed."""
        if not self.access_token:
            self.authenticate()
            return

        # Check if token is about to expire (within 5 minutes)
        if self.token_expires_at and datetime.utcnow() >= self.token_expires_at - timedelta(minutes=5):
            try:
                self.refresh_access_token()
            except Exception:
                # If refresh fails, re-authenticate
                self.authenticate()

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with authentication."""
        self._ensure_authenticated()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: requests.Response) -> None:
        """Handle API error responses."""
        try:
            error_data = response.json()
            error = error_data.get("error", {})
            code = error.get("code", "UNKNOWN_ERROR")
            message = error.get("message", "Unknown error")
            details = error.get("details", {})
            request_id = error.get("request_id")

            error_msg = f"{code}: {message}"
            if details:
                error_msg += f" | Details: {details}"
            if request_id:
                error_msg += f" | Request ID: {request_id}"

            raise Exception(error_msg)
        except ValueError:
            # Response is not JSON
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def get(self, path: str, **kwargs) -> Any:
        """
        Make GET request.

        Args:
            path: API path (e.g., "/auth/me")
            **kwargs: Additional arguments for requests.get

        Returns:
            Response JSON data

        Example:
            >>> user = client.get("/auth/me")
            >>> print(user["username"])
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_headers())

        response = self.session.get(
            f"{self.base_url}{path}",
            headers=headers,
            **kwargs,
        )

        if response.status_code != 200:
            self._handle_error(response)

        return response.json()

    def post(self, path: str, data: Any = None, **kwargs) -> Any:
        """
        Make POST request.

        Args:
            path: API path
            data: Request body data
            **kwargs: Additional arguments for requests.post

        Returns:
            Response JSON data

        Example:
            >>> result = client.post("/agents", {"name": "MyAgent"})
        """
        headers = kwargs.pop("headers", {})
        headers.update(self._get_headers())

        response = self.session.post(
            f"{self.base_url}{path}",
            json=data,
            headers=headers,
            **kwargs,
        )

        if response.status_code not in (200, 201):
            self._handle_error(response)

        return response.json() if response.text else None

    def get_current_user(self) -> dict[str, Any]:
        """
        Get current authenticated user information.

        Returns:
            User information

        Example:
            >>> user = client.get_current_user()
            >>> print(f"Logged in as: {user['username']}")
        """
        return self.get("/auth/me")

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List active sessions for current user.

        Returns:
            List of active sessions

        Example:
            >>> sessions = client.list_sessions()
            >>> for session in sessions:
            ...     print(f"Session: {session['session_id']}")
        """
        response = self.get("/auth/sessions")
        return response.get("sessions", [])

    def logout(self) -> None:
        """
        Logout and invalidate current session.

        Example:
            >>> client.logout()
        """
        self.post("/auth/logout")
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None


# Usage Examples

def example_basic_authentication():
    """Example: Basic authentication and API calls."""
    print("Example: Basic Authentication")
    print("-" * 50)

    # Create client and authenticate
    client = AgentCoreClient(username="user", password="user123")

    # Get user information
    user = client.get_current_user()
    print(f"Logged in as: {user['username']}")
    print(f"Email: {user['email']}")
    print(f"Roles: {user['roles']}")

    # List active sessions
    sessions = client.list_sessions()
    print(f"\nActive sessions: {len(sessions)}")
    for session in sessions:
        print(f"  - {session['session_id']} from {session['ip_address']}")

    # Logout
    client.logout()
    print("\nLogged out successfully")


def example_error_handling():
    """Example: Error handling and retry logic."""
    print("\nExample: Error Handling")
    print("-" * 50)

    client = AgentCoreClient()

    try:
        # Invalid credentials
        client.authenticate("invalid", "invalid")
    except Exception as e:
        print(f"Authentication failed: {e}")

    try:
        # Access protected endpoint without authentication
        client.access_token = None
        client.get_current_user()
    except Exception as e:
        print(f"Unauthorized access: {e}")


def example_rate_limit_handling():
    """Example: Handle rate limits gracefully."""
    print("\nExample: Rate Limit Handling")
    print("-" * 50)

    client = AgentCoreClient(username="user", password="user123")

    for i in range(5):
        try:
            user = client.get_current_user()
            print(f"Request {i + 1}: Success")
        except Exception as e:
            if "RATE_LIMIT_EXCEEDED" in str(e):
                print(f"Rate limited. Waiting 5 seconds...")
                time.sleep(5)
                user = client.get_current_user()
                print(f"Request {i + 1}: Success (after retry)")
            else:
                print(f"Request {i + 1}: Error - {e}")


def example_token_refresh():
    """Example: Automatic token refresh."""
    print("\nExample: Token Refresh")
    print("-" * 50)

    client = AgentCoreClient(username="user", password="user123")

    # Initial authentication
    print(f"Access token expires at: {client.token_expires_at}")

    # Simulate token near expiration
    client.token_expires_at = datetime.utcnow() + timedelta(minutes=3)

    # This will automatically refresh the token
    user = client.get_current_user()
    print(f"New token expires at: {client.token_expires_at}")


if __name__ == "__main__":
    # Run examples
    example_basic_authentication()
    example_error_handling()
    example_rate_limit_handling()
    example_token_refresh()
