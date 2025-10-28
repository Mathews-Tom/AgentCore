"""
Integration tests for authentication endpoints.

Tests complete authentication flows including token generation, refresh, and protected routes.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from testcontainers.redis import RedisContainer

from gateway.auth.jwt import jwt_manager
from gateway.auth.session import session_manager
from gateway.main import create_app


@pytest.fixture
async def redis_container():
    """Start Redis container for testing."""
    container = RedisContainer("redis:7-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def app_with_auth(redis_container):
    """Create FastAPI app with auth configured."""
    # Configure Redis URL for testing
    redis_url = f"redis://localhost:{redis_container.get_exposed_port(6379)}/0"
    session_manager.redis_url = redis_url

    # Initialize managers
    await jwt_manager.initialize()
    await session_manager.initialize()

    app = create_app()
    yield app

    # Cleanup
    await session_manager.close()


@pytest.fixture
async def client(app_with_auth):
    """Create async HTTP client."""
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestAuthenticationEndpoints:
    """Test authentication endpoint integration."""

    @pytest.mark.asyncio
    async def test_token_password_grant(self, client: AsyncClient) -> None:
        """Test token generation with password grant."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "Bearer"
        assert data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_token_invalid_credentials(self, client: AsyncClient) -> None:
        """Test token generation with invalid credentials."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "wrongpassword",
            })

        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_token_client_credentials_grant(self, client: AsyncClient) -> None:
        """Test token generation with client credentials grant."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": "service",
                "client_secret": "service123",
            })

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data

    @pytest.mark.asyncio
    async def test_token_missing_username(self, client: AsyncClient) -> None:
        """Test token generation with missing username."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "password": "admin123",
            })

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_token_unsupported_grant_type(self, client: AsyncClient) -> None:
        """Test token generation with unsupported grant type."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "implicit",
                "username": "admin",
                "password": "admin123",
            })

        assert response.status_code == 400
        assert "Unsupported grant type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_refresh_token(self, client: AsyncClient) -> None:
        """Test token refresh flow."""
        # Get initial tokens
        login_response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        assert login_response.status_code == 200
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        refresh_response = await client.post(
            "/auth/refresh",
            params={"refresh_token": refresh_token})

        assert refresh_response.status_code == 200
        data = refresh_response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["access_token"] != login_response.json()["access_token"]

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, client: AsyncClient) -> None:
        """Test refresh with invalid token."""
        response = await client.post(
            "/auth/refresh",
            params={"refresh_token": "invalid.token.here"})

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user(self, client: AsyncClient) -> None:
        """Test getting current user information."""
        # Login first
        login_response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        access_token = login_response.json()["access_token"]

        # Get current user
        response = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {access_token}"})

        assert response.status_code == 200
        data = response.json()

        assert data["username"] == "admin"
        assert "id" in data
        assert "roles" in data

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, client: AsyncClient) -> None:
        """Test getting current user without token."""
        response = await client.get("/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, client: AsyncClient) -> None:
        """Test getting current user with invalid token."""
        response = await client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"})

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_logout(self, client: AsyncClient) -> None:
        """Test logout endpoint."""
        # Login first
        login_response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        access_token = login_response.json()["access_token"]

        # Logout
        response = await client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"})

        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]

        # Token should be invalid after logout
        response = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {access_token}"})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, client: AsyncClient) -> None:
        """Test getting user sessions."""
        # Login first
        login_response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        access_token = login_response.json()["access_token"]

        # Get sessions
        response = await client.get(
            "/auth/sessions",
            headers={"Authorization": f"Bearer {access_token}"})

        assert response.status_code == 200
        data = response.json()

        assert "sessions" in data
        assert len(data["sessions"]) > 0

    @pytest.mark.asyncio
    async def test_delete_user_session(self, client: AsyncClient) -> None:
        """Test deleting specific user session."""
        # Create two sessions
        login1 = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        token1 = login1.json()["access_token"]

        login2 = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        token2 = login2.json()["access_token"]

        # Get sessions
        sessions_response = await client.get(
            "/auth/sessions",
            headers={"Authorization": f"Bearer {token1}"})
        sessions = sessions_response.json()["sessions"]
        assert len(sessions) >= 2

        # Delete one session
        session_to_delete = sessions[0]["session_id"]
        delete_response = await client.delete(
            f"/auth/sessions/{session_to_delete}",
            headers={"Authorization": f"Bearer {token1}"})

        assert delete_response.status_code == 200

    @pytest.mark.asyncio
    async def test_token_with_scope(self, client: AsyncClient) -> None:
        """Test token generation with custom scope."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
                "scope": "read:agents write:tasks",
            })

        assert response.status_code == 200
        data = response.json()

        assert data["scope"] == "read:agents write:tasks"

    @pytest.mark.asyncio
    async def test_protected_route_requires_auth(self, client: AsyncClient) -> None:
        """Test that protected routes require authentication."""
        # Try to access protected route without token
        response = await client.get("/auth/sessions")

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers


class TestAuthenticationSecurity:
    """Test authentication security features."""

    @pytest.mark.asyncio
    async def test_token_includes_user_roles(self, client: AsyncClient) -> None:
        """Test that tokens include user roles."""
        # Login as admin
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "password",
                "username": "admin",
                "password": "admin123",
            })
        access_token = response.json()["access_token"]

        # Get user info
        user_response = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {access_token}"})

        assert user_response.status_code == 200
        roles = user_response.json()["roles"]

        assert "admin" in roles
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_service_account_authentication(self, client: AsyncClient) -> None:
        """Test service account authentication."""
        response = await client.post(
            "/auth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": "service",
                "client_secret": "service123",
            })

        assert response.status_code == 200
        access_token = response.json()["access_token"]

        # Verify service role
        user_response = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {access_token}"})

        assert user_response.status_code == 200
        roles = user_response.json()["roles"]
        assert "service" in roles

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self, client: AsyncClient) -> None:
        """Test multiple concurrent sessions for same user."""
        # Create multiple sessions
        tokens = []
        for _ in range(3):
            response = await client.post(
                "/auth/token",
                json={
                    "grant_type": "password",
                    "username": "admin",
                    "password": "admin123",
                })
            tokens.append(response.json()["access_token"])

        # All tokens should be valid
        for token in tokens:
            response = await client.get(
                "/auth/me",
                headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 200
