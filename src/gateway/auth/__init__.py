"""
Gateway Authentication Module

JWT-based authentication system with RSA-256 signing, session management,
and token refresh mechanisms for the AgentCore API Gateway.
"""

from gateway.auth.dependencies import get_current_user, require_auth
from gateway.auth.jwt import JWTManager
from gateway.auth.models import TokenResponse, User, UserRole
from gateway.auth.session import SessionManager

__all__ = [
    "JWTManager",
    "SessionManager",
    "TokenResponse",
    "User",
    "UserRole",
    "get_current_user",
    "require_auth",
]
