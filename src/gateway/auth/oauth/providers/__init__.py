"""
OAuth Provider Implementations

Concrete implementations for supported OAuth providers (Google, GitHub, Microsoft, etc.).
"""

from gateway.auth.oauth.providers.github import GitHubOAuthProvider
from gateway.auth.oauth.providers.google import GoogleOAuthProvider

__all__ = [
    "GoogleOAuthProvider",
    "GitHubOAuthProvider",
]
