"""
PKCE (Proof Key for Code Exchange) Implementation

RFC 7636 compliant PKCE generator for OAuth 2.0 authorization code flow.
Provides enhanced security by preventing authorization code interception attacks.
"""

from __future__ import annotations

import hashlib
import secrets
from base64 import urlsafe_b64encode
from datetime import UTC, datetime, timedelta

from gateway.auth.oauth.models import PKCEChallenge, PKCEChallengeMethod


class PKCEGenerator:
    """
    PKCE code verifier and challenge generator.

    Implements RFC 7636 specification for OAuth 2.0 PKCE.
    """

    # RFC 7636: code_verifier must be 43-128 characters
    MIN_VERIFIER_LENGTH = 43
    MAX_VERIFIER_LENGTH = 128
    DEFAULT_VERIFIER_LENGTH = 128

    # Challenge validity period
    DEFAULT_TTL_MINUTES = 10

    @staticmethod
    def generate_code_verifier(length: int | None = None) -> str:
        """
        Generate cryptographically secure code verifier.

        The code verifier is a high-entropy cryptographic random string
        using unreserved characters [A-Z] / [a-z] / [0-9] / "-" / "." / "_" / "~"

        Args:
            length: Verifier length (43-128 chars, default 128)

        Returns:
            URL-safe base64 encoded random string

        Raises:
            ValueError: If length is outside valid range
        """
        if length is None:
            length = PKCEGenerator.DEFAULT_VERIFIER_LENGTH

        if not (PKCEGenerator.MIN_VERIFIER_LENGTH <= length <= PKCEGenerator.MAX_VERIFIER_LENGTH):
            raise ValueError(
                f"Code verifier length must be between {PKCEGenerator.MIN_VERIFIER_LENGTH} "
                f"and {PKCEGenerator.MAX_VERIFIER_LENGTH} characters"
            )

        # Generate random bytes and encode as URL-safe base64
        # We need more bytes than the target length since base64 encoding expands
        num_bytes = (length * 3) // 4 + 1
        random_bytes = secrets.token_bytes(num_bytes)

        # URL-safe base64 encode and remove padding
        verifier = urlsafe_b64encode(random_bytes).decode("utf-8").rstrip("=")

        # Truncate to exact length
        return verifier[:length]

    @staticmethod
    def generate_code_challenge(
        code_verifier: str,
        method: PKCEChallengeMethod = PKCEChallengeMethod.S256,
    ) -> str:
        """
        Generate code challenge from verifier.

        Args:
            code_verifier: Code verifier string
            method: Challenge method (S256 or PLAIN)

        Returns:
            Code challenge string

        Raises:
            ValueError: If verifier length is invalid
        """
        verifier_length = len(code_verifier)

        if not (PKCEGenerator.MIN_VERIFIER_LENGTH <= verifier_length <= PKCEGenerator.MAX_VERIFIER_LENGTH):
            raise ValueError(
                f"Code verifier length must be between {PKCEGenerator.MIN_VERIFIER_LENGTH} "
                f"and {PKCEGenerator.MAX_VERIFIER_LENGTH} characters"
            )

        if method == PKCEChallengeMethod.S256:
            # S256: BASE64URL(SHA256(ASCII(code_verifier)))
            digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
            challenge = urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
            return challenge
        elif method == PKCEChallengeMethod.PLAIN:
            # PLAIN: code_verifier
            return code_verifier
        else:
            raise ValueError(f"Unsupported PKCE method: {method}")

    @staticmethod
    def verify_challenge(
        code_verifier: str,
        code_challenge: str,
        method: PKCEChallengeMethod = PKCEChallengeMethod.S256,
    ) -> bool:
        """
        Verify code challenge against verifier.

        Args:
            code_verifier: Code verifier from client
            code_challenge: Code challenge to verify
            method: Challenge method used

        Returns:
            True if challenge matches verifier, False otherwise
        """
        try:
            expected_challenge = PKCEGenerator.generate_code_challenge(code_verifier, method)
            return secrets.compare_digest(expected_challenge, code_challenge)
        except (ValueError, AttributeError):
            return False

    @classmethod
    def generate_pkce_pair(
        cls,
        method: PKCEChallengeMethod = PKCEChallengeMethod.S256,
        verifier_length: int | None = None,
        ttl_minutes: int | None = None,
    ) -> PKCEChallenge:
        """
        Generate complete PKCE challenge/verifier pair.

        Args:
            method: Challenge method (S256 or PLAIN)
            verifier_length: Code verifier length (default 128)
            ttl_minutes: Time-to-live in minutes (default 10)

        Returns:
            PKCEChallenge model with verifier and challenge
        """
        # Generate verifier
        code_verifier = cls.generate_code_verifier(verifier_length)

        # Generate challenge
        code_challenge = cls.generate_code_challenge(code_verifier, method)

        # Calculate expiration
        ttl = ttl_minutes if ttl_minutes is not None else cls.DEFAULT_TTL_MINUTES
        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(minutes=ttl)

        return PKCEChallenge(
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            code_challenge_method=method,
            created_at=created_at,
            expires_at=expires_at,
        )

    @staticmethod
    def is_pkce_required(client_type: str = "public") -> bool:
        """
        Determine if PKCE is required for client type.

        OAuth 2.1 requires PKCE for all clients, but OAuth 2.0 only
        recommends it for public clients (mobile, SPA, etc.).

        Args:
            client_type: Client type ("public" or "confidential")

        Returns:
            True if PKCE is required/recommended
        """
        # OAuth 2.1 / OAuth 3.0 direction: PKCE for all clients
        # For now, we require PKCE for public clients and recommend for confidential
        return client_type == "public"
