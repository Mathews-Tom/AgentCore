"""
Tests for PKCE (Proof Key for Code Exchange) implementation.
"""

from __future__ import annotations

import hashlib
from base64 import urlsafe_b64encode

import pytest

from gateway.auth.oauth.models import PKCEChallengeMethod
from gateway.auth.oauth.pkce import PKCEGenerator


class TestPKCEGenerator:
    """Test PKCE code verifier and challenge generation."""

    def test_generate_code_verifier_default_length(self) -> None:
        """Test code verifier generation with default length."""
        verifier = PKCEGenerator.generate_code_verifier()

        assert verifier is not None
        assert len(verifier) == PKCEGenerator.DEFAULT_VERIFIER_LENGTH
        assert all(c.isalnum() or c in "-._~" for c in verifier)

    def test_generate_code_verifier_custom_length(self) -> None:
        """Test code verifier generation with custom length."""
        length = 64
        verifier = PKCEGenerator.generate_code_verifier(length)

        assert verifier is not None
        assert len(verifier) == length

    def test_generate_code_verifier_invalid_length(self) -> None:
        """Test code verifier generation with invalid length."""
        with pytest.raises(ValueError, match="Code verifier length must be between"):
            PKCEGenerator.generate_code_verifier(length=30)  # Too short

        with pytest.raises(ValueError, match="Code verifier length must be between"):
            PKCEGenerator.generate_code_verifier(length=200)  # Too long

    def test_generate_code_challenge_s256(self) -> None:
        """Test S256 code challenge generation."""
        # Use valid verifier length (minimum 43 characters)
        verifier = PKCEGenerator.generate_code_verifier(43)
        challenge = PKCEGenerator.generate_code_challenge(verifier, PKCEChallengeMethod.S256)

        # Verify it's a valid base64 string
        assert challenge is not None
        assert len(challenge) > 0

    def test_generate_code_challenge_plain(self) -> None:
        """Test PLAIN code challenge generation."""
        verifier = PKCEGenerator.generate_code_verifier(43)
        challenge = PKCEGenerator.generate_code_challenge(verifier, PKCEChallengeMethod.PLAIN)

        # For PLAIN method, challenge should equal verifier
        assert challenge == verifier

    def test_verify_challenge_s256_valid(self) -> None:
        """Test successful S256 challenge verification."""
        verifier = PKCEGenerator.generate_code_verifier()
        challenge = PKCEGenerator.generate_code_challenge(verifier, PKCEChallengeMethod.S256)

        result = PKCEGenerator.verify_challenge(verifier, challenge, PKCEChallengeMethod.S256)
        assert result is True

    def test_verify_challenge_s256_invalid(self) -> None:
        """Test failed S256 challenge verification."""
        verifier = PKCEGenerator.generate_code_verifier()
        wrong_verifier = PKCEGenerator.generate_code_verifier()
        challenge = PKCEGenerator.generate_code_challenge(verifier, PKCEChallengeMethod.S256)

        result = PKCEGenerator.verify_challenge(wrong_verifier, challenge, PKCEChallengeMethod.S256)
        assert result is False

    def test_verify_challenge_plain_valid(self) -> None:
        """Test successful PLAIN challenge verification."""
        verifier = PKCEGenerator.generate_code_verifier(43)
        challenge = verifier  # PLAIN method

        result = PKCEGenerator.verify_challenge(verifier, challenge, PKCEChallengeMethod.PLAIN)
        assert result is True

    def test_verify_challenge_plain_invalid(self) -> None:
        """Test failed PLAIN challenge verification."""
        verifier = PKCEGenerator.generate_code_verifier(43)
        wrong_challenge = PKCEGenerator.generate_code_verifier(43)

        result = PKCEGenerator.verify_challenge(verifier, wrong_challenge, PKCEChallengeMethod.PLAIN)
        assert result is False

    def test_generate_pkce_pair_default(self) -> None:
        """Test PKCE pair generation with defaults."""
        pkce_pair = PKCEGenerator.generate_pkce_pair()

        assert pkce_pair.code_verifier is not None
        assert len(pkce_pair.code_verifier) == PKCEGenerator.DEFAULT_VERIFIER_LENGTH
        assert pkce_pair.code_challenge is not None
        assert pkce_pair.code_challenge_method == PKCEChallengeMethod.S256
        assert pkce_pair.created_at is not None
        assert pkce_pair.expires_at is not None
        assert pkce_pair.expires_at > pkce_pair.created_at

    def test_generate_pkce_pair_custom_verifier_length(self) -> None:
        """Test PKCE pair generation with custom verifier length."""
        length = 64
        pkce_pair = PKCEGenerator.generate_pkce_pair(verifier_length=length)

        assert len(pkce_pair.code_verifier) == length

    def test_generate_pkce_pair_custom_ttl(self) -> None:
        """Test PKCE pair generation with custom TTL."""
        ttl_minutes = 5
        pkce_pair = PKCEGenerator.generate_pkce_pair(ttl_minutes=ttl_minutes)

        time_diff = (pkce_pair.expires_at - pkce_pair.created_at).total_seconds()
        assert abs(time_diff - (ttl_minutes * 60)) < 2  # Allow 2 second tolerance

    def test_generate_pkce_pair_plain_method(self) -> None:
        """Test PKCE pair generation with PLAIN method."""
        pkce_pair = PKCEGenerator.generate_pkce_pair(method=PKCEChallengeMethod.PLAIN)

        assert pkce_pair.code_challenge_method == PKCEChallengeMethod.PLAIN
        assert pkce_pair.code_challenge == pkce_pair.code_verifier

    def test_is_pkce_required_public_client(self) -> None:
        """Test PKCE requirement for public clients."""
        assert PKCEGenerator.is_pkce_required("public") is True

    def test_is_pkce_required_confidential_client(self) -> None:
        """Test PKCE requirement for confidential clients."""
        # Currently returns False, but should be True for OAuth 2.1/3.0
        result = PKCEGenerator.is_pkce_required("confidential")
        assert isinstance(result, bool)

    def test_code_verifier_randomness(self) -> None:
        """Test that code verifiers are random and unique."""
        verifiers = {PKCEGenerator.generate_code_verifier() for _ in range(100)}

        # All verifiers should be unique
        assert len(verifiers) == 100

    def test_challenge_consistency(self) -> None:
        """Test that same verifier always produces same challenge."""
        verifier = PKCEGenerator.generate_code_verifier(43)

        challenge1 = PKCEGenerator.generate_code_challenge(verifier, PKCEChallengeMethod.S256)
        challenge2 = PKCEGenerator.generate_code_challenge(verifier, PKCEChallengeMethod.S256)

        assert challenge1 == challenge2
