"""Unit tests for credential manager.

Tests encryption, decryption, rotation, and key derivation.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import InvalidToken

from agentcore.integration.security.credential_manager import (
    CredentialManager,
    CredentialStatus,
    CredentialType,
    EncryptedCredential,
)


class TestCredentialManager:
    """Test credential manager functionality."""

    def test_initialization(self) -> None:
        """Test credential manager initialization."""
        manager = CredentialManager()
        assert manager is not None
        assert manager.get_master_key_b64() is not None

    def test_initialization_with_key(self) -> None:
        """Test initialization with existing master key."""
        key = CredentialManager.generate_master_key()
        manager1 = CredentialManager(master_key=key)
        manager2 = CredentialManager(master_key=key)

        # Both managers should use same key
        assert manager1.get_master_key_b64() == manager2.get_master_key_b64()

    def test_encrypt_credential(self) -> None:
        """Test credential encryption."""
        manager = CredentialManager()

        cred = manager.encrypt_credential(
            credential_id="test-001",
            service_name="test-service",
            credential_type=CredentialType.API_KEY,
            credential_value="super-secret-api-key-12345",
        )

        assert cred.credential_id == "test-001"
        assert cred.service_name == "test-service"
        assert cred.credential_type == CredentialType.API_KEY
        assert cred.status == CredentialStatus.ACTIVE
        assert cred.encrypted_value is not None
        assert "super-secret" not in cred.encrypted_value

    def test_decrypt_credential(self) -> None:
        """Test credential decryption."""
        manager = CredentialManager()
        original_value = "my-secret-password-123"

        manager.encrypt_credential(
            credential_id="test-002",
            service_name="database",
            credential_type=CredentialType.DATABASE,
            credential_value=original_value,
        )

        decrypted = manager.decrypt_credential("test-002")
        assert decrypted == original_value

    def test_encrypt_decrypt_roundtrip(self) -> None:
        """Test encryption/decryption roundtrip with various values."""
        manager = CredentialManager()
        test_values = [
            "simple",
            "with spaces",
            "with-special-chars!@#$%",
            "unicode-文字-emoji-🔐",
            "very-long-" + "x" * 1000,
        ]

        for i, value in enumerate(test_values):
            cred_id = f"test-{i:03d}"
            manager.encrypt_credential(
                credential_id=cred_id,
                service_name="test",
                credential_type=CredentialType.API_KEY,
                credential_value=value,
            )

            decrypted = manager.decrypt_credential(cred_id)
            assert decrypted == value

    def test_duplicate_credential_id_fails(self) -> None:
        """Test that duplicate credential IDs are rejected."""
        manager = CredentialManager()

        manager.encrypt_credential(
            credential_id="duplicate",
            service_name="service1",
            credential_type=CredentialType.API_KEY,
            credential_value="value1",
        )

        with pytest.raises(ValueError, match="already exists"):
            manager.encrypt_credential(
                credential_id="duplicate",
                service_name="service2",
                credential_type=CredentialType.API_KEY,
                credential_value="value2",
            )

    def test_decrypt_nonexistent_fails(self) -> None:
        """Test that decrypting nonexistent credential fails."""
        manager = CredentialManager()

        with pytest.raises(KeyError, match="not found"):
            manager.decrypt_credential("nonexistent")

    def test_decrypt_with_wrong_key_fails(self) -> None:
        """Test that decryption with wrong key fails."""
        manager1 = CredentialManager()
        manager2 = CredentialManager()  # Different key

        cred = manager1.encrypt_credential(
            credential_id="test-wrong-key",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
        )

        # Try to decrypt with different manager (different key)
        manager2._credentials["test-wrong-key"] = cred

        with pytest.raises(InvalidToken):
            manager2.decrypt_credential("test-wrong-key")

    def test_rotate_credential(self) -> None:
        """Test credential rotation."""
        manager = CredentialManager()
        original_value = "original-secret"
        new_value = "new-secret-after-rotation"

        manager.encrypt_credential(
            credential_id="rotate-test",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value=original_value,
        )

        # Get initial state
        cred_before = manager.get_credential("rotate-test")
        version_before = cred_before.version

        # Rotate
        rotated = manager.rotate_credential("rotate-test", new_value)

        # Check version incremented
        assert rotated.version == version_before + 1
        assert rotated.last_rotated_at is not None

        # Check new value is set
        decrypted = manager.decrypt_credential("rotate-test")
        assert decrypted == new_value
        assert decrypted != original_value

    def test_rotate_nonexistent_fails(self) -> None:
        """Test rotating nonexistent credential fails."""
        manager = CredentialManager()

        with pytest.raises(KeyError, match="not found"):
            manager.rotate_credential("nonexistent", "new-value")

    def test_revoke_credential(self) -> None:
        """Test credential revocation."""
        manager = CredentialManager()

        manager.encrypt_credential(
            credential_id="revoke-test",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
        )

        # Revoke
        manager.revoke_credential("revoke-test")

        # Check status
        cred = manager.get_credential("revoke-test")
        assert cred.status == CredentialStatus.REVOKED

        # Try to decrypt revoked credential
        with pytest.raises(ValueError, match="revoked"):
            manager.decrypt_credential("revoke-test")

    def test_credential_expiration(self) -> None:
        """Test credential expiration handling."""
        manager = CredentialManager()

        # Create expired credential
        expires_at = datetime.now(timezone.utc) - timedelta(days=1)

        manager.encrypt_credential(
            credential_id="expired-test",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
            expires_at=expires_at,
        )

        # Check is_expired
        cred = manager.get_credential("expired-test")
        assert cred.is_expired()

        # Try to decrypt expired credential
        with pytest.raises(ValueError, match="expired"):
            manager.decrypt_credential("expired-test")

    def test_credential_needs_rotation(self) -> None:
        """Test rotation detection logic."""
        manager = CredentialManager()

        # Create credential with rotation interval
        manager.encrypt_credential(
            credential_id="rotation-due",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
            rotation_interval_days=30,
        )

        # Set last rotation to past
        cred = manager.get_credential("rotation-due")
        cred.last_rotated_at = datetime.now(timezone.utc) - timedelta(days=31)

        # Should need rotation
        assert cred.needs_rotation()

    def test_list_credentials_empty(self) -> None:
        """Test listing credentials when empty."""
        manager = CredentialManager()
        credentials = manager.list_credentials()
        assert credentials == []

    def test_list_credentials_all(self) -> None:
        """Test listing all credentials."""
        manager = CredentialManager()

        # Create multiple credentials
        for i in range(5):
            manager.encrypt_credential(
                credential_id=f"cred-{i}",
                service_name=f"service-{i}",
                credential_type=CredentialType.API_KEY,
                credential_value=f"secret-{i}",
            )

        credentials = manager.list_credentials()
        assert len(credentials) == 5

    def test_list_credentials_filtered_by_service(self) -> None:
        """Test listing credentials filtered by service name."""
        manager = CredentialManager()

        # Create credentials for different services
        manager.encrypt_credential(
            credential_id="service1-cred1",
            service_name="service1",
            credential_type=CredentialType.API_KEY,
            credential_value="secret1",
        )
        manager.encrypt_credential(
            credential_id="service1-cred2",
            service_name="service1",
            credential_type=CredentialType.DATABASE,
            credential_value="secret2",
        )
        manager.encrypt_credential(
            credential_id="service2-cred1",
            service_name="service2",
            credential_type=CredentialType.API_KEY,
            credential_value="secret3",
        )

        # Filter by service
        credentials = manager.list_credentials(service_name="service1")
        assert len(credentials) == 2
        assert all(c.service_name == "service1" for c in credentials)

    def test_list_credentials_filtered_by_type(self) -> None:
        """Test listing credentials filtered by type."""
        manager = CredentialManager()

        # Create credentials of different types
        manager.encrypt_credential(
            credential_id="api-key-1",
            service_name="service",
            credential_type=CredentialType.API_KEY,
            credential_value="secret1",
        )
        manager.encrypt_credential(
            credential_id="oauth-1",
            service_name="service",
            credential_type=CredentialType.OAUTH2,
            credential_value="secret2",
        )

        # Filter by type
        credentials = manager.list_credentials(credential_type=CredentialType.API_KEY)
        assert len(credentials) == 1
        assert credentials[0].credential_type == CredentialType.API_KEY

    def test_list_credentials_filtered_by_status(self) -> None:
        """Test listing credentials filtered by status."""
        manager = CredentialManager()

        # Create and revoke some credentials
        manager.encrypt_credential(
            credential_id="active-1",
            service_name="service",
            credential_type=CredentialType.API_KEY,
            credential_value="secret1",
        )
        manager.encrypt_credential(
            credential_id="revoked-1",
            service_name="service",
            credential_type=CredentialType.API_KEY,
            credential_value="secret2",
        )
        manager.revoke_credential("revoked-1")

        # Filter by status
        active = manager.list_credentials(status=CredentialStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].credential_id == "active-1"

        revoked = manager.list_credentials(status=CredentialStatus.REVOKED)
        assert len(revoked) == 1
        assert revoked[0].credential_id == "revoked-1"

    def test_check_rotations_needed(self) -> None:
        """Test checking which credentials need rotation."""
        manager = CredentialManager()

        # Create credential that needs rotation
        manager.encrypt_credential(
            credential_id="needs-rotation",
            service_name="service",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
            rotation_interval_days=30,
        )
        cred = manager.get_credential("needs-rotation")
        cred.last_rotated_at = datetime.now(timezone.utc) - timedelta(days=31)

        # Create credential that doesn't need rotation
        manager.encrypt_credential(
            credential_id="no-rotation-needed",
            service_name="service",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
        )

        # Check
        needs_rotation = manager.check_rotations_needed()
        assert len(needs_rotation) == 1
        assert needs_rotation[0].credential_id == "needs-rotation"

    def test_generate_master_key(self) -> None:
        """Test master key generation."""
        key1 = CredentialManager.generate_master_key()
        key2 = CredentialManager.generate_master_key()

        # Keys should be different
        assert key1 != key2

        # Keys should be valid base64
        decoded1 = base64.urlsafe_b64decode(key1)
        decoded2 = base64.urlsafe_b64decode(key2)

        # Keys should be 32 bytes
        assert len(decoded1) == 32
        assert len(decoded2) == 32

    def test_constant_time_compare(self) -> None:
        """Test constant-time string comparison."""
        # Equal strings
        assert CredentialManager.constant_time_compare("secret", "secret")

        # Different strings
        assert not CredentialManager.constant_time_compare("secret", "different")

        # Different lengths
        assert not CredentialManager.constant_time_compare("short", "longer")

    def test_credential_with_metadata(self) -> None:
        """Test storing and retrieving credential metadata."""
        manager = CredentialManager()

        metadata = {
            "scope": "read:write",
            "owner": "admin@example.com",
            "environment": "production",
        }

        cred = manager.encrypt_credential(
            credential_id="with-metadata",
            service_name="api",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
            metadata=metadata,
        )

        # Retrieve and check metadata
        retrieved = manager.get_credential("with-metadata")
        assert retrieved.metadata == metadata

    def test_pbkdf2_iterations_configurable(self) -> None:
        """Test PBKDF2 iterations can be configured."""
        manager1 = CredentialManager(pbkdf2_iterations=100000)
        manager2 = CredentialManager(pbkdf2_iterations=600000)

        # Both should work but use different iteration counts
        manager1.encrypt_credential(
            credential_id="test1",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
        )

        manager2.encrypt_credential(
            credential_id="test2",
            service_name="test",
            credential_type=CredentialType.API_KEY,
            credential_value="secret",
        )

        # Should decrypt successfully
        assert manager1.decrypt_credential("test1") == "secret"
        assert manager2.decrypt_credential("test2") == "secret"
