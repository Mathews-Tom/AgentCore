"""
Tests for model encryption
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentcore.dspy_optimization.security.encryption import (
    EncryptedModel,
    EncryptionConfig,
    ModelEncryption,
)


class TestEncryptionConfig:
    """Tests for EncryptionConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = EncryptionConfig()
        assert config.key_size == 32
        assert config.salt_size == 16
        assert config.iterations == 100000
        assert config.enable_key_rotation is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = EncryptionConfig(
            key_size=32,
            iterations=200000,
            enable_key_rotation=False,
        )
        assert config.key_size == 32
        assert config.iterations == 200000
        assert config.enable_key_rotation is False


class TestModelEncryption:
    """Tests for ModelEncryption"""

    @pytest.fixture
    def encryption(self) -> ModelEncryption:
        """Create encryption instance"""
        return ModelEncryption()

    @pytest.fixture
    def sample_data(self) -> bytes:
        """Sample model data"""
        return b"This is a test model with sensitive data"

    def test_initialization(self, encryption: ModelEncryption):
        """Test encryption service initialization"""
        assert encryption.config.key_size == 32
        assert encryption._current_key_version == 1
        assert encryption._master_key is not None

    def test_encrypt_model(self, encryption: ModelEncryption, sample_data: bytes):
        """Test model encryption"""
        metadata = {"model_name": "test_model", "version": "1.0"}
        encrypted = encryption.encrypt_model(sample_data, metadata)

        assert isinstance(encrypted, EncryptedModel)
        assert encrypted.encrypted_data != sample_data
        assert len(encrypted.encrypted_data) > 0
        assert len(encrypted.salt) == 16
        assert len(encrypted.iv) == 16
        assert encrypted.metadata == metadata
        assert encrypted.key_version == 1

    def test_decrypt_model(self, encryption: ModelEncryption, sample_data: bytes):
        """Test model decryption"""
        encrypted = encryption.encrypt_model(sample_data)
        decrypted = encryption.decrypt_model(encrypted)

        assert decrypted == sample_data

    def test_encrypt_decrypt_roundtrip(
        self, encryption: ModelEncryption, sample_data: bytes
    ):
        """Test encryption-decryption roundtrip"""
        encrypted = encryption.encrypt_model(sample_data)
        decrypted = encryption.decrypt_model(encrypted)

        assert decrypted == sample_data

    def test_different_encryptions_produce_different_output(
        self, encryption: ModelEncryption, sample_data: bytes
    ):
        """Test that same data produces different encrypted output"""
        encrypted1 = encryption.encrypt_model(sample_data)
        encrypted2 = encryption.encrypt_model(sample_data)

        # Different salt/IV should produce different output
        assert encrypted1.encrypted_data != encrypted2.encrypted_data
        assert encrypted1.salt != encrypted2.salt
        assert encrypted1.iv != encrypted2.iv

    def test_tampered_data_fails_decryption(
        self, encryption: ModelEncryption, sample_data: bytes
    ):
        """Test that tampered data fails decryption"""
        encrypted = encryption.encrypt_model(sample_data)

        # Tamper with encrypted data
        tampered_data = bytearray(encrypted.encrypted_data)
        tampered_data[0] ^= 1
        encrypted.encrypted_data = bytes(tampered_data)

        with pytest.raises(ValueError, match="Model decryption failed"):
            encryption.decrypt_model(encrypted)

    def test_save_and_load_encrypted_model(
        self, encryption: ModelEncryption, sample_data: bytes, tmp_path: Path
    ):
        """Test saving and loading encrypted model"""
        encrypted = encryption.encrypt_model(sample_data)
        filepath = tmp_path / "encrypted_model.json"

        encryption.save_encrypted_model(encrypted, filepath)
        assert filepath.exists()

        loaded = encryption.load_encrypted_model(filepath)
        assert loaded.encrypted_data == encrypted.encrypted_data
        assert loaded.salt == encrypted.salt
        assert loaded.iv == encrypted.iv
        assert loaded.metadata == encrypted.metadata
        assert loaded.key_version == encrypted.key_version

    def test_saved_model_can_be_decrypted(
        self, encryption: ModelEncryption, sample_data: bytes, tmp_path: Path
    ):
        """Test that saved model can be decrypted"""
        encrypted = encryption.encrypt_model(sample_data)
        filepath = tmp_path / "encrypted_model.json"

        encryption.save_encrypted_model(encrypted, filepath)
        loaded = encryption.load_encrypted_model(filepath)
        decrypted = encryption.decrypt_model(loaded)

        assert decrypted == sample_data

    def test_key_rotation(self, encryption: ModelEncryption):
        """Test key rotation"""
        initial_version = encryption._current_key_version
        initial_key = encryption._master_key

        new_version = encryption.rotate_key()

        assert new_version == initial_version + 1
        assert encryption._current_key_version == new_version
        assert encryption._master_key != initial_key

    def test_encryption_after_key_rotation(
        self, encryption: ModelEncryption, sample_data: bytes
    ):
        """Test encryption after key rotation"""
        encrypted_v1 = encryption.encrypt_model(sample_data)
        assert encrypted_v1.key_version == 1

        encryption.rotate_key()

        encrypted_v2 = encryption.encrypt_model(sample_data)
        assert encrypted_v2.key_version == 2

    def test_get_key_info(self, encryption: ModelEncryption):
        """Test getting key information"""
        info = encryption.get_key_info()

        assert info["current_version"] == 1
        assert info["last_rotation"] is not None
        assert info["total_rotations"] == 1
        assert "days_since_rotation" in info
        assert "rotation_needed" in info

    def test_verify_integrity_valid(
        self, encryption: ModelEncryption, sample_data: bytes
    ):
        """Test integrity verification of valid model"""
        encrypted = encryption.encrypt_model(sample_data)
        assert encryption.verify_integrity(encrypted) is True

    def test_verify_integrity_tampered(
        self, encryption: ModelEncryption, sample_data: bytes
    ):
        """Test integrity verification of tampered model"""
        encrypted = encryption.encrypt_model(sample_data)

        # Tamper with data
        tampered = bytearray(encrypted.encrypted_data)
        tampered[0] ^= 1
        encrypted.encrypted_data = bytes(tampered)

        assert encryption.verify_integrity(encrypted) is False

    def test_large_model_encryption(self, encryption: ModelEncryption):
        """Test encryption of large model"""
        large_data = b"x" * 1_000_000  # 1MB
        encrypted = encryption.encrypt_model(large_data)
        decrypted = encryption.decrypt_model(encrypted)

        assert decrypted == large_data

    def test_empty_model_encryption(self, encryption: ModelEncryption):
        """Test encryption of empty model"""
        empty_data = b""
        encrypted = encryption.encrypt_model(empty_data)
        decrypted = encryption.decrypt_model(encrypted)

        assert decrypted == empty_data

    def test_metadata_preservation(self, encryption: ModelEncryption, sample_data: bytes):
        """Test that metadata is preserved"""
        metadata = {
            "model_name": "test_model",
            "version": "2.0",
            "author": "test_user",
            "created_at": datetime.now(UTC).isoformat(),
        }

        encrypted = encryption.encrypt_model(sample_data, metadata)
        assert encrypted.metadata == metadata

    def test_encryption_timestamp(self, encryption: ModelEncryption, sample_data: bytes):
        """Test that encryption timestamp is recorded"""
        before = datetime.now(UTC)
        encrypted = encryption.encrypt_model(sample_data)
        after = datetime.now(UTC)

        assert before <= encrypted.encrypted_at <= after


class TestEncryptionIntegration:
    """Integration tests for encryption"""

    def test_multiple_models_with_same_service(self):
        """Test encrypting multiple models with same service"""
        encryption = ModelEncryption()

        models = [
            b"model_1_data",
            b"model_2_data",
            b"model_3_data",
        ]

        encrypted_models = [encryption.encrypt_model(m) for m in models]
        decrypted_models = [encryption.decrypt_model(e) for e in encrypted_models]

        assert decrypted_models == models

    def test_cross_version_decryption_same_key(self):
        """Test decryption across versions with same key"""
        encryption = ModelEncryption()
        data = b"test_data"

        # Encrypt with version 1
        encrypted_v1 = encryption.encrypt_model(data)
        assert encrypted_v1.key_version == 1

        # Rotate key
        encryption.rotate_key()

        # Should not be able to decrypt v1 with v2 key
        with pytest.raises(ValueError):
            encryption.decrypt_model(encrypted_v1)
