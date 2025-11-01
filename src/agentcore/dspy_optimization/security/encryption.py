"""
Model Encryption

Provides AES-256 encryption for model artifacts with key management and rotation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class EncryptionConfig(BaseModel):
    """Configuration for model encryption"""

    key_size: int = Field(default=32, description="Key size in bytes (32 = AES-256)")
    salt_size: int = Field(default=16, description="Salt size in bytes")
    iterations: int = Field(default=100000, description="PBKDF2 iterations")
    key_env_var: str = Field(
        default="DSPY_ENCRYPTION_KEY", description="Environment variable for master key"
    )
    enable_key_rotation: bool = Field(
        default=True, description="Enable automatic key rotation"
    )
    key_rotation_days: int = Field(default=90, description="Days between key rotations")


@dataclass
class EncryptedModel:
    """Encrypted model artifact"""

    encrypted_data: bytes
    salt: bytes
    iv: bytes
    metadata: dict[str, Any]
    key_version: int
    encrypted_at: datetime


class ModelEncryption:
    """
    Model encryption service using AES-256-GCM.

    Provides secure encryption/decryption of model artifacts with key management.
    """

    def __init__(self, config: EncryptionConfig | None = None):
        self.config = config or EncryptionConfig()
        self.logger = structlog.get_logger()

        # Key management
        self._master_key: bytes | None = None
        self._current_key_version = 1
        self._key_rotation_history: dict[int, datetime] = {}

        # Initialize master key
        self._initialize_master_key()

        self.logger.info(
            "model_encryption_initialized",
            key_size=self.config.key_size,
            key_rotation_enabled=self.config.enable_key_rotation,
        )

    def _initialize_master_key(self) -> None:
        """Initialize or load master encryption key"""
        key_str = os.getenv(self.config.key_env_var)

        if not key_str:
            self.logger.warning(
                "master_key_not_found",
                env_var=self.config.key_env_var,
                action="generating_temporary_key",
            )
            # Generate temporary key for development
            self._master_key = os.urandom(self.config.key_size)
        else:
            # Derive key from environment variable
            self._master_key = base64.b64decode(key_str.encode())

            if len(self._master_key) != self.config.key_size:
                raise ValueError(
                    f"Master key must be {self.config.key_size} bytes, "
                    f"got {len(self._master_key)}"
                )

        self._key_rotation_history[self._current_key_version] = datetime.now(UTC)

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_size,
            salt=salt,
            iterations=self.config.iterations,
            backend=default_backend(),
        )
        return kdf.derive(self._master_key)

    def encrypt_model(
        self, model_data: bytes, metadata: dict[str, Any] | None = None
    ) -> EncryptedModel:
        """
        Encrypt model artifact using AES-256-GCM.

        Args:
            model_data: Raw model data to encrypt
            metadata: Optional metadata to store with encrypted model

        Returns:
            EncryptedModel with encrypted data and metadata

        Raises:
            ValueError: If encryption fails
        """
        try:
            # Generate salt and IV
            salt = os.urandom(self.config.salt_size)
            iv = os.urandom(16)  # AES GCM IV size

            # Derive encryption key
            key = self._derive_key(salt)

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key), modes.GCM(iv), backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Encrypt data
            encrypted_data = encryptor.update(model_data) + encryptor.finalize()

            # Get authentication tag
            tag = encryptor.tag

            # Combine encrypted data with tag
            encrypted_with_tag = encrypted_data + tag

            encrypted_model = EncryptedModel(
                encrypted_data=encrypted_with_tag,
                salt=salt,
                iv=iv,
                metadata=metadata or {},
                key_version=self._current_key_version,
                encrypted_at=datetime.now(UTC),
            )

            self.logger.info(
                "model_encrypted",
                data_size=len(model_data),
                encrypted_size=len(encrypted_with_tag),
                key_version=self._current_key_version,
            )

            return encrypted_model

        except Exception as e:
            self.logger.error("model_encryption_failed", error=str(e))
            raise ValueError(f"Model encryption failed: {e}") from e

    def decrypt_model(self, encrypted_model: EncryptedModel) -> bytes:
        """
        Decrypt model artifact.

        Args:
            encrypted_model: Encrypted model to decrypt

        Returns:
            Decrypted model data

        Raises:
            ValueError: If decryption fails
        """
        try:
            # Derive decryption key
            key = self._derive_key(encrypted_model.salt)

            # Split encrypted data and tag
            encrypted_data = encrypted_model.encrypted_data[:-16]
            tag = encrypted_model.encrypted_data[-16:]

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(encrypted_model.iv, tag),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()

            # Decrypt data
            decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

            self.logger.info(
                "model_decrypted",
                encrypted_size=len(encrypted_model.encrypted_data),
                decrypted_size=len(decrypted_data),
                key_version=encrypted_model.key_version,
            )

            return decrypted_data

        except Exception as e:
            self.logger.error("model_decryption_failed", error=str(e))
            raise ValueError(f"Model decryption failed: {e}") from e

    def save_encrypted_model(
        self, encrypted_model: EncryptedModel, filepath: Path
    ) -> None:
        """
        Save encrypted model to file.

        Args:
            encrypted_model: Encrypted model to save
            filepath: Path to save encrypted model
        """
        try:
            data = {
                "encrypted_data": base64.b64encode(
                    encrypted_model.encrypted_data
                ).decode(),
                "salt": base64.b64encode(encrypted_model.salt).decode(),
                "iv": base64.b64encode(encrypted_model.iv).decode(),
                "metadata": encrypted_model.metadata,
                "key_version": encrypted_model.key_version,
                "encrypted_at": encrypted_model.encrypted_at.isoformat(),
            }

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f)

            self.logger.info(
                "encrypted_model_saved",
                filepath=str(filepath),
                key_version=encrypted_model.key_version,
            )

        except Exception as e:
            self.logger.error("save_encrypted_model_failed", error=str(e))
            raise ValueError(f"Failed to save encrypted model: {e}") from e

    def load_encrypted_model(self, filepath: Path) -> EncryptedModel:
        """
        Load encrypted model from file.

        Args:
            filepath: Path to encrypted model file

        Returns:
            EncryptedModel instance

        Raises:
            ValueError: If loading fails
        """
        try:
            with open(filepath) as f:
                data = json.load(f)

            encrypted_model = EncryptedModel(
                encrypted_data=base64.b64decode(data["encrypted_data"]),
                salt=base64.b64decode(data["salt"]),
                iv=base64.b64decode(data["iv"]),
                metadata=data["metadata"],
                key_version=data["key_version"],
                encrypted_at=datetime.fromisoformat(data["encrypted_at"]),
            )

            self.logger.info(
                "encrypted_model_loaded",
                filepath=str(filepath),
                key_version=encrypted_model.key_version,
            )

            return encrypted_model

        except Exception as e:
            self.logger.error("load_encrypted_model_failed", error=str(e))
            raise ValueError(f"Failed to load encrypted model: {e}") from e

    def rotate_key(self) -> int:
        """
        Rotate encryption key.

        Returns:
            New key version

        Raises:
            ValueError: If key rotation fails
        """
        try:
            # Generate new master key
            self._master_key = os.urandom(self.config.key_size)
            self._current_key_version += 1
            self._key_rotation_history[self._current_key_version] = datetime.now(UTC)

            self.logger.info(
                "key_rotated", new_version=self._current_key_version, timestamp=datetime.now(UTC)
            )

            return self._current_key_version

        except Exception as e:
            self.logger.error("key_rotation_failed", error=str(e))
            raise ValueError(f"Key rotation failed: {e}") from e

    def get_key_info(self) -> dict[str, Any]:
        """Get current key information"""
        last_rotation = self._key_rotation_history.get(self._current_key_version)
        days_since_rotation = (
            (datetime.now(UTC) - last_rotation).days if last_rotation else None
        )

        return {
            "current_version": self._current_key_version,
            "last_rotation": last_rotation.isoformat() if last_rotation else None,
            "days_since_rotation": days_since_rotation,
            "rotation_needed": (
                days_since_rotation >= self.config.key_rotation_days
                if days_since_rotation
                else False
            ),
            "total_rotations": len(self._key_rotation_history),
        }

    def verify_integrity(self, encrypted_model: EncryptedModel) -> bool:
        """
        Verify integrity of encrypted model.

        Args:
            encrypted_model: Encrypted model to verify

        Returns:
            True if integrity check passes
        """
        try:
            # Attempt decryption (GCM mode verifies integrity automatically)
            self.decrypt_model(encrypted_model)
            return True
        except Exception:
            return False
