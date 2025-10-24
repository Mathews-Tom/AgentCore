"""Credential encryption and rotation management.

Provides secure storage and management of external service credentials
with encryption at rest, version tracking, and automatic rotation support.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, SecretStr

logger = structlog.get_logger(__name__)


class CredentialType(str, Enum):
    """Supported credential types."""

    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    JWT = "jwt"
    SSH_KEY = "ssh_key"
    DATABASE = "database"
    CERTIFICATE = "certificate"


class CredentialStatus(str, Enum):
    """Credential lifecycle status."""

    ACTIVE = "active"
    ROTATING = "rotating"
    EXPIRED = "expired"
    REVOKED = "revoked"


class EncryptedCredential(BaseModel):
    """Encrypted credential with metadata.

    Stores encrypted credentials with version tracking, rotation support,
    and security metadata for audit and compliance.
    """

    credential_id: str = Field(
        description="Unique credential identifier",
    )
    service_name: str = Field(
        description="Service this credential is for",
    )
    credential_type: CredentialType = Field(
        description="Type of credential",
    )
    encrypted_value: str = Field(
        description="Base64-encoded encrypted credential value",
    )
    version: int = Field(
        default=1,
        description="Credential version for rotation tracking",
        ge=1,
    )
    status: CredentialStatus = Field(
        default=CredentialStatus.ACTIVE,
        description="Current credential status",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Expiration timestamp (None = no expiration)",
    )
    last_rotated_at: datetime | None = Field(
        default=None,
        description="Last rotation timestamp",
    )
    rotation_interval_days: int | None = Field(
        default=None,
        description="Automatic rotation interval in days",
        ge=1,
        le=365,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., scopes, permissions)",
    )

    def needs_rotation(self) -> bool:
        """Check if credential needs rotation.

        Returns:
            True if credential should be rotated
        """
        if self.status != CredentialStatus.ACTIVE:
            return False

        # Check expiration
        if self.expires_at:
            now = datetime.now(timezone.utc)
            if now >= self.expires_at:
                return True

        # Check rotation interval
        if self.rotation_interval_days and self.last_rotated_at:
            now = datetime.now(timezone.utc)
            rotation_due = self.last_rotated_at + timedelta(
                days=self.rotation_interval_days
            )
            return now >= rotation_due

        return False

    def is_expired(self) -> bool:
        """Check if credential is expired.

        Returns:
            True if credential has expired
        """
        if self.expires_at:
            now = datetime.now(timezone.utc)
            return now >= self.expires_at
        return False


class CredentialManager:
    """Credential encryption and rotation manager.

    Provides secure credential storage with encryption at rest using Fernet
    symmetric encryption, key derivation with PBKDF2, and automatic rotation
    support with version tracking.
    """

    def __init__(
        self,
        master_key: str | None = None,
        pbkdf2_iterations: int = 600000,
    ) -> None:
        """Initialize credential manager.

        Args:
            master_key: Base64-encoded master key (32 bytes). If None, generates new key.
            pbkdf2_iterations: PBKDF2 iterations for key derivation (default: 600000)
        """
        if master_key:
            self._master_key = base64.urlsafe_b64decode(master_key)
        else:
            self._master_key = secrets.token_bytes(32)

        self._pbkdf2_iterations = pbkdf2_iterations
        self._credentials: dict[str, EncryptedCredential] = {}

        logger.info(
            "credential_manager_initialized",
            pbkdf2_iterations=pbkdf2_iterations,
        )

    def get_master_key_b64(self) -> str:
        """Get base64-encoded master key for storage.

        Returns:
            Base64-encoded master key
        """
        return base64.urlsafe_b64encode(self._master_key).decode()

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2.

        Args:
            salt: 16-byte salt for key derivation

        Returns:
            32-byte derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._pbkdf2_iterations,
        )
        return kdf.derive(self._master_key)

    def _get_fernet(self, salt: bytes) -> Fernet:
        """Get Fernet instance with derived key.

        Args:
            salt: Salt for key derivation

        Returns:
            Fernet instance for encryption/decryption
        """
        derived_key = self._derive_key(salt)
        key_b64 = base64.urlsafe_b64encode(derived_key)
        return Fernet(key_b64)

    def encrypt_credential(
        self,
        credential_id: str,
        service_name: str,
        credential_type: CredentialType,
        credential_value: str,
        expires_at: datetime | None = None,
        rotation_interval_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EncryptedCredential:
        """Encrypt and store a credential.

        Args:
            credential_id: Unique identifier for credential
            service_name: Name of service credential is for
            credential_type: Type of credential
            credential_value: Plain-text credential value
            expires_at: Optional expiration timestamp
            rotation_interval_days: Optional automatic rotation interval
            metadata: Optional additional metadata

        Returns:
            Encrypted credential object

        Raises:
            ValueError: If credential_id already exists
        """
        if credential_id in self._credentials:
            raise ValueError(f"Credential {credential_id} already exists")

        # Generate salt for this credential
        salt = secrets.token_bytes(16)

        # Encrypt credential value
        fernet = self._get_fernet(salt)
        encrypted_bytes = fernet.encrypt(credential_value.encode())

        # Combine salt + encrypted data for storage
        combined = salt + encrypted_bytes
        encrypted_b64 = base64.urlsafe_b64encode(combined).decode()

        # Create encrypted credential object
        encrypted_cred = EncryptedCredential(
            credential_id=credential_id,
            service_name=service_name,
            credential_type=credential_type,
            encrypted_value=encrypted_b64,
            expires_at=expires_at,
            rotation_interval_days=rotation_interval_days,
            metadata=metadata or {},
        )

        # Store credential
        self._credentials[credential_id] = encrypted_cred

        logger.info(
            "credential_encrypted",
            credential_id=credential_id,
            service_name=service_name,
            credential_type=credential_type.value,
            has_expiration=expires_at is not None,
            rotation_interval_days=rotation_interval_days,
        )

        return encrypted_cred

    def decrypt_credential(self, credential_id: str) -> str:
        """Decrypt and retrieve credential value.

        Args:
            credential_id: Credential identifier

        Returns:
            Decrypted credential value

        Raises:
            KeyError: If credential not found
            ValueError: If credential is expired or revoked
            InvalidToken: If decryption fails
        """
        if credential_id not in self._credentials:
            raise KeyError(f"Credential {credential_id} not found")

        encrypted_cred = self._credentials[credential_id]

        # Check status
        if encrypted_cred.status == CredentialStatus.REVOKED:
            raise ValueError(f"Credential {credential_id} is revoked")

        if encrypted_cred.is_expired():
            raise ValueError(f"Credential {credential_id} is expired")

        # Decode combined data
        combined = base64.urlsafe_b64decode(encrypted_cred.encrypted_value)

        # Extract salt and encrypted data
        salt = combined[:16]
        encrypted_bytes = combined[16:]

        # Decrypt
        fernet = self._get_fernet(salt)
        try:
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            credential_value = decrypted_bytes.decode()

            logger.debug(
                "credential_decrypted",
                credential_id=credential_id,
                service_name=encrypted_cred.service_name,
            )

            return credential_value

        except InvalidToken as e:
            logger.error(
                "credential_decryption_failed",
                credential_id=credential_id,
                error=str(e),
            )
            raise

    def rotate_credential(
        self,
        credential_id: str,
        new_credential_value: str,
    ) -> EncryptedCredential:
        """Rotate credential to new value.

        Updates credential with new encrypted value, increments version,
        and updates rotation timestamp.

        Args:
            credential_id: Credential to rotate
            new_credential_value: New credential value

        Returns:
            Updated encrypted credential

        Raises:
            KeyError: If credential not found
        """
        if credential_id not in self._credentials:
            raise KeyError(f"Credential {credential_id} not found")

        old_cred = self._credentials[credential_id]

        # Generate new salt
        salt = secrets.token_bytes(16)

        # Encrypt new value
        fernet = self._get_fernet(salt)
        encrypted_bytes = fernet.encrypt(new_credential_value.encode())

        # Combine salt + encrypted data
        combined = salt + encrypted_bytes
        encrypted_b64 = base64.urlsafe_b64encode(combined).decode()

        # Update credential
        old_cred.encrypted_value = encrypted_b64
        old_cred.version += 1
        old_cred.last_rotated_at = datetime.now(timezone.utc)
        old_cred.status = CredentialStatus.ACTIVE

        logger.info(
            "credential_rotated",
            credential_id=credential_id,
            service_name=old_cred.service_name,
            version=old_cred.version,
        )

        return old_cred

    def revoke_credential(self, credential_id: str) -> None:
        """Revoke a credential.

        Args:
            credential_id: Credential to revoke

        Raises:
            KeyError: If credential not found
        """
        if credential_id not in self._credentials:
            raise KeyError(f"Credential {credential_id} not found")

        self._credentials[credential_id].status = CredentialStatus.REVOKED

        logger.warning(
            "credential_revoked",
            credential_id=credential_id,
            service_name=self._credentials[credential_id].service_name,
        )

    def get_credential(self, credential_id: str) -> EncryptedCredential:
        """Get encrypted credential metadata.

        Args:
            credential_id: Credential identifier

        Returns:
            Encrypted credential object (value still encrypted)

        Raises:
            KeyError: If credential not found
        """
        if credential_id not in self._credentials:
            raise KeyError(f"Credential {credential_id} not found")

        return self._credentials[credential_id]

    def list_credentials(
        self,
        service_name: str | None = None,
        credential_type: CredentialType | None = None,
        status: CredentialStatus | None = None,
    ) -> list[EncryptedCredential]:
        """List credentials with optional filtering.

        Args:
            service_name: Filter by service name
            credential_type: Filter by credential type
            status: Filter by status

        Returns:
            List of encrypted credentials
        """
        credentials = list(self._credentials.values())

        if service_name:
            credentials = [c for c in credentials if c.service_name == service_name]

        if credential_type:
            credentials = [c for c in credentials if c.credential_type == credential_type]

        if status:
            credentials = [c for c in credentials if c.status == status]

        return credentials

    def check_rotations_needed(self) -> list[EncryptedCredential]:
        """Check which credentials need rotation.

        Returns:
            List of credentials that need rotation
        """
        return [
            cred
            for cred in self._credentials.values()
            if cred.needs_rotation()
        ]

    @staticmethod
    def generate_master_key() -> str:
        """Generate a new master key for credential encryption.

        Returns:
            Base64-encoded 32-byte master key
        """
        key = secrets.token_bytes(32)
        return base64.urlsafe_b64encode(key).decode()

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Compare two strings in constant time.

        Prevents timing attacks when comparing sensitive data.

        Args:
            a: First string
            b: Second string

        Returns:
            True if strings are equal
        """
        return hmac.compare_digest(a.encode(), b.encode())
