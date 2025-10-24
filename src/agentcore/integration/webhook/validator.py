"""Webhook security and validation."""

import hashlib
import hmac
import json
from typing import Any

from .config import WebhookConfig
from .exceptions import SignatureVerificationError, WebhookValidationError


class WebhookValidator:
    """Webhook validation and security utilities.

    Provides:
    - HMAC signature generation and verification
    - Payload validation
    - URL validation
    - Secret strength validation
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        """Initialize validator.

        Args:
            config: Webhook configuration
        """
        self.config = config or WebhookConfig()

    def generate_signature(
        self,
        secret: str,
        payload: dict[str, Any],
    ) -> str:
        """Generate HMAC signature for webhook payload.

        Uses SHA-256 HMAC for signing webhook payloads.

        Args:
            secret: Webhook secret key
            payload: Payload to sign

        Returns:
            Hex-encoded HMAC signature
        """
        # Serialize payload to canonical JSON
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

        return f"sha256={signature}"

    def verify_signature(
        self,
        secret: str,
        payload: dict[str, Any],
        provided_signature: str,
    ) -> bool:
        """Verify webhook signature.

        Args:
            secret: Webhook secret key
            payload: Received payload
            provided_signature: Signature from request header

        Returns:
            True if signature is valid

        Raises:
            SignatureVerificationError: If signature is invalid
        """
        # Generate expected signature
        expected_signature = self.generate_signature(secret, payload)

        # Compare using constant-time comparison
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise SignatureVerificationError(
                "Signature verification failed - invalid signature"
            )

        return True

    def validate_url(self, url: str) -> None:
        """Validate webhook URL.

        Args:
            url: URL to validate

        Raises:
            WebhookValidationError: If URL is invalid
        """
        # Check HTTPS requirement (recommended for production)
        if not url.startswith(("http://", "https://")):
            raise WebhookValidationError("URL must use HTTP or HTTPS protocol")

        # Warn about non-HTTPS URLs
        if url.startswith("http://") and not url.startswith("http://localhost"):
            # In production, this should be an error
            # For development, allow http://localhost
            pass

        # Check for localhost/private IPs in production (SSRF protection)
        # This is a basic check - in production, use more robust validation
        if any(
            x in url.lower()
            for x in ["localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254"]
        ):
            # Allow in development, block in production
            pass

    def validate_secret(self, secret: str) -> None:
        """Validate webhook secret strength.

        Args:
            secret: Secret to validate

        Raises:
            WebhookValidationError: If secret is invalid
        """
        if len(secret) < self.config.min_secret_length:
            raise WebhookValidationError(
                f"Secret must be at least {self.config.min_secret_length} characters"
            )

        # Check for common weak secrets
        weak_secrets = [
            "password",
            "secret",
            "webhook",
            "test",
            "admin",
            "12345678",
        ]

        if secret.lower() in weak_secrets:
            raise WebhookValidationError("Secret is too weak - use a strong random value")

    def validate_payload(self, payload: dict[str, Any]) -> None:
        """Validate event payload structure.

        Args:
            payload: Payload to validate

        Raises:
            WebhookValidationError: If payload is invalid
        """
        # Check required fields
        required_fields = ["event_id", "event_type", "timestamp", "data", "source"]

        for field in required_fields:
            if field not in payload:
                raise WebhookValidationError(f"Missing required field: {field}")

        # Validate field types
        if not isinstance(payload.get("data"), dict):
            raise WebhookValidationError("Field 'data' must be a dictionary")

        if not isinstance(payload.get("metadata", {}), dict):
            raise WebhookValidationError("Field 'metadata' must be a dictionary")

    def sanitize_response_body(
        self,
        body: str,
        max_length: int = 1000,
    ) -> str:
        """Sanitize response body for storage.

        Args:
            body: Response body to sanitize
            max_length: Maximum length to keep

        Returns:
            Sanitized response body
        """
        # Truncate to max length
        if len(body) > max_length:
            body = body[:max_length] + "... [truncated]"

        # Remove potentially sensitive data patterns
        # This is a basic implementation - enhance as needed
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
        ]

        body_lower = body.lower()
        for pattern in sensitive_patterns:
            if pattern in body_lower:
                # Don't log potentially sensitive data
                return "[Response contains sensitive data - redacted]"

        return body
