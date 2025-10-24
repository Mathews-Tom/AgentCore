"""
TLS 1.3 Configuration

Secure TLS configuration with modern cipher suites and security hardening.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class TLSConfig:
    """TLS configuration for HTTPS connections."""

    cert_path: str | Path
    """Path to TLS certificate file"""

    key_path: str | Path
    """Path to TLS private key file"""

    ca_cert_path: str | Path | None = None
    """Path to CA certificate bundle (for client verification)"""

    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3
    """Minimum TLS version (default: TLS 1.3)"""

    max_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3
    """Maximum TLS version (default: TLS 1.3)"""

    verify_client: bool = False
    """Require client certificate verification"""

    cipher_suites: list[str] | None = None
    """Custom cipher suites (None = use secure defaults)"""

    enable_session_tickets: bool = True
    """Enable TLS session tickets for resumption"""

    enable_early_data: bool = False
    """Enable TLS 1.3 0-RTT early data (use with caution)"""


# Recommended TLS 1.3 cipher suites (ordered by preference)
TLS13_CIPHER_SUITES = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
]

# Fallback TLS 1.2 cipher suites (for compatibility, if enabled)
TLS12_CIPHER_SUITES = [
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-CHACHA20-POLY1305",
    "ECDHE-RSA-CHACHA20-POLY1305",
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES128-GCM-SHA256",
]


def create_ssl_context(config: TLSConfig) -> ssl.SSLContext:
    """
    Create SSL context with secure TLS 1.3 configuration.

    Args:
        config: TLS configuration

    Returns:
        Configured SSL context

    Raises:
        FileNotFoundError: If certificate or key files not found
        ssl.SSLError: On SSL configuration errors
    """
    # Verify certificate files exist
    cert_path = Path(config.cert_path)
    key_path = Path(config.key_path)

    if not cert_path.exists():
        raise FileNotFoundError(f"TLS certificate not found: {cert_path}")

    if not key_path.exists():
        raise FileNotFoundError(f"TLS private key not found: {key_path}")

    # Create SSL context with secure defaults
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

    # Load certificate and private key
    context.load_cert_chain(
        certfile=str(cert_path),
        keyfile=str(key_path),
    )

    # Set TLS version constraints
    context.minimum_version = config.min_version
    context.maximum_version = config.max_version

    # Configure cipher suites
    if config.cipher_suites:
        # Use custom cipher suites
        cipher_string = ":".join(config.cipher_suites)
        context.set_ciphers(cipher_string)
    else:
        # Use recommended cipher suites
        if config.min_version >= ssl.TLSVersion.TLSv1_3:
            # TLS 1.3 only
            cipher_string = ":".join(TLS13_CIPHER_SUITES)
        else:
            # TLS 1.2 + 1.3
            all_ciphers = TLS12_CIPHER_SUITES + TLS13_CIPHER_SUITES
            cipher_string = ":".join(all_ciphers)

        context.set_ciphers(cipher_string)

    # Client certificate verification
    if config.verify_client:
        context.verify_mode = ssl.CERT_REQUIRED

        if config.ca_cert_path:
            ca_path = Path(config.ca_cert_path)
            if ca_path.exists():
                context.load_verify_locations(cafile=str(ca_path))
            else:
                logger.warning(
                    "CA certificate not found",
                    ca_path=str(ca_path),
                )
    else:
        context.verify_mode = ssl.CERT_NONE

    # Security hardening options
    context.options |= ssl.OP_NO_COMPRESSION  # Disable TLS compression (CRIME)
    context.options |= ssl.OP_NO_RENEGOTIATION  # Disable renegotiation
    context.options |= ssl.OP_SINGLE_DH_USE  # Generate new DH key for each connection
    context.options |= (
        ssl.OP_SINGLE_ECDH_USE
    )  # Generate new ECDH key for each connection

    # TLS 1.3 specific options
    if config.min_version >= ssl.TLSVersion.TLSv1_3:
        # Session tickets (enabled by default in TLS 1.3)
        if not config.enable_session_tickets:
            context.options |= ssl.OP_NO_TICKET

        # Early data / 0-RTT (disabled by default for security)
        if config.enable_early_data:
            logger.warning(
                "TLS 1.3 early data (0-RTT) enabled",
                warning="Vulnerable to replay attacks - use with caution",
            )
            # Note: Python's ssl module doesn't expose OP_NO_ANTI_REPLAY
            # Early data is disabled by default

    # Set server name indication (SNI) callback
    # This would be set on the server side for multi-domain support

    logger.info(
        "SSL context created",
        min_version=config.min_version.name,
        max_version=config.max_version.name,
        verify_client=config.verify_client,
        session_tickets=config.enable_session_tickets,
        early_data=config.enable_early_data,
    )

    return context


def validate_certificate(cert_path: str | Path) -> dict[str, any]:
    """
    Validate TLS certificate and extract information.

    Args:
        cert_path: Path to certificate file

    Returns:
        Dictionary with certificate information

    Raises:
        FileNotFoundError: If certificate not found
        ssl.SSLError: On certificate validation errors
    """
    from datetime import UTC, datetime

    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    cert_path = Path(cert_path)

    if not cert_path.exists():
        raise FileNotFoundError(f"Certificate not found: {cert_path}")

    # Load certificate
    with open(cert_path, "rb") as f:
        cert_data = f.read()

    cert = x509.load_pem_x509_certificate(cert_data, default_backend())

    # Extract information
    subject = cert.subject
    issuer = cert.issuer
    not_before = cert.not_valid_before_utc
    not_after = cert.not_valid_after_utc
    now = datetime.now(UTC)

    # Check validity
    is_valid = not_before <= now <= not_after
    days_until_expiry = (not_after - now).days

    info = {
        "subject": str(subject),
        "issuer": str(issuer),
        "not_before": not_before.isoformat(),
        "not_after": not_after.isoformat(),
        "is_valid": is_valid,
        "days_until_expiry": days_until_expiry,
        "serial_number": cert.serial_number,
        "version": cert.version.name,
    }

    # Extract SAN (Subject Alternative Names)
    try:
        san_extension = cert.extensions.get_extension_for_oid(
            x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
        san_names = san_extension.value.get_values_for_type(x509.DNSName)
        info["san_names"] = list(san_names)
    except x509.ExtensionNotFound:
        info["san_names"] = []

    return info
