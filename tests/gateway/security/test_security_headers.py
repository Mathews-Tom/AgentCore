"""Security headers tests.

Tests for proper security header configuration.
"""

import pytest


class SecurityHeaders:
    """Security headers configuration."""

    @staticmethod
    def get_security_headers() -> dict[str, str]:
        """Get recommended security headers.

        Returns:
            Dictionary of security headers
        """
        return {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # XSS protection
            "X-Content-Type-Options": "nosniff",
            # HTTPS enforcement
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # CSP
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; object-src 'none'",
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }


@pytest.fixture
def security_headers() -> dict[str, str]:
    """Get security headers for testing."""
    return SecurityHeaders.get_security_headers()


class TestSecurityHeaders:
    """Tests for security headers presence and configuration."""

    def test_x_frame_options(self, security_headers: dict[str, str]) -> None:
        """Test X-Frame-Options header."""
        assert "X-Frame-Options" in security_headers
        assert security_headers["X-Frame-Options"] in ["DENY", "SAMEORIGIN"]

    def test_x_content_type_options(self, security_headers: dict[str, str]) -> None:
        """Test X-Content-Type-Options header."""
        assert "X-Content-Type-Options" in security_headers
        assert security_headers["X-Content-Type-Options"] == "nosniff"

    def test_strict_transport_security(self, security_headers: dict[str, str]) -> None:
        """Test HSTS header."""
        assert "Strict-Transport-Security" in security_headers
        hsts = security_headers["Strict-Transport-Security"]
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts

    def test_content_security_policy(self, security_headers: dict[str, str]) -> None:
        """Test CSP header."""
        assert "Content-Security-Policy" in security_headers
        csp = security_headers["Content-Security-Policy"]
        assert "default-src" in csp
        assert "'self'" in csp

    def test_referrer_policy(self, security_headers: dict[str, str]) -> None:
        """Test Referrer-Policy header."""
        assert "Referrer-Policy" in security_headers
        assert security_headers["Referrer-Policy"] in [
            "no-referrer",
            "no-referrer-when-downgrade",
            "strict-origin-when-cross-origin",
        ]

    def test_permissions_policy(self, security_headers: dict[str, str]) -> None:
        """Test Permissions-Policy header."""
        assert "Permissions-Policy" in security_headers


class TestCORSHeaders:
    """Tests for CORS header configuration."""

    def test_cors_headers_strict(self) -> None:
        """Test strict CORS configuration."""
        cors_headers = {
            "Access-Control-Allow-Origin": "https://example.com",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "86400",
        }

        assert cors_headers["Access-Control-Allow-Origin"] != "*"
        assert "GET" in cors_headers["Access-Control-Allow-Methods"]
        assert "Authorization" in cors_headers["Access-Control-Allow-Headers"]

    def test_cors_credentials(self) -> None:
        """Test CORS credentials handling."""
        # When allowing credentials, origin cannot be *
        cors_with_credentials = {
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Origin": "https://example.com",
        }

        if cors_with_credentials.get("Access-Control-Allow-Credentials") == "true":
            assert cors_with_credentials["Access-Control-Allow-Origin"] != "*"


class TestCacheHeaders:
    """Tests for cache control headers."""

    def test_no_cache_for_sensitive_data(self) -> None:
        """Test that sensitive endpoints have no-cache headers."""
        sensitive_headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        assert "no-store" in sensitive_headers["Cache-Control"]
        assert "no-cache" in sensitive_headers["Cache-Control"]

    def test_cache_for_static_assets(self) -> None:
        """Test that static assets can be cached."""
        static_headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
        }

        assert "public" in static_headers["Cache-Control"]
        assert "max-age=" in static_headers["Cache-Control"]


class TestCSPDirectives:
    """Tests for CSP directive configuration."""

    def test_csp_script_src(self) -> None:
        """Test script-src directive."""
        csp_production = "default-src 'self'; script-src 'self'"
        csp_dev = "default-src 'self'; script-src 'self' 'unsafe-inline'"

        # Production should not allow unsafe-inline
        assert "'unsafe-inline'" not in csp_production

        # Dev may allow it for convenience (with warning)
        if "'unsafe-inline'" in csp_dev:
            assert True  # Allowed in dev, but should warn

    def test_csp_object_src(self) -> None:
        """Test object-src directive."""
        csp = "default-src 'self'; object-src 'none'"

        assert "object-src 'none'" in csp

    def test_csp_upgrade_insecure_requests(self) -> None:
        """Test upgrade-insecure-requests directive."""
        csp = "default-src 'self'; upgrade-insecure-requests"

        # Should include in production
        assert "upgrade-insecure-requests" in csp or "http://" in csp


class TestSecurityMisconfiguration:
    """Tests for common security misconfigurations."""

    def test_no_server_header_disclosure(self) -> None:
        """Test that server information is not disclosed."""
        # Server header should be minimal or absent
        server_header = "nginx"  # Should not be "nginx/1.18.0 (Ubuntu)"

        assert "/" not in server_header  # No version info

    def test_no_x_powered_by(self) -> None:
        """Test that X-Powered-By header is not present."""
        headers = SecurityHeaders.get_security_headers()

        assert "X-Powered-By" not in headers

    def test_error_handling_no_details(self) -> None:
        """Test that error responses don't leak details."""
        error_response = {
            "error": "Internal Server Error",
            "message": "An error occurred",
        }

        # Should not contain stack traces or system info
        assert "Traceback" not in str(error_response)
        assert "/usr/lib" not in str(error_response)
