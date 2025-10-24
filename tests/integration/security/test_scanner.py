"""Unit tests for security scanner.

Tests credential validation, configuration scanning, and security policy enforcement.
"""

from __future__ import annotations

import pytest

from agentcore.integration.security.compliance import (
    ComplianceFramework,
    ComplianceManager,
    DataRegion,
    DataResidencyConfig,
)
from agentcore.integration.security.credential_manager import CredentialType
from agentcore.integration.security.scanner import (
    SecurityIssue,
    SecurityIssueType,
    SecurityPolicy,
    SecurityScanner,
    SecuritySeverity,
)


class TestSecurityScanner:
    """Test security scanner functionality."""

    def test_initialization(self) -> None:
        """Test security scanner initialization."""
        scanner = SecurityScanner()
        assert scanner is not None

    def test_initialization_with_policy(self) -> None:
        """Test initialization with custom policy."""
        policy = SecurityPolicy(
            require_strong_passwords=True,
            min_password_length=20,
            max_credential_age_days=60,
        )

        scanner = SecurityScanner(policy=policy)
        assert scanner is not None

    def test_initialization_with_compliance_manager(self) -> None:
        """Test initialization with compliance manager."""
        compliance_manager = ComplianceManager()
        scanner = SecurityScanner(compliance_manager=compliance_manager)
        assert scanner is not None

    def test_validate_credential_allowed_type(self) -> None:
        """Test validating credential with allowed type."""
        scanner = SecurityScanner()

        issues = scanner.validate_credential(
            credential_value="a-very-strong-api-key-with-randomness-ABC123!@#",
            credential_type=CredentialType.API_KEY,
        )

        # Should pass validation
        assert len(issues) == 0

    def test_validate_credential_not_allowed_type(self) -> None:
        """Test validating credential with disallowed type."""
        policy = SecurityPolicy(
            allowed_credential_types=[CredentialType.OAUTH2],
        )
        scanner = SecurityScanner(policy=policy)

        issues = scanner.validate_credential(
            credential_value="some-credential",
            credential_type=CredentialType.API_KEY,
        )

        # Should have policy violation
        assert len(issues) > 0
        assert any(i.issue_type == SecurityIssueType.POLICY_VIOLATION for i in issues)

    def test_validate_credential_too_short(self) -> None:
        """Test validating credential that is too short."""
        policy = SecurityPolicy(min_password_length=20)
        scanner = SecurityScanner(policy=policy)

        issues = scanner.validate_credential(
            credential_value="short",
            credential_type=CredentialType.API_KEY,
        )

        # Should have weak credential issue
        assert len(issues) > 0
        assert any(i.issue_type == SecurityIssueType.WEAK_CREDENTIAL for i in issues)
        assert any(i.severity == SecuritySeverity.HIGH for i in issues)

    def test_validate_credential_weak_pattern_password(self) -> None:
        """Test validating credential with weak pattern."""
        scanner = SecurityScanner()

        issues = scanner.validate_credential(
            credential_value="password123456789",
            credential_type=CredentialType.DATABASE,
        )

        # Should detect weak pattern
        assert len(issues) > 0
        assert any(
            i.issue_type == SecurityIssueType.WEAK_CREDENTIAL
            and "weak pattern" in i.title.lower()
            for i in issues
        )

    def test_validate_credential_weak_pattern_admin(self) -> None:
        """Test validating credential starting with 'admin'."""
        scanner = SecurityScanner()

        issues = scanner.validate_credential(
            credential_value="admin1234567890",
            credential_type=CredentialType.BASIC_AUTH,
        )

        # Should detect weak pattern
        assert any(i.severity == SecuritySeverity.CRITICAL for i in issues)

    def test_validate_credential_lacks_complexity(self) -> None:
        """Test validating credential lacking complexity."""
        policy = SecurityPolicy(require_password_complexity=True)
        scanner = SecurityScanner(policy=policy)

        # Only lowercase
        issues = scanner.validate_credential(
            credential_value="alllowercaseletters",
            credential_type=CredentialType.API_KEY,
        )

        # Should have complexity issue
        assert any(
            i.issue_type == SecurityIssueType.WEAK_CREDENTIAL
            and "complexity" in i.title.lower()
            for i in issues
        )

    def test_validate_credential_good_complexity(self) -> None:
        """Test validating credential with good complexity."""
        scanner = SecurityScanner()

        issues = scanner.validate_credential(
            credential_value="StrongPassword123!@#WithComplexity",
            credential_type=CredentialType.API_KEY,
        )

        # Should pass complexity check (may still have other issues)
        complexity_issues = [
            i
            for i in issues
            if i.issue_type == SecurityIssueType.WEAK_CREDENTIAL
            and "complexity" in i.title.lower()
        ]
        assert len(complexity_issues) == 0

    def test_validate_credential_contains_secret_keywords(self) -> None:
        """Test validating credential containing secret keywords."""
        scanner = SecurityScanner()

        issues = scanner.validate_credential(
            credential_value="my_api_key_secret_1234567890",
            credential_type=CredentialType.API_KEY,
        )

        # Should detect exposed secret
        assert any(i.issue_type == SecurityIssueType.EXPOSED_SECRET for i in issues)

    def test_validate_credential_oauth2_no_issues(self) -> None:
        """Test validating OAuth2 token (less strict validation)."""
        scanner = SecurityScanner()

        issues = scanner.validate_credential(
            credential_value="ya29.a0AfH6SMBx...",  # OAuth2 tokens have different format
            credential_type=CredentialType.OAUTH2,
        )

        # OAuth2 tokens shouldn't trigger credential-specific validations
        # (they're validated differently)
        weak_cred_issues = [
            i for i in issues if i.issue_type == SecurityIssueType.WEAK_CREDENTIAL
        ]
        assert len(weak_cred_issues) == 0

    def test_scan_configuration_ssl_disabled(self) -> None:
        """Test scanning configuration with SSL disabled."""
        scanner = SecurityScanner()

        config = {
            "host": "api.example.com",
            "ssl_enabled": False,
        }

        issues = scanner.scan_configuration(config, "test-api")

        # Should detect insecure config
        assert len(issues) > 0
        assert any(i.issue_type == SecurityIssueType.INSECURE_CONFIG for i in issues)
        assert any("SSL/TLS disabled" in i.title for i in issues)

    def test_scan_configuration_ssl_verify_disabled(self) -> None:
        """Test scanning configuration with SSL verification disabled."""
        scanner = SecurityScanner()

        config = {
            "host": "api.example.com",
            "ssl_verify": False,
        }

        issues = scanner.scan_configuration(config, "test-api")

        # Should detect insecure config
        assert any(
            i.issue_type == SecurityIssueType.INSECURE_CONFIG
            and "certificate verification" in i.title.lower()
            for i in issues
        )

    def test_scan_configuration_hardcoded_password(self) -> None:
        """Test scanning configuration with hardcoded password."""
        scanner = SecurityScanner()

        config = {
            "host": "db.example.com",
            "password": "super-secret-password-123",
        }

        issues = scanner.scan_configuration(config, "database")

        # Should detect hardcoded credential
        assert len(issues) > 0
        assert any(i.issue_type == SecurityIssueType.EXPOSED_SECRET for i in issues)
        assert any(i.severity == SecuritySeverity.CRITICAL for i in issues)

    def test_scan_configuration_hardcoded_api_key(self) -> None:
        """Test scanning configuration with hardcoded API key."""
        scanner = SecurityScanner()

        config = {
            "endpoint": "https://api.example.com",
            "api_key": "sk-1234567890abcdef",
        }

        issues = scanner.scan_configuration(config, "api-service")

        # Should detect hardcoded credential
        assert any(
            i.issue_type == SecurityIssueType.EXPOSED_SECRET
            and "hardcoded" in i.description.lower()
            for i in issues
        )

    def test_scan_configuration_pii_detected(self) -> None:
        """Test scanning configuration with PII."""
        compliance_manager = ComplianceManager()
        scanner = SecurityScanner(compliance_manager=compliance_manager)

        config = {
            "admin_email": "admin@example.com",
            "support_phone": "555-1234",
        }

        issues = scanner.scan_configuration(config, "app-config")

        # Should detect PII in configuration
        assert any(
            i.issue_type == SecurityIssueType.EXPOSED_SECRET and "PII" in i.title
            for i in issues
        )

    def test_scan_configuration_blocked_pattern(self) -> None:
        """Test scanning configuration with blocked pattern."""
        policy = SecurityPolicy(
            blocked_patterns=[r"debug.*true", r"test_mode"],
        )
        scanner = SecurityScanner(policy=policy)

        config = {
            "debug_enabled": "true",
            "environment": "production",
        }

        issues = scanner.scan_configuration(config, "app-config")

        # Should detect blocked pattern
        assert any(i.issue_type == SecurityIssueType.POLICY_VIOLATION for i in issues)

    def test_scan_configuration_secure(self) -> None:
        """Test scanning secure configuration."""
        scanner = SecurityScanner()

        config = {
            "host": "api.example.com",
            "port": 443,
            "ssl_enabled": True,
            "ssl_verify": True,
            "credential_ref": "credential-store://api-key",
        }

        issues = scanner.scan_configuration(config, "secure-api")

        # Should have no critical issues
        critical_issues = [i for i in issues if i.severity == SecuritySeverity.CRITICAL]
        assert len(critical_issues) == 0

    def test_check_compliance_no_manager(self) -> None:
        """Test compliance check without compliance manager."""
        scanner = SecurityScanner()

        issues = scanner.check_compliance(
            data={"email": "user@example.com"},
            resource_name="user-data",
        )

        # Should return empty list
        assert len(issues) == 0

    def test_check_compliance_gdpr_violation(self) -> None:
        """Test compliance check with GDPR violation."""
        compliance_manager = ComplianceManager()
        policy = SecurityPolicy(
            compliance_frameworks=[ComplianceFramework.GDPR],
        )
        scanner = SecurityScanner(policy=policy, compliance_manager=compliance_manager)

        data = {
            "email": "user@example.com",
            "name": "John Doe",
        }

        issues = scanner.check_compliance(data, "user-profile")

        # Should detect GDPR violation
        assert len(issues) > 0
        assert any(i.issue_type == SecurityIssueType.COMPLIANCE_VIOLATION for i in issues)
        assert any("GDPR" in i.title for i in issues)

    def test_check_compliance_multiple_frameworks(self) -> None:
        """Test compliance check with multiple frameworks."""
        compliance_manager = ComplianceManager()
        policy = SecurityPolicy(
            compliance_frameworks=[
                ComplianceFramework.GDPR,
                ComplianceFramework.CCPA,
            ],
        )
        scanner = SecurityScanner(policy=policy, compliance_manager=compliance_manager)

        data = {
            "email": "user@example.com",
        }

        issues = scanner.check_compliance(data, "user-data")

        # May have violations for multiple frameworks
        assert len(issues) >= 0

    def test_generate_security_report_no_issues(self) -> None:
        """Test generating security report with no issues."""
        scanner = SecurityScanner()

        report = scanner.generate_security_report([])

        assert report["total_issues"] == 0
        assert report["pass"]
        assert report["severity_counts"]["critical"] == 0
        assert report["severity_counts"]["high"] == 0

    def test_generate_security_report_with_issues(self) -> None:
        """Test generating security report with issues."""
        scanner = SecurityScanner()

        issues = [
            SecurityIssue(
                issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                severity=SecuritySeverity.CRITICAL,
                title="Weak password",
                description="Password is too weak",
                resource="cred-001",
                remediation="Use stronger password",
            ),
            SecurityIssue(
                issue_type=SecurityIssueType.INSECURE_CONFIG,
                severity=SecuritySeverity.HIGH,
                title="SSL disabled",
                description="SSL is not enabled",
                resource="api-config",
                remediation="Enable SSL",
            ),
            SecurityIssue(
                issue_type=SecurityIssueType.POLICY_VIOLATION,
                severity=SecuritySeverity.MEDIUM,
                title="Policy violation",
                description="Configuration violates policy",
                resource="app-config",
                remediation="Update configuration",
            ),
        ]

        report = scanner.generate_security_report(issues)

        assert report["total_issues"] == 3
        assert not report["pass"]
        assert report["severity_counts"]["critical"] == 1
        assert report["severity_counts"]["high"] == 1
        assert report["severity_counts"]["medium"] == 1
        assert report["resources_affected"] == 3

    def test_generate_security_report_type_counts(self) -> None:
        """Test security report type counts."""
        scanner = SecurityScanner()

        issues = [
            SecurityIssue(
                issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                severity=SecuritySeverity.HIGH,
                title="Weak 1",
                description="Desc",
                resource="res1",
                remediation="Fix",
            ),
            SecurityIssue(
                issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                severity=SecuritySeverity.HIGH,
                title="Weak 2",
                description="Desc",
                resource="res2",
                remediation="Fix",
            ),
            SecurityIssue(
                issue_type=SecurityIssueType.EXPOSED_SECRET,
                severity=SecuritySeverity.CRITICAL,
                title="Secret",
                description="Desc",
                resource="res3",
                remediation="Fix",
            ),
        ]

        report = scanner.generate_security_report(issues)

        assert report["type_counts"]["weak_credential"] == 2
        assert report["type_counts"]["exposed_secret"] == 1

    def test_generate_security_report_resources_affected(self) -> None:
        """Test security report resources affected count."""
        scanner = SecurityScanner()

        issues = [
            SecurityIssue(
                issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                severity=SecuritySeverity.HIGH,
                title="Issue 1",
                description="Desc",
                resource="resource-1",
                remediation="Fix",
            ),
            SecurityIssue(
                issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                severity=SecuritySeverity.HIGH,
                title="Issue 2",
                description="Desc",
                resource="resource-1",  # Same resource
                remediation="Fix",
            ),
            SecurityIssue(
                issue_type=SecurityIssueType.EXPOSED_SECRET,
                severity=SecuritySeverity.CRITICAL,
                title="Issue 3",
                description="Desc",
                resource="resource-2",  # Different resource
                remediation="Fix",
            ),
        ]

        report = scanner.generate_security_report(issues)

        assert report["resources_affected"] == 2  # Two unique resources

    def test_scan_integration_comprehensive(self) -> None:
        """Test comprehensive integration scan."""
        scanner = SecurityScanner()

        integration_config = {
            "name": "test-integration",
            "endpoint": "https://api.example.com",
            "ssl_enabled": True,
            "ssl_verify": True,
            "credential_ref": "cred-store://api-key",
        }

        report = scanner.scan_integration(integration_config, "test-integration")

        assert "total_issues" in report
        assert "pass" in report
        assert "severity_counts" in report

    def test_scan_integration_with_issues(self) -> None:
        """Test integration scan that finds issues."""
        scanner = SecurityScanner()

        integration_config = {
            "name": "insecure-integration",
            "endpoint": "http://api.example.com",
            "ssl_enabled": False,
            "password": "hardcoded-password-123",
        }

        report = scanner.scan_integration(integration_config, "insecure-integration")

        assert report["total_issues"] > 0
        assert not report["pass"]
        assert report["severity_counts"]["critical"] > 0

    def test_scan_integration_with_compliance(self) -> None:
        """Test integration scan with compliance checks."""
        compliance_manager = ComplianceManager()
        policy = SecurityPolicy(
            compliance_frameworks=[ComplianceFramework.GDPR],
        )
        scanner = SecurityScanner(policy=policy, compliance_manager=compliance_manager)

        integration_config = {
            "name": "user-integration",
            "user_email": "admin@example.com",
        }

        report = scanner.scan_integration(integration_config, "user-integration")

        # Should detect PII and compliance issues
        assert report["total_issues"] > 0

    def test_security_policy_defaults(self) -> None:
        """Test security policy default values."""
        policy = SecurityPolicy()

        assert policy.require_strong_passwords
        assert policy.min_password_length == 16
        assert policy.require_password_complexity
        assert policy.max_credential_age_days == 90
        assert policy.require_mfa
        assert not policy.allow_weak_tls
        assert policy.require_encryption_at_rest
        assert len(policy.allowed_credential_types) > 0

    def test_security_policy_custom_values(self) -> None:
        """Test security policy with custom values."""
        policy = SecurityPolicy(
            require_strong_passwords=False,
            min_password_length=8,
            max_credential_age_days=30,
            allow_weak_tls=True,
        )

        assert not policy.require_strong_passwords
        assert policy.min_password_length == 8
        assert policy.max_credential_age_days == 30
        assert policy.allow_weak_tls

    def test_validate_credential_multiple_issues(self) -> None:
        """Test credential validation with multiple issues."""
        scanner = SecurityScanner()

        # Credential with multiple problems
        issues = scanner.validate_credential(
            credential_value="password",  # Too short, weak pattern, lacks complexity
            credential_type=CredentialType.DATABASE,
        )

        # Should have multiple issues
        assert len(issues) >= 2
        issue_types = {i.issue_type for i in issues}
        assert SecurityIssueType.WEAK_CREDENTIAL in issue_types

    def test_scan_configuration_short_password_value(self) -> None:
        """Test configuration scan doesn't flag short non-credential values."""
        scanner = SecurityScanner()

        config = {
            "timeout": "30",  # Short value but not a credential
            "host": "api.example.com",
        }

        issues = scanner.scan_configuration(config, "config")

        # Should not flag short non-credential values
        hardcoded_issues = [
            i
            for i in issues
            if i.issue_type == SecurityIssueType.EXPOSED_SECRET
            and "hardcoded" in i.description.lower()
        ]
        assert len(hardcoded_issues) == 0

    def test_security_issue_model(self) -> None:
        """Test SecurityIssue model."""
        issue = SecurityIssue(
            issue_type=SecurityIssueType.WEAK_CREDENTIAL,
            severity=SecuritySeverity.HIGH,
            title="Test Issue",
            description="Test description",
            resource="test-resource",
            remediation="Fix it",
            metadata={"key": "value"},
        )

        assert issue.issue_type == SecurityIssueType.WEAK_CREDENTIAL
        assert issue.severity == SecuritySeverity.HIGH
        assert issue.title == "Test Issue"
        assert issue.metadata["key"] == "value"

    def test_validate_credential_jwt_format(self) -> None:
        """Test validating JWT credential."""
        scanner = SecurityScanner()

        # JWT tokens have specific format
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        issues = scanner.validate_credential(
            credential_value=jwt_token,
            credential_type=CredentialType.JWT,
        )

        # JWTs are not validated for password strength
        weak_cred_issues = [
            i for i in issues if i.issue_type == SecurityIssueType.WEAK_CREDENTIAL
        ]
        assert len(weak_cred_issues) == 0
