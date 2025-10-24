"""Security scanning and validation.

Provides credential validation, security policy enforcement,
vulnerability scanning, and compliance checks.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from agentcore.integration.security.compliance import (
    ComplianceFramework,
    ComplianceManager,
    DataClassification,
)
from agentcore.integration.security.credential_manager import CredentialType

logger = structlog.get_logger(__name__)


class SecuritySeverity(str, Enum):
    """Security issue severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityIssueType(str, Enum):
    """Types of security issues."""

    WEAK_CREDENTIAL = "weak_credential"
    EXPIRED_CREDENTIAL = "expired_credential"
    EXPOSED_SECRET = "exposed_secret"
    POLICY_VIOLATION = "policy_violation"
    VULNERABILITY = "vulnerability"
    COMPLIANCE_VIOLATION = "compliance_violation"
    INSECURE_CONFIG = "insecure_config"


class SecurityIssue(BaseModel):
    """Security issue detected by scanner.

    Represents a security vulnerability or policy violation with
    details for remediation.
    """

    issue_type: SecurityIssueType = Field(
        description="Type of security issue",
    )
    severity: SecuritySeverity = Field(
        description="Issue severity level",
    )
    title: str = Field(
        description="Issue title",
    )
    description: str = Field(
        description="Detailed description",
    )
    resource: str = Field(
        description="Affected resource identifier",
    )
    remediation: str = Field(
        description="Recommended remediation steps",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional issue metadata",
    )


class SecurityPolicy(BaseModel):
    """Security policy configuration.

    Defines security requirements and validation rules for
    credentials, configurations, and data handling.
    """

    require_strong_passwords: bool = Field(
        default=True,
        description="Require strong passwords (min length, complexity)",
    )
    min_password_length: int = Field(
        default=16,
        description="Minimum password length",
        ge=8,
        le=128,
    )
    require_password_complexity: bool = Field(
        default=True,
        description="Require uppercase, lowercase, numbers, symbols",
    )
    max_credential_age_days: int = Field(
        default=90,
        description="Maximum credential age before rotation required",
        ge=30,
        le=365,
    )
    require_mfa: bool = Field(
        default=True,
        description="Require multi-factor authentication",
    )
    allow_weak_tls: bool = Field(
        default=False,
        description="Allow TLS versions below 1.3",
    )
    require_encryption_at_rest: bool = Field(
        default=True,
        description="Require encryption at rest for sensitive data",
    )
    allowed_credential_types: list[CredentialType] = Field(
        default_factory=lambda: [
            CredentialType.API_KEY,
            CredentialType.OAUTH2,
            CredentialType.JWT,
        ],
        description="Allowed credential types",
    )
    blocked_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for blocked content (e.g., hardcoded secrets)",
    )
    compliance_frameworks: list[ComplianceFramework] = Field(
        default_factory=list,
        description="Required compliance frameworks",
    )


class SecurityScanner:
    """Security scanner and validation system.

    Provides credential validation, security policy enforcement,
    vulnerability scanning, and compliance checks.
    """

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        compliance_manager: ComplianceManager | None = None,
    ) -> None:
        """Initialize security scanner.

        Args:
            policy: Security policy configuration
            compliance_manager: Compliance manager for checks
        """
        self._policy = policy or SecurityPolicy()
        self._compliance_manager = compliance_manager

        # Common weak password patterns (case-insensitive matching in validation)
        self._weak_patterns = [
            "password",
            "123",
            "admin",
            "default",
            "guest",
            "root",
            "test",
        ]

        logger.info(
            "security_scanner_initialized",
            has_policy=policy is not None,
            has_compliance_manager=compliance_manager is not None,
        )

    def validate_credential(
        self,
        credential_value: str,
        credential_type: CredentialType,
    ) -> list[SecurityIssue]:
        """Validate credential strength and format.

        Args:
            credential_value: Credential to validate
            credential_type: Type of credential

        Returns:
            List of security issues found (empty if valid)
        """
        issues: list[SecurityIssue] = []

        # Check credential type is allowed
        if credential_type not in self._policy.allowed_credential_types:
            issues.append(
                SecurityIssue(
                    issue_type=SecurityIssueType.POLICY_VIOLATION,
                    severity=SecuritySeverity.HIGH,
                    title="Credential type not allowed",
                    description=f"Credential type {credential_type.value} is not in allowed types",
                    resource=credential_type.value,
                    remediation=f"Use one of allowed types: {[t.value for t in self._policy.allowed_credential_types]}",
                )
            )

        # Validate based on credential type
        if credential_type in (
            CredentialType.API_KEY,
            CredentialType.BASIC_AUTH,
            CredentialType.DATABASE,
        ):
            # Check length
            if len(credential_value) < self._policy.min_password_length:
                issues.append(
                    SecurityIssue(
                        issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                        severity=SecuritySeverity.HIGH,
                        title="Credential too short",
                        description=f"Credential length {len(credential_value)} is below minimum {self._policy.min_password_length}",
                        resource="credential",
                        remediation=f"Use credential with at least {self._policy.min_password_length} characters",
                    )
                )

            # Check for weak patterns
            if self._policy.require_strong_passwords:
                credential_lower = credential_value.lower()
                for pattern in self._weak_patterns:
                    if credential_lower.startswith(pattern):
                        issues.append(
                            SecurityIssue(
                                issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                                severity=SecuritySeverity.CRITICAL,
                                title="Weak pattern detected",
                                description=f"Credential starts with weak pattern: {pattern}",
                                resource="credential",
                                remediation="Use a strong, randomly generated credential",
                            )
                        )
                        break

            # Check complexity
            if self._policy.require_password_complexity:
                has_upper = any(c.isupper() for c in credential_value)
                has_lower = any(c.islower() for c in credential_value)
                has_digit = any(c.isdigit() for c in credential_value)
                has_symbol = any(not c.isalnum() for c in credential_value)

                if not all([has_upper, has_lower, has_digit, has_symbol]):
                    issues.append(
                        SecurityIssue(
                            issue_type=SecurityIssueType.WEAK_CREDENTIAL,
                            severity=SecuritySeverity.MEDIUM,
                            title="Credential lacks complexity",
                            description="Credential must contain uppercase, lowercase, numbers, and symbols",
                            resource="credential",
                            remediation="Add uppercase, lowercase, numbers, and special characters",
                        )
                    )

        # Check for common secrets
        common_secrets = ["api_key", "secret", "password", "token"]
        if any(secret in credential_value.lower() for secret in common_secrets):
            issues.append(
                SecurityIssue(
                    issue_type=SecurityIssueType.EXPOSED_SECRET,
                    severity=SecuritySeverity.HIGH,
                    title="Credential contains secret keywords",
                    description="Credential should not contain identifiable secret keywords",
                    resource="credential",
                    remediation="Use randomly generated credential without keywords",
                )
            )

        if issues:
            logger.warning(
                "credential_validation_failed",
                credential_type=credential_type.value,
                issue_count=len(issues),
                severities=[i.severity.value for i in issues],
            )
        else:
            logger.debug(
                "credential_validated",
                credential_type=credential_type.value,
            )

        return issues

    def scan_configuration(
        self,
        config: dict[str, Any],
        resource_name: str,
    ) -> list[SecurityIssue]:
        """Scan configuration for security issues.

        Args:
            config: Configuration to scan
            resource_name: Name of resource being configured

        Returns:
            List of security issues found
        """
        issues: list[SecurityIssue] = []

        # Check for insecure TLS settings
        if "ssl_enabled" in config and not config["ssl_enabled"]:
            issues.append(
                SecurityIssue(
                    issue_type=SecurityIssueType.INSECURE_CONFIG,
                    severity=SecuritySeverity.HIGH,
                    title="SSL/TLS disabled",
                    description="SSL/TLS encryption is disabled",
                    resource=resource_name,
                    remediation="Enable SSL/TLS encryption",
                )
            )

        if "ssl_verify" in config and not config["ssl_verify"]:
            issues.append(
                SecurityIssue(
                    issue_type=SecurityIssueType.INSECURE_CONFIG,
                    severity=SecuritySeverity.MEDIUM,
                    title="SSL certificate verification disabled",
                    description="SSL certificate verification is disabled",
                    resource=resource_name,
                    remediation="Enable SSL certificate verification",
                )
            )

        # Check for exposed secrets in config
        config_str = str(config)
        if self._compliance_manager:
            pii_matches = self._compliance_manager.detect_pii(config_str)
            if pii_matches:
                issues.append(
                    SecurityIssue(
                        issue_type=SecurityIssueType.EXPOSED_SECRET,
                        severity=SecuritySeverity.CRITICAL,
                        title="PII detected in configuration",
                        description=f"PII types detected: {list(pii_matches.keys())}",
                        resource=resource_name,
                        remediation="Remove PII from configuration, use credential references",
                    )
                )

        # Check for hardcoded credentials (but skip credential references)
        credential_keywords = ["password", "secret", "key", "token", "credential"]
        for key, value in config.items():
            if isinstance(value, str):
                if any(keyword in key.lower() for keyword in credential_keywords):
                    # Skip if it's a reference (e.g., credential-store://, ${VAR}, etc.)
                    if not value.startswith(("credential-store://", "${", "$", "arn:", "vault:")):
                        if len(value) > 8:  # Likely a real credential
                            issues.append(
                                SecurityIssue(
                                    issue_type=SecurityIssueType.EXPOSED_SECRET,
                                    severity=SecuritySeverity.CRITICAL,
                                    title="Hardcoded credential detected",
                                    description=f"Configuration key '{key}' contains hardcoded credential",
                                    resource=resource_name,
                                    remediation="Use credential manager references instead of hardcoded values",
                                )
                            )

        # Check blocked patterns
        for pattern in self._policy.blocked_patterns:
            if re.search(pattern, config_str):
                issues.append(
                    SecurityIssue(
                        issue_type=SecurityIssueType.POLICY_VIOLATION,
                        severity=SecuritySeverity.HIGH,
                        title="Blocked pattern detected",
                        description=f"Configuration matches blocked pattern: {pattern}",
                        resource=resource_name,
                        remediation="Remove blocked content from configuration",
                    )
                )

        if issues:
            logger.warning(
                "configuration_scan_issues",
                resource=resource_name,
                issue_count=len(issues),
                critical_count=sum(1 for i in issues if i.severity == SecuritySeverity.CRITICAL),
            )

        return issues

    def check_compliance(
        self,
        data: dict[str, Any],
        resource_name: str,
    ) -> list[SecurityIssue]:
        """Check compliance with required frameworks.

        Args:
            data: Data to check
            resource_name: Resource being checked

        Returns:
            List of compliance issues
        """
        if not self._compliance_manager:
            return []

        issues: list[SecurityIssue] = []

        # Check each required framework
        for framework in self._policy.compliance_frameworks:
            result = self._compliance_manager.check_compliance(framework, data)

            if not result["compliant"]:
                for violation in result["violations"]:
                    issues.append(
                        SecurityIssue(
                            issue_type=SecurityIssueType.COMPLIANCE_VIOLATION,
                            severity=SecuritySeverity.HIGH,
                            title=f"{framework.value.upper()} compliance violation",
                            description=violation,
                            resource=resource_name,
                            remediation=f"Address {framework.value.upper()} requirements",
                            metadata={"framework": framework.value},
                        )
                    )

        return issues

    def generate_security_report(
        self,
        issues: list[SecurityIssue],
    ) -> dict[str, Any]:
        """Generate security scan report.

        Args:
            issues: List of security issues found

        Returns:
            Security report with summary and details
        """
        # Count by severity
        severity_counts = {
            "critical": sum(1 for i in issues if i.severity == SecuritySeverity.CRITICAL),
            "high": sum(1 for i in issues if i.severity == SecuritySeverity.HIGH),
            "medium": sum(1 for i in issues if i.severity == SecuritySeverity.MEDIUM),
            "low": sum(1 for i in issues if i.severity == SecuritySeverity.LOW),
            "info": sum(1 for i in issues if i.severity == SecuritySeverity.INFO),
        }

        # Count by type
        type_counts: dict[str, int] = {}
        for issue in issues:
            type_counts[issue.issue_type.value] = type_counts.get(issue.issue_type.value, 0) + 1

        # Group by resource
        resource_issues: dict[str, list[SecurityIssue]] = {}
        for issue in issues:
            if issue.resource not in resource_issues:
                resource_issues[issue.resource] = []
            resource_issues[issue.resource].append(issue)

        report = {
            "total_issues": len(issues),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "resources_affected": len(resource_issues),
            "pass": len(issues) == 0,
            "issues": [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity.value,
                    "title": i.title,
                    "description": i.description,
                    "resource": i.resource,
                    "remediation": i.remediation,
                }
                for i in issues
            ],
        }

        logger.info(
            "security_report_generated",
            total_issues=report["total_issues"],
            critical_count=severity_counts["critical"],
            high_count=severity_counts["high"],
            pass_status=report["pass"],
        )

        return report

    def scan_integration(
        self,
        integration_config: dict[str, Any],
        integration_name: str,
    ) -> dict[str, Any]:
        """Comprehensive security scan of integration.

        Args:
            integration_config: Integration configuration
            integration_name: Integration name

        Returns:
            Security scan report
        """
        all_issues: list[SecurityIssue] = []

        # Scan configuration
        config_issues = self.scan_configuration(integration_config, integration_name)
        all_issues.extend(config_issues)

        # Check compliance
        if self._compliance_manager:
            compliance_issues = self.check_compliance(integration_config, integration_name)
            all_issues.extend(compliance_issues)

        # Generate report
        report = self.generate_security_report(all_issues)

        logger.info(
            "integration_scan_completed",
            integration_name=integration_name,
            total_issues=len(all_issues),
            pass_status=report["pass"],
        )

        return report
