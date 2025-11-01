"""
Compliance Validation

Provides GDPR, SOC 2, and security best practices validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class ComplianceStandard(str, Enum):
    """Compliance standards"""

    GDPR = "gdpr"  # General Data Protection Regulation
    SOC2 = "soc2"  # Service Organization Control 2
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # Information Security Management


class ComplianceStatus(str, Enum):
    """Compliance status"""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class ViolationSeverity(str, Enum):
    """Violation severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Compliance violation record"""

    standard: ComplianceStandard
    rule: str
    description: str
    severity: ViolationSeverity
    remediation: str
    detected_at: datetime


class ComplianceConfig(BaseModel):
    """Configuration for compliance validation"""

    enabled_standards: list[ComplianceStandard] = Field(
        default=[ComplianceStandard.GDPR, ComplianceStandard.SOC2],
        description="Enabled compliance standards",
    )
    auto_remediation: bool = Field(
        default=False, description="Enable automatic remediation"
    )
    violation_reporting: bool = Field(
        default=True, description="Enable violation reporting"
    )
    continuous_monitoring: bool = Field(
        default=True, description="Enable continuous compliance monitoring"
    )


class ComplianceReport(BaseModel):
    """Compliance assessment report"""

    report_id: str = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    standards_checked: list[ComplianceStandard]
    overall_status: ComplianceStatus
    compliance_score: float = Field(ge=0.0, le=100.0)
    violations: list[dict[str, Any]]
    recommendations: list[str]
    next_review_date: datetime | None = None


class ComplianceValidator:
    """
    Compliance validation service.

    Provides GDPR, SOC 2, and security best practices validation.
    """

    def __init__(self, config: ComplianceConfig | None = None):
        self.config = config or ComplianceConfig()
        self.logger = structlog.get_logger()

        # Compliance rules
        self._rules = self._initialize_rules()

        # Violation tracking
        self._violations: list[ComplianceViolation] = []

        # Statistics
        self._compliance_stats = {
            "assessments_performed": 0,
            "violations_detected": 0,
            "violations_by_severity": {},
            "remediations_applied": 0,
        }

        self.logger.info(
            "compliance_validator_initialized",
            standards=self.config.enabled_standards,
            auto_remediation=self.config.auto_remediation,
        )

    def _initialize_rules(self) -> dict[ComplianceStandard, list[dict[str, Any]]]:
        """Initialize compliance rules"""
        return {
            ComplianceStandard.GDPR: [
                {
                    "rule": "data_encryption",
                    "description": "Personal data must be encrypted at rest and in transit",
                    "severity": ViolationSeverity.CRITICAL,
                    "check": self._check_data_encryption,
                },
                {
                    "rule": "data_retention",
                    "description": "Data must not be retained longer than necessary",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_data_retention,
                },
                {
                    "rule": "consent_tracking",
                    "description": "User consent must be tracked and documented",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_consent_tracking,
                },
                {
                    "rule": "data_portability",
                    "description": "Users must be able to export their data",
                    "severity": ViolationSeverity.MEDIUM,
                    "check": self._check_data_portability,
                },
                {
                    "rule": "right_to_erasure",
                    "description": "Users must be able to request data deletion",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_right_to_erasure,
                },
            ],
            ComplianceStandard.SOC2: [
                {
                    "rule": "access_control",
                    "description": "Access controls must be implemented",
                    "severity": ViolationSeverity.CRITICAL,
                    "check": self._check_access_control,
                },
                {
                    "rule": "audit_logging",
                    "description": "All operations must be logged",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_audit_logging,
                },
                {
                    "rule": "change_management",
                    "description": "Changes must be tracked and approved",
                    "severity": ViolationSeverity.MEDIUM,
                    "check": self._check_change_management,
                },
                {
                    "rule": "incident_response",
                    "description": "Incident response procedures must be defined",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_incident_response,
                },
                {
                    "rule": "vulnerability_management",
                    "description": "Vulnerabilities must be tracked and remediated",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_vulnerability_management,
                },
            ],
            ComplianceStandard.ISO27001: [
                {
                    "rule": "security_policy",
                    "description": "Security policies must be documented",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_security_policy,
                },
                {
                    "rule": "risk_assessment",
                    "description": "Regular risk assessments must be performed",
                    "severity": ViolationSeverity.HIGH,
                    "check": self._check_risk_assessment,
                },
                {
                    "rule": "employee_training",
                    "description": "Security training must be provided",
                    "severity": ViolationSeverity.MEDIUM,
                    "check": self._check_employee_training,
                },
            ],
        }

    def validate_compliance(
        self, context: dict[str, Any] | None = None
    ) -> ComplianceReport:
        """
        Validate compliance across enabled standards.

        Args:
            context: Validation context with system state

        Returns:
            ComplianceReport with assessment results
        """
        context = context or {}
        self._compliance_stats["assessments_performed"] += 1

        violations: list[ComplianceViolation] = []
        total_checks = 0
        passed_checks = 0

        for standard in self.config.enabled_standards:
            rules = self._rules.get(standard, [])

            for rule in rules:
                total_checks += 1
                check_result = rule["check"](context)

                if not check_result["compliant"]:
                    violation = ComplianceViolation(
                        standard=standard,
                        rule=rule["rule"],
                        description=rule["description"],
                        severity=rule["severity"],
                        remediation=check_result.get(
                            "remediation", "No remediation available"
                        ),
                        detected_at=datetime.now(UTC),
                    )
                    violations.append(violation)
                    self._violations.append(violation)
                    self._compliance_stats["violations_detected"] += 1
                    self._compliance_stats["violations_by_severity"][
                        violation.severity.value
                    ] = (
                        self._compliance_stats["violations_by_severity"].get(
                            violation.severity.value, 0
                        )
                        + 1
                    )
                else:
                    passed_checks += 1

        # Calculate compliance score
        compliance_score = (
            (passed_checks / total_checks * 100) if total_checks > 0 else 100.0
        )

        # Determine overall status
        if compliance_score == 100.0:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 80.0:
            overall_status = ComplianceStatus.PARTIAL
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT

        # Generate recommendations
        recommendations = self._generate_recommendations(violations)

        report = ComplianceReport(
            standards_checked=self.config.enabled_standards,
            overall_status=overall_status,
            compliance_score=compliance_score,
            violations=[
                {
                    "standard": v.standard.value,
                    "rule": v.rule,
                    "description": v.description,
                    "severity": v.severity.value,
                    "remediation": v.remediation,
                    "detected_at": v.detected_at.isoformat(),
                }
                for v in violations
            ],
            recommendations=recommendations,
        )

        self.logger.info(
            "compliance_validation_completed",
            status=overall_status.value,
            score=compliance_score,
            violations=len(violations),
        )

        return report

    def _check_data_encryption(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check if data encryption is enabled"""
        encryption_enabled = context.get("encryption_enabled", False)
        return {
            "compliant": encryption_enabled,
            "remediation": "Enable AES-256 encryption for all sensitive data",
        }

    def _check_data_retention(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check data retention policies"""
        retention_policy = context.get("retention_policy_enabled", False)
        return {
            "compliant": retention_policy,
            "remediation": "Implement data retention policy with automatic cleanup",
        }

    def _check_consent_tracking(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check consent tracking"""
        consent_tracking = context.get("consent_tracking_enabled", False)
        return {
            "compliant": consent_tracking,
            "remediation": "Implement user consent tracking and documentation",
        }

    def _check_data_portability(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check data portability"""
        export_enabled = context.get("data_export_enabled", False)
        return {
            "compliant": export_enabled,
            "remediation": "Implement data export functionality for users",
        }

    def _check_right_to_erasure(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check right to erasure"""
        deletion_enabled = context.get("data_deletion_enabled", False)
        return {
            "compliant": deletion_enabled,
            "remediation": "Implement user data deletion functionality",
        }

    def _check_access_control(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check access control implementation"""
        rbac_enabled = context.get("rbac_enabled", False)
        return {
            "compliant": rbac_enabled,
            "remediation": "Implement role-based access control (RBAC)",
        }

    def _check_audit_logging(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check audit logging"""
        audit_enabled = context.get("audit_logging_enabled", False)
        return {
            "compliant": audit_enabled,
            "remediation": "Enable comprehensive audit logging for all operations",
        }

    def _check_change_management(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check change management"""
        change_tracking = context.get("change_tracking_enabled", False)
        return {
            "compliant": change_tracking,
            "remediation": "Implement change tracking and approval workflows",
        }

    def _check_incident_response(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check incident response procedures"""
        incident_response = context.get("incident_response_plan", False)
        return {
            "compliant": incident_response,
            "remediation": "Document incident response procedures and train staff",
        }

    def _check_vulnerability_management(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check vulnerability management"""
        vuln_scanning = context.get("vulnerability_scanning_enabled", False)
        return {
            "compliant": vuln_scanning,
            "remediation": "Implement regular vulnerability scanning and remediation",
        }

    def _check_security_policy(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check security policy documentation"""
        policy_documented = context.get("security_policy_documented", False)
        return {
            "compliant": policy_documented,
            "remediation": "Document comprehensive security policies",
        }

    def _check_risk_assessment(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check risk assessment"""
        risk_assessment = context.get("risk_assessment_performed", False)
        return {
            "compliant": risk_assessment,
            "remediation": "Perform regular security risk assessments",
        }

    def _check_employee_training(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check employee security training"""
        training_provided = context.get("security_training_provided", False)
        return {
            "compliant": training_provided,
            "remediation": "Provide regular security awareness training",
        }

    def _generate_recommendations(
        self, violations: list[ComplianceViolation]
    ) -> list[str]:
        """Generate recommendations based on violations"""
        recommendations = []

        # Group by severity
        critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        high = [v for v in violations if v.severity == ViolationSeverity.HIGH]

        if critical:
            recommendations.append(
                f"Address {len(critical)} critical violations immediately"
            )

        if high:
            recommendations.append(
                f"Remediate {len(high)} high-severity violations within 30 days"
            )

        # Standard-specific recommendations
        gdpr_violations = [v for v in violations if v.standard == ComplianceStandard.GDPR]
        if gdpr_violations:
            recommendations.append("Review GDPR compliance requirements and implement missing controls")

        soc2_violations = [v for v in violations if v.standard == ComplianceStandard.SOC2]
        if soc2_violations:
            recommendations.append("Strengthen SOC 2 controls for security and availability")

        if not violations:
            recommendations.append("Maintain current compliance posture with regular monitoring")

        return recommendations

    def get_violations(
        self,
        standard: ComplianceStandard | None = None,
        severity: ViolationSeverity | None = None,
    ) -> list[ComplianceViolation]:
        """
        Get compliance violations.

        Args:
            standard: Filter by standard
            severity: Filter by severity

        Returns:
            List of violations
        """
        filtered = self._violations

        if standard:
            filtered = [v for v in filtered if v.standard == standard]

        if severity:
            filtered = [v for v in filtered if v.severity == severity]

        return filtered

    def get_compliance_stats(self) -> dict[str, Any]:
        """Get compliance statistics"""
        return {
            **self._compliance_stats,
            "total_violations": len(self._violations),
            "enabled_standards": [s.value for s in self.config.enabled_standards],
        }
