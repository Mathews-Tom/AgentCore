"""
Tests for compliance validation
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentcore.dspy_optimization.security.compliance import (
    ComplianceConfig,
    ComplianceReport,
    ComplianceStandard,
    ComplianceStatus,
    ComplianceValidator,
    ViolationSeverity,
)


class TestComplianceConfig:
    """Tests for ComplianceConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = ComplianceConfig()
        assert ComplianceStandard.GDPR in config.enabled_standards
        assert ComplianceStandard.SOC2 in config.enabled_standards
        assert config.auto_remediation is False
        assert config.violation_reporting is True


class TestComplianceValidator:
    """Tests for ComplianceValidator"""

    @pytest.fixture
    def validator(self) -> ComplianceValidator:
        """Create compliance validator"""
        return ComplianceValidator()

    def test_initialization(self, validator: ComplianceValidator):
        """Test validator initialization"""
        assert len(validator.config.enabled_standards) > 0
        assert len(validator._rules) > 0

    def test_validate_compliance_all_compliant(self, validator: ComplianceValidator):
        """Test validation when all checks pass"""
        context = {
            "encryption_enabled": True,
            "retention_policy_enabled": True,
            "consent_tracking_enabled": True,
            "data_export_enabled": True,
            "data_deletion_enabled": True,
            "rbac_enabled": True,
            "audit_logging_enabled": True,
            "change_tracking_enabled": True,
            "incident_response_plan": True,
            "vulnerability_scanning_enabled": True,
        }

        report = validator.validate_compliance(context)

        assert report.overall_status == ComplianceStatus.COMPLIANT
        assert report.compliance_score == 100.0
        assert len(report.violations) == 0

    def test_validate_compliance_non_compliant(self, validator: ComplianceValidator):
        """Test validation when checks fail"""
        context = {
            "encryption_enabled": False,
            "rbac_enabled": False,
        }

        report = validator.validate_compliance(context)

        assert report.overall_status == ComplianceStatus.NON_COMPLIANT
        assert report.compliance_score < 100.0
        assert len(report.violations) > 0

    def test_validate_compliance_partial(self, validator: ComplianceValidator):
        """Test partial compliance"""
        context = {
            "encryption_enabled": True,
            "rbac_enabled": True,
            "audit_logging_enabled": True,
            "retention_policy_enabled": False,
        }

        report = validator.validate_compliance(context)

        # Should have some violations but not all
        assert report.compliance_score > 0
        assert report.compliance_score < 100
        assert len(report.violations) > 0

    def test_gdpr_data_encryption_check(self, validator: ComplianceValidator):
        """Test GDPR data encryption check"""
        context_compliant = {"encryption_enabled": True}
        result = validator._check_data_encryption(context_compliant)
        assert result["compliant"] is True

        context_non_compliant = {"encryption_enabled": False}
        result = validator._check_data_encryption(context_non_compliant)
        assert result["compliant"] is False
        assert "remediation" in result

    def test_gdpr_data_retention_check(self, validator: ComplianceValidator):
        """Test GDPR data retention check"""
        context = {"retention_policy_enabled": True}
        result = validator._check_data_retention(context)
        assert result["compliant"] is True

    def test_soc2_access_control_check(self, validator: ComplianceValidator):
        """Test SOC2 access control check"""
        context = {"rbac_enabled": True}
        result = validator._check_access_control(context)
        assert result["compliant"] is True

    def test_soc2_audit_logging_check(self, validator: ComplianceValidator):
        """Test SOC2 audit logging check"""
        context = {"audit_logging_enabled": True}
        result = validator._check_audit_logging(context)
        assert result["compliant"] is True

    def test_violation_severity_tracking(self, validator: ComplianceValidator):
        """Test violation severity tracking"""
        context = {
            "encryption_enabled": False,  # CRITICAL
            "rbac_enabled": False,  # CRITICAL
        }

        validator.validate_compliance(context)

        critical_violations = validator.get_violations(severity=ViolationSeverity.CRITICAL)
        assert len(critical_violations) >= 2

    def test_get_violations_by_standard(self, validator: ComplianceValidator):
        """Test filtering violations by standard"""
        context = {
            "encryption_enabled": False,  # GDPR
            "rbac_enabled": False,  # SOC2
        }

        validator.validate_compliance(context)

        gdpr_violations = validator.get_violations(standard=ComplianceStandard.GDPR)
        assert len(gdpr_violations) > 0
        assert all(v.standard == ComplianceStandard.GDPR for v in gdpr_violations)

    def test_generate_recommendations(self, validator: ComplianceValidator):
        """Test recommendation generation"""
        context = {
            "encryption_enabled": False,  # CRITICAL
            "retention_policy_enabled": False,  # HIGH
        }

        report = validator.validate_compliance(context)

        assert len(report.recommendations) > 0
        assert any("critical" in rec.lower() for rec in report.recommendations)

    def test_compliance_report_structure(self, validator: ComplianceValidator):
        """Test compliance report structure"""
        context = {"encryption_enabled": True}
        report = validator.validate_compliance(context)

        assert isinstance(report, ComplianceReport)
        assert report.report_id is not None
        assert isinstance(report.generated_at, datetime)
        assert isinstance(report.standards_checked, list)
        assert isinstance(report.overall_status, ComplianceStatus)
        assert 0.0 <= report.compliance_score <= 100.0
        assert isinstance(report.violations, list)
        assert isinstance(report.recommendations, list)

    def test_compliance_score_calculation(self, validator: ComplianceValidator):
        """Test compliance score calculation"""
        # All compliant
        context_full = {
            "encryption_enabled": True,
            "retention_policy_enabled": True,
            "consent_tracking_enabled": True,
            "data_export_enabled": True,
            "data_deletion_enabled": True,
            "rbac_enabled": True,
            "audit_logging_enabled": True,
            "change_tracking_enabled": True,
            "incident_response_plan": True,
            "vulnerability_scanning_enabled": True,
        }
        report = validator.validate_compliance(context_full)
        assert report.compliance_score == 100.0

        # None compliant
        context_empty = {}
        report = validator.validate_compliance(context_empty)
        assert report.compliance_score < 100.0

    def test_multiple_standards(self):
        """Test validation across multiple standards"""
        config = ComplianceConfig(
            enabled_standards=[
                ComplianceStandard.GDPR,
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ]
        )
        validator = ComplianceValidator(config)

        context = {}
        report = validator.validate_compliance(context)

        assert len(report.standards_checked) == 3
        assert ComplianceStandard.GDPR in report.standards_checked
        assert ComplianceStandard.SOC2 in report.standards_checked
        assert ComplianceStandard.ISO27001 in report.standards_checked

    def test_get_compliance_stats(self, validator: ComplianceValidator):
        """Test getting compliance statistics"""
        context = {"encryption_enabled": False}
        validator.validate_compliance(context)

        stats = validator.get_compliance_stats()

        assert stats["assessments_performed"] >= 1
        assert stats["violations_detected"] > 0
        assert "enabled_standards" in stats

    def test_violation_tracking(self, validator: ComplianceValidator):
        """Test violation tracking across assessments"""
        context = {"encryption_enabled": False}

        # Run multiple assessments
        validator.validate_compliance(context)
        validator.validate_compliance(context)

        violations = validator.get_violations()
        assert len(violations) >= 2  # Should track all violations

    def test_remediation_recommendations(self, validator: ComplianceValidator):
        """Test remediation recommendations are provided"""
        context = {"encryption_enabled": False}
        validator.validate_compliance(context)

        violations = validator.get_violations()
        for violation in violations:
            assert violation.remediation is not None
            assert len(violation.remediation) > 0

    def test_iso27001_checks(self):
        """Test ISO27001 specific checks"""
        config = ComplianceConfig(enabled_standards=[ComplianceStandard.ISO27001])
        validator = ComplianceValidator(config)

        context = {
            "security_policy_documented": True,
            "risk_assessment_performed": True,
            "security_training_provided": True,
        }

        report = validator.validate_compliance(context)
        assert report.compliance_score == 100.0

    def test_empty_context(self, validator: ComplianceValidator):
        """Test validation with empty context"""
        report = validator.validate_compliance({})

        assert report.overall_status == ComplianceStatus.NON_COMPLIANT
        assert len(report.violations) > 0
        assert report.compliance_score < 100.0


class TestComplianceIntegration:
    """Integration tests for compliance validation"""

    def test_complete_compliance_workflow(self):
        """Test complete compliance workflow"""
        validator = ComplianceValidator()

        # Initial assessment - non-compliant
        context_initial = {
            "encryption_enabled": False,
            "rbac_enabled": False,
        }
        report_initial = validator.validate_compliance(context_initial)
        assert report_initial.overall_status != ComplianceStatus.COMPLIANT
        initial_score = report_initial.compliance_score

        # After remediation - compliant
        context_remediated = {
            "encryption_enabled": True,
            "retention_policy_enabled": True,
            "consent_tracking_enabled": True,
            "data_export_enabled": True,
            "data_deletion_enabled": True,
            "rbac_enabled": True,
            "audit_logging_enabled": True,
            "change_tracking_enabled": True,
            "incident_response_plan": True,
            "vulnerability_scanning_enabled": True,
        }
        report_remediated = validator.validate_compliance(context_remediated)
        assert report_remediated.overall_status == ComplianceStatus.COMPLIANT
        assert report_remediated.compliance_score > initial_score

    def test_progressive_compliance_improvement(self):
        """Test progressive compliance improvement"""
        validator = ComplianceValidator()

        # Stage 1: No compliance
        context1 = {}
        report1 = validator.validate_compliance(context1)
        score1 = report1.compliance_score

        # Stage 2: Some compliance
        context2 = {
            "encryption_enabled": True,
            "rbac_enabled": True,
        }
        report2 = validator.validate_compliance(context2)
        score2 = report2.compliance_score

        # Stage 3: Full compliance
        context3 = {
            "encryption_enabled": True,
            "retention_policy_enabled": True,
            "consent_tracking_enabled": True,
            "data_export_enabled": True,
            "data_deletion_enabled": True,
            "rbac_enabled": True,
            "audit_logging_enabled": True,
            "change_tracking_enabled": True,
            "incident_response_plan": True,
            "vulnerability_scanning_enabled": True,
        }
        report3 = validator.validate_compliance(context3)
        score3 = report3.compliance_score

        # Scores should improve
        assert score1 < score2 < score3

    def test_all_standards_compliance(self):
        """Test compliance across all standards"""
        config = ComplianceConfig(
            enabled_standards=[
                ComplianceStandard.GDPR,
                ComplianceStandard.SOC2,
                ComplianceStandard.ISO27001,
            ]
        )
        validator = ComplianceValidator(config)

        context = {
            "encryption_enabled": True,
            "retention_policy_enabled": True,
            "consent_tracking_enabled": True,
            "data_export_enabled": True,
            "data_deletion_enabled": True,
            "rbac_enabled": True,
            "audit_logging_enabled": True,
            "change_tracking_enabled": True,
            "incident_response_plan": True,
            "vulnerability_scanning_enabled": True,
            "security_policy_documented": True,
            "risk_assessment_performed": True,
            "security_training_provided": True,
        }

        report = validator.validate_compliance(context)

        assert report.overall_status == ComplianceStatus.COMPLIANT
        assert report.compliance_score == 100.0
        assert len(report.violations) == 0

    def test_critical_violations_flagged(self):
        """Test critical violations are properly flagged"""
        validator = ComplianceValidator()

        context = {
            "encryption_enabled": False,  # CRITICAL
            "rbac_enabled": False,  # CRITICAL
        }

        report = validator.validate_compliance(context)

        critical_violations = [
            v for v in report.violations if v["severity"] == ViolationSeverity.CRITICAL.value
        ]

        assert len(critical_violations) >= 2
        assert any("critical" in rec.lower() for rec in report.recommendations)
